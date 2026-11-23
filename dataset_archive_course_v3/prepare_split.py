"""Build the expanded three-class course split from the project's archived image index.

Run from the project directory with an empty or new destination:
    python dataset_archive_course_v3/prepare_split.py /path/to/new_split
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path.cwd()
ARCHIVE = ROOT / "archive/data/data"
PILOT = ROOT / "dataset_archive_pilot_v2"
REPLICATION = ROOT / "dataset_cam_replication_v1"
LABELS = {"fake": "fake money", "real": "OR money"}
DENOMS = ("10", "20", "50", "100", "200", "500", "2000")
SPLITS = ("train", "val", "test")
SEED = 20261121
CAP = 55
DISTANCE = 10


def read_json(path):
    return json.loads(path.read_text())


def order(value):
    return hashlib.sha256(f"{SEED}/{value}".encode()).hexdigest()


def family_key(row):
    # Camera timestamps survive changes in folder. Bare sequence numbers recur across denominations.
    name = row["family"]
    if name.startswith(("whatsapp image", "img", "screenshot", "pxl", "dsc")):
        return name
    return "/".join([row["folder_label"], row["denomination"], name])


def close_to(row, view_hashes):
    base = row["phash_d4"][0]
    return any((base ^ h).bit_count() <= DISTANCE for h in view_hashes)


def noise_ratio(path):
    with Image.open(path) as im:
        im.thumbnail((256, 256))
        gray = np.asarray(im.convert("L"), dtype=np.float32)
    return float(np.median(np.abs(gray - cv2.GaussianBlur(gray, (3, 3), 0))) / (gray.std() + 1e-6))


def build(destination):
    index_file = REPLICATION / "index.json"
    images = read_json(index_file)["images"]
    actual = {str(p.relative_to(ROOT)) for p in ARCHIVE.rglob("*")
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".avif", ".webp", ".bmp"}}
    if actual != {r["path"] for r in images}:
        raise ValueError("Archive contents changed. Rebuild the archive image index first.")
    by_path = {r["path"]: r for r in images}
    groups = defaultdict(list)
    pixel_labels = defaultdict(set)
    for row in images:
        groups[family_key(row)].append(row)
        pixel_labels[row["pixel_hash"]].add(row["folder_label"])
    pilot = read_json(PILOT / "manifest.json")
    # Keep source families and near-duplicate representatives in their historical pilot split.
    old_splits = defaultdict(set)
    old_hashes = defaultdict(set)
    old_by_path = {}
    for version in ("dataset_archive_pilot_v1", "dataset_archive_pilot_v2"):
        for row in read_json(ROOT / version / "manifest.json")["samples"]:
            indexed = by_path.get(row["path"])
            if indexed is not None:
                old_splits[family_key(indexed)].add(row["split"])
                old_hashes[row["split"]].update(indexed["phash_d4"])
                old_by_path[row["path"]] = row["split"]
    reserved_keys = {family_key(by_path[r["path"]]) for r in
                     read_json(REPLICATION / "candidate_manifest_before_review.json")["samples"]}
    reserved_hashes = {h for key in reserved_keys for row in groups[key] for h in row["phash_d4"]}
    denied = Counter()
    strata = defaultdict(list)
    for key, members in sorted(groups.items()):
        if key in reserved_keys or len(old_splits[key]) > 1:
            denied["reserved_or_historically_conflicting_family"] += 1
            continue
        if len({(r["folder_label"], r["denomination"]) for r in members}) != 1:
            denied["ambiguous_family_stratum"] += 1
            continue
        if any(len(pixel_labels[r["pixel_hash"]]) > 1 for r in members):
            denied["decoded_label_conflict"] += 1
            continue
        # Prefer original photographs, followed by known pilot representatives.
        members = sorted(members, key=lambda r: (r["is_augmented"], r["path"] not in old_by_path,
                                                  order(r["path"])))
        for row in members:
            if min(row["size"]) < 64 or Path(row["path"]).suffix.lower() not in {".jpg",".jpeg",".png",".avif"}:
                continue
            if close_to(row, reserved_hashes) or noise_ratio(ROOT / row["path"]) > .30:
                continue
            linked = old_splits[key] | {s for s, hashes in old_hashes.items() if close_to(row, hashes)}
            if len(linked) > 1:
                continue
            record = dict(row, family_key=key, assigned=next(iter(linked), None))
            strata[(row["folder_label"], row["denomination"])].append(record)
            break
        else:
            denied["no_eligible_representative"] += 1
    samples = []
    used_hashes = set()
    used_pixels = set()
    matched_counts = {}
    for label in ("fake", "real"):
        for denom in DENOMS:
            chosen = []
            limit = CAP if label == "fake" else matched_counts[denom]
            for row in sorted(strata[label, denom], key=lambda r: order(r["family_key"])):
                if len(chosen) == limit:
                    break
                if row["pixel_hash"] in used_pixels or close_to(row, used_hashes):
                    denied["near_selected_representative"] += 1
                    continue
                used_hashes.update(row["phash_d4"])
                used_pixels.add(row["pixel_hash"])
                chosen.append(row)
            if label == "fake":
                matched_counts[denom] = len(chosen)
            # Fill the most underrepresented split in each class/denomination stratum.
            n = len(chosen)
            target = {"test": round(.20*n), "val": round(.15*n)}
            target["train"] = n - sum(target.values())
            counts = Counter(r["assigned"] for r in chosen if r["assigned"])
            for row in chosen:
                split = row["assigned"]
                if split is None:
                    split = max(SPLITS, key=lambda s: target[s] - counts[s])
                    counts[split] += 1
                cls = LABELS[label]
                path = Path(row["path"])
                samples.append(dict(path=row["path"], class_name=cls, denomination=denom,
                    split=split, sha256=row["sha256"], pixel_hash=row["pixel_hash"], size=row["size"],
                    source_family=row["family_key"], group_id=order(row["family_key"])[:24],
                    is_augmented=row["is_augmented"], label_status="dataset_folder_provisional",
                    physical_identity_verified=False,
                    destination=f"{split}/{cls}/{denom}_{row['sha256'][:16]}{path.suffix.lower()}"))
    for row in pilot["samples"]:
        if row["class_name"] == "not a banknote":
            samples.append(dict(row))
    counts = {s:{name:sum(r["split"]==s and r["class_name"]==name for r in samples)
                 for name in pilot["class_to_idx"]} for s in SPLITS}
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    roster = []
    for r in samples:
        src = ROOT/r["path"] if r["class_name"] != "not a banknote" else PILOT/r["destination"]
        if hashlib.sha256(src.read_bytes()).hexdigest() != r["sha256"]:
            raise ValueError(f"Archived image changed: {src}")
        target = destination/r["destination"]
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target)
        roster.append(dict(split=r["split"], label=r["class_name"], denomination=r.get("denomination",""),
                           split_path=r["destination"],
                           sha256=r["sha256"], group_id=r["group_id"]))
    with (destination/"image_paths.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(roster[0]))
        writer.writeheader()
        writer.writerows(roster)
    manifest = dict(seed=SEED, class_to_idx=pilot["class_to_idx"], counts=counts, samples=samples,
                    archive_index_sha256=hashlib.sha256(index_file.read_bytes()).hexdigest(),
                    archived_banknote_files=len(images), max_per_label_denomination=CAP,
                    excluded_family_or_candidate_counts=dict(denied), min_d4_phash_distance=DISTANCE+1,
                    previous_pilot_banknote_paths_reused=sum(r["path"] in old_by_path for r in samples),
                    selection="One representative per normalized filename family. Prefer originals. Match real counts to available fake counts per denomination. Preserve known pilot split when linked. Exclude prior replication candidates and visually near representatives. Fixed seed, no model predictions used.",
                    limitations=["Folder labels, physical identity, and denomination metadata are unverified.",
                                 "Some families contain only augmented images. Near-duplicate screening cannot establish physical independence.",
                                 "The non-banknote examples and some pilot banknotes were previously used."])
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(counts), "rows", len(samples), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    build(parser.parse_args().destination)

