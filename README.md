# 假钞分类与可疑文字复核

课设有两部分。第一部分比较四个模型识别假标签纸币的表现。第二部分用 Grad-CAM 和 Score-CAM 查看第 6.10 节误判右图的 EfficientNetV2-S 对假钞类别的响应，再读取原图文字，与印度央行公布的发行机构名称核对。模型判真且候选名称有出入时，提示人工对照原图复核。

## 运行顺序

从项目根目录打开[分类与解释 notebook](moneyDetectionFlassfier.ipynb)，选择安装了 OpenCV、PyTorch、torchvision、timm、grad-cam、NumPy、pandas 和 Matplotlib 的 Jupyter 内核，从第一个单元格顺序运行。附加依赖见 [CAM 依赖](requirements-cam.txt)和[模型依赖](requirements-models.txt)。

1. 第 1–7 节比较四个分类器。第 1–4 节准备数据、搭建模型和训练，第 5 节展示三个随机种子的逐轮记录，第 6–7 节比较测试结果。默认读取已有训练记录。将开头的 `TRAIN_AGAIN` 设为 `True` 后重新运行，可在 `runs/` 下的新目录训练并保存模型。
2. 第 8–10 节用第 6.10 节的 100 卢比测试照片和 EfficientNetV2-S（种子 42），查看预测、两种 CAM 热图和像素排序结果。检查点为 `runs/course_models_archive_v3/efficientnet_v2_s/seed_42/best.pt`。第 8 节使用存档运行的抗锯齿双线性缩放，复现 63.9% 的判真分数。
3. 第 11 节用 macOS 上的 Apple Vision 先转录两张原图中的文字，再将含 BANK 的候选行与官方发行机构名核对，在原图和 Score-CAM 上标出文字框，并用 OpenCV 模糊框内像素检查假钞分数的变化。macOS 内核安装依赖后可以重算。其他平台可直接查看 notebook 已保存的两张图和说明。[100 卢比复核图](docs/review_100_rupees.png) 第 11.5 节还会逐张评分 48 张事先选好的同来源图片，比较同样复核 5 张时两种排序找到的漏检数。[新增样本清单](dataset_issuer_review_v1/manifest.json) · [逐图结果](runs/issuer_review_v1/per_photo.csv)

## 使用原缩放方式的历史结果

以下结果和图片保留了早期实验。目前重新训练使用 OpenCV 双线性缩放。第 8–11 节按存档 EfficientNet 的缩放方式复核第 6.10 节的判决。

**分类。** 扩充划分有 187 张训练图、38 张验证图和 53 张测试图。CNN、ResNet18、EfficientNetV2-S 和 DINOv3 各训练三个种子。在该测试集上，EfficientNetV2-S 平均准确率为 90.57%，假钞召回率为 93.06%，真钞误报率为 13.33%。DINOv3 平均准确率为 89.94%，合计 159 次预测中相差一次正确预测。当前解释实验复核该 EfficientNetV2-S 的一次漏检。[四模型比较](runs/course_models_archive_v3/comparison.csv) · [混淆矩阵](runs/course_models_archive_v3/confusion_matrices.png)

**基础热图。** 固定先导 ResNet18、`layer2[-1]`、Fake 类目标和每次修改的面积。两种方法在全部 32 张先导验证图上均可排序。Deletion 从原图逐步移除热区，曲线面积越小越好。Insertion 从替换图逐步恢复热区，曲线面积越大越好。[逐图配对结果](runs/cam_baselines_layer2_val_v1/gradcam/paired_comparison.json)

| 替换方式 | 方法 | Deletion AUC ↓ | Insertion AUC ↑ |
|---|---|---:|---:|
| 均值色 | Grad-CAM | 0.2611 | 0.4094 |
| 均值色 | Score-CAM | 0.2635 | 0.3889 |
| 模糊 | Grad-CAM | 0.3080 | 0.4521 |
| 模糊 | Score-CAM | 0.2871 | 0.4434 |

**局部核验。** 对等面积候选区域，比较两种选法：按 Score-CAM 平均热度选，或按一次局部编辑的实测影响选。另一种编辑负责复查，三个同形对照位置用于扣除修改本身的影响。先按图平均，再按样本汇总。18 张先导测试图的实测结果均与原记录一致。其中 7 张暂定假钞上，中值滤波选区、形态学滤波复查，平均多降低 0.047 的假－真 logit 差；交换顺序后为 0.119。同来源另 10 张此前未使用的假标签图上，这两项差值分别为 0.056 和 0.119。[本次逐图核验](runs/cam_regions_pilot_test_v1/summary.json) · [同来源复核](runs/cam_replication_v1/findings_zh.md)

**历史选区示例。** 2000 卢比图中，同样编辑 357 个像素，按热度选区使假向得分下降 0.58 分，按实测选区下降 2.15 分。100 卢比图中，ResNet18 判真，区域编辑测出了偏假、偏真两个方向。OCR 同时读到玩具钞文字，可供人工复核。[等面积选区对照图](runs/advantages_zh_v1/01_local_selection_zh.png) · [漏检和局部方向图](runs/advantages_zh_v1/02_conflict_evidence_zh.png)

## 课设交付物清单

| 老师要求 | 当前状态 | 提交内容 |
| --- | --- | --- |
| 英文报告，单栏正文至少 7 页 | 旧版已完成，第 11 节改写后案例待同步 | [正式英文报告（Word）](docs/Banknote_Classification_and_Local_Verification_Revised.docx)。已核对排版，共 12 页。前 9 页为正文，第 10 页为结果表，第 11 页为两张纸币的案例，第 12 页为参考文献。 |
| 15 页 PPT | 待制作 | 用第 6 节讲分类对比，用第 11 节的 100 卢比示例讲人工复核。 |
| 15 秒演示视频 | 待制作 | 依次显示 100 卢比原图、ResNet18 判真、OCR 放大的玩具钞字样和复核提示。 |
| 代码及运行说明 | 实验已运行，待整理提交包 | 从头运行[分类与解释 notebook](moneyDetectionFlassfier.ipynb)。随附本 README、两份补充依赖清单，以及下列数据和记录或获取方式。 |

### 代码交付核对

- **四模型比较。** 第 1–7 节使用 `dataset_archive_course_v3/` 和 `runs/course_models_archive_v3/`。默认显示的排名复核还会读取 `dataset_threeclass/manifest.json`。若要重新评估旧测试集，还需要对应的测试图片和检查点。
- **CAM 与文字复核。** 第 8–11 节使用 `runs/course_models_archive_v3/efficientnet_v2_s/seed_42/best.pt`。100 卢比主图来自 `dataset_archive_course_v3/test/fake money/`，2000 卢比补充图来自 `dataset_archive_pilot_v2/test/fake money/`。重跑现场 OCR 需要 macOS 和 `ocrmac`。
- **交付前验收。** 在提交环境检查四模型结果、两张案例图及复核提示，并核对报告与演示中的结果。

- **局限。** 真假标签来自目录，实体身份未经独立核验，验证和测试材料已在项目中查看过。两张示例展示提示的读法。额外 48 张同来源图片评估了提示和误报；跨来源泛化仍需独立验证。
