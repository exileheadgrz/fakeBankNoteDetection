# Indian Banknote classification + explanability reimplementation and one alarm method over suspicious text 

This computer vision project compares four models that classify images as real banknotes, fake banknotes, or non-banknotes. It uses Grad-CAM and Score-CAM to examine an EfficientNetV2-S prediction and RISE to test how the fake score changes when image regions are hidden or restored. A customized OCR check flags images for alarm when the model predicts a real banknote but the issuer name differs from “Reserve Bank of India”.

## Run the notebook

From the repository root, run the following commands in a Python virtual environment. Section 11 uses Apple Vision for OCR and requires macOS.

```bash
python -m pip install jupyterlab torch torchvision numpy pandas matplotlib -r requirements-models.txt -r requirements-cam.txt
python -m jupyterlab moneyDetectionFlassfier.ipynb
```

In JupyterLab, select the same Python environment and choose **Run → Run All Cells**. The notebook reads the saved training runs by default. To retrain all four models, set `TRAIN_AGAIN = True` in the first code cell before running it.
