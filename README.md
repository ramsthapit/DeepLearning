# Signature Verification System

Multimodal signature verification using CNN detector + BERT text classifier with late fusion.

## 🎯 Overview

This system combines:
1. **CNN Detector** - Visual signature detection (Faster R-CNN)
2. **BERT Text Classifier** - Text-based signature detection (OCR + DistilBERT)
3. **Fusion Model** - Combines both modalities for final prediction

---

## 📦 Installation

```bash
cd "/home/ram-sthapit/programming/Signature Verification"
source venv/bin/activate

pip install -U pip
pip install -U pillow tqdm transformers

# Install PyTorch + torchvision
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you have CUDA, install the CUDA build that matches your GPU/driver instead.

---

## 🚀 Quick Start

### Step 1: Train CNN Signature Detector

Trains a Faster R-CNN model to detect signatures visually in document images.

**Data format:**
- Images: `export_images/train/dataset/document/`
- Labels: `export_images/train/dataset/labels.csv` (contains `bbox_json` with normalized `[x1, y1, x2, y2]` boxes)

**Train with chunk-based approach (1000 samples at a time):**

```bash
python train_signature_detector.py \
  --labels_csv "/home/ram-sthapit/programming/Signature Verification/export_images/train/dataset/labels.csv" \
  --document_dir "/home/ram-sthapit/programming/Signature Verification/export_images/train/dataset/document" \
  --chunk_size 1000 \
  --epochs 5 \
  --batch_size 1 \
  --lr 2e-4
```

**Output:** `signature_detector.pt`

**Test the detector:**

```bash
python infer_signature_detector.py \
  --ckpt "/home/ram-sthapit/programming/Signature Verification/signature_detector.pt" \
  --image "/home/ram-sthapit/programming/Signature Verification/export_images/train/dataset/document/test_000003.png" \
  --score 0.5
```

Outputs annotated image to `export_images/predict/` directory.

---

### Step 2: Train BERT Text Classifier

Trains a DistilBERT model to classify signature presence from OCR text.

**Train with chunk-based approach (1000 images at a time):**

```bash
python ocr_to_bert_simple.py \
  --labels_csv "/home/ram-sthapit/programming/Signature Verification/export_images/train/dataset/labels.csv" \
  --document_root "/home/ram-sthapit/programming/Signature Verification/export_images/train/dataset/document" \
  --out_dir "/home/ram-sthapit/programming/Signature Verification/text_model" \
  --chunk_size 1000 \
  --epochs 3 \
  --batch_size 8 \
  --lr 2e-5
```

**Output:** `text_model/` directory

**Test the text model:**

```bash
python infer_text_bert_simple.py \
  --model_dir "/home/ram-sthapit/programming/Signature Verification/text_model" \
  --image "/home/ram-sthapit/programming/Signature Verification/export_images/train/dataset/document/test_000003.png"
```

** Test the fusion model


```
python infer_weighted_avg.py --image validation_00411.jpg
