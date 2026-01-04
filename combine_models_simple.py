"""
Simple combined model: CNN Detector + BERT Text Classifier → Late Fusion

This combines:
  1. CNN Detector (signature_detector.pt) → visual evidence (max detection score)
  2. BERT Text Classifier (text_model_simple/) → text evidence (probability from OCR)

Fusion: Concatenate both features → Linear layer → Final prediction

Simple PyTorch style:
  - training_epoch(epoch)
  - evaluate()
  - predict_combined(image_path, ...)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from infer_signature_detector import load_model as load_detector, predict as detector_predict
from ocr_text_dataset import extract_text_with_cache


CLASS_NAMES = ["not_signed", "signed"]


@dataclass(frozen=True)
class Row:
    id: str
    label: int
    image_path: str


def read_rows(labels_csv: str, *, document_root: Optional[str] = None) -> List[Row]:
    """Read rows from labels.csv"""
    rows: List[Row] = []
    labels_dir = Path(labels_csv).resolve().parent
    with open(labels_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_path = r["document_path"]
            if not os.path.isabs(img_path):
                p1 = (labels_dir / img_path).resolve()
                if p1.exists():
                    img_path = str(p1)
                elif document_root:
                    img_path = str((Path(document_root) / Path(img_path).name).resolve())
            rows.append(Row(id=r["id"], label=int(r["label"]), image_path=img_path))
    return rows


@dataclass(frozen=True)
class CombinedFeatures:
    """Combined features from CNN detector + BERT text model"""
    detector_score: float  # Max detection score from CNN
    text_prob: float        # Probability from BERT text classifier
    label: int


class CombinedDataset(Dataset):
    """Dataset of combined features (detector_score, text_prob)"""
    def __init__(self, features: List[CombinedFeatures]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features[idx]
        x = torch.tensor([feat.detector_score, feat.text_prob], dtype=torch.float32)
        y = torch.tensor(feat.label, dtype=torch.long)
        return x, y


def extract_features(
    rows: List[Row],
    *,
    detector,
    text_model,
    tokenizer,
    device: torch.device,
    cache_dir: str,
    score_thresh: float = 0.05,
    max_length: int = 256,
) -> List[CombinedFeatures]:
    """
    Extract combined features from all images:
      - CNN detector: max detection score
      - BERT text model: probability from OCR text
    """
    features: List[CombinedFeatures] = []

    @torch.no_grad()
    def get_detector_score(img_path: str) -> float:
        """Get max detection score from CNN detector"""
        _annot, preds = detector_predict(detector, img_path, device, score_thresh=score_thresh)
        if not preds:
            return 0.0
        scores = [p[-1] for p in preds]  # p[-1] is the score
        return float(max(scores))

    @torch.no_grad()
    def get_text_prob(img_path: str) -> float:
        """Get probability from BERT text classifier"""
        txt = extract_text_with_cache(img_path, cache_dir=cache_dir)
        enc = tokenizer([txt], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = text_model(**enc)
        probs = torch.softmax(out.logits[0], dim=-1)
        return float(probs[1].item())  # prob of "signed" class

    for r in tqdm(rows, desc="Extracting features", unit="img"):
        det_score = get_detector_score(r.image_path)
        txt_prob = get_text_prob(r.image_path)
        features.append(CombinedFeatures(
            detector_score=det_score,
            text_prob=txt_prob,
            label=r.label
        ))

    return features


def training_epoch(epoch: int, model, train_loader, criterion, optimizer, device):
    """Train the fusion model for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_id, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_id + 1) % 100 == 0:
            print(f'Epoch [{epoch}] Batch [{batch_id+1}/{len(train_loader)}]  '
                  f'Loss: {running_loss/100:.4f}  Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0


def evaluate(model, val_loader, criterion, device):
    """Evaluate the fusion model on validation set"""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total
    avg_loss = running_loss / len(val_loader)
    print(f'Validation Loss: {avg_loss:.4f}  Accuracy: {acc:.2f}%')
    return acc, avg_loss


def predict_combined(
    image_path: str,
    fusion_model,
    detector,
    text_model,
    tokenizer,
    class_names: List[str],
    device: str = 'cpu',
    cache_dir: str = ".cache/ocr",
    score_thresh: float = 0.05,
    max_length: int = 256,
) -> str:
    """
    Predict signature presence using combined CNN + BERT model.

    Args:
        image_path: Path to document image
        fusion_model: Trained fusion model (Linear layer)
        detector: CNN signature detector
        text_model: BERT text classifier
        tokenizer: BERT tokenizer
        class_names: List of class names ["not_signed", "signed"]
        device: 'cpu' or 'cuda'
        cache_dir: OCR cache directory
        score_thresh: Detector score threshold
        max_length: Max token length for BERT

    Returns:
        Predicted class name ("signed" or "not_signed")
    """
    device_t = torch.device(device)

    @torch.no_grad()
    def get_detector_score(img_path: str) -> float:
        _annot, preds = detector_predict(detector, img_path, device_t, score_thresh=score_thresh)
        if not preds:
            return 0.0
        scores = [p[-1] for p in preds]
        return float(max(scores))

    @torch.no_grad()
    def get_text_prob(img_path: str) -> float:
        txt = extract_text_with_cache(img_path, cache_dir=cache_dir)
        enc = tokenizer([txt], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device_t) for k, v in enc.items()}
        out = text_model(**enc)
        probs = torch.softmax(out.logits[0], dim=-1)
        return float(probs[1].item())

    # Extract features
    det_score = get_detector_score(image_path)
    txt_prob = get_text_prob(image_path)

    # Combine features and predict
    x = torch.tensor([[det_score, txt_prob]], dtype=torch.float32).to(device_t)
    
    fusion_model.eval()
    with torch.no_grad():
        outputs = fusion_model(x)
        _, predicted = outputs.max(1)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    print(f"Detector score: {det_score:.4f}")
    print(f"Text probability: {txt_prob:.4f}")
    print(f"Output shape: {outputs.shape}")
    print(f"Top probability: {probabilities.max().item():.4f}")

    class_index = predicted.item()
    print(f"Predicted class: {class_names[class_index]}")

    return class_names[class_index]


def main():
    ap = argparse.ArgumentParser(description="Train combined CNN + BERT fusion model")
    ap.add_argument(
        "--labels_csv",
        default="/home/ram-sthapit/programming/Signature Verification/export_images/train/labels.csv",
    )
    ap.add_argument(
        "--document_root",
        default="/home/ram-sthapit/programming/Signature Verification/export_images/train/document",
    )
    ap.add_argument(
        "--detector_ckpt",
        default="/home/ram-sthapit/programming/Signature Verification/signature_detector.pt",
        help="Path to trained CNN detector checkpoint",
    )
    ap.add_argument(
        "--text_model_dir",
        default="/home/ram-sthapit/programming/Signature Verification/text_model_simple",
        help="Directory containing trained BERT text model",
    )
    ap.add_argument(
        "--out_path",
        default="/home/ram-sthapit/programming/Signature Verification/fusion_model_simple.pt",
        help="Where to save the trained fusion model",
    )
    ap.add_argument(
        "--cache_dir",
        default="/home/ram-sthapit/programming/Signature Verification/.cache/ocr",
    )
    ap.add_argument("--max_samples", type=int, default=500, help="Max samples to use (<=0 = all)")
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--score_thresh", type=float, default=0.05, help="Detector score threshold")
    ap.add_argument("--max_length", type=int, default=256, help="Max token length for BERT")
    args = ap.parse_args()

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise RuntimeError(
            "Missing dependency: transformers. Install with:\n"
            "  source venv/bin/activate && pip install -U transformers\n"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load pre-trained models (frozen)
    print("Loading CNN detector...")
    detector = load_detector(args.detector_ckpt, device)
    detector.eval()

    print("Loading BERT text model...")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_dir, use_fast=True)
    text_model = AutoModelForSequenceClassification.from_pretrained(args.text_model_dir).to(device).eval()

    # Read data
    print("Reading labels.csv...")
    rows = read_rows(args.labels_csv, document_root=args.document_root)
    if args.max_samples > 0:
        rows = rows[:args.max_samples]
    print(f"Loaded {len(rows)} samples")

    # Extract combined features
    print("\nExtracting features from CNN detector + BERT text model...")
    features = extract_features(
        rows,
        detector=detector,
        text_model=text_model,
        tokenizer=tokenizer,
        device=device,
        cache_dir=args.cache_dir,
        score_thresh=args.score_thresh,
        max_length=args.max_length,
    )

    # Train/val split
    random.shuffle(features)
    val_len = max(1, int(len(features) * args.val_split))
    train_features = features[val_len:]
    val_features = features[:val_len]
    print(f"Train: {len(train_features)}, Val: {len(val_features)}")

    train_dataset = CombinedDataset(train_features)
    val_dataset = CombinedDataset(val_features)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Fusion model: Linear(2 -> 2) for binary classification
    model = torch.nn.Linear(2, 2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    print(f"\nTraining fusion model for {args.epochs} epochs...")
    best_acc = 0.0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        training_epoch(epoch, model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                args.out_path,
            )
            print(f"✓ Saved best model (acc={val_acc:.2f}%, loss={val_loss:.4f}) -> {args.out_path}")

    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()

