"""
Late fusion: combine CNN detector evidence + BERT text evidence.

This is the practical multimodal fusion for your repo:
- Visual evidence: signature detector max score (from signature_detector.pt)
- Text evidence: p(signature_present) from trained text model (train_text_bert.py)

Fusion model: simple logistic regression (2 -> 1).
"""

from __future__ import annotations

import argparse
import csv
import json
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


@dataclass(frozen=True)
class Row:
    id: str
    label: int
    image_path: str


def read_rows(labels_csv: str, *, document_root: Optional[str]) -> List[Row]:
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


class FeatDs(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def train() -> None:
    ap = argparse.ArgumentParser()
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
    )
    ap.add_argument(
        "--text_model_dir",
        default="/home/ram-sthapit/programming/Signature Verification/text_model",
        help="Directory produced by train_text_bert.py",
    )
    ap.add_argument(
        "--out_path",
        default="/home/ram-sthapit/programming/Signature Verification/fusion_model.pt",
    )
    ap.add_argument(
        "--cache_dir",
        default="/home/ram-sthapit/programming/Signature Verification/.cache/ocr",
    )
    ap.add_argument("--max_samples", type=int, default=500, help="<=0 means no limit")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--score_thresh", type=float, default=0.05, help="Detector threshold for considering boxes.")
    args = ap.parse_args()

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: transformers. Install with:\n"
            "  source venv/bin/activate && pip install -U transformers\n"
        ) from e

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load frozen models
    detector = load_detector(args.detector_ckpt, device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_dir, use_fast=True)
    text_model = AutoModelForSequenceClassification.from_pretrained(args.text_model_dir).to(device).eval()

    @torch.no_grad()
    def text_prob(img_path: str) -> float:
        txt = extract_text_with_cache(img_path, cache_dir=args.cache_dir)
        enc = tokenizer([txt], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = text_model(**enc)
        probs = torch.softmax(out.logits[0], dim=-1)
        return float(probs[1].item())

    @torch.no_grad()
    def det_feats(img_path: str) -> Tuple[float, float]:
        _annot, preds = detector_predict(detector, img_path, device, score_thresh=args.score_thresh)
        if not preds:
            return 0.0, 0.0
        scores = [p[-1] for p in preds]
        return float(max(scores)), float(len(scores))

    rows = read_rows(args.labels_csv, document_root=args.document_root)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    X: List[List[float]] = []
    Y: List[int] = []

    for r in tqdm(rows, desc="features", unit="img"):
        p_txt = text_prob(r.image_path)
        det_max, det_n = det_feats(r.image_path)
        X.append([det_max, p_txt])  # 2D late-fusion features
        Y.append(int(r.label))

    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    # Train/val split
    idx = list(range(x.shape[0]))
    random.shuffle(idx)
    val_n = max(1, int(len(idx) * args.val_split))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[val_idx], y[val_idx]
    print(f"samples: train={len(tr_idx)} val={len(val_idx)}")

    model = torch.nn.Linear(2, 1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    tr_loader = DataLoader(FeatDs(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(FeatDs(x_va, y_va), batch_size=args.batch_size, shuffle=False)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # eval
        model.eval()
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                losses.append(float(loss.item()))
                preds = (torch.sigmoid(logits) >= 0.5).to(torch.float32)
                correct += int((preds == yb).sum().item())
                total += int(yb.numel())
        val_loss = float(sum(losses) / max(1, len(losses)))
        val_acc = float(correct / max(1, total))
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best:
            best = val_loss
            Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "linear_state": model.state_dict(),
                    "features": ["detector_max_score", "text_prob"],
                    "val_loss": best,
                    "val_acc": val_acc,
                },
                args.out_path,
            )
    print("saved:", args.out_path)


if __name__ == "__main__":
    train()


