"""
Train a text-only classifier (BERT/DistilBERT/BART-for-classification) on OCR text.

Input: JSONL produced by `ocr_text_dataset.py`
  {"id": ..., "label": 0/1, "text": "...", ...}

Output: HuggingFace model directory with tokenizer + weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Example:
    text: str
    label: int


def read_jsonl(path: str, *, max_samples: int) -> List[Example]:
    ex: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ex.append(Example(text=str(r.get("text", "")), label=int(r["label"])))
            if max_samples > 0 and len(ex) >= max_samples:
                break
    return ex


class TextDs(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


def make_collate(tokenizer, *, max_length: int):
    def collate(batch: List[Example]) -> Dict[str, torch.Tensor]:
        texts = [b.text for b in batch]
        labels = torch.tensor([b.label for b in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return collate


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    losses: List[float] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = float(out.loss.item())
        logits = out.logits
        preds = torch.argmax(logits, dim=-1)
        correct += int((preds == batch["labels"]).sum().item())
        total += int(batch["labels"].shape[0])
        losses.append(loss)
    avg_loss = float(sum(losses) / max(1, len(losses)))
    acc = float(correct / max(1, total))
    return avg_loss, acc


def train() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_jsonl",
        default="/home/ram-sthapit/programming/Signature Verification/export_images/train/ocr_texts.jsonl",
        help="JSONL from ocr_text_dataset.py",
    )
    ap.add_argument(
        "--out_dir",
        default="/home/ram-sthapit/programming/Signature Verification/text_model",
        help="Where to save the HF model+tokenizer",
    )
    ap.add_argument(
        "--model_name",
        default="distilbert-base-uncased",
        help="Any HuggingFace checkpoint. For BART classification you can try `facebook/bart-base`.",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_samples", type=int, default=0, help="<=0 means no limit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    # Lazy import so the file can be imported without transformers installed.
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import get_linear_schedule_with_warmup
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: transformers. Install with:\n"
            "  source venv/bin/activate && pip install -U transformers\n"
        ) from e

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    examples = read_jsonl(args.data_jsonl, max_samples=args.max_samples)
    if len(examples) < 10:
        raise ValueError(f"Need more samples; got {len(examples)} from {args.data_jsonl}")

    random.shuffle(examples)
    val_n = max(1, int(len(examples) * args.val_split))
    train_ex = examples[:-val_n]
    val_ex = examples[-val_n:]
    print(f"samples: train={len(train_ex)} val={len(val_ex)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    train_loader = DataLoader(
        TextDs(train_ex),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer, max_length=args.max_length),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        TextDs(val_ex),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer, max_length=args.max_length),
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * args.epochs)

    warmup_steps = int(0.06 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", unit="batch", leave=False)
        running = 0.0
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running += float(loss.item())
            pbar.set_postfix(train_loss=running / step)

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            (out_dir / "metrics.json").write_text(
                json.dumps({"best_val_loss": best_val_loss, "val_acc": val_acc}, indent=2),
                encoding="utf-8",
            )
            print("saved best ->", str(out_dir))


if __name__ == "__main__":
    train()


