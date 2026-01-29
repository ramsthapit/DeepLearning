"""
Simple BERT text classifier training (OCR -> BERT).

Trains incrementally in chunks of 1000 images at a time.
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ocr_text_dataset import extract_text_with_cache


CLASS_NAMES = ["not_signed", "signed"]


@dataclass(frozen=True)
class Row:
    id: str
    label: int
    image_path: str


@dataclass(frozen=True)
class Example:
    text: str
    label: int


def read_rows(labels_csv: str, *, document_root: Optional[str] = None) -> List[Row]:
    """Read rows from labels.csv."""
    rows = []
    labels_dir = Path(labels_csv).parent
    
    with open(labels_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            img_path = r["document_path"]
            
            # Resolve path: absolute -> relative to CSV -> document_root fallback
            if not Path(img_path).is_absolute():
                p = (labels_dir / img_path).resolve()
                if not p.exists() and document_root:
                    p = Path(document_root) / Path(img_path).name
                img_path = str(p.resolve())
            
            rows.append(Row(id=r["id"], label=int(r["label"]), image_path=img_path))
    
    return rows


def build_examples(rows: List[Row], *, cache_dir: str) -> List[Example]:
    """Extract OCR text from images."""
    examples = []
    for r in tqdm(rows, desc="OCR", unit="img"):
        txt = extract_text_with_cache(r.image_path, cache_dir=cache_dir)
        examples.append(Example(text=txt, label=r.label))
    return examples


class TextDataset(Dataset):
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def make_collate(tokenizer, *, max_length: int):
    def collate(batch: List[Example]):
        texts = [b.text for b in batch]
        labels = torch.tensor([b.label for b in batch], dtype=torch.long)
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return enc, labels
    return collate


def training_epoch(epoch: int, model, train_loader, optimizer, criterion, device, chunk_num: int, total_chunks: int):
    """Train for one epoch."""
    model.train()
    desc = f"Chunk {chunk_num}/{total_chunks} - Epoch {epoch}"
    
    for inputs, targets in tqdm(train_loader, desc=desc, leave=False):
        inputs = {k: v.to(device, non_blocking=False) for k, v in inputs.items()}
        targets = targets.to(device, non_blocking=False)
        
        optimizer.zero_grad()
        outputs = model(**inputs).logits
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, val_loader, device) -> float:
    """Evaluate model."""
    model.eval()
    correct = 0
    total = 0
    
    for inputs, targets in val_loader:
        inputs = {k: v.to(device, non_blocking=False) for k, v in inputs.items()}
        targets = targets.to(device, non_blocking=False)
        
        outputs = model(**inputs).logits
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / max(1, total)


def train_chunk(
    rows_chunk: List[Row],
    model,
    tokenizer,
    optimizer,
    criterion,
    device,
    epochs: int,
    batch_size: int,
    val_split: float,
    cache_dir: str,
    max_length: int,
    chunk_num: int,
    total_chunks: int,
):
    """Train on a chunk of data."""
    # Extract OCR text
    examples = build_examples(rows_chunk, cache_dir=cache_dir)
    random.shuffle(examples)
    
    # Train/val split
    val_n = max(1, int(len(examples) * val_split))
    train_ex, val_ex = examples[:-val_n], examples[-val_n:]
    print(f"Chunk {chunk_num}: train={len(train_ex)}, val={len(val_ex)}")
    
    # DataLoaders
    collate_fn = make_collate(tokenizer, max_length=max_length)
    train_loader = DataLoader(
        TextDataset(train_ex), batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=False, num_workers=0
    )
    val_loader = DataLoader(
        TextDataset(val_ex), batch_size=max(1, batch_size // 2), shuffle=False,
        collate_fn=collate_fn, pin_memory=False, num_workers=0
    )
    
    # Training
    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        training_epoch(epoch, model, train_loader, optimizer, criterion, device, chunk_num, total_chunks)
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        acc = evaluate(model, val_loader, device)
        print(f"Chunk {chunk_num}/{total_chunks} - Epoch {epoch}: val_acc={acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
    
    return best_acc


@torch.no_grad()
def predict_image_class(
    image_path: str,
    model,
    tokenizer,
    class_names: List[str],
    device: str | torch.device = "cpu",
    *,
    cache_dir: str,
    max_length: int = 256,
) -> str:
    """Predict class from image via OCR + BERT."""
    device = torch.device(device)
    model.eval().to(device)
    
    text = extract_text_with_cache(image_path, cache_dir=cache_dir)
    enc = tokenizer([text], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    
    logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=0)
    class_index = int(probs.argmax().item())
    
    print(f"Top probability: {float(probs.max().item()):.4f}")
    print(f"Predicted class: {class_names[class_index]}")
    return class_names[class_index]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train BERT text classifier")
    ap.add_argument("--labels_csv", default="/home/ram-sthapit/programming/Signature Verification/export_images/train/labels.csv")
    ap.add_argument("--document_root", default="/home/ram-sthapit/programming/Signature Verification/export_images/train/document")
    ap.add_argument("--cache_dir", default="/home/ram-sthapit/programming/Signature Verification/.cache/ocr")
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--out_dir", default="/home/ram-sthapit/programming/Signature Verification/text_model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--chunk_size", type=int, default=1000, help="Images per chunk")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise RuntimeError("Missing transformers. Install: pip install -U transformers")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data and split into chunks
    rows = read_rows(args.labels_csv, document_root=args.document_root)
    chunks = [rows[i:i + args.chunk_size] for i in range(0, len(rows), args.chunk_size)]
    print(f"Total samples: {len(rows)}, Chunks: {len(chunks)}\n")

    # Initialize model
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if (out_dir / "config.json").exists():
        print(f"Loading existing model from {out_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(out_dir)).to(device)
    else:
        print(f"Creating new model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Train on each chunk
    best_acc = -1.0
    for chunk_num, rows_chunk in enumerate(chunks, 1):
        print(f"\n{'='*60}")
        print(f"Training on chunk {chunk_num}/{len(chunks)} ({len(rows_chunk)} samples)")
        print(f"{'='*60}\n")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        acc = train_chunk(
            rows_chunk, model, tokenizer, optimizer, criterion, device,
            args.epochs, args.batch_size, args.val_split, args.cache_dir,
            args.max_length, chunk_num, len(chunks)
        )
        
        # Save after each chunk
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"✓ Saved model (val_acc={acc:.2f}%) -> {out_dir}")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        if acc > best_acc:
            best_acc = acc
    
    print(f"\n{'='*60}")
    print(f"Training complete! Processed {len(rows)} samples in {len(chunks)} chunks")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
