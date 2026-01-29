"""
Simple CNN signature detector training (Faster R-CNN with ResNet50 backbone).
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from tqdm import tqdm


@dataclass
class Row:
    image_path: str
    bboxes: List[List[float]]


def read_rows(labels_csv: str, document_root: str | None = None) -> List[Row]:
    """Read rows from labels.csv with bbox annotations."""
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
            
            # Parse bboxes
            bboxes = json.loads(r.get("bbox_json", "[]"))
            if not isinstance(bboxes, list):
                bboxes = []
            
            rows.append(Row(image_path=img_path, bboxes=bboxes))
    
    return rows


class SignatureDetectionDataset(Dataset):
    def __init__(self, rows: List[Row]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(row.image_path).convert("RGB")
        w, h = img.size

        # Convert normalized bboxes to pixel coordinates
        boxes = []
        for b in row.bboxes:
            if not (isinstance(b, list) and len(b) == 4):
                continue
            x1, y1, x2, y2 = [max(0.0, min(1.0, float(x))) for x in b]
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            boxes.append([x_min * w, y_min * h, x_max * w, y_max * h])

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.ones(len(boxes), dtype=torch.int64)
        
        return F.to_tensor(img), {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }


def make_model(num_classes: int = 2) -> FasterRCNN:
    """Create Faster R-CNN model with ResNet50-FPN backbone."""
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Evaluate model on validation set."""
    model.train()  # Need train mode to compute losses
    losses = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        if isinstance(loss_dict, dict):
            losses.append(sum(loss_dict.values()).item())
        else:
            # Fallback if model returns something unexpected
            losses.append(0.0)
    model.eval()  # Set back to eval mode
    return sum(losses) / len(losses) if losses else 0.0


def train_chunk(
    rows_chunk: List[Row],
    model,
    optimizer,
    device,
    epochs: int,
    batch_size: int,
    val_split: float,
    num_workers: int,
    chunk_num: int,
    total_chunks: int,
    gradient_accumulation: int = 1,
):
    """Train on a chunk of data with memory optimization."""
    dataset = SignatureDetectionDataset(rows_chunk)
    
    # Train/val split
    val_len = max(1, int(len(dataset) * val_split))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_len, val_len])
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
    
    # Memory-optimized DataLoader settings
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(num_workers, 2),  # Limit workers to save memory
        collate_fn=collate_fn,
        pin_memory=False,  # Disable pin_memory to save GPU memory
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=max(1, batch_size // 2),  # Smaller batch for validation
        shuffle=False,
        num_workers=0,  # No workers for validation
        collate_fn=collate_fn,
        pin_memory=False,
    )

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Chunk {chunk_num}/{total_chunks} - Epoch {epoch}/{epochs}")
        accumulated_loss = 0.0
        step_count = 0
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = [img.to(device, non_blocking=False) for img in images]
            targets = [{k: v.to(device, non_blocking=False) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) / gradient_accumulation  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item() * gradient_accumulation
            step_count += 1
            
            # Update weights every gradient_accumulation steps
            if (batch_idx + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                optimizer.zero_grad()
                
                pbar.set_postfix(loss=f"{accumulated_loss/step_count:.4f}")
        
        # Final update if needed
        if step_count % gradient_accumulation != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Clear cache before validation
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        val_loss = evaluate(model, val_loader, device)
        print(f"Chunk {chunk_num}/{total_chunks} - Epoch {epoch}: val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
    
    return best_val


def train(
    labels_csv: str,
    document_dir: str | None,
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    num_workers: int,
    chunk_size: int = 1000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all data
    rows = read_rows(labels_csv, document_root=document_dir)
    print(f"Total samples: {len(rows)}")
    print(f"Training in chunks of {chunk_size} samples\n")

    # Split into chunks
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    total_chunks = len(chunks)
    print(f"Split into {total_chunks} chunks\n")

    # Initialize model
    model = make_model(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Calculate optimal gradient accumulation based on batch size
    # For small GPUs, use gradient accumulation to simulate larger batches
    effective_batch_size = batch_size
    if device.type == "cuda":
        # Auto-adjust for memory constraints
        if batch_size > 2:
            gradient_accumulation = max(1, batch_size // 2)
            effective_batch_size = batch_size
        else:
            gradient_accumulation = 1
    else:
        gradient_accumulation = 1
    
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation}")
    print(f"Effective batch size: {effective_batch_size * gradient_accumulation}\n")

    # Train on each chunk incrementally
    best_val = float("inf")
    for chunk_num, rows_chunk in enumerate(chunks, 1):
        print(f"\n{'='*60}")
        print(f"Training on chunk {chunk_num}/{total_chunks} ({len(rows_chunk)} samples)")
        print(f"{'='*60}\n")
        
        # Clear cache before each chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        val_loss = train_chunk(
            rows_chunk, model, optimizer, device,
            epochs, batch_size, val_split, num_workers,
            chunk_num, total_chunks, gradient_accumulation
        )
        
        # Save model after each chunk
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "num_classes": 2,
            "chunk": chunk_num,
            "total_chunks": total_chunks,
            "val_loss": val_loss,
        }, out_path)
        print(f"✓ Saved model after chunk {chunk_num} (val_loss={val_loss:.4f}) -> {out_path}")
        
        # Clear cache after saving
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        if val_loss < best_val:
            best_val = val_loss
    
    print(f"\n{'='*60}")
    print(f"Training complete! Processed {len(rows)} samples in {total_chunks} chunks")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Final model saved to: {out_path}")
    print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser(description="Train CNN signature detector")
    ap.add_argument("--labels_csv", default="/home/ram-sthapit/programming/Signature Verification/export_images/train/labels.csv")
    ap.add_argument("--document_dir", default="/home/ram-sthapit/programming/Signature Verification/export_images/train/document")
    ap.add_argument("--out", default="/home/ram-sthapit/programming/Signature Verification/signature_detector.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size (use 1 for small GPUs)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0-2 recommended for small GPUs)")
    ap.add_argument("--chunk_size", type=int, default=1000, help="Number of samples to train on at a time")
    args, _ = ap.parse_known_args()

    train(
        labels_csv=args.labels_csv,
        document_dir=args.document_dir,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
