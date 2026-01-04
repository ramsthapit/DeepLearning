import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    bboxes: List[List[float]]  # normalized [x1,y1,x2,y2]


def read_rows(labels_csv: str, document_root: str | None = None) -> List[Row]:
    rows: List[Row] = []
    labels_dir = Path(labels_csv).resolve().parent
    with open(labels_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_path = r["document_path"]
            # Resolve relative paths in a robust way:
            # 1) absolute paths as-is
            # 2) relative to the labels.csv directory (common for exports)
            # 3) if a document_root is provided, fallback to document_root/<filename>
            if not os.path.isabs(img_path):
                p1 = (labels_dir / img_path).resolve()
                if p1.exists():
                    img_path = str(p1)
                elif document_root:
                    img_path = str((Path(document_root) / Path(img_path).name).resolve())
            bboxes = json.loads(r["bbox_json"]) if r.get("bbox_json") else []
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

        # Convert to absolute pixel coords (clamped)
        boxes_abs: List[List[float]] = []
        for b in row.bboxes:
            if not (isinstance(b, list) and len(b) == 4):
                continue
            x1, y1, x2, y2 = b
            x1 = max(0.0, min(1.0, float(x1)))
            y1 = max(0.0, min(1.0, float(y1)))
            x2 = max(0.0, min(1.0, float(x2)))
            y2 = max(0.0, min(1.0, float(y2)))
            # Ensure proper ordering
            x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
            # Convert to pixels
            boxes_abs.append([x_min * w, y_min * h, x_max * w, y_max * h])

        boxes = torch.tensor(boxes_abs, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # 1 = "signature"

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        img_t = F.to_tensor(img)  # float32 [0,1]
        return img_t, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_model(num_classes: int = 2) -> FasterRCNN:
    # CNN backbone (ResNet50-FPN) + detector head
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate_one_epoch(model, loader, device) -> float:
    model.eval()
    losses: List[float] = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values()).item()
        losses.append(loss)
    return float(sum(losses) / max(1, len(losses)))


def train(
    labels_csv: str,
    document_dir: str | None,
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    num_workers: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    rows = read_rows(labels_csv, document_root=document_dir)
    dataset = SignatureDetectionDataset(rows)

    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = make_model(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

        val_loss = evaluate_one_epoch(model, val_loader, device)
        print(f"epoch {epoch}: val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": 2,
                },
                out_path,
            )
            print(f"saved best model -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--document_dir",
        default="/home/ram-sthapit/programming/Signature Verification/export_images/train/document",
        help="Directory containing document images (png/jpg).",
    )
    ap.add_argument(
        "--labels_csv",
        default="/home/ram-sthapit/programming/Signature Verification/export_images/train/labels.csv",
        help="CSV with bbox_json and document_path.",
    )
    ap.add_argument(
        "--out",
        default="/home/ram-sthapit/programming/Signature Verification/signature_detector.pt",
        help="Where to save the trained model checkpoint.",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=2)
    # Jupyter/IPython sometimes injects extra args like `--f=...`; ignore unknown args.
    args, _unknown = ap.parse_known_args()

    train(
        labels_csv=args.labels_csv,
        document_dir=args.document_dir,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()


