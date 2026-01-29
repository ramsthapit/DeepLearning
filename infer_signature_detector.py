import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms import functional as F


MODEL_BUILDERS = {
    "fasterrcnn_resnet50_fpn": lambda: fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None),
    "fasterrcnn_mobilenet_v3_large_320_fpn": lambda: fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None, weights_backbone=None
    ),
}


def _guess_model_type_from_state_dict(model_state: Dict[str, torch.Tensor]) -> str | None:
    keys = model_state.keys()
    # ResNet backbone has conv1/bn1/layer{1..} keys.
    if "backbone.body.conv1.weight" in keys or any(k.startswith("backbone.body.layer1.") for k in keys):
        return "fasterrcnn_resnet50_fpn"
    # MobileNetV3 backbone tends to have backbone.body.0.*, backbone.body.1.*, etc.
    if any(k.startswith("backbone.body.0.") for k in keys) or any(k.startswith("backbone.body.1.") for k in keys):
        return "fasterrcnn_mobilenet_v3_large_320_fpn"
    return None


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = int(ckpt.get("num_classes", 2))

    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError("Checkpoint missing 'model_state' dict.")

    model_type = ckpt.get("model_type") or ckpt.get("arch") or _guess_model_type_from_state_dict(model_state)
    if model_type not in MODEL_BUILDERS:
        raise RuntimeError(
            f"Unsupported/unknown model_type {model_type!r}. Supported: {sorted(MODEL_BUILDERS.keys())}"
        )

    model = MODEL_BUILDERS[model_type]()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(
    model,
    image_path: str,
    device: torch.device,
    score_thresh: float,
) -> Tuple[Image.Image, List[Tuple[float, float, float, float, float]]]:
    img = Image.open(image_path).convert("RGB")
    x = F.to_tensor(img).to(device)
    # Torchvision detectors apply their own score threshold internally (roi_heads.score_thresh).
    # If we want to see/draw lower-confidence boxes, we must lower that too.
    prev_thresh = getattr(model.roi_heads, "score_thresh", None)
    if prev_thresh is not None and score_thresh < float(prev_thresh):
        model.roi_heads.score_thresh = float(score_thresh)
    out = model([x])[0]
    if prev_thresh is not None:
        model.roi_heads.score_thresh = prev_thresh

    boxes = out["boxes"].cpu().tolist()
    scores = out["scores"].cpu().tolist()
    keep = [(b, s) for b, s in zip(boxes, scores) if s >= score_thresh]

    draw = ImageDraw.Draw(img)
    preds: List[Tuple[float, float, float, float, float]] = []
    for (x1, y1, x2, y2), s in keep:
        preds.append((x1, y1, x2, y2, s))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), f"{s:.2f}", fill="red")
    return img, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="/home/ram-sthapit/programming/Signature Verification/signature_detector.pt",
        help="Path to trained checkpoint from train_signature_detector.py",
    )
    ap.add_argument("--image", required=True, help="Path to a document image (png/jpg).")
    ap.add_argument(
        "--out",
        default=None,
        help="Optional: specific output file path. If not provided, uses --out_dir.",
    )
    ap.add_argument(
        "--out_dir",
        default="/home/ram-sthapit/programming/Signature Verification/export_images/predict",
        help="Directory to save annotated images. Default: export_images/predict/",
    )
    ap.add_argument("--score", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    annotated, preds = predict(model, args.image, device, score_thresh=args.score)
    for x1, y1, x2, y2, s in preds:
        print(f"box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) score={s:.3f}")

    if args.out:
        out_path = args.out
    else:
        predict_dir = Path(args.out_dir)
        predict_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(predict_dir / (Path(args.image).stem + "_pred.png"))
    annotated.save(out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main()


