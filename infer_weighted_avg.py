"""
Simple inference using weighted average of CNN detector + BERT text model.
No fusion model needed - just combine scores with weights.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from infer_signature_detector import load_model as load_detector, predict as detector_predict
from ocr_text_dataset import extract_text_with_cache


def get_cnn_score(detector, image_path: str, device: torch.device, score_thresh: float = 0.05) -> float:
    """Get max detection score from CNN detector"""
    with torch.no_grad():
        _annot, preds = detector_predict(detector, image_path, device, score_thresh=score_thresh)
        if not preds:
            return 0.0
        scores = [p[-1] for p in preds]
        return float(max(scores))


def get_bert_prob(text_model, tokenizer, image_path: str, device: torch.device, 
                  cache_dir: str, max_length: int = 256) -> float:
    """Get probability from BERT text classifier"""
    with torch.no_grad():
        txt = extract_text_with_cache(image_path, cache_dir=cache_dir)
        enc = tokenizer([txt], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = text_model(**enc)
        probs = torch.softmax(out.logits[0], dim=-1)
        return float(probs[1].item())  # Probability of "signed" class


def main():
    ap = argparse.ArgumentParser(description="Inference using weighted average of CNN + BERT")
    ap.add_argument(
        "--detector_ckpt",
        default="/home/ram-sthapit/programming/Signature Verification/signature_detector.pt",
        help="Path to trained CNN detector checkpoint",
    )
    ap.add_argument(
        "--text_model_dir",
        default="/home/ram-sthapit/programming/Signature Verification/text_model_final",
        help="Directory containing trained BERT text model",
    )
    ap.add_argument(
        "--image",
        required=True,
        help="Path to document image (png/jpg) to test",
    )
    ap.add_argument(
        "--cache_dir",
        default="/home/ram-sthapit/programming/Signature Verification/.cache/ocr",
        help="OCR cache directory",
    )
    ap.add_argument(
        "--score_thresh",
        type=float,
        default=0.05,
        help="Detector score threshold",
    )
    ap.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max token length for BERT",
    )
    ap.add_argument(
        "--cnn_weight",
        type=float,
        default=0.7,
        help="Weight for CNN detector (default: 0.7 = 70%%)",
    )
    ap.add_argument(
        "--bert_weight",
        type=float,
        default=0.3,
        help="Weight for BERT text model (default: 0.3 = 30%%)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for final prediction (default: 0.5)",
    )
    args = ap.parse_args()

    # Validate weights sum to 1.0
    total_weight = args.cnn_weight + args.bert_weight
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: Weights sum to {total_weight:.3f}, normalizing to 1.0")
        args.cnn_weight /= total_weight
        args.bert_weight /= total_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load CNN detector
    print(f"Loading CNN detector from {args.detector_ckpt}...")
    detector = load_detector(args.detector_ckpt, device)
    detector.eval()

    # Load BERT text model
    print(f"Loading BERT text model from {args.text_model_dir}...")
    text_model_path = Path(args.text_model_dir).resolve()
    if not text_model_path.exists():
        raise FileNotFoundError(
            f"Text model directory not found: {text_model_path}\n"
            f"Please train a text model first using:\n"
            f"  python ocr_to_bert_simple.py --out_dir {text_model_path}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(text_model_path), use_fast=True)
    text_model = AutoModelForSequenceClassification.from_pretrained(str(text_model_path)).to(device).eval()

    print(f"\n{'='*60}")
    print(f"Processing image: {args.image}")
    print(f"{'='*60}\n")

    # Get scores from both models
    print("Running models...")
    cnn_score = get_cnn_score(detector, args.image, device, args.score_thresh)
    bert_prob = get_bert_prob(text_model, tokenizer, args.image, device, args.cache_dir, args.max_length)

    # Weighted average
    final_score = args.cnn_weight * cnn_score + args.bert_weight * bert_prob
    prediction = "signed" if final_score >= args.threshold else "not_signed"

    # Display results
    print(f"{'='*60}")
    print("Model Scores:")
    print(f"{'='*60}")
    print(f"  CNN Detector Score:  {cnn_score:.4f}  (weight: {args.cnn_weight*100:.0f}%)")
    print(f"  BERT Text Prob:      {bert_prob:.4f}  (weight: {args.bert_weight*100:.0f}%)")
    print(f"\n  Weighted Average:    {final_score:.4f}")
    print(f"  Threshold:           {args.threshold:.4f}")
    print(f"\n{'='*60}")
    print(f"✓ Final Prediction: {prediction}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

