"""
Simple inference script for BERT text model.

Tests the trained BERT text classifier on a single document image.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ocr_text_dataset import extract_text_with_cache
from ocr_to_bert_simple import predict_image_class, CLASS_NAMES


def main():
    ap = argparse.ArgumentParser(description="Test BERT text model on a document image")
    ap.add_argument(
        "--model_dir",
        default="/home/ram-sthapit/programming/Signature Verification/text_model",
        help="Directory containing trained BERT model (from ocr_to_bert_simple.py)",
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
        "--max_length",
        type=int,
        default=256,
        help="Max token length for BERT",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model_path = Path(args.model_dir).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please train a text model first using:\n"
            f"  python ocr_to_bert_simple.py --out_dir {model_path}"
        )

    print(f"Loading BERT model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device).eval()

    # Predict
    print(f"\nProcessing image: {args.image}")
    predicted_class = predict_image_class(
        args.image,
        model,
        tokenizer,
        CLASS_NAMES,
        device=device,
        cache_dir=args.cache_dir,
        max_length=args.max_length,
    )

    print(f"\n✓ Final prediction: {predicted_class}")


if __name__ == "__main__":
    main()

