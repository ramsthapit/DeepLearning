"""
OCR → Text dataset builder for signature presence classification.

This module reads your existing `export_images/train/labels.csv`, runs OCR on the
document images, and writes a JSONL file that can be used to train a text model
(BERT/DistilBERT/BART-for-classification).

It intentionally does NOT require pytesseract; it shells out to the system
`tesseract` binary (which is already present on many Linux installs).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm


SIGNATURE_KEYWORDS = (
    "signature",
    "sign",
    "signed",
    "signatory",
    "signature:",
    "sign:",
    "authorized signature",
    "authorised signature",
    "signed by",
)


def _normalize_text(t: str) -> str:
    t = t.replace("\x0c", " ")  # tesseract page break
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _keyword_features(text: str) -> Dict[str, int]:
    low = text.lower()
    feats: Dict[str, int] = {}
    for kw in SIGNATURE_KEYWORDS:
        feats[f"kw:{kw}"] = low.count(kw)
    feats["kw_any"] = int(any(v > 0 for v in feats.values()))
    feats["kw_total"] = int(sum(v for v in feats.values()))
    return feats


def _stable_cache_key(image_path: str) -> str:
    # Cache key based on (path, mtime, size). Fast and stable across runs.
    p = Path(image_path)
    st = p.stat()
    h = hashlib.sha1()
    h.update(str(p.resolve()).encode("utf-8"))
    h.update(str(int(st.st_mtime_ns)).encode("utf-8"))
    h.update(str(int(st.st_size)).encode("utf-8"))
    return h.hexdigest()


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Lightweight preprocessing: grayscale + autocontrast.
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    return g


def run_tesseract_ocr(
    image_path: str,
    *,
    lang: str = "eng",
    psm: int = 6,
    oem: int = 3,
    dpi: int = 300,
) -> str:
    """
    OCR using system `tesseract` CLI:
      tesseract <image> stdout -l <lang> --psm <psm> --oem <oem> --dpi <dpi>
    """
    cmd = [
        "tesseract",
        image_path,
        "stdout",
        "-l",
        lang,
        "--psm",
        str(psm),
        "--oem",
        str(oem),
        "--dpi",
        str(dpi),
    ]
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or "").strip()
        raise RuntimeError(f"tesseract failed (code={p.returncode}) for {image_path}: {msg}")
    return _normalize_text(p.stdout or "")


def extract_text_with_cache(
    image_path: str,
    *,
    cache_dir: str,
    lang: str = "eng",
    psm: int = 6,
    oem: int = 3,
    dpi: int = 300,
) -> str:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    key = _stable_cache_key(image_path)
    cache_path = Path(cache_dir) / f"{key}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    # Preprocess via PIL and OCR the temp image to stabilize results across formats.
    img = Image.open(image_path).convert("RGB")
    pre = _preprocess_for_ocr(img)
    tmp_path = Path(cache_dir) / f"{key}.png"
    pre.save(tmp_path)

    text = run_tesseract_ocr(str(tmp_path), lang=lang, psm=psm, oem=oem, dpi=dpi)
    cache_path.write_text(text, encoding="utf-8")

    # Clean up the temp png (keep only .txt).
    try:
        tmp_path.unlink(missing_ok=True)  # py3.8+: missing_ok
    except TypeError:
        if tmp_path.exists():
            tmp_path.unlink()

    return text


@dataclass(frozen=True)
class CsvRow:
    id: str
    label: int
    document_path: str


def read_label_rows(labels_csv: str, *, document_root: Optional[str] = None) -> List[CsvRow]:
    rows: List[CsvRow] = []
    labels_dir = Path(labels_csv).resolve().parent
    with open(labels_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            doc_path = r["document_path"]
            if not os.path.isabs(doc_path):
                p1 = (labels_dir / doc_path).resolve()
                if p1.exists():
                    doc_path = str(p1)
                elif document_root:
                    doc_path = str((Path(document_root) / Path(doc_path).name).resolve())
            rows.append(CsvRow(id=r["id"], label=int(r["label"]), document_path=doc_path))
    return rows


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_ocr_jsonl(
    *,
    labels_csv: str,
    document_root: Optional[str],
    out_jsonl: str,
    cache_dir: str,
    lang: str,
    psm: int,
    oem: int,
    dpi: int,
    max_samples: int,
) -> None:
    rows = read_label_rows(labels_csv, document_root=document_root)
    if max_samples > 0:
        rows = rows[:max_samples]

    def gen() -> Iterator[Dict]:
        for row in tqdm(rows, desc="OCR", unit="img"):
            text = extract_text_with_cache(
                row.document_path,
                cache_dir=cache_dir,
                lang=lang,
                psm=psm,
                oem=oem,
                dpi=dpi,
            )
            feats = _keyword_features(text)
            yield {
                "id": row.id,
                "label": int(row.label),
                "image_path": row.document_path,
                "text": text,
                "features": feats,
            }

    write_jsonl(out_jsonl, gen())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labels_csv",
        default="/home/ram-sthapit/programming/Signature Verification/export_images/train2/dataset/labels.csv",
        help="Path to labels.csv (expects columns: id,label,document_path,...)",
    )
    ap.add_argument(
        "--document_root",
        default="/home/ram-sthapit/programming/Signature Verification/export_images/train2/dataset/document",
        help="Optional root dir for document images (used if document_path is not resolvable).",
    )
    ap.add_argument(
        "--out_jsonl",
        default="/home/ram-sthapit/programming/Signature Verification/export_images/train2/ocr_texts.jsonl",
        help="Output JSONL with OCR text + label.",
    )
    ap.add_argument(
        "--cache_dir",
        default="/home/ram-sthapit/programming/Signature Verification/.cache/ocr",
        help="Cache directory to avoid re-running OCR for the same image.",
    )
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--oem", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max_samples", type=int, default=500, help="<=0 means no limit")
    args = ap.parse_args()

    build_ocr_jsonl(
        labels_csv=args.labels_csv,
        document_root=args.document_root,
        out_jsonl=args.out_jsonl,
        cache_dir=args.cache_dir,
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        dpi=args.dpi,
        max_samples=args.max_samples,
    )
    print("wrote:", args.out_jsonl)


if __name__ == "__main__":
    main()


