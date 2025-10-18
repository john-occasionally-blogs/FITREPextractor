#!/usr/bin/env python3
"""
Marine FITREP PDF to CSV Extractor (robust OCC, Grade, and 8-box detection)

Usage:
    python fitrep_extractor_fixed.py --input <pdf_or_dir_or_list.txt> --output out.csv

Notes:
- This version fixes:
  * OCC not recognized (normalizes OCR confusions, uses label proximity on same row)
  * Grade incorrectly grabbing RS/RO grades (chooses the top-most 'Grade' on page 1)
  * 8-box rows defaulting to 4 (uses OpenCV to detect boxes and score diagonal 'X')
"""

import os
import re
import csv
import sys
import io
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# OCR / Imaging
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import cv2


# ----------------------------
# Configuration / Valid sets
# ----------------------------

VALID_GRADES = [
    "PVT","PFC","LCPL","CPL","SGT","SSGT","GYSGT","MGYSGT","1STSGT","SGTMAJ",
    "2NDLT","1STLT","CAPT","MAJ","LTCOL","COL",
    "WO","CWO2","CWO3","CWO4","CWO5",
    "BGEN","MAJGEN","LTGEN","GEN"
]

VALID_OCC_CODES = [
    "GC","DC","CH","TR","CD","TD","FD","EN","CS","AN","AR","SA","RT"
]


# ----------------------------
# Utility / OCR helpers
# ----------------------------

def pdf_page_to_image(pdf: fitz.Document, page_index: int, scale: float = 3.0) -> Image.Image:
    """Render a PDF page to a PIL Image at high DPI for better OCR."""
    page = pdf[page_index]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))


def get_ocr(img: Image.Image, psm: int = 6):
    """Return (plain_text, ocr_data dict) for an image using Tesseract."""
    cfg = f"--psm {psm}"
    text = pytesseract.image_to_string(img, config=cfg)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg)
    return text, data


def norm_token(s: str) -> str:
    """Normalize common OCR confusions and strip punctuation."""
    return (
        s.strip().upper()
         .replace("0", "O")
         .replace("1", "I")
         .replace("5", "S")
         .replace(":", "")
         .replace(".", "")
    )


def nearest_right_token(ocr, i: int, max_dx: int = 450, max_dy: int = 35) -> Optional[int]:
    """From token i, find nearest token to the right within a band."""
    base_y = ocr["top"][i]
    base_x = ocr["left"][i] + ocr["width"][i]
    best = None
    best_dx = 10**9
    for j, t in enumerate(ocr["text"]):
        if not t:
            continue
        x = ocr["left"][j]
        y = ocr["top"][j]
        if y < base_y - max_dy or y > base_y + max_dy:
            continue
        dx = x - base_x
        if dx < 0 or dx > max_dx:
            continue
        if dx < best_dx:
            best_dx = dx
            best = j
    return best


def find_label_indices(ocr, labels: List[str]) -> List[int]:
    labset = {norm_token(l) for l in labels}
    idxs = []
    for i, t in enumerate(ocr["text"]):
        if not t:
            continue
        if norm_token(t) in labset:
            idxs.append(i)
    return idxs


# ----------------------------
# Field Extractors
# ----------------------------

def extract_last_name_from_page1_text(text: str) -> Optional[str]:
    """
    Very simple last name pullers—kept broad due to layout variance.
    You can refine to your form template if needed.
    """
    patterns = [
        r'Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
        r'a\.\s*Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
        r'Last Name.*?\n\s*([A-Z]+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # fallback: the first ALLCAPS token of length >= 2 near "Last Name"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "LAST" in line.upper() and "NAME" in line.upper():
            for j in range(0, min(3, len(lines) - i - 1)):
                caps = re.findall(r"\b[A-Z]{2,}\b", lines[i + 1 + j])
                if caps:
                    return caps[0]
    return None


def extract_grade_page1(img: Image.Image, valid_grades: List[str]) -> Optional[str]:
    """
    Robust Grade extractor:
    - Find all 'GRADE' tokens in the top third of page 1
    - Choose the top-most occurrence (smallest y) to target MRO's grade
    - Read rightward (tight x/y band) to get the nearest valid rank token
    """
    text, ocr = get_ocr(img, psm=6)
    h = img.height
    top_third = h // 3

    # All Grade tokens in top third
    grade_idxs = [
        i for i, t in enumerate(ocr["text"])
        if t and "GRADE" in t.upper() and ocr["top"][i] < top_third
    ]
    if not grade_idxs:
        # Regex fallback in top third chunk
        lines = text.splitlines()
        top_lines = lines[: max(1, len(lines)//3)]
        tt = "\n".join(top_lines).upper()
        for g in valid_grades:
            if re.search(rf"\b{re.escape(g)}\b", tt):
                return g
        return None

    grade_idxs.sort(key=lambda i: ocr["top"][i])  # smallest y (top-most)
    candidate = grade_idxs[0]

    # Look right within tight window and gather a small token window
    j = nearest_right_token(ocr, candidate, max_dx=450, max_dy=30)
    window = []
    if j is not None:
        for k in range(j, min(j + 10, len(ocr["text"]))):
            if abs(ocr["top"][k] - ocr["top"][candidate]) > 30:
                break
            tok = norm_token(ocr["text"][k])
            if tok:
                window.append(tok)

    # direct token match
    for tok in window:
        if tok in valid_grades:
            return tok
    # joined pieces like "1ST" + "LT"
    joined = "".join(window)
    for g in valid_grades:
        if g in joined:
            return g

    # last-chance fallback: search top third raw text
    lines = text.splitlines()
    tt = "\n".join(lines[: max(1, len(lines)//3)]).upper()
    for g in valid_grades:
        if re.search(rf"\b{re.escape(g)}\b", tt):
            return g
    return None


def extract_occ_page1(img: Image.Image, valid_occ_codes: List[str]) -> Optional[str]:
    """
    OCC extractor:
    - Locate label variants (OCC/Occasion) on page 1 and search rightward on the same row
    - Normalize OCR confusions (0->O, etc.)
    - Fallback: search top half for isolated 2-letter codes from the valid list
    """
    text, ocr = get_ocr(img, psm=6)
    labels = ["OCC", "OCC.", "OCC:", "OCCASION", "OCCASION:", "OCCASION.", "Occasion"]
    idxs = find_label_indices(ocr, labels)
    idxs.sort(key=lambda i: (ocr["top"][i], ocr["left"][i]))  # top-left preference

    for idx in idxs:
        row_y = ocr["top"][idx]
        # scan a limited number of tokens to the right on same row band
        for k in range(idx + 1, min(idx + 20, len(ocr["text"]))):
            if abs(ocr["top"][k] - row_y) > 30:
                break
            tok = norm_token(ocr["text"][k])
            if len(tok) == 2 and tok in valid_occ_codes:
                return tok
            if len(tok) == 2:
                corr = tok.replace("0", "O").replace("1", "I").replace("5", "S")
                if corr in valid_occ_codes:
                    return corr

    # Fallback: search top half of page 1 for any valid 2-letter code
    half = img.height // 2
    for i, t in enumerate(ocr["text"]):
        if not t:
            continue
        if ocr["top"][i] > half:
            continue
        tok = norm_token(t)
        if len(tok) == 2 and tok in valid_occ_codes:
            return tok

    return None


def extract_8box_rows(img: Image.Image, expected_count: int) -> List[int]:
    """
    Detect 8-box rows (A..H) and pick which box has the 'X' by diagonal energy.

    Returns: list of length expected_count, values in 1..8
    """
    # PIL -> cv2
    cv_img = np.array(img)
    if cv_img.ndim == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img

    # Adaptive binarization for varied scans
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 35, 9)

    # Slight close to strengthen edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = gray.shape[:2]
    min_area = (w_img * h_img) * 0.00002
    max_area = (w_img * h_img) * 0.01

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar = w / float(h)
        if area < min_area or area > max_area:
            continue
        if 0.75 < ar < 1.33:
            # check inner brightness to reject filled blobs
            pad = max(1, min(w, h) // 10)
            inner = thr[y + pad:y + h - pad, x + pad:x + w - pad]
            if inner.size == 0:
                continue
            if inner.mean() < 50:
                # likely a filled region (text blob), not a hollow checkbox
                continue
            boxes.append((x, y, w, h))

    if not boxes:
        return [4] * expected_count  # neutral fallback

    # Group into rows by y proximity
    boxes.sort(key=lambda b: b[1])  # by y
    rows = []
    current = [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - current[-1][1]) < 18:  # same row if y close
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)

    # Keep rows that plausibly contain 8 boxes; pick the leftmost 8 if >8
    cleaned_rows = []
    for r in rows:
        r_sorted = sorted(r, key=lambda b: b[0])  # by x
        if len(r_sorted) >= 7:
            cleaned_rows.append(r_sorted[:8])

    cleaned_rows.sort(key=lambda row: sum(b[1] for b in row) / len(row))  # top -> bottom

    values = []
    for row in cleaned_rows[:expected_count]:
        row = sorted(row, key=lambda b: b[0])
        # ensure exactly 8
        if len(row) < 8:
            row = (row + [row[-1]] * 8)[:8]
        elif len(row) > 8:
            row = row[:8]

        scores = []
        for (x, y, w, h) in row:
            # crop to inner ROI
            ix = x + w // 6
            iw = max(3, w - 2 * (w // 6))
            iy = y + h // 6
            ih = max(3, h - 2 * (h // 6))
            roi = gray[iy:iy + ih, ix:ix + iw]
            roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Diagonal 'X' energy via simple diagonal kernels
            k = max(3, (min(roi_bin.shape[:2]) // 6) | 1)  # odd
            diag1 = np.eye(k, dtype=np.float32)
            diag2 = np.flipud(diag1)
            d1 = cv2.filter2D(roi_bin.astype(np.float32), -1, diag1)
            d2 = cv2.filter2D(roi_bin.astype(np.float32), -1, diag2)
            score = float(d1.mean() + d2.mean())
            scores.append(score)

        col = int(np.argmax(scores)) + 1  # 1..8
        values.append(col)

    # pad if fewer rows found
    while len(values) < expected_count:
        values.append(4)

    return values


# ----------------------------
# Core extraction per PDF
# ----------------------------

def extract_from_pdf(pdf_path: Path) -> Dict[str, object]:
    """
    Extracts fields from a single FITREP PDF.
    Returns a dict with keys:
      last_name, grade, occ,
      page2_values (list of 5), page3_values (list of 5), page4_values (list of 4)
    """
    data: Dict[str, object] = {
        "file": str(pdf_path),
        "last_name": None,
        "grade": None,
        "occ": None,
        "page2_values": [4]*5,
        "page3_values": [4]*5,
        "page4_values": [4]*4,
    }

    with fitz.open(pdf_path) as doc:
        # Page 1
        if len(doc) >= 1:
            img1 = pdf_page_to_image(doc, 0, scale=3.0)
            text1, _ = get_ocr(img1, psm=6)

            # Last name (simple patterns; adjust if you have a stable template)
            ln = extract_last_name_from_page1_text(text1)
            if ln:
                data["last_name"] = ln

            # Grade (top-most 'GRADE' row)
            grade = extract_grade_page1(img1, VALID_GRADES)
            if grade:
                data["grade"] = grade

            # OCC code
            occ = extract_occ_page1(img1, VALID_OCC_CODES)
            if occ:
                data["occ"] = occ

        # Page 2: 5 rows of 8
        if len(doc) >= 2:
            img2 = pdf_page_to_image(doc, 1, scale=3.0)
            data["page2_values"] = extract_8box_rows(img2, expected_count=5)

        # Page 3: 5 rows of 8
        if len(doc) >= 3:
            img3 = pdf_page_to_image(doc, 2, scale=3.0)
            data["page3_values"] = extract_8box_rows(img3, expected_count=5)

        # Page 4: 4 rows of 8
        if len(doc) >= 4:
            img4 = pdf_page_to_image(doc, 3, scale=3.0)
            data["page4_values"] = extract_8box_rows(img4, expected_count=4)

    return data


# ----------------------------
# I/O / CLI
# ----------------------------

def discover_pdfs(input_path: Path) -> List[Path]:
    """
    Accepts:
      - PDF file
      - Directory (recursively finds *.pdf)
      - Text file with one PDF path per line
    Returns list of PDF paths.
    """
    pdfs: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            pdfs = [input_path]
        elif input_path.suffix.lower() in [".txt", ".lst"]:
            for line in input_path.read_text().splitlines():
                p = Path(line.strip())
                if p.suffix.lower() == ".pdf" and p.exists():
                    pdfs.append(p)
        else:
            raise ValueError(f"Unsupported file type: {input_path}")
    elif input_path.is_dir():
        pdfs = sorted([p for p in input_path.rglob("*.pdf")])
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")
    return pdfs


def write_csv(out_path: Path, records: List[Dict[str, object]]) -> None:
    # Flatten page values into separate columns
    fieldnames = [
        "file", "last_name", "grade", "occ",
        *(f"p2_{i+1}" for i in range(5)),
        *(f"p3_{i+1}" for i in range(5)),
        *(f"p4_{i+1}" for i in range(4)),
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            row = {
                "file": rec.get("file"),
                "last_name": rec.get("last_name"),
                "grade": rec.get("grade"),
                "occ": rec.get("occ"),
            }
            for i, v in enumerate(rec.get("page2_values", [])):
                row[f"p2_{i+1}"] = v
            for i, v in enumerate(rec.get("page3_values", [])):
                row[f"p3_{i+1}"] = v
            for i, v in enumerate(rec.get("page4_values", [])):
                row[f"p4_{i+1}"] = v
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Marine FITREP PDF to CSV Extractor (robust)")
    ap.add_argument("--input", required=True, help="PDF, directory of PDFs, or text file of PDF paths")
    ap.add_argument("--output", required=True, help="Output CSV path")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    pdfs = discover_pdfs(in_path)

    if not pdfs:
        print("No PDFs found. Nothing to do.")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s). Extracting...")
    records: List[Dict[str, object]] = []
    for i, pdf in enumerate(pdfs, 1):
        try:
            rec = extract_from_pdf(pdf)
            records.append(rec)
            print(f"[{i}/{len(pdfs)}] OK: {pdf.name}  Grade={rec.get('grade')}  OCC={rec.get('occ')}")
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] ERROR: {pdf} -> {e}")

    write_csv(out_path, records)
    print(f"\nDone. Wrote: {out_path}")


if __name__ == "__main__":
    main()

