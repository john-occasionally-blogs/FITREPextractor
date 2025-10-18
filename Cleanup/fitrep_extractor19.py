#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import sys
import io
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Make stdout more eager when possible (3.7-safe)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# OCR / Imaging
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import cv2


# =========================
# Valid sets / configuration
# =========================
VALID_GRADES = [
    "PVT","PFC","LCPL","CPL","SGT","SSGT","GYSGT","MGYSGT","1STSGT","SGTMAJ",
    "2NDLT","1STLT","CAPT","MAJ","LTCOL","COL",
    "WO","CWO2","CWO3","CWO4","CWO5",
    "BGEN","MAJGEN","LTGEN","GEN"
]
VALID_OCC_CODES = ["GC","DC","CH","TR","CD","TD","FD","EN","CS","AN","AR","SA","RT"]


# =========================
# Helpers
# =========================
def pdf_page_to_image(pdf: "fitz.Document", page_index: int, scale: float = 3.0) -> Image.Image:
    page = pdf[page_index]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

def get_ocr(img: Image.Image, psm: int = 6):
    cfg = f"--psm {psm}"
    text = pytesseract.image_to_string(img, config=cfg)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg)
    return text, data

def norm_token(s: str) -> str:
    return (
        s.strip().upper()
         .replace("0", "O")
         .replace("1", "I")
         .replace("5", "S")
         .replace(":", "")
         .replace(".", "")
    )

def nearest_right_token(ocr, i: int, max_dx: int = 450, max_dy: int = 35) -> Optional[int]:
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


# =========================
# Field extractors (Page 1)
# =========================
def extract_last_name_page1_STRICT(text1: str) -> Optional[str]:
    """
    EXACTLY the original last-name logic (no extra fallbacks),
    because you said it was previously 100% for you.  :contentReference[oaicite:3]{index=3}
    """
    patterns = [
        r'Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
        r'a\.\s*Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
        r'Last Name.*?\n\s*([A-Z]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text1, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def extract_grade_page1(img: Image.Image) -> Optional[str]:
    """
    Robust Grade (keeps improved logic; picks top-most 'Grade' and reads right). 
    """
    text, ocr = get_ocr(img, psm=6)
    h = img.height
    top_third = h // 3

    grade_idxs = [
        i for i, t in enumerate(ocr["text"])
        if t and "GRADE" in t.upper() and ocr["top"][i] < top_third
    ]
    if not grade_idxs:
        # Regex fallback in top third text
        lines = text.splitlines()
        top_lines = lines[: max(1, len(lines)//3)]
        tt = "\n".join(top_lines).upper()
        for g in VALID_GRADES:
            if re.search(rf"\b{re.escape(g)}\b", tt):
                return g
        return None

    grade_idxs.sort(key=lambda i: ocr["top"][i])
    candidate = grade_idxs[0]
    j = nearest_right_token(ocr, candidate, max_dx=450, max_dy=30)

    window = []
    if j is not None:
        for k in range(j, min(j + 12, len(ocr["text"]))):
            if abs(ocr["top"][k] - ocr["top"][candidate]) > 30:
                break
            tok = norm_token(ocr["text"][k])
            if tok:
                window.append(tok)

    for tok in window:
        if tok in VALID_GRADES:
            return tok
    joined = "".join(window)
    for g in VALID_GRADES:
        if g in joined:
            return g

    # Last-chance fallback
    lines = text.splitlines()
    top_lines = lines[: max(1, len(lines)//3)]
    tt = "\n".join(top_lines).upper()
    for g in VALID_GRADES:
        if re.search(rf"\b{re.escape(g)}\b", tt):
            return g
    return None


def extract_occ_page1(img: Image.Image) -> Optional[str]:
    """
    Robust OCC (keeps improved logic; label-nearby and normalized tokens).
    """
    text, ocr = get_ocr(img, psm=6)
    labels = ["OCC","OCC.","OCC:","OCCASION","OCCASION:","OCCASION.","Occasion"]
    idxs = find_label_indices(ocr, labels)
    idxs.sort(key=lambda i: (ocr["top"][i], ocr["left"][i]))

    for idx in idxs:
        row_y = ocr["top"][idx]
        for k in range(idx + 1, min(idx + 20, len(ocr["text"]))):
            if abs(ocr["top"][k] - row_y) > 30:
                break
            tok = norm_token(ocr["text"][k])
            if len(tok) == 2 and tok in VALID_OCC_CODES:
                return tok
            if len(tok) == 2:
                corr = tok.replace("0","O").replace("1","I").replace("5","S")
                if corr in VALID_OCC_CODES:
                    return corr

    # Fallback: scan top half for a valid 2-letter code
    half = img.height // 2
    for i, t in enumerate(ocr["text"]):
        if not t:
            continue
        if ocr["top"][i] > half:
            continue
        tok = norm_token(t)
        if len(tok) == 2 and tok in VALID_OCC_CODES:
            return tok
    return None


def extract_to_date_page1(text1: str, ocr) -> Optional[str]:
    """
    EXACTLY the prior 'To date' approach you used (two-step).  :contentReference[oaicite:4]{index=4}
    - Method 1: find FROM ... TO in OCR stream and read the 8-digit number after TO
    - Method 2: regex for From:######## To:######## , else second 8-digit date in text
    """
    to_value = None

    # Method 1: walk OCR tokens to find FROM then TO, then an 8-digit date
    from_index = -1
    for i, word in enumerate(ocr['text']):
        if not word:
            continue
        w = word.upper()
        if w == 'FROM':
            from_index = i
        elif w == 'TO' and from_index != -1:
            # look ahead for 8-digit date
            for j in range(1, min(10, len(ocr['text']) - i)):
                next_word = ocr['text'][i + j].strip()
                if re.match(r'^\d{8}$', next_word):
                    to_value = next_word
                    break
            if to_value:
                break

    # Method 2: regex fallbacks in full page text
    if not to_value:
        m = re.search(r'From[:\s]*(\d{8})[:\s]*To[:\s]*(\d{8})', text1, re.IGNORECASE)
        if m:
            to_value = m.group(2)
        else:
            two_dates = re.findall(r'\d{8}', text1)
            if len(two_dates) >= 2:
                to_value = two_dates[1]

    return to_value


# =========================
# 8-box detection (Pages 2–4)
# =========================
def _extract_8box_rows_cv(img: Image.Image, expected_count: int) -> List[int]:
    """
    Pass 1: OpenCV shape + diagonal 'X' energy (as before, with a few robustness tweaks).
    """
    cv_img = np.array(img)
    if cv_img.ndim == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img

    # Normalize a bit to reduce lighting variance
    gray = cv2.equalizeHist(gray)

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = gray.shape[:2]
    min_area = (w_img * h_img) * 0.000015
    max_area = (w_img * h_img) * 0.02

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area or area > max_area:
            continue
        ar = w / float(h)
        if 0.7 < ar < 1.4:  # square-ish
            pad = max(1, min(w, h) // 8)
            inner = thr[y + pad:y + h - pad, x + pad:x + w - pad]
            if inner.size == 0:
                continue
            # inner should not be fully filled (when inverted threshold, empty centers are dark)
            if inner.mean() < 35:
                continue
            boxes.append((x, y, w, h))

    if not boxes:
        return [4] * expected_count

    # group by rows
    boxes.sort(key=lambda b: b[1])  # by y
    rows = []
    current = [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - current[-1][1]) < 22:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)

    # keep rows that plausibly contain >=7 boxes; enforce 8 by trimming/duplicating
    cleaned_rows = []
    for r in rows:
        r_sorted = sorted(r, key=lambda b: b[0])  # by x
        if len(r_sorted) >= 7:
            # Try to pick 8 most even by spacing: simple left-to-right slice after sort
            cleaned_rows.append(r_sorted[:8])

    cleaned_rows.sort(key=lambda row: sum(b[1] for b in row) / len(row))  # top→bottom

    # score X
    values = []
    for row in cleaned_rows[:expected_count]:
        row = sorted(row, key=lambda b: b[0])
        if len(row) < 8:
            row = (row + [row[-1]] * 8)[:8]
        elif len(row) > 8:
            row = row[:8]

        scores = []
        for (x, y, w, h) in row:
            ix = x + w // 6
            iw = max(5, w - 2 * (w // 6))
            iy = y + h // 6
            ih = max(5, h - 2 * (h // 6))
            roi = gray[iy:iy + ih, ix:ix + iw]
            roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # thin/skeletonize light pen marks
            roi_bin = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

            k = max(3, (min(roi_bin.shape[:2]) // 6) | 1)  # odd
            diag1 = np.eye(k, dtype=np.float32)
            diag2 = np.flipud(diag1)
            d1 = cv2.filter2D(roi_bin.astype(np.float32), -1, diag1)
            d2 = cv2.filter2D(roi_bin.astype(np.float32), -1, diag2)
            score = float(d1.mean() + d2.mean())

            # also add Hough line evidence for diagonals
            lines = cv2.HoughLinesP(roi_bin, 1, np.pi / 180, threshold=12, minLineLength=max(5, iw // 3), maxLineGap=3)
            if lines is not None:
                for l in lines[:8]:
                    x1, y1, x2, y2 = l[0]
                    slope = (y2 - y1) / (x2 - x1 + 1e-5)
                    if abs(abs(slope) - 1.0) < 0.5:  # roughly diagonal
                        score += 5.0

            scores.append(score)

        col = int(np.argmax(scores)) + 1
        values.append(col)

    # pad if not enough
    while len(values) < expected_count:
        values.append(4)

    return values


def _extract_8box_rows_ocr_headers(img: Image.Image, expected_count: int) -> List[int]:
    """
    Pass 2 (fallback): your previous OCR method that uses A–H header letters
    and finds an 'X' token below to decide the column.  :contentReference[oaicite:5]{index=5}
    """
    values: List[int] = []
    ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    # Find header letters A..H
    headers = {ch: [] for ch in list("ABCDEFGH")}
    for i, text in enumerate(ocr["text"]):
        t = (text or "").strip().upper()
        if len(t) == 1 and t in headers:
            try:
                conf = int(ocr["conf"][i])
            except Exception:
                conf = 0
            if conf > 20:
                headers[t].append({
                    "x": ocr["left"][i] + ocr["width"][i] // 2,
                    "y": ocr["top"][i],
                    "conf": conf
                })

    # group by y; choose the row with most unique letters
    y_groups: Dict[int, Dict[str, Dict[str, int]]] = {}
    for letter, positions in headers.items():
        for pos in positions:
            y_range = pos["y"] // 30 * 30
            y_groups.setdefault(y_range, {})
            if letter not in y_groups[y_range]:
                y_groups[y_range][letter] = pos

    best_header_row = None
    best_header_y = None
    max_headers_found = 0
    for y_range, letters in y_groups.items():
        if len(letters) > max_headers_found:
            max_headers_found = len(letters)
            best_header_row = letters
            best_header_y = y_range

    # Find X-like tokens below headers
    x_marks = []
    for i, text in enumerate(ocr["text"]):
        t = (text or "").strip().upper()
        if not t:
            continue
        try:
            conf = int(ocr["conf"][i])
        except Exception:
            conf = 0
        y_pos = ocr["top"][i]

        if best_header_y and y_pos > best_header_y + 50 and conf > 20:
            # treat single letters that aren't headers as potential X
            if t in ["X", "K", "*", "x"] or (len(t) == 1 and t not in list("ABCDEFGH")):
                x_marks.append({
                    "x": ocr["left"][i] + ocr["width"][i] // 2,
                    "y": y_pos,
                    "conf": conf
                })

    # group x marks into rows by y
    x_marks.sort(key=lambda m: m["y"])
    x_rows = []
    current = []
    last_y = -1
    for m in x_marks:
        if last_y == -1 or abs(m["y"] - last_y) < 40:
            current.append(m)
        else:
            if current:
                x_rows.append(current)
            current = [m]
        last_y = m["y"]
    if current:
        x_rows.append(current)

    # map each row to closest header column by x-distance
    if best_header_row:
        for x_row in x_rows[:expected_count]:
            best_x = max(x_row, key=lambda m: m["conf"])
            best_col = 4
            best_dist = float("inf")
            for letter, pos in best_header_row.items():
                dist = abs(best_x["x"] - pos["x"])
                if dist < best_dist:
                    best_dist = dist
                    best_col = ord(letter) - ord('A') + 1
            if best_dist < 150:
                values.append(best_col)
            else:
                values.append(4)

    while len(values) < expected_count:
        values.append(4)
    return values[:expected_count]


def extract_8box_rows(img: Image.Image, expected_count: int) -> List[int]:
    """
    Two-pass strategy:
      1) CV pass (shapes + X-energy)
      2) If results look doubtful, OCR-header fallback (your original approach)
    """
    cv_vals = _extract_8box_rows_cv(img, expected_count)

    # Heuristics: if all defaults or zero variance, fall back
    if (all(v == 4 for v in cv_vals)) or (len(set(cv_vals)) <= 1):
        ocr_vals = _extract_8box_rows_ocr_headers(img, expected_count)
        return ocr_vals
    return cv_vals


# =========================
# Core extraction per PDF
# =========================
def extract_from_pdf(pdf_path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {
        "file": str(pdf_path),
        "last_name": None,
        "grade": None,
        "occ": None,
        "to_date": None,
        "page2_values": [4]*5,
        "page3_values": [4]*5,
        "page4_values": [4]*4,
    }

    with fitz.open(pdf_path) as doc:
        # Page 1
        if len(doc) >= 1:
            img1 = pdf_page_to_image(doc, 0, scale=3.0)
            text1, ocr1 = get_ocr(img1, psm=6)

            # Last name (strictly revert to prior logic)
            data["last_name"] = extract_last_name_page1_STRICT(text1)

            # Grade
            data["grade"] = extract_grade_page1(img1)

            # OCC
            data["occ"] = extract_occ_page1(img1)

            # To date (restored)
            to_date = extract_to_date_page1(text1, ocr1)
            if to_date:
                data["to_date"] = to_date

        # Page 2: 5 rows
        if len(doc) >= 2:
            img2 = pdf_page_to_image(doc, 1, scale=3.0)
            data["page2_values"] = extract_8box_rows(img2, expected_count=5)

        # Page 3: 5 rows
        if len(doc) >= 3:
            img3 = pdf_page_to_image(doc, 2, scale=3.0)
            data["page3_values"] = extract_8box_rows(img3, expected_count=5)

        # Page 4: 4 rows
        if len(doc) >= 4:
            img4 = pdf_page_to_image(doc, 3, scale=3.0)
            data["page4_values"] = extract_8box_rows(img4, expected_count=4)

    return data


# =========================
# CSV + discovery
# =========================
def write_csv(out_path: Path, records: List[Dict[str, object]]) -> None:
    fieldnames = [
        "file","last_name","grade","occ","to_date",
        *(f"p2_{i+1}" for i in range(5)),
        *(f"p3_{i+1}" for i in range(5)),
        *(f"p4_{i+1}" for i in range(4)),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            row = {k: rec.get(k) for k in ["file","last_name","grade","occ","to_date"]}
            for i, v in enumerate(rec.get("page2_values", [])):
                row[f"p2_{i+1}"] = v
            for i, v in enumerate(rec.get("page3_values", [])):
                row[f"p3_{i+1}"] = v
            for i, v in enumerate(rec.get("page4_values", [])):
                row[f"p4_{i+1}"] = v
            w.writerow(row)

def discover_pdfs_auto(base_dir: Path, recursive: bool) -> List[Path]:
    return sorted(base_dir.rglob("*.pdf")) if recursive else sorted(base_dir.glob("*.pdf"))

def discover_pdfs_from_path(input_path: Path) -> List[Path]:
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
        pdfs = sorted(input_path.rglob("*.pdf"))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")
    return pdfs


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="Marine FITREP PDF → CSV (restored Last Name & To date + better 8-box)")
    ap.add_argument("--input", help="Optional: a PDF, a directory of PDFs, or a text file with PDF paths")
    ap.add_argument("--output", default="fitreps.csv", help="Output CSV path (default: fitreps.csv)")
    ap.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    ap.add_argument("--use-script-dir", action="store_true", help="Scan the folder this script lives in")
    args = ap.parse_args()

    print("Starting FITREP extractor…", flush=True)
    print(f"Python: {sys.version.split()[0]}  Executable: {sys.executable}", flush=True)

    if args.input:
        in_path = Path(args.input)
        print(f"Discovering PDFs from: {in_path} …", flush=True)
        pdfs = discover_pdfs_from_path(in_path)
        base_dir = in_path if in_path.is_dir() else Path.cwd()
    else:
        base_dir = Path(__file__).resolve().parent if args.use_script_dir else Path.cwd()
        print(f"Discovering PDFs under: {base_dir} (recursive={args.recursive}) …", flush=True)
        pdfs = discover_pdfs_auto(base_dir, recursive=args.recursive)

    if not pdfs:
        print("No PDFs found. Nothing to do.", flush=True)
        sys.exit(1)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = base_dir / out_path.name

    print(f"Found {len(pdfs)} PDF(s).", flush=True)
    print(f"Writing CSV to: {out_path}", flush=True)

    records: List[Dict[str, object]] = []
    for i, pdf in enumerate(pdfs, 1):
        try:
            rec = extract_from_pdf(pdf)
            records.append(rec)
            print(f"[{i}/{len(pdfs)}] OK: {pdf.name}  Last={rec.get('last_name')}  Grade={rec.get('grade')}  OCC={rec.get('occ')}  To={rec.get('to_date')}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] ERROR: {pdf} -> {e}", flush=True)

    write_csv(out_path, records)
    print(f"Done. Wrote: {out_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        raise

