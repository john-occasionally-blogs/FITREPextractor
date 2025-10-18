#!/usr/bin/env python3

import os
import re
import csv
import sys
import io
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Make stdout more eager when possible (Python 3.7-safe)
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

# ----------------------------
# Config / Valid sets
# ----------------------------
VALID_GRADES = [
    "PVT","PFC","LCPL","CPL","SGT","SSGT","GYSGT","MGYSGT","1STSGT","SGTMAJ",
    "2NDLT","1STLT","CAPT","MAJ","LTCOL","COL",
    "WO","CWO2","CWO3","CWO4","CWO5",
    "BGEN","MAJGEN","LTGEN","GEN"
]
VALID_OCC_CODES = ["GC","DC","CH","TR","CD","TD","FD","EN","CS","AN","AR","SA","RT"]

# ----------------------------
# Utils / OCR helpers
# ----------------------------
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

# ----------------------------
# Field extractors
# ----------------------------
def extract_last_name_from_page1_text(text: str) -> Optional[str]:
    patterns = [
        r'Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
        r'a\.\s*Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
        r'Last Name.*?\n\s*([A-Z]+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "LAST" in line.upper() and "NAME" in line.upper():
            for j in range(0, min(3, len(lines) - i - 1)):
                caps = re.findall(r"\b[A-Z]{2,}\b", lines[i + 1 + j])
                if caps:
                    return caps[0]
    return None

def extract_grade_page1(img: Image.Image, valid_grades: List[str]) -> Optional[str]:
    text, ocr = get_ocr(img, psm=6)
    h = img.height
    top_third = h // 3
    grade_idxs = [
        i for i, t in enumerate(ocr["text"])
        if t and "GRADE" in t.upper() and ocr["top"][i] < top_third
    ]
    if not grade_idxs:
        lines = text.splitlines()
        top_lines = lines[: max(1, len(lines)//3)]
        tt = "\n".join(top_lines).upper()
        for g in valid_grades:
            if re.search(rf"\b{re.escape(g)}\b", tt):
                return g
        return None

    grade_idxs.sort(key=lambda i: ocr["top"][i])  # smallest y = top-most
    candidate = grade_idxs[0]
    j = nearest_right_token(ocr, candidate, max_dx=450, max_dy=30)
    window = []
    if j is not None:
        for k in range(j, min(j + 10, len(ocr["text"]))):
            if abs(ocr["top"][k] - ocr["top"][candidate]) > 30:
                break
            tok = norm_token(ocr["text"][k])
            if tok:
                window.append(tok)

    for tok in window:
        if tok in valid_grades:
            return tok
    joined = "".join(window)
    for g in valid_grades:
        if g in joined:
            return g

    lines = text.splitlines()
    tt = "\n".join(lines[: max(1, len(lines)//3)]).upper()
    for g in valid_grades:
        if re.search(rf"\b{re.escape(g)}\b", tt):
            return g
    return None

def extract_occ_page1(img: Image.Image, valid_occ_codes: List[str]) -> Optional[str]:
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
            if len(tok) == 2 and tok in valid_occ_codes:
                return tok
            if len(tok) == 2:
                corr = tok.replace("0","O").replace("1","I").replace("5","S")
                if corr in valid_occ_codes:
                    return corr

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
    cv_img = np.array(img)
    if cv_img.ndim == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img

    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 35, 9)
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
            pad = max(1, min(w, h) // 10)
            inner = thr[y + pad:y + h - pad, x + pad:x + w - pad]
            if inner.size == 0:
                continue
            if inner.mean() < 50:
                continue
            boxes.append((x, y, w, h))

    if not boxes:
        return [4] * expected_count  # neutral fallback

    boxes.sort(key=lambda b: b[1])  # by y
    rows = []
    current = [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - current[-1][1]) < 18:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)

    cleaned_rows = []
    for r in rows:
        r_sorted = sorted(r, key=lambda b: b[0])
        if len(r_sorted) >= 7:
            cleaned_rows.append(r_sorted[:8])

    cleaned_rows.sort(key=lambda row: sum(b[1] for b in row) / len(row))  # top -> bottom

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
            iw = max(3, w - 2 * (w // 6))
            iy = y + h // 6
            ih = max(3, h - 2 * (h // 6))
            roi = gray[iy:iy + ih, ix:ix + iw]
            roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            k = max(3, (min(roi_bin.shape[:2]) // 6) | 1)  # odd
            diag1 = np.eye(k, dtype=np.float32)
            diag2 = np.flipud(diag1)
            d1 = cv2.filter2D(roi_bin.astype(np.float32), -1, diag1)
            d2 = cv2.filter2D(roi_bin.astype(np.float32), -1, diag2)
            score = float(d1.mean() + d2.mean())
            scores.append(score)

        col = int(np.argmax(scores)) + 1  # 1..8
        values.append(col)

    while len(values) < expected_count:
        values.append(4)

    return values

# ----------------------------
# Core extraction per PDF
# ----------------------------
def extract_from_pdf(pdf_path: Path) -> Dict[str, object]:
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
        if len(doc) >= 1:
            img1 = pdf_page_to_image(doc, 0, scale=3.0)
            text1, _ = get_ocr(img1, psm=6)

            ln = extract_last_name_from_page1_text(text1)
            if ln:
                data["last_name"] = ln

            grade = extract_grade_page1(img1, VALID_GRADES)
            if grade:
                data["grade"] = grade

            occ = extract_occ_page1(img1, VALID_OCC_CODES)
            if occ:
                data["occ"] = occ

        if len(doc) >= 2:
            img2 = pdf_page_to_image(doc, 1, scale=3.0)
            data["page2_values"] = extract_8box_rows(img2, expected_count=5)

        if len(doc) >= 3:
            img3 = pdf_page_to_image(doc, 2, scale=3.0)
            data["page3_values"] = extract_8box_rows(img3, expected_count=5)

        if len(doc) >= 4:
            img4 = pdf_page_to_image(doc, 3, scale=3.0)
            data["page4_values"] = extract_8box_rows(img4, expected_count=4)

    return data

# ----------------------------
# Discovery / CSV
# ----------------------------
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

def write_csv(out_path: Path, records: List[Dict[str, object]]) -> None:
    fieldnames = [
        "file","last_name","grade","occ",
        *(f"p2_{i+1}" for i in range(5)),
        *(f"p3_{i+1}" for i in range(5)),
        *(f"p4_{i+1}" for i in range(4)),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
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

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Marine FITREP PDF to CSV Extractor (verbose)")
    ap.add_argument("--input", help="Optional: a PDF, a directory of PDFs, or a text file with PDF paths")
    ap.add_argument("--output", default="fitreps.csv", help="Output CSV path (default: fitreps.csv)")
    ap.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    ap.add_argument("--use-script-dir", action="store_true",
                    help="Scan the folder this script lives in (instead of current working directory)")
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
            print(f"[{i}/{len(pdfs)}] OK: {pdf.name}  Grade={rec.get('grade')}  OCC={rec.get('occ')}", flush=True)
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

