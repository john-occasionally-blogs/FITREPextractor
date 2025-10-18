script = r'''
#!/usr/bin/env python3
import os, re, csv, sys, io, argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import fitz
from PIL import Image
import pytesseract
import numpy as np
import cv2

VALID_GRADES = [
    "PVT","PFC","LCPL","CPL","SGT","SSGT","GYSGT","MGYSGT","1STSGT","SGTMAJ",
    "2NDLT","1STLT","CAPT","MAJ","LTCOL","COL",
    "WO","CWO2","CWO3","CWO4","CWO5",
    "BGEN","MAJGEN","LTGEN","GEN"
]
VALID_OCC_CODES = ["GC","DC","CH","TR","CD","TD","FD","EN","CS","AN","AR","SA","RT"]

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
    return (s.strip().upper()
            .replace("0","O")
            .replace("1","I")
            .replace("5","S")
            .replace(":","")
            .replace(".",""))

def nearest_right_token(ocr, i: int, max_dx: int = 450, max_dy: int = 35) -> Optional[int]:
    base_y = ocr["top"][i]
    base_x = ocr["left"][i] + ocr["width"][i]
    best, best_dx = None, 10**9
    for j, t in enumerate(ocr["text"]):
        if not t: continue
        x, y = ocr["left"][j], ocr["top"][j]
        if y < base_y - max_dy or y > base_y + max_dy: continue
        dx = x - base_x
        if dx < 0 or dx > max_dx: continue
        if dx < best_dx:
            best_dx, best = dx, j
    return best

def find_label_indices(ocr, labels: List[str]) -> List[int]:
    labset = {norm_token(l) for l in labels}
    idxs = []
    for i, t in enumerate(ocr["text"]):
        if not t: continue
        if norm_token(t) in labset:
            idxs.append(i)
    return idxs

def extract_last_name_page1(img: Image.Image) -> Optional[str]:
    # Revert to a stricter, label-anchored method:
    # Look for line beginning with "a. Last Name" (typical FITREP label)
    text, ocr = get_ocr(img, psm=6)
    lines = text.splitlines()
    # First, try explicit "a. Last Name" anchor
    for idx, line in enumerate(lines):
        if re.search(r'^\s*a\.\s*last\s*name\b', line, re.IGNORECASE):
            for look_ahead in range(1, 4):
                if idx + look_ahead >= len(lines): break
                # choose first ALLCAPS token (incl hyphen)
                caps = re.findall(r"\b[A-Z][A-Z\-']{1,}\b", lines[idx+look_ahead])
                caps = [c for c in caps if c not in ("LAST","NAME")]
                if caps:
                    return caps[0]
    # Next, any line containing "Last Name" anchor
    for idx, line in enumerate(lines):
        if "LAST" in line.upper() and "NAME" in line.upper():
            for look_ahead in range(1, 4):
                if idx + look_ahead >= len(lines): break
                caps = re.findall(r"\b[A-Z][A-Z\-']{1,}\b", lines[idx+look_ahead])
                caps = [c for c in caps if c not in ("LAST","NAME")]
                if caps:
                    return caps[0]
    # As a final fallback, use positional OCR near a token 'LAST' 'NAME' sequence
    # Find token "LAST" then "NAME" on same y-row and take the next token to the right
    words = ocr["text"]; tops = ocr["top"]; lefts = ocr["left"]; widths = ocr["width"]
    indices_last = [i for i,w in enumerate(words) if w and norm_token(w)=="LAST"]
    for i in indices_last:
        row_y = tops[i]
        # find NAME on same row close to the right
        name_idx = None
        for j in range(i+1, min(i+15, len(words))):
            if abs(tops[j]-row_y) > 30: break
            if words[j] and norm_token(words[j])=="NAME":
                name_idx = j; break
        if name_idx is None: continue
        k = nearest_right_token(ocr, name_idx, max_dx=600, max_dy=35)
        if k is not None:
            tok = norm_token(words[k])
            if re.fullmatch(r"[A-Z][A-Z\-']{1,}", tok):
                return tok
    return None

def extract_grade_page1(img: Image.Image, valid_grades: List[str]) -> Optional[str]:
    text, ocr = get_ocr(img, psm=6)
    h = img.height; top_third = h // 3
    grade_idxs = [i for i,t in enumerate(ocr["text"]) if t and "GRADE" in t.upper() and ocr["top"][i] < top_third]
    if not grade_idxs:
        top_lines = text.splitlines()[: max(1, len(text.splitlines())//3)]
        tt = "\n".join(top_lines).upper()
        for g in valid_grades:
            if re.search(rf"\b{re.escape(g)}\b", tt): return g
        return None
    grade_idxs.sort(key=lambda i: ocr["top"][i])
    candidate = grade_idxs[0]
    j = nearest_right_token(ocr, candidate, max_dx=450, max_dy=30)
    window = []
    if j is not None:
        for k in range(j, min(j+10, len(ocr["text"]))):
            if abs(ocr["top"][k] - ocr["top"][candidate]) > 30: break
            tok = norm_token(ocr["text"][k])
            if tok: window.append(tok)
    for tok in window:
        if tok in valid_grades: return tok
    joined = "".join(window)
    for g in valid_grades:
        if g in joined: return g
    top_lines = text.splitlines()[: max(1, len(text.splitlines())//3)]
    tt = "\n".join(top_lines).upper()
    for g in valid_grades:
        if re.search(rf"\b{re.escape(g)}\b", tt): return g
    return None

def extract_occ_page1(img: Image.Image, valid_occ_codes: List[str]) -> Optional[str]:
    text, ocr = get_ocr(img, psm=6)
    labels = ["OCC","OCC.","OCC:","OCCASION","OCCASION:","OCCASION.","Occasion"]
    idxs = find_label_indices(ocr, labels)
    idxs.sort(key=lambda i: (ocr["top"][i], ocr["left"][i]))
    for idx in idxs:
        row_y = ocr["top"][idx]
        for k in range(idx+1, min(idx+20, len(ocr["text"]))):
            if abs(ocr["top"][k]-row_y) > 30: break
            tok = norm_token(ocr["text"][k])
            if len(tok)==2 and tok in valid_occ_codes: return tok
            if len(tok)==2:
                corr = tok.replace("0","O").replace("1","I").replace("5","S")
                if corr in valid_occ_codes: return corr
    # fallback top half
    half = img.height//2
    for i,t in enumerate(ocr["text"]):
        if not t: continue
        if ocr["top"][i] > half: continue
        tok = norm_token(t)
        if len(tok)==2 and tok in valid_occ_codes: return tok
    return None

def parse_date_token(tok: str) -> Optional[str]:
    tok = tok.strip()
    # formats: YYYYMMDD, YYYY-MM-DD, MM/DD/YYYY, M/D/YYYY
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", tok)
    if m:
        y, mo, d = m.groups()
        try:
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", tok)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", tok)
    if m:
        mo, d, y = map(int, m.groups())
        try:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except ValueError:
            return None
    return None

def extract_period_dates_page1(img: Image.Image) -> Tuple[Optional[str], Optional[str]]:
    # Look for "FROM ... TO ..." in page 1 text
    text, ocr = get_ocr(img, psm=6)
    up = text.upper().replace("\n", " ")
    # Find two dates around FROM...TO
    # capture tokens after FROM and after TO
    m = re.search(r"\bFROM\b\s*([0-9/ -]{8,12})\s*\bTO\b\s*([0-9/ -]{8,12})", up)
    if m:
        from_raw, to_raw = m.groups()
        f = parse_date_token(from_raw.replace(" ", "").replace("--","-"))
        t = parse_date_token(to_raw.replace(" ", "").replace("--","-"))
        return f, t
    # fallback: scan for tokens "FROM" and "TO" in OCR positions
    words = [w for w in ocr["text"]]
    tops, lefts = ocr["top"], ocr["left"]
    idxs_from = [i for i,w in enumerate(words) if w and w.strip().upper()=="FROM"]
    idxs_to = [i for i,w in enumerate(words) if w and w.strip().upper()=="TO"]
    def next_date_after(idx):
        row_y = tops[idx]
        for j in range(idx+1, min(idx+12, len(words))):
            if abs(tops[j]-row_y) > 35: break
            if not words[j]: continue
            tok = re.sub(r"[^\d/ -]", "", words[j])
            dt = parse_date_token(tok)
            if dt: return dt
        return None
    f = None
    for i in idxs_from:
        f = next_date_after(i)
        if f: break
    t = None
    for i in idxs_to:
        t = next_date_after(i)
        if t: break
    return f, t

def extract_8box_rows(img: Image.Image, expected_count: int, y_min_ratio: float=0.18, y_max_ratio: float=0.92) -> List[int]:
    """Detect 8-box rows and choose the column with the X.
       y_min_ratio/y_max_ratio restrict vertical search to ignore headers/footers."""
    cv_img = np.array(img)
    if cv_img.ndim == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv_img
    h_img, w_img = gray.shape[:2]
    
    # vertical crop to likely area
    y0 = int(h_img * y_min_ratio)
    y1 = int(h_img * y_max_ratio)
    gray_c = gray[y0:y1, :]
    
    # binarize
    thr = cv2.adaptiveThreshold(gray_c, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 35, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = (w_img*h_img) * 0.000015
    max_area = (w_img*h_img) * 0.008
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        y += y0  # offset due to cropping
        area = w*h; ar = w/float(h)
        if area < min_area or area > max_area: continue
        if 0.75 < ar < 1.33:
            pad = max(1, min(w,h)//8)
            inner = thr[(y-y0)+pad:(y-y0)+h-pad, x+pad:x+w-pad]
            if inner.size == 0: continue
            if inner.mean() < 40: # filled region -> not a hollow box
                continue
            boxes.append((x,y,w,h))
    if not boxes:
        return [4]*expected_count
    # group into rows by y
    boxes.sort(key=lambda b: b[1])
    rows = []
    current = [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1]-current[-1][1]) < 22:
            current.append(b)
        else:
            rows.append(current)
            current = [b]
    rows.append(current)
    # keep candidate rows with >=7 boxes and choose leftmost 8
    cleaned_rows = []
    for r in rows:
        r_sorted = sorted(r, key=lambda b: b[0])
        if len(r_sorted)>=7:
            # try to find 8 with near-uniform spacing
            cleaned_rows.append(r_sorted[:8])
    # sort rows by vertical position
    cleaned_rows.sort(key=lambda r: sum(b[1] for b in r)/len(r))
    values = []
    for row in cleaned_rows[:expected_count]:
        row = sorted(row, key=lambda b: b[0])
        if len(row) < 8:
            row = (row + [row[-1]]*8)[:8]
        elif len(row) > 8:
            row = row[:8]
        # compute X score: combine diagonal correlation and Hough diagonals
        scores = []
        for (x,y,w,h) in row:
            ix = x + w//6; iw = max(3, w - 2*(w//6))
            iy = y + h//6; ih = max(3, h - 2*(h//6))
            roi = gray[iy:iy+ih, ix:ix+iw]
            roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            # diagonal kernels
            k = max(3, (min(roi_bin.shape[:2])//6) | 1)
            diag1 = np.eye(k, dtype=np.float32)
            diag2 = np.flipud(diag1)
            d1 = cv2.filter2D(roi_bin.astype(np.float32), -1, diag1)
            d2 = cv2.filter2D(roi_bin.astype(np.float32), -1, diag2)
            score_diag = float(d1.mean() + d2.mean())
            # Hough lines (diagonals)
            edges = cv2.Canny(roi, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=max(5, min(iw,ih)//2), maxLineGap=3)
            score_hough = 0.0
            if lines is not None:
                for l in lines:
                    x1,y1,x2,y2 = l[0]
                    angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                    angle = abs(angle)
                    angle = 180-angle if angle>90 else angle
                    # near diagonal ~45 deg ± 15
                    if 30 <= angle <= 60:
                        score_hough += 1.0
            scores.append(score_diag + 10.0*score_hough)
        col = int(np.argmax(scores)) + 1
        values.append(col)
    while len(values) < expected_count:
        values.append(4)
    return values

def extract_from_pdf(pdf_path: Path) -> Dict[str, object]:
    data = {
        "file": str(pdf_path),
        "last_name": None,
        "grade": None,
        "occ": None,
        "from_date": None,
        "to_date": None,
        "p1_1": None, "p1_2": None, "p1_3": None, "p1_4": None, "p1_5": None,
        "p2_1": None, "p2_2": None, "p2_3": None, "p2_4": None, "p2_5": None,
        "p3_1": None, "p3_2": None, "p3_3": None, "p3_4": None, "p3_5": None,
        "p4_1": None, "p4_2": None, "p4_3": None, "p4_4": None
    }
    with fitz.open(pdf_path) as doc:
        if len(doc)>=1:
            img1 = pdf_page_to_image(doc, 0, scale=3.0)
            # Last name (reverted)
            ln = extract_last_name_page1(img1)
            if ln: data["last_name"] = ln
            # Grade
            g = extract_grade_page1(img1, VALID_GRADES)
            if g: data["grade"] = g
            # OCC
            occ = extract_occ_page1(img1, VALID_OCC_CODES)
            if occ: data["occ"] = occ
            # Dates
            fdate, tdate = extract_period_dates_page1(img1)
            if fdate: data["from_date"] = fdate
            if tdate: data["to_date"] = tdate
            # Page 1 8-box rows (expected 5)
            vals = extract_8box_rows(img1, expected_count=5, y_min_ratio=0.25, y_max_ratio=0.88)
            for i,v in enumerate(vals[:5], start=1):
                data[f"p1_{i}"] = v
        if len(doc)>=2:
            img2 = pdf_page_to_image(doc, 1, scale=3.0)
            vals = extract_8box_rows(img2, expected_count=5, y_min_ratio=0.18, y_max_ratio=0.92)
            for i,v in enumerate(vals[:5], start=1):
                data[f"p2_{i}"] = v
        if len(doc)>=3:
            img3 = pdf_page_to_image(doc, 2, scale=3.0)
            vals = extract_8box_rows(img3, expected_count=5, y_min_ratio=0.18, y_max_ratio=0.92)
            for i,v in enumerate(vals[:5], start=1):
                data[f"p3_{i}"] = v
        if len(doc)>=4:
            img4 = pdf_page_to_image(doc, 3, scale=3.0)
            vals = extract_8box_rows(img4, expected_count=4, y_min_ratio=0.18, y_max_ratio=0.92)
            for i,v in enumerate(vals[:4], start=1):
                data[f"p4_{i}"] = v
    return data

def discover_pdfs_auto(base_dir: Path, recursive: bool) -> List[Path]:
    return sorted(base_dir.rglob("*.pdf")) if recursive else sorted(base_dir.glob("*.pdf"))

def write_csv(out_path: Path, records: List[Dict[str, object]]) -> None:
    fieldnames = ["file","last_name","grade","occ","from_date","to_date"]
    fieldnames += [f"p1_{i}" for i in range(1,6)]
    fieldnames += [f"p2_{i}" for i in range(1,6)]
    fieldnames += [f"p3_{i}" for i in range(1,6)]
    fieldnames += [f"p4_{i}" for i in range(1,5)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            row = {k: rec.get(k) for k in fieldnames}
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="FITREP PDF extractor (with robust OCC/Grade and 8-box detection)")
    ap.add_argument("--recursive", action="store_true", help="Scan subdirectories")
    ap.add_argument("--use-script-dir", action="store_true", help="Scan the script's directory instead of CWD")
    ap.add_argument("--output", default="fitreps.csv", help="Output CSV path (default: fitreps.csv)")
    args = ap.parse_args()
    print("Starting FITREP extractor...", flush=True)
    base_dir = Path(__file__).resolve().parent if args.use_script_dir else Path.cwd()
    print(f"Discovering PDFs under: {base_dir} (recursive={args.recursive}) ...", flush=True)
    pdfs = discover_pdfs_auto(base_dir, args.recursive)
    if not pdfs:
        print("No PDFs found. Nothing to do.", flush=True)
        sys.exit(1)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = base_dir / out_path.name
    print(f"Found {len(pdfs)} PDF(s).", flush=True)
    print(f"Writing CSV to: {out_path}", flush=True)
    records = []
    for i,pdf in enumerate(pdfs,1):
        try:
            rec = extract_from_pdf(pdf)
            records.append(rec)
            print(f"[{i}/{len(pdfs)}] OK: {pdf.name}  Grade={rec.get('grade')}  OCC={rec.get('occ')}  To={rec.get('to_date')}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] ERROR: {pdf} -> {e}", flush=True)
    write_csv(out_path, records)
    print(f"Done. Wrote: {out_path}", flush=True)

if __name__ == "__main__":
    main()
'''
# Try compiling to check syntax
compile(script, 'fitrep_extractor_new.py', 'exec')
print("Syntax OK. length:", len(script))


