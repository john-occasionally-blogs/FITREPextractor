#!/usr/bin/env python3
import os
from pathlib import Path
import argparse

import fitz  # PyMuPDF
from PIL import Image
import io
import pytesseract


LETTERS = list('ABCDEFGH')


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def render_gray(page, scale=3.0):
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert('L')
    return img


def moving_average(vals, k):
    if k <= 1:
        return vals[:]
    n = len(vals)
    out = [0.0] * n
    window = 0.0
    half = k // 2
    # simple centered moving average via cumulative sums
    csum = [0.0]
    for v in vals:
        csum.append(csum[-1] + v)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = (csum[b] - csum[a]) / max(1, (b - a))
    return out


def find_rows_and_cols_gray(img_gray: Image.Image, expected_rows: int):
    W, H = img_gray.size
    # Build "darkness" per pixel lazily using getpixel
    px = img_gray.load()

    # Restrict vertical search band to avoid headers/footers
    y_top = int(H * 0.20)
    y_bot = int(H * 0.90)
    # Row projection across width within [y_top, y_bot)
    row_proj = []
    for y in range(y_top, y_bot):
        s = 0
        for x in range(W):
            s += 255 - px[x, y]
        row_proj.append(float(s))

    # Row projection and smoothing
    row_sm = moving_average(row_proj, 41)

    # Peak picking with separation
    peaks = []
    last = -10_000
    min_sep = max(40, (y_bot - y_top) // (expected_rows + 1))
    for y in range(1, len(row_sm) - 1):
        if row_sm[y] > row_sm[y - 1] and row_sm[y] >= row_sm[y + 1]:
            if y - last >= min_sep:
                peaks.append((row_sm[y], y))
                last = y
    if not peaks:
        return [], []
    peaks.sort(key=lambda a: a[0], reverse=True)
    chosen = sorted(peaks[:expected_rows], key=lambda a: a[1])
    row_centers = [y_top + int(y) for (_, y) in chosen]

    # Column projection restricted to bands around detected rows
    band_half = 30
    col_proj = [0.0] * W
    for yc in row_centers:
        y0 = max(0, yc - band_half)
        y1 = min(H, yc + band_half)
        for x in range(W):
            s = 0
            for y in range(y0, y1):
                s += 255 - px[x, y]
            col_proj[x] += float(s)
    col_sm = moving_average(col_proj, 21)

    thr = max(col_sm) * 0.35 if col_sm else 0.0
    xs = [i for i, v in enumerate(col_sm) if v >= thr]
    if len(xs) < 8 and col_sm:
        thr = max(col_sm) * 0.20
        xs = [i for i, v in enumerate(col_sm) if v >= thr]
    if len(xs) == 0:
        return row_centers, []
    x_left = int(min(xs))
    x_right = int(max(xs))
    pad = max(0, (x_right - x_left) // 40)
    x_left += pad
    x_right -= pad
    if x_right <= x_left + 8:
        return row_centers, []

    cols = []
    for i in range(8):
        x0 = int(x_left + i * (x_right - x_left) / 8.0)
        x1 = int(x_left + (i + 1) * (x_right - x_left) / 8.0)
        cols.append((x0, max(x0 + 1, x1)))
    return row_centers, cols


def score_cell(crop_gray: Image.Image):
    # Darkness + diagonal energy + OCR "X" boost
    w, h = crop_gray.size
    cx = crop_gray.crop((0, 0, w, h)).load()
    def dark_at(x, y):
        return 255 - cx[x, y]
    s_dark = 0.0
    for yy in range(h):
        for xx in range(w):
            s_dark += dark_at(xx, yy)
    diag = 0.0
    for d in range(-2, 3):
        for y in range(h):
            x = y + d
            if 0 <= x < w:
                diag += dark_at(x, y)
            x2 = (w - 1 - y) + d
            if 0 <= x2 < w:
                diag += dark_at(x2, y)
    try:
        ocr_txt = pytesseract.image_to_string(crop_gray, config='--psm 10 -c tessedit_char_whitelist=Xx')
        has_x = 'x' in (ocr_txt or '').lower()
    except Exception:
        has_x = False
    score = 0.25 * s_dark + 1.0 * diag + (50000.0 if has_x else 0.0)
    return float(score), bool(has_x), float(s_dark), float(diag)


def dump_cells(pdf_path: Path, out_dir: Path):
    ensure_dir(out_dir)
    with fitz.open(str(pdf_path)) as doc:
        for page_num, expected in [(1, 5), (2, 5), (3, 4)]:
            if page_num >= len(doc):
                break
            page = doc[page_num]
            img_gray = render_gray(page, scale=3.0)
            W, H = img_gray.size
            row_centers, cols = find_rows_and_cols_gray(img_gray, expected)
            print(f"page {page_num+1}")
            print(f"  rows: {row_centers}")
            print(f"  cols: {cols[:3]} ... {cols[-3:] if cols else []}")
            if not row_centers or not cols:
                print("  (no rows/cols detected)")
                continue
            box_h = 36
            for r_idx, yc in enumerate(row_centers, 1):
                y0 = max(0, yc - box_h // 2)
                y1 = min(H, yc + box_h // 2)
                row_scores = []
                for c_idx, (x0, x1) in enumerate(cols, 1):
                    width = x1 - x0
                    inner_w = max(6, int(width * 0.4))
                    cx = (x0 + x1) // 2
                    ix0 = max(0, cx - inner_w // 2)
                    ix1 = min(W, cx + inner_w // 2)
                    crop = img_gray.crop((ix0, y0, ix1, y1))
                    score, has_x, s_dark, diag = score_cell(crop)
                    row_scores.append((score, has_x, s_dark, diag))
                    # Save crop image for inspection
                    out_name = f"{pdf_path.stem}_p{page_num+1}_r{r_idx}_c{LETTERS[c_idx-1]}.png"
                    crop.save(out_dir / out_name)
                # Print per-row summary
                best = max(enumerate(row_scores, 1), key=lambda kv: kv[1][0])
                best_col = best[0]
                print(f"  row {r_idx} best={best_col} scores=[{', '.join(str(int(s[0])) for s in row_scores)}]")


def main():
    ap = argparse.ArgumentParser(description='Dump per-cell crops and scores for checkbox grids')
    ap.add_argument('pdf', help='Path to FITREP PDF')
    ap.add_argument('--out', default='examples/debug_cells', help='Output directory for cell crops')
    args = ap.parse_args()
    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    dump_cells(pdf_path, out_dir)


if __name__ == '__main__':
    main()
