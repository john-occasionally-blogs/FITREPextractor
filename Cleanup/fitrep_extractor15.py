#!/usr/bin/env python3
"""
Marine FITREP PDF to CSV Extractor (verbose + immediate feedback)

- Auto-discovers PDFs (current dir by default; optional --recursive)
- Robust OCC + Grade OCR
- Robust 8-box (A..H) detection via OpenCV
- Loud status messages printed immediately (flush=True)
"""

import os
import re
import csv
import sys
import io
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Make stdout line-buffered so prints appear immediately in most shells
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger("fitrep")

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
def pdf_page_to_image(pdf: fitz.Document, page_index: int, scale: float = 3.0) -> Image.Image:
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
        if not t: continue
        x = ocr["left"][j]; y = ocr["top"][j]
        if y < base_y - max_dy or y > base_y + max_dy: continue
        dx = x - base_x
        if dx < 0 or dx > max_dx: continue
        if dx < best_dx:
            best_dx = dx; best = j
    return best

def find_label_indices(ocr, labels: List[str]) -> List[int]:
    labset = {norm_token(l) for l in lab_

