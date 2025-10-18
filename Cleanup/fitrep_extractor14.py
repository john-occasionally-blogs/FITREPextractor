#!/usr/bin/env python3
"""
Marine FITREP PDF to CSV Extractor (auto-scan current directory)

Default behavior (no args):
- Scans the current working directory for *.pdf (non-recursive)
- Extracts fields with robust Grade/OCC logic + OpenCV 8-box detection
- Writes fitreps.csv in the current working directory

Optional:
  --recursive          scan subdirectories too
  --use-script-dir     scan the folder the script lives in (instead of CWD)
  --input PATH         process a specific PDF, directory, or text file of paths
  --output FILE        output CSV name/path (default: fitreps.csv)
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

