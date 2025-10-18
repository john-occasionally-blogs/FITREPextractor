#!/usr/bin/env python3
import os
from pathlib import Path

import fitz  # PyMuPDF

from fitrep_extractor import FITREPExtractor


def run_one(pdf_path: Path):
    ex = FITREPExtractor()
    with fitz.open(str(pdf_path)) as doc:
        for page_num, expected in [(1,5),(2,5),(3,4)]:
            dbg = ex.debug_checkbox_diagnostics(doc, page_num, expected)
            vals_tb = dbg.get('values_text_based')
            vals_rb = dbg.get('values_row_bands')
            print(f"page {page_num+1}")
            print(f"  header_y_px: {dbg.get('header_y_px')}")
            print(f"  centers_px:  {dbg.get('centers_px')}")
            print(f"  row peaks:   {[ (round(s,1), y) for (s,y) in dbg.get('row_energy_peaks', []) ]}")
            print(f"  chosen rows: {dbg.get('chosen_rows_y')}")
            cols = dbg.get('col_darkness_per_row') or []
            for i, row_scores in enumerate(cols, 1):
                print(f"  row {i} col darkness: {row_scores}")
            print(f"  text_based:  {vals_tb}")
            print(f"  row_bands:   {vals_rb}")
            print("---")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Print numeric diagnostics for checkbox detection")
    ap.add_argument('pdf', help='Path to PDF to diagnose')
    args = ap.parse_args()
    run_one(Path(args.pdf))


if __name__ == '__main__':
    main()

