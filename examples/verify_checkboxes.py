#!/usr/bin/env python3
import json
import os
from pathlib import Path

import fitz  # PyMuPDF

from fitrep_extractor import FITREPExtractor


def extract_ah_values(path: Path, mode: str):
    os.environ['FITREP_CHECKBOX_FALLBACK'] = mode
    ex = FITREPExtractor()
    data = ex.extract_from_pdf(path)
    if not data:
        return []
    v2 = data.get('page2_values') or []
    v3 = data.get('page3_values') or []
    v4 = data.get('page4_values') or []
    return v2 + v3 + v4


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Verify checkbox A–H values against ground truth")
    ap.add_argument('-m', '--manifest', default=str(Path(__file__).with_name('ground_truth_checkboxes.json')),
                    help='Path to ground truth JSON manifest')
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    entries = json.loads(manifest_path.read_text())
    any_fail = False
    for e in entries:
        path = Path(e['path'])
        expected = e['expected']
        print(path.name)
        old_vals = extract_ah_values(path, mode='off')
        auto_vals = extract_ah_values(path, mode='auto')
        print(f"  old:  {old_vals}")
        print(f"  auto: {auto_vals}")
        print(f"  exp:  {expected}")
        ok_old = (old_vals == expected)
        ok_auto = (auto_vals == expected)
        status = (
            "OK(old)" if ok_old else ("OK(auto)" if ok_auto else "MISMATCH")
        )
        if not (ok_old or ok_auto):
            any_fail = True
        print(f"  => {status}")
        print("---")

    if any_fail:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
