#!/usr/bin/env python3
import os
import json
import itertools
from pathlib import Path

from fitrep_extractor import FITREPExtractor


def extract_14(ex: FITREPExtractor, pdf_path: Path):
    d = ex.extract_from_pdf(pdf_path)
    if not d:
        return None
    v2 = d.get('page2_values') or [4]*5
    v3 = d.get('page3_values') or [4]*5
    v4 = d.get('page4_values') or [4]*4
    vals = (v2 + v3 + v4)[:14]
    return vals


def hamming(a, b):
    return sum(1 for x,y in zip(a,b) if x!=y)


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Grid-search calibrator for checkbox scoring')
    ap.add_argument('-m', '--manifest', required=True, help='JSON with entries: {path, expected[14]}')
    args = ap.parse_args()

    entries = json.loads(Path(args.manifest).read_text())
    samples = []
    for e in entries:
        p = Path(e['path'])
        if not p.exists():
            print(f"Missing: {p}")
            return 2
        exp = e['expected']
        if len(exp) != 14:
            print(f"Bad expected length for {p}")
            return 2
        samples.append((p, exp))

    # Search space (kept modest)
    ROI_W = [0.5, 0.6, 0.7]
    ROI_V = [0.4, 0.5, 0.6]
    W_EDGE = [0.4, 0.6, 0.8]
    W_DIAG = [0.6, 0.8, 1.0]
    CONF = [1.10, 1.15, 1.20]
    BETA = [0.85, 0.90, 0.95]

    best = None
    best_err = 9999
    tried = 0

    for rw, rv, we, wd, cf, bt in itertools.product(ROI_W, ROI_V, W_EDGE, W_DIAG, CONF, BETA):
        os.environ['FITREP_ROI_W'] = str(rw)
        os.environ['FITREP_ROI_V'] = str(rv)
        os.environ['FITREP_W_DARK'] = '1.0'
        os.environ['FITREP_W_EDGE'] = str(we)
        os.environ['FITREP_W_DIAG'] = str(wd)
        os.environ['FITREP_W_OCRX'] = '50000'
        os.environ['FITREP_BETA'] = str(bt)
        os.environ['FITREP_CONF'] = str(cf)
        os.environ['FITREP_STRICT'] = 'true'
        os.environ['FITREP_CHECKBOX_FALLBACK'] = 'auto'
        ex = FITREPExtractor()
        total_err = 0
        ok = True
        for p, exp in samples:
            pred = extract_14(ex, p)
            if pred is None:
                ok = False
                break
            total_err += hamming(pred, exp)
        if not ok:
            continue
        tried += 1
        if total_err < best_err:
            best_err = total_err
            best = (rw, rv, we, wd, cf, bt)

    if best is None:
        print("Calibration failed to find a configuration")
        return 3

    rw, rv, we, wd, cf, bt = best
    print("Best configuration:")
    print(json.dumps({
        'FITREP_ROI_W': rw,
        'FITREP_ROI_V': rv,
        'FITREP_W_DARK': 1.0,
        'FITREP_W_EDGE': we,
        'FITREP_W_DIAG': wd,
        'FITREP_W_OCRX': 50000,
        'FITREP_BETA': bt,
        'FITREP_CONF': cf,
        'total_hamming_error': best_err,
        'tried': tried,
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

