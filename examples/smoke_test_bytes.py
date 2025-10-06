#!/usr/bin/env python3
"""
Minimal smoke test for bytes-based extraction.

Usage:
  python3 examples/smoke_test_bytes.py /path/to/file.pdf [--async]

Exits 0 on success, non-zero on failure.
"""

import sys
import json
import asyncio
from pathlib import Path

try:
    from fitrep_extractor import FITREPExtractor
except Exception as e:
    print("Failed to import FITREPExtractor: {0}".format(e))
    sys.exit(2)


def run_sync(pdf_path: Path) -> int:
    try:
        pdf_bytes = pdf_path.read_bytes()
    except Exception as e:
        print("Failed to read file: {0}".format(e))
        return 2

    extractor = FITREPExtractor()
    data = extractor.extract_from_bytes(pdf_bytes)
    if not data:
        print("No data extracted (None returned)")
        return 1

    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


async def run_async(pdf_path: Path) -> int:
    try:
        pdf_bytes = pdf_path.read_bytes()
    except Exception as e:
        print("Failed to read file: {0}".format(e))
        return 2

    extractor = FITREPExtractor()
    data = await extractor.extract_fitrep_data_bytes(pdf_bytes)
    if not data:
        print("No data extracted (None returned)")
        return 1

    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2

    pdf_path = Path(argv[1])
    use_async = len(argv) > 2 and argv[2] == "--async"

    if not pdf_path.exists():
        print("File not found: {0}".format(pdf_path))
        return 2

    if use_async:
        return asyncio.run(run_async(pdf_path))
    else:
        return run_sync(pdf_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

