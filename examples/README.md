Checkbox Ground-Truth Verification
=================================

Files
- `verify_checkboxes.py` — runs extraction and compares against a local manifest.
- `ground_truth_checkboxes.sample.json` — template showing the expected format.

Usage
1) Copy the template and fill in your own files and values:
   cp examples/ground_truth_checkboxes.sample.json examples/ground_truth_checkboxes.json
   # edit examples/ground_truth_checkboxes.json

   Each entry:
   {
     "path": "/absolute/path/to/file.pdf",
     "expected": [a,b,c,d,e, f,g,h,i,j, k,l,m,n]
   }

2) Run the verifier:
   PYTHONPATH=. python3 examples/verify_checkboxes.py

Notes
- The manifest and any optional overrides JSON are git-ignored to avoid committing sensitive data.
- You can set `FITREP_CHECKBOX_FALLBACK=auto` to enable the guarded OCR fallback during development.

