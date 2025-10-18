#!/usr/bin/env python3
"""
Debug OCC extraction to see why it's finding AN instead of TR
"""

import fitz
from PIL import Image
import pytesseract
import io

def debug_occ_extraction():
    """Debug the OCC extraction process"""
    pdf_path = "/Users/John/Desktop/FITREP Test/fitrepPdf.pdf"
    doc = fitz.open(pdf_path)
    
    # Process Page 1
    page = doc[0]
    mat = fitz.Matrix(3, 3)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    
    # Get OCR data
    text1 = pytesseract.image_to_string(img)
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    print("=== DEBUGGING OCC EXTRACTION ===\n")
    
    # Show raw text to look for TR
    print("Raw OCR text (first 1000 chars):")
    print(repr(text1[:1000]))
    print("\n" + "="*50 + "\n")
    
    # Valid OCC codes for reference
    valid_occ_codes = ['GC', 'DC', 'CH', 'TR', 'CD', 'TD', 'FD', 'EN', 'CS', 'AN', 'AR', 'SA', 'RT']
    
    # Look for all 2-letter codes in top half
    print("All 2-letter codes found in top half:")
    half_height = img.height // 2
    found_codes = []
    
    for i, text in enumerate(ocr_data["text"]):
        if not text or ocr_data["top"][i] > half_height:
            continue
        
        text_clean = text.strip().upper().replace("0", "O").replace("1", "I").replace("5", "S").replace(":", "").replace(".", "")
        
        if len(text_clean) == 2:
            found_codes.append({
                'text': text_clean,
                'original': text,
                'position': (ocr_data["left"][i], ocr_data["top"][i]),
                'is_valid': text_clean in valid_occ_codes
            })
    
    # Sort by position (top to bottom, left to right)
    found_codes.sort(key=lambda x: (x['position'][1], x['position'][0]))
    
    for i, code in enumerate(found_codes):
        status = "✅ VALID" if code['is_valid'] else "❌ invalid"
        print(f"{i+1:2d}. '{code['text']}' (orig: '{code['original']}') at {code['position']} - {status}")
    
    print("\n" + "="*50 + "\n")
    
    # Look specifically for TR in the text
    print("Looking specifically for 'TR' patterns:")
    lines = text1.split('\n')
    for i, line in enumerate(lines):
        if 'TR' in line.upper():
            print(f"Line {i}: '{line.strip()}'")
    
    # Also check all text tokens for TR
    print("\nAll tokens containing 'TR':")
    for i, text in enumerate(ocr_data["text"]):
        if text and 'TR' in text.upper():
            print(f"Token {i}: '{text}' at ({ocr_data['left'][i]}, {ocr_data['top'][i]})")
    
    doc.close()

if __name__ == "__main__":
    debug_occ_extraction()