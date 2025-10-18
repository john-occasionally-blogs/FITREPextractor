#!/usr/bin/env python3
"""
Debug to find where TR should be - look at the actual form structure
"""

import fitz
from PIL import Image  
import pytesseract
import io

def debug_form_fields():
    """Look for the actual OCC field area"""
    pdf_path = "/Users/John/Desktop/FITREP Test/fitrepPdf.pdf"
    doc = fitz.open(pdf_path)
    
    # Check all pages for TR
    for page_num in range(min(len(doc), 3)):
        page = doc[page_num]
        
        print(f"=== PAGE {page_num + 1} ===")
        
        # Get text blocks
        blocks = page.get_text("blocks")
        
        print(f"Text blocks: {len(blocks)}")
        
        for i, block in enumerate(blocks):
            if len(block) >= 5:
                x0, y0, x1, y1, text = block[:5]
                text_clean = text.strip()
                
                # Look for blocks containing TR
                if 'TR' in text_clean and len(text_clean) < 100:
                    print(f"Block {i} contains TR: '{text_clean}'")
                    print(f"  Position: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
                
                # Also look for blocks with OCC patterns
                if 'OCC' in text_clean.upper() or any(code in text_clean for code in ['TR', 'AN', 'GC', 'DC']):
                    print(f"Block {i} relevant: '{text_clean}'")
                    print(f"  Position: ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f})")
        
        # Also try raw text
        text = page.get_text()
        lines = text.split('\n')
        
        print("\nLines containing TR:")
        for i, line in enumerate(lines):
            if 'TR' in line:
                print(f"Line {i}: '{line.strip()}'")
        
        print("\n" + "="*50 + "\n")
    
    doc.close()

if __name__ == "__main__":
    debug_form_fields()