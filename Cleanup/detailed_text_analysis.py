#!/usr/bin/env python3
"""
More detailed text analysis of checkbox pages
"""

import fitz

def detailed_analysis():
    """More thorough text analysis"""
    pdf_path = "/Users/John/Desktop/FITREP Test/fitrepPdf.pdf"
    doc = fitz.open(pdf_path)
    
    for page_num in [1, 2, 3]:  # Pages 2, 3, 4
        if page_num >= len(doc):
            continue
            
        page = doc[page_num]
        
        print(f"=== PAGE {page_num + 1} DETAILED ANALYSIS ===")
        
        # Method 1: Simple text extraction
        text = page.get_text()
        print(f"Simple text length: {len(text)}")
        if text.strip():
            print("Sample text:", repr(text[:200]))
        
        # Method 2: Text blocks
        blocks = page.get_text("blocks")
        print(f"Text blocks: {len(blocks)}")
        
        text_blocks = []
        for i, block in enumerate(blocks):
            if len(block) >= 5:
                x0, y0, x1, y1, text = block[:5]
                text_clean = text.strip()
                if text_clean:
                    text_blocks.append({
                        "idx": i,
                        "text": text_clean,
                        "x": (x0 + x1) / 2,
                        "y": (y0 + y1) / 2
                    })
        
        print(f"Non-empty text blocks: {len(text_blocks)}")
        for block in text_blocks:
            if len(block['text']) < 100:  # Don't show very long blocks
                print(f"  Block {block['idx']}: '{block['text']}' at ({block['x']:.1f}, {block['y']:.1f})")
        
        # Method 3: Text dictionary (structured)
        text_dict = page.get_text("dict")
        if "blocks" in text_dict:
            print(f"Dict blocks: {len(text_dict['blocks'])}")
        
        print("\n" + "="*50 + "\n")
    
    doc.close()

if __name__ == "__main__":
    detailed_analysis()