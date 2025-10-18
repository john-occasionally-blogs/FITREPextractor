#!/usr/bin/env python3
"""
Extract full page text to see if we can find column headers
"""

import fitz

def extract_page_text():
    """Extract full text from each page to look for column headers"""
    pdf_path = "/Users/John/Desktop/FITREP Test/fitrepPdf.pdf"
    doc = fitz.open(pdf_path)
    
    for page_num in [1, 2, 3]:  # Pages 2, 3, 4
        if page_num >= len(doc):
            continue
            
        page = doc[page_num]
        
        print(f"=== PAGE {page_num + 1} FULL TEXT ===")
        text = page.get_text()
        
        # Look for lines that might contain column headers
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Look for lines with A B C D E pattern or similar
            if ('A' in line_stripped and 'B' in line_stripped and 
                'C' in line_stripped and 'D' in line_stripped):
                print(f"Potential header line {i}: '{line_stripped}'")
                
                # Also show the next few lines for context
                for j in range(1, 4):
                    if i + j < len(lines):
                        context_line = lines[i + j].strip()
                        if context_line:
                            print(f"  +{j}: '{context_line}'")
                print()
            
            # Also look for lines with just single letters
            if line_stripped in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                print(f"Single letter line {i}: '{line_stripped}'")
        
        print("\n" + "="*50 + "\n")
    
    doc.close()

if __name__ == "__main__":
    extract_page_text()