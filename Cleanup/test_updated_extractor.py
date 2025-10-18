#!/usr/bin/env python3
"""
Test the updated fitrep_extractor27.py with text-based checkbox detection
"""

import sys
from pathlib import Path

# Add current directory to path to import from our modules
sys.path.append(str(Path(__file__).parent))

from fitrep_extractor27 import FITREPExtractor

def test_updated_extractor():
    """Test the updated extractor"""
    pdf_path = Path("/Users/John/Desktop/FITREP Test/fitrepPdf.pdf")
    
    print("=== Testing Updated FITREP Extractor ===\n")
    
    extractor = FITREPExtractor()
    data = extractor.extract_from_pdf(pdf_path)
    
    if data:
        print("\n=== UPDATED EXTRACTOR RESULTS ===")
        print(f"Last Name: {data.get('last_name', 'N/A')}")
        print(f"Grade: {data.get('grade', 'N/A')}")
        print(f"To Date: {data.get('to_date', 'N/A')}")
        print(f"Page 2 values: {data.get('page2_values', 'N/A')}")
        print(f"Page 3 values: {data.get('page3_values', 'N/A')}")
        print(f"Page 4 values: {data.get('page4_values', 'N/A')}")
        
        # Format complete CSV row
        row = [
            data.get('last_name', ''),
            data.get('grade', ''),
            data.get('occ', ''),
            data.get('to_date', '')
        ]
        row.extend(data.get('page2_values', [''] * 5))
        row.extend(data.get('page3_values', [''] * 5))
        row.extend(data.get('page4_values', [''] * 4))
        
        print(f"\nComplete CSV row: {row}")
        print(f"Total fields: {len(row)}")
        print("\n✅ Updated extractor working successfully!")
        
        return True
    else:
        print("❌ Updated extractor failed to extract data")
        return False

if __name__ == "__main__":
    success = test_updated_extractor()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")