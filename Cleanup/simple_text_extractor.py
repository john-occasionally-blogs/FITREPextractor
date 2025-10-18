#!/usr/bin/env python3
"""
Simplified text-based checkbox extractor using text blocks
"""

import fitz  # PyMuPDF

def extract_checkbox_values_simple(pdf_path, page_num, expected_count):
    """Extract checkbox values using simple text blocks"""
    doc = fitz.open(pdf_path)
    
    if page_num >= len(doc):
        doc.close()
        return [4] * expected_count
    
    page = doc[page_num]
    
    # Get text blocks with position info
    blocks = page.get_text("blocks")
    
    print(f"\n  Page {page_num + 1} has {len(blocks)} text blocks")
    
    # Find X marks - they should be in separate blocks
    x_marks = []
    
    for i, block in enumerate(blocks):
        # block is a tuple: (x0, y0, x1, y1, "text content", block_no, block_type)
        if len(block) >= 5:
            x0, y0, x1, y1, text = block[:5]
            
            # Look for isolated X marks
            text_clean = text.strip()
            if text_clean == "X":
                x_marks.append({
                    "text": text_clean,
                    "x": (x0 + x1) / 2,  # Center X
                    "y": (y0 + y1) / 2,  # Center Y
                    "block_idx": i
                })
                print(f"    Found X at block {i}: position ({(x0+x1)/2:.1f}, {(y0+y1)/2:.1f})")
    
    print(f"  Total X marks found: {len(x_marks)}")
    
    if not x_marks:
        print(f"  No X marks found, using defaults")
        doc.close()
        return [4] * expected_count
    
    # Sort by Y position (top to bottom)
    x_marks.sort(key=lambda x: x["y"])
    
    # Group into rows by Y position
    rows = []
    if x_marks:
        current_row = [x_marks[0]]
        
        for x_mark in x_marks[1:]:
            # If Y positions are close (within 20 points), same row
            if abs(x_mark["y"] - current_row[0]["y"]) < 20:
                current_row.append(x_mark)
            else:
                rows.append(current_row)
                current_row = [x_mark]
        
        if current_row:
            rows.append(current_row)
    
    print(f"  Grouped into {len(rows)} rows")
    
    # Convert to values
    values = []
    
    for row_idx in range(expected_count):
        if row_idx < len(rows):
            row_x_marks = rows[row_idx]
            
            if row_x_marks:
                # Use the first (or only) X mark in the row
                x_pos = row_x_marks[0]["x"]
                
                # Map X position to column (1-8)
                # These boundaries may need adjustment based on actual FITREP layout
                if x_pos < 150:
                    column = 8  # H 
                elif x_pos < 200:
                    column = 7  # G
                elif x_pos < 250:
                    column = 6  # F
                elif x_pos < 300:
                    column = 5  # E
                elif x_pos < 350:
                    column = 4  # D
                elif x_pos < 400:
                    column = 3  # C
                elif x_pos < 450:
                    column = 2  # B
                else:
                    column = 1  # A
                
                values.append(column)
                print(f"    Row {row_idx + 1}: X at x={x_pos:.1f} -> Column {chr(64 + column)} (value {column})")
            else:
                values.append(4)  # Default
                print(f"    Row {row_idx + 1}: No X mark, default value 4")
        else:
            values.append(4)  # Default
            print(f"    Row {row_idx + 1}: No row data, default value 4")
    
    doc.close()
    return values

def test_simple_extraction():
    """Test the simplified extraction"""
    pdf_path = "/Users/John/Desktop/FITREP Test/fitrepPdf.pdf"
    
    print("Testing simplified text-based checkbox extraction:")
    
    # Test all pages
    print("\n=== Page 2 (5 expected) ===")
    page2_values = extract_checkbox_values_simple(pdf_path, 1, 5)
    print(f"Result: {page2_values}")
    
    print("\n=== Page 3 (5 expected) ===")  
    page3_values = extract_checkbox_values_simple(pdf_path, 2, 5)
    print(f"Result: {page3_values}")
    
    print("\n=== Page 4 (4 expected) ===")
    page4_values = extract_checkbox_values_simple(pdf_path, 3, 4)
    print(f"Result: {page4_values}")
    
    return page2_values, page3_values, page4_values

if __name__ == "__main__":
    test_simple_extraction()