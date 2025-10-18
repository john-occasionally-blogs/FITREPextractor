#!/usr/bin/env python3
"""
Text-based checkbox extractor - cleaner approach using direct PDF text extraction
"""

import fitz  # PyMuPDF
import re
from collections import defaultdict

def extract_checkbox_values_text_based(pdf_path, page_num, expected_count):
    """Extract checkbox values using direct text extraction"""
    doc = fitz.open(pdf_path)
    
    if page_num >= len(doc):
        doc.close()
        return [4] * expected_count  # Default values
    
    page = doc[page_num]
    
    # Get structured text with position information
    text_dict = page.get_text("dict")
    
    # Extract all text blocks with their positions
    text_blocks = []
    
    def extract_blocks(obj, x_offset=0, y_offset=0):
        """Recursively extract text blocks with positions"""
        if isinstance(obj, dict):
            if obj.get("type") == "char":
                # Character level - record position and text
                char_info = {
                    "text": obj.get("c", ""),
                    "x": obj.get("origin", [0, 0])[0] + x_offset,
                    "y": obj.get("origin", [0, 0])[1] + y_offset,
                    "size": obj.get("size", 12)
                }
                text_blocks.append(char_info)
            elif "bbox" in obj:
                # Block with bbox - update offset
                bbox = obj["bbox"]
                new_x_offset = x_offset + bbox[0] if bbox else x_offset
                new_y_offset = y_offset + bbox[1] if bbox else y_offset
                
                for key, value in obj.items():
                    if key not in ["bbox", "type"]:
                        if isinstance(value, (list, tuple)):
                            for item in value:
                                extract_blocks(item, new_x_offset, new_y_offset)
                        else:
                            extract_blocks(value, new_x_offset, new_y_offset)
            else:
                # Other dict - recurse through values
                for value in obj.values():
                    if isinstance(value, (list, tuple)):
                        for item in value:
                            extract_blocks(item, x_offset, y_offset)
                    else:
                        extract_blocks(value, x_offset, y_offset)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                extract_blocks(item, x_offset, y_offset)
    
    extract_blocks(text_dict)
    
    # Filter for X marks only
    x_marks = [block for block in text_blocks if block["text"].strip().upper() == "X"]
    
    print(f"  Found {len(x_marks)} X marks on page {page_num + 1}")
    for i, x_mark in enumerate(x_marks):
        print(f"    X {i+1}: position ({x_mark['x']:.1f}, {x_mark['y']:.1f})")
    
    # Sort X marks by Y position (top to bottom) to group into rows
    x_marks.sort(key=lambda x: x["y"])
    
    # Group X marks into rows based on Y position
    rows = []
    if x_marks:
        current_row = [x_marks[0]]
        
        for x_mark in x_marks[1:]:
            # If Y position is close to current row (within ~15 points), add to same row
            if abs(x_mark["y"] - current_row[0]["y"]) < 15:
                current_row.append(x_mark)
            else:
                # Start new row
                rows.append(current_row)
                current_row = [x_mark]
        
        # Add the last row
        if current_row:
            rows.append(current_row)
    
    print(f"  Grouped into {len(rows)} rows: {[len(row) for row in rows]}")
    
    # Extract values for each expected row
    values = []
    
    for row_idx in range(expected_count):
        if row_idx < len(rows):
            row_x_marks = rows[row_idx]
            
            if len(row_x_marks) == 1:
                # Single X mark in this row - determine column based on X position
                x_pos = row_x_marks[0]["x"]
                
                # Convert X position to column value (1-8 scale)
                # Based on standard FITREP layout: A(1), B(2), C(3), D(4), E(5), F(6), G(7), H(8)
                # These boundaries need calibration based on actual FITREP layout
                if x_pos < 100:
                    column = 8  # H (rightmost visually appears leftmost in extracted coordinates)
                elif x_pos < 150:
                    column = 7  # G
                elif x_pos < 200:
                    column = 6  # F
                elif x_pos < 250:
                    column = 5  # E
                elif x_pos < 300:
                    column = 4  # D
                elif x_pos < 350:
                    column = 3  # C
                elif x_pos < 400:
                    column = 2  # B
                else:
                    column = 1  # A
                
                values.append(column)
                print(f"    Row {row_idx + 1}: X at x={x_pos:.1f} -> Column {chr(64 + column)} (value {column})")
            
            elif len(row_x_marks) > 1:
                # Multiple X marks - might be noise, take the leftmost
                leftmost = min(row_x_marks, key=lambda x: x["x"])
                x_pos = leftmost["x"]
                
                # Same mapping as above
                if x_pos < 100:
                    column = 8
                elif x_pos < 150:
                    column = 7
                elif x_pos < 200:
                    column = 6
                elif x_pos < 250:
                    column = 5
                elif x_pos < 300:
                    column = 4
                elif x_pos < 350:
                    column = 3
                elif x_pos < 400:
                    column = 2
                else:
                    column = 1
                
                values.append(column)
                print(f"    Row {row_idx + 1}: Multiple X marks, using leftmost at x={x_pos:.1f} -> Column {chr(64 + column)} (value {column})")
            
            else:
                # No X marks in this row
                values.append(4)  # Default to D
                print(f"    Row {row_idx + 1}: No X marks, using default value 4")
        else:
            # No row data for this index
            values.append(4)  # Default to D
            print(f"    Row {row_idx + 1}: No row found, using default value 4")
    
    doc.close()
    return values

def test_extraction():
    """Test the text-based extraction on sample PDF"""
    pdf_path = "/Users/John/Desktop/FITREP Test/fitrepPdf.pdf"
    
    print("Testing text-based checkbox extraction:\n")
    
    # Test page 2 (index 1) - should have 5 checkboxes
    print("Page 2 (5 expected checkboxes):")
    page2_values = extract_checkbox_values_text_based(pdf_path, 1, 5)
    print(f"Result: {page2_values}\n")
    
    # Test page 3 (index 2) - should have 5 checkboxes  
    print("Page 3 (5 expected checkboxes):")
    page3_values = extract_checkbox_values_text_based(pdf_path, 2, 5)
    print(f"Result: {page3_values}\n")
    
    # Test page 4 (index 3) - should have 4 checkboxes
    print("Page 4 (4 expected checkboxes):")  
    page4_values = extract_checkbox_values_text_based(pdf_path, 3, 4)
    print(f"Result: {page4_values}\n")
    
    return page2_values, page3_values, page4_values

if __name__ == "__main__":
    test_extraction()