#!/usr/bin/env python3
"""
Analyze checkbox positions in detail to improve accuracy
"""

import fitz
from pathlib import Path

def analyze_checkbox_positions():
    """Analyze the exact positions and see what we can learn"""
    pdf_path = "/Users/John/Desktop/FITREP Test/fitrepPdf.pdf"
    doc = fitz.open(pdf_path)
    
    print("=== DETAILED CHECKBOX POSITION ANALYSIS ===\n")
    
    for page_num in [1, 2, 3]:  # Pages 2, 3, 4 (0-indexed: 1, 2, 3)
        if page_num >= len(doc):
            continue
            
        page = doc[page_num]
        blocks = page.get_text("blocks")
        
        print(f"PAGE {page_num + 1} ANALYSIS:")
        print(f"Total blocks: {len(blocks)}")
        
        # Find all X marks with detailed position info
        x_marks = []
        other_blocks = []
        
        for i, block in enumerate(blocks):
            if len(block) >= 5:
                x0, y0, x1, y1, text = block[:5]
                text_clean = text.strip()
                
                if text_clean == "X":
                    x_marks.append({
                        "block_idx": i,
                        "text": text_clean,
                        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                        "center_x": (x0 + x1) / 2,
                        "center_y": (y0 + y1) / 2,
                        "width": x1 - x0,
                        "height": y1 - y0
                    })
                else:
                    # Keep some other blocks for context
                    if text_clean and len(text_clean) < 50:
                        other_blocks.append({
                            "block_idx": i,
                            "text": text_clean,
                            "center_x": (x0 + x1) / 2,
                            "center_y": (y0 + y1) / 2
                        })
        
        print(f"Found {len(x_marks)} X marks:")
        for i, x_mark in enumerate(x_marks):
            print(f"  X {i+1}: center=({x_mark['center_x']:.1f}, {x_mark['center_y']:.1f}), "
                  f"bbox=({x_mark['x0']:.1f}, {x_mark['y0']:.1f}, {x_mark['x1']:.1f}, {x_mark['y1']:.1f})")
        
        # Look for column headers (A, B, C, D, E, F, G, H)
        print(f"\nLooking for column headers:")
        potential_headers = []
        for block in other_blocks:
            if block['text'] in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                potential_headers.append(block)
                print(f"  Found '{block['text']}' at x={block['center_x']:.1f}, y={block['center_y']:.1f}")
        
        # If we found headers, try to map X marks to columns
        if potential_headers:
            print(f"\nMapping X marks to columns based on headers:")
            potential_headers.sort(key=lambda h: h['center_x'])
            
            for i, x_mark in enumerate(x_marks):
                # Find closest header by X position
                distances = []
                for header in potential_headers:
                    distance = abs(x_mark['center_x'] - header['center_x'])
                    distances.append((distance, header['text']))
                
                if distances:
                    closest_distance, closest_header = min(distances)
                    column_value = ord(closest_header) - ord('A') + 1  # A=1, B=2, etc.
                    print(f"    X {i+1} at x={x_mark['center_x']:.1f} closest to '{closest_header}' "
                          f"(distance={closest_distance:.1f}) -> value {column_value}")
        
        # Also show the current mapping results
        print(f"\nCurrent text-based method results:")
        x_marks_sorted = sorted(x_marks, key=lambda x: x['center_y'])
        
        for i, x_mark in enumerate(x_marks_sorted):
            x_pos = x_mark['center_x']
            
            # Current mapping logic
            if x_pos < 150:
                current_value = 8  # H 
            elif x_pos < 200:
                current_value = 7  # G
            elif x_pos < 275:
                current_value = 6  # F
            elif x_pos < 350:
                current_value = 5  # E
            elif x_pos < 425:
                current_value = 4  # D
            elif x_pos < 500:
                current_value = 3  # C
            elif x_pos < 575:
                current_value = 2  # B
            else:
                current_value = 1  # A
            
            print(f"    Row {i+1}: X at x={x_pos:.1f} -> Current mapping: {current_value} ({chr(64 + current_value)})")
        
        print("\n" + "="*60 + "\n")
    
    doc.close()

if __name__ == "__main__":
    analyze_checkbox_positions()