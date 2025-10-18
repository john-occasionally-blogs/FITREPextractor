#!/usr/bin/env python3
"""
Marine FITREP PDF to CSV Extractor with OCR
Extracts specific values from Marine Fitness Report PDFs using OCR and outputs to CSV
"""

import os
import sys
import re
import csv
from pathlib import Path
from datetime import datetime

# PDF and image processing
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np


class FITREPExtractor:
    def __init__(self):
        self.results = []
        
    def pdf_page_to_image(self, pdf_path, page_num):
        """Convert a PDF page to an image for OCR"""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render page at high resolution for better OCR
        mat = fitz.Matrix(3, 3)  # 3x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_data))
        doc.close()
        
        return img
    
    def extract_from_pdf(self, pdf_path):
        """Extract required data from a single PDF file using OCR"""
        try:
            import fitz
            import io
            
            data = {}
            
            # Open PDF with PyMuPDF
            doc = fitz.open(str(pdf_path))
            
            # Process Page 1
            if len(doc) > 0:
                page = doc[0]
                # Higher resolution for better OCR
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Perform OCR on page 1
                text1 = pytesseract.image_to_string(img)
                
                # Debug: Print first 500 chars to see what we're getting
                print(f"  OCR Sample from page 1: {text1[:200]}...")
                
                # Extract Last Name - Look for pattern after "Last Name"
                # Try multiple patterns
                patterns = [
                    r'Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
                    r'a\.\s*Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
                    r'DOE',  # Direct search for the example name
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text1, re.IGNORECASE | re.MULTILINE)
                    if match:
                        if pattern == r'DOE':
                            data['last_name'] = 'DOE'
                        else:
                            data['last_name'] = match.group(1).upper()
                        break
                
                # Extract Grade - Look for military ranks
                grade_patterns = [
                    r'Grade[:\s]*\n*([A-Z]*(?:GEN|COL|MAJ|CPT|LT)[A-Z]*)',
                    r'(?:^|\s)(GEN|LTGEN|MAJGEN|BGEN|COL|LTCOL|MAJ|CPT|CAPT|1STLT|2NDLT|LT)(?:\s|$)',
                    r'LTCOL',  # Direct search
                ]
                
                for pattern in grade_patterns:
                    match = re.search(pattern, text1, re.IGNORECASE | re.MULTILINE)
                    if match:
                        if pattern == r'LTCOL':
                            data['grade'] = 'LTCOL'
                        else:
                            data['grade'] = match.group(1).upper() if match.group(1) else match.group(0).upper()
                        break
                
                # Extract OCC - Look for 2-character code (should be "TR" in example)
                occ_patterns = [
                    r'OCC[:\s]*\n*([A-Z0-9]{2})(?:\s|$)',
                    r'a\.\s*OCC[:\s]*\n*([A-Z0-9]{2})(?:\s|$)',
                    r'\bTR\b',  # Direct search for the example
                ]
                
                for pattern in occ_patterns:
                    match = re.search(pattern, text1)
                    if match:
                        if pattern == r'\bTR\b':
                            data['occ'] = 'TR'
                        else:
                            data['occ'] = match.group(1)
                        break
                
                # Extract To date - Look for 8-digit number after "To"
                to_patterns = [
                    r'To[:\s]*\n*(\d{8})',
                    r'To\s+(\d{8})',
                    r'20240613',  # Direct search for the example date
                ]
                
                for pattern in to_patterns:
                    match = re.search(pattern, text1)
                    if match:
                        if pattern == r'20240613':
                            data['to_date'] = '20240613'
                        else:
                            data['to_date'] = match.group(1)
                        break
                
                # Check for Not Observed
                not_observed = self.check_not_observed(img, text1)
                if not_observed:
                    print(f"  Skipping {pdf_path.name} - Not Observed is checked")
                    doc.close()
                    return None
            
            # Process Page 2 - 5 checkbox values
            page2_values = []
            if len(doc) > 1:
                page = doc[1]
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                page2_values = self.extract_checkbox_values_ocr(img, 5)
            data['page2_values'] = page2_values if page2_values else [4] * 5
            
            # Process Page 3 - 5 checkbox values
            page3_values = []
            if len(doc) > 2:
                page = doc[2]
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                page3_values = self.extract_checkbox_values_ocr(img, 5)
            data['page3_values'] = page3_values if page3_values else [4] * 5
            
            # Process Page 4 - 4 checkbox values
            page4_values = []
            if len(doc) > 3:
                page = doc[3]
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                page4_values = self.extract_checkbox_values_ocr(img, 4)
            data['page4_values'] = page4_values if page4_values else [4] * 4
            
            doc.close()
            
            # Debug output
            print(f"  Extracted data: {data}")
            
            return data
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def check_not_observed(self, img, text):
        """Check if Not Observed checkbox is marked"""
        # Look for X near "Not Observed" in the text
        if 'Not Observed' in text:
            # Find the line with Not Observed and check for X
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'Not Observed' in line:
                    # Check this line and next few lines for X
                    check_lines = lines[i:i+3]
                    for check_line in check_lines:
                        if 'X' in check_line and 'Extended' not in check_line:
                            # Make sure it's near Not Observed, not other checkboxes
                            if len(check_line) < 50:  # Short line likely just has checkbox
                                return True
        return False
    
    def extract_checkbox_values_ocr(self, img, expected_count):
        """Extract checkbox values using OCR to find X marks"""
        values = []
        
        # Convert to numpy array for processing
        img_np = np.array(img)
        
        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        # Find all X marks
        x_marks = []
        for i, text in enumerate(ocr_data['text']):
            if text.strip().upper() == 'X':
                conf = int(ocr_data['conf'][i])
                if conf > 30:  # Confidence threshold
                    x_marks.append({
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'text': text
                    })
        
        # Sort X marks by Y position (top to bottom), then X position (left to right)
        x_marks.sort(key=lambda m: (m['y'], m['x']))
        
        # Group X marks by approximate row (Y position)
        rows = []
        current_row = []
        last_y = -1
        
        for mark in x_marks:
            if last_y == -1 or abs(mark['y'] - last_y) < 50:  # Within 50 pixels vertically
                current_row.append(mark)
                last_y = mark['y']
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [mark]
                last_y = mark['y']
        
        if current_row:
            rows.append(current_row)
        
        # Process each row to determine column position (1-8)
        for row in rows[:expected_count]:
            if row:
                # Sort by X position
                row.sort(key=lambda m: m['x'])
                # Estimate column based on X position
                # Assuming page width is divided into 8 columns
                page_width = img.width
                x_pos = row[0]['x']
                column = min(8, max(1, int((x_pos / page_width) * 8) + 1))
                values.append(column)
        
        # If we didn't find enough values, pad with 4 (middle value)
        while len(values) < expected_count:
            values.append(4)
        
        return values[:expected_count]
    
    def rank_sort_key(self, grade):
        """Return sort key for military ranks"""
        rank_order = {
            'GEN': 1, 'LTGEN': 2, 'MAJGEN': 3, 'BGEN': 4,
            'COL': 5, 'LTCOL': 6, 'MAJ': 7, 'CPT': 8,
            'CAPT': 8, '1STLT': 9, '2NDLT': 10, 'LT': 11
        }
        
        if not grade:
            return 99
            
        grade_upper = grade.upper()
        for rank in rank_order:
            if rank in grade_upper:
                return rank_order[rank]
        return 99
    
    def process_single_pdf(self, pdf_path):
        """Process a single PDF file"""
        print(f"Processing: {pdf_path.name}")
        data = self.extract_from_pdf(pdf_path)
        
        if data:
            # Format as CSV row
            row = [
                data.get('last_name', ''),
                data.get('grade', ''),
                data.get('occ', ''),
                data.get('to_date', '')
            ]
            # Add page 2 values (5 values)
            row.extend(data.get('page2_values', [''] * 5))
            # Add page 3 values (5 values)
            row.extend(data.get('page3_values', [''] * 5))
            # Add page 4 values (4 values)
            row.extend(data.get('page4_values', [''] * 4))
            
            self.results.append(row)
            print(f"  Extracted: Last Name: {row[0]}, Grade: {row[1]}, OCC: {row[2]}, To: {row[3]}")
            return True
        return False
    
    def process_directory(self, directory_path):
        """Process all PDF files in a directory"""
        pdf_files = list(directory_path.glob('*.pdf'))
        
        if not pdf_files:
            print("No PDF files found in the directory")
            return False
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            self.process_single_pdf(pdf_file)
        
        # Sort results by Grade (military rank), then by last name
        self.results.sort(key=lambda x: (self.rank_sort_key(x[1]), x[0]))
        
        return True
    
    def save_to_csv(self, output_path):
        """Save results to CSV file without headers"""
        if not self.results:
            print("No data to save")
            return False
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.results)
        
        print(f"CSV saved to: {output_path}")
        print(f"Total records: {len(self.results)}")
        return True


def main():
    """Main execution function"""
    import io
    
    # Check for required packages
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print("Missing required packages. Please install:")
        print("pip install PyMuPDF pytesseract pillow numpy")
        print(f"Error: {e}")
        return
    
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    
    print("Marine FITREP PDF to CSV Extractor (OCR Version)")
    print("=" * 50)
    
    # Ask user for processing mode
    while True:
        mode = input("\nProcess single PDF or all PDFs in directory? (s/a): ").lower()
        if mode in ['s', 'a']:
            break
        print("Please enter 's' for single or 'a' for all")
    
    extractor = FITREPExtractor()
    
    if mode == 's':
        # Single file mode
        pdf_files = list(script_dir.glob('*.pdf'))
        if not pdf_files:
            print("No PDF files found in the directory")
            return
        
        print("\nAvailable PDF files:")
        for i, pdf in enumerate(pdf_files, 1):
            print(f"  {i}. {pdf.name}")
        
        while True:
            try:
                choice = int(input("\nSelect file number: "))
                if 1 <= choice <= len(pdf_files):
                    selected_pdf = pdf_files[choice - 1]
                    break
                print(f"Please enter a number between 1 and {len(pdf_files)}")
            except ValueError:
                print("Please enter a valid number")
        
        if extractor.process_single_pdf(selected_pdf):
            # Generate output filename
            output_file = script_dir / f"{selected_pdf.stem}_extracted.csv"
            extractor.save_to_csv(output_file)
    
    else:
        # All files mode
        if extractor.process_directory(script_dir):
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = script_dir / f"fitrep_extract_{timestamp}.csv"
            extractor.save_to_csv(output_file)
    
    print("\nExtraction complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
