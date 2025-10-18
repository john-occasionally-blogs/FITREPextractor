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
                
                # Get OCR with position data for better extraction
                ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                
                # Extract Last Name - Generic pattern without specific names
                patterns = [
                    r'Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
                    r'a\.\s*Last\s+Name[:\s]*\n*([A-Z][A-Z]+)',
                    r'Last Name.*?\n\s*([A-Z]+)',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text1, re.IGNORECASE | re.MULTILINE)
                    if match:
                        data['last_name'] = match.group(1).upper()
                        break
                
                # Extract FIRST Grade only - look in top 1/3 of page
                # Split text into lines and only look at first portion
                lines = text1.split('\n')
                top_third_lines = lines[:len(lines)//3]
                top_third_text = '\n'.join(top_third_lines)
                
                # Valid military grades
                valid_grades = ['SGT', 'SSGT', 'GYSGT', 'MGYSGT', '1STSGT', 'SGTMAJ', 
                               '2NDLT', '1STLT', 'CAPT', 'MAJ', 'LTCOL', 'COL', 
                               'WO', 'CWO2', 'CWO3', 'CWO4', 'CWO5', 
                               'BGEN', 'MAJGEN', 'LTGEN', 'GEN']
                
                # Look for Grade label and then find valid grade
                grade_value = None
                
                # Method 1: Look near Grade label in OCR data
                for i, word in enumerate(ocr_data['text']):
                    if 'Grade' in word and ocr_data['top'][i] < img.height // 3:  # Top third
                        # Look at next few words for a valid grade
                        for j in range(1, min(15, len(ocr_data['text']) - i)):
                            next_word = ocr_data['text'][i + j].strip().upper()
                            if next_word in valid_grades:
                                grade_value = next_word
                                break
                            # Check for partial matches (OCR might split words)
                            for grade in valid_grades:
                                if grade in next_word or next_word in grade:
                                    grade_value = grade
                                    break
                        if grade_value:
                            break
                
                # Method 2: Pattern search in top third text
                if not grade_value:
                    for grade in valid_grades:
                        pattern = r'\b' + re.escape(grade) + r'\b'
                        if re.search(pattern, top_third_text, re.IGNORECASE):
                            grade_value = grade
                            break
                
                if grade_value:
                    data['grade'] = grade_value
                else:
                    print("  Warning: Could not find valid Grade")
                
                # Extract OCC - Look for 2-letter code specifically near "OCC" label
                # Valid OCC codes provided by user
                valid_occ_codes = ['GC', 'DC', 'CH', 'TR', 'CD', 'TD', 'FD', 'EN', 'CS', 'AN', 'AR', 'SA', 'RT']
                
                occ_value = None
                
                # First try: Look in OCR data near OCC label
                for i, word in enumerate(ocr_data['text']):
                    if 'OCC' in word.upper():
                        # Look at next several words for a 2-character value
                        for j in range(1, min(15, len(ocr_data['text']) - i)):
                            next_word = ocr_data['text'][i + j].strip().upper()
                            # Check if it's in our valid codes list
                            if next_word in valid_occ_codes:
                                occ_value = next_word
                                break
                            # Also check if it's 2 characters that might be misread
                            if len(next_word) == 2:
                                # Check for common OCR mistakes (0 vs O, etc)
                                corrected = next_word.replace('0', 'O').replace('1', 'I').replace('5', 'S')
                                if corrected in valid_occ_codes:
                                    occ_value = corrected
                                    break
                        if occ_value:
                            break
                
                # Second try: Direct search for valid codes in the top portion
                if not occ_value:
                    # Look in the top half of the page for any valid code
                    for code in valid_occ_codes:
                        pattern = r'\b' + re.escape(code) + r'\b'
                        if re.search(pattern, text1[:len(text1)//2]):
                            occ_value = code
                            break
                
                if occ_value:
                    data['occ'] = occ_value
                else:
                    print("  Warning: Could not find valid OCC code")
                
                # Extract To date - MUST be the second date after From
                # Look for From and To pattern together
                to_value = None
                
                # Method 1: Find "From" and "To" labels and get dates near them
                from_index = -1
                to_index = -1
                for i, word in enumerate(ocr_data['text']):
                    if word.upper() == 'FROM':
                        from_index = i
                    elif word.upper() == 'TO' and from_index != -1:
                        to_index = i
                        # Look for 8-digit number after TO
                        for j in range(1, min(10, len(ocr_data['text']) - i)):
                            next_word = ocr_data['text'][i + j].strip()
                            if re.match(r'^\d{8}$', next_word):
                                to_value = next_word
                                break
                        if to_value:
                            break
                
                # Method 2: Find pattern with From and To dates
                if not to_value:
                    # Look for two consecutive 8-digit dates
                    date_pattern = r'From[:\s]*(\d{8})[:\s]*To[:\s]*(\d{8})'
                    match = re.search(date_pattern, text1, re.IGNORECASE)
                    if match:
                        to_value = match.group(2)
                    else:
                        # Look for any pattern with two 8-digit numbers close together
                        two_dates = re.findall(r'\d{8}', text1)
                        if len(two_dates) >= 2:
                            # The second one should be the To date
                            to_value = two_dates[1]
                
                if to_value:
                    data['to_date'] = to_value
                else:
                    print("  Warning: Could not find To date")
                
                # Check for Not Observed
                not_observed = self.check_not_observed(img, text1)
                if not_observed:
                    print("  Skipping {0} - Not Observed is checked".format(pdf_path.name))
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
                page2_values = self.extract_checkbox_values_with_headers(img, 5)
            data['page2_values'] = page2_values if page2_values else [4] * 5
            
            # Process Page 3 - 5 checkbox values
            page3_values = []
            if len(doc) > 2:
                page = doc[2]
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                page3_values = self.extract_checkbox_values_with_headers(img, 5)
            data['page3_values'] = page3_values if page3_values else [4] * 5
            
            # Process Page 4 - 4 checkbox values
            page4_values = []
            if len(doc) > 3:
                page = doc[3]
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                page4_values = self.extract_checkbox_values_with_headers(img, 4)
            data['page4_values'] = page4_values if page4_values else [4] * 4
            
            doc.close()
            
            # Debug output - using format instead of f-strings
            print("  Extracted - Last Name: {0}, Grade: {1}, OCC: {2}, To: {3}".format(
                data.get('last_name'), data.get('grade'), data.get('occ'), data.get('to_date')))
            print("  Page 2 values: {0}".format(data.get('page2_values')))
            print("  Page 3 values: {0}".format(data.get('page3_values')))
            print("  Page 4 values: {0}".format(data.get('page4_values')))
            
            return data
            
        except Exception as e:
            print("Error processing {0}: {1}".format(pdf_path, str(e)))
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
    
    def extract_checkbox_values_with_headers(self, img, expected_count):
        """Extract checkbox values using A-H headers to identify columns"""
        values = []
        
        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        # Debug: Let's see what we're finding
        print("    Looking for checkboxes (expecting {0} values)".format(expected_count))
        
        # Find all instances of single letters A through H (column headers)
        headers = {'A': [], 'B': [], 'C': [], 'D': [], 'E': [], 'F': [], 'G': [], 'H': []}
        
        for i, text in enumerate(ocr_data['text']):
            text_clean = text.strip().upper()
            if text_clean in headers.keys() and len(text.strip()) == 1:
                conf = int(ocr_data['conf'][i])
                if conf > 20:  # Lower threshold for headers
                    headers[text_clean].append({
                        'x': ocr_data['left'][i] + ocr_data['width'][i] // 2,
                        'y': ocr_data['top'][i],
                        'conf': conf
                    })
        
        # Find the most likely header row (should have most letters in a line)
        best_header_row = None
        best_header_y = None
        max_headers_found = 0
        
        # Group potential headers by Y position
        y_groups = {}
        for letter, positions in headers.items():
            for pos in positions:
                y_range = pos['y'] // 30 * 30  # Group by 30-pixel ranges
                if y_range not in y_groups:
                    y_groups[y_range] = {}
                if letter not in y_groups[y_range]:
                    y_groups[y_range][letter] = pos
        
        # Find the Y range with the most unique letters
        for y_range, letters in y_groups.items():
            if len(letters) > max_headers_found:
                max_headers_found = len(letters)
                best_header_row = letters
                best_header_y = y_range
        
        print("    Found {0} column headers at Y position ~{1}".format(max_headers_found, best_header_y))
        
        # Now find all X marks
        x_marks = []
        for i, text in enumerate(ocr_data['text']):
            text_clean = text.strip().upper()
            # Look for X, but also handle common OCR misreads of X
            if text_clean in ['X', 'K', '*', 'x'] or (len(text_clean) == 1 and text_clean.isalpha()):
                # Check if this might be an X by looking at confidence and context
                conf = int(ocr_data['conf'][i])
                y_pos = ocr_data['top'][i]
                
                # Make sure it's below the headers and has reasonable confidence
                if best_header_y and y_pos > best_header_y + 50 and conf > 20:
                    # Check if it's likely an X (not A-H which are headers)
                    if text_clean == 'X' or (text_clean not in ['A','B','C','D','E','F','G','H']):
                        x_marks.append({
                            'x': ocr_data['left'][i] + ocr_data['width'][i] // 2,
                            'y': y_pos,
                            'text': text_clean,
                            'conf': conf
                        })
        
        print("    Found {0} potential X marks".format(len(x_marks)))
        
        # Sort X marks by Y position (top to bottom)
        x_marks.sort(key=lambda m: m['y'])
        
        # Group X marks into rows
        x_rows = []
        current_row = []
        last_y = -1
        
        for mark in x_marks:
            if last_y == -1 or abs(mark['y'] - last_y) < 40:  # Same row if within 40 pixels
                current_row.append(mark)
            else:
                if current_row:
                    x_rows.append(current_row)
                current_row = [mark]
            last_y = mark['y']
        
        if current_row:
            x_rows.append(current_row)
        
        print("    Grouped into {0} rows of X marks".format(len(x_rows)))
        
        # For each row of X marks, determine which column it belongs to
        for row_idx, x_row in enumerate(x_rows[:expected_count]):
            if best_header_row:
                # Find the X mark in this row (should be only one per row)
                if x_row:
                    # Use the first (or most confident) X in the row
                    best_x = max(x_row, key=lambda m: m['conf'])
                    
                    # Find which column this X belongs to
                    min_distance = float('inf')
                    best_column = 4  # Default
                    
                    for letter, header_pos in best_header_row.items():
                        distance = abs(best_x['x'] - header_pos['x'])
                        if distance < min_distance:
                            min_distance = distance
                            best_column = ord(letter) - ord('A') + 1  # A=1, B=2, etc.
                    
                    # Only accept if X is reasonably aligned with a column
                    if min_distance < 150:  # Within 150 pixels of a header
                        values.append(best_column)
                        print("      Row {0}: X at position {1} -> Column {2}".format(
                            row_idx + 1, best_x['x'], best_column))
                    else:
                        values.append(4)  # Default if too far from any header
                        print("      Row {0}: X too far from headers, using default".format(row_idx + 1))
        
        # Pad with default values if needed
        while len(values) < expected_count:
            values.append(4)
            print("      Row {0}: No X found, using default value 4".format(len(values)))
        
        return values[:expected_count]
    
    def rank_sort_key(self, grade):
        """Return sort key for military ranks"""
        rank_order = {
            'GEN': 1, 'LTGEN': 2, 'MAJGEN': 3, 'BGEN': 4,
            'COL': 5, 'LTCOL': 6, 'MAJ': 7, 'CAPT': 8,
            '1STLT': 9, '2NDLT': 10,
            'CWO5': 11, 'CWO4': 12, 'CWO3': 13, 'CWO2': 14, 'WO': 15,
            'SGTMAJ': 16, '1STSGT': 17, 'MGYSGT': 18, 'GYSGT': 19, 
            'SSGT': 20, 'SGT': 21
        }
        
        if not grade:
            return 99
            
        grade_upper = grade.upper()
        if grade_upper in rank_order:
            return rank_order[grade_upper]
        return 99
    
    def process_single_pdf(self, pdf_path):
        """Process a single PDF file"""
        print("\nProcessing: {0}".format(pdf_path.name))
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
            return True
        return False
    
    def process_directory(self, directory_path):
        """Process all PDF files in a directory"""
        pdf_files = list(directory_path.glob('*.pdf'))
        
        if not pdf_files:
            print("No PDF files found in the directory")
            return False
        
        print("Found {0} PDF files".format(len(pdf_files)))
        
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
        
        print("\nCSV saved to: {0}".format(output_path))
        print("Total records: {0}".format(len(self.results)))
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
        print("Error: {0}".format(e))
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
            print("  {0}. {1}".format(i, pdf.name))
        
        while True:
            try:
                choice = int(input("\nSelect file number: "))
                if 1 <= choice <= len(pdf_files):
                    selected_pdf = pdf_files[choice - 1]
                    break
                print("Please enter a number between 1 and {0}".format(len(pdf_files)))
            except ValueError:
                print("Please enter a valid number")
        
        if extractor.process_single_pdf(selected_pdf):
            # Generate output filename
            output_file = script_dir / "{0}_extracted.csv".format(selected_pdf.stem)
            extractor.save_to_csv(output_file)
    
    else:
        # All files mode
        if extractor.process_directory(script_dir):
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = script_dir / "fitrep_extract_{0}.csv".format(timestamp)
            extractor.save_to_csv(output_file)
    
    print("\nExtraction complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        print("\nError: {0}".format(str(e)))
        import traceback
        traceback.print_exc()
        sys.exit(1)
