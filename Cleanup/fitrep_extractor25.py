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
import cv2


class FITREPExtractor:
    def __init__(self):
        self.results = []
        # Valid military grades
        self.valid_grades = [
            'SGT', 'SSGT', 'GYSGT', 'MSGT', 'MGYSGT', '1STSGT', 'SGTMAJ',
            '2NDLT', '1STLT', 'CAPT', 'MAJ', 'LTCOL', 'COL',
            'WO', 'CWO2', 'CWO3', 'CWO4', 'CWO5',
            'BGEN', 'MAJGEN', 'LTGEN', 'GEN'
        ]
        # Valid OCC codes
        self.valid_occ_codes = ['GC', 'DC', 'CH', 'TR', 'CD', 'TD', 'FD', 'EN', 'CS', 'AN', 'AR', 'SA', 'RT']
    
    def normalize_token(self, s):
        """Normalize a token for better matching"""
        return (s.strip().upper()
                .replace("0", "O")
                .replace("1", "I") 
                .replace("5", "S")
                .replace(":", "")
                .replace(".", ""))
    
    def find_label_indices(self, ocr_data, labels):
        """Find indices of labels in OCR data"""
        label_set = {self.normalize_token(l) for l in labels}
        indices = []
        for i, text in enumerate(ocr_data["text"]):
            if not text:
                continue
            if self.normalize_token(text) in label_set:
                indices.append(i)
        return indices
    
    def extract_from_pdf(self, pdf_path):
        """Extract required data from a single PDF file using OCR"""
        try:
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
                
                # Extract Last Name - Keep working patterns
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
                
                # Extract Grade - simplified approach looking for any valid grade in top third
                grade_value = None
                top_third_height = img.height // 3
                
                # Debug: Let's see what tokens we're finding near Grade
                print("  DEBUG Grade extraction:")
                
                # First find Grade label positions
                grade_positions = []
                for i, text in enumerate(ocr_data["text"]):
                    if text and "Grade" in text and ocr_data["top"][i] < top_third_height:
                        grade_positions.append(i)
                        print("    Found 'Grade' at index {0}, text: '{1}'".format(i, text))
                
                # Method 1: Look for valid grades anywhere in top third
                # Sometimes the grade isn't directly tied to the label due to OCR issues
                all_top_third_tokens = []
                for i, text in enumerate(ocr_data["text"]):
                    if text and ocr_data["top"][i] < top_third_height:
                        tok = self.normalize_token(text)
                        all_top_third_tokens.append(tok)
                        # Direct match
                        if tok in self.valid_grades:
                            print("    Found grade '{0}' at position {1}".format(tok, i))
                            if not grade_value:  # Take first valid grade found
                                grade_value = tok
                
                # Method 2: If no direct match, look for partial matches that might be grades
                if not grade_value:
                    # Check for grades that might be misread - based on actual OCR errors observed
                    grade_mapping = {
                        'MAJ': ['MAJ', 'MAS', 'MA', 'MJ', 'MAJOR', 'MAI', 'MAT'],
                        'LTCOL': ['LTCOL', 'LRCOL', 'LTCO', 'LTC', 'LTCL', 'LICOL', 'IRCOL'],
                        'MGYSGT': ['MGYSGT', 'MGYST', 'MGSG', 'MGYSG', 'MGYSGI'],
                        'MSGT': ['MSGT', 'SCR', 'MSG', 'MSGI', 'MST'],  # SCR is for MSGT not MGYSGT
                        'GYSGT': ['GYSGT', 'GYSG', 'GYST', 'GSGT'],
                        'SSGT': ['SSGT', 'SSG', 'SSGI', 'SST'],
                        '1STSGT': ['1STSGT', 'ISTSGT', '1STSG', 'ISTSG'],
                        'SGTMAJ': ['SGTMAJ', 'SGTMA', 'SGMAJ', 'SGTMAS'],
                    }
                    
                    for correct_grade, variations in grade_mapping.items():
                        if correct_grade in self.valid_grades:
                            for variation in variations:
                                if variation in all_top_third_tokens:
                                    print("    Found grade variation '{0}' mapping to '{1}'".format(variation, correct_grade))
                                    grade_value = correct_grade
                                    break
                            if grade_value:
                                break
                
                # Method 3: Look specifically near the first Grade label if found
                if not grade_value and grade_positions:
                    idx = grade_positions[0]
                    print("    Checking near first Grade label at index {0}".format(idx))
                    # Check next 30 tokens
                    for k in range(idx + 1, min(idx + 30, len(ocr_data["text"]))):
                        tok = self.normalize_token(ocr_data["text"][k])
                        if tok:
                            print("      Token at {0}: '{1}'".format(k, tok))
                            if tok in self.valid_grades:
                                grade_value = tok
                                break
                
                if grade_value:
                    data['grade'] = grade_value
                    print("    Final grade selected: {0}".format(grade_value))
                else:
                    print("    WARNING: No valid grade found")
                    # Last resort: print all tokens in top third for debugging
                    print("    All top third tokens: {0}".format(all_top_third_tokens[:20]))
                
                # Extract OCC using improved ChatGPT approach
                occ_value = None
                labels = ["OCC", "OCC.", "OCC:", "OCCASION", "OCCASION:", "OCCASION.", "Occasion"]
                label_indices = self.find_label_indices(ocr_data, labels)
                
                # Sort by position (top to bottom, left to right)
                label_indices.sort(key=lambda i: (ocr_data["top"][i], ocr_data["left"][i]))
                
                for idx in label_indices:
                    row_y = ocr_data["top"][idx]
                    # Look at tokens in the same row
                    for k in range(idx + 1, min(idx + 20, len(ocr_data["text"]))):
                        if abs(ocr_data["top"][k] - row_y) > 30:
                            break
                        tok = self.normalize_token(ocr_data["text"][k])
                        if len(tok) == 2 and tok in self.valid_occ_codes:
                            occ_value = tok
                            break
                        # Try correction for OCR errors
                        if len(tok) == 2:
                            corrected = tok.replace("0", "O").replace("1", "I").replace("5", "S")
                            if corrected in self.valid_occ_codes:
                                occ_value = corrected
                                break
                    if occ_value:
                        break
                
                # Fallback: scan top half for valid 2-letter code
                if not occ_value:
                    half_height = img.height // 2
                    for i, text in enumerate(ocr_data["text"]):
                        if not text or ocr_data["top"][i] > half_height:
                            continue
                        tok = self.normalize_token(text)
                        if len(tok) == 2 and tok in self.valid_occ_codes:
                            occ_value = tok
                            break
                
                if occ_value:
                    data['occ'] = occ_value
                
                # Extract To date - MUST be the second date after From
                to_value = None
                
                # Method 1: Find "From" and "To" labels and get dates near them
                from_index = -1
                for i, word in enumerate(ocr_data['text']):
                    if not word:
                        continue
                    w = word.upper()
                    if w == 'FROM':
                        from_index = i
                    elif w == 'TO' and from_index != -1:
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
                    date_pattern = r'From[:\s]*(\d{8})[:\s]*To[:\s]*(\d{8})'
                    match = re.search(date_pattern, text1, re.IGNORECASE)
                    if match:
                        to_value = match.group(2)
                    else:
                        # Look for any two 8-digit numbers close together
                        two_dates = re.findall(r'\d{8}', text1)
                        if len(two_dates) >= 2:
                            to_value = two_dates[1]
                
                if to_value:
                    data['to_date'] = to_value
                
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
                page2_values = self.extract_checkbox_values_cv2(img, 5)
            data['page2_values'] = page2_values if page2_values else [4] * 5
            
            # Process Page 3 - 5 checkbox values
            page3_values = []
            if len(doc) > 2:
                page = doc[2]
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                page3_values = self.extract_checkbox_values_cv2(img, 5)
            data['page3_values'] = page3_values if page3_values else [4] * 5
            
            # Process Page 4 - 4 checkbox values
            page4_values = []
            if len(doc) > 3:
                page = doc[3]
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                page4_values = self.extract_checkbox_values_cv2(img, 4)
            data['page4_values'] = page4_values if page4_values else [4] * 4
            
            doc.close()
            
            # Debug output
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
        if 'Not Observed' in text:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'Not Observed' in line:
                    check_lines = lines[i:i+3]
                    for check_line in check_lines:
                        if 'X' in check_line and 'Extended' not in check_line:
                            if len(check_line) < 50:
                                return True
        return False
    
    def extract_checkbox_values_cv2(self, img, expected_count):
        """Extract checkbox values using OpenCV to detect boxes and X marks"""
        values = []
        
        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (potential boxes)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for square-ish boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Check if it's roughly square and reasonable size
            aspect_ratio = float(w) / h
            area = w * h
            if 0.7 < aspect_ratio < 1.3 and 400 < area < 10000:
                boxes.append((x, y, w, h))
        
        print("    Found {0} potential checkbox candidates".format(len(boxes)))
        
        # Group boxes by rows (similar Y coordinates)
        if boxes:
            boxes.sort(key=lambda b: b[1])  # Sort by Y
            rows = []
            current_row = [boxes[0]]
            
            for box in boxes[1:]:
                if abs(box[1] - current_row[-1][1]) < 30:  # Same row
                    current_row.append(box)
                else:
                    if len(current_row) >= 6:  # Likely a checkbox row
                        rows.append(current_row)
                    current_row = [box]
            
            if len(current_row) >= 6:
                rows.append(current_row)
            
            print("    Found {0} rows with checkboxes".format(len(rows)))
            
            # Process each row
            for row_idx, row in enumerate(rows[:expected_count]):
                row.sort(key=lambda b: b[0])  # Sort by X
                
                # Ensure we have exactly 8 boxes
                if len(row) > 8:
                    row = row[:8]
                elif len(row) < 8:
                    # Try to interpolate missing boxes
                    while len(row) < 8:
                        row.append(row[-1])
                
                # Check which box has an X
                max_intensity_idx = 0
                max_intensity = 0
                
                for idx, (x, y, w, h) in enumerate(row):
                    # Extract the region inside the box
                    roi = gray[y+5:y+h-5, x+5:x+w-5]
                    
                    # Look for X pattern (diagonal lines)
                    if roi.size > 0:
                        # Apply edge detection
                        edges = cv2.Canny(roi, 50, 150)
                        
                        # Count edge pixels (more edges = likely has X)
                        edge_count = np.sum(edges > 0)
                        
                        if edge_count > max_intensity:
                            max_intensity = edge_count
                            max_intensity_idx = idx
                
                # Map to 1-8 value
                values.append(max_intensity_idx + 1)
                print("      Row {0}: Selected column {1}".format(row_idx + 1, max_intensity_idx + 1))
        
        # Pad with default values if needed
        while len(values) < expected_count:
            values.append(4)
        
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
        import cv2
    except ImportError as e:
        print("Missing required packages. Please install:")
        print("pip install PyMuPDF pytesseract pillow numpy opencv-python")
        print("Error: {0}".format(e))
        return
    
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    
    print("Marine FITREP PDF to CSV Extractor")
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
