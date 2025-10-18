#!/usr/bin/env python3
"""
Marine FITREP PDF to CSV Extractor with Improved Text-Based Checkbox Detection
Extracts specific values from Marine Fitness Report PDFs using direct text extraction
"""

import os
import sys
import re
import csv
from pathlib import Path
from datetime import datetime
import io

# PDF processing
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


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
    
    def extract_checkbox_values_text_based(self, pdf_doc, page_num, expected_count):
        """Extract checkbox values using direct PDF text extraction - much more reliable"""
        if page_num >= len(pdf_doc):
            return [4] * expected_count
        
        page = pdf_doc[page_num]
        
        # Get text blocks with position info
        blocks = page.get_text("blocks")
        
        print(f"    Using text-based extraction on page {page_num + 1}")
        print(f"    Found {len(blocks)} text blocks")
        
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
        
        print(f"    Found {len(x_marks)} X marks")
        
        if not x_marks:
            print(f"    No X marks found, using defaults")
            return [4] * expected_count
        
        # Sort by Y position (top to bottom)
        x_marks.sort(key=lambda x: x["y"])
        
        # Group into rows by Y position
        rows = []
        if x_marks:
            current_row = [x_marks[0]]
            
            for x_mark in x_marks[1:]:
                # If Y positions are close (within 30 points), same row
                if abs(x_mark["y"] - current_row[0]["y"]) < 30:
                    current_row.append(x_mark)
                else:
                    rows.append(current_row)
                    current_row = [x_mark]
            
            if current_row:
                rows.append(current_row)
        
        print(f"    Grouped into {len(rows)} rows")
        
        # Convert to values
        values = []
        
        for row_idx in range(expected_count):
            if row_idx < len(rows):
                row_x_marks = rows[row_idx]
                
                if row_x_marks:
                    # Use the first (or only) X mark in the row
                    x_pos = row_x_marks[0]["x"]
                    
                    # Improved position mapping based on FITREP layout
                    # Calibrated boundaries for standard 8-column checkbox layout
                    # A=1 (rightmost), B=2, C=3, D=4, E=5, F=6, G=7, H=8 (leftmost)
                    if x_pos < 150:
                        column = 8  # H 
                    elif x_pos < 200:
                        column = 7  # G
                    elif x_pos < 275:
                        column = 6  # F
                    elif x_pos < 350:
                        column = 5  # E
                    elif x_pos < 425:
                        column = 4  # D
                    elif x_pos < 500:
                        column = 3  # C
                    elif x_pos < 575:
                        column = 2  # B
                    else:
                        column = 1  # A
                    
                    values.append(column)
                    print(f"      Row {row_idx + 1}: X at x={x_pos:.1f} -> Column {chr(64 + column)} (value {column})")
                else:
                    values.append(4)  # Default to D
                    print(f"      Row {row_idx + 1}: No X mark, default value 4")
            else:
                values.append(4)  # Default to D
                print(f"      Row {row_idx + 1}: No row data, default value 4")
        
        return values
    
    def extract_from_pdf(self, pdf_path):
        """Extract required data from a single PDF file"""
        try:
            data = {}
            
            # Open PDF with PyMuPDF
            doc = fitz.open(str(pdf_path))
            
            # Process Page 1 for metadata
            if len(doc) > 0:
                page = doc[0]
                # Higher resolution for better OCR on page 1
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Perform OCR on page 1 for text fields
                text1 = pytesseract.image_to_string(img)
                ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                
                # Extract Last Name
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
                
                # Extract Grade using existing method (it works well)
                grade_value = None
                top_third_height = img.height // 3
                
                print("  DEBUG Grade extraction:")
                
                all_top_third_tokens = []
                for i, text in enumerate(ocr_data["text"]):
                    if text and ocr_data["top"][i] < top_third_height:
                        tok = self.normalize_token(text)
                        all_top_third_tokens.append(tok)
                        if tok in self.valid_grades:
                            print("    Found grade '{0}' at position {1}".format(tok, i))
                            if not grade_value:
                                grade_value = tok
                
                # Grade mapping fallback
                if not grade_value:
                    grade_mapping = {
                        'MAJ': ['MAJ', 'MAS', 'MA', 'MJ', 'MAJOR', 'MAI', 'MAT'],
                        'LTCOL': ['LTCOL', 'LRCOL', 'LTCO', 'LTC', 'LTCL', 'LICOL', 'IRCOL'],
                        'MGYSGT': ['MGYSGT', 'MGYST', 'MGSG', 'MGYSG', 'MGYSGI'],
                        'MSGT': ['MSGT', 'SCR', 'MSG', 'MSGI', 'MST'],
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
                
                if grade_value:
                    data['grade'] = grade_value
                    print("    Final grade selected: {0}".format(grade_value))
                else:
                    print("    WARNING: No valid grade found")
                
                # Extract To date using existing method
                to_value = None
                to_labels = ["To", "TO", "To:", "TO:"]
                to_indices = self.find_label_indices(ocr_data, to_labels)
                to_indices = [i for i in to_indices if ocr_data["top"][i] < top_third_height]
                
                print("  DEBUG To date extraction:")
                print("    Found {0} 'To' labels in top third".format(len(to_indices)))
                
                if to_indices:
                    to_indices.sort(key=lambda i: (ocr_data["top"][i], ocr_data["left"][i]))
                    
                    best_to_idx = None
                    for idx in to_indices:
                        row_y = ocr_data["top"][idx]
                        found_from_before = False
                        
                        for j in range(max(0, idx - 10), idx):
                            if abs(ocr_data["top"][j] - row_y) < 30:
                                tok = self.normalize_token(ocr_data["text"][j])
                                if tok in ["FROM", "FRON"]:
                                    found_from_before = True
                                    break
                        
                        if found_from_before:
                            best_to_idx = idx
                            print("    Using 'To' at index {0} (has 'From' before it)".format(idx))
                            break
                    
                    if best_to_idx is None:
                        best_to_idx = to_indices[0]
                        print("    Using first 'To' at index {0}".format(best_to_idx))
                    
                    row_y = ocr_data["top"][best_to_idx]
                    
                    for k in range(best_to_idx + 1, min(best_to_idx + 20, len(ocr_data["text"]))):
                        if ocr_data["top"][k] < row_y - 10:
                            continue
                        if ocr_data["top"][k] > row_y + 60:
                            break
                        
                        tok = ocr_data["text"][k].strip()
                        
                        if re.match(r'^\d{8}$', tok):
                            to_value = tok
                            print("    Found date: {0}".format(tok))
                            break
                        
                        tok_no_space = tok.replace(" ", "").replace("-", "")
                        if re.match(r'^\d{8}$', tok_no_space):
                            to_value = tok_no_space
                            print("    Found date (after removing spaces): {0}".format(tok_no_space))
                            break
                
                if to_value:
                    data['to_date'] = to_value
                else:
                    print("    WARNING: Could not find To date")
                
                # Check for Not Observed
                not_observed = self.check_not_observed(img, text1)
                if not_observed:
                    print("  Skipping {0} - Not Observed is checked".format(pdf_path.name))
                    doc.close()
                    return None
            
            # Process Pages 2-4 using improved text-based checkbox extraction
            print("  Extracting checkboxes using text-based method:")
            
            # Process Page 2 - 5 checkbox values
            page2_values = []
            if len(doc) > 1:
                page2_values = self.extract_checkbox_values_text_based(doc, 1, 5)
            data['page2_values'] = page2_values if page2_values else [4] * 5
            
            # Process Page 3 - 5 checkbox values
            page3_values = []
            if len(doc) > 2:
                page3_values = self.extract_checkbox_values_text_based(doc, 2, 5)
            data['page3_values'] = page3_values if page3_values else [4] * 5
            
            # Process Page 4 - 4 checkbox values
            page4_values = []
            if len(doc) > 3:
                page4_values = self.extract_checkbox_values_text_based(doc, 3, 4)
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
            # Add page values
            row.extend(data.get('page2_values', [''] * 5))
            row.extend(data.get('page3_values', [''] * 5))
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
    # Check for required packages
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError as e:
        print("Missing required packages. Please install:")
        print("pip install PyMuPDF pytesseract pillow")
        print("Error: {0}".format(e))
        return
    
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    
    print("Marine FITREP PDF to CSV Extractor (Improved Text-Based)")
    print("=" * 60)
    
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
            output_file = script_dir / "{0}_improved_extracted.csv".format(selected_pdf.stem)
            extractor.save_to_csv(output_file)
    
    else:
        # All files mode
        if extractor.process_directory(script_dir):
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = script_dir / "fitrep_improved_extract_{0}.csv".format(timestamp)
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