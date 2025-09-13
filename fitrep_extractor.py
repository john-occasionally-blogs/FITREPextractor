#!/usr/bin/env python3
"""
Marine FITREP PDF to CSV Extractor with OCR
Extracts specific values from Marine Fitness Report PDFs using OCR and outputs to CSV
"""

import os
import sys
import re
import csv
import time
from pathlib import Path
from datetime import datetime
import io

# PDF and image processing
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


class FITREPExtractor:
    def __init__(self):
        self.results = []
        self.start_time = None
        self.pdf_count = 0
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
                
                # Extract FITREP ID from top of document
                fitrep_id = None
                
                # Pattern: USMC FITNESS REPORT (1610)  FITREP ID #  followed by 7-digit number
                fitrep_patterns = [
                    r'FITREP\s+ID\s*#?\s*(\d{7})',
                    r'FITNESS\s+REPORT.*?FITREP\s+ID\s*#?\s*(\d{7})',
                    r'1610.*?FITREP\s+ID\s*#?\s*(\d{7})',
                    r'USMC.*?FITREP\s+ID\s*#?\s*(\d{7})',
                    r'ID\s*#?\s*(\d{7})',  # Fallback pattern
                ]
                
                for pattern in fitrep_patterns:
                    match = re.search(pattern, text1, re.IGNORECASE | re.MULTILINE)
                    if match:
                        fitrep_id = match.group(1)
                        break
                
                if fitrep_id:
                    data['fitrep_id'] = fitrep_id
                
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
                
                # Extract all EDIPIs in sequence (Marine, RS, RO)
                edipi_data = self.extract_all_edipis(text1)
                if edipi_data:
                    data.update(edipi_data)
                
                # Extract RS and RO last names using coordinate-based method
                name_data = self.extract_rs_ro_names_coordinate_based(doc)
                if name_data:
                    data.update(name_data)
                else:
                    # Fallback to regex-based method if coordinate method fails
                    name_data = self.extract_rs_ro_names(text1)
                    if name_data:
                        data.update(name_data)
                
                # Extract Grade - simplified approach looking for any valid grade in top third
                grade_value = None
                top_third_height = img.height // 3
                
                # First find Grade label positions
                grade_positions = []
                for i, text in enumerate(ocr_data["text"]):
                    if text and "Grade" in text and ocr_data["top"][i] < top_third_height:
                        grade_positions.append(i)
                
                # Method 1: Look for valid grades anywhere in top third
                # Sometimes the grade isn't directly tied to the label due to OCR issues
                all_top_third_tokens = []
                for i, text in enumerate(ocr_data["text"]):
                    if text and ocr_data["top"][i] < top_third_height:
                        tok = self.normalize_token(text)
                        all_top_third_tokens.append(tok)
                        # Direct match
                        if tok in self.valid_grades:
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
                                    grade_value = correct_grade
                                    break
                            if grade_value:
                                break
                
                # Method 3: Look specifically near the first Grade label if found
                if not grade_value and grade_positions:
                    idx = grade_positions[0]
                    # Check next 30 tokens
                    for k in range(idx + 1, min(idx + 30, len(ocr_data["text"]))):
                        tok = self.normalize_token(ocr_data["text"][k])
                        if tok:
                            if tok in self.valid_grades:
                                grade_value = tok
                                break
                
                if grade_value:
                    data['grade'] = grade_value
                
                # Extract OCC - use text blocks to find form data areas
                occ_value = None
                
                # Use same page object as the checkbox detection uses
                page = doc[0]
                blocks = page.get_text("blocks")
                
                for i, block in enumerate(blocks):
                    if len(block) >= 5:
                        x0, y0, x1, y1, text = block[:5]
                        text_lines = text.strip().split('\n')
                        
                        # Look for blocks containing form data (names, dates, OCC codes)
                        has_form_data = False
                        for line in text_lines:
                            line_clean = line.strip().upper()
                            # Check if this block has form-like data
                            if (re.search(r'\b[A-Z]{2,}\b', line_clean) or
                                any(line_clean.isdigit() and len(line_clean) == 8 for line_clean in [line_clean])):
                                has_form_data = True
                                break
                        
                        if has_form_data:
                            # Look for OCC codes in this block
                            for line in text_lines:
                                line_clean = line.strip().upper()
                                
                                # Check if line is exactly a valid OCC code
                                if line_clean in self.valid_occ_codes:
                                    occ_value = line_clean
                                    break
                                
                                # Also try with normalization
                                normalized = self.normalize_token(line_clean)
                                if normalized in self.valid_occ_codes:
                                    occ_value = normalized
                                    break
                            
                            if occ_value:
                                break
                
                if occ_value:
                    data['occ'] = occ_value
                
                # Extract To date - use the same text block approach as OCC
                to_value = None
                
                # Look in the same form data blocks where we found OCC and names
                for i, block in enumerate(blocks):
                    if len(block) >= 5:
                        x0, y0, x1, y1, text = block[:5]
                        text_lines = text.strip().split('\n')
                        
                        # Look for blocks with dates (form data)
                        has_dates = False
                        for line in text_lines:
                            line_clean = line.strip()
                            if re.match(r'^\d{8}$', line_clean):  # 8-digit date
                                has_dates = True
                                break
                        
                        if has_dates:
                            # Look for the To date - should be the second date in blocks with TR/OCC
                            dates_found = []
                            has_occ_context = False
                            
                            for line in text_lines:
                                line_clean = line.strip()
                                
                                # Check if this block has OCC context (TR, etc.)
                                if line_clean in self.valid_occ_codes:
                                    has_occ_context = True
                                
                                # Collect 8-digit dates
                                if re.match(r'^\d{8}$', line_clean):
                                    dates_found.append(line_clean)
                            
                            # If this block has OCC context and multiple dates, take the second one (To date)
                            if has_occ_context and len(dates_found) >= 2:
                                to_value = dates_found[1]  # Second date is "To" date
                                break
                            # If only one date in OCC context, it might still be the To date
                            elif has_occ_context and len(dates_found) == 1:
                                # Use the date found in OCC context as To date
                                to_value = dates_found[0]
                                break
                
                # Fallback: look for any 8-digit dates in form blocks and take the latest/highest one
                if not to_value:
                    all_dates = []
                    
                    for i, block in enumerate(blocks):
                        if len(block) >= 5:
                            x0, y0, x1, y1, text = block[:5]
                            text_lines = text.strip().split('\n')
                            
                            for line in text_lines:
                                line_clean = line.strip()
                                if re.match(r'^\d{8}$', line_clean):
                                    all_dates.append(line_clean)
                    
                    if all_dates:
                        # Sort dates and take the latest one (likely the To date)
                        all_dates.sort()
                        to_value = all_dates[-1]
                
                if to_value:
                    data['to_date'] = to_value
                
                # Check for Not Observed
                not_observed = self.check_not_observed(img, text1)
                if not_observed:
                    print("  Skipping {0} - Not Observed is checked".format(pdf_path.name))
                    doc.close()
                    return None
            
            # Process Pages 2-4 using improved text-based checkbox extraction
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
            
            return data
            
        except Exception as e:
            print("Error processing {0}: {1}".format(pdf_path, str(e)))
            import traceback
            traceback.print_exc()
            return None
    
    def extract_reporting_senior_info(self, text, ocr_data):
        """Extract Reporting Senior name, rank, and EDIPI from the PDF"""
        rs_info = {}
        
        # Look for Reporting Senior section
        lines = text.split('\n')
        
        # Find the reporting senior section (usually in lower portion of page 1)
        rs_section_start = -1
        for i, line in enumerate(lines):
            # Look for "Reporting Senior" or similar indicators
            if any(phrase in line.upper() for phrase in ['REPORTING SENIOR', 'REPORT SENIOR', 'RS:']):
                rs_section_start = i
                break
            # Sometimes it's just labeled with rank/name patterns
            if any(rank in line.upper() for rank in ['MAJ', 'LTCOL', 'COL', 'CAPT', '1STLT', '2NDLT']):
                # Check if this looks like a reporting senior line with name patterns
                if re.search(r'\b[A-Z]{2,}\b.*\b[A-Z]{2,}\b', line.upper()):
                    rs_section_start = i
                    break
        
        if rs_section_start == -1:
            # Try alternative approach - look for name/rank patterns
            for i, line in enumerate(lines):
                line_upper = line.strip().upper()
                # Look for patterns that might indicate reporting senior info
                if (re.search(r'\b[A-Z]{2,}\b', line_upper) and 
                    any(rank in line_upper for rank in ['MAJ', 'LTCOL', 'COL', 'CAPT']) and
                    len(line.strip()) < 50):  # Avoid long text blocks
                    rs_section_start = i
                    break
        
        if rs_section_start >= 0:
            # Extract from surrounding lines
            search_range = lines[max(0, rs_section_start-3):rs_section_start+10]
            
            # Look for rank
            for line in search_range:
                line_clean = line.strip().upper()
                for rank in ['MAJ', 'LTCOL', 'COL', 'CAPT', '1STLT', '2NDLT']:
                    if rank in line_clean and len(line_clean) < 30:  # Avoid long descriptive text
                        rs_info['rs_rank'] = rank
                        break
                if 'rs_rank' in rs_info:
                    break
            
            # Look for name patterns
            name_patterns = [
                r'([A-Z]{2,}),\s*([A-Z][A-Z]+)',  # LASTNAME, FIRSTNAME
                r'([A-Z][A-Z]+)\s+([A-Z])\s+([A-Z]{2,})',  # FIRSTNAME M LASTNAME
            ]
            
            for line in search_range:
                for pattern in name_patterns:
                    match = re.search(pattern, line.upper())
                    if match and len(match.groups()) >= 2:
                        # Try to parse the match
                        if ',' in match.group(0):  # LASTNAME, FIRSTNAME format
                            rs_info['rs_last_name'] = match.group(1).strip()
                            rs_info['rs_first_name'] = match.group(2).strip()
                        elif len(match.groups()) >= 3:  # FIRSTNAME M LASTNAME format
                            rs_info['rs_first_name'] = match.group(1).strip()
                            rs_info['rs_last_name'] = match.group(3).strip()
                        break
                if 'rs_first_name' in rs_info:
                    break
            
            # Look for EDIPI (service number)
            edipi_patterns = [
                r'\b(\d{10})\b',  # Generic 10-digit number
                r'EDIPI[:\s]*(\d+)',  # Labeled EDIPI
                r'Service.*?(\d{10})',  # Service number
            ]
            
            # Check wider range for EDIPI
            full_text = '\n'.join(lines)
            for pattern in edipi_patterns:
                match = re.search(pattern, full_text)
                if match:
                    potential_edipi = match.group(1)
                    if len(potential_edipi) == 10:  # Standard EDIPI length
                        rs_info['rs_edipi'] = potential_edipi
                        break
        
        return rs_info if rs_info else None
    
    def extract_all_edipis(self, text):
        """Extract Marine, Reporting Senior, and Reviewing Officer EDIPIs in sequence"""
        edipis = {}
        
        # Find all 10-digit numbers in the document
        edipi_pattern = r'\b(\d{10})\b'
        matches = re.finditer(edipi_pattern, text)
        
        all_edipis = []
        for match in matches:
            edipi = match.group(1)
            all_edipis.append(edipi)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_edipis = []
        for edipi in all_edipis:
            if edipi not in seen:
                seen.add(edipi)
                unique_edipis.append(edipi)
        
        
        # Assign based on position (1st = Marine, 2nd = RS, 3rd = RO)
        if len(unique_edipis) >= 1:
            edipis['marine_edipi'] = unique_edipis[0]
        
        if len(unique_edipis) >= 2:
            edipis['rs_edipi'] = unique_edipis[1]
        
        if len(unique_edipis) >= 3:
            edipis['ro_edipi'] = unique_edipis[2]
        
        return edipis if edipis else None
    
    def extract_rs_ro_names_coordinate_based(self, pdf_doc):
        """Extract RS and RO last names using coordinate-based positioning"""
        names = {}
        
        # Get the EDIPIs we found earlier using text method
        page = pdf_doc[0]
        full_text = page.get_text()
        edipi_data = self.extract_all_edipis(full_text)
        if not edipi_data:
            return None
        
        # Get detailed text with coordinates
        text_dict = page.get_text("dict")
        
        # Find coordinates of EDIPIs
        rs_edipi_y = None
        ro_edipi_y = None
        
        if 'rs_edipi' in edipi_data:
            rs_edipi = edipi_data['rs_edipi']
            rs_edipi_y = self.find_text_y_coordinate(text_dict, rs_edipi)
        
        if 'ro_edipi' in edipi_data:
            ro_edipi = edipi_data['ro_edipi']
            ro_edipi_y = self.find_text_y_coordinate(text_dict, ro_edipi)
        
        # Extract leftmost names on the same lines as EDIPIs
        if rs_edipi_y is not None:
            rs_name = self.find_leftmost_name_on_line(text_dict, rs_edipi_y)
            if rs_name:
                names['rs_last_name'] = rs_name
        
        if ro_edipi_y is not None:
            ro_name = self.find_leftmost_name_on_line(text_dict, ro_edipi_y)
            if ro_name:
                names['ro_last_name'] = ro_name
        
        return names if names else None
    
    def find_text_y_coordinate(self, text_dict, search_text):
        """Find the Y-coordinate of specific text in the PDF"""
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if search_text in span.get("text", ""):
                            return line["bbox"][1]  # Y-coordinate of line
        return None
    
    def find_leftmost_name_on_line(self, text_dict, target_y, tolerance=10):
        """Find the leftmost name-like text on a specific Y-coordinate line"""
        line_texts = []
        
        # Find all text elements near the target Y-coordinate
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_y = line["bbox"][1]  # Y-coordinate of line
                    
                    # Check if this line is close to our target Y
                    if abs(line_y - target_y) <= tolerance:
                        for span in line["spans"]:
                            text = span.get("text", "").strip()
                            if text and len(text) >= 3:  # Minimum name length
                                # Check if it looks like a name (all caps, letters only)
                                if text.isalpha() and text.isupper():
                                    # Filter out common field labels
                                    field_labels = {'DUTY', 'ASSIGNMENT', 'GRADE', 'SERVICE', 'LAST', 'NAME', 
                                                   'INITIALS', 'COMMANDER', 'OFFICER', 'SENIOR', 'REPORTING', 
                                                   'REVIEWING', 'USMC', 'ANG', 'USA', 'AFNG', 'USAF', 'USN', 
                                                   'FMS', 'USCG', 'USSF'}
                                    if text not in field_labels:
                                        line_texts.append({
                                            'text': text,
                                            'x': span["bbox"][0]  # Left X-coordinate
                                        })
        
        # Sort by X-coordinate and return the leftmost
        if line_texts:
            line_texts.sort(key=lambda x: x['x'])
            return line_texts[0]['text']
        
        return None

    def extract_rs_ro_names(self, text):
        """Legacy regex-based extraction - kept as fallback"""
        names = {}
        
        # Valid service branches
        valid_services = ['USMC', 'ANG', 'USA', 'AFNG', 'USAF', 'USN', 'FMS', 'USCG', 'USSF']
        
        # Get the EDIPIs we found earlier
        edipi_data = self.extract_all_edipis(text)
        if not edipi_data:
            return None
        
        # Look for RS EDIPI (second EDIPI)
        if 'rs_edipi' in edipi_data:
            rs_edipi = edipi_data['rs_edipi']
            
            
            # Pattern variations for RS based on actual format observed
            patterns_to_try = [
                # Pattern: LASTNAME INITIALS| SERVICE | EDIPI (mixed case initials)
                r'\b([A-Za-z]{2,})\s+[A-Za-z]+\s*\|\s*(' + '|'.join(valid_services) + r')\s*\|\s*' + rs_edipi + r'\b',
                # Pattern: LASTNAME INITIALS SERVICE | EDIPI (mixed case service and initials)
                r'\b([A-Za-z]{2,})\s+[A-Za-z]+\s+([A-Za-z]+)\s*\|\s*' + rs_edipi + r'\b',
                # Pattern: LASTNAME INITIALS} SERVICE | EDIPI] (using } instead of |, ends with ])
                r'\b([A-Za-z]{2,})\s+[A-Za-z]+\s*\}\s*(' + '|'.join(valid_services) + r')\s*\|\s*' + rs_edipi + r'\]',
                # Pattern: LASTNAME | INITIALS | SERVICE | EDIPI (original, mixed case)
                r'\b([A-Za-z]{2,})\s*\|\s*[A-Za-z]+\s*\|\s*(' + '|'.join(valid_services) + r')\s*\|\s*' + rs_edipi + r'\b',
            ]
            
            for pattern in patterns_to_try:
                match = re.search(pattern, text)
                if match:
                    potential_name = match.group(1).upper()
                    # Filter out field labels that aren't actual names
                    field_labels = ['DUTY', 'ASSIGNMENT', 'GRADE', 'SERVICE', 'LAST', 'NAME', 'INITIALS', 
                                   'COMMANDER', 'OFFICER', 'SENIOR', 'REPORTING', 'REVIEWING']
                    if potential_name not in field_labels:
                        names['rs_last_name'] = potential_name
                        break
            
        
        # Look for RO EDIPI (third EDIPI)  
        if 'ro_edipi' in edipi_data:
            ro_edipi = edipi_data['ro_edipi']
            
            
            # Pattern variations for RO based on actual format observed
            # Create case-insensitive service patterns
            services_pattern = '|'.join(valid_services)
            
            patterns_to_try = [
                # Pattern: LASTNAME INITIALS SERVICE | EDIPI (require lastname 3+ chars, initials 1-2 chars)
                r'(?i)\b([A-Za-z]{3,})\s+[A-Za-z]{1,2}\s+([A-Za-z]+)\s*\|\s*' + ro_edipi + r'\b',
                # Pattern: LASTNAME INITIALS SERVICE | EDIPI] (ends with ], 3+ char lastname)
                r'(?i)\b([A-Za-z]{3,})\s+[A-Za-z]{1,2}\s+([A-Za-z]+)\s*\|\s*' + ro_edipi + r'\]',
                # Pattern: LASTNAME | INITIALS | SERVICE | EDIPI (pipe-separated, 3+ char lastname)
                r'(?i)\b([A-Za-z]{3,})\s*\|\s*[A-Za-z]{1,2}\s*\|\s*(' + services_pattern + r')\s*\|\s*' + ro_edipi + r'\b',
            ]
            
            for pattern in patterns_to_try:
                match = re.search(pattern, text)
                if match:
                    potential_name = match.group(1).upper()
                    # Filter out field labels that aren't actual names
                    field_labels = ['DUTY', 'ASSIGNMENT', 'GRADE', 'SERVICE', 'LAST', 'NAME', 'INITIALS', 
                                   'COMMANDER', 'OFFICER', 'SENIOR', 'REPORTING', 'REVIEWING']
                    if potential_name not in field_labels:
                        names['ro_last_name'] = potential_name
                        break
            
        
        return names if names else None
    
    def extract_reviewing_officer_info(self, text, ocr_data):
        """Extract Reviewing Officer name and EDIPI from the PDF"""
        ro_info = {}
        
        # Look for Reviewing Officer section
        lines = text.split('\n')
        
        # Find the reviewing officer section (usually in lower portion of page 1)
        ro_section_start = -1
        for i, line in enumerate(lines):
            # Look for "Reviewing Officer" or similar indicators
            if any(phrase in line.upper() for phrase in ['REVIEWING OFFICER', 'REVIEW OFFICER', 'RO:']):
                ro_section_start = i
                break
            # Sometimes it's just labeled with rank/name patterns (different from RS)
            # Look for reviewing officer indicators like different EDIPI patterns
            line_upper = line.strip().upper()
            if re.search(r'\b\d{10}\b', line_upper):
                # This might be RO section with EDIPI
                ro_section_start = i
                break
        
        if ro_section_start == -1:
            # Try alternative approach - look for section after reporting senior
            # Usually RO comes after RS in the document
            for i, line in enumerate(lines):
                line_upper = line.strip().upper()
                # Look for different name patterns (generic name detection)
                if re.search(r'\b[A-Z]{2,}\b.*\b[A-Z]{2,}\b', line_upper) and len(line.strip()) < 50:
                    ro_section_start = i
                    break
        
        if ro_section_start >= 0:
            # Extract from surrounding lines
            search_range = lines[max(0, ro_section_start-3):ro_section_start+10]
            
            # Look for rank
            for line in search_range:
                line_clean = line.strip().upper()
                for rank in ['COL', 'MAJ', 'LTCOL', 'CAPT', '1STLT', '2NDLT', 'BGEN', 'MAJGEN']:
                    if rank in line_clean and len(line_clean) < 30:  # Avoid long descriptive text
                        ro_info['ro_rank'] = rank
                        break
                if 'ro_rank' in ro_info:
                    break
            
            # Look for name patterns (different from reporting senior)
            name_patterns = [
                r'([A-Z]{2,}),\s*([A-Z][A-Z]+)',  # LASTNAME, FIRSTNAME
                r'([A-Z][A-Z]+)\s+([A-Z])\s+([A-Z]{2,})',  # FIRSTNAME M LASTNAME
            ]
            
            for line in search_range:
                for pattern in name_patterns:
                    match = re.search(pattern, line.upper())
                    if match:  # Generic name pattern matching
                        # Parse based on format
                        if ',' in match.group(0):  # LASTNAME, FIRSTNAME format
                            ro_info['ro_last_name'] = match.group(1).strip()
                            ro_info['ro_first_name'] = match.group(2).strip()
                        elif len(match.groups()) >= 3:  # FIRSTNAME M LASTNAME format
                            ro_info['ro_first_name'] = match.group(1).strip()
                            ro_info['ro_last_name'] = match.group(3).strip()
                        break
                if 'ro_first_name' in ro_info:
                    break
            
            # Look for EDIPI (different from reporting senior EDIPI)
            edipi_patterns = [
                r'\b(\d{10})\b',  # Generic 10-digit number
                r'EDIPI[:\s]*(\d+)',  # Labeled EDIPI
                r'Service.*?(\d{10})',  # Service number
            ]
            
            # Check wider range for EDIPI but exclude the known RS EDIPI
            full_text = '\n'.join(lines)
            for pattern in edipi_patterns:
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    potential_edipi = match.group(1)
                    if len(potential_edipi) == 10:  # Standard EDIPI length
                        ro_info['ro_edipi'] = potential_edipi
                        break
                if 'ro_edipi' in ro_info:
                    break
        
        return ro_info if ro_info else None
    
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
    
    def extract_checkbox_values_text_based(self, pdf_doc, page_num, expected_count):
        """Extract checkbox values using direct PDF text extraction - much more reliable"""
        if page_num >= len(pdf_doc):
            return [4] * expected_count
        
        page = pdf_doc[page_num]
        
        # Get text blocks with position info
        blocks = page.get_text("blocks")
        
        
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
        
        if not x_marks:
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
        
        
        # Convert to values
        values = []
        
        for row_idx in range(expected_count):
            if row_idx < len(rows):
                row_x_marks = rows[row_idx]
                
                if row_x_marks:
                    # Use the first (or only) X mark in the row
                    x_pos = row_x_marks[0]["x"]
                    
                    # Corrected position mapping based on expected values
                    # Analysis of actual vs expected shows the mapping needs to be adjusted:
                    # X position clusters: ~225-228, ~317-319, ~416, ~513-516
                    # Issue: H column (value 8) being read as 3, C column (value 3) being read as 8
                    
                    # Fixed mapping with C and H columns corrected
                    if x_pos < 180:  # Cluster around ~150-170 (if any)
                        column = 3  # C (corrected - was showing as 8)
                    elif x_pos < 250:  # Cluster around ~225-228  
                        column = 4  # D
                    elif x_pos < 340:  # Cluster around ~317-319
                        column = 5  # E
                    elif x_pos < 440:  # Cluster around ~416
                        column = 6  # F
                    elif x_pos < 520:  # Cluster around ~513-516
                        column = 7  # G
                    elif x_pos < 600:  # Potential cluster (not seen yet)
                        column = 8  # H (corrected - was showing as 3)
                    elif x_pos < 700:  # Potential cluster (not seen yet)
                        column = 2  # B
                    else:  # Far right
                        column = 1  # A
                    
                    values.append(column)
                else:
                    values.append(4)  # Default to D
            else:
                values.append(4)  # Default to D
        
        return values

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
        pdf_start_time = time.time()
        data = self.extract_from_pdf(pdf_path)
        
        if data:
            # Format as CSV row
            row = [
                data.get('fitrep_id', ''),
                data.get('last_name', ''),
                data.get('grade', ''),
                data.get('occ', ''),
                data.get('to_date', ''),
                data.get('marine_edipi', ''),
                data.get('rs_last_name', ''),
                data.get('rs_edipi', ''),
                data.get('ro_last_name', ''),
                data.get('ro_edipi', '')
            ]
            # Add page 2 values (5 values)
            row.extend(data.get('page2_values', [''] * 5))
            # Add page 3 values (5 values)
            row.extend(data.get('page3_values', [''] * 5))
            # Add page 4 values (4 values)
            row.extend(data.get('page4_values', [''] * 4))
            
            self.results.append(row)
            pdf_end_time = time.time()
            pdf_time = pdf_end_time - pdf_start_time
            self.pdf_count += 1
            print("  Processing time: {:.2f} seconds".format(pdf_time))
            return True
        return False
    
    def process_directory(self, directory_path):
        """Process all PDF files in a directory"""
        pdf_files = list(directory_path.glob('*.pdf'))
        
        if not pdf_files:
            print("No PDF files found in the directory")
            return False
        
        print("Found {0} PDF files".format(len(pdf_files)))
        self.start_time = time.time()
        
        for pdf_file in pdf_files:
            self.process_single_pdf(pdf_file)
        
        # Sort results by Grade (military rank), then by last name
        self.results.sort(key=lambda x: (self.rank_sort_key(x[1]), x[0]))
        
        end_time = time.time()
        total_time = end_time - self.start_time
        avg_time = total_time / self.pdf_count if self.pdf_count > 0 else 0
        
        print("\n" + "=" * 50)
        print("TIMING SUMMARY:")
        print("Total processing time: {:.2f} seconds ({:.1f} minutes)".format(total_time, total_time / 60))
        print("PDFs processed: {0}".format(self.pdf_count))
        print("Average time per PDF: {:.2f} seconds".format(avg_time))
        print("=" * 50)
        
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
        
        extractor.start_time = time.time()
        if extractor.process_single_pdf(selected_pdf):
            # Generate output filename
            output_file = script_dir / "{0}_extracted.csv".format(selected_pdf.stem)
            extractor.save_to_csv(output_file)
            
            # Display timing for single file
            end_time = time.time()
            total_time = end_time - extractor.start_time
            print("\n" + "=" * 50)
            print("TIMING SUMMARY:")
            print("Total processing time: {:.2f} seconds".format(total_time))
            print("=" * 50)
    
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