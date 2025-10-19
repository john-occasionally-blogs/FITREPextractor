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
        # Checkbox fallback mode: 'off' | 'auto' | 'force'
        # - off: use text-based only
        # - auto: use OCR fallback only when text-based looks suspicious
        # - force: always use OCR fallback for pages 2–4
        self.checkbox_fallback_mode = os.getenv('FITREP_CHECKBOX_FALLBACK', 'off').lower()
        # Strict mode: do not silently substitute default checkbox values when uncertain
        self.strict_no_defaults = str(os.getenv('FITREP_STRICT', 'true')).lower() in {'1','true','yes','on'}
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

    def normalize_grade_token(self, s):
        """Normalize a token specifically for grade matching (preserve digits)."""
        s = s.strip().upper()
        # Keep only alphanumeric (drop punctuation/spaces), but DO NOT map digits
        return ''.join(ch for ch in s if ch.isalnum())
    
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
    
    def _extract_from_document(self, doc, label="document"):
        """Core extraction from an open PyMuPDF document. Returns dict or None."""
        try:
            data = {}

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

                # Marine last name - anchor on Marine EDIPI line and walk up lines
                try:
                    if edipi_data and edipi_data.get('marine_edipi'):
                        mname = self.extract_marine_last_name_by_edipi(doc, edipi_data['marine_edipi'])
                        if mname:
                            data['last_name'] = mname
                except Exception:
                    pass
                
                # Extract Grade - fixed-region OCR (simple + accurate) with text fallback
                grade_value = None

                # Attempt 0: Small fixed region OCR where the Marine's grade always appears
                # Try a sequence of calibrated normalized windows (x0,y0,x1,y1)
                # First, the window calibrated from a marked sample; then a broader default.
                try:
                    w, h = img.width, img.height
                    # List of windows: (left, top, right, bottom) in normalized [0,1]
                    grade_windows = [
                        # Calibrated from user-marked file (fitrepPdf (3) marked.pdf)
                        (0.60, 0.17, 0.685, 0.205),
                        # Original default
                        (0.58, 0.16, 0.69, 0.22),
                        # Slightly wider fallback
                        (0.56, 0.15, 0.71, 0.24),
                    ]
                    for (lx, ty, rx, by) in grade_windows:
                        left = int(lx * w)
                        top = int(ty * h)
                        right = int(rx * w)
                        bottom = int(by * h)
                        crop = img.crop((left, top, right, bottom))
                        ocr_fixed = pytesseract.image_to_data(
                            crop, output_type=pytesseract.Output.DICT,
                            config='--psm 7 --oem 1 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/'
                        )
                        for t in ocr_fixed.get('text', []):
                            if not t:
                                continue
                            tok = self.normalize_grade_token(t)
                            if tok in self.valid_grades:
                                grade_value = tok
                                break
                        if grade_value:
                            break
                except Exception:
                    pass

                # Attempt 1: direct PDF text spans near a GRADE label or in top section
                if not grade_value:
                    try:
                        page0 = doc[0]
                        text_dict = page0.get_text("dict")
                        top_third_limit = page0.rect.height / 3.0

                        def token_iter():
                            for block in text_dict.get("blocks", []):
                                for line in block.get("lines", []):
                                    y_top = line.get("bbox", [0, 0, 0, 0])[1]
                                    for span in line.get("spans", []):
                                        txt = (span.get("text", "") or "").strip()
                                        if not txt:
                                            continue
                                        yield txt, y_top, span.get("bbox", [0, 0, 0, 0])

                        # Find any grade tokens in top third
                        for txt, y_top, bbox in token_iter():
                            if y_top <= top_third_limit:
                                # Split span text into alnum tokens to catch cases like "1234567890 1STLT"
                                for piece in re.findall(r"[A-Za-z0-9]+", txt):
                                    tok = self.normalize_grade_token(piece)
                                    if tok in self.valid_grades:
                                        grade_value = tok
                                        break
                                if grade_value:
                                    break

                        # If not found, look for a GRADE label and then take the first valid token to its right on same line
                        if not grade_value:
                            for block in text_dict.get("blocks", []):
                                for line in block.get("lines", []):
                                    spans = line.get("spans", [])
                                    # Identify index of label-like span
                                    label_idx = None
                                    for i, sp in enumerate(spans):
                                        label_text = self.normalize_token(sp.get("text", ""))
                                        if label_text in {"GRADE", "RANK", "GRADERANK", "GRADE/ RANK", "GRADE/RANK"}:
                                            label_idx = i
                                            break
                                    if label_idx is not None:
                                        # Search subsequent spans on same line, prefer ones to the right
                                        label_right = spans[label_idx].get("bbox", [0, 0, 0, 0])[2]
                                        for j in range(label_idx + 1, len(spans)):
                                            sp = spans[j]
                                            if sp.get("bbox", [0, 0, 0, 0])[0] < label_right:
                                                continue
                                            # Split span into tokens and check each
                                            for piece in re.findall(r"[A-Za-z0-9]+", sp.get("text", "")):
                                                tok = self.normalize_grade_token(piece)
                                                if tok in self.valid_grades:
                                                    grade_value = tok
                                                    break
                                            if grade_value:
                                                break
                                        if grade_value:
                                            break
                                if grade_value:
                                    break
                    except Exception:
                        # Ignore and fall back to OCR
                        pass

                # Attempt 2: OCR-based heuristics (fallback)
                if not grade_value:
                    top_third_height = img.height // 3

                    # First find Grade label positions
                    grade_positions = []
                    for i, text in enumerate(ocr_data["text"]):
                        if text and "Grade" in text and ocr_data["top"][i] < top_third_height:
                            grade_positions.append(i)

                    # Look for valid grades anywhere in top third
                    all_top_third_tokens = []
                    for i, text in enumerate(ocr_data["text"]):
                        if text and ocr_data["top"][i] < top_third_height:
                            tok = self.normalize_grade_token(text)
                            all_top_third_tokens.append(tok)
                            if tok in self.valid_grades and not grade_value:
                                grade_value = tok

                    # Partial/variation mapping
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
                            for variation in variations:
                                if variation in all_top_third_tokens:
                                    grade_value = correct_grade
                                    break
                            if grade_value:
                                break

                    # Look near the first Grade label
                    if not grade_value and grade_positions:
                        idx = grade_positions[0]
                        for k in range(idx + 1, min(idx + 30, len(ocr_data["text"]))):
                            tok = self.normalize_grade_token(ocr_data["text"][k])
                            if tok and tok in self.valid_grades:
                                grade_value = tok
                                break
                
                if grade_value:
                    data['grade'] = grade_value
                
                # Extract OCC - use text blocks to find form data areas
                occ_value = None
                occ_block_lines = None
                
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
                                    occ_block_lines = text_lines
                                    break
                            
                            if occ_value:
                                break
                
                if occ_value:
                    data['occ'] = occ_value

                # If Grade still missing, try to pull it from the same OCC block context
                if not data.get('grade') and occ_block_lines:
                    for line in occ_block_lines:
                        for tok in re.findall(r"[A-Z0-9]+", line.upper()):
                            nt = self.normalize_grade_token(tok)
                            if nt in self.valid_grades:
                                data['grade'] = nt
                                break
                        if data.get('grade'):
                            break
                
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

                # Additional Grade heuristic: look near Marine's last name line in page text
                if not data.get('grade') and data.get('last_name'):
                    try:
                        page0 = doc[0]
                        text_dict2 = page0.get_text("dict")
                        lname = data['last_name'].upper()
                        page_height = float(page0.rect.height)
                        top_third_limit = page_height / 3.0
                        target_ys = []
                        # Find Y positions of lines containing the last name token
                        for block in text_dict2.get("blocks", []):
                            for line in block.get("lines", []):
                                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                                if lname in line_text.upper():
                                    y_line = line.get("bbox", [0, 0, 0, 0])[1]
                                    if y_line <= top_third_limit:
                                        target_ys.append(y_line)
                        # Search same/nearby lines for a grade token
                        for block in text_dict2.get("blocks", []):
                            for line in block.get("lines", []):
                                y_top = line.get("bbox", [0, 0, 0, 0])[1]
                                if any(abs(y_top - ty) <= 20 for ty in target_ys):
                                    for span in line.get("spans", []):
                                        tok = self.normalize_grade_token(span.get("text", ""))
                                        if tok in self.valid_grades:
                                            data['grade'] = tok
                                            raise StopIteration
                    except StopIteration:
                        pass
                    except Exception:
                        pass
                
                # Check for Not Observed
                not_observed = self.check_not_observed(img, text1)
                if not_observed:
                    print("  Skipping {0} - Not Observed is checked".format(label))
                    doc.close()
                    return None
            
            # Process Pages 2-4 using improved text-based checkbox extraction
            # Process Page 2 - 5 checkbox values
            page2_values = []
            if len(doc) > 1:
                page2_values = self.extract_checkbox_values_auto(doc, 1, 5)
            data['page2_values'] = page2_values if page2_values else None
            
            # Process Page 3 - 5 checkbox values
            page3_values = []
            if len(doc) > 2:
                page3_values = self.extract_checkbox_values_auto(doc, 2, 5)
            data['page3_values'] = page3_values if page3_values else None
            
            # Process Page 4 - 4 checkbox values
            page4_values = []
            if len(doc) > 3:
                page4_values = self.extract_checkbox_values_auto(doc, 3, 4)
            data['page4_values'] = page4_values if page4_values else None

            # Do not apply any overrides keyed by FITREP ID.
            # Verification against known values should be performed via example scripts,
            # not by mutating extracted results.
            
            doc.close()

            # Debug output
            print("  Extracted - Last Name: {0}, Grade: {1}, OCC: {2}, To: {3}".format(
                data.get('last_name'), data.get('grade'), data.get('occ'), data.get('to_date')))
            
            return data
            
        except Exception as e:
            print("Error processing {0}: {1}".format(label, str(e)))
            import traceback
            traceback.print_exc()
            return None

        finally:
            try:
                doc.close()
            except Exception:
                pass

    def extract_from_pdf(self, pdf_path):
        """Extract required data from a single PDF file using OCR"""
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            print("Error opening {0}: {1}".format(pdf_path, str(e)))
            import traceback
            traceback.print_exc()
            return None
        label = getattr(pdf_path, 'name', str(pdf_path))
        return self._extract_from_document(doc, label=label)

    def extract_from_bytes(self, pdf_bytes):
        """Extract data from in-memory PDF bytes. Returns dict or None."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            print("Error opening <bytes>: {0}".format(str(e)))
            import traceback
            traceback.print_exc()
            return None
        return self._extract_from_document(doc, label="<bytes>")

    async def extract_fitrep_data(self, pdf_path):
        """
        Async wrapper for extract_from_pdf - provides API interface for pdf-processor service
        """
        return self.extract_from_pdf(pdf_path)

    async def extract_fitrep_data_bytes(self, pdf_bytes):
        """Async wrapper for bytes-based extraction for API/microservice use."""
        return self.extract_from_bytes(pdf_bytes)

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
        """Check if Not Observed checkbox is marked using position-aware OCR.

        Strategy:
        - Use OCR tokens with coordinates to find the horizontal centers of
          labels "Adverse", "Not Observed", and "Extended" (columns a/b/c).
        - In the band below those labels, locate any isolated X marks or
          bracketed variants (e.g., "X", "[x]").
        - Determine if the X nearest to the Not Observed column center is
          within a reasonable tolerance, indicating that b. Not Observed is
          the checked option. This avoids flagging when only c. Extended is
          checked.
        """
        try:
            # OCR with positional data
            ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            n = len(ocr.get('text', []))
            if n == 0:
                return False

            # Locate label centers and reference Y for the label row
            import re as _re
            centers = {}
            label_y = None
            for i in range(n):
                t = (ocr['text'][i] or '').strip()
                if not t:
                    continue
                # a. Adverse
                if _re.fullmatch(r"Adverse", t, flags=_re.I):
                    centers['adverse'] = ocr['left'][i] + ocr['width'][i] / 2.0
                    label_y = ocr['top'][i]
                # b. Not Observed (split across two tokens: Not + Observed)
                if _re.fullmatch(r"Not", t, flags=_re.I):
                    j = i + 1
                    if j < n:
                        t2 = (ocr['text'][j] or '').strip()
                        if _re.fullmatch(r"Observed", t2, flags=_re.I) and abs(ocr['top'][j] - ocr['top'][i]) < 12:
                            centers['not_observed'] = (ocr['left'][i] + (ocr['left'][j] + ocr['width'][j])) / 2.0
                            label_y = ocr['top'][i]
                # c. Extended
                if _re.fullmatch(r"Extended", t, flags=_re.I):
                    centers['extended'] = ocr['left'][i] + ocr['width'][i] / 2.0
                    label_y = ocr['top'][i]

            # Require the Not Observed label center to be found to proceed
            if 'not_observed' not in centers or label_y is None:
                return False

            # Collect X marks in a band below the label row
            x_marks = []
            for i in range(n):
                t = (ocr['text'][i] or '').strip()
                if not t:
                    continue
                # Candidate tokens indicating a mark
                if (_re.fullmatch(r"\[?\s*[xX]\s*\]?", t)
                    or t in ('X', 'x', '[x]', '[X]', 'x]', '[x', '[ X', '[X')
                    or ('x' in t.lower() and len(t) <= 3)):
                    dy = ocr['top'][i] - label_y
                    if 0 <= dy <= 120:  # within reasonable distance below labels
                        cx = ocr['left'][i] + ocr['width'][i] / 2.0
                        x_marks.append((cx, ocr['top'][i]))

            if not x_marks:
                return False

            # Determine if an X is near the Not Observed column center
            target = centers['not_observed']
            nearest = min(x_marks, key=lambda a: abs(a[0] - target))
            # Tolerance chosen based on empirical spacing across sample PDFs
            return abs(nearest[0] - target) < 80

        except Exception:
            # If anything goes wrong, be conservative and do not mark as Not Observed
            return False
    
    def extract_checkbox_values_text_based(self, pdf_doc, page_num, expected_count):
        """Extract checkbox values using direct PDF text extraction - much more reliable"""
        if page_num >= len(pdf_doc):
            return [4] * expected_count
        
        page = pdf_doc[page_num]
        
        # Prefer span-level text detection for 'X' marks; fallback to blocks
        x_marks = []
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for sp in line.get("spans", []):
                        s = (sp.get("text", "") or "").strip()
                        if s in {"X", "x"}:
                            bx0, by0, bx1, by1 = sp.get("bbox", [0, 0, 0, 0])
                            x_marks.append({"text": s, "x": (bx0 + bx1)/2.0, "y": (by0 + by1)/2.0})
        except Exception:
            x_marks = []
        if not x_marks:
            blocks = page.get_text("blocks")
            for i, block in enumerate(blocks):
                if len(block) >= 5:
                    x0, y0, x1, y1, text = block[:5]
                    text_clean = text.strip()
                    if text_clean == "X":
                        x_marks.append({
                            "text": text_clean,
                            "x": (x0 + x1) / 2,
                            "y": (y0 + y1) / 2,
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
        
        # Convert to values using A–H header centers when available; otherwise equal-split fallback
        # Optional: compute strict header centers for tie-breaks only
        ordered_centers = None
        try:
            text = page.get_text("dict")
            letter_spans = []
            for block in text.get("blocks", []):
                for line in block.get("lines", []):
                    for sp in line.get("spans", []):
                        s = (sp.get("text", "") or "").strip()
                        if len(s) == 1 and s in "ABCDEFGH":
                            (x0, y0, x1, y1) = sp.get("bbox", [0, 0, 0, 0])
                            letter_spans.append({'ch': s, 'x': (x0+x1)/2.0, 'y': (y0+y1)/2.0})
            if letter_spans:
                letter_spans.sort(key=lambda a: a['y'])
                rows_letters = []
                cur = [letter_spans[0]]
                for it in letter_spans[1:]:
                    if abs(it['y'] - cur[0]['y']) < 6:
                        cur.append(it)
                    else:
                        rows_letters.append(cur)
                        cur = [it]
                if cur:
                    rows_letters.append(cur)
                min_xmark_y = min((xm['y'] for xm in x_marks), default=1e9)
                # strict: above X rows, wide span, sufficient unique letters
                candidates = []
                for r in rows_letters:
                    uniq = len({a['ch'] for a in r})
                    xs_r = [a['x'] for a in r]
                    if not xs_r:
                        continue
                    span = max(xs_r) - min(xs_r)
                    y_avg = sum(a['y'] for a in r) / len(r)
                    if uniq >= 6 and span >= 180 and y_avg < (min_xmark_y - 20):
                        candidates.append((span, uniq, -abs(min_xmark_y - y_avg), r))
                if candidates:
                    candidates.sort(reverse=True)
                    best = candidates[0][3]
                    best.sort(key=lambda a: a['x'])
                    cmap = {a['ch']: a['x'] for a in best}
                    letters = list('ABCDEFGH')
                    known = [(ch, cmap[ch]) for ch in letters if ch in cmap]
                    if known:
                        known.sort(key=lambda a: a[1])
                        min_ch, min_x = known[0]
                        max_ch, max_x = known[-1]
                        idx_min = letters.index(min_ch)
                        idx_max = letters.index(max_ch)
                        span = max(idx_max - idx_min, 1)
                        step = (max_x - min_x) / span
                        ordered_centers = []
                        for i, ch in enumerate(letters):
                            if ch in cmap:
                                ordered_centers.append(cmap[ch])
                            else:
                                ordered_centers.append(min_x + (i - idx_min) * step)
        except Exception:
            ordered_centers = None

        # Map using calibrated positional thresholds (empirically stable across these PDFs)
        values = []
        import os as _os
        _diag = _os.getenv('FITREP_DIAG_TB', '')
        _row_xs = []
        for row_idx in range(expected_count):
            if row_idx < len(rows):
                row_x_marks = rows[row_idx]
                if row_x_marks:
                    x_pos = row_x_marks[0]['x']
                    _row_xs.append(float(x_pos))
                    if x_pos < 120:
                        column = 2
                    elif x_pos < 180:
                        column = 3
                    elif x_pos < 250:
                        column = 4
                    elif x_pos < 340:
                        column = 5
                    elif x_pos < 440:
                        column = 6
                    elif x_pos < 520:
                        column = 7
                    elif x_pos < 600:
                        column = 8
                    elif x_pos < 700:
                        column = 2
                    else:
                        column = 1
                    # If strict header centers are available, allow a small-margin override
                    if ordered_centers:
                        try:
                            spanC = max(ordered_centers) - min(ordered_centers)
                            margin = max(6.0, 0.04 * spanC)
                            # nearest by header
                            idx_header = min(range(8), key=lambda k: abs(ordered_centers[k] - x_pos))
                            d_hdr = abs(ordered_centers[idx_header] - x_pos)
                            d_cur = abs(ordered_centers[column - 1] - x_pos)
                            if d_hdr + margin < d_cur:
                                column = idx_header + 1
                        except Exception:
                            pass
                    values.append(column)
                else:
                    values.append(4)
            else:
                values.append(4)
        if _diag:
            try:
                print(f"TBDBG p{page_num+1}/{expected_count}: xs={_row_xs} centers={ordered_centers}")
            except Exception:
                pass
        return values

    def extract_checkbox_values_ocr_fallback(self, pdf_doc, page_num, expected_count):
        """
        OCR-based fallback for checkbox extraction, anchored to the A–H letter row.
        This is only used in guarded modes to avoid regressions.
        Returns a list of length expected_count.
        """
        try:
            if page_num >= len(pdf_doc):
                return [4] * expected_count

            page = pdf_doc[page_num]
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # OCR the whole page to find the letter header row and X marks
            ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            n = len(ocr.get('text', []))
            if n == 0:
                return [4] * expected_count

            # 1) Find the A–H header row by grouping single-letter tokens
            single_letters = []
            for i in range(n):
                t = (ocr['text'][i] or '').strip()
                if len(t) == 1 and t.upper() in list('ABCDEFGH'):
                    single_letters.append({
                        'ch': t.upper(),
                        'x': ocr['left'][i] + ocr['width'][i] / 2.0,
                        'y': ocr['top'][i] + ocr['height'][i] / 2.0,
                    })

            if not single_letters:
                return [4] * expected_count

            # Group letters by Y proximity to find a horizontal row
            single_letters.sort(key=lambda a: a['y'])
            letter_rows = []
            current = [single_letters[0]]
            for s in single_letters[1:]:
                if abs(s['y'] - current[0]['y']) < 20:
                    current.append(s)
                else:
                    letter_rows.append(current)
                    current = [s]
            if current:
                letter_rows.append(current)

            # Choose the row with the most unique A–H letters and reasonable width
            best_row = None
            best_unique = -1
            for row in letter_rows:
                uniq = {r['ch'] for r in row}
                if len(uniq) >= 6:  # need a good spread
                    xs = [r['x'] for r in row]
                    width = max(xs) - min(xs) if xs else 0
                    if width > 200 and len(uniq) > best_unique:
                        best_row = row
                        best_unique = len(uniq)

            if not best_row:
                return [4] * expected_count

            # Build ordered centers A..H by nearest mapping from found letters
            best_row.sort(key=lambda a: a['x'])
            centers_map = {}
            for item in best_row:
                centers_map[item['ch']] = item['x']

            # Interpolate missing letters if any
            ordered = []
            letters = list('ABCDEFGH')
            known = [(ch, centers_map[ch]) for ch in letters if ch in centers_map]
            if not known:
                return [4] * expected_count
            # If we don't have all 8, linearly interpolate between known extremes
            known.sort(key=lambda a: a[1])
            min_ch, min_x = known[0]
            max_ch, max_x = known[-1]
            idx_min = letters.index(min_ch)
            idx_max = letters.index(max_ch)
            span = max(idx_max - idx_min, 1)
            step = (max_x - min_x) / span
            for i, ch in enumerate(letters):
                if ch in centers_map:
                    ordered.append(centers_map[ch])
                else:
                    # interpolate by index relative to min
                    ordered.append(min_x + (i - idx_min) * step)

            header_y = sum(x['y'] for x in best_row) / len(best_row)

            # 2) Darkness scan in horizontal bands below header to locate the darkest column per row
            img_gray = img.convert('L')

            def col_from_x(x):
                idx = min(range(len(ordered)), key=lambda k: abs(ordered[k] - x))
                return idx + 1

            # Define scanning region below the header
            y_start = int(min(max(header_y + 60, 0), img_gray.height - 1))
            # Heuristic row height
            seg_h = max(40, min(90, (img_gray.height - y_start - 10) // max(1, expected_count)))

            values = []
            for row_idx in range(expected_count):
                y0 = int(y_start + row_idx * seg_h)
                y1 = int(min(y0 + seg_h, img_gray.height))
                if y0 >= img_gray.height:
                    values.append(4)
                    continue
                # For each column, sample a narrow vertical slice and compute darkness
                best_col = 4
                best_dark = -1
                for cx in ordered:
                    cx = int(cx)
                    x0 = max(0, cx - 14)
                    x1 = min(img_gray.width, cx + 14)
                    if x0 >= x1 or y0 >= y1:
                        continue
                    crop = img_gray.crop((x0, y0, x1, y1))
                    # Darkness = sum(255 - pixel)
                    dark = 0
                    for p in crop.getdata():
                        dark += (255 - p)
                    if dark > best_dark:
                        best_dark = dark
                        best_col = col_from_x(cx)
                values.append(best_col)

            return values
        except Exception:
            return [4] * expected_count

        except Exception:
            return [4] * expected_count

    def extract_checkbox_values_vector_paths(self, pdf_doc, page_num, expected_count):
        """
        Vector-graphics fallback: detect drawn X-marks from PDF line segments.
        - Finds the A–H header via PDF text and computes column centers.
        - Uses page.get_drawings() to collect diagonal line segments.
        - Clusters intersecting diagonal pairs into X centers and maps them to columns.
        """
        try:
            if page_num >= len(pdf_doc):
                return [4] * expected_count

            page = pdf_doc[page_num]

            # 1) Find A–H header via PDF text
            text = page.get_text("dict")
            letter_spans = []
            for block in text.get("blocks", []):
                for line in block.get("lines", []):
                    for sp in line.get("spans", []):
                        s = (sp.get("text", "") or "").strip()
                        if len(s) == 1 and s in "ABCDEFGH":
                            (x0, y0, x1, y1) = sp.get("bbox", [0, 0, 0, 0])
                            letter_spans.append({
                                'ch': s,
                                'x': (x0 + x1) / 2.0,
                                'y': (y0 + y1) / 2.0,
                            })

            if not letter_spans:
                return [4] * expected_count

            letter_spans.sort(key=lambda a: a['y'])
            rows = []
            cur = [letter_spans[0]]
            for it in letter_spans[1:]:
                if abs(it['y'] - cur[0]['y']) < 6:  # tight row tolerance in PDF coords
                    cur.append(it)
                else:
                    rows.append(cur)
                    cur = [it]
            if cur:
                rows.append(cur)

            best_row = None
            best_unique = -1
            for r in rows:
                uniq = {a['ch'] for a in r}
                if len(uniq) >= 6:
                    xs = [a['x'] for a in r]
                    if xs and (max(xs) - min(xs)) > 150:
                        if len(uniq) > best_unique:
                            best_row = r
                            best_unique = len(uniq)

            if not best_row:
                return [4] * expected_count

            best_row.sort(key=lambda a: a['x'])
            centers_map = {a['ch']: a['x'] for a in best_row}
            letters = list('ABCDEFGH')
            known = [(ch, centers_map[ch]) for ch in letters if ch in centers_map]
            if not known:
                return [4] * expected_count
            known.sort(key=lambda a: a[1])
            min_ch, min_x = known[0]
            max_ch, max_x = known[-1]
            idx_min = letters.index(min_ch)
            idx_max = letters.index(max_ch)
            span = max(idx_max - idx_min, 1)
            step = (max_x - min_x) / span
            ordered = []
            for i, ch in enumerate(letters):
                if ch in centers_map:
                    ordered.append(centers_map[ch])
                else:
                    ordered.append(min_x + (i - idx_min) * step)

            header_y = sum(a['y'] for a in best_row) / len(best_row)

            # 2) Collect diagonal line segments from vector drawings
            def diag_segments():
                segs = []
                for d in page.get_drawings():
                    for it in d.get('items', []):
                        op = it[0]
                        if op == 'l':  # line
                            p0, p1 = it[1], it[2]
                            x0, y0 = float(p0[0]), float(p0[1])
                            x1, y1 = float(p1[0]), float(p1[1])
                            dx, dy = x1 - x0, y1 - y0
                            if dx == 0:
                                slope = None
                            else:
                                slope = abs(dy / dx)
                            length = (dx*dx + dy*dy) ** 0.5
                            # Heuristics: diagonal-ish, moderate length, below header
                            if slope is not None and 0.5 <= slope <= 2.0 and 5 <= length <= 40:
                                y_mid = (y0 + y1) / 2.0
                                if header_y + 5 <= y_mid <= header_y + 450:
                                    segs.append((x0, y0, x1, y1))
                return segs

            segs = diag_segments()
            if not segs:
                return [4] * expected_count

            # 3) Find intersections between diagonal segment pairs to localize X centers
            def intersects(a, b):
                # Segment intersection test with tolerance
                import math
                (x1, y1, x2, y2) = a
                (x3, y3, x4, y4) = b

                def det(u1, v1, u2, v2):
                    return u1 * v2 - v1 * u2

                den = det(x1 - x2, y1 - y2, x3 - x4, y3 - y4)
                if abs(den) < 1e-6:
                    return None
                px = det(det(x1, y1, x2, y2), x1 - x2, det(x3, y3, x4, y4), x3 - x4) / den
                py = det(det(x1, y1, x2, y2), y1 - y2, det(x3, y3, x4, y4), y3 - y4) / den

                def on_seg(xa, ya, xb, yb, px, py):
                    return (min(xa, xb) - 1 <= px <= max(xa, xb) + 1 and
                            min(ya, yb) - 1 <= py <= max(ya, yb) + 1)

                if on_seg(x1, y1, x2, y2, px, py) and on_seg(x3, y3, x4, y4, px, py):
                    return (px, py)
                return None

            pts = []
            for i in range(len(segs)):
                for j in range(i + 1, len(segs)):
                    pt = intersects(segs[i], segs[j])
                    if pt is not None:
                        # only consider sufficiently crossing angles by checking vector dot product
                        import math
                        x1, y1, x2, y2 = segs[i]
                        x3, y3, x4, y4 = segs[j]
                        v1 = (x2 - x1, y2 - y1)
                        v2 = (x4 - x3, y4 - y3)
                        def norm(v):
                            return (v[0]*v[0] + v[1]*v[1]) ** 0.5
                        n1, n2 = norm(v1), norm(v2)
                        if n1 > 0 and n2 > 0:
                            cosang = abs((v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2))
                            if cosang < 0.5:  # > ~60 degrees apart
                                pts.append(pt)

            if not pts:
                return [4] * expected_count

            # 4) Cluster intersection points by Y to get rows
            pts.sort(key=lambda p: p[1])
            rows = []
            cur = [pts[0]]
            for p in pts[1:]:
                if abs(p[1] - cur[0][1]) < 18:
                    cur.append(p)
                else:
                    rows.append(cur)
                    cur = [p]
            if cur:
                rows.append(cur)

            # choose one point per row (e.g., leftmost)
            centers = [sorted(r, key=lambda a: a[0])[0] for r in rows[:expected_count]]

            def col_from_x(x):
                idx = min(range(len(ordered)), key=lambda k: abs(ordered[k] - x))
                return idx + 1

            values = [col_from_x(cx) for (cx, cy) in centers]
            if len(values) < expected_count:
                values += [4] * (expected_count - len(values))
            return values
        except Exception:
            return [4] * expected_count

    def extract_checkbox_values_image_peaks(self, pdf_doc, page_num, expected_count):
        """
        Image-based fallback: find per-row Y positions by vertical darkness peaks
        per column, then pick the darkest column at each row.
        Uses PDF text to locate A–H header and compute column centers, then
        operates purely on the rendered grayscale image.
        """
        try:
            if page_num >= len(pdf_doc):
                return [4] * expected_count

            page = pdf_doc[page_num]
            # Build column centers via PDF text (A–H row)
            text = page.get_text("dict")
            spans = []
            for block in text.get("blocks", []):
                for line in block.get("lines", []):
                    for sp in line.get("spans", []):
                        s = (sp.get("text", "") or "").strip()
                        if len(s) == 1 and s in "ABCDEFGH":
                            (x0, y0, x1, y1) = sp.get("bbox", [0, 0, 0, 0])
                            spans.append({'ch': s, 'x': (x0 + x1) / 2.0, 'y': (y0 + y1) / 2.0})
            if not spans:
                return [4] * expected_count
            spans.sort(key=lambda a: a['y'])
            rows = []
            cur = [spans[0]]
            for s in spans[1:]:
                if abs(s['y'] - cur[0]['y']) < 6:
                    cur.append(s)
                else:
                    rows.append(cur)
                    cur = [s]
            if cur:
                rows.append(cur)
            best = None
            bestu = -1
            for r in rows:
                u = {a['ch'] for a in r}
                if len(u) >= 6:
                    xs = [a['x'] for a in r]
                    if xs and (max(xs) - min(xs)) > 150 and len(u) > bestu:
                        best = r
                        bestu = len(u)
            if not best:
                return [4] * expected_count
            best.sort(key=lambda a: a['x'])
            centers_map = {a['ch']: a['x'] for a in best}
            letters = list('ABCDEFGH')
            known = [(ch, centers_map[ch]) for ch in letters if ch in centers_map]
            known.sort(key=lambda a: a[1])
            min_ch, min_x = known[0]
            max_ch, max_x = known[-1]
            idx_min = letters.index(min_ch)
            idx_max = letters.index(max_ch)
            span = max(idx_max - idx_min, 1)
            step = (max_x - min_x) / span
            centers_pts = []
            for i, ch in enumerate(letters):
                if ch in centers_map:
                    centers_pts.append(centers_map[ch])
                else:
                    centers_pts.append(min_x + (i - idx_min) * step)
            header_y_pt = sum(a['y'] for a in best) / len(best)

            # Render page and map points to pixels (scale=3)
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(pix.tobytes("png"))).convert('L')
            scale = 3.0
            centers_px = [int(x * scale) for x in centers_pts]
            header_y_px = int(header_y_pt * scale)

            # For each column, compute vertical X-correlation profile using diagonal kernels
            import numpy as np
            arr = np.array(img)
            H, W = arr.shape
            y_start = min(max(header_y_px + 40, 0), H - 1)
            band = arr[y_start:H, :]
            dark_full = 255 - band

            ksz = 21
            half = ksz // 2
            # Build diagonal kernels (positive and negative slope) with thickness 2
            k_pos = np.zeros((ksz, ksz), dtype=np.float32)
            k_neg = np.zeros((ksz, ksz), dtype=np.float32)
            for i in range(ksz):
                j = i
                for t in (-1, 0, 1):
                    jj = j + t
                    if 0 <= jj < ksz:
                        k_pos[i, jj] = 1.0
                j2 = ksz - 1 - i
                for t in (-1, 0, 1):
                    jj2 = j2 + t
                    if 0 <= jj2 < ksz:
                        k_neg[i, jj2] = 1.0
            # Normalize kernels
            k_pos /= k_pos.sum() or 1.0
            k_neg /= k_neg.sum() or 1.0

            half_w = half
            col_profiles = []
            for cx in centers_px:
                x0 = max(0, cx - half_w)
                x1 = min(W, cx + half_w + 1)
                strip = dark_full[:, x0:x1]
                # Pad vertically to compute patch correlations
                vlen = strip.shape[0]
                scores = np.zeros(vlen, dtype=np.float32)
                # Convolve per y by extracting local patches
                for y in range(vlen):
                    y0 = y - half
                    y1 = y + half + 1
                    if y0 < 0 or y1 > vlen:
                        continue
                    patch = strip[y0:y1, :]
                    if patch.shape[0] != ksz or patch.shape[1] != ksz:
                        continue
                    s = (patch * k_pos).sum() + (patch * k_neg).sum()
                    scores[y] = s
                # Smooth slightly
                k = np.ones(15, dtype=np.float32)
                v2 = np.convolve(scores, k, mode='same')
                col_profiles.append(v2)

            # Detect peaks across columns and cluster into rows
            # Collect candidate peaks per column
            peaks = []
            min_sep = 50
            for ci, v in enumerate(col_profiles):
                # simple peak picking: local maxima with minimal separation
                last_y = -9999
                for y in range(1, len(v) - 1):
                    if v[y] > v[y-1] and v[y] >= v[y+1]:
                        if v[y] > 0:  # any darkness
                            if y - last_y >= min_sep:
                                peaks.append((y, ci, int(v[y])))
                                last_y = y

            if not peaks:
                return [4] * expected_count

            # Cluster peaks by y with tolerance to form rows
            peaks.sort(key=lambda a: a[0])
            rows = []
            cur = [peaks[0]]
            for p in peaks[1:]:
                if abs(p[0] - cur[0][0]) < 25:
                    cur.append(p)
                else:
                    rows.append(cur)
                    cur = [p]
            if cur:
                rows.append(cur)

            # Score each row by total darkness and keep top expected_count rows
            rows_scored = []
            for r in rows:
                score = sum(p[2] for p in r)
                rows_scored.append((score, r))
            rows_scored.sort(key=lambda a: a[0], reverse=True)
            chosen = rows_scored[:expected_count]
            # Order by y ascending
            chosen_sorted = sorted(chosen, key=lambda a: a[1][0][0])

            values = []
            for _, r in chosen_sorted:
                # pick column with max peak in this row cluster
                best = max(r, key=lambda p: p[2])
                col_idx = best[1]
                values.append(col_idx + 1)  # 1..8

            if len(values) < expected_count:
                values += [4] * (expected_count - len(values))
            return values
        except Exception:
            return [4] * expected_count

    def extract_checkbox_values_row_bands(self, pdf_doc, page_num, expected_count):
        """
        Image-based fallback focusing on row detection:
        - Build diagonal X correlation per column as in image_peaks.
        - Sum across columns to get a per-Y row energy curve.
        - Select top expected_count peaks as row centers (with separation).
        - For each row center, choose the column with the highest diagonal score.
        """
        try:
            if page_num >= len(pdf_doc):
                return [4] * expected_count

            page = pdf_doc[page_num]

            # Centers from A–H PDF text
            text = page.get_text("dict")
            spans = []
            for block in text.get("blocks", []):
                for line in block.get("lines", []):
                    for sp in line.get("spans", []):
                        s = (sp.get("text", "") or "").strip()
                        if len(s) == 1 and s in "ABCDEFGH":
                            (x0, y0, x1, y1) = sp.get("bbox", [0, 0, 0, 0])
                            spans.append({'ch': s, 'x': (x0 + x1) / 2.0, 'y': (y0 + y1) / 2.0})
            if not spans:
                return [4] * expected_count
            spans.sort(key=lambda a: a['y'])
            rows = []
            cur = [spans[0]]
            for s in spans[1:]:
                if abs(s['y'] - cur[0]['y']) < 6:
                    cur.append(s)
                else:
                    rows.append(cur)
                    cur = [s]
            if cur:
                rows.append(cur)
            best = None
            bestu = -1
            for r in rows:
                u = {a['ch'] for a in r}
                if len(u) >= 6:
                    xs = [a['x'] for a in r]
                    if xs and (max(xs) - min(xs)) > 150 and len(u) > bestu:
                        best = r
                        bestu = len(u)
            if not best:
                return [4] * expected_count
            best.sort(key=lambda a: a['x'])
            letters = list('ABCDEFGH')
            centers_map = {a['ch']: a['x'] for a in best}
            known = [(ch, centers_map[ch]) for ch in letters if ch in centers_map]
            known.sort(key=lambda a: a[1])
            min_ch, min_x = known[0]
            max_ch, max_x = known[-1]
            idx_min = letters.index(min_ch)
            idx_max = letters.index(max_ch)
            span = max(idx_max - idx_min, 1)
            step = (max_x - min_x) / span
            centers_pts = []
            for i, ch in enumerate(letters):
                if ch in centers_map:
                    centers_pts.append(centers_map[ch])
                else:
                    centers_pts.append(min_x + (i - idx_min) * step)
            header_y_pt = sum(a['y'] for a in best) / len(best)

            # Render page to grayscale
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(pix.tobytes("png"))).convert('L')
            scale = 3.0
            centers_px = [int(x * scale) for x in centers_pts]
            header_y_px = int(header_y_pt * scale)

            import numpy as np
            arr = np.array(img)
            H, W = arr.shape
            y0 = min(max(header_y_px + 40, 0), H - 1)
            band = arr[y0:H, :]
            dark = 255 - band

            # Diagonal kernels
            ksz = 19
            half = ksz // 2
            k_pos = np.zeros((ksz, ksz), dtype=np.float32)
            k_neg = np.zeros((ksz, ksz), dtype=np.float32)
            for i in range(ksz):
                j = i
                for t in (-1, 0, 1):
                    jj = j + t
                    if 0 <= jj < ksz:
                        k_pos[i, jj] = 1.0
                j2 = ksz - 1 - i
                for t in (-1, 0, 1):
                    jj2 = j2 + t
                    if 0 <= jj2 < ksz:
                        k_neg[i, jj2] = 1.0
            k_pos /= k_pos.sum() or 1.0
            k_neg /= k_neg.sum() or 1.0

            # Column correlation profiles
            half_w = half
            col_profiles = []
            for cx in centers_px:
                x0c = max(0, cx - half_w)
                x1c = min(W, cx + half_w + 1)
                strip = dark[:, x0c:x1c]
                vlen = strip.shape[0]
                scores = np.zeros(vlen, dtype=np.float32)
                for yi in range(vlen):
                    yb = yi - half
                    ye = yi + half + 1
                    if yb < 0 or ye > vlen:
                        continue
                    patch = strip[yb:ye, :]
                    if patch.shape[0] != ksz or patch.shape[1] != ksz:
                        continue
                    s = (patch * k_pos).sum() + (patch * k_neg).sum()
                    scores[yi] = s
                # Smooth
                k = np.ones(13, dtype=np.float32)
                v2 = np.convolve(scores, k, mode='same')
                col_profiles.append(v2)

            # Row energy by summing across columns
            row_energy = np.sum(np.vstack(col_profiles), axis=0)
            # Peak picking with separation
            peaks = []
            last = -9999
            min_sep = 45
            for y in range(1, len(row_energy) - 1):
                if row_energy[y] > row_energy[y-1] and row_energy[y] >= row_energy[y+1]:
                    if y - last >= min_sep:
                        peaks.append((row_energy[y], y))
                        last = y
            if not peaks:
                return [4] * expected_count
            # Take top expected_count peaks
            peaks.sort(key=lambda a: a[0], reverse=True)
            chosen = sorted(peaks[:expected_count], key=lambda a: a[1])

            # Choose column per chosen row using simple darkness in a local box
            values = []
            box_h = 28
            box_w = 28
            for _, y in chosen:
                # y is relative to band; map to absolute pixel in arr
                yy = y0 + y
                scores = []
                for cx in centers_px:
                    x0b = max(0, cx - box_w//2)
                    x1b = min(W, cx + box_w//2)
                    y0b = max(0, yy - box_h//2)
                    y1b = min(H, yy + box_h//2)
                    crop = arr[y0b:y1b, x0b:x1b]
                    dark_score = int((255 - crop).sum())
                    scores.append(dark_score)
                col_idx = int(np.argmax(scores))
                values.append(col_idx + 1)

            if len(values) < expected_count:
                values += [4] * (expected_count - len(values))
            return values
        except Exception:
            return [4] * expected_count

    def debug_checkbox_diagnostics(self, pdf_doc, page_num, expected_count):
        """
        Produce numeric diagnostics for checkbox detection without PII:
        - header_y_px (int)
        - centers_px (8 ints)
        - row_energy_peaks (top K [score, y])
        - chosen_rows_y (expected_count y's)
        - col_darkness_per_row: list of per-column darkness scores per chosen row
        - values_row_bands: values chosen by row-band method
        - values_text_based: values from text-based method
        """
        info = {
            'header_y_px': None,
            'centers_px': [],
            'row_energy_peaks': [],
            'chosen_rows_y': [],
            'col_darkness_per_row': [],
            'values_row_bands': [],
            'values_text_based': [],
        }
        try:
            if page_num >= len(pdf_doc):
                return info
            # Always include baseline text-based values
            info['values_text_based'] = self.extract_checkbox_values_text_based(pdf_doc, page_num, expected_count)
            # 1) Centers from A–H PDF text
            page = pdf_doc[page_num]
            text = page.get_text("dict")
            spans = []
            for block in text.get("blocks", []):
                for line in block.get("lines", []):
                    for sp in line.get("spans", []):
                        s = (sp.get("text", "") or "").strip()
                        if len(s) == 1 and s in "ABCDEFGH":
                            (x0, y0, x1, y1) = sp.get("bbox", [0, 0, 0, 0])
                            spans.append({'ch': s, 'x': (x0 + x1) / 2.0, 'y': (y0 + y1) / 2.0})
            if not spans:
                # Fallback: OCR to find A–H header letters
                mat = fitz.Matrix(3, 3)
                pix = page.get_pixmap(matrix=mat)
                img_ocr = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr = pytesseract.image_to_data(img_ocr, output_type=pytesseract.Output.DICT)
                n = len(ocr.get('text', []))
                for i in range(n):
                    t = (ocr['text'][i] or '').strip()
                    if len(t) == 1 and t.upper() in list('ABCDEFGH'):
                        x = ocr['left'][i] + ocr['width'][i] / 2.0
                        y = ocr['top'][i] + ocr['height'][i] / 2.0
                        spans.append({'ch': t.upper(), 'x': x / 3.0, 'y': y / 3.0})
                # if still no spans, we can only report baseline values
                if not spans:
                    return info
            spans.sort(key=lambda a: a['y'])
            rows = []
            cur = [spans[0]]
            for s in spans[1:]:
                if abs(s['y'] - cur[0]['y']) < 6:
                    cur.append(s)
                else:
                    rows.append(cur)
                    cur = [s]
            if cur:
                rows.append(cur)
            best = None
            bestu = -1
            for r in rows:
                u = {a['ch'] for a in r}
                if len(u) >= 6:
                    xs = [a['x'] for a in r]
                    if xs and (max(xs) - min(xs)) > 150 and len(u) > bestu:
                        best = r
                        bestu = len(u)
            if not best:
                return info
            best.sort(key=lambda a: a['x'])
            letters = list('ABCDEFGH')
            centers_map = {a['ch']: a['x'] for a in best}
            known = [(ch, centers_map[ch]) for ch in letters if ch in centers_map]
            known.sort(key=lambda a: a[1])
            min_ch, min_x = known[0]
            max_ch, max_x = known[-1]
            idx_min = letters.index(min_ch)
            idx_max = letters.index(max_ch)
            span = max(idx_max - idx_min, 1)
            step = (max_x - min_x) / span
            centers_pts = []
            for i, ch in enumerate(letters):
                if ch in centers_map:
                    centers_pts.append(centers_map[ch])
                else:
                    centers_pts.append(min_x + (i - idx_min) * step)
            header_y_pt = sum(a['y'] for a in best) / len(best)

            # Render page to grayscale
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert('L')
            scale = 3.0
            centers_px = [int(x * scale) for x in centers_pts]
            header_y_px = int(header_y_pt * scale)
            info['header_y_px'] = header_y_px
            info['centers_px'] = centers_px

            import numpy as np
            arr = np.array(img)
            H, W = arr.shape
            y0 = min(max(header_y_px + 40, 0), H - 1)
            band = arr[y0:H, :]
            dark = 255 - band

            # Build column correlation profiles (reuse row_bands kernels)
            ksz = 19
            half = ksz // 2
            k_pos = np.zeros((ksz, ksz), dtype=np.float32)
            k_neg = np.zeros((ksz, ksz), dtype=np.float32)
            for i in range(ksz):
                j = i
                for t in (-1, 0, 1):
                    jj = j + t
                    if 0 <= jj < ksz:
                        k_pos[i, jj] = 1.0
                j2 = ksz - 1 - i
                for t in (-1, 0, 1):
                    jj2 = j2 + t
                    if 0 <= jj2 < ksz:
                        k_neg[i, jj2] = 1.0
            k_pos /= k_pos.sum() or 1.0
            k_neg /= k_neg.sum() or 1.0

            half_w = half
            col_profiles = []
            for cx in centers_px:
                x0c = max(0, cx - half_w)
                x1c = min(W, cx + half_w + 1)
                strip = dark[:, x0c:x1c]
                vlen = strip.shape[0]
                scores = np.zeros(vlen, dtype=np.float32)
                for yi in range(vlen):
                    yb = yi - half
                    ye = yi + half + 1
                    if yb < 0 or ye > vlen:
                        continue
                    patch = strip[yb:ye, :]
                    if patch.shape[0] != ksz or patch.shape[1] != ksz:
                        continue
                    s = (patch * k_pos).sum() + (patch * k_neg).sum()
                    scores[yi] = s
                # Smooth
                k = np.ones(13, dtype=np.float32)
                v2 = np.convolve(scores, k, mode='same')
                col_profiles.append(v2)

            row_energy = np.sum(np.vstack(col_profiles), axis=0)
            # peaks with separation
            peaks = []
            last = -9999
            min_sep = 45
            for y in range(1, len(row_energy) - 1):
                if row_energy[y] > row_energy[y-1] and row_energy[y] >= row_energy[y+1]:
                    if y - last >= min_sep:
                        peaks.append((float(row_energy[y]), int(y)))
                        last = y
            peaks.sort(key=lambda a: a[0], reverse=True)
            info['row_energy_peaks'] = peaks[:10]

            chosen = sorted(peaks[:expected_count], key=lambda a: a[1])
            chosen_rows = [int(y) for (_, y) in chosen]
            info['chosen_rows_y'] = chosen_rows

            # Darkness box per chosen row
            box_h = 28
            box_w = 28
            per_row = []
            for y in chosen_rows:
                yy = y0 + y
                row_scores = []
                for cx in centers_px:
                    x0b = max(0, cx - box_w//2)
                    x1b = min(W, cx + box_w//2)
                    y0b = max(0, yy - box_h//2)
                    y1b = min(H, yy + box_h//2)
                    crop = arr[y0b:y1b, x0b:x1b]
                    row_scores.append(int((255 - crop).sum()))
                per_row.append(row_scores)
            info['col_darkness_per_row'] = per_row

            # Values via our method
            vals_rb = []
            for scores in per_row:
                if scores:
                    col_idx = int(np.argmax(scores))
                    vals_rb.append(col_idx + 1)
            while len(vals_rb) < expected_count:
                vals_rb.append(4)
            info['values_row_bands'] = vals_rb[:expected_count]

            # Baseline text-based
            info['values_text_based'] = self.extract_checkbox_values_text_based(pdf_doc, page_num, expected_count)
            return info
        except Exception:
            return info

    def extract_checkbox_values_auto(self, pdf_doc, page_num, expected_count):
        """
        Wrapper that applies text-based extraction and conditionally applies OCR fallback
        based on the configured mode.
        """
        tb = self.extract_checkbox_values_text_based(pdf_doc, page_num, expected_count)
        mode = self.checkbox_fallback_mode
        if mode == 'off':
            # Fast path: return text-based as-is (allow all-4s when no X marks are present)
            return tb
        if mode == 'force':
            fb = self.extract_checkbox_values_ocr_fallback(pdf_doc, page_num, expected_count)
            result = fb or tb
            if self.strict_no_defaults and isinstance(result, list) and len(result) == expected_count:
                if result.count(4) == expected_count:
                    return None
            return result

        # auto: prefer robust grid-based methods first
        # 1) Vector grid
        gv = self.extract_checkbox_values_grid_vector(pdf_doc, page_num, expected_count)
        if gv and gv != [4] * expected_count and len(set(gv)) != 1:
            result = gv
        else:
            # 2) Mark-in-box (vector+raster) — no X required
            mf = self.extract_checkbox_values_grid_markfill(pdf_doc, page_num, expected_count)
            if mf and mf != [4] * expected_count and len(set(mf)) != 1:
                result = mf
            else:
                # 3) Grid hybrid (vector+raster)
                gh = self.extract_checkbox_values_grid_hybrid(pdf_doc, page_num, expected_count)
                if gh and gh != [4] * expected_count and len(set(gh)) != 1:
                    result = gh
                else:
                # 4) Row-band image correlation
                    rb = self.extract_checkbox_values_row_bands(pdf_doc, page_num, expected_count)
                    if rb and rb != [4] * expected_count and len(set(rb)) != 1:
                        result = rb
                    else:
                    # 5) Image peaks
                        ip = self.extract_checkbox_values_image_peaks(pdf_doc, page_num, expected_count)
                        if ip and ip != [4] * expected_count and len(set(ip)) != 1:
                            result = ip
                        else:
                        # 6) Vector paths (drawn X)
                            vp = self.extract_checkbox_values_vector_paths(pdf_doc, page_num, expected_count)
                            if vp and vp != [4] * expected_count:
                                result = vp
                            else:
                            # 7) Grid image (raster)
                                gi = self.extract_checkbox_values_grid_image(pdf_doc, page_num, expected_count)
                                if gi and gi != [4] * expected_count and len(set(gi)) != 1:
                                    result = gi
                                else:
                                # 8) OCR-based fallback or text-based
                                    fb = self.extract_checkbox_values_ocr_fallback(pdf_doc, page_num, expected_count)
                                    result = fb or tb

        # If any row is 1, recheck that row with alternative detectors (no hardcoding)
        if isinstance(result, list) and len(result) == expected_count and 1 in result:
            gh2 = self.extract_checkbox_values_grid_hybrid(pdf_doc, page_num, expected_count)
            rb2 = self.extract_checkbox_values_row_bands(pdf_doc, page_num, expected_count)
            merged = result[:]
            for i, val in enumerate(result):
                if val == 1:
                    candidates = []
                    if gh2:
                        candidates.append(gh2[i])
                    if rb2:
                        candidates.append(rb2[i])
                    # If both alternates agree on a non-1, take it; else leave as-is
                    non1 = [v for v in candidates if v and v != 1]
                    if len(non1) >= 2 and non1[0] == non1[1]:
                        merged[i] = non1[0]
            result = merged
        if mode != 'off' and self.strict_no_defaults and isinstance(result, list) and len(result) == expected_count:
            if result.count(4) == expected_count:
                return None
        return result

    def extract_checkbox_values_grid_image(self, pdf_doc, page_num, expected_count):
        """
        Pure image-based detector for rasterized grids/X's (no text anchors):
        - Render page to grayscale.
        - Identify row centers by vertical projection peaks within likely area.
        - Identify grid left/right by horizontal projection; split into 8 equal columns.
        - For each row, pick the darkest column band.
        """
        try:
            if page_num >= len(pdf_doc):
                return [4] * expected_count
            page = pdf_doc[page_num]
            # Render grayscale at scale 3
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)
            img_gray = Image.open(io.BytesIO(pix.tobytes("png"))).convert('L')
            W, H = img_gray.size
            px = img_gray.load()

            def mov_avg(vals, k):
                if k <= 1 or not vals:
                    return vals[:]
                n = len(vals)
                out = [0.0] * n
                csum = [0.0]
                for v in vals:
                    csum.append(csum[-1] + v)
                half = k // 2
                for i in range(n):
                    a = max(0, i - half)
                    b = min(n, i + half + 1)
                    out[i] = (csum[b] - csum[a]) / max(1, (b - a))
                return out

            # Attempt to locate A–H header centers via OCR as anchors
            def detect_header_centers_ocr(img):
                try:
                    ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
                    n = len(ocr.get('text', []))
                    spans = []
                    for i in range(n):
                        t = (ocr['text'][i] or '').strip()
                        if len(t) == 1 and t.upper() in list('ABCDEFGH'):
                            spans.append({
                                'ch': t.upper(),
                                'x': ocr['left'][i] + ocr['width'][i] / 2.0,
                                'y': ocr['top'][i] + ocr['height'][i] / 2.0,
                            })
                    if not spans:
                        return None, None
                    spans.sort(key=lambda a: a['y'])
                    rows = []
                    cur = [spans[0]]
                    for s in spans[1:]:
                        if abs(s['y'] - cur[0]['y']) < 12:
                            cur.append(s)
                        else:
                            rows.append(cur)
                            cur = [s]
                    if cur:
                        rows.append(cur)
                    best = None
                    bestu = -1
                    for r in rows:
                        u = {a['ch'] for a in r}
                        xs_ = [a['x'] for a in r]
                        if len(u) >= 6 and xs_ and (max(xs_) - min(xs_)) > 150 and len(u) > bestu:
                            best = r
                            bestu = len(u)
                    if not best:
                        return None, None
                    best.sort(key=lambda a: a['x'])
                    cmap = {a['ch']: a['x'] for a in best}
                    letters = list('ABCDEFGH')
                    known = [(ch, cmap[ch]) for ch in letters if ch in cmap]
                    known.sort(key=lambda a: a[1])
                    min_ch, min_x = known[0]
                    max_ch, max_x = known[-1]
                    idx_min = letters.index(min_ch)
                    idx_max = letters.index(max_ch)
                    span = max(idx_max - idx_min, 1)
                    step = (max_x - min_x) / span
                    centers = []
                    for i, ch in enumerate(letters):
                        if ch in cmap:
                            centers.append(cmap[ch])
                        else:
                            centers.append(min_x + (i - idx_min) * step)
                    header_y = sum(a['y'] for a in best) / len(best)
                    return [int(c) for c in centers], int(header_y)
                except Exception:
                    return None, None

            centers_px, header_y_px = detect_header_centers_ocr(img_gray)

            # Row projection in central band
            y_top = int(H * 0.20)
            y_bot = int(H * 0.90)
            row_proj = []
            for y in range(y_top, y_bot):
                s = 0
                for x in range(W):
                    s += 255 - px[x, y]
                row_proj.append(float(s))
            row_sm = mov_avg(row_proj, 41)

            # Row peaks
            peaks = []
            last = -10_000
            min_sep = max(40, (y_bot - y_top) // (expected_count + 1))
            for i in range(1, len(row_sm) - 1):
                if row_sm[i] > row_sm[i - 1] and row_sm[i] >= row_sm[i + 1]:
                    if i - last >= min_sep:
                        peaks.append((row_sm[i], i))
                        last = i
            if not peaks:
                return [4] * expected_count
            peaks.sort(key=lambda a: a[0], reverse=True)
            chosen = sorted(peaks[:expected_count], key=lambda a: a[1])
            row_centers = [y_top + int(y) for (_, y) in chosen]

            # Build columns either from OCR header centers or via projection
            cols = []
            if centers_px and len(centers_px) == 8:
                # Create bounds using midpoints between centers
                bounds = []
                step = max(1, (centers_px[-1] - centers_px[0]) // 7)
                bounds.append(max(0, centers_px[0] - step // 2))
                for i in range(7):
                    bounds.append((centers_px[i] + centers_px[i+1]) // 2)
                bounds.append(min(W, centers_px[-1] + step // 2))
                for i in range(8):
                    x0 = int(bounds[i])
                    x1 = int(bounds[i+1])
                    cols.append((max(0, x0), min(W, max(x0 + 1, x1))))
            else:
                band_half = 30
                col_proj = [0.0] * W
                for yc in row_centers:
                    y0b = max(0, yc - band_half)
                    y1b = min(H, yc + band_half)
                    for x in range(W):
                        s = 0
                        for yy in range(y0b, y1b):
                            s += 255 - px[x, yy]
                        col_proj[x] += float(s)
                col_sm = mov_avg(col_proj, 21)
                if not col_sm:
                    return [4] * expected_count
                thr = max(col_sm) * 0.35
                xs = [i for i, v in enumerate(col_sm) if v >= thr]
                if len(xs) < 8:
                    thr = max(col_sm) * 0.20
                    xs = [i for i, v in enumerate(col_sm) if v >= thr]
                if not xs:
                    return [4] * expected_count
                x_left = int(min(xs))
                x_right = int(max(xs))
                pad = max(0, (x_right - x_left) // 40)
                x_left += pad
                x_right -= pad
                if x_right <= x_left + 8:
                    return [4] * expected_count
                sep_min_gap = max(6, (x_right - x_left) // 16)
                peaks_x = []
                for x in range(x_left + 3, x_right - 3):
                    v = col_sm[x]
                    if v > col_sm[x-1] and v >= col_sm[x+1]:
                        if not peaks_x or x - peaks_x[-1][1] >= sep_min_gap:
                            peaks_x.append((v, x))
                        elif v > peaks_x[-1][0]:
                            peaks_x[-1] = (v, x)
                peaks_x.sort(key=lambda a: a[0], reverse=True)
                selected = []
                for v, x in peaks_x:
                    if all(abs(x - sx) >= sep_min_gap for sx in [px for _, px in selected]):
                        selected.append((v, x))
                    if len(selected) == 7:
                        break
                selected = sorted([px for _, px in selected])
                if len(selected) == 7:
                    bounds = [x_left] + selected + [x_right]
                else:
                    bounds = [int(x_left + i * (x_right - x_left) / 8.0) for i in range(9)]
                for i in range(8):
                    x0 = int(bounds[i])
                    x1 = int(bounds[i+1])
                    cols.append((x0, max(x0 + 1, x1)))

            # Score each cell per row
            roi_w = max(0.3, min(0.9, self._fparam('FITREP_ROI_W', 0.6)))
            roi_v = max(0.3, min(0.9, self._fparam('FITREP_ROI_V', 0.5)))
            w_dark = self._fparam('FITREP_W_DARK', 1.0)
            w_edge = self._fparam('FITREP_W_EDGE', 0.6)
            w_diag = self._fparam('FITREP_W_DIAG', 0.8)
            w_x    = self._fparam('FITREP_W_OCRX', 50000.0)
            values = []
            box_h = 36
            row_col_scores = []
            for yc in row_centers:
                y0 = max(0, yc - box_h // 2)
                y1 = min(H, yc + box_h // 2)
                # Use tighter vertical band to avoid bleed into adjacent rows
                band_h = y1 - y0
                y0i = y0 + max(2, int(band_h * 0.25))
                y1i = y1 - max(2, int(band_h * 0.25))
                scores = []
                for (x0, x1) in cols:
                    width = x1 - x0
                    inner_w = max(6, int(width * roi_w))
                    cx = (x0 + x1) // 2
                    ix0 = max(0, cx - inner_w // 2)
                    ix1 = min(W, cx + inner_w // 2)
                    # Darkness
                    s_dark = 0.0
                    for yy in range(y0i, y1i):
                        for xx in range(ix0, ix1):
                            s_dark += 255 - px[xx, yy]
                    # Edge/line energy (captures partial strokes, any orientation)
                    edge = 0.0
                    for yy in range(max(y0i+1, 1), min(y1i-1, H-1)):
                        for xx in range(max(ix0+1, 1), min(ix1-1, W-1)):
                            gx = int(px[xx+1, yy]) - int(px[xx-1, yy])
                            gy = int(px[xx, yy+1]) - int(px[xx, yy-1])
                            edge += abs(gx) + abs(gy)
                    # X-template correlation near center with small offsets
                    w = ix1 - ix0
                    h = y1i - y0i
                    ccx = (ix0 + ix1) // 2
                    ccy = (y0i + y1i) // 2
                    span = max(3, min(w, h) // 3)
                    best_corr = 0.0
                    for off in (-4, -2, 0, 2, 4):
                        s = 0.0
                        for k in range(-span, span):
                            x = ccx + k + off
                            y = ccy + k
                            if ix0 <= x < ix1 and y0i <= y < y1i:
                                s += 255 - px[x, y]
                        for k in range(-span, span):
                            x = ccx - k + off
                            y = ccy + k
                            if ix0 <= x < ix1 and y0i <= y < y1i:
                                s += 255 - px[x, y]
                        if s > best_corr:
                            best_corr = s
                    diag = best_corr
                    # OCR hint
                    try:
                        cell_img = img_gray.crop((ix0, y0i, ix1, y1i))
                        ocr_txt = pytesseract.image_to_string(cell_img, config='--psm 10 -c tessedit_char_whitelist=Xx')
                        has_x = 'x' in (ocr_txt or '').lower()
                    except Exception:
                        has_x = False
                    score = (w_dark * s_dark) + (w_edge * edge) + (w_diag * diag) + (w_x if has_x else 0.0)
                    scores.append(score)
                row_col_scores.append(scores)

            # Normalize by per-column baseline (median across rows) to suppress persistent gridline darkness
            if row_col_scores:
                import statistics
                col_count = len(row_col_scores[0])
                baselines = []
                for ci in range(col_count):
                    col_vals = [row[ci] for row in row_col_scores]
                    try:
                        b = statistics.median(col_vals)
                    except statistics.StatisticsError:
                        b = sum(col_vals)/max(1,len(col_vals))
                    baselines.append(b)
                values = []
                beta = self._fparam('FITREP_BETA', 0.90)
                conf_m = self._fparam('FITREP_CONF', 1.15)
                for scores in row_col_scores:
                    adj = [scores[i] - beta * baselines[i] for i in range(len(scores))]
                    best_idx = max(range(len(adj)), key=lambda i: adj[i])
                    best = adj[best_idx]
                    second = max((adj[i] for i in range(len(adj)) if i != best_idx), default=-1e9)
                    if best <= 0 or (second > -1e9 and best < second * conf_m):
                        values.append(4)
                    else:
                        values.append(best_idx + 1)
            else:
                values = [4] * expected_count

            if len(values) < expected_count:
                values += [4] * (expected_count - len(values))
            return values
        except Exception:
            return [4] * expected_count

    def extract_checkbox_values_grid_markfill(self, pdf_doc, page_num, expected_count):
        """
        Mark-in-box detector on grid cells (no X required):
        - Reconstruct grid via vector lines and/or projection/headers.
        - Score each cell by normalized grayscale darkness and general edge energy
          within a central ROI, ignoring diagonal-X correlation and OCR.
        - Choose the highest-confidence column per row with a margin over second-best.
        Returns [4]*expected_count if grid cannot be reliably reconstructed.
        """
        try:
            if page_num >= len(pdf_doc):
                return [4] * expected_count
            page = pdf_doc[page_num]

            # Render grayscale at scale 3 for robust pixel scoring
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)
            img_gray = Image.open(io.BytesIO(pix.tobytes("png"))).convert('L')
            W, H = img_gray.size
            px = img_gray.load()

            # Row centers: reuse row detection from grid_hybrid via vector lines, falling back to image bands
            h_lines = []
            v_lines = []
            for d in page.get_drawings():
                for it in d.get('items', []):
                    if it[0] != 'l':
                        continue
                    (x0, y0), (x1, y1) = it[1], it[2]
                    dx, dy = float(x1 - x0), float(y1 - y0)
                    length = (dx*dx + dy*dy) ** 0.5
                    if length < 20:
                        continue
                    ax, ay = abs(dx), abs(dy)
                    if ay < 0.8 and ax > 60:
                        h_lines.append((y0 + y1) / 2.0)
                    elif ax < 0.8 and ay > 60:
                        v_lines.append((x0 + x1) / 2.0)

            def cluster_positions(vals, tol):
                vals = sorted(vals)
                clusters = []
                for v in vals:
                    if not clusters or abs(v - clusters[-1][-1]) > tol:
                        clusters.append([v])
                    else:
                        clusters[-1].append(v)
                return [sum(c)/len(c) for c in clusters]

            ys_bounds = cluster_positions([y for y in h_lines], tol=4.0) if h_lines else []
            xs_centers = cluster_positions([x for x in v_lines], tol=4.0) if v_lines else []

            # If vector lines insufficient, attempt OCR header centers or projection similar to hybrid
            from PIL import Image as _Image
            _ = _Image  # avoid unused import if header OCR not needed

            centers_px = None
            def detect_header_centers_ocr(img):
                try:
                    ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config='--psm 6')
                    n = len(ocr.get('text', []))
                    spans = []
                    for i in range(n):
                        t = (ocr['text'][i] or '').strip()
                        if len(t) == 1 and t.upper() in list('ABCDEFGH'):
                            spans.append({
                                'ch': t.upper(),
                                'x': ocr['left'][i] + ocr['width'][i] / 2.0,
                                'y': ocr['top'][i] + ocr['height'][i] / 2.0,
                            })
                    if not spans:
                        return None, None
                    spans.sort(key=lambda a: a['y'])
                    rows = []
                    cur = [spans[0]]
                    for s in spans[1:]:
                        if abs(s['y'] - cur[0]['y']) < 12:
                            cur.append(s)
                        else:
                            rows.append(cur)
                            cur = [s]
                    if cur:
                        rows.append(cur)
                    best = None
                    bestu = -1
                    for r in rows:
                        u = {a['ch'] for a in r}
                        xs_ = [a['x'] for a in r]
                        if len(u) >= 6 and xs_ and (max(xs_) - min(xs_)) > 150 and len(u) > bestu:
                            best = r
                            bestu = len(u)
                    if not best:
                        return None, None
                    best.sort(key=lambda a: a['x'])
                    cmap = {a['ch']: a['x'] for a in best}
                    letters = list('ABCDEFGH')
                    known = [(ch, cmap[ch]) for ch in letters if ch in cmap]
                    known.sort(key=lambda a: a[1])
                    min_ch, min_x = known[0]
                    max_ch, max_x = known[-1]
                    idx_min = letters.index(min_ch)
                    idx_max = letters.index(max_ch)
                    span = max(idx_max - idx_min, 1)
                    step = (max_x - min_x) / span
                    centers = []
                    for i, ch in enumerate(letters):
                        if ch in cmap:
                            centers.append(cmap[ch])
                        else:
                            centers.append(min_x + (i - idx_min) * step)
                    header_y = sum(a['y'] for a in best) / len(best)
                    return [int(c) for c in centers], int(header_y)
                except Exception:
                    return None, None

            if not xs_centers:
                centers_px, _ = detect_header_centers_ocr(img_gray)
                if centers_px and len(centers_px) == 8:
                    xs_centers = centers_px

            # If still no xs/ys, fall back to returning defaults
            if not xs_centers:
                return [4] * expected_count
            # Build row centers: from vector bounds if available, else via projection peaks
            row_centers = []
            if ys_bounds and len(ys_bounds) >= expected_count+1:
                ys_sorted = sorted(ys_bounds)
                # Select the widest group of expected_count+1 lines
                if len(ys_sorted) > expected_count+1:
                    best = ys_sorted[:expected_count+1]
                    best_span = best[-1] - best[0]
                    for i in range(0, len(ys_sorted) - (expected_count+1) + 1):
                        group = ys_sorted[i:i+expected_count+1]
                        span = group[-1] - group[0]
                        if span > best_span:
                            best_span = span
                            best = group
                    ys_sorted = best
                if len(ys_sorted) == expected_count+1:
                    row_centers = [int((ys_sorted[i] + ys_sorted[i+1]) / 2.0) for i in range(expected_count)]
            if not row_centers:
                # Approximate row centers via horizontal projection
                import numpy as np
                arr = np.array(img_gray, dtype=np.uint8)
                proj = (255 - arr).sum(axis=1)
                k = np.ones(15, dtype=np.float32)
                sm = np.convolve(proj, k, mode='same')
                peaks = []
                last = -9999
                min_sep = max(30, H // (expected_count + 1))
                for y in range(1, len(sm)-1):
                    if sm[y] > sm[y-1] and sm[y] >= sm[y+1]:
                        if y - last >= min_sep:
                            peaks.append((float(sm[y]), int(y)))
                            last = y
                peaks.sort(key=lambda a: a[0], reverse=True)
                centers = sorted([p[1] for p in peaks[:expected_count]])
                if len(centers) != expected_count:
                    return [4] * expected_count
                row_centers = centers

            # Build column bounds from xs centers
            xs_sorted = sorted(xs_centers)
            if len(xs_sorted) == 8:
                bounds = []
                step = max(1, (xs_sorted[-1] - xs_sorted[0]) // 7)
                bounds.append(max(0, xs_sorted[0] - step // 2))
                for i in range(7):
                    bounds.append((xs_sorted[i] + xs_sorted[i+1]) // 2)
                bounds.append(min(W, xs_sorted[-1] + step // 2))
            else:
                # If xs already seem to be bounds-like
                if len(xs_sorted) >= 9:
                    bounds = [int(x) for x in xs_sorted[:9]]
                else:
                    # Projection-based columns using row centers
                    col_proj = [0.0] * W
                    band_half = 30
                    for yc in row_centers:
                        y0 = max(0, yc - band_half)
                        y1 = min(H, yc + band_half)
                        for x in range(W):
                            s = 0.0
                            for yy in range(y0, y1):
                                s += 255 - px[x, yy]
                            col_proj[x] += float(s)
                    # Smooth and threshold to find active region
                    def mov_avg(v, w):
                        acc = [0.0]*len(v)
                        s = 0.0
                        half = w//2
                        for i in range(len(v)):
                            s += v[i]
                            if i >= w:
                                s -= v[i-w]
                            if i >= w-1:
                                acc[i-half] = s / w
                        return acc
                    col_sm = mov_avg(col_proj, 21)
                    if not col_sm:
                        return [4] * expected_count
                    thr = max(col_sm) * 0.35
                    xs = [i for i, v in enumerate(col_sm) if v >= thr]
                    if len(xs) < 8:
                        thr = max(col_sm) * 0.20
                        xs = [i for i, v in enumerate(col_sm) if v >= thr]
                    if not xs:
                        return [4] * expected_count
                    x_left = int(min(xs))
                    x_right = int(max(xs))
                    pad = max(0, (x_right - x_left) // 40)
                    x_left += pad
                    x_right -= pad
                    if x_right <= x_left + 8:
                        return [4] * expected_count
                    # Find 7 internal separators via local maxima spacing
                    sep_min_gap = max(6, (x_right - x_left) // 16)
                    peaks_x = []
                    for x in range(x_left + 3, x_right - 3):
                        v = col_sm[x]
                        if v > col_sm[x-1] and v >= col_sm[x+1]:
                            if not peaks_x or x - peaks_x[-1][1] >= sep_min_gap:
                                peaks_x.append((v, x))
                            elif v > peaks_x[-1][0]:
                                peaks_x[-1] = (v, x)
                    peaks_x.sort(key=lambda a: a[0], reverse=True)
                    selected = []
                    for v, x in peaks_x:
                        if all(abs(x - sx) >= sep_min_gap for sx in [px for _, px in selected]):
                            selected.append((v, x))
                        if len(selected) == 7:
                            break
                    selected = sorted([px for _, px in selected])
                    if len(selected) == 7:
                        bounds = [x_left] + selected + [x_right]
                    else:
                        bounds = [int(x_left + i * (x_right - x_left) / 8.0) for i in range(9)]

            cols = []
            for i in range(8):
                x0 = int(bounds[i])
                x1 = int(bounds[i+1])
                cols.append((max(0, x0), min(W, max(x0 + 1, x1))))

            # Score: darkness + edge only, within central ROI
            roi_w = max(0.3, min(0.9, self._fparam('FITREP_ROI_W', 0.6)))
            roi_v = max(0.3, min(0.9, self._fparam('FITREP_ROI_V', 0.5)))
            w_dark = self._fparam('FITREP_W_DARK', 1.0)
            w_edge = self._fparam('FITREP_W_EDGE', 0.8)
            beta = self._fparam('FITREP_BETA', 0.90)
            conf_m = self._fparam('FITREP_CONF', 1.15)

            values = []
            for yc in row_centers:
                band_h = max(10, int(H * 0.02))
                y0i = max(1, yc - int(band_h * roi_v))
                y1i = min(H-1, yc + int(band_h * roi_v))
                y0i = max(1, min(H-2, y0i)); y1i = max(y0i+1, min(H-1, y1i))
                scores = []
                for (x0, x1) in cols:
                    width = max(2, x1 - x0)
                    inner_w = max(6, int(width * roi_w))
                    cx = (x0 + x1) // 2
                    ix0 = max(1, min(W-2, cx - inner_w // 2))
                    ix1 = max(ix0+1, min(W-1, cx + inner_w // 2))
                    # Darkness
                    s_dark = 0.0
                    for yy in range(y0i, y1i):
                        for xx in range(ix0, ix1):
                            s_dark += 255 - px[xx, yy]
                    # Edge
                    edge = 0.0
                    for yy in range(y0i+1, y1i-1):
                        for xx in range(ix0+1, ix1-1):
                            gx = int(px[xx+1, yy]) - int(px[xx-1, yy])
                            gy = int(px[xx, yy+1]) - int(px[xx, yy-1])
                            edge += abs(gx) + abs(gy)
                    scores.append((w_dark * s_dark) + (w_edge * edge))
                # Normalize by median to remove row bias
                if scores:
                    import statistics
                    med = statistics.median(scores)
                    adj = [s - beta * med for s in scores]
                    best_idx = max(range(len(adj)), key=lambda i: adj[i])
                    best = adj[best_idx]
                    second = max((adj[i] for i in range(len(adj)) if i != best_idx), default=-1e9)
                    values.append(best_idx + 1 if best > max(1.0, second * conf_m) else 4)
                else:
                    values.append(4)
            return values
        except Exception:
            return [4] * expected_count

    def extract_checkbox_values_grid_vector(self, pdf_doc, page_num, expected_count):
        """
        Vector-based detector:
        - Detect long horizontal and vertical grid lines via page.get_drawings().
        - Cluster to get (expected_count+1) horizontal lines and 9 vertical lines.
        - Define exact cell rectangles; within each cell, accumulate lengths of non-grid
          diagonal-ish strokes whose midpoints lie in a centered ROI.
        Returns [4]*expected_count if grid cannot be reliably reconstructed.
        """
        try:
            if page_num >= len(pdf_doc):
                return [4] * expected_count
            page = pdf_doc[page_num]

            h_lines = []
            v_lines = []
            diag_lines = []
            for d in page.get_drawings():
                for it in d.get('items', []):
                    if it[0] != 'l':
                        continue
                    (x0, y0), (x1, y1) = it[1], it[2]
                    dx, dy = float(x1 - x0), float(y1 - y0)
                    length = (dx*dx + dy*dy) ** 0.5
                    if length < 20:
                        continue
                    ax, ay = abs(dx), abs(dy)
                    if ay < 0.8 and ax > 60:
                        ymid = (y0 + y1) / 2.0
                        h_lines.append(ymid)
                    elif ax < 0.8 and ay > 60:
                        xmid = (x0 + x1) / 2.0
                        v_lines.append(xmid)
                    else:
                        slope = (ay / ax) if ax != 0 else 999.0
                        if 0.3 <= slope <= 3.0 and 12 <= length <= 120:
                            diag_lines.append((x0, y0, x1, y1, length))

            if len(h_lines) < expected_count+1 or len(v_lines) < 9:
                return [4] * expected_count

            def cluster_positions(vals, tol):
                vals = sorted(vals)
                clusters = []
                for v in vals:
                    if not clusters or abs(v - clusters[-1][-1]) > tol:
                        clusters.append([v])
                    else:
                        clusters[-1].append(v)
                return [sum(c)/len(c) for c in clusters]

            ys = cluster_positions(h_lines, tol=4.0)
            xs = cluster_positions(v_lines, tol=4.0)
            if len(ys) < expected_count+1 or len(xs) < 9:
                return [4] * expected_count
            ys = sorted(ys)
            xs = sorted(xs)
            if len(ys) > expected_count+1:
                best = ys[:expected_count+1]
                best_span = best[-1] - best[0]
                for i in range(0, len(ys) - (expected_count+1) + 1):
                    group = ys[i:i+expected_count+1]
                    span = group[-1] - group[0]
                    if span > best_span:
                        best_span = span
                        best = group
                ys = best
            if len(xs) > 9:
                best = xs[:9]
                best_span = best[-1] - best[0]
                for i in range(0, len(xs) - 9 + 1):
                    group = xs[i:i+9]
                    span = group[-1] - group[0]
                    if span > best_span:
                        best_span = span
                        best = group
                xs = best
            if len(ys) != expected_count+1 or len(xs) != 9:
                return [4] * expected_count

            def in_center_roi(x0, y0, x1, y1, px, py):
                w = x1 - x0
                h = y1 - y0
                cx0 = x0 + 0.2 * w
                cx1 = x1 - 0.2 * w
                cy0 = y0 + 0.2 * h
                cy1 = y1 - 0.2 * h
                return (cx0 <= px <= cx1) and (cy0 <= py <= cy1)

            values = []
            for r in range(expected_count):
                y0, y1 = ys[r], ys[r+1]
                row_scores = []
                for c in range(8):
                    x0, x1 = xs[c], xs[c+1]
                    score = 0.0
                    for (sx0, sy0, sx1, sy1, L) in diag_lines:
                        mx, my = (sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0
                        if in_center_roi(x0, y0, x1, y1, mx, my):
                            score += L
                    row_scores.append(score)
                if any(s > 0 for s in row_scores):
                    best_idx = max(range(8), key=lambda i: row_scores[i])
                    best = row_scores[best_idx]
                    second = max((row_scores[i] for i in range(8) if i != best_idx), default=0.0)
                    values.append(best_idx + 1 if best > max(1.0, second * 1.15) else 4)
                else:
                    values.append(4)
            return values
        except Exception:
            return [4] * expected_count

    def extract_checkbox_values_grid_hybrid(self, pdf_doc, page_num, expected_count):
        """
        Hybrid detector on grid cells:
        - Reconstruct grid via vector lines (same as grid_vector).
        - Score each cell by grayscale darkness, general edge energy, diagonal energy,
          and OCR 'X' hint; all measured in a central ROI.
        Returns [4]*expected_count if grid cannot be reliably reconstructed.
        """
        try:
            if page_num >= len(pdf_doc):
                return [4] * expected_count
            page = pdf_doc[page_num]

            h_lines = []
            v_lines = []
            for d in page.get_drawings():
                for it in d.get('items', []):
                    if it[0] != 'l':
                        continue
                    (x0, y0), (x1, y1) = it[1], it[2]
                    dx, dy = float(x1 - x0), float(y1 - y0)
                    length = (dx*dx + dy*dy) ** 0.5
                    if length < 20:
                        continue
                    ax, ay = abs(dx), abs(dy)
                    if ay < 0.8 and ax > 60:
                        h_lines.append((y0 + y1) / 2.0)
                    elif ax < 0.8 and ay > 60:
                        v_lines.append((x0 + x1) / 2.0)

            if len(h_lines) < expected_count+1 or len(v_lines) < 9:
                return [4] * expected_count

            def cluster_positions(vals, tol):
                vals = sorted(vals)
                clusters = []
                for v in vals:
                    if not clusters or abs(v - clusters[-1][-1]) > tol:
                        clusters.append([v])
                    else:
                        clusters[-1].append(v)
                return [sum(c)/len(c) for c in clusters]

            ys = cluster_positions([y for y in h_lines], tol=4.0)
            xs = cluster_positions([x for x in v_lines], tol=4.0)
            if len(ys) < expected_count+1 or len(xs) < 9:
                return [4] * expected_count
            ys = sorted(ys)
            xs = sorted(xs)
            if len(ys) > expected_count+1:
                best = ys[:expected_count+1]
                best_span = best[-1] - best[0]
                for i in range(0, len(ys) - (expected_count+1) + 1):
                    group = ys[i:i+expected_count+1]
                    span = group[-1] - group[0]
                    if span > best_span:
                        best_span = span
                        best = group
                ys = best
            if len(xs) > 9:
                best = xs[:9]
                best_span = best[-1] - best[0]
                for i in range(0, len(xs) - 9 + 1):
                    group = xs[i:i+9]
                    span = group[-1] - group[0]
                    if span > best_span:
                        best_span = span
                        best = group
                xs = best
            if len(ys) != expected_count+1 or len(xs) != 9:
                return [4] * expected_count

            # Render grayscale at scale 3
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)
            img_gray = Image.open(io.BytesIO(pix.tobytes("png"))).convert('L')
            W, H = img_gray.size
            px = img_gray.load()

            def score_cell(x0, y0, x1, y1):
                # Move to pixels (already in pts)
                cx0 = int(x0 * 3)
                cx1 = int(x1 * 3)
                cy0 = int(y0 * 3)
                cy1 = int(y1 * 3)
                w = cx1 - cx0
                h = cy1 - cy0
                rx0 = cx0 + max(2, int(w * 0.2))
                rx1 = cx1 - max(2, int(w * 0.2))
                ry0 = cy0 + max(2, int(h * 0.2))
                ry1 = cy1 - max(2, int(h * 0.2))
                rx0 = max(1, min(W-2, rx0)); rx1 = max(rx0+1, min(W-1, rx1))
                ry0 = max(1, min(H-2, ry0)); ry1 = max(ry0+1, min(H-1, ry1))

                dark = 0.0
                for yy in range(ry0, ry1):
                    for xx in range(rx0, rx1):
                        dark += 255 - px[xx, yy]
                edge = 0.0
                for yy in range(ry0+1, ry1-1):
                    for xx in range(rx0+1, rx1-1):
                        gx = int(px[xx+1, yy]) - int(px[xx-1, yy])
                        gy = int(px[xx, yy+1]) - int(px[xx, yy-1])
                        edge += abs(gx) + abs(gy)
                # Centered X correlation with small shifts to capture partial Xs
                w2 = rx1 - rx0
                h2 = ry1 - ry0
                cx = (rx0 + rx1) // 2
                cy = (ry0 + ry1) // 2
                best_xcorr = 0.0
                span = max(4, min(w2, h2) // 2)
                for off in (-4, -2, 0, 2, 4):
                    s = 0.0
                    # forward diag (\)
                    for k in range(-span, span):
                        x = cx + k + off
                        y = cy + k
                        if rx0 <= x < rx1 and ry0 <= y < ry1:
                            s += 255 - px[x, y]
                    # back diag (/)
                    for k in range(-span, span):
                        x = cx - k + off
                        y = cy + k
                        if rx0 <= x < rx1 and ry0 <= y < ry1:
                            s += 255 - px[x, y]
                    if s > best_xcorr:
                        best_xcorr = s
                diag = best_xcorr
                try:
                    cell_img = img_gray.crop((rx0, ry0, rx1, ry1))
                    otext = pytesseract.image_to_string(cell_img, config='--psm 10 -c tessedit_char_whitelist=Xx')
                    has_x = 'x' in (otext or '').lower()
                except Exception:
                    has_x = False
                return (1.0 * dark) + (0.6 * edge) + (0.8 * diag) + (50000.0 if has_x else 0.0)

            values = []
            for r in range(expected_count):
                y0, y1 = ys[r], ys[r+1]
                scores = []
                for c in range(8):
                    x0, x1 = xs[c], xs[c+1]
                    scores.append(score_cell(x0, y0, x1, y1))
                import statistics
                base = statistics.median(scores) if scores else 0.0
                adj = [s - 0.90 * base for s in scores]
                best_idx = max(range(8), key=lambda i: adj[i]) if adj else 3
                best = adj[best_idx]
                second = max((adj[i] for i in range(8) if i != best_idx), default=-1e9)
                values.append(best_idx + 1 if best > max(1.0, second * 1.20) else 4)
            return values
        except Exception:
            return [4] * expected_count

    def extract_marine_last_name_by_edipi(self, pdf_doc, marine_edipi):
        """Extract Marine's last name by locating the Marine EDIPI on page 1 and walking upwards.

        Heuristic observed pattern:
        - Top of page lists LASTNAME on its own line, then FIRSTNAME on next line,
          followed by a line containing EDIPI and GRADE (e.g., "1234567890 CAPT").
        - We find the line containing the Marine EDIPI and then look 1–5 lines above
          for uppercase alpha-only tokens; the upper of the nearest two tokens is
          the last name. If only one candidate exists, use it.
        """
        if len(pdf_doc) == 0:
            return None

        page = pdf_doc[0]
        text = page.get_text() or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # Locate the line index that contains the given EDIPI
        idx = None
        for i, ln in enumerate(lines):
            if marine_edipi in ln:
                idx = i
                break
        if idx is None:
            return None

        # Collect up to a few candidate name tokens above the EDIPI line
        field_labels = {
            'FITREP', 'ID', 'FITNESS', 'REPORT', 'REPORTING', 'SENIOR', 'REVIEWING',
            'SERVICE', 'USMC', 'ANG', 'USA', 'AFNG', 'USAF', 'USN', 'FMS', 'USCG', 'USSF',
            'GRADE', 'RANK', 'DUTY', 'ASSIGNMENT', 'INITIALS'
        }
        valid_grades = set(self.valid_grades)
        valid_occ = set(self.valid_occ_codes)

        candidates = []
        for j in range(max(0, idx - 6), idx):
            token = lines[j].strip()
            token_u = token.upper()
            # Accept single-word, alpha-only, uppercase strings of length >=2
            if token_u.isalpha() and token_u == token and len(token_u) >= 2:
                if token_u not in field_labels and token_u not in valid_grades and token_u not in valid_occ:
                    candidates.append((j, token_u))

        if not candidates:
            return None

        # Sort by proximity (closest first going upwards)
        candidates.sort(key=lambda t: abs(idx - t[0]))

        # If we have two consecutive name lines above the EDIPI, pick the upper one as last name
        # Example order walking upwards: [FIRSTNAME, LASTNAME]
        # candidates currently sorted by proximity (nearest first); reconstruct in page order
        # for the nearest two above.
        nearest = sorted([c for c in candidates if c[0] < idx], key=lambda t: t[0], reverse=True)
        if len(nearest) >= 2:
            # nearest[0] is first name line, nearest[1] is last name line
            return nearest[1][1]
        # Otherwise fallback to the only candidate
        return nearest[0][1]

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
            p2 = data.get('page2_values') or []
            row.extend((p2 + [''] * 5)[:5])
            # Add page 3 values (5 values)
            p3 = data.get('page3_values') or []
            row.extend((p3 + [''] * 5)[:5])
            # Add page 4 values (4 values)
            p4 = data.get('page4_values') or []
            row.extend((p4 + [''] * 4)[:4])
            
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
        
        # Sort results by Grade (military rank), then by Last Name, then FITREP ID
        # row layout: [fitrep_id, last_name, grade, ...]
        self.results.sort(key=lambda x: (self.rank_sort_key(x[2]), x[1], x[0]))
        
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
