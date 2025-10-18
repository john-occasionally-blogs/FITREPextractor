#!/usr/bin/env python3
"""
Marine FITREP PDF to CSV Extractor
Extracts specific values from Marine Fitness Report PDFs and outputs to CSV
"""

import os
import sys
import re
import csv
from pathlib import Path
import PyPDF2
import pdfplumber
from datetime import datetime


class FITREPExtractor:
    def __init__(self):
        self.results = []
        
    def extract_from_pdf(self, pdf_path):
        """Extract required data from a single PDF file"""
        try:
            data = {}
            
            # Try with pdfplumber first (better for form data)
            with pdfplumber.open(pdf_path) as pdf:
                # Page 1 extraction
                page1 = pdf.pages[0]
                text1 = page1.extract_text()
                
                # Extract Last Name (Line after "1. Marine Reported On:")
                last_name_match = re.search(r'Last Name.*?\n\s*([A-Z]+)', text1)
                if last_name_match:
                    data['last_name'] = last_name_match.group(1)
                
                # Extract Grade (LTCOL, COL, etc.)
                grade_match = re.search(r'Grade.*?\n.*?([A-Z]+(?:COL|GEN|MAJ|CPT|LT))', text1)
                if grade_match:
                    data['grade'] = grade_match.group(1)
                
                # Extract OCC (2-character code from "a. OCC")
                occ_match = re.search(r'OCC.*?\n.*?([A-Z0-9]{2})\s', text1)
                if occ_match:
                    data['occ'] = occ_match.group(1)
                
                # Extract To date (YYYYMMDD format)
                to_match = re.search(r'To\s*\n.*?(\d{8})', text1)
                if to_match:
                    data['to_date'] = to_match.group(1)
                
                # Check for Not Observed (look for X in that section)
                not_observed = False
                not_obs_section = re.search(r'Not Observed.*?\n.*?([X\s]{1,3})', text1, re.DOTALL)
                if not_obs_section and 'X' in not_obs_section.group(1):
                    not_observed = True
                    
                # If Not Observed is checked, skip this report
                if not_observed:
                    print(f"  Skipping {pdf_path.name} - Not Observed is checked")
                    return None
                
                # Page 2 - Extract 5 checkbox values
                page2_values = []
                if len(pdf.pages) > 1:
                    page2 = pdf.pages[1]
                    text2 = page2.extract_text()
                    page2_values = self.extract_checkbox_values(text2, 5)
                data['page2_values'] = page2_values
                
                # Page 3 - Extract 5 checkbox values
                page3_values = []
                if len(pdf.pages) > 2:
                    page3 = pdf.pages[2]
                    text3 = page3.extract_text()
                    page3_values = self.extract_checkbox_values(text3, 5)
                data['page3_values'] = page3_values
                
                # Page 4 - Extract 4 checkbox values
                page4_values = []
                if len(pdf.pages) > 3:
                    page4 = pdf.pages[3]
                    text4 = page4.extract_text()
                    page4_values = self.extract_checkbox_values(text4, 4)
                data['page4_values'] = page4_values
                
            return data
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None
    
    def extract_checkbox_values(self, text, expected_count):
        """Extract checkbox values from a page (1-8 based on X position)"""
        values = []
        
        # Look for patterns of X marks in evaluation sections
        # Pattern looks for lines with checkbox indicators (A-H columns with X marks)
        
        # Split text into lines and look for evaluation rows
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Look for lines that appear to be evaluation rows
            # They typically have letter grades A through H with X marks
            if re.search(r'[A-H]\s+[A-H]', line) or 'X' in line:
                # Count X positions in the line
                x_positions = [m.start() for m in re.finditer(r'X', line)]
                if x_positions:
                    # Try to determine which column (1-8) the X is in
                    # This is approximate based on position in line
                    for pos in x_positions:
                        # Rough estimation: divide line into 8 sections
                        if len(line) > 0:
                            column = min(8, max(1, int((pos / len(line)) * 8) + 1))
                            values.append(column)
                            if len(values) >= expected_count:
                                break
            if len(values) >= expected_count:
                break
        
        # If we didn't find enough values, pad with defaults
        while len(values) < expected_count:
            values.append(4)  # Default to middle value if not found
            
        return values[:expected_count]
    
    def rank_sort_key(self, grade):
        """Return sort key for military ranks"""
        rank_order = {
            'GEN': 1, 'LTGEN': 2, 'MAJGEN': 3, 'BGEN': 4,
            'COL': 5, 'LTCOL': 6, 'MAJ': 7, 'CPT': 8,
            'CAPT': 8, '1STLT': 9, '2NDLT': 10, 'LT': 11
        }
        # Extract the rank portion from the grade
        for rank in rank_order:
            if rank in grade.upper():
                return rank_order[rank]
        return 99  # Unknown ranks go to the end
    
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
            print(f"  Extracted: {row[:4]}...")  # Show first 4 fields for confirmation
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
        
        # Sort results by Grade (military rank)
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
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    
    print("Marine FITREP PDF to CSV Extractor")
    print("=" * 40)
    
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
        sys.exit(1)
