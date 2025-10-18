#!/usr/bin/env python3
"""
PDF OCR Form Data Extractor for Mac
Extracts text from static/scanned PDF forms using OCR
"""

import sys
import os
import re
from pathlib import Path
import json

# Import required libraries with error handling
try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("OCR libraries not installed. Install with:")
    print("pip3 install pytesseract pillow")
    print("\nAlso install tesseract OCR engine:")
    print("brew install tesseract  # On macOS with Homebrew")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
except ImportError:
    print("pdf2image not installed. Install with:")
    print("pip3 install pdf2image")
    print("\nAlso install poppler:")
    print("brew install poppler  # On macOS with Homebrew")
    sys.exit(1)

def check_tesseract():
    """Check if tesseract is installed and accessible"""
    try:
        # Try to get tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        return True
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract OCR engine not found!")
        print("\nInstall tesseract:")
        print("brew install tesseract  # On macOS with Homebrew")
        print("\nOr download from: https://github.com/tesseract-ocr/tesseract")
        return False
    except Exception as e:
        print(f"Error checking tesseract: {e}")
        return False

def pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF pages to images for OCR processing
    
    Args:
        pdf_path (str): Path to PDF file
        dpi (int): Resolution for image conversion
        
    Returns:
        list: List of PIL Image objects
    """
    try:
        print(f"Converting PDF to images (DPI: {dpi})...")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Converted {len(images)} pages to images")
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def extract_text_from_image(image, page_num):
    """
    Extract text from a single image using OCR
    
    Args:
        image (PIL.Image): Image to process
        page_num (int): Page number for reference
        
    Returns:
        dict: Extracted text data
    """
    try:
        print(f"Processing page {page_num} with OCR...")
        
        # Configure tesseract for better accuracy
        custom_config = r'--oem 3 --psm 6'
        
        # Extract text with bounding boxes
        ocr_data = pytesseract.image_to_data(
            image, 
            config=custom_config, 
            output_type=pytesseract.Output.DICT
        )
        
        # Extract plain text
        text = pytesseract.image_to_string(image, config=custom_config)
        
        return {
            'page': page_num,
            'text': text.strip(),
            'data': ocr_data,
            'lines': text.strip().split('\n') if text.strip() else []
        }
        
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        return {
            'page': page_num,
            'text': '',
            'data': None,
            'lines': []
        }

def find_form_patterns(text_data):
    """
    Find common form field patterns in extracted text
    
    Args:
        text_data (list): List of text data from all pages
        
    Returns:
        dict: Structured form data
    """
    form_fields = {}
    
    # Common form field patterns
    patterns = {
        'name': [
            r'name[:\s]*([^\n\r]+)',
            r'first\s*name[:\s]*([^\n\r]+)',
            r'last\s*name[:\s]*([^\n\r]+)',
            r'full\s*name[:\s]*([^\n\r]+)'
        ],
        'email': [
            r'email[:\s]*([^\s]+@[^\s]+\.[^\s]+)',
            r'e-mail[:\s]*([^\s]+@[^\s]+\.[^\s]+)'
        ],
        'phone': [
            r'phone[:\s]*([0-9\-\(\)\s\+]+)',
            r'tel[:\s]*([0-9\-\(\)\s\+]+)',
            r'mobile[:\s]*([0-9\-\(\)\s\+]+)'
        ],
        'date': [
            r'date[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'
        ],
        'address': [
            r'address[:\s]*([^\n\r]+)',
            r'street[:\s]*([^\n\r]+)',
            r'city[:\s]*([^\n\r]+)',
            r'state[:\s]*([^\n\r]+)',
            r'zip[:\s]*([0-9\-\s]+)'
        ],
        'checkbox_yes': [
            r'☑.*?yes',
            r'✓.*?yes',
            r'\[x\].*?yes'
        ],
        'checkbox_no': [
            r'☑.*?no',
            r'✓.*?no',
            r'\[x\].*?no'
        ]
    }
    
    # Process each page's text
    for page_data in text_data:
        page_num = page_data['page']
        text = page_data['text'].lower()
        lines = page_data['lines']
        
        print(f"Analyzing page {page_num} for form patterns...")
        
        # Search for patterns
        for field_type, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    field_name = f"{field_type}_page_{page_num}"
                    form_fields[field_name] = {
                        'value': matches[0].strip() if isinstance(matches[0], str) else str(matches[0]),
                        'type': field_type,
                        'page': page_num,
                        'pattern': pattern
                    }
        
        # Extract lines that look like filled fields (contain colons or underscores)
        for i, line in enumerate(lines):
            line = line.strip()
            if line and (':' in line or '_' in line):
                # Split on colon or look for filled underscores
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2 and parts[1].strip():
                        field_name = f"field_{parts[0].strip().replace(' ', '_').lower()}_page_{page_num}"
                        form_fields[field_name] = {
                            'value': parts[1].strip(),
                            'type': 'extracted_field',
                            'page': page_num,
                            'pattern': 'colon_separator'
                        }
    
    return form_fields

def extract_form_data_ocr(pdf_path, dpi=300):
    """
    Main function to extract form data using OCR
    
    Args:
        pdf_path (str): Path to PDF file
        dpi (int): Image resolution for OCR
        
    Returns:
        dict: Extracted form data
    """
    # Check if tesseract is available
    if not check_tesseract():
        return {}
    
    # Convert PDF to images
    images = pdf_to_images(pdf_path, dpi)
    if not images:
        return {}
    
    # Extract text from each page
    text_data = []
    for i, image in enumerate(images, 1):
        page_text = extract_text_from_image(image, i)
        text_data.append(page_text)
    
    # Find form patterns
    form_data = find_form_patterns(text_data)
    
    return {
        'form_fields': form_data,
        'raw_text': text_data
    }

def print_ocr_results(extraction_results, show_raw=False):
    """
    Print OCR extraction results
    
    Args:
        extraction_results (dict): Results from OCR extraction
        show_raw (bool): Whether to show raw text
    """
    form_data = extraction_results.get('form_fields', {})
    raw_text = extraction_results.get('raw_text', [])
    
    if not form_data:
        print("No form field patterns detected")
        if show_raw and raw_text:
            print("\n" + "="*60)
            print("RAW EXTRACTED TEXT")
            print("="*60)
            for page_data in raw_text:
                print(f"\n--- Page {page_data['page']} ---")
                print(page_data['text'])
        return
    
    print("\n" + "="*60)
    print("OCR FORM DATA EXTRACTION RESULTS")
    print("="*60)
    
    for field_name, field_info in form_data.items():
        print(f"\nField: {field_name}")
        print(f"  Type: {field_info['type']}")
        print(f"  Value: {field_info['value']}")
        print(f"  Page: {field_info['page']}")
        print(f"  Pattern: {field_info['pattern']}")
    
    print(f"\n" + "-"*60)
    print(f"Summary: {len(form_data)} fields detected")
    print("-"*60)
    
    if show_raw and raw_text:
        print("\n" + "="*60)
        print("RAW EXTRACTED TEXT")
        print("="*60)
        for page_data in raw_text:
            print(f"\n--- Page {page_data['page']} ---")
            print(page_data['text'])

def save_ocr_results(extraction_results, output_path):
    """
    Save OCR results to JSON file
    
    Args:
        extraction_results (dict): Extraction results
        output_path (str): Output file path
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

def save_to_csv_ocr(form_data, output_path):
    """
    Save form data to CSV file
    
    Args:
        form_data (dict): Form data dictionary
        output_path (str): Path for output CSV file
    """
    try:
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Field Name', 'Type', 'Value', 'Page', 'Pattern'])
            
            for field_name, field_info in form_data.items():
                writer.writerow([
                    field_name,
                    field_info['type'],
                    field_info['value'],
                    field_info['page'],
                    field_info['pattern']
                ])
        
        print(f"CSV data saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python3 pdf_ocr_extractor.py <pdf_file> [options]")
        print("\nOptions:")
        print("  --dpi <number>     Set image resolution (default: 300)")
        print("  --show-raw        Show raw extracted text")
        print("  --csv <file>      Save results to CSV file")
        print("  --json <file>     Save results to JSON file")
        print("\nExample:")
        print("  python3 pdf_ocr_extractor.py form.pdf --dpi 400 --csv output.csv --show-raw")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Parse options
    dpi = 300
    show_raw = '--show-raw' in sys.argv
    
    if '--dpi' in sys.argv:
        try:
            dpi_index = sys.argv.index('--dpi') + 1
            if dpi_index < len(sys.argv):
                dpi = int(sys.argv[dpi_index])
        except (ValueError, IndexError):
            print("Warning: Invalid DPI value, using default 300")
    
    # Check if PDF exists
    if not Path(pdf_path).exists():
        print(f"Error: File '{pdf_path}' not found")
        sys.exit(1)
    
    print(f"Starting OCR extraction from: {pdf_path}")
    print("This may take a while depending on PDF size and complexity...")
    
    # Extract data
    results = extract_form_data_ocr(pdf_path, dpi)
    
    # Print results
    print_ocr_results(results, show_raw)
    
    # Save to CSV if requested
    if '--csv' in sys.argv:
        try:
            csv_index = sys.argv.index('--csv') + 1
            if csv_index < len(sys.argv):
                csv_path = sys.argv[csv_index]
                save_to_csv_ocr(results.get('form_fields', {}), csv_path)
        except (ValueError, IndexError):
            print("Error: --csv option requires a filename")
    
    # Save to JSON if requested
    if '--json' in sys.argv:
        try:
            json_index = sys.argv.index('--json') + 1
            if json_index < len(sys.argv):
                json_path = sys.argv[json_index]
                save_ocr_results(results, json_path)
        except (ValueError, IndexError):
            print("Error: --json option requires a filename")

if __name__ == "__main__":
    main()
