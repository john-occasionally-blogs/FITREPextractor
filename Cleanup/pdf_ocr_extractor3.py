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
    Find common form field patterns in extracted text, focusing on actual values
    
    Args:
        text_data (list): List of text data from all pages
        
    Returns:
        dict: Structured form data
    """
    form_fields = {}
    
    # Improved patterns that capture values AFTER field labels
    patterns = {
        'first_name': [
            r'first\s*name[:\s_]*([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'fname[:\s_]*([A-Za-z]+(?:\s+[A-Za-z]+)*)'
        ],
        'last_name': [
            r'last\s*name[:\s_]*([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'lname[:\s_]*([A-Za-z]+(?:\s+[A-Za-z]+)*)',
            r'surname[:\s_]*([A-Za-z]+(?:\s+[A-Za-z]+)*)'
        ],
        'full_name': [
            r'(?:full\s*name|name)[:\s_]*([A-Za-z]+(?:\s+[A-Za-z]+)+)',
        ],
        'email': [
            r'(?:email|e-mail)[:\s_]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        ],
        'phone': [
            r'(?:phone|tel|mobile|cell)[:\s_]*([0-9\-\(\)\s\+\.]{7,})',
        ],
        'date': [
            r'(?:date|dob|birth)[:\s_]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})'
        ],
        'address': [
            r'(?:address|street)[:\s_]*([0-9]+[^\n\r]*)',
        ],
        'city': [
            r'city[:\s_]*([A-Za-z\s]+)',
        ],
        'state': [
            r'state[:\s_]*([A-Za-z\s]{2,})',
        ],
        'zip': [
            r'(?:zip|postal)[:\s_]*([0-9\-\s]{5,})',
        ],
        'age': [
            r'age[:\s_]*([0-9]{1,3})',
        ],
        'id_number': [
            r'(?:id|number|ssn)[:\s_]*([0-9\-\s]{3,})',
        ]
    }
    
    # Process each page's text
    for page_data in text_data:
        page_num = page_data['page']
        text = page_data['text']
        lines = page_data['lines']
        
        print(f"Analyzing page {page_num} for form patterns...")
        
        # Search for field-value patterns
        for field_type, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                for i, match in enumerate(matches):
                    if match and match.strip() and len(match.strip()) > 1:
                        # Filter out obvious labels/field names
                        match_clean = match.strip()
                        if not re.match(r'^(first|last|full|name|email|phone|address|city|state|zip|date)

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
            print("RAW EXTRACTED TEXT (check for missed values)")
            print("="*60)
            for page_data in raw_text:
                print(f"\n--- Page {page_data['page']} ---")
                print(page_data['text'])
        return
    
    print("\n" + "="*60)
    print("OCR FORM DATA EXTRACTION RESULTS")
    print("="*60)
    
    # Sort by confidence and page
    sorted_fields = sorted(form_data.items(), 
                          key=lambda x: (x[1]['page'], x[1].get('confidence', 'low') == 'high'), 
                          reverse=True)
    
    high_confidence = []
    medium_confidence = []
    low_confidence = []
    
    for field_name, field_info in sorted_fields:
        confidence = field_info.get('confidence', 'low')
        field_data = {
            'name': field_name,
            'info': field_info
        }
        
        if confidence == 'high':
            high_confidence.append(field_data)
        elif confidence == 'medium':
            medium_confidence.append(field_data)
        else:
            low_confidence.append(field_data)
    
    # Print high confidence fields first
    if high_confidence:
        print("\n🟢 HIGH CONFIDENCE FIELDS:")
        for field_data in high_confidence:
            field_name = field_data['name']
            field_info = field_data['info']
            print(f"\nField: {field_name}")
            print(f"  Type: {field_info['type']}")
            print(f"  Value: '{field_info['value']}'")
            print(f"  Page: {field_info['page']}")
            if 'label' in field_info:
                print(f"  Label: {field_info['label']}")
    
    if medium_confidence:
        print(f"\n🟡 MEDIUM CONFIDENCE FIELDS:")
        for field_data in medium_confidence:
            field_name = field_data['name']
            field_info = field_data['info']
            print(f"\nField: {field_name}")
            print(f"  Type: {field_info['type']}")
            print(f"  Value: '{field_info['value']}'")
            print(f"  Page: {field_info['page']}")
    
    if low_confidence:
        print(f"\n🔴 LOW CONFIDENCE FIELDS:")
        for field_data in low_confidence:
            field_name = field_data['name']
            field_info = field_data['info']
            print(f"\nField: {field_name}")
            print(f"  Type: {field_info['type']}")
            print(f"  Value: '{field_info['value']}'")
            print(f"  Page: {field_info['page']}")
    
    total_high = len(high_confidence)
    total_medium = len(medium_confidence)
    total_low = len(low_confidence)
    total_fields = total_high + total_medium + total_low
    
    print(f"\n" + "-"*60)
    print(f"Summary: {total_fields} fields detected")
    print(f"  🟢 High confidence: {total_high}")
    print(f"  🟡 Medium confidence: {total_medium}")
    print(f"  🔴 Low confidence: {total_low}")
    print("-"*60)
    
    if show_raw and raw_text:
        print("\n" + "="*60)
        print("RAW EXTRACTED TEXT (for manual review)")
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
, match_clean.lower()):
                            field_name = f"{field_type}_{i+1}_page_{page_num}" if i > 0 else f"{field_type}_page_{page_num}"
                            form_fields[field_name] = {
                                'value': match_clean,
                                'type': field_type,
                                'page': page_num,
                                'pattern': pattern,
                                'confidence': 'high' if any(char.isdigit() for char in match_clean) or '@' in match_clean else 'medium'
                            }
        
        # Look for filled underlines or boxes (common in filled forms)
        underline_patterns = [
            r'_{3,}\s*([A-Za-z0-9@\.\-\s]+)\s*_{0,}',  # Text over underlines
            r'([A-Za-z0-9@\.\-\s]+)\s*_{3,}',          # Text before underlines
        ]
        
        for pattern in underline_patterns:
            matches = re.findall(pattern, text)
            for i, match in enumerate(matches):
                if match and match.strip() and len(match.strip()) > 1:
                    match_clean = match.strip()
                    # Skip common field labels
                    if not re.match(r'^(first|last|name|email|phone|address|city|state|date|zip).*', match_clean.lower()):
                        field_name = f"underline_field_{i+1}_page_{page_num}"
                        form_fields[field_name] = {
                            'value': match_clean,
                            'type': 'underline_field',
                            'page': page_num,
                            'pattern': 'underline',
                            'confidence': 'medium'
                        }
        
        # Extract values from colon-separated lines (Label: Value)
        for line in lines:
            line = line.strip()
            if ':' in line and len(line) > 3:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    label = parts[0].strip().lower()
                    value = parts[1].strip()
                    
                    # Only keep if value is substantial and not just another label
                    if (value and len(value) > 1 and 
                        not re.match(r'^(yes|no|true|false|male|female)

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
, value.lower()) and
                        not value.lower() in ['first', 'last', 'name', 'email', 'phone', 'address']):
                        
                        field_name = f"labeled_field_{label.replace(' ', '_')}_page_{page_num}"
                        form_fields[field_name] = {
                            'value': value,
                            'type': 'labeled_field',
                            'page': page_num,
                            'pattern': 'colon_separator',
                            'label': label,
                            'confidence': 'high'
                        }
        
        # Look for checkbox/radio selections
        checkbox_patterns = [
            r'[☑✓✗×]\s*([A-Za-z\s]+)',  # Checked boxes followed by text
            r'\[x\]\s*([A-Za-z\s]+)',    # [x] style checkboxes
            r'•\s*([A-Za-z\s]+)',        # Bullet points (sometimes used for selections)
        ]
        
        for pattern in checkbox_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for i, match in enumerate(matches):
                if match and match.strip():
                    field_name = f"selected_option_{i+1}_page_{page_num}"
                    form_fields[field_name] = {
                        'value': match.strip(),
                        'type': 'checkbox_selection',
                        'page': page_num,
                        'pattern': 'checkbox',
                        'confidence': 'high'
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
