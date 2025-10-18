#!/usr/bin/env python3
"""
PDF Form Data Extractor for Mac
Extracts text fields and radio button values from PDF forms
"""

import sys
from pathlib import Path
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
        print("Warning: PyPDF2 is deprecated. Consider upgrading to pypdf with: pip install pypdf")
    except ImportError:
        print("PDF library not installed. Install with: pip install pypdf")
        print("Alternative: pip install PyPDF2 (deprecated)")
        sys.exit(1)

def extract_form_data(pdf_path):
    """
    Extract form field data from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: Dictionary containing field names and their values
    """
    try:
        reader = PdfReader(pdf_path)
        
        # Check if PDF has form fields
        if reader.metadata is None or '/AcroForm' not in str(reader.metadata):
            print(f"Warning: {pdf_path} may not contain fillable form fields")
        
        form_data = {}
        
        # Extract form fields from all pages
        for page_num, page in enumerate(reader.pages):
            if '/Annots' in page:
                annotations = page['/Annots']
                
                for annotation in annotations:
                    annotation_obj = annotation.get_object()
                    
                    # Check if it's a form field
                    if '/T' in annotation_obj:  # Field name
                        field_name = annotation_obj['/T']
                        field_value = None
                        field_type = "unknown"
                        
                        # Get field value
                        if '/V' in annotation_obj:
                            field_value = annotation_obj['/V']
                            
                        # Determine field type
                        if '/FT' in annotation_obj:
                            field_type_obj = annotation_obj['/FT']
                            if field_type_obj == '/Tx':  # Text field
                                field_type = "text"
                            elif field_type_obj == '/Btn':  # Button field (radio/checkbox)
                                field_type = "button"
                                # For radio buttons, check if it's selected
                                if '/AS' in annotation_obj:
                                    appearance_state = annotation_obj['/AS']
                                    # Radio button is selected if appearance state is not '/Off'
                                    if appearance_state != '/Off':
                                        field_value = str(appearance_state).replace('/', '')
                                    else:
                                        field_value = None
                            elif field_type_obj == '/Ch':  # Choice field (dropdown/list)
                                field_type = "choice"
                        
                        # Store the field data
                        if field_name:
                            form_data[str(field_name)] = {
                                'value': str(field_value) if field_value else None,
                                'type': field_type,
                                'page': page_num + 1
                            }
        
        return form_data
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return {}

def print_form_data(form_data, show_empty=False):
    """
    Print form data in a readable format
    
    Args:
        form_data (dict): Form data dictionary
        show_empty (bool): Whether to show fields with no values
    """
    if not form_data:
        print("No form fields found or error reading PDF")
        return
    
    print("\n" + "="*60)
    print("PDF FORM DATA EXTRACTION RESULTS")
    print("="*60)
    
    filled_fields = 0
    empty_fields = 0
    
    for field_name, field_info in form_data.items():
        value = field_info['value']
        field_type = field_info['type']
        page = field_info['page']
        
        if value and value != 'None':
            filled_fields += 1
            print(f"\nField: {field_name}")
            print(f"  Type: {field_type}")
            print(f"  Value: {value}")
            print(f"  Page: {page}")
        elif show_empty:
            empty_fields += 1
            print(f"\nField: {field_name}")
            print(f"  Type: {field_type}")
            print(f"  Value: [EMPTY]")
            print(f"  Page: {page}")
        else:
            empty_fields += 1
    
    print(f"\n" + "-"*60)
    print(f"Summary: {filled_fields} filled fields, {empty_fields} empty fields")
    print("-"*60)

def save_to_csv(form_data, output_path):
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
            writer.writerow(['Field Name', 'Type', 'Value', 'Page'])
            
            for field_name, field_info in form_data.items():
                writer.writerow([
                    field_name,
                    field_info['type'],
                    field_info['value'] or '',
                    field_info['page']
                ])
        
        print(f"Data saved to: {output_path}")
        
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def main():
    """Main function to handle command line arguments"""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python pdf_form_extractor.py <pdf_file> [options]")
        print("\nOptions:")
        print("  --show-empty    Show fields with no values")
        print("  --csv <file>    Save results to CSV file")
        print("\nExample:")
        print("  python pdf_form_extractor.py form.pdf --show-empty --csv output.csv")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    show_empty = '--show-empty' in sys.argv
    
    # Check if PDF file exists
    if not Path(pdf_path).exists():
        print(f"Error: File '{pdf_path}' not found")
        sys.exit(1)
    
    print(f"Extracting form data from: {pdf_path}")
    
    # Extract form data
    form_data = extract_form_data(pdf_path)
    
    # Print results
    print_form_data(form_data, show_empty)
    
    # Save to CSV if requested
    if '--csv' in sys.argv:
        try:
            csv_index = sys.argv.index('--csv') + 1
            if csv_index < len(sys.argv):
                csv_path = sys.argv[csv_index]
                save_to_csv(form_data, csv_path)
            else:
                print("Error: --csv option requires a filename")
        except ValueError:
            pass

if __name__ == "__main__":
    main()
