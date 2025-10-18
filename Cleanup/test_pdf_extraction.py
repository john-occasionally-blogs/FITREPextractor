#!/usr/bin/env python3
"""
Test PDF text extraction approaches to see what works best for checkbox detection
"""

import fitz  # PyMuPDF
import json
from pathlib import Path

def extract_pdf_to_text(pdf_path):
    """Extract raw text from PDF"""
    doc = fitz.open(pdf_path)
    full_text = ""
    pages_text = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages_text.append(text)
        full_text += f"\n--- PAGE {page_num + 1} ---\n{text}\n"
    
    doc.close()
    return full_text, pages_text

def extract_pdf_to_dict(pdf_path):
    """Extract PDF content to structured dictionary"""
    doc = fitz.open(pdf_path)
    result = {"pages": []}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        page_data = {
            "page_number": page_num + 1,
            "text": page.get_text(),
            "text_dict": page.get_text("dict"),  # Structured text with fonts, positions
            "blocks": page.get_text("blocks"),   # Text blocks with coordinates
        }
        
        # Try to extract form fields if any
        try:
            widgets = page.widgets()
            if widgets:
                page_data["form_fields"] = []
                for widget in widgets:
                    field_info = {
                        "field_name": widget.field_name,
                        "field_type": widget.field_type,
                        "field_value": widget.field_value,
                        "rect": widget.rect,
                        "choices": getattr(widget, 'choice_values', None)
                    }
                    page_data["form_fields"].append(field_info)
        except:
            pass
        
        result["pages"].append(page_data)
    
    doc.close()
    return result

def main():
    pdf_path = "/Users/John/Desktop/FITREP Test/fitrepPdf.pdf"
    
    print("=== Testing PDF Text Extraction ===\n")
    
    # Method 1: Raw text extraction
    print("1. Raw text extraction:")
    try:
        full_text, pages_text = extract_pdf_to_text(pdf_path)
        
        print(f"Total pages: {len(pages_text)}")
        print(f"Page 2 text preview (first 500 chars):")
        print(repr(pages_text[1][:500]) if len(pages_text) > 1 else "No page 2")
        print(f"\nPage 3 text preview (first 500 chars):")
        print(repr(pages_text[2][:500]) if len(pages_text) > 2 else "No page 3")
        
        # Look for checkbox patterns
        page2_text = pages_text[1] if len(pages_text) > 1 else ""
        
        print(f"\nChecking for checkbox patterns in page 2:")
        print(f"- Contains 'X': {'X' in page2_text}")
        print(f"- Contains brackets: {'[' in page2_text or ']' in page2_text}")
        print(f"- Contains A B C D E: {all(letter in page2_text for letter in ['A', 'B', 'C', 'D', 'E'])}")
        
    except Exception as e:
        print(f"Error with raw text: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Method 2: Structured extraction
    print("2. Structured extraction:")
    try:
        pdf_dict = extract_pdf_to_dict(pdf_path)
        
        print(f"Extracted {len(pdf_dict['pages'])} pages")
        
        # Check if there are form fields
        has_form_fields = any('form_fields' in page for page in pdf_dict['pages'])
        print(f"Has form fields: {has_form_fields}")
        
        if has_form_fields:
            for page in pdf_dict['pages']:
                if 'form_fields' in page:
                    print(f"Page {page['page_number']} form fields: {len(page['form_fields'])}")
                    for field in page['form_fields'][:3]:  # Show first 3
                        print(f"  - {field['field_name']}: {field['field_type']} = {field['field_value']}")
        
        # Save structured data for analysis
        with open("/Users/John/Desktop/FITREP Test/pdf_structure.json", "w") as f:
            json.dump(pdf_dict, f, indent=2, default=str)
        print("Saved structure to pdf_structure.json")
        
    except Exception as e:
        print(f"Error with structured extraction: {e}")

if __name__ == "__main__":
    main()