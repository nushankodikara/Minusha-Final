import os
import pytesseract
import cv2
import numpy as np
import re
import fitz  # PyMuPDF
from PIL import Image
import io
from django.conf import settings

class OCRProcessor:
    """Class for processing sugar reports using OCR"""
    
    def __init__(self):
        # Set Tesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # For Windows
        pass
    
    def process_report(self, report_file_path):
        """
        Process a sugar report file using OCR
        
        Parameters:
        report_file_path (str): Path to the report file
        
        Returns:
        dict: Extracted data from the report
        """
        try:
            # Get file extension
            ext = os.path.splitext(report_file_path)[1].lower()
            
            # Extract text based on file type
            if ext == '.pdf':
                text = self._extract_text_from_pdf(report_file_path)
            elif ext in ['.jpg', '.jpeg', '.png']:
                text = self._extract_text_from_image(report_file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Extract relevant data from the text
            data = self._extract_data_from_text(text)
            
            return data
        
        except Exception as e:
            # Log the error
            print(f"OCR processing error: {str(e)}")
            # Return empty data
            return {
                'glucose_level': None,
                'hba1c': None,
                'fasting_glucose': None,
                'postprandial_glucose': None,
                'error': str(e)
            }
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        text = ""
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Iterate through pages
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text from page
            page_text = page.get_text()
            text += page_text
            
            # Extract images if text extraction doesn't yield good results
            if not self._contains_glucose_data(page_text):
                # Get images from page
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert image bytes to OpenCV format
                    image = Image.open(io.BytesIO(image_bytes))
                    image_np = np.array(image)
                    
                    # Process image with OCR
                    img_text = pytesseract.image_to_string(image_np)
                    text += img_text
        
        return text
    
    def _extract_text_from_image(self, image_path):
        """Extract text from an image file"""
        # Read the image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Perform OCR
        text = pytesseract.image_to_string(thresh)
        
        return text
    
    def _contains_glucose_data(self, text):
        """Check if text contains glucose-related data"""
        glucose_keywords = [
            'glucose', 'blood sugar', 'hba1c', 'a1c', 'fasting', 'postprandial',
            'mg/dl', 'mmol/l', 'glycated', 'hemoglobin'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in glucose_keywords)
    
    def _extract_data_from_text(self, text):
        """Extract relevant data from the OCR text"""
        data = {
            'glucose_level': None,
            'hba1c': None,
            'fasting_glucose': None,
            'postprandial_glucose': None
        }
        
        # Regular expressions for extracting values
        # Glucose level (general)
        glucose_pattern = r'(?:glucose|blood sugar)[:\s]+(\d+\.?\d*)'
        # HbA1c
        hba1c_pattern = r'(?:hba1c|a1c|glycated hemoglobin)[:\s]+(\d+\.?\d*)'
        # Fasting glucose
        fasting_pattern = r'(?:fasting|fbg)[:\s]+(\d+\.?\d*)'
        # Postprandial glucose
        postprandial_pattern = r'(?:postprandial|pp|after meal)[:\s]+(\d+\.?\d*)'
        
        # Search for patterns in the text
        text_lower = text.lower()
        
        # Extract glucose level
        glucose_match = re.search(glucose_pattern, text_lower)
        if glucose_match:
            data['glucose_level'] = float(glucose_match.group(1))
        
        # Extract HbA1c
        hba1c_match = re.search(hba1c_pattern, text_lower)
        if hba1c_match:
            data['hba1c'] = float(hba1c_match.group(1))
        
        # Extract fasting glucose
        fasting_match = re.search(fasting_pattern, text_lower)
        if fasting_match:
            data['fasting_glucose'] = float(fasting_match.group(1))
        
        # Extract postprandial glucose
        postprandial_match = re.search(postprandial_pattern, text_lower)
        if postprandial_match:
            data['postprandial_glucose'] = float(postprandial_match.group(1))
        
        return data