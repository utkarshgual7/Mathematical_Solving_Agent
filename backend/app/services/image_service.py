import os
import uuid
from typing import Dict, Optional
from PIL import Image
import pytesseract
import easyocr
import cv2
import numpy as np
from fastapi import UploadFile
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ImageProcessingService:
    def __init__(self):
        self.upload_dir = "uploads"
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize EasyOCR reader for better mathematical text recognition
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def process_image(self, file: UploadFile) -> Dict:
        """Process uploaded image and extract mathematical text"""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
            filename = f"{file_id}{file_extension}"
            file_path = os.path.join(self.upload_dir, filename)
            
            # Save uploaded file
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Process image in thread pool to avoid blocking
            extracted_text = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._extract_text_from_image, file_path
            )
            
            return {
                "success": True,
                "imageUrl": f"/uploads/{filename}",
                "extractedText": extracted_text,
                "filename": filename
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "imageUrl": "",
                "extractedText": ""
            }
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using multiple OCR methods"""
        try:
            # Load image
            image = cv2.imread(image_path)
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)
            
            # Try EasyOCR first (better for mathematical content)
            easyocr_text = self._extract_with_easyocr(processed_image)
            
            # Try Tesseract as fallback
            tesseract_text = self._extract_with_tesseract(processed_image)
            
            # Choose the better result (longer text usually means better extraction)
            if len(easyocr_text.strip()) > len(tesseract_text.strip()):
                extracted_text = easyocr_text
            else:
                extracted_text = tesseract_text
            
            # Clean and format the extracted text
            cleaned_text = self._clean_extracted_text(extracted_text)
            
            return cleaned_text if cleaned_text.strip() else "Mathematical problem detected in image. Please describe the problem or provide additional context."
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return "Error extracting text from image. Please try again or describe the problem manually."
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        return processed
    
    def _extract_with_easyocr(self, image: np.ndarray) -> str:
        """Extract text using EasyOCR"""
        try:
            results = self.ocr_reader.readtext(image)
            text_parts = []
            
            # Sort results by vertical position (top to bottom)
            results.sort(key=lambda x: x[0][0][1])
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter out low-confidence results
                    text_parts.append(text)
            
            return ' '.join(text_parts)
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def _extract_with_tesseract(self, image: np.ndarray) -> str:
        """Extract text using Tesseract OCR"""
        try:
            # Configure Tesseract for mathematical content
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-=()[]{}/<>.,;:!?\"\' '
            
            text = pytesseract.image_to_string(image, config=custom_config)
            return text
        except Exception as e:
            print(f"Tesseract error: {e}")
            return ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and format extracted text"""
        # Remove extra whitespace and newlines
        cleaned = ' '.join(text.split())
        
        # Fix common OCR errors for mathematical symbols
        replacements = {
            'x': '×',  # multiplication
            '/': '÷',  # division (in some contexts)
            '|': '1',  # common OCR error
            'O': '0',  # zero vs letter O
            'l': '1',  # one vs letter l
            'S': '5',  # five vs letter S (in some contexts)
        }
        
        # Apply replacements cautiously (only for obvious mathematical contexts)
        # This is a simplified approach - in production, you'd want more sophisticated logic
        
        return cleaned
    
    def get_image_path(self, filename: str) -> Optional[str]:
        """Get full path for uploaded image"""
        file_path = os.path.join(self.upload_dir, filename)
        if os.path.exists(file_path):
            return file_path
        return None

# Global instance
image_service = ImageProcessingService()