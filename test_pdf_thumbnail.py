#!/usr/bin/env python3
"""
Test script for PDF thumbnail generation.
This can be run directly to test if PDF thumbnailing is working correctly.
"""
import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Check if pdf2image and poppler are installed
try:
    import pdf2image
    print("pdf2image is installed correctly")
except ImportError:
    print("ERROR: pdf2image is not installed")
    print("Try: pip install pdf2image")
    sys.exit(1)

# Check if Poppler utilities are installed
try:
    result = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True)
    print(f"Poppler utilities found: {result.stderr.strip()}")
except FileNotFoundError:
    print("ERROR: Poppler utilities (pdftoppm) not found")
    print("Install poppler-utils package:")
    print("  - For Ubuntu/Debian: sudo apt-get install poppler-utils")
    print("  - For macOS: brew install poppler")
    print("  - For Windows: See pdf2image documentation")
    sys.exit(1)

# Create a test PDF using reportlab
test_pdf_path = None
try:
    print("\nCreating test PDF with reportlab...")
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create PDF in a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            test_pdf_path = tmp.name
            c = canvas.Canvas(test_pdf_path, pagesize=letter)
            c.setFont("Helvetica", 24)
            c.drawString(100, 750, "Test PDF Document")
            c.save()
            print(f"Test PDF created with reportlab at: {test_pdf_path}")
    except ImportError:
        print("reportlab not available, creating a simpler PDF...")
        # Create a simpler test file for the PDF thumbnail test
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            test_pdf_path = tmp.name
            # This is a valid 1-page PDF with minimal content
            simple_pdf = b"""%PDF-1.7
1 0 obj<</Pages 2 0 R/Type/Catalog>>endobj
2 0 obj<</Count 1/Kids[3 0 R]/Type/Pages>>endobj
3 0 obj<</Contents 4 0 R/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>/Type/Page>>endobj
4 0 obj<</Length 18>>stream
BT /F1 12 Tf ET
endstream
endobj
5 0 obj<</Author/Unknown/CreationDate(D:20140324120910)/Creator/Unspecified/Keywords/None/ModDate(D:20140324120910)/Producer/Unspecified/Subject/None/Title/Test PDF>>endobj
xref
0 6
0000000000 65535 f 
0000000015 00000 n 
0000000060 00000 n 
0000000111 00000 n 
0000000200 00000 n 
0000000266 00000 n 
trailer<</Root 1 0 R/Info 5 0 R/Size 6>>
startxref
437
%%EOF"""
            tmp.write(simple_pdf)
            print(f"Simple test PDF created at: {test_pdf_path}")
    
    # Check if the file was created and has content
    if not test_pdf_path or not os.path.exists(test_pdf_path):
        print("ERROR: Failed to create test PDF file")
        sys.exit(1)
        
    file_size = os.path.getsize(test_pdf_path)
    print(f"Test PDF file size: {file_size} bytes")
    
    if file_size == 0:
        print("ERROR: Test PDF is empty")
        sys.exit(1)
        
    # Attempt to process the PDF
    print("\nExtracting PDF info...")
    info = pdf2image.pdfinfo_from_path(test_pdf_path)
    print(f"PDF info: {info}")
    
    print("\nConverting PDF to image...")
    # Size param matches what we use in the real code
    images = pdf2image.convert_from_path(test_pdf_path, first_page=1, last_page=1, size=(150, 150))
    print(f"PDF converted successfully into {len(images)} image(s)")
    
    # Save the image to verify file writing works
    if images:
        output_path = os.path.join(tempfile.gettempdir(), "test_thumbnail.jpg")
        images[0].save(output_path, "JPEG")
        print(f"Thumbnail saved to: {output_path}")
        if os.path.exists(output_path):
            print(f"Thumbnail file exists and is {os.path.getsize(output_path)} bytes")
            print("PDF THUMBNAIL TEST SUCCESSFUL!")
        else:
            print("ERROR: Failed to save thumbnail file")
    else:
        print("ERROR: No images returned from conversion")
        
except Exception as e:
    print(f"ERROR: PDF processing failed: {e}")
    
finally:
    # Clean up the test PDF
    if test_pdf_path and os.path.exists(test_pdf_path):
        os.unlink(test_pdf_path)
        print(f"\nTest PDF cleaned up: {test_pdf_path}") 