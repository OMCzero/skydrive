FROM python:3.12

WORKDIR /app

# Install system dependencies
# poppler-utils is required for PDF to image conversion
# ghostscript helps with PDF rendering
# ffmpeg is required for video thumbnail generation
RUN apt-get update && apt-get install -y \
    libimage-exiftool-perl \
    tesseract-ocr \
    poppler-utils \
    ghostscript \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt 
RUN pip install --no-cache-dir ollama

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p uploads
RUN mkdir -p uploads/thumbnails
RUN mkdir -p static

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
