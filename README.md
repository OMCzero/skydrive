# OMCzero SkyDrive

> [!WARNING]
> This is PRE-ALPHA software and should not be used in production.
>
> It is under active development, working towards a v0 release in the near future. Stay tuned for updates.



## Features

- Asynchronous file processing with background task queue
- File upload and metadata extraction
- Advanced file type detection using Google's Magika
- Multiple hash calculations (MD5, SHA1, SHA256)
- TLSH fuzzy hash for file similarity detection
- Perceptual hash for images (pHash, dHash, aHash, wHash)
- EXIF data extraction using ExifTool
- Text extraction from various file formats:
  - OCR for images using Tesseract
  - PDF text extraction with OCR fallback
  - Word document text extraction
  - Plain text files with encoding detection
  - Audio transcription using OpenAI's Whisper (runs locally)
- Modular enrichment system for different file types
- Task status tracking and task history
- LLM-based text summarization and image description
- Simple web interface for testing
- Duplicate file detection
- Persistent task storage with Redis

## Getting Started

### Docker Deployment (Recommended)

The easiest way to run the application is with Docker Compose:

1. Make sure you have Docker and Docker Compose installed:
   ```bash
   docker --version
   docker-compose --version
   ```

2. Ensure Ollama is installed and running on your host machine:
   ```bash
   # Install Ollama from https://ollama.com/download
   # Start the Ollama application
   
   # Pull required models
   ollama pull llama3.2
   ollama pull llama3.2-vision
   ```

3. Run the application:
   ```bash
   docker-compose up -d
   ```

4. Access the application at http://localhost:8000

5. To stop the application:
   ```bash
   docker-compose down
   ```

> **Note:** The Docker configuration includes `ASSUME_OLLAMA_MODELS_EXIST=true` and `SKIP_OLLAMA_WARMUP=true` to prevent automatic checking and downloading of Ollama models inside containers. Ollama runs on your host machine for better performance, not in Docker. The containers will connect to the Ollama service running on your host.

### Manual Installation

#### Prerequisites

- Python 3.8+
- Redis (for task queue)
- ExifTool (for enhanced metadata extraction)
- Tesseract OCR (for text extraction from images)
- Poppler (for PDF to image conversion for OCR)
- FFmpeg (for audio processing with Whisper)
- Meilisearch (for search functionality)
- Ollama (for LLM text summarization and image description)

#### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd file_system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Redis (for task queue):
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis:alpine
   
   # Or if you have Redis installed locally
   redis-server
   ```

4. Start Meilisearch (for search):
   ```bash
   docker run -d -p 7700:7700 -v $(pwd)/meili_data:/meili_data -e MEILI_NO_ANALYTICS=true -e MEILI_MASTER_KEY="masterKey" getmeili/meilisearch:latest
   ```

5. Start the Celery worker (in a separate terminal):
   ```bash
   python worker.py
   ```

6. Run the application:
   ```bash
   export MEILISEARCH_API_KEY="masterKey"
   uvicorn app:app --reload
   ```

7. Access the web interface at http://localhost:8000

## API Documentation

When the server is running, you can access the API documentation at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

### Key Endpoints

- `POST /upload` - Upload a file and get a task ID for tracking
- `GET /task/{task_id}` - Get the status of a specific task
- `GET /tasks` - List recent tasks with optional status filtering
- `GET /search` - Search for files using various filters
- `GET /download/{doc_id}` - Download a processed file
- `GET /status` - Check system status (all components)

## Adding New File Type Support

See the [CLAUDE.md](CLAUDE.md) file for detailed documentation on how to add support for additional file types through the enrichment system.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
