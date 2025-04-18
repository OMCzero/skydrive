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
- Automatic thumbnail generation for:
  - Images (using PIL)
  - Videos (using FFmpeg)
  - PDFs (using pdf2image/Poppler)
- Text extraction from various file formats:
  - OCR for images using Tesseract
  - PDF text extraction with OCR fallback
  - Word document text extraction
  - Plain text files with encoding detection
  - Audio transcription using an external Transcription API (requires separate `transcription_server.py`)
  - Enhanced file type detection using Magika for more accurate text extraction
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

3. Generate the MeiliSearch API key:
   ```bash
   cp .env.example .env
   # Edit the .env file to set a secure random key
   # You can generate one with: openssl rand -hex 16
   ```

4. Run the application:
   ```bash
   docker-compose up -d
   ```

   > **Note on External Services:** This setup assumes **Ollama** and the **Transcription Server** (`transcription_server.py`) are running on your **host machine**, not within Docker containers. The containers are configured to connect to these services via `host.docker.internal`.
   >
   > - **Ollama:** Ensure Ollama is installed and running locally. The `OLLAMA_HOST` environment variable in `.env` points to it.
   > - **Transcription Server:** You need to run the `transcription_server.py` script separately on a machine with sufficient resources (CPU/GPU) for audio processing. Start it using `python transcription_server.py`. The `TRANSCRIPTION_API_URL` environment variable in `.env` points to its endpoint (default: `http://host.docker.internal:9000/transcribe/`).

5. Access the application at http://localhost:8000

6. To stop the application:
   ```bash
   docker-compose down
   ```

> **Development:** The quick command to re-run the app every time: `rm -rf meili_data/ uploads/ && docker compose down && docker system prune -f && docker compose down && clear && docker compose up --build`

### Manual Installation

#### Prerequisites

- Python 3.8+
- Redis (for task queue)
- ExifTool (for enhanced metadata extraction)
- Tesseract OCR (for text extraction from images)
- Poppler (for PDF to image conversion for OCR)
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

6. Run the Transcription Server (in a separate terminal, on a machine with suitable resources):
   ```bash
   # Ensure you have installed requirements for the server (e.g., whisper, fastapi, uvicorn)
   # pip install openai-whisper fastapi uvicorn python-multipart
   python transcription_server.py
   ```

7. Run the main application (in another terminal):
   ```bash
   export MEILISEARCH_API_KEY="masterKey"
   export TRANSCRIPTION_API_URL="http://localhost:9000/transcribe/" # Adjust if server runs elsewhere
   uvicorn app:app --reload
   ```

8. Access the web interface at http://localhost:8000

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
- `GET /admin` - Access the admin dashboard with file statistics
- `GET /admin/stats` - Get detailed stats about files (JSON endpoint for admin dashboard)

## Admin Dashboard

The system includes an admin dashboard accessible at http://localhost:8000/admin that provides statistics about the files stored in the system:

- Biggest file stored (with download link)
- Pie chart showing distribution of file types
- Pie chart showing distribution of file uploaders (users)

This dashboard is useful for monitoring system usage and understanding the composition of stored files.

## Adding New File Type Support

See the [CLAUDE.md](CLAUDE.md) file for detailed documentation on how to add support for additional file types through the enrichment system.

## Continuous Integration

This project uses GitHub Actions for continuous integration to ensure Docker builds remain stable:

- **Docker Build Validation**: Automatically runs on every PR and push to main that affects Docker-related files
- **Validation Process**:
  - Builds the Docker image to ensure all dependencies are installable
  - Validates the docker-compose.yml file
  - Fails early if any issues would prevent deployment

You can see the workflow configuration in the `.github/workflows/docker-build.yml` file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### PDF Thumbnail Generation Issues

If you're experiencing issues with PDF thumbnail generation in Docker, you can run the included test script to diagnose the problem:

1. Access the container shell:
   ```bash
   docker exec -it skydrive-app-1 /bin/bash
   ```

2. Run the PDF thumbnail test script:
   ```bash
   python test_pdf_thumbnail.py
   ```

3. Check the output for any errors:
   - If the script runs successfully, it will create and then process a test PDF file
   - Look for "PDF THUMBNAIL TEST SUCCESSFUL!" message
   - If errors occur, they will be displayed with details

Common issues include:
- Missing system dependencies (solved by ensuring Poppler and GhostScript are installed)
- PDF file format issues (some PDFs may not be processable)
- Permission problems in the container environment

For local development, make sure you have:
- Poppler utilities installed (`apt-get install poppler-utils` on Debian/Ubuntu or `brew install poppler` on macOS)
- GhostScript installed for PDF processing
- The pdf2image Python package installed (`pip install pdf2image`)
