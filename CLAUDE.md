# FileSystem Project Documentation

## Project Overview
A modular file/media management system built with Python. The system can ingest various file types (audio, video, PDFs, photos, etc.) and perform enrichment operations on them to extract metadata. File processing is handled asynchronously using a task queue.

## Project Structure
- `app.py` - Main FastAPI application with API endpoints
- `basic_metadata.py` - Module for extracting basic file metadata (hashes, exif data)
- `task_queue.py` - Async task queue implementation using Celery
- `worker.py` - Celery worker startup script
- `enrichment/` - Directory containing file enrichment modules
  - `__init__.py` - Registers and provides access to all enrichment functions
  - `image_enrichment.py` - Specific enrichment for image files
  - `text_extraction.py` - Text extraction from various file formats
  - (additional modules can be added for different file types)
- `static/` - Directory containing static files for the frontend
  - `index.html` - Simple web interface for testing uploads

## Development Principles
1. **Modularity**: Each enrichment function should be in its own file for easy maintenance
2. **Extensibility**: System should allow easy addition of new file types and enrichment functions
3. **Documentation**: Keep this file updated with all commands, patterns, and project information

## Development Workflow
1. Always feel free to ask clarifying questions
2. Suggest better/different/faster approaches when appropriate
3. Document new functionality in this file as it's developed

## Dependencies
- FastAPI: Web framework for building APIs
- python-multipart: For handling file uploads in FastAPI
- imagehash: For generating perceptual hashes of images
- pillow: For image processing
- PyExifTool: For extracting metadata from files
- Meilisearch: Fast, typo-tolerant search engine for file metadata
- uuid: For generating unique document IDs for search
- Google Magika: ML-based file type detection
- py-tlsh: Trend Micro Locality Sensitive Hash for fuzzy matching
- OpenAI Whisper: Speech recognition for audio/video transcription
- Ollama: Local LLM for text summarization and image description
  - llama3.2: Text analysis and summarization
  - llama3.2-vision: Image description and understanding

## Running the Application

### Using Docker (Recommended)

The easiest way to run the application is using Docker Compose:

```bash
# Install Ollama on your host machine from https://ollama.com/download
# Start the Ollama application
# Pull required models
ollama pull llama3.2
ollama pull llama3.2-vision

# Run the application stack (FastAPI, Celery, Redis, Meilisearch)
docker-compose up -d

# Access the application at http://localhost:8000

# To stop the application
docker-compose down
```

**Docker Configuration Notes:**
- The Docker configuration uses `SKIP_OLLAMA_WARMUP=true` environment variable to prevent the containers from trying to download models at startup
- Ollama runs on the host machine (not in Docker) for better performance with GPU access
- The containers communicate with Ollama on the host via `host.docker.internal`
- Volumes are configured for persistent storage of uploaded files and search indexes
- Redis data is persisted through a named volume

### Manual Setup

If you prefer to run the components individually:

```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis for task queue (using Docker)
docker run -d -p 6379:6379 redis:alpine

# Start Meilisearch (using Docker) with master key for authentication
docker run -p 7700:7700 -v $(pwd)/meili_data:/meili_data -e MEILI_NO_ANALYTICS=true -e MEILI_MASTER_KEY="masterKey" getmeili/meilisearch:latest

# Start Celery worker for background processing (in a separate terminal)
python worker.py

# Set the API key for the application
export MEILISEARCH_API_KEY="masterKey"

# Run the development server
uvicorn app:app --reload
```

## Task Queue System

The file management system uses Celery with Redis for asynchronous file processing:

1. **Upload Process Flow**:
   - User uploads a file via the `/upload` endpoint
   - File is temporarily saved to disk
   - A Celery task is created and queued
   - The API immediately returns a task ID
   - The file is processed in the background
   - Results are stored and indexed

2. **Task Status Tracking**:
   - Each task has a unique ID and status (PENDING, PROCESSING, SUCCESS, ERROR)
   - Users can check task status via the `/task/{task_id}` endpoint
   - List of tasks available via the `/tasks` endpoint with optional status filtering
   - Task results include full file metadata when processing is complete
   - Tasks are stored in Redis for persistence across server restarts
   - Automatic task cleanup removes old tasks after 7 days (configurable)
   - Metadata storage is optimized with summary data in task records and full metadata in separate keys
   - UI includes pagination controls for browsing large numbers of tasks

3. **System Status Monitoring**:
   - `/status` endpoint checks all components including Redis and Celery
   - Redis connectivity checked via direct connection ping
   - Celery worker availability verified by sending test ping task
   - Status page shows real-time component availability
   - Clear error messages when queue services are down
   - Worker setup instructions included in error messages

4. **Configuration**:
   - Redis connection configurable via REDIS_URL environment variable
   - Worker concurrency auto-scales based on available CPUs
   - Task time limit set to 1 hour for long-running processes (e.g., transcription)
   - Task results stored in Redis with configurable TTL (default: 7 days)
   - Periodic task cleanup runs hourly to remove expired task records
   - Task pagination configurable with adjustable page size
   - Full metadata storage optimization with separate keys for large task results

## API Endpoints
- POST `/upload`: Upload a file and get a task ID for background processing
  - Optional query parameter: `allow_duplicates=true` to bypass duplicate detection
- GET `/task/{task_id}`: Get the status and result of a specific task
  - Optional query parameter: `full_metadata=true` to include complete metadata for SUCCESS tasks
- GET `/tasks`: List recent tasks with optional status filtering and pagination
  - Parameters: `limit`, `page`, `status`, `refresh`, `include_metadata`
  - Returns pagination info with total count and page navigation
- GET `/`: Serve the web interface
- GET `/status`: Get system status for all components (TLSH, Magika, Whisper, Ollama, ExifTool, Meilisearch)
- GET `/search`: Search files using Meilisearch with filters for MIME type, extension, and size
  - Additional filters: `filter_id`, `hash_md5`, `hash_sha256` for direct and hash-based lookups
- DELETE `/search/{doc_id}`: Delete a document from the search index
- GET `/search/stats`: Get statistics about the search index
- GET `/download/{doc_id}`: Download a processed file

## Enrichment System

The file management system uses a modular enrichment architecture with searchable storage:

1. **Basic Metadata** (basic_metadata.py)
   - File information (size, type, name)
   - Advanced file type detection using Google's Magika
   - Cryptographic hashes (MD5, SHA1, SHA256)
   - TLSH fuzzy hash for similarity matching
   - Perceptual hashes for images
   - EXIF data extraction

2. **Type-Specific Enrichment** (enrichment/ directory)
   - Each module handles a specific file type or category
   - Modules export a single enrichment function
   - Enrichment functions are registered in `enrichment/__init__.py`
   - Detected by MIME type prefix matching
   - **Image Enrichment** includes:
     - Basic image metadata (dimensions, format, color mode)
     - Image statistics and color information
     - Animation details for GIFs
     - **LLM Image Description** using multimodal capabilities:
       - Generates concise, plain-text descriptions of image content (2-3 sentences)
       - Identifies objects, scenes, people, and text in images
       - Consistent format with text summarization (no markdown, single paragraph)
       - Uses fixed seed (42) for deterministic outputs
       - Post-processes descriptions to remove formatting artifacts
       - Enforces length limits (approximately 50 words)
       - Uses the same Ollama integration as text summarization
       - Configurable via:
         - OLLAMA_MODEL environment variable (default: "llama3.2-vision" for image description)
         - OLLAMA_HOST environment variable (default: "http://localhost:11434")
       - Descriptions are included in search index for semantic image search
       - Example description format: "A police car with 'INTERCEPTOR' markings drives down a road with lights illuminated. Another police vehicle is visible in the background, appearing blurred due to motion. The cars are responding to an emergency as indicated by their activated lights."

3. **Text Extraction** (enrichment/text_extraction.py)
   - OCR for images using pytesseract
   - PDF text extraction using pdfminer.six
   - PDF OCR fallback using pdf2image + pytesseract
   - Word document parsing using python-docx
   - Plain text file reading with encoding detection
   - Structured data extraction (JSON, XML, YAML) with format validation
   - Audio/video transcription using OpenAI's Whisper (local)
     - Configurable via WHISPER_MODEL environment variable
     - Options: "tiny", "base", "small", "medium", "large", "turbo"
     - Default: "turbo" (optimized version of large-v3, ~8x faster than large with minimal accuracy loss)
   - LLM summarization of extracted text using Ollama
     - Automatically summarizes any extracted text longer than 100 characters
     - Works with any file type that yields text (documents, images via OCR, audio via transcription, etc.)
     - Uses locally-run Llama 3.2 model by default
     - Configurable via:
       - OLLAMA_MODEL environment variable (default: "llama3.2")
       - OLLAMA_HOST environment variable (default: "http://localhost:11434")
     - Uses the official Ollama Python library for integration
     - Provides concise 2-3 sentence summaries of text content
     - Deterministic generation with fixed seed (42) for consistent results
     - Model warmup at server startup:
       - Pre-loads the model into memory
       - Keeps model warm for faster responses
       - Runs in background thread
     - Timeouts:
       - Client connection timeout: 1 hour (3600 seconds)
       - Summary generation timeout: 1 hour (3600 seconds)
     - Setup:
       - Install Python dependency: `pip install ollama`
       - Install Ollama server from https://ollama.com/download
       - Run Ollama service: start the Ollama application
       - Pull the text model: `ollama pull llama3.2`
       - Pull the vision model: `ollama pull llama3.2-vision` (for image descriptions)
       - Verify models: `ollama list` should show both models
       - Set model: `export OLLAMA_MODEL="llama3.2-vision"` (for image description)

The text extraction system provides a consistent output format:
```json
{
  "file_path": "path/to/file",
  "mime_type": "detected/mime-type",
  "extension": ".ext",
  "extracted_text": "The extracted text content...",
  "llm_summary": "A concise summary of the extracted text...",
  "metadata": {
    "method": "extraction_method_used",
    "format": "file_format",
    "encoding": "detected_encoding",
    "text_length": 1234,
    "word_count": 256,
    "llm_summary_model": "llama3.2",
    "llm_summary_tokens_processed": 512,
    "llm_summary_tokens_generated": 48,
    "additional_format_specific_data": "..."
  }
}
```

## Search Integration

The system integrates Meilisearch for powerful full-text search capabilities:

1. **Meilisearch Integration** (search_service.py)
   - Indexing of processed file metadata into searchable documents
   - Fast full-text search across all file metadata and extracted text
   - Search-as-you-type with typo tolerance
   - Faceted filtering by file type, extension, and size
   - Document management (adding/deleting)

   > **IMPORTANT CONFIGURATION NOTE**:
   > When adding new fields that need to be filtered or used for lookups in Meilisearch:
   > 1. Always add the field to BOTH the `searchable_attributes` and `filterable_attributes` arrays in `search_service.py`.
   > 2. For hash-based operations like duplicate detection, the hash fields MUST be included in `filterable_attributes`.
   > 3. After adding new filterable fields, you may need to reindex your data or restart the Meilisearch service.
   > 4. Fields used in filters with the `search()` method must be defined in the `filterable_attributes` array.
   > 5. Errors like `MeilisearchApiError. Error code: invalid_search_filter. Error message: Index \`files\`: Attribute X is not filterable` indicate that a field needs to be added to `filterable_attributes`.
   >
   > **Example**: For hash-based duplicate detection, we use:
   > ```python
   > # In the filterable_attributes array:
   > "hashes.md5",
   > "hashes.sha256",
   > 
   > # When filtering in the search API:
   > filters["hashes.md5"] = hash_value
   > ```

2. **Search Features**
   - Automatic indexing of uploaded files
   - Searching by keywords (in filenames, metadata, text extraction, summaries, image descriptions)
   - Filtering results by MIME type, file extension, and size category
   - Frontend integration with search-as-you-type interface
   - Document deduplication via content hashing
   - Results displayed with summaries and metadata
   - Semantic image search via LLM-generated image descriptions
   - Find images by content (e.g., "sunset", "car", "people") regardless of filename

3. **Duplicate Detection** (app.py)
   - Automatic detection of duplicate files during upload
   - Hash-based comparison (MD5 for speed, SHA256 for confirmation)
   - User notification when duplicates are detected
   - Options to:
     - View the existing duplicate file in search
     - Force upload anyway (with `allow_duplicates=true` parameter)
     - Upload a different file
   - Hash-based search capability (`hash_md5` and `hash_sha256` parameters)
   - Direct document lookup by ID (`filter_id` parameter)

4. **Search Configuration**
   - Environment variable configuration (MEILISEARCH_HOST, MEILISEARCH_API_KEY, MEILISEARCH_INDEX)
   - Customizable relevance rules and searchable attributes
   - Size categorization for easy filtering (tiny, small, medium, large, huge)
   - Graceful fallback when search service is unavailable

## File Type Detection and Processing Flow

The file management system uses multiple layers of file type detection to determine how to process each file:

### Detection Process

1. **Initial MIME Type Detection**:
   - Uses Python's `mimetypes` module to guess type from file extension
   - This is fast but relies on correct file extensions

2. **Advanced Content-Based Detection**:
   - Uses Google's Magika ML-based detection to analyze file content
   - Works even when file extensions are missing or incorrect
   - Provides confidence scores and detailed file type information

3. **Secondary Metadata Extraction**:
   - ExifTool provides additional format-specific metadata
   - This helps with encoding detection, dimensions, and other properties

### Processing Flow

When a file is uploaded, it follows this processing flow:

1. File is saved to a temporary location with original extension
2. Basic metadata is extracted (size, timestamps, etc.)
3. File type detection is performed (MIME type + Magika)
4. Cryptographic hashes are calculated (MD5, SHA1, SHA256, TLSH)
5. If applicable, perceptual hashes are calculated for images
6. EXIF metadata is extracted if available
7. If file type matches a registered enrichment function, it's applied
8. If file type supports text extraction, it's performed

The routing logic is defined in these locations:

- `basic_metadata.py`: Main processing orchestration
- `enrichment/__init__.py`: Defines mappings between MIME types and enrichment functions
- `enrichment/text_extraction.py`: Contains logic for extracting text from different file types

### MIME Type Routing

File processing is determined by MIME type mappings:

```python
# For enrichment functions
ENRICHMENT_FUNCTIONS = {
    "image/": enrich_image,
    # Additional mappings can be added here
}

# For text extraction
TEXT_EXTRACTION_MIME_TYPES = [
    "image/",           # Images (OCR)
    "application/pdf",  # PDF
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/msword",  # DOC
    "text/",            # Text files
]
```

### Adding a New Enrichment Module

To add support for a new file type:

1. Create a new Python file in the `enrichment/` directory (e.g., `pdf_enrichment.py`)
2. Implement the enrichment function that takes a file path and returns metadata
3. Add the function to `enrichment/__init__.py`:
   - Import the function
   - Add it to the `ENRICHMENT_FUNCTIONS` dictionary with appropriate MIME type prefix

Example:
```python
# In enrichment/pdf_enrichment.py
def enrich_pdf(file_path: str) -> dict:
    # Extract PDF-specific metadata
    return {"pages": 10, "author": "Example"}

# In enrichment/__init__.py
try:
    from .pdf_enrichment import enrich_pdf
except ImportError:
    enrich_pdf = None
    
# Add to the dictionary
ENRICHMENT_FUNCTIONS = {
    "image/": enrich_image,
    "application/pdf": enrich_pdf,
}
```

### Adding Text Extraction for a New File Type

To add text extraction support for a new file type:

1. Create a new extraction function in `text_extraction.py` 
2. Add the MIME type to `TEXT_EXTRACTION_MIME_TYPES` in `enrichment/__init__.py`
3. Update the routing in the `extract_text` function

Example for EPUB files:
```python
# In text_extraction.py
def extract_text_from_epub(file_path: str) -> Dict[str, Any]:
    # Extract text from EPUB
    return {"extracted_text": "Content from EPUB", "method": "epub_extraction"}

# In __init__.py, add to TEXT_EXTRACTION_MIME_TYPES
TEXT_EXTRACTION_MIME_TYPES = [
    "image/",
    "application/pdf",
    "application/epub+zip",  # Add EPUB MIME type
    # ...
]

# In extract_text function, add routing
elif mime_type == 'application/epub+zip':
    extraction_result = extract_text_from_epub(file_path)
```