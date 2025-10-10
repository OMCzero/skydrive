# File Processing System

A comprehensive file processing system built with **Cloudflare Workers** and **Workflows** that provides:

- **File Upload** via web UI
- **SHA-256 and ssdeep hashing**
- **File type detection** using Google's Magika
- **Image captioning** with Moondream AI
- **PDF text extraction and summarization** with Workers AI
- **R2 storage** for files
- **D1 database** for metadata
- **Search functionality** across filenames, hashes, and metadata

## Architecture

- **Cloudflare Worker**: Serves web UI and handles API requests
- **Cloudflare Workflows**: Multi-step file processing pipeline with automatic retries
- **Cloudflare R2**: Object storage for uploaded files
- **Cloudflare D1**: SQLite database for file metadata
- **Workers AI**: PDF text summarization
- **Moondream API**: Image captioning (optional)

## Prerequisites

1. [Cloudflare account](https://dash.cloudflare.com/sign-up/workers-and-pages)
2. [Node.js](https://nodejs.org/) (v16.17.0 or later)
3. [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/install-and-update/)

## Setup Instructions

### 1. Install Dependencies

```bash
cd file-processor
npm install
```

### 2. Create D1 Database

```bash
npx wrangler d1 create file-metadata-db
```

Copy the `database_id` from the output and update `wrangler.jsonc`:

```jsonc
"d1_databases": [
  {
    "binding": "DB",
    "database_name": "file-metadata-db",
    "database_id": "<YOUR_DATABASE_ID_HERE>"
  }
]
```

### 3. Initialize Database Schema

```bash
npx wrangler d1 execute file-metadata-db --local --file=./schema.sql
npx wrangler d1 execute file-metadata-db --remote --file=./schema.sql
```

### 4. Create R2 Bucket

```bash
npx wrangler r2 bucket create uploaded-files
```

### 5. Configure Moondream API (Optional)

If you want image captioning, sign up for [Moondream API](https://moondream.ai) and set your API key:

```bash
npx wrangler secret put MOONDREAM_API_KEY
```

When prompted, paste your Moondream API key.

If you skip this step, image captioning will be skipped with a warning message.

### 6. Development

Start the local development server:

```bash
npm run start
```

Visit `http://localhost:8787` to access the web UI.

### 7. Deploy to Production

```bash
npm run deploy
```

Your application will be deployed to `https://file-processor.<your-subdomain>.workers.dev`

## Usage

### Upload Files

1. Open the web UI
2. Click the upload area or drag and drop a file
3. Click "Upload & Process"
4. The file will be processed through the workflow automatically

### Search Files

1. Enter a search query (filename, hash, or content)
2. Click "Search" or press Enter
3. View matching files with their metadata

### Supported Features by File Type

- **Images** (PNG, JPEG, etc.):
  - SHA-256 and ssdeep hashing
  - File type detection
  - Image captioning (if Moondream API key is set)
  - Storage in R2

- **PDFs**:
  - SHA-256 and ssdeep hashing
  - File type detection
  - Text extraction
  - AI-powered summarization
  - Storage in R2

- **Other Files**:
  - SHA-256 and ssdeep hashing
  - File type detection
  - Storage in R2

## Project Structure

```
file-processor/
├── src/
│   └── index.ts              # Main Worker and Workflow implementation
├── public/
│   └── index.html            # Web UI
├── schema.sql                # D1 database schema
├── wrangler.jsonc            # Cloudflare configuration
├── package.json              # Dependencies
└── README.md                 # This file
```

## API Endpoints

### POST `/api/upload`
Upload a file for processing

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "success": true,
  "id": "file-uuid",
  "workflowInstanceId": "workflow-instance-id",
  "message": "File uploaded and processing started"
}
```

### GET `/api/search?q=query`
Search for files

**Response**:
```json
{
  "success": true,
  "files": [
    {
      "id": "file-uuid",
      "filename": "example.pdf",
      "content_type": "application/pdf",
      "file_type": "pdf",
      "size": 12345,
      "sha256_hash": "abc123...",
      "ssdeep_hash": "xyz789...",
      "r2_key": "files/uuid/example.pdf",
      "created_at": 1234567890,
      "metadata": "summary: This is a summary..."
    }
  ],
  "count": 1
}
```

### GET `/api/status?id=workflow-instance-id`
Get workflow processing status

**Response**:
```json
{
  "success": true,
  "status": {
    "status": "running",
    "output": null
  }
}
```

## Database Schema

### `files` table
- `id` - Unique file identifier (UUID)
- `filename` - Original filename
- `content_type` - MIME type
- `file_type` - Detected file type (via Magika)
- `size` - File size in bytes
- `sha256_hash` - SHA-256 hash
- `ssdeep_hash` - ssdeep fuzzy hash
- `r2_key` - R2 storage key
- `created_at` - Upload timestamp

### `file_metadata` table
- `id` - Auto-increment ID
- `file_id` - Foreign key to `files.id`
- `metadata_type` - Type: 'caption', 'summary', or 'raw_text'
- `content` - Metadata content

## Technologies Used

- **Cloudflare Workers** - Serverless compute platform
- **Cloudflare Workflows** - Durable execution for multi-step processes
- **Cloudflare R2** - Object storage (S3-compatible)
- **Cloudflare D1** - Serverless SQLite database
- **Workers AI** - GPU-powered AI inference
- **Google Magika** - AI-powered file type detection
- **Moondream** - Vision language model for image captioning
- **ssdeep** - Fuzzy hashing for similarity detection
- **unpdf** - PDF text extraction

## Development Notes

### ssdeep Hashing
This project uses `ssdeep.js`, a pure JavaScript implementation of ssdeep, which works in Cloudflare Workers without native dependencies.

### Moondream Integration
Moondream Cloud API is used instead of the local model because Cloudflare Workers doesn't support running the full model. The API offers 5,000 free requests per day.

### PDF Processing
The `unpdf` library is used for PDF text extraction as it's lightweight and compatible with the Workers runtime.

### File Size Limits
- **Workers**: 100 MB request size limit
- **Workflows**: Can handle larger files by processing in chunks
- Consider implementing chunked upload for files larger than 100 MB

## Troubleshooting

### Database ID Error
Make sure you've replaced `<DATABASE_ID>` in `wrangler.jsonc` with your actual D1 database ID from the `wrangler d1 create` command.

### R2 Bucket Not Found
Ensure the R2 bucket name in `wrangler.jsonc` matches the bucket you created with `wrangler r2 bucket create`.

### Image Captioning Not Working
If you see "Image captioning skipped" messages, you need to set the `MOONDREAM_API_KEY` secret as described in the setup instructions.

### Type Errors
Run `npm run cf-typegen` to regenerate TypeScript type definitions after modifying `wrangler.jsonc`.

## Contributing

Feel free to submit issues or pull requests to improve this project!

## License

MIT
