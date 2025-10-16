# SkyDrive Architecture

## File Processing Flow

This document describes how different file types are processed when uploaded to SkyDrive.

```mermaid
flowchart TD
    Start([File Upload]) --> ValidateSize{Size OK?<br/>< 100MB}
    ValidateSize -->|No| Error1[❌ Error: File too large]
    ValidateSize -->|Yes| HashFile[🔐 Hash File<br/>SHA-256 + ssdeep]

    HashFile --> DetectType[🔍 Detect File Type<br/>Google Magika]

    DetectType --> TypeCheck{What type?}

    TypeCheck -->|Image<br/>png/jpg/jpeg| ImagePath[🖼️ Image Processing]
    TypeCheck -->|PDF| PDFPath[📄 PDF Processing]
    TypeCheck -->|Other| OtherPath[📦 Generic Processing]

    %% Image Processing Path
    ImagePath --> CheckMoondream{Moondream<br/>API Key?}
    CheckMoondream -->|Yes| Moondream[🤖 Image Captioning<br/>Moondream API]
    CheckMoondream -->|No| SkipCaption[⚠️ Skip Captioning]
    Moondream --> CaptionStore[💾 Store Caption<br/>metadata.caption]
    SkipCaption --> CaptionStore
    CaptionStore --> StoreR2

    %% PDF Processing Path
    PDFPath --> ExtractText[📝 Extract Text<br/>Page by Page]
    ExtractText --> CheckPages{Any page<br/>< 50 chars?}
    CheckPages -->|Yes + Total < 1000 chars| OCRPath[👁️ OCR Processing]
    CheckPages -->|No| TextOK[✅ Use Extracted Text]

    OCRPath --> ExtractImages[🖼️ Extract Embedded<br/>Images from PDF]
    ExtractImages --> VisionOCR[🤖 Vision Model OCR<br/>Llama 3.2 Vision<br/>per image]
    VisionOCR --> CombineOCR[📋 Combine OCR Text]
    CombineOCR --> PDFTextReady

    TextOK --> PDFTextReady{Text<br/>Available?}
    PDFTextReady -->|Yes| Summarize[🤖 Summarize<br/>Llama 3.1 Fast<br/>First 4000 chars]
    PDFTextReady -->|No| EmptyPDF[⚠️ Empty PDF]

    Summarize --> PDFMetadata[💾 Store PDF Data<br/>metadata.rawText<br/>metadata.summary<br/>metadata.ocrPerformed]
    EmptyPDF --> PDFMetadata
    PDFMetadata --> StoreR2

    %% Other Files Path
    OtherPath --> NoProcessing[ℹ️ No Special Processing]
    NoProcessing --> StoreR2

    %% Storage Phase
    StoreR2[☁️ Store in R2<br/>files/fileId/filename]
    StoreR2 --> StoreD1[💾 Store Metadata in D1<br/>- ID, filename, type, size<br/>- Hashes<br/>- JSON metadata]

    StoreD1 --> Complete([✅ Processing Complete<br/>File Searchable])

    %% Search Capabilities
    Complete -.->|Searchable by| Search[🔍 Search Fields]
    Search -.-> SearchFields[📋 Search Across:<br/>• Filename<br/>• SHA-256 hash<br/>• ssdeep hash<br/>• Image captions<br/>• PDF summaries<br/>• PDF raw text/OCR]

    %% Styling
    classDef imageClass fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef pdfClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef ocrClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef storageClass fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef errorClass fill:#ffebee,stroke:#c62828,stroke-width:2px

    class ImagePath,Moondream,CaptionStore,CheckMoondream,SkipCaption imageClass
    class PDFPath,ExtractText,TextOK,Summarize,PDFMetadata,PDFTextReady,EmptyPDF pdfClass
    class OCRPath,ExtractImages,VisionOCR,CombineOCR,CheckPages ocrClass
    class StoreR2,StoreD1,Complete,Search,SearchFields storageClass
    class Error1,EmptyPDF errorClass
```

## Supported File Types

### 1. Images (PNG, JPG, JPEG)
- **Detection**: Google Magika file type detection
- **Processing**:
  - Image captioning via Moondream API (if `MOONDREAM_API_KEY` is set)
  - Generates natural language description of image content
- **Storage**:
  - Original image → R2
  - Caption → `metadata.caption` in D1
- **Searchable**: Filename, hashes, caption text

### 2. PDFs
- **Detection**: Google Magika file type detection
- **Text Extraction**: Page-by-page text extraction using `unpdf`
- **OCR Trigger**: Automatic OCR if any page has < 50 characters AND total text < 1000 chars
- **OCR Method**:
  1. Extract embedded images from PDF using `pdf-lib`
  2. Run Llama 3.2 Vision model on each image
  3. Combine OCR'd text from all images
- **Summarization**: First 4000 chars sent to Llama 3.1 Fast for summary
- **Storage**:
  - Original PDF → R2
  - Text/OCR → `metadata.rawText` (first 50k chars)
  - Summary → `metadata.summary`
  - OCR flag → `metadata.ocrPerformed`
- **Searchable**: Filename, hashes, raw text, summary

### 3. All Other Files
- **Detection**: Google Magika file type detection
- **Processing**: Hashing only (SHA-256 + ssdeep)
- **Storage**:
  - Original file → R2
  - Basic metadata → D1
- **Searchable**: Filename, hashes

## Processing Steps

### Step 1: File Validation
- Max size: 100 MB
- Must not be empty (0 bytes)

### Step 2: Hashing
- **SHA-256**: Cryptographic hash for exact file matching
- **ssdeep**: Fuzzy hash for similarity detection

### Step 3: File Type Detection
- Uses Google Magika ML model for accurate type detection
- Handles files with missing or incorrect extensions

### Step 4: Content Processing
- **Images**: Captioning with Moondream
- **PDFs**: Text extraction → OCR (if needed) → Summarization
- **Other**: No content processing

### Step 5: Storage
- **R2**: File stored at `files/{fileId}/{filename}`
- **D1**: Metadata stored in `files` table with JSON metadata column

## Technology Stack

### Cloudflare Services
- **Workers**: Serverless compute
- **Workflows**: Multi-step processing with retries
- **R2**: Object storage
- **D1**: SQL database
- **Workers AI**:
  - Llama 3.2 Vision (11B) - Image OCR
  - Llama 3.1 Fast (8B) - Text summarization

### External Services
- **Moondream API**: Image captioning (optional)

### Libraries
- **Magika**: File type detection
- **ssdeep.js**: Fuzzy hashing
- **unpdf**: PDF text extraction
- **pdf-lib**: PDF image extraction

## Search Capabilities

Search queries match across:
1. Filename (partial match)
2. SHA-256 hash (exact match)
3. ssdeep hash (exact match)
4. Image captions (partial match)
5. PDF summaries (partial match)
6. PDF raw text/OCR (partial match)

Results limited to 50 files, sorted by newest first.

## Admin Features

The admin dashboard (`/admin`) provides:
- View all uploaded files
- Multi-select file deletion
- Delete from both R2 and D1
- Detailed error messages for failed operations
- Statistics (total files, total storage)

## Cost Monitoring

**OCR Costs**: When OCR is triggered:
- 1 vision model inference per embedded image in PDF
- For a 100-page scanned PDF with 1 image per page = 100 inferences
- Monitor costs in Cloudflare dashboard under Workers AI usage
