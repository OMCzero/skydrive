from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

# Import our modules
from basic_metadata import extract_basic_metadata
from system_check import get_system_status
from search_service import search_service

# Define uploads directory
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="File Management System")

# Add CORS middleware to allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the index.html page"""
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/status")
async def check_status():
    """
    Get system status for all components.
    
    This endpoint checks the availability of:
    - TLSH fuzzy hashing
    - Magika file type detection
    - Whisper transcription
    - Ollama LLM
    - ExifTool metadata extraction
    - Redis (task queue backend)
    - Celery (task queue workers)
    - Meilisearch (search engine)
    """
    status = get_system_status()
    
    # Add Meilisearch status
    search_stats = search_service.get_stats()
    status["meilisearch"] = {
        "status": "available" if search_stats.get("available", False) else "unavailable",
        "message": f"Meilisearch is available with {search_stats.get('documents_count', 0)} documents" 
                  if search_stats.get("available", False) 
                  else "Meilisearch is not available"
    }
    
    # Check queue system and update total count of components
    queue_components = ["redis", "celery"]
    for component in queue_components:
        if component in status:
            if status[component]["status"] == "available":
                status["available_count"] += 1
            status["total_components"] += 1
    
    return JSONResponse(content=status)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), allow_duplicates: bool = Query(False, description="Allow duplicate file uploads")):
    """
    Upload a file for asynchronous processing.
    
    This endpoint accepts any file type and initiates a background task that will:
    - Extract basic metadata (hashes, EXIF, etc.)
    - Apply type-specific enrichment
    - Extract and process text when applicable
    - Index the file in the search system
    
    By default, it prevents duplicate uploads by comparing file hashes.
    Use the allow_duplicates parameter to override this behavior.
    
    Returns a task ID that can be used to check processing status or
    a duplicate notification if the file was already uploaded.
    """
    from task_queue import process_file
    from basic_metadata import calculate_file_hashes
    
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Get file extension from original filename
    ext = os.path.splitext(file.filename)[1]
    
    # Create a temporary file with the correct extension for processing
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        # Copy uploaded file to temporary file
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    # Check for duplicates unless explicitly allowed
    if not allow_duplicates and search_service.available:
        # Calculate file hashes
        try:
            # Calculate minimal hashes needed for duplicate detection
            hashes = calculate_file_hashes(temp_file_path)
            
            # Query Meilisearch for files with matching MD5 hash
            search_results = search_service.search("", {"hashes.md5": hashes["md5"]}, limit=1)
            
            # If we found a match, check SHA256 for confirmation
            if search_results.get("hits") and len(search_results["hits"]) > 0:
                matched_doc = search_results["hits"][0]
                matched_hashes = matched_doc.get("hashes", {})
                
                # Confirm with SHA256 (stronger hash) to avoid false positives
                if matched_hashes.get("sha256") == hashes["sha256"]:
                    # This is a duplicate - clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    
                    return JSONResponse(
                        status_code=409,  # Conflict
                        content={
                            "status": "DUPLICATE",
                            "message": f"Duplicate file detected: {file.filename}",
                            "filename": file.filename,
                            "duplicate_id": matched_doc.get("id"),
                            "duplicate_filename": matched_doc.get("file_info", {}).get("filename")
                        }
                    )
        except Exception as e:
            # Log the error but continue with upload if duplicate check fails
            print(f"Error checking for duplicates: {str(e)}")
    
    # Start background processing task
    task = process_file.delay(
        temp_file_path=temp_file_path,
        original_filename=file.filename or "unknown"
    )
    
    # Return task ID so client can check status
    return JSONResponse(
        content={
            "task_id": task.id,
            "status": "PENDING",
            "message": f"Processing started for file: {file.filename}",
            "filename": file.filename
        }
    )

@app.get("/search")
async def search_files(
    q: str = Query("", description="Search query"),
    # ID search (for direct document lookup)
    filter_id: Optional[str] = Query(None, description="Filter by document ID"),
    # Basic file filters
    mime_type: Optional[str] = Query(None, description="Filter by MIME type"),
    extension: Optional[str] = Query(None, description="Filter by file extension"),
    size: Optional[str] = Query(None, description="Filter by size category (tiny, small, medium, large, huge)"),
    # Image-specific filters
    exif_Make: Optional[str] = Query(None, description="Filter by camera make", alias="exif.Make"),
    enrichment_color_mode: Optional[str] = Query(None, description="Filter by color mode", alias="enrichment.color_mode"),
    enrichment_animation_is_animated: Optional[str] = Query(None, description="Filter by animation status", alias="enrichment.animation.is_animated"),
    # Text-specific filters
    text_extraction_metadata_encoding: Optional[str] = Query(None, description="Filter by text encoding", alias="text_extraction.metadata.encoding"),
    text_extraction_metadata_format: Optional[str] = Query(None, description="Filter by text format", alias="text_extraction.metadata.format"),
    # Hash-based filters for duplicate detection
    hash_md5: Optional[str] = Query(None, description="Filter by MD5 hash"),
    hash_sha256: Optional[str] = Query(None, description="Filter by SHA256 hash"),
    # Pagination and limits
    limit: int = Query(20, description="Maximum number of results to return")
):
    """
    Search for files in the system using Meilisearch.
    
    This endpoint allows searching across all indexed file metadata using a query string.
    Filters can be applied to narrow results by various attributes including file type,
    extension, size, camera make, color mode, animation status, text encoding, and format.
    
    Returns a list of matching files with their metadata.
    """
    if not search_service.available:
        return JSONResponse(content={
            "hits": [],
            "error": "Search service is not available"
        })
    
    # Build filters
    filters = {}
    
    # Document ID filter for direct lookup
    if filter_id:
        filters["id"] = filter_id
    
    # Hash-based filters for duplicate detection
    if hash_md5:
        filters["hashes.md5"] = hash_md5
    if hash_sha256:
        filters["hashes.sha256"] = hash_sha256
    
    # Basic file filters
    if mime_type:
        filters["file_info.mime_type"] = mime_type
    if extension:
        filters["file_info.extension"] = extension
    if size:
        filters["file_info.size_category"] = size
    
    # Image-specific filters
    if exif_Make:
        filters["exif.Make"] = exif_Make
    if enrichment_color_mode:
        filters["enrichment.color_mode"] = enrichment_color_mode
    if enrichment_animation_is_animated:
        filters["enrichment.animation.is_animated"] = enrichment_animation_is_animated
    
    # Text-specific filters
    if text_extraction_metadata_encoding:
        filters["text_extraction.metadata.encoding"] = text_extraction_metadata_encoding
    if text_extraction_metadata_format:
        filters["text_extraction.metadata.format"] = text_extraction_metadata_format
    
    # Execute search
    results = search_service.search(q, filters, limit)
    
    return JSONResponse(content=results)

@app.delete("/search/{doc_id}")
async def delete_from_search(doc_id: str):
    """
    Delete a document from the search index.
    
    Args:
        doc_id: The ID of the document to delete
    """
    if not search_service.available:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    success = search_service.delete_document(doc_id)
    
    if success:
        return JSONResponse(content={"status": "success", "message": f"Document {doc_id} deleted"})
    else:
        raise HTTPException(status_code=500, detail=f"Failed to delete document {doc_id}")

@app.get("/search/stats")
async def get_search_stats():
    """
    Get statistics about the search index.
    """
    stats = search_service.get_stats()
    
    return JSONResponse(content=stats)
    
@app.get("/task/{task_id}")
async def get_task_status(task_id: str, full_metadata: bool = Query(True, description="Include full metadata")):
    """
    Get the status of a task.
    
    Args:
        task_id: The ID of the task to check
        full_metadata: Whether to include full metadata for SUCCESS tasks
        
    Returns:
        The task status information
    """
    from task_queue import get_task_status, update_task_status
    from celery.result import AsyncResult
    
    # Get the task status from Redis first, with optional full metadata
    status_info = get_task_status(task_id, include_full_metadata=full_metadata)
    
    if status_info:
        return JSONResponse(content=status_info)
    
    # If not found in Redis, check with Celery directly
    task_result = AsyncResult(task_id)
    
    # Map Celery states to our status format
    status_mapping = {
        "PENDING": "PENDING",
        "STARTED": "PROCESSING",
        "SUCCESS": "SUCCESS",
        "FAILURE": "ERROR",
        "REVOKED": "CANCELLED",
        "RETRY": "PROCESSING"
    }
    
    # Create status object with mapped status
    mapped_status = status_mapping.get(task_result.state, "UNKNOWN")
    
    # Create a properly formatted task status object
    status = {
        "task_id": task_id,
        "status": mapped_status,
        "message": f"Task is {task_result.state}",
        "updated_at": datetime.utcnow().isoformat(),
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Get additional task info if available
    task_info = None
    if hasattr(task_result, 'info') and task_result.info:
        task_info = task_result.info
        
        # If task info has a filename, include it
        if isinstance(task_info, dict) and 'filename' in task_info:
            status['filename'] = task_info['filename']
    
    # If the task was successful, include the result
    if task_result.successful():
        status["metadata"] = task_result.result
    
    # If the task failed, include the error
    if task_result.failed():
        status["error"] = str(task_result.result)
    
    # Update Redis storage with this information
    # so it will appear in the tasks list
    update_task_status(
        task_id=task_id,
        status=mapped_status,
        message=status["message"],
        metadata=status.get("metadata"),
        filename=status.get("filename")
    )
    
    return JSONResponse(content=status)

@app.get("/tasks")
async def list_tasks(
    limit: int = Query(10, description="Maximum number of tasks to return"),
    status: Optional[str] = Query(None, description="Filter by status (PENDING, PROCESSING, SUCCESS, ERROR)"),
    refresh: bool = Query(False, description="Force refresh task status from Celery"),
    page: int = Query(1, description="Page number for pagination"),
    include_metadata: bool = Query(False, description="Include full metadata in results")
):
    """
    List recent tasks with pagination.
    
    Args:
        limit: Maximum number of tasks to return
        status: Filter by status
        refresh: Force refresh task status from Celery
        page: Page number for pagination
        include_metadata: Include full metadata in results
        
    Returns:
        List of recent tasks with pagination info
    """
    from task_queue import list_tasks as get_task_list, celery_app, update_task_status
    from celery.result import AsyncResult
    
    # Calculate offset for pagination
    offset = (page - 1) * limit
    
    # If refresh is requested, try to query active tasks from Celery
    if refresh:
        try:
            # Try to get Celery control interface to inspect workers
            inspector = celery_app.control.inspect()
            
            # Get all active, reserved, and scheduled tasks
            active_tasks = inspector.active() or {}
            reserved_tasks = inspector.reserved() or {}
            
            # Status mapping from Celery states to our format
            status_mapping = {
                "PENDING": "PENDING",
                "STARTED": "PROCESSING",
                "SUCCESS": "SUCCESS",
                "FAILURE": "ERROR",
                "REVOKED": "CANCELLED",
                "RETRY": "PROCESSING"
            }
            
            # Process active tasks - these are currently running
            all_active_task_ids = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    task_id = task.get('id')
                    if task_id:
                        all_active_task_ids.append(task_id)
                        update_task_status(
                            task_id=task_id,
                            status="PROCESSING",
                            message=f"Task is currently running on worker {worker}",
                            metadata={"task_name": task.get('name', 'unknown')}
                        )
                            
            # Process reserved tasks - these are waiting to be processed
            for worker, tasks in reserved_tasks.items():
                for task in tasks:
                    task_id = task.get('id')
                    if task_id and task_id not in all_active_task_ids:
                        all_active_task_ids.append(task_id)
                        update_task_status(
                            task_id=task_id,
                            status="PENDING",
                            message=f"Task is reserved by worker {worker}",
                            metadata={"task_name": task.get('name', 'unknown')}
                        )
        except Exception as e:
            # Log the error but continue with what we have
            print(f"Error refreshing tasks from Celery: {str(e)}")
    
    # Get tasks from Redis with pagination
    tasks, total_count = get_task_list(
        limit=limit, 
        status_filter=status, 
        offset=offset,
        include_metadata=include_metadata
    )
    
    # Calculate pagination info
    total_pages = max(1, (total_count + limit - 1) // limit)
    
    return JSONResponse(content={
        "tasks": tasks,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total_count,
            "pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    })

@app.get("/download/{doc_id}")
async def download_file(doc_id: str):
    """
    Download a file by its document ID.
    
    Args:
        doc_id: The search document ID of the file to download
    
    Returns:
        The file as a downloadable attachment
    """
    if not search_service.available:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        # Search for the document in the index
        # Using an empty query with a filter on the ID
        results = search_service.search("", {"id": doc_id}, limit=1)
        
        if not results or not results.get("hits") or len(results["hits"]) == 0:
            # If the first approach fails, try getting all documents and filter in memory
            # This is a fallback in case the id field is not properly configured as filterable
            print(f"Could not find document with ID {doc_id} using filter, trying fallback method")
            all_results = search_service.search("", {}, limit=100)
            
            # Find the document with matching ID
            matching_docs = [doc for doc in all_results.get("hits", []) if doc.get("id") == doc_id]
            
            if not matching_docs:
                raise HTTPException(status_code=404, detail=f"File with ID {doc_id} not found")
            
            document = matching_docs[0]
        else:
            document = results["hits"][0]
        
        # Get the stored path from the document
        if not document.get("file_info") or not document["file_info"].get("stored_path"):
            raise HTTPException(status_code=404, detail="File path not found in metadata")
        
        stored_path = document["file_info"]["stored_path"]
        
        # Check if the file exists
        if not os.path.exists(stored_path):
            raise HTTPException(status_code=404, detail=f"File not found on disk at {stored_path}")
        
        # Get the original filename
        filename = document["file_info"].get("filename", "download")
        
        # Return the file as a downloadable attachment
        return FileResponse(
            path=stored_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error downloading file {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)