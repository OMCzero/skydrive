from celery import Celery
import os
import tempfile
import shutil
import uuid
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import redis

# Import our modules
from basic_metadata import extract_basic_metadata
from search_service import search_service

# Configure Redis connection
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
# Parse Redis URL to get information for creating the Redis connection
if redis_url.startswith('redis://'):
    redis_host = redis_url.split('redis://')[1].split(':')[0]
    redis_port = int(redis_url.split(':')[-1].split('/')[0])
    redis_db = int(redis_url.split('/')[-1])
else:
    # Default values if parsing fails
    redis_host = 'localhost'
    redis_port = 6379
    redis_db = 0

# Create Redis client for task status storage
redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

# Configure Celery
celery_app = Celery('file_system_tasks', broker=redis_url, backend=redis_url)

# Configure Celery with periodic tasks
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour time limit for tasks
    worker_prefetch_multiplier=1,  # Process one task at a time,
    
    # Configure periodic tasks (beat scheduler)
    beat_schedule={
        'cleanup-old-tasks': {
            'task': 'task_queue.periodic_task_cleanup',
            'schedule': 3600.0,  # Run every hour (3600 seconds)
            'args': (7,)  # Remove tasks older than 7 days
        }
    }
)

# Define uploads directory
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Redis keys and configuration
TASK_STATUS_PREFIX = "task:status:"
TASK_LIST_KEY = "task:list"
TASK_TTL = 60 * 60 * 24 * 7  # 7 days TTL for task records
TASK_PAGE_SIZE = 100  # Number of tasks to load in each page for pagination

@celery_app.task
def ping():
    """
    Simple task to check if workers are running.
    Returns "pong" if successful.
    """
    return "pong"

@celery_app.task(bind=True)
def process_file(self, temp_file_path: str, original_filename: str, 
                 tags: Optional[List[str]] = None, 
                 user_login: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a file asynchronously.
    
    Args:
        temp_file_path: Path to the temporary file
        original_filename: Original filename of the uploaded file
        tags: Optional list of user-provided tags
        user_login: Optional user login identifier from Tailscale
        
    Returns:
        Dict containing the file metadata
    """
    task_id = self.request.id
    tags_list = tags if tags is not None else []
    
    # Initial status update
    update_task_status(
        task_id=task_id, 
        status="PROCESSING", 
        message=f"Starting processing for: {original_filename}",
        filename=original_filename
    )
    
    try:
        unique_id = str(uuid.uuid4())
        
        # --- Step: Metadata Extraction (includes enrichment, thumbnail, text) ---
        # This function now handles its own internal status updates
        metadata = extract_basic_metadata(
            file_path=temp_file_path,
            original_filename=original_filename,
            unique_id=unique_id,
            task_id=task_id, # Pass task ID
            update_status_func=update_task_status # Pass the update function
        )
        
        # --- Status Update: Storing File ---
        # Keep this update after metadata extraction is fully complete
        update_task_status(task_id, "PROCESSING", "Storing file and metadata...", filename=original_filename)

        # Add user-provided tags to the metadata
        metadata["tags"] = tags_list
        
        # Add user login to metadata if provided
        if user_login:
            metadata["user_login"] = user_login
        
        # Get file extension from original filename
        ext = os.path.splitext(original_filename)[1]
        
        # Use the pre-generated unique_id for the filename
        safe_filename = f"{unique_id}{ext}"
        stored_path = os.path.join(UPLOADS_DIR, safe_filename)
        
        # Copy the processed file to the permanent storage location
        shutil.copy2(temp_file_path, stored_path)
        
        # Store the file path in the metadata
        if "file_info" not in metadata:
            metadata["file_info"] = {}
        metadata["file_info"]["stored_path"] = stored_path
        # unique_id is already in file_info from extract_basic_metadata

        # Save metadata to a JSON file (optional step, perhaps remove later)
        metadata_file_path = os.path.join(UPLOADS_DIR, f"{unique_id}.json")
        with open(metadata_file_path, "w") as f:
            json.dump(metadata, f)
        
        # --- Step: Indexing ---
        search_id = None
        if search_service.available:
            # metadata already contains id, file_info, hashes, tags, user_login, exif, thumbnail_info, enrichment, etc.
            # The 'id' field in metadata should match unique_id and be used by Meilisearch
            update_task_status(task_id, "PROCESSING", "Indexing file in search engine...", filename=original_filename)
            search_id = search_service.add_document(metadata)
            if search_id:
                metadata["search_id"] = search_id # Keep search_id for reference, though id should be the main key
                metadata["searchable"] = True
        
        # --- Step: Complete ---
        # Use original_filename in the final success message
        update_task_status(task_id, "SUCCESS", f"File '{original_filename}' processed successfully", metadata=metadata, filename=original_filename)
        return metadata
    
    except Exception as e:
        # Handle any errors during processing
        error_msg = f"Error processing file '{original_filename}': {str(e)}"
        update_task_status(task_id, "ERROR", error_msg, filename=original_filename)
        # Log traceback for detailed debugging
        import traceback
        print(f"Task {task_id} failed for {original_filename}:")
        traceback.print_exc()
        raise # Re-raise the exception for Celery to mark as FAILURE
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def update_task_status(task_id: str, status: str, message: str, metadata: Optional[Dict[str, Any]] = None, 
                    filename: Optional[str] = None) -> None:
    """
    Update the status of a task in Redis.
    
    Args:
        task_id: The Celery task ID
        status: The status (PENDING, PROCESSING, SUCCESS, ERROR)
        message: Status message
        metadata: Optional metadata to store with the status
        filename: Optional filename for better task identification
    """
    try:
        # Get the current time in ISO format
        now = datetime.utcnow().isoformat()
        
        # Fetch existing data first to preserve created_at and filename if not provided now
        existing_data_json = redis_client.get(f"{TASK_STATUS_PREFIX}{task_id}")
        existing_data = {}
        created_at = now # Default created_at
        existing_filename = None # Default filename
        
        if existing_data_json:
            try:
                existing_data = json.loads(existing_data_json)
                created_at = existing_data.get("created_at", now)
                existing_filename = existing_data.get("filename")
            except (json.JSONDecodeError, TypeError):
                # If we can't decode the existing data, just use current time
                pass
        
        # Use provided filename, fallback to existing, else None
        current_filename = filename if filename is not None else existing_filename

        # Create task status data
        task_data = {
            "task_id": task_id,
            "status": status,
            "message": message,
            "updated_at": now,
            "created_at": created_at,  # Preserve original creation time
        }
        
        # Add filename if available
        if current_filename:
            task_data["filename"] = current_filename
            
        # Add metadata if provided (may be large, so we store it separately for completed tasks)
        if metadata:
            # For success tasks with large metadata, store a summary in the task status
            # and the full metadata in a separate key
            if status == "SUCCESS" and metadata.get("file_info"):
                # Extract just the essential info for the task list view
                file_info = metadata.get("file_info", {})
                thumbnail_info = metadata.get("thumbnail_info") # Get thumbnail info if present
                
                summary_metadata = {
                    "file_info": {
                        # Use current_filename if available, otherwise fallback to filename in file_info
                        "filename": current_filename or file_info.get("filename", "Unknown"), 
                        "mime_type": file_info.get("mime_type", "Unknown"),
                        "size_bytes": file_info.get("size_bytes", 0),
                        "extension": file_info.get("extension", ""),
                        "thumbnail_path": thumbnail_info.get("thumbnail_path") if thumbnail_info else None
                    },
                    "search_id": metadata.get("search_id"),
                    "searchable": metadata.get("searchable", False)
                }
                task_data["metadata"] = summary_metadata # Store summary in main task data
                
                # Store the full metadata in a separate key
                full_metadata_key = f"{TASK_STATUS_PREFIX}{task_id}:metadata"
                redis_client.set(full_metadata_key, json.dumps(metadata))
                redis_client.expire(full_metadata_key, TASK_TTL)
            else:
                # For other statuses or small metadata, include it directly
                # Ensure it's JSON serializable - converting potential complex objects might be needed here
                try:
                   # Attempt to serialize metadata directly
                   json.dumps(metadata) 
                   task_data["metadata"] = metadata
                except TypeError:
                   # Fallback if direct serialization fails (e.g., contains non-serializable objects)
                   # Could convert to string or a simplified representation
                   task_data["metadata"] = {"error": "Metadata not JSON serializable for direct storage"} 
                   print(f"Warning: Metadata for task {task_id} (status {status}) is not directly JSON serializable.")
        
        # Store task status in Redis with TTL
        redis_client.set(
            f"{TASK_STATUS_PREFIX}{task_id}", 
            json.dumps(task_data),
            ex=TASK_TTL
        )
        
        # Add to the sorted set of tasks by updated_at time
        # Using the timestamp as score for sorting (newest first)
        score = time.time()
        redis_client.zadd(TASK_LIST_KEY, {task_id: score})
        
        # Ensure the list itself has TTL to prevent infinite growth (might be redundant if items have TTL)
        # redis_client.expire(TASK_LIST_KEY, TASK_TTL) # Consider if this is needed
        
    except Exception as e:
        # If Redis operations fail, log the error but don't crash the task
        print(f"Error updating task status in Redis for task {task_id}: {str(e)}")

def get_task_status(task_id: str, include_full_metadata: bool = False) -> Optional[Dict[str, Any]]:
    """
    Get the status of a task from Redis.
    
    Args:
        task_id: The Celery task ID
        include_full_metadata: Whether to include full metadata (for SUCCESS tasks)
        
    Returns:
        Dict with task status information or None if not found
    """
    try:
        # Get task status from Redis
        data = redis_client.get(f"{TASK_STATUS_PREFIX}{task_id}")
        if not data:
            return None
        
        # Parse JSON data
        task_data = json.loads(data)
        
        # If task was successful and we want full metadata, get it from separate key
        if include_full_metadata and task_data.get("status") == "SUCCESS":
            full_metadata_key = f"{TASK_STATUS_PREFIX}{task_id}:metadata"
            full_metadata = redis_client.get(full_metadata_key)
            if full_metadata:
                try:
                    task_data["metadata"] = json.loads(full_metadata)
                except (json.JSONDecodeError, TypeError):
                    # If metadata is corrupted, keep what we have
                    pass
        
        return task_data
    
    except Exception as e:
        # If Redis operations fail, log the error and return None
        print(f"Error getting task status from Redis: {str(e)}")
        return None

def list_tasks(limit: int = 20, status_filter: Optional[str] = None, 
              offset: int = 0, include_metadata: bool = False) -> Tuple[List[Dict[str, Any]], int]:
    """
    List recent tasks from Redis with pagination.
    
    Args:
        limit: Maximum number of tasks to return
        status_filter: Filter by status (PENDING, PROCESSING, SUCCESS, ERROR)
        offset: Offset for pagination
        include_metadata: Whether to include metadata in results
        
    Returns:
        Tuple of (list of task dicts, total count)
    """
    try:
        # Get total count of tasks
        total_count = redis_client.zcard(TASK_LIST_KEY)
        
        # Get task IDs from sorted set (newest first)
        task_ids = redis_client.zrevrange(TASK_LIST_KEY, offset, offset + limit - 1)
        
        tasks = []
        for task_id in task_ids:
            # Get task data
            task_data = get_task_status(task_id, include_full_metadata=include_metadata)
            if task_data:
                # Filter by status if requested
                if status_filter and task_data.get("status") != status_filter:
                    continue
                
                # Add task to results
                tasks.append(task_data)
        
        # If status filter was applied, total might be different
        if status_filter:
            # For filtered results, the actual count needs to be calculated
            # This could be optimized in a production system
            filtered_count = len(tasks)
            if offset > 0 or len(tasks) == limit:
                # We need to get all tasks to count if we're paginating with a filter
                all_task_ids = redis_client.zrevrange(TASK_LIST_KEY, 0, -1)
                filtered_count = 0
                for task_id in all_task_ids:
                    task_data = get_task_status(task_id)
                    if task_data and task_data.get("status") == status_filter:
                        filtered_count += 1
            
            return tasks, filtered_count
        else:
            return tasks, total_count
    
    except Exception as e:
        # If Redis operations fail, log the error and return empty results
        print(f"Error listing tasks from Redis: {str(e)}")
        return [], 0

def cleanup_old_tasks(days: int = 7) -> int:
    """
    Cleanup tasks older than the specified number of days.
    This function is used internally by the periodic task.
    
    Args:
        days: Age in days of tasks to remove
        
    Returns:
        Number of tasks removed
    """
    try:
        # Calculate cutoff time (score) for old tasks
        cutoff_time = time.time() - (days * 86400)  # 86400 seconds in a day
        
        # Get task IDs older than cutoff time
        old_task_ids = redis_client.zrangebyscore(TASK_LIST_KEY, 0, cutoff_time)
        
        # Remove tasks from sorted set and delete keys
        if old_task_ids:
            # Remove tasks from sorted set
            redis_client.zremrangebyscore(TASK_LIST_KEY, 0, cutoff_time)
            
            # Delete task keys
            for task_id in old_task_ids:
                redis_client.delete(f"{TASK_STATUS_PREFIX}{task_id}")
                redis_client.delete(f"{TASK_STATUS_PREFIX}{task_id}:metadata")
            
            return len(old_task_ids)
        
        return 0
    
    except Exception as e:
        # If Redis operations fail, log the error
        print(f"Error cleaning up old tasks: {str(e)}")
        return 0

@celery_app.task
def periodic_task_cleanup(days: int = 7) -> Dict[str, Any]:
    """
    Periodic task that cleans up old tasks in Redis.
    
    Args:
        days: Age in days of tasks to remove
        
    Returns:
        Dict with cleanup results
    """
    start_time = time.time()
    
    try:
        # Call the cleanup function
        removed_count = cleanup_old_tasks(days)
        
        # Calculate execution time
        duration = time.time() - start_time
        
        return {
            "status": "success",
            "removed_count": removed_count,
            "days_threshold": days,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        # Log error details
        error_msg = f"Error in periodic task cleanup: {str(e)}"
        print(error_msg)
        
        return {
            "status": "error",
            "error": error_msg,
            "days_threshold": days,
            "duration_seconds": round(time.time() - start_time, 2),
            "timestamp": datetime.utcnow().isoformat()
        }