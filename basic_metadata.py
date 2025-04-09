import os
import hashlib
import json
from typing import Dict, Any, List, Optional
import subprocess
import tempfile
from datetime import datetime
import mimetypes
import c2pa
import pathlib
import uuid

# For perceptual hashing (primarily for images)
try:
    import imagehash
    from PIL import Image, ImageOps
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    Image = None
    ImageOps = None

# For fuzzy hashing using TLSH (Trend Micro Locality Sensitive Hash)
try:
    import tlsh
    # Test if TLSH works
    test_hash = tlsh.hash(b"test" * 50)  # TLSH needs at least 50 bytes
    TLSH_AVAILABLE = True
    print("TLSH fuzzy hashing is available")
except (ImportError, RuntimeError, Exception) as e:
    TLSH_AVAILABLE = False
    print(f"TLSH fuzzy hashing not available: {str(e)}")

# Import Magika for advanced file type detection
try:
    from magika import Magika, ContentTypeLabel
    MAGIKA_AVAILABLE = True
    try:
        # Initialize Magika with default model
        magika = Magika()
        # Test that Magika works by analyzing a simple string
        test_result = magika.identify_bytes(b"test")
        # Access properties to verify correct API
        if test_result.ok:
            _ = test_result.output.mime_type
            _ = test_result.output.label  # In v0.6.1, ct_label is renamed to label
            _ = test_result.score         # In v0.6.1, score is at top level, not in output
            print("Magika initialized successfully")
        else:
            print(f"Magika test failed: {test_result.status}")
            MAGIKA_AVAILABLE = False
    except Exception as init_error:
        print(f"Magika initialization failed: {str(init_error)}")
        MAGIKA_AVAILABLE = False
except ImportError:
    MAGIKA_AVAILABLE = False
    print("Magika not available - advanced file type detection disabled")

# Import type-specific enrichment functions
try:
    from enrichment import get_enrichment_function, supports_text_extraction, get_text_extraction_function
    ENRICHMENT_AVAILABLE = True
except ImportError:
    ENRICHMENT_AVAILABLE = False
    get_enrichment_function = lambda mime_type: None
    supports_text_extraction = lambda mime_type: False
    get_text_extraction_function = lambda: None

# Thumbnail constants
THUMBNAIL_DIR = "uploads/thumbnails"
THUMBNAIL_SIZE = (150, 150) # Width, Height
THUMBNAIL_FORMAT = "JPEG"
os.makedirs(THUMBNAIL_DIR, exist_ok=True)

def _is_ffmpeg_available() -> bool:
    """Check if ffmpeg command is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

FFMPEG_AVAILABLE = _is_ffmpeg_available()
print(f"FFmpeg available: {FFMPEG_AVAILABLE}")

def generate_thumbnail(file_path: str, unique_id: str, mime_type: str) -> Optional[Dict[str, Any]]:
    """
    Generate a thumbnail for image or video files.

    Args:
        file_path: Path to the original file.
        unique_id: Unique ID associated with the file.
        mime_type: The MIME type of the file.

    Returns:
        Dict containing thumbnail info (path, dimensions) or None if unsupported/error.
    """
    thumbnail_filename = f"{unique_id}_thumb.jpg"
    thumbnail_path = os.path.join(THUMBNAIL_DIR, thumbnail_filename)
    thumbnail_rel_path = os.path.join("thumbnails", thumbnail_filename) # Relative path for storage

    try:
        # --- Image Thumbnail Generation ---
        if mime_type.startswith("image/") and IMAGEHASH_AVAILABLE and Image and ImageOps:
            try:
                with Image.open(file_path) as img:
                    # Ensure image is in RGB mode for JPEG saving
                    if img.mode not in ('RGB', 'L'): # L is grayscale
                        img = img.convert('RGB')

                    # Create thumbnail maintaining aspect ratio
                    img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

                    # Optional: Create a square canvas and paste the thumbnail in the center
                    # square_thumb = ImageOps.pad(img, THUMBNAIL_SIZE, color=(255, 255, 255)) # White background
                    # square_thumb.save(thumbnail_path, THUMBNAIL_FORMAT)
                    # thumb_w, thumb_h = square_thumb.size

                    # Save the potentially non-square thumbnail directly
                    img.save(thumbnail_path, THUMBNAIL_FORMAT)
                    thumb_w, thumb_h = img.size

                    return {
                        "thumbnail_path": thumbnail_rel_path,
                        "width": thumb_w,
                        "height": thumb_h,
                        "format": THUMBNAIL_FORMAT
                    }
            except Exception as img_err:
                print(f"Error generating image thumbnail for {file_path}: {img_err}")
                return None # Fall through or return error?

        # --- Video Thumbnail Generation ---
        elif mime_type.startswith("video/") and FFMPEG_AVAILABLE:
            try:
                # Extract frame at 1 second, scale to fit width 150, maintain aspect ratio
                cmd = [
                    "ffmpeg",
                    "-i", file_path,      # Input file
                    "-ss", "00:00:01.000", # Seek to 1 second
                    "-vframes", "1",      # Extract one frame
                    "-vf", f"scale={THUMBNAIL_SIZE[0]}:-1", # Scale width to 150px, auto height
                    "-q:v", "3",          # Quality (2-5 is good)
                    thumbnail_path        # Output path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # Check if thumbnail was created
                if os.path.exists(thumbnail_path):
                     # We don't easily know the dimensions without reading the thumb back
                    return {
                        "thumbnail_path": thumbnail_rel_path,
                        "width": None, # Could read image back, but maybe not worth it
                        "height": None,
                        "format": THUMBNAIL_FORMAT
                    }
                else:
                     print(f"FFmpeg ran but thumbnail not found for {file_path}.")
                     print(f"FFmpeg stderr: {result.stderr}")
                     return None
            except subprocess.CalledProcessError as vid_err:
                print(f"Error generating video thumbnail for {file_path}: {vid_err.stderr}")
                return None
            except Exception as vid_err_generic:
                print(f"Generic error generating video thumbnail for {file_path}: {vid_err_generic}")
                return None

        # --- Unsupported Type ---
        else:
            return None # Not an image or video, or dependencies missing

    except Exception as e:
        print(f"Failed to generate thumbnail for {file_path}: {e}")
        return None

def calculate_file_hashes(file_path: str) -> Dict[str, str]:
    """
    Calculate multiple hash types for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict containing different hash types (md5, sha1, sha256, ssdeep)
    """
    # Initialize hash objects
    md5_hash = hashlib.md5()
    sha1_hash = hashlib.sha1()
    sha256_hash = hashlib.sha256()
    
    # Read and update hash in chunks for memory efficiency
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
            sha1_hash.update(byte_block)
            sha256_hash.update(byte_block)
    
    result = {
        "md5": md5_hash.hexdigest(),
        "sha1": sha1_hash.hexdigest(),
        "sha256": sha256_hash.hexdigest()
    }
    
    # Add TLSH fuzzy hash if available
    if TLSH_AVAILABLE:
        try:
            with open(file_path, 'rb') as f:
                # Read the file content
                data = f.read()
                # TLSH requires at least 50 bytes of data
                if len(data) >= 50:
                    # Generate the TLSH hash
                    hash_value = tlsh.hash(data)
                    result["tlsh"] = hash_value
                else:
                    # File is too small for TLSH
                    result["tlsh"] = None
        except Exception as e:
            result["tlsh"] = {"error": str(e)}
    
    return result

def calculate_perceptual_hash(file_path: str) -> Optional[Dict[str, str]]:
    """
    Calculate perceptual hashes for image files.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dict containing different perceptual hash types or None if not an image
    """
    if not IMAGEHASH_AVAILABLE:
        return None
    
    try:
        # Try to open as image
        img = Image.open(file_path)
        
        # Calculate different types of perceptual hashes
        phash = str(imagehash.phash(img))
        dhash = str(imagehash.dhash(img))
        ahash = str(imagehash.average_hash(img))
        whash = str(imagehash.whash(img))
        
        return {
            "phash": phash,  # Perceptual hash
            "dhash": dhash,  # Difference hash
            "ahash": ahash,  # Average hash
            "whash": whash,  # Wavelet hash
        }
    except Exception:
        # Not an image or other error
        return None

def extract_exif_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract EXIF metadata using exiftool if available.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict containing EXIF metadata or None if exiftool not available
    """
    try:
        # Run exiftool and capture JSON output
        result = subprocess.run(
            ["exiftool", "-j", "-n", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse JSON output (exiftool returns a list with one item)
        exif_data = json.loads(result.stdout)
        if exif_data and isinstance(exif_data, list):
            return exif_data[0]
        return None
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        # exiftool not available or other error
        return {"error": "exiftool not available"}

def get_basic_file_info(file_path: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Get basic file information.
    
    Args:
        file_path: Path to the file
        original_filename: Original filename if different from file_path
        
    Returns:
        Dict containing basic file information
    """
    filename = original_filename or os.path.basename(file_path)
    file_stats = os.stat(file_path)
    
    # Guess the MIME type using standard library
    mime_type, _ = mimetypes.guess_type(filename)
    
    # Create basic file info
    file_info = {
        "filename": filename,
        "size_bytes": file_stats.st_size,
        "mime_type": mime_type or "application/octet-stream",
        "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
        "extension": os.path.splitext(filename)[1].lower() if '.' in filename else ""
    }
    
    # Use Magika for advanced file type detection
    if MAGIKA_AVAILABLE:
        try:
            # First try with Path object
            try:
                # Convert string path to Path object
                path_obj = pathlib.Path(file_path)
                if not path_obj.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                    
                # Analyze the file using Magika path API
                result = magika.identify_path(path_obj)
                
                # Validate that we got a proper result
                if not result.ok:
                    raise AttributeError(f"Magika analysis failed: {result.status}")
                
                if not hasattr(result, "output") or not hasattr(result.output, "label"):
                    raise AttributeError("Invalid result structure")
                    
            except Exception as path_error:
                # If path method fails, try with file content directly
                print(f"Magika path analysis failed: {str(path_error)}, trying content analysis")
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        result = magika.identify_bytes(content)
                        
                        # Validate that we got a proper result
                        if not result.ok:
                            raise AttributeError(f"Magika analysis failed: {result.status}")
                            
                        if not hasattr(result, "output") or not hasattr(result.output, "label"):
                            raise AttributeError("Invalid result structure")
                except Exception as bytes_error:
                    raise Exception(f"Both path and bytes analysis failed: {str(path_error)}, {str(bytes_error)}")
            
            # Add Magika results - using the updated structure for v0.6.1
            file_info["magika"] = {
                "mime_type": result.output.mime_type,
                "label": str(result.output.label),
                "confidence": float(result.score),  # score is now at top level
                "group": result.output.group or None,
                # 'magic' field has been removed in v0.6.1
                "description": result.output.description or None,
                # Add new fields available in v0.6.1
                "is_text": result.output.is_text if hasattr(result.output, "is_text") else None,
                "extensions": result.output.extensions if hasattr(result.output, "extensions") else None
            }
            
            # If standard MIME type detection failed, use Magika's result
            if not mime_type and result.output.mime_type:
                file_info["mime_type"] = result.output.mime_type

        except Exception as e:
            print(f"Magika analysis error: {str(e)}")
            file_info["magika"] = {"error": str(e)}
    try:
        reader = c2pa.Reader.from_file(file_path)
        # Parse the JSON string into a Python object
        file_info["c2pa"] = {"manifest": json.loads(reader.json())}
    except c2pa.Error.ManifestNotFound as e:
        file_info["c2pa"] = "No manifest found"
    except Exception as e:
        file_info["c2pa"] = {"error": str(e)}
                

    
    return file_info

def extract_basic_metadata(file_path: str, original_filename: Optional[str] = None, unique_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract basic metadata from a file including hashes, EXIF data, and thumbnail.

    Args:
        file_path: Path to the file
        original_filename: Original filename if different from file_path
        unique_id: The unique identifier assigned to this file storage.

    Returns:
        Dict containing all metadata
    """
    if unique_id is None:
        # This should ideally always be provided by the caller (process_file task)
        # If not, generate one, but this might cause issues if the caller expects consistency.
        print("Warning: unique_id not provided to extract_basic_metadata. Generating one.")
        unique_id = str(uuid.uuid4())


    # Get basic file info first (includes MIME type detection)
    file_info = get_basic_file_info(file_path, original_filename)
    # Add the unique_id here as it's fundamental file info now
    file_info["unique_id"] = unique_id

    metadata = {
        "id": unique_id, # Use the unique ID as the document ID for search
        "file_info": file_info,
        "hashes": calculate_file_hashes(file_path),
        "upload_date": datetime.utcnow().isoformat(), # Add upload timestamp
    }

    # Add perceptual hash for images
    perceptual_hash = calculate_perceptual_hash(file_path)
    if perceptual_hash:
        metadata["perceptual_hashes"] = perceptual_hash
    
    # Add EXIF data if available
    exif_data = extract_exif_data(file_path)
    if exif_data:
        metadata["exif"] = exif_data

    # Generate and add thumbnail info if applicable
    thumbnail_info = generate_thumbnail(file_path, unique_id, file_info.get("mime_type", ""))
    if thumbnail_info:
        metadata["thumbnail_info"] = thumbnail_info

    # Apply file type specific enrichment if available
    if ENRICHMENT_AVAILABLE:
        mime_type = file_info.get("mime_type", "")
        enrichment_func = get_enrichment_function(mime_type)
        
        if enrichment_func:
            try:
                enrichment_data = enrichment_func(file_path)
                if enrichment_data:
                    # Add the enrichment data under its own key
                    metadata["enrichment"] = enrichment_data
            except Exception as e:
                metadata["enrichment"] = {"error": str(e)}
        
        # Apply text extraction if supported for this file type
        if supports_text_extraction(mime_type, file_path):
            text_extraction_func = get_text_extraction_function()
            if text_extraction_func:
                try:
                    print(f"Attempting text extraction for file: {file_path} with mime type: {mime_type}")
                    # Pass the mime_type to the text extraction function
                    # LLM summarization is now always attempted in the extraction functions if text is substantial
                    text_data = text_extraction_func(file_path, mime_type=mime_type)
                    print(f"Text extraction result: {text_data.keys() if text_data else None}")
                    
                    # Always include text extraction results, even if there was an error
                    # (this helps debug what's happening)
                    metadata["text_extraction"] = text_data
                except Exception as e:
                    metadata["text_extraction"] = {"error": str(e)}
    
    return metadata