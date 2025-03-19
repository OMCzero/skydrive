import os
import hashlib
import json
from typing import Dict, Any, List, Optional
import subprocess
import tempfile
from datetime import datetime
import mimetypes

# For perceptual hashing (primarily for images)
try:
    import imagehash
    from PIL import Image
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

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
    from magika import Magika
    import pathlib
    MAGIKA_AVAILABLE = True
    try:
        # Initialize Magika with default model
        magika = Magika()
        # Test that Magika works by analyzing a simple string
        test_result = magika.identify_bytes(b"test")
        # Access properties to verify correct API
        _ = test_result.output.mime_type
        _ = test_result.output.ct_label  # Correct field name is ct_label, not label
        _ = test_result.output.score     # Score is part of output, not at top level
        print("Magika initialized successfully")
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
            print(f"Error generating TLSH hash: {str(e)}")
            result["tlsh"] = None
    
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
        return None

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
                if not hasattr(result, "output") or not hasattr(result.output, "ct_label"):
                    raise AttributeError("Invalid result structure")
                    
            except Exception as path_error:
                # If path method fails, try with file content directly
                print(f"Magika path analysis failed: {str(path_error)}, trying content analysis")
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        result = magika.identify_bytes(content)
                        
                        # Validate that we got a proper result
                        if not hasattr(result, "output") or not hasattr(result.output, "ct_label"):
                            raise AttributeError("Invalid result structure")
                except Exception as bytes_error:
                    raise Exception(f"Both path and bytes analysis failed: {str(path_error)}, {str(bytes_error)}")
            
            # Add Magika results - using the correct output structure
            file_info["magika"] = {
                "mime_type": result.output.mime_type,
                "label": str(result.output.ct_label),
                "confidence": float(result.output.score),
                "group": result.output.group or None,
                "magic": result.output.magic or None,
                "description": result.output.description or None,
            }
            
            # If standard MIME type detection failed, use Magika's result
            if not mime_type and result.output.mime_type:
                file_info["mime_type"] = result.output.mime_type
                
        except Exception as e:
            print(f"Magika analysis error: {str(e)}")
            file_info["magika"] = {"error": str(e)}
    
    return file_info

def extract_basic_metadata(file_path: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract basic metadata from a file including hashes and EXIF data.
    
    Args:
        file_path: Path to the file
        original_filename: Original filename if different from file_path
        
    Returns:
        Dict containing all metadata
    """
    # Get basic file info first (includes MIME type detection)
    file_info = get_basic_file_info(file_path, original_filename)
    
    metadata = {
        "file_info": file_info,
        "hashes": calculate_file_hashes(file_path),
    }
    
    # Add perceptual hash for images
    perceptual_hash = calculate_perceptual_hash(file_path)
    if perceptual_hash:
        metadata["perceptual_hashes"] = perceptual_hash
    
    # Add EXIF data if available
    exif_data = extract_exif_data(file_path)
    if exif_data:
        metadata["exif"] = exif_data
    
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
                # Log the error but continue processing
                print(f"Error in enrichment: {str(e)}")
        
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
                    # Log the error but continue processing
                    error_msg = f"Error in text extraction: {str(e)}"
                    print(error_msg)
                    metadata["text_extraction"] = {"error": error_msg}
    
    return metadata