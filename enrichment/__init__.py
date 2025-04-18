"""
Enrichment modules package for the file management system.
Each module in this package handles specific file type enrichment.
"""
import os

# Import all enrichment modules
try:
    from .image_enrichment import enrich_image
except ImportError:
    enrich_image = None

try:
    from .text_extraction import extract_text
    TEXT_EXTRACTION_AVAILABLE = True
except ImportError:
    extract_text = None
    TEXT_EXTRACTION_AVAILABLE = False

# Map of mime type prefixes to enrichment functions
ENRICHMENT_FUNCTIONS = {
    "image/": enrich_image,
}

# Text extraction supported mime types
TEXT_EXTRACTION_MIME_TYPES = [
    "image/",           # Images (OCR)
    "application/pdf",  # PDF
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/msword",  # DOC
    "text/",            # Text files
    "audio/",           # Audio files (transcription)
    "video/",           # Video files (transcription)
    "application/ogg",  # OGG audio files
    "application/json", # JSON files
    "application/xml",  # XML files
    "application/yaml", # YAML files
    "application/x-yaml", # Alternative YAML MIME type
]

def get_enrichment_function(mime_type: str):
    """
    Get the appropriate enrichment function for a MIME type.
    
    Args:
        mime_type: The MIME type of the file
        
    Returns:
        Enrichment function or None if no specific function is available
    """
    if not mime_type:
        return None
        
    # Check for matching MIME type prefixes
    for prefix, func in ENRICHMENT_FUNCTIONS.items():
        if mime_type.startswith(prefix) and func is not None:
            return func
            
    return None

def supports_text_extraction(mime_type: str, file_path: str = None, file_info: dict = None) -> bool:
    """
    Check if a MIME type is supported for text extraction.
    
    Args:
        mime_type: The MIME type to check
        file_path: Optional file path to check extension if MIME type fails
        file_info: Optional file info dict containing Magika data
        
    Returns:
        Boolean indicating whether text extraction is supported
    """
    if not TEXT_EXTRACTION_AVAILABLE:
        return False
    
    # Check if Magika data indicates this is a text file
    if file_info and "magika" in file_info:
        magika_data = file_info["magika"]
        # If Magika explicitly says it's text, process it
        if isinstance(magika_data, dict) and magika_data.get("is_text", False) and magika_data.get("confidence", 0) > 0.7:
            return True
            
        # If Magika detected a supported mime type with high confidence, use that
        magika_mime = magika_data.get("mime_type") if isinstance(magika_data, dict) else None
        if magika_mime and magika_data.get("confidence", 0) > 0.7:
            for supported_type in TEXT_EXTRACTION_MIME_TYPES:
                if magika_mime.startswith(supported_type):
                    return True
                # Special case for XML-like formats
                elif supported_type == "application/xml" and (
                    magika_mime == "text/xml" or "xml" in magika_mime
                ):
                    return True
    
    # First check by MIME type    
    if mime_type:
        for supported_type in TEXT_EXTRACTION_MIME_TYPES:
            if mime_type.startswith(supported_type):
                return True
            # Special case for XML-like formats
            elif supported_type == "application/xml" and (
                mime_type == "text/xml" or "xml" in mime_type
            ):
                return True
    
    # If MIME type check fails and file_path is provided, check by extension
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',  # Images
                  '.pdf',  # PDF
                  '.docx', '.doc',  # Word
                  '.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm', '.rss',  # Text
                  '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.opus', '.wma',  # Audio
                  '.mp4', '.mkv', '.webm', '.avi', '.mov', '.wmv', '.flv', '.mpg', '.mpeg']:  # Video
            return True
            
    return False

def get_text_extraction_function():
    """
    Get the text extraction function if available.
    The returned function accepts parameters:
    - file_path: Path to the file
    - mime_type: Optional MIME type if already determined elsewhere
    
    Returns:
        Text extraction function or None if not available
    """
    return extract_text if TEXT_EXTRACTION_AVAILABLE else None