"""
Image enrichment module for extracting specific metadata from image files.
"""
from typing import Dict, Any, Optional
import io
import os
import re
import base64
import traceback

try:
    from PIL import Image, ImageStat
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

# LLM for image description with Ollama
try:
    import ollama
    from ollama import Client, ResponseError
    
    # Configure Ollama
    OLLAMA_AVAILABLE = False
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")  # Using llama3.2-vision as default for image description
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_VISION_AVAILABLE = False
    
    # Check for environment variables to control Ollama behavior
    ASSUME_OLLAMA_MODELS_EXIST = os.environ.get("ASSUME_OLLAMA_MODELS_EXIST", "").lower() in ["true", "1", "yes"]
    SKIP_OLLAMA_WARMUP = os.environ.get("SKIP_OLLAMA_WARMUP", "").lower() in ["true", "1", "yes"]
    
    # Create client with timeout
    try:
        ollama_client = Client(host=OLLAMA_HOST, timeout=3600.0)  # Increased timeout to 1 hour for vision tasks
        
        # In Docker, we can assume models exist on the host
        if ASSUME_OLLAMA_MODELS_EXIST:
            print(f"Assuming Ollama is available with vision model: {OLLAMA_MODEL} (ASSUME_OLLAMA_MODELS_EXIST=true)")
            OLLAMA_AVAILABLE = True
            OLLAMA_VISION_AVAILABLE = True
            
            if SKIP_OLLAMA_WARMUP:
                print(f"Skipping {OLLAMA_MODEL} vision model warmup as SKIP_OLLAMA_WARMUP is set")
        else:
            # Check if the model supports vision capabilities
            try:
                model_list = ollama_client.list()
                model_info = None
                
                # Try to find our model's info
                if hasattr(model_list, 'models'):
                    for m in model_list.models:
                        if m.model.split(':')[0] == OLLAMA_MODEL:
                            model_info = m
                            break
                
                # Check if the model supports vision
                if model_info and hasattr(model_info, 'details'):
                    print(f"Model info details: {dir(model_info.details)}")
                    if hasattr(model_info.details, 'multimodal') and model_info.details.multimodal:
                        OLLAMA_VISION_AVAILABLE = True
                        print(f"Ollama model {OLLAMA_MODEL} supports vision/multimodal capabilities")
                    else:
                        print(f"Ollama model {OLLAMA_MODEL} does not support vision/multimodal capabilities according to API")
                        # For llama3.2-vision, we'll force it to be available regardless of the API response
                        if "vision" in OLLAMA_MODEL.lower():
                            print(f"Model name contains 'vision', forcing vision capabilities to be enabled")
                            OLLAMA_VISION_AVAILABLE = True
                        else:
                            print("For image description, use a multimodal model like 'llama3.2-vision', 'llava' or 'bakllava'")
                else:
                    # Simplified check - if the model has "vision" in its name, assume it's vision-capable
                    print(f"Could not determine if {OLLAMA_MODEL} supports vision via API")
                    if "vision" in OLLAMA_MODEL.lower():
                        print(f"Model name contains 'vision', assuming it has vision capabilities")
                        OLLAMA_VISION_AVAILABLE = True
                    else:
                        print(f"Will attempt to use {OLLAMA_MODEL} for vision tasks anyway")
                        OLLAMA_VISION_AVAILABLE = True
                
                OLLAMA_AVAILABLE = True
                
                # Check if we should skip model warmup
                if SKIP_OLLAMA_WARMUP:
                    print(f"Skipping {OLLAMA_MODEL} vision model warmup as SKIP_OLLAMA_WARMUP is set")
            except Exception as e:
                print(f"Error checking Ollama model capabilities: {str(e)}")
                OLLAMA_AVAILABLE = False
    except Exception as client_error:
        print(f"Error creating Ollama client: {str(client_error)}")
        ollama_client = None
        OLLAMA_AVAILABLE = False
        
except ImportError:
    OLLAMA_AVAILABLE = False
    OLLAMA_VISION_AVAILABLE = False
    print("Ollama not available for image description - install the 'ollama' Python package")

def generate_image_description(file_path: str, 
                             task_id: Optional[str] = None, 
                             original_filename: Optional[str] = None, 
                             update_status_func: Optional[callable] = None) -> Dict[str, Any]:
    """
    Generate a description of an image using Ollama's multimodal capabilities.
    
    Args:
        file_path: Path to the image file
        task_id: Optional task ID for status updates
        original_filename: Optional original filename for status updates
        update_status_func: Optional function to update status
        
    Returns:
        Dict containing the image description and metadata
    """
    # Helper function to safely call the update status function
    def _update_status(message: str):
        if task_id and update_status_func and original_filename:
            try:
                update_status_func(task_id, "PROCESSING", message, filename=original_filename)
            except Exception as e:
                print(f"Error calling update_status_func from image_enrichment: {e}")
                
    if not OLLAMA_AVAILABLE or not OLLAMA_VISION_AVAILABLE:
        return {
            "error": "Ollama vision not available",
            "possible_solution": "Ensure Ollama is running and a multimodal model (like llava or bakllava) is installed"
        }
    
    try:
        _update_status(f"Preparing image for vision analysis...")
        # Read the image file
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Set a seed for more consistent generations
        generation_seed = 42
        temperature = 0.3
        top_p = 0.9
        num_predict = 100
        
        # Prepare the prompt for image description
        prompt = """
        Describe this image in detail. Provide a plain text description (no markdown) that:
        1. Describes the main content of the image.
        2. Identifies people, objects, setting, and any text visible.
        3. Is concise but informative (about 50 words).
        
        Format the response as a single paragraph with no headings or bullet points.
        """
        
        # --- Call LLM (Potentially long) ---
        _update_status(f"Generating description of image...")
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            images=[base64_image],
            options={
                "seed": generation_seed,
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict
            }
        )
        _update_status(f"Processing...")
        
        # Extract and clean up the description
        if hasattr(response, 'response'):
            description = response.response
        else:
            description = response.get("response", "")
        
        # Process the description to remove common artifacts
        description = re.sub(r'^\s*(\*|\-|\#|\>|\d+\.)\s*', '', description)  # Remove leading markdown formatting
        description = re.sub(r'\n+', ' ', description)  # Replace newlines with spaces
        description = re.sub(r'\s+', ' ', description)  # Replace multiple spaces with a single space
        description = description.strip()
        
        # Create the result dictionary
        result = {
            "description": description,
            "metadata": {
                "model": OLLAMA_MODEL,
                "provider": "ollama",
                "generated_tokens": getattr(response, 'eval_count', 0) if hasattr(response, 'eval_count') else response.get("eval_count", 0),
                "generation_settings": {
                    "seed": generation_seed,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error generating image description: {str(e)}")
        traceback.print_exc()
        return {
            "error": f"Failed to generate image description: {str(e)}",
            "traceback": traceback.format_exc()
        }

def enrich_image(file_path: str,
                 task_id: Optional[str] = None, 
                 original_filename: Optional[str] = None, 
                 update_status_func: Optional[callable] = None) -> Dict[str, Any]:
    """
    Extract image-specific metadata from an image file, including LLM description if available.
    
    Args:
        file_path: Path to the image file
        task_id: Optional task ID for status updates
        original_filename: Optional original filename for status updates
        update_status_func: Optional function to update status
        
    Returns:
        Dict containing image metadata (including llm_description)
    """
    if not PILLOW_AVAILABLE:
        return {"error": "Image enrichment unavailable - PIL/Pillow library not installed"}
    
    # Helper function to safely call the update status function
    def _update_status(message: str):
        if task_id and update_status_func and original_filename:
            try:
                update_status_func(task_id, "PROCESSING", message, filename=original_filename)
            except Exception as e:
                print(f"Error calling update_status_func from image_enrichment: {e}")

    try:
        # --- Basic Image Info ---
        _update_status("Extracting basic image properties (dimensions, format, mode)...")
        img = Image.open(file_path)
        info = {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "has_alpha": img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info),
            "animated": hasattr(img, "is_animated") and img.is_animated
        }
        
        # If the image is animated (like a GIF), get frame information
        if info["animated"]:
            try:
                info["frames"] = getattr(img, "n_frames", 0)
                info["duration"] = [frame.info.get('duration', 0) for frame in 
                                    [img.seek(i) or img for i in range(info["frames"])]]
                # Reset the frame position to avoid issues
                img.seek(0)
            except Exception as e:
                info["animation"] = {"error": str(e)}
        
        # Image statistics if available (for non-animated images or first frame)
        try:
            # Convert to RGB if in another mode
            if img.mode != 'RGB':
                stat_img = img.convert('RGB')
            else:
                stat_img = img
                
            stats = ImageStat.Stat(stat_img)
            info["statistics"] = {
                "mean": stats.mean,
                "median": stats.median,
                "stddev": stats.stddev,
                "extrema": stats.extrema
            }
        except Exception as e:
            info["statistics"] = {"error": str(e)}
            
        # Get the color palette if it's a palettized image
        if img.mode == 'P':
            try:
                palette = img.getpalette()
                if palette:
                    info["palette"] = palette[:768] # Store RGB palette (max 256 colors)
            except Exception as e:
                info["palette"] = {"error": str(e)}
        
        # --- LLM Description ---
        # Generate description using the helper function if vision is available
        if OLLAMA_AVAILABLE and OLLAMA_VISION_AVAILABLE:
            # Pass status update args to the description function
            llm_result = generate_image_description(
                file_path,
                task_id=task_id, 
                original_filename=original_filename, 
                update_status_func=update_status_func
            )
            
            if "error" not in llm_result:
                info["llm_description"] = llm_result.get("description")
                info["llm_metadata"] = llm_result.get("metadata")
            else:
                # Store the error if generation failed
                info["llm_description_error"] = llm_result.get("error")
                _update_status(f"Vision LLM description failed: {llm_result.get('error')}")
        
        # Return the enriched information
        return info
        
    except Exception as e:
        print(f"Error enriching image: {str(e)}")
        traceback.print_exc()
        return {
            "error": f"Failed to enrich image: {str(e)}",
            "traceback": traceback.format_exc()
        }