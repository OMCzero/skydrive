"""
Text extraction module for different file types.

This module provides functions to extract text from various file formats
including PDF, Word documents, images (via OCR), and plain text files.
"""
import os
import re
import tempfile
import subprocess
import chardet
import json
import yaml
import xml.etree.ElementTree as ET
import io
import re
from typing import Dict, Any, Optional
import requests

# OCR for images
try:
    import pytesseract
    from PIL import Image
    
    # Test if Tesseract is actually installed on the system
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except (ImportError, pytesseract.TesseractNotFoundError, Exception) as e:
    OCR_AVAILABLE = False
    print(f"OCR not available: {str(e)} - install pytesseract and Tesseract OCR")

# PDF processing
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False
    print("PDF extraction not available - install pdfminer.six")

# Word document processing
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("DOCX extraction not available - install python-docx")

# PDF to image conversion for OCR
try:
    import pdf2image
    PDF_TO_IMAGE_AVAILABLE = True
except ImportError:
    PDF_TO_IMAGE_AVAILABLE = False
    print("PDF to image conversion not available - install pdf2image and poppler")

# Transcription API configuration
TRANSCRIPTION_API_URL = os.environ.get("TRANSCRIPTION_API_URL")
TRANSCRIPTION_API_AVAILABLE = bool(TRANSCRIPTION_API_URL)

if TRANSCRIPTION_API_AVAILABLE:
    print(f"Transcription API configured at: {TRANSCRIPTION_API_URL}")
    # Test connectivity? Maybe later.
else:
    print("Transcription API URL not configured (TRANSCRIPTION_API_URL env var). Audio/video transcription disabled.")

# LLM for text summarization with Ollama
try:
    import ollama
    from ollama import Client, ResponseError
    
    # Configure Ollama
    OLLAMA_AVAILABLE = False
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")  # Using llama3.2 as default
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    # Import necessary modules
    import threading
    
    # Create client with timeout
    try:
        ollama_client = Client(host=OLLAMA_HOST, timeout=3600.0)  # Increased timeout to 1 hour
        print(f"Created Ollama client for host: {OLLAMA_HOST}")
    except Exception as client_error:
        ollama_client = {"error": str(client_error)}
    
    # Test if Ollama is available
    if ollama_client is None:
        OLLAMA_AVAILABLE = False
        print("Cannot check Ollama availability - client initialization failed")
        print("LLM summarization will be disabled")
    else:
        # Check for environment variables to control Ollama behavior
        SKIP_OLLAMA_WARMUP = os.environ.get("SKIP_OLLAMA_WARMUP", "").lower() in ["true", "1", "yes"]
        ASSUME_OLLAMA_MODELS_EXIST = os.environ.get("ASSUME_OLLAMA_MODELS_EXIST", "").lower() in ["true", "1", "yes"]
        
        # In Docker, we can assume models exist on the host
        if ASSUME_OLLAMA_MODELS_EXIST:
            print(f"Assuming Ollama is available with model: {OLLAMA_MODEL} (ASSUME_OLLAMA_MODELS_EXIST=true)")
            OLLAMA_AVAILABLE = True
            
            if SKIP_OLLAMA_WARMUP:
                print(f"Skipping {OLLAMA_MODEL} model warmup as SKIP_OLLAMA_WARMUP is set")
        else:
            try:
                # First, try a simple endpoint like list that should always work if Ollama is running
                try:
                    # Simple connectivity test
                    model_list = ollama_client.list()
                    
                    # Now check if our specific model is available
                    model_available = False
                    available_models = []
                    
                    # Process the model list - it's a ListResponse object with models attribute
                    try:
                        # Extract model names from the models attribute (which is a list)
                        if hasattr(model_list, 'models'):
                            # For newer versions of the Ollama library
                            available_models = [m.model.split(':')[0] for m in model_list.models]
                        elif isinstance(model_list, dict) and "models" in model_list:
                            # For older versions of the Ollama library
                            available_models = [m.get('name', m.get('model', 'unknown')).split(':')[0] 
                                               for m in model_list.get('models', [])]
                        else:
                            # Direct list response
                            if isinstance(model_list, list):
                                available_models = [m.get('name', m.get('model', 'unknown')).split(':')[0]
                                                  for m in model_list]
                            else:
                                print(f"Unexpected model list format: {type(model_list)}")
                        
                        # Strip the version tag if present (e.g., "llama3.2:latest" -> "llama3.2")
                        model_available = any(OLLAMA_MODEL == m.split(':')[0] for m in available_models)
                    except Exception as parse_error:
                        print(f"Could not parse model list: {str(parse_error)}")
                    
                    if model_available:
                        OLLAMA_AVAILABLE = True
                        print(f"Ollama is available with model: {OLLAMA_MODEL}")
                        
                        # Check if we should skip model warmup
                        if not SKIP_OLLAMA_WARMUP:
                            # Warm up the model by sending a simple request
                            try:
                                print(f"Warming up {OLLAMA_MODEL} to keep it loaded in memory...")
                                warmup_thread = threading.Thread(
                                    target=lambda: ollama_client.generate(
                                        model=OLLAMA_MODEL,
                                        prompt="Hello",
                                        options={"num_predict": 1}
                                    )
                                )
                                warmup_thread.daemon = True  # Make thread exit when main program exits
                                warmup_thread.start()
                                print(f"Model {OLLAMA_MODEL} warming up in background")
                            except Exception as warmup_error:
                                print(f"Warning: Model warmup failed: {str(warmup_error)}")
                                print(f"First request may be slower, but this won't affect functionality")
                        else:
                            print(f"Skipping {OLLAMA_MODEL} warmup as SKIP_OLLAMA_WARMUP is set")
                    else:
                        OLLAMA_AVAILABLE = False
                        print(f"Ollama is running but model '{OLLAMA_MODEL}' is not available.")
                        print(f"Available models: {available_models}")
                        print(f"Please run: ollama pull {OLLAMA_MODEL}")
                    
                except Exception as connectivity_error:
                    print(f"Could not connect to Ollama: {str(connectivity_error)}")
                    OLLAMA_AVAILABLE = False
                    print("LLM summarization will be disabled")
                    
            except Exception as outer_error:
                print(f"Error testing Ollama availability: {str(outer_error)}")
                OLLAMA_AVAILABLE = False
                print("LLM summarization will be disabled")
                
except ImportError:
    OLLAMA_AVAILABLE = False
    print("LLM summarization not available - ensure 'ollama' Python package is installed")

def extract_text_from_image(file_path: str) -> Dict[str, Any]:
    """
    Extract text from an image using OCR.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Dict with extracted text and metadata
    """
    if not OCR_AVAILABLE:
        return {"extracted_text": "", "error": "OCR not available", "method": "none"}
    
    try:
        # Open the image
        image = Image.open(file_path)
        
        # Use pytesseract for OCR
        extracted_text = pytesseract.image_to_string(image)
        
        # Create result
        result = {
            "extracted_text": extracted_text.strip(),
            "method": "tesseract_ocr",
            "text_length": len(extracted_text),
            "word_count": len(extracted_text.split()) if extracted_text else 0
        }
        
        # Add LLM summary if available and text is substantial
        if OLLAMA_AVAILABLE and len(extracted_text.strip()) > 100:
            result.update(generate_text_summary(extracted_text))
            
        return result
    except Exception as e:
        return {
            "extracted_text": "",
            "error": f"Failed to extract text from image: {str(e)}",
            "method": "tesseract_ocr_failed"
        }

def extract_text_from_pdf(file_path: str) -> Dict[str, Any]:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dict with extracted text and metadata
    """
    if not PDF_EXTRACTION_AVAILABLE:
        return {"extracted_text": "", "error": "PDF extraction not available", "method": "none"}
    
    try:
        # Use pdfminer to extract text
        extracted_text = pdf_extract_text(file_path)
        
        # If text extraction didn't work well (too little text), try OCR
        if PDF_TO_IMAGE_AVAILABLE and OCR_AVAILABLE and len(extracted_text.strip()) < 100:
            # Convert PDF to images
            pdf_images = pdf2image.convert_from_path(file_path)
            
            # Use OCR on each image and combine results
            ocr_text = ""
            for image in pdf_images:
                ocr_text += pytesseract.image_to_string(image) + "\n\n"
                
            # If OCR yielded more text, use that instead
            if len(ocr_text.strip()) > len(extracted_text.strip()):
                extracted_text = ocr_text
                method = "pdf2image_ocr"
            else:
                method = "pdfminer"
        else:
            method = "pdfminer"
            
        # Create result
        result = {
            "extracted_text": extracted_text.strip(),
            "method": method,
            "text_length": len(extracted_text),
            "word_count": len(extracted_text.split()) if extracted_text else 0
        }
        
        # Add LLM summary if available and text is substantial
        if OLLAMA_AVAILABLE and len(extracted_text.strip()) > 100:
            result.update(generate_text_summary(extracted_text))
            
        return result
    except Exception as e:
        return {
            "extracted_text": "",
            "error": f"Failed to extract text from PDF: {str(e)}",
            "method": "pdf_extract_failed"
        }

def extract_text_from_docx(file_path: str) -> Dict[str, Any]:
    """
    Extract text from a DOCX (Word) file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Dict with extracted text and metadata
    """
    if not DOCX_AVAILABLE:
        return {"extracted_text": "", "error": "DOCX extraction not available", "method": "none"}
    
    try:
        # Load the document
        doc = docx.Document(file_path)
        
        # Extract text from paragraphs
        extracted_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        # Create result
        result = {
            "extracted_text": extracted_text.strip(),
            "method": "python-docx",
            "text_length": len(extracted_text),
            "word_count": len(extracted_text.split()) if extracted_text else 0
        }
        
        # Add document metadata if available
        try:
            core_properties = doc.core_properties
            result["metadata"] = {
                "title": core_properties.title,
                "author": core_properties.author,
                "created": str(core_properties.created) if core_properties.created else None,
                "modified": str(core_properties.modified) if core_properties.modified else None,
            }
        except:
            pass
        
        # Add LLM summary if available and text is substantial
        if OLLAMA_AVAILABLE and len(extracted_text.strip()) > 100:
            result.update(generate_text_summary(extracted_text))
            
        return result
    except Exception as e:
        return {
            "extracted_text": "",
            "error": f"Failed to extract text from DOCX: {str(e)}",
            "method": "docx_extract_failed"
        }

def extract_text_from_plaintext(file_path: str) -> Dict[str, Any]:
    """
    Extract text from a plain text file with encoding detection.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Dict with extracted text and metadata
    """
    try:
        # Read the file as binary to detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
        # Detect encoding
        detection = chardet.detect(raw_data)
        encoding = detection['encoding'] or 'utf-8'
        confidence = detection['confidence']
        
        # Read with detected encoding
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                extracted_text = f.read()
        except UnicodeDecodeError:
            # Fallback to utf-8 if detected encoding fails
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                extracted_text = f.read()
                encoding = 'utf-8 (fallback)'
                confidence = 0.0
        
        # Create result
        result = {
            "extracted_text": extracted_text.strip(),
            "method": "plaintext",
            "encoding": encoding,
            "encoding_confidence": confidence,
            "text_length": len(extracted_text),
            "word_count": len(extracted_text.split()) if extracted_text else 0
        }
        
        # Try to determine if it's a structured format
        if file_path.endswith(('.json', '.js')):
            # Validate JSON syntax
            try:
                json.loads(extracted_text)
                result["format"] = "json"
            except:
                result["format"] = "text"
        elif file_path.endswith(('.yaml', '.yml')):
            # Validate YAML syntax
            try:
                yaml.safe_load(extracted_text)
                result["format"] = "yaml"
            except:
                result["format"] = "text"
        elif file_path.endswith(('.xml', '.html', '.htm')):
            # Validate XML/HTML syntax
            try:
                ET.fromstring(extracted_text)
                result["format"] = "xml"
            except:
                result["format"] = "text"
        else:
            result["format"] = "text"
        
        # Add LLM summary if available and text is substantial
        if OLLAMA_AVAILABLE and len(extracted_text.strip()) > 100:
            result.update(generate_text_summary(extracted_text))
            
        return result
    except Exception as e:
        return {
            "extracted_text": "",
            "error": f"Failed to extract text from text file: {str(e)}",
            "method": "plaintext_failed"
        }

def transcribe_audio_video(file_path: str, 
                           task_id: Optional[str] = None, 
                           original_filename: Optional[str] = None, 
                           update_status_func: Optional[callable] = None) -> Dict[str, Any]:
    """
    Transcribe audio or video using an external API.
    
    Args:
        file_path: Path to the audio/video file
        task_id: Optional task ID for status updates
        original_filename: Optional original filename for status updates
        update_status_func: Optional function to update status
        
    Returns:
        Dict with transcription result
    """
    # Helper function to safely call the update status function
    def _update_status(message: str):
        if task_id and update_status_func and original_filename:
            try:
                update_status_func(task_id, "PROCESSING", message, filename=original_filename)
            except Exception as e:
                print(f"Error calling update_status_func from text_extraction: {e}")
                
    if not TRANSCRIPTION_API_AVAILABLE:
        return {"extracted_text": "", "error": "Transcription API not configured", "method": "none"}
    
    try:
        _update_status("Preparing file for transcription API...")
        with open(file_path, 'rb') as f:
            files = {'file': f}
            
            # --- Submit job to Transcription API ---
            _update_status("Submitting audio/video to transcription API...")
            response = requests.post(TRANSCRIPTION_API_URL, files=files)
            
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data.get('job_id')
            
            if not job_id:
                error_msg = f"Transcription API did not return a job_id: {job_data}"
                _update_status("Transcription failed: No job ID returned")
                return {"extracted_text": "", "error": error_msg, "method": "external_transcription_api_failed"}
            
            # Poll for results
            _update_status(f"Transcription job submitted (ID: {job_id}). Waiting for results...")
            
            import time
            max_attempts = 60  # Wait up to 5 minutes (60 x 5 seconds)
            for attempt in range(max_attempts):
                time.sleep(5)  # Wait 5 seconds between polling attempts
                
                try:
                    poll_url = f"{TRANSCRIPTION_API_URL}/{job_id}" if not TRANSCRIPTION_API_URL.endswith('/') else f"{TRANSCRIPTION_API_URL}{job_id}"
                    poll_response = requests.get(poll_url)
                    
                    if poll_response.status_code == 200:
                        result = poll_response.json()
                        status = result.get('status')
                        
                        if status == "completed":
                            extracted_text = result.get('text', '')
                            _update_status("Transcription completed successfully.")
                            
                            transcription_result = {
                                "extracted_text": extracted_text.strip(),
                                "method": "external_transcription_api",
                                "text_length": len(extracted_text),
                                "word_count": len(extracted_text.split()) if extracted_text else 0,
                                "api_metadata": {"job_id": job_id}
                            }
                            
                            # Add LLM summary if available and text is substantial
                            if OLLAMA_AVAILABLE and len(extracted_text.strip()) > 100:
                                # Pass status update args down
                                summary_result = generate_text_summary(
                                    extracted_text, 
                                    task_id=task_id, 
                                    original_filename=original_filename, 
                                    update_status_func=update_status_func
                                )
                                transcription_result.update(summary_result)
                            
                            return transcription_result
                        
                        elif status == "error":
                            error_msg = f"Transcription job failed: {result.get('error', 'Unknown error')}"
                            _update_status(f"Transcription failed: {error_msg}")
                            return {"extracted_text": "", "error": error_msg, "method": "external_transcription_api_failed"}
                        
                        else:  # status still "processing"
                            if attempt % 6 == 0:  # Update status every ~30 seconds
                                _update_status(f"Still waiting for transcription... (attempt {attempt+1}/{max_attempts})")
                    
                    else:
                        _update_status(f"Error checking transcription status: {poll_response.status_code}")
                
                except Exception as poll_error:
                    _update_status(f"Error polling for results: {str(poll_error)}")
            
            # If we get here, we've exceeded max attempts
            error_msg = f"Transcription timed out after {max_attempts} polling attempts"
            _update_status("Transcription failed: Timeout")
            return {"extracted_text": "", "error": error_msg, "method": "external_transcription_api_timeout"}
            
        else:
            error_msg = f"Transcription API failed with status {response.status_code}: {response.text}"
            _update_status(f"Transcription failed: {response.status_code}")
            return {"extracted_text": "", "error": error_msg, "method": "external_transcription_api_failed"}
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Error connecting to transcription API: {str(e)}"
        _update_status("Transcription failed: Connection error")
        return {"extracted_text": "", "error": error_msg, "method": "external_transcription_api_failed"}
    except Exception as e:
        error_msg = f"Error during transcription: {str(e)}"
        _update_status("Transcription failed: Unknown error")
        return {"extracted_text": "", "error": error_msg, "method": "external_transcription_api_failed"}

def generate_text_summary(text: str, 
                          task_id: Optional[str] = None, 
                          original_filename: Optional[str] = None, 
                          update_status_func: Optional[callable] = None) -> Dict[str, Any]:
    """
    Generate a summary of the provided text using Ollama.
    
    Args:
        text: The text to summarize
        task_id: Optional task ID for status updates
        original_filename: Optional original filename for status updates
        update_status_func: Optional function to update status
        
    Returns:
        Dict containing the summary and metadata, or an error
    """
    # Helper function to safely call the update status function
    def _update_status(message: str):
        if task_id and update_status_func and original_filename:
            try:
                update_status_func(task_id, "PROCESSING", message, filename=original_filename)
            except Exception as e:
                print(f"Error calling update_status_func from text_extraction: {e}")
                
    if not OLLAMA_AVAILABLE:
        return {"llm_summary": "", "llm_error": "Ollama not available for summarization"}
    
    if not text or len(text.strip()) < 100:
        return {"llm_summary": "", "llm_metadata": {"reason": "Text too short for summary"}}
    
    try:
        # Limit text length to avoid excessive token usage (e.g., first 5000 chars)
        max_chars = 5000
        truncated_text = text[:max_chars]
        if len(text) > max_chars:
            truncated_text += "... (text truncated for summarization)"
            
        # Prepare the prompt for summarization
        prompt = f"""
        Summarize the following text in about 50-70 words. 
        Focus on the key topics and entities mentioned. 
        Provide the summary as a single paragraph.
        
        Text:
        {truncated_text}
        
        Summary:
        """
        
        # Configure generation parameters
        generation_seed = 42
        temperature = 0.5
        top_p = 0.9
        num_predict = 150 # Limit summary tokens
        
        # --- Call LLM for Summarization (Potentially long) ---
        _update_status(f"Generating summary...")
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                "seed": generation_seed,
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": num_predict
            }
        )
        _update_status(f"LLM summary response received.")
        
        # Extract and clean up the summary
        if hasattr(response, 'response'):
            summary = response.response
        else:
            summary = response.get("response", "")
            
        summary = re.sub(r'^\s*(\*|\-|\#|\>|\d+\.)\s*', '', summary)  # Remove leading markdown
        summary = re.sub(r'\n+', ' ', summary).strip() # Replace newlines, strip whitespace
        
        return {
            "llm_summary": summary,
            "llm_metadata": {
                "model": OLLAMA_MODEL,
                "provider": "ollama",
                "generated_tokens": getattr(response, 'eval_count', 0) if hasattr(response, 'eval_count') else response.get("eval_count", 0),
                "generation_settings": {
                    "seed": generation_seed,
                    "temperature": temperature,
                    "top_p": top_p,
                    "truncated_input": len(text) > max_chars
                }
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to generate text summary: {str(e)}"
        _update_status(f"LLM summarization failed: {str(e)}")
        return {"llm_summary": "", "llm_error": error_msg}

def extract_text(file_path: str, mime_type: str, 
                 task_id: Optional[str] = None, 
                 original_filename: Optional[str] = None, 
                 update_status_func: Optional[callable] = None) -> Dict[str, Any]:
    """
    Extract text content from a file based on its MIME type.
    Handles different file types and calls appropriate extraction functions.
    Also attempts to generate a summary using an LLM if text is found.
    
    Args:
        file_path: Path to the file
        mime_type: MIME type of the file
        task_id: Optional task ID for status updates
        original_filename: Optional original filename for status updates
        update_status_func: Optional function to update status
        
    Returns:
        Dict containing extracted text, metadata, and potentially a summary
    """
    result = {
        "extracted_text": "",
        "method": "none"
    }
    
    try:
        if mime_type.startswith('image/'):
            result = extract_text_from_image(file_path)
        elif mime_type == 'application/pdf':
            result = extract_text_from_pdf(file_path)
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            result = extract_text_from_docx(file_path)
        elif mime_type.startswith('text/'):
            result = extract_text_from_plaintext(file_path)
        elif mime_type.startswith('audio/') or mime_type.startswith('video/'):
             # Pass status update args down to transcription
             result = transcribe_audio_video(
                 file_path, 
                 task_id=task_id, 
                 original_filename=original_filename, 
                 update_status_func=update_status_func
             )
        else:
            result["method"] = "unsupported_mime_type"
        
        # If text was extracted (and not via transcription which handles its own summary), try summarizing
        extracted_text = result.get("extracted_text", "")
        if extracted_text and OLLAMA_AVAILABLE and result.get("method", "none") not in ["external_transcription_api", "none", "unsupported_mime_type"]:
            # Check if it's substantial enough for summary
            if len(extracted_text.strip()) > 100:
                 # Pass status update args down to summarization
                 summary_result = generate_text_summary(
                     extracted_text, 
                     task_id=task_id, 
                     original_filename=original_filename, 
                     update_status_func=update_status_func
                 )
                 result.update(summary_result) # Add summary fields to the main result
            else:
                result["llm_summary"] = "" # Indicate no summary generated due to length
                result["llm_metadata"] = {"reason": "Text too short for summary"}
                
    except Exception as e:
        result["error"] = f"Error during text extraction: {str(e)}"
        
    return result