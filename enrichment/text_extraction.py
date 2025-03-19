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

# Audio/video transcription
try:
    import whisper
    
    # Configure whisper
    WHISPER_AVAILABLE = False
    WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "turbo")  # Options: tiny, base, small, medium, large
    
    # Initialize whisper model
    try:
        # Check which version of Whisper we're using (newer versions have available_device())
        if hasattr(whisper, "available_device"):
            device = whisper.available_device()
            print(f"Initializing Whisper with '{WHISPER_MODEL}' model on {device}")
        else:
            # Older versions default to CPU or CUDA if available
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Initializing Whisper with '{WHISPER_MODEL}' model on {device}")
            
        # Initialize the model (this will download it if not present)
        whisper_model = whisper.load_model(WHISPER_MODEL)
        WHISPER_AVAILABLE = True
        print(f"Whisper media transcription available (using {WHISPER_MODEL} model on {device})")
    except Exception as whisper_error:
        print(f"Whisper initialization failed: {str(whisper_error)}")
        WHISPER_AVAILABLE = False
        
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper transcription not available - install openai-whisper")

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
        print(f"Error creating Ollama client: {str(client_error)}")
        ollama_client = None
    
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

def transcribe_audio_video(file_path: str) -> Dict[str, Any]:
    """
    Transcribe speech from audio or video files using Whisper.
    
    Args:
        file_path: Path to the audio or video file
        
    Returns:
        Dict with transcribed text and metadata
    """
    if not WHISPER_AVAILABLE:
        return {"extracted_text": "", "error": "Whisper transcription not available", "method": "none"}
    
    try:
        # Use whisper to transcribe
        result = whisper_model.transcribe(file_path)
        
        # Extract transcription text
        transcribed_text = result.get("text", "")
        
        # Create result
        output = {
            "extracted_text": transcribed_text.strip(),
            "method": "whisper_transcription",
            "model": WHISPER_MODEL,
            "text_length": len(transcribed_text),
            "word_count": len(transcribed_text.split()) if transcribed_text else 0
        }
        
        # Add segments if available
        if "segments" in result:
            output["segments"] = [
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text", "").strip()
                }
                for segment in result["segments"]
            ]
        
        # Add LLM summary if available and text is substantial
        if OLLAMA_AVAILABLE and len(transcribed_text.strip()) > 100:
            output.update(generate_text_summary(transcribed_text))
            
        return output
    except Exception as e:
        return {
            "extracted_text": "",
            "error": f"Failed to transcribe audio/video: {str(e)}",
            "method": "whisper_failed"
        }

def generate_text_summary(text: str) -> Dict[str, Any]:
    """
    Generate a summary of the extracted text using Ollama LLM.
    
    Args:
        text: The text to summarize
        
    Returns:
        Dict with summary and metadata
    """
    if not OLLAMA_AVAILABLE:
        return {"llm_summary": "", "llm_error": "Ollama not available"}
    
    try:
        # Prepare the prompt for summarization
        prompt = f"""
        Here's some text extracted from a document. Please provide a concise 2-3 sentence summary:

        {text[:10000]}  # Limit to first 10K chars to avoid context overflow
        
        Provide only the summary in plain text format. No preamble or explanation.
        """
        
        # Set a seed for more consistent summaries
        generation_seed = 42
        
        # Query the model
        response = ollama_client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                "seed": generation_seed,
                "temperature": 0.3,
                "top_p": 0.9
            }
        )
        
        # Extract the summary
        if hasattr(response, 'response'):
            summary = response.response
        else:
            summary = response.get("response", "")
        
        # Process the summary to remove common artifacts
        summary = re.sub(r'^\s*(\*|\-|\#|\>|\d+\.)\s*', '', summary)  # Remove leading markdown formatting
        summary = re.sub(r'\n+', ' ', summary)  # Replace newlines with spaces
        summary = re.sub(r'\s+', ' ', summary)  # Replace multiple spaces with a single space
        summary = summary.strip()
        
        # Return the summary
        return {
            "llm_summary": summary,
            "llm_summary_model": OLLAMA_MODEL,
            "llm_summary_tokens_processed": getattr(response, 'prompt_eval_count', 0) if hasattr(response, 'prompt_eval_count') else response.get("prompt_eval_count", 0),
            "llm_summary_tokens_generated": getattr(response, 'eval_count', 0) if hasattr(response, 'eval_count') else response.get("eval_count", 0),
        }
    except Exception as e:
        return {
            "llm_summary": "",
            "llm_error": f"Failed to generate summary: {str(e)}"
        }

def extract_text(file_path: str, mime_type: str) -> Dict[str, Any]:
    """
    Extract text from a file based on its MIME type.
    
    Args:
        file_path: Path to the file
        mime_type: MIME type of the file
        
    Returns:
        Dict containing the extracted text and metadata
    """
    result = {
        "file_path": file_path,
        "mime_type": mime_type,
        "extension": os.path.splitext(file_path)[1].lower()
    }
    
    # Determine how to extract text based on MIME type
    if mime_type.startswith('image/'):
        extraction_result = extract_text_from_image(file_path)
    elif mime_type == 'application/pdf':
        extraction_result = extract_text_from_pdf(file_path)
    elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        extraction_result = extract_text_from_docx(file_path)
    elif mime_type.startswith('text/'):
        extraction_result = extract_text_from_plaintext(file_path)
    elif mime_type.startswith(('audio/', 'video/')) and WHISPER_AVAILABLE:
        extraction_result = transcribe_audio_video(file_path)
    else:
        # Try plaintext for unknown types
        extraction_result = extract_text_from_plaintext(file_path)
        if not extraction_result.get("extracted_text"):
            extraction_result = {
                "extracted_text": "",
                "error": f"No text extraction method available for MIME type: {mime_type}",
                "method": "none"
            }
    
    # Add the extraction result to the overall result
    result.update(extraction_result)
    
    return result