"""
System status check functionality for the file management system.
"""
import importlib
import os
import json
import sys
from typing import Dict, Any, List, Optional
import subprocess

def check_tlsh():
    """Check if TLSH fuzzy hashing is available"""
    try:
        import tlsh
        # Test if TLSH works
        test_hash = tlsh.hash(b"test" * 50)  # TLSH needs at least 50 bytes
        return {"status": "available", "message": "TLSH fuzzy hashing is available"}
    except (ImportError, RuntimeError, Exception) as e:
        return {"status": "unavailable", "message": f"TLSH fuzzy hashing not available: {str(e)}"}

def check_magika():
    """Check if Magika advanced file type detection is available"""
    try:
        from magika import Magika, ContentTypeLabel
        import pathlib
        
        try:
            # Initialize Magika with default model
            magika = Magika()
            print("Magika object created successfully")
            
            # Test that Magika works by analyzing a simple string
            print("Testing Magika with simple string...")
            test_result = magika.identify_bytes(b"test")
            print(f"Got result type: {type(test_result)}, dir: {dir(test_result)}")
            
            # Check if the operation was successful
            if hasattr(test_result, 'ok'):
                print(f"Result has 'ok' attribute, type: {type(test_result.ok)}, value: {test_result.ok}")
                if not test_result.ok:  # Changed from ok() to ok - it's a property, not a method
                    return {"status": "error", "message": f"Magika test failed: {test_result.status}"}
            else:
                return {"status": "error", "message": f"Magika result missing 'ok' attribute: {dir(test_result)}"}
                
            # Access properties to verify correct API
            print("Checking result properties...")
            _ = test_result.output.mime_type
            _ = test_result.output.label  # In v0.6.1, ct_label is renamed to label
            _ = test_result.score         # In v0.6.1, score is at top level, not in output
            return {"status": "available", "message": "Magika initialized successfully"}
        except Exception as init_error:
            print(f"Detailed Magika init error: {str(init_error)}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"Magika initialization failed: {str(init_error)}"}
    except ImportError:
        return {"status": "unavailable", "message": "Magika not available - advanced file type detection disabled"}

def check_whisper():
    """Check if Whisper transcription is available"""
    try:
        # Import the module dynamically
        text_extraction = importlib.import_module("enrichment.text_extraction")
        
        # Check whisper-related variables
        whisper_model = getattr(text_extraction, "WHISPER_MODEL", None)
        whisper_available = getattr(text_extraction, "WHISPER_AVAILABLE", False)
        whisper_model_obj = getattr(text_extraction, "whisper_model", None)
        
        if whisper_available and whisper_model_obj is not None:
            # Newer versions have available_device(), older versions don't
            try:
                if hasattr(text_extraction.whisper, "available_device"):
                    device = text_extraction.whisper.available_device()
                else:
                    # For older Whisper versions
                    device = "cpu"  # Default to CPU
            except:
                device = "cpu"  # Fallback
                
            return {
                "status": "available", 
                "message": f"Whisper media transcription available (using {whisper_model} model on {device})"
            }
        else:
            return {
                "status": "unavailable", 
                "message": "Whisper transcription is not available"
            }
    except Exception as e:
        return {"status": "error", "message": f"Error checking Whisper: {str(e)}"}

def check_ollama():
    """Check if Ollama LLM is available"""
    try:
        # Import the module dynamically
        text_extraction = importlib.import_module("enrichment.text_extraction")
        
        # Check ollama-related variables
        ollama_available = getattr(text_extraction, "OLLAMA_AVAILABLE", False)
        ollama_host = getattr(text_extraction, "OLLAMA_HOST", None)
        ollama_model = getattr(text_extraction, "OLLAMA_MODEL", None)
        ollama_client = getattr(text_extraction, "ollama_client", None)
        
        if ollama_available:
            # Try to verify the model is actually available
            try:
                # Get client and make a test request
                if ollama_client:
                    model_list = ollama_client.list()
                    
                    # Try multiple approaches to extract model names, similar to text_extraction.py
                    model_names = []
                    
                    # Approach 1: Check if it's a dict with 'models' key containing a list of dicts with 'name' key
                    if isinstance(model_list, dict) and 'models' in model_list:
                        try:
                            model_names = [model['name'] for model in model_list['models'] 
                                          if isinstance(model, dict) and 'name' in model]
                        except (TypeError, KeyError):
                            pass
                    
                    # Approach 2: Check if it's an object with 'models' attribute that's iterable
                    if not model_names and hasattr(model_list, 'models'):
                        try:
                            if hasattr(model_list.models, '__iter__'):
                                for model in model_list.models:
                                    if hasattr(model, 'name'):
                                        model_names.append(model.name)
                                    elif isinstance(model, dict) and 'name' in model:
                                        model_names.append(model['name'])
                        except (TypeError, AttributeError):
                            pass
                    
                    # Approach 3: Check if it's just a list or if it's directly iterable
                    if not model_names:
                        try:
                            if hasattr(model_list, '__iter__') and not isinstance(model_list, (str, bytes)):
                                for model in model_list:
                                    if hasattr(model, 'name'):
                                        model_names.append(model.name)
                                    elif isinstance(model, dict) and 'name' in model:
                                        model_names.append(model['name'])
                        except (TypeError, AttributeError):
                            pass
                    
                    # Now check if our model is in the list
                    if ollama_model in model_names:
                        return {
                            "status": "available", 
                            "message": f"Ollama is available with model: {ollama_model}"
                        }
                    elif model_names:
                        # Service is running but our model isn't available
                        available_models = ", ".join(model_names)
                        return {
                            "status": "partial", 
                            "message": f"Ollama service is running with models: {available_models}, but '{ollama_model}' is not available"
                        }
                    else:
                        # Could connect but couldn't determine model list
                        return {
                            "status": "partial", 
                            "message": f"Ollama service is running, but couldn't determine available models"
                        }
                else:
                    return {
                        "status": "partial", 
                        "message": f"Ollama client wasn't initialized properly"
                    }
            except Exception as verify_error:
                return {
                    "status": "error", 
                    "message": f"Ollama connection error: {str(verify_error)}"
                }
        else:
            return {
                "status": "unavailable", 
                "message": f"Ollama LLM is not available. Check if service is running at {ollama_host}"
            }
    except (ImportError, AttributeError) as e:
        return {"status": "error", "message": f"Error checking Ollama: {str(e)}"}

def check_exiftool():
    """Check if exiftool is available"""
    try:
        result = subprocess.run(
            ["exiftool", "-ver"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        return {"status": "available", "message": f"ExifTool version {version} is available"}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"status": "unavailable", "message": "ExifTool is not installed or not in PATH"}

def check_redis():
    """Check if Redis is available for the task queue"""
    try:
        import redis
        
        # Try to get Redis URL from environment or use default
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        
        # Parse the URL to get host and port
        host = 'localhost'
        port = 6379
        if '://' in redis_url:
            parts = redis_url.split('://', 1)[1]
            if '@' in parts:
                auth, location = parts.split('@', 1)
            else:
                location = parts
                
            if ':' in location and '/' in location:
                host_port = location.split('/', 1)[0]
                if ':' in host_port:
                    host, port_str = host_port.split(':', 1)
                    try:
                        port = int(port_str)
                    except ValueError:
                        pass
        
        # Create Redis client and test connection
        r = redis.Redis(host=host, port=port, socket_connect_timeout=2)
        ping_result = r.ping()
        
        if ping_result:
            return {"status": "available", "message": f"Redis is available at {host}:{port}"}
        else:
            return {"status": "error", "message": f"Redis ping failed at {host}:{port}"}
            
    except ImportError:
        return {"status": "error", "message": "Redis Python package is not installed"}
    except Exception as e:
        return {"status": "unavailable", "message": f"Redis connection error: {str(e)}"}

def check_celery():
    """Check if Celery is properly configured and workers are running"""
    try:
        # Try to import and access Celery
        import importlib
        
        try:
            # Import the task queue module
            task_queue = importlib.import_module("task_queue")
            celery_app = getattr(task_queue, "celery_app", None)
            
            if celery_app:
                # Check broker connection
                try:
                    # Get broker URL
                    broker_url = celery_app.conf.broker_url
                    
                    # Now check if any workers are actually running
                    try:
                        # This is a quick test to see if workers are running
                        # We'll send a simple ping task with a short timeout
                        import uuid
                        from celery.exceptions import TimeoutError
                        
                        # Use our custom ping task
                        ping_id = str(uuid.uuid4())
                        ping_task = celery_app.send_task(
                            'task_queue.ping',  # Use our custom ping task
                            args=[],
                            kwargs={},
                            task_id=ping_id,
                            expires=5  # 5 seconds
                        )
                        
                        try:
                            # Wait for result with a short timeout
                            result = ping_task.get(timeout=2)
                            
                            if result == 'pong':
                                return {
                                    "status": "available", 
                                    "message": f"Celery workers are running with broker: {broker_url}"
                                }
                            else:
                                return {
                                    "status": "partial", 
                                    "message": f"Celery is configured but returned unexpected response: {result}"
                                }
                        except TimeoutError:
                            # No worker responded in time
                            return {
                                "status": "error", 
                                "message": f"Celery is configured but no workers responded within timeout. Start workers with 'python worker.py'"
                            }
                        except Exception as e:
                            # Something else went wrong with the ping
                            return {
                                "status": "partial", 
                                "message": f"Celery is configured but couldn't verify workers: {str(e)}"
                            }
                    
                    except Exception as worker_err:
                        # Fall back to just reporting that Celery is configured
                        return {
                            "status": "partial", 
                            "message": f"Celery is configured with broker: {broker_url} but couldn't verify workers"
                        }
                
                except AttributeError:
                    return {
                        "status": "partial", 
                        "message": "Celery is available but broker URL could not be determined"
                    }
                except Exception as conn_err:
                    return {
                        "status": "error", 
                        "message": f"Celery broker connection error: {str(conn_err)}"
                    }
            else:
                return {"status": "unavailable", "message": "Celery app was not initialized in task_queue module"}
                
        except (ImportError, AttributeError) as e:
            return {"status": "error", "message": f"Error accessing task queue module: {str(e)}"}
            
    except ImportError:
        return {"status": "unavailable", "message": "Celery package is not installed"}

def get_system_status() -> Dict[str, Any]:
    """
    Get status of all system components.
    
    Returns:
        Dict containing status of all components
    """
    status = {
        "tlsh": check_tlsh(),
        "magika": check_magika(),
        "whisper": check_whisper(),
        "ollama": check_ollama(),
        "exiftool": check_exiftool(),
        "redis": check_redis(),
        "celery": check_celery(),
    }
    
    # Add timestamp
    from datetime import datetime
    status["timestamp"] = datetime.now().isoformat()
    
    # Count availability
    available_count = sum(1 for component in status.values() 
                         if isinstance(component, dict) and component.get("status") == "available")
    total_components = sum(1 for component in status 
                          if component not in ["timestamp", "available_count", "total_components"])
    
    status["available_count"] = available_count
    status["total_components"] = total_components
    
    return status