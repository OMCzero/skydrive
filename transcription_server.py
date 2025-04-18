from fastapi import FastAPI, File, UploadFile, Response, BackgroundTasks
import whisper
import uvicorn
import tempfile
import os
from typing import Dict, List
import threading

app = FastAPI()

# Load the Whisper model
model = whisper.load_model("turbo")  # or "medium", "small" based on your hardware

# Store results in memory (you might want to use a proper database for production)
results_store: Dict[str, Dict] = {}
active_transcriptions: List[str] = []
lock = threading.Lock()

@app.head("/transcribe/")
async def health_check():
    # Return a 200 status code for connectivity checks
    return Response(status_code=200)

def process_transcription(temp_path: str, job_id: str):
    try:
        # Transcribe with Whisper
        result = model.transcribe(temp_path)
        
        # Print transcription for debugging
        print(f"Job {job_id} completed. Transcription: {result['text']}")
        
        # Store the result
        with lock:
            results_store[job_id] = {"status": "completed", "text": result["text"]}
            if job_id in active_transcriptions:
                active_transcriptions.remove(job_id)
        
        # Delete the temporary file
        os.unlink(temp_path)
    except Exception as e:
        print(f"Error in job {job_id}: {str(e)}")
        with lock:
            results_store[job_id] = {"status": "error", "error": str(e)}
            if job_id in active_transcriptions:
                active_transcriptions.remove(job_id)
        # Clean up in case of error
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile, background_tasks: BackgroundTasks):
    # Generate a job ID
    job_id = f"job_{len(results_store) + 1}"
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
        temp.write(await file.read())
        temp_path = temp.name
    
    # Add to active transcriptions
    with lock:
        active_transcriptions.append(job_id)
        results_store[job_id] = {"status": "processing"}
    
    # Add the transcription task to background tasks
    background_tasks.add_task(process_transcription, temp_path, job_id)
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/transcribe/{job_id}")
async def get_transcription_result(job_id: str):
    if job_id not in results_store:
        return Response(status_code=404, content={"error": "Job not found"})
    
    return results_store[job_id]

@app.get("/status/")
async def get_server_status():
    with lock:
        return {
            "active_transcriptions": len(active_transcriptions),
            "completed_transcriptions": len(results_store) - len(active_transcriptions)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)