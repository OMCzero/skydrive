from fastapi import FastAPI, File, UploadFile, Response
import whisper
import uvicorn
import tempfile
import os

app = FastAPI()

# Load the Whisper model
model = whisper.load_model("turbo")  # or "medium", "small" based on your hardware

@app.head("/")
async def health_check():
    # Return a 200 status code for connectivity checks
    return Response(status_code=200)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
        temp.write(await file.read())
        temp_path = temp.name
    
    # Transcribe with Whisper
    result = model.transcribe(temp_path)
    
    # Delete the temporary file
    os.unlink(temp_path)
    
    return {"text": result["text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)