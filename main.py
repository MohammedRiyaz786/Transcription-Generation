import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil, uvicorn
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify a list of domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

model = whisper.load_model("base", device='cpu')

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


# API for video file transcription
@app.post("/transcribe/")
async def transcribe_video(file: UploadFile = File(...)):
    try:
        file_extension = Path(file.filename).suffix.lower()
        print(f"Received file: {file.filename}, Extension: {file_extension}")


        audios_dir = os.path.join(os.getcwd(), 'audios')
        os.makedirs(audios_dir, exist_ok=True)

        # Save the video file temporarily
        temp_path = os.path.join(audios_dir, f"temp_video_{file.filename}")
        print(f"Attempting to save file to: {temp_path}")

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("File saved successfully")

        
        audio = whisper.load_audio(file=temp_path)
        print("Audio loaded successfully")
        
        result = model.transcribe(audio)
        transcription = result["text"]
        print("Transcription completed")


        # Clean up the temporary file
        os.unlink(temp_path)

        return {
            "filename": file.filename,
            "transcription": transcription,
            #"output_file": output_file_path,
        }
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
