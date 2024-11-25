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


@app.post("/transcribe_audio/")
async def transcribe_audio(file: UploadFile = File(...)):
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in AUDIO_EXTENSIONS:
        return {"error": f"Unsupported audio file format: {file_extension}. Supported formats: {AUDIO_EXTENSIONS}"}
    temp_path = f"temp_audio_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        audio = whisper.load_audio(temp_path)
        result = model.transcribe(audio)
        transcription = result["text"]

        # Save transcription to a text file
        output_file_path = f"{Path(file.filename).stem}_audio_transcription.txt"
        with open(output_file_path, "w") as transcription_file:
            transcription_file.write(transcription)

        # Clean up the temporary file
        Path(temp_path).unlink()

        return {
            "filename": file.filename,
            "transcription": transcription,
            "output_file": output_file_path,
        }
    except Exception as e:
        print(str(e),e)
        return {"error": str(e)}

# API for video file transcription
@app.post("/transcribe_video/")
async def transcribe_video(file: UploadFile = File(...)):
    file_extension = Path(file.filename).suffix.lower()
    print("inside the function")
    if file_extension not in VIDEO_EXTENSIONS:
        return {"error": f"Unsupported video file format: {file_extension}. Supported formats: {VIDEO_EXTENSIONS}"}

    # Save the video file temporarily
    dirr = os.getcwd()
    temp_path = f"{dirr}/audios/temp_video_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print("file opened")

    # Load and transcribe the audio extracted from the video
    try:
        print("inside the try")
        print(temp_path)
        audio = whisper.load_audio(file=temp_path)
        print("audio loaded succes")
        result = model.transcribe(audio)
        transcription = result["text"]
        print("transcription done")

        # Save transcription to a text file
        output_file_path = f"{Path(file.filename).stem}_video_transcription.txt"
        with open(output_file_path, "w") as transcription_file:
            transcription_file.write(transcription)

        # Clean up the temporary file
        Path(temp_path).unlink()

        return {
            "filename": file.filename,
            "transcription": transcription,
            "output_file": output_file_path,
        }
    except Exception as e:
        print(str(e),e)
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
