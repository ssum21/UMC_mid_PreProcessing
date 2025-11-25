from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import uuid
import shutil
from video_Processor import GeminiVideoAnalyzer
from media_utils import downsample_video, transcribe_audio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL")
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

analyzer = GeminiVideoAnalyzer(api_key=GEMINI_API_KEY)

async def process_and_send_to_n8n(file_path: str, filename: str, webhook_url: str):
    downsampled_path = None
    try:
        print(f"Starting processing for {filename}...")
        
        # 1. Downsample Video
        downsampled_filename = f"downsampled_{os.path.basename(file_path)}"
        downsampled_path = os.path.join("/tmp", downsampled_filename)
        downsample_video(file_path, downsampled_path)
        
        # 2. Transcribe Audio (using original file for better quality, or downsampled if speed matters)
        # Using original file for potentially better audio quality
        transcript = transcribe_audio(file_path)
        print(f"Transcript generated: {transcript[:50]}...")

        # 3. Analyze with Gemini (using downsampled video)
        print(f"Starting Gemini analysis...")
        analysis_result = analyzer.process_video(downsampled_path, transcript=transcript)
        gemini_data = analysis_result.get('gemini_analysis', {})
        
        print(f"Analysis complete. Sending to n8n: {webhook_url}")
        
        payload = {
            "filename": filename,
            "analysis": gemini_data,
            "suno_request": gemini_data.get('suno_request'),
            "transcript": transcript
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
            print(f"Successfully sent to n8n. Status: {response.status_code}")

    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        # Cleanup temp files
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up original file: {file_path}")
        if downsampled_path and os.path.exists(downsampled_path):
            os.remove(downsampled_path)
            print(f"Cleaned up downsampled file: {downsampled_path}")

@app.post("/api/analyze-video")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validate file size (approximate, as we stream)
    # Note: UploadFile doesn't have a size attribute until read, 
    # but we can check Content-Length header if available, or check during read.
    
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join("/tmp", temp_filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Check size after saving
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE:
            os.remove(temp_path)
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")

        # Process in background to not block the response
        background_tasks.add_task(process_and_send_to_n8n, temp_path, file.filename, N8N_WEBHOOK_URL)
        
        return {
            "message": "Video received and processing started",
            "filename": file.filename,
            "status": "processing"
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
