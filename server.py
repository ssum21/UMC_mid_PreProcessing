from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import uuid
import shutil
from video_Processor import GeminiVideoAnalyzer
from media_utils import downsample_video, transcribe_audio, check_ffmpeg, mix_audio_video
from storage_utils import upload_to_r2, download_from_r2, generate_presigned_url
from pydantic import BaseModel
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# In-memory task store
# Structure: {task_id: {"status": str, "music_url": str, "final_video_url": str, "video_object_name": str}}
tasks: Dict[str, dict] = {}

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

class FinalizeRequest(BaseModel):
    video_object_name: str
    music_url: str
    start_time: float = 0.0
    audio_volume: float = 0.3

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

analyzer = GeminiVideoAnalyzer(api_key=GEMINI_API_KEY)

async def process_and_send_to_n8n(file_path: str, filename: str, webhook_url: str, video_object_name: str, task_id: str):
    downsampled_path = None
    try:
        print(f"Starting processing for {filename} (Task: {task_id})...")
        
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
            "task_id": task_id, # Send task_id to n8n so it can pass it back
            "video_object_name": video_object_name, 
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
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
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
        
        # Upload original video to R2 immediately
        print(f"Uploading original video to R2: {file.filename}")
        video_object_name = upload_to_r2(temp_path, f"{uuid.uuid4()}_{file.filename}")

        # Create task
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "processing",
            "video_object_name": video_object_name,
            "music_url": None,
            "final_video_url": None
        }

        # Process in background to not block the response
        background_tasks.add_task(process_and_send_to_n8n, temp_path, file.filename, N8N_WEBHOOK_URL, video_object_name, task_id)
        
        return {
            "message": "Video received and processing started",
            "filename": file.filename,
            "task_id": task_id,
            "status": "processing"
        }

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

class MusicItem(BaseModel):
    title: str
    url: str
    image: str

class MusicCallbackRequest(BaseModel):
    task_id: str
    music_list: list[MusicItem]

@app.post("/api/receive-music")
async def receive_music(request: MusicCallbackRequest, background_tasks: BackgroundTasks):
    """
    Callback endpoint for n8n to send generated music.
    Triggers auto-mixing.
    """
    task_id = request.task_id
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if not request.music_list:
        raise HTTPException(status_code=400, detail="No music generated")

    print(f"Received {len(request.music_list)} songs for task {task_id}")
    
    # Update task
    tasks[task_id]["status"] = "music_ready"
    tasks[task_id]["music_list"] = [m.dict() for m in request.music_list]
    
    return {"message": "Music received, waiting for selection"}

@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

async def process_auto_mixing(task_id: str):
    video_path = None
    music_path = None
    output_path = None
    try:
        task = tasks[task_id]
        video_object_name = task["video_object_name"]
        music_url = task["music_url"]
        
        print(f"Starting auto-mixing for task {task_id}...")
        
        # 1. Download Video from R2
        video_filename = f"temp_video_{task_id}.mp4"
        video_path = os.path.join("/tmp", video_filename)
        download_from_r2(video_object_name, video_path)
        
        # 2. Download Music from URL
        music_filename = f"temp_music_{task_id}.mp3"
        music_path = os.path.join("/tmp", music_filename)
        async with httpx.AsyncClient() as client:
            resp = await client.get(music_url)
            resp.raise_for_status()
            with open(music_path, "wb") as f:
                f.write(resp.content)
                
        # 3. Mix Audio and Video (Default settings: start 0, vol 0.3)
        output_filename = f"final_{task_id}.mp4"
        output_path = os.path.join("/tmp", output_filename)
        mix_audio_video(video_path, music_path, output_path, video_duration=10, audio_volume=0.3)
        
        # 4. Upload Result to R2
        final_object_name = f"final/{output_filename}"
        upload_to_r2(output_path, final_object_name)
        
        # 5. Update Task
        # Construct public URL if possible, otherwise just object name
        # Assuming R2 public bucket or similar. For now returning object name.
        # If you have a custom domain, you can prepend it here.
        final_url = f"{os.environ.get('R2_ENDPOINT_URL')}/{os.environ.get('R2_BUCKET_NAME')}/{final_object_name}"
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["final_video_url"] = final_object_name # Or full URL
        
        print(f"Auto-mixing complete for task {task_id}")
        
    except Exception as e:
        print(f"Error in auto-mixing task {task_id}: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
    finally:
        # Cleanup
        for p in [video_path, music_path, output_path]:
            if p and os.path.exists(p):
                os.remove(p)

@app.post("/api/finalize-video")
async def finalize_video(request: FinalizeRequest, background_tasks: BackgroundTasks):
    """
    Downloads video and music, mixes them, and uploads the result.
    """
    try:
        # Generate a unique task ID for tracking (optional)
        # Create task
        task_id = str(uuid.uuid4())
        tasks[task_id] = {
            "status": "mixing",
            "video_object_name": request.video_object_name,
            "music_url": request.music_url,
            "final_video_url": None
        }
        
        print(f"Received finalization request: {request}")
        
        # We'll do this in a background task too, to not block
        background_tasks.add_task(process_finalization, request, task_id)
        
        return {"message": "Finalization started", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_finalization(request: FinalizeRequest, task_id: str):
    video_path = None
    music_path = None
    output_path = None
    try:
        print(f"Starting finalization task {task_id}...")
        
        # 1. Download Video from R2
        video_filename = f"temp_video_{task_id}.mp4"
        video_path = os.path.join("/tmp", video_filename)
        download_from_r2(request.video_object_name, video_path)
        
        # 2. Download Music from URL
        music_filename = f"temp_music_{task_id}.mp3"
        music_path = os.path.join("/tmp", music_filename)
        async with httpx.AsyncClient() as client:
            resp = await client.get(request.music_url)
            resp.raise_for_status()
            with open(music_path, "wb") as f:
                f.write(resp.content)
                
        # 3. Mix Audio and Video
        output_filename = f"final_{task_id}.mp4"
        output_path = os.path.join("/tmp", output_filename)
        mix_audio_video(video_path, music_path, output_path, request.start_time, request.audio_volume)
        
        # 4. Upload Result to R2
        final_object_name = f"final/{output_filename}"
        upload_to_r2(output_path, final_object_name)
        
        # Generate Presigned URL
        final_url = generate_presigned_url(final_object_name)
        
        if task_id in tasks:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["final_video_url"] = final_url
        
        print(f"Finalization complete. Uploaded to {final_object_name}")
        
    except Exception as e:
        print(f"Error in finalization task {task_id}: {e}")
        if task_id in tasks:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)
    finally:
        # Cleanup
        for p in [video_path, music_path, output_path]:
            if p and os.path.exists(p):
                os.remove(p)

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    check_ffmpeg()
