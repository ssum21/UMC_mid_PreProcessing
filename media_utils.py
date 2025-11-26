import os
import ffmpeg
import whisper
import torch
import shutil

# Add current directory to PATH so local ffmpeg binary can be found
os.environ["PATH"] += os.pathsep + os.getcwd()

def check_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        print(f"✅ FFmpeg found at: {path}")
        return True
    else:
        print("❌ FFmpeg NOT found in PATH")
        # Try to find it in common locations
        common_paths = ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
        for p in common_paths:
            if os.path.exists(p):
                print(f"⚠️ Found ffmpeg at {p} but it's not in PATH")
        return False

def downsample_video(input_path: str, output_path: str, height: int = 360):
    """
    Downsamples the video to the specified height while maintaining aspect ratio.
    """
    try:
        check_ffmpeg()
        print(f"Downsampling video to {height}p: {input_path} -> {output_path}")
        (
            ffmpeg
            .input(input_path)
            .filter('scale', -2, height)  # -2 ensures width is even (required by some codecs)
            .output(output_path, vcodec='libx264', crf=28, preset='fast', acodec='aac')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("Downsampling complete.")
        return output_path
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode('utf8')}")
        raise

def transcribe_audio(video_path: str, model_size: str = "base"):
    """
    Extracts audio and transcribes it using OpenAI Whisper.
    """
    print(f"Transcribing audio from: {video_path} using model '{model_size}'...")
    
    # Check if CUDA is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = whisper.load_model(model_size, device=device)
        result = model.transcribe(video_path)
        transcript = result['text']
        print(f"Transcription complete. Length: {len(transcript)} chars")
        return transcript
    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""
