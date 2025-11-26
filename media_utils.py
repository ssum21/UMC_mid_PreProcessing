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

def mix_audio_video(video_path: str, audio_path: str, output_path: str, start_time: float = 0.0, audio_volume: float = 0.3):
    """
    Mixes the audio into the video at the specified start time with the given volume.
    The original video audio is kept.
    """
    try:
        print(f"Mixing audio {audio_path} into {video_path} at {start_time}s with volume {audio_volume}...")
        
        input_video = ffmpeg.input(video_path)
        input_audio = ffmpeg.input(audio_path)
        
        # Delay the audio start
        delayed_audio = input_audio.filter('adelay', f"{int(start_time * 1000)}|{int(start_time * 1000)}")
        
        # Adjust volume
        adjusted_audio = delayed_audio.filter('volume', audio_volume)
        
        # Mix with original audio
        # We assume the video has audio. If not, we might need a check.
        # For simplicity, we mix the original video audio (0:a) with the new audio.
        mixed_audio = ffmpeg.filter([input_video.audio, adjusted_audio], 'amix', inputs=2, duration='first')
        
        (
            ffmpeg
            .output(input_video.video, mixed_audio, output_path, vcodec='copy', acodec='aac')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print("Mixing complete.")
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
        # Set fp16 based on device
        fp16 = False if device == "cpu" else True
        result = model.transcribe(video_path, fp16=fp16)
        transcript = result['text']
        print(f"Transcription complete. Length: {len(transcript)} chars")
        return transcript
    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""
