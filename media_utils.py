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

def mix_audio_video(video_path: str, audio_path: str, output_path: str, start_time: float = 0.0, audio_volume: float = 0.7):
    """
    Mixes audio into video using ffmpeg-python.
    
    :param video_path: Path to input video
    :param audio_path: Path to input audio (music)
    :param output_path: Path to output video
    :param start_time: Time in seconds to start playing the music
    :param audio_volume: Volume of the music (0.0 to 1.0)
    """
    try:
        # 0. Probe video duration
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        video_duration = float(video_info['duration'])
        
        print(f"Video duration: {video_duration}s")
        
        # Calculate fade out start
        fade_out_start = video_duration - 5.0
        if fade_out_start < 0:
            fade_out_start = 0.0

        # 1. Define Inputs
        video_input = ffmpeg.input(video_path)
        music_input = ffmpeg.input(audio_path)
        
        # 2. Process Music
        # Apply volume to music
        music_processed = music_input.audio.filter('volume', audio_volume)
        
        # Apply delay if start_time > 0
        if start_time > 0:
            # adelay uses milliseconds. "delays" arg: "1000|1000" for stereo
            delay_ms = int(start_time * 1000)
            music_processed = music_processed.filter('adelay', f"{delay_ms}|{delay_ms}")
        
        # Apply Fade In (starts at start_time)
        music_processed = music_processed.filter('afade', type='in', start_time=start_time, duration=3)
        
        # Apply Fade Out (ends at video_duration)
        music_processed = music_processed.filter('afade', type='out', start_time=fade_out_start, duration=5)

        # 3. Mix with Original Audio
        # Check if video has audio stream
        has_audio = False
        for s in probe['streams']:
            if s['codec_type'] == 'audio':
                has_audio = True
                break
        
        if has_audio:
            original_audio = video_input.audio
            # amix: inputs=2, duration=first (video length)
            # Note: 'duration=first' ensures output is same length as first input (video)
            mixed_audio = ffmpeg.filter([original_audio, music_processed], 'amix', inputs=2, duration='first')
        else:
            # If no original audio, just use music (trimmed to video length)
            mixed_audio = music_processed.filter('atrim', duration=video_duration)

        # 4. Output
        stream = (
            ffmpeg
            .output(video_input.video, mixed_audio, output_path, 
                    vcodec='copy',
                    acodec='aac',
                    audio_bitrate='192k'
            )
            .overwrite_output()
        )
        
        # 5. Run
        print("Executing FFmpeg command...")
        stream.run(capture_stdout=True, capture_stderr=True)
        print(f"✅ Mixing complete: {output_path}")

    except ffmpeg.Error as e:
        print("❌ FFmpeg Error:")
        if e.stderr:
            print(e.stderr.decode('utf8'))
        else:
            print("Unknown FFmpeg error (no stderr captured)")
        raise
    except Exception as e:
        print(f"❌ Error in mix_audio_video: {e}")
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
