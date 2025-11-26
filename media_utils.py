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

#def mix_audio_video(video_path: str, audio_path: str, output_path: str, start_time: float = 0.0, audio_volume: float = 0.3):
def mix_audio_video(video_path: str, audio_path: str, output_path: str, video_duration: float, audio_volume: float = 0.):
    """
    FFmpeg 명령줄과 동일한 로직을 ffmpeg-python 라이브러리로 구현합니다.
    (기존 영상 소리 유지 + 배경 음악 페이드 인/아웃 적용 및 믹싱)
    
    :param video_path: 원본 비디오 파일 경로
    :param audio_path: 배경 음악 파일 경로
    :param output_path: 최종 출력 비디오 파일 경로
    :param video_duration: 원본 비디오의 총 길이(초) - 페이드 아웃 시작점 계산에 사용됨
    :audio_volume: 합성될 음악의 소리 크기
    """
    
    # 배경 음악의 페이드 아웃 시작 시간을 계산합니다.
    # st=(끝나기 5초 전) 로직을 구현합니다.
    # amix 필터에서 duration=shortest를 사용하므로, video_duration이 최종 클립의 길이와 같다고 가정합니다.
    fade_out_start = video_duration - 5.0
    
    # 페이드 아웃 시작 시간이 음수가 되지 않도록 보호합니다.
    if fade_out_start < 0:
        fade_out_start = 0.0
        print("경고: 비디오 길이가 5초 미만이므로 페이드 아웃 시작 시간을 0초로 설정합니다.")


    try:
        # 1. 입력 스트림 정의
        video_input = ffmpeg.input(video_path)
        music_input = ffmpeg.input(audio_path)
        
        # 2. 오디오 스트림 추출 및 필터링
        # [0:a]volume=1.0[a1]
        original_audio = video_input.audio.filter('volume', 1.0)
        
        # [1:a]volume=0.5[a2_1]
        music_vol = music_input.audio.filter('volume', audio_volume)
        
        # [a2_1]afade=t=in:d=3[a2_2]
        music_fade_in = music_vol.filter('afade', type='in', duration=3)
        
        # [a2_2]afade=t=out:d=5:st=(끝나기 5초 전)[a2]
        music_fade_out = music_fade_in.filter('afade', type='out', duration=5, start_time=fade_out_start)

        # 3. 오디오 믹싱 (amix)
        # [a1][a2]amix=inputs=2:duration=shortest[aout]
        mixed_audio = ffmpeg.filter([original_audio, music_fade_out], 'amix', inputs=2, duration='shortest')

        # 4. 최종 출력 스트림 구성
        # 비디오 스트림은 copy, 오디오 스트림은 믹싱된 오디오 사용
        stream = (
            ffmpeg
            .output(video_input.video, mixed_audio, output_path, 
                    # 출력 옵션 지정
                    vcodec='copy',      # -c:v copy
                    acodec='aac',       # -c:a aac
                    audio_bitrate='192k', # -b:a 192k
                    map=['0:v:0', '[aout]'] # stream mapping은 .output에서 자동으로 처리되므로,
                                            # 여기서는 vcodec='copy'가 0:v:0을, mixed_audio가 [aout]을 대체합니다.
                                            # 명시적인 map 옵션은 복잡도를 높일 수 있어 생략해도 됩니다.
            )
            .overwrite_output() # 기존 파일 덮어쓰기
        )
        
        # 5. FFmpeg 명령어 실행
        print("FFmpeg 명령어 실행 중...")
        stream.run()
        print(f"✅ 합성 완료: {output_path}")

    except ffmpeg.Error as e:
        print("❌ FFmpeg 오류 발생:")
        # 에러 메시지를 디코딩하여 출력
        print(e.stderr.decode('utf8'))

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
