import os
import time
import json
import subprocess
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class GeminiVideoAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        # Using the model name from previous code/user request
        self.model_name = 'gemini-2.5-flash-lite' 
        self.model = genai.GenerativeModel(self.model_name)
        
        # Optimization settings
        self.target_resolution = {
            'width': 640,   # 360p (16:9)
            'height': 360,
            'bitrate': '250k',  # 250kbps
            'audio_bitrate': '64k',
            'fps': 15  # 15fps
        }

    def get_video_info(self, video_path):
        """Extract video info using ffprobe"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            video_stream = next(
                (s for s in data['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            return {
                'duration': float(data['format']['duration']),
                'size_mb': float(data['format']['size']) / (1024 * 1024),
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'bitrate': int(data['format'].get('bit_rate', 0)) / 1000,
                'fps': eval(video_stream.get('r_frame_rate', '30/1')),
            }
        except Exception as e:
            print(f"Error getting video info: {e}")
            # Fallback or re-raise
            return {'duration': 0, 'size_mb': 0, 'width': 0, 'height': 0, 'bitrate': 0, 'fps': 0}

    def optimize_video(self, input_path, output_path=None):
        """
        Downsample video for speed optimization (<20MB target).
        """
        if output_path is None:
            base = Path(input_path).stem
            output_path = f"/tmp/{base}_opt.mp4"
        
        print(f"⚡️ Starting optimization: {input_path}")
        
        try:
            info = self.get_video_info(input_path)
            print(f"  Original: {info['width']}x{info['height']}, {info['size_mb']:.1f}MB")
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', f"scale={self.target_resolution['width']}:{self.target_resolution['height']}:force_original_aspect_ratio=decrease,pad={self.target_resolution['width']}:{self.target_resolution['height']}:(ow-iw)/2:(oh-ih)/2",
                '-b:v', self.target_resolution['bitrate'],
                '-b:a', self.target_resolution['audio_bitrate'],
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-r', str(self.target_resolution['fps']),
                '-preset', 'fast',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            new_info = self.get_video_info(output_path)
            print(f"  Completed: {new_info['width']}x{new_info['height']}, {new_info['size_mb']:.1f}MB")
            
            return output_path
        except Exception as e:
            print(f"Optimization failed: {e}")
            return input_path # Fallback to original

    def analyze_with_gemini(self, video_path, video_duration=None, custom_prompt=None):
        """
        Analyze video using Gemini API.
        """
        print(f"🔍 Starting Gemini analysis: {video_path}")
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f}MB")
        
        if custom_prompt is None:
            duration_info = f"Video duration is approximately {int(video_duration)} seconds." if video_duration else ""
            
            prompt = f"""
                Analyze this video in detail. {duration_info} Provide the following information in JSON format:

                {{
                "video_duration": "Video duration (seconds, number only)",
                "scene_type": "Scene type (e.g., Nature, Indoor Party, Sports, Cooking, etc.)",
                "main_subjects": ["Top 3 main subjects"],
                "activities": ["Top 3 main activities"],
                "mood": "Overall mood (Bright/Dark/Calm/Energetic/Emotional, etc.)",
                "color_tone": "Main color tone (Warm/Cool/Neutral/Vibrant, etc.)",
                "movement_speed": "Movement speed (Fast/Medium/Slow/Static)",
                "time_of_day": "Time of day (Sunrise/Day/Sunset/Night or Indoor)",
                "audio_description": "Audio description (Dialogue/Music/Nature sounds, etc.)",
                "key_moments": [
                    {{"timestamp": "MM:SS", "description": "Description of key moment"}}
                ],
                "detected_event": "Detected event (Birthday, Wedding, Trip, Workout, etc. or null)",
                "emotional_tone": "Emotional tone (Joyful/Sad/Tense/Peaceful, etc.)",
                "recommended_music_style": "Recommended music style (Specific)",
                "recommended_tempo": "Recommended tempo (Slow/Medium/Fast or BPM)",
                "music_mood_keywords": ["5 music mood keywords"],
                "suno_request": {{
                    "prompt": "Lyrics or music description based on video mood and content (Max 3000 chars). If instrumental is true, describe style; if false, include lyrics.",
                    "style": "Music style (e.g., Pop, Electronic, Jazz). Max 1000 chars.",
                    "title": "Music title (Max 80 chars)",
                    "instrumental": false, 
                    "customMode": true,
                    "model": "V5",
                    "callBackUrl": "https://example.com/callback",
                    "vocalGender": "m or f (optional, based on mood)",
                    "styleWeight": 0.65,
                    "weirdnessConstraint": 0.65,
                    "audioWeight": 0.65
                }}
                }}

                Output ONLY valid JSON. No additional explanation.
                Set instrumental to false (include lyrics) unless the video has heavy dialogue that clashes with vocals.
                If instrumental is false, YOU MUST generate creative lyrics in the 'prompt' field based on the video content.
                Include specific genre and mood in style.
                """
        else:
            prompt = custom_prompt

        try:
            if file_size_mb < 20:
                print("  Mode: Inline (<20MB)")
                response = self._analyze_inline(video_path, prompt)
            else:
                print("  Mode: File API (>=20MB)")
                response = self._analyze_with_file_api(video_path, prompt)
            
            # Parse JSON
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split('\n')
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)
            
            result = json.loads(text)
            print("✅ Gemini analysis complete")
            return result
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_inline(self, video_path, prompt):
        """Inline analysis for small files"""
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
            
        # google-generativeai expects a dict for inline data
        inline_data = {
            "mime_type": "video/mp4",
            "data": video_bytes
        }
        
        return self.model.generate_content(
            [inline_data, prompt],
            generation_config={"response_mime_type": "application/json"}
        )

    def _analyze_with_file_api(self, video_path, prompt):
        """File API analysis for large files"""
        print("  Uploading file...")
        video_file = genai.upload_file(path=video_path)
        print(f"  Upload complete: {video_file.name}")
        
        while video_file.state.name == "PROCESSING":
            print('.', end='', flush=True)
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
            
        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")
            
        print(" Ready!")
        
        try:
            response = self.model.generate_content(
                [video_file, prompt],
                generation_config={"response_mime_type": "application/json"}
            )
            return response
        finally:
            print(f"  Deleting file: {video_file.name}")
            genai.delete_file(video_file.name)

    def process_video(self, video_path, transcript=""):
        """
        Main processing pipeline.
        """
        print(f"\n{'='*60}")
        print(f"🎬 Processing Video: {Path(video_path).name}")
        print(f"{'='*60}\n")
        
        # 1. Get original info
        original_info = self.get_video_info(video_path)
        
        # 2. Optimize (Downsample)
        # We can skip downsampling if file is already small, but for consistency we might just run it
        # or check size. The user logic always ran it unless skipped.
        sampled_path = self.optimize_video(video_path)
        sampled_info = self.get_video_info(sampled_path)
        
        # 3. Analyze
        # Pass original duration to prompt
        gemini_result = self.analyze_with_gemini(sampled_path, video_duration=original_info.get('duration'))
        
        # Inject correct duration
        if 'video_duration' in gemini_result:
             gemini_result['video_duration'] = original_info.get('duration')
        
        # Ensure suno_request exists (wrapper logic from before, though prompt should handle it)
        if 'suno_request' not in gemini_result:
             # Try to find it or wrap it? The prompt is strict now.
             pass

        print(f"\n{'='*60}")
        print(f"✅ Processing Complete")
        print(f"{'='*60}\n")
        
        return {
            'gemini_analysis': gemini_result,
            'original_info': original_info,
            'sampled_info': sampled_info,
            'sampled_video_path': sampled_path
        }
