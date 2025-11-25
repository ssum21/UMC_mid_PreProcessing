import os
import time
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class GeminiVideoAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def process_video(self, video_path, transcript=""):
        """
        Uploads video to Gemini, waits for processing, and generates analysis.
        """
        print(f"Uploading file: {video_path}...")
        video_file = genai.upload_file(path=video_path)
        print(f"Completed upload: {video_file.uri}")

        # Wait for the file to be ready
        while video_file.state.name == "PROCESSING":
            print('.', end='', flush=True)
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")

        print(f"\nFile is ready: {video_file.name}")

        # Create the prompt
        prompt = f"""
        Analyze this video and generate a JSON response for music generation.
        
        Here is the audio transcript of the video for context:
        "{transcript}"
        
        The JSON should have a 'suno_request' key with the following fields:
        - title: A creative title for the music.
        - style: The musical style (e.g., 'lo-fi', 'upbeat pop', 'cinematic').
        - prompt: A detailed description of the music to generate.
        - customMode: true
        - instrumental: true or false based on the video vibe.
        
        Output ONLY valid JSON.
        """

        print("Generating content...")
        response = self.model.generate_content(
            [video_file, prompt],
            generation_config={"response_mime_type": "application/json"}
        )

        try:
            result_json = json.loads(response.text)
            # Ensure the structure matches what main2.py expects
            if 'suno_request' not in result_json:
                # If the model returned a flat JSON, wrap it
                result_json = {'suno_request': result_json}
            
            return {'gemini_analysis': result_json}
        except json.JSONDecodeError:
            print("Failed to parse JSON response")
            print(response.text)
            return {'gemini_analysis': {'suno_request': {}, 'error': 'JSON parse error'}}
        finally:
            # Clean up the file from Gemini storage
            print(f"Deleting file: {video_file.name}")
            genai.delete_file(video_file.name)
