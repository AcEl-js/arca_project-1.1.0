"""
Gemini API Key Manager with automatic rotation on rate limit errors
"""
import os
import time
import random
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiKeyManager:
    def __init__(self):
        # Load all available API keys
        self.api_keys = [
            os.getenv("GEMINI_API_KEY"),
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3"),
            os.getenv("GEMINI_API_KEY_4"),
            os.getenv("GEMINI_API_KEY_5"),
            os.getenv("GEMINI_API_KEY_6"),
            os.getenv("GEMINI_API_KEY_7"),
        ]
        
        # Filter out None values and duplicates
        self.api_keys = list(set([key for key in self.api_keys if key]))
        
        if not self.api_keys:
            raise Exception("No Gemini API keys found in environment variables")
        
        self.current_key_index = 0
        self.configure_current_key()
        
        print(f"✅ Loaded {len(self.api_keys)} unique Gemini API keys")
    
    def configure_current_key(self):
        """Configure genai with current API key"""
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        # print(f"🔑 Switched to API key #{self.current_key_index + 1}")
    
    def rotate_key(self):
        """Switch to next available API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_current_key()
        print(f"🔄 Rotated to API Key #{self.current_key_index + 1} due to rate limit")
        return self.api_keys[self.current_key_index]
    
    def get_current_key(self):
        """Get current API key"""
        return self.api_keys[self.current_key_index]

    def embed_content_with_retry(self, text, task_type="retrieval_document", max_retries=None):
        if max_retries is None:
            max_retries = len(self.api_keys) * 2
        
        for attempt in range(max_retries):
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type=task_type
                )
                return response["embedding"]
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
                    print(f"⚠️ Embed Rate Limit hit. Rotating key...")
                    self.rotate_key()
                    time.sleep(1)
                    continue
                raise e
        raise Exception("Failed to embed content after retries")

_key_manager = None

def get_key_manager():
    global _key_manager
    if _key_manager is None:
        _key_manager = GeminiKeyManager()
    return _key_manager