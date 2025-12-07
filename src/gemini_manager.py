"""
Gemini API Key Manager with automatic rotation on rate limit errors
"""
import os
import time
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
        
        # Filter out None values
        self.api_keys = [key for key in self.api_keys if key]
        
        if not self.api_keys:
            raise Exception("No Gemini API keys found in environment variables")
        
        self.current_key_index = 0
        self.configure_current_key()
        
        print(f"✅ Loaded {len(self.api_keys)} Gemini API keys")
    
    def configure_current_key(self):
        """Configure genai with current API key"""
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        print(f"🔑 Using API key #{self.current_key_index + 1}/{len(self.api_keys)}")
    
    def rotate_key(self):
        """Switch to next available API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.configure_current_key()
        print(f"🔄 Rotated to API key #{self.current_key_index + 1}")
    
    def get_current_key(self):
        """Get current API key"""
        return self.api_keys[self.current_key_index]
    
    def embed_content_with_retry(self, text, task_type="retrieval_document", max_retries=None):
        """
        Embed content with automatic key rotation on rate limit errors
        """
        if max_retries is None:
            max_retries = len(self.api_keys)
        
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
                
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
                    print(f"⚠️ Rate limit hit on key #{self.current_key_index + 1}: {e}")
                    
                    if attempt < max_retries - 1:
                        self.rotate_key()
                        time.sleep(1)
                        continue
                    else:
                        print("❌ All API keys exhausted")
                        raise Exception(f"All {len(self.api_keys)} API keys hit rate limits")
                else:
                    # Non-rate-limit error, don't retry
                    raise e
        
        raise Exception("Failed to embed content after all retries")


# Global instance
_key_manager = None

def get_key_manager():
    global _key_manager
    if _key_manager is None:
        _key_manager = GeminiKeyManager()
    return _key_manager