import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import uuid
import time

# Import shared components to ensure we use the SAME collection and embedding logic
from src.vector_db_search import get_db, GeminiEmbeddingFunction
from src.utils import CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

# Verify keys
if not os.getenv("CHROMA_API_KEY") or not os.getenv("GEMINI_API_KEY"):
    raise Exception("❌ Missing Environment Variables")

# ------------------------------
# Simple Text Splitter
# ------------------------------
class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> list:
        if not text: return []
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                last_para = text.rfind('\n\n', start, end)
                if last_para > start: end = last_para + 2
                else:
                    for sep in ['. ', '! ', '? ', '.\n']:
                        last_sent = text.rfind(sep, start, end)
                        if last_sent > start: 
                            end = last_sent + len(sep)
                            break
            
            chunk = text[start:end].strip()
            if chunk: chunks.append(chunk)
            if end >= text_len: break
            start = end - self.chunk_overlap
        return chunks

# ------------------------------
# Seed Logic
# ------------------------------
def seed_policies():
    print("🚀 Starting Seeding Process for Collection: arca_policies_gemini_v3")
    
    # Use the central DB instance (points to v3)
    db = get_db()
    splitter = SimpleTextSplitter(CHUNK_SIZE, CHUNK_OVERLAP)
    
    policies_dir = "data/policies"
    if not os.path.exists(policies_dir):
        print(f"❌ Directory not found: {policies_dir}")
        return

    files = [f for f in os.listdir(policies_dir) if f.endswith(".md")]
    print(f"📂 Found {len(files)} policy files")

    for i, filename in enumerate(files):
        filepath = os.path.join(policies_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        if not content: continue

        chunks = splitter.split_text(content)
        ids = [f"default-{uuid.uuid4()}" for _ in chunks]
        metadatas = [{"source": filename, "user_id": "default"} for _ in chunks]

        print(f"Processing {filename} ({len(chunks)} chunks)...")

        # Add to DB
        # We access the underlying collection directly for batching if needed, 
        # or use the wrapper. The wrapper add_document does splitting, 
        # but here we want control or to match previous logic.
        # Let's use the wrapper's logic but adapted for this script
        
        try:
            db.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Added {filename}")
            time.sleep(1) # Rate limit safety
        except Exception as e:
            print(f"❌ Failed to add {filename}: {e}")

    print("\n✅ Seeding Complete!")

if __name__ == "__main__":
    seed_policies()