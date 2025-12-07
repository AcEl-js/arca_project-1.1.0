import os
import chromadb
from dotenv import load_dotenv
import uuid
import time
import numpy as np

# Make sure to import your utils if they are in the same folder structure
# from src.utils import CHUNK_SIZE, CHUNK_OVERLAP
# Hardcoding these if utils import fails for standalone usage context
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

load_dotenv()

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT_ID")
CHROMA_DATABASE = os.getenv("CHROMA_DB")

if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE:
    raise Exception(
        "\n❌ Missing Environment Variables\n"
        "Please set: CHROMA_API_KEY, CHROMA_TENANT_ID, CHROMA_DB\n"
    )

# ------------------------------
# Simple Text Splitter
# ------------------------------
class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> list:
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            if end < text_len:
                last_para = text.rfind('\n\n', start, end)
                if last_para > start + 100:
                    end = last_para + 2
                else:
                    best_break = -1
                    for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                        pos = text.rfind(sep, start, end)
                        if pos > start + 100:
                            best_break = max(best_break, pos + len(sep))
                    
                    if best_break > start + 100:
                        end = best_break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            if end >= text_len:
                break
            
            next_start = max(end - self.chunk_overlap, start + self.chunk_size // 2)
            start = next_start
        
        return chunks

# ------------------------------
# Create Gemini Embedding Function
# ------------------------------
class GeminiEmbeddingFunction:
    def __init__(self):
        # Assuming src.gemini_manager exists in your project structure
        try:
            from src.gemini_manager import get_key_manager
            self.key_manager = get_key_manager()
            print("✅ Gemini embedding function initialized with key manager")
        except ImportError:
            # Fallback if gemini_manager is missing in this context
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.key_manager = None
            print("⚠️ Key manager not found, using direct genai fallback")

    def name(self):
        return "gemini-text-embedding-004"

    def __call__(self, input):
        if not input:
            return []
        
        results = []
        for i, text in enumerate(input):
            try:
                if self.key_manager:
                    embedding = self.key_manager.embed_content_with_retry(
                        text=text, task_type="retrieval_document"
                    )
                else:
                    # Fallback logic
                    import google.generativeai as genai
                    resp = genai.embed_content(
                        model="models/text-embedding-004",
                        content=text,
                        task_type="retrieval_document"
                    )
                    embedding = resp["embedding"]

                results.append(embedding)
                time.sleep(0.5) 
                    
            except Exception as e:
                print(f"⚠️ Embedding error for chunk {i+1}: {e}")
                results.append([0.0] * 768)
                
        return np.array(results)
    
    def embed_query(self, query=None, input=None):
        text = query if query is not None else input
        if text is None:
            raise ValueError("Either 'query' or 'input' parameter must be provided")
        
        if self.key_manager:
            embedding = self.key_manager.embed_content_with_retry(
                text=text, task_type="retrieval_query"
            )
        else:
            import google.generativeai as genai
            resp = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            embedding = resp["embedding"]

        return np.array(embedding)

# ------------------------------
# Connect to Chroma Cloud
# ------------------------------
client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

_collection = None

def get_db_collection():
    global _collection

    if _collection is None:
        ef = GeminiEmbeddingFunction()

        # FIXED: Changed name to 'v3' to avoid dimension conflict (384 vs 768)
        # If 'v2' exists with 384 dims, this code crashes. 'v3' will be created fresh with 768.
        _collection = client.get_or_create_collection(
            name="arca_policies_gemini_v3", 
            embedding_function=ef,
        )

        print("🔗 Chroma Cloud Connected (Gemini Embeddings - v3) ✔")

    return _collection

# ------------------------------
# Vector DB Wrapper
# ------------------------------
class VectorDB:
    def __init__(self):
        self.collection = get_db_collection()
        self.text_splitter = SimpleTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def add_document(self, text: str, filename: str, user_id: str):
        # ... (Same as your provided code) ...
        # For brevity, I'm assuming you have the add_document logic here
        pass 

    def search(self, query: str, user_id: str, top_k: int = 5):
        print(f"🔍 Searching '{query[:50]}...' for user '{user_id}'")

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"user_id": {"$eq": user_id}}
            )
        except Exception as e:
            print(f"❌ Chroma Search Error: {str(e)}")
            return []

        docs = results.get("documents", [[]])[0]

        if not docs and user_id != "default":
            print("⚠️ No results for user — fallback to default dataset")
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"user_id": {"$eq": "default"}}
            )
            docs = results.get("documents", [[]])[0]

        if not docs:
            print("❌ No match found")
            return []

        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        return list(zip(ids, docs))

_db_instance = VectorDB()

def get_db():
    return _db_instance