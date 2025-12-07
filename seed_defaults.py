import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import uuid

from src.utils import CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT_ID")
CHROMA_DATABASE = os.getenv("CHROMA_DB")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE or not GEMINI_API_KEY:
    raise Exception(
        "\n❌ Missing Environment Variables\n"
        "Please set: CHROMA_API_KEY, CHROMA_TENANT_ID, CHROMA_DB, and GEMINI_API_KEY\n"
    )


# ------------------------------
# Simple Text Splitter (Replaces LangChain)
# ------------------------------
class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> list:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # Try to break at natural boundaries if not at end
            if end < text_len:
                # Look for paragraph break
                last_para = text.rfind('\n\n', start, end)
                if last_para > start:
                    end = last_para + 2
                else:
                    # Look for sentence break
                    for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                        last_sent = text.rfind(sep, start, end)
                        if last_sent > start:
                            end = last_sent + len(sep)
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move forward with overlap
            if end >= text_len:
                break
            start = end - self.chunk_overlap
        
        return chunks


# ------------------------------
# Create Gemini Embedding Wrapper
# ------------------------------
class LightweightGeminiEmbeddingFunction:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)

    def name(self):
        """Required by ChromaDB to identify the embedding function"""
        return "gemini-text-embedding-004"

    def __call__(self, input):
        """
        IMPORTANT: Parameter MUST be named 'input' for ChromaDB compatibility
        ChromaDB passes a list of strings to embed
        """
        if not input:
            return []

        results = []
        import time
        
        for i, text in enumerate(input):
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                results.append(response["embedding"])
                
                # Small delay to avoid rate limiting
                if i > 0 and i % 5 == 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"⚠️ Embedding error for chunk {i+1}: {e}")
                # Return zero vector as fallback
                results.append([0.0] * 768)
                
        return results


# ------------------------------
# Connect to Chroma Cloud
# ------------------------------
_client = None
_collection = None


def get_chroma_client():
    global _client
    if _client is None:
        print("🔄 Connecting to Chroma Cloud...")
        _client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
        print("✅ Chroma Cloud client connected")
    return _client


def get_db_collection():
    global _collection

    if _collection is None:
        client = get_chroma_client()
        ef = LightweightGeminiEmbeddingFunction(GEMINI_API_KEY)

        _collection = client.get_or_create_collection(
            name="arca_policies_gemini",
            embedding_function=ef,
        )

        print("🔗 Chroma Cloud Connected (Gemini Embeddings) ✔")

    return _collection


# ------------------------------
# Vector DB Wrapper
# ------------------------------
class VectorDB:
    def __init__(self):
        self.collection = get_db_collection()
        # Use our simple text splitter instead of LangChain
        self.text_splitter = SimpleTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def add_document(self, text: str, filename: str, user_id: str):
        chunks = self.text_splitter.split_text(text)
        ids = [f"{user_id}-{uuid.uuid4()}" for _ in chunks]
        metadata = [{"source": filename, "user_id": user_id} for _ in chunks]

        self.collection.add(
            documents=chunks,
            metadatas=metadata,
            ids=ids,
        )

        print(f"☁️ Added {len(chunks)} chunks for user={user_id}, file={filename}")

    def search(self, query: str, user_id: str, top_k: int = 5):
        print(f"🔍 Searching '{query[:50]}...' for user '{user_id}'")

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"user_id": {"$eq": user_id}}
        )

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

        return list(zip(results["ids"][0], docs))


_db_instance = None


def get_db():
    global _db_instance
    if _db_instance is None:
        print("🔄 Initializing VectorDB instance...")
        _db_instance = VectorDB()
        print("✅ VectorDB instance ready")
    return _db_instance