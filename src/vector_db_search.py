import os
import chromadb
import google.generativeai as genai  # Lightweight direct import
from dotenv import load_dotenv
import uuid

from src.utils import CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

# Load env for cloud connection
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT_ID")
CHROMA_DATABASE = os.getenv("CHROMA_DB")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE or not GEMINI_API_KEY:
    raise Exception(
        "\n❌ Missing Environment Variables\n"
        "Please set: CHROMA_API_KEY, CHROMA_TENANT_ID, CHROMA_DB, and GEMINI_API_KEY\n"
    )

# --- 1. Define Lightweight Embedding Function ---
# This replaces the heavy "chromadb.utils.embedding_functions" 
# to save ~150MB of space on Vercel.
class LightweightGeminiEmbeddingFunction:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)

    def __call__(self, input):
        # ChromaDB expects a list of text strings and requires a list of embedding arrays back.
        if not input:
            return []
        
        # You can use "models/text-embedding-004" or "models/embedding-001"
        model = "models/text-embedding-004"
        
        embeddings = []
        # Loop through inputs (simple batching)
        for text in input:
            # Generate embedding
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
            
        return embeddings

# --- 2. Initialize Chroma Cloud Client ---
# Note: Ensure you are using 'chromadb-client' in requirements.txt
client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE,
)

# global collection cache
_collection = None

def get_db_collection():
    global _collection

    if _collection is None:
        # Instantiate our custom lightweight class
        ef = LightweightGeminiEmbeddingFunction(api_key=GEMINI_API_KEY)

      _collection = client.get_or_create_collection(
    name="arca_policies_gemini",  # IMPORTANT!! Must be different
    embedding_function=ef,
)


        print("🔗 Chroma Cloud Collection Connected (Lightweight Gemini) ✓")

    return _collection


class VectorDB:
    def __init__(self):
        self.collection = get_db_collection()

        # Import splitter inside the class to avoid top-level overhead if not needed immediately
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def add_document(self, text: str, filename: str, user_id: str):
        chunks = self.text_splitter.split_text(text)
        ids = [f"{user_id}-{uuid.uuid4()}" for _ in chunks]
        metadatas = [{"source": filename, "user_id": user_id} for _ in chunks]

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"☁️ Added {len(chunks)} chunks → user={user_id}, file={filename}")

    def search(self, query: str, user_id: str, top_k: int = 5):
        print(f"🔍 Searching query='{query}' user='{user_id}'")

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"user_id": {"$eq": user_id}}
        )

        docs = results.get("documents", [[]])[0]

        # Fallback to default corpus if user has no data
        if (not docs or len(docs) == 0) and user_id != "default":
            print("⚠️ No user match → fallback to default corpus")
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"user_id": {"$eq": "default"}}
            )
            docs = results.get("documents", [[]])[0]

        if not docs or len(docs) == 0:
            print("❌ No match found")
            return []

        return list(zip(results["ids"][0], docs))

# singleton instance
_db_instance = VectorDB()

def get_db():
    return _db_instance