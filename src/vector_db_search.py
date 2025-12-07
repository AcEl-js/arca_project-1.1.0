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
# Create Gemini Embedding Wrapper
# ------------------------------
class LightweightGeminiEmbeddingFunction:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)

    # REQUIRED BY CHROMADB
    def name(self):
        return "gemini_lightweight_v1"  # any unique name works

    def __call__(self, input_list):
        if not input_list:
            return []

        results = []
        for text in input_list:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            results.append(response["embedding"])
        return results


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

        ef = LightweightGeminiEmbeddingFunction(GEMINI_API_KEY)

        _collection = client.get_or_create_collection(
            name="arca_policies_gemini",  # MUST BE UNIQUE
            embedding_function=ef,
        )

        print("🔗 Chroma Cloud Connected (Gemini Embeddings) ✓")

    return _collection


# ------------------------------
# Vector DB Wrapper
# ------------------------------
class VectorDB:
    def __init__(self):
        self.collection = get_db_collection()

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
        metadata = [{"source": filename, "user_id": user_id} for _ in chunks]

        self.collection.add(
            documents=chunks,
            metadatas=metadata,
            ids=ids,
        )

        print(f"☁️ Added {len(chunks)} chunks for user={user_id}, file={filename}")

    def search(self, query: str, user_id: str, top_k: int = 5):
        print(f"🔍 Searching '{query}' for user '{user_id}'")

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


_db_instance = VectorDB()


def get_db():
    return _db_instance
