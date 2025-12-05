import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import uuid

from src.utils import CHUNK_SIZE, CHUNK_OVERLAP  # 400 / 50 as required

load_dotenv()

# Use environment variable or default to ./chroma_db
DB_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# Singleton clients
_client = None
_collection = None


def get_db_collection():
    global _client, _collection
    if _collection is None:
        # 1. Initialize Persistent Client
        _client = chromadb.PersistentClient(path=DB_DIR)

        # 2. Embedding: all-MiniLM-L6-v2 (free, local-friendly)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # 3. Get or Create Collection
        _collection = _client.get_or_create_collection(
            name="arca_policies",
            embedding_function=ef,
        )
    return _collection


class VectorDB:
    def __init__(self):
        self.collection = get_db_collection()

        # ARCA-compliant chunking: 400 size / 50 overlap
        try:
            # New style
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            # Fallback (older langchain)
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,        # 400
            chunk_overlap=CHUNK_OVERLAP,  # 50
        )

    def add_document(self, text: str, filename: str, user_id: str):
        """
        Splits text and saves it with user_id metadata.
        """
        chunks = self.text_splitter.split_text(text)

        # Generate IDs: "user_123-uuid"
        ids = [f"{user_id}-{uuid.uuid4()}" for _ in range(len(chunks))]

        # Tag Metadata: This is what allows User Isolation (SaaS extension)
        metadatas = [{"source": filename, "user_id": user_id} for _ in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"DEBUG: Added {len(chunks)} chunks for user '{user_id}' from '{filename}'")

    def search(self, query: str, user_id: str, top_k: int = 5):
        """
        Searches User Data first, then falls back to Default Data.
        This is the engine behind the vector_db_search tool.
        """
        print(f"DEBUG: Searching for user '{user_id}'...")

        # 1. Try User's Private Data
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"user_id": user_id},
        )

        found_ids = results["ids"][0] if results["ids"] else []

        # 2. Fallback to 'default' corpus
        if not found_ids and user_id != "default":
            print(f"DEBUG: User '{user_id}' has no matching policies. Falling back to 'default'.")
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"user_id": "default"},
            )

        if results["ids"]:
            return list(zip(results["ids"][0], results["documents"][0]))
        return []


# Singleton Instance
_db_instance = VectorDB()


def get_db():
    return _db_instance
