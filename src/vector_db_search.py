import os
import glob
import pathlib
import argparse
import sys
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from src.utils import CHUNK_SIZE, CHUNK_OVERLAP

DB_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
MODEL_NAME = "all-MiniLM-L6-v2"


class VectorDB:
    def __init__(self, persist_directory: str = DB_DIR):
        self.persist_directory = persist_directory

        # NEW Chroma client API
        self.client = Client(
            Settings(
                is_persistent=True,
                persist_directory=persist_directory,
            )
        )

        # SentenceTransformers embedding wrapper
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )

        # Load or create collection
        self._create_or_load_collection()

    def _create_or_load_collection(self):
        collections = [c.name for c in self.client.list_collections()]
        if "arca_policies" in collections:
            self.collection = self.client.get_collection("arca_policies")
        else:
            self.collection = self.client.create_collection(
                name="arca_policies",
                embedding_function=self.embedding_function
            )

    def build_from_folder(self, folder: str):
        files = glob.glob(os.path.join(folder, "**/*"), recursive=True)

        docs = []
        ids = []
        metas = []

        for f in files:
            if not any(f.endswith(ext) for ext in (".md", ".txt")):
                continue

            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read()

            start, idx = 0, 0
            while start < len(text):
                chunk = text[start:start+CHUNK_SIZE]
                chunk_id = f"{pathlib.Path(f).stem}-chunk-{idx}"

                docs.append(chunk)
                ids.append(chunk_id)
                metas.append({"source": f})

                start += CHUNK_SIZE - CHUNK_OVERLAP
                idx += 1

        if docs:
            self.collection.add(
                ids=ids,
                documents=docs,
                metadatas=metas
            )
           

        return len(docs)

    def search(self, query: str, top_k: int = 5):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        ids = results["ids"][0]
        docs = results["documents"][0]
        return list(zip(ids, docs))

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARCA Vector DB Utility")
    parser.add_argument("--build", "-b", help="Folder of policies to index")
    parser.add_argument("--list", action="store_true", help="List all stored chunks")
    parser.add_argument("--clean", action="store_true", help="Delete DB directory")
    args = parser.parse_args()

    db = VectorDB()

    if args.clean:
        db.clean_db()
        sys.exit(0)

    if args.list:
        chunks = db.list_chunks()
        for cid, text in chunks:
            print(f"ID: {cid}\n{text[:200]}\n{'-'*40}")
        sys.exit(0)

    if args.build:
        count = db.build_from_folder(args.build)
        print(f"Indexed {count} chunks.")
        sys.exit(0)

    print("No arguments provided. Use --build, --list, or --clean.")


# ---------------- Singleton for ARCA Agents ----------------
_db_instance = None

def get_db():
    global _db_instance
    if _db_instance is None:
        _db_instance = VectorDB()
    return _db_instance
