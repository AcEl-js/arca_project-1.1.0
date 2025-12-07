import os
import uuid
import time
import chromadb
from dotenv import load_dotenv

# Import your existing logic to ensure we test the ACTUAL app code
from src.vector_db_search import get_db, GeminiEmbeddingFunction
from src.utils import CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

def run_diagnostic():
    print("🔍 DIAGNOSTIC: Connecting to Chroma Cloud...")
    
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT_ID"),
        database=os.getenv("CHROMA_DB"),
    )
    
    print("\n📋 Current Collections Status:")
    collections = client.list_collections()
    target_col = None
    
    for col in collections:
        cnt = col.count()
        print(f"   • Name: {col.name: <30} | Count: {cnt} documents")
        if col.name == "arca_policies_gemini_v3":
            target_col = col

    if not target_col:
        print("\n❌ Collection 'arca_policies_gemini_v3' DOES NOT EXIST yet.")
    else:
        print(f"\n✅ Target collection found. Current count: {target_col.count()}")

    return target_col

def force_seed():
    print("\n🚀 STARTING FORCE SEED (Script Mode)...")
    db = get_db()
    
    # Check if we have policies
    policies_dir = "data/policies"
    if not os.path.exists(policies_dir):
        print(f"❌ ERROR: {policies_dir} directory not found!")
        return

    files = [f for f in os.listdir(policies_dir) if f.endswith(".md")]
    print(f"📂 Found {len(files)} policy files to process.")

    total_chunks_added = 0

    for filename in files:
        filepath = os.path.join(policies_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        if not content.strip():
            print(f"⚠️ Skipped empty file: {filename}")
            continue

        # Use the DB's splitter directly to debug chunking
        chunks = db.text_splitter.split_text(content)
        
        print(f"   📄 {filename}: Found {len(content)} chars -> {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print("      ⚠️ WARNING: Splitter returned 0 chunks!")
            continue

        ids = [f"default-{uuid.uuid4()}" for _ in chunks]
        metadatas = [{"source": filename, "user_id": "default"} for _ in chunks]

        # Direct add to collection to bypass any wrapper silence
        try:
            db.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"      ✅ Successfully pushed {len(chunks)} chunks to Chroma")
            total_chunks_added += len(chunks)
            time.sleep(0.5) # Avoid rate limits
        except Exception as e:
            print(f"      ❌ FAILED to add chunks: {e}")

    print(f"\n🏁 Seeding Finished. Total chunks in DB: {total_chunks_added}")

def test_search():
    print("\n🔎 TESTING SEARCH...")
    db = get_db()
    results = db.search("sécurité", "default", top_k=3)
    
    if results:
        print("✅ Search SUCCESS! Found results:")
        for doc_id, text in results:
            print(f"   - {doc_id}: {text[:100]}...")
    else:
        print("❌ Search FAILED: No results found.")

if __name__ == "__main__":
    col = run_diagnostic()
    
    # If empty or missing, run seed
    if not col or col.count() == 0:
        force_seed()
    else:
        choice = input("\nData already exists. Force re-seed? (y/n): ")
        if choice.lower() == 'y':
            force_seed()
    
    test_search()