#!/usr/bin/env python3
"""
Test script to verify the embedding function works correctly
Run this before starting the server to validate setup
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in environment")
    exit(1)

print("✅ GEMINI_API_KEY found")
genai.configure(api_key=GEMINI_API_KEY)

# Test 1: Direct Gemini embedding
print("\n=== Test 1: Direct Gemini Embedding ===")
try:
    response = genai.embed_content(
        model="models/text-embedding-004",
        content="test query",
        task_type="retrieval_query"
    )
    embedding = response["embedding"]
    print(f"✅ Direct embedding successful")
    print(f"   Type: {type(embedding)}")
    print(f"   Length: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"❌ Direct embedding failed: {e}")
    exit(1)

# Test 2: Embedding function
print("\n=== Test 2: GeminiEmbeddingFunction (from src) ===")
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # UPDATED: Import the correct class name
    from src.vector_db_search import GeminiEmbeddingFunction
    
    # UPDATED: No arguments needed (it handles env vars internally)
    ef = GeminiEmbeddingFunction()
    print(f"✅ Embedding function created")
    
    # Test __call__ (batch embeddings)
    batch_result = ef(["test document 1", "test document 2"])
    print(f"✅ Batch embedding successful")
    print(f"   Number of embeddings: {len(batch_result)}")
    print(f"   First embedding type: {type(batch_result[0])}")
    print(f"   First embedding shape: {batch_result[0].shape if hasattr(batch_result[0], 'shape') else len(batch_result[0])}")
    
    # Test embed_query
    query_result = ef.embed_query(query="test query")
    print(f"✅ Query embedding successful")
    print(f"   Type: {type(query_result)}")
    print(f"   Shape/Length: {query_result.shape if hasattr(query_result, 'shape') else len(query_result)}")
    
except Exception as e:
    print(f"❌ Embedding function test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Database connection
print("\n=== Test 3: Database Connection ===")
try:
    from src.vector_db_search import get_db
    
    db = get_db()
    print(f"✅ Database instance created")
    print(f"   Type: {type(db)}")
    print(f"   Collection: {type(db.collection)}")
    
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Search Test
print("\n=== Test 4: Search Test ===")
try:
    results = db.search("test query", "default", top_k=3)
    print(f"✅ Search executed")
    print(f"   Number of results: {len(results)}")
    if results:
        print(f"   First result ID: {results[0][0]}")
        # Handle tuple return (id, text)
        text_preview = results[0][1][:100] if len(results[0]) > 1 else "N/A"
        print(f"   First result text: {text_preview}...")
    else:
        print(f"   ⚠️ No results found (Expected if v3 is new. Run seed_defaults.py)")
        
except Exception as e:
    print(f"❌ Search test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✅ TEST SEQUENCE COMPLETE")
print("="*60)