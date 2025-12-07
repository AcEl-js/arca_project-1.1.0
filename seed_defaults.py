"""
Seed script to populate ChromaDB with default policies from data/policies folder
Run this once to initialize your database with your existing policies
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from src.vector_db_search import get_db

load_dotenv()


def read_policy_file(filepath: Path) -> str:
    """Read a policy file and return its content"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding if utf-8 fails
        with open(filepath, 'r', encoding='latin-1') as f:
            return f.read()


def seed_default_policies():
    """Seed ChromaDB with policies from data/policies folder"""
    print("\n" + "="*60)
    print("🌱 SEEDING DEFAULT POLICIES TO CHROMADB")
    print("="*60 + "\n")
    
    # Path to policies folder
    policies_dir = Path("data/policies")
    
    if not policies_dir.exists():
        print(f"❌ Error: {policies_dir} folder not found!")
        print("Please make sure the data/policies folder exists.\n")
        return
    
    # Get all markdown files
    policy_files = list(policies_dir.glob("*.md"))
    
    if not policy_files:
        print(f"❌ No .md files found in {policies_dir}")
        return
    
    print(f"📁 Found {len(policy_files)} policy files\n")
    
    db = get_db()
    successful = 0
    failed = 0
    
    for idx, filepath in enumerate(policy_files, 1):
        filename = filepath.name
        print(f"[{idx}/{len(policy_files)}] Processing: {filename}")
        
        try:
            content = read_policy_file(filepath)
            
            if not content.strip():
                print(f"    ⚠️  Skipping {filename} (empty file)\n")
                continue
            
            db.add_document(
                text=content,
                filename=filename,
                user_id="default"  # Using "default" as user_id for shared policies
            )
            
            print(f"    ✅ Successfully added {filename}")
            print(f"    📝 Content length: {len(content)} characters\n")
            successful += 1
            
        except Exception as e:
            print(f"    ❌ Error adding {filename}: {e}\n")
            failed += 1
    
    print("="*60)
    print(f"✅ SEEDING COMPLETE!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print("="*60)
    print("\nYour ChromaDB now contains default policies.")
    print("Users without custom policies will fallback to these.\n")


if __name__ == "__main__":
    seed_default_policies()