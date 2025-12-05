import os
import sys
import pypdf # Make sure to pip install pypdf

# 1. Setup paths so we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.vector_db_search import get_db

# 2. Define where your default policies are
POLICIES_DIR = os.path.join(parent_dir, "data", "policies")

def extract_content(filepath):
    """Reads PDF, TXT, or MD files."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            reader = pypdf.PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        elif ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

def seed_defaults():
    print(f"--- ARCA: Seeding Default Knowledge Base ---")
    print(f"Scanning folder: {POLICIES_DIR}")
    
    if not os.path.exists(POLICIES_DIR):
        print(f"ERROR: The folder {POLICIES_DIR} does not exist!")
        print("Please create it and put your default policy files there.")
        return

    # Find all supported files
    files = [f for f in os.listdir(POLICIES_DIR) if f.lower().endswith(('.pdf', '.txt', '.md'))]
    
    if not files:
        print("No files found! Please add .pdf, .txt, or .md files to data/policies/")
        return

    db = get_db()
    count = 0
    
    for filename in files:
        filepath = os.path.join(POLICIES_DIR, filename)
        print(f"Processing: {filename}...")
        
        text_content = extract_content(filepath)
        
        if text_content and len(text_content.strip()) > 10:
            # CRITICAL: We tag these as 'default' so the backend fallback logic works
            db.add_document(text=text_content, filename=filename, user_id="default")
            count += 1
            print(f" -> Indexed successfully.")
        else:
            print(f" -> Skipped (Empty or unreadable).")

    print(f"--- Finished. {count} default policies are now in the database. ---")

if __name__ == "__main__":
    seed_defaults()