from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import io
import pypdf 

# 1. Load environment variables FIRST
load_dotenv()

# 2. Import agents AFTER .env is loaded
# NOTE: We do NOT import 'load_gemini_keys' because CrewAI handles keys automatically now.
from src.agents import run_arca_pipeline
from src.vector_db_search import get_db

# 3. Initialize App
app = FastAPI(title="ARCA SaaS API")

# 4. Add CORS Middleware (Fixes the 405/Network Error on Frontend)
app.add_middleware(
    CORSMiddleware,
    # Allow your frontend URL (adjust port if needed)
    allow_origins=["http://localhost:3000", "http://localhost:3001"], 
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows x-user-id header
)

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Helper to extract text from PDF or decode Text files."""
    try:
        if filename.lower().endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            # Assume text/markdown
            return file_bytes.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

# --- ENDPOINT 1: Upload Policy (SaaS Feature) ---
@app.post("/upload_policy")
async def upload_policy(
    file: UploadFile = File(...),
    x_user_id: str = Header(...) # Enforce User ID header
):
    """Uploads a policy file into the user's private vector store."""
    if not x_user_id:
         raise HTTPException(status_code=400, detail="Missing x-user-id header")

    try:
        content = await file.read()
        text = extract_text(content, file.filename)
        
        # Add to Vector DB tagged with user_id
        get_db().add_document(text, file.filename, x_user_id)
        
        return {"status": "success", "message": f"Indexed {file.filename} for user {x_user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT 2: Analyze Regulation (CrewAI Trigger) ---
@app.post("/analyze_regulation")
async def analyze(
    new_regulation_text: str | None = Form(None),
    regulation_file: UploadFile | None = File(None),
    date_of_law: str | None = Form(None),
    x_user_id: str = Header(...) # Enforce User ID header
):
    final_text = ""
    
    # 1. Handle File Input (PDF)
    if regulation_file:
        content = await regulation_file.read()
        final_text = extract_text(content, regulation_file.filename)
    
    # 2. Handle Text Input
    elif new_regulation_text:
        final_text = new_regulation_text.strip()

    if not final_text:
        raise HTTPException(status_code=400, detail="No input provided")

    # 3. Run Pipeline with User ID
    # CrewAI will automatically find GEMINI_API_KEY in os.environ
    result = run_arca_pipeline(final_text, x_user_id, date_of_law)
    return result