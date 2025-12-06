from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import pypdf
import io


load_dotenv()

from src.agents import run_arca_pipeline
from src.vector_db_search import get_db


MAX_SIZE = 8 * 1024 * 1024  # 8MB limit for safer Vercel execution


class UploadLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        size = int(request.headers.get("content-length", 0))

        if size > MAX_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": "Uploaded file too large"},
            )

        return await call_next(request)


app = FastAPI(title="ARCA SaaS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://arca-project-frontend.vercel.app",  # change this to your deployed frontend
        "*"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


app.add_middleware(UploadLimitMiddleware)


def extract_text_stream(file: UploadFile, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        buffer = io.BytesIO()

        while True:
            chunk = file.file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            buffer.write(chunk)

        buffer.seek(0)
        pdf_reader = pypdf.PdfReader(buffer)

        text = ""
        for page in pdf_reader.pages:
            try:
                # Limit per-page extraction to stay safe in memory
                page_text = page.extract_text() or ""
                text += page_text[:2000] + "\n"
            except Exception:
                continue

        return text.strip()

    else:
        return file.file.read().decode("utf-8")


@app.post("/upload_policy")
async def upload_policy(
    file: UploadFile = File(...),
    x_user_id: str = Header(...)
):
    if not x_user_id:
        raise HTTPException(status_code=400, detail="Missing x-user-id header")

    try:
        text = extract_text_stream(file, file.filename)
        get_db().add_document(text, file.filename, x_user_id)

        return {
            "status": "success",
            "message": f"Indexed {file.filename} for user {x_user_id}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_regulation")
async def analyze(
    new_regulation_text: str | None = Form(None),
    regulation_file: UploadFile | None = File(None),
    date_of_law: str | None = Form(None),
    x_user_id: str = Header(...)
):
    final_text = ""

    if regulation_file:
        final_text = extract_text_stream(regulation_file, regulation_file.filename)

    elif new_regulation_text:
        final_text = new_regulation_text.strip()

    if not final_text:
        raise HTTPException(status_code=400, detail="No input provided")

    result = run_arca_pipeline(final_text, x_user_id, date_of_law)
    return result
