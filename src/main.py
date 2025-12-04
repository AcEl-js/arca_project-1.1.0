from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.agents import load_gemini_keys

load_gemini_keys()

import os


from src.agents import run_arca_pipeline, init_gemini_llm

load_dotenv()

# Initialize Gemini globally
init_gemini_llm(os.getenv("GEMINI_API_KEY"))

app = FastAPI(title="ARCA - Regulatory Analysis API")


class AnalyzeRequest(BaseModel):
    new_regulation_text: str
    date_of_law: str | None = None


@app.post("/analyze_regulation")
async def analyze(req: AnalyzeRequest):
    if not req.new_regulation_text.strip():
        raise HTTPException(status_code=400, detail="new_regulation_text is required")

    result = run_arca_pipeline(req.new_regulation_text, req.date_of_law)
    return result
