# ARCA — Agile Regulatory Compliance Agent

ARCA (Agile Regulatory Compliance Agent) is an automated system for analyzing new regulations and comparing them against internal company policies. It uses a vector database for semantic retrieval, a multi-agent reasoning pipeline, and Google's Gemini model for legal conflict assessment.

---

## 1. Overview

ARCA streamlines the compliance review process by:

- Indexing internal policies using a vector database (ChromaDB)
- Retrieving the most relevant policy excerpts using semantic search
- Comparing each excerpt to a new regulation using a reasoning model
- Producing a structured compliance report in JSON format

The system is built around a three-agent pipeline:

1. **Policy Researcher** – Retrieves relevant internal policies  
2. **Compliance Auditor** – Assesses conflict severity using Gemini  
3. **Report Generator** – Produces the final JSON compliance report  

A FastAPI server exposes a single endpoint for analyzing regulations.

---

## 2. Features

- Semantic policy retrieval (RAG)
- Automatic document chunking and embedding
- Conflict classification (HIGH, MEDIUM, LOW)
- Machine-readable compliance reports
- FastAPI endpoint for external integrations
- CLI for vector database management

---

## 3. Project Structure
arca_project/
│
├── src/
│ ├── main.py # FastAPI entry point
│ ├── agents.py # Multi-agent compliance workflow
│ ├── vector_db_search.py # Vector DB utilities and CLI
│ ├── utils.py # Hashing, timestamps, constants
│ └── init.py
│
├── data/
│ └── policies/ # Internal policy documents (.md, .txt)
│
├── chroma_db/ # Persisted vector database
│
├── requirements.txt
├── .gitignore
└── README.md


---

## 4. Installation

### Step 1: Clone the repository

git clone <your-repository-url>

### Step 2: Create a virtual environment

python3 -m venv venv
source venv/bin/activate



### Step 3: Install dependencies

pip install -r requirements.txt


### Step 4: Configure API keys

Create a `.env` file in the project root:
GEMINI_API_KEY=your_api_key_here
CHROMA_DIR=./chroma_db


---

## 5. Preparing the Policy Database

Place your policy files (`.md` or `.txt`) inside:

data/policies/


Then build the vector database: python3 -m src.vector_db_search --build data/policies



Expected output: Indexed X chunks.


---

## 6. Running the API

Start the FastAPI server:

uvicorn src.main:app --reload


API documentation will be available at: http://127.0.0.1:8000/docs



---

## 7. API Usage

### Endpoint

POST /analyze_regulation


### Example request body

```json
{
  "new_regulation_text": "Les entreprises doivent conserver les logs pendant trois ans.",
  "date_of_law": "2025-01-01"
}

{
  "regulation_id": "48fd582f...",
  "date_processed": "2025-12-03",
  "total_risks_flagged": 2,
  "risks": [
    {
      "policy_id": "RetentionPolicy-chunk-3",
      "severity": "MEDIUM",
      "divergence_summary": "Internal policy keeps logs for 1 year while the regulation requires 3 years.",
      "conflicting_policy_excerpt": "...",
      "new_rule_excerpt": "..."
    }
  ],
  "recommendation": "Review HIGH severity conflicts with the legal team."
}






