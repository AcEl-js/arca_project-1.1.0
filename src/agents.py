import json
import os
import google.generativeai as genai
from src.vector_db_search import get_db
from src.utils import regulation_id_for, today_iso

# =========================================================
#  MULTI-KEY GEMINI CLIENT WITH AUTOMATIC ROTATION
# =========================================================

GEMINI_KEYS = []
CURRENT_KEY_INDEX = 0


def load_gemini_keys():
    """Load GEMINI_API_KEY_1 ... GEMINI_API_KEY_10 dynamically."""
    global GEMINI_KEYS

    GEMINI_KEYS = []
    for i in range(1, 21):  # Supports up to 20 keys
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            GEMINI_KEYS.append(key)

    if not GEMINI_KEYS:
        raise ValueError("No Gemini API keys were found in environment variables.")

    configure_gemini(GEMINI_KEYS[0])


def configure_gemini(key):
    """Configure Gemini with a specific key."""
    genai.configure(api_key=key)


def rotate_key():
    """Switch to the next Gemini key in the list."""
    global CURRENT_KEY_INDEX

    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(GEMINI_KEYS)
    new_key = GEMINI_KEYS[CURRENT_KEY_INDEX]
    configure_gemini(new_key)
    return new_key


def safe_gemini_call(model, prompt):
    """
    Executes a Gemini API call safely.
    Automatically rotates API keys if an error occurs.
    """

    for attempt in range(len(GEMINI_KEYS)):
        try:
            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            error_msg = str(e)

            if any(code in error_msg for code in ["quota", "429", "403", "exceeded", "API key"]):
                rotate_key()
                continue

            if "500" in error_msg or "503" in error_msg:
                rotate_key()
                continue

            raise e

    raise RuntimeError("All Gemini API keys failed.")


# =========================================================
#  POLICY RESEARCHER
# =========================================================
class PolicyResearcher:
    def __init__(self):
        self.db = get_db()

    def run(self, new_text: str):
        results = self.db.search(new_text, top_k=5)
        return [{"policy_id": pid, "excerpt": doc} for pid, doc in results]


# =========================================================
#  COMPLIANCE AUDITOR
# =========================================================
class ComplianceAuditor:
    def analyze_one(self, policy_excerpt: str, new_rule: str):

        schema = {
            "type": "OBJECT",
            "properties": {
                "severity": {"type": "STRING", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "justification": {"type": "STRING"}
            },
            "required": ["severity", "justification"]
        }

        generation_cfg = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema
        )

        model = genai.GenerativeModel(
            "models/gemini-2.0-pro",
            generation_config=generation_cfg
        )

        prompt = f"""
Compare the following:

Policy Excerpt:
{policy_excerpt}

New Regulation:
{new_rule}

Determine if there is a conflict. Classify: HIGH, MEDIUM, or LOW.
Justify briefly in one sentence.
"""

        try:
            response_text = safe_gemini_call(model, prompt)
            return json.loads(response_text)

        except Exception as e:
            return {
                "severity": "LOW",
                "justification": f"API error: {str(e)}"
            }

    def run(self, items, new_text):
        out = []
        for it in items:
            analysis = self.analyze_one(it["excerpt"], new_text)

            out.append({
                "policy_id": it["policy_id"],
                "severity": analysis.get("severity", "LOW"),
                "divergence_summary": analysis.get("justification", "No justification provided"),
                "conflicting_policy_excerpt": it["excerpt"],
                "new_rule_excerpt": new_text[:500]
            })
        return out


# =========================================================
#  REPORT GENERATOR
# =========================================================
class ReportGenerator:
    def build_report(self, reg_text: str, date_of_law: str, risks):
        return {
            "regulation_id": regulation_id_for(reg_text, date_of_law),
            "date_processed": today_iso(),
            "total_risks_flagged": len(risks),
            "risks": risks,
            "recommendation": (
                "Immediate review required for HIGH severity conflicts. "
                "MEDIUM conflicts should be assessed by legal counsel. "
                "LOW conflicts typically require no action unless part of a wider pattern."
            )
        }


# =========================================================
#  ORCHESTRATOR
# =========================================================
def run_arca_pipeline(new_regulation_text: str, date_of_law: str = None):
    researcher = PolicyResearcher()
    auditor = ComplianceAuditor()
    generator = ReportGenerator()

    top_policies = researcher.run(new_regulation_text)
    risks = auditor.run(top_policies, new_regulation_text)
    return generator.build_report(new_regulation_text, date_of_law, risks)
