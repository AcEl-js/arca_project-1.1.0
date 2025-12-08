import os
import sys
import traceback
import time
import re
from typing import List, Type

from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

from src.vector_db_search import get_db
from src.utils import regulation_id_for, today_iso
from src.gemini_manager import get_key_manager

# ============================================================
# 1. TOOL DEFINITION
# ============================================================
class VectorDBSearchInput(BaseModel):
    query: str
    user_id: str

class VectorDBSearchTool(BaseTool):
    name: str = "vector_db_search"
    description: str = (
        "Search internal policy documents for relevant excerpts. "
        "Provide a search query describing what you're looking for, and the user_id. "
        "Returns the top 5 most relevant policy excerpts with their IDs."
    )
    args_schema: Type[BaseModel] = VectorDBSearchInput

    def _run(self, query: str, user_id: str) -> str:
        try:
            results = get_db().search(query, user_id, top_k=5)

            # Fallback to default user if specific user has no data
            if not results and user_id != "default":
                results = get_db().search(query, "default", top_k=5)

            if not results:
                return "NO RESULT"

            blocks = []
            for doc_id, text in results[:5]:
                # Clean text to save tokens and improve readability
                cleaned = text.replace("\n", " ").replace("\r", " ").strip()
                cleaned = cleaned[:600]
                blocks.append(f"POLICY_ID: {doc_id}\nEXCERPT: {cleaned}\n---")

            return "\n".join(blocks)
            
        except Exception as e:
            return f"ERROR: {str(e)}"

vector_db_search_tool = VectorDBSearchTool()

# ============================================================
# 2. PYDANTIC MODELS (OUTPUT STRUCTURE)
# ============================================================
class PolicyMatch(BaseModel):
    policy_id: str
    excerpt: str
    relevance_reason: str = Field(description="Why this policy is relevant")

class PolicyReport(BaseModel):
    policies: List[PolicyMatch] = Field(default=[], description="List of relevant policies found")

class RiskItem(BaseModel):
    policy_id: str
    severity: str = Field(description="HIGH, MEDIUM, or LOW")
    divergence_summary: str = Field(description="Clear description of the conflict")
    conflicting_policy_excerpt: str
    new_rule_excerpt: str
    recommended_action: str = Field(description="Specific action to resolve the divergence")

class RiskAnalysisReport(BaseModel):
    risks: List[RiskItem] = Field(default=[], description="List of identified compliance risks")

class FinalRecommendation(BaseModel):
    recommendation: str = Field(description="Executive summary and prioritized action plan")
    compliance_score: str = Field(default="UNKNOWN", description="COMPLIANT, NEEDS_UPDATES, or NON_COMPLIANT")

# ============================================================
# 3. DYNAMIC CREW BUILDER
# ============================================================
def create_crew(api_key):
    """Creates a fresh crew instance with a specific API Key"""
    
    # Use Gemini 1.5 Flash for best balance of speed, cost, and JSON instruction following
    llm = LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.1,  # Low temperature for strict adherence to formats
        api_key=api_key, 
    )

    # ==== AGENTS ====
    policy_researcher = Agent(
        role="Policy Research Specialist",
        goal="Find relevant internal policy documents using semantic search.",
        backstory=(
            "You are an expert at breaking down regulations into key topics and "
            "finding the exact internal policy clauses that apply."
        ),
        tools=[vector_db_search_tool],
        llm=llm,
        verbose=True,
    )

    compliance_auditor = Agent(
        role="Compliance Risk Analyst",
        goal="Identify specific conflicts (High/Medium/Low) between regulation and policy.",
        backstory=(
            "You are a meticulous auditor. You compare texts line-by-line to find "
            "missing requirements, contradictory rules, or vague definitions."
        ),
        llm=llm,
        verbose=True,
    )

    report_generator = Agent(
        role="Executive Compliance Advisor",
        goal="Generate a structured action plan and executive summary.",
        backstory="Synthesizes technical findings into business-ready recommendations.",
        llm=llm,
        verbose=True,
    )

    # ==== TASKS ====
    # We explicitly forbid Markdown code blocks to prevent JSON parsing errors
    
    task_research = Task(
        description=(
            "Analyze the new regulation and search for related internal policies.\n"
            "NEW REGULATION: {new_regulation_text}\n"
            "USER_ID: {user_id}\n\n"
            "1. Identify key topics.\n"
            "2. Use vector_db_search for each topic.\n"
            "3. Return a JSON list of matches.\n\n"
            "CRITICAL: Output MUST be raw JSON only. "
            "Do NOT use markdown blocks (```json). "
            "Do NOT add introductory text."
        ),
        expected_output="Raw JSON object matching PolicyReport schema",
        agent=policy_researcher,
        tools=[vector_db_search_tool],
        output_pydantic=PolicyReport,
    )

    task_audit = Task(
        description=(
            "Compare regulation vs policies.\n"
            "NEW REGULATION: {new_regulation_text}\n"
            "POLICIES FOUND: (from previous task)\n\n"
            "Identify risks (HIGH/MEDIUM/LOW).\n"
            "If no conflicts, return empty list.\n\n"
            "CRITICAL: Output MUST be raw JSON only. "
            "Do NOT use markdown blocks (```json). "
            "Do NOT add text before or after the JSON."
        ),
        agent=compliance_auditor,
        context=[task_research],
        expected_output="Raw JSON object matching RiskAnalysisReport schema",
        output_pydantic=RiskAnalysisReport,
    )

    task_recommend = Task(
        description=(
            "Create executive summary and action plan.\n"
            "Based on identified risks.\n\n"
            "CRITICAL: Output MUST be raw JSON only. "
            "Do NOT use markdown blocks (```json)."
        ),
        expected_output="Raw JSON object matching FinalRecommendation schema",
        agent=report_generator,
        context=[task_audit],
        output_pydantic=FinalRecommendation,
    )

    crew = Crew(
        agents=[policy_researcher, compliance_auditor, report_generator],
        tasks=[task_research, task_audit, task_recommend],
        process=Process.sequential,
        verbose=True,
    )
    
    return crew, task_audit, task_recommend

# ============================================================
# 4. EXECUTION PIPELINE WITH RETRY & LIMITS
# ============================================================
def run_arca_pipeline(new_regulation_text: str, user_id: str, date_of_law: str | None = None):
    key_manager = get_key_manager()
    max_retries = 3
    
    risk_report = None
    reco = None
    last_error = None

    for attempt in range(max_retries):
        try:
            # 1. Get current valid key
            current_key = key_manager.get_current_key()
            
            # 2. Build Crew
            crew, task_audit, task_recommend = create_crew(current_key)
            
            print(f"🚀 Launching Crew (Attempt {attempt+1}/{max_retries})")
            
            # 3. Run
            crew.kickoff(inputs={
                "new_regulation_text": new_regulation_text,
                "user_id": user_id,
            })

            # 4. Safe Extraction
            if hasattr(task_audit.output, 'pydantic'):
                risk_report = task_audit.output.pydantic
            
            if hasattr(task_recommend.output, 'pydantic'):
                reco = task_recommend.output.pydantic
            
            # If we got results, successful break
            if risk_report:
                break

        except Exception as e:
            error_msg = str(e).lower()
            last_error = e
            print(f"⚠️ Error in attempt {attempt+1}: {e}")
            
            # Rate Limit Handling (429)
            if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
                print("🔄 Rate limit hit. Rotating key...")
                key_manager.rotate_key()
                time.sleep(2)
            
            # JSON/Validation Error Handling (Bad LLM output)
            elif "validation error" in error_msg or "json" in error_msg:
                print("🔄 JSON parsing error. Retrying generation...")
                time.sleep(1)
            
            # Generic error
            else:
                traceback.print_exc()
                time.sleep(1)

    # ========================================
    # Process Results & Fallback
    # ========================================
    risks = risk_report.risks if risk_report else []
    
    # Case A: Total Failure
    if not risk_report and last_error:
        risks = [
            RiskItem(
                policy_id="system_error",
                severity="HIGH",
                divergence_summary=f"Analysis Error: {str(last_error)[:200]}",
                conflicting_policy_excerpt="N/A",
                new_rule_excerpt=new_regulation_text[:300],
                recommended_action="Please try again later or check input size."
            )
        ]
    
    # Case B: Compliant (No risks found)
    elif len(risks) == 0:
        risks = [
            RiskItem(
                policy_id="compliant",
                severity="LOW",
                divergence_summary="No significant compliance gaps identified.",
                conflicting_policy_excerpt="Policies align with the new regulation.",
                new_rule_excerpt=new_regulation_text[:300],
                recommended_action="Conduct periodic review."
            )
        ]

    # Case C: Limit Results to Max 5 (Requested feature)
    risks = risks[:5]

    return {
        "regulation_id": regulation_id_for(new_regulation_text, date_of_law),
        "date_processed": today_iso(),
        "total_risks_flagged": len(risks),
        "risks": [
            {
                "policy_id": r.policy_id,
                "severity": r.severity,
                "divergence_summary": r.divergence_summary,
                "conflicting_policy_excerpt": r.conflicting_policy_excerpt[:800],
                "new_rule_excerpt": r.new_rule_excerpt[:800],
                "recommended_action": r.recommended_action[:500] if hasattr(r, 'recommended_action') else ""
            }
            for r in risks
        ],
        "recommendation": reco.recommendation if reco else "Analysis completed.",
        "compliance_score": reco.compliance_score if reco and hasattr(reco, 'compliance_score') else "UNKNOWN"
    }
