import os
import sys
import traceback
import time
from typing import List, Type

from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

from src.vector_db_search import get_db
from src.utils import regulation_id_for, today_iso
from src.gemini_manager import get_key_manager

# ============================================================
# 1. TOOL 
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

            if not results and user_id != "default":
                results = get_db().search(query, "default", top_k=5)

            if not results:
                return "NO RESULT"

            blocks = []
            for doc_id, text in results[:5]:
                cleaned = text.replace("\n", " ").replace("\r", " ").strip()
                cleaned = cleaned[:600]
                blocks.append(f"POLICY_ID: {doc_id}\nEXCERPT: {cleaned}\n---")

            return "\n".join(blocks)
            
        except Exception as e:
            return f"ERROR: {str(e)}"

vector_db_search_tool = VectorDBSearchTool()

# ============================================================
# 2. Pydantic MODELS
# ============================================================
class PolicyMatch(BaseModel):
    policy_id: str
    excerpt: str
    relevance_reason: str = Field(description="Why this policy is relevant to the regulation")

class PolicyReport(BaseModel):
    policies: List[PolicyMatch] = Field(default=[], description="List of relevant policies found")

class RiskItem(BaseModel):
    policy_id: str
    severity: str = Field(description="HIGH, MEDIUM, or LOW")
    divergence_summary: str = Field(description="Clear description of what conflicts or is missing")
    conflicting_policy_excerpt: str
    new_rule_excerpt: str
    recommended_action: str = Field(description="Specific action to resolve the divergence")

class RiskAnalysisReport(BaseModel):
    risks: List[RiskItem] = Field(default=[], description="List of identified compliance risks")

class FinalRecommendation(BaseModel):
    recommendation: str = Field(description="Executive summary and prioritized action plan")
    compliance_score: str = Field(default="UNKNOWN", description="Overall compliance level: COMPLIANT, NEEDS_UPDATES, or NON_COMPLIANT")

# ============================================================
# 3. Dynamic Crew Builder
# ============================================================
def create_crew(api_key):
    """Creates a fresh crew instance with a specific API Key"""
    
    # CHANGED: Switched to gemini-1.5-flash for better JSON stability
    llm = LLM(
        model="gemini/gemini-2.5-flash-lite",
        temperature=0.1, # Lower temperature for stricter output
        api_key=api_key, 
    )

    # ==== AGENTS ====
    policy_researcher = Agent(
        role="Policy Research Specialist",
        goal=(
            "Find all internal policy documents that relate to the new regulation. "
            "Search for policies covering the same topics, requirements, or business areas."
        ),
        backstory=(
            "You are an expert at understanding regulatory requirements and finding "
            "relevant internal policies. You break down regulations into key topics "
            "and search for each topic separately to ensure comprehensive coverage."
        ),
        tools=[vector_db_search_tool],
        llm=llm,
        verbose=True,
    )

    compliance_auditor = Agent(
        role="Compliance Risk Analyst",
        goal=(
            "Identify specific conflicts, gaps, or weaknesses between the new regulation "
            "and existing policies. Focus on HIGH severity risks first."
        ),
        backstory=(
            "You are a meticulous compliance auditor. You compare requirements line-by-line, identifying:\n"
            "- HIGH: Direct conflicts or missing critical requirements\n"
            "- MEDIUM: Incomplete implementations or unclear procedures\n"
            "- LOW: Minor gaps or opportunities for improvement\n"
        ),
        llm=llm,
        verbose=True,
    )

    report_generator = Agent(
        role="Executive Compliance Advisor",
        goal="Create clear, actionable recommendations prioritized by business impact",
        backstory=(
            "You synthesize compliance findings into executive-ready reports. "
            "You prioritize actions, estimate effort, and provide clear next steps."
        ),
        llm=llm,
        verbose=True,
    )

    # ==== TASKS ====
    task_research = Task(
        description=(
            "Analyze the new regulation and search for related internal policies.\n\n"
            "NEW REGULATION:\n{new_regulation_text}\n\n"
            "USER_ID: {user_id}\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify key topics (e.g., 'data retention', 'access controls')\n"
            "2. For EACH topic, use vector_db_search\n"
            "3. Collect all relevant policy excerpts\n"
            "4. Return a JSON list of the most relevant policies found\n\n"
            "IMPORTANT: Output MUST be raw JSON. Do not use markdown code blocks. "
            "Do not add any text before or after the JSON."
        ),
        expected_output="Valid JSON object matching the PolicyReport schema",
        agent=policy_researcher,
        tools=[vector_db_search_tool],
        output_pydantic=PolicyReport,
    )

    task_audit = Task(
        description=(
            "Compare the new regulation against the policies found.\n\n"
            "NEW REGULATION:\n{new_regulation_text}\n\n"
            "POLICIES FOUND: (from previous task)\n\n"
            "INSTRUCTIONS:\n"
            "1. Check coverage for each requirement\n"
            "2. Identify conflicts, gaps, or weaknesses\n"
            "3. Assess severity (HIGH/MEDIUM/LOW)\n"
            "4. Provide specific recommended action\n\n"
            "IMPORTANT: Output MUST be raw JSON. Do not use markdown code blocks. "
            "If policies are COMPLIANT, return an empty risks array."
        ),
        agent=compliance_auditor,
        context=[task_research],
        expected_output="Valid JSON object matching the RiskAnalysisReport schema",
        output_pydantic=RiskAnalysisReport,
    )

    task_recommend = Task(
        description=(
            "Create an executive summary and action plan based on the risks identified.\n"
            "IMPORTANT: Output MUST be raw JSON. Do not use markdown code blocks."
        ),
        expected_output="Valid JSON object matching the FinalRecommendation schema",
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
# 4. EXECUTION PIPELINE WITH RETRY
# ============================================================
def run_arca_pipeline(new_regulation_text: str, user_id: str, date_of_law: str | None = None):
    key_manager = get_key_manager()
    max_retries = 3
    
    risk_report = None
    reco = None
    last_error = None

    for attempt in range(max_retries):
        try:
            current_key = key_manager.get_current_key()
            crew, task_audit, task_recommend = create_crew(current_key)
            
            print(f"🚀 Launching Crew (Attempt {attempt+1}/{max_retries}) using Gemini 1.5 Flash")
            
            crew.kickoff(inputs={
                "new_regulation_text": new_regulation_text,
                "user_id": user_id,
            })

            risk_report = task_audit.output.pydantic
            reco = task_recommend.output.pydantic
            break

        except Exception as e:
            error_msg = str(e).lower()
            last_error = e
            
            # Check for Rate Limits (429 or Resource Exhausted)
            if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
                print(f"⚠️ Rate Limit Hit! Rotating key...")
                key_manager.rotate_key()
                time.sleep(2)
            else:
                print(f"❌ Error during execution: {e}")
                # Retry even on validation errors, sometimes it's just a bad generation
                if attempt < max_retries - 1:
                    print("🔄 Retrying due to validation/parsing error...")
                    time.sleep(1)
                else:
                    traceback.print_exc()
                    break

    # ========================================
    # Process Results & Fallback
    # ========================================
    risks = risk_report.risks if risk_report else []
    
    # If total failure
    if not risk_report and last_error:
        risks = [
            RiskItem(
                policy_id="system_error",
                severity="HIGH",
                divergence_summary=f"Analysis failed: {str(last_error)[:200]}",
                conflicting_policy_excerpt="N/A",
                new_rule_excerpt=new_regulation_text[:300],
                recommended_action="Please try again. If the issue persists, check API quotas."
            )
        ]
    
    # If compliant (no risks found)
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

    # Limit to 10 items to prevent UI overload
    risks = risks[:10]

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