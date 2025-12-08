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
# 1. TOOL with Smart Deduplication
# ============================================================
class VectorDBSearchInput(BaseModel):
    query: str
    user_id: str

class VectorDBSearchTool(BaseTool):
    name: str = "vector_db_search"
    description: str = (
        "Search internal policy documents for relevant excerpts. "
        "Returns deduplicated results from unique policy sources. "
        "Provide a descriptive search query and the user_id."
    )
    args_schema: Type[BaseModel] = VectorDBSearchInput

    def _run(self, query: str, user_id: str) -> str:
        try:
            # Fetch more results to handle deduplication
            results = get_db().search(query, user_id, top_k=10)

            if not results and user_id != "default":
                results = get_db().search(query, "default", top_k=10)

            if not results:
                return "NO RESULT"

            # Deduplicate by policy content similarity
            unique_results = []
            seen_fingerprints = set()
            
            for doc_id, text in results:
                # Create fingerprint from first 150 chars
                fingerprint = text[:150].strip().lower()
                
                if fingerprint not in seen_fingerprints:
                    seen_fingerprints.add(fingerprint)
                    unique_results.append((doc_id, text))
                
                # Limit to 3 unique policy excerpts per search
                if len(unique_results) >= 3:
                    break

            if not unique_results:
                return "NO RESULT"

            blocks = []
            for doc_id, text in unique_results:
                cleaned = text.replace("\n", " ").replace("\r", " ").strip()
                cleaned = cleaned[:700]  # More context
                blocks.append(f"POLICY_ID: {doc_id}\nCONTENT: {cleaned}\n---")

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
    relevance_reason: str = Field(description="Why this policy is relevant")

class PolicyReport(BaseModel):
    policies: List[PolicyMatch] = Field(
        default=[], 
        description="List of relevant policies found",
        max_length=5
    )

class RiskItem(BaseModel):
    policy_id: str
    severity: str = Field(description="HIGH, MEDIUM, or LOW")
    divergence_summary: str = Field(description="What conflicts or is missing")
    conflicting_policy_excerpt: str
    new_rule_excerpt: str
    recommended_action: str = Field(description="Specific action to resolve")

class RiskAnalysisReport(BaseModel):
    risks: List[RiskItem] = Field(
        default=[], 
        description="List of identified compliance risks (max 5)",
        max_length=5
    )

class FinalRecommendation(BaseModel):
    recommendation: str = Field(description="Executive summary and action plan")
    compliance_score: str = Field(
        default="UNKNOWN", 
        description="COMPLIANT, NEEDS_UPDATES, or NON_COMPLIANT"
    )

# ============================================================
# 3. Dynamic Crew Builder
# ============================================================
def create_crew(api_key):
    """Creates a fresh crew instance with a specific API Key"""
    
    llm = LLM(
        model="gemini/gemini-2.5-flash-lite",
        temperature=0.1,
        api_key=api_key, 
    )

    # ==== AGENTS ====
    policy_researcher = Agent(
        role="Policy Research Specialist",
        goal=(
            "Find the TOP 3-5 most relevant internal policies related to the regulation. "
            "Focus on quality over quantity."
        ),
        backstory=(
            "You are an expert at finding relevant policies. You search for key topics "
            "separately and identify the MOST important policies, avoiding duplicates."
        ),
        tools=[vector_db_search_tool],
        llm=llm,
        verbose=True,
    )

    compliance_auditor = Agent(
        role="Compliance Risk Analyst",
        goal=(
            "Identify the TOP 5 MOST CRITICAL compliance risks. "
            "Prioritize HIGH severity issues. One risk per policy maximum."
        ),
        backstory=(
            "You are a meticulous auditor who identifies the most critical issues:\n"
            "- HIGH: Direct conflicts or missing critical requirements\n"
            "- MEDIUM: Incomplete implementations\n"
            "- LOW: Minor improvement opportunities\n"
            "You focus on the BIGGEST risks, not every minor detail."
        ),
        llm=llm,
        verbose=True,
    )

    report_generator = Agent(
        role="Executive Compliance Advisor",
        goal="Create a concise, actionable executive summary",
        backstory=(
            "You synthesize findings into clear, prioritized recommendations. "
            "You focus on business impact and next steps."
        ),
        llm=llm,
        verbose=True,
    )

    # ==== TASKS ====
    task_research = Task(
        description=(
            "Analyze the regulation and find the 3-5 most relevant policies.\n\n"
            "NEW REGULATION:\n{new_regulation_text}\n\n"
            "USER_ID: {user_id}\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify 2-3 key topics from the regulation\n"
            "2. Search for each topic using vector_db_search\n"
            "3. Select the TOP 3-5 most relevant unique policies\n"
            "4. Return JSON with policies array (max 5 items)\n\n"
            "IMPORTANT: Return ONLY raw JSON. No markdown, no extra text."
        ),
        expected_output="JSON with max 5 policies",
        agent=policy_researcher,
        tools=[vector_db_search_tool],
        output_pydantic=PolicyReport,
    )

    task_audit = Task(
        description=(
            "Compare regulation vs policies. Find TOP 5 CRITICAL risks.\n\n"
            "NEW REGULATION:\n{new_regulation_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. For each requirement, check if policies comply\n"
            "2. Identify the 5 MOST CRITICAL issues (prioritize HIGH severity)\n"
            "3. Maximum ONE risk per policy (focus on biggest issue per policy)\n"
            "4. If < 5 risks found, that's OK - only report real issues\n"
            "5. Return JSON with risks array (max 5 items)\n\n"
            "IMPORTANT: Return ONLY raw JSON. No markdown. "
            "If fully compliant, return empty risks array."
        ),
        agent=compliance_auditor,
        context=[task_research],
        expected_output="JSON with max 5 risks",
        output_pydantic=RiskAnalysisReport,
    )

    task_recommend = Task(
        description=(
            "Create executive summary based on identified risks.\n"
            "Be concise and action-oriented.\n"
            "IMPORTANT: Return ONLY raw JSON."
        ),
        expected_output="JSON with recommendation and compliance_score",
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
# 4. EXECUTION PIPELINE
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
            
            print(f"🚀 Launching Crew (Attempt {attempt+1}/{max_retries})")
            
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
            
            if "429" in error_msg or "resource_exhausted" in error_msg or "quota" in error_msg:
                print(f"⚠️ Rate Limit! Rotating key...")
                key_manager.rotate_key()
                time.sleep(2)
            else:
                print(f"❌ Error: {e}")
                if attempt < max_retries - 1:
                    print("🔄 Retrying...")
                    time.sleep(1)
                else:
                    traceback.print_exc()
                    break

    # ========================================
    # Process Results
    # ========================================
    risks = risk_report.risks if risk_report else []
    
    # System error fallback
    if not risk_report and last_error:
        risks = [
            RiskItem(
                policy_id="system_error",
                severity="HIGH",
                divergence_summary=f"Analysis failed: {str(last_error)[:200]}",
                conflicting_policy_excerpt="N/A",
                new_rule_excerpt=new_regulation_text[:300],
                recommended_action="Retry analysis or contact support"
            )
        ]
    
    # Fully compliant fallback
    elif len(risks) == 0:
        risks = [
            RiskItem(
                policy_id="compliant",
                severity="LOW",
                divergence_summary="No significant compliance gaps identified",
                conflicting_policy_excerpt="Existing policies align with regulation",
                new_rule_excerpt=new_regulation_text[:300],
                recommended_action="Conduct periodic review to ensure continued compliance"
            )
        ]

    # ⭐ ENFORCE MAXIMUM 5 RISKS ⭐
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
        "compliance_score": reco.compliance_score if (reco and hasattr(reco, 'compliance_score')) else "UNKNOWN"
    }
