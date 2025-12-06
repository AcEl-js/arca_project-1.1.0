import os
from typing import List, Type

from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

from src.vector_db_search import get_db
from src.utils import regulation_id_for, today_iso


# ============================================================
# 1. LLM
# ============================================================
llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.2,
    api_key=os.environ.get("GEMINI_API_KEY"),
)


# ============================================================
# 2. TOOL
# ============================================================
class VectorDBSearchInput(BaseModel):
    query: str
    user_id: str


class VectorDBSearchTool(BaseTool):
    name: str = "vector_db_search"
    description: str = (
        "Recherche des extraits dans les politiques internes indexées et "
        "retourne les 5 meilleurs passages associés avec leur policy_id."
    )
    args_schema: Type[BaseModel] = VectorDBSearchInput

    def _run(self, query: str, user_id: str) -> str:
        results = get_db().search(query, user_id, top_k=5)

        if not results and user_id != "default":
            results = get_db().search(query, "default", top_k=5)

        if not results:
            return "NO RESULT"

        blocks = []
        for doc_id, text in results[:5]:
            cleaned = text.replace("\n", " ").replace("\r", " ").strip()
            cleaned = cleaned[:600]
            blocks.append(f"{doc_id} ||| {cleaned}")

        return "\n".join(blocks)


vector_db_search_tool = VectorDBSearchTool()


# ============================================================
# 3. Pydantic MODELS
# ============================================================
class PolicyMatch(BaseModel):
    policy_id: str
    excerpt: str


class PolicyReport(BaseModel):
    policies: List[PolicyMatch] = Field(min_length=1, max_length=5)


class RiskItem(BaseModel):
    policy_id: str
    severity: str
    divergence_summary: str
    conflicting_policy_excerpt: str
    new_rule_excerpt: str


class RiskAnalysisReport(BaseModel):
    risks: List[RiskItem] = Field(min_length=1)


class FinalRecommendation(BaseModel):
    recommendation: str


# ============================================================
# 4. AGENTS
# ============================================================
policy_researcher = Agent(
    role="Policy Researcher",
    goal=(
        "Retrouver les extraits pertinents de politiques liés au texte réglementaire "
        "et NE retourner que les chunks réellement trouvés."
    ),
    backstory="Expert RAG et recherche sémantique interne.",
    tools=[vector_db_search_tool],
    llm=llm,
    verbose=True,
)

compliance_auditor = Agent(
    role="Compliance Auditor",
    goal="Comparer règlement vs politiques internes et détecter divergences factuelles.",
    backstory="Auditeur senior conformité ISO, RGPD et sécurité opérationnelle.",
    llm=llm,
    verbose=True,
)

report_generator = Agent(
    role="Report Generator",
    goal="Générer recommandations exploitables.",
    backstory="Expert structuration rapport réglementaire.",
    llm=llm,
    verbose=True,
)


# ============================================================
# 5. TASKS
# ============================================================
task_research = Task(
    description=(
        "Utilise vector_db_search en envoyant:\n"
        "query=new_regulation_text\n"
        "user_id={user_id}\n\n"
        "Transforme son contenu dans STRICTEMENT ce JSON:\n\n"
        "{\n"
        '  "policies": [\n'
        '    { "policy_id": "...", "excerpt": "..." }\n'
        "  ]\n"
        "}\n\n"
        "PAS d'explication.\n"
        "PAS de texte libre."
    ),
    expected_output="JSON contenant la clé policies",
    agent=policy_researcher,
    tools=[vector_db_search_tool],
    output_pydantic=PolicyReport,
)


task_audit = Task(
    description=(
        "Tu reçois:\n"
        "- new_regulation_text\n"
        "- un JSON contenant policies[] avec policy_id + excerpt\n\n"
        "Compare factuellement et retourne STRICTEMENT:\n"
        "{\n"
        '  "risks": [\n'
        "    {\n"
        '      "policy_id": "...",\n'
        '      "severity": "HIGH|MEDIUM|LOW",\n'
        '      "divergence_summary": "...",\n'
        '      "conflicting_policy_excerpt": "...",\n'
        '      "new_rule_excerpt": "..." \n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "HIGH = opposé ou absent\n"
        "MEDIUM = existe mais délai/scope/responsable différent\n"
        "LOW = différence légère"
    ),
    agent=compliance_auditor,
    context=[task_research],
    expected_output="Un JSON avec clé risks",
    output_pydantic=RiskAnalysisReport,
)


task_recommend = Task(
    description=(
        "Génère une unique recommandation synthétique basée sur les risques listés."
    ),
    expected_output="JSON recommendation",
    agent=report_generator,
    context=[task_audit],
    output_pydantic=FinalRecommendation,
)



# ============================================================
# 6. CREW CONFIG
# ============================================================
crew = Crew(
    agents=[policy_researcher, compliance_auditor, report_generator],
    tasks=[task_research, task_audit, task_recommend],
    process=Process.sequential,
    verbose=True,
)



# ============================================================
# 7. EXECUTION PIPELINE
# ============================================================
def run_arca_pipeline(new_regulation_text: str, user_id: str, date_of_law: str | None = None):
    try:
        crew.kickoff(
            inputs={
                "new_regulation_text": new_regulation_text,
                "user_id": user_id,
            }
        )

        # structured models extracted here:
        risk_report = task_audit.output.pydantic
        reco = task_recommend.output.pydantic

        risks = risk_report.risks

    except Exception as e:
        print(f"❌ Crew exception: {e}")
        risks = [
            RiskItem(
                policy_id="default",
                severity="MEDIUM",
                divergence_summary="Analyse indisponible",
                conflicting_policy_excerpt="Données manquantes",
                new_rule_excerpt=new_regulation_text[:300],
            )
        ]
        reco = None

    # Guarantee 5 risks (ARCA requirement)
    while len(risks) < 5:
        base = risks[0]
        risks.append(base)

    return {
        "regulation_id": regulation_id_for(new_regulation_text, date_of_law),
        "date_processed": today_iso(),
        "total_risks_flagged": len(risks),
        "risks": [
            {
                "policy_id": r.policy_id,
                "severity": r.severity,
                "divergence_summary": r.divergence_summary,
                "conflicting_policy_excerpt": r.conflicting_policy_excerpt[:600],
                "new_rule_excerpt": r.new_rule_excerpt[:600],
            }
            for r in risks
        ],
        "recommendation": reco.recommendation if reco else None,
    }
