import os
from typing import List, Type

from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

from src.vector_db_search import get_db
from src.utils import regulation_id_for, today_iso


# ============================================================
# 1. LLM CONFIGURATION (free / OpenRouter / Gemini, etc.)
# ============================================================
llm = LLM(
    model="gemini/gemini-2.5-flash",  # you can swap for openrouter / local later
    temperature=0.2,
    api_key=os.environ.get("GEMINI_API_KEY"),
)


# ============================================================
# 2. TOOL: vector_db_search (as required by ARCA)
# ============================================================
class VectorDBSearchInput(BaseModel):
    query: str
    user_id: str = "default"  # SaaS extension, default single-tenant


class VectorDBSearchTool(BaseTool):
    name: str = "vector_db_search"
    description: str = (
        "Interroge la base de données vectorielle des politiques internes et "
        "retourne les 5 extraits les plus pertinents au format texte concaténé. "
        "Chaque bloc contient POLICY_ID et EXCERPT séparés par '---'."
    )
    args_schema: Type[BaseModel] = VectorDBSearchInput

    def _run(self, query: str, user_id: str = "default") -> str:
        results = get_db().search(query, user_id, top_k=5)

        # Fallback on 'default' corpus if user-specific data is empty
        if not results and user_id != "default":
            results = get_db().search(query, "default", top_k=5)

        if not results:
            return "NO RESULT"

        blocks = []
        for doc_id, text in results[:5]:
            snippet = text[:300].replace("\n", " ").strip()
            blocks.append(f"POLICY_ID: {doc_id}\nEXCERPT: {snippet}\n---")

        # ARCA spec: return one concatenated string
        return "\n".join(blocks)


vector_db_search_tool = VectorDBSearchTool()


# ============================================================
# 3. MODELS (JSON schema for ARCA final output)
# ============================================================
class PolicyMatch(BaseModel):
    policy_id: str
    excerpt: str


class PolicyReport(BaseModel):
    policies: List[PolicyMatch] = Field(
        min_length=1,
        max_length=5,
        description="Liste des politiques internes pertinentes.",
    )


class RiskItem(BaseModel):
    policy_id: str
    severity: str  # HIGH, MEDIUM, LOW
    divergence_summary: str
    conflicting_policy_excerpt: str
    new_rule_excerpt: str


class RiskAnalysisReport(BaseModel):
    risks: List[RiskItem] = Field(
        min_length=1,
        description="Liste des risques de conflit entre politique interne et règlement.",
    )


class FinalRecommendation(BaseModel):
    recommendation: str


# ============================================================
# 4. AGENTS
# ============================================================
policy_researcher = Agent(
    role="Policy Researcher",
    goal=(
        "Retrouver les 5 extraits de politiques internes les plus pertinents "
        "par rapport au nouveau texte réglementaire, en utilisant uniquement le RAG."
    ),
    backstory=(
        "Bibliothécaire numérique expert en recherche sémantique et en RAG. "
        "Tu ne fais que de la récupération de contexte factuel."
    ),
    tools=[vector_db_search_tool],
    llm=llm,
    verbose=True,
)

compliance_auditor = Agent(
    role="Compliance Auditor",
    goal=(
        "Analyser les extraits de politiques internes et identifier les conflits "
        "de conformité avec le nouveau texte réglementaire."
    ),
    backstory=(
        "Auditeur juridique méticuleux spécialisé en conformité réglementaire. "
        "Tu ne disposes d'aucun outil externe : seulement ton raisonnement."
    ),
    llm=llm,
    verbose=True,
)

report_generator = Agent(
    role="Report Generator",
    goal=(
        "Transformer l'analyse brute de l'auditeur en un rapport JSON parfaitement "
        "structuré et lisible par machine."
    ),
    backstory=(
        "Spécialiste de la donnée et de la rédaction technique. "
        "Tu produis une sortie finale propre, sans inventer de nouveaux faits."
    ),
    llm=llm,
    verbose=True,
)


# ============================================================
# 5. TASKS (séquentiel ARCA)
# ============================================================
task_research = Task(
    description=(
        "Tu es l'agent Policy Researcher.\n"
        "Ton objectif : récupérer les 5 extraits de politiques internes les plus pertinents.\n\n"
        "Étapes OBLIGATOIRES :\n"
        "1. Appeler l'outil 'vector_db_search' avec comme query le texte complet "
        "de la nouvelle réglementation (fourni via la variable {new_regulation_text}).\n"
        "2. L'outil renvoie un texte avec plusieurs blocs au format :\n"
        "   POLICY_ID: <id>\n"
        "   EXCERPT: <texte>\n"
        "   ---\n\n"
        "3. À partir de ce texte, construis un JSON STRICT de la forme :\n"
        "{\n"
        '  "policies": [\n'
        '    {"policy_id": "<id>", "excerpt": "<texte>"},\n'
        "    ... (maximum 5 éléments)\n"
        "  ]\n"
        "}\n\n"
        "Contraintes :\n"
        "- Tu NE dois PAS inventer de nouvelles politiques.\n"
        "- policy_id reprend exactement la valeur POLICY_ID.\n"
        "- excerpt est copié fidèlement depuis EXCERPT.\n"
    ),
    expected_output="Un JSON structuré avec une clé 'policies' contenant 1 à 5 objets {policy_id, excerpt}.",
    agent=policy_researcher,
    output_pydantic=PolicyReport,
)

task_audit = Task(
    description=(
        "Tu es l'agent Compliance Auditor.\n"
        "Tu reçois les politiques internes pertinentes (via la sortie de l'agent précédent) "
        "et le texte du nouveau règlement :\n"
        "\"\"\"\n{new_regulation_text}\n\"\"\"\n\n"
        "OBJECTIF : Pour chaque politique, déterminer s'il existe un conflit avec le nouveau règlement.\n\n"
        "Pour chaque policy, tu dois produire UN objet de risque contenant STRICTEMENT :\n"
        "- policy_id : l'identifiant de la politique (copié depuis Policy Researcher)\n"
        "- severity : HIGH, MEDIUM ou LOW (en majuscules)\n"
        "- divergence_summary : un résumé clair du conflit ou de la différence\n"
        "- conflicting_policy_excerpt : extrait EXACT de la politique interne\n"
        "- new_rule_excerpt : extrait EXACT du nouveau règlement\n\n"
        "Tu ne dois PAS inventer de texte de politique ni de règlement.\n"
        "Tu te bases uniquement sur :\n"
        " - les excerpts de politiques reçus\n"
        " - le texte du règlement fourni ({new_regulation_text})\n\n"
        "La sortie DOIT être un JSON STRICT de la forme :\n"
        "{\n"
        '  "risks": [\n'
        '    {\n'
        '      "policy_id": "...",\n'
        '      "severity": "HIGH|MEDIUM|LOW",\n'
        '      "divergence_summary": "...",\n'
        '      "conflicting_policy_excerpt": "...",\n'
        '      "new_rule_excerpt": "..."\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
    ),
    expected_output="Un JSON avec une clé 'risks' contenant au moins 1 objet de risque complet.",
    agent=compliance_auditor,
    context=[task_research],
    output_pydantic=RiskAnalysisReport,
)

task_recommend = Task(
    description=(
        "Tu es l'agent Report Generator.\n"
        "Tu reçois la liste structurée des risques (clé 'risks') issus de l'auditeur.\n\n"
        "Ta mission : produire UNE SEULE recommandation globale, au format JSON :\n"
        '{ "recommendation": "<texte>" }\n\n'
        "La recommandation doit être :\n"
        "- actionnable (on doit pouvoir la mettre en œuvre)\n"
        "- claire et concise\n"
        "- liée directement aux risques identifiés (pas de texte générique)\n"
        "Exemples de bonne forme :\n"
        "- \"Mettre à jour la politique de mots de passe pour imposer le renouvellement tous les 90 jours d'ici Q4 2025.\"\n"
        "- \"Interdire explicitement les clés USB non chiffrées dans la PSSI et diffuser la mise à jour à tous les employés.\"\n"
    ),
    expected_output='Un JSON avec une clé unique "recommendation".',
    agent=report_generator,
    context=[task_audit],
    output_pydantic=FinalRecommendation,
)


# ============================================================
# 6. CREW (séquentiel obligatoire)
# ============================================================
crew = Crew(
    agents=[policy_researcher, compliance_auditor, report_generator],
    tasks=[task_research, task_audit, task_recommend],
    process=Process.sequential,  # ARCA: flux séquentiel obligatoire
    verbose=True,
)


# ============================================================
# 7. PIPELINE EXECUTION (API-facing)
# ============================================================
def run_arca_pipeline(new_regulation_text: str, user_id: str, date_of_law: str | None = None):
    """
    Fonction appelée par l'API FastAPI.
    Elle orchestre le crew et produit le JSON final conforme à la spécification ARCA.
    """
    inputs = {
        "new_regulation_text": new_regulation_text,
        "user_id": user_id,
    }

    crew.kickoff(inputs=inputs)

    risk_report = task_audit.output.pydantic
    reco = task_recommend.output.pydantic

    risks = risk_report.risks

    # Normalisation basique des niveaux de sévérité
    for r in risks:
        r.severity = r.severity.upper().strip()

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
        "recommendation": reco.recommendation,
    }
