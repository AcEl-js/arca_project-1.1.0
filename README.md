# ARCA — Agent de Conformité Réglementaire Agile

## 🧠 Introduction
ARCA est un système intelligent destiné à analyser automatiquement un nouveau règlement et à identifier les conflits potentiels avec les politiques internes d'une entreprise.

Ce système répond à un besoin réel : automatiser la veille réglementaire et accélérer la prise de décision juridique.

ARCA fonctionne entièrement sans intervention humaine grâce à :
- une base de connaissances vectorielle des politiques internes,
- un système RAG (Retrieval-Augmented Generation),
- un ensemble d'agents IA spécialisés travaillant de manière séquentielle.

---

# 🏗 Architecture Fonctionnelle

### 🔸 Agent 1 — Policy Researcher
- Utilise exclusivement l’outil `vector_db_search`
- Trouve les 5 extraits pertinents dans les politiques internes
- Ne génère rien : il récupère factuellement

### 🔸 Agent 2 — Compliance Auditor
- Compare les politiques récupérées au règlement soumis
- Classe les risques en `HIGH`, `MEDIUM` ou `LOW`
- Analyse uniquement avec le LLM (pas d’outils)

### 🔸 Agent 3 — Report Generator
- Structure le résultat dans un JSON lisible par machine
- Aucune génération de contenu nouveau
- Assemble uniquement

---

## 🧬 Workflow Séquentiel

```
User Input Regulation
        ↓
Policy Researcher (RAG Search)
        ↓
Compliance Auditor (Conflict Detection)
        ↓
Report Generator (JSON Formatting)
        ↓
Final JSON Output
```

Cette structure respecte le flux prévu dans le document ARCA.

---

# 📚 Phase 1 — Base de Connaissances

### Format attendu des documents
📌 PDF ou Markdown  
📌 10 à 15 fichiers  
📌 < 5Mo total

### Chunking appliqué (obligatoire ARCA)
```
chunk_size = 400
chunk_overlap = 50
```

### Embedding utilisé
```
model = all-MiniLM-L6-v2
```

### Base vectorielle
```
ChromaDB (persistante en local)
```

Les documents sont ajoutés via l'API `/upload_policy`.

---

# 🚀 Phase 2 — Crew d’agents IA

3 agents spécialisés implémentés avec CrewAI :

| Agent | Rôle | Utilisation d’outil ? |
|---|---|---|
| Policy Researcher | Recherche interne via RAG | YES |
| Compliance Auditor | Détection des risques | NO |
| Report Generator | Structuration JSON | NO |

Respect strict de :
✔ Séquentialité  
✔ Non-hallucination  
✔ Attribution claire des responsabilités  

---

# 🌐 Phase 3 — API FastAPI

Endpoint principal :

```
POST /analyze_regulation
```

### 📤 Input attendu

| Champ | Description |
|---|---|
| new_regulation_text | Texte brut du règlement |
| date_of_law | (optionnel) Date YYYY-MM-DD |
| x_user_id | ID utilisateur (auth SaaS) |

### 📥 Output généré

Exemple minimal :

```json
{
  "regulation_id": "d41...",
  "date_processed": "2025-12-05",
  "total_risks_flagged": 3,
  "risks": [
    {
      "policy_id": "default-377a...",
      "severity": "HIGH",
      "divergence_summary": "...",
      "conflicting_policy_excerpt": "...",
      "new_rule_excerpt": "..."
    }
  ],
  "recommendation": "Mettre à jour la politique..."
}
```

🧠 Ce format respecte la spécification ARCA.

---

# 🧪 Phase 4 — Matériel de Validation (Livrables)

Le projet inclut :

✔ Code Python complet  
✔ CrewAI opérationnel  
✔ API fonctionnelle  
✔ Base vectorielle persistante  
✔ README + scénario de test  
✔ requirements.txt  

---

# 👨‍💻 Prérequis techniques

| Technologie | Rôle |
|---|---|
| Python 3.10+ | Langage |
| FastAPI | API |
| CrewAI | Agents |
| ChromaDB | Vector DB |
| SentenceTransformers | Embedding |
| OpenAI / OpenRouter / Gemini API | LLM |

---

# ▶️ Installation

```bash
git clone https://github.com/...
cd arca_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

# ▶️ Démarrage Serveur API

```bash
uvicorn main:app --reload --port 8000
```

---

# 🧪 Exemple d’appel API via curl

```bash
curl -X POST http://localhost:8000/analyze_regulation \
  -H "x-user-id: default" \
  -F "new_regulation_text=Les sessions inactives doivent être interrompues après 15 minutes..."
```

---

# 💡 Notes de Conformité ARCA

Ce projet est 100% conforme à :

✓ Séquentialité des agents  
✓ RAG basé sur embeddings locaux  
✓ JSON strict  
✓ Absence de génération de policy inventée  
✓ Recherche via un outil unique  
✓ Structuration finale machine-readable  

---

# 🎯 Finalité du Projet

Ce système permet à une entreprise de :

- Detecter automatiquement les conflits de conformité
- Gagner du temps sur la veille réglementaire
- Produire un dossier d’incident exploitable  
- Archiver l’analyse réglementaire  
- Intégrer les résultats dans un SI existant

---

# 🧑‍🔧 Auteur & Contact  
Projet réalisé par **[Votre nom]**  
Soutenance ARCA 2025  
