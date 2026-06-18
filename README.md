# 🏥 Agentic Medical RAG (Clinical Decision Support)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Orchestration-orange)
![ChromaDB](https://img.shields.io/badge/Chroma-VectorDB-purple)

**Agentic Medical RAG** is an advanced, multi-agent Clinical Decision Support (CDS) system designed to ingest complex patient data, securely retrieve relevant medical evidence, and generate highly accurate, citation-backed clinical briefings for healthcare professionals.

## 🌟 Why This Project Stands Out

Traditional RAG systems can hallucinate, which is unacceptable in healthcare. This system solves that by employing an **Agentic Workflow** built on `LangGraph`. It introduces specialized AI agents that debate, verify, and cross-check information against medical guidelines *before* presenting it to the clinician.

## ⚙️ How It Works (The Agentic Architecture)

The system operates as a directed state graph where multiple specialized agents collaborate:

1.  **🕵️‍♂️ The Historian Agent:** Parses unstructured patient documents (like raw clinical notes or PDFs) and identifies critical clinical trends (e.g., rising creatinine levels).
2.  **🔬 The Researcher Agent:** Takes the clinical trends and queries internal medical vector databases (ChromaDB) and external APIs (like PubMed) using Adaptive RAG techniques.
3.  **⚖️ The Skeptic Agent (Clinical Validator):** The crucial safety layer. It cross-checks the researcher's findings against gold-standard clinical guidelines to detect any contradictions or contraindications based on the patient's specific history.
4.  **📝 The Scribe Agent:** If the findings pass the skeptic's validation, the Scribe consolidates the verified data into a professional, easy-to-read briefing, complete with mandatory citations linked directly to the source text.

## 🚀 Key Features

- **Stateful Orchestration:** Utilizes LangGraph for complex agent routing, allowing for cycles, self-correction, and human-in-the-loop (HITL) interventions.
- **Zero-Hallucination Focus:** The "Skeptic Agent" ensures a strict verification loop. Unverified claims are rejected.
- **Local Privacy:** Designed to be compatible with local, open-weights models (via Ollama) to ensure patient data (PHI) never leaves the secure environment.
- **Semantic Data Ingestion:** Processes PDFs and extracts medical context intelligently before chunking into a local vector database.

## 📂 Project Structure

```text
Agentic_Medical_RAG/
├── data/               # Raw patient data for ingestion
├── vector_db/          # Local ChromaDB persistence for medical literature
├── src/
│   ├── agents/         # historian.py, researcher.py, skeptic.py, scribe.py
│   ├── tools/          # setup_vector_db.py
│   ├── graph.py        # LangGraph workflow definition
│   └── state.py        # TypedDict State definition
├── main.py             # Main entry point to run the pipeline
└── TECHNICAL_DESIGN.md # Detailed architectural design
```

## 🛠️ Setup and Execution

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AbhijeetArunGore/Agentic_Medical_RAG.git
    cd Agentic_Medical_RAG
    ```

2.  **Install dependencies:**
    *(Ensure you have Python 3.10+ installed)*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the multi-agent pipeline:**
    ```bash
    python main.py
    ```
    *Observe the terminal as the agents process the data, reason through the clinical problem, and output the final verified briefing.*

---
*Disclaimer: This is a technical demonstration of Agentic AI patterns and should not be used for actual medical diagnosis or treatment without proper clinical oversight and regulatory approval.*
