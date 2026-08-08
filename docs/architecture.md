# Agentic Medical RAG Technical Architecture

## Overview
Agentic Medical RAG is a stateful multi-agent Clinical Decision Support (CDS) platform designed to ingest unstructured clinical records, perform vector-based evidence retrieval, enforce strict clinical safety cross-validation, and generate citation-backed clinical briefings.

## Architectural Layers

```mermaid
flowchart TD
    subgraph Data Layer
        A[Unstructured Clinical Notes / PDFs] --> B[Data Ingestion Engine]
        JSON[FHIR JSON Lab Feeds] --> B
    end

    subgraph RAG & Embedding Layer
        B --> C[Text Chunking & Preprocessing]
        C --> D[Embedding Manager]
        D --> E[Persistent ChromaDB Store]
    end

    subgraph Agentic Orchestration Layer (LangGraph)
        E <--> H[Historian Agent]
        H --> R[Researcher Agent]
        R --> S[Skeptic Safety Agent]
        S -- Verified --> SC[Scribe Agent]
        S -- Contraindication Flagged --> HALT[Safety Halt & Alert]
        SC --> F[Final Verified CDS Briefing]
    end

    subgraph Service & Interface Layer
        REST[FastAPI REST Service] --> Agentic Orchestration Layer
        CLI[Python CLI Interface] --> Agentic Orchestration Layer
        ST[Streamlit Dashboard] --> Agentic Orchestration Layer
    end
```

## Specialized AI Agents

1. **Historian Agent**: Parses raw patient record narratives and detects clinical lab trends (e.g. rising serum creatinine, uncontrolled HbA1c).
2. **Researcher Agent**: Executes targeted k-NN semantic queries against ChromaDB guidelines index using adaptive RAG patterns.
3. **Skeptic Safety Agent**: Acts as a clinical validator by cross-referencing proposed treatment plans against patient medical history to detect contraindications (e.g., ACE inhibitors in bilateral renal artery stenosis).
4. **Scribe Agent**: Synthesizes verified findings into a structured Markdown briefing with mandatory source citations.

## Safety & Compliance
- **Zero-Hallucination Gate**: Unverified recommendations flagged by Skeptic Agent are blocked from output.
- **HIPAA Privacy**: Supports local vector persistence and local open-weights LLMs (Ollama) to keep PHI on-premise.
