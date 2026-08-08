"""FastAPI REST API routes for Agentic Medical RAG.

Architectural Mechanics:
- Request Validation: Pydantic schemas enforce type safety, default payloads, and documentation examples.
- Exception Handling: Wrap RAG and graph pipeline calls with HTTP 500 error boundaries and structured logs.
- Response Models: Standardized Pydantic schemas for health, query, and agentic evaluation responses.
- Medical Safety Disclaimer: Responses explicitly communicate safety verification status and Skeptic contraindications.
"""

import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.agents.orchestrator import create_medical_graph
from app.agents.medical_agent import MedicalAgent
from app.utils.logger import get_logger

logger = get_logger("API")

# Step 1: Initialize FastAPI Application Instance with OpenAPI metadata
app = FastAPI(
    title="Agentic Medical RAG API",
    description="Production-Grade Clinical Decision Support System via Agentic Multi-Agent Workflows.",
    version="1.0.0",
)

# Step 2: Global singletons for compiled LangGraph workflow app and MedicalAgent
workflow_app = create_medical_graph()
medical_agent = MedicalAgent()


# Step 3: Define Pydantic Request & Response Validation Schemas
class HealthResponse(BaseModel):
    """Schema for health status endpoint."""

    status: str = Field(..., json_schema_extra={"example": "ok"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})


class QueryRequest(BaseModel):
    """Schema for single-turn RAG query requests."""

    query: str = Field(
        ...,
        json_schema_extra={"example": "Treatment options for rising creatinine in T2D patient."},
    )
    patient_id: Optional[str] = Field(
        default="PAT-2026-X88", json_schema_extra={"example": "PAT-2026-X88"}
    )


class QueryResponse(BaseModel):
    """Schema for single-turn RAG query responses."""

    query: str
    patient_id: str
    retrieved_docs: List[str]
    briefing: str


class EvaluationRequest(BaseModel):
    """Schema for multi-agent stateful evaluation requests."""

    patient_id: str = Field(default="PAT-2026-X88", json_schema_extra={"example": "PAT-2026-X88"})
    query: Optional[str] = Field(
        default="Treatment for rising creatinine and high HbA1c",
        json_schema_extra={"example": "Treatment for rising creatinine and high HbA1c"},
    )


class EvaluationResponse(BaseModel):
    """Schema for multi-agent stateful evaluation responses."""

    patient_id: str
    verified: bool
    clinical_trends: List[str]
    contradictions: List[str]
    final_report: Optional[str]
    citations: List[str]


# Step 4: API Endpoints Implementation
@app.get("/", tags=["Root"])
def root():
    """Root endpoint providing service metadata and API documentation URL."""
    return {
        "message": "Welcome to Agentic Medical RAG API",
        "documentation": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint to verify microservice operational status."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/query", response_model=QueryResponse, tags=["CDS RAG"])
def query_rag(request: QueryRequest):
    """Executes single-turn RAG retrieval and medical briefing synthesis.

    Receives user query, retrieves top guidelines from ChromaDB, and formats clinical briefing.
    """
    try:
        result = medical_agent.query(query_text=request.query, patient_id=request.patient_id)
        return QueryResponse(
            query=result["query"],
            patient_id=result["patient_id"],
            retrieved_docs=result["retrieved_docs"],
            briefing=result["briefing"],
        )
    except Exception as e:
        logger.error(f"Error executing RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal system error during query execution: {str(e)}",
        )


@app.post("/evaluate-patient", response_model=EvaluationResponse, tags=["Agentic Workflow"])
def evaluate_patient(request: EvaluationRequest):
    """Executes stateful multi-agent LangGraph workflow for patient diagnostic evaluation.

    Streams state transitions across Historian -> Researcher -> Skeptic -> Scribe agents.
    Enforces safety halt if Skeptic agent flags contraindications.
    """
    try:
        initial_state = {
            "patient_id": request.patient_id,
            "patient_history": "",
            "clinical_trends": [],
            "query": request.query,
            "retrieved_docs": [],
            "draft_briefing": "",
            "reasoning": "Starting diagnostic flow.",
            "verified": False,
            "contradictions": [],
            "final_report": "",
            "citations": [],
        }

        # Step 5: Execute stateful LangGraph stream with thread isolation
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        for _ in workflow_app.stream(initial_state, config):
            pass

        # Step 6: Extract final state values after workflow termination
        final_state = workflow_app.get_state(config).values

        return EvaluationResponse(
            patient_id=final_state.get("patient_id", request.patient_id),
            verified=final_state.get("verified", False),
            clinical_trends=final_state.get("clinical_trends", []),
            contradictions=final_state.get("contradictions", []),
            final_report=final_state.get("final_report", None),
            citations=final_state.get("citations", []),
        )
    except Exception as e:
        logger.error(f"Error executing agentic evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal agentic evaluation error: {str(e)}",
        )
