"""FastAPI REST API routes for Agentic Medical RAG.

Architectural Mechanics:
- Request Validation: Pydantic schemas enforce type safety, payload limits, and documentation examples.
- Request ID Middleware: Generates unique UUIDs and binds X-Request-ID to each HTTP request.
- Exception Handling: Safe boundaries preventing unhandled stack trace leakage.
- Observability: Diagnostic endpoints for readiness, collection status, and retrieval telemetry.
- Backward Compatibility: Preserves /health, /query, and /evaluate-patient response contracts.
"""

import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, Response, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.orchestrator import create_medical_graph
from app.agents.medical_agent import MedicalAgent
from app.rag.retriever import MedicalRetriever
from app.rag.vector_store import VectorStoreManager
from app.rag.ingestion import DocumentIngestor, IngestionSummary, DocumentIngestionError
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("API")

# Initialize FastAPI instance
app = FastAPI(
    title="Agentic Medical RAG API",
    description="Production-Grade Clinical Decision Support System via Agentic Multi-Agent Workflows.",
    version="1.1.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID and Timing Middleware
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()

    response: Response = await call_next(request)

    duration_ms = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-MS"] = str(duration_ms)
    return response


# Lazy-loaded singletons to avoid startup bottlenecks
_retriever: Optional[MedicalRetriever] = None
_workflow_app = None
_medical_agent: Optional[MedicalAgent] = None
_ingestor: Optional[DocumentIngestor] = None


def get_retriever() -> MedicalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MedicalRetriever()
    return _retriever


def get_workflow_app():
    global _workflow_app
    if _workflow_app is None:
        _workflow_app = create_medical_graph(retriever=get_retriever())
    return _workflow_app


def get_medical_agent() -> MedicalAgent:
    global _medical_agent
    if _medical_agent is None:
        _medical_agent = MedicalAgent(retriever=get_retriever())
    return _medical_agent


def get_ingestor() -> DocumentIngestor:
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor()
    return _ingestor


# ------------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ------------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    version: str = Field(..., json_schema_extra={"example": "1.1.0"})
    environment: str = Field(..., json_schema_extra={"example": "development"})


class ReadyResponse(BaseModel):
    ready: bool
    indexed_documents: int
    vector_store_status: str


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        json_schema_extra={"example": "Treatment options for rising creatinine in T2D patient."},
    )
    patient_id: Optional[str] = Field(
        default="PAT-2026-X88", json_schema_extra={"example": "PAT-2026-X88"}
    )


class QueryResponse(BaseModel):
    query: str
    patient_id: str
    retrieved_docs: List[str]
    briefing: str
    diagnostics: Optional[Dict[str, Any]] = None


class EvaluationRequest(BaseModel):
    patient_id: str = Field(default="PAT-2026-X88", json_schema_extra={"example": "PAT-2026-X88"})
    query: Optional[str] = Field(
        default="Treatment for rising creatinine and high HbA1c",
        max_length=2000,
        json_schema_extra={"example": "Treatment for rising creatinine and high HbA1c"},
    )
    patient_history: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "example": "Type 2 diabetes, creatinine 1.4 mg/dL, renal artery stenosis."
        },
    )


class EvaluationResponse(BaseModel):
    patient_id: str
    verified: bool
    clinical_trends: List[str]
    contradictions: List[str]
    final_report: Optional[str]
    citations: List[str]
    abstained: bool = False
    abstention_reason: Optional[str] = None


class RawIngestRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=1000000)
    source_name: str = Field(default="manual_entry")
    document_id: Optional[str] = None


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/", tags=["Root"])
def root():
    """Root endpoint providing service metadata and API documentation URL."""
    return {
        "message": "Welcome to Agentic Medical RAG API",
        "documentation": "/docs",
        "health": "/health",
        "ready": "/ready",
        "version": "1.1.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint to verify microservice operational status."""
    return HealthResponse(status="ok", version="1.1.0", environment=settings.ENVIRONMENT)


@app.get("/ready", response_model=ReadyResponse, tags=["Health"])
def ready():
    """Readiness probe verifying vector store accessibility and guideline seeding."""
    try:
        retriever = get_retriever()
        count = retriever.vector_store.count()
        return ReadyResponse(ready=True, indexed_documents=count, vector_store_status="online")
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse, tags=["CDS RAG"])
def query_rag(request: QueryRequest):
    """Executes single-turn RAG retrieval and medical briefing synthesis."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be empty or purely whitespace.",
        )

    try:
        agent = get_medical_agent()
        result = agent.query(query_text=request.query, patient_id=request.patient_id)
        return QueryResponse(
            query=result["query"],
            patient_id=result["patient_id"],
            retrieved_docs=result["retrieved_docs"],
            briefing=result["briefing"],
            diagnostics=result.get("diagnostics"),
        )
    except Exception as e:
        logger.error(f"Error executing RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal system error during query execution: {str(e)}",
        )


@app.post("/evaluate-patient", response_model=EvaluationResponse, tags=["Agentic Workflow"])
def evaluate_patient(request: EvaluationRequest):
    """Executes stateful multi-agent LangGraph workflow for patient diagnostic evaluation."""
    try:
        workflow = get_workflow_app()
        initial_state = {
            "patient_id": request.patient_id,
            "patient_history": request.patient_history or "",
            "clinical_trends": [],
            "query": request.query or "Treatment for rising creatinine and high HbA1c",
            "retrieved_docs": [],
            "retrieved_chunks": [],
            "draft_briefing": "",
            "reasoning": "Starting diagnostic flow.",
            "verified": False,
            "abstained": False,
            "abstention_reason": None,
            "contradictions": [],
            "safety_flags": [],
            "final_report": "",
            "citations": [],
            "validated_citations": [],
            "iteration_count": 0,
            "max_iterations": settings.MAX_AGENT_ITERATIONS,
        }

        # Thread isolation for multi-turn state execution
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        for _ in workflow.stream(initial_state, config):
            pass

        final_state = workflow.get_state(config).values

        return EvaluationResponse(
            patient_id=final_state.get("patient_id", request.patient_id),
            verified=final_state.get("verified", False),
            clinical_trends=final_state.get("clinical_trends", []),
            contradictions=final_state.get("contradictions", []),
            final_report=final_state.get("final_report", None),
            citations=final_state.get("citations", []),
            abstained=final_state.get("abstained", False),
            abstention_reason=final_state.get("abstention_reason", None),
        )
    except Exception as e:
        logger.error(f"Error executing agentic evaluation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal agentic evaluation error: {str(e)}",
        )


@app.post("/documents/ingest", response_model=IngestionSummary, tags=["Knowledge Base"])
def ingest_raw_text(request: RawIngestRequest):
    """Ingests raw clinical guideline or patient text into the vector knowledge base."""
    try:
        ingestor = get_ingestor()
        summary, chunks = ingestor.process_raw_text(
            text=request.text,
            source_name=request.source_name,
            document_id=request.document_id,
        )
        if chunks:
            retriever = get_retriever()
            retriever.vector_store.add_chunks(chunks)
        return summary
    except DocumentIngestionError as de:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(de))
    except Exception as e:
        logger.error(f"Error during raw text ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}",
        )


@app.get("/documents", tags=["Knowledge Base"])
def list_documents():
    """Returns knowledge base collection statistics."""
    try:
        retriever = get_retriever()
        total_chunks = retriever.vector_store.count()
        return {
            "collection_name": retriever.vector_store.collection_name,
            "total_chunks": total_chunks,
            "retrieval_strategy": retriever.retrieval_mode,
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document statistics: {str(e)}",
        )


@app.get("/diagnostics", tags=["Observability"])
def get_diagnostics():
    """Returns runtime telemetry without exposing internal keys or sensitive content."""
    return {
        "status": "operational",
        "environment": settings.ENVIRONMENT,
        "mock_llm": settings.MOCK_LLM,
        "retrieval_mode": settings.RETRIEVAL_MODE,
        "top_k": settings.RETRIEVAL_TOP_K,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
        "chunking_strategy": settings.CHUNKING_STRATEGY,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
    }
