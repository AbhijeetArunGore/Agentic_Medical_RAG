"""API endpoint tests covering health, readiness, query, evaluate, and ingestion."""

import pytest
from app.api.routes import (
    health,
    ready,
    get_diagnostics,
    query_rag,
    evaluate_patient,
    ingest_raw_text,
    list_documents,
    QueryRequest,
    EvaluationRequest,
    RawIngestRequest,
)
from fastapi import HTTPException


def test_health_endpoint():
    """Verify /health returns 200 and operational status."""
    res = health()
    assert res.status == "ok"
    assert res.version == "1.1.0"
    assert res.environment is not None


def test_readiness_endpoint():
    """Verify /ready reports vector store status and indexed document count."""
    res = ready()
    assert res.ready is True
    assert res.indexed_documents >= 0
    assert res.vector_store_status == "online"


def test_diagnostics_endpoint_no_secrets():
    """Verify /diagnostics does not leak secret keys or private paths."""
    res = get_diagnostics()
    assert res["status"] == "operational"
    assert "OPENAI_API_KEY" not in res
    assert "HUGGINGFACE_API_KEY" not in res
    assert "retrieval_mode" in res
    assert "chunk_size" in res


def test_query_endpoint_valid():
    """Verify POST /query returns retrieved documents and briefing."""
    req = QueryRequest(query="Recommended SGLT2 inhibitors for diabetic CKD", patient_id="TEST-P1")
    res = query_rag(req)
    assert res.patient_id == "TEST-P1"
    assert isinstance(res.retrieved_docs, list)
    assert len(res.briefing) > 0


def test_query_endpoint_empty_raises_422():
    """Verify empty query raises 422 Unprocessable Entity."""
    req = QueryRequest(query="   ", patient_id="TEST-P1")
    with pytest.raises(HTTPException) as exc:
        query_rag(req)
    assert exc.value.status_code == 422


def test_evaluate_patient_endpoint():
    """Verify POST /evaluate-patient executes multi-agent graph."""
    req = EvaluationRequest(patient_id="PAT-API-01", query="Evaluate renal safety constraints")
    res = evaluate_patient(req)
    assert res.patient_id == "PAT-API-01"
    assert isinstance(res.verified, bool)
    assert isinstance(res.clinical_trends, list)
    assert isinstance(res.citations, list)


def test_documents_ingest_and_list():
    """Verify raw text ingestion and document listing."""
    req = RawIngestRequest(
        text="Clinical guideline: Regular blood glucose monitoring reduces retinopathy incidence.",
        source_name="Endocrine Society",
        document_id="test_ingest_api_1",
    )
    res = ingest_raw_text(req)
    assert res.status == "success"
    assert res.chunks_created > 0

    doc_list = list_documents()
    assert "total_chunks" in doc_list
    assert doc_list["total_chunks"] > 0
