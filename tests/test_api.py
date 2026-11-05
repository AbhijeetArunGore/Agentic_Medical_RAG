"""Integration tests for FastAPI REST endpoints."""

from app.api.routes import (
    health,
    query_rag,
    evaluate_patient,
    QueryRequest,
    EvaluationRequest,
)


def test_health_endpoint():
    """Test /health endpoint returns 200 OK and status ok."""
    data = health()
    assert data.status == "ok"
    assert data.version == "1.1.0"


def test_query_endpoint():
    """Test POST /query endpoint returns QueryResponse."""
    req = QueryRequest(
        query="What is the recommended treatment for T2D with kidney stress?",
        patient_id="PAT-TEST-01",
    )
    data = query_rag(req)
    assert data.patient_id == "PAT-TEST-01"
    assert isinstance(data.retrieved_docs, list)
    assert data.briefing is not None


def test_evaluate_patient_endpoint():
    """Test POST /evaluate-patient endpoint triggers multi-agent state graph."""
    req = EvaluationRequest(
        patient_id="PAT-TEST-02",
        query="Evaluate renal function and diabetes control",
    )
    data = evaluate_patient(req)
    assert data.patient_id == "PAT-TEST-02"
    assert hasattr(data, "verified")
    assert hasattr(data, "clinical_trends")
