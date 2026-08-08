"""Integration tests for FastAPI REST endpoints using TestClient."""

from fastapi.testclient import TestClient
from app.api.routes import app

client = TestClient(app)


def test_health_endpoint():
    """Test /health endpoint returns 200 OK and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_query_endpoint():
    """Test POST /query endpoint returns QueryResponse."""
    payload = {
        "query": "What is the recommended treatment for T2D with kidney stress?",
        "patient_id": "PAT-TEST-01",
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "PAT-TEST-01"
    assert "retrieved_docs" in data
    assert "briefing" in data


def test_evaluate_patient_endpoint():
    """Test POST /evaluate-patient endpoint triggers multi-agent state graph."""
    payload = {
        "patient_id": "PAT-TEST-02",
        "query": "Evaluate renal function and diabetes control",
    }
    response = client.post("/evaluate-patient", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "PAT-TEST-02"
    assert "verified" in data
    assert "clinical_trends" in data
