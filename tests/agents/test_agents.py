"""Unit tests for individual agent node behaviors and orchestrator safeguards."""

import pytest
from app.agents.historian import HistorianAgent
from app.agents.researcher import ResearcherAgent
from app.agents.skeptic import SkepticAgent
from app.agents.scribe import ScribeAgent
from app.agents.orchestrator import create_medical_graph


def test_historian_trend_extraction():
    """Verify Historian extracts lab trends and stenosis history."""
    agent = HistorianAgent()
    state = {
        "patient_id": "PAT-H1",
        "patient_history": "Patient with Creatinine 1.5 mg/dL, HbA1c 7.8%, and history of bilateral renal artery stenosis.",
        "iteration_count": 0,
    }
    update = agent.process(state)
    assert len(update["clinical_trends"]) >= 2
    assert any("Creatinine" in t for t in update["clinical_trends"])
    assert any("HbA1c" in t for t in update["clinical_trends"])
    assert update["iteration_count"] == 1


def test_researcher_evidence_retrieval(tmp_path):
    """Verify Researcher retrieves matching documents and builds draft briefing."""
    agent = ResearcherAgent()
    state = {
        "patient_id": "PAT-R1",
        "query": "SGLT2 inhibitors for diabetic nephropathy",
        "clinical_trends": ["Creatinine: Elevated"],
        "iteration_count": 1,
    }
    update = agent.process(state)
    assert "retrieved_docs" in update
    assert "retrieved_chunks" in update
    assert "draft_briefing" in update
    assert update["iteration_count"] == 2


def test_skeptic_contraindication_detection():
    """Verify Skeptic flags ACE inhibitors in renal artery stenosis."""
    agent = SkepticAgent()
    state = {
        "draft_briefing": "Recommend initiating ACE inhibitor therapy (Lisinopril 10mg) for kidney protection.",
        "patient_history": "Confirmed bilateral renal artery stenosis.",
        "retrieved_docs": ["Guideline: ACE inhibitors recommended for proteinuria."],
        "retrieved_chunks": [{"score": 0.8, "chunk_id": "c1"}],
        "iteration_count": 2,
    }
    update = agent.process(state)
    assert update["verified"] is False
    assert len(update["contradictions"]) > 0
    assert "CRITICAL CONTRAINDICATION" in update["contradictions"][0]


def test_skeptic_abstention_on_empty_retrieval():
    """Verify Skeptic initiates abstention when retrieval yields zero documents."""
    agent = SkepticAgent()
    state = {
        "draft_briefing": "No guidelines found.",
        "patient_history": "Normal patient history.",
        "retrieved_docs": [],
        "retrieved_chunks": [],
        "iteration_count": 2,
    }
    update = agent.process(state)
    assert update["verified"] is False
    assert update["abstained"] is True
    assert "INSUFFICIENT EVIDENCE" in update["abstention_reason"]


def test_scribe_report_and_citations():
    """Verify Scribe attaches citations and clinical disclaimer."""
    agent = ScribeAgent()
    state = {
        "patient_id": "PAT-S1",
        "clinical_trends": ["Creatinine: Normal"],
        "draft_briefing": "Prescribe standard protective therapy.",
        "retrieved_chunks": [
            {"chunk_id": "ada_c1", "source": "ADA 2026", "page_number": 14, "section": "CKD", "score": 0.95}
        ],
        "verified": True,
        "abstained": False,
        "contradictions": [],
    }
    update = agent.process(state)
    assert "final_report" in update
    assert "DISCLAIMER" in update["final_report"]
    assert len(update["citations"]) == 1
    assert "ADA 2026" in update["citations"][0]
    assert "ada_c1" in update["citations"][0]
