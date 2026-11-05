"""Integration tests for end-to-end LangGraph multi-agent clinical workflow."""

import uuid
from app.agents.orchestrator import create_medical_graph, AgentState
from app.rag.retriever import MedicalRetriever
from app.rag.vector_store import VectorStoreManager
from tests.fixtures.sample_guidelines import SAMPLE_CLINICAL_GUIDELINES


def test_safe_patient_pipeline_flow(tmp_path):
    """Test safe patient without stenosis contraindications compiles verified report."""
    db_dir = tmp_path / "test_safe_db"
    vm = VectorStoreManager(persist_directory=db_dir, collection_name="safe_eval")
    vm.seed_initial_knowledge_base(SAMPLE_CLINICAL_GUIDELINES)
    retriever = MedicalRetriever(vector_store=vm, retrieval_mode="hybrid")

    graph = create_medical_graph(retriever=retriever)

    initial_state = {
        "patient_id": "PAT-SAFE-01",
        "patient_history": "Patient ID: PAT-SAFE-01. Type 2 Diabetes. Creatinine: 1.0 mg/dL. No stenosis.",
        "clinical_trends": [],
        "query": "SGLT2 inhibitors for diabetes with kidney protection",
        "retrieved_docs": [],
        "retrieved_chunks": [],
        "draft_briefing": "",
        "reasoning": "Starting flow",
        "verified": False,
        "abstained": False,
        "abstention_reason": None,
        "contradictions": [],
        "safety_flags": [],
        "final_report": "",
        "citations": [],
        "validated_citations": [],
        "iteration_count": 0,
        "max_iterations": 5,
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    for _ in graph.stream(initial_state, config):
        pass

    final_state = graph.get_state(config).values
    assert final_state["verified"] is True
    assert final_state["abstained"] is False
    assert "CLINICAL BRIEFING & DECISION SUPPORT REPORT" in final_state["final_report"]
    assert len(final_state["citations"]) > 0


def test_contraindicated_patient_pipeline_flow(tmp_path):
    """Test patient with bilateral renal artery stenosis triggers Skeptic safety halt."""
    db_dir = tmp_path / "test_contra_db"
    vm = VectorStoreManager(persist_directory=db_dir, collection_name="contra_eval")
    vm.seed_initial_knowledge_base(SAMPLE_CLINICAL_GUIDELINES)
    retriever = MedicalRetriever(vector_store=vm, retrieval_mode="hybrid")

    graph = create_medical_graph(retriever=retriever)

    initial_state = {
        "patient_id": "PAT-RISK-02",
        "patient_history": "Patient ID: PAT-RISK-02. History of bilateral renal artery stenosis. Creatinine 1.4 mg/dL.",
        "clinical_trends": [],
        "query": "Treatment for diabetic nephropathy with rising creatinine",
        "retrieved_docs": [],
        "retrieved_chunks": [],
        "draft_briefing": "",
        "reasoning": "Starting flow",
        "verified": False,
        "abstained": False,
        "abstention_reason": None,
        "contradictions": [],
        "safety_flags": [],
        "final_report": "",
        "citations": [],
        "validated_citations": [],
        "iteration_count": 0,
        "max_iterations": 5,
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    for _ in graph.stream(initial_state, config):
        pass

    final_state = graph.get_state(config).values
    assert final_state["verified"] is False
    assert len(final_state["contradictions"]) > 0
    assert "renal artery stenosis" in final_state["contradictions"][0].lower()
    assert "SAFETY HALT" in final_state["final_report"]
