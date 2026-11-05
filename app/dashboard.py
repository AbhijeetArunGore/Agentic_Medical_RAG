"""Streamlit Dashboard for visual demonstration of Agentic Medical RAG."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uuid
import time
import streamlit as st

from app.agents.orchestrator import create_medical_graph
from app.rag.retriever import MedicalRetriever
from app.utils.config import settings

# Page Config
st.set_page_config(
    page_title="Agentic Medical RAG CDS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 Agentic Clinical Decision Support System")
st.caption("Stateful Multi-Agent Diagnostic Workflow Powered by LangGraph & ChromaDB")
st.markdown("---")

# Sidebar for Inputs
st.sidebar.header("Patient Data Input")
patient_id = st.sidebar.text_input("Patient ID", "PAT-2026-X88")
clinical_query = st.sidebar.text_area(
    "Clinical Query / Focus Area", "Treatment for rising creatinine and uncontrolled HbA1c"
)
retrieval_strategy = st.sidebar.selectbox(
    "Retrieval Strategy", ["hybrid", "dense", "bm25"]
)

if st.sidebar.button("Run Agentic CDS Pipeline", type="primary"):
    with st.status("🚀 Orchestrating Multi-Agent Workflow...", expanded=True) as status:
        t0 = time.time()
        # 1. Initialize Retriever & Graph
        retriever = MedicalRetriever(retrieval_mode=retrieval_strategy)
        workflow_app = create_medical_graph(retriever=retriever)

        # 2. Initial State Setup
        initial_state = {
            "patient_id": patient_id,
            "patient_history": "",
            "clinical_trends": [],
            "query": clinical_query,
            "retrieved_docs": [],
            "retrieved_chunks": [],
            "draft_briefing": "",
            "reasoning": "Initializing workflow...",
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

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # 3. Stream Events
        for event in workflow_app.stream(initial_state, config):
            for node_name, state_update in event.items():
                st.write(f"✅ Agent Node Completed: **{node_name.upper()}**")
                if "reasoning" in state_update:
                    st.info(f"**Reasoning Trace:** {state_update['reasoning']}")
                time.sleep(0.1)

        total_latency_ms = round((time.time() - t0) * 1000, 2)
        status.update(label=f"✅ Diagnostic Workflow Complete ({total_latency_ms}ms)!", state="complete", expanded=False)

    # 4. Display Workflow Results
    final_state = workflow_app.get_state(config).values

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Identified Laboratory Trends")
        trends = final_state.get("clinical_trends", [])
        if trends:
            for trend in trends:
                st.warning(trend)
        else:
            st.write("No critical trends flagged.")

        st.subheader("🛡️ Skeptic Safety Verification")
        if final_state.get("verified", False):
            st.success("PASSED: Evidence grounded and no contraindications detected.")
        else:
            contradictions = final_state.get("contradictions", [])
            st.error(
                f"SAFETY HALT / CONTRAINDICATION: {contradictions[0] if contradictions else 'Contraindication flagged.'}"
            )

        if final_state.get("citations"):
            st.subheader("📚 Evidence Citations")
            for c in final_state["citations"]:
                st.markdown(f"- {c}")

    with col2:
        st.subheader("📝 Clinical Report Output")
        if final_state.get("final_report"):
            st.code(final_state["final_report"], language="text")
        else:
            st.warning("⚠️ No report compiled.")

    # 5. Live Runtime Diagnostics (Honest Telemetry)
    st.markdown("---")
    st.subheader("📈 Live Execution Telemetry")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pipeline Latency", f"{total_latency_ms} ms")
    m2.metric("Retrieval Strategy", retrieval_strategy.upper())
    m3.metric("Chunks Evaluated", len(final_state.get("retrieved_chunks", [])))
    m4.metric("Verification Status", "VERIFIED" if final_state.get("verified") else "HALTED")

    st.caption("ℹ️ To inspect offline RAG benchmark evaluation figures (Precision@K, Recall@K, MRR), run `python -m evaluation.runners.evaluate_retrieval`.")

else:
    st.info("👈 Select patient parameters in the sidebar and click 'Run Agentic CDS Pipeline'.")
