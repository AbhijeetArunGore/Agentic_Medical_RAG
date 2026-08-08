"""Main entry point for Agentic Medical RAG application.

Supports running in CLI mode, starting the FastAPI server, or setting up the vector store.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for standalone module execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import argparse
import uvicorn
import uuid

from app.agents.orchestrator import create_medical_graph
from app.rag.vector_store import VectorStoreManager
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("Main")


def setup_knowledge_base():
    """Seed vector database with clinical guidelines."""
    logger.info("Initializing vector store knowledge base...")
    vm = VectorStoreManager()
    count = vm.seed_initial_knowledge_base()
    logger.info(f"Knowledge base seeded with {count} guidelines.")


def run_cli_pipeline(patient_id: str = "PAT-2026-X88", query: str = None):
    """Executes stateful multi-agent LangGraph workflow via CLI.

    Args:
        patient_id (str): Patient ID identifier.
        query (str): Clinical evaluation query.
    """
    print("\n" + "=" * 80)
    print("=== STARTING AGENTIC MEDICAL RAG (LangGraph Multi-Agent Pipeline) ===")
    print("=" * 80 + "\n")

    app = create_medical_graph()

    initial_state = {
        "patient_id": patient_id,
        "patient_history": "",
        "clinical_trends": [],
        "query": query or "Treatment for rising creatinine and high HbA1c",
        "retrieved_docs": [],
        "draft_briefing": "",
        "reasoning": "Starting diagnostic flow.",
        "verified": False,
        "contradictions": [],
        "final_report": "",
        "citations": [],
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    for event in app.stream(initial_state, config):
        for node_name, state_update in event.items():
            print(f"\n[NODE COMPLETED: {node_name.upper()}]")
            if "reasoning" in state_update:
                print(f"Agent Reasoning Trace: {state_update['reasoning']}")

    final_state = app.get_state(config).values

    print("\n" + "=" * 80)
    if final_state.get("final_report"):
        print("=== FINAL GENERATED BRIEFING ===")
        print(final_state["final_report"])
    else:
        print("=== PIPELINE TERMINATED: Safety Halt Flagged ===")
        print(f"Safety Warnings: {final_state.get('contradictions')}")
    print("=" * 80 + "\n")


def start_server():
    """Starts FastAPI REST server with Uvicorn."""
    print("\n🚀 FastAPI Web Server running!")
    print(f"👉 Access in browser: http://localhost:{settings.API_PORT}/")
    print(f"📚 Interactive Swagger API Docs: http://localhost:{settings.API_PORT}/docs\n")
    logger.info(f"Binding host {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "app.api.routes:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )


def main():
    """Main execution router handling CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic Medical RAG - Clinical Decision Support System"
    )
    parser.add_argument("--server", action="store_true", help="Start FastAPI REST server")
    parser.add_argument(
        "--setup-db", action="store_true", help="Initialize and seed ChromaDB vector store"
    )
    parser.add_argument(
        "--patient-id", type=str, default="PAT-2026-X88", help="Patient ID for CLI run"
    )
    parser.add_argument("--query", type=str, default=None, help="Custom query for CLI run")

    args = parser.parse_args()

    if args.setup_db:
        setup_knowledge_base()
    elif args.server:
        start_server()
    else:
        run_cli_pipeline(patient_id=args.patient_id, query=args.query)


if __name__ == "__main__":
    main()
