"""High-level Medical Agent interface for direct CDS query evaluation."""

from typing import Dict, Any
from app.rag.retriever import MedicalRetriever
from app.rag.generator import MedicalGenerator
from app.utils.logger import get_logger

logger = get_logger("MedicalAgent")


class MedicalAgent:
    """Standalone Medical Agent for executing quick RAG queries outside graph loop."""

    def __init__(self):
        """Initialize retriever and generator dependencies."""
        self.retriever = MedicalRetriever()
        self.generator = MedicalGenerator()

    def query(self, query_text: str, patient_id: str = "PAT-2026-X88") -> Dict[str, Any]:
        """Execute single-turn medical retrieval and briefing generation.

        Args:
            query_text (str): Medical query or prompt.
            patient_id (str): Optional patient identifier.

        Returns:
            Dict[str, Any]: Query result containing retrieved docs and briefing.
        """
        logger.info(f"MedicalAgent processing query: {query_text}")
        docs = self.retriever.retrieve_documents(query=query_text, top_k=3)
        briefing = self.generator.generate_briefing(
            patient_id=patient_id, trends=["Query-based evaluation"], documents=docs
        )

        return {
            "query": query_text,
            "patient_id": patient_id,
            "retrieved_docs": docs,
            "briefing": briefing,
        }
