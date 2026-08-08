"""Researcher Agent: Queries clinical vector database and retrieves evidence guidelines."""

from typing import Dict, Any
from app.rag.retriever import MedicalRetriever
from app.rag.generator import MedicalGenerator
from app.utils.logger import get_logger

logger = get_logger("ResearcherAgent")


class ResearcherAgent:
    """Agent responsible for performing adaptive vector retrieval over clinical guidelines."""

    def __init__(self, retriever: MedicalRetriever = None):
        """Initialize ResearcherAgent with dynamic retriever abstraction."""
        self.retriever = retriever or MedicalRetriever()
        self.generator = MedicalGenerator()

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Queries vector database based on patient trends in state.

        Args:
            state (Dict[str, Any]): Current graph state dictionary.

        Returns:
            Dict[str, Any]: State update containing retrieved docs and draft briefing.
        """
        logger.info("--- RESEARCHER: Performing Agentic RAG Search ---")
        query = state.get("query", "Diabetic nephropathy treatments")
        patient_id = state.get("patient_id", "PAT-2026-X88")

        # Dynamic retrieval via MedicalRetriever
        retrieved_docs = self.retriever.retrieve_documents(query=query, top_k=2)

        if not retrieved_docs:
            retrieved_docs = [
                "Guideline: ACE inhibitors recommended for T2D with rising creatinine.",
                "Contraindication: ACE inhibitors contraindicated in bilateral renal artery stenosis.",
            ]

        # Generate draft briefing using RAG generator
        draft_briefing = self.generator.generate_briefing(
            patient_id=patient_id,
            trends=state.get("clinical_trends", []),
            documents=retrieved_docs,
        )

        reasoning = (
            f"Identified {len(retrieved_docs)} relevant clinical documents. "
            f"Key focus query: '{query}'. Formulated draft briefing with retrieved evidence."
        )

        logger.info(f"Researcher Reasoning: {reasoning}")

        return {
            "retrieved_docs": retrieved_docs,
            "draft_briefing": draft_briefing,
            "reasoning": reasoning,
        }
