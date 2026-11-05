"""Researcher Agent: Executes adaptive RAG retrieval and synthesizes draft clinical briefing.

Architectural Mechanics:
- Adaptive Retrieval: Queries hybrid vector + BM25 retriever for relevant clinical evidence.
- Provenance Preservation: Passes retrieved chunk IDs, source literature, and similarity scores.
- Threshold Awareness: Flags weak or empty retrieval candidates for downstream Skeptic evaluation.
- Synthesis: Generates structured draft clinical briefing incorporating retrieved citations.
"""

from typing import Dict, Any, List, Optional
from app.rag.retriever import MedicalRetriever, RetrievalResult
from app.rag.generator import MedicalGenerator
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("Researcher")


class ResearcherAgent:
    """Agent responsible for performing adaptive vector retrieval over clinical guidelines."""

    def __init__(
        self,
        retriever: Optional[MedicalRetriever] = None,
        generator: Optional[MedicalGenerator] = None,
    ):
        self.retriever = retriever or MedicalRetriever()
        self.generator = generator or MedicalGenerator()

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Queries vector database and generates evidence-grounded draft briefing.

        Returns updated state with retrieved_docs, retrieved_chunks, and draft_briefing.
        """
        logger.info("--- RESEARCHER AGENT: Performing Evidence Retrieval ---")
        query = state.get("query", "Diabetic nephropathy treatments")
        patient_id = state.get("patient_id", "PAT-2026-X88")
        trends = state.get("clinical_trends", [])

        # Execute retrieval with full diagnostics
        top_k = settings.RETRIEVAL_TOP_K
        retrieval_res: RetrievalResult = self.retriever.retrieve_with_diagnostics(
            query=query, top_k=top_k
        )

        chunks = retrieval_res.chunks
        retrieved_docs = [c.text for c in chunks]

        # Serialized chunk provenance for downstream citation validation
        chunk_provenance = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "score": c.score,
                "source": c.source,
                "page_number": c.page_number,
                "section": c.section,
            }
            for c in chunks
        ]

        # Generate draft briefing using RAG generator
        if retrieved_docs:
            draft_briefing = self.generator.generate_briefing(
                patient_id=patient_id,
                trends=trends,
                documents=retrieved_docs,
            )
            reasoning = (
                f"Retrieved {len(chunks)} evidence chunks using '{retrieval_res.diagnostics.strategy}' strategy. "
                f"Top similarity score: {retrieval_res.diagnostics.top_scores[0] if retrieval_res.diagnostics.top_scores else 0.0}. "
                f"Drafted clinical recommendation briefing."
            )
        else:
            draft_briefing = "No sufficiently relevant clinical guidelines retrieved for the query."
            reasoning = "Retriever returned 0 matching guidelines meeting the similarity threshold."

        logger.info(f"Researcher Reasoning: {reasoning}")

        return {
            "retrieved_docs": retrieved_docs,
            "retrieved_chunks": chunk_provenance,
            "draft_briefing": draft_briefing,
            "reasoning": reasoning,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
