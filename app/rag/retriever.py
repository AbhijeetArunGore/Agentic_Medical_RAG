"""Document retriever implementation for Agentic RAG."""

from typing import List, Optional
from app.rag.vector_store import VectorStoreManager
from app.utils.logger import get_logger

logger = get_logger("MedicalRetriever")


class MedicalRetriever:
    """Medical document retriever abstraction over vector store."""

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        """Initialize MedicalRetriever.

        Args:
            vector_store (Optional[VectorStoreManager]): Vector store manager instance.
        """
        self.vector_store = vector_store or VectorStoreManager()

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant medical guidelines for a query string.

        Args:
            query (str): Search query or clinical trends phrase.
            top_k (int): Top K matching results.

        Returns:
            List[str]: List of document texts.
        """
        logger.info(f"Retrieving top {top_k} documents for query: '{query}'")
        results = self.vector_store.query(query_text=query, top_k=top_k)
        logger.info(f"Retrieved {len(results)} relevant documents.")
        return results
