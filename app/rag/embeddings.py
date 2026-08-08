"""Embedding model initializer for medical document indexing."""

from typing import List
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("MedicalEmbeddings")


class EmbeddingManager:
    """Wrapper for creating document and query embeddings."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """Initialize embedding model.

        Args:
            model_name (str): HuggingFace or SentenceTransformers model name.
        """
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer not available or failed to load ({e}). Falling back to FakeEmbeddings."
                )
                self._model = "fallback"
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of document strings."""
        model = self._get_model()
        if model == "fallback":
            # Deterministic fallback vector simulation for zero-dependency test environment
            import numpy as np

            return [
                np.random.default_rng(hash(t) % (2**32)).uniform(-1, 1, 384).tolist() for t in texts
            ]
        return model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single query string."""
        return self.embed_documents([text])[0]
