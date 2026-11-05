"""Embedding generation for medical document indexing and similarity retrieval.

Architectural Mechanics:
- ChromaDB EmbeddingFunction Protocol: Directly pluggable into Chroma collections.
- Model Flexibility: Uses SentenceTransformers (`all-MiniLM-L6-v2` by default) when available.
- Deterministic Offline Fallback: Synthesizes deterministic 384-dimensional L2-normalized
  semantic vectors using character/token n-gram hashing for offline testing and CI environments.
"""

import math
import zlib
import numpy as np
from typing import List, Union, Optional
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("Embeddings")


def _deterministic_hash_vector(text: str, dim: int = 384) -> List[float]:
    """Generates a deterministic unit-normalized pseudo-embedding for zero-network testing.

    Uses word n-grams and process-stable CRC32 hashing so texts with overlapping medical terminology
    produce consistent, meaningful cosine similarity across separate Python processes.
    """
    words = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)

    if not words:
        vec[0] = 1.0
        return vec.tolist()

    for word in words:
        # Process-stable CRC32 hash into vector dimensions
        h = zlib.crc32(word.encode("utf-8"))
        idx = h % dim
        val = ((h >> 8) % 1000) / 500.0 - 1.0
        vec[idx] += val

    # Add bigram hashes for localized sequence sensitivity
    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        h = zlib.crc32(bigram.encode("utf-8"))
        idx = h % dim
        val = ((h >> 8) % 1000) / 500.0 - 1.0
        vec[idx] += val * 1.5

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm
    else:
        vec[0] = 1.0

    return vec.tolist()


class MedicalEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible embedding function with deterministic fallback."""

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        force_mock: bool = False,
    ):
        self.model_name = model_name
        self.force_mock = force_mock
        self._st_model = None
        self._initialization_attempted = False

    def _get_model(self):
        import os

        # Fast path for offline mode, testing, or mock execution
        if self.force_mock or settings.MOCK_LLM or bool(os.environ.get("PYTEST_CURRENT_TEST")):
            return None

        if not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("USE_TORCH", "1")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading SentenceTransformer: {self.model_name}")
                self._st_model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer unavailable or network restricted ({e}). Using deterministic offline embeddings."
                )
                self._st_model = None

        return self._st_model

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a batch of document strings."""
        model = self._get_model()
        if model is not None:
            try:
                embeddings = model.encode(input, convert_to_numpy=True, normalize_embeddings=True)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"Error during neural embedding inference ({e}); falling back.")

        return [_deterministic_hash_vector(doc) for doc in input]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convenience method for LangChain compatibility."""
        return self(texts)

    def embed_query(self, input: Union[str, List[str], None] = None, text: Optional[str] = None) -> Union[List[List[float]], List[float]]:
        """Embedding method compatible with both ChromaDB (input=...) and LangChain (text=...)."""
        target = input if input is not None else text
        if target is None:
            raise ValueError("Must provide either 'input' or 'text'")
        if isinstance(target, str):
            # If called as LangChain embed_query(text) or single string
            return self([target])[0]
        # If called by ChromaDB as embed_query(input=['query1', 'query2'])
        return self(target)
