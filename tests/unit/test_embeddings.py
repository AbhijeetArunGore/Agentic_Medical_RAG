"""Unit tests for MedicalEmbeddingFunction and deterministic offline vectors."""

import math
import numpy as np
from app.rag.embeddings import MedicalEmbeddingFunction, _deterministic_hash_vector


def test_deterministic_hash_vector_properties():
    """Verify generated vector has dimension 384 and unit L2 norm."""
    text = "Treatment guidelines for chronic kidney disease in diabetic patients"
    vec = _deterministic_hash_vector(text, dim=384)
    assert len(vec) == 384
    norm = np.linalg.norm(vec)
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_embedding_function_batch_and_query():
    """Verify MedicalEmbeddingFunction embeds batches and queries consistently."""
    embedder = MedicalEmbeddingFunction(force_mock=True)

    docs = ["Doc A: ACE inhibitors", "Doc B: SGLT2 inhibitors"]
    embeddings = embedder.embed_documents(docs)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    query_vec = embedder.embed_query("ACE inhibitors")
    assert len(query_vec) == 384
    assert math.isclose(np.linalg.norm(query_vec), 1.0, rel_tol=1e-5)


def test_chromadb_embed_query_protocol():
    """Verify embed_query accepts both input=... (ChromaDB) and text=... (LangChain)."""
    embedder = MedicalEmbeddingFunction(force_mock=True)

    res_chroma = embedder.embed_query(input=["Query text 1", "Query text 2"])
    assert len(res_chroma) == 2

    res_langchain = embedder.embed_query(text="Query text 1")
    assert len(res_langchain) == 384


def test_mock_embeddings_configuration_flag(monkeypatch):
    """Verify MOCK_EMBEDDINGS setting activates deterministic CRC32 vectors."""
    from app.utils.config import settings

    monkeypatch.setattr(settings, "MOCK_EMBEDDINGS", True)
    embedder = MedicalEmbeddingFunction(force_mock=False)
    assert embedder._get_model() is None
    vec = embedder.embed_query(text="Hypertension treatment")
    assert len(vec) == 384
    assert math.isclose(np.linalg.norm(vec), 1.0, rel_tol=1e-5)


def test_neural_embedding_mode_execution():
    """Verify that neural embedding mode initializes SentenceTransformer when available."""
    embedder = MedicalEmbeddingFunction(force_mock=False)
    model = embedder._get_model()
    # If SentenceTransformer is installed and weights are cached, verify neural output
    vec = embedder.embed_query(text="Renal artery stenosis contraindications")
    assert len(vec) == 384
    assert math.isclose(np.linalg.norm(vec), 1.0, rel_tol=1e-5)
    if model is not None:
        assert hasattr(model, "encode")
