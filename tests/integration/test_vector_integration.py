"""Integration tests for Ingestion -> VectorStore -> Retrieval pipeline."""

import pytest
from app.rag.chunking import MedicalChunker
from app.rag.ingestion import DocumentIngestor
from app.rag.vector_store import VectorStoreManager
from app.rag.retriever import MedicalRetriever
from tests.fixtures.sample_guidelines import SAMPLE_CLINICAL_GUIDELINES


def test_full_ingestion_and_retrieval_flow(tmp_path):
    """Test creating documents, ingesting into isolated ChromaDB, and retrieving via hybrid search."""
    db_dir = tmp_path / "test_vector_db"
    vm = VectorStoreManager(persist_directory=db_dir, collection_name="test_integration")
    vm.seed_initial_knowledge_base(SAMPLE_CLINICAL_GUIDELINES)
    assert vm.count() == 4

    retriever = MedicalRetriever(vector_store=vm, retrieval_mode="hybrid")
    res = retriever.retrieve_with_diagnostics("bilateral renal artery stenosis contraindication", top_k=2)

    assert len(res.chunks) > 0
    assert res.diagnostics.strategy == "hybrid"
    assert any("renal artery stenosis" in c.text.lower() for c in res.chunks)


def test_retrieval_strategies_comparison(tmp_path):
    """Test dense, bm25, and hybrid retrieval modes over the same index."""
    db_dir = tmp_path / "test_strat_db"
    vm = VectorStoreManager(persist_directory=db_dir, collection_name="test_strat")
    vm.seed_initial_knowledge_base(SAMPLE_CLINICAL_GUIDELINES)

    retriever = MedicalRetriever(vector_store=vm)

    # 1. BM25
    bm25_res = retriever.retrieve_with_diagnostics("metformin egfr", top_k=2, strategy="bm25")
    assert len(bm25_res.chunks) > 0
    assert any("metformin" in c.text.lower() for c in bm25_res.chunks)

    # 2. Dense
    dense_res = retriever.retrieve_with_diagnostics("metformin egfr", top_k=2, strategy="dense")
    assert isinstance(dense_res.chunks, list)

    # 3. Hybrid
    hybrid_res = retriever.retrieve_with_diagnostics("metformin egfr", top_k=2, strategy="hybrid")
    assert len(hybrid_res.chunks) > 0
