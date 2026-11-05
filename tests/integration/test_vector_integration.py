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
    res = retriever.retrieve_with_diagnostics(
        "bilateral renal artery stenosis contraindication", top_k=2
    )

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


def test_embedding_space_transition_guard(tmp_path):
    """Verify VectorStoreManager rejects mismatched persisted index unless reseeded."""
    from app.rag.embeddings import MedicalEmbeddingFunction
    from app.rag.vector_store import IncompatibleEmbeddingSpaceError

    db_dir = tmp_path / "test_transition_db"
    mock_embed = MedicalEmbeddingFunction(force_mock=True)

    # 1. Initialize and seed with mock embedding
    vm1 = VectorStoreManager(
        persist_directory=db_dir,
        collection_name="test_col",
        embedding_function=mock_embed,
    )
    vm1.seed_initial_knowledge_base()
    assert vm1.count() > 0

    # 2. Attempt opening existing index with a distinct model identifier
    class DistinctEmbeddingFunction(MedicalEmbeddingFunction):
        @property
        def model_identifier(self) -> str:
            return "distinct-different-model"

        @property
        def is_mock(self) -> bool:
            return False

    distinct_embed = DistinctEmbeddingFunction(force_mock=False)

    with pytest.raises(IncompatibleEmbeddingSpaceError) as exc_info:
        VectorStoreManager(
            persist_directory=db_dir,
            collection_name="test_col",
            embedding_function=distinct_embed,
        )
    assert "was indexed using embedding_model=" in str(exc_info.value)
    assert "force_reseed=true" in str(exc_info.value).lower()

    # 3. Explicit reseed with force_recreate allows clean transition
    vm3 = VectorStoreManager(
        persist_directory=db_dir,
        collection_name="test_col",
        embedding_function=distinct_embed,
        force_reseed=True,
    )
    vm3.seed_initial_knowledge_base()
    assert vm3.count() > 0
    assert vm3.collection.metadata.get("embedding_model") == "distinct-different-model"


def test_neural_fallback_on_existing_neural_index_raises_error(tmp_path, monkeypatch):
    """Verify that when neural loading fails, fallback changes identity and raises error on existing neural index."""
    import chromadb
    from app.rag.embeddings import MedicalEmbeddingFunction
    from app.rag.vector_store import IncompatibleEmbeddingSpaceError

    db_dir = tmp_path / "test_fallback_db"

    # 1. Seed collection originally identified as neural
    client = chromadb.PersistentClient(path=str(db_dir))
    col = client.create_collection(
        name="neural_col",
        metadata={
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_mode": "neural",
            "embedding_dim": 384,
        },
    )
    col.add(ids=["doc1"], documents=["Clinical guideline text"], metadatas=[{"source": "test"}])
    assert col.count() == 1

    # 2. Simulate neural model failure so embedding function falls back to CRC32
    failing_embedder = MedicalEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2", force_mock=False
    )
    # Monkeypatch _get_model to simulate SentenceTransformer loading failure
    monkeypatch.setattr(failing_embedder, "_get_model", lambda: None)

    # 3. Verify active identity is now mock-crc32-384d
    assert failing_embedder.is_mock is True
    assert failing_embedder.model_identifier == "mock-crc32-384d"

    # 4. Attempt to initialize VectorStoreManager against existing neural collection
    with pytest.raises(IncompatibleEmbeddingSpaceError) as exc_info:
        VectorStoreManager(
            persist_directory=db_dir,
            collection_name="neural_col",
            embedding_function=failing_embedder,
        )
    assert "was indexed using embedding_model='sentence-transformers/all-MiniLM-L6-v2'" in str(
        exc_info.value
    )
    assert "'mock-crc32-384d'" in str(exc_info.value)


def test_legacy_unversioned_index_without_metadata_rejected(tmp_path):
    """Verify that legacy indexes lacking embedding metadata are rejected explicitly rather than silently accepted."""
    import chromadb
    from app.rag.embeddings import MedicalEmbeddingFunction
    from app.rag.vector_store import IncompatibleEmbeddingSpaceError

    db_dir = tmp_path / "test_legacy_db"

    # 1. Create a legacy collection without embedding_model metadata
    client = chromadb.PersistentClient(path=str(db_dir))
    legacy_col = client.create_collection(
        name="legacy_col",
        metadata={"hnsw:space": "cosine"},  # No embedding_model or embedding_mode
    )
    legacy_col.add(
        ids=["legacy_1"], documents=["Unversioned guideline text"], metadatas=[{"source": "legacy"}]
    )
    assert legacy_col.count() == 1

    # 2. Attempt opening with VectorStoreManager
    embedder = MedicalEmbeddingFunction(force_mock=True)
    with pytest.raises(IncompatibleEmbeddingSpaceError) as exc_info:
        VectorStoreManager(
            persist_directory=db_dir,
            collection_name="legacy_col",
            embedding_function=embedder,
        )
    assert "lacks embedding provenance metadata" in str(exc_info.value)
    assert "force_reseed=true" in str(exc_info.value).lower()
