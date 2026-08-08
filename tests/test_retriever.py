"""Unit tests for MedicalRetriever and VectorStoreManager."""

import pytest
import shutil
from app.rag.vector_store import VectorStoreManager
from app.rag.retriever import MedicalRetriever


@pytest.fixture
def temp_vector_dir(tmp_path):
    """Fixture providing a temporary directory for ChromaDB vector store."""
    db_dir = tmp_path / "test_vector_db"
    yield db_dir
    if db_dir.exists():
        shutil.rmtree(db_dir, ignore_errors=True)


def test_vector_store_initialization_and_seeding(temp_vector_dir):
    """Test ChromaDB client creation and document seeding."""
    vm = VectorStoreManager(persist_directory=temp_vector_dir, collection_name="test_guidelines")
    count = vm.seed_initial_knowledge_base()
    assert count > 0
    assert vm.collection.count() == count


def test_retriever_document_query(temp_vector_dir):
    """Test MedicalRetriever returns relevant document snippets."""
    vm = VectorStoreManager(persist_directory=temp_vector_dir, collection_name="test_guidelines")
    vm.seed_initial_knowledge_base()

    retriever = MedicalRetriever(vector_store=vm)
    results = retriever.retrieve_documents(query="ACE inhibitors renal artery stenosis", top_k=2)

    assert len(results) > 0
    assert isinstance(results[0], str)
