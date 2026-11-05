"""Unit tests for MedicalChunker and TextChunk."""

import pytest
from app.rag.chunking import MedicalChunker, TextChunk


def test_chunker_initialization_validation():
    """Verify ValueError is raised if overlap >= chunk size."""
    with pytest.raises(ValueError):
        MedicalChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        MedicalChunker(chunk_size=100, chunk_overlap=150)


def test_chunk_document_recursive():
    """Verify recursive chunking splits paragraphs and generates stable IDs."""
    chunker = MedicalChunker(strategy="recursive", chunk_size=80, chunk_overlap=20)
    text = (
        "First clinical paragraph about hypertension management.\n\n"
        "Second clinical paragraph discussing renal artery stenosis contraindications.\n\n"
        "Third section covering metformin renal dosage protocols."
    )
    chunks = chunker.chunk_document(
        document_id="doc_unit_test",
        text=text,
        source="clinical_memo.txt",
        page_number=2,
    )
    assert len(chunks) >= 3
    for c in chunks:
        assert isinstance(c, TextChunk)
        assert c.document_id == "doc_unit_test"
        assert c.source == "clinical_memo.txt"
        assert c.page_number == 2
        assert len(c.content_hash) == 64
        assert c.chunk_id.startswith("doc_unit_test_c")


def test_chunk_document_fixed():
    """Verify fixed-size chunking correctly slices long strings."""
    chunker = MedicalChunker(strategy="fixed", chunk_size=50, chunk_overlap=10)
    text = "A" * 120
    chunks = chunker.chunk_document("doc_fixed", text, "source.txt")
    assert len(chunks) == 3


def test_chunk_document_structure_aware():
    """Verify clinical headers are detected and preserved."""
    chunker = MedicalChunker(strategy="structure_aware", chunk_size=200, chunk_overlap=20)
    text = (
        "CHIEF COMPLAINT:\nPatient reports bilateral ankle edema.\n\n"
        "PAST MEDICAL HISTORY:\nType 2 Diabetes and bilateral renal artery stenosis.\n\n"
        "LABORATORY TRENDS:\nCreatinine 1.4 mg/dL."
    )
    chunks = chunker.chunk_document("doc_struct", text, "report.txt")
    assert len(chunks) >= 3
    sections = [c.section for c in chunks]
    assert any("CHIEF COMPLAINT" in s for s in sections)
    assert any("LABORATORY TRENDS" in s for s in sections)


def test_chunk_empty_text():
    """Verify empty or purely whitespace strings yield empty list."""
    chunker = MedicalChunker()
    assert chunker.chunk_document("doc_empty", "", "source.txt") == []
    assert chunker.chunk_document("doc_empty", "   \n\n\t  ", "source.txt") == []
