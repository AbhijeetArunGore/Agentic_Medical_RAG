"""Unit tests for DocumentIngestor."""

import json
import pytest
from app.rag.ingestion import DocumentIngestor, DocumentIngestionError


def test_ingest_raw_text():
    """Verify raw in-memory clinical text is properly chunked and reported."""
    ingestor = DocumentIngestor()
    summary, chunks = ingestor.process_raw_text(
        text="Clinical guideline: SGLT2 inhibitors provide renal protection in CKD Stage 3.",
        source_name="raw_test",
        document_id="doc_raw_1",
    )
    assert summary.status == "success"
    assert summary.chunks_created == 1
    assert len(chunks) == 1
    assert chunks[0].document_id == "doc_raw_1"


def test_ingest_empty_raw_text_raises_error():
    """Verify empty raw string raises DocumentIngestionError."""
    ingestor = DocumentIngestor()
    with pytest.raises(DocumentIngestionError):
        ingestor.process_raw_text(text="   \n\t  ")


def test_ingest_text_file(tmp_path):
    """Verify text file ingestion and chunk generation."""
    sample_file = tmp_path / "sample_guideline.txt"
    sample_file.write_text("Guideline A: ACE inhibitors. Guideline B: SGLT2 inhibitors.", encoding="utf-8")

    ingestor = DocumentIngestor()
    summary, chunks = ingestor.process_file(sample_file)
    assert summary.status == "success"
    assert summary.chunks_created > 0
    assert summary.duplicate is False


def test_ingest_duplicate_file_detection(tmp_path):
    """Verify identical file is marked duplicate on second ingestion."""
    file1 = tmp_path / "file1.txt"
    content = "Exact same clinical text for duplicate verification."
    file1.write_text(content, encoding="utf-8")

    ingestor = DocumentIngestor()
    summary1, chunks1 = ingestor.process_file(file1)
    assert summary1.status == "success"

    summary2, chunks2 = ingestor.process_file(file1)
    assert summary2.duplicate is True
    assert summary2.status == "duplicate_skipped"
    assert len(chunks2) == 0


def test_ingest_empty_file_raises_error(tmp_path):
    """Verify empty 0-byte file raises DocumentIngestionError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    ingestor = DocumentIngestor()
    with pytest.raises(DocumentIngestionError, match="Document is empty"):
        ingestor.process_file(empty_file)


def test_ingest_unsupported_extension_raises_error(tmp_path):
    """Verify unsupported file type raises DocumentIngestionError."""
    bad_file = tmp_path / "malicious.exe"
    bad_file.write_text("not a real exe", encoding="utf-8")

    ingestor = DocumentIngestor()
    with pytest.raises(DocumentIngestionError, match="Unsupported file format"):
        ingestor.process_file(bad_file)


def test_ingest_json_file(tmp_path):
    """Verify structured JSON record is ingested."""
    json_file = tmp_path / "patient.json"
    data = {"patient_id": "P100", "condition": "Diabetic Nephropathy", "creatinine": 1.4}
    json_file.write_text(json.dumps(data), encoding="utf-8")

    ingestor = DocumentIngestor()
    summary, chunks = ingestor.process_file(json_file)
    assert summary.status == "success"
    assert summary.chunks_created > 0
