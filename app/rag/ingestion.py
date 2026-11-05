"""Document ingestion pipeline for medical literature, guidelines, and EHR narratives.

Architectural Mechanics:
- Document Format Support: .txt, .md, .json, and .pdf documents.
- Validation: Size restriction (default 10MB), empty file checks, non-printable character scrubbing.
- Deduplication: SHA256 document content hashing prevents redundant chunk embeddings.
- Metadata Provenance: Captures file source, page numbers, ingest timestamp, and stable IDs.
- Deterministic IDs: Generates stable document IDs and chunk IDs.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from app.rag.chunking import MedicalChunker, TextChunk
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("Ingestion")


class IngestionSummary(BaseModel):
    """Pydantic schema reporting ingestion pipeline diagnostics."""

    document_id: str
    source_filename: str
    total_characters: int
    chunks_created: int
    duplicate: bool = False
    duration_ms: float
    chunk_ids: List[str] = Field(default_factory=list)
    status: str = "success"
    error: Optional[str] = None


class DocumentIngestionError(Exception):
    """Base exception for document ingestion failures."""
    pass


class DocumentIngestor:
    """Production-grade document validation, extraction, chunking, and indexing engine."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}

    def __init__(
        self,
        chunker: Optional[MedicalChunker] = None,
        max_file_size_bytes: int = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024,
    ):
        self.chunker = chunker or MedicalChunker()
        self.max_file_size_bytes = max_file_size_bytes
        self._ingested_hashes: Dict[str, str] = {}  # hash -> document_id

    def extract_text(self, file_path: Path) -> List[tuple]:
        """Extract text content and page numbers from supported file types.

        Returns:
            List of (page_number: int, page_text: str) tuples.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise DocumentIngestionError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        if file_size == 0:
            raise DocumentIngestionError(f"Document is empty (0 bytes): {file_path.name}")
        if file_size > self.max_file_size_bytes:
            raise DocumentIngestionError(
                f"File size {file_size} bytes exceeds maximum allowed limit {self.max_file_size_bytes} bytes."
            )

        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise DocumentIngestionError(
                f"Unsupported file format '{suffix}'. Allowed: {sorted(list(self.SUPPORTED_EXTENSIONS))}"
            )

        pages: List[tuple] = []

        if suffix in {".txt", ".md"}:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            pages.append((1, text))

        elif suffix == ".json":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            # Normalize JSON into structured clinical text
            text = json.dumps(data, indent=2)
            pages.append((1, text))

        elif suffix == ".pdf":
            try:
                import pypdf

                reader = pypdf.PdfReader(str(file_path))
                for page_num, page in enumerate(reader.pages, start=1):
                    extracted = page.extract_text() or ""
                    if extracted.strip():
                        pages.append((page_num, extracted))
                if not pages:
                    raise DocumentIngestionError(f"No extractable text found in PDF: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to parse PDF {file_path}: {e}")
                raise DocumentIngestionError(f"PDF extraction error: {str(e)}")

        return pages

    def process_file(
        self,
        file_path: Path,
        custom_document_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[IngestionSummary, List[TextChunk]]:
        """Validate, extract, deduplicate, and chunk a document.

        Returns:
            (IngestionSummary, List[TextChunk])
        """
        start_time = time.time()
        file_path = Path(file_path)

        pages = self.extract_text(file_path)
        combined_text = "\n\n".join(text for _, text in pages)
        doc_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()

        # Check duplicate
        if doc_hash in self._ingested_hashes:
            existing_doc_id = self._ingested_hashes[doc_hash]
            duration = (time.time() - start_time) * 1000
            logger.info(f"Duplicate document detected ({file_path.name}); maps to {existing_doc_id}")
            summary = IngestionSummary(
                document_id=existing_doc_id,
                source_filename=file_path.name,
                total_characters=len(combined_text),
                chunks_created=0,
                duplicate=True,
                duration_ms=round(duration, 2),
                status="duplicate_skipped",
            )
            return summary, []

        doc_id = custom_document_id or f"doc_{doc_hash[:10]}"
        self._ingested_hashes[doc_hash] = doc_id

        all_chunks: List[TextChunk] = []
        for page_num, page_text in pages:
            page_chunks = self.chunker.chunk_document(
                document_id=doc_id,
                text=page_text,
                source=file_path.name,
                page_number=page_num,
                extra_metadata=extra_metadata,
            )
            all_chunks.extend(page_chunks)

        duration = (time.time() - start_time) * 1000
        summary = IngestionSummary(
            document_id=doc_id,
            source_filename=file_path.name,
            total_characters=len(combined_text),
            chunks_created=len(all_chunks),
            duplicate=False,
            duration_ms=round(duration, 2),
            chunk_ids=[c.chunk_id for c in all_chunks],
            status="success",
        )

        logger.info(
            f"Successfully ingested {file_path.name}: {len(all_chunks)} chunks created in {duration:.2f}ms."
        )
        return summary, all_chunks

    def process_raw_text(
        self,
        text: str,
        source_name: str = "raw_input",
        document_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[IngestionSummary, List[TextChunk]]:
        """Ingest raw in-memory clinical text strings."""
        start_time = time.time()
        if not text.strip():
            raise DocumentIngestionError("Cannot ingest empty raw text.")

        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = document_id or f"doc_{doc_hash[:10]}"

        chunks = self.chunker.chunk_document(
            document_id=doc_id,
            text=text,
            source=source_name,
            page_number=1,
            extra_metadata=extra_metadata,
        )

        duration = (time.time() - start_time) * 1000
        summary = IngestionSummary(
            document_id=doc_id,
            source_filename=source_name,
            total_characters=len(text),
            chunks_created=len(chunks),
            duplicate=False,
            duration_ms=round(duration, 2),
            chunk_ids=[c.chunk_id for c in chunks],
            status="success",
        )
        return summary, chunks
