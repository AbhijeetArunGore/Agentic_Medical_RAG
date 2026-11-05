"""Chunking strategies and metadata preservation for medical documents.

Architectural Mechanics:
- Recursive Character Chunking: Breaks text hierarchically on paragraphs, sentences, and words.
- Fixed Size Chunking: Slices text into uniform token-approximate windows with overlap.
- Structure-Aware Chunking: Respects clinical section headers (e.g., CHIEF COMPLAINT, LABS).
- Metadata Enrichment: Computes content hashes, stable chunk IDs, source document tracking.
"""

import hashlib
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from app.utils.config import settings


class TextChunk:
    """Represents an ingested text chunk with clinical provenance metadata."""

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        text: str,
        chunk_index: int,
        source: str,
        page_number: int = 1,
        section: Optional[str] = None,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.text = text
        self.chunk_index = chunk_index
        self.source = source
        self.page_number = page_number
        self.section = section or "General"
        self.content_hash = content_hash or hashlib.sha256(text.encode("utf-8")).hexdigest()
        self.ingestion_timestamp = datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk and its metadata to a flat dictionary for vector store storage."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "page_number": self.page_number,
            "section": self.section,
            "content_hash": self.content_hash,
            "ingestion_timestamp": self.ingestion_timestamp,
            **self.metadata,
        }


class MedicalChunker:
    """Configurable text chunker tailored for clinical guidelines and records."""

    def __init__(
        self,
        strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.strategy = strategy or settings.CHUNKING_STRATEGY
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

    def chunk_document(
        self,
        document_id: str,
        text: str,
        source: str,
        page_number: int = 1,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Chunk a single document text according to the configured strategy."""
        normalized_text = self._normalize_text(text)
        if not normalized_text.strip():
            return []

        if self.strategy == "structure_aware":
            raw_chunks = self._chunk_structure_aware(normalized_text)
        elif self.strategy == "fixed":
            raw_chunks = self._chunk_fixed(normalized_text)
        else:
            raw_chunks = self._chunk_recursive(normalized_text)

        chunks: List[TextChunk] = []
        for idx, (chunk_text, section) in enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunk_id = f"{document_id}_c{idx}_{chunk_hash[:8]}"
            chunk = TextChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                chunk_index=idx,
                source=source,
                page_number=page_number,
                section=section,
                content_hash=chunk_hash,
                metadata=extra_metadata or {},
            )
            chunks.append(chunk)

        return chunks

    def _normalize_text(self, text: str) -> str:
        """Strip control characters and excessive whitespaces while preserving paragraphs."""
        # Replace Windows carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple horizontal whitespaces
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive newlines (> 2) to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_fixed(self, text: str) -> List[tuple]:
        """Fixed size window chunking with step = size - overlap."""
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(text), step):
            slice_text = text[i : i + self.chunk_size].strip()
            if slice_text:
                chunks.append((slice_text, "Fixed"))
        return chunks

    def _chunk_recursive(self, text: str) -> List[tuple]:
        """Hierarchical recursive chunking: split by paragraphs, then sentences, then spaces."""
        separators = ["\n\n", "\n", ". ", "; ", " ", ""]
        raw_pieces = self._split_recursive_helper(text, separators, self.chunk_size)

        # Merge pieces into overlapping chunks
        merged_chunks: List[tuple] = []
        current_chunk = ""

        for piece in raw_pieces:
            if not current_chunk:
                current_chunk = piece
            elif len(current_chunk) + len(piece) + 1 <= self.chunk_size:
                current_chunk += " " + piece
            else:
                merged_chunks.append((current_chunk.strip(), "General"))
                # Overlap tail
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_seed = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_seed + " " + piece
                else:
                    current_chunk = piece

        if current_chunk.strip():
            merged_chunks.append((current_chunk.strip(), "General"))

        return merged_chunks

    def _split_recursive_helper(self, text: str, separators: List[str], max_size: int) -> List[str]:
        """Recursive helper that descends separator hierarchy."""
        if len(text) <= max_size or not separators:
            return [text] if text else []

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return [text[i : i + max_size] for i in range(0, len(text), max_size)]

        parts = text.split(sep)
        result: List[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= max_size:
                result.append(part)
            else:
                result.extend(self._split_recursive_helper(part, remaining_seps, max_size))

        return result

    def _chunk_structure_aware(self, text: str) -> List[tuple]:
        """Detect clinical section headers (e.g. 'CHIEF COMPLAINT:', 'LABORATORY TRENDS:')."""
        header_pattern = re.compile(
            r"^(?:[A-Z0-9\s\-_]{3,40}:|[0-9]+\.\s+[A-Z\s]{3,40})",
            re.MULTILINE,
        )

        matches = list(header_pattern.finditer(text))
        if not matches:
            # Fall back to recursive if no headers found
            return self._chunk_recursive(text)

        sections: List[tuple] = []
        for i, match in enumerate(matches):
            section_name = match.group(0).rstrip(":").strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_content = text[start:end].strip()

            # If section content is larger than chunk_size, recursively sub-chunk
            if len(section_content) > self.chunk_size:
                sub_chunks = self._chunk_recursive(section_content)
                for sub_text, _ in sub_chunks:
                    sections.append((sub_text, section_name))
            else:
                sections.append((section_content, section_name))

        return sections
