"""Advanced Medical Document Retriever supporting Dense, Lexical (BM25), and Hybrid strategies.

Architectural Mechanics:
- Dense Vector Retrieval: ChromaDB cosine similarity over clinical embedding space.
- BM25 Lexical Retrieval: Pure-Python Okapi BM25 implementation for exact medical keyword matching.
- Calibrated Hybrid Scoring: Weighted linear combination of absolute dense cosine similarity and normalized BM25 query coverage.
- Absolute Threshold Filtering: Drops candidates below similarity threshold, enabling genuine zero-candidate abstention.
- Provenance Preservation: Preserves chunk IDs, source literature, page numbers, and scores.
"""

import math
import re
import time
from typing import List, Dict, Any, Optional, Set
from pydantic import BaseModel, Field

from app.rag.vector_store import VectorStoreManager, RetrievedChunk
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("Retriever")


class RetrievalDiagnostics(BaseModel):
    """Diagnostics capturing retrieval performance and provenance metadata."""

    query: str
    strategy: str
    candidates_retrieved: int
    chunks_retained: int
    duration_ms: float
    top_scores: List[float] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Complete retrieval output bundle separating text from diagnostic metadata."""

    chunks: List[RetrievedChunk]
    diagnostics: RetrievalDiagnostics

    @property
    def texts(self) -> List[str]:
        return [c.text for c in self.chunks]


class BM25SearchIndex:
    """Okapi BM25 lexical search index implemented in pure Python."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_docs: List[RetrievedChunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into lowercase word tokens."""
        tokens = re.findall(r"\b[a-z0-9_\-\./%]{2,}\b", text.lower())
        return tokens

    def index_documents(self, documents: List[RetrievedChunk]) -> None:
        """Build term frequency and inverted index over the provided documents."""
        self.corpus_docs = documents
        total_docs = len(documents)
        if total_docs == 0:
            return

        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc.text)
            self.doc_lengths.append(len(tokens))
            unique_terms: Set[str] = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / max(1, total_docs)

        # Compute Robertson-Spärck Jones IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            val = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[term] = max(0.1, val)

    def search(self, query: str, top_k: int = 5) -> List[tuple[RetrievedChunk, float]]:
        """Compute BM25 scores for the query across indexed documents.

        Returns:
            List of (RetrievedChunk, normalized_bm25_score: float in [0.0, 1.0])
        """
        if not self.corpus_docs:
            return []

        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []

        # Total theoretical maximum IDF for normalization
        max_query_idf = sum(self.idf.get(t, 0.5) for t in set(q_tokens))
        if max_query_idf <= 0:
            max_query_idf = 1.0
        norm_factor = max_query_idf * (self.k1 + 1.0)

        raw_scores: List[float] = [0.0] * len(self.corpus_docs)

        for i, doc in enumerate(self.corpus_docs):
            doc_len = self.doc_lengths[i]
            tokens = self._tokenize(doc.text)
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            doc_score = 0.0
            for term in q_tokens:
                if term in tf_map:
                    tf = tf_map[term]
                    idf = self.idf.get(term, 0.5)
                    denom = tf + self.k1 * (
                        1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len))
                    )
                    num = tf * (self.k1 + 1.0)
                    doc_score += idf * (num / denom)

            raw_scores[i] = doc_score

        # Normalize score into [0.0, 1.0] range
        scored: List[tuple[RetrievedChunk, float]] = []
        for i, raw_s in enumerate(raw_scores):
            if raw_s > 0:
                normalized_score = min(1.0, round(raw_s / norm_factor, 4))
                scored.append((self.corpus_docs[i], normalized_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class MedicalRetriever:
    """Production clinical retriever providing Dense, BM25, and Hybrid RAG retrieval."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        retrieval_mode: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.retrieval_mode = retrieval_mode or settings.RETRIEVAL_MODE
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.SIMILARITY_THRESHOLD
        )
        self.bm25_index = BM25SearchIndex()
        self._bm25_initialized = False

    def _sync_bm25_index(self) -> None:
        """Initializes or updates the in-memory BM25 index from ChromaDB documents."""
        count = self.vector_store.count()
        if count == 0:
            self.vector_store.seed_initial_knowledge_base()
            count = self.vector_store.count()

        if not self._bm25_initialized and count > 0:
            all_docs = self.vector_store.collection.get(include=["documents", "metadatas"])
            docs_list = all_docs.get("documents", [])
            ids_list = all_docs.get("ids", [])
            meta_list = all_docs.get("metadatas", [])

            chunk_objects = []
            for cid, doc_text, meta in zip(ids_list, docs_list, meta_list):
                meta = meta or {}
                chunk_objects.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        document_id=meta.get("document_id", "doc_default"),
                        text=doc_text,
                        score=1.0,
                        source=meta.get("source", "Standard Guidelines"),
                        page_number=meta.get("page_number", 1),
                        section=meta.get("section", "General"),
                        metadata=meta,
                    )
                )

            self.bm25_index.index_documents(chunk_objects)
            self._bm25_initialized = True

    def retrieve_with_diagnostics(
        self,
        query: str,
        top_k: int = 4,
        strategy: Optional[str] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """Executes retrieval with full telemetry and metadata provenance."""
        start_time = time.time()
        mode = strategy or self.retrieval_mode
        threshold = (
            similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        )

        logger.info(f"Retrieving for query: '{query}' [Strategy: {mode}, TopK: {top_k}]")

        if mode == "bm25":
            self._sync_bm25_index()
            scored_bm25 = self.bm25_index.search(query, top_k=top_k)
            candidates = len(scored_bm25)
            final_chunks = []
            for chunk, s in scored_bm25:
                if s >= threshold:
                    chunk.score = s
                    final_chunks.append(chunk)

        elif mode == "dense":
            dense_chunks = self.vector_store.query(
                query_text=query,
                top_k=top_k,
                where_filter=where_filter,
                similarity_threshold=threshold,
            )
            candidates = len(dense_chunks)
            final_chunks = dense_chunks

        else:
            # HYBRID RETRIEVAL: Calibrated linear combination of Dense Cosine + Normalized BM25
            self._sync_bm25_index()
            dense_chunks = self.vector_store.query(
                query_text=query,
                top_k=top_k * 2,
                where_filter=where_filter,
                similarity_threshold=0.0,
            )
            bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
            candidates = len(dense_chunks) + len(bm25_results)

            bm25_scores = {chunk.chunk_id: score for chunk, score in bm25_results}
            dense_scores = {chunk.chunk_id: chunk.score for chunk in dense_chunks}

            all_chunk_map: Dict[str, RetrievedChunk] = {}
            for chunk in dense_chunks:
                all_chunk_map[chunk.chunk_id] = chunk
            for chunk, _ in bm25_results:
                if chunk.chunk_id not in all_chunk_map:
                    all_chunk_map[chunk.chunk_id] = chunk

            # Calculate absolute calibrated hybrid score
            scored_candidates: List[tuple[RetrievedChunk, float]] = []
            for cid, chunk in all_chunk_map.items():
                s_dense = dense_scores.get(cid, 0.0)
                s_bm25 = bm25_scores.get(cid, 0.0)

                # Calibrated hybrid score in [0.0, 1.0]
                hybrid_score = round(
                    (settings.DENSE_WEIGHT * s_dense) + (settings.BM25_WEIGHT * s_bm25),
                    4,
                )

                if hybrid_score >= threshold:
                    chunk.score = hybrid_score
                    scored_candidates.append((chunk, hybrid_score))

            # Rank by hybrid score descending
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            final_chunks = [chunk for chunk, _ in scored_candidates[:top_k]]

        duration = (time.time() - start_time) * 1000

        diagnostics = RetrievalDiagnostics(
            query=query,
            strategy=mode,
            candidates_retrieved=candidates,
            chunks_retained=len(final_chunks),
            duration_ms=round(duration, 2),
            top_scores=[c.score for c in final_chunks],
            chunk_ids=[c.chunk_id for c in final_chunks],
            sources=[c.source for c in final_chunks],
            sections=[c.section for c in final_chunks],
        )

        logger.info(
            f"Retrieval complete in {duration:.2f}ms. Retained {len(final_chunks)}/{candidates} chunks."
        )
        return RetrievalResult(chunks=final_chunks, diagnostics=diagnostics)

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[str]:
        """Backward-compatible retrieval returning document text snippets."""
        res = self.retrieve_with_diagnostics(query=query, top_k=top_k)
        return res.texts
