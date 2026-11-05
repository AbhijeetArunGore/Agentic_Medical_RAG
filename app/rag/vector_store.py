"""Persistent ChromaDB vector store manager for clinical knowledge retrieval.

Architectural Mechanics:
- ChromaDB Persistence: Local SQLite + HNSW index at VECTOR_DB_PATH.
- Native Embedding Function: Uses MedicalEmbeddingFunction to prevent unexpected default downloads.
- Metadata Preservation: Indexes document_id, chunk_id, page_number, source, and clinical section.
- Safe Seeding: Prevents duplicate insertions by checking existing chunk IDs.
- Diagnostic Retrieval: Returns document texts, scores, and chunk provenance metadata.
"""

import math
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.rag.embeddings import MedicalEmbeddingFunction
from app.rag.chunking import TextChunk
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("VectorStore")

DEFAULT_GUIDELINES = [
    {
        "id": "guideline_ada_2026_c0",
        "text": "Guideline for Diabetic Nephropathy 2026: ACE inhibitors or ARBs are recommended for T2D patients with rising creatinine and persistent proteinuria to slow kidney disease progression.",
        "metadata": {
            "source": "ADA 2026 Standards of Care",
            "type": "Guideline",
            "document_id": "guideline_ada_2026",
            "section": "Diabetic Kidney Disease",
            "page_number": 14,
        },
    },
    {
        "id": "guideline_fda_contraindication_c0",
        "text": "Critical Safety Contraindication: ACE inhibitors and ARBs are absolutely contraindicated in bilateral renal artery stenosis due to risk of precipitating acute renal shutdown and ischemic nephropathy.",
        "metadata": {
            "source": "FDA Safety Alert",
            "type": "Contraindication",
            "document_id": "guideline_fda_contraindication",
            "section": "Black Box Warnings",
            "page_number": 1,
        },
    },
    {
        "id": "guideline_kdigo_2025_c0",
        "text": "KDIGO 2025 Standard: SGLT2 inhibitors (e.g., Empagliflozin, Dapagliflozin) reduce kidney failure progression and cardiovascular mortality in patients with CKD and Type 2 Diabetes.",
        "metadata": {
            "source": "KDIGO 2025",
            "type": "Guideline",
            "document_id": "guideline_kdigo_2025",
            "section": "Pharmacotherapy",
            "page_number": 32,
        },
    },
    {
        "id": "guideline_metformin_protocol_c0",
        "text": "Clinical Dosing Protocol: Metformin dose reduction to 1000mg daily is required when eGFR falls between 30-44 mL/min/1.73m2. Metformin is contraindicated if eGFR is below 30 due to lactic acidosis risk.",
        "metadata": {
            "source": "Clinical Pharmacy Guide",
            "type": "Dosing Protocol",
            "document_id": "guideline_metformin_protocol",
            "section": "Renal Dosing Adjustments",
            "page_number": 5,
        },
    },
    {
        "id": "guideline_hypertension_jnc8_c0",
        "text": "JNC-8 Hypertension Guideline: Target blood pressure for patients with diabetes and chronic kidney disease is < 130/80 mmHg to reduce macrovascular and microvascular renal damage.",
        "metadata": {
            "source": "JNC-8 Guidelines",
            "type": "Guideline",
            "document_id": "guideline_hypertension_jnc8",
            "section": "Hypertension Targets",
            "page_number": 8,
        },
    },
    {
        "id": "guideline_insulin_renal_c0",
        "text": "Insulin Glargine Dosing in Renal Impairment: Renal clearance of insulin declines in CKD Stage 3b-5. Titrate basal insulin conservatively by 25-50% to prevent severe prolonged hypoglycemia.",
        "metadata": {
            "source": "Endocrine Society Clinical Practice",
            "type": "Dosing Protocol",
            "document_id": "guideline_insulin_renal",
            "section": "Insulin Titration",
            "page_number": 22,
        },
    },
]


class RetrievedChunk(BaseModel):
    """Schema for retrieved document chunks with scoring and metadata."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str
    page_number: int
    section: str
    metadata: Dict[str, Any] = {}


class IncompatibleEmbeddingSpaceError(RuntimeError):
    """Raised when active embedding model does not match persisted vector collection metadata."""

    pass


class VectorStoreManager:
    """Manages persistent ChromaDB vector storage with safe lifecycle handling."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[MedicalEmbeddingFunction] = None,
        force_reseed: bool = False,
        force_recreate: Optional[bool] = None,
    ):
        self.persist_directory = Path(persist_directory or settings.VECTOR_DB_DIR)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.embedding_function = embedding_function or MedicalEmbeddingFunction()
        should_reseed = force_reseed or (force_recreate is True)

        logger.info(f"Initializing ChromaDB client at: {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        active_model = self.embedding_function.model_identifier
        active_mode = "mock" if self.embedding_function.is_mock else "neural"
        target_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": active_model,
            "embedding_mode": active_mode,
            "embedding_dim": 384,
        }

        if should_reseed:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(
                    f"Deleted existing collection '{self.collection_name}' on force_reseed."
                )
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata=target_metadata,
            )
            return

        # Check existing collection compatibility
        try:
            # First fetch raw collection to inspect metadata without triggering Chroma internal EF conflict
            existing_col = self.client.get_collection(name=self.collection_name)
            meta = existing_col.metadata or {}
            stored_model = meta.get("embedding_model")
            stored_mode = meta.get("embedding_mode")

            if existing_col.count() > 0:
                if not stored_model:
                    raise IncompatibleEmbeddingSpaceError(
                        f"Persisted ChromaDB collection '{self.collection_name}' contains "
                        f"{existing_col.count()} documents but lacks embedding provenance metadata "
                        f"('embedding_model'). Operating against an unversioned legacy index risks "
                        f"coordinate space corruption. Reseed explicitly via force_reseed=True or "
                        f"'python main.py --setup-db'."
                    )
                if stored_model != active_model:
                    raise IncompatibleEmbeddingSpaceError(
                        f"Persisted ChromaDB collection '{self.collection_name}' was indexed using "
                        f"embedding_model='{stored_model}' (mode='{stored_mode}'), but active configuration is "
                        f"'{active_model}' (mode='{active_mode}'). Querying across mismatched embedding coordinate "
                        f"spaces produces corrupt cosine distances. Reseed explicitly via force_reseed=True or "
                        f"'python main.py --setup-db'."
                    )
            # Re-bind active embedding function
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        except IncompatibleEmbeddingSpaceError:
            raise
        except ValueError as e:
            # ChromaDB internal embedding function configuration conflict
            raise IncompatibleEmbeddingSpaceError(
                f"Persisted ChromaDB collection '{self.collection_name}' has an incompatible "
                f"embedding function configuration ({e}). Reseed explicitly via force_reseed=True or "
                f"'python main.py --setup-db'."
            ) from e
        except Exception:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata=target_metadata,
            )

    def count(self) -> int:
        """Returns the number of indexed records in the collection."""
        return self.collection.count()

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """Indexes TextChunk objects, skipping any chunks already present."""
        if not chunks:
            return 0

        chunk_ids = [c.chunk_id for c in chunks]
        # Check existing
        existing = self.collection.get(ids=chunk_ids)
        existing_ids = set(existing.get("ids", []))

        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            logger.info("All provided chunks already indexed; no new chunks to add.")
            return 0

        documents = [c.text for c in new_chunks]
        ids = [c.chunk_id for c in new_chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "source": c.source,
                "page_number": c.page_number,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "content_hash": c.content_hash,
                **c.metadata,
            }
            for c in new_chunks
        ]

        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f"Indexed {len(new_chunks)} new chunks into '{self.collection_name}'.")
        return len(new_chunks)

    def seed_initial_knowledge_base(
        self,
        guidelines: Optional[List[Dict[str, Any]]] = None,
        force_reseed: bool = False,
    ) -> int:
        """Seeds default clinical guidelines if not already present or if force_reseed=True."""
        data_to_seed = guidelines or DEFAULT_GUIDELINES
        ids = [g["id"] for g in data_to_seed]

        active_model = self.embedding_function.model_identifier
        active_mode = "mock" if self.embedding_function.is_mock else "neural"
        target_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": active_model,
            "embedding_mode": active_mode,
            "embedding_dim": 384,
        }

        if force_reseed:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}' for clean reseed.")
            except Exception as e:
                logger.info(f"Collection deletion during force_reseed: {e}")

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata=target_metadata,
            )

        existing = self.collection.get(ids=ids)
        existing_ids = set(existing.get("ids", []))

        items_to_add = [g for g in data_to_seed if g["id"] not in existing_ids]
        if not items_to_add:
            logger.info("Default guidelines already seeded in vector collection.")
            return self.collection.count()

        documents = [g["text"] for g in items_to_add]
        add_ids = [g["id"] for g in items_to_add]
        metadatas = [g.get("metadata", {}) for g in items_to_add]

        self.collection.add(documents=documents, metadatas=metadatas, ids=add_ids)
        logger.info(
            f"Seeded {len(items_to_add)} documents into collection '{self.collection_name}' with model '{active_model}'."
        )
        return self.collection.count()

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        where_filter: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Perform semantic similarity search and return structured RetrievedChunk items."""
        if self.collection.count() == 0:
            logger.info("Collection empty during query; seeding default guidelines.")
            self.seed_initial_knowledge_base()

        n_results = min(top_k, self.collection.count())
        if n_results == 0:
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.SIMILARITY_THRESHOLD
        )
        chunks: List[RetrievedChunk] = []

        for chunk_id, doc_text, meta, dist in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine distance: dist = 1 - cosine_similarity
            # Similarity score = 1 - dist
            similarity = max(0.0, min(1.0, 1.0 - dist))
            if similarity < threshold:
                continue

            meta = meta or {}
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", "unknown_doc"),
                text=doc_text,
                score=round(similarity, 4),
                source=meta.get("source", "Clinical Guidelines"),
                page_number=meta.get("page_number", 1),
                section=meta.get("section", "General"),
                metadata=meta,
            )
            chunks.append(chunk)

        return chunks

    def reset_collection(self) -> None:
        """Deletes and recreates the current vector collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Vector collection '{self.collection_name}' has been reset.")
