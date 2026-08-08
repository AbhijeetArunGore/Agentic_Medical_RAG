"""Vector store management wrapper using ChromaDB.

Architectural Mechanics:
- Persistent ChromaDB Initialization: Initializes a local SQLite & HNSW index at VECTOR_DB_PATH.
- Collection Creation: Manages collection 'clinical_guidelines' with metadata filtering capability.
- Embedding Storage: Embeds guidelines and clinical trial evidence into 384-dimensional vector space.
- Retrieval Strategy: Performs k-Nearest Neighbors (k-NN) cosine similarity search for adaptive RAG.
"""

import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("VectorStoreManager")

DEFAULT_GUIDELINES = [
    {
        "id": "G1",
        "text": "Guideline for Diabetic Nephropathy 2026: ACE inhibitors or ARBs are recommended for T2D patients with rising creatinine and proteinuria.",
        "metadata": {"source": "ADA 2026", "type": "Guideline"},
    },
    {
        "id": "G2",
        "text": "Contraindication Notice: ACE inhibitors and ARBs are absolutely contraindicated in bilateral renal artery stenosis due to risk of acute renal failure.",
        "metadata": {"source": "FDA Safety Alert", "type": "Contraindication"},
    },
    {
        "id": "G3",
        "text": "KDIGO 2025 Standard: SGLT2 inhibitors reduce kidney failure progression and cardiovascular events in patients with CKD and T2D.",
        "metadata": {"source": "KDIGO 2025", "type": "Guideline"},
    },
    {
        "id": "G4",
        "text": "Clinical Protocol: Metformin dose reduction is required when eGFR falls below 45 mL/min/1.73m2, and contraindicated if eGFR < 30.",
        "metadata": {"source": "Clinical Pharmacy", "type": "Dosing Protocol"},
    },
]


class VectorStoreManager:
    """Manages persistent ChromaDB vector storage and clinical collection indexing."""

    def __init__(
        self,
        persist_directory: Path = settings.VECTOR_DB_DIR,
        collection_name: str = "clinical_guidelines",
    ):
        """Initialize persistent ChromaDB client.

        Args:
            persist_directory (Path): Directory where vector indices are stored.
            collection_name (str): Name of the Chroma collection.
        """
        # Step 1: Resolve persistence path dynamically using settings (avoids hardcoded OS paths)
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(f"Initializing ChromaDB persistent client at: {self.persist_directory}")
        # Step 2: Initialize ChromaDB persistent storage engine
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        # Step 3: Retrieve or create target vector collection
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def seed_initial_knowledge_base(self, guidelines: Optional[List[Dict[str, Any]]] = None) -> int:
        """Indexes initial clinical guidelines into the vector store with metadata tags.

        Args:
            guidelines (List[Dict]): Custom guidelines list or defaults to DEFAULT_GUIDELINES.

        Returns:
            int: Number of indexed documents.
        """
        data_to_seed = guidelines or DEFAULT_GUIDELINES
        documents = [g["text"] for g in data_to_seed]
        metadatas = [g.get("metadata", {}) for g in data_to_seed]
        ids = [g.get("id", f"doc_{i}") for i, g in enumerate(data_to_seed)]

        # Step 4: Index documents and metadata into persistent HNSW vector store
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(
            f"Successfully seeded {len(documents)} documents into vector collection '{self.collection_name}'."
        )
        return len(documents)

    def query(self, query_text: str, top_k: int = 3) -> List[str]:
        """Perform semantic k-NN vector search query against indexed medical documents.

        Args:
            query_text (str): Clinical search query string.
            top_k (int): Number of top matching documents to return.

        Returns:
            List[str]: List of matching document texts.
        """
        # Step 5: Auto-seed default clinical guidelines if collection is empty upon query
        if self.collection.count() == 0:
            logger.info("Collection empty during query; seeding default guidelines.")
            self.seed_initial_knowledge_base()

        # Step 6: Execute semantic vector retrieval query
        results = self.collection.query(
            query_texts=[query_text], n_results=min(top_k, self.collection.count())
        )
        docs = results.get("documents", [[]])[0]
        return docs
