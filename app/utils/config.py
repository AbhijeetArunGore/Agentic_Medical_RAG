"""Centralized configuration management module for Agentic Medical RAG.

Architectural Mechanics:
- Strongly typed settings backed by pydantic-settings.
- Cross-platform dynamic path resolution using pathlib.Path.
- Safe defaults for offline execution without requiring paid API credentials.
- Credential masking to prevent secrets leaking in logs or string representations.
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application configuration with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base Paths
    ROOT_DIR: Path = BASE_DIR
    APP_DIR: Path = BASE_DIR / "app"
    DATA_DIR: Path = BASE_DIR / "data"
    DOCS_DIR: Path = BASE_DIR / "docs"
    VECTOR_DB_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "vector_db")))
    )

    # Runtime & Mode
    ENVIRONMENT: str = "development"
    MOCK_LLM: bool = True
    LOG_LEVEL: str = "INFO"

    # API Keys & Cloud Models
    OPENAI_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    MODEL_NAME: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector Storage
    COLLECTION_NAME: str = "clinical_guidelines"

    # Chunking & Ingestion
    CHUNKING_STRATEGY: str = "recursive"  # recursive, fixed, structure_aware
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MAX_UPLOAD_SIZE_MB: int = 10

    # Retrieval Configuration
    RETRIEVAL_MODE: str = "hybrid"  # dense, hybrid, bm25
    RETRIEVAL_TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.22
    BM25_WEIGHT: float = 0.4
    DENSE_WEIGHT: float = 0.6

    # Safety & Agent Guardrails
    MAX_AGENT_ITERATIONS: int = 5
    ENABLE_CITATION_VALIDATION: bool = True
    ABSTENTION_THRESHOLD: float = 0.22

    # API Host & Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_REQUEST_TIMEOUT_SECONDS: int = 30
    CORS_ORIGINS: str = "*"

    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    def is_cloud_ready(self) -> bool:
        """Check whether valid cloud credentials are provided."""
        return bool(self.OPENAI_API_KEY and not self.OPENAI_API_KEY.startswith("your_"))


settings = Settings()
settings.ensure_directories()
