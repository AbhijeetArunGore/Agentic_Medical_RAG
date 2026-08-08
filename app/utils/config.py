"""Central configuration management module for Agentic Medical RAG.

Provides dynamic path resolution and environment variable access without hardcoded paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory: root of the repository
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file if present
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Application settings with environment fallback defaults."""

    # Project Paths
    ROOT_DIR: Path = BASE_DIR
    APP_DIR: Path = BASE_DIR / "app"
    DATA_DIR: Path = BASE_DIR / "data"
    VECTOR_DB_DIR: Path = Path(os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "vector_db")))
    DOCS_DIR: Path = BASE_DIR / "docs"

    # Model & API Configurations
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Logging & Debugging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist on startup."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        cls.DOCS_DIR.mkdir(parents=True, exist_ok=True)


# Instantiated Singleton Configuration
settings = Settings()
settings.ensure_directories()
