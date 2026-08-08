"""Helper utility functions for string formatting, file IO, and data sanitization."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def safe_read_file(file_path: Path) -> str:
    """Safely read text content from a file path.

    Args:
        file_path (Path): Absolute or relative Path object.

    Returns:
        str: File content or empty string if file does not exist.
    """
    if not file_path.exists():
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def safe_read_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely read JSON content from a file path.

    Args:
        file_path (Path): Path to JSON file.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON content or None.
    """
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None


def clean_clinical_text(text: str) -> str:
    """Clean raw clinical text by stripping whitespace and normalizing line breaks.

    Args:
        text (str): Raw text string.

    Returns:
        str: Cleaned text.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)
