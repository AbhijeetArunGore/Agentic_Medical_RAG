"""Centralized logger for Agentic Medical RAG."""

import logging
import sys
from app.utils.config import settings


def get_logger(name: str) -> logging.Logger:
    """Configures and returns a standardized logger instance.

    Args:
        name (str): Name of the logger module.

    Returns:
        logging.Logger: Standard python logger configured with custom format.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
