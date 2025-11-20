"""Lightweight logging helpers shared by CLI scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .config import LOG_DIR


def get_logger(name: str, logfile: Optional[Path] = None) -> logging.Logger:
    """Return configured logger writing to LOG_DIR."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    log_path = logfile if logfile else LOG_DIR / "etl.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger
