"""
Structured logging setup.
All modules import `get_logger` from here.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    log_dir = os.environ.get("LOG_DIR", "logs")
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        filename=Path(log_dir) / "bot.log",
        maxBytes=20 * 1024 * 1024,   # 20 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
