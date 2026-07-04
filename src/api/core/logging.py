"""Structured logging configuration using loguru.

Import and call setup_logging() once in main.py lifespan.
All other modules just do: from loguru import logger
"""
from __future__ import annotations

import logging
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route stdlib logging through loguru so uvicorn/SQLAlchemy logs are unified."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure loguru.

    Parameters
    ----------
    level : str
        Minimum log level (DEBUG | INFO | WARNING | ERROR | CRITICAL).
    json : bool
        If True, emit JSON lines (suited for production log aggregation).
    """
    logger.remove()

    if json:
        logger.add(
            sys.stdout,
            level=level,
            serialize=True,
            enqueue=True,
        )
    else:
        logger.add(
            sys.stdout,
            level=level,
            colorize=True,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            enqueue=True,
        )

    # Intercept stdlib loggers
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "sqlalchemy.engine", "apscheduler"):
        _lib_logger = logging.getLogger(name)
        _lib_logger.handlers = [_InterceptHandler()]
        _lib_logger.propagate = False
