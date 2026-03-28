"""
Structlog configuration for Hex Tac Toe AlphaZero.

Phase 1: JSON logs to file only (no rich dashboard — that's Phase 2).
Phase 2: Add rich live console display.

Usage:
    from python.logging.setup import configure_logging
    log = configure_logging(log_dir="logs", run_name="phase1_debug")
    log.info("train_step", iteration=1, loss=0.5)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import structlog


def configure_logging(
    log_dir: str = "logs",
    run_name: str | None = None,
    level: int = logging.INFO,
) -> structlog.BoundLogger:
    """Configure structlog to write JSON lines to a file.

    Args:
        log_dir:  Directory for log files (created if absent).
        run_name: Identifier for this run (default: timestamp).
        level:    Minimum log level (default INFO).

    Returns:
        A structlog bound logger ready to use.
    """
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{run_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.WriteLoggerFactory(file=log_path.open("a")),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()
