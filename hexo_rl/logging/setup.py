"""
Structlog configuration for Hex Tac Toe AlphaZero.

Phase 2: JSON logs to file + pretty console output via rich.

Usage:
    from hexo_rl.logging.setup import configure_logging
    log = configure_logging(log_dir="logs", run_name="phase2_train")
    log.info("train_step", iteration=1, policy_loss=0.4, value_loss=0.1)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog


def configure_logging(
    log_dir: str = "logs",
    run_name: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> structlog.BoundLogger:
    """Configure structlog for structured JSON logging to file.

    Writes one JSON object per line to ``<log_dir>/<run_name>.jsonl``.
    Optionally also prints a human-readable version to stdout.

    Args:
        log_dir:  Directory for log files (created if absent).
        run_name: Identifier for this run (default: timestamp).
        level:    Minimum log level (default INFO).
        console:  If True, also emit pretty output to stdout.

    Returns:
        A structlog bound logger ready to use.
    """
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{run_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = log_path.open("a")

    # ── File handler: JSON lines ──────────────────────────────────────────
    # Configure stdlib root logger to route through structlog's file output.
    file_handler = logging.StreamHandler(log_file)
    file_handler.setLevel(level)

    # ── Console handler (optional) ────────────────────────────────────────
    handlers: list[logging.Handler] = [file_handler]
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        handlers.append(console_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )

    # ── Structlog processors ──────────────────────────────────────────────
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.WriteLoggerFactory(file=log_file),
        cache_logger_on_first_use=True,
    )

    log = structlog.get_logger()
    log.info("logging_configured", log_path=str(log_path), run_name=run_name)
    return log
