"""
Structlog configuration for Hex Tac Toe AlphaZero.

JSON logs to rotating gzip-compressed file + optional console output.
Rotation fires at max_log_bytes (default 500 MB); up to backup_count
rotated files are kept, each gzip-compressed.

Usage:
    from hexo_rl.monitoring.configure import configure_logging
    log, handler = configure_logging(log_dir="logs", run_name="phase4_train")
    log.info("train_step", iteration=1, policy_loss=0.4, value_loss=0.1)
    # on shutdown:
    handler.close()
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional

import structlog
from structlog.stdlib import ProcessorFormatter


def _gzip_rotator(source: str, dest: str) -> None:
    """Rotate ``source`` → ``dest`` with gzip compression."""
    import gzip
    with open(source, "rb") as f_in, gzip.open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(source)


def configure_logging(
    log_dir: str = "logs",
    run_name: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    max_log_bytes: int = 500 * 1024 * 1024,
    backup_count: int = 5,
) -> Tuple[structlog.BoundLogger, logging.Handler]:
    """Configure structlog for structured JSON logging with size-based rotation.

    Writes one JSON object per line to ``<log_dir>/<run_name>.jsonl``.
    Files rotate at ``max_log_bytes``; rotated files are gzip-compressed
    (suffix ``.jsonl.1.gz``, ``.jsonl.2.gz``, …).
    Optionally also emits JSON to stdout.

    Args:
        log_dir:       Directory for log files (created if absent).
        run_name:      Identifier for this run (default: timestamp).
        level:         Minimum log level (default INFO).
        console:       If True, also emit JSON to stdout.
        max_log_bytes: Rotate when file exceeds this size (default 500 MB).
        backup_count:  Number of rotated files to keep (default 5).

    Returns:
        (bound_logger, file_handler) — call ``file_handler.close()`` on shutdown.
    """
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{run_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Shared pre-processors applied to all structlog events before rendering.
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Route structlog calls through stdlib so RotatingFileHandler handles rotation.
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    json_formatter = ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    # ── Rotating file handler ─────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_log_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.rotator = _gzip_rotator
    file_handler.namer = lambda name: name + ".gz"
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(level)

    handlers: list[logging.Handler] = [file_handler]

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        console_handler.setLevel(level)
        handlers.append(console_handler)

    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)

    log = structlog.get_logger()
    log.info("logging_configured", log_path=str(log_path), run_name=run_name)
    return log, file_handler
