"""Replay-buffer persistence helper. Extracted from orchestrator.py per §176 P15."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import structlog

log = structlog.get_logger(__name__)


def try_save_buffer(
    buffer: Any,
    mixing_cfg: dict[str, Any],
    trigger: str,
    recent_buffer: Optional[Any] = None,
) -> None:
    """Save replay buffer (and optionally recent_buffer) if buffer_persist is enabled."""
    if not mixing_cfg.get("buffer_persist", False):
        return
    bp = Path(mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
    try:
        buffer.save_to_path(str(bp))
        log.info("buffer_saved", path=str(bp), positions=buffer.size, trigger=trigger)
    except Exception as exc:
        log.warning("buffer_save_failed", path=str(bp), error=str(exc))
    if recent_buffer is not None and recent_buffer.size > 0:
        rbp = Path(str(bp) + ".recent")
        try:
            n = recent_buffer.save_to_path(str(rbp))
            log.info("recent_buffer_saved", path=str(rbp), positions=n, trigger=trigger)
        except Exception as exc:
            log.warning("recent_buffer_save_failed", path=str(rbp), error=str(exc))
