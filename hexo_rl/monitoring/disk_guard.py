"""Disk-space monitor: emit disk_free_gb events, WARN at < 10 GB, HARD FAIL at < 5 GB."""
from __future__ import annotations

import gzip
import os
import shutil
import signal
import threading
from pathlib import Path
from typing import Optional

import structlog

from hexo_rl.monitoring.events import emit_event

log = structlog.get_logger()


class DiskGuard:
    """Background thread monitoring disk free space.

    Emits ``disk_free`` events every ``interval_sec`` seconds.
    Logs a warning if free < warn_gb; sends SIGTERM (graceful shutdown) if free < fail_gb.
    SIGTERM triggers the existing shutdown handler in loop.py — buffer is saved before exit.

    ``keep_all`` is passed through to call sites for pruning policy; it does NOT
    disable the disk-space thresholds (those are a safety guard, not a pruning knob).
    """

    def __init__(
        self,
        watch_path: str | Path = ".",
        interval_sec: float = 60.0,
        warn_gb: float = 10.0,
        fail_gb: float = 5.0,
        keep_all: bool = False,
    ) -> None:
        self._path = Path(watch_path)
        self._interval = interval_sec
        self._warn_gb = warn_gb
        self._fail_gb = fail_gb
        self.keep_all = keep_all
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="disk-guard",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def check_once(self) -> float:
        """Check disk free, emit event, handle thresholds. Returns free_gb."""
        usage = shutil.disk_usage(self._path)
        free_gb = usage.free / 1e9
        emit_event({"event": "disk_free", "disk_free_gb": round(free_gb, 2)})

        if free_gb < self._fail_gb:
            log.error(
                "disk_critical",
                free_gb=round(free_gb, 2),
                fail_threshold_gb=self._fail_gb,
                msg="Disk critically low — sending SIGTERM to halt training cleanly",
            )
            emit_event({"event": "disk_alert", "level": "critical", "disk_free_gb": round(free_gb, 2)})
            os.kill(os.getpid(), signal.SIGTERM)
        elif free_gb < self._warn_gb:
            log.warning(
                "disk_low_warn",
                free_gb=round(free_gb, 2),
                warn_threshold_gb=self._warn_gb,
            )
            emit_event({"event": "disk_alert", "level": "warn", "disk_free_gb": round(free_gb, 2)})

        return free_gb

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            try:
                self.check_once()
            except Exception as exc:
                log.warning("disk_guard_error", error=str(exc))


def gzip_rotate(source: str, dest: str) -> None:
    """Gzip ``source`` to ``dest``, then remove ``source``. Used by RotatingFileHandler."""
    with open(source, "rb") as f_in, gzip.open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(source)
