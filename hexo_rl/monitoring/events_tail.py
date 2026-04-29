"""Tail events_*.jsonl files written by JSONLSink and dispatch each line.

Used by ``scripts/serve_dashboard.py`` so an out-of-process dashboard server
can ingest live events from a separately-running training process. The
training process writes to ``logs/events_<run_id>.jsonl`` via JSONLSink;
this tailer follows the newest such file and forwards each parsed event
to the supplied callback (typically ``WebDashboard.on_event``).

Behaviour:
- Polls the log directory at ``poll_interval_s`` for the newest matching
  file (by mtime). Switches to a newer file when one appears (new run).
- On first attach to a file, replays from offset 0 so the dashboard sees
  the run's history; bounded deques inside WebDashboard cap memory.
- Detects rotation via inode change (handles structlog-style rotators if
  applied later — currently events JSONL doesn't rotate).
- Skips partial trailing lines (no terminating newline) — retried next tick.
- Daemon thread; never raises out of the loop.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Optional

import structlog

log = structlog.get_logger(__name__)


class EventsTailer:
    def __init__(
        self,
        log_dir: str | Path,
        callback: Callable[[dict], None],
        glob_pattern: str = "events_*.jsonl",
        poll_interval_s: float = 0.5,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._callback = callback
        self._glob = glob_pattern
        self._poll = poll_interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._current_path: Optional[Path] = None
        self._inode: Optional[int] = None
        self._offset: int = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="events-tailer"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    @property
    def current_path(self) -> Optional[Path]:
        return self._current_path

    # ── internals ────────────────────────────────────────────────────────────

    def _select_latest(self) -> Optional[Path]:
        if not self._log_dir.is_dir():
            return None
        try:
            candidates = list(self._log_dir.glob(self._glob))
        except OSError:
            return None
        if not candidates:
            return None
        try:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except OSError:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                log.warning("events_tailer_tick_failed", error=str(exc))
            self._stop.wait(self._poll)

    def _tick(self) -> None:
        latest = self._select_latest()
        if latest is None:
            return

        switched = False
        if self._current_path != latest:
            try:
                inode = latest.stat().st_ino
            except OSError:
                return
            if self._current_path is None:
                log.info("events_tailer_attached", path=str(latest))
            else:
                log.info(
                    "events_tailer_switch",
                    previous=str(self._current_path),
                    current=str(latest),
                )
            self._current_path = latest
            self._inode = inode
            self._offset = 0
            switched = True
        else:
            try:
                inode = latest.stat().st_ino
            except OSError:
                return
            if inode != self._inode:
                log.info("events_tailer_rotated", path=str(latest))
                self._inode = inode
                self._offset = 0
                switched = True

        # Read any new bytes since last offset.
        try:
            with open(self._current_path, "rb") as f:
                f.seek(self._offset)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        # Partial line — wait for the writer to finish it.
                        break
                    self._offset += len(line)
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    try:
                        record = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    try:
                        self._callback(record)
                    except Exception as exc:
                        log.warning(
                            "events_tailer_callback_failed",
                            error=str(exc),
                            event=record.get("event"),
                        )
        except OSError as exc:
            log.warning(
                "events_tailer_read_failed",
                error=str(exc),
                path=str(self._current_path),
            )
            return

        if switched:
            log.info(
                "events_tailer_replay_done",
                path=str(self._current_path),
                bytes_read=self._offset,
            )
