"""Event emitter for training monitoring.

Single fan-out dispatch: train.py calls emit_event(payload), which adds a
timestamp and dispatches to all registered renderers.  Never raises —
renderer failures are caught and printed to stderr.

Renderers implement: on_event(self, payload: dict) -> None

JSONLSink renderer
------------------
``register_jsonl_sink(path)`` attaches a renderer that appends each event
payload as one JSON line to ``path``. This lets an out-of-process dashboard
tail the file (see hexo_rl.monitoring.events_tail) and reconstruct the live
event stream without any IPC. Single shared file per run; line-buffered
writes; thread-safe.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

_renderers: list = []
_lock = threading.Lock()


def register_renderer(renderer: Any) -> None:
    """Register a renderer that will receive all events via on_event()."""
    with _lock:
        _renderers.append(renderer)


def emit_event(payload: dict[str, Any]) -> None:
    """Add ts, then dispatch to all registered renderers.

    Never raises — failures are caught and logged to stderr only.
    """
    payload = {"ts": time.time(), **payload}
    with _lock:
        targets = list(_renderers)
    for r in targets:
        try:
            r.on_event(payload)
        except Exception as exc:
            print(f"[dashboard] renderer {r} failed: {exc}", file=sys.stderr)


class JSONLSink:
    """Append every event payload as one JSON line to ``path``.

    Used so an out-of-process dashboard server can tail the file and replay
    events into its own renderer chain — no shared memory, no socket.
    """

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append; one open file shared by all event sources.
        self._fh = open(p, "a", buffering=1, encoding="utf-8")
        self._path = p
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def on_event(self, payload: dict) -> None:
        try:
            line = json.dumps(payload, default=str, ensure_ascii=False) + "\n"
        except Exception:
            return
        with self._lock:
            try:
                self._fh.write(line)
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:
                pass


def register_jsonl_sink(path: str | Path) -> JSONLSink:
    """Register a JSONLSink writing every event to ``path``. Returns the sink."""
    sink = JSONLSink(path)
    register_renderer(sink)
    return sink
