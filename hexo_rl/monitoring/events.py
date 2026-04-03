"""Event emitter for training monitoring.

Single fan-out dispatch: train.py calls emit_event(payload), which adds a
timestamp and dispatches to all registered renderers.  Never raises —
renderer failures are caught and printed to stderr.

Renderers implement: on_event(self, payload: dict) -> None
"""

from __future__ import annotations

import sys
import threading
import time
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
