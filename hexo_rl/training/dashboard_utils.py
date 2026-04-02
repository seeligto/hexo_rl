"""
Fire-and-forget dashboard bridge.

Pushes training data to a running dashboard.py server via HTTP POST.
Uses a background daemon thread + queue so the training loop never blocks.
If the server is not running, logs a warning once and silently discards data.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)


class DashboardClient:
    def __init__(self, base_url: str = "http://localhost:5001") -> None:
        self._base_url = base_url.rstrip("/")
        self._q: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(maxsize=512)
        self._warned = False    # emit "server not reachable" warning only once
        self._connected = False  # track recovery to log reconnection
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="dashboard-client"
        )
        self._thread.start()

    def send_game(
        self,
        moves: List,
        result: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Enqueue a completed game for the dashboard. Non-blocking."""
        self._enqueue({
            "endpoint": "/api/game",
            "payload": {"moves": moves, "result": result, "metadata": metadata or {}},
        })

    def send_metrics(
        self,
        iteration: int,
        loss: float,
        elo: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Enqueue training metrics. Non-blocking."""
        payload: Dict[str, Any] = {"iteration": iteration, "loss": loss}
        if elo is not None:
            payload["elo"] = elo
        payload.update(kwargs)
        self._enqueue({"endpoint": "/api/metric", "payload": payload})

    def stop(self) -> None:
        """Signal the worker thread to drain and exit."""
        self._q.put(None)

    # ── internals ─────────────────────────────────────────────────────────────

    def _enqueue(self, item: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            pass  # drop silently — dashboard is a best-effort side channel

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            self._post(item["endpoint"], item["payload"])

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> None:
        url = self._base_url + endpoint
        try:
            requests.post(url, json=payload, timeout=0.05)
            if not self._connected:
                if self._warned:
                    log.info("dashboard reconnected at %s", self._base_url)
                self._connected = True
        except Exception:
            self._connected = False
            if not self._warned:
                log.warning(
                    "Dashboard server not reachable at %s — "
                    "game/metric data will be discarded silently.",
                    self._base_url,
                )
                self._warned = True
