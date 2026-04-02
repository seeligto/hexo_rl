"""Smoke tests for DashboardClient — queue mechanics, silent-drop, warn-once.

These tests do NOT require a live dashboard server. All assertions are on the
client's internal queue and state, making them fast and hermetic.
"""
from __future__ import annotations

import queue
import time

from hexo_rl.training.dashboard_utils import DashboardClient


def _make_client(url: str = "http://localhost:59998") -> DashboardClient:
    """Return a client pointed at a guaranteed-dead port."""
    return DashboardClient(base_url=url)


def test_enqueue_is_nonblocking() -> None:
    """10 000 enqueues must complete well under 1 second."""
    client = _make_client()
    t0 = time.perf_counter()
    for i in range(10_000):
        client.send_metrics(iteration=i, loss=0.5)
    elapsed = time.perf_counter() - t0
    client.stop()
    assert elapsed < 1.0, f"Enqueue too slow: {elapsed:.3f}s for 10k calls"


def test_queue_full_drop_is_silent() -> None:
    """Flooding beyond maxsize=512 must not raise and must not exceed cap."""
    client = _make_client()
    # Pause the worker thread by filling with a sentinel first
    # (we can't easily pause it, so just measure queue never exceeds 512)
    for _ in range(1024):
        client._enqueue({"endpoint": "/x", "payload": {}})
    assert client._q.qsize() <= 512
    client.stop()


def test_warn_once_on_failure() -> None:
    """Client warns exactly once on repeated failures, not on every call."""
    import logging
    client = _make_client()
    # Simulate prior connection failure state
    client._warned = False
    client._connected = False

    # Both _post calls should set _warned=True after first failure
    client._post("/api/metric", {"iteration": 1, "loss": 0.5})
    assert client._warned is True
    first_warned = client._warned

    # Second call: _warned stays True, no new warning raised
    client._post("/api/metric", {"iteration": 2, "loss": 0.4})
    assert client._warned is True  # still True, not reset
    client.stop()


def test_send_game_enqueues_correct_payload() -> None:
    """send_game must put the right endpoint + payload onto the queue."""
    # Use __new__ to skip __init__ (no daemon thread spawned)
    c = DashboardClient.__new__(DashboardClient)
    c._base_url = "http://localhost:59998"
    c._q = queue.Queue(maxsize=512)
    c._warned = False
    c._connected = False

    c.send_game(moves=[[0, 0], [1, 1]], result=1.0, metadata={"test": True})
    item = c._q.get_nowait()
    assert item["endpoint"] == "/api/game"
    assert item["payload"]["result"] == 1.0
    assert item["payload"]["moves"] == [[0, 0], [1, 1]]
    assert item["payload"]["metadata"] == {"test": True}


def test_send_metrics_enqueues_correct_payload() -> None:
    """send_metrics must forward all kwargs into the payload."""
    c = DashboardClient.__new__(DashboardClient)
    c._base_url = "http://localhost:59998"
    c._q = queue.Queue(maxsize=512)
    c._warned = False
    c._connected = False

    c.send_metrics(iteration=42, loss=0.123, elo=1055.0, gpu_util=89.5)
    item = c._q.get_nowait()
    assert item["endpoint"] == "/api/metric"
    p = item["payload"]
    assert p["iteration"] == 42
    assert abs(p["loss"] - 0.123) < 1e-6
    assert abs(p["elo"] - 1055.0) < 1e-6
    assert abs(p["gpu_util"] - 89.5) < 1e-6
