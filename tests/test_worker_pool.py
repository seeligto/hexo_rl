"""
Tests for WorkerPool — multiprocess self-play.

These tests start real worker processes and verify end-to-end behaviour.
They use a tiny config (n_sims=5, n_workers=2, small model) to run fast.

Because spawning processes has overhead, the test allows up to 60 s for
games to appear in the replay buffer.
"""

from __future__ import annotations

import time
from typing import Dict, Any

import pytest
import torch
import torch.multiprocessing as mp

from python.model.network import HexTacToeNet
from python.training.replay_buffer import ReplayBuffer
from python.selfplay.pool import WorkerPool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_config() -> Dict[str, Any]:
    """Minimal config for fast test runs."""
    return {
        "mcts": {
            "n_simulations": 5,
            "c_puct": 1.5,
            "temperature_threshold_ply": 30,
        },
        "selfplay": {
            "n_workers": 2,
            "inference_batch_size": 4,
            "inference_max_wait_ms": 20.0,
        },
    }


def _small_model(device: torch.device) -> HexTacToeNet:
    return HexTacToeNet(
        board_size=19,
        in_channels=18,
        filters=32,
        res_blocks=2,
    ).to(device)


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.timeout(90)
def test_worker_pool_produces_positions():
    """Start a pool with 2 workers, wait until ≥1 game completes."""
    # torch.multiprocessing requires spawn start method.
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # already set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _small_model(device)
    buffer = ReplayBuffer(capacity=10_000)
    config = _tiny_config()

    try:
        pool = WorkerPool(model, config, device, buffer, n_workers=2)
    except PermissionError as exc:
        # Some sandboxed environments block POSIX semaphore creation, which
        # prevents multiprocessing queues from being constructed at all.
        pytest.skip(f"multiprocessing queues not permitted here: {exc}")
    pool.start()

    try:
        # Wait up to 60 s for at least 1 game to finish.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if pool.games_completed >= 1:
                break
            time.sleep(0.5)

        assert pool.games_completed >= 1, (
            f"No games completed within timeout. "
            f"positions_pushed={pool.positions_pushed}"
        )
        assert pool.positions_pushed > 0, "No positions pushed to replay buffer"
        assert buffer.size > 0, "Replay buffer is still empty"
    finally:
        pool.stop()


@pytest.mark.timeout(90)
def test_worker_pool_respects_backpressure():
    """Pool should not overflow the request queue; it stays bounded."""
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = _small_model(device)
    buffer = ReplayBuffer(capacity=10_000)
    config = _tiny_config()

    try:
        pool = WorkerPool(model, config, device, buffer, n_workers=2)
    except PermissionError as exc:
        pytest.skip(f"multiprocessing queues not permitted here: {exc}")
    pool.start()

    # Let it run for a short time, verify no exceptions and request queue stays bounded.
    time.sleep(5.0)
    qsize = pool._request_queue.qsize()
    maxsize = pool._request_queue._maxsize  # type: ignore[attr-defined]

    pool.stop()

    # The queue should never exceed its maxsize (backpressure is working).
    assert qsize <= maxsize, f"Queue overflow: qsize={qsize}, maxsize={maxsize}"
