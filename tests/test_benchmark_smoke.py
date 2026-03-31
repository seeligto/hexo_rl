from __future__ import annotations

import torch

from python.model.network import HexTacToeNet
from native_core import RustReplayBuffer
from scripts.benchmark import benchmark_replay_buffer, benchmark_worker_pool


def test_replay_buffer_benchmark_smoke() -> None:
    """Replay benchmark should run and return expected metrics keys."""
    replay = RustReplayBuffer(capacity=1024)
    for _ in range(512):
        replay.push(
            torch.zeros((18, 19, 19), dtype=torch.float16).numpy(),
            (torch.ones((362,), dtype=torch.float32) / 362.0).numpy(),
            0.0,
        )

    result = benchmark_replay_buffer(replay)
    assert result["name"] == "Replay buffer sample (batch=256)"
    assert result["us_per_sample"] >= 0.0
    assert result["aug_ms"] >= 0.0


def test_worker_pool_benchmark_smoke() -> None:
    """Worker pool benchmark should execute briefly and return throughput stats."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexTacToeNet(board_size=19, in_channels=18, filters=32, res_blocks=2).to(device)

    config = {
        "n_simulations": 5,
        "c_puct": 1.5,
        "temperature_threshold_ply": 30,
        "inference_batch_size": 4,
        "inference_max_wait_ms": 10.0,
        "dispatch_wait_ms": 2.0,
        "leaf_batch_size": 4,
    }

    result = benchmark_worker_pool(
        model=model,
        config=config,
        device=device,
        duration_sec=3,
        n_workers=1,
    )

    assert result["name"] == "Worker pool throughput"
    assert result["elapsed_sec"] > 0.0
    assert result["games_completed"] >= 0
    assert result["positions_pushed"] >= 0
    assert result["games_per_hour"] >= 0.0
