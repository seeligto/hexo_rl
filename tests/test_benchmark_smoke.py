from __future__ import annotations

import numpy as np
import torch

from hexo_rl.model.network import HexTacToeNet
from engine import ReplayBuffer
from scripts.benchmark import benchmark_replay_buffer, benchmark_worker_pool


def test_replay_buffer_benchmark_smoke() -> None:
    """Replay benchmark should run and return expected metrics keys."""
    replay = ReplayBuffer(capacity=1024)
    own   = np.ones(361, dtype=np.uint8)
    wl    = np.zeros(361, dtype=np.uint8)
    chain = np.zeros((6, 19, 19), dtype=np.float16)
    for _ in range(512):
        replay.push(
            torch.zeros((18, 19, 19), dtype=torch.float16).numpy(),
            chain,
            (torch.ones((362,), dtype=torch.float32) / 362.0).numpy(),
            0.0,
            own, wl,
        )

    result = benchmark_replay_buffer(replay, n_runs=1, warmup_sec=0.5)
    assert result["name"] == "Replay buffer"
    assert result["push"]["value"] > 0.0
    assert result["raw"]["value"] >= 0.0
    assert result["aug"]["value"] >= 0.0


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
        n_runs=1,
        warmup_sec=3.0,
    )

    assert result["name"] == "Worker pool throughput"
    assert result["pph"]["value"] >= 0.0
    assert result["gph"]["value"] >= 0.0
    assert result["bat"]["value"] >= 0.0
