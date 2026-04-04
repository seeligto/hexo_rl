"""Tests for Phase 3.5 self-play concurrency interfaces.

Legacy multiprocessing queue tests were removed. These tests validate:
1) InferenceBatcher block/batch/unblock behavior.
2) WorkerPool basic in-process threaded self-play smoke path.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from engine import InferenceBatcher, SelfPlayRunner
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference_server import InferenceServer
from hexo_rl.selfplay.pool import WorkerPool
from engine import ReplayBuffer


@pytest.mark.timeout(30)
def test_rust_batcher_blocks_batches_and_unblocks():
    feature_len = 18 * 19 * 19
    policy_len = 19 * 19 + 1

    batcher = InferenceBatcher(feature_len=feature_len, policy_len=policy_len)
    try:
        batcher.spawn_mock_games(10)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not batcher.has_pending_requests():
            time.sleep(0.01)

        assert batcher.has_pending_requests(), "mock game requests did not reach Rust queue"

        all_ids: list[int] = []
        all_rows: list[np.ndarray] = []
        collect_deadline = time.monotonic() + 5.0
        while len(all_ids) < 10 and time.monotonic() < collect_deadline:
            req_ids, fused = batcher.next_inference_batch(batch_size=16, max_wait_ms=500)
            if not req_ids:
                continue
            fused_np = np.asarray(fused, dtype=np.float32)
            all_ids.extend(req_ids)
            all_rows.append(fused_np)

        assert len(all_ids) == 10
        merged = np.concatenate(all_rows, axis=0)
        assert merged.shape == (10, feature_len)

        policies = np.full((10, policy_len), 1.0 / float(policy_len), dtype=np.float32)
        values = np.zeros((10,), dtype=np.float32)
        batcher.submit_inference_results(all_ids, policies, values)

        done_deadline = time.monotonic() + 5.0
        while time.monotonic() < done_deadline and batcher.completed_mock_games() < 10:
            time.sleep(0.01)

        assert batcher.completed_mock_games() == 10
    finally:
        batcher.close()


@pytest.mark.timeout(60)
def test_worker_pool_produces_positions_threaded_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexTacToeNet(board_size=19, in_channels=18, filters=32, res_blocks=2).to(device)
    buffer = ReplayBuffer(capacity=10_000)

    config = {
        "mcts": {
            "n_simulations": 1,
            "c_puct": 1.0,
            "temperature_threshold_ply": 8,
        },
        "selfplay": {
            "n_workers": 1,
        },
    }

    pool = WorkerPool(model, config, device, buffer, n_workers=1)
    pool.start()
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if pool.games_completed >= 1 and pool.positions_pushed > 0 and buffer.size > 0:
                break
            time.sleep(0.1)

        assert pool.games_completed >= 1
        assert pool.positions_pushed > 0
        assert buffer.size > 0
    finally:
        pool.stop()


@pytest.mark.timeout(30)
def test_rust_runner_with_inference_server_generates_positions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexTacToeNet(board_size=19, in_channels=18, filters=32, res_blocks=2).to(device)

    runner = SelfPlayRunner(n_workers=2, max_moves_per_game=16, n_simulations=1, leaf_batch_size=1)
    server = InferenceServer(
        model,
        device,
        {"selfplay": {"inference_batch_size": 32, "inference_max_wait_ms": 5.0}},
        batcher=runner.batcher,
    )

    server.start()
    runner.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and runner.games_completed <= 0:
            time.sleep(0.05)

        assert runner.positions_generated > 0
        assert runner.games_completed >= 1
    finally:
        runner.stop()
        server.stop()
        server.join(timeout=5.0)


@pytest.mark.timeout(30)
def test_rust_runner_collect_data_format():
    """Verify that collect_data returns correctly shaped tensors and policies."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexTacToeNet(board_size=19, in_channels=18, filters=32, res_blocks=2).to(device)
    
    # Run with 1 worker and 1 simulation for speed
    runner = SelfPlayRunner(n_workers=1, max_moves_per_game=4, n_simulations=1, leaf_batch_size=1)
    server = InferenceServer(
        model,
        device,
        {"selfplay": {"inference_batch_size": 8, "inference_max_wait_ms": 5.0}},
        batcher=runner.batcher,
    )
    
    server.start()
    runner.start()
    try:
        # Wait for at least one game to complete
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and runner.games_completed <= 0:
            time.sleep(0.1)
            
        assert runner.games_completed >= 1
        data = runner.collect_data()
        assert len(data) > 0
        
        feat, pol, outcome, plies = data[0]
        assert len(feat) == 18 * 19 * 19
        assert len(pol) == 19 * 19 + 1
        assert isinstance(outcome, float)
        assert outcome == -1.0 or outcome == 1.0 or abs(outcome - (-0.1)) < 1e-5
        assert isinstance(plies, int)
        assert plies >= 0
        
        # Check that we can reshape and use as numpy
        feat_np = np.array(feat).reshape(18, 19, 19)
        assert feat_np.shape == (18, 19, 19)
    finally:
        runner.stop()
        server.stop()
        server.join(timeout=5.0)
