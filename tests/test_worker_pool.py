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
    feature_len = 8 * 19 * 19
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
    model = HexTacToeNet(board_size=19, in_channels=8, filters=32, res_blocks=2).to(device)
    buffer = ReplayBuffer(capacity=10_000)

    config = {
        "mcts": {
            "n_simulations": 1,
            "c_puct": 1.0,
            "temperature_threshold_ply": 8,
        },
        "selfplay": {
            "n_workers": 1,
            "playout_cap": {
                "fast_sims": 1,
                "fast_prob": 0.0,
                "standard_sims": 1,
            },
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
    model = HexTacToeNet(board_size=19, in_channels=8, filters=32, res_blocks=2).to(device)

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
    model = HexTacToeNet(board_size=19, in_channels=8, filters=32, res_blocks=2).to(device)
    
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

        # drain_game_results returns metadata 8-tuples (Phase B' instrumentation:
        # adds terminal_reason + model_version_min/max/distinct). Spatial aux
        # flows per-row via collect_data() instead.
        drained = runner.drain_game_results()
        assert len(drained) > 0
        (
            plies_drain, winner_code, move_history, worker_id,
            terminal_reason, mv_min, mv_max, mv_distinct,
        ) = drained[0]
        assert isinstance(worker_id, int)
        assert worker_id == 0
        assert isinstance(terminal_reason, int) and 0 <= terminal_reason <= 3
        assert mv_max >= mv_min
        assert mv_distinct >= 1 if plies_drain > 0 else mv_distinct >= 0

        # collect_data returns 8 numpy arrays:
        # (feats, chain, pols, vals, plies, own, wl, is_full_search)
        feats_np, chain_np, pols_np, vals_np, plies_np, own_np, wl_np, ifs_np = runner.collect_data()
        assert isinstance(feats_np, np.ndarray)
        assert isinstance(chain_np, np.ndarray)
        assert isinstance(pols_np, np.ndarray)
        assert isinstance(vals_np, np.ndarray)
        assert isinstance(plies_np, np.ndarray)
        assert isinstance(own_np, np.ndarray)
        assert isinstance(wl_np, np.ndarray)
        assert isinstance(ifs_np, np.ndarray)
        assert ifs_np.dtype == np.uint8
        n = len(vals_np)
        assert n > 0
        assert feats_np.shape == (n, 8 * 19 * 19)
        assert chain_np.shape == (n, 6 * 19 * 19)
        assert pols_np.shape == (n, 19 * 19 + 1)
        assert vals_np.shape == (n,)
        assert plies_np.shape == (n,)
        assert own_np.shape == (n, 19 * 19)
        assert wl_np.shape == (n, 19 * 19)
        assert ifs_np.shape == (n,)
        assert feats_np.dtype == np.float32
        assert chain_np.dtype == np.float32
        assert pols_np.dtype == np.float32
        assert vals_np.dtype == np.float32
        assert plies_np.dtype == np.uint64
        assert own_np.dtype == np.uint8
        assert wl_np.dtype == np.uint8
        outcome = float(vals_np[0])
        assert outcome == -1.0 or outcome == 1.0 or abs(outcome - (-0.1)) < 1e-5
        # Verify features can be reshaped to (8, 19, 19) per position — HEXB v6
        feat_2d = feats_np[0].reshape(8, 19, 19)
        assert feat_2d.shape == (8, 19, 19)
    finally:
        runner.stop()
        server.stop()
        server.join(timeout=5.0)


# ── F-004: fast_prob + full_search_prob mutex ──────────────────────────────────

def test_pool_init_rejects_both_playout_caps():
    """WorkerPool.__init__ must raise ValueError when both fast_prob and
    full_search_prob are non-zero (§100 mutex — F-004).

    This guards against refactors that drop the mutex check at pool init,
    which would silently allow both caps to coexist — the move-level cap
    (full_search_prob) would override the game-level cap (fast_prob) with
    no warning, making training harder to reproduce.
    """
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=32, res_blocks=2).to(device)
    buffer = ReplayBuffer(capacity=100)

    bad_cfg = {
        "selfplay": {
            "playout_cap": {
                "fast_sims": 50,           # required to pass the earlier guard
                "fast_prob": 0.25,
                "full_search_prob": 0.25,
                "n_sims_quick": 100,
                "n_sims_full": 600,
            },
        },
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        WorkerPool(model, bad_cfg, device, buffer, n_workers=1)
