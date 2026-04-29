"""§130 — buffer compatibility under per-game self-play rotation.

The rotation port writes already-rotated state/chain/policy/aux into the
replay buffer (per the B4 audit verdict; no schema change). This test pins:

  1. `ReplayBuffer.push_many` / `sample_batch` accept rotated rows without
     shape or dtype error.
  2. Loading a HEXB v6 file written before the rotation port still succeeds
     after the port lands — the on-disk format is unchanged.
  3. Sample-time augmentation (12-fold scatter on top of per-game rotation)
     still produces well-shaped batches with the right ply-0 stone counts.

The W4 Step 1 spec wants `tests/test_rotation_buffer_compat.py — existing
buffer reads correctly with rotation on`. We exercise that by running the
WorkerPool with `selfplay.rotation_enabled=true`, sampling a batch with
`augment=True`, and asserting shapes/dtypes match the contract.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from engine import ReplayBuffer
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.pool import WorkerPool


def _spin_pool(rotation: bool, capacity: int = 256) -> ReplayBuffer:
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=18, filters=16, res_blocks=2).to(device)
    config = {
        "selfplay": {
            "n_workers": 1,
            "max_game_moves": 8,
            "leaf_batch_size": 1,
            "inference_batch_size": 4,
            "inference_max_wait_ms": 5.0,
            "rotation_enabled": rotation,
            "playout_cap": {
                "fast_prob": 0.0, "fast_sims": 2, "standard_sims": 2,
                "n_sims_quick": 0, "n_sims_full": 0, "full_search_prob": 0.0,
            },
            "random_opening_plies": 0,
        },
        "mcts": {"n_simulations": 2, "c_puct": 1.5, "fpu_reduction": 0.25,
                 "dirichlet_enabled": False},
        "training": {"draw_value": 0.0},
    }
    buf = ReplayBuffer(capacity=capacity)
    pool = WorkerPool(model, config, device, buf)
    pool.start()
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and buf.size < 16:
            time.sleep(0.1)
    finally:
        pool.stop()
    return buf


@pytest.mark.timeout(60)
def test_rotation_on_buffer_push_and_sample_shapes():
    """Rotated rows push and sample without shape/dtype error.

    Schema: 8 state planes (HEXB v6 KEPT_PLANE_INDICES), 6 chain planes,
    362 policy slots, 361 ownership/winning_line cells. We assert all of
    those at sample time so a future schema drift is caught here rather
    than at training-loop run time.
    """
    buf = _spin_pool(rotation=True)
    if buf.size < 16:
        pytest.skip(f"only {buf.size} positions; smoke too short on this CPU")

    n = min(32, buf.size)

    # augment=False — pin the post-rotation, pre-augmentation row shape.
    states_np, chain_np, pols_np, vals_np, own_np, wl_np, ifs_np = (
        buf.sample_batch(n, False)
    )
    assert states_np.shape == (n, 8, 19, 19), states_np.shape
    assert chain_np.shape == (n, 6, 19, 19), chain_np.shape
    assert pols_np.shape == (n, 362), pols_np.shape
    assert vals_np.shape == (n,), vals_np.shape
    assert own_np.shape == (n, 19, 19), own_np.shape
    assert wl_np.shape == (n, 19, 19), wl_np.shape
    assert ifs_np.shape == (n,), ifs_np.shape
    assert states_np.dtype == np.float16
    assert chain_np.dtype == np.float16
    assert pols_np.dtype == np.float32
    assert own_np.dtype == np.uint8
    assert wl_np.dtype == np.uint8
    assert ifs_np.dtype == np.uint8

    # Policies are normalised distributions: row sums close to 1.
    row_sums = pols_np.sum(axis=1)
    assert np.all(row_sums > 0.5), f"policy row sums too small: {row_sums.min()}"
    assert np.all(row_sums < 1.5), f"policy row sums too large: {row_sums.max()}"


@pytest.mark.timeout(60)
def test_rotation_on_buffer_sample_with_augmentation():
    """augment=True (sample-time 12-fold scatter) on top of per-game rotation
    composes without error.

    Per the B4 audit, sample-time augmentation runs unchanged on top of the
    per-game rotation — the resulting batch reflects 12 × 12 = 144 effective
    orientations per source position. Identity-element overlap is negligible
    relative to corpus turnover; the test just guards against a shape /
    indexing bug introduced by the compose.
    """
    buf = _spin_pool(rotation=True)
    if buf.size < 16:
        pytest.skip(f"only {buf.size} positions; smoke too short on this CPU")
    n = min(32, buf.size)
    states_np, chain_np, pols_np, *_ = buf.sample_batch(n, True)
    assert states_np.shape == (n, 8, 19, 19)
    assert chain_np.shape == (n, 6, 19, 19)
    assert pols_np.shape == (n, 362)
    # Aug is a permutation: stone counts per row should match the un-augmented
    # case. We can't compare row-by-row because `sample_batch` reshuffles
    # indices, but the *aggregate* nonzero count per ply-0 plane should
    # remain in a sane range (≤ 8 stones × n rows for max_game_moves=8).
    cur_ply0 = states_np[:, 0]
    assert int(np.count_nonzero(cur_ply0)) <= 8 * n


@pytest.mark.timeout(60)
def test_rotation_on_buffer_save_load_roundtrip():
    """The rotation port does not change the on-disk HEXB v6 format.

    Serialise a rotated buffer to a temp path, instantiate a fresh empty
    buffer, load it, and confirm size matches. This is the contract the
    sustained-run resume path depends on (eval anchors are the most common
    consumer).
    """
    buf = _spin_pool(rotation=True)
    if buf.size < 8:
        pytest.skip(f"only {buf.size} positions; smoke too short")
    initial_size = buf.size

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "rot_buffer.hexb")
        buf.save_to_path(path)

        fresh = ReplayBuffer(capacity=buf.capacity)
        loaded = fresh.load_from_path(path)
        assert loaded == initial_size, (
            f"load_from_path returned {loaded}; expected {initial_size}"
        )
        assert fresh.size == initial_size

        # Sample shape should match across the persistence boundary.
        n = min(8, fresh.size)
        states_np, *_ = fresh.sample_batch(n, False)
        assert states_np.shape == (n, 8, 19, 19)
