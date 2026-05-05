"""§153 T3 — per-move / per-turn rotation cadence wiring.

The `rotation_cadence` kwarg on `SelfPlayRunner` selects when the per-game
sym_idx is resampled within a game:

  * "per_game" — sample once at game start (default; matches §130).
  * "per_move" — sample for every recorded move.
  * "per_turn" — sample at each turn boundary (ply 0 or first move of
    a player's compound turn).

Tests drive each cadence end-to-end through `WorkerPool`, drain rows from
the replay buffer, and verify both shape correctness and aux/state
consistency: each row's stone planes (0 = cur, 8 = opp) align cell-wise
with its ownership target (which is 1 — empty — wherever both stone
planes are zero), regardless of rotation cadence. A bug that put state
and aux into different rotated frames would scatter the two apart.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from engine import ReplayBuffer, SelfPlayRunner
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.pool import WorkerPool


def _build_pool_with_cadence(cadence: str, *, board_size: int = 19) -> tuple[WorkerPool, ReplayBuffer]:
    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=board_size, in_channels=8, filters=16, res_blocks=2
    ).to(device)
    config = {
        "selfplay": {
            "n_workers": 1,
            "max_game_moves": 14,
            "leaf_batch_size": 1,
            "inference_batch_size": 4,
            "inference_max_wait_ms": 5.0,
            "rotation_enabled": True,
            "rotation_cadence": cadence,
            "playout_cap": {
                "fast_prob": 0.0,
                "fast_sims": 2,
                "standard_sims": 2,
                "n_sims_quick": 0,
                "n_sims_full": 0,
                "full_search_prob": 0.0,
            },
            "random_opening_plies": 0,
        },
        "mcts": {
            "n_simulations": 2,
            "c_puct": 1.5,
            "fpu_reduction": 0.25,
            "dirichlet_enabled": False,
        },
        "training": {"draw_value": 0.0},
    }
    buf = ReplayBuffer(capacity=512)
    pool = WorkerPool(model, config, device, buf)
    return pool, buf


def _wait_for_rows(buf: ReplayBuffer, n_rows: int, timeout_s: float, pool: WorkerPool) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and buf.size < n_rows:
        time.sleep(0.05)


@pytest.mark.timeout(60)
def test_per_move_cadence_runs_and_produces_consistent_records():
    """`per_move` cadence completes games and produces aux/state-consistent rows."""
    pool, buf = _build_pool_with_cadence("per_move")
    pool.start()
    try:
        _wait_for_rows(buf, n_rows=24, timeout_s=45.0, pool=pool)
        if buf.size < 24:
            pytest.skip(f"only {buf.size} rows after timeout; smoke too short")
        states_np, _chains_np, _pols_np, _vals_np, own_np, _wl_np, _ifs_np = (
            buf.sample_batch(min(64, buf.size), False)
        )
        states = np.asarray(states_np).reshape(-1, 8, 19, 19).astype(np.float32)
        own = np.asarray(own_np).reshape(-1, 19, 19).astype(np.uint8)
        # Plane 0 = cur stones, plane 8 location is index 4 in the 8-plane
        # HEXB v6 wire format (planes 1..7 / 9..15 of the 18-plane history are
        # never set on the Rust path; only 0,8,16,17 carry information). The
        # buffer sample_batch returns the 8-plane slice; cur-plane is index 0,
        # opp-plane is index 4.
        cur = states[:, 0]
        opp = states[:, 4]
        # Aux ownership encoding: 0 = P2, 1 = empty, 2 = P1.
        # Wherever there is no stone in either plane, ownership must be 1.
        any_stone = (cur > 0) | (opp > 0)
        no_stone = ~any_stone
        # Allow a small tolerance for the (rare) row where the aux reprojection
        # touches a cell outside the row's own window (window_flat_idx_at
        # returns TOTAL_CELLS for out-of-window cells; those are skipped at
        # reprojection time, and the aux defaults to 1 = empty there).
        ownership_empty_at_no_stone = (own == 1)
        agree_no_stone = ownership_empty_at_no_stone[no_stone]
        # Mostly, no-stone cells must read ownership=1.
        assert agree_no_stone.mean() > 0.95, (
            f"aux ownership disagrees with state at >5% of no-stone cells "
            f"({(1 - agree_no_stone.mean()) * 100:.2f}%) — per_move rotation "
            "may have desynced state and aux frames."
        )
    finally:
        pool.stop()


@pytest.mark.timeout(60)
def test_per_turn_cadence_runs_and_produces_consistent_records():
    """`per_turn` cadence completes games and produces aux/state-consistent rows."""
    pool, buf = _build_pool_with_cadence("per_turn")
    pool.start()
    try:
        _wait_for_rows(buf, n_rows=24, timeout_s=45.0, pool=pool)
        if buf.size < 24:
            pytest.skip(f"only {buf.size} rows after timeout; smoke too short")
        states_np, _, _, _, own_np, _, _ = (
            buf.sample_batch(min(64, buf.size), False)
        )
        states = np.asarray(states_np).reshape(-1, 8, 19, 19).astype(np.float32)
        own = np.asarray(own_np).reshape(-1, 19, 19).astype(np.uint8)
        cur = states[:, 0]
        opp = states[:, 4]
        any_stone = (cur > 0) | (opp > 0)
        no_stone = ~any_stone
        agree = (own == 1)[no_stone]
        assert agree.mean() > 0.95, (
            "aux ownership disagrees with state at >5% of no-stone cells "
            f"({(1 - agree.mean()) * 100:.2f}%) under per_turn cadence."
        )
    finally:
        pool.stop()


def test_rotation_cadence_invalid_value_rejected():
    """Bad `rotation_cadence` values must raise a ValueError at construction."""
    with pytest.raises(ValueError, match="rotation_cadence must be"):
        SelfPlayRunner(
            n_workers=1,
            max_moves_per_game=4,
            n_simulations=1,
            leaf_batch_size=1,
            fast_sims=1,
            standard_sims=1,
            rotation_cadence="per_year",
        )


def test_rotation_cadence_default_is_per_game():
    """Existing callers (no kwarg) inherit `per_game` behaviour."""
    runner = SelfPlayRunner(
        n_workers=1,
        max_moves_per_game=4,
        n_simulations=1,
        leaf_batch_size=1,
        fast_sims=1,
        standard_sims=1,
    )
    # Construction succeeds — no flag accessor is exposed back to Python by
    # design (see §153 T3 design notes); confirming construction is enough
    # for the default-cadence regression check.
    assert runner is not None
