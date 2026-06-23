"""§173 A8' — v6w25 multi-window selfplay micro-smoke.

After lifting the α multi-window guard at `WorkerPool.__init__`, this
test exercises the full Python→Rust selfplay loop on v6w25:

  - `WorkerPool` constructs (no NotImplementedError).
  - Workers spin up, run MCTS, push buffer rows.
  - `ReplayBuffer.size > 0`; `pool.positions_pushed > 0`.
  - Pushed rows have the v6w25 shape (8 planes × 25 × 25 = 5000;
    policy = 626; chain = 6 × 625 = 3750). No NaN / Inf in any sampled
    row.

Test uses a randomly-initialised 25×25×8 net so the suite runs without
needing the `bootstrap_model_v6w25.pt` artifact. Pool runs with
n_workers=2, n_simulations=4, max_moves=12 → completes well under 30s.

Discriminator: ANY panic / NaN / shape mismatch surfaces α gaps that
the guard previously hid.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pytest
import torch

from engine import ReplayBuffer  # type: ignore[attr-defined]
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.pool import WorkerPool


def _v6w25_microsmoke_cfg() -> Dict[str, Any]:
    """Minimal cfg sized for fast smoke — 2 workers, 4 sims/move, 12-move cap."""
    sp = {
        "n_workers": 2,
        "max_game_moves": 12,
        "leaf_batch_size": 1,
        "inference_batch_size": 4,
        "inference_max_wait_ms": 5.0,
        "rotation_enabled": False,
        "playout_cap": {
            "fast_prob": 0.0,
            "fast_sims": 4,
            "standard_sims": 4,
            "n_sims_quick": 0,
            "n_sims_full": 0,
            "full_search_prob": 0.0,
        },
        "random_opening_plies": 0,
        "trace_inference": False,
        "results_queue_cap": 1024,
    }
    return {
        "selfplay": sp,
        "mcts": {
            "interior_selector": "puct",
            "n_simulations": 4,
            "c_puct": 1.5,
            "fpu_reduction": 0.25,
            "dirichlet_enabled": False,
        },
        "training": {"draw_value": 0.0},
        "encoding": {"version": "v6w25"},
        "game_replay": {"enabled": False},
    }


@pytest.mark.timeout(60)
def test_v6w25_pool_smoke_no_nan_correct_shapes():
    """§173 A8'' closed the two Rust α gaps surfaced by A8':
      (1) records.rs::reproject_game_end_row now takes a spec-derived
          n_cells param (replaces TOTAL_CELLS=361 hardcode); aux buffer
          sized to 2 × n_cells matches collect_data's reshape.
      (2) board/state.rs::window_flat_idx now dispatches via
          Board::cluster_window_size (= spec.trunk_size) so the policy
          index map honours the v6w25 25×25 frame, not the v6 19×19.
    The xfail marker is gone — this test now asserts end-to-end v6w25
    multi-window selfplay (Python pool + Rust worker_loop + ReplayBuffer).

    Constructs + runs a v6w25 WorkerPool for a handful of games.
    """
    device = torch.device("cpu")
    # Random-init net — model + buffer shape parity is what's under test;
    # quality of play is out of scope.
    model = HexTacToeNet(
        board_size=25, in_channels=8, filters=8, res_blocks=1,
        encoding="v6w25",
    ).to(device)
    # §173 A8'': pool pushes v6w25-shaped rows (8 × 25 × 25); buffer must
    # be initialised with the matching encoding so state_stride / aux_stride
    # align. Default `ReplayBuffer(capacity)` keeps v6 (8 × 19 × 19) which
    # would loud-fail at push_many.
    buf = ReplayBuffer(capacity=2048, encoding="v6w25")
    cfg = _v6w25_microsmoke_cfg()

    pool = WorkerPool(model, cfg, device, buf, n_workers=2)

    # Geometry sanity (pre-start) — confirms §173 A8' refactor wired
    # trunk_size through correctly.
    assert pool._board_size == 25
    assert pool._trunk_size == 25
    assert pool._feat_len == 8 * 25 * 25       # n_kept_planes * trunk_size²
    assert pool._pol_len == 626                # policy_logit_count
    assert pool._chain_len == 6 * 25 * 25

    pool.start()
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if pool.games_completed >= 1 and pool.positions_pushed > 0 and buf.size > 0:
                break
            time.sleep(0.1)

        # Hard asserts — α path must produce real rows, not silent zeros.
        assert pool.games_completed >= 1, (
            f"no games completed in 15 s; multi-window dispatch likely "
            f"stalled (positions_pushed={pool.positions_pushed}, "
            f"buf.size={buf.size})"
        )
        assert pool.positions_pushed > 0
        assert buf.size > 0

        # Sample rows from the replay buffer; verify shape + NaN-free.
        bs = min(8, buf.size)
        states, chain_planes, policies, outcomes, ownership, wl, ifs, _vv = (
            buf.sample_batch(batch_size=bs, augment=False)
        )

        # v6w25 wire format: 8 planes × 25 × 25 features; 626 policy logits.
        assert states.shape == (bs, 8, 25, 25), (
            f"v6w25 sampled states shape {states.shape} != expected "
            f"({bs}, 8, 25, 25); buffer-push geometry diverged from trunk_size."
        )
        assert chain_planes.shape == (bs, 6, 25, 25), (
            f"chain_planes shape {chain_planes.shape} != ({bs}, 6, 25, 25)"
        )
        assert policies.shape == (bs, 626), (
            f"policies shape {policies.shape} != ({bs}, 626)"
        )
        assert outcomes.shape == (bs,)

        # No NaN / Inf in any sampled column — α path must emit real values.
        assert np.isfinite(states.astype(np.float32)).all(), "NaN/Inf in features"
        assert np.isfinite(chain_planes.astype(np.float32)).all(), "NaN/Inf in chain"
        assert np.isfinite(policies.astype(np.float32)).all(), "NaN/Inf in policy"
        assert np.isfinite(outcomes.astype(np.float32)).all(), "NaN/Inf in value"
    finally:
        pool.stop()
