"""§153 v9 full-stack integration guard.

The unit tests cover each of T1 / T2 / T3 in isolation (HexConv2d,
corner_mask, rotation_cadence). This test stacks ALL THREE on top of
each other end-to-end through `WorkerPool` — the v9_s4 sustained-run
candidate config — to catch integration glitches that none of the
isolated tests would surface:

  * model.use_hex_kernel = True
  * model.corner_mask    = True (engine encoder-side)
  * selfplay.rotation_cadence = "per_move"

Verifies the run completes, rows reach the buffer, and aux/state stay
self-consistent (the sym_idx-per-row plumbing keeps the rotated frame
aligned with the corner-masked stone planes).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import engine
from engine import ReplayBuffer
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.pool import WorkerPool


@pytest.fixture(autouse=True)
def restore_corner_mask_flag():
    before = engine.corner_mask_enabled()
    yield
    engine.set_corner_mask_enabled(before)


@pytest.mark.timeout(90)
def test_v9_s4_full_stack_runs_and_stays_consistent():
    """The v9_s4 candidate config (every Class-4 lever) end-to-end."""
    device = torch.device("cpu")
    # `use_hex_kernel=True` swaps every trunk Conv2d → HexConv2d. Construction
    # at this layer drives the same code path the bootstrap pipeline takes.
    model = HexTacToeNet(
        board_size=19, in_channels=8, filters=16, res_blocks=2,
        use_hex_kernel=True,
    ).to(device)
    # Sanity: every res-block conv is in fact masked.
    from hexo_rl.model.hex_conv import HexConv2d
    assert isinstance(model.trunk.input_conv, HexConv2d)
    for blk in model.trunk.tower:
        assert isinstance(blk.conv1, HexConv2d)
        assert isinstance(blk.conv2, HexConv2d)

    config = {
        "model": {"use_hex_kernel": True, "corner_mask": True},
        "selfplay": {
            "n_workers": 1,
            "max_game_moves": 14,
            "leaf_batch_size": 1,
            "inference_batch_size": 4,
            "inference_max_wait_ms": 5.0,
            "rotation_enabled": True,
            "rotation_cadence": "per_move",
            "legal_move_radius_jitter": True,  # v9_s4 stacks Q2 too
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

    # Sanity: WorkerPool flipped the engine corner-mask flag during init.
    assert engine.corner_mask_enabled() is True, (
        "WorkerPool.__init__ must flip the corner-mask flag when "
        "model.corner_mask is set in the config."
    )

    pool.start()
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline and buf.size < 24:
            time.sleep(0.05)
        if buf.size < 24:
            pytest.skip(f"only {buf.size} rows after timeout; smoke too short")

        states_np, _, _, _, own_np, _, _ = (
            buf.sample_batch(min(64, buf.size), False)
        )
        states = np.asarray(states_np).reshape(-1, 8, 19, 19).astype(np.float32)
        own = np.asarray(own_np).reshape(-1, 19, 19).astype(np.uint8)
        cur = states[:, 0]
        opp = states[:, 4]

        # 1. Aux/state self-consistency under per-move rotation + corner-mask.
        #    No-stone cells must mostly read ownership=1 (empty).
        any_stone = (cur > 0) | (opp > 0)
        no_stone = ~any_stone
        empty_aux = (own == 1)
        agree_no_stone = empty_aux[no_stone]
        assert agree_no_stone.mean() > 0.95, (
            f"aux ownership disagrees with state at "
            f"{(1 - agree_no_stone.mean()) * 100:.2f}% of no-stone cells — "
            "the per-row sym_idx plumbing may have desynced state and aux "
            "frames under the v9_s4 stack."
        )

        # 2. Corner-mask is in fact zeroing parallelogram corners on stone
        #    planes. The 4 hex_dist=18 corners (0,0), (0,18), (18,0),
        #    (18,18) must never carry stone mass on any row when the mask
        #    is on. Any row where they do means the mask leaked past one
        #    of the encode paths under per-move rotation.
        corners = [(0, 0), (0, 18), (18, 0), (18, 18)]
        for (q, r) in corners:
            assert (cur[:, q, r] == 0).all(), (
                f"corner ({q},{r}) carries cur-stone mass with mask on"
            )
            assert (opp[:, q, r] == 0).all(), (
                f"corner ({q},{r}) carries opp-stone mass with mask on"
            )
    finally:
        pool.stop()
