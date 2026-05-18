"""INV26 — `ply_cap_value` distinct outcome path (§178, T10).

Pins the new `ply_cap_value` split introduced in T2 (commit `b84006d`).
The Rust `finalize_game` outcome branch now reads:

    None => if terminal_reason == 2 { ply_cap_value } else { draw_reward },

Pre-T2 the branch unconditionally paid `draw_reward`. The Python side
exercises the live PyO3 wire (`SelfPlayRunner(SelfPlayRunnerConfig(...))`)
with `InferenceServer` so MCTS runs and rows reach the results queue.

Cells:
  1. Ply-cap path + split values (`draw_reward=-0.1, ply_cap_value=-0.5`)
     → every drained `vals_np` row is exactly `-0.5` (within f32 epsilon).
  2. Organic-draw path (winner=None, ply<max, legal_move_count==0) is
     unreachable from natural play on a 19×19 board in <180 plies; the
     branch is value-equivalent to the pre-T2 path by inspection of
     `finalize_game` source — covered by code review + Cell 4 back-compat.
  3. Winner=Some path pays ±1.0 — skipped from this isolated test because
     forcing a 6-in-a-row under random+MCTS-1 on 19×19 in <30 plies is
     non-deterministic. Covered by other test suites end-to-end.
  4. Back-compat default — when `ply_cap_value == draw_reward`, ply-cap
     rows pay that single value (matches pre-T2 byte-for-byte).

Pre-registered invariant (design §11 line 3): INV26 must FAIL fast if
`ply_cap_value` silently equals `draw_reward` when YAML specifies a
distinct value. Cell 1 establishes the positive path; Cell 4 establishes
back-compat parity.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from engine import SelfPlayRunner, SelfPlayRunnerConfig
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference_server import InferenceServer


def _build_runner(draw_reward: float, ply_cap_value: float) -> SelfPlayRunner:
    """Tiny 8-ply game so ply-cap fires fast under MCTS-1."""
    return SelfPlayRunner(SelfPlayRunnerConfig(
        n_workers=2,
        max_moves_per_game=8,
        n_simulations=1,
        leaf_batch_size=1,
        c_puct=1.5,
        fpu_reduction=0.0,
        fast_prob=0.0,
        fast_sims=1,
        standard_sims=1,
        draw_reward=draw_reward,
        ply_cap_value=ply_cap_value,
        encoding_name="v6",
    ))


def _drive_one_round(runner: SelfPlayRunner, model, device) -> tuple[np.ndarray, list[int]]:
    """Spin up InferenceServer, run until at least one game completes, return
    (vals_np, terminal_reasons)."""
    server = InferenceServer(
        model,
        device,
        {"selfplay": {"inference_batch_size": 32, "inference_max_wait_ms": 5.0}},
        batcher=runner.batcher,
    )
    server.start()
    runner.start()
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and runner.games_completed <= 0:
            time.sleep(0.05)
        assert runner.games_completed >= 1, "at least one game must complete in 20s"
        # Let a few more games complete for sample size.
        more_deadline = time.monotonic() + 2.0
        while time.monotonic() < more_deadline:
            time.sleep(0.05)
    finally:
        runner.stop()
        server.stop()
        server.join(timeout=5.0)

    drained = runner.drain_game_results()
    terminal_reasons = [row[4] for row in drained]
    feats_np, chain_np, pols_np, vals_np, plies_np, own_np, wl_np, ifs_np = (
        runner.collect_data()
    )
    return vals_np, terminal_reasons


@pytest.mark.timeout(60)
def test_inv26_cell_1_ply_cap_value_distinct_writes_minus_0_5():
    """Cell 1: draw_reward=-0.1 + ply_cap_value=-0.5 → every row pays -0.5."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=32, res_blocks=2).to(device)
    runner = _build_runner(draw_reward=-0.1, ply_cap_value=-0.5)
    vals_np, terminal_reasons = _drive_one_round(runner, model, device)

    # Every completed game in a 19×19 board with max_moves=8 + MCTS-1 reaches
    # ply-cap (no 6-in-a-row achievable in 8 plies; no legal-move exhaustion).
    assert terminal_reasons, "must observe at least one terminal_reason"
    assert all(r == 2 for r in terminal_reasons), (
        f"INV26 Cell 1: all terminal_reasons must be 2 (ply-cap); "
        f"got {terminal_reasons}"
    )
    # Per-row outcome equality is the load-bearing assertion. With
    # ply_cap_value=-0.5 every row from a ply-cap game pays -0.5.
    assert vals_np.size > 0, "at least one row must be recorded (max_moves=8)"
    assert np.allclose(vals_np, -0.5, atol=1e-6), (
        f"INV26 Cell 1: every row must have outcome==-0.5 (ply_cap_value); "
        f"got unique={np.unique(vals_np)} (sample {vals_np[:5]})"
    )


@pytest.mark.timeout(60)
def test_inv26_cell_4_back_compat_default_writes_minus_0_1():
    """Cell 4: draw_reward=-0.1 == ply_cap_value=-0.1 → behavior matches pre-T2.

    Catches the failure mode 'design implements knob but plumbing reverts
    default at top-level wiring' — when both values equal -0.1, ply-cap rows
    pay -0.1 byte-for-byte.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=32, res_blocks=2).to(device)
    runner = _build_runner(draw_reward=-0.1, ply_cap_value=-0.1)
    vals_np, terminal_reasons = _drive_one_round(runner, model, device)

    assert terminal_reasons, "must observe at least one terminal_reason"
    assert all(r == 2 for r in terminal_reasons), (
        f"INV26 Cell 4: all terminal_reasons must be 2 (ply-cap); "
        f"got {terminal_reasons}"
    )
    assert vals_np.size > 0, "at least one row must be recorded"
    assert np.allclose(vals_np, -0.1, atol=1e-6), (
        f"INV26 Cell 4 back-compat: every row must pay -0.1 (both split values); "
        f"got unique={np.unique(vals_np)}"
    )


@pytest.mark.timeout(30)
def test_inv26_pyo3_kwarg_accepts_ply_cap_value():
    """Pin: `SelfPlayRunnerConfig(...)` PyO3 kwarg surface accepts the new
    `ply_cap_value` field. INV19 Test 1 (Rust side) already pins the default
    `-0.1`; this Python-side cell pins the kwarg name is exposed through PyO3.

    A regression that drops the kwarg from the `#[pyo3(signature=...)]` block
    at `engine/src/game_runner/config.rs` would surface as a TypeError here.
    """
    # Explicit value path — kwarg name must be accepted.
    cfg = SelfPlayRunnerConfig(encoding_name="v6", ply_cap_value=-0.42)
    assert cfg is not None
    # Default-omitted path — must still construct (back-compat default = -0.1
    # per Rust INV19 Test 1).
    cfg_default = SelfPlayRunnerConfig(encoding_name="v6")
    assert cfg_default is not None
