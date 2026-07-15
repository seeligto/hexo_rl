"""GNN-integration WP-5b commit A — live worker-loop graph-recording dispatch,
end-to-end integration test (delta doc §10 test-plan rows: "Record dispatch
(R4) + legal_set force (R1)", "Drain (R7) -> push (P5)", "End-to-end
(part-3 gate)").

Drives the REAL ``SelfPlayRunner`` self-play loop on a ``representation=
"graph"`` spec end-to-end — the thing WP-5a's report named as NOT wired:

    SelfPlayRunner(encoding_name="gnn_axis_v1").start()
      -> worker_loop is_graph()-gated dispatch (R1 legal_set force,
         R4 record_position_graph_dispatch, R5 finalize_game_graph)
      -> graph_results queue
      -> collect_graph_data() (R7) -> HexgBuffer.push_graph_position (P5)
      -> save_to_path / load_from_path round trip -> get_buffer_stats
      -> sample_graph_batch -> collate_graph_batch -> GnnNet.forward_batch
         (finite losses)

If R1 (the `legal_set` force) were NOT wired, `record_position_graph_dispatch`
would hit its `MovePolicy::Dense` `unreachable!()` arm on the very first
recorded move and the whole worker thread would panic (the crate's release
profile is `panic="abort"`) — so a run that completes normally with non-empty
drained rows is itself a structural proof that R1+R4 are wired, not just that
the pure `record_position_graph`/`finalize_graph_outcome` fns work (WP-5a
already proved those in isolation).

FORMERLY (WP-5b commit A): `hexo_graph::legal_moves_from_stones` (WP-1's
native builder) had no empty-board fallback — an actual game-0 (0-stone)
position yielded ZERO legal graph nodes, so an MCTS search rooted at a fresh
board raised `EmptyLegalSet` inside `collate_graph_batch` on the very first
leaf-batch and the game could never progress (`RootExpansionFailed` on every
attempt, game finalized at `ply==0`). This test originally routed around the
gap with `random_opening_plies=2` (Board-level opening moves that bypass
MCTS/graph-inference entirely, seeding stones before the first graph search).

FIXED (WP-1 launch-blocker fix): `legal_moves_from_stones` now special-cases
`n_stones == 0` with a fixed 5x5 region mirroring the dense engine's own
empty-board rule EXACTLY (`Board::legal_moves_set()`, `engine/src/board/
moves.rs:95-101`) — see that fn's Rust doc comment for the full rationale.
This test is now PARAMETRIZED over `random_opening_plies in {0, 2}`: `0`
proves organic graph self-play starts a game from an actual empty board
(the regression pin for this fix — EmptyLegalSet must not fire at ply 0);
`2` is retained as the original workaround-path coverage (both must record,
drain, and train-step-finite identically).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

import engine
from engine import HexgBuffer
from hexo_rl.encoding import lookup as registry_lookup
from hexo_rl.model.gnn_net import GnnNet, load_representation_policy_from_bc
from hexo_rl.selfplay.graph_collate import collate_graph_batch, stone_mask_from_batch
from hexo_rl.selfplay.inference_server import InferenceServer
from hexo_rl.training.binned_value import binned_value_loss
from hexo_rl.training.losses import ragged_policy_ce

pytestmark = pytest.mark.integration

_BC_CKPT = "checkpoints/probes/gnn_bc/gnn_bc_040000.pt"
ENC = "gnn_axis_v1"
_TRUNK = 19


def _trunc_div2(x: int) -> int:
    """Integer /2 truncated toward zero — matches Rust `(a+b)/2`, NOT Python //."""
    return -((-x) // 2) if x < 0 else x // 2


def _window_center(stones):
    """Bbox-midpoint window centre — the SAME convention `Board::window_center`
    / `record_position_graph` uses (mirrors `test_gnn_seam_smoke.py`)."""
    qs = [q for q, _r, _p in stones]
    rs = [r for _q, r, _p in stones]
    return _trunc_div2(min(qs) + max(qs)), _trunc_div2(min(rs) + max(rs))


def _window_slot(q: int, r: int, cq: int, cr: int, trunk: int = _TRUNK) -> int:
    half = (trunk - 1) // 2
    wq, wr = q - cq + half, r - cr + half
    if 0 <= wq < trunk and 0 <= wr < trunk:
        return wq * trunk + wr
    return -1


def _load_net(device: torch.device) -> GnnNet:
    ckpt = torch.load(_BC_CKPT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    net = GnnNet()
    load_representation_policy_from_bc(net, state)
    return net.to(device).eval()


def _run_selfplay_to_n_games(
    net: GnnNet, device: torch.device, *, n_games: int, timeout_s: float, random_opening_plies: int
):
    """Drives real self-play (`.start()`/`.stop()`) until `n_games` complete
    (or `timeout_s` elapses), then stops and returns
    `(games_batch, graph_rows)` — both drained ONCE, after `stop()`, so the
    two totals are mutually consistent (no in-flight race)."""
    spec = registry_lookup(ENC)
    runner = engine.SelfPlayRunner(
        engine.SelfPlayRunnerConfig(
            n_workers=3,
            encoding_name=ENC,
            n_simulations=6,
            leaf_batch_size=4,
            standard_sims=6,
            max_moves_per_game=24,
            # random_opening_plies=0 drives the game to start ORGANICALLY
            # from an actual empty board — the WP-1 empty-board fix's
            # regression pin (module docstring). random_opening_plies=2
            # (Board-level opening moves that bypass MCTS/graph-inference)
            # is retained as the original workaround-path coverage.
            random_opening_plies=random_opening_plies,
            selfplay_rotation_enabled=False,
        )
    )
    server = InferenceServer(
        net, device,
        {"selfplay": {"inference_batch_size": 8, "inference_max_wait_ms": 10}},
        batcher=runner.batcher, encoding_spec=spec,
    )
    t0 = time.monotonic()
    runner.start()
    server.start()
    try:
        while runner.games_completed < n_games and time.monotonic() - t0 < timeout_s:
            time.sleep(0.1)
    finally:
        runner.stop()
        server.stop()
    assert runner.games_completed >= n_games, (
        f"self-play did not complete {n_games} games within {timeout_s}s "
        f"(got {runner.games_completed}) — see module docstring; a regression "
        f"to 0 at random_opening_plies=0 means the WP-1 empty-board fix broke"
    )
    games_batch = runner.drain_game_results()
    graph_rows = runner.collect_graph_data()
    return games_batch, graph_rows


@pytest.mark.skipif(not os.path.exists(_BC_CKPT), reason="banked GNN-BC checkpoint absent")
@pytest.mark.parametrize("random_opening_plies", [0, 2], ids=["organic-ply0", "workaround-ply2"])
def test_graph_selfplay_records_drain_and_train_step_is_finite(tmp_path: Path, random_opening_plies: int):
    device = torch.device("cpu")
    net = _load_net(device)

    games_batch, rows = _run_selfplay_to_n_games(
        net, device, n_games=3, timeout_s=90.0, random_opening_plies=random_opening_plies
    )

    # ── R7 drain correctness: row count == positions actually recorded ──
    # `random_opening_plies` moves per game are Board-level (no MCTS/record);
    # every OTHER move records exactly one GraphRecord (R4), so the total
    # drained row count must equal sum(plies - random_opening_plies) across
    # every COMPLETED game this run drained. At random_opening_plies=0 this
    # also proves the game started ORGANICALLY from an empty board (WP-1 fix
    # regression pin) — no workaround stones were seeded before ply 1's real
    # graph-MCTS root expansion.
    expected_rows = sum(max(0, plies - random_opening_plies) for (plies, *_rest) in games_batch)
    assert len(rows) == expected_rows, (
        f"collect_graph_data() drained {len(rows)} rows, expected {expected_rows} "
        f"from {len(games_batch)} completed games (plies={[g[0] for g in games_batch]})"
    )
    assert len(rows) > 0

    # ── ADV-C (WP5b_commitA_redteam.md #1) live-drain-seam finiteness canary:
    # every drained row's `outcome` field must be finite — cheap regression
    # pin on the new finalize->drain seam (`finalize_graph_outcome` stamping
    # inside `finalize_game_graph`), near-free given the e2e already computes
    # losses below. ──
    for _stones, _visits, _cp, _mr, _ply, _ifs, outcome, _vv, _gl in rows:
        assert np.isfinite(outcome), f"drained row outcome must be finite, got {outcome}"

    # ── R1 legal_set force + R4 record dispatch: off-window mass reachable ──
    # (records.rs:62's off-window skip must NOT be inherited on the LIVE path
    # — WP-5a proved the pure fn; this proves the wiring threads a REAL
    # multi-window-capable LegalSetPolicy, not a degenerate/empty one).
    total_off_window = 0
    for stones, visits, _cp, _mr, _ply, _ifs, _outcome, _vv, _gl in rows:
        if not stones:
            continue
        cq, cr = _window_center(stones)
        for q, r, _p in visits:
            if _window_slot(q, r, cq, cr) == -1:
                total_off_window += 1
    assert total_off_window > 0, (
        "no off-window visited cell across the whole self-play run — either "
        "the boards stayed compact (unlucky) or legal_set was not forced "
        "(R1) and the recorded target degenerated to the dense in-window-only "
        "view; re-run or widen the smoke if this flakes"
    )

    # ── P5 / Adjudication 3: every drained row round-trips push_graph_position
    # without tripping the Rust validate_visit_prob guard (finite, non-negative
    # visit mass by construction — the record fn's p>0.0 filter suffices). ──
    buf = HexgBuffer(max(len(rows) * 2, 64), ENC)
    for rec in rows:
        buf.push_graph_position(*rec, game_id=-1)  # game_id=-1: untagged (§4.2)
    assert buf.size == len(rows)

    # ── Buffer persist round trip (part-3 gate) ──
    persist_path = tmp_path / "graph_run.hexg"
    buf.save_to_path(str(persist_path))
    buf2 = HexgBuffer(max(len(rows) * 2, 64), ENC)
    n_loaded = buf2.load_from_path(str(persist_path))
    assert n_loaded == len(rows)
    size, capacity, hist = buf2.get_buffer_stats()
    assert size == len(rows)
    assert capacity == buf2.capacity
    assert sum(hist) == size, "weight-bucket histogram must sum to buffer size"

    # ── sample -> collate -> forward -> losses finite (part-3 gate) ──
    batch_size = min(16, buf2.size)
    wire, tg = buf2.sample_graph_batch(batch_size, augment=True)
    assert int(wire.builder_impl) == 1, "F7: sample must rebuild via the native builder"
    batch = collate_graph_batch(
        wire, expected_version=1, device="cpu", semantic="full",
        target_argmax_cells=tg.target_argmax_cells,
    )
    assert batch.n_graphs == batch_size

    stone_mask = stone_mask_from_batch(batch)
    policy_logits, value, bin_logits = net.forward_batch(
        batch.x, batch.edge_index, batch.edge_attr, batch.legal_mask,
        stone_mask, batch.node_offsets,
    )
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(bin_logits).all()

    policy_target = torch.tensor(np.asarray(tg.policy_target), dtype=torch.float32)
    outcomes = torch.tensor(np.asarray(tg.outcomes), dtype=torch.float32)
    value_mask = torch.tensor(np.asarray(tg.value_valid), dtype=torch.uint8)
    ifs = torch.tensor(np.asarray(tg.is_full_search), dtype=torch.uint8)

    # ── ADV-C: every sampled tg.outcomes entry finite (post write-seam guard,
    # post-sample-rebuild — the same field the ADV-A push guard now protects). ──
    assert torch.isfinite(outcomes).all(), "sampled tg.outcomes must all be finite"

    l_pol = ragged_policy_ce(policy_logits, policy_target, batch.legal_offsets, ifs)
    l_val = binned_value_loss(bin_logits, outcomes, value_mask=value_mask)
    loss = l_pol + l_val
    assert torch.isfinite(loss), f"loss must be finite (pol={l_pol}, val={l_val})"
