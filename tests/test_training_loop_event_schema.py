"""Regression tests for structlog event cadence in the training loop.

Q27 smoke 2026-04-19 observed 6050 structlog ``train_step`` entries for
5500 real training steps — 550 spurious duplicates at log_interval
cadence. Two call sites emitted the same event name: trainer (per step)
and loop._emit_training_events (per log_interval). The loop-side call was
removed; these tests pin the invariant.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import structlog.testing

from engine import ReplayBuffer
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer


FAST_CONFIG = {
    "board_size":           19,
    "res_blocks":           2,
    "filters":              32,
    "batch_size":           8,
    "lr":                   2e-3,
    "weight_decay":         1e-4,
    "checkpoint_interval":  1,
    "log_interval":         1,
    "torch_compile":        False,
}


def _fill_buffer(size: int = 32) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=200)
    rng = np.random.default_rng(0)
    own = np.ones(361, dtype=np.uint8)
    wl  = np.zeros(361, dtype=np.uint8)
    chain = np.zeros((6, 19, 19), dtype=np.float16)
    for _ in range(size):
        state   = rng.random((8, 19, 19), dtype=np.float32).astype(np.float16)
        policy  = rng.dirichlet(np.ones(362)).astype(np.float32)
        outcome = float(rng.choice([-1.0, 0.0, 1.0]))
        buf.push(state, chain, policy, outcome, own, wl)
    return buf


def test_no_duplicate_train_step_at_checkpoint(tmp_path: Path) -> None:
    """Two trainer steps with checkpoint_interval=1 must emit exactly two
    structlog ``train_step`` events. Q27 smoke showed 550 duplicates at
    log_interval cadence across a 5500-step run."""
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path)
    buf = _fill_buffer()

    with structlog.testing.capture_logs() as captured:
        trainer.train_step(buf, augment=False)
        trainer.train_step(buf, augment=False)

    train_step_events = [e for e in captured if e.get("event") == "train_step"]
    assert len(train_step_events) == 2, (
        f"expected 2 train_step events for 2 trainer steps, got "
        f"{len(train_step_events)}: "
        f"{[e.get('step') for e in train_step_events]}"
    )


def test_loop_does_not_duplicate_train_step_log() -> None:
    """Static source guard: neither ``hexo_rl/training/loop.py`` nor
    ``hexo_rl/training/step_coordinator.py`` may call
    ``log.info('train_step', ...)``. The authoritative cadence is
    trainer.train_step_from_tensors (one emission per real step). A
    log_interval-gated duplicate in loop.py was the Q27 smoke root cause;
    §159a moved the per-step body into step_coordinator.py so the guard
    must scan both files.
    """
    pattern = re.compile(r"(?:log|self\._logger)\.info\(\s*[\"']train_step[\"']")
    base = Path(__file__).resolve().parents[1] / "hexo_rl" / "training"
    for fname in ("loop.py", "step_coordinator.py"):
        src = (base / fname).read_text()
        matches = pattern.findall(src)
        assert not matches, (
            f"hexo_rl/training/{fname} has {len(matches)} "
            "log.info('train_step', ...) call(s); this duplicates "
            "trainer.train_step_from_tensors's per-step emission and "
            "caused 6050 events for 5500 steps in Q27 smoke 2026-04-19."
        )


# ── WP-5b commit B (P1/§10) — graph loss_info contract ──────────────────────
#
# Hard-required keys: `emit_training_events` (events.py:235-237,339-341)
# direct-indexes loss_info["loss"]/["policy_loss"]/["value_loss"];
# step_coordinator.py:981 direct-indexes loss_info["policy_loss"]. Aux keys
# (ownership/threat/chain/opp_reply/uncertainty) are `.get(...)`-guarded
# (events.py:366) — a graph loss_info supplying none of them must NOT raise.

class _FakeRunnerStats:
    mcts_mean_depth = 0.0
    solver_moves_eligible = 0
    solver_win_proven = 0
    solver_injected = 0
    solver_injected_offwindow = 0
    solver_budget_exhausted = 0
    solver_moves_eligible_seeded = 0
    solver_injected_seeded = 0
    seeded_games_started = 0


class _FakeInferenceStats:
    forward_count = 0
    total_requests = 0


class _FakePool:
    avg_game_length = 0.0
    gumbel_mcts = True  # graph self-play is Gumbel-only; skips the PUCT-only rstats reads
    batch_fill_pct = 0.0
    sims_per_sec = 0.0
    x_winrate = 0.0
    o_winrate = 0.0
    x_wins = 0
    o_wins = 0
    draws = 0
    self_play_positions_pushed = 0

    def runner_stats(self) -> _FakeRunnerStats:
        return _FakeRunnerStats()

    def inference_stats(self) -> _FakeInferenceStats:
        return _FakeInferenceStats()


class _FakeGpuMonitor:
    gpu_util_pct = 0.0
    vram_used_gb = 0.0


def _push_wpa_positions(buf, n: int) -> None:
    import json as _json

    wpa = Path("reports/probes/gnn_integration/wpa_positions.json")
    if not wpa.exists():
        import pytest as _pytest
        _pytest.skip(f"{wpa} not present (WP-A frozen position set)")
    data = _json.loads(wpa.read_text())
    positions = data["positions"][:n]
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    def _empty_neighbor(stones):
        occ = {(q, r) for q, r, _ in stones}
        for q0, r0, _ in stones:
            for dq, dr in neighbors:
                c = (q0 + dq, r0 + dr)
                if c not in occ:
                    return c
        raise AssertionError("no empty neighbor — degenerate position")

    for i, p in enumerate(positions):
        stones = [(int(q), int(r), int(pl)) for q, r, pl in p["stones"]]
        nq, nr = _empty_neighbor(stones)
        buf.push_graph_position(
            stones, [(int(nq), int(nr), 1.0)], int(p["current_player"]),
            int(p["moves_remaining"]), int(p.get("ply", 0)) & 0xFFFF, True,
            1.0 if i % 2 == 0 else -1.0, True, 30, i,
        )


def test_graph_loss_info_satisfies_emit_training_events_contract() -> None:
    """A graph `train_step`'s `loss_info` must survive `emit_training_events`
    (the dashboard producer, `events.py:183`) with no KeyError — the SAME
    keys the existing loss panels already read, no new panel needed (delta
    doc §10 ruling)."""
    import torch

    from engine import HexgBuffer
    from hexo_rl.model.gnn_net import GnnNet
    from hexo_rl.training.events import emit_training_events
    from hexo_rl.training.trainer import Trainer

    buf = HexgBuffer(64, "gnn_axis_v1")
    _push_wpa_positions(buf, 32)

    config = {
        "encoding": "gnn_axis_v1", "fp16": False, "lr": 1e-3, "weight_decay": 1e-4,
        "batch_size": 16, "checkpoint_interval": 1000, "grad_clip": 1.0,
    }
    trainer = Trainer(GnnNet(), config, checkpoint_dir="/tmp", device=torch.device("cpu"))
    loss_info = trainer.train_step(buf, augment=False)

    # Hard-required (direct-indexed) keys, finite.
    for key in ("loss", "policy_loss", "value_loss", "grad_norm"):
        assert key in loss_info, f"missing hard-required key {key!r}"
        import math
        assert math.isfinite(loss_info[key]), f"{key}={loss_info[key]} not finite"

    # No CNN aux keys — GnnNet has no aux heads (standing §6.3).
    for aux_key in ("opp_reply_loss", "ownership_loss", "threat_loss", "chain_loss", "uncertainty_loss"):
        assert aux_key not in loss_info

    # Must not KeyError through the real dashboard producer.
    emit_training_events(
        train_step=trainer.step,
        loss_info=loss_info,
        w_pre=0.0,
        games_played=0,
        last_iter_games=0,
        pool=_FakePool(),
        buffer=buf,
        gpu_monitor=_FakeGpuMonitor(),
        config={},
        mcts_config={},
        capacity=buf.capacity,
        games_per_hour_fn=lambda: 0.0,
        qfire_delta=0,
    )
