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
    """Static source guard: ``hexo_rl/training/loop.py`` must not call
    ``log.info('train_step', ...)``. The authoritative cadence is
    trainer.train_step_from_tensors (one emission per real step). A
    log_interval-gated duplicate in loop.py was the Q27 smoke root cause.
    """
    src = (Path(__file__).resolve().parents[1]
           / "hexo_rl" / "training" / "loop.py").read_text()
    pattern = re.compile(r"log\.info\(\s*[\"']train_step[\"']")
    matches = pattern.findall(src)
    assert not matches, (
        f"hexo_rl/training/loop.py has {len(matches)} log.info('train_step', ...) "
        "call(s); this duplicates trainer.train_step_from_tensors's per-step "
        "emission and caused 6050 events for 5500 steps in Q27 smoke 2026-04-19."
    )
