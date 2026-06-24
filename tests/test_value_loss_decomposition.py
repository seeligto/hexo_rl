"""§D-VALPROBE Phase 3 — value-axis loss-logging decomposition (logging-only).

Pins three properties of the train_step logging surface:
  1. Explicit decomposition keys exist: value_loss (the historical main BCE
     term — the redundant value_loss_main alias was deleted in B5),
     value_loss_uncertainty / value_loss_aux (WEIGHTED contributions as they
     enter the total), and value_loss_composite == their exact sum.
  2. The decomposition accounts for the optimized total:
     loss == policy_loss + value_loss_composite when no other heads are on.
  3. Logging cannot perturb training math: identically-seeded trainers produce
     bit-identical losses and post-step parameters (the determinism harness
     used for the cross-commit bit-identity check).
"""
from pathlib import Path

import numpy as np
import pytest
import torch

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
    "checkpoint_interval":  1000,
    "log_interval":         1,
    "torch_compile":        False,
}

FULL_CFG = {
    **FAST_CONFIG,
    "uncertainty_weight": 0.1,
    "aux_opp_reply_weight": 0.15,
}


def make_trainer(tmp_path: Path, cfg: dict, seed: int = 1234) -> Trainer:
    torch.manual_seed(seed)
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    return Trainer(model, cfg, checkpoint_dir=tmp_path)


def fill_buffer(size: int = 32, seed: int = 0) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=200)
    rng   = np.random.default_rng(seed)
    own   = np.ones(361, dtype=np.uint8)
    wl    = np.zeros(361, dtype=np.uint8)
    chain = np.zeros((6, 19, 19), dtype=np.float16)
    for _ in range(size):
        state   = rng.random((8, 19, 19), dtype=np.float32).astype(np.float16)
        policy  = rng.dirichlet(np.ones(362)).astype(np.float32)
        outcome = float(rng.choice([-1.0, 0.0, 1.0]))
        buf.push(state, chain, policy, outcome, own, wl)
    return buf


# ── 1. decomposition keys + exact sum ────────────────────────────────────────


def test_decomposition_keys_and_sum_with_heads_on(tmp_path: Path):
    trainer = make_trainer(tmp_path, FULL_CFG)
    result = trainer.train_step(fill_buffer(), augment=False)
    # value_loss_main alias deleted in B5; value_loss is the main BCE term.
    assert "value_loss_main" not in result
    for k in ("value_loss", "value_loss_uncertainty",
              "value_loss_aux", "value_loss_composite"):
        assert k in result, f"{k} missing from loss_info"
    # weighted contributions, not raw terms
    assert result["value_loss_uncertainty"] == pytest.approx(
        0.1 * result["uncertainty_loss"], rel=1e-9)
    assert result["value_loss_aux"] == pytest.approx(
        0.15 * result["opp_reply_loss"], rel=1e-9)
    # composite is the exact sum of the three logged parts
    assert result["value_loss_composite"] == pytest.approx(
        result["value_loss"]
        + result["value_loss_uncertainty"]
        + result["value_loss_aux"],
        rel=1e-12,
    )


def test_decomposition_zero_weights_collapse_to_main(tmp_path: Path):
    trainer = make_trainer(tmp_path, FAST_CONFIG)
    result = trainer.train_step(fill_buffer(), augment=False)
    assert result["value_loss_uncertainty"] == 0.0
    assert result["value_loss_aux"] == 0.0
    assert result["value_loss_composite"] == result["value_loss"]


# ── 2. decomposition accounts for the optimized total ────────────────────────


def test_total_loss_equals_policy_plus_value_composite(tmp_path: Path):
    trainer = make_trainer(tmp_path, FULL_CFG)
    result = trainer.train_step(fill_buffer(), augment=False)
    # only policy/value/aux/uncertainty heads are on in FULL_CFG
    assert result["loss"] == pytest.approx(
        result["policy_loss"] + result["value_loss_composite"],
        rel=1e-4, abs=1e-5,
    )


# ── decomposition reaches the structlog surface ──────────────────────────────


def test_train_step_log_carries_decomposition(tmp_path: Path):
    import structlog.testing

    trainer = make_trainer(tmp_path, FULL_CFG)
    buf = fill_buffer()
    with structlog.testing.capture_logs() as captured:
        trainer.train_step(buf, augment=False)
    evt = [e for e in captured if e.get("event") == "train_step"][0]
    assert "value_loss_main" not in evt  # alias deleted in B5
    for k in ("value_loss", "value_loss_uncertainty",
              "value_loss_aux", "value_loss_composite"):
        assert k in evt, f"{k} missing from train_step structlog event"


# ── 3. bit-identity determinism harness ──────────────────────────────────────
#
# ReplayBuffer.sample_batch draws indices from an unseeded RNG, and CUDA conv
# backward is non-deterministic — so the harness (a) fills the buffer with
# IDENTICAL rows (any sampled batch is content-identical) and (b) runs on CPU.
# This isolates exactly what the cross-commit check needs: with the same
# inputs, a logging-only change must leave loss and post-step parameters
# bit-identical.


def fill_buffer_identical_rows(size: int = 32) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=200)
    rng   = np.random.default_rng(7)
    own   = np.ones(361, dtype=np.uint8)
    wl    = np.zeros(361, dtype=np.uint8)
    chain = np.zeros((6, 19, 19), dtype=np.float16)
    state  = rng.random((8, 19, 19), dtype=np.float32).astype(np.float16)
    policy = rng.dirichlet(np.ones(362)).astype(np.float32)
    for _ in range(size):
        buf.push(state, chain, policy, 1.0, own, wl)
    return buf


def _run_one(tmp_path: Path, tag: str):
    torch.manual_seed(99)
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, FULL_CFG, checkpoint_dir=tmp_path / tag,
                      device=torch.device("cpu"))
    result = trainer.train_step(fill_buffer_identical_rows(), augment=False)
    params = [p.detach().cpu().clone() for p in trainer.model.parameters()]
    return result, params


def test_train_step_bit_identical_across_identical_runs(tmp_path: Path):
    r1, p1 = _run_one(tmp_path, "a")
    r2, p2 = _run_one(tmp_path, "b")
    assert r1["loss"] == r2["loss"]
    assert r1["value_loss_composite"] == r2["value_loss_composite"]
    for a, b in zip(p1, p2):
        assert torch.equal(a, b), "post-step parameters diverged between identical runs"
