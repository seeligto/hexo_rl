"""
Phase 4.0 smoke tests: mixed-buffer training, buffer resize, playout cap config.

Run with: pytest tests/test_phase4_smoke.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from engine import ReplayBuffer

CHANNELS = 24
BOARD_SIZE = 19
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1  # 362


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_model(device: torch.device):
    from hexo_rl.model.network import HexTacToeNet
    return HexTacToeNet(board_size=BOARD_SIZE, res_blocks=2, filters=32).to(device)


def _fill_buffer(buf: ReplayBuffer, n: int, value_only_frac: float = 0.0) -> None:
    """Push n random entries. A fraction have zero-policy (value-only)."""
    own = np.ones(361, dtype=np.uint8)
    wl  = np.zeros(361, dtype=np.uint8)
    for i in range(n):
        state = np.random.randn(CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)
        if np.random.random() < value_only_frac:
            policy = np.zeros(N_ACTIONS, dtype=np.float32)
        else:
            policy = np.abs(np.random.randn(N_ACTIONS).astype(np.float32))
            policy /= policy.sum()
        outcome = float(np.random.choice([-1.0, 0.0, 1.0]))
        buf.push(state, policy, outcome, own, wl)


# ── Test: mixed-buffer training (10 steps) ───────────────────────────────────

def test_mixed_buffer_training_10_steps():
    """10 training steps with mixed pretrained + self-play buffers complete without error."""
    from hexo_rl.training.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(device)
    config = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 16,
        "checkpoint_interval": 999,
        "log_interval": 5,
        "aux_opp_reply_weight": 0.0,
        "lr_schedule": "none",
    }

    trainer = Trainer(model, config, checkpoint_dir="/tmp/test_phase4_ckpt", device=device)

    # Pretrained buffer (static).
    pretrained_buf = ReplayBuffer(capacity=100)
    _fill_buffer(pretrained_buf, 100)

    # Self-play buffer (with some value-only entries from fast games).
    selfplay_buf = ReplayBuffer(capacity=200)
    _fill_buffer(selfplay_buf, 200, value_only_frac=0.25)

    batch_size = config["batch_size"]
    decay_steps = 1_000_000.0
    losses = []

    for step in range(10):
        w_pre = max(0.1, 0.8 * math.exp(-step / decay_steps))
        n_pre = max(1, int(math.ceil(batch_size * w_pre)))
        n_self = batch_size - n_pre

        s_pre, p_pre, o_pre, own_pre, wl_pre = pretrained_buf.sample_batch(n_pre, True)
        s_self, p_self, o_self, own_self, wl_self = selfplay_buf.sample_batch(max(1, n_self), True)

        states = np.concatenate([s_pre, s_self], axis=0)
        policies = np.concatenate([p_pre, p_self], axis=0)
        outcomes = np.concatenate([o_pre, o_self], axis=0)
        ownership = np.concatenate([own_pre, own_self], axis=0)
        winning_line = np.concatenate([wl_pre, wl_self], axis=0)

        loss_info = trainer.train_step_from_tensors(
            states, policies, outcomes,
            ownership_targets=ownership, threat_targets=winning_line,
            n_pretrain=n_pre,
        )
        losses.append(loss_info["loss"])
        assert "policy_loss" in loss_info
        assert "value_loss" in loss_info

    assert len(losses) == 10
    assert all(math.isfinite(l) for l in losses), f"Non-finite loss: {losses}"


# ── Test: value-only policy masking ──────────────────────────────────────────

def test_value_only_batch_does_not_crash():
    """A batch of all-zero policies (value-only) should not crash training."""
    from hexo_rl.training.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(device)
    config = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "checkpoint_interval": 999,
        "log_interval": 5,
        "aux_opp_reply_weight": 0.0,
        "lr_schedule": "none",
    }
    trainer = Trainer(model, config, checkpoint_dir="/tmp/test_phase4_vo", device=device)

    # All-zero policies = value-only.
    states = np.random.randn(8, CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float16)
    policies = np.zeros((8, N_ACTIONS), dtype=np.float32)
    outcomes = np.array([1.0, -1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0], dtype=np.float32)

    loss_info = trainer.train_step_from_tensors(states, policies, outcomes)
    assert loss_info["policy_loss"] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(loss_info["value_loss"])


# ── Test: buffer resize during training ──────────────────────────────────────

def test_buffer_resize_during_training():
    """Resize mid-training: push, resize, push more, sample, train."""
    from hexo_rl.training.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(device)
    config = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "checkpoint_interval": 999,
        "log_interval": 5,
        "aux_opp_reply_weight": 0.0,
        "lr_schedule": "none",
    }
    trainer = Trainer(model, config, checkpoint_dir="/tmp/test_phase4_resize", device=device)

    buf = ReplayBuffer(capacity=50)
    _fill_buffer(buf, 50)

    # Train 3 steps.
    for _ in range(3):
        trainer.train_step(buf)

    # Resize.
    buf.resize(100)
    assert buf.capacity == 100
    assert buf.size == 50

    # Push more.
    _fill_buffer(buf, 30)
    assert buf.size == 80

    # Train 3 more steps.
    for _ in range(3):
        loss_info = trainer.train_step(buf)
        assert math.isfinite(loss_info["loss"])


# ── Test: pretrained weight decay ────────────────────────────────────────────

def test_pretrained_weight_schedule():
    """Verify the pretrained weight formula decays correctly."""
    min_w = 0.1
    initial_w = 0.8
    decay_steps = 1_000_000.0

    def w(step):
        return max(min_w, initial_w * math.exp(-step / decay_steps))

    assert w(0) == pytest.approx(0.8, abs=1e-6)
    assert w(1_000_000) == pytest.approx(max(0.1, 0.8 * math.exp(-1)), abs=1e-3)
    # At very large step, should floor at min_w.
    assert w(100_000_000) == pytest.approx(0.1, abs=1e-6)
    # Monotonically decreasing.
    prev = w(0)
    for s in range(0, 3_000_000, 100_000):
        cur = w(s)
        assert cur <= prev + 1e-9
        prev = cur


# ── Test: SelfPlayRunner accepts playout cap params ──────────────────────

def test_runner_accepts_playout_cap_params():
    """SelfPlayRunner constructor accepts the new playout cap kwargs."""
    from engine import SelfPlayRunner

    runner = SelfPlayRunner(
        n_workers=1,
        n_simulations=50,
        fast_prob=0.25,
        fast_sims=50,
        standard_sims=400,
        temp_threshold_compound_moves=15,
    )
    assert not runner.is_running()
