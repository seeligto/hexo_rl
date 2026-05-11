"""Test Trainer training path accepts v6w25 (25×25) tensor shapes end-to-end.

§173 A8-fix: catches any remaining 19×19 hardcode in the Python training loop
(aux decode, batch assembly, recent-buffer reshape).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer


def _v6w25_config() -> dict:
    return {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "checkpoint_interval": 10,
        "log_interval": 10,
        "board_size": 25,
        "encoding": {"version": "v6w25"},
        "fp16": False,
        "aux_opp_reply_weight": 0.0,
        "uncertainty_weight": 0.0,
        "ownership_weight": 0.0,
        "threat_weight": 0.0,
        "aux_chain_weight": 0.0,
        "policy_prune_frac": 0.0,
        "entropy_reg_weight": 0.0,
        "grad_clip": 1.0,
    }


@pytest.mark.timeout(30)
def test_trainer_v6w25_train_step_from_tensors():
    """Trainer._train_on_batch with v6w25-shaped numpy arrays must not crash."""
    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=25, in_channels=8, filters=8, res_blocks=1, encoding="v6w25"
    )
    cfg = _v6w25_config()
    trainer = Trainer(model, cfg, checkpoint_dir="/tmp/hexo_test_ckpts", device=device)

    B = cfg["batch_size"]
    # v6w25 wire shapes
    states = np.random.randn(B, 8, 25, 25).astype(np.float16)
    policies = np.random.dirichlet(np.ones(626), size=B).astype(np.float32)
    outcomes = np.random.choice([-1.0, 0.0, 1.0], size=B).astype(np.float32)
    chain_planes = np.random.randn(B, 6, 25, 25).astype(np.float16)
    ownership = np.random.randint(0, 3, size=(B, 25, 25), dtype=np.uint8)
    winning_line = np.random.randint(0, 2, size=(B, 25, 25), dtype=np.uint8)
    is_full_search = np.ones(B, dtype=np.uint8)

    result = trainer.train_step_from_tensors(
        states=states,
        policies=policies,
        outcomes=outcomes,
        chain_planes=chain_planes,
        ownership_targets=ownership,
        threat_targets=winning_line,
        is_full_search=is_full_search,
        n_pretrain=0,
        n_recent=0,
    )

    assert math.isfinite(result["loss"])
    assert math.isfinite(result["grad_norm"])
    assert result["policy_loss"] >= 0.0  # CE loss is non-negative


@pytest.mark.timeout(30)
def test_trainer_v6w25_recent_buffer_reshape():
    """train_step with a RecentBuffer holding v6w25 rows must reshape aux correctly."""
    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=25, in_channels=8, filters=8, res_blocks=1, encoding="v6w25"
    )
    cfg = _v6w25_config()
    cfg["recency_weight"] = 0.5
    trainer = Trainer(model, cfg, checkpoint_dir="/tmp/hexo_test_ckpts2", device=device)

    from hexo_rl.training.recency_buffer import RecentBuffer
    recent = RecentBuffer(
        capacity=16,
        state_shape=(8, 25, 25),
        policy_len=626,
        aux_stride=625,
    )
    for _ in range(8):
        recent.push(
            state=np.random.randn(8, 25, 25).astype(np.float16),
            chain_planes=np.random.randn(6, 25, 25).astype(np.float16),
            policy=np.random.dirichlet(np.ones(626)).astype(np.float32),
            outcome=1.0,
            ownership=np.random.randint(0, 3, size=625, dtype=np.uint8),
            winning_line=np.random.randint(0, 2, size=625, dtype=np.uint8),
        )

    # We need a real Rust ReplayBuffer for the uniform side.
    # Skip if engine extension not built.
    try:
        from engine import ReplayBuffer
    except ImportError:
        pytest.skip("engine extension not built")

    buf = ReplayBuffer(capacity=64, encoding="v6w25")
    for _ in range(16):
        buf.push_game(
            states=np.random.randn(1, 8, 25, 25).astype(np.float16),
            chain_planes=np.random.randn(1, 6, 25, 25).astype(np.float16),
            policies=np.random.dirichlet(np.ones(626), size=1).astype(np.float32),
            outcomes=np.array([1.0], dtype=np.float32),
            ownership=np.random.randint(0, 3, size=(1, 625), dtype=np.uint8),
            winning_line=np.random.randint(0, 2, size=(1, 625), dtype=np.uint8),
        )

    result = trainer.train_step(buf, augment=False, recent_buffer=recent)
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["grad_norm"])
