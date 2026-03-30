"""
Phase 1 tests for Trainer (training step + checkpoint round-trip).

Run with: .venv/bin/pytest tests/test_trainer.py -v
"""
import json
from pathlib import Path

import numpy as np
import torch

from python.model.network import HexTacToeNet
from python.training.replay_buffer import ReplayBuffer
from python.training.trainer import Trainer


FAST_CONFIG = {
    "board_size":           9,
    "res_blocks":           2,
    "filters":              32,
    "batch_size":           8,
    "lr":                   2e-3,
    "weight_decay":         1e-4,
    "checkpoint_interval":  5,
    "log_interval":         1,
}


def make_trainer(tmp_path: Path) -> Trainer:
    model = HexTacToeNet(board_size=9, res_blocks=2, filters=32)
    return Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path)


def fill_buffer(size: int = 32, board_size: int = 9) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=200, board_channels=18, board_size=board_size)
    rng = np.random.default_rng(0)
    n_actions = board_size * board_size + 1
    for _ in range(size):
        state   = rng.random((18, board_size, board_size), dtype=np.float32).astype(np.float16)
        policy  = rng.dirichlet(np.ones(n_actions)).astype(np.float32)
        outcome = float(rng.choice([-1.0, 0.0, 1.0]))
        buf.push(state, policy, outcome)
    return buf


# ── Training step ─────────────────────────────────────────────────────────────

def test_train_step_returns_loss_keys(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    result  = trainer.train_step(buf)
    assert "loss"        in result
    assert "policy_loss" in result
    assert "value_loss"  in result


def test_train_step_loss_is_finite(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    result  = trainer.train_step(buf)
    for k, v in result.items():
        assert np.isfinite(v), f"{k} = {v} is not finite"


def test_train_step_increments_step(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    assert trainer.step == 0
    trainer.train_step(buf)
    assert trainer.step == 1
    trainer.train_step(buf)
    assert trainer.step == 2


def test_loss_decreases_over_multiple_steps(tmp_path: Path):
    """Policy loss should decrease when the model is trained on fixed data."""
    torch.manual_seed(42)
    model   = HexTacToeNet(board_size=9, res_blocks=2, filters=32)
    trainer = Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path)
    buf     = fill_buffer(size=64)

    first_loss = trainer.train_step(buf)["loss"]
    last_loss  = first_loss
    for _ in range(199):
        last_loss = trainer.train_step(buf)["loss"]

    # Over 200 steps the loss should drop noticeably from the initial value.
    assert last_loss < first_loss, (
        f"loss did not decrease: first={first_loss:.4f} last={last_loss:.4f}"
    )


# ── Checkpoint save ───────────────────────────────────────────────────────────

def test_checkpoint_saved_at_interval(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    # checkpoint_interval = 5, so step 5 triggers a save
    for _ in range(5):
        trainer.train_step(buf)
    ckpt_files = list(tmp_path.glob("checkpoint_*.pt"))
    assert len(ckpt_files) == 1
    assert ckpt_files[0].name == "checkpoint_00000005.pt"


def test_inference_only_saved_with_checkpoint(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    for _ in range(5):
        trainer.train_step(buf)
    assert (tmp_path / "inference_only.pt").exists()


def test_checkpoint_log_json_written(tmp_path: Path):
    trainer = make_trainer(tmp_path)
    buf     = fill_buffer()
    for _ in range(5):
        trainer.train_step(buf)
    log_path = tmp_path / "checkpoint_log.json"
    assert log_path.exists()
    with open(log_path) as f:
        log = json.load(f)
    assert len(log) == 1
    assert log[0]["step"] == 5
    assert "loss" in log[0]


# ── Checkpoint round-trip ─────────────────────────────────────────────────────

def test_checkpoint_round_trip(tmp_path: Path):
    """Load a checkpoint and verify model outputs match before/after."""
    torch.manual_seed(0)
    model   = HexTacToeNet(board_size=9, res_blocks=2, filters=32)
    trainer = Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path)
    buf     = fill_buffer()

    for _ in range(5):
        trainer.train_step(buf)

    ckpt_path = tmp_path / "checkpoint_00000005.pt"
    assert ckpt_path.exists()

    # Record model outputs before reload.
    x = torch.zeros(1, 18, 9, 9, device=trainer.device)
    trainer.model.eval()
    with torch.no_grad():
        log_p_before, v_before = trainer.model(x)

    # Reload.
    restored = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path)
    assert restored.step == 5

    x_r = torch.zeros(1, 18, 9, 9, device=restored.device)
    restored.model.eval()
    with torch.no_grad():
        log_p_after, v_after = restored.model(x_r)

    # Compare on CPU.
    assert torch.allclose(log_p_before.cpu(), log_p_after.cpu(), atol=1e-4), \
        "policy mismatch after reload"
    assert torch.allclose(v_before.cpu(), v_after.cpu(), atol=1e-4), \
        "value mismatch after reload"


def test_checkpoint_optimizer_state_preserved(tmp_path: Path):
    """AdamW momentum state should survive a round-trip."""
    model   = HexTacToeNet(board_size=9, res_blocks=2, filters=32)
    trainer = Trainer(model, FAST_CONFIG, checkpoint_dir=tmp_path)
    buf     = fill_buffer()
    for _ in range(5):
        trainer.train_step(buf)

    ckpt_path = tmp_path / "checkpoint_00000005.pt"
    restored  = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path)

    orig_state = trainer.optimizer.state_dict()
    rest_state = restored.optimizer.state_dict()
    assert len(orig_state["state"]) == len(rest_state["state"]), \
        "optimizer state group count mismatch"


def test_scheduler_steps_each_train_step(tmp_path: Path):
    cfg = {
        **FAST_CONFIG,
        "lr_schedule": "cosine",
        "total_steps": 20,
        "min_lr": 1e-5,
    }
    model = HexTacToeNet(board_size=9, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    assert trainer.scheduler is not None
    start_epoch = trainer.scheduler.last_epoch

    trainer.train_step(buf)
    assert trainer.scheduler.last_epoch == start_epoch + 1

    trainer.train_step(buf)
    assert trainer.scheduler.last_epoch == start_epoch + 2


def test_scheduler_state_round_trip(tmp_path: Path):
    cfg = {
        **FAST_CONFIG,
        "lr_schedule": "cosine",
        "total_steps": 20,
        "min_lr": 1e-5,
    }
    model = HexTacToeNet(board_size=9, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    for _ in range(5):
        trainer.train_step(buf)

    ckpt_path = tmp_path / "checkpoint_00000005.pt"
    restored = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path)

    assert trainer.scheduler is not None
    assert restored.scheduler is not None
    assert restored.scheduler.last_epoch == trainer.scheduler.last_epoch


def test_load_checkpoint_allows_config_override(tmp_path: Path):
    cfg = {
        **FAST_CONFIG,
        "lr_schedule": "cosine",
        "total_steps": 20,
        "min_lr": 1e-5,
    }
    model = HexTacToeNet(board_size=9, res_blocks=2, filters=32)
    trainer = Trainer(model, cfg, checkpoint_dir=tmp_path)
    buf = fill_buffer()

    for _ in range(5):
        trainer.train_step(buf)

    ckpt_path = tmp_path / "checkpoint_00000005.pt"
    restored = Trainer.load_checkpoint(
        ckpt_path,
        checkpoint_dir=tmp_path,
        config_overrides={"total_steps": 100},
    )

    assert restored.config.get("total_steps") == 100
    assert restored.scheduler is not None
    assert int(restored.scheduler.T_max) == 100


def test_normalize_state_dict_adds_tower_aliases():
    state = {
        "trunk.tower.0.conv1.weight": torch.randn(8, 8, 3, 3),
        "_orig_mod.module.policy_fc.weight": torch.randn(10, 20),
    }

    normalized = Trainer._normalize_model_state_dict_keys(state)

    assert "trunk.tower.0.conv1.weight" in normalized
    assert "tower.0.conv1.weight" in normalized
    assert "policy_fc.weight" in normalized


def test_load_weights_only_checkpoint_infers_architecture(tmp_path: Path):
    base = HexTacToeNet(board_size=9, in_channels=18, res_blocks=2, filters=32)
    base_state = base.state_dict()

    # Simulate a bootstrap-style checkpoint that only has trunk.* keys.
    trunk_only = {
        k: v
        for k, v in base_state.items()
        if not k.startswith("tower.")
    }

    ckpt_path = tmp_path / "bootstrap_like.pt"
    torch.save(trunk_only, ckpt_path)

    # Intentionally mismatched fallback config: loader should reconcile from state_dict.
    fallback = {
        "board_size": 19,
        "res_blocks": 1,
        "filters": 16,
        "in_channels": 18,
        "batch_size": 8,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "checkpoint_interval": 5,
        "log_interval": 1,
    }

    restored = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path, fallback_config=fallback)

    assert restored.model.board_size == 9
    assert restored.model.res_blocks == 2
    assert restored.model.filters == 32
