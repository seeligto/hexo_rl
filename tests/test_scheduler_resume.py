"""Regression tests for E-004: --override-scheduler-horizon gate.

Verifies that Trainer.load_checkpoint only mutates scheduler T_max when
config_overrides explicitly contains 'total_steps' (i.e. the flag was set),
and preserves the checkpoint T_max otherwise.
"""
from pathlib import Path

import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer


COSINE_CONFIG = {
    "board_size":          19,
    "res_blocks":          2,
    "filters":             32,
    "batch_size":          8,
    "lr":                  1e-3,
    "weight_decay":        1e-4,
    "checkpoint_interval": 5,
    "log_interval":        1,
    "torch_compile":       False,
    "lr_schedule":         "cosine",
    "total_steps":         1_000_000,
    "eta_min":             1e-6,
}


def _make_trainer(cfg: dict, tmp_path: Path) -> Trainer:
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    return Trainer(model, cfg, checkpoint_dir=tmp_path)


def _save(trainer: Trainer, tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoint_00000000.pt"
    from hexo_rl.training.checkpoints import save_full_checkpoint
    save_full_checkpoint(
        trainer.model, trainer.optimizer, trainer.scaler,
        trainer.scheduler, trainer.step, trainer.config, ckpt,
    )
    return ckpt


def test_resume_without_flag_preserves_scheduler_T_max(tmp_path: Path):
    """No total_steps in config_overrides → checkpoint T_max survives."""
    trainer = _make_trainer(COSINE_CONFIG, tmp_path)
    assert trainer.scheduler is not None
    assert trainer.scheduler.T_max == 1_000_000

    ckpt = _save(trainer, tmp_path)

    # Simulate train.py resume WITHOUT --override-scheduler-horizon:
    # config_overrides does NOT contain total_steps.
    config_overrides = {"torch_compile": False}
    resumed = Trainer.load_checkpoint(
        ckpt,
        checkpoint_dir=tmp_path,
        fallback_config=COSINE_CONFIG,
        config_overrides=config_overrides,
    )
    assert resumed.scheduler is not None
    assert resumed.scheduler.T_max == 1_000_000, (
        f"T_max mutated without flag: got {resumed.scheduler.T_max}"
    )


def test_resume_with_flag_rehorizons_to_iterations(tmp_path: Path):
    """total_steps in config_overrides (flag set) → T_max re-set to new value."""
    trainer = _make_trainer(COSINE_CONFIG, tmp_path)
    ckpt = _save(trainer, tmp_path)

    # Simulate train.py resume WITH --override-scheduler-horizon --iterations 50000:
    config_overrides = {"torch_compile": False, "total_steps": 50_000}
    resumed = Trainer.load_checkpoint(
        ckpt,
        checkpoint_dir=tmp_path,
        fallback_config=COSINE_CONFIG,
        config_overrides=config_overrides,
    )
    assert resumed.scheduler is not None
    assert resumed.scheduler.T_max == 50_000, (
        f"T_max not re-horizoned: got {resumed.scheduler.T_max}"
    )
