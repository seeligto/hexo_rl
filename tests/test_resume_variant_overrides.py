"""Regression tests for D-FULLSPEC E0 — resume --variant override precedence.

Pre-E0 bug: on a full-checkpoint resume, the checkpoint-baked config WON for
every key outside a tiny whitelist, so a resumed --variant silently dropped its
runtime knobs (completed_q_values, aux_*_weight, entropy_reg_weight, grad_clip,
per_class_target_temperature, policy_prune_frac, value_distill_*) and re-baked
the reverted defaults of bot_batch_share / draw_value / ply_cap_value into the
next checkpoint (provenance corruption).

The E0 fix inverts the precedence in
``hexo_rl.training.orchestrator.build_resume_config_overrides``: seed the
overrides from the full launch config minus ``RESUME_CHECKPOINT_OWNED_KEYS``
(encoding/arch pins + optimizer/scheduler/step state). These tests pin:
  1. every vulnerable runtime + provenance key resumes to the LAUNCH value;
  2. encoding/arch keys still come from the CHECKPOINT (helper excludes them);
  3. scheduler horizon stays checkpoint-owned WITHOUT
     --override-scheduler-horizon, and re-horizons WITH it (mirrors
     tests/test_scheduler_resume.py).
"""
from pathlib import Path

import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer
from hexo_rl.training.orchestrator import (
    build_resume_config_overrides,
    RESUME_CHECKPOINT_OWNED_KEYS,
)


# Checkpoint (pre-resume) config — the "old run" baked values. Arch + scheduler
# horizon live here and must survive the resume; the runtime/provenance knobs
# carry the OLD values that the variant flips below.
CHECKPOINT_CONFIG = {
    # arch / encoding (checkpoint-owned)
    "board_size":   19,
    "res_blocks":   2,
    "filters":      32,
    # optimizer / scheduler state (checkpoint-owned)
    "batch_size":   8,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "lr_schedule":  "cosine",
    "total_steps":  1_000_000,
    "eta_min":      1e-6,
    # checkpoint cadence
    "checkpoint_interval": 5,
    "log_interval":        1,
    "torch_compile":       False,
    # RUNTIME-vulnerable knobs — OLD values
    "completed_q_values":            False,
    "aux_opp_reply_weight":          0.0,
    "aux_chain_weight":              0.0,
    "ply_index_weight":              0.0,
    "policy_prune_frac":             0.0,
    "entropy_reg_weight":            0.0,
    "grad_clip":                     1.0,
    "uncertainty_weight":            0.0,
    "recency_weight":                0.0,
    "ownership_weight":              0.0,
    "threat_weight":                 0.0,
    "per_class_target_temperature":  {"enabled": False},
    "value_distill_weight":          0.0,
    # PROVENANCE-only knobs — OLD values
    "draw_value":      -0.5,
    "ply_cap_value":   -0.5,
    "bot_batch_share": 0.0,
    "mixing":          {"bot_batch_share": 0.0},
}

# The full set of knobs the variant flips, with their LAUNCH values. Every one
# of these must equal the launch value in resumed.config after the fix.
VULNERABLE_LAUNCH_VALUES = {
    # runtime-vulnerable (read via trainer.config / self.config)
    "completed_q_values":            True,
    "aux_opp_reply_weight":          0.25,
    "aux_chain_weight":              0.10,
    "ply_index_weight":              0.05,
    "policy_prune_frac":             0.30,
    "entropy_reg_weight":            0.01,
    "grad_clip":                     0.5,
    "uncertainty_weight":            0.2,
    "recency_weight":                0.3,
    "ownership_weight":              0.15,
    "threat_weight":                 0.4,
    "per_class_target_temperature":  {"enabled": True, "temperature": 1.5},
    "value_distill_weight":          0.7,
    # provenance-only (loop/pool read launch cfg; checkpoint bake must be right)
    "draw_value":      -0.1,
    "ply_cap_value":   0.0,
    "bot_batch_share": 0.15,
    "mixing":          {"bot_batch_share": 0.15},
}


def _variant_config() -> dict:
    """The launch (--variant) config: checkpoint base, with the vulnerable
    knobs flipped to launch values AND a deliberately MISMATCHED arch + horizon
    that the EXCLUDE-set must keep checkpoint-owned."""
    cfg = dict(CHECKPOINT_CONFIG)
    cfg.update(VULNERABLE_LAUNCH_VALUES)
    # Arch + horizon the variant should NOT be able to impose on a resume:
    cfg["res_blocks"]     = 4            # checkpoint has 2 — must stay 2
    cfg["filters"]        = 64           # checkpoint has 32 — must stay 32
    cfg["total_steps"]    = 2_000_000    # checkpoint has 1_000_000 — stays 1M w/o flag
    cfg["eta_min"]        = 9e-9         # checkpoint has 1e-6 — checkpoint-owned
    return cfg


def _make_checkpoint_trainer(tmp_path: Path) -> Trainer:
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    return Trainer(model, dict(CHECKPOINT_CONFIG), checkpoint_dir=tmp_path)


def _save(trainer: Trainer, tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoint_00000000.pt"
    from hexo_rl.training.checkpoints import save_full_checkpoint
    save_full_checkpoint(
        trainer.model, trainer.optimizer, trainer.scaler,
        trainer.scheduler, trainer.step, trainer.config, ckpt,
    )
    return ckpt


def test_helper_excludes_checkpoint_owned_keys():
    """build_resume_config_overrides must drop every EXCLUDE-set key (without
    the flag) and keep every vulnerable knob."""
    overrides = build_resume_config_overrides(
        _variant_config(), override_scheduler_horizon=False,
    )
    # Encoding/arch/optimizer-scheduler keys are checkpoint-owned → excluded.
    for owned in ("res_blocks", "filters", "in_channels", "lr", "weight_decay",
                  "eta_min", "total_steps", "scheduler_t_max", "lr_schedule"):
        assert owned not in overrides, f"{owned!r} leaked into resume overrides"
    # Vulnerable knobs travel.
    for k, v in VULNERABLE_LAUNCH_VALUES.items():
        assert overrides.get(k) == v, f"{k!r} missing/wrong in overrides"


def test_resume_applies_launch_vulnerable_keys(tmp_path: Path):
    """End-to-end: resumed.config carries the LAUNCH value for the full
    vulnerable key-set, and the checkpoint value for arch/horizon."""
    trainer = _make_checkpoint_trainer(tmp_path)
    # sanity: checkpoint really baked the OLD values
    assert trainer.config["completed_q_values"] is False
    assert trainer.config["bot_batch_share"] == 0.0
    ckpt = _save(trainer, tmp_path)

    variant_cfg = _variant_config()
    overrides = build_resume_config_overrides(
        variant_cfg, override_scheduler_horizon=False,
    )
    resumed = Trainer.load_checkpoint(
        ckpt,
        checkpoint_dir=tmp_path,
        fallback_config=variant_cfg,
        config_overrides=overrides,
    )

    # (1) every vulnerable key resumes to the LAUNCH value
    for k, v in VULNERABLE_LAUNCH_VALUES.items():
        assert resumed.config.get(k) == v, (
            f"vulnerable key {k!r} = {resumed.config.get(k)!r}, "
            f"expected launch {v!r}"
        )
    # explicit spot-checks for the two consumer classes
    assert resumed.config["completed_q_values"] is True        # trainer.py:724
    assert resumed.config["grad_clip"] == 0.5                  # trainer.py:879
    assert resumed.config["mixing"]["bot_batch_share"] == 0.15  # loop.py:278
    assert resumed.config["draw_value"] == -0.1                # pool.py:365
    assert resumed.config["ply_cap_value"] == 0.0             # pool.py:369

    # (2) arch keys stay CHECKPOINT-owned (variant tried 4/64; real model = 2/32)
    assert resumed.config["res_blocks"] == 2
    assert resumed.config["filters"] == 32

    # (3) scheduler horizon stays checkpoint-owned without the flag
    assert resumed.scheduler is not None
    assert resumed.scheduler.T_max == 1_000_000, (
        f"T_max leaked launch value without flag: {resumed.scheduler.T_max}"
    )


def test_resume_negative_pin_scheduler_T_max(tmp_path: Path):
    """Without --override-scheduler-horizon the checkpoint T_max survives even
    though the variant asks for a different total_steps (mirrors
    tests/test_scheduler_resume.py)."""
    trainer = _make_checkpoint_trainer(tmp_path)
    assert trainer.scheduler.T_max == 1_000_000
    ckpt = _save(trainer, tmp_path)

    overrides = build_resume_config_overrides(
        _variant_config(), override_scheduler_horizon=False,
    )
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=_variant_config(), config_overrides=overrides,
    )
    assert resumed.scheduler.T_max == 1_000_000


def test_resume_with_horizon_flag_rehorizons(tmp_path: Path):
    """WITH --override-scheduler-horizon the variant total_steps re-horizons the
    scheduler (gate preserved)."""
    trainer = _make_checkpoint_trainer(tmp_path)
    ckpt = _save(trainer, tmp_path)

    overrides = build_resume_config_overrides(
        _variant_config(), override_scheduler_horizon=True,
    )
    assert overrides.get("total_steps") == 2_000_000
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=_variant_config(), config_overrides=overrides,
    )
    assert resumed.scheduler.T_max == 2_000_000


def test_exclude_set_membership():
    """Pin the EXCLUDE-set so a future edit can't accidentally re-capture a
    runtime knob (e.g. completed_q_values) into checkpoint ownership."""
    for runtime_knob in ("completed_q_values", "aux_opp_reply_weight",
                         "aux_chain_weight", "ply_index_weight",
                         "per_class_target_temperature", "policy_prune_frac",
                         "entropy_reg_weight", "grad_clip", "value_distill_weight",
                         "draw_value", "ply_cap_value", "bot_batch_share", "mixing"):
        assert runtime_knob not in RESUME_CHECKPOINT_OWNED_KEYS
    for owned in ("encoding", "res_blocks", "filters", "in_channels",
                  "total_steps", "scheduler_t_max", "eta_min", "lr"):
        assert owned in RESUME_CHECKPOINT_OWNED_KEYS
