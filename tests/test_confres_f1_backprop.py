"""CONFRES F1(A) split-brain fix — F1-deferred loop-read knobs are back-propagated into combined_config.

F1 defer-to-baked preserves a base-inherited knob's BAKED value into ``trainer.config``. But the
training loop reads ``combined_config`` (``scripts/train.py:274 run_training_loop(config=combined_config)``),
and ``orchestrator.init_trainer`` back-propagated ONLY the 4 ENCODING keys (``cluster_window_size``,
``cluster_threshold``, ``legal_move_radius``, ``encoding``) from ``trainer.config`` → ``combined_config``.
So for every OTHER F1-deferred loop-read knob (``ply_cap_value``, ``draw_value``,
``completed_q_values``, ``bot_batch_share``, ``min_buffer_size``, ``amp_dtype``, …) the loop / self-play
/ inference EXECUTED the launch-merge (base-default) value while:
  - the ``resume_base_default_deferred_to_baked`` WARN asserted "deferring to the baked value",
  - the re-baked checkpoint recorded the baked value,
  - the ``resolved_config`` emission reported ``source: checkpoint``.
== a split-brain: three surfaces reported the baked value the run did NOT execute. The ``amp_dtype``
case is a numerical trainer⊥inference autocast-dtype split (Trainer reads trainer.config; InferenceServer
reads combined_config).

FIX: after F1 defer-to-baked runs, back-propagate the FULL set of F1-DEFERRED keys (programmatically —
the exact keys F1 fired the defer on, from ``trainer.f1_deferred_keys``) from ``trainer.config`` →
``combined_config`` so the loop/self-play/inference EXECUTE the values F1 preserved + the WARN/emission
report. Must NOT break F2 (load-time assert), owned-key semantics, or E0 (declared knobs still win).

TDD — written first, watched fail.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import structlog
import structlog.testing

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.orchestrator import init_trainer
from hexo_rl.training.trainer import Trainer


# ── Minimal resumeable checkpoint config ─────────────────────────────────────
def _ckpt_config(**over) -> dict:
    cfg = {
        "encoding":     "v6_live2_ls",
        "res_blocks":   2,
        "filters":      32,
        "batch_size":   8,
        "lr":           1e-3,
        "weight_decay": 1e-4,
        "lr_schedule":  "cosine",
        "total_steps":  1_000_000,
        "eta_min":      1e-6,
        "checkpoint_interval": 5,
        "log_interval":        1,
        "torch_compile":       False,
    }
    cfg.update(over)
    return cfg


def _make_and_save_ckpt(cfg: dict, tmp_path: Path) -> Path:
    model = HexTacToeNet(board_size=19, in_channels=4, res_blocks=2, filters=32, encoding="v6_live2_ls")
    trainer = Trainer(model, dict(cfg), checkpoint_dir=tmp_path)
    ckpt = tmp_path / "checkpoint_00000000.pt"
    from hexo_rl.training.checkpoints import save_full_checkpoint
    save_full_checkpoint(
        trainer.model, trainer.optimizer, trainer.scaler,
        trainer.scheduler, trainer.step, trainer.config, ckpt,
    )
    return ckpt


def _args(ckpt: Path, tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(ckpt),
        checkpoint_dir=str(tmp_path),
        override_scheduler_horizon=False,
        allow_fresh_scheduler=False,
        iterations=None,
    )


def _resume(ckpt_baked: dict, combined_over: dict, declared: frozenset, tmp_path: Path):
    """Drive the LIVE resume path (init_trainer) and return (trainer, combined_config, cap_logs)."""
    ckpt = _make_and_save_ckpt(_ckpt_config(**ckpt_baked), tmp_path)
    combined = _ckpt_config(**combined_over)
    import torch
    log = structlog.get_logger()
    with structlog.testing.capture_logs() as cap_logs:
        trainer, _bs = init_trainer(
            _args(ckpt, tmp_path), combined, torch.device("cpu"),
            board_size=19, res_blocks=2, filters=32, log=log,
            declared_keys=declared,
        )
    return trainer, combined, cap_logs


# ── THE split-brain fix: F1-deferred loop-read knob is executed by combined_config ────
def test_f1_deferred_ply_cap_value_backpropped_to_combined_config(tmp_path):
    """ply_cap_value baked -0.8, base default -0.5, NOT declared → F1 defers to -0.8.
    The loop reads combined_config, so combined_config MUST carry -0.8 (not the base -0.5)."""
    trainer, combined, cap_logs = _resume(
        ckpt_baked={"ply_cap_value": -0.8},
        combined_over={"ply_cap_value": -0.5},
        declared=frozenset(),
        tmp_path=tmp_path,
    )
    # trainer.config already had the baked value (pre-fix F1) — the split-brain is combined_config.
    assert trainer.config["ply_cap_value"] == -0.8
    # THE FIX: the loop-read config now AGREES with the preserved/baked value.
    assert combined["ply_cap_value"] == -0.8, (
        f"split-brain: loop reads combined_config[ply_cap_value]={combined['ply_cap_value']!r} "
        "but F1 preserved -0.8 (baked); loop executes a value the WARN/emission does not describe"
    )
    # The WARN fired for ply_cap_value (F1 mechanics unchanged).
    warns = [
        e for e in cap_logs
        if e.get("event") == "resume_base_default_deferred_to_baked"
        and e.get("knob") == "ply_cap_value"
    ]
    assert len(warns) == 1
    assert warns[0]["checkpoint_baked"] == -0.8
    # All three surfaces now AGREE: WARN(baked=-0.8), trainer.config(-0.8), combined_config(-0.8).
    assert warns[0]["checkpoint_baked"] == trainer.config["ply_cap_value"] == combined["ply_cap_value"]


def test_f1_deferred_amp_dtype_backpropped_closes_autocast_split(tmp_path):
    """amp_dtype is the trainer⊥inference autocast split: Trainer reads trainer.config, InferenceServer
    reads combined_config. F1 baked bf16, base default fp16, NOT declared → BOTH must read bf16."""
    trainer, combined, _ = _resume(
        ckpt_baked={"amp_dtype": "bf16"},
        combined_over={"amp_dtype": "fp16"},
        declared=frozenset(),
        tmp_path=tmp_path,
    )
    assert trainer.config["amp_dtype"] == "bf16"          # Trainer's autocast (baked, F1)
    assert combined["amp_dtype"] == "bf16", (             # InferenceServer's autocast (the fix)
        f"autocast split: InferenceServer reads combined_config[amp_dtype]={combined['amp_dtype']!r} "
        "but the trainer autocasts bf16 — a numerical trainer⊥inference mismatch"
    )


# ── E0 / owned-key / non-regression guards ────────────────────────────────────
def test_declared_knob_still_wins_and_backpropped(tmp_path):
    """E0: a DECLARED knob wins over baked, and the DECLARED value lands in combined_config
    (the back-prop must not resurrect the baked value for a declared knob)."""
    trainer, combined, cap_logs = _resume(
        ckpt_baked={"ply_cap_value": -0.8},
        combined_over={"ply_cap_value": -0.3},
        declared=frozenset({"ply_cap_value"}),   # variant DECLARED it
        tmp_path=tmp_path,
    )
    assert trainer.config["ply_cap_value"] == -0.3, "declaration must win (E0)"
    assert combined["ply_cap_value"] == -0.3, "declared value must survive in combined_config"
    # a declared override must NOT emit the F1-suppress WARN.
    assert not [
        e for e in cap_logs if e.get("event") == "resume_base_default_deferred_to_baked"
        and e.get("knob") == "ply_cap_value"
    ]


def test_non_deferred_base_knob_not_touched(tmp_path):
    """A base knob the checkpoint did NOT bake → base default applies (NOT deferred), and the
    back-prop must not spuriously overwrite combined_config with a baked value that doesn't exist."""
    trainer, combined, _ = _resume(
        ckpt_baked={},                              # novel_base_knob NOT baked
        combined_over={"novel_base_knob": 0.42},
        declared=frozenset(),
        tmp_path=tmp_path,
    )
    assert trainer.config["novel_base_knob"] == 0.42
    assert combined["novel_base_knob"] == 0.42     # base default preserved, unchanged


def test_matched_value_no_split_and_no_spurious_change(tmp_path):
    """When base default == baked value, F1 does a silent no-op defer; combined_config keeps the
    (identical) value — no spurious back-prop delta, no WARN."""
    trainer, combined, cap_logs = _resume(
        ckpt_baked={"ply_cap_value": -0.5},
        combined_over={"ply_cap_value": -0.5},     # identical
        declared=frozenset(),
        tmp_path=tmp_path,
    )
    assert trainer.config["ply_cap_value"] == -0.5
    assert combined["ply_cap_value"] == -0.5
    assert not [
        e for e in cap_logs if e.get("event") == "resume_base_default_deferred_to_baked"
    ], "identical value must not emit the F1-suppress WARN"


def test_owned_key_not_resurrected_into_combined_config(tmp_path):
    """Owned-key boundary through the back-prop path: an owned key (lr — checkpoint-STATE-owned,
    excluded from config_overrides upstream) is NOT in f1_deferred_keys and must NOT be back-propped
    into combined_config even when the baked lr differs from the launch lr. The loop-read lr stays the
    launch value; back-prop must not resurrect the baked value for an owned key."""
    # baked lr 5e-4, combined (launch) lr 1e-3 — differ. lr is in RESUME_CHECKPOINT_OWNED_KEYS.
    trainer, combined, _ = _resume(
        ckpt_baked={"lr": 5e-4},
        combined_over={"lr": 1e-3},
        declared=frozenset(),
        tmp_path=tmp_path,
    )
    assert "lr" not in trainer.f1_deferred_keys, "lr is owned — must never enter the F1 deferred set"
    assert combined["lr"] == 1e-3, (
        f"owned key lr was resurrected into combined_config: {combined['lr']!r} (expected launch 1e-3)"
    )


def test_f1_deferred_keys_recorded_on_trainer(tmp_path):
    """The deferred-key set is discoverable programmatically (drives the programmatic back-prop —
    NOT a hardcoded list)."""
    trainer, _combined, _ = _resume(
        ckpt_baked={"ply_cap_value": -0.8, "amp_dtype": "bf16"},
        combined_over={"ply_cap_value": -0.5, "amp_dtype": "fp16"},
        declared=frozenset(),
        tmp_path=tmp_path,
    )
    deferred = getattr(trainer, "f1_deferred_keys", None)
    assert deferred is not None, "trainer must expose the F1-deferred key set"
    assert "ply_cap_value" in deferred
    assert "amp_dtype" in deferred
