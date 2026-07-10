"""CONFRES Phase-3 batch 6b — F1(A) + B3: base-inherited defers to ckpt-baked on resume.

The ONE intentional behavior change of the CONFRES refactor (operator-approved F1=(A),
2026-07-10). On a FULL-checkpoint resume the effective precedence for a variant-wins knob
(NOT in ``RESUME_CHECKPOINT_OWNED_KEYS``) becomes::

    cli → variant/--config declaration → checkpoint-baked config → base default

- A knob DECLARED in a ``--config``/variant layer OVERRIDES the ckpt-baked value (E0 fix; preserve).
- A knob only INHERITED from a base layer now DEFERS to the ckpt-baked value when the checkpoint
  baked it (F1 fix; was: base default silently clobbered the baked value). If the checkpoint did
  NOT bake the key → base default applies (unchanged).
- A base-inherited default that is SUPPRESSED because it differs from the baked value it defers to
  emits a loud, structured WARN (``resume_base_default_deferred_to_baked``).
- B3 null: an EXPLICIT ``key: null`` in a declaration layer is a DECLARATION (overrides to null);
  a base-inherited ``null`` is a default (null-skip; does not clobber a baked value).

These tests drive the impl (TDD — written first, watched fail). The GOLDEN resume-arm (design §9
regime c) pins the operator's exact §S178 example: resume-without-variant of a checkpoint baked
with a variant-tuned knob → the tuned knob is PRESERVED, not reverted to base.
"""
from pathlib import Path

import structlog.testing

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer
from hexo_rl.training.orchestrator import (
    build_resume_config_overrides,
    compute_declared_keys,
)


# ── Minimal checkpoint config the resume machinery accepts ───────────────────
def _base_ckpt_config(**over) -> dict:
    cfg = {
        "board_size":   19,
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


def _make_ckpt_trainer(cfg: dict, tmp_path: Path) -> Trainer:
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    return Trainer(model, dict(cfg), checkpoint_dir=tmp_path)


def _save(trainer: Trainer, tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoint_00000000.pt"
    from hexo_rl.training.checkpoints import save_full_checkpoint
    save_full_checkpoint(
        trainer.model, trainer.optimizer, trainer.scaler,
        trainer.scheduler, trainer.step, trainer.config, ckpt,
    )
    return ckpt


# ── compute_declared_keys: top-level keys present in config/variant layers ────
def test_compute_declared_keys_only_config_and_variant_layers():
    layers = [
        {"label": "configs/model.yaml", "kind": "base",
         "raw": {"encoding": "v6", "bot_batch_share": 0.0, "seed": 42}},
        {"label": "configs/override.yaml", "kind": "config",
         "raw": {"grad_clip": 0.5}},
        {"label": "configs/variants/x.yaml", "kind": "variant",
         "raw": {"completed_q_values": True, "ply_cap_value": None}},
    ]
    declared = compute_declared_keys(layers)
    # config + variant declared keys present; explicit-null is STILL a declaration.
    assert "grad_clip" in declared
    assert "completed_q_values" in declared
    assert "ply_cap_value" in declared          # explicit null is a declaration (B3)
    # base-only keys are NOT declarations.
    assert "encoding" not in declared
    assert "bot_batch_share" not in declared
    assert "seed" not in declared


def test_compute_declared_keys_none_layers_empty():
    assert compute_declared_keys(None) == frozenset()
    assert compute_declared_keys([]) == frozenset()


# ── B3 null semantics in build_resume_config_overrides ───────────────────────
def test_b3_explicit_null_declaration_travels():
    """A variant-declared explicit null OVERRIDES (must travel into overrides, to null)."""
    combined = {"some_key": None, "grad_clip": 0.5}
    declared = frozenset({"some_key"})
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    # declared explicit null present in overrides (so config.update sets it to null → override)
    assert "some_key" in overrides
    assert overrides["some_key"] is None


def test_b3_base_inherited_null_skipped():
    """A base-inherited null is a default: null-skip so it can't clobber a baked value."""
    combined = {"some_key": None, "grad_clip": 0.5}
    declared = frozenset()  # some_key inherited from base, not declared
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    assert "some_key" not in overrides


# ── F1(A): base-inherited defers to baked (end-to-end via load_checkpoint) ────
def test_f1_base_inherited_defers_to_baked(tmp_path):
    """The operator's exact F1 example: resume-WITHOUT-variant, base default
    bot_batch_share=0.0, checkpoint baked 0.15, no declaration →
    effective config bot_batch_share == 0.15 (baked preserved) + a WARN."""
    ckpt_cfg = _base_ckpt_config(bot_batch_share=0.15)
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    assert trainer.config["bot_batch_share"] == 0.15
    ckpt = _save(trainer, tmp_path)

    # resume-WITHOUT-variant: combined_config carries the BASE default 0.0, nothing declared it.
    combined = _base_ckpt_config(bot_batch_share=0.0)
    declared = frozenset()  # no config/variant layer declared bot_batch_share
    overrides = build_resume_config_overrides(combined, declared_keys=declared)

    with structlog.testing.capture_logs() as cap_logs:
        resumed = Trainer.load_checkpoint(
            ckpt, checkpoint_dir=tmp_path,
            fallback_config=combined, config_overrides=overrides,
            declared_keys=declared,
        )
    assert resumed.config["bot_batch_share"] == 0.15, (
        f"baked value not preserved: {resumed.config['bot_batch_share']!r}"
    )
    # loud, structured WARN naming the suppressed base default + the baked value it defers to.
    warns = [
        e for e in cap_logs
        if e.get("event") == "resume_base_default_deferred_to_baked"
        and e.get("log_level") == "warning"
    ]
    assert len(warns) == 1, f"expected exactly one F1-suppress WARN, got {cap_logs}"
    w = warns[0]
    assert w["knob"] == "bot_batch_share"
    assert w["base_default"] == 0.0
    assert w["checkpoint_baked"] == 0.15


def test_f1_no_warn_when_declared(tmp_path):
    """A declared override that differs from baked must NOT emit the F1-suppress WARN."""
    ckpt_cfg = _base_ckpt_config(bot_batch_share=0.15)
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    ckpt = _save(trainer, tmp_path)

    combined = _base_ckpt_config(bot_batch_share=0.30)
    declared = frozenset({"bot_batch_share"})
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    with structlog.testing.capture_logs() as cap_logs:
        Trainer.load_checkpoint(
            ckpt, checkpoint_dir=tmp_path,
            fallback_config=combined, config_overrides=overrides,
            declared_keys=declared,
        )
    assert not [
        e for e in cap_logs if e.get("event") == "resume_base_default_deferred_to_baked"
    ], "declared override must not emit an F1-suppress WARN"


def test_f1_declared_overrides_baked(tmp_path):
    """A variant DECLARING bot_batch_share: 0.30 → effective 0.30 (declaration wins), no suppress."""
    ckpt_cfg = _base_ckpt_config(bot_batch_share=0.15)
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    ckpt = _save(trainer, tmp_path)

    combined = _base_ckpt_config(bot_batch_share=0.30)
    declared = frozenset({"bot_batch_share"})  # variant DECLARED it
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=combined, config_overrides=overrides,
        declared_keys=declared,
    )
    assert resumed.config["bot_batch_share"] == 0.30, (
        f"declaration did not win: {resumed.config['bot_batch_share']!r}"
    )


def test_f1_base_only_not_baked_applies(tmp_path):
    """A base key ABSENT from the ckpt-baked config → base default applies (unchanged)."""
    ckpt_cfg = _base_ckpt_config()          # no 'novel_base_knob' baked
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    ckpt = _save(trainer, tmp_path)
    assert "novel_base_knob" not in trainer.config

    combined = _base_ckpt_config(novel_base_knob=0.42)
    declared = frozenset()  # inherited from base
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=combined, config_overrides=overrides,
        declared_keys=declared,
    )
    # no baked value to preserve → base default applies.
    assert resumed.config["novel_base_knob"] == 0.42


def test_b3_explicit_null_declaration_overrides_baked(tmp_path):
    """A variant-declared explicit null overrides the baked value TO null (declaration wins)."""
    ckpt_cfg = _base_ckpt_config(some_path="/baked/path")
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    ckpt = _save(trainer, tmp_path)

    combined = _base_ckpt_config(some_path=None)
    declared = frozenset({"some_path"})  # variant DECLARED some_path: null
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=combined, config_overrides=overrides,
        declared_keys=declared,
    )
    assert resumed.config["some_path"] is None, (
        f"explicit-null declaration did not override baked: {resumed.config['some_path']!r}"
    )


def test_b3_base_inherited_null_preserves_baked(tmp_path):
    """A base-inherited null does NOT clobber a baked value (null-skip applies)."""
    ckpt_cfg = _base_ckpt_config(some_path="/baked/path")
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    ckpt = _save(trainer, tmp_path)

    combined = _base_ckpt_config(some_path=None)  # base default null, NOT declared
    declared = frozenset()
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=combined, config_overrides=overrides,
        declared_keys=declared,
    )
    assert resumed.config["some_path"] == "/baked/path", (
        f"base-inherited null clobbered baked path: {resumed.config['some_path']!r}"
    )


# ── GOLDEN resume-arm (design §9 regime c) ────────────────────────────────────
def test_golden_resume_without_variant_preserves_tuned_knob(tmp_path):
    """design §9 regime c — resume-from-full-checkpoint WITHOUT the variant, of a checkpoint
    baked with a variant-tuned knob → the tuned knob is PRESERVED (not reverted to base).

    Mirrors the operator's §S178 example (a follow-up resume that forgets to re-pass the variant):
    ckpt baked completed_q_values=True + ply_cap_value=-0.8; base defaults are False / -0.5.
    """
    ckpt_cfg = _base_ckpt_config(
        completed_q_values=True,
        ply_cap_value=-0.8,
        aux_chain_weight=0.10,
    )
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    ckpt = _save(trainer, tmp_path)

    # resume-WITHOUT-variant → combined carries the BASE defaults, declared_keys empty.
    combined = _base_ckpt_config(
        completed_q_values=False,
        ply_cap_value=-0.5,
        aux_chain_weight=0.0,
    )
    declared = frozenset()
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=combined, config_overrides=overrides,
        declared_keys=declared,
    )
    # tuned knobs PRESERVED from the checkpoint (F1(A)).
    assert resumed.config["completed_q_values"] is True
    assert resumed.config["ply_cap_value"] == -0.8
    assert resumed.config["aux_chain_weight"] == 0.10


def test_f1_horizon_gate_rehorizons_with_declared_keys(tmp_path):
    """The --override-scheduler-horizon gate survives the F1 apply on the LIVE launch path.

    On the real path init_trainer threads a NON-None declared_keys, so the F1 helper is active. The
    horizon keys (total_steps/scheduler_t_max) enter config_overrides ONLY under the flag; even when
    the operator did NOT declare total_steps in a variant layer (it came via --iterations / base),
    the re-horizon must still fire — deferring total_steps to the baked value would silently ignore
    the operator's explicit --override-scheduler-horizon intent.
    """
    ckpt_cfg = _base_ckpt_config(total_steps=1_000_000)
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    assert trainer.scheduler.T_max == 1_000_000
    ckpt = _save(trainer, tmp_path)

    combined = _base_ckpt_config(total_steps=2_000_000)
    declared = frozenset()  # total_steps NOT declared in a variant layer (came via --iterations)
    overrides = build_resume_config_overrides(
        combined, override_scheduler_horizon=True, declared_keys=declared,
    )
    assert overrides.get("total_steps") == 2_000_000
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=combined, config_overrides=overrides,
        declared_keys=declared,
    )
    assert resumed.scheduler.T_max == 2_000_000, (
        f"F1 defer swallowed the --override-scheduler-horizon re-horizon: {resumed.scheduler.T_max}"
    )


def test_f1_horizon_gate_off_preserves_baked_T_max(tmp_path):
    """Without the flag (horizon keys absent from overrides) the baked T_max survives under F1."""
    ckpt_cfg = _base_ckpt_config(total_steps=1_000_000)
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    ckpt = _save(trainer, tmp_path)

    combined = _base_ckpt_config(total_steps=2_000_000)
    declared = frozenset()
    overrides = build_resume_config_overrides(
        combined, override_scheduler_horizon=False, declared_keys=declared,
    )
    assert "total_steps" not in overrides  # owned key, not re-added without the flag
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=combined, config_overrides=overrides,
        declared_keys=declared,
    )
    assert resumed.scheduler.T_max == 1_000_000


def test_f1_torch_compile_handling_preserved(tmp_path):
    """torch_compile/torch_compile_mode handling is preserved verbatim (not F1-deferred).

    build_resume_config_overrides ALWAYS re-adds torch_compile (so --no-compile / a variant flip is
    honoured on resume). Deferring it to the baked value would swallow that intent. The prompt pins
    'preserve torch_compile handling verbatim'.
    """
    ckpt_cfg = _base_ckpt_config(torch_compile=True)   # ckpt baked compile ON
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    ckpt = _save(trainer, tmp_path)

    # resume with --no-compile → combined torch_compile False, NOT declared in a variant layer.
    combined = _base_ckpt_config(torch_compile=False)
    declared = frozenset()
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    assert overrides["torch_compile"] is False
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=combined, config_overrides=overrides,
        declared_keys=declared,
    )
    assert resumed.config["torch_compile"] is False, (
        "torch_compile handling not preserved — F1 deferred it to the baked True"
    )


def test_owned_keys_still_owned_under_f1(tmp_path):
    """F1 does not touch the EXCLUDE-set: lr/arch stay checkpoint-owned even when declared."""
    ckpt_cfg = _base_ckpt_config(lr=1e-3, res_blocks=2, filters=32)
    trainer = _make_ckpt_trainer(ckpt_cfg, tmp_path)
    ckpt = _save(trainer, tmp_path)

    combined = _base_ckpt_config(lr=5e-4, res_blocks=4, filters=64)
    declared = frozenset({"lr", "res_blocks", "filters"})  # even if declared, owned wins
    overrides = build_resume_config_overrides(combined, declared_keys=declared)
    # owned keys never enter overrides.
    assert "lr" not in overrides
    assert "res_blocks" not in overrides
    assert "filters" not in overrides
    resumed = Trainer.load_checkpoint(
        ckpt, checkpoint_dir=tmp_path,
        fallback_config=combined, config_overrides=overrides,
        declared_keys=declared,
    )
    assert resumed.config["res_blocks"] == 2
    assert resumed.config["filters"] == 32
