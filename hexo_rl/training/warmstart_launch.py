"""T8 — E1 warm-start launch hook.

Wires the T6 value-head loader (`warmstart_value_head.load_value_head`) into the
real launch path (`orchestrator.init_trainer`, resume branch). It is the seam
that makes E1 launchable: after the 248k weights-only trunk loads into a
freshly-built (scalar OR dist65) net, this hook overwrites the net's VALUE HEAD
tensors from the pre-registered HEADSWAP head selected BY value_head_type.

ONE-KEY-DIFF invariant (INV-D1 / R5): the two E1 arm yamls differ ONLY in
`model.value_head_type`. The concrete head file is therefore NOT a yaml key — it
is SELECTED HERE, in code, from value_head_type:

    scalar  -> <head_dir>/arm_A_seed0/head_A_seed0.pt
    dist65  -> <head_dir>/arm_B_seed0/head_B_seed0.pt

Both arm yamls carry the SAME `warm_start.head_dir` (the HEADSWAP `ab` dir).

Default-OFF: `warm_start.enabled` defaults False, so every non-E1 launch is
byte-identical (the hook is a no-op that touches nothing).

RESUME GUARD: fires ONLY on a fresh/weights-only warm launch
(`trainer.loaded_from_full_checkpoint is False`). A mid-run FULL-checkpoint
resume already has the TRAINED value head restored from the checkpoint;
re-overwriting it from the HEADSWAP head would corrupt training — so the hook
skips (and WARNs) on a full resume even when warm_start is enabled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from hexo_rl.training.warmstart_value_head import load_value_head

# Arm-selection by value_head_type (relative to warm_start.head_dir).
# Pre-registered (mirrors warmstart_value_head.default_head_for_arm): scalar arm
# <- arm_A_seed0, dist arm <- arm_B_seed0.
HEAD_FILE_BY_TYPE: Dict[str, str] = {
    "scalar": "arm_A_seed0/head_A_seed0.pt",
    "dist65": "arm_B_seed0/head_B_seed0.pt",
}

_log = structlog.get_logger("hexo_rl.training.warmstart_launch")


def resolve_warmstart_head_file(head_dir: str, value_head_type: str) -> str:
    """Resolve the HEADSWAP head `.pt` for ``value_head_type`` under ``head_dir``.

    Raises:
        ValueError:        unknown value_head_type.
        FileNotFoundError: the resolved head file does not exist (a misconfigured
                           head_dir must fail LOUDLY at launch, not silently seed
                           nothing).
    """
    rel = HEAD_FILE_BY_TYPE.get(value_head_type)
    if rel is None:
        raise ValueError(
            f"value_head_type={value_head_type!r} has no warm-start head mapping "
            f"(known: {sorted(HEAD_FILE_BY_TYPE)})."
        )
    head_file = str(Path(head_dir) / rel)
    if not Path(head_file).exists():
        raise FileNotFoundError(
            f"warm-start head file not found: {head_file} "
            f"(head_dir={head_dir!r}, value_head_type={value_head_type!r}). "
            "Check warm_start.head_dir points at the HEADSWAP `ab` dir."
        )
    return head_file


def maybe_warmstart_value_head(
    trainer: Any,
    combined_config: Dict[str, Any],
    log: Optional[Any] = None,
) -> bool:
    """Seed ``trainer.model``'s value head from the pre-registered HEADSWAP head,
    IF warm_start is enabled AND this is a weights-only warm launch.

    Returns True iff the value head was actually seeded (a
    ``warmstart_value_head_loaded`` event is emitted on success). Returns False —
    a byte-identical no-op — when warm_start is disabled/absent, or when this is a
    full-checkpoint resume (RESUME GUARD).

    Args:
        trainer:          the loaded Trainer (needs ``.model`` and
                          ``.loaded_from_full_checkpoint``).
        combined_config:  the flattened launch config (reads ``warm_start`` +
                          ``value_head_type``).
        log:              optional structlog logger; falls back to the module
                          logger.
    """
    logger = log if log is not None else _log

    ws_cfg = combined_config.get("warm_start") or {}
    if not isinstance(ws_cfg, dict) or not ws_cfg.get("enabled", False):
        return False

    head_dir = ws_cfg.get("head_dir")
    if not head_dir:
        raise ValueError(
            "warm_start.enabled is true but warm_start.head_dir is unset — "
            "cannot resolve the HEADSWAP head to seed the value head."
        )

    # RESUME GUARD — a full-checkpoint resume already restored the TRAINED value
    # head; re-seeding from HEADSWAP would corrupt it. Skip + WARN (visible, not
    # silent) so an accidental warm_start-on resume is diagnosable.
    loaded_from_full = getattr(trainer, "loaded_from_full_checkpoint", None)
    if loaded_from_full is None:
        # Fresh (non-resume) run: the trunk was never loaded, so there is no
        # weights-only warm-start to seed onto. Skip loudly.
        logger.warning(
            "warmstart_value_head_skipped",
            reason="no_checkpoint_loaded",
            msg=(
                "warm_start.enabled is true but no checkpoint was loaded "
                "(fresh run) — nothing to warm-start; skipping."
            ),
        )
        return False
    if loaded_from_full:
        logger.warning(
            "warmstart_value_head_skipped",
            reason="full_checkpoint_resume",
            msg=(
                "warm_start.enabled is true but this is a FULL-checkpoint "
                "resume — the trained value head is already restored; re-seeding "
                "from HEADSWAP would corrupt it. Skipping (this is expected on a "
                "mid-run E1 resume)."
            ),
        )
        return False

    value_head_type = combined_config.get("value_head_type", "scalar")
    head_file = resolve_warmstart_head_file(head_dir, value_head_type)

    load_value_head(trainer.model, head_file, value_head_type)

    arm = "A" if value_head_type == "scalar" else "B"
    logger.info(
        "warmstart_value_head_loaded",
        arm=arm,
        head_file=head_file,
        head_type=value_head_type,
        head_dir=str(head_dir),
    )
    return True


def assert_dist65_bins_seeded(
    trainer: Any,
    combined_config: Dict[str, Any],
    warmstart_fired: bool,
) -> None:
    """Belt-and-suspenders guard: raise if a dist65 net's value_fc2_bins are
    untrained/random (neither loaded from the checkpoint NOR seeded by the
    warm-start hook).

    Called by the orchestrator immediately after ``maybe_warmstart_value_head``.
    No-op for scalar nets (any config), dist65 + warm-start ON (E1 path), or
    a genuine dist65 checkpoint resume (bins were in the checkpoint).

    Raises:
        RuntimeError: dist65 + scalar trunk (no bins in ckpt) + no warm-start.
    """
    value_head_type = combined_config.get("value_head_type", "scalar")
    if value_head_type != "dist65":
        return  # scalar arm — always safe
    ckpt_had_bins = getattr(trainer, "ckpt_had_value_fc2_bins", True)
    if ckpt_had_bins:
        return  # checkpoint was a dist65 ckpt — bins were loaded
    if warmstart_fired:
        return  # warm-start seeded the bins — safe
    raise RuntimeError(
        "dist65 value head has untrained/random value_fc2_bins and no "
        "warm-start seeded them — refusing to train. The loaded checkpoint "
        "is a SCALAR trunk (no value_fc2_bins.*) and warm_start.enabled is "
        "false (or warm-start was skipped). Fix: set warm_start.enabled=true "
        "and point warm_start.head_dir at the HEADSWAP `ab` dir, OR resume "
        "from a full dist65 checkpoint that already has trained bins."
    )


__all__ = [
    "HEAD_FILE_BY_TYPE",
    "assert_dist65_bins_seeded",
    "maybe_warmstart_value_head",
    "resolve_warmstart_head_file",
]
