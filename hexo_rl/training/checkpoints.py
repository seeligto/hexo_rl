"""Shared checkpoint save/load utilities for Trainer and BootstrapTrainer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import structlog

log = structlog.get_logger()


def get_base_model(model: nn.Module) -> nn.Module:
    """Unwrap torch.compile / DataParallel wrapper to get the raw model."""
    return getattr(model, "_orig_mod", model)


def save_full_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: Optional[Any],
    step: int,
    config: Dict[str, Any],
    path: Path,
) -> None:
    """Save a full training checkpoint (model + optimizer + meta)."""
    base_model = get_base_model(model)
    torch.save(
        {
            "step": step,
            "model_state": base_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
        },
        path,
    )


def save_inference_weights(model: nn.Module, path: Path) -> None:
    """Save model weights only (no optimizer state)."""
    base_model = get_base_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base_model.state_dict(), path)


def prune_checkpoints(
    checkpoint_dir: Path,
    max_kept: Optional[int],
    pattern_str: str = r"^checkpoint_(\d+)\.pt$",
    preserve_predicate: Optional[Callable[[int], bool]] = None,
    keep_all: bool = False,
    anchor_every_steps: Optional[int] = None,
) -> None:
    """Delete old checkpoint files beyond max_kept.

    Keeps the N most recent *rolling* checkpoints by step number.
    Steps for which preserve_predicate(step) is True are never deleted,
    regardless of max_kept — use this to permanently retain eval checkpoints.

    Args:
        keep_all:          If True, skip all pruning (KEEP_ALL=true mode for debug runs).
        anchor_every_steps: Also preserve steps that are exact multiples of this value.
                           Additive with preserve_predicate. None = disabled.
    """
    if keep_all or max_kept is None:
        return

    pattern = re.compile(pattern_str)
    candidates: List[tuple] = []
    for p in checkpoint_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    candidates.sort(key=lambda x: x[0])

    def _is_preserved(step: int) -> bool:
        if preserve_predicate is not None and preserve_predicate(step):
            return True
        if anchor_every_steps and step > 0 and step % anchor_every_steps == 0:
            return True
        return False

    rolling = [
        (step, p) for step, p in candidates
        if not _is_preserved(step)
    ]
    to_delete = rolling[:-max_kept] if len(rolling) > max_kept else []
    for _, p in to_delete:
        try:
            p.unlink()
            log.info("checkpoint_pruned", path=str(p))
        except OSError as exc:
            log.warning("checkpoint_prune_failed", path=str(p), error=str(exc))


_BN_KEY_PATTERNS = (
    ".input_bn.", ".bn1.", ".bn2.", ".policy_bn.", ".opp_reply_bn.",
)


def normalize_model_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Normalize model state dict keys across save variants.

    Handles _orig_mod., module. prefixes and tower/trunk.tower aliasing.

    Raises RuntimeError on pre-GroupNorm checkpoints (keys containing BatchNorm
    layer names). These checkpoints are incompatible with the current model —
    start a fresh training run rather than silently loading random trunk weights.
    """
    if not state_dict:
        return state_dict

    # Detect pre-GN checkpoints before any remapping so the error fires on raw
    # keys too (e.g. _orig_mod.trunk.input_bn.weight).
    bn_keys = [k for k in state_dict if any(pat in k for pat in _BN_KEY_PATTERNS)]
    if bn_keys:
        examples = ", ".join(bn_keys[:3])
        raise RuntimeError(
            "Checkpoint contains BatchNorm keys incompatible with the current "
            "GroupNorm model (BN→GN migration, 2026-04-16). "
            f"Example keys: {examples}. "
            "Start a fresh training run — do not attempt to resume from a pre-GN checkpoint."
        )

    prefixes = ("_orig_mod.", "module.")
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        norm_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if norm_key.startswith(prefix):
                    norm_key = norm_key[len(prefix):]
                    changed = True
        normalized[norm_key] = value

    has_tower = any(k.startswith("tower.") for k in normalized.keys())
    has_trunk_tower = any(k.startswith("trunk.tower.") for k in normalized.keys())
    if has_tower and not has_trunk_tower:
        for key, value in list(normalized.items()):
            if key.startswith("tower."):
                normalized.setdefault(f"trunk.{key}", value)
    elif has_trunk_tower and not has_tower:
        for key, value in list(normalized.items()):
            if key.startswith("trunk.tower."):
                normalized.setdefault(key[len("trunk."):], value)

    return normalized
