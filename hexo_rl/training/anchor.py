"""Best-model anchor management — atomic save, resilient load, quarantine.

Owns ``best_model.pt`` lifecycle: torch.save round-trip verify + .bak rotation
on save; best → .bak → bootstrap_*.pt fallback chain on load; corrupt-anchor
quarantine with timestamp suffix.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from hexo_rl.training.trainer import Trainer

log = structlog.get_logger(__name__)


# Bootstrap candidates tried (in order) when no usable best_model.pt exists.
# Fresh runs anchor against the trained bootstrap, not a random fresh-init
# copy of trainer.model.
_BOOTSTRAP_ANCHOR_CANDIDATES: tuple[str, ...] = (
    "checkpoints/bootstrap_model_v7full.pt",
    "checkpoints/bootstrap_model.pt",
)


def save_best_model_atomic(model: torch.nn.Module, path: Path) -> None:
    """Save ``state_dict()`` to ``path`` atomically with one-revision backup.

    Sequence:
      1. write to ``path.tmp``,
      2. verify the tmp file actually loads (catches partial writes),
      3. rotate any existing ``path`` to ``path.bak`` (clobbers an older bak),
      4. rename ``path.tmp`` → ``path``.

    A SIGKILL between (3) and (4) leaves ``.bak`` as the recovery copy;
    ``load_best_model_resilient`` knows to fall through to it.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    bak = path.with_suffix(path.suffix + ".bak")
    sd = model.state_dict()
    torch.save(sd, tmp)
    # Round-trip verify — torch.save is not atomic on some filesystems and
    # a process kill mid-write produces exactly the truncated zip we are
    # trying to defend against.
    torch.load(tmp, map_location="cpu", weights_only=True)
    if path.exists():
        path.replace(bak)
    tmp.replace(path)


def _quarantine_corrupt(path: Path) -> Path:
    """Move a corrupt anchor aside with a unique suffix so it isn't overwritten
    by the next write. Returns the destination path for logging."""
    ts = time.strftime("%Y%m%dT%H%M%S")
    dest = path.with_suffix(path.suffix + f".corrupt-{ts}")
    path.replace(dest)
    return dest


def _try_load_anchor(
    candidate: Path,
    *,
    checkpoint_dir: str,
    device: torch.device,
    fallback_config: dict[str, Any],
) -> Optional[Trainer]:
    """Attempt to load ``candidate`` as an anchor checkpoint.

    Returns the loaded Trainer on success, None on any failure (corrupt zip,
    arch mismatch in the load path, unreadable file). Failures are logged
    but not raised — the caller decides what to fall back to.
    """
    if not candidate.exists():
        return None
    try:
        return Trainer.load_checkpoint(
            candidate,
            checkpoint_dir=checkpoint_dir,
            device=device,
            fallback_config=fallback_config,
            config_overrides={"input_channels": None, "in_channels": None},
        )
    except Exception as exc:
        log.warning(
            "anchor_load_failed",
            path=str(candidate),
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None


def load_best_model_resilient(
    best_model_path: Path,
    *,
    checkpoint_dir: str,
    device: torch.device,
    config: dict[str, Any],
) -> Optional[Trainer]:
    """Try best_model.pt → its .bak → bootstrap candidates. None if all fail.

    On corruption of ``best_model.pt`` the file is quarantined and the next
    candidate is tried; the eventual successful candidate is then promoted
    in-place to ``best_model.pt`` by the caller (atomic save).
    """
    fallback_cfg = {k: v for k, v in config.items()
                    if k not in ("in_channels", "input_channels")}

    # 1. Live anchor.
    if best_model_path.exists():
        ref = _try_load_anchor(
            best_model_path,
            checkpoint_dir=checkpoint_dir,
            device=device,
            fallback_config=fallback_cfg,
        )
        if ref is not None:
            return ref
        quarantined = _quarantine_corrupt(best_model_path)
        log.warning(
            "anchor_quarantined",
            original=str(best_model_path),
            quarantined=str(quarantined),
            msg="best_model.pt was unreadable — moved aside, falling through to backup/bootstrap",
        )

    # 2. One-revision backup written by the previous atomic save.
    bak = best_model_path.with_suffix(best_model_path.suffix + ".bak")
    if bak.exists():
        ref = _try_load_anchor(
            bak,
            checkpoint_dir=checkpoint_dir,
            device=device,
            fallback_config=fallback_cfg,
        )
        if ref is not None:
            log.info("anchor_recovered_from_bak", path=str(bak))
            return ref

    # 3. Repo-level bootstrap candidates.
    for rel in _BOOTSTRAP_ANCHOR_CANDIDATES:
        cand = Path(rel)
        ref = _try_load_anchor(
            cand,
            checkpoint_dir=checkpoint_dir,
            device=device,
            fallback_config=fallback_cfg,
        )
        if ref is not None:
            log.info("anchor_loaded_from_bootstrap", path=str(cand))
            return ref

    return None
