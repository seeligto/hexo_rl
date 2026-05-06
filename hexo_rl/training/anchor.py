"""Best-model anchor management — atomic save, resilient load, quarantine.

Owns ``best_model.pt`` lifecycle: torch.save round-trip verify + .bak rotation
on save; best → .bak → bootstrap_*.pt fallback chain on load; corrupt-anchor
quarantine with timestamp suffix.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from hexo_rl.model.network import HexTacToeNet
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


@dataclass
class AnchorState:
    """Resolved best-model anchor + provenance.

    ``best_model`` is None only when no eval pipeline is configured (the
    caller supplies eval_pipeline=None) — this matches the pre-refactor
    invariant where best_model stayed None outside the eval branch.
    ``best_model_step`` is None when no anchor step was recoverable from
    the loaded checkpoint.
    """

    best_model: Optional[HexTacToeNet]
    best_model_step: Optional[int]
    best_model_path: Path


def resolve_anchor(
    *,
    eval_pipeline: Any,                      # EvalPipeline | None
    eval_ext_config: dict[str, Any],
    inf_model: torch.nn.Module,              # mutated when arch matches
    trainer: Trainer,
    args: Any,                               # argparse.Namespace
    config: dict[str, Any],
    device: torch.device,
    board_size: int,
    res_blocks: int,
    filters: int,
    in_channels: int,
    input_channels: Any,
    se_reduction_ratio: int,
) -> AnchorState:
    """Resolve the best-model anchor and sync ``inf_model`` to it.

    Steps:
      1. Resolve ``best_model_path`` from eval.yaml gating config.
      2. If eval_pipeline is None → return AnchorState(None, None, path).
      3. Try resilient load (best.pt → .bak → bootstrap candidates).
      4. On success: unwrap torch.compile, persist to live path if recovered
         from a fallback, sync inf_model when in_channels match (preserves
         the input_channel_index buffer when present), warn on
         trainer.step ≠ best_model_step (M2 invariant).
      5. On total failure: fresh-init from trainer.model, save atomically.

    Mutates ``inf_model`` via ``load_state_dict`` when the architecture
    matches the loaded anchor. Sweep variants intentionally leave inf_model
    on trainer.model weights (arch-mismatch logged, no sync).
    """
    best_model_path = Path(
        eval_ext_config.get("eval_pipeline", {}).get("gating", {}).get(
            "best_model_path", "checkpoints/best_model.pt"
        )
    )
    best_model: HexTacToeNet | None = None
    best_model_step: int | None = None
    if eval_pipeline is None:
        return AnchorState(None, None, best_model_path)

    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    # Resilient anchor load: tries best_model.pt → its .bak →
    # _BOOTSTRAP_ANCHOR_CANDIDATES → fresh init from trainer.model.
    # A corrupt best_model.pt is quarantined with a timestamp suffix
    # instead of being silently discarded.
    best_ref = load_best_model_resilient(
        best_model_path,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        config=config,
    )
    if best_ref is not None:
        # Unwrap torch.compile — best_model's state_dict() is consumed
        # at multiple load_state_dict call sites below; leaving the
        # OptimizedModule wrapper would inject `_orig_mod.*` prefixes
        # into every subsequent state_dict() call.
        best_model = getattr(best_ref.model, "_orig_mod", best_ref.model)
        best_model.eval()
        best_model_step = best_ref.step
        # If best_model.pt was missing/corrupt and we recovered from a
        # bootstrap or .bak, persist the chosen anchor as the live
        # best_model.pt so subsequent runs find it directly.
        if not best_model_path.exists():
            save_best_model_atomic(best_model, best_model_path)
            log.info("anchor_persisted_from_fallback", path=str(best_model_path))
        # Graduation gate: self-play consumes anchor weights, not trainer.model.
        # Sync inf_model to the loaded anchor before workers start — but only
        # when architectures match. Sweep variants train a reduced-channel model
        # against an 18-channel anchor; syncing architectures is impossible and
        # wrong (sweep inf_model should start from trainer.model, not the anchor).
        _inf_base = getattr(inf_model, "_orig_mod", inf_model)
        if _inf_base.in_channels == best_model.in_channels:
            _best_sd = best_model.state_dict()
            # Anchor is always loaded without input_channels (see _try_load_anchor
            # config_overrides). If _inf_base was built with input_channels, inject
            # its own buffer so load_state_dict sees a consistent state_dict.
            _inf_idx = getattr(_inf_base, "input_channel_index", None)
            if "input_channel_index" not in _best_sd and _inf_idx is not None:
                _best_sd = dict(_best_sd)
                _best_sd["input_channel_index"] = _inf_idx.detach().clone()
            _inf_base.load_state_dict(_best_sd)
        else:
            log.info(
                "inf_model_anchor_arch_mismatch_skip_sync",
                inf_in_channels=_inf_base.in_channels,
                anchor_in_channels=best_model.in_channels,
                msg="inf_model starts from trainer.model (sweep variant)",
            )
        log.info("best_model_loaded", path=str(best_model_path), step=best_model_step)
        # M2: warn if resumed trainer.model and loaded anchor diverge on step.
        # Either side may legitimately be ahead (anchor rollback, or training
        # continued past last promotion) but a silent mismatch can produce a
        # trivially-promoted first eval that wipes a hand-picked anchor.
        if best_model_step is not None and trainer.step != best_model_step:
            log.warning(
                "resume_anchor_step_mismatch",
                trainer_step=trainer.step,
                best_model_step=best_model_step,
                msg=(
                    "trainer.model and best_model.pt were loaded from different "
                    "training steps. First eval will compare the current trainer "
                    "weights against this anchor; confirm this is intended."
                ),
            )
    else:
        # No usable anchor anywhere — last-resort fresh init from trainer.model.
        # On a clean run this should rarely fire: the bootstrap candidates above
        # are the canonical first-run anchor. Reaching this branch means the box
        # has neither best_model.pt, its .bak, nor any bootstrap_*.pt — flag it.
        log.warning(
            "anchor_fresh_init_no_bootstrap",
            tried=list(_BOOTSTRAP_ANCHOR_CANDIDATES),
            msg="No anchor or bootstrap available — initialising best_model.pt from current trainer.model. "
                "Drop a bootstrap_model.pt (or one of _BOOTSTRAP_ANCHOR_CANDIDATES) into checkpoints/ to anchor wr_best meaningfully.",
        )
        base_model = getattr(trainer.model, "_orig_mod", trainer.model)
        best_model = HexTacToeNet(
            board_size=board_size, res_blocks=res_blocks, filters=filters,
            in_channels=in_channels, input_channels=input_channels,
            se_reduction_ratio=se_reduction_ratio,
        ).to(device)
        best_model.load_state_dict(base_model.state_dict())
        best_model.eval()
        save_best_model_atomic(best_model, best_model_path)
        best_model_step = trainer.step
        log.info("best_model_initialized", path=str(best_model_path), step=best_model_step)

    return AnchorState(best_model, best_model_step, best_model_path)
