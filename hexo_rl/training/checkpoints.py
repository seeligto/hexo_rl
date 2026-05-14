"""Shared checkpoint save/load utilities for Trainer and BootstrapTrainer."""

from __future__ import annotations

import datetime as _datetime
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import structlog

log = structlog.get_logger()

if TYPE_CHECKING:
    from hexo_rl.encoding import EncodingSpec
    from hexo_rl.model.network import HexTacToeNet


# ── State-dict inspection helpers ────────────────────────────────────────────
# §176 P79: lifted from Trainer staticmethods so viewer/model_loader.py and
# probe scripts can dedup against a single source. Trainer keeps its
# @staticmethod surface as a thin delegate (back-compat for our_model_bot
# and probe scripts that call Trainer._extract_model_state etc.).

_TOWER_RES_BLOCK_PATTERN = re.compile(r"^(?:trunk\.)?tower\.(\d+)\.")


def extract_model_state(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Extract the model state dict from common checkpoint payload layouts."""
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(ckpt)!r}")

    for key in ("model_state", "model_state_dict", "state_dict"):
        maybe_state = ckpt.get(key)
        if isinstance(maybe_state, dict):
            return maybe_state

    # Weights-only checkpoints are plain state_dict payloads.
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt

    raise ValueError("Unable to locate model state dict in checkpoint payload")


def infer_res_blocks_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Optional[int]:
    idxs = {
        int(match.group(1))
        for key in state_dict.keys()
        for match in [_TOWER_RES_BLOCK_PATTERN.search(key)]
        if match is not None
    }
    if not idxs:
        return None
    return max(idxs) + 1


def infer_model_hparams(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infer model hyperparameters directly from a checkpoint state_dict."""
    inferred: Dict[str, int] = {}

    conv_w = state_dict.get("trunk.input_conv.weight")
    if conv_w is not None and conv_w.ndim == 4:
        inferred["filters"] = int(conv_w.shape[0])
        inferred["in_channels"] = int(conv_w.shape[1])

    policy_fc_w = state_dict.get("policy_fc.weight")
    if policy_fc_w is not None and policy_fc_w.ndim == 2:
        two_spatial = int(policy_fc_w.shape[1])
        if two_spatial % 2 == 0:
            spatial = two_spatial // 2
            board_size = int(math.isqrt(spatial))
            if board_size * board_size == spatial:
                inferred["board_size"] = board_size

    res_blocks = infer_res_blocks_from_state_dict(state_dict)
    if res_blocks is not None:
        inferred["res_blocks"] = int(res_blocks)

    return inferred


# §172 A5.1 metadata schema version. Bump if a metadata key is renamed/removed.
CHECKPOINT_METADATA_SCHEMA_VERSION = 1


def get_base_model(model: nn.Module) -> nn.Module:
    """Unwrap torch.compile / DataParallel wrapper to get the raw model."""
    return getattr(model, "_orig_mod", model)


def _resolve_commit_sha() -> str:
    """`git rev-parse HEAD` — best-effort.

    Returns "unknown" outside a git checkout (e.g. wheel install, CI fixture
    sandbox). Never raises; metadata write must not be blocked by VCS state.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
            timeout=2.0,
        )
        return out.decode("ascii", errors="replace").strip() or "unknown"
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ):
        return "unknown"


def build_checkpoint_metadata(
    *,
    encoding_name: str,
    train_config_path: Optional[str | Path] = None,
    corpus_sha256: Optional[str] = None,
    model_architecture: str = "HexTacToeNet",
    model_variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct the §172 A2 §8 checkpoint metadata dict.

    `encoding_name` MANDATORY. Caller is responsible for resolving it from
    the active EncodingSpec (e.g. spec.name) or via
    `hexo_rl.encoding.resolve_from_config(self.config).name`.
    """
    if not encoding_name or not isinstance(encoding_name, str):
        raise ValueError(
            f"encoding_name must be a non-empty str; got {encoding_name!r}. "
            "Resolve via hexo_rl.encoding.resolve_from_config(cfg).name."
        )
    return {
        "encoding_name": encoding_name,
        "commit_sha": _resolve_commit_sha(),
        # Timezone-aware UTC; suffix 'Z' kept for ISO 8601 readability.
        "training_date": _datetime.datetime.now(_datetime.timezone.utc)
            .replace(tzinfo=None)
            .isoformat()
            + "Z",
        "train_config_path": str(train_config_path) if train_config_path is not None else None,
        "corpus_sha256": corpus_sha256,
        "model_architecture": model_architecture,
        "model_variant": model_variant,
        "schema_version": CHECKPOINT_METADATA_SCHEMA_VERSION,
    }


def save_full_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: Optional[Any],
    step: int,
    config: Dict[str, Any],
    path: Path,
    *,
    encoding_name: Optional[str] = None,
    train_config_path: Optional[str | Path] = None,
    corpus_sha256: Optional[str] = None,
    model_architecture: str = "HexTacToeNet",
    model_variant: Optional[str] = None,
) -> None:
    """Save a full training checkpoint (model + optimizer + meta).

    §172 A5.1: writes a top-level `metadata` dict per design §8. All new
    callers MUST pass `encoding_name`; legacy callers (no kwarg) get a
    DeprecationWarning and the metadata block is omitted — the resulting
    ckpt resolves via `compat.infer_encoding_from_state_dict` on load.
    """
    base_model = get_base_model(model)
    payload: Dict[str, Any] = {
        "step": step,
        "model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    if encoding_name is not None:
        payload["metadata"] = build_checkpoint_metadata(
            encoding_name=encoding_name,
            train_config_path=train_config_path,
            corpus_sha256=corpus_sha256,
            model_architecture=model_architecture,
            model_variant=model_variant,
        )
    else:
        # Legacy call site — metadata absent, load path will fall back to
        # shape inference. Logged so operator can grep for unstamped saves.
        log.warning(
            "checkpoint_save_no_encoding_name",
            path=str(path),
            msg=(
                "save_full_checkpoint called without encoding_name; "
                "metadata block omitted. Update caller per §172 A5.1."
            ),
        )
    torch.save(payload, path)


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

    # Reject pre-P3 18-plane checkpoints. The P3 model has in_channels=8;
    # silently loading an 18-plane trunk would produce garbage.
    _conv_key = "trunk.input_conv.weight"
    _conv_w = state_dict.get(_conv_key)
    if _conv_w is None:
        # Try after prefix stripping
        for k, v in state_dict.items():
            if k.endswith("trunk.input_conv.weight"):
                _conv_w = v
                break
    if _conv_w is not None and _conv_w.dim() == 4 and _conv_w.shape[1] == 18:
        raise RuntimeError(
            "Checkpoint has 18-plane input conv (pre-P3 model, in_channels=18). "
            "Current model expects in_channels=8 (HEXB v6 wire format). "
            "Retrain from the 8-plane bootstrap — do not load a pre-P3 checkpoint."
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


def load_state_dict_strict(
    model: nn.Module, state_dict: Dict[str, torch.Tensor]
) -> None:
    """Load state_dict with explicit missing/unexpected key reporting.

    `normalize_model_state_dict_keys` adds tower/trunk.tower aliases; one side
    is always "unexpected" to the live model. Those are filtered. Anything
    else missing or unexpected raises — pre-§99 BN keys are rejected at
    normalize time (F-002), and silent drops at load time hide the same
    class of bug (B-002).

    §176 P47: lifted from Trainer._load_state_dict_strict so the public
    `load_inference_model` API and probe scripts share one implementation.
    Trainer._load_state_dict_strict remains as a thin delegate.
    """
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    model_key_set = set(model.state_dict().keys())
    benign_unexpected = []
    real_unexpected = []
    for k in unexpected_keys:
        if k.startswith("tower."):
            alias = f"trunk.{k}"
        elif k.startswith("trunk.tower."):
            alias = k[len("trunk."):]
        else:
            alias = None
        if alias is not None and alias in model_key_set:
            benign_unexpected.append(k)
        else:
            real_unexpected.append(k)

    if missing_keys or real_unexpected:
        log.error(
            "checkpoint_key_mismatch",
            missing_count=len(missing_keys),
            unexpected_count=len(real_unexpected),
            missing_examples=missing_keys[:5],
            unexpected_examples=real_unexpected[:5],
        )
        raise RuntimeError(
            f"Checkpoint load_state_dict mismatch: missing={len(missing_keys)} keys, "
            f"unexpected={len(real_unexpected)} keys (after filtering tower/trunk.tower aliases). "
            f"Missing examples: {missing_keys[:3]}. Unexpected examples: {real_unexpected[:3]}. "
            "If this is an intentional architecture change, retrain from bootstrap_model.pt."
        )


def load_inference_model(
    checkpoint_path: str | Path,
    config: Dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> Tuple["HexTacToeNet", "EncodingSpec", str]:
    """Load a checkpoint into an inference-ready HexTacToeNet.

    Public entry point for ckpt → (model, encoding_spec, label) used by
    bots, viewer, and probe scripts. Delegates encoding-aware construction
    (pool_type / gpool_bias / canvas_realness detection, v6 vs v8
    branching) to ``eval.checkpoint_loader.load_model_with_encoding``;
    ``config`` is reserved for future hparam fallbacks the eval loader
    does not need (the eval loader infers everything from the state dict
    today). The arg is kept in the signature so call sites that pass it
    (OurModelBot, future bot adapters) don't need to change when those
    fallbacks are added.

    Args:
        checkpoint_path: path to a .pt checkpoint.
        config:          optional config dict — **currently unused** (the
                         body discards it via ``del config``). Reserved
                         for future hparam fallbacks (``filters``,
                         ``res_blocks``, ``se_reduction_ratio``) when
                         state-dict inference is insufficient. The
                         state-dict shape is the **sole** source of arch
                         dims today — pre-§176-P47 callers (e.g. the old
                         ``our_model_bot.py`` path) that relied on
                         ``config['model']['filters']`` /
                         ``['res_blocks']`` / ``['se_reduction_ratio']``
                         overrides must ensure the state-dict carries
                         matching shapes; ``load_state_dict`` raises
                         loudly on any mismatch (see
                         ``load_state_dict_strict`` above).
        device:          target device. Defaults to `best_device()`.

    Returns:
        (model, EncodingSpec, label) — model is `.to(device).eval()`.
        `label` is the canonical encoding name (e.g. "v6", "v6w25", "v8").

    Security:
        Loads checkpoints with ``weights_only=False`` (inherited from
        ``load_model_with_encoding``'s eval-path default — the pre-§176-P47
        ``our_model_bot.py`` path used ``weights_only=True``, but every
        other eval-side loader has been ``weights_only=False`` since the
        registry landed, so this is documented consistency, not a
        regression). Only load checkpoints from trusted sources;
        ``weights_only=False`` will execute pickled Python during load.

    §176 P47.
    """
    # Local imports to keep checkpoints.py free of model/encoding cycles
    # at module-import time.
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    if device is None:
        from hexo_rl.utils.device import best_device
        device = best_device()

    del config  # reserved for future hparam fallbacks; see docstring.
    return load_model_with_encoding(checkpoint_path, device)
