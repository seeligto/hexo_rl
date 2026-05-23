"""Lightweight checkpoint -> HexTacToeNet loader for the /analyze path.

Deliberately avoids importing Trainer to keep the import graph minimal
(monitoring invariant enforced by tests/test_analyze_api.py
``TestMonitoringInvariant``). State-dict inspection helpers live in
``hexo_rl.training.checkpoints`` and are shared with Trainer
(§176 P79 dedup; previously copied byte-for-byte from Trainer with a
"keep in sync" note). Paired follow-up: P47 (deferred to Phase 5 / W3)
will lift this to a public ``load_model_for_inference`` API used by
viewer + bots alike.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.model.network_min_max_head import VALUE_FC1_MULTIPLIER
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.training.checkpoints import (
    extract_model_state,
    infer_model_hparams,
    normalize_model_state_dict_keys,
)

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
BUFFER_CHANNELS: int = _V6.n_planes


# Underscore-prefixed re-exports for back-compat with our_model_bot and
# tests/test_analyze_api parity checks. New callers should import the
# unprefixed names from hexo_rl.training.checkpoints directly.
_extract_model_state = extract_model_state
_infer_model_hparams = infer_model_hparams


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[HexTacToeNet, dict, torch.device]:
    """Load checkpoint -> (model in eval mode, metadata dict, device).

    The model is ready for inference with torch.no_grad() + autocast.
    """
    if device is None:
        from hexo_rl.utils.device import best_device
        device = best_device()

    checkpoint_path = Path(checkpoint_path)
    payload: Any = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = normalize_model_state_dict_keys(extract_model_state(payload))
    hparams = infer_model_hparams(state_dict)

    net = HexTacToeNet(
        board_size=hparams.get("board_size", BOARD_SIZE),
        in_channels=hparams.get("in_channels", BUFFER_CHANNELS),
        filters=hparams.get("filters", 128),
        res_blocks=hparams.get("res_blocks", 12),
        se_reduction_ratio=hparams.get("se_reduction_ratio", 4),
    )
    # Strict-equivalent: filter tower/trunk aliases added by normalize, raise on real mismatches.
    model_keys = set(net.state_dict().keys())
    missing = [k for k in model_keys if k not in state_dict]
    real_unexpected = [
        k for k in state_dict if k not in model_keys
        and not ((k.startswith("tower.") and f"trunk.{k}" in model_keys)
                 or (k.startswith("trunk.tower.") and k[len("trunk."):] in model_keys))
    ]
    if missing or real_unexpected:
        raise ValueError(f"Checkpoint mismatch — missing: {missing}, unexpected: {real_unexpected}")
    # §S181 FU-2 A2 — pre-A2 ckpts have value_fc1 input dim 2*filters; A2
    # model expects VALUE_FC1_MULTIPLIER * filters. Surface a clear error
    # rather than letting load_state_dict leak the raw torch shape mismatch.
    fc1_w = state_dict.get("value_fc1.weight")
    if fc1_w is not None and int(fc1_w.shape[1]) != net.value_fc1.in_features:
        raise RuntimeError(
            f"value_fc1 shape mismatch in {checkpoint_path}: checkpoint has "
            f"in_features={int(fc1_w.shape[1])}, A2 model expects "
            f"{net.value_fc1.in_features} (VALUE_FC1_MULTIPLIER="
            f"{VALUE_FC1_MULTIPLIER} * filters). Pre-§S181-FU-2 A2 "
            f"checkpoints (GAP+GMP 2*filters) are INVALID for the A2 "
            f"multi-scale avg-pool value head — pretrain a new A2 anchor."
        )
    net.load_state_dict(state_dict, strict=False)
    net.to(device).eval()

    metadata = {
        "hparams": hparams,
        "config": payload.get("config", {}),
        "step": payload.get("step"),
        "checkpoint_path": str(checkpoint_path),
    }
    return net, metadata, device
