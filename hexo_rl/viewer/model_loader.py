"""Lightweight checkpoint -> HexTacToeNet loader for the /analyze path.

Deliberately avoids importing Trainer to keep the import graph minimal.
The _extract_model_state and _infer_model_hparams functions are copied from
Trainer (hexo_rl/training/trainer.py) -- keep in sync if those change.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import normalize_model_state_dict_keys


# -- Copied from Trainer._extract_model_state ---------------------------------

def _extract_model_state(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Extract model state dict from common checkpoint payload layouts."""
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(ckpt)!r}")
    for key in ("model_state", "model_state_dict", "state_dict"):
        maybe_state = ckpt.get(key)
        if isinstance(maybe_state, dict):
            return maybe_state
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt
    raise ValueError("Unable to locate model state dict in checkpoint payload")


# -- Copied from Trainer._infer_res_blocks_from_state_dict --------------------

def _infer_res_blocks(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    pattern = re.compile(r"^(?:trunk\.)?tower\.(\d+)\.")
    idxs = {
        int(m.group(1))
        for key in state_dict.keys()
        for m in [pattern.search(key)]
        if m is not None
    }
    return (max(idxs) + 1) if idxs else None


# -- Copied from Trainer._infer_model_hparams ---------------------------------

def _infer_model_hparams(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infer model hyperparameters from checkpoint weights alone."""
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
    res_blocks = _infer_res_blocks(state_dict)
    if res_blocks is not None:
        inferred["res_blocks"] = int(res_blocks)
    return inferred


# -- Public API ----------------------------------------------------------------

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
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = normalize_model_state_dict_keys(_extract_model_state(payload))
    hparams = _infer_model_hparams(state_dict)

    net = HexTacToeNet(
        board_size=hparams.get("board_size", 19),
        in_channels=hparams.get("in_channels", 24),
        filters=hparams.get("filters", 128),
        res_blocks=hparams.get("res_blocks", 12),
        se_reduction_ratio=hparams.get("se_reduction_ratio", 4),
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
