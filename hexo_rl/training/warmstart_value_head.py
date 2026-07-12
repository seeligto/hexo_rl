"""T6 — E1 warm-start value-head loader.

Seeds an E1 net's VALUE HEAD from a converged HEADSWAP head. Both E1 arms
warm-start: the scalar arm from arm_A_seed0, the dist arm from arm_B_seed0
(pre-registered — see PINNED constants below). The 248k trunk is loaded
SEPARATELY by the launch wiring (Trainer.load_checkpoint in
orchestrator.init_trainer); this module ONLY overwrites the value-head tensors
afterwards — it does not touch trunk / policy / aux params.

Scope (INV-D1 / R5): this LOADS weights only. No gradient / target change; no
teacher / TD / distill / solver. The scalar path stays byte-identical to E1's
scalar arm.

HEADSWAP head `.pt` layout (authoritative source:
scripts/headswap/train_arm.py:218-233 `save_blob`; head module
scripts/headswap/model_heads.py ScalarHead / BinHead):

    {
      "arm": "A"|"B"|..., "seed": int, "lr": float, "steps": int,
      "head_shape": "scalar" | "bin65",     # <- dist marker
      "head_state": {                       # <- the tensors (NO "value_" prefix)
          "fc1.weight": (256, 2C),  "fc1.bias": (256,),
          "fc2.weight": (1|65, 256), "fc2.bias": (1|65,),
      },
      "trunk_ckpt": str, "buffer_sha": str,
    }

Key remap onto the net (network.py:558-585):
    fc1.*  ->  value_fc1.*         (both head types)
    fc2.*  ->  value_fc2.*         (scalar)
    fc2.*  ->  value_fc2_bins.*    (dist65)

C1-consistency (mirrors hexo_rl/eval/checkpoint_loader.py d695208): a
scalar↔dist mismatch RAISES loudly (no silent random-head fallback), and the
loaded head tensors are VERIFIED to have landed via ``allclose`` against the
source after the copy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from hexo_rl.training.checkpoints import get_base_model

# ── Pinned warm-start selection (pre-registered — do NOT change) ─────────────
# The launch config supplies the concrete path per arm; these are the
# documented defaults / canonical selection.
_HEADSWAP_AB_DIR = "/home/timmy/headswap_safe/box_results/headswap/ab"
ARM_A_SCALAR_HEAD = f"{_HEADSWAP_AB_DIR}/arm_A_seed0/head_A_seed0.pt"
ARM_B_DIST_HEAD = f"{_HEADSWAP_AB_DIR}/arm_B_seed0/head_B_seed0.pt"

# head_shape metadata string -> expected head_type
_SHAPE_TO_HEAD_TYPE = {"scalar": "scalar", "bin65": "dist65"}
_VALID_HEAD_TYPES = {"scalar", "dist65"}


def load_value_head(net: Any, head_pt_path: str, head_type: str) -> None:
    """Seed ``net``'s value head from a HEADSWAP head `.pt`.

    Args:
        net:          a built HexTacToeNet (or a torch.compile wrapper of one).
                      Its value_head_type must match ``head_type``.
        head_pt_path: path to a HEADSWAP head `.pt` (wrapper dict, see module doc).
        head_type:    "scalar" | "dist65" — the EXPECTED head kind. A mismatch
                      against the `.pt`'s ``head_shape`` (or against the net's
                      built head) RAISES (C1 guard, no silent fallback).

    Raises:
        ValueError:   unknown head_type; scalar↔dist mismatch (`.pt` vs
                      head_type, or head_type vs net's built head); shape
                      mismatch on any value-head tensor.
        RuntimeError: post-load verification failed (a tensor did not land).

    Does NOT touch trunk / policy / aux params.
    """
    if head_type not in _VALID_HEAD_TYPES:
        raise ValueError(
            f"head_type={head_type!r} not in {sorted(_VALID_HEAD_TYPES)}"
        )

    base = get_base_model(net)

    # Net-vs-head_type consistency: a scalar request against a dist net (has
    # value_fc2_bins) would leave value_fc2_bins RANDOM; reject the combo.
    net_has_bins = getattr(base, "value_fc2_bins", None) is not None
    net_head_type = getattr(base, "value_head_type", "scalar")
    if head_type == "dist65" and not net_has_bins:
        raise ValueError(
            f"head_type='dist65' but the net has no value_fc2_bins layer "
            f"(net value_head_type={net_head_type!r}). Build the net with "
            "value_head_type='dist65' before warm-starting a dist head."
        )
    if head_type == "scalar" and net_has_bins:
        raise ValueError(
            "head_type='scalar' but the net is a dist65 net (has "
            "value_fc2_bins). Seeding a scalar head onto a dist net would "
            "leave value_fc2_bins random. Match head_type to the net's "
            f"built head (net value_head_type={net_head_type!r})."
        )

    blob = torch.load(head_pt_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "head_state" not in blob:
        raise ValueError(
            f"{head_pt_path}: not a HEADSWAP head `.pt` "
            "(missing 'head_state' wrapper key). Expected the save_blob format "
            "from scripts/headswap/train_arm.py."
        )

    head_shape = blob.get("head_shape")
    pt_head_type = _SHAPE_TO_HEAD_TYPE.get(head_shape)
    if pt_head_type is None:
        raise ValueError(
            f"{head_pt_path}: unknown head_shape={head_shape!r} "
            f"(expected one of {sorted(_SHAPE_TO_HEAD_TYPE)})."
        )
    # C1 mismatch guard: `.pt`'s dist-ness must match the requested head_type.
    if pt_head_type != head_type:
        raise ValueError(
            f"scalar/dist mismatch: head `.pt` head_shape={head_shape!r} "
            f"(=> {pt_head_type!r}) but head_type={head_type!r} was requested. "
            "Refusing to load — mismatched value-head kind would silently drop "
            "the trained bin/scalar weights (C1 regression). Check the arm "
            "selection: scalar arm <- arm_A_seed0, dist arm <- arm_B_seed0."
        )

    head_state: Dict[str, torch.Tensor] = blob["head_state"]

    # Destination FC2 module: value_fc2 (scalar) | value_fc2_bins (dist).
    fc2_dst = base.value_fc2 if head_type == "scalar" else base.value_fc2_bins

    # (source key, destination Parameter, human label)
    mapping = [
        ("fc1.weight", base.value_fc1.weight, "value_fc1.weight"),
        ("fc1.bias", base.value_fc1.bias, "value_fc1.bias"),
        ("fc2.weight", fc2_dst.weight, "value_fc2.weight" if head_type == "scalar" else "value_fc2_bins.weight"),
        ("fc2.bias", fc2_dst.bias, "value_fc2.bias" if head_type == "scalar" else "value_fc2_bins.bias"),
    ]

    for src_key, dst_param, label in mapping:
        if src_key not in head_state:
            raise ValueError(
                f"{head_pt_path}: head_state missing key {src_key!r} "
                f"(have {sorted(head_state)})."
            )
        src_tensor = head_state[src_key]
        if tuple(src_tensor.shape) != tuple(dst_param.shape):
            raise ValueError(
                f"shape mismatch for {label}: head `.pt` {src_key} is "
                f"{tuple(src_tensor.shape)} but net expects "
                f"{tuple(dst_param.shape)}. Trunk filters / bin count "
                "disagree between the HEADSWAP head and the target net."
            )
        with torch.no_grad():
            dst_param.data.copy_(
                src_tensor.to(device=dst_param.device, dtype=dst_param.dtype)
            )

    # Post-load verification (mirrors eval checkpoint_loader d695208 allclose
    # guard): confirm every value-head tensor actually landed.
    for src_key, dst_param, label in mapping:
        src_tensor = head_state[src_key].to(
            device=dst_param.device, dtype=dst_param.dtype
        )
        if not torch.allclose(dst_param.data, src_tensor):
            raise RuntimeError(
                f"warm-start value-head verify FAILED for {label}: the tensor "
                f"did not land (post-copy mismatch). head_pt={head_pt_path}."
            )


def default_head_for_arm(head_type: str) -> str:
    """Return the pre-registered default HEADSWAP head path for an arm's head_type.

    scalar -> arm_A_seed0/head_A_seed0.pt ; dist65 -> arm_B_seed0/head_B_seed0.pt.
    The launch config MAY override; this encodes the pre-registered selection.
    """
    if head_type == "scalar":
        return ARM_A_SCALAR_HEAD
    if head_type == "dist65":
        return ARM_B_DIST_HEAD
    raise ValueError(
        f"head_type={head_type!r} not in {sorted(_VALID_HEAD_TYPES)}"
    )


# Path import kept for callers that want to existence-check the pinned defaults.
__all__ = [
    "ARM_A_SCALAR_HEAD",
    "ARM_B_DIST_HEAD",
    "load_value_head",
    "default_head_for_arm",
]

# Silence unused-import lint (Path re-exported for launch-wiring convenience).
_ = Path
