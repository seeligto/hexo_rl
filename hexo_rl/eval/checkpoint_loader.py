"""Encoding-aware checkpoint loader.

Single entry point that detects the encoding version from a checkpoint
file and instantiates the matching `HexTacToeNet`. Used by the
generalized SealBot eval harness (`scripts/run_sealbot_eval.py`) and any
other downstream code that must accept a checkpoint without knowing the
encoding upfront.

Detection priority:
1. Explicit `encoding` field in the checkpoint dict (preferred for new
   checkpoints — pretrain should write this).
2. Filename heuristic: `v6w25` substring → v6w25.
3. `trunk.input_conv.weight` shape:
   - `in_channels == 11` → v8
   - `in_channels == 8`  → v6 (default; v6w25 falls under v6 by shape but is
     disambiguated via the filename heuristic above).

v6w25 shares wire format with v6 (8 planes, K-cluster) but uses a 25×25
cluster window + R=8 perception. Both fact families resolve at the
inference adapter (V6ArgmaxBot, etc.) — the loader only needs to surface
the EncodingSpec correctly so dispatch downstream picks the right bot.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch

from hexo_rl.encoding import (
    EncodingSpec,
    detect_encoding_from_state_dict as _registry_detect_from_state_dict,
    lookup as _registry_lookup,
)
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import normalize_model_state_dict_keys

BUFFER_CHANNELS: int = _registry_lookup("v6").n_planes


def _strip_compile_prefixes(state: dict) -> dict:
    """Strip `_orig_mod.` / `module.` wrapper prefixes from state-dict keys.

    Lighter than `normalize_model_state_dict_keys` — does NOT add
    `tower.*` ↔ `trunk.tower.*` aliases. Aliasing breaks strict-load on
    v8 checkpoints whose state dicts already carry both.
    """
    prefixes = ("_orig_mod.", "module.")
    out: dict = {}
    for key, value in state.items():
        norm_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if norm_key.startswith(prefix):
                    norm_key = norm_key[len(prefix):]
                    changed = True
        out[norm_key] = value
    return out



def detect_encoding_label(ckpt_path: Path, state: dict) -> str:
    """Return the encoding label string: 'v6', 'v6w25', or 'v8'.

    Pure-detection helper — does not load model. Useful for tests and the
    inference-method dispatcher when only the label is needed.

    §176 P6: thin shim over
    ``hexo_rl.encoding.resolvers.detect_encoding_from_state_dict``
    (strict=True). The shared helper raises ValueError on missing keys
    and unsupported in_channels, defaults `in_ch=8` with no n_actions
    probe to v6 (preserves the previous fallback at this site), and
    handles the v6w25 filename-substring disambiguator via the
    ``ckpt_label`` parameter (we pass the basename only to keep the
    historical scoping — full path could match `v6w25` in a parent dir).
    """
    spec = _registry_detect_from_state_dict(
        state, ckpt_path.name, strict=True,
    )
    # strict=True guarantees a non-None spec.
    assert spec is not None
    return spec.name


def load_model_with_encoding(
    ckpt_path: str | Path,
    device: torch.device,
) -> Tuple[HexTacToeNet, EncodingSpec, str]:
    """Load checkpoint, detect encoding, return (model, spec, label).

    The label is one of {'v6', 'v6w25', 'v8'} and matches ``spec.name``.
    Use the label to drive bot dispatch; use the spec for numeric constants.
    """
    ckpt_path = Path(ckpt_path)
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state: dict = raw
    explicit_label: str | None = None
    if isinstance(raw, dict):
        explicit_label = raw.get("encoding")
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break

    # v6 path uses normalize_model_state_dict_keys (handles tower↔trunk.tower
    # aliasing for legacy v7full / v6 / v7 checkpoints saved without the
    # `trunk.` prefix). v8 checkpoints already use `trunk.tower.*` and
    # break under aliasing — they get the lighter strip-prefixes-only path.
    inp_w = state.get("trunk.input_conv.weight")
    if inp_w is None:
        inp_w = state.get("trunk.input_conv.conv.weight")
    if inp_w is None:
        inp_w = state.get("_orig_mod.trunk.input_conv.weight")
    if inp_w is None:
        inp_w = state.get("_orig_mod.trunk.input_conv.conv.weight")
    if inp_w is None:
        # Fall back: try after light strip — may surface a v8 checkpoint
        # under a wrapper prefix.
        light = _strip_compile_prefixes(state)
        inp_w = light.get("trunk.input_conv.weight") \
            or light.get("trunk.input_conv.conv.weight")
    if inp_w is not None and int(inp_w.shape[1]) == 11:
        state = _strip_compile_prefixes(state)
    else:
        state = normalize_model_state_dict_keys(state)
    label = explicit_label or detect_encoding_label(ckpt_path, state)
    if label not in ("v6", "v6tp", "v6_live2", "v6w25", "v8"):
        raise ValueError(
            f"checkpoint {ckpt_path}: unknown encoding label {label!r}"
        )

    if label == "v8":
        spec = _registry_lookup("v8")
    elif label == "v6w25":
        spec = _registry_lookup("v6w25")
    elif label == "v6tp":
        # §P5-CT CF-2 — 10-plane (v6 + turn-phase 16/17); builds like v6
        # (min_max head, in_channels read from the conv weight = 10).
        spec = _registry_lookup("v6tp")
    elif label == "v6_live2":
        # §P5-CT H-PLANE fix — 4-plane (v6 minus history); builds like v6
        # (min_max head, in_channels read from the conv weight = 4).
        spec = _registry_lookup("v6_live2")
    else:
        spec = _registry_lookup("v6")
    model = _build_model_from_spec(state, spec)

    model.to(device)
    model.eval()
    return model, spec, label


def _build_model_from_spec(state: dict, spec: EncodingSpec) -> HexTacToeNet:
    """Unified entry point — dispatches to the per-family builder by
    ``spec.has_pass_slot``.

    Branch:
      - ``has_pass_slot=True``  → ``_build_min_max_model`` (v6 / v6w25 /
                                  v7full / v7 / v7e30 / v7mw family;
                                  min_max policy head with optional
                                  pma / pma_global pool variants and the
                                  §170 P3 gpool_bias side-branch).
      - ``has_pass_slot=False`` → ``_build_kata_model`` (v8 / v8_canvas_realness
                                  family; KataGoPolicyHead + per-block gpool +
                                  optional PartialConv2d canvas_realness wrap).

    Cycle 3 Wave 8 Batch D (2026-05-17): renamed from
    ``_build_v6_model`` / ``_build_v8_model`` (GENERICISE #4 fold);
    bodies preserved architecturally distinct because feature-detection
    + ``strict`` load policy differ per family.
    """
    if spec.has_pass_slot:
        return _build_min_max_model(state, spec)
    return _build_kata_model(state, spec)


def _build_min_max_model(state: dict, spec: EncodingSpec) -> HexTacToeNet:
    inp_w = state["trunk.input_conv.weight"]
    filters = int(inp_w.shape[0])
    in_channels = int(inp_w.shape[1])
    block_indices = sorted({
        int(k.split(".")[2]) for k in state.keys()
        if k.startswith("trunk.tower.") and len(k.split(".")) >= 4
    })
    res_blocks = max(block_indices) + 1 if block_indices else 12
    # §169 A2 / A3 — detect pool type from state dict.
    #   global_encoder.* + cluster_pool.global_gate ⇒ pma_global (A3).
    #   cluster_pool.* without global branch ⇒ pma (A2).
    #   Otherwise ⇒ min_max (A1, default).
    has_global_encoder = any(k.startswith("global_encoder.") for k in state)
    has_cluster_pool = any(k.startswith("cluster_pool.") for k in state)
    if has_global_encoder:
        pool_type = "pma_global"
    elif has_cluster_pool:
        pool_type = "pma"
    else:
        pool_type = "min_max"
    # §170 P3 — detect the gpool-bias side-branch via state-dict keys. Only
    # valid in tandem with pool_type='min_max' (the constructor enforces
    # this); presence of `gpool_bias_branch.*` keys is the unambiguous flag.
    gpool_bias_active = any(
        k.startswith("gpool_bias_branch.") for k in state
    )
    if gpool_bias_active and pool_type != "min_max":
        raise ValueError(
            "checkpoint has both gpool_bias_branch.* and "
            f"cluster_pool.* keys (pool_type={pool_type!r}); the side-branch "
            "is A1-only (pool_type='min_max'). Inspect the checkpoint."
        )
    model = HexTacToeNet(
        board_size=spec.board_size,
        in_channels=in_channels,
        filters=filters,
        res_blocks=res_blocks,
        encoding="v6",
        pool_type=pool_type,
        gpool_bias_active=gpool_bias_active,
    )
    # strict=False because v6 / v6w25 checkpoints may carry tower.* duplicates
    # left over from older save formats (see eval_pipeline._load_anchor_model).
    model.load_state_dict(state, strict=False)
    return model


def _build_kata_model(state: dict, spec: EncodingSpec) -> HexTacToeNet:
    # §169 A4 — under canvas_realness the trunk-entry conv is wrapped in a
    # PartialConv2d, so the weight key shifts from `trunk.input_conv.weight`
    # to `trunk.input_conv.conv.weight`. Detection is unambiguous because
    # the regular Conv2d path never has a `.conv.weight` sub-key.
    canvas_realness = "trunk.input_conv.conv.weight" in state
    inp_w_key = (
        "trunk.input_conv.conv.weight" if canvas_realness
        else "trunk.input_conv.weight"
    )
    inp_w = state[inp_w_key]
    filters = int(inp_w.shape[0])
    in_channels = int(inp_w.shape[1])
    if in_channels != 11:
        raise ValueError(
            f"v8 checkpoint expects in_channels=11; got {in_channels}"
        )
    block_indices = sorted({
        int(k.split(".")[2]) for k in state.keys()
        if k.startswith("trunk.tower.") and len(k.split(".")) >= 4
    })
    res_blocks = max(block_indices) + 1 if block_indices else 0
    gpool_indices = sorted({
        i for i in block_indices
        if f"trunk.tower.{i}.conv1.conv1g.weight" in state
    })
    head_use_gpool = "policy_head.conv1g.weight" in state
    model = HexTacToeNet(
        board_size=spec.board_size,
        in_channels=in_channels,
        filters=filters,
        res_blocks=res_blocks,
        encoding="v8",
        gpool_indices=gpool_indices if gpool_indices else None,
        head_use_gpool=head_use_gpool,
        canvas_realness=canvas_realness,
    )
    model.load_state_dict(state, strict=True)
    return model
