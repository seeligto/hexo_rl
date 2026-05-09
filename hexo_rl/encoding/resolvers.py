"""Encoding resolvers — config-form, checkpoint-form, state-dict validation.

Authored §172 Phase A3 (2026-05-09). The two `resolve_*` functions are
the only A4-blessed paths to construct an `EncodingSpec` outside the
registry itself; consumer call sites should never call `lookup` with a
hard-coded string except for explicit defaults.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

from hexo_rl.encoding import compat
from hexo_rl.encoding.registry import EncodingRegistryError, lookup
from hexo_rl.encoding.spec import EncodingSpec


class ShapeMismatchError(Exception):
    """Raised when state-dict shapes contradict an EncodingSpec."""


def resolve_from_config(cfg: Mapping[str, Any] | None) -> EncodingSpec:
    """Return an `EncodingSpec` from a config mapping.

    Accepts BOTH legacy forms:
      - `cfg['encoding'] = "v6w25"`            (string form)
      - `cfg['encoding'] = {'version': 'v6'}`  (mapping form, current canonical)

    Default: `"v6"` if no encoding section present (preserves byte-exact
    pre-§166 behavior).
    """
    if cfg is None:
        return lookup("v6")
    section = cfg.get("encoding")
    if section is None:
        return lookup("v6")
    if isinstance(section, str):
        return lookup(section)
    if isinstance(section, Mapping):
        version = section.get("version", "v6")
        if not isinstance(version, str):
            raise EncodingRegistryError(
                f"encoding.version must be a string; got {type(version).__name__}"
            )
        return lookup(version)
    raise EncodingRegistryError(
        f"encoding section must be str or mapping; got {type(section).__name__}"
    )


def resolve_from_checkpoint(path: str | Path) -> EncodingSpec:
    """Return an `EncodingSpec` for a saved checkpoint.

    Reads `ckpt['metadata']['encoding_name']` if present. Otherwise
    falls back to `compat.infer_encoding_from_state_dict` and emits a
    `DeprecationWarning` directing to §172 A5 stamping.
    """
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    meta = d.get("metadata") if isinstance(d, dict) else None
    if isinstance(meta, dict) and "encoding_name" in meta:
        name = meta["encoding_name"]
        if not isinstance(name, str):
            raise EncodingRegistryError(
                f"checkpoint {path}: metadata['encoding_name'] is "
                f"{type(name).__name__}, expected str"
            )
        return lookup(name)

    if isinstance(d, dict) and "model_state" in d:
        sd = d["model_state"]
    else:
        sd = d
    if not isinstance(sd, Mapping):
        raise EncodingRegistryError(
            f"checkpoint {path}: cannot extract state-dict for shape inference"
        )
    name = compat.infer_encoding_from_state_dict(sd, str(path))
    warnings.warn(
        f"checkpoint {path} has no metadata['encoding_name']; "
        f"inferred {name!r} from state-dict + filename. Stamp metadata via "
        f"§172 A5 migration script.",
        DeprecationWarning,
        stacklevel=2,
    )
    return lookup(name)


def validate_against_state_dict(
    spec: EncodingSpec, state_dict: Mapping[str, Any]
) -> None:
    """Cross-check spec.policy_logit_count + spec.n_planes against a state-dict.

    Probes a list of common key names for the policy fc and first conv.
    Silently no-ops for keys that don't appear (caller's responsibility
    to know which architecture they hold). Raises `ShapeMismatchError`
    on disagreement.
    """
    policy_keys = (
        "policy_fc.weight",
        "policy_head.fc.weight",
        "policy.fc.weight",
        "policy.weight",
    )
    conv_keys = (
        "trunk.0.weight",
        "trunk.conv.weight",
        "input_conv.weight",
        "stem.0.weight",
        "conv1.weight",
    )

    pfc = None
    for k in policy_keys:
        if k in state_dict:
            pfc = state_dict[k]
            break
    if pfc is not None:
        out_features = int(pfc.shape[0])
        if out_features != spec.policy_logit_count:
            raise ShapeMismatchError(
                f"policy_fc out_features {out_features} != "
                f"spec.policy_logit_count {spec.policy_logit_count} "
                f"for encoding {spec.name!r}"
            )

    conv = None
    for k in conv_keys:
        if k in state_dict:
            conv = state_dict[k]
            break
    if conv is not None:
        in_channels = int(conv.shape[1])
        if in_channels != spec.n_planes:
            raise ShapeMismatchError(
                f"first conv in_channels {in_channels} != "
                f"spec.n_planes {spec.n_planes} for encoding {spec.name!r}"
            )
