"""Canonical encoding registry — §172 Phase A3.

Public API:
  - `EncodingSpec`               immutable dataclass (full schema)
  - `lookup(name)`               by-name registry access
  - `all_specs()`                iterate every registered spec
  - `resolve_from_config(cfg)`   resolve from a config mapping
  - `resolve_from_checkpoint(p)` resolve from a saved checkpoint
  - `validate_against_state_dict(spec, sd)` cross-check shapes
  - `EncodingRegistryError`      raised on parse / lookup failure
  - `ShapeMismatchError`         raised by validate_against_state_dict

Schema authoring lives in `engine/src/encoding/registry.toml`; both
Rust (`engine/src/encoding/registry.rs`) and Python parse the same
file.

A4 will migrate consumers off the legacy `hexo_rl.utils.encoding`
NamedTuple shim and extend the PyO3 binding to surface the full
schema.
"""
from hexo_rl.encoding.registry import (
    EncodingRegistryError,
    all_specs,
    lookup,
)
from hexo_rl.encoding.resolvers import (
    ShapeMismatchError,
    expand_auto_paths,
    resolve_anchor_path,
    resolve_corpus_path,
    resolve_from_checkpoint,
    resolve_from_config,
    validate_against_state_dict,
)
from hexo_rl.encoding.spec import EncodingSpec

__all__ = [
    "EncodingSpec",
    "EncodingRegistryError",
    "ShapeMismatchError",
    "all_specs",
    "expand_auto_paths",
    "lookup",
    "resolve_anchor_path",
    "resolve_corpus_path",
    "resolve_from_checkpoint",
    "resolve_from_config",
    "validate_against_state_dict",
]
