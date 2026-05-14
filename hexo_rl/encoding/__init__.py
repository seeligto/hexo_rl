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

§176 P3 retired the legacy `hexo_rl.utils.encoding` NamedTuple shim;
the wire-format scalars consumers (trainer ckpt-load propagation +
pool runner-kwarg construction) used to read off it now live at
`hexo_rl.encoding.compat.WIRE_FORMAT_SPECS`.
"""
from hexo_rl.encoding.registry import (
    EncodingRegistryError,
    all_specs,
    lookup,
)
from hexo_rl.encoding.resolvers import (
    ShapeMismatchError,
    expand_auto_paths,
    normalize_encoding_name,
    resolve_anchor_path,
    resolve_corpus_path,
    resolve_encoding_for_eval,
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
    "normalize_encoding_name",
    "resolve_anchor_path",
    "resolve_corpus_path",
    "resolve_encoding_for_eval",
    "resolve_from_checkpoint",
    "resolve_from_config",
    "validate_against_state_dict",
]
