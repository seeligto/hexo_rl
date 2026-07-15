"""Canonical encoding registry — §172 Phase A3.

Public API:
  - `EncodingSpec`               type alias for `engine.RegistrySpec`
                                 (cycle 3 Wave 8 Batch A FF.2 — retired
                                 the parallel Python @dataclass mirror)
  - `lookup(name)`               by-name registry access
  - `all_specs()`                iterate every registered spec
  - `resolve_from_config(cfg)`   resolve from a config mapping
  - `resolve_from_checkpoint(p)` resolve from a saved checkpoint
  - `validate_against_state_dict(spec, sd)` cross-check shapes
  - `EncodingRegistryError`      raised on parse / lookup failure
  - `ShapeMismatchError`         raised by validate_against_state_dict

Schema authoring lives in `engine/src/encoding/registry.toml`; the Rust
parser at `engine/src/encoding/registry.rs` is the single source of
truth. The retired Python parser at `hexo_rl/encoding/registry.py` is
preserved as a thin delegating shim over `engine.RegistrySpec.from_registry`.

§176 P3 retired the legacy `hexo_rl.utils.encoding` NamedTuple shim;
wire-format scalars (cluster_window_size / cluster_threshold /
legal_move_radius / board_size) are read directly off the registry
record at `hexo_rl.encoding.lookup(name)`.

Cycle 3 Wave 8 Batch C (FF.10, 2026-05-17): retired the `WireFormatSpec`
dataclass + `WIRE_FORMAT_SPECS` table + `legacy_spec_for_registry_name`
shim at `hexo_rl/encoding/compat.py`. The trainer ckpt-load propagation
and the pool runner-kwarg construction now read the registry record
directly; the Rust SelfPlayRunner takes `encoding_name: Optional[str]`
and resolves the spec once on the Rust side.

§178 cycle 3 Wave 8 Batch A FF.2 (2026-05-17): retired the parallel
Python `hexo_rl.encoding.spec.EncodingSpec` @dataclass mirror; the
canonical Rust `engine.RegistrySpec` PyO3 wrapper is now the sole
encoding record type. `EncodingSpec` survives as a type alias for
backwards compatibility at every consumer call site.
"""
from engine import RegistrySpec as EncodingSpec  # type: ignore[attr-defined]

from hexo_rl.encoding.registry import (
    EncodingRegistryError,
    all_specs,
    lookup,
)
from hexo_rl.encoding.resolvers import (
    ArchSpec,
    ShapeMismatchError,
    assert_not_heldout_sha,
    cur_stone_slot,
    detect_encoding_from_state_dict,
    expand_auto_paths,
    held_out_shas,
    heldout_size_bytes,
    normalize_encoding_name,
    opp_stone_slot,
    resolve_anchor_path,
    resolve_arch,
    resolve_corpus_path,
    resolve_corpus_sha_pin,
    resolve_encoding_for_eval,
    resolve_from_checkpoint,
    resolve_from_config,
    validate_against_state_dict,
)

__all__ = [
    "ArchSpec",
    "EncodingSpec",
    "EncodingRegistryError",
    "ShapeMismatchError",
    "all_specs",
    "assert_not_heldout_sha",
    "cur_stone_slot",
    "detect_encoding_from_state_dict",
    "expand_auto_paths",
    "held_out_shas",
    "heldout_size_bytes",
    "lookup",
    "normalize_encoding_name",
    "opp_stone_slot",
    "resolve_anchor_path",
    "resolve_arch",
    "resolve_corpus_path",
    "resolve_corpus_sha_pin",
    "resolve_encoding_for_eval",
    "resolve_from_checkpoint",
    "resolve_from_config",
    "validate_against_state_dict",
]
