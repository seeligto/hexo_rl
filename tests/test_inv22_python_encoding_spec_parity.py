"""INV22 — Python `EncodingSpec` is the `engine.RegistrySpec` type alias.

Cycle 3 Wave 8 Batch A FF.2 (2026-05-17). Pins:

  1. ``hexo_rl.encoding.EncodingSpec is engine.RegistrySpec`` — type alias
     byte-identity (no parallel @dataclass mirror).
  2. For every registered encoding, the alias exposes all 18 schema fields
     plus the 6 derived accessors (`n_actions`, `n_cells`, `state_stride`,
     `chain_stride`, `aux_stride`, `policy_stride`) as Python attributes
     (not method calls), with values mirroring the Rust `RegistrySpec`
     impl.
  3. Consumer-side migration smoke: ``from hexo_rl.encoding import
     EncodingSpec`` resolves, ``EncodingSpec`` constructs via
     ``from_registry``, and field reads return values identical to a
     direct ``engine.RegistrySpec.from_registry(name).<field>`` read.

The pin guards against drift if a future commit re-introduces a Python
shim (e.g. wrapper class) without preserving the engine.RegistrySpec
identity contract.
"""
from __future__ import annotations

import engine
import pytest

from hexo_rl.encoding import EncodingSpec, all_specs

# 8 canonical registered encodings — sourced from
# `engine/src/encoding/registry.toml`. Parametrise across the full set so
# adding a new TOML entry surfaces missing accessor coverage immediately.
_REGISTERED_NAMES: tuple[str, ...] = tuple(sorted(s.name for s in all_specs()))


# Required attribute surface (post-Wave 8 Batch A retirement of the Python
# @dataclass mirror). The 18 schema fields are TOML-parsed by the Rust
# loader; the 6 derived accessors are exposed as `#[getter]` methods on
# `engine.PyRegistrySpec`.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "name",
    "board_size",
    "trunk_size",
    "cluster_window_size",
    "cluster_threshold",
    "legal_move_radius",
    "n_planes",
    "plane_layout",
    "policy_logit_count",
    "has_pass_slot",
    "is_multi_window",
    "value_pool",
    "policy_pool",
    "sym_table_id",
    "schema_version",
    "notes",
    "kept_plane_indices",
    "n_source_planes",
)
_REQUIRED_DERIVED: tuple[str, ...] = (
    "n_actions",
    "n_cells",
    "state_stride",
    "chain_stride",
    "aux_stride",
    "policy_stride",
)


def test_inv22_python_encoding_spec_is_engine_registry_spec_alias() -> None:
    """INV22 #1 — ``EncodingSpec`` IS ``engine.RegistrySpec`` (type alias).

    Guards against accidental re-introduction of a Python @dataclass mirror
    or a wrapper class that breaks ``isinstance`` checks across the
    boundary.
    """
    assert EncodingSpec is engine.RegistrySpec, (
        f"hexo_rl.encoding.EncodingSpec must be the engine.RegistrySpec "
        f"type alias; got {EncodingSpec!r} vs {engine.RegistrySpec!r}"
    )


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_inv22_required_attribute_surface(name: str) -> None:
    """INV22 #2 — every registered encoding exposes 18 fields + 6 derived
    accessors as Python attributes (not method calls)."""
    spec = engine.RegistrySpec.from_registry(name)
    for field in _REQUIRED_FIELDS:
        assert hasattr(spec, field), (
            f"{name}: missing required schema field {field!r} on "
            f"engine.RegistrySpec — PyO3 accessor gap"
        )
        # Property-style access: attribute read must NOT be a method
        # handle. (Catches a regression where `pub fn` is restored over
        # `#[getter]`.)
        value = getattr(spec, field)
        assert not callable(value), (
            f"{name}.{field} is callable; Python @dataclass mirror "
            f"exposed it as a field/property, not a method"
        )
    for derived in _REQUIRED_DERIVED:
        assert hasattr(spec, derived), (
            f"{name}: missing required derived accessor {derived!r} on "
            f"engine.RegistrySpec"
        )
        value = getattr(spec, derived)
        assert not callable(value), (
            f"{name}.{derived} is callable; the Python @dataclass mirror "
            f"used `@property`, not a method"
        )


@pytest.mark.parametrize("name", _REGISTERED_NAMES)
def test_inv22_consumer_migration_smoke(name: str) -> None:
    """INV22 #3 — consumer-side migration: import via hexo_rl.encoding,
    construct via from_registry, field reads identical to direct engine
    access."""
    direct = engine.RegistrySpec.from_registry(name)
    via_alias = EncodingSpec.from_registry(name)
    # Field reads must agree across the alias and the direct symbol.
    for field in _REQUIRED_FIELDS + _REQUIRED_DERIVED:
        d_val = getattr(direct, field)
        a_val = getattr(via_alias, field)
        # Some fields return Vec<...>/list (e.g. plane_layout,
        # kept_plane_indices). Coerce to tuple for stable equality
        # comparison across PyO3 list returns.
        if isinstance(d_val, list):
            d_val = tuple(d_val)
            a_val = tuple(a_val)
        assert d_val == a_val, (
            f"{name}.{field}: alias-read {a_val!r} != direct-read {d_val!r}"
        )
