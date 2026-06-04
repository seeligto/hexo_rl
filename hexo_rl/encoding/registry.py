"""Encoding registry — thin delegating shim over `engine.RegistrySpec`.

Authored §172 Phase A3 (2026-05-09). Counterpart to
`engine/src/encoding/registry.rs`. Originally re-parsed the canonical
TOML at `engine/src/encoding/registry.toml` independently of the Rust
parser; cycle 3 Wave 8 Batch A FF.2 (2026-05-17) retired the parallel
Python parser + @dataclass mirror, making the Rust loader the single
source of truth.

The Python module surface is preserved (`lookup`, `all_specs`,
`EncodingRegistryError`, plus the `_load` private helper used by
`compat.py` / `resolvers.py` / `audit_sections.py` / the §172 A5
metadata migration script). All entry points delegate to
`engine.RegistrySpec.from_registry()`.
"""
from __future__ import annotations

from typing import Iterable

from engine import RegistrySpec as _EngineRegistrySpec  # type: ignore[attr-defined]


# Canonical registered encoding names — must mirror
# `engine/src/encoding/registry.toml`. Used by `_load` to populate the
# Python-side cache. Adding a new encoding requires adding the TOML
# entry AND appending the name here. (Rust still owns parse + validation.)
_REGISTERED_NAMES: tuple[str, ...] = (
    "v6",
    "v6tp",
    "v6_live2",
    "v6_live2_anchored",
    "v7full",
    "v7",
    "v7e30",
    "v6w25",
    "v7mw",
    "v8",
    "v8_canvas_realness",
)


class EncodingRegistryError(Exception):
    """Raised on registry parse failure or unknown encoding lookup.

    Preserved for backwards compatibility with consumers that catch this
    exception class. The underlying Rust parser raises Python `ValueError`
    on unknown lookup; `lookup` translates that into this exception type.
    """


_REGISTRY_CACHE: dict[str, _EngineRegistrySpec] | None = None


def _load() -> dict[str, _EngineRegistrySpec]:
    """Return the cached registry dict.

    Preserved as a private helper because `compat.py`, `resolvers.py`,
    `audit_sections.py`, and the §172 A5 metadata migration script all
    import it. Iterates registered names exactly once (the underlying
    `engine.RegistrySpec.from_registry` is itself O(1) on a LazyLock).
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    specs: dict[str, _EngineRegistrySpec] = {}
    for name in _REGISTERED_NAMES:
        try:
            specs[name] = _EngineRegistrySpec.from_registry(name)
        except ValueError as exc:  # pragma: no cover — registry/toml mismatch
            raise EncodingRegistryError(
                f"engine registry missing canonical name {name!r}; "
                f"engine.RegistrySpec.from_registry raised: {exc}"
            ) from exc
    _REGISTRY_CACHE = specs
    return specs


def lookup(name: str) -> _EngineRegistrySpec:
    """Return spec for `name` or raise `EncodingRegistryError`.

    Cached: repeated calls with the same name return the byte-identical
    Python `engine.RegistrySpec` wrapper instance, preserving the
    pre-Wave-8 ``lookup(name) is lookup(name)`` contract that the
    round-trip test pins at `tests/test_encoding_round_trip.py:115`.
    """
    cache = _load()
    spec = cache.get(name)
    if spec is None:
        registered = sorted(_REGISTERED_NAMES)
        raise EncodingRegistryError(
            f"unknown encoding {name!r}; registered: {registered}"
        )
    return spec


def all_specs() -> Iterable[_EngineRegistrySpec]:
    """Iterate every registered spec (insertion order — TOML preserves)."""
    return _load().values()
