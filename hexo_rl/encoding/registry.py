"""Encoding registry — tomllib parser + cached lookup.

Authored §172 Phase A3 (2026-05-09). Counterpart to
`engine/src/encoding/registry.rs`; both parse the SAME canonical TOML
file (`engine/src/encoding/registry.toml`).

Discovery: walk up from this module until we find
`engine/src/encoding/registry.toml`. Repo dev install only — A2 §5.4.
Resolution result is cached at module-import for the rest of the
process.
"""
from __future__ import annotations

import pathlib
import tomllib
from typing import Any, Iterable, Mapping

from hexo_rl.encoding._validators import (
    validate_cross_field_consistency,
    validate_kept_plane_indices,
    validate_multi_window_invariants,
    validate_plane_layout,
    validate_pool_enums,
)
from hexo_rl.encoding.spec import EncodingSpec, PolicyPool, ValuePool


_NONE_SENTINEL = "none"


class EncodingRegistryError(Exception):
    """Raised on registry parse failure or unknown encoding lookup."""


def _find_registry_toml() -> pathlib.Path:
    """Walk up from this module to locate engine/src/encoding/registry.toml."""
    here = pathlib.Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        candidate = ancestor / "engine" / "src" / "encoding" / "registry.toml"
        if candidate.is_file():
            return candidate
    raise EncodingRegistryError(
        f"could not locate engine/src/encoding/registry.toml by walking up from {here}"
    )


_REGISTRY_TOML_PATH: pathlib.Path = _find_registry_toml()


def _coerce_optional_int(field: str, raw: Any, errors: list[str]) -> int | None:
    """Sentinel "none" → None; int → int; anything else → error."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw == _NONE_SENTINEL:
            return None
        errors.append(f"{field}: expected int or 'none' sentinel; got string {raw!r}")
        return None
    if isinstance(raw, bool):
        # bool is subclass of int — reject explicitly
        errors.append(f"{field}: expected int or 'none'; got bool {raw!r}")
        return None
    if isinstance(raw, int):
        return raw
    errors.append(f"{field}: expected int or 'none'; got {type(raw).__name__}")
    return None


def _build_and_validate(name: str, body: Mapping[str, Any]) -> EncodingSpec:
    """Construct + validate one EncodingSpec from a TOML table.

    Collects ALL field/invariant violations into a single multi-line
    error (mirrors Rust validator).
    """
    errors: list[str] = []

    def _req(field: str, type_: type) -> Any:
        if field not in body:
            errors.append(f"{field}: missing")
            return None
        v = body[field]
        if type_ is int and isinstance(v, bool):
            errors.append(f"{field}: expected int; got bool {v!r}")
            return None
        if not isinstance(v, type_):
            errors.append(f"{field}: expected {type_.__name__}; got {type(v).__name__}")
            return None
        return v

    board_size = _req("board_size", int)
    trunk_size = _req("trunk_size", int)
    cluster_window_size = _coerce_optional_int(
        "cluster_window_size", body.get("cluster_window_size"), errors
    )
    cluster_threshold = _coerce_optional_int(
        "cluster_threshold", body.get("cluster_threshold"), errors
    )
    legal_move_radius = _req("legal_move_radius", int)
    n_planes = _req("n_planes", int)
    plane_layout_raw = body.get("plane_layout")
    if not isinstance(plane_layout_raw, list) or not all(
        isinstance(p, str) for p in plane_layout_raw
    ):
        errors.append("plane_layout: expected list[str]")
        plane_layout: tuple[str, ...] = ()
    else:
        plane_layout = tuple(plane_layout_raw)

    policy_logit_count = _req("policy_logit_count", int)
    has_pass_slot = _req("has_pass_slot", bool)
    is_multi_window = _req("is_multi_window", bool)
    value_pool = _req("value_pool", str)
    policy_pool = _req("policy_pool", str)
    sym_table_id = _req("sym_table_id", str)
    schema_version = _req("schema_version", int)
    notes = _req("notes", str)

    # §173 A3 — kept_plane_indices + n_source_planes.
    kept_plane_indices_raw = body.get("kept_plane_indices")
    if not isinstance(kept_plane_indices_raw, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in kept_plane_indices_raw
    ):
        errors.append("kept_plane_indices: expected list[int]")
        kept_plane_indices: tuple[int, ...] = ()
    else:
        if any(v < 0 for v in kept_plane_indices_raw):
            errors.append("kept_plane_indices: all values must be >= 0")
            kept_plane_indices = ()
        else:
            kept_plane_indices = tuple(kept_plane_indices_raw)
    n_source_planes = _req("n_source_planes", int)

    validate_pool_enums(value_pool, policy_pool, errors)
    validate_cross_field_consistency(
        plane_layout,
        n_planes,
        board_size,
        policy_logit_count,
        has_pass_slot,
        legal_move_radius,
        errors,
    )
    validate_multi_window_invariants(
        is_multi_window,
        cluster_window_size,
        cluster_threshold,
        value_pool,
        policy_pool,
        trunk_size,
        board_size,
        errors,
    )
    validate_plane_layout(plane_layout, errors)
    validate_kept_plane_indices(
        kept_plane_indices, n_planes, n_source_planes, errors
    )

    if errors:
        raise EncodingRegistryError(
            f"encoding {name!r} validation failed:\n  - " + "\n  - ".join(errors)
        )

    return EncodingSpec(
        name=name,
        board_size=board_size,  # type: ignore[arg-type]
        trunk_size=trunk_size,  # type: ignore[arg-type]
        cluster_window_size=cluster_window_size,
        cluster_threshold=cluster_threshold,
        legal_move_radius=legal_move_radius,  # type: ignore[arg-type]
        n_planes=n_planes,  # type: ignore[arg-type]
        plane_layout=plane_layout,
        policy_logit_count=policy_logit_count,  # type: ignore[arg-type]
        has_pass_slot=has_pass_slot,  # type: ignore[arg-type]
        is_multi_window=is_multi_window,  # type: ignore[arg-type]
        value_pool=value_pool,  # type: ignore[arg-type]
        policy_pool=policy_pool,  # type: ignore[arg-type]
        sym_table_id=sym_table_id,  # type: ignore[arg-type]
        schema_version=schema_version,  # type: ignore[arg-type]
        notes=notes,  # type: ignore[arg-type]
        kept_plane_indices=kept_plane_indices,
        n_source_planes=n_source_planes,  # type: ignore[arg-type]
    )


_REGISTRY_CACHE: dict[str, EncodingSpec] | None = None


def _load() -> dict[str, EncodingSpec]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    raw = tomllib.loads(_REGISTRY_TOML_PATH.read_text())
    if "encodings" not in raw or not isinstance(raw["encodings"], dict):
        raise EncodingRegistryError(
            f"{_REGISTRY_TOML_PATH}: missing top-level [encodings] table"
        )
    specs: dict[str, EncodingSpec] = {}
    errors: list[str] = []
    for name, body in raw["encodings"].items():
        if not isinstance(body, dict):
            errors.append(f"encoding {name!r}: not a TOML table")
            continue
        try:
            specs[name] = _build_and_validate(name, body)
        except EncodingRegistryError as e:
            errors.append(str(e))
    if errors:
        raise EncodingRegistryError(
            "registry validation failed:\n" + "\n".join(errors)
        )
    _REGISTRY_CACHE = specs
    return specs


def lookup(name: str) -> EncodingSpec:
    """Return spec for `name` or raise EncodingRegistryError."""
    registry = _load()
    spec = registry.get(name)
    if spec is None:
        raise EncodingRegistryError(
            f"unknown encoding {name!r}; registered: {sorted(registry)}"
        )
    return spec


def all_specs() -> Iterable[EncodingSpec]:
    """Iterate every registered spec (insertion order — TOML preserves)."""
    return _load().values()
