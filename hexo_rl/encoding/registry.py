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

import functools
import pathlib
import tomllib
from typing import Any, Iterable, Mapping

from hexo_rl.encoding.spec import EncodingSpec, PolicyPool, ValuePool


_VALID_VALUE_POOLS = ("none", "min", "max", "mean")
_VALID_POLICY_POOLS = ("none", "scatter_max", "scatter_mean")
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

    if value_pool is not None and value_pool not in _VALID_VALUE_POOLS:
        errors.append(
            f"value_pool: must be one of {list(_VALID_VALUE_POOLS)}; got {value_pool!r}"
        )
    if policy_pool is not None and policy_pool not in _VALID_POLICY_POOLS:
        errors.append(
            f"policy_pool: must be one of {list(_VALID_POLICY_POOLS)}; got {policy_pool!r}"
        )

    # Cross-field invariants (mirror Rust validator).
    if plane_layout and n_planes is not None and len(plane_layout) != n_planes:
        errors.append(
            f"len(plane_layout)={len(plane_layout)} != n_planes={n_planes}"
        )

    if board_size is not None and policy_logit_count is not None and has_pass_slot is not None:
        expected_logits = board_size * board_size + (1 if has_pass_slot else 0)
        if policy_logit_count != expected_logits:
            errors.append(
                f"policy_logit_count={policy_logit_count} != board_size²+(pass_slot?1:0)"
                f"={expected_logits} (board_size={board_size}, has_pass_slot={has_pass_slot})"
            )

    if is_multi_window is not None:
        cw_some = cluster_window_size is not None
        ct_some = cluster_threshold is not None
        if cw_some != is_multi_window:
            errors.append(
                f"is_multi_window={is_multi_window} != cluster_window_size.is_some()={cw_some}"
            )
        if ct_some != is_multi_window:
            errors.append(
                f"is_multi_window={is_multi_window} != cluster_threshold.is_some()={ct_some}"
            )
        if is_multi_window:
            if value_pool not in ("min", "max", "mean"):
                errors.append(
                    f"is_multi_window=true requires value_pool ∈ {{min,max,mean}}; got {value_pool!r}"
                )
            if policy_pool not in ("scatter_max", "scatter_mean"):
                errors.append(
                    f"is_multi_window=true requires policy_pool ∈ {{scatter_max,scatter_mean}}; "
                    f"got {policy_pool!r}"
                )
            if cluster_window_size is not None and trunk_size is not None and trunk_size != cluster_window_size:
                errors.append(
                    f"is_multi_window=true requires trunk_size==cluster_window_size; "
                    f"got trunk_size={trunk_size}, cluster_window_size={cluster_window_size}"
                )
        else:
            if value_pool != "none":
                errors.append(
                    f"is_multi_window=false requires value_pool='none'; got {value_pool!r}"
                )
            if policy_pool != "none":
                errors.append(
                    f"is_multi_window=false requires policy_pool='none'; got {policy_pool!r}"
                )
            if trunk_size is not None and board_size is not None and trunk_size != board_size:
                errors.append(
                    f"is_multi_window=false requires trunk_size==board_size; "
                    f"got trunk_size={trunk_size}, board_size={board_size}"
                )

    if legal_move_radius is not None and legal_move_radius <= 0:
        errors.append("legal_move_radius must be > 0")

    # plane_layout no-empty + no-dup
    seen: set[str] = set()
    for idx, p in enumerate(plane_layout):
        if not p:
            errors.append(f"plane_layout[{idx}] is empty")
        if p in seen:
            errors.append(f"plane_layout[{idx}] duplicate name {p!r}")
        seen.add(p)

    # §173 A3 — kept_plane_indices validators (mirror Rust spec.rs).
    if kept_plane_indices and n_planes is not None:
        # 3.1: len == n_planes
        if len(kept_plane_indices) != n_planes:
            errors.append(
                f"len(kept_plane_indices)={len(kept_plane_indices)} != n_planes={n_planes}"
            )
        # 3.2: no duplicates
        if len(set(kept_plane_indices)) != len(kept_plane_indices):
            errors.append("kept_plane_indices: duplicate index")
        # 3.3: max < n_source_planes
        if n_source_planes is not None and kept_plane_indices:
            max_idx = max(kept_plane_indices)
            if max_idx >= n_source_planes:
                errors.append(
                    f"kept_plane_indices: max index {max_idx} >= n_source_planes={n_source_planes}"
                )
        # 3.5: n_source_planes >= n_planes
        if n_source_planes is not None and n_planes is not None and n_source_planes < n_planes:
            errors.append(
                f"n_source_planes={n_source_planes} < n_planes={n_planes} "
                f"(kept set must be a subset of source)"
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


@functools.cache
def _load() -> dict[str, EncodingSpec]:
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
