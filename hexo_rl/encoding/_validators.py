"""Per-stage validators for the encoding registry.

Extracted from `registry._build_and_validate` (§176 P61). Each function
appends violations to the caller's `errors: list[str]` accumulator
(mirrors the Rust validator's collect-all-then-raise pattern). Error
message format is preserved byte-identical — tests substring-pin the
strings.

Call order is fixed by `_build_and_validate`; see that function for the
canonical sequence.
"""
from __future__ import annotations

from typing import Sequence


_VALID_VALUE_POOLS = ("none", "min", "max", "mean")
_VALID_POLICY_POOLS = ("none", "scatter_max", "scatter_mean")


def validate_pool_enums(
    value_pool: str | None,
    policy_pool: str | None,
    errors: list[str],
) -> None:
    """value_pool / policy_pool must be members of the legal enum sets."""
    if value_pool is not None and value_pool not in _VALID_VALUE_POOLS:
        errors.append(
            f"value_pool: must be one of {list(_VALID_VALUE_POOLS)}; got {value_pool!r}"
        )
    if policy_pool is not None and policy_pool not in _VALID_POLICY_POOLS:
        errors.append(
            f"policy_pool: must be one of {list(_VALID_POLICY_POOLS)}; got {policy_pool!r}"
        )


def validate_cross_field_consistency(
    plane_layout: Sequence[str],
    n_planes: int | None,
    board_size: int | None,
    policy_logit_count: int | None,
    has_pass_slot: bool | None,
    legal_move_radius: int | None,
    errors: list[str],
) -> None:
    """plane_layout len / policy_logit_count / legal_move_radius invariants."""
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

    if legal_move_radius is not None and legal_move_radius <= 0:
        errors.append("legal_move_radius must be > 0")


def validate_multi_window_invariants(
    is_multi_window: bool | None,
    cluster_window_size: int | None,
    cluster_threshold: int | None,
    value_pool: str | None,
    policy_pool: str | None,
    trunk_size: int | None,
    board_size: int | None,
    errors: list[str],
) -> None:
    """is_multi_window coverage: presence of cluster_* + pool / trunk constraints."""
    if is_multi_window is None:
        return

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


def validate_plane_layout(
    plane_layout: Sequence[str],
    errors: list[str],
) -> None:
    """plane_layout: no empty names, no duplicates."""
    seen: set[str] = set()
    for idx, p in enumerate(plane_layout):
        if not p:
            errors.append(f"plane_layout[{idx}] is empty")
        if p in seen:
            errors.append(f"plane_layout[{idx}] duplicate name {p!r}")
        seen.add(p)


def validate_kept_plane_indices(
    kept_plane_indices: Sequence[int],
    n_planes: int | None,
    n_source_planes: int | None,
    errors: list[str],
) -> None:
    """§173 A3 — kept_plane_indices invariants (mirror Rust spec.rs)."""
    if not (kept_plane_indices and n_planes is not None):
        return

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
