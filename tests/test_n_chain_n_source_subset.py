"""§173 A3 — Runtime invariant tests for kept_plane_indices across live registry.

Belt-and-braces: same as validator 3.3 + 3.5, but asserted at runtime
across the live registry rather than parse-time. Ensures the registry
TOML and parser stay in sync across all 5 encodings.
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import all_specs, lookup


_ENCODING_NAMES = sorted(s.name for s in all_specs())


@pytest.mark.parametrize("name", _ENCODING_NAMES)
def test_kept_indices_subset_of_source(name: str) -> None:
    """max(kept_plane_indices) < n_source_planes for all registered encodings."""
    spec = lookup(name)
    assert len(spec.kept_plane_indices) > 0, (
        f"{name}: kept_plane_indices is empty"
    )
    assert max(spec.kept_plane_indices) < spec.n_source_planes, (
        f"{name}: max(kept_plane_indices)={max(spec.kept_plane_indices)} "
        f">= n_source_planes={spec.n_source_planes}"
    )
    assert len(spec.kept_plane_indices) <= spec.n_source_planes, (
        f"{name}: len(kept_plane_indices)={len(spec.kept_plane_indices)} "
        f"> n_source_planes={spec.n_source_planes}"
    )


@pytest.mark.parametrize("name", _ENCODING_NAMES)
def test_kept_indices_len_equals_n_planes(name: str) -> None:
    """len(kept_plane_indices) == n_planes (also enforced by validator — belt-and-braces)."""
    spec = lookup(name)
    assert len(spec.kept_plane_indices) == spec.n_planes, (
        f"{name}: len(kept_plane_indices)={len(spec.kept_plane_indices)} "
        f"!= n_planes={spec.n_planes}"
    )


@pytest.mark.parametrize("name", _ENCODING_NAMES)
def test_kept_indices_no_duplicates(name: str) -> None:
    """No duplicate indices in kept_plane_indices."""
    spec = lookup(name)
    indices = spec.kept_plane_indices
    assert len(set(indices)) == len(indices), (
        f"{name}: kept_plane_indices has duplicates: {indices}"
    )


@pytest.mark.parametrize("name", _ENCODING_NAMES)
def test_n_source_planes_at_least_n_planes(name: str) -> None:
    """n_source_planes >= n_planes (kept set is a subset)."""
    spec = lookup(name)
    assert spec.n_source_planes >= spec.n_planes, (
        f"{name}: n_source_planes={spec.n_source_planes} < n_planes={spec.n_planes}"
    )


def test_v6_family_has_18_source_planes() -> None:
    """v6, v7full, v6w25 all have n_source_planes=18 (v6 source tensor)."""
    for name in ("v6", "v7full", "v6w25"):
        spec = lookup(name)
        assert spec.n_source_planes == 18, (
            f"{name}: n_source_planes={spec.n_source_planes}, expected 18"
        )


def test_v8_family_has_21_source_planes() -> None:
    """v8, v8_canvas_realness have n_source_planes=21 (v8 source tensor)."""
    for name in ("v8", "v8_canvas_realness"):
        spec = lookup(name)
        assert spec.n_source_planes == 21, (
            f"{name}: n_source_planes={spec.n_source_planes}, expected 21"
        )


def test_v6_family_canonical_indices() -> None:
    """v6 family: canonical [0,1,2,3,8,9,10,11] X+history, O+history block."""
    expected = (0, 1, 2, 3, 8, 9, 10, 11)
    for name in ("v6", "v7full", "v6w25"):
        spec = lookup(name)
        assert spec.kept_plane_indices == expected, (
            f"{name}: kept_plane_indices={spec.kept_plane_indices}, "
            f"expected {expected}"
        )


def test_v8_family_canonical_indices() -> None:
    """v8 family: canonical [0,1,2,3,8,9,10,11,18,19,20]."""
    expected = (0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 20)
    for name in ("v8", "v8_canvas_realness"):
        spec = lookup(name)
        assert spec.kept_plane_indices == expected, (
            f"{name}: kept_plane_indices={spec.kept_plane_indices}, "
            f"expected {expected}"
        )
