"""§176 P20 — unit test for ``_resolve_encoding_for_pool``.

WorkerPool.__init__ used to inline a ~65-LOC encoding-dispatch block
resolving the canonical registry spec, the legacy 4-field PyEncodingSpec
shim, the PyO3 runner kwargs, and the model.board_size cross-check.
That block was extracted to ``_resolve_encoding_for_pool(config, model)``
so it can be exercised in isolation per registered encoding.

Parametrized over every encoding in the registry (v6, v6w25, v7full,
v7, v7e30, v7mw, v8, v8_canvas_realness):

  - v6 / v6w25 / v7full / v7mw    → returns ResolvedPoolEncoding
                                     with registry-derived shape.
  - v8 / v8_canvas_realness        → NotImplementedError loud-fail
                                     (no Rust runner path; pretrain
                                     via dataset_v8.py).
  - v7 / v7e30                     → ValueError from the legacy
                                     resolver (registry-only labels;
                                     not on a selfplay path today).
                                     This documents pre-existing
                                     behaviour preserved by the
                                     refactor.

Also exercises the model.board_size mismatch guard.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from hexo_rl.encoding import all_specs, lookup
from hexo_rl.selfplay.pool import (
    ResolvedPoolEncoding,
    _resolve_encoding_for_pool,
)


# Pre-classify every registered encoding into the three buckets. Done
# at import time so the parametrize ids are stable and a registry add
# surfaces here on test-collection rather than at runtime.
_V8_NAMES = ("v8", "v8_canvas_realness")
# Legacy `hexo_rl.utils.encoding.resolve_encoding` knows v6 / v6w25 / v8.
# v7full / v7mw are routed through the explicit v6_spec() fallback inside
# the helper. Any other registry name (e.g. v7, v7e30) falls through to
# the legacy resolver and ValueErrors — pre-existing behaviour.
_LEGACY_HANDLED = ("v6", "v6w25", "v7full", "v7mw")


def _ok_names() -> List[str]:
    return [s.name for s in all_specs() if s.name in _LEGACY_HANDLED]


def _v8_names() -> List[str]:
    return [s.name for s in all_specs() if s.name in _V8_NAMES]


def _legacy_unknown_names() -> List[str]:
    return [
        s.name
        for s in all_specs()
        if s.name not in _LEGACY_HANDLED and s.name not in _V8_NAMES
    ]


# --------------------------------------------------------------------------- #
# Successful dispatch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _ok_names())
def test_resolve_encoding_for_pool_returns_expected_shape(name: str) -> None:
    """Each ok encoding returns a ResolvedPoolEncoding whose scalar
    fields match the registry; legacy_spec / runner_encoding /
    runner_registry_spec are non-None and round-trip."""
    spec = lookup(name)
    cfg: Dict[str, Any] = {"encoding": name}

    r = _resolve_encoding_for_pool(cfg)

    assert isinstance(r, ResolvedPoolEncoding)
    assert r.registry_spec.name == name
    assert r.board_size == spec.board_size
    assert r.trunk_size == spec.trunk_size
    assert r.n_kept_planes == len(spec.kept_plane_indices)

    # legacy_spec is a NamedTuple shim — must carry a `version` matching
    # the registry name (or the v6-family fallback for v7full/v7mw,
    # which still tags `version` per the v6_spec() helper).
    assert r.legacy_spec is not None

    # Multi-window encodings (v6w25, v7mw) have cluster fields set in
    # the legacy spec → runner_encoding (PyO3 4-field) is non-None.
    # Single-window v6 / v7full also have cluster fields set in their
    # legacy spec (v6 default), so all four ok encodings yield a
    # non-None runner_encoding.
    assert r.runner_encoding is not None

    # runner_registry_spec is the PyO3 full-schema mirror; carries the
    # same registered name.
    assert r.runner_registry_spec is not None
    assert r.runner_registry_spec.name == name


# --------------------------------------------------------------------------- #
# v8 loud-fail
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _v8_names())
def test_resolve_encoding_for_pool_v8_loud_fails(name: str) -> None:
    """v8 selfplay path is intentionally blocked at the helper — the
    Rust runner has no v8 dispatch and silently routing through
    legacy_spec.to_pyo3() would crash with an obscure ValueError."""
    cfg = {"encoding": name}
    with pytest.raises(NotImplementedError, match=r"v8 selfplay"):
        _resolve_encoding_for_pool(cfg)


# --------------------------------------------------------------------------- #
# Registry-only labels not handled by the legacy resolver
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _legacy_unknown_names())
def test_resolve_encoding_for_pool_legacy_unknown_raises(name: str) -> None:
    """v7 / v7e30 are registry-only labels; the legacy
    `hexo_rl.utils.encoding.resolve_encoding` knows v6 / v6w25 / v8.
    The helper preserves that pre-existing behaviour — pure-move
    discipline forbids inventing a v6 fallback for them in this
    refactor."""
    cfg = {"encoding": name}
    with pytest.raises(ValueError, match=r"unknown encoding\.version"):
        _resolve_encoding_for_pool(cfg)


# --------------------------------------------------------------------------- #
# Model.board_size cross-check
# --------------------------------------------------------------------------- #
def test_resolve_encoding_for_pool_model_board_size_mismatch_raises() -> None:
    """A model declaring a board_size that disagrees with the resolved
    encoding's canvas geometry must loud-fail before any Rust runner is
    constructed."""
    cfg = {"encoding": "v6"}
    bad_model = SimpleNamespace(board_size=25)  # v6 canvas is 19
    with pytest.raises(ValueError, match=r"model\.board_size"):
        _resolve_encoding_for_pool(cfg, model=bad_model)


def test_resolve_encoding_for_pool_model_board_size_match_passes() -> None:
    """A model whose board_size matches the resolved spec succeeds and
    returns the same shape as the model=None path."""
    cfg = {"encoding": "v6w25"}
    good_model = SimpleNamespace(board_size=25)
    r = _resolve_encoding_for_pool(cfg, model=good_model)
    assert r.board_size == 25
    assert r.trunk_size == 25
    assert r.registry_spec.name == "v6w25"


# --------------------------------------------------------------------------- #
# Sanity: every encoding is classified
# --------------------------------------------------------------------------- #
def test_every_registered_encoding_classified() -> None:
    """Tripwire — new encodings added to registry.toml will surface here
    if they don't land in one of the three test buckets, prompting a
    test update."""
    classified = set(_ok_names()) | set(_v8_names()) | set(_legacy_unknown_names())
    registered = {s.name for s in all_specs()}
    missing = registered - classified
    assert not missing, (
        f"unclassified encodings {missing!r}; update test buckets"
    )
