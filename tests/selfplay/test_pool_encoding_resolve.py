"""§176 P20 — unit test for ``_resolve_encoding_for_pool``.

WorkerPool.__init__ used to inline a ~65-LOC encoding-dispatch block
resolving the canonical registry spec, the legacy 4-field PyEncodingSpec
shim, the PyO3 runner kwargs, and the model.board_size cross-check.
That block was extracted to ``_resolve_encoding_for_pool(config, model)``
so it can be exercised in isolation per registered encoding.

Cycle 3 Wave 8 Batch C (FF.10, 2026-05-17): the WireFormatSpec +
PyO3 ``encoding_spec=PyRegistrySpec`` round-trip retired. The Rust
runner now takes a registry name string via ``encoding_name=`` and
resolves the record once on the Rust side. ``ResolvedPoolEncoding``
drops the ``wire_format_spec`` + ``runner_registry_spec`` fields and
exposes ``encoding_name`` directly.

Parametrized over every encoding in the registry (v6, v6w25, v7full,
v7, v7e30, v7mw, v8, v8_canvas_realness):

  - v6 / v6w25 / v7full / v7 / v7e30 / v7mw → returns ResolvedPoolEncoding
                                     with registry-derived shape.
  - v8 / v8_canvas_realness        → NotImplementedError loud-fail
                                     (no Rust runner path; pretrain
                                     via dataset_v8.py).

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


# Pre-classify every registered encoding into two buckets. Done
# at import time so the parametrize ids are stable and a registry add
# surfaces here on test-collection rather than at runtime.
#
# §176 P3: WIRE_FORMAT_SPECS covers every registered name; the previous
# "legacy resolver doesn't know v7 / v7e30" bucket is gone.
_V8_NAMES = ("v8", "v8_canvas_realness")


def _ok_names() -> List[str]:
    return [s.name for s in all_specs() if s.name not in _V8_NAMES]


def _v8_names() -> List[str]:
    return [s.name for s in all_specs() if s.name in _V8_NAMES]


# --------------------------------------------------------------------------- #
# Successful dispatch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _ok_names())
def test_resolve_encoding_for_pool_returns_expected_shape(name: str) -> None:
    """Each ok encoding returns a ResolvedPoolEncoding whose scalar
    fields match the registry; ``encoding_name`` round-trips.

    Cycle 3 Wave 8 Batch C (FF.10): `wire_format_spec` /
    `runner_registry_spec` fields retired; the Rust runner takes the
    registry name string directly via ``encoding_name=``.
    """
    spec = lookup(name)
    cfg: Dict[str, Any] = {"encoding": name}

    r = _resolve_encoding_for_pool(cfg)

    assert isinstance(r, ResolvedPoolEncoding)
    assert r.registry_spec.name == name
    assert r.board_size == spec.board_size
    assert r.trunk_size == spec.trunk_size
    assert r.n_kept_planes == len(spec.kept_plane_indices)
    # `encoding_name` is the single Rust-bound surface post-FF.10.
    assert r.encoding_name == name


# --------------------------------------------------------------------------- #
# v8 loud-fail
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", _v8_names())
def test_resolve_encoding_for_pool_v8_loud_fails(name: str) -> None:
    """v8 selfplay path is intentionally blocked at the helper — the
    Rust runner has no v8 dispatch and silently routing through
    PyEncodingSpec construction would crash with an obscure ValueError."""
    cfg = {"encoding": name}
    with pytest.raises(NotImplementedError, match=r"v8 selfplay"):
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
    if they don't land in one of the test buckets, prompting a test
    update."""
    classified = set(_ok_names()) | set(_v8_names())
    registered = {s.name for s in all_specs()}
    missing = registered - classified
    assert not missing, (
        f"unclassified encodings {missing!r}; update test buckets"
    )
