"""§171 P2 reopen — engine.EncodingSpec PyO3 binding + Board.with_encoding.

Tests cover:
- engine.EncodingSpec construction with kwargs (validation in Rust).
- Board.with_encoding produces a board with the correct window / threshold / radius.
- Default Board() and Board.with_encoding(v6_spec) are encoding-equivalent.
- Python EncodingSpec.to_pyo3() round-trip for v6 and v6w25.
- v8 EncodingSpec.to_pyo3() raises ValueError (no cluster plumbing).
- Validation rejects bad inputs.
"""
from __future__ import annotations

import pytest

from engine import Board, EncodingSpec as PyEncodingSpec
from hexo_rl.utils.encoding import (
    EncodingSpec,
    resolve_encoding,
    v6_spec,
    v6w25_spec,
    v8_spec,
)


def test_pyencoding_kwargs_construction():
    spec = PyEncodingSpec(
        cluster_window_size=25, cluster_threshold=8, legal_move_radius=8, board_size=19
    )
    assert spec.cluster_window_size == 25
    assert spec.cluster_threshold == 8
    assert spec.legal_move_radius == 8
    assert spec.board_size == 19


def test_pyboard_with_encoding_v6w25_constructs():
    spec = PyEncodingSpec(
        cluster_window_size=25, cluster_threshold=8, legal_move_radius=8, board_size=19
    )
    b = Board.with_encoding(spec)
    assert b.cluster_window_size() == 25
    assert b.cluster_threshold() == 8
    assert b.legal_move_radius() == 8


def test_pyboard_with_encoding_v6_matches_default():
    spec = PyEncodingSpec(
        cluster_window_size=19, cluster_threshold=5, legal_move_radius=5, board_size=19
    )
    b_with = Board.with_encoding(spec)
    b_new = Board()
    assert b_with.cluster_window_size() == b_new.cluster_window_size()
    assert b_with.cluster_threshold() == b_new.cluster_threshold()
    assert b_with.legal_move_radius() == b_new.legal_move_radius()


def test_pyboard_with_encoding_validates_inputs():
    with pytest.raises(ValueError, match="cluster_window_size"):
        PyEncodingSpec(
            cluster_window_size=20,  # even
            cluster_threshold=5, legal_move_radius=5, board_size=19,
        )
    with pytest.raises(ValueError):
        PyEncodingSpec(
            cluster_window_size=7, cluster_threshold=9,  # window < threshold
            legal_move_radius=5, board_size=19,
        )


def test_to_pyo3_v6_round_trip():
    py = v6_spec().to_pyo3()
    assert py.cluster_window_size == 19
    assert py.cluster_threshold == 5
    assert py.legal_move_radius == 5
    assert py.board_size == 19


def test_to_pyo3_v6w25_round_trip():
    py = v6w25_spec().to_pyo3()
    assert py.cluster_window_size == 25
    assert py.cluster_threshold == 8
    assert py.legal_move_radius == 8


def test_to_pyo3_v8_raises():
    with pytest.raises(ValueError, match="cluster"):
        v8_spec().to_pyo3()


def test_resolve_encoding_v6w25():
    spec = resolve_encoding({"encoding": {"version": "v6w25"}})
    assert spec.version == "v6w25"
    assert spec.cluster_window_size == 25


def test_python_namedtuple_has_cluster_window_size():
    assert v6_spec().cluster_window_size == 19
    assert v6w25_spec().cluster_window_size == 25
    assert v8_spec().cluster_window_size is None
