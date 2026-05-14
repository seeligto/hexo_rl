"""Unified replay_game_to_triples dispatch — §176 P77.

Verifies the new ReplayTriples wrapper at hexo_rl/bootstrap/replay_triples.py
routes by encoding_spec.name and preserves byte-exact arrays vs the legacy
per-encoding fns.
"""
from __future__ import annotations

import numpy as np
import pytest

from hexo_rl.encoding import lookup
from hexo_rl.bootstrap.replay_triples import (
    ReplayTriples,
    replay_game_to_triples,
)


# Short legal games matching the construction in the per-encoding tests.
# v6 first move is ply-0 single; both players alternate 2 moves per turn.
_V6_MOVES = [
    (0, 0),                # P1 ply-0 single
    (0, 1), (1, 0),        # P2 turn 1
    (1, 1), (2, 0),        # P1 turn 2
]
_V6W25_MOVES = list(_V6_MOVES)


def test_v6_dispatch_shapes():
    triples = replay_game_to_triples(
        _V6_MOVES, winner=1, encoding_spec=lookup("v6")
    )
    assert isinstance(triples, ReplayTriples)
    T = triples.states.shape[0]
    assert T > 0
    assert triples.states.shape == (T, 18, 19, 19)
    assert triples.chain_planes.shape == (T, 6, 19, 19)
    assert triples.policies.shape == (T, 362)
    assert triples.outcomes.shape == (T,)
    assert triples.global_crops is None
    assert triples.n_clipped is None


def test_v6_byte_parity_with_legacy():
    """Unified dispatch returns the exact same arrays as the legacy v6 fn."""
    from hexo_rl.bootstrap.dataset import replay_game_to_triples as _legacy

    legacy = _legacy(_V6_MOVES, winner=1)
    unified = replay_game_to_triples(
        _V6_MOVES, winner=1, encoding_spec=lookup("v6")
    )
    assert np.array_equal(unified.states, legacy[0])
    assert np.array_equal(unified.chain_planes, legacy[1])
    assert np.array_equal(unified.policies, legacy[2])
    assert np.array_equal(unified.outcomes, legacy[3])


def test_v6w25_dispatch_default_no_global_crop():
    triples = replay_game_to_triples(
        _V6W25_MOVES, winner=1, encoding_spec=lookup("v6w25")
    )
    T = triples.states.shape[0]
    assert T > 0
    assert triples.states.shape == (T, 8, 25, 25)
    assert triples.chain_planes.shape == (T, 6, 25, 25)
    assert triples.policies.shape == (T, 626)
    assert triples.outcomes.shape == (T,)
    assert triples.global_crops is None
    assert triples.n_clipped is None


def test_v6w25_dispatch_with_global_crop():
    triples = replay_game_to_triples(
        _V6W25_MOVES,
        winner=1,
        encoding_spec=lookup("v6w25"),
        with_global_crop=True,
    )
    T = triples.states.shape[0]
    assert T > 0
    assert triples.global_crops is not None
    assert triples.global_crops.shape == (T, 3, 32, 32)
    assert triples.global_crops.dtype == np.float16


def test_v6w25_byte_parity_with_legacy():
    from hexo_rl.bootstrap.dataset_v6w25 import (
        replay_game_to_triples_v6w25 as _legacy,
    )

    legacy_default = _legacy(_V6W25_MOVES, winner=1)
    unified_default = replay_game_to_triples(
        _V6W25_MOVES, winner=1, encoding_spec=lookup("v6w25")
    )
    assert np.array_equal(unified_default.states, legacy_default[0])
    assert np.array_equal(unified_default.chain_planes, legacy_default[1])
    assert np.array_equal(unified_default.policies, legacy_default[2])
    assert np.array_equal(unified_default.outcomes, legacy_default[3])

    legacy_gc = _legacy(_V6W25_MOVES, winner=1, with_global_crop=True)
    unified_gc = replay_game_to_triples(
        _V6W25_MOVES,
        winner=1,
        encoding_spec=lookup("v6w25"),
        with_global_crop=True,
    )
    assert np.array_equal(unified_gc.global_crops, legacy_gc[4])


def test_v8_dispatch_returns_n_clipped():
    triples = replay_game_to_triples(
        [(0, 0)], winner=1, encoding_spec=lookup("v8")
    )
    assert triples.states.shape == (1, 11, 25, 25)
    assert triples.chain_planes.shape == (1, 6, 25, 25)
    assert triples.policies.shape == (1, 625)
    assert triples.outcomes.shape == (1,)
    assert triples.n_clipped is not None
    assert triples.n_clipped >= 0
    assert triples.global_crops is None


def test_v8_canvas_realness_dispatch_implies_flag():
    """v8_canvas_realness spec name forces canvas_realness=True regardless
    of the kwarg, matching the dispatch semantics."""
    from hexo_rl.bootstrap.dataset_v8 import (
        replay_game_to_triples_v8 as _legacy,
    )

    moves = [(0, 0)]
    legacy_real = _legacy(moves, winner=1, canvas_realness=True)
    unified_real = replay_game_to_triples(
        moves, winner=1, encoding_spec=lookup("v8_canvas_realness")
    )
    assert np.array_equal(unified_real.states, legacy_real[0])
    assert unified_real.n_clipped == legacy_real[4]


def test_v8_byte_parity_with_legacy():
    from hexo_rl.bootstrap.dataset_v8 import (
        replay_game_to_triples_v8 as _legacy,
    )

    moves = [(0, 0)]
    legacy = _legacy(moves, winner=1, canvas_realness=False)
    unified = replay_game_to_triples(
        moves, winner=1, encoding_spec=lookup("v8")
    )
    assert np.array_equal(unified.states, legacy[0])
    assert np.array_equal(unified.chain_planes, legacy[1])
    assert np.array_equal(unified.policies, legacy[2])
    assert np.array_equal(unified.outcomes, legacy[3])
    assert unified.n_clipped == legacy[4]


def test_unknown_encoding_raises():
    fake_spec = type("FakeSpec", (), {"name": "v999_unknown"})()
    with pytest.raises(ValueError, match="no replayer"):
        replay_game_to_triples(
            [(0, 0)], winner=1, encoding_spec=fake_spec
        )
