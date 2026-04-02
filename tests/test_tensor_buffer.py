"""
test_tensor_buffer.py — isolated tests for TensorBuffer.assemble().

All tests use mock 2-plane NumPy arrays (simulating Rust's get_cluster_views output)
instead of real Board objects. This lets us verify the history assembly logic in
complete isolation from board logic, win detection, and PyO3 bindings.

Run with: pytest tests/test_tensor_buffer.py -v
"""
from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

from python.env.game_state import GameState, HISTORY_LEN
from python.selfplay.tensor_buffer import TensorBuffer

BOARD_SIZE = 19


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_views(K: int, fill_my: float, fill_opp: float) -> list:
    """Return K fake (2, 19, 19) float32 view arrays simulating get_cluster_views output.

    fill_my fills plane 0 (current player's stones); fill_opp fills plane 1 (opponent).
    Each cluster k adds k*0.01 to both fills so clusters are distinguishable.
    """
    views = []
    for k in range(K):
        v = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        v[0] = fill_my + k * 0.01
        v[1] = fill_opp + k * 0.01
        views.append(v)
    return views


def _make_mock_state(
    K: int = 1,
    ply: int = 0,
    moves_remaining: int = 2,
    history: deque | None = None,
    fill_my: float = 1.0,
    fill_opp: float = 2.0,
) -> MagicMock:
    """Return a MagicMock that behaves like a GameState for TensorBuffer.assemble()."""
    state = MagicMock(spec=GameState)
    state.views = _make_mock_views(K, fill_my, fill_opp)
    state.centers = [(0, 0)] * K
    state.ply = ply
    state.moves_remaining = moves_remaining
    state.move_history = history if history is not None else deque(maxlen=HISTORY_LEN)
    return state


def _make_history_chain(n_steps: int) -> list:
    """Return a list of n_steps mock states with strictly distinct fill values.

    State i has fill_my = float(i+1) and fill_opp = float(i+100).
    Each state's move_history contains all prior states, mirroring how
    GameState.apply_move() builds the deque.
    """
    states: list = []
    for i in range(n_steps):
        hist: deque = deque(states, maxlen=HISTORY_LEN)
        s = _make_mock_state(K=1, ply=i, fill_my=float(i + 1), fill_opp=float(i + 100))
        s.move_history = hist
        states.append(s)
    return states


# ── Initial assembly (rebuild path) ──────────────────────────────────────────

class TestTensorBufferInitialAssembly:
    """Given: a fresh TensorBuffer and a state with no history.
    When: assemble() is called for the first time.
    Then: output has the correct shape, dtype, and plane layout.
    """

    def test_output_shape_single_cluster(self) -> None:
        buf = TensorBuffer()
        state = _make_mock_state(K=1)
        out, centers = buf.assemble(state)
        assert out.shape == (1, 18, BOARD_SIZE, BOARD_SIZE)

    def test_output_dtype_is_float16(self) -> None:
        buf = TensorBuffer()
        state = _make_mock_state(K=1)
        out, _ = buf.assemble(state)
        assert out.dtype == np.float16

    def test_output_shape_two_clusters(self) -> None:
        buf = TensorBuffer()
        state = _make_mock_state(K=2)
        out, _ = buf.assemble(state)
        assert out.shape == (2, 18, BOARD_SIZE, BOARD_SIZE)

    def test_centers_returned_unchanged(self) -> None:
        expected_centers = [(3, 5), (-1, 2)]
        state = _make_mock_state(K=2)
        state.centers = expected_centers
        buf = TensorBuffer()
        _, centers = buf.assemble(state)
        assert centers == expected_centers

    def test_plane_0_is_current_my_stones(self) -> None:
        """Plane 0 must hold the current player's stones from state.views[k][0]."""
        buf = TensorBuffer()
        state = _make_mock_state(K=1, fill_my=7.0, fill_opp=0.5)
        out, _ = buf.assemble(state)
        expected = state.views[0][0].astype(np.float16)
        np.testing.assert_allclose(
            out[0, 0], expected, rtol=1e-2,
            err_msg="plane 0 must hold current my-stones from state.views[0][0]",
        )

    def test_plane_8_is_current_opp_stones(self) -> None:
        """Plane 8 must hold the opponent's stones from state.views[k][1]."""
        buf = TensorBuffer()
        state = _make_mock_state(K=1, fill_my=1.0, fill_opp=5.0)
        out, _ = buf.assemble(state)
        expected = state.views[0][1].astype(np.float16)
        np.testing.assert_allclose(
            out[0, 8], expected, rtol=1e-2,
            err_msg="plane 8 must hold current opp-stones from state.views[0][1]",
        )

    def test_history_planes_zero_when_no_history(self) -> None:
        """History planes 1-7 and 9-15 must be zero when move_history is empty."""
        buf = TensorBuffer()
        state = _make_mock_state(K=1)
        out, _ = buf.assemble(state)
        for t in range(1, 8):
            assert out[0, t].max() == 0.0, \
                f"my-stones history plane {t} must be zero on first call with no history"
            assert out[0, 8 + t].max() == 0.0, \
                f"opp-stones history plane {8 + t} must be zero on first call with no history"

    @pytest.mark.parametrize("moves_remaining,expected", [
        (1, 0.0),   # mid-turn (only 1 move left) → flag OFF
        (2, 1.0),   # start of 2-stone turn → flag ON
    ])
    def test_plane_16_moves_remaining_flag(self, moves_remaining: int, expected: float) -> None:
        """Plane 16 encodes whether we are at the start of a 2-stone turn."""
        buf = TensorBuffer()
        state = _make_mock_state(K=1, moves_remaining=moves_remaining)
        out, _ = buf.assemble(state)
        assert float(out[0, 16, 0, 0]) == pytest.approx(expected, abs=0.01), \
            f"plane 16 with moves_remaining={moves_remaining} must be {expected}"

    @pytest.mark.parametrize("ply,expected", [(0, 0.0), (1, 1.0), (2, 0.0), (7, 1.0)])
    def test_plane_17_ply_parity(self, ply: int, expected: float) -> None:
        """Plane 17 encodes ply % 2 (which player moved on this timestep)."""
        buf = TensorBuffer()
        state = _make_mock_state(K=1, ply=ply)
        out, _ = buf.assemble(state)
        assert float(out[0, 17, 0, 0]) == pytest.approx(expected, abs=0.01), \
            f"plane 17 with ply={ply} must be {expected}"


# ── Rolling history assembly (shift path) ─────────────────────────────────────

class TestTensorBufferHistoryAssembly:
    """Given: a TensorBuffer called on a chain of states.
    When: assemble() is called repeatedly with the same K.
    Then: history planes accumulate via circular shift without data leaks.
    """

    def test_plane_1_reflects_t_minus_1_after_two_calls(self) -> None:
        """After 2 calls, plane 1 (my-stones t-1) must equal the first state's view."""
        states = _make_history_chain(2)
        buf = TensorBuffer()
        buf.assemble(states[0])          # prime: buf[0, 0] = fill_my=1.0
        out, _ = buf.assemble(states[1])  # shift: buf[0, 1] ← buf[0, 0] = 1.0

        # states[1].move_history[-1] is states[0] with fill_my=1.0
        expected = states[1].move_history[-1].views[0][0].astype(np.float16)
        np.testing.assert_allclose(
            out[0, 1], expected, rtol=1e-2,
            err_msg="plane 1 (t-1 my-stones) must hold the previous call's plane 0",
        )

    def test_circular_shift_correct_after_eight_calls(self) -> None:
        """After 8 calls each plane t should contain data from step (8-t) back.

        Fill values are 1..8 (states[0]..states[7]).
        After 8 calls:
          plane 0 = 8.0   (most recent)
          plane 1 = 7.0   (one step older)
          ...
          plane 7 = 1.0   (oldest in window)
        """
        states = _make_history_chain(8)
        buf = TensorBuffer()
        out = None
        for s in states:
            out, _ = buf.assemble(s)

        assert out is not None
        for t in range(1, 8):
            plane_val = float(out[0, t, 0, 0])
            expected_fill = float(8 - t)
            assert plane_val == pytest.approx(expected_fill, abs=0.05), (
                f"plane {t} after 8 calls must hold my-stones from step {8 - t} "
                f"(fill={expected_fill:.0f}), got {plane_val}"
            )

    def test_opp_stones_history_shifts_independently(self) -> None:
        """Opponent-stones planes (9-15) shift independently of my-stones (1-7)."""
        states = _make_history_chain(4)
        buf = TensorBuffer()
        out = None
        for s in states:
            out, _ = buf.assemble(s)

        assert out is not None
        # After 4 calls: plane 9 (opp t-1) = fill_opp from states[2] = 102.0
        opp_plane9 = float(out[0, 9, 0, 0])
        assert opp_plane9 == pytest.approx(102.0, abs=0.1), (
            f"plane 9 (opp t-1) after 4 calls must be 102.0, got {opp_plane9}"
        )

    def test_k_change_triggers_full_rebuild_clears_history(self) -> None:
        """When K changes, the buffer fully rebuilds — no stale history from old K."""
        buf = TensorBuffer()
        # First call with K=1, fill with sentinel value 99.0
        state_k1 = _make_mock_state(K=1, fill_my=99.0, fill_opp=99.0)
        buf.assemble(state_k1)

        # Second call with K=2 — must trigger full rebuild
        state_k2 = _make_mock_state(K=2, fill_my=0.5, fill_opp=0.5)
        out, _ = buf.assemble(state_k2)

        assert out.shape == (2, 18, BOARD_SIZE, BOARD_SIZE), \
            "output shape must match new K=2"
        for t in range(1, 8):
            assert out[0, t].max() == 0.0, \
                f"after K-change, cluster 0 plane {t} must be zero (no valid history)"
            assert out[1, t].max() == 0.0, \
                f"after K-change, cluster 1 plane {t} must be zero (no valid history)"

    def test_reset_then_rebuild_clears_history(self) -> None:
        """After reset(), assemble() must treat the next call as a fresh game (no history)."""
        states = _make_history_chain(3)
        buf = TensorBuffer()
        for s in states:
            buf.assemble(s)

        # After 3 calls, plane 1 has non-zero history. Reset must clear it.
        buf.reset()
        fresh_state = _make_mock_state(K=1, fill_my=0.5, fill_opp=0.5)
        out, _ = buf.assemble(fresh_state)

        for t in range(1, 8):
            assert out[0, t].max() == 0.0, \
                f"after reset(), plane {t} must be zero (no prior history)"


# ── Regression guards ─────────────────────────────────────────────────────────

class TestTensorBufferRegressionGuards:
    """Regression tests that fail if specific architectural bugs are re-introduced."""

    def test_get_cluster_views_called_exactly_once_per_from_board(self) -> None:
        """Regression guard: get_cluster_views must be called exactly once per from_board().

        This test fails if the 'double evaluation' bug is re-introduced — i.e. if
        Python code calls get_cluster_views twice for the same board state.
        """
        from engine import Board as NativeBoard

        class _BoardSpy:
            """Wraps a native Board and counts get_cluster_views calls."""
            def __init__(self, board: NativeBoard) -> None:
                self._board = board
                self.gcv_calls = 0

            def get_cluster_views(self):
                self.gcv_calls += 1
                return self._board.get_cluster_views()

            def __getattr__(self, name: str):
                return getattr(self._board, name)

        board = NativeBoard()
        board.apply_move(0, 0)
        spy = _BoardSpy(board)

        _ = GameState.from_board(spy)  # type: ignore[arg-type]

        assert spy.gcv_calls == 1, (
            f"get_cluster_views was called {spy.gcv_calls} times in from_board() — "
            "expected exactly 1. The double-evaluation bug has been re-introduced."
        )

    def test_to_tensor_does_not_call_get_cluster_views(self) -> None:
        """Regression guard: to_tensor() must use cached views, not re-call get_cluster_views.

        The views are cached in GameState.views when from_board() is called.
        to_tensor() must read from that cache, never crossing the PyO3 boundary again.
        """
        from engine import Board as NativeBoard

        class _BoardSpy:
            def __init__(self, board: NativeBoard) -> None:
                self._board = board
                self.gcv_calls = 0

            def get_cluster_views(self):
                self.gcv_calls += 1
                return self._board.get_cluster_views()

            def __getattr__(self, name: str):
                return getattr(self._board, name)

        board = NativeBoard()
        board.apply_move(0, 0)
        spy = _BoardSpy(board)

        # Construct the state (one legitimate call to get_cluster_views)
        state = GameState.from_board(spy)  # type: ignore[arg-type]
        calls_after_from_board = spy.gcv_calls

        # to_tensor() must not trigger any further calls
        _ = state.to_tensor()

        assert spy.gcv_calls == calls_after_from_board, (
            f"to_tensor() caused {spy.gcv_calls - calls_after_from_board} additional "
            "get_cluster_views call(s) — views must be read from the cached GameState.views."
        )
