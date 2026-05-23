"""Tests for §S181-AUDIT Wave 1 Track B / B2 — buffer position-class snapshot."""
from __future__ import annotations

import numpy as np

from hexo_rl.training.track_b_buffer_snapshot import (
    snapshot_buffer_position_classes,
)


class _FakeBuffer:
    """In-memory drop-in for the Rust ReplayBuffer, just enough for B2."""

    def __init__(self, states: np.ndarray, outcomes: np.ndarray) -> None:
        self._states = states
        self._outcomes = outcomes
        self.size = int(states.shape[0])

    def sample_batch(self, n: int, augment: bool):
        # Match the Rust API tuple length (7): states, chain, policies,
        # outcomes, ownership, winning_line, is_full_search. Only states +
        # outcomes are inspected by the snapshot module.
        idx = np.arange(min(n, self.size))
        states = self._states[idx]
        outcomes = self._outcomes[idx]
        zero_aux = np.zeros((states.shape[0], 19, 19), dtype=np.float32)
        chain = np.zeros((states.shape[0], 6, 19, 19), dtype=np.float16)
        policies = np.zeros((states.shape[0], 19 * 19 + 1), dtype=np.float32)
        ifs = np.ones(states.shape[0], dtype=np.uint8)
        return states, chain, policies, outcomes, zero_aux, zero_aux, ifs


def _make_colony_state() -> np.ndarray:
    """8-plane (8, 19, 19) state with a tight colony cluster."""
    s = np.zeros((8, 19, 19), dtype=np.float32)
    # plane 0 = current player; plane 4 = opponent.
    centre = 9
    for di in range(-1, 2):
        for dj in range(-1, 2):
            s[0, centre + di, centre + dj] = 1.0
            s[4, centre + di + 1, centre + dj + 1] = 1.0
    return s


def _make_extension_state() -> np.ndarray:
    """8-plane state with an open run of 5 stones along an axis (extension)."""
    s = np.zeros((8, 19, 19), dtype=np.float32)
    # Player stones at (5..9, 5) along axis (1, 0); flanks empty.
    for i in range(5, 10):
        s[0, i, 5] = 1.0
    # Opponent: well-separated single stone (does not form a run).
    s[4, 15, 15] = 1.0
    return s


def test_snapshot_emits_payload_with_class_fractions():
    n_col = 5
    n_ext = 7
    states = np.stack(
        [_make_colony_state()] * n_col + [_make_extension_state()] * n_ext
    )
    outcomes = np.array([1.0] * n_col + [-1.0] * n_ext, dtype=np.float32)
    buf = _FakeBuffer(states, outcomes)

    payload = snapshot_buffer_position_classes(buf, step=42, n_sample=100)
    assert payload is not None
    assert payload["event"] == "buffer_position_class_snapshot"
    assert payload["step"] == 42
    assert payload["n_sampled"] == n_col + n_ext
    assert payload["colony_n"] == n_col
    assert payload["extension_n"] == n_ext
    assert payload["neither_n"] == 0
    # Payload rounds to 4 decimals; allow 5e-4 absolute tolerance.
    assert abs(payload["colony_frac"] - n_col / (n_col + n_ext)) < 5e-4
    assert abs(payload["extension_frac"] - n_ext / (n_col + n_ext)) < 5e-4
    # Per-class outcome means: colony rows had +1, extension rows −1.
    assert abs(payload["colony_mean_value_target"] - 1.0) < 5e-4
    assert abs(payload["extension_mean_value_target"] - (-1.0)) < 5e-4


def test_snapshot_empty_buffer_returns_none():
    buf = _FakeBuffer(
        np.zeros((0, 8, 19, 19), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
    payload = snapshot_buffer_position_classes(buf, step=0, n_sample=100)
    assert payload is None


def test_snapshot_clamps_sample_to_buffer_size():
    n = 3
    states = np.stack([_make_colony_state()] * n)
    outcomes = np.array([1.0] * n, dtype=np.float32)
    buf = _FakeBuffer(states, outcomes)
    payload = snapshot_buffer_position_classes(buf, step=1, n_sample=1000)
    assert payload is not None
    assert payload["n_sampled"] == n


def test_snapshot_handles_zero_class_count_without_nan_keys():
    """A class with zero hits must produce a sentinel (None) — not NaN — so the
    emitted JSONL stays parseable downstream."""
    n_col = 4
    states = np.stack([_make_colony_state()] * n_col)
    outcomes = np.array([0.5] * n_col, dtype=np.float32)
    buf = _FakeBuffer(states, outcomes)
    payload = snapshot_buffer_position_classes(buf, step=2, n_sample=10)
    assert payload is not None
    assert payload["extension_n"] == 0
    # Mean value target for an empty class is sentinel None (not NaN).
    assert payload["extension_mean_value_target"] is None
    assert payload["colony_mean_value_target"] is not None
