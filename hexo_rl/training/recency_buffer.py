"""Lightweight Python-side ring buffer for recent self-play positions.

Maintains the last ``capacity`` positions in pre-allocated NumPy arrays.
Used alongside the Rust ReplayBuffer for recency-weighted batch sampling,
biasing training toward newer self-play data without requiring any Rust changes.

Thread-safety: push() and sample() are protected by a threading.Lock so they
can be called concurrently from the pool stats thread and the training loop.
"""

from __future__ import annotations

import threading

import numpy as np


class RecentBuffer:
    """Rolling window ring buffer over recent (state, policy, outcome) triples.

    Args:
        capacity:    Maximum number of positions to store.  Oldest entries are
                     overwritten once full (ring semantics).
        state_shape: Shape of one state tensor, default (18, 19, 19).
        policy_len:  Number of policy logits per position, default 362.
    """

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...] = (18, 19, 19),
        policy_len: int = 362,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._states   = np.zeros((capacity, *state_shape), dtype=np.float16)
        self._policies = np.zeros((capacity, policy_len),   dtype=np.float32)
        self._outcomes = np.zeros(capacity,                  dtype=np.float32)
        self._head = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    def push(
        self,
        state:   np.ndarray,
        policy:  np.ndarray,
        outcome: float,
    ) -> None:
        """Add one (state, policy, outcome) triple; overwrites oldest when full."""
        with self._lock:
            self._states[self._head]   = state
            self._policies[self._head] = policy
            self._outcomes[self._head] = float(outcome)
            self._head = (self._head + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniform random sample of ``n`` positions from the stored recent window.

        Returns copies so the caller is free to mutate the arrays.

        Raises:
            ValueError: If the buffer is empty.
        """
        with self._lock:
            if self._size == 0:
                raise ValueError("Cannot sample from empty RecentBuffer")
            indices = np.random.randint(0, self._size, n)
            return (
                self._states[indices].copy(),
                self._policies[indices].copy(),
                self._outcomes[indices].copy(),
            )
