"""Lightweight Python-side ring buffer for recent self-play positions.

Maintains the last ``capacity`` positions in pre-allocated NumPy arrays.
Used alongside the Rust ReplayBuffer for recency-weighted batch sampling,
biasing training toward newer self-play data without requiring any Rust changes.

Thread-safety: push() and sample() are protected by a threading.Lock so they
can be called concurrently from the pool stats thread and the training loop.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class RecentBuffer:
    """Rolling window ring buffer over recent self-play positions with aux columns.

    Args:
        capacity:    Maximum number of positions to store.  Oldest entries are
                     overwritten once full (ring semantics).
        state_shape: Shape of one state tensor, default (24, 19, 19).
        policy_len:  Number of policy logits per position, default 362.
        aux_stride:  Flat length of one ownership/winning_line plane, default 361.
    """

    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...] = (18, 19, 19),
        policy_len: int = 362,
        aux_stride: int = 361,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._states       = np.zeros((capacity, *state_shape), dtype=np.float16)
        self._chain_planes = np.zeros((capacity, 6, 19, 19),   dtype=np.float16)
        self._policies     = np.zeros((capacity, policy_len),   dtype=np.float32)
        self._outcomes     = np.zeros(capacity,                  dtype=np.float32)
        # Default ownership=1 ("empty" per Rust encoding), winning_line=0 — neutral fallback
        # so unset slots decode to harmless aux targets if ever sampled.
        self._ownership    = np.ones((capacity, aux_stride),  dtype=np.uint8)
        self._winning_line = np.zeros((capacity, aux_stride), dtype=np.uint8)
        self._head = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    def push(
        self,
        state:        np.ndarray,
        chain_planes: Optional[np.ndarray] = None,
        policy:       Optional[np.ndarray] = None,
        outcome:      float = 0.0,
        ownership:    Optional[np.ndarray] = None,
        winning_line: Optional[np.ndarray] = None,
    ) -> None:
        """Add one position; overwrites oldest when full.

        chain_planes, ownership, and winning_line default to None which leaves
        the pre-allocated zeros/ones in place.
        """
        with self._lock:
            self._states[self._head]   = state
            if chain_planes is not None:
                self._chain_planes[self._head] = chain_planes
            else:
                self._chain_planes[self._head] = 0
            if policy is not None:
                self._policies[self._head] = policy
            self._outcomes[self._head] = float(outcome)
            if ownership is not None:
                self._ownership[self._head] = ownership
            else:
                self._ownership[self._head] = 1  # "empty" encoding
            if winning_line is not None:
                self._winning_line[self._head] = winning_line
            else:
                self._winning_line[self._head] = 0
            self._head = (self._head + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform random sample of ``n`` positions from the stored recent window.

        Returns 6-tuple (states, chain_planes, policies, outcomes, ownership, winning_line).
        ownership and winning_line are returned as (n, aux_stride) u8 — caller
        reshapes to (n, 19, 19) if needed.

        NumPy fancy indexing (`arr[index_array]`) returns a newly-allocated
        array that is NOT aliased to the underlying ring buffer storage —
        unlike slice indexing, which does return a view. That means:
          - No defensive `.copy()` is needed before returning to the caller.
          - The caller is free to mutate the returned arrays in place without
            corrupting the ring; the buffer only sees writes that go through
            `add()` / `add_batch()`.
        If this function is ever switched to slice indexing in the future
        (e.g. contiguous range reads for a dataloader), the aliasing
        assumption reverses and the caller contract must be revisited.

        Raises:
            ValueError: If the buffer is empty.
        """
        with self._lock:
            if self._size == 0:
                raise ValueError("Cannot sample from empty RecentBuffer")
            indices = np.random.randint(0, self._size, n)
            return (
                self._states[indices],
                self._chain_planes[indices],
                self._policies[indices],
                self._outcomes[indices],
                self._ownership[indices],
                self._winning_line[indices],
            )
