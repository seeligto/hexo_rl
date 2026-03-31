"""
ReplayBuffer — pre-allocated NumPy ring arrays for storing self-play data.

Never allocates after __init__. All arrays are fixed-size; overflow wraps
around (oldest data is overwritten). Designed for FP16 state storage to
minimise VRAM and RAM bandwidth during training.

Architecture spec (docs/01_architecture.md §4):
    states:   (capacity, board_channels, board_size, board_size)  float16
    policies: (capacity, board_size*board_size + 1)               float32
    outcomes: (capacity,)                                          float32
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """Ring-buffer replay buffer with pre-allocated NumPy arrays.

    Args:
        capacity:      Maximum number of positions stored (oldest overwritten on wrap).
        board_channels: Number of input tensor channels (default 18).
        board_size:    Spatial dimension of the board (default 19).
    """

    def __init__(
        self,
        capacity: int = 500_000,
        board_channels: int = 18,
        board_size: int = 19,
    ) -> None:
        self.capacity      = capacity
        self.board_channels = board_channels
        self.board_size    = board_size
        self.spatial       = board_size * board_size
        self._n_actions    = self.spatial + 1

        # Pre-allocate all storage up front — never re-allocate.
        self.states   = np.zeros(
            (capacity, board_channels, board_size, board_size), dtype=np.float16
        )
        self.policies = np.zeros((capacity, self._n_actions), dtype=np.float32)
        self.outcomes = np.zeros((capacity,),                 dtype=np.float32)
        # Game-position ID per slot — used to prevent same-position cluster pairs
        # from landing in the same training batch (Multi-Window correlation guard).
        self.game_ids = np.full(capacity, -1, dtype=np.int64)

        self._ptr     = 0    # next write index
        self._size    = 0    # number of valid entries
        self._game_id = 0    # monotonic counter; bumped by push_new_position()
        self._rng     = np.random.default_rng()

    # ── Write ─────────────────────────────────────────────────────────────────

    def push(
        self,
        state:   "np.ndarray",  # (board_channels, board_size, board_size) float16
        policy:  "np.ndarray",  # (n_actions,) float32
        outcome: float,
        game_id: int = -1,
    ) -> None:
        """Store a single (state, policy, outcome) triple.

        Pass `game_id` (from `next_game_id()`) to tag which board position this
        cluster belongs to — enables correlation-safe sampling.

        Overwrites the oldest entry once capacity is reached.
        No allocation occurs — writes directly into the pre-allocated arrays.
        """
        self.states  [self._ptr] = state
        self.policies[self._ptr] = policy
        self.outcomes[self._ptr] = outcome
        self.game_ids[self._ptr] = game_id
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def next_game_id(self) -> int:
        """Return a fresh monotonic game-position ID and advance the counter.

        Call once per board position (not once per cluster).  Pass the returned
        ID to every cluster's `push()` call so the sampler can group them.
        """
        gid = self._game_id
        self._game_id += 1
        return gid

    def push_game(
        self,
        states:   "np.ndarray",  # (T, board_channels, board_size, board_size)
        policies: "np.ndarray",  # (T, n_actions)
        outcomes: "np.ndarray",  # (T,)
    ) -> None:
        """Store all positions from a completed game efficiently.

        Handles wrap-around correctly even when the game straddles the
        end of the ring buffer.
        """
        t = len(states)
        if t == 0:
            return
        end = self._ptr + t
        if end <= self.capacity:
            # Common case: fits without wrap.
            self.states  [self._ptr:end] = states
            self.policies[self._ptr:end] = policies
            self.outcomes[self._ptr:end] = outcomes
        else:
            # Wrap around: split into two memcpy operations.
            first = self.capacity - self._ptr
            self.states  [self._ptr:]  = states  [:first]
            self.policies[self._ptr:]  = policies[:first]
            self.outcomes[self._ptr:]  = outcomes[:first]
            self.states  [:t - first]  = states  [first:]
            self.policies[:t - first]  = policies[first:]
            self.outcomes[:t - first]  = outcomes[first:]
        self._ptr  = end % self.capacity
        self._size = min(self._size + t, self.capacity)

    # ── Read ──────────────────────────────────────────────────────────────────

    def sample(
        self, batch_size: int
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Sample `batch_size` entries with Multi-Window correlation guard.

        Entries belonging to the same board position (same game_id) are never
        allowed into the same batch together — at most one cluster per position
        is included.  Falls back to plain uniform sampling if game_ids are not
        set (all -1, e.g. data loaded from an older checkpoint).

        Note: augmentation is handled by RustReplayBuffer.sample_batch. Use
        that for training; this method is for testing/inspection only.

        Returns:
            states:   (batch_size, board_channels, board_size, board_size) float16
            policies: (batch_size, n_actions)                               float32
            outcomes: (batch_size,)                                          float32
        """
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        valid_ids = self.game_ids[:self._size]
        use_dedup = valid_ids[0] != -1  # fast check: ids assigned?

        if use_dedup:
            perm = self._rng.permutation(self._size)
            shuffled_ids = valid_ids[perm]
            _, first_occurrence = np.unique(shuffled_ids, return_index=True)
            candidate_indices = perm[first_occurrence]
            if len(candidate_indices) >= batch_size:
                chosen = self._rng.choice(candidate_indices, size=batch_size, replace=False)
            else:
                chosen = self._rng.choice(self._size, size=batch_size, replace=True)
            indices = chosen
        else:
            indices = np.random.randint(0, self._size, size=batch_size)

        return self.states[indices], self.policies[indices], self.outcomes[indices]
    
    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of valid entries currently stored."""
        return self._size

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self._size}/{self.capacity}, "
            f"channels={self.board_channels}, board={self.board_size}x{self.board_size})"
        )
