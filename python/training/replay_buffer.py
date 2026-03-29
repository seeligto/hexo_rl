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
        self._n_actions    = board_size * board_size + 1

        # Pre-allocate all storage up front — never re-allocate.
        self.states   = np.zeros(
            (capacity, board_channels, board_size, board_size), dtype=np.float16
        )
        self.policies = np.zeros((capacity, self._n_actions), dtype=np.float32)
        self.outcomes = np.zeros((capacity,),                 dtype=np.float32)

        self._ptr  = 0   # next write index
        self._size = 0   # number of valid entries

    # ── Write ─────────────────────────────────────────────────────────────────

    def push(
        self,
        state:   "np.ndarray",  # (board_channels, board_size, board_size) float16
        policy:  "np.ndarray",  # (n_actions,) float32
        outcome: float,
    ) -> None:
        """Store a single (state, policy, outcome) triple.

        Overwrites the oldest entry once capacity is reached.
        No allocation occurs — writes directly into the pre-allocated arrays.
        """
        self.states  [self._ptr] = state
        self.policies[self._ptr] = policy
        self.outcomes[self._ptr] = outcome
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

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
        self, batch_size: int, augment: bool = True
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Sample `batch_size` entries uniformly at random.

        Args:
            batch_size: Number of entries to sample.
            augment:    If True, apply random 12-fold hexagonal symmetry.

        Returns:
            states:   (batch_size, board_channels, board_size, board_size) float16
            policies: (batch_size, n_actions)                               float32
            outcomes: (batch_size,)                                          float32
        """
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self.states[indices].copy()
        policies = self.policies[indices].copy()
        outcomes = self.outcomes[indices]

        if augment:
            for i in range(batch_size):
                sym_idx = np.random.randint(0, 12)
                if sym_idx > 0:
                    states[i], policies[i] = self._apply_symmetry(states[i], policies[i], sym_idx)

        return states, policies, outcomes

    def _apply_symmetry(self, state: np.ndarray, policy: np.ndarray, sym_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply one of the 12 hexagonal symmetries to a state and policy.
        
        sym_idx: 0-5 rotations, 6-11 reflected rotations.
        """
        # 1. Parse policy
        pass_move = policy[-1]
        policy_grid = policy[:-1].reshape(self.board_size, self.board_size)
        
        # 2. Hexagonal symmetry on axial coordinates (q, r)
        # In our window, (q, r) are relative to center, mapped to [0, 18].
        # Let's shift to [-9, 9] range.
        half = (self.board_size - 1) // 2
        
        new_state = np.zeros_like(state)
        new_policy_grid = np.zeros_like(policy_grid)
        
        reflect = sym_idx >= 6
        rot = sym_idx % 6
        
        # Pre-calculate coordinate mapping for efficiency if needed, but for now
        # we'll do it point-by-point.
        for i in range(self.board_size):
            for j in range(self.board_size):
                q, r = i - half, j - half
                
                # Apply reflection: (q, r) -> (r, q)
                if reflect:
                    q, r = r, q
                
                # Apply rotation (rot * 60 degrees)
                # 0: (q, r)
                # 1: (-r, q+r)
                # 2: (-q-r, q)
                # 3: (-q, -r)
                # 4: (r, -q-r)
                # 5: (q+r, -q)
                for _ in range(rot):
                    q, r = -r, q + r
                
                # Map back to [0, 18]
                ni, nj = q + half, r + half
                
                # Check if in bounds (window might shift, but within the window it should be fine)
                if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                    new_state[:, ni, nj] = state[:, i, j]
                    new_policy_grid[ni, nj] = policy_grid[i, j]
        
        new_policy = np.append(new_policy_grid.flatten(), pass_move)
        return new_state, new_policy

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
