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

        # Precompute flat index maps for all 12 hexagonal symmetries.
        self._sym_indices = self._precompute_symmetry_indices(board_size)

    @staticmethod
    def _precompute_symmetry_indices(board_size: int) -> np.ndarray:
        """Return (12, H*W) array of flat indices mapping src -> dst."""
        half = (board_size - 1) // 2
        H = W = board_size
        i_grid, j_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        q_base = i_grid - half
        r_base = j_grid - half

        sym_indices = np.zeros((12, board_size * board_size), dtype=np.int64)

        for sym_idx in range(12):
            reflect = sym_idx >= 6
            rot = sym_idx % 6
            q = q_base.copy()
            r = r_base.copy()
            if reflect:
                q, r = r.copy(), q.copy()
            for _ in range(rot):
                q, r = -r, q + r
            
            dst_i = q + half
            dst_j = r + half
            
            valid = (dst_i >= 0) & (dst_i < H) & (dst_j >= 0) & (dst_j < W)
            flat_dst = np.where(valid, dst_i * board_size + dst_j, -1).flatten()
            sym_indices[sym_idx] = flat_dst
            
        return sym_indices

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
        self, batch_size: int, augment: bool = True
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Sample `batch_size` entries with Multi-Window correlation guard.

        Entries belonging to the same board position (same game_id) are never
        allowed into the same batch together — at most one cluster per position
        is included.  Falls back to plain uniform sampling if game_ids are not
        set (all -1, e.g. data loaded from an older checkpoint).

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

        valid_ids = self.game_ids[:self._size]
        use_dedup = valid_ids[0] != -1  # fast check: ids assigned?

        if use_dedup:
            # Shuffle all valid indices, then pick the first occurrence of each
            # unique game_id.  This is O(size) but uses no Python loops.
            perm = self._rng.permutation(self._size)
            shuffled_ids = valid_ids[perm]
            _, first_occurrence = np.unique(shuffled_ids, return_index=True)
            candidate_indices = perm[first_occurrence]
            if len(candidate_indices) >= batch_size:
                chosen = self._rng.choice(candidate_indices, size=batch_size, replace=False)
            else:
                # Fewer distinct positions than batch_size — fall back to allowing
                # duplicates rather than under-filling the batch.
                chosen = self._rng.choice(self._size, size=batch_size, replace=True)
            indices = chosen
        else:
            indices = np.random.randint(0, self._size, size=batch_size)
        
        if not augment:
            return self.states[indices], self.policies[indices], self.outcomes[indices]

        # Use views for source data to avoid copying before transformation
        states = self.states[indices]
        policies = self.policies[indices]
        outcomes = self.outcomes[indices]

        sym_choices = np.random.randint(0, 12, size=batch_size)
        # Create buffers for transformed data
        new_states = np.zeros_like(states)
        new_policies = np.zeros_like(policies)
        new_policies[:, -1] = policies[:, -1] # pass move stays

        for sym_idx in range(12):
            mask = sym_choices == sym_idx
            if not mask.any():
                continue
            
            if sym_idx == 0: # identity
                new_states[mask] = states[mask]
                new_policies[mask, :-1] = policies[mask, :-1]
                continue

            flat_map = self._sym_indices[sym_idx]
            valid_mask = flat_map >= 0
            valid_src = np.where(valid_mask)[0]
            valid_dst = flat_map[valid_mask]

            # Vectorized scatter for this symmetry group
            mask_indices = mask.nonzero()[0]
            K = len(mask_indices)
            
            # Write to flattened view of the target buffer
            s_batch = states[mask].reshape(K, self.board_channels, self.spatial)
            target_view = new_states[mask_indices].reshape(K, self.board_channels, self.spatial)
            target_view[:, :, valid_dst] = s_batch[:, :, valid_src]
            new_states[mask_indices] = target_view.reshape(K, self.board_channels, self.board_size, self.board_size)
            
            # Policy grids (K, spatial)
            p_batch = policies[mask, :-1]
            target_p_view = new_policies[mask_indices, :-1]
            target_p_view[:, valid_dst] = p_batch[:, valid_src]
            new_policies[mask_indices, :-1] = target_p_view
        
        return new_states, new_policies, outcomes
    
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
