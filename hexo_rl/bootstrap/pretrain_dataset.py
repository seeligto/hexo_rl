"""Pretrain dataset + augmentation collate (§176 P39 split from pretrain.py).

Contains:
  - AugmentedBootstrapDataset — torch Dataset over flat (state, policy, outcome)
    triples (optionally with global_crops for §169 A3 / §170 P3 paths).
  - make_augmented_collate — factory returning a collate_fn that batches the
    triples and applies the 12-fold hex augmentation via either the Rust
    `engine.apply_symmetries_batch` binding (v6) or pure-numpy scatter
    (v6w25 / v8). Chain planes are recomputed post-augment from stone planes.
  - _game_winner_from_replay — replays a move list on a Board and returns the
    winner. Used by load_corpus (pretrain_legacy.py) and
    scripts/export_corpus_npz.py.

Per §176 P39 pure-move discipline: bodies moved verbatim from pretrain.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

import engine
from engine import Board
from hexo_rl.augment.luts import get_policy_scatters
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.encoding.resolvers import opp_stone_slot
from hexo_rl.env.game_state import _compute_chain_planes


# ── Hex augmentation ──────────────────────────────────────────────────────────
#
# Q13 chain planes live at state[18..24] and require an axis-plane remap under
# augmentation in addition to the coordinate scatter — see
# `engine/src/replay_buffer/sample.rs::apply_symmetry_24plane` + the
# `axis_perm[sym_idx]` table in `sym_tables.rs`. The pre-Q13 pretrain path
# replicated the coord scatter in Python but had two bugs:
#   (1) no axis_perm remap, so chain planes contradicted the stones in 11/12
#       augmented samples (§92/§93 F1);
#   (2) (col, row) vs (row, col) coordinate convention mismatch with the
#       Rust SymTables / _compute_chain_planes.
# Both bugs are eliminated by routing through the Rust `apply_symmetries_batch`
# binding — the same kernel ReplayBuffer.sample_batch uses, guaranteed to
# match byte-exact via `tests/test_pretrain_aug.py` (F1 guard).
#
# Policies are scattered in numpy via a once-computed per-sym index map — they
# have no chain-plane structure and the 12 tables fit in ~17 KB.
# LUTs are in hexo_rl/augment/luts.py (shared with batch_assembly).

# ── Dataset ───────────────────────────────────────────────────────────────────

class AugmentedBootstrapDataset(torch.utils.data.Dataset):
    """Pretrain dataset that yields raw (state, policy, outcome) triples.

    The 12-fold hex augmentation is applied in `augmented_collate` via the
    Rust `engine.apply_symmetries_batch` binding — this is the same scatter
    kernel the ReplayBuffer uses at sample time, guaranteed byte-exact via
    the F1 parity test (`tests/test_pretrain_aug.py`). Moving augmentation
    to collate amortises the PyO3 boundary over an entire batch and
    eliminates the Python `_apply_hex_sym` path that was corrupting Q13
    chain planes in pretrain v3.

    Chain planes (6 planes, aux head target) are computed in collate from
    the augmented stone planes (0=cur, 8=opp) — recomputing post-augment
    avoids the axis-perm remap and is self-consistent.

    §169 A3 — when ``global_crops`` is provided, ``__getitem__`` returns
    a 4-tuple including the per-position global summary crop. Augmentation
    of the crop is intentionally skipped (the GlobalTokenEncoder is
    near-rotation-invariant via KataGo gpool, so feeding the canonical-
    orientation crop alongside augmented cluster windows is a tractable
    approximation; documented as a known A3 simplification).

    Args:
        states:        (N, C, H, W) float16 array (mmap-compatible).
        policies:      (N, n_actions) float32 array.
        outcomes:      (N,) float32 array, values in {-1, 0, +1}.
        global_crops:  Optional (N, 3, 32, 32) float16 array. When present,
                       __getitem__ returns a 4-tuple.
    """

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        outcomes: np.ndarray,
        global_crops: Optional[np.ndarray] = None,
    ) -> None:
        self.states = states
        self.policies = policies
        self.outcomes = outcomes
        self.global_crops = global_crops

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(self, idx: int):
        # Copy out of the (possibly mmapped) backing store so downstream
        # collate can batch-concat without aliasing.
        if self.global_crops is not None:
            return (
                self.states[idx].copy(),
                self.policies[idx].copy(),
                float(self.outcomes[idx]),
                self.global_crops[idx].copy(),
            )
        return (
            self.states[idx].copy(),
            self.policies[idx].copy(),
            float(self.outcomes[idx]),
        )


def make_augmented_collate(
    augment: bool,
    encoding: str = "v6",
    with_global_crop: bool = False,
):
    """Return a collate_fn that batches triples and applies hex augmentation.

    Two paths:

    v6 (default, `encoding="v6"`, trunk_size=19):
      - Stack states (N, 8, 19, 19) f16, upcast to f32.
      - Rust `engine.apply_symmetries_batch` for state scatter (one PyO3 hop).
      - Policy scatter via precomputed numpy index tables (12 × 362).
      - Chain planes recomputed from augmented stone planes 0 (cur) / 4 (opp).

    v8 (`encoding="v8"`, trunk_size=25, `has_pass=False`):
      - Stack states (N, 11, 25, 25) f16.
      - Pure-numpy state scatter (same hex-symmetry math as Rust kernel) —
        the Rust `apply_symmetries_batch` PyO3 binding hardcodes BOARD_SIZE=19
        and is v6-only. Per S2 §2.4 v8 splice spec, Phase B keeps state aug
        in Python; the v8 PyO3 path is a Phase D concern when self-play
        replay buffers must scatter v8 buffer rows at runtime.
      - Policy scatter (N, 625) using `get_policy_scatters(25, has_pass=False)`.
      - Chain planes recomputed at 25×25 (planes 8/9/10 are
        symmetry-invariant: off_window mask is hex-symmetric and the broadcast
        scalars are constants).

    With `augment=False`:
      - Stack as-is, no scatter.
      - Compute chain_planes from raw stone planes.

    §176 P76: dropped redundant `board_size` parameter — read off
    `spec.trunk_size` (canonical spatial size of trunk input tensor).
    """
    _enc_spec = _lookup_encoding(encoding)
    has_pass = _enc_spec.has_pass_slot
    board_size = _enc_spec.trunk_size
    _opp_slot = opp_stone_slot(_enc_spec)  # opp t0 slice idx: 4 (v6 family) / 1 (v6_live2)
    scatters_np = get_policy_scatters(board_size, has_pass=has_pass) if augment else None

    def _collate(batch):
        n = len(batch)
        states = np.stack([b[0] for b in batch], axis=0)
        policies = np.stack([b[1] for b in batch], axis=0)
        outcomes = np.asarray([b[2] for b in batch], dtype=np.float32)
        # §169 A3 — global crop stays canonical (no per-batch sym applied).
        # GlobalTokenEncoder ends in KataGo gpool which is near-spatially-
        # invariant, so the orientation mismatch with augmented cluster
        # windows is bounded; documented as an A3 simplification.
        global_crops = (
            np.stack([b[3] for b in batch], axis=0)
            if with_global_crop
            else None
        )

        if augment and scatters_np is not None:
            sym_indices = np.random.randint(0, 12, size=n).astype(np.int64)

            if encoding == "v6":
                states_f32 = states.astype(np.float32, copy=False)
                states_f32 = engine.apply_symmetries_batch(
                    states_f32, sym_indices.astype(np.uint64).tolist()
                )
                scattered = np.empty_like(policies)
                for i in range(n):
                    scattered[i] = policies[i][scatters_np[int(sym_indices[i])]]
                policies = scattered
                states = states_f32.astype(np.float16, copy=False)
            else:
                # v8 / v6w25 path — pure-numpy scatter, batched per-sym to avoid
                # the per-sample take_along_axis cost. The Rust
                # apply_symmetries_batch PyO3 binding hardcodes BOARD_SIZE=19;
                # both v6w25 (25×25 K-cluster) and v8 (25×25 bbox) need numpy.
                C = states.shape[1]
                spatial = board_size * board_size
                states_flat = states.reshape(n, C, spatial)
                augmented = np.empty_like(states_flat)
                policy_aug = np.empty_like(policies)
                for sym in range(12):
                    mask_idx = np.where(sym_indices == sym)[0]
                    if mask_idx.size == 0:
                        continue
                    # For v6w25 the policy scatter is `spatial+1` long (cells
                    # + pass). State scatter must slice to spatial-only.
                    sc = scatters_np[sym]
                    state_scatter = sc[:spatial] if has_pass else sc
                    augmented[mask_idx] = states_flat[mask_idx][:, :, state_scatter]
                    policy_aug[mask_idx] = policies[mask_idx][:, sc]
                states = augmented.reshape(n, C, board_size, board_size)
                policies = policy_aug

        # Chain planes — recomputed post-augmentation from stone planes.
        # cur t0 is always slot 0; opp t0 (source plane 8) sits at slot 4 for
        # the v6 family but slot 1 for v6_live2 — derive from the registry.
        chain_np = np.zeros((n, 6, board_size, board_size), dtype=np.float16)
        for i in range(n):
            chain_np[i] = _compute_chain_planes(
                states[i, 0].astype(np.float32),
                states[i, _opp_slot].astype(np.float32),
            ).astype(np.float16) / 6.0

        if global_crops is not None:
            return (
                torch.from_numpy(states),
                torch.from_numpy(chain_np),
                torch.from_numpy(policies),
                torch.from_numpy(outcomes),
                torch.from_numpy(global_crops),
            )
        return (
            torch.from_numpy(states),
            torch.from_numpy(chain_np),
            torch.from_numpy(policies),
            torch.from_numpy(outcomes),
        )

    return _collate


# ── Corpus loading ────────────────────────────────────────────────────────────

def _game_winner_from_replay(moves: List[Tuple[int, int]]) -> Optional[int]:
    """Replay a move sequence and return winner (+1 or -1), or None."""
    board = Board()
    for q, r in moves:
        try:
            board.apply_move(q, r)
        except Exception:
            break
    w = board.winner()
    return int(w) if w is not None else None
