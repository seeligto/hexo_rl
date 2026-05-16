"""F1 guard — pretrain Rust-kernel aug parity (apply_symmetries_batch vs ReplayBuffer).

After the Q13 chain-plane landing (§92), pretrain no longer uses a Python
`_apply_hex_sym` scatter — it routes through `engine.apply_symmetries_batch`,
which shares the exact Rust scatter kernel used inside
`ReplayBuffer.sample_batch`. This test is the byte-exact guard against
anyone re-introducing a divergent Python augmentation path.

Post-`00b7d2b` the single-state `engine.apply_symmetry` PyO3 free function
is retired; only the batch form `apply_symmetries_batch` remains. Single-
state calls are emulated via batch-of-1 + index-0 — the inner kernel
(`apply_symmetry_state`) is the same.

Strategy:
  1. Build a known (18, 19, 19) state tensor via `to_tensor()` for a few
     hand-picked positions.
  2. For each of the 12 hex symmetries:
       - Compute `apply_symmetries_batch(state[None], [sym_idx])[0]`.
       - Compute the reference path: push into a fresh ReplayBuffer with a
         marker game_id, sample-loop many times with `augment=True`, and
         collect every unique output; the one whose raw stone scatter
         matches `sym_idx` must equal the batch-of-1 output byte-exact.
"""
from __future__ import annotations

import numpy as np
import pytest

import engine
from engine import Board, ReplayBuffer
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.env.game_state import GameState

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size
KEPT_PLANE_INDICES = list(_V6.kept_plane_indices)

CHANNELS = 18  # to_tensor() output; sliced to 8 before buffer push
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1
AUX_STRIDE = BOARD_SIZE * BOARD_SIZE
HALF = (BOARD_SIZE - 1) // 2


def _state_from_moves(moves: list[tuple[int, int]]) -> np.ndarray:
    """Play `moves` into a fresh Board and return the (18, 19, 19) tensor for
    cluster 0. Float32 copy of the f16 to_tensor output."""
    board = Board()
    state = GameState.from_board(board)
    for q, r in moves:
        state = state.apply_move(board, q, r)
    tensor, _ = state.to_tensor()
    return tensor[0].astype(np.float32)


POSITION_MOVES: list[tuple[str, list[tuple[int, int]]]] = [
    ("single_opening", [(0, 0)]),
    ("triangle", [(0, 0), (1, 0), (0, 1), (-1, 1)]),
    (
        "mid_game_axis_runs",
        [
            (0, 0), (1, 0), (2, 0), (-1, 1),
            (-2, 2), (3, -1), (-3, 3), (1, -1),
            (-1, 0), (2, -2),
        ],
    ),
]


def _collect_buffer_unique_outputs(state8: np.ndarray, n_draws: int = 4000) -> set[bytes]:
    """Push `state8` (8-plane, HEXB v6) into a fresh buffer and draw `n_draws`
    augmented samples. Return unique (8, 19, 19) f16-byte keys observed."""
    buf    = ReplayBuffer(4)
    s16    = state8.astype(np.float16)
    chain  = np.zeros((6, 19, 19), dtype=np.float16)
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    policy[0] = 1.0
    own = np.ones(AUX_STRIDE, dtype=np.uint8)
    wl  = np.zeros(AUX_STRIDE, dtype=np.uint8)
    buf.push(s16, chain, policy, 0.0, own, wl)

    seen: set[bytes] = set()
    for _ in range(n_draws):
        # sample_batch returns 7-tuple: (states, chain_planes, policies, outcomes, own, wl, is_full_search)
        states_out, _, _, _, _, _, _ = buf.sample_batch(1, augment=True)
        sampled = np.asarray(states_out[0]).astype(np.float16)
        seen.add(sampled.tobytes())
    return seen


def _binding_unique_outputs(state8: np.ndarray) -> dict[bytes, int]:
    """Compute {f16 bytes → sym_idx} for all 12 syms via
    `engine.apply_symmetries_batch` (batch-of-1) on the 8-plane state.
    Collapses duplicates for symmetric positions."""
    out: dict[bytes, int] = {}
    state_f32 = state8.astype(np.float32)
    for sym_idx in range(12):
        result = engine.apply_symmetries_batch(state_f32[None], [sym_idx])[0].astype(np.float16)
        key = result.tobytes()
        out.setdefault(key, sym_idx)
    return out


@pytest.mark.parametrize("name,moves", POSITION_MOVES)
def test_apply_symmetry_matches_replay_buffer_path(name, moves):
    """Every unique state output produced by the buffer's augment path must
    match one of the 12 binding outputs byte-exact. P(hitting all syms in
    4000 uniform draws) > 1 − 12·(11/12)^4000, well above 1 − 1e-150."""
    state18 = _state_from_moves(moves)
    # HEXB v6: buffer stores 8 planes; compare on the same 8-plane slice.
    state8 = state18[KEPT_PLANE_INDICES]

    binding_keys = _binding_unique_outputs(state8)
    buffer_keys = _collect_buffer_unique_outputs(state8)

    unknown = buffer_keys - set(binding_keys.keys())
    assert not unknown, (
        f"[{name}] {len(unknown)} buffer-sampled outputs did not match any "
        f"apply_symmetries_batch output across 12 syms — kernel divergence."
    )
    # The buffer should exhaust every unique binding output class in 4000 draws.
    missing = set(binding_keys.keys()) - buffer_keys
    assert not missing, (
        f"[{name}] binding produced {len(binding_keys)} unique outputs but "
        f"only {len(buffer_keys)} seen from buffer — {len(missing)} unseen "
        f"(coverage gap, not a kernel drift; increase n_draws if flaky)."
    )
