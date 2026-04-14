"""F1 guard — pretrain Rust-kernel aug parity (apply_symmetry vs ReplayBuffer).

After the Q13 chain-plane landing (§92), pretrain no longer uses a Python
`_apply_hex_sym` scatter — it routes through `engine.apply_symmetry` /
`engine.apply_symmetries_batch`, which share the exact Rust scatter kernel
used inside `ReplayBuffer.sample_batch`. This test is the byte-exact guard
against anyone re-introducing a divergent Python augmentation path.

Strategy:
  1. Build a known (24, 19, 19) state tensor via `to_tensor()` for a few
     hand-picked positions.
  2. For each of the 12 hex symmetries:
       - Compute `engine.apply_symmetry(state, sym_idx)`.
       - Compute the reference path: push into a fresh ReplayBuffer with a
         marker game_id, sample-loop many times with `augment=True`, and
         collect every unique output; the one whose raw stone scatter
         matches `sym_idx` must equal the `apply_symmetry` output byte-exact.
"""
from __future__ import annotations

import numpy as np
import pytest

import engine
from engine import Board, ReplayBuffer
from hexo_rl.env.game_state import GameState
from hexo_rl.utils.constants import BOARD_SIZE

CHANNELS = 24
N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1
AUX_STRIDE = BOARD_SIZE * BOARD_SIZE
HALF = (BOARD_SIZE - 1) // 2


def _state_from_moves(moves: list[tuple[int, int]]) -> np.ndarray:
    """Play `moves` into a fresh Board and return the (24, 19, 19) tensor for
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


def _collect_buffer_unique_outputs(state: np.ndarray, n_draws: int = 4000) -> set[bytes]:
    """Push `state` into a fresh buffer and draw `n_draws` augmented samples.
    Return the set of unique (24, 19, 19) f16-byte keys observed.
    """
    buf = ReplayBuffer(4)
    s16 = state.astype(np.float16)
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    policy[0] = 1.0
    own = np.ones(AUX_STRIDE, dtype=np.uint8)
    wl = np.zeros(AUX_STRIDE, dtype=np.uint8)
    buf.push(s16, policy, 0.0, own, wl)

    seen: set[bytes] = set()
    for _ in range(n_draws):
        states_out, _, _, _, _ = buf.sample_batch(1, augment=True)
        sampled = np.asarray(states_out[0]).astype(np.float16)
        seen.add(sampled.tobytes())
    return seen


def _binding_unique_outputs(state: np.ndarray) -> dict[bytes, int]:
    """Compute {f16 bytes → sym_idx} for all 12 syms via `engine.apply_symmetry`.

    Collapses duplicates for symmetric positions — the key is the f16 byte
    image of the output state so multiple syms that produce the same image
    all end up under one entry."""
    out: dict[bytes, int] = {}
    for sym_idx in range(12):
        result = engine.apply_symmetry(state, sym_idx).astype(np.float16)
        key = result.tobytes()
        out.setdefault(key, sym_idx)
    return out


@pytest.mark.parametrize("name,moves", POSITION_MOVES)
def test_apply_symmetry_matches_replay_buffer_path(name, moves):
    """Every unique state output produced by the buffer's augment path must
    match one of the 12 binding outputs byte-exact. P(hitting all syms in
    4000 uniform draws) > 1 − 12·(11/12)^4000, well above 1 − 1e-150."""
    state = _state_from_moves(moves)

    binding_keys = _binding_unique_outputs(state)
    buffer_keys = _collect_buffer_unique_outputs(state)

    unknown = buffer_keys - set(binding_keys.keys())
    assert not unknown, (
        f"[{name}] {len(unknown)} buffer-sampled outputs did not match any "
        f"engine.apply_symmetry output across 12 syms — kernel divergence."
    )
    # The buffer should exhaust every unique binding output class in 4000 draws.
    missing = set(binding_keys.keys()) - buffer_keys
    assert not missing, (
        f"[{name}] binding produced {len(binding_keys)} unique outputs but "
        f"only {len(buffer_keys)} seen from buffer — {len(missing)} unseen "
        f"(coverage gap, not a kernel drift; increase n_draws if flaky)."
    )
