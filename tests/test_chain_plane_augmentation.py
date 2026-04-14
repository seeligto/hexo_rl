"""Byte-exact augmentation invariance for Q13 chain-length planes (C3 gate).

Verifies that the Rust `ReplayBuffer.sample_batch(augment=True)` chain-plane
scatter matches a ground-truth path that transforms stones by each hex
symmetry first and then computes fresh chain planes.

This is THE load-bearing correctness test for the axis-permutation scatter
landed in C3 (engine/src/replay_buffer/sample.rs). Failure means the 18..23
plane block is being scattered with the wrong source plane index and training
data is silently corrupted on augmentation.

Test positions use stones near the board origin so every hex symmetry
transform keeps them in-window — edge-mapping asymmetry would introduce
unrelated failure modes.
"""
from __future__ import annotations
import numpy as np
import pytest

from engine import ReplayBuffer
from hexo_rl.env.game_state import _CHAIN_CAP, _compute_chain_planes
from hexo_rl.utils.constants import BOARD_SIZE

CHANNELS   = 24
N_ACTIONS  = BOARD_SIZE * BOARD_SIZE + 1
AUX_STRIDE = BOARD_SIZE * BOARD_SIZE
HALF       = (BOARD_SIZE - 1) // 2  # 9


def _flat_to_axial(idx: int) -> tuple[int, int]:
    qi, ri = divmod(idx, BOARD_SIZE)
    return qi - HALF, ri - HALF


def _axial_to_flat(q: int, r: int) -> int | None:
    qi, ri = q + HALF, r + HALF
    if 0 <= qi < BOARD_SIZE and 0 <= ri < BOARD_SIZE:
        return qi * BOARD_SIZE + ri
    return None


def _apply_sym_to_coord(q: int, r: int, sym_idx: int) -> tuple[int, int]:
    """Replicate engine/src/replay_buffer/sym_tables.rs:95-119 exactly:
    reflection (q,r)->(r,q) applied first, then n_rot × 60° via (q,r)->(-r, q+r)."""
    reflect = sym_idx >= 6
    n_rot = sym_idx % 6
    if reflect:
        q, r = r, q
    for _ in range(n_rot):
        q, r = -r, q + r
    return q, r


def _transform_stones(stones: np.ndarray, sym_idx: int) -> np.ndarray:
    """Apply hex symmetry to a (19,19) stone mask by scattering individual cells.
    Cells mapping out of the window are dropped — matches the scatter-drop
    behaviour of the Rust apply_sym kernel."""
    out = np.zeros_like(stones)
    for flat in range(BOARD_SIZE * BOARD_SIZE):
        if stones.flat[flat] == 0:
            continue
        src_q, src_r = _flat_to_axial(flat)
        dst_q, dst_r = _apply_sym_to_coord(src_q, src_r, sym_idx)
        dst = _axial_to_flat(dst_q, dst_r)
        if dst is not None:
            out.flat[dst] = stones.flat[flat]
    return out


def _build_state_tensor(cur: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Build a (24, 19, 19) float16 state tensor matching the to_tensor layout.
    Planes 0 = cur, 8 = opp, 16/17 = scalar zero, 1..7 + 9..15 = zero,
    18..23 = Q13 chain planes computed from cur/opp."""
    tensor = np.zeros((CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    tensor[0] = cur
    tensor[8] = opp
    chain = _compute_chain_planes(cur.astype(np.float32), opp.astype(np.float32))
    tensor[18:24] = chain.astype(np.float16) / np.float16(_CHAIN_CAP)
    return tensor


def _decode_chain_planes(state_out: np.ndarray) -> np.ndarray:
    """Extract chain planes from a sampled (24, 19, 19) state and denormalize
    back to int8 values in [0, 6] for comparison with fresh recomputation."""
    chain_f16 = state_out[18:24]
    # Round-trip /6 then *6 is not byte-exact in f16, so compare values
    # rounded to nearest integer and then cast to int8.
    chain_f32 = chain_f16.astype(np.float32) * float(_CHAIN_CAP)
    return np.round(chain_f32).astype(np.int8)


# ---------------------------------------------------------------------------
# Test positions — stones near origin so every symmetry keeps them in-window
# ---------------------------------------------------------------------------

def _pos_isolated_stone() -> tuple[np.ndarray, np.ndarray]:
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opp = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    cur[HALF + 0, HALF + 0] = 1.0  # (0, 0)
    return cur, opp


def _pos_open_three_axis0() -> tuple[np.ndarray, np.ndarray]:
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opp = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for q in (-1, 0, 1):
        cur[HALF + q, HALF + 0] = 1.0
    return cur, opp


def _pos_mixed_crosses() -> tuple[np.ndarray, np.ndarray]:
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opp = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for q in (0, 1, 2):
        cur[HALF + q, HALF + 0] = 1.0  # axis0 run
    for r in (1, 2):
        cur[HALF + 0, HALF + r] = 1.0  # axis1 run sharing (0,0)
    # Opponent stones nearby but not adjacent.
    opp[HALF + (-2), HALF + 0] = 1.0
    opp[HALF + 0, HALF + (-2)] = 1.0
    return cur, opp


def _pos_asymmetric() -> tuple[np.ndarray, np.ndarray]:
    cur = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    opp = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Pure axis1-only pattern — every sym maps it into a different axis pair,
    # so all 12 outputs should be distinguishable from each other.
    for r in (0, 1, 2, 3):
        cur[HALF + 0, HALF + r] = 1.0
    return cur, opp


POSITIONS = [
    ("isolated_stone", _pos_isolated_stone),
    ("open_three_axis0", _pos_open_three_axis0),
    ("mixed_crosses", _pos_mixed_crosses),
    ("asymmetric_axis1", _pos_asymmetric),
]


@pytest.mark.parametrize("pos_name,pos_fn", POSITIONS)
def test_chain_plane_augmentation_byte_exact(pos_name, pos_fn):
    """For each of 4 test positions, sample the buffer many times with
    augment=True and verify every unique chain-plane output matches a
    ground-truth computed on symmetry-transformed stones."""
    cur, opp = pos_fn()

    # Ground-truth: for each of the 12 syms, transform stones and recompute
    # chain planes fresh. The transform uses the same reflect-then-rotate
    # composition as SymTables::new().
    expected_chain_by_key: dict[bytes, int] = {}
    for sym_idx in range(12):
        rot_cur = _transform_stones(cur, sym_idx)
        rot_opp = _transform_stones(opp, sym_idx)
        expected_chain = _compute_chain_planes(rot_cur, rot_opp)  # int8 (6,19,19)
        expected_chain_by_key[expected_chain.tobytes()] = sym_idx

    # Push a single original sample into a fresh buffer.
    buf = ReplayBuffer(4)
    state = _build_state_tensor(cur, opp)
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    policy[0] = 1.0
    own = np.ones(AUX_STRIDE, dtype=np.uint8)
    wl = np.zeros(AUX_STRIDE, dtype=np.uint8)
    buf.push(state, policy, 0.0, own, wl)

    # Sample many times with augment=True, collect which syms appear.
    seen_syms: set[int] = set()
    unknown_keys: list[bytes] = []
    N_SAMPLES = 400
    for _ in range(N_SAMPLES):
        states_out, _, _, _, _ = buf.sample_batch(1, augment=True)
        sampled = _decode_chain_planes(np.asarray(states_out[0]))
        key = sampled.tobytes()
        if key in expected_chain_by_key:
            seen_syms.add(expected_chain_by_key[key])
        else:
            unknown_keys.append(key)

    assert not unknown_keys, (
        f"[{pos_name}] {len(unknown_keys)} / {N_SAMPLES} sampled chain-plane "
        f"outputs did not match any ground-truth symmetry transform"
    )
    # For truly symmetric positions (e.g. isolated_stone, which is invariant
    # under all 12 hex symmetries), many syms collapse to a single output —
    # the dict dedup keeps only one. Assert we see EVERY unique expected output.
    unique_expected = set(expected_chain_by_key.values())
    assert seen_syms == unique_expected, (
        f"[{pos_name}] seen syms {sorted(seen_syms)} != expected unique "
        f"syms {sorted(unique_expected)} after {N_SAMPLES} samples"
    )


def test_positions_span_multiple_sym_outputs():
    """Sanity check: our four test positions collectively exercise more than
    one unique symmetry output. An isolated stone is fully symmetric (all 12
    outputs identical); the other three positions should each produce at
    least 2 distinct outputs, giving the main invariance test discriminatory
    power against a scatter-plane-remap that silently does nothing."""
    for name, pos_fn in POSITIONS:
        if name == "isolated_stone":
            continue  # fully symmetric, skip
        cur, opp = pos_fn()
        keys = set()
        for sym_idx in range(12):
            rot_cur = _transform_stones(cur, sym_idx)
            rot_opp = _transform_stones(opp, sym_idx)
            chain = _compute_chain_planes(rot_cur, rot_opp)
            keys.add(chain.tobytes())
        assert len(keys) >= 2, (
            f"[{name}] has only {len(keys)} unique sym outputs — too symmetric to "
            f"exercise axis permutation logic"
        )
