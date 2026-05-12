"""§173 A8 — Buffer push/sample byte-roundtrip identity, parameterized over registry.

Validates the α multi-window self-play prerequisite: a position pushed with
spec X reappears bit-identical when sampled with augment=False (sym 0 ==
identity) under the same encoding.

Per §173 A8 design (`docs/designs/encoding_alpha_multiwindow_selfplay_design.md`
§8.1 G1): zero shape errors / zero spec-mismatch errors / byte-identity for
every registered encoding. v8 family additionally checks `has_pass_slot=False`
path (sample.rs:220 H1-α fix from A4).

Storage contract (engine/src/replay_buffer/{push,sample}.rs):
  - state:        f16 of len = n_planes × trunk_size²
  - chain_planes: f16 of len = 6 × trunk_size²       (N_CHAIN_PLANES constant)
  - policy:       f32 of len = policy_logit_count
  - outcome:      f32 scalar
  - ownership:    u8  of len = aux_stride (trunk_size²)
  - winning_line: u8  of len = aux_stride

augment=False forces sym_idx=0 (identity); byte-identity must hold across the
full row irrespective of encoding.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine import ReplayBuffer
from hexo_rl.encoding import all_specs, lookup

_REGISTERED: list[str] = sorted(s.name for s in all_specs())
_N_CHAIN_PLANES = 6  # mirrors engine/src/replay_buffer/sym_tables.rs::N_CHAIN_PLANES


def _make_row(spec, rng: np.random.Generator) -> tuple:
    """Build one synthetic (state, chain, policy, outcome, own, wl) row for `spec`.

    Uses small finite f16-representable values so cast through the f16 storage
    is exact (no rounding). Policy is left in raw f32 — sample preserves bytes.
    """
    n_cells = spec.trunk_size * spec.trunk_size
    # Use a coarse uniform grid that is exactly f16-representable to guarantee
    # byte-identity through f16 storage. Values in {-1.0, 0.0, 0.5, 1.0} are
    # exact in f16.
    state_grid = rng.choice(
        np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float16),
        size=(spec.n_planes, spec.trunk_size, spec.trunk_size),
    ).astype(np.float16)
    chain_grid = rng.choice(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float16),
        size=(_N_CHAIN_PLANES, spec.trunk_size, spec.trunk_size),
    ).astype(np.float16)
    # Policy: normalize a positive vector → f32 byte-pattern preserved by sample.
    p = rng.random(spec.policy_logit_count).astype(np.float32) + 1e-6
    p /= p.sum()
    outcome = float(rng.choice([-1.0, 0.0, 1.0]))
    # Ownership values {0,1,2}; winning_line bit-mask values {0,1}.
    own = rng.integers(0, 3, size=n_cells, dtype=np.uint8)
    wl = rng.integers(0, 2, size=n_cells, dtype=np.uint8)
    return state_grid, chain_grid, p, outcome, own, wl


@pytest.mark.parametrize("encoding_name", _REGISTERED)
def test_buffer_byte_roundtrip(encoding_name: str) -> None:
    """Push N rows, sample 1-by-1 with augment=False, assert byte-identity per row."""
    spec = lookup(encoding_name)
    rng = np.random.default_rng(seed=hash(encoding_name) & 0xFFFFFFFF)

    n_rows = 32
    buf = ReplayBuffer(n_rows, encoding=encoding_name)

    pushed: list[tuple] = []
    for _ in range(n_rows):
        row = _make_row(spec, rng)
        buf.push(*row)
        pushed.append(row)

    assert buf.size == n_rows, (
        f"{encoding_name}: buf.size={buf.size} != pushed {n_rows}"
    )

    # Sample one full batch (all rows). sample_batch with augment=False uses
    # sym 0 (identity). Index selection is randomized — match by outcome value
    # OR re-sample until we cover every slot. Easier: sample a much larger
    # batch and verify EVERY pushed row appears unchanged at least once.
    out = buf.sample_batch(n_rows * 8, augment=False)
    s_out, c_out, p_out, o_out, own_out, wl_out, _ifs = out

    # Output shapes must match spec.
    assert s_out.shape == (n_rows * 8, spec.n_planes, spec.trunk_size, spec.trunk_size), (
        f"{encoding_name}: state out shape {s_out.shape} != "
        f"({n_rows*8}, {spec.n_planes}, {spec.trunk_size}, {spec.trunk_size})"
    )
    assert c_out.shape == (n_rows * 8, _N_CHAIN_PLANES, spec.trunk_size, spec.trunk_size), (
        f"{encoding_name}: chain out shape {c_out.shape}"
    )
    assert p_out.shape == (n_rows * 8, spec.policy_logit_count), (
        f"{encoding_name}: policy out shape {p_out.shape}"
    )
    assert s_out.dtype == np.float16
    assert c_out.dtype == np.float16
    assert p_out.dtype == np.float32
    assert own_out.dtype == np.uint8
    assert wl_out.dtype == np.uint8

    # Build identity-keyed lookup of pushed rows. Key = (outcome, state.tobytes())
    # — outcome alone disambiguates ~3 buckets, full state-bytes settles uniqueness.
    pushed_keys = {
        (row[3], row[0].tobytes()): row for row in pushed
    }
    # Track which pushed rows we've matched.
    matched: set[bytes] = set()

    for b in range(s_out.shape[0]):
        s_b = np.ascontiguousarray(s_out[b])
        key = (float(o_out[b]), s_b.tobytes())
        if key not in pushed_keys:
            # Outcomes are randomized; if state-bytes don't appear in pushed,
            # the buffer corrupted a row. Fail loud.
            pytest.fail(
                f"{encoding_name}: sampled row {b} state-bytes (outcome={o_out[b]}) "
                f"not found among pushed rows (corruption)"
            )
        ref = pushed_keys[key]
        # Byte-exact per field.
        assert s_b.tobytes() == ref[0].tobytes(), f"{encoding_name}: state bytes drift row {b}"
        assert np.ascontiguousarray(c_out[b]).tobytes() == ref[1].tobytes(), (
            f"{encoding_name}: chain bytes drift row {b}"
        )
        assert np.ascontiguousarray(p_out[b]).tobytes() == ref[2].tobytes(), (
            f"{encoding_name}: policy bytes drift row {b}"
        )
        # Outcome already matched via key, but assert for symmetry.
        assert float(o_out[b]) == ref[3], f"{encoding_name}: outcome drift row {b}"
        assert np.ascontiguousarray(own_out[b]).tobytes() == ref[4].tobytes(), (
            f"{encoding_name}: ownership bytes drift row {b}"
        )
        assert np.ascontiguousarray(wl_out[b]).tobytes() == ref[5].tobytes(), (
            f"{encoding_name}: winning_line bytes drift row {b}"
        )
        matched.add(ref[0].tobytes())

    # With batch_size = 8 × capacity and weighted sampling, every slot should
    # appear ≥ 1 time with overwhelming probability. We tolerate up to 1
    # un-sampled slot rather than re-running until full coverage (probabilistic).
    assert len(matched) >= n_rows - 1, (
        f"{encoding_name}: only matched {len(matched)} of {n_rows} pushed rows "
        f"(weighted sampler skipped > 1 slot — unexpected at batch×8 cover)"
    )


@pytest.mark.parametrize("encoding_name", _REGISTERED)
def test_buffer_ctor_storage_sizing(encoding_name: str) -> None:
    """Sanity: buffer ctor pre-allocates without crash; size 0 at start.

    Regression guard for §173 A4 stride wiring — if any of state_stride /
    chain_stride / policy_stride / aux_stride is mis-computed, allocation
    capacity vs push contract diverges and push fails at row 0.
    """
    spec = lookup(encoding_name)
    buf = ReplayBuffer(8, encoding=encoding_name)
    assert buf.size == 0
    assert buf.capacity == 8

    # Single push smoke — exercises every storage stride at row index 0.
    rng = np.random.default_rng(0)
    row = _make_row(spec, rng)
    buf.push(*row)
    assert buf.size == 1


@pytest.mark.parametrize("encoding_name", _REGISTERED)
def test_buffer_push_rejects_wrong_shape(encoding_name: str) -> None:
    """Wrong-shape push must raise (not silently corrupt). Defense for §173 A4
    stride-wiring regressions: shape gate must use spec.*_stride, not const.
    """
    spec = lookup(encoding_name)
    buf = ReplayBuffer(4, encoding=encoding_name)
    # Build a row sized for the *other* board_size — exercises shape rejection.
    other_size = 19 if spec.trunk_size != 19 else 25
    n_cells_wrong = other_size * other_size
    bad_state = np.zeros((spec.n_planes, other_size, other_size), dtype=np.float16)
    bad_chain = np.zeros((_N_CHAIN_PLANES, other_size, other_size), dtype=np.float16)
    bad_policy = np.zeros(spec.policy_logit_count, dtype=np.float32)
    bad_own = np.ones(n_cells_wrong, dtype=np.uint8)
    bad_wl = np.zeros(n_cells_wrong, dtype=np.uint8)
    with pytest.raises(Exception):  # PyValueError surfaces as ValueError
        buf.push(bad_state, bad_chain, bad_policy, 0.0, bad_own, bad_wl)
    assert buf.size == 0, f"{encoding_name}: buffer accepted wrong-shape push"
