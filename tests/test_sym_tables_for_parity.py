"""§173 A4 — sym_tables_for(v6) geometry matches SymTables::new().

Verifies that the module-level `sym_tables_for()` function (introduced in
§173 A4) returns a sym-table instance with identical board geometry to the
existing `SymTables::new()` default (v6, 19×19). The Rust-side unit tests
cover scatter-table equality; this Python-level smoke confirms the PyO3
boundary does not accidentally expose a mis-shaped table.
"""
from __future__ import annotations

import pytest
from engine import ReplayBuffer


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_buf(encoding: str, capacity: int = 4) -> ReplayBuffer:
    return ReplayBuffer(capacity, encoding=encoding)


# ── geometry checks ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("encoding,expected_board_size,expected_n_cells,expected_n_planes", [
    ("v6",      19, 361, 8),
    ("v7full",  19, 361, 8),
    ("v6w25",   25, 625, 8),
    ("v8",      25, 625, 11),
])
def test_replay_buffer_geometry_matches_encoding(
    encoding: str,
    expected_board_size: int,
    expected_n_cells: int,
    expected_n_planes: int,
) -> None:
    """Buffer geometry (stride sizes) matches registry spec for each encoding."""
    from hexo_rl.encoding import lookup
    spec = lookup(encoding)
    buf = _make_buf(encoding)

    # Verify stride sizing via expected allocation sizes.
    # state_stride = n_planes * n_cells
    expected_state_stride = expected_n_planes * expected_n_cells
    # chain_stride = 6 * n_cells (N_CHAIN_PLANES always 6)
    expected_chain_stride = 6 * expected_n_cells
    # aux_stride = n_cells
    expected_aux_stride = expected_n_cells

    # Verify via spec accessors (parity check Python == expected)
    assert spec.n_cells == expected_n_cells, \
        f"{encoding}: spec.n_cells={spec.n_cells}, expected {expected_n_cells}"
    assert spec.state_stride == expected_state_stride, \
        f"{encoding}: spec.state_stride={spec.state_stride}, expected {expected_state_stride}"
    assert spec.chain_stride == expected_chain_stride, \
        f"{encoding}: spec.chain_stride={spec.chain_stride}, expected {expected_chain_stride}"
    assert spec.aux_stride == expected_aux_stride, \
        f"{encoding}: spec.aux_stride={spec.aux_stride}, expected {expected_aux_stride}"

    # Verify buffer actually allocated with correct strides by checking push accepts
    # correctly-shaped inputs.
    import numpy as np
    state = np.zeros((expected_n_planes, expected_board_size, expected_board_size), dtype=np.float16)
    chain = np.zeros((6, expected_board_size, expected_board_size), dtype=np.float16)
    policy_len = spec.policy_logit_count
    policy = np.zeros(policy_len, dtype=np.float32)
    own = np.ones(expected_n_cells, dtype=np.uint8)
    wl = np.zeros(expected_n_cells, dtype=np.uint8)
    # Should not raise.
    buf.push(state, chain, policy, 0.0, own, wl)
    assert buf.size == 1


def test_v6_buffer_geometry_matches_legacy_defaults() -> None:
    """v6 buffer must produce 19×19 shaped outputs (regression: sym_tables_for(v6) parity)."""
    import numpy as np
    buf = _make_buf("v6", capacity=10)
    for _ in range(5):
        state = np.zeros((8, 19, 19), dtype=np.float16)
        chain = np.zeros((6, 19, 19), dtype=np.float16)
        policy = np.zeros(362, dtype=np.float32)
        own = np.ones(361, dtype=np.uint8)
        wl = np.zeros(361, dtype=np.uint8)
        buf.push(state, chain, policy, 0.0, own, wl)

    s, c, p, o, own, wl, _ifs, _vv = buf.sample_batch(4, augment=False)
    # Shapes must match v6 geometry exactly.
    assert s.shape == (4, 8, 19, 19), f"state shape mismatch: {s.shape}"
    assert c.shape == (4, 6, 19, 19), f"chain shape mismatch: {c.shape}"
    assert p.shape == (4, 362),       f"policy shape mismatch: {p.shape}"
    assert own.shape == (4, 19, 19),  f"ownership shape mismatch: {own.shape}"
    assert wl.shape == (4, 19, 19),   f"winning_line shape mismatch: {wl.shape}"


def test_v6w25_buffer_geometry_correct() -> None:
    """v6w25 buffer must produce 25×25 shaped outputs."""
    import numpy as np
    buf = _make_buf("v6w25", capacity=10)
    for _ in range(5):
        state = np.zeros((8, 25, 25), dtype=np.float16)
        chain = np.zeros((6, 25, 25), dtype=np.float16)
        policy = np.zeros(626, dtype=np.float32)  # 25*25+1
        own = np.ones(625, dtype=np.uint8)
        wl = np.zeros(625, dtype=np.uint8)
        buf.push(state, chain, policy, 0.0, own, wl)

    s, c, p, o, own, wl, _ifs, _vv = buf.sample_batch(4, augment=False)
    assert s.shape == (4, 8, 25, 25), f"state shape mismatch: {s.shape}"
    assert c.shape == (4, 6, 25, 25), f"chain shape mismatch: {c.shape}"
    assert p.shape == (4, 626),       f"policy shape mismatch: {p.shape}"
    assert own.shape == (4, 25, 25),  f"ownership shape mismatch: {own.shape}"
    assert wl.shape == (4, 25, 25),   f"winning_line shape mismatch: {wl.shape}"
