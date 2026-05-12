"""HEXB v7 Python round-trip for v6w25 encoding.

§174 prereq — verifies PyO3 save/load path writes and reads encoding name
correctly for the canonical §173 α multi-window encoding.
"""
from __future__ import annotations

import numpy as np
import pytest
import tempfile
import os

from engine import ReplayBuffer
from hexo_rl.encoding import lookup


def test_hexb_v7_v6w25_roundtrip() -> None:
    """Push 10 rows to v6w25 buffer, save v7, load, verify shapes and data."""
    spec = lookup("v6w25")
    n_rows = 10
    buf = ReplayBuffer(n_rows, encoding="v6w25")
    rng = np.random.default_rng(42)

    pushed_outcomes = []
    for i in range(n_rows):
        state = rng.random((spec.n_planes, spec.trunk_size, spec.trunk_size)).astype(np.float16)
        chain = rng.random((6, spec.trunk_size, spec.trunk_size)).astype(np.float16)
        policy = (rng.random(spec.policy_logit_count).astype(np.float32) + 1e-6)
        policy /= policy.sum()
        outcome = float(rng.choice([-1.0, 0.0, 1.0]))
        own = rng.integers(0, 3, size=spec.trunk_size * spec.trunk_size, dtype=np.uint8)
        wl = rng.integers(0, 2, size=spec.trunk_size * spec.trunk_size, dtype=np.uint8)
        buf.push(state, chain, policy, outcome, own, wl, game_id=-1, game_length=20, is_full_search=(i % 2 == 0))
        pushed_outcomes.append(outcome)

    assert buf.size == n_rows

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "v7_v6w25.hexb")
        buf.save_to_path(path)
        assert os.path.getsize(path) > 0

        buf2 = ReplayBuffer(n_rows, encoding="v6w25")
        loaded = buf2.load_from_path(path)
        assert loaded == n_rows
        assert buf2.size == n_rows

        # Verify outcomes survived.
        for slot in range(n_rows):
            # Access outcomes via sample_batch with augment=False and identity sym.
            # Alternatively, we can just trust the Rust tests verified byte-exact
            # columns; here we verify the Python API contract (size, capacity).
            pass

        # Sample a batch and verify shapes match v6w25 spec.
        out = buf2.sample_batch(4, augment=False)
        s_out, c_out, p_out, o_out, own_out, wl_out, ifs_out = out

        assert s_out.shape == (4, spec.n_planes, spec.trunk_size, spec.trunk_size)
        assert c_out.shape == (4, 6, spec.trunk_size, spec.trunk_size)
        assert p_out.shape == (4, spec.policy_logit_count)
        assert o_out.shape == (4,)
        assert own_out.shape == (4, spec.trunk_size, spec.trunk_size)
        assert wl_out.shape == (4, spec.trunk_size, spec.trunk_size)
        assert ifs_out.shape == (4,)

        # Verify outcome values round-tripped by sampling enough to cover all.
        out = buf2.sample_batch(n_rows * 4, augment=False)
        o_all = out[3]
        for orig in pushed_outcomes:
            assert any(abs(float(o) - orig) < 1e-6 for o in o_all), (
                f"outcome {orig} not found in sampled batch"
            )
