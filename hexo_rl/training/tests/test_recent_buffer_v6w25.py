"""Test RecentBuffer generalises to v6w25 (25×25) board geometry.

§173 A8-fix: RecentBuffer default was hardcoded (8, 19, 19) which caused
grad_norm spikes when recency buffer was disabled as a workaround.
"""
from __future__ import annotations

import numpy as np
import pytest

from hexo_rl.training.recency_buffer import RecentBuffer


def test_recent_buffer_v6w25_shapes():
    """Construct, push, and sample with v6w25 dimensions."""
    buf = RecentBuffer(
        capacity=32,
        state_shape=(8, 25, 25),
        policy_len=626,
        aux_stride=625,
    )

    # Push 10 dummy positions
    for _ in range(10):
        buf.push(
            state=np.random.randn(8, 25, 25).astype(np.float16),
            chain_planes=np.random.randn(6, 25, 25).astype(np.float16),
            policy=np.random.dirichlet(np.ones(626)).astype(np.float32),
            outcome=1.0,
            ownership=np.random.randint(0, 3, size=625, dtype=np.uint8),
            winning_line=np.random.randint(0, 2, size=625, dtype=np.uint8),
            is_full_search=True,
        )

    assert buf.size == 10

    s, c, p, o, own, wl, ifs, vv = buf.sample(4)
    assert s.shape == (4, 8, 25, 25)
    assert c.shape == (4, 6, 25, 25)
    assert p.shape == (4, 626)
    assert o.shape == (4,)
    assert own.shape == (4, 625)
    assert wl.shape == (4, 625)
    assert ifs.shape == (4,)


def test_recent_buffer_v6_default_backward_compat():
    """Default ctor still produces v6 shapes (backward compat)."""
    buf = RecentBuffer(capacity=8)
    buf.push(
        state=np.zeros((8, 19, 19), dtype=np.float16),
        policy=np.zeros(362, dtype=np.float32),
        ownership=np.ones(361, dtype=np.uint8),
    )
    s, c, p, o, own, wl, ifs, vv = buf.sample(1)
    assert s.shape == (1, 8, 19, 19)
    assert p.shape == (1, 362)
    assert own.shape == (1, 361)
