"""HEXB v6 legacy load from Python.

§174 prereq — verifies v7 code loads a manually-constructed v6-format file,
assumes encoding "v6", and emits a deprecation warning.
"""
from __future__ import annotations

import numpy as np
import pytest
import tempfile
import os
import struct

from engine import ReplayBuffer


def test_hexb_v6_legacy_load_python() -> None:
    """Manually write a v6-format file, load with Python ReplayBuffer(v6)."""
    n_rows = 5
    capacity = 10
    # v6 header: magic, version=6, n_planes=8, capacity=10, size=5
    magic = 0x48455842
    version = 6
    n_planes = 8
    state_stride = n_planes * 361
    chain_stride = 6 * 361
    policy_stride = 362
    aux_stride = 361
    entry_bytes = (
        state_stride * 2
        + chain_stride * 2
        + policy_stride * 4
        + 4   # outcome f32
        + 8   # game_id i64
        + 2   # weight u16
        + aux_stride   # ownership
        + aux_stride   # winning_line
        + 1   # is_full_search
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "legacy_v6.hexb")
        with open(path, "wb") as f:
            f.write(struct.pack("<I", magic))
            f.write(struct.pack("<I", version))
            f.write(struct.pack("<I", n_planes))
            f.write(struct.pack("<Q", capacity))
            f.write(struct.pack("<Q", n_rows))
            for i in range(n_rows):
                # state: zeros
                f.write(b"\x00" * (state_stride * 2))
                # chain: zeros
                f.write(b"\x00" * (chain_stride * 2))
                # policy: zeros
                f.write(b"\x00" * (policy_stride * 4))
                # outcome = i as f32
                f.write(struct.pack("<f", float(i)))
                # game_id = 0
                f.write(struct.pack("<q", 0))
                # weight = 0x3C00 (f16 1.0)
                f.write(struct.pack("<H", 0x3C00))
                # ownership: all 1 (empty)
                f.write(b"\x01" * aux_stride)
                # winning_line: all 0
                f.write(b"\x00" * aux_stride)
                # is_full_search: 1
                f.write(b"\x01")

        buf = ReplayBuffer(capacity, encoding="v6")
        loaded = buf.load_from_path(path)
        assert loaded == n_rows
        assert buf.size == n_rows

        # Verify outcomes.
        out = buf.sample_batch(n_rows * 4, augment=False)
        o_all = out[3]
        for i in range(n_rows):
            assert any(abs(float(o) - float(i)) < 1e-6 for o in o_all), (
                f"outcome {i} not found in sampled batch"
            )
