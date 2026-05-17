"""§172 A4.3 — training/scripts registry plumbing.

Covers the new spec-derived buffer allocators + the metadata-preference
load path on Trainer.load_checkpoint. The trainer-load tests are
augmented in tests/test_trainer_encoding_load.py with explicit
metadata-vs-shape-inference assertions; this file owns the buffer-shape
contract.
"""
from __future__ import annotations

import numpy as np

from hexo_rl.encoding import lookup
from hexo_rl.training.batch_assembly import BatchBuffers, allocate_batch_buffers
from hexo_rl.training.recency_buffer import RecentBuffer


def test_allocate_batch_buffers_v6_default_trunk_19() -> None:
    """Legacy callers (no trunk_size kwarg) get v6 (B, 8, 19, 19) shapes."""
    buf = allocate_batch_buffers(batch_size=4, n_actions=362)
    assert isinstance(buf, BatchBuffers)
    assert buf.states.shape == (4, 8, 19, 19)
    assert buf.chain_planes.shape == (4, 6, 19, 19)
    assert buf.ownership.shape == (4, 19, 19)
    assert buf.winning_line.shape == (4, 19, 19)
    assert buf.policies.shape == (4, 362)


def test_allocate_batch_buffers_v6w25_trunk_25() -> None:
    """trunk_size=25 returns 25×25 spatials and the v6w25 policy width."""
    spec = lookup("v6w25")
    buf = allocate_batch_buffers(
        batch_size=4,
        n_actions=spec.policy_logit_count,
        trunk_size=spec.trunk_size,
    )
    assert buf.states.shape == (4, 8, 25, 25)
    assert buf.chain_planes.shape == (4, 6, 25, 25)
    assert buf.ownership.shape == (4, 25, 25)
    assert buf.winning_line.shape == (4, 25, 25)
    assert buf.policies.shape == (4, 626)


def test_allocate_batch_buffers_explicit_aux_stride_accepted() -> None:
    """aux_stride parameter accepted (currently reserved hook)."""
    buf = allocate_batch_buffers(
        batch_size=2, n_actions=626, trunk_size=25, aux_stride=625,
    )
    assert buf.states.shape == (2, 8, 25, 25)


def test_recency_buffer_state_shape_drives_chain_shape() -> None:
    """v6w25 RecentBuffer derives chain spatials from state_shape."""
    rb = RecentBuffer(
        capacity=8,
        state_shape=(8, 25, 25),
        policy_len=626,
        aux_stride=625,
    )
    assert rb._states.shape == (8, 8, 25, 25)
    assert rb._chain_planes.shape == (8, 6, 25, 25)
    assert rb._policies.shape == (8, 626)
    assert rb._ownership.shape == (8, 625)
    assert rb._winning_line.shape == (8, 625)


def test_recency_buffer_v6_default_unchanged() -> None:
    """Default ctor still produces v6 (capacity, 6, 19, 19) chain planes."""
    rb = RecentBuffer(capacity=4)
    assert rb._states.shape == (4, 8, 19, 19)
    assert rb._chain_planes.shape == (4, 6, 19, 19)
    assert rb._policies.shape == (4, 362)
    assert rb._ownership.shape == (4, 361)


def test_recency_buffer_v8_state_shape_drives_chain_shape() -> None:
    """v8 (11, 25, 25) state shape produces (capacity, 6, 25, 25) chains."""
    rb = RecentBuffer(
        capacity=4,
        state_shape=(11, 25, 25),
        policy_len=625,
        aux_stride=625,
    )
    assert rb._states.shape == (4, 11, 25, 25)
    # Chain plane channel count is 6 regardless of state-channel count.
    assert rb._chain_planes.shape == (4, 6, 25, 25)


# Cycle 3 Wave 8 Batch C (FF.10, 2026-05-17) retired the
# `_legacy_spec_for_registry_name` shim that bridged registry names into
# the WireFormatSpec dataclass. Wire-format scalars are now read directly
# off the registry record at `hexo_rl.encoding.lookup(name)`.
#
# The 4 retired tests below covered:
#   - v7full → v6 wire (name="v6", board_size=19, cw=19): registry
#     lookup("v7full") returns spec.name="v7full" (registry-name, not
#     wire-family) but spec.board_size=19 / spec.legal_move_radius=5
#     (v6 wire geometry inherited via the registry's v7full alias).
#   - v8_canvas_realness → v8 wire (name="v8"): registry lookup gives
#     spec.name="v8_canvas_realness" (registry-name) with v8 geometry.
#   - unknown name → ValueError: registry lookup raises
#     `EncodingRegistryError` rather than the legacy "no wire-format
#     mapping" message.
#   - canonical names round-trip: superseded by
#     `tests/test_encoding_registry.py::test_lookup_round_trips`.
