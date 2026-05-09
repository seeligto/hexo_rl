"""§172 A4.3 — training/scripts registry plumbing.

Covers the new spec-derived buffer allocators + the metadata-preference
load path on Trainer.load_checkpoint. The trainer-load tests are
augmented in tests/test_trainer_encoding_load.py with explicit
metadata-vs-shape-inference assertions; this file owns the buffer-shape
contract.
"""
from __future__ import annotations

import numpy as np
import pytest

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


def test_legacy_bridge_v7full_to_v6_spec() -> None:
    """v7full (registry-only) bridges to legacy v6 spec for downstream consumers."""
    from hexo_rl.training.trainer import _legacy_spec_for_registry_name

    legacy = _legacy_spec_for_registry_name("v7full")
    assert legacy.version == "v6"
    assert legacy.board_size == 19
    assert legacy.cluster_window_size == 19


def test_legacy_bridge_v8_canvas_realness_to_v8_spec() -> None:
    """v8_canvas_realness (registry-only) bridges to legacy v8 spec."""
    from hexo_rl.training.trainer import _legacy_spec_for_registry_name

    legacy = _legacy_spec_for_registry_name("v8_canvas_realness")
    assert legacy.version == "v8"


def test_legacy_bridge_unknown_name_raises() -> None:
    from hexo_rl.training.trainer import _legacy_spec_for_registry_name

    with pytest.raises(ValueError, match="no legacy bridge"):
        _legacy_spec_for_registry_name("v999_imaginary")


def test_legacy_bridge_canonical_names_round_trip() -> None:
    """v6 / v6w25 / v8 names go through resolve_encoding cleanly."""
    from hexo_rl.training.trainer import _legacy_spec_for_registry_name

    assert _legacy_spec_for_registry_name("v6").version == "v6"
    assert _legacy_spec_for_registry_name("v6w25").version == "v6w25"
    assert _legacy_spec_for_registry_name("v8").version == "v8"
