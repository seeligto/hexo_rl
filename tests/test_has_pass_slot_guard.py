"""§173 A4 — H5-α guard: has_pass_slot=false encodings don't OOB on apply_sym.

v8 has has_pass_slot=false and policy_logit_count=625 (25×25, no extra slot).
If apply_sym unconditionally copied dst_policy[n_cells] = src_policy[n_cells]
(as it did pre-A4), that would be index 625 into a 625-element slice — OOB.

This test verifies:
1. v8 ReplayBuffer accepts the correct policy shape (625 elements, no pass slot).
2. Push + augmented sample_batch completes without panic or OOB for v8.
3. v6 pass slot (index 361) is still preserved under augmentation.

Closes HAZARD H5-α (§173 A4).
"""
from __future__ import annotations

import numpy as np
import pytest
from engine import ReplayBuffer


# ── v8 (has_pass_slot=false) ─────────────────────────────────────────────────

def _v8_entry():
    """Return a random (state, chain, policy, outcome, own, wl) for v8 encoding."""
    state  = np.random.randn(11, 25, 25).astype(np.float16)
    chain  = np.zeros((6, 25, 25), dtype=np.float16)
    policy = np.abs(np.random.randn(625).astype(np.float32))
    policy /= policy.sum()
    own    = np.ones(625, dtype=np.uint8)
    wl     = np.zeros(625, dtype=np.uint8)
    return state, chain, policy, 0.0, own, wl


def test_v8_push_accepts_correct_policy_shape() -> None:
    """v8 buffer must accept 625-element policy (no pass slot)."""
    buf = ReplayBuffer(10, encoding="v8")
    state, chain, policy, out, own, wl = _v8_entry()
    # Should not raise.
    buf.push(state, chain, policy, out, own, wl)
    assert buf.size == 1


def test_v8_push_rejects_v6_policy_shape() -> None:
    """v8 buffer must reject 362-element policy (v6 shape)."""
    buf = ReplayBuffer(10, encoding="v8")
    state = np.zeros((11, 25, 25), dtype=np.float16)
    chain = np.zeros((6, 25, 25), dtype=np.float16)
    policy_wrong = np.zeros(362, dtype=np.float32)  # v6 shape
    own = np.ones(625, dtype=np.uint8)
    wl  = np.zeros(625, dtype=np.uint8)
    with pytest.raises((ValueError, Exception)):
        buf.push(state, chain, policy_wrong, 0.0, own, wl)


def test_v8_sample_batch_no_oob() -> None:
    """sample_batch with augment=True must not panic for v8 (H5-α guard)."""
    buf = ReplayBuffer(20, encoding="v8")
    for _ in range(10):
        buf.push(*_v8_entry())
    assert buf.size == 10

    # With augment=True, apply_sym is called for each sample.
    # Pre-A4 this would OOB on the pass-slot copy path.
    for _ in range(5):
        s, c, p, o, own, wl, _ifs = buf.sample_batch(8, augment=True)
        assert s.shape   == (8, 11, 25, 25), f"state shape: {s.shape}"
        assert c.shape   == (8, 6,  25, 25), f"chain shape: {c.shape}"
        assert p.shape   == (8, 625),        f"policy shape: {p.shape}"
        assert own.shape == (8, 25, 25),     f"own shape: {own.shape}"
        assert wl.shape  == (8, 25, 25),     f"wl shape: {wl.shape}"


def test_v8_policy_values_in_range_after_augment() -> None:
    """Augmented v8 policy must stay non-negative (scatter, not OOB garbage)."""
    buf = ReplayBuffer(20, encoding="v8")
    for _ in range(15):
        buf.push(*_v8_entry())

    _, _, p, _, _, _, _ = buf.sample_batch(15, augment=True)
    assert (p >= 0.0).all(), "v8 augmented policy must have no negative values"


# ── v6 pass slot still works ─────────────────────────────────────────────────

def _v6_entry_with_known_pass(pass_val: float):
    """v6 entry with a known pass-slot value."""
    state  = np.zeros((8, 19, 19), dtype=np.float16)
    chain  = np.zeros((6, 19, 19), dtype=np.float16)
    policy = np.zeros(362, dtype=np.float32)
    policy[361] = pass_val  # pass slot at index 361 = n_cells
    own    = np.ones(361, dtype=np.uint8)
    wl     = np.zeros(361, dtype=np.uint8)
    return state, chain, policy, 0.0, own, wl


def test_v6_pass_slot_preserved_under_augmentation() -> None:
    """v6 pass action (index 361) must survive all 12 symmetries unchanged."""
    buf = ReplayBuffer(20, encoding="v6")
    pass_val = 0.123
    for _ in range(15):
        buf.push(*_v6_entry_with_known_pass(pass_val))

    # Without augmentation — should be exact.
    _, _, p_noaug, _, _, _, _ = buf.sample_batch(15, augment=False)
    pass_logits = p_noaug[:, 361]
    assert np.allclose(pass_logits, pass_val, atol=1e-3), \
        f"no-aug pass slot mismatch: {pass_logits}"

    # With augmentation — pass slot is invariant under hex symmetry.
    _, _, p_aug, _, _, _, _ = buf.sample_batch(15, augment=True)
    pass_logits_aug = p_aug[:, 361]
    assert np.allclose(pass_logits_aug, pass_val, atol=1e-3), \
        f"aug pass slot mismatch: {pass_logits_aug}"
