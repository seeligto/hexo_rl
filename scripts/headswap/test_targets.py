"""Unit tests for the two-hot 65-bin target + scoring primitives.

Run: .venv/bin/python -m pytest scripts/headswap/test_targets.py -q
The red-team's declared attack surface (bin construction, tail-mass rule) —
these must be provably correct BEFORE any arm trains.
"""

import torch

from scripts.headswap.targets import (
    VALUE_BINS,
    BIN_WIDTH,
    LOSS_TAIL_BIN,
    support,
    scalar_to_two_hot,
    two_hot_ce_loss,
    decode_expected_value,
    loss_tail_mass,
)


def test_support_anchors():
    s = support()
    assert s.shape == (65,)
    assert s.dtype == torch.float32
    assert torch.equal(s[0], torch.tensor(-1.0))
    assert torch.equal(s[64], torch.tensor(1.0))
    assert torch.equal(s[32], torch.tensor(0.0))
    assert torch.equal(s[LOSS_TAIL_BIN], torch.tensor(-0.5))  # exact
    assert abs(BIN_WIDTH - 1.0 / 32.0) < 1e-12


def test_two_hot_mass_sums_to_one():
    z = torch.linspace(-1.0, 1.0, 401)
    t = scalar_to_two_hot(z)
    assert t.dtype == torch.float32
    assert t.shape == (401, VALUE_BINS)
    assert torch.allclose(t.sum(dim=1), torch.ones(401), atol=1e-6)
    assert (t >= 0).all()


def test_two_hot_hand_checked_bins():
    cases = {
        -1.0: {0: 1.0},
        -0.5: {16: 1.0},
        0.0: {32: 1.0},
        0.5: {48: 1.0},
        1.0: {64: 1.0},
        -0.9: {3: 0.8, 4: 0.2},   # pos = 0.1*32 = 3.2
        0.25: {40: 1.0},          # pos = 1.25*32 = 40
    }
    for z, expect in cases.items():
        t = scalar_to_two_hot(torch.tensor([z]))[0]
        for b, m in expect.items():
            assert abs(float(t[b]) - m) < 1e-5, (z, b, float(t[b]), m)
        assert abs(float(t.sum()) - 1.0) < 1e-6


def test_decode_is_exact_left_inverse():
    # decode(two_hot(z)) == z exactly (two-hot expectation over adjacent support
    # atoms equals z for any z in [-1, 1]). Log-of-target as logits -> softmax
    # recovers the target only approximately, so decode directly from the target
    # probabilities instead of round-tripping through log_softmax.
    z = torch.linspace(-1.0, 1.0, 257)
    t = scalar_to_two_hot(z)
    v = (t @ support()).clamp(-1.0, 1.0)
    assert torch.allclose(v, z, atol=1e-5), (v - z).abs().max()


def test_tail_mass_monotone_decreasing_in_z():
    z = torch.linspace(-1.0, 1.0, 401)
    # feed two-hot targets as near-one-hot logits (log of target, floored)
    logits = torch.log(scalar_to_two_hot(z).clamp_min(1e-12))
    tm = loss_tail_mass(logits)
    # non-increasing as z rises
    assert (tm[1:] - tm[:-1] <= 1e-4).all()
    # anchors: all loss mass at z=-1, none at z=+1, exactly-included at -0.5
    assert float(tm[0]) > 0.99          # z = -1 -> bin 0 (in tail)
    assert float(tm[-1]) < 0.01         # z = +1 -> bin 64 (out of tail)


def test_tail_mass_threshold_boundary_inclusive():
    # z = -0.5 lands entirely in bin 16 which IS in the tail sum -> ~1.0
    logits = torch.log(scalar_to_two_hot(torch.tensor([-0.5])).clamp_min(1e-12))
    assert float(loss_tail_mass(logits)[0]) > 0.99
    # z just above -0.5 spills mass into bin 17 (out of tail) -> < 1
    logits2 = torch.log(scalar_to_two_hot(torch.tensor([-0.49])).clamp_min(1e-12))
    assert float(loss_tail_mass(logits2)[0]) < 0.9


def test_ce_loss_mask_and_zero_guard():
    torch.manual_seed(0)
    logits = torch.randn(8, VALUE_BINS, requires_grad=True)
    z = torch.linspace(-1, 1, 8)
    # all-zero mask -> differentiable zero
    loss0 = two_hot_ce_loss(logits, z, mask=torch.zeros(8))
    assert float(loss0) == 0.0
    loss0.backward()  # must not error
    # partial mask matches manual mean over kept rows
    mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    loss = two_hot_ce_loss(logits, z, mask=mask)
    t = scalar_to_two_hot(z)
    logp = torch.log_softmax(logits.float(), dim=1)
    ce = -(t * logp).sum(dim=1)
    manual = ce[mask.bool()].mean()
    assert torch.allclose(loss, manual, atol=1e-6)


def test_ce_loss_minimized_at_true_two_hot():
    # CE is lower when logits match the true two-hot than for a wrong target
    z = torch.tensor([-0.8, 0.3, 0.9])
    good = torch.log(scalar_to_two_hot(z).clamp_min(1e-9))
    bad = torch.log(scalar_to_two_hot(-z).clamp_min(1e-9))
    lg = two_hot_ce_loss(good, z)
    lb = two_hot_ce_loss(bad, z)
    assert float(lg) < float(lb)


def test_decode_matches_scalar_range():
    torch.manual_seed(1)
    logits = torch.randn(100, VALUE_BINS)
    v = decode_expected_value(logits)
    assert v.shape == (100,)
    assert (v >= -1.0).all() and (v <= 1.0).all()
