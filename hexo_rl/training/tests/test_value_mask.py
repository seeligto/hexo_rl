"""DRAW-MASK (Phase 6) — value_mask gating in compute_value_loss.

Ply-capped games are horizon truncation with no real outcome: the value head
must NOT be supervised on those rows. ``value_mask`` (1 = supervise, 0 = mask)
mirrors the ``full_search_mask`` plumbing in compute_policy_loss.

compute_value_loss maps outcome {-1,+1} -> {0,1} internally and squeezes a
(B,1) logit, so these fixtures use the real signature.
"""

import torch
import torch.nn.functional as F

from hexo_rl.training.losses import compute_value_loss


def _bce(logit, outcome):
    """Reference: the exact unmasked computation compute_value_loss does."""
    target = (outcome + 1.0) / 2.0
    return F.binary_cross_entropy_with_logits(
        logit.squeeze(1), target, reduction="none"
    )


def test_value_mask_all_true_equals_unmasked():
    """All-ones mask == current unmasked value (regression pin)."""
    torch.manual_seed(0)
    logit = torch.randn(8, 1)
    outcome = torch.empty(8).uniform_(-1.0, 1.0)
    mask = torch.ones(8, dtype=torch.uint8)
    masked = compute_value_loss(logit, outcome, value_mask=mask)
    unmasked = compute_value_loss(logit, outcome)
    assert torch.allclose(masked, unmasked, atol=1e-6)


def test_value_mask_partial_is_bce_over_true_rows():
    torch.manual_seed(1)
    logit = torch.randn(8, 1)
    outcome = torch.empty(8).uniform_(-1.0, 1.0)
    mask = torch.tensor([1, 0, 1, 0, 1, 1, 0, 1], dtype=torch.uint8)
    masked = compute_value_loss(logit, outcome, value_mask=mask)
    expected = _bce(logit, outcome)[mask.bool()].mean()
    assert torch.allclose(masked, expected, atol=1e-6)
    # And it must differ from the unmasked mean (the masked rows mattered).
    assert not torch.allclose(masked, compute_value_loss(logit, outcome), atol=1e-4)


def test_value_mask_all_false_is_zero_scalar():
    torch.manual_seed(2)
    logit = torch.randn(8, 1)
    outcome = torch.empty(8).uniform_(-1.0, 1.0)
    mask = torch.zeros(8, dtype=torch.uint8)
    masked = compute_value_loss(logit, outcome, value_mask=mask)
    assert masked.numel() == 1
    assert float(masked) == 0.0


def test_value_mask_default_none_unchanged():
    """Default (no mask) == prior unmasked BCE — backward-compat pin."""
    torch.manual_seed(3)
    logit = torch.randn(8, 1)
    outcome = torch.empty(8).uniform_(-1.0, 1.0)
    target = (outcome + 1.0) / 2.0
    expected = F.binary_cross_entropy_with_logits(logit.squeeze(1), target)
    assert torch.allclose(compute_value_loss(logit, outcome), expected, atol=1e-6)


def test_value_mask_uint8_numpy_array_wiring_contract():
    """DRAW-MASK plumbing contract: a uint8 numpy value_target_valid column —
    exactly what ``ReplayBuffer.sample_batch_with_pos`` returns and the trainer
    converts via ``torch.from_numpy(...).bool()`` — masks the value loss to the
    valid rows only. Pins the dtype/conversion the Rust→Python wiring relies on.
    """
    import numpy as np

    torch.manual_seed(4)
    logit = torch.randn(8, 1)
    outcome = torch.empty(8).uniform_(-1.0, 1.0)
    # Simulate the buffer column: rows 1,3,6 are ply-capped (0 = mask value).
    vv_np = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    # Reproduce the exact trainer conversion (trainer.py _train_on_batch).
    value_mask_t = torch.from_numpy(np.asarray(vv_np, dtype=np.uint8)).bool()
    masked = compute_value_loss(logit, outcome, value_mask=value_mask_t)
    expected = _bce(logit, outcome)[value_mask_t].mean()
    assert torch.allclose(masked, expected, atol=1e-6)
    # The masked rows must matter (capped rows excluded from the mean).
    assert not torch.allclose(masked, compute_value_loss(logit, outcome), atol=1e-4)
