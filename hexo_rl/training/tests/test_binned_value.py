import torch
from hexo_rl.training.binned_value import (
    VALUE_SUPPORT, scalar_to_two_hot, decode_binned_value, binned_value_loss,
)

def test_support_shape_and_endpoints():
    assert VALUE_SUPPORT.shape == (65,)
    assert torch.allclose(VALUE_SUPPORT[0], torch.tensor(-1.0))
    assert torch.allclose(VALUE_SUPPORT[-1], torch.tensor(1.0))
    assert torch.allclose(VALUE_SUPPORT[32], torch.tensor(0.0), atol=1e-6)

def test_two_hot_rows_sum_to_one_and_place_on_bin_centers():
    z = torch.tensor([-1.0, 0.0, 1.0])
    th = scalar_to_two_hot(z)
    assert th.shape == (3, 65)
    assert torch.allclose(th.sum(dim=1), torch.ones(3))
    assert th[0, 0] == 1.0            # z=-1 → bin 0
    assert th[2, 64] == 1.0           # z=+1 → bin 64
    assert th[1, 32] == 1.0           # z=0 → bin 32 (exact center)

def test_two_hot_splits_between_adjacent_bins():
    z = torch.tensor([0.0 + 1.0 / 64.0])   # +half a bin above center
    th = scalar_to_two_hot(z)               # pos=(z+1)*32=32.5 → split 32/33
    assert torch.allclose(th[0, 32], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(th[0, 33], torch.tensor(0.5), atol=1e-6)

def test_decode_is_left_inverse_of_two_hot():
    z = torch.linspace(-1, 1, 21)
    logits = torch.log(scalar_to_two_hot(z).clamp_min(1e-9))
    dec = decode_binned_value(logits).squeeze(1)
    assert torch.allclose(dec, z, atol=1e-3)

def test_binned_loss_masks_out_invalid_rows():
    logits = torch.zeros(4, 65, requires_grad=True)
    outcome = torch.tensor([1.0, -1.0, 0.0, 1.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    loss = binned_value_loss(logits, outcome, value_mask=mask)
    assert loss.requires_grad and torch.isfinite(loss)
    # all-invalid → exact zero scalar, no NaN
    z0 = binned_value_loss(logits, outcome, value_mask=torch.zeros(4))
    assert float(z0) == 0.0
