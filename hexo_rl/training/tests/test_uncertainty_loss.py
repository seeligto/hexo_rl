"""§S181-AUDIT Wave 4 4B-impl-5 — value uncertainty head loss tests.

Validates the Huber-on-squared-error formulation replaces the pre-Wave-4
Gaussian-NLL formulation that diverged as σ²→0.
"""

import pytest
import torch

from hexo_rl.training.losses import compute_uncertainty_loss


def _z():
    return torch.tensor([1.0, 0.0, -1.0])


def _v():
    return torch.tensor([[0.5], [0.0], [-0.5]])


def test_finite_on_subnormal_sigma2():
    sigma2 = torch.tensor([[6e-8], [1e-5], [1.0]], dtype=torch.float16)
    loss = compute_uncertainty_loss(sigma2, _z(), _v())
    assert torch.isfinite(loss).item()
    assert loss.item() < 10.0, f"Huber loss should stay bounded; got {loss.item()}"


def test_finite_on_zero_sigma2():
    sigma2 = torch.zeros((3, 1), dtype=torch.float16)
    loss = compute_uncertainty_loss(sigma2, _z(), _v())
    assert torch.isfinite(loss).item()
    # target = (z - v)^2 → max 0.25 across the 3 batch rows; loss ≤ 0.25
    assert loss.item() < 1.0


def test_finite_on_large_sigma2():
    sigma2 = torch.full((3, 1), 10.0, dtype=torch.float32)
    loss = compute_uncertainty_loss(sigma2, _z(), _v())
    assert torch.isfinite(loss).item()


def test_zero_loss_when_perfect_prediction():
    """If sigma2 == (z - v)^2 exactly, Huber returns 0."""
    z = _z()
    v = _v()
    target = (z.unsqueeze(1) - v).pow(2)
    sigma2 = target.clone()
    loss = compute_uncertainty_loss(sigma2, z, v)
    assert loss.item() == 0.0


def test_gradient_flows_through_sigma2():
    """Gradient must be nonzero so the head can learn."""
    sigma2 = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    z = torch.tensor([1.0])
    v_det = torch.tensor([[0.0]])
    loss = compute_uncertainty_loss(sigma2, z, v_det)
    loss.backward()
    assert sigma2.grad is not None
    assert sigma2.grad.abs().item() > 1e-4


def test_gradient_flows_at_zero():
    """Old Gaussian NLL had grad=0 at clamp floor; Huber must not."""
    sigma2 = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    z = torch.tensor([1.0])
    v_det = torch.tensor([[0.0]])
    loss = compute_uncertainty_loss(sigma2, z, v_det)
    loss.backward()
    assert sigma2.grad is not None
    assert sigma2.grad.abs().item() > 0.0, "Huber must produce gradient at sigma2=0"


def test_does_not_backprop_into_value():
    """value_detached must not receive gradient (caller passes .detach())."""
    sigma2 = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    v_with_grad = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    z = torch.tensor([1.0])
    loss = compute_uncertainty_loss(sigma2, z, v_with_grad.detach())
    loss.backward()
    assert v_with_grad.grad is None
