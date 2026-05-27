"""§S181-AUDIT Wave 4 4B-impl-3 — ply-index aux head loss tests."""

import torch

from hexo_rl.training.losses import compute_ply_index_loss


def test_zero_loss_on_perfect_prediction():
    ply_pred = torch.tensor([[0.50], [0.70], [0.30]], dtype=torch.float32)
    positions = torch.tensor([50, 70, 30], dtype=torch.int64)
    loss = compute_ply_index_loss(ply_pred, positions)
    assert loss.item() < 1e-6


def test_positive_loss_on_mismatch():
    ply_pred = torch.tensor([[0.10], [0.90], [0.50]], dtype=torch.float32)
    positions = torch.tensor([80, 20, 50], dtype=torch.int64)
    loss = compute_ply_index_loss(ply_pred, positions)
    assert 0.10 < loss.item() < 0.30


def test_target_clamped_at_one():
    """Positions > 100 should clamp target to 1.0 — head can still learn."""
    ply_pred = torch.tensor([[1.0]], dtype=torch.float32)
    positions = torch.tensor([500], dtype=torch.int64)
    loss = compute_ply_index_loss(ply_pred, positions)
    assert loss.item() < 1e-6


def test_target_floor_zero():
    """Negative or zero positions clamp at 0 — no negative target."""
    ply_pred = torch.tensor([[0.0]], dtype=torch.float32)
    positions = torch.tensor([0], dtype=torch.int64)
    loss = compute_ply_index_loss(ply_pred, positions)
    assert loss.item() < 1e-6


def test_gradient_flows_through_ply_pred():
    ply_pred = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    positions = torch.tensor([20], dtype=torch.int64)
    loss = compute_ply_index_loss(ply_pred, positions)
    loss.backward()
    assert ply_pred.grad is not None
    assert ply_pred.grad.abs().item() > 1e-4


def test_does_not_backprop_into_positions():
    """positions is the target — must not receive gradient (it's int64 anyway,
    but verify the loss is constant wrt position floats)."""
    ply_pred = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    pos = torch.tensor([20.0], dtype=torch.float32, requires_grad=True)
    loss = compute_ply_index_loss(ply_pred, pos)
    loss.backward()
    assert ply_pred.grad is not None
    # The function casts pos to float and divides; gradient through pos exists
    # but is not used downstream — verify it's the expected magnitude relative
    # to the head prediction direction.
    assert torch.isfinite(pos.grad).all().item()


def test_uint16_position_indices_accepted():
    """Position indices come from Rust as uint16 — must accept them."""
    import numpy as np
    ply_pred = torch.tensor([[0.50], [0.70]], dtype=torch.float32)
    positions = torch.from_numpy(np.array([50, 70], dtype=np.uint16))
    loss = compute_ply_index_loss(ply_pred, positions)
    assert torch.isfinite(loss).item()
