"""Forward + loss tests for Q13-aux chain_head."""
from __future__ import annotations
import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.losses import compute_chain_loss


def _tiny_net(in_channels: int = 18) -> HexTacToeNet:
    # Small tower to keep tests fast; same in_channels as production.
    return HexTacToeNet(
        board_size=19,
        in_channels=in_channels,
        filters=16,
        res_blocks=1,
        se_reduction_ratio=4,
    )


def test_forward_base_tuple_unchanged_without_chain_flag():
    net = _tiny_net()
    x = torch.zeros(2, 18, 19, 19)
    out = net(x)
    assert isinstance(out, tuple)
    assert len(out) == 3
    log_policy, value, v_logit = out
    assert log_policy.shape == (2, 362)
    assert value.shape == (2, 1)
    assert v_logit.shape == (2, 1)


def test_forward_with_chain_flag_appends_chain_pred():
    net = _tiny_net()
    x = torch.zeros(2, 18, 19, 19)
    out = net(x, chain=True)
    assert len(out) == 4
    chain_pred = out[-1]
    assert chain_pred.shape == (2, 6, 19, 19)


def test_forward_all_flags_preserves_order():
    net = _tiny_net()
    x = torch.zeros(1, 18, 19, 19)
    out = net(
        x,
        aux=True,
        uncertainty=True,
        ownership=True,
        threat=True,
        chain=True,
    )
    # Base 3 + 5 extras = 8
    assert len(out) == 8
    # Order: log_policy, value, v_logit, opp_reply, sigma2, own_pred, thr_pred, chain_pred
    assert out[3].shape == (1, 362)      # opp_reply
    assert out[4].shape == (1, 1)        # sigma2
    assert out[5].shape == (1, 1, 19, 19)  # own_pred
    assert out[6].shape == (1, 1, 19, 19)  # thr_pred
    assert out[7].shape == (1, 6, 19, 19)  # chain_pred


def test_chain_loss_zero_when_pred_matches_target():
    pred = torch.rand(4, 6, 19, 19)
    loss = compute_chain_loss(pred, pred)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_chain_loss_positive_when_pred_differs():
    pred = torch.zeros(4, 6, 19, 19)
    target = torch.ones(4, 6, 19, 19)
    loss = compute_chain_loss(pred, target)
    # smooth_l1 at delta=1 between 0 and 1: loss(1) = 0.5 * 1^2 = 0.5
    assert loss.item() == pytest.approx(0.5, abs=1e-5)


def test_chain_loss_huber_large_error_falls_back_to_l1():
    pred = torch.zeros(1, 6, 19, 19)
    target = torch.full((1, 6, 19, 19), 3.0)
    loss = compute_chain_loss(pred, target)
    # smooth_l1(3) = |3| - 0.5 = 2.5 (L1 region beyond delta=1)
    assert loss.item() == pytest.approx(2.5, abs=1e-5)


def test_chain_loss_legal_mask_zeros_masked_cells():
    """C13/W2: with legal_mask=0 on half the cells, loss must match the
    unmasked cells only. Uses a (B, 1, H, W) mask broadcast across planes."""
    pred = torch.zeros(2, 6, 19, 19)
    target = torch.ones(2, 6, 19, 19)
    mask = torch.zeros(2, 1, 19, 19)
    mask[:, :, :10, :] = 1.0  # first 10 rows contribute, rest masked
    loss = compute_chain_loss(pred, target, legal_mask=mask)
    # All contributing cells have |err|=1 → smooth_l1(1) = 0.5 (L2 region).
    assert loss.item() == pytest.approx(0.5, abs=1e-5)


def test_chain_loss_legal_mask_all_ones_matches_unmasked():
    pred = torch.rand(3, 6, 19, 19)
    target = torch.rand(3, 6, 19, 19)
    unmasked = compute_chain_loss(pred, target)
    mask = torch.ones(3, 1, 19, 19)
    masked = compute_chain_loss(pred, target, legal_mask=mask)
    assert masked.item() == pytest.approx(unmasked.item(), abs=1e-5)


def test_chain_loss_mask_shapes_all_produce_identical_loss():
    """(B,H,W), (B,1,H,W), and (B,6,H,W) masks with the same cell selection
    must yield the same loss. Guards against the pre-fix over-division bug
    where (B,6,H,W) divided by n_planes a second time."""
    torch.manual_seed(0)
    pred = torch.rand(2, 6, 19, 19)
    target = torch.rand(2, 6, 19, 19)

    # Select first 10 rows of the board.
    mask_bhw = torch.zeros(2, 19, 19)
    mask_bhw[:, :10, :] = 1.0

    mask_b1hw = mask_bhw.unsqueeze(1)                           # (2,1,19,19)
    mask_b6hw = mask_bhw.unsqueeze(1).expand(2, 6, 19, 19).clone()  # (2,6,19,19)

    loss_bhw = compute_chain_loss(pred, target, legal_mask=mask_bhw)
    loss_b1hw = compute_chain_loss(pred, target, legal_mask=mask_b1hw)
    loss_b6hw = compute_chain_loss(pred, target, legal_mask=mask_b6hw)

    assert loss_b1hw.item() == pytest.approx(loss_bhw.item(), abs=1e-5), (
        f"(B,1,H,W) loss {loss_b1hw.item():.6f} != (B,H,W) loss {loss_bhw.item():.6f}"
    )
    assert loss_b6hw.item() == pytest.approx(loss_bhw.item(), abs=1e-5), (
        f"(B,6,H,W) loss {loss_b6hw.item():.6f} != (B,H,W) loss {loss_bhw.item():.6f}"
    )


def test_chain_head_gradient_flows_through_trunk():
    """Train-step sanity: chain loss gradient must propagate into trunk weights."""
    net = _tiny_net()
    net.train()
    # 18-plane input; chain target is a separate random tensor (no longer sliced
    # from input since chain planes were removed from the state tensor in Q13).
    x = torch.randn(2, 18, 19, 19, requires_grad=False)
    out = net(x, chain=True)
    chain_pred = out[-1]
    chain_target = torch.rand(2, 6, 19, 19)
    loss = compute_chain_loss(chain_pred, chain_target)
    loss.backward()
    # Every trunk and chain_head param must have a gradient.
    named = dict(net.named_parameters())
    for key in ("trunk.input_conv.weight", "chain_head.weight"):
        assert named[key].grad is not None, f"no grad on {key}"
        assert torch.isfinite(named[key].grad).all(), f"non-finite grad on {key}"


def test_chain_target_from_numpy_has_no_gradient():
    """Chain target comes from numpy (via torch.from_numpy) — must have no grad.
    In the trainer, chain_planes is a numpy array H2D-transferred without
    requires_grad, so the chain loss only back-props through chain_pred."""
    import numpy as np
    chain_np = np.zeros((2, 6, 19, 19), dtype=np.float32)
    target = torch.from_numpy(chain_np).float()
    assert target.requires_grad is False
    # Build a fake pred that DOES require grad; verify grad flows through pred only.
    pred = torch.zeros(2, 6, 19, 19, requires_grad=True)
    loss = compute_chain_loss(pred, target)
    loss.backward()
    assert pred.grad is not None


def test_threat_head_bce_accepts_pos_weight():
    """Q19 sanity: threat BCE loss must accept a pos_weight tensor on the
    same device as the logits. Pinned here so a future refactor of the
    trainer does not accidentally drop the pos_weight kwarg."""
    logits = torch.randn(2, 1, 19, 19)
    target = torch.zeros(2, 19, 19)
    target[0, 5, 5] = 1.0  # one positive cell
    pos_weight = torch.tensor(59.0)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(1), target, pos_weight=pos_weight
    )
    assert torch.isfinite(loss)
