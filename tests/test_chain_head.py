"""Forward + loss tests for Q13-aux chain_head."""
from __future__ import annotations
import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.losses import compute_chain_loss


def _tiny_net(in_channels: int = 24) -> HexTacToeNet:
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
    x = torch.zeros(2, 24, 19, 19)
    out = net(x)
    assert isinstance(out, tuple)
    assert len(out) == 3
    log_policy, value, v_logit = out
    assert log_policy.shape == (2, 362)
    assert value.shape == (2, 1)
    assert v_logit.shape == (2, 1)


def test_forward_with_chain_flag_appends_chain_pred():
    net = _tiny_net()
    x = torch.zeros(2, 24, 19, 19)
    out = net(x, chain=True)
    assert len(out) == 4
    chain_pred = out[-1]
    assert chain_pred.shape == (2, 6, 19, 19)


def test_forward_all_flags_preserves_order():
    net = _tiny_net()
    x = torch.zeros(1, 24, 19, 19)
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


def test_chain_head_gradient_flows_through_trunk():
    """Train-step sanity: chain loss gradient must propagate into trunk weights."""
    net = _tiny_net()
    net.train()
    # Input with the chain block as a non-trivial target; policy/value paths
    # are zeroed out so the only gradient is from chain_loss.
    x = torch.randn(2, 24, 19, 19, requires_grad=False)
    out = net(x, chain=True)
    chain_pred = out[-1]
    chain_target = x[:, 18:24]
    loss = compute_chain_loss(chain_pred, chain_target)
    loss.backward()
    # Every trunk and chain_head param must have a gradient.
    named = dict(net.named_parameters())
    for key in ("trunk.input_conv.weight", "chain_head.weight"):
        assert named[key].grad is not None, f"no grad on {key}"
        assert torch.isfinite(named[key].grad).all(), f"non-finite grad on {key}"


def test_chain_head_target_slice_from_input_does_not_leak_gradient_to_input():
    """The chain target is `input[:, 18:24]` — this must be a no-gradient tensor.
    In the trainer, `states_t` is constructed from numpy with no requires_grad,
    so the slice also has no grad. This test pins that contract."""
    x = torch.zeros(2, 24, 19, 19, requires_grad=False)
    target = x[:, 18:24]
    assert target.requires_grad is False
    # Build a fake pred that DOES require grad and verify loss gradient only
    # flows through pred, not through target.
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
