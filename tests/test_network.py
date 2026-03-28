"""
Phase 1 smoke tests for HexTacToeNet.

Run with: .venv/bin/pytest tests/test_network.py -v
"""
import pytest
import torch
from python.model.network import HexTacToeNet


def make_net(board_size: int = 9, filters: int = 32, res_blocks: int = 2) -> HexTacToeNet:
    """Small net for fast smoke tests."""
    return HexTacToeNet(board_size=board_size, filters=filters, res_blocks=res_blocks)


def test_output_shapes_small_board():
    net = make_net(board_size=9)
    x = torch.zeros(2, 18, 9, 9)
    log_policy, value = net(x)
    assert log_policy.shape == (2, 9 * 9 + 1), f"policy shape {log_policy.shape}"
    assert value.shape == (2, 1), f"value shape {value.shape}"


def test_output_shapes_full_board():
    net = HexTacToeNet(board_size=19, filters=32, res_blocks=2)
    x = torch.zeros(4, 18, 19, 19)
    log_policy, value = net(x)
    assert log_policy.shape == (4, 19 * 19 + 1)
    assert value.shape == (4, 1)


def test_policy_is_log_softmax():
    """log_softmax outputs must sum to 1 in probability space."""
    net = make_net()
    x = torch.randn(3, 18, 9, 9)
    log_policy, _ = net(x)
    probs_sum = log_policy.exp().sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones(3), atol=1e-4), f"probs sum: {probs_sum}"


def test_value_in_range():
    net = make_net()
    x = torch.randn(8, 18, 9, 9)
    _, value = net(x)
    assert (value >= -1.0).all() and (value <= 1.0).all(), f"value out of range: {value}"


def test_forward_with_float16_input():
    """Network should handle float16 input (autocast context)."""
    net = make_net().half()
    x = torch.zeros(2, 18, 9, 9, dtype=torch.float16)
    log_policy, value = net(x)
    assert log_policy.dtype == torch.float16
    assert value.dtype == torch.float16


def test_default_architecture_params():
    """Default params match the full Phase 2 architecture spec."""
    net = HexTacToeNet()
    assert net.board_size == 19
    # Count residual blocks
    res_count = sum(1 for _ in net.tower.children())
    assert res_count == 10
