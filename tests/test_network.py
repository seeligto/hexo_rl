"""
Phase 1 smoke tests for HexTacToeNet.

Run with: .venv/bin/pytest tests/test_network.py -v
"""
import pytest
import torch
from hexo_rl.model.network import HexTacToeNet, compile_model


def make_net(board_size: int = 9, filters: int = 32, res_blocks: int = 2) -> HexTacToeNet:
    """Small net for fast smoke tests."""
    return HexTacToeNet(board_size=board_size, filters=filters, res_blocks=res_blocks)


def test_output_shapes_small_board():
    net = make_net(board_size=9)
    x = torch.zeros(2, 8, 9, 9)
    log_policy, value, v_logit = net(x)
    assert log_policy.shape == (2, 9 * 9 + 1), f"policy shape {log_policy.shape}"
    assert value.shape == (2, 1), f"value shape {value.shape}"
    assert v_logit.shape == (2, 1), f"v_logit shape {v_logit.shape}"


def test_output_shapes_full_board():
    net = HexTacToeNet(board_size=19, filters=32, res_blocks=2)
    x = torch.zeros(4, 8, 19, 19)
    log_policy, value, v_logit = net(x)
    assert log_policy.shape == (4, 19 * 19 + 1)
    assert value.shape == (4, 1)
    assert v_logit.shape == (4, 1)


def test_policy_is_log_softmax():
    """log_softmax outputs must sum to 1 in probability space."""
    net = make_net()
    x = torch.randn(3, 8, 9, 9)
    log_policy, _, _ = net(x)
    probs_sum = log_policy.exp().sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones(3), atol=1e-4), f"probs sum: {probs_sum}"


def test_value_in_range():
    net = make_net()
    x = torch.randn(8, 8, 9, 9)
    _, value, _ = net(x)
    assert (value >= -1.0).all() and (value <= 1.0).all(), f"value out of range: {value}"


def test_value_is_tanh_of_logit():
    """value should equal tanh(value_logit)."""
    net = make_net()
    x = torch.randn(4, 8, 9, 9)
    _, value, v_logit = net(x)
    assert torch.allclose(value, torch.tanh(v_logit), atol=1e-6)


def test_forward_with_float16_input():
    """Network should handle float16 input (autocast context)."""
    net = make_net().half()
    x = torch.zeros(2, 8, 9, 9, dtype=torch.float16)
    log_policy, value, v_logit = net(x)
    assert log_policy.dtype == torch.float16
    assert value.dtype == torch.float16


def test_default_architecture_params():
    """Default params match the Phase 4 architecture spec."""
    net = HexTacToeNet()
    assert net.board_size == 19
    # Count residual blocks
    res_count = sum(1 for _ in net.tower.children())
    assert res_count == 12


def test_se_block_present():
    """Each residual block should contain an SE block."""
    net = make_net(filters=32, res_blocks=2)
    for block in net.tower:
        assert hasattr(block, 'se'), "ResidualBlock missing SE block"


def test_value_head_uses_global_pooling():
    """Value head FC1 should accept 2*filters (avg+max pool concat)."""
    net = make_net(filters=32)
    assert net.value_fc1.in_features == 64  # 2 * 32


def test_aux_head_output_shapes():
    """Auxiliary opponent-reply head returns correct shapes."""
    net = make_net(board_size=9)
    x = torch.randn(2, 8, 9, 9)
    log_policy, value, v_logit, opp_reply = net(x, aux=True)
    assert opp_reply.shape == (2, 9 * 9 + 1)
    # opp_reply should be log-softmax
    probs_sum = opp_reply.exp().sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones(2), atol=1e-4)


def test_aux_false_returns_three_outputs():
    """Without aux=True, forward returns (log_policy, value, value_logit)."""
    net = make_net(board_size=9)
    x = torch.randn(2, 8, 9, 9)
    result = net(x)
    assert len(result) == 3


def test_compile_model_produces_correct_output():
    """compile_model returns a model that produces the same output shapes."""
    net = make_net(board_size=9)
    compiled = compile_model(net)
    x = torch.zeros(2, 8, 9, 9)
    log_policy, value, v_logit = compiled(x)
    assert log_policy.shape == (2, 9 * 9 + 1)
    assert value.shape == (2, 1)


def test_compile_model_policy_still_sums_to_one():
    """Compiled model policy head still produces valid log-probabilities."""
    net = make_net()
    compiled = compile_model(net)
    x = torch.randn(3, 8, 9, 9)
    log_policy, _, _ = compiled(x)
    probs_sum = log_policy.exp().sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones(3), atol=1e-4), (
        f"probs sum after compile: {probs_sum}"
    )


def test_compile_model_fallback_on_bad_mode():
    """compile_model with an invalid mode falls back to the uncompiled model."""
    net = make_net()
    result = compile_model(net, mode="this_mode_does_not_exist")
    # Should return something callable (either compiled or original model).
    x = torch.zeros(2, 8, 9, 9)
    log_policy, value, v_logit = result(x)
    assert log_policy.shape == (2, 9 * 9 + 1)


# ── Value uncertainty head ────────────────────────────────────────────────────

def test_uncertainty_flag_returns_four_outputs():
    """uncertainty=True adds sigma2 as a 4th return value."""
    net = make_net(board_size=9)
    x = torch.randn(2, 8, 9, 9)
    result = net(x, uncertainty=True)
    assert len(result) == 4, f"expected 4-tuple, got {len(result)}"
    log_policy, value, v_logit, sigma2 = result
    assert log_policy.shape == (2, 9 * 9 + 1)
    assert sigma2.shape == (2, 1)


def test_uncertainty_sigma2_positive():
    """sigma2 must be strictly positive (Softplus ensures this)."""
    net = make_net(board_size=9)
    x = torch.randn(4, 8, 9, 9)
    _, _, _, sigma2 = net(x, uncertainty=True)
    assert (sigma2 > 0).all(), "sigma2 must be strictly positive"


def test_uncertainty_aux_returns_five_outputs():
    """aux=True + uncertainty=True returns a 5-tuple."""
    net = make_net(board_size=9)
    x = torch.randn(2, 8, 9, 9)
    result = net(x, aux=True, uncertainty=True)
    assert len(result) == 5, f"expected 5-tuple, got {len(result)}"
    log_policy, value, v_logit, opp_reply, sigma2 = result
    assert opp_reply.shape == (2, 9 * 9 + 1)
    assert sigma2.shape == (2, 1)


def test_uncertainty_false_still_returns_three_outputs():
    """uncertainty=False (default) must preserve the 3-tuple contract."""
    net = make_net(board_size=9)
    x = torch.randn(2, 8, 9, 9)
    result = net(x)
    assert len(result) == 3


def test_uncertainty_head_does_not_appear_in_normal_forward():
    """Calling forward() without uncertainty=True must not execute value_var."""
    net = make_net(board_size=9)
    x = torch.randn(1, 8, 9, 9)
    # Patch value_var to raise if called
    original_var = net.value_var

    class _Sentinel(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("value_var called unexpectedly in non-uncertainty forward")

    net.value_var = _Sentinel()
    try:
        net(x)            # aux=False, uncertainty=False — must not raise
        net(x, aux=True)  # aux=True, uncertainty=False — must not raise
    finally:
        net.value_var = original_var
