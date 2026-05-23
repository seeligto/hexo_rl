"""§S181 FU-2 A2 INV pin — value_fc1 in_features matches multi-scale concat.

Locks the post-A2 value-head shape so an accidental revert to GAP+GMP
(``2*filters``) or any other concat reshape surfaces as a clear unit
failure, not a silent ckpt-load break in production.

Verifies (in this order):
  1. ``VALUE_FC1_MULTIPLIER`` is the *single* source of truth — the ctor
     reads it and any concat-shape change requires bumping the constant.
  2. The multi-scale pool's output dim matches the constant.
  3. The pool actually distinguishes a single-quadrant extension from a
     uniformly-distributed colony pre-``fc1`` (the audit's whole point —
     T2 §1.3 max|diff|=0 for the old GMP-only path).
  4. ``forward()`` still returns the canonical 3-tuple
     ``(log_policy, value, v_logit)`` with correct shapes — the
     inference contract MUST not have shifted.
  5. The ckpt-load shape guard fires loudly on a pre-A2 state_dict
     (``2*filters`` GAP+GMP weight) instead of leaking a raw torch error.
"""
from __future__ import annotations

import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.model.network_min_max_head import (
    VALUE_FC1_MULTIPLIER,
    multi_scale_avg_pool,
)


def _net(filters: int = 32, encoding: str = "v6") -> HexTacToeNet:
    return HexTacToeNet(
        board_size=9 if encoding == "v6" else 25,
        in_channels=8 if encoding == "v6" else 11,
        filters=filters,
        res_blocks=2,
        encoding=encoding,
    )


def test_value_fc1_in_features_equals_multiplier_times_filters():
    """Constant matches every supported filters width."""
    for filters in (8, 32, 64, 128):
        net = _net(filters=filters)
        assert net.value_fc1.in_features == VALUE_FC1_MULTIPLIER * filters, (
            f"filters={filters}: value_fc1.in_features="
            f"{net.value_fc1.in_features} expected "
            f"{VALUE_FC1_MULTIPLIER * filters}"
        )


def test_value_fc1_in_features_consistent_v6_and_v8():
    """v6 (has_pass_slot=true) and v8 (has_pass_slot=false) share fc1 sizing."""
    v6 = _net(filters=32, encoding="v6")
    v8 = _net(filters=32, encoding="v8")
    assert v6.value_fc1.in_features == v8.value_fc1.in_features
    assert v6.value_fc1.in_features == VALUE_FC1_MULTIPLIER * 32


def test_multi_scale_avg_pool_concat_shape():
    """Helper output dim matches the constant."""
    out = torch.randn(3, 16, 19, 19)
    v = multi_scale_avg_pool(out)
    assert v.shape == (3, VALUE_FC1_MULTIPLIER * 16)


def test_multi_scale_avg_pool_distinguishes_quadrant_vs_uniform():
    """Quadrant-localised coverage ≠ uniformly-spread coverage post-pool.

    The whole point of A2: pre-fc1, a single-quadrant extension and a
    uniformly distributed colony must produce DIFFERENT pooled vectors,
    not the equality GMP imposed (T2 §1.3 max|diff|=0 for matched peaks).
    """
    quadrant = torch.zeros(1, 4, 19, 19)
    quadrant[0, :, 0:10, 0:10] = 1.0
    uniform = torch.zeros(1, 4, 19, 19)
    uniform[0, :, ::2, ::2] = 4.0
    v_q = multi_scale_avg_pool(quadrant)
    v_u = multi_scale_avg_pool(uniform)
    # GAP component identical by construction (same total mass) — the 2×2
    # quadrant component must separate them.
    assert not torch.allclose(v_q, v_u, atol=1e-3), (
        "A2 multi-scale pool must distinguish quadrant vs uniform coverage; "
        "this is the structural property the GAP+GMP head lacked."
    )


def test_forward_returns_three_tuple_correct_shapes():
    """Base inference contract preserved: (log_policy, value, v_logit)."""
    net = _net(filters=32)
    x = torch.randn(2, 8, 9, 9)
    result = net(x)
    assert len(result) == 3, "forward() inference contract = 3-tuple"
    log_policy, value, v_logit = result
    assert log_policy.shape == (2, 9 * 9 + 1)
    assert value.shape == (2, 1)
    assert v_logit.shape == (2, 1)
    assert torch.allclose(
        log_policy.exp().sum(dim=1), torch.ones(2), atol=1e-4,
    ), "log_policy must remain a log-softmax distribution"


def test_pre_a2_checkpoint_load_raises_clear_error(tmp_path):
    """§S181 FU-2 ckpt guard — pre-A2 (2*filters) state_dict fails loud.

    A pre-A2 anchor (e.g. bootstrap_model_v6.pt) loaded into an A2 model
    must raise a RuntimeError that mentions value_fc1 + A2, not leak a
    raw deep-PyTorch shape mismatch.
    """
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    filters = 32
    pre_a2 = _net(filters=filters, encoding="v6")
    # Replace value_fc1 with pre-A2 (2*filters) weight to simulate an old
    # GAP+GMP checkpoint. Everything else stays current-shape.
    pre_a2.value_fc1 = torch.nn.Linear(2 * filters, 256)
    state = pre_a2.state_dict()
    state["encoding"] = "v6"  # explicit label so detector doesn't probe
    ckpt = tmp_path / "pre_a2_bootstrap.pt"
    torch.save({"model_state": pre_a2.state_dict(), "encoding": "v6"}, ckpt)
    try:
        load_model_with_encoding(ckpt, torch.device("cpu"))
    except RuntimeError as exc:
        msg = str(exc)
        assert "value_fc1" in msg
        assert "A2" in msg
        assert "in_features=64" in msg  # 2 * 32
        assert f"expects {VALUE_FC1_MULTIPLIER * filters}" in msg
    else:
        raise AssertionError(
            "load_model_with_encoding must raise RuntimeError on pre-A2 ckpt"
        )
