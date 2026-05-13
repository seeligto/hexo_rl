"""Integration tests for HexTacToeNet under v8 encoding (Path β).

Covers all 5 Phase B variant arms (B0-B4):
- B0: 128×12, no GPool, head_use_gpool=False — control.
- B1: 128×12, GPool {6,10}, full v8 spec.
- B2: 96×12,  GPool {6,10} — capacity probe.
- B3: 128×10, GPool {6,10} — depth probe.
- B4: 160×12, GPool {6,10} — width probe.

Per arm we check forward shape, log-probability normalization, and that
the mask plumbing routes through both trunk gpool sites and the policy head.
"""
from __future__ import annotations

from typing import Optional

import pytest
import torch

from hexo_rl.model.network import HexTacToeNet


_V8_BOARD = 25
_V8_PLANES = 11
_V8_OFF_WINDOW_PLANE = 8
_V8_N_ACTIONS = 625


def _make_v8_input(batch: int) -> torch.Tensor:
    """Synthesize a (batch, 11, 25, 25) v8 input with a hex-shaped off_window."""
    x = torch.zeros(batch, _V8_PLANES, _V8_BOARD, _V8_BOARD)
    # Off_window plane: 1.0 outside dilated hex of radius 8 around centroid (12, 12).
    half = (_V8_BOARD - 1) // 2  # 12
    radius = 8
    for q in range(_V8_BOARD):
        for r in range(_V8_BOARD):
            lq = q - half
            lr = r - half
            ls = -(lq + lr)
            dist = max(abs(lq), abs(lr), abs(ls))
            if dist > radius:
                x[:, _V8_OFF_WINDOW_PLANE, q, r] = 1.0
    # A few stones on plane 0 / 4 inside the hex.
    x[:, 0, 12, 12] = 1.0
    x[:, 4, 13, 12] = 1.0
    # Plane 9 (moves_remaining_bcast) and plane 10 (ply_parity_bcast).
    x[:, 9, :, :] = 0.5
    x[:, 10, :, :] = 1.0
    return x


def _make_v8_model(
    filters: int,
    res_blocks: int,
    gpool_indices: Optional[list[int]],
    head_use_gpool: bool,
) -> HexTacToeNet:
    return HexTacToeNet(
        board_size=_V8_BOARD,
        in_channels=_V8_PLANES,
        filters=filters,
        res_blocks=res_blocks,
        encoding="v8",
        gpool_indices=gpool_indices,
        head_use_gpool=head_use_gpool,
    )


def test_v6_default_unchanged() -> None:
    """v6 default construction must remain byte-shape-compatible."""
    m = HexTacToeNet()
    assert m.encoding == "v6"
    assert m.has_pass is True
    assert m.n_actions == 19 * 19 + 1  # 362
    x = torch.randn(1, 8, 19, 19)
    log_p, value, v_logit = m(x)
    assert log_p.shape == (1, 362)
    assert value.shape == (1, 1)
    assert v_logit.shape == (1, 1)


def test_v6_blocks_input_channels_under_v8() -> None:
    with pytest.raises(ValueError, match="v6-only knob"):
        HexTacToeNet(
            encoding="v8",
            board_size=25,
            in_channels=11,
            input_channels=[0, 4],
        )


def test_v6_invalid_encoding_rejected() -> None:
    with pytest.raises(ValueError, match=r"not in registry"):
        HexTacToeNet(encoding="v9")


@pytest.mark.parametrize(
    "arm,filters,res_blocks,gpool_indices,head_use_gpool",
    [
        ("B0", 128, 12, None, False),       # control — encoding shape only
        ("B1", 128, 12, [6, 10], True),     # primary candidate
        ("B2", 96,  12, [6, 10], True),     # capacity probe
        # B3 depth probe: 12-block plan said {6, 10}, but depth-10 trunk
        # only has indices [0, 10). Re-derived to KataGo b10c128's {5, 8}
        # (same 50% / 80% depth fractions). Variant summary captures the
        # adjustment.
        ("B3", 128, 10, [5, 8],  True),     # depth probe
        ("B4", 160, 12, [6, 10], True),     # width probe
    ],
)
def test_v8_variant_forward(
    arm: str,
    filters: int,
    res_blocks: int,
    gpool_indices: Optional[list[int]],
    head_use_gpool: bool,
) -> None:
    model = _make_v8_model(filters, res_blocks, gpool_indices, head_use_gpool)
    assert model.encoding == "v8"
    assert model.has_pass is False
    assert model.n_actions == _V8_N_ACTIONS

    x = _make_v8_input(batch=2)
    log_p, value, v_logit = model(x)
    assert log_p.shape == (2, _V8_N_ACTIONS), f"{arm}: {log_p.shape}"
    assert value.shape == (2, 1)
    assert v_logit.shape == (2, 1)
    # log_softmax → probabilities sum to ≈1 per row.
    probs = log_p.exp().sum(dim=1)
    assert torch.allclose(probs, torch.ones(2), atol=1e-3), \
        f"{arm} policy not normalized: {probs}"


def test_v8_b1_aux_head_returns_log_softmax() -> None:
    model = _make_v8_model(128, 12, [6, 10], True)
    x = _make_v8_input(1)
    out = model(x, aux=True)
    assert len(out) == 4
    log_p, value, v_logit, opp_reply = out
    assert opp_reply.shape == (1, _V8_N_ACTIONS)
    probs = opp_reply.exp().sum(dim=1)
    assert torch.allclose(probs, torch.ones(1), atol=1e-3)


def test_v8_b0_offboard_policy_probs_near_zero() -> None:
    """B0 with no gpool still applies the off-board logit bias in the head."""
    model = _make_v8_model(128, 12, None, False)
    x = _make_v8_input(1)
    log_p, _v, _vl = model(x)
    probs = log_p.exp().squeeze(0)
    # Build the same off_window mask used in _make_v8_input and test that
    # off-board cells received negligible probability.
    half = (_V8_BOARD - 1) // 2
    offboard = []
    for q in range(_V8_BOARD):
        for r in range(_V8_BOARD):
            lq = q - half
            lr = r - half
            ls = -(lq + lr)
            if max(abs(lq), abs(lr), abs(ls)) > 8:
                offboard.append(q * _V8_BOARD + r)
    offboard_total = probs[offboard].sum().item()
    assert offboard_total < 1e-3, f"off-board probability mass {offboard_total} too large"


def test_v8_trunk_gpool_indices_validation() -> None:
    """Out-of-range gpool indices must surface at construction time."""
    with pytest.raises(ValueError, match="out of"):
        _make_v8_model(128, 10, [12], True)  # idx 12 ≥ 10


def test_v8_b2_param_count_smaller_than_b1() -> None:
    """B2 (96 ch) must have fewer trunk params than B1 (128 ch)."""
    b1 = _make_v8_model(128, 12, [6, 10], True)
    b2 = _make_v8_model(96, 12, [6, 10], True)
    n1 = sum(p.numel() for p in b1.parameters())
    n2 = sum(p.numel() for p in b2.parameters())
    assert n2 < n1, f"capacity probe param count not smaller: B1={n1}, B2={n2}"
    # Roughly capacity ratio ≈ (96/128)² ≈ 0.56× for trunk-dominated params.
    assert n2 / n1 < 0.85, f"B2/B1 ratio {n2 / n1} unexpectedly high"


def test_v8_b3_depth_smaller_than_b1() -> None:
    """B3 (depth 10) must have fewer params than B1 (depth 12).

    B3 gpool indices re-derived to {5, 8} (KataGo b10c128's depth-fraction
    pattern) since {6, 10} would index past the 10-block trunk.
    """
    b1 = _make_v8_model(128, 12, [6, 10], True)
    b3 = _make_v8_model(128, 10, [5, 8], True)
    n1 = sum(p.numel() for p in b1.parameters())
    n3 = sum(p.numel() for p in b3.parameters())
    assert n3 < n1, f"depth probe param count not smaller: B1={n1}, B3={n3}"


def test_v8_backward_pass() -> None:
    """Backward pass through B1 (gpool trunk + gpool head) must produce grads
    on the trunk gpool blocks AND both heads (with aux=True so opp_reply runs).
    """
    model = _make_v8_model(128, 12, [6, 10], True)
    x = _make_v8_input(2)
    log_p, value, _v_logit, opp_reply = model(x, aux=True)
    loss = (-log_p.mean()
            - opp_reply.mean()
            + value.pow(2).mean())
    loss.backward()
    expected_modules = (
        "trunk.tower.6.conv1.conv1g",
        "trunk.tower.10.conv1.conv1g",
        "trunk.tower.6.conv1.linear_g",
        "trunk.tower.10.conv1.linear_g",
        "policy_head.conv1g",
        "policy_head.linear_g",
        "opp_reply_head.conv1g",
        "opp_reply_head.linear_g",
    )
    seen: set[str] = set()
    for name, p in model.named_parameters():
        for prefix in expected_modules:
            if name.startswith(prefix):
                assert p.grad is not None, f"no gradient for {name}"
                seen.add(prefix)
    missing = set(expected_modules) - seen
    assert not missing, f"missing expected gpool params: {missing}"
