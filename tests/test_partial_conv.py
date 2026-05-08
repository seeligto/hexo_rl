"""Tests for ``hexo_rl.model.partial_conv.PartialConv2d`` and the §169 A4
``canvas_realness`` wiring through ``HexTacToeNet``.

Subspike basis: ``audit/encoding_spikes/s4_a4_se_partial_conv.py`` (local
artifact, audit/ gitignored). The script measured forward / backward
correctness, off-canvas zero, and full-model latency Δ +0.51% at b=1
on the laptop 4060 Max-Q. These tests pin the on-CI invariants.
"""
from __future__ import annotations

import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.model.partial_conv import PartialConv2d


def _canvas_realness_mask(side: int = 25, half: int = 12, radius: int = 8) -> torch.Tensor:
    mask = torch.zeros(1, 1, side, side)
    for wq in range(side):
        for wr in range(side):
            lq = wq - half
            lr = wr - half
            ls = -(lq + lr)
            if max(abs(lq), abs(lr), abs(ls)) <= radius:
                mask[0, 0, wq, wr] = 1.0
    return mask


def test_partial_conv_forward_shape_and_finite() -> None:
    pc = PartialConv2d(11, 128, 3, 1, bias=False)
    x = torch.randn(4, 11, 25, 25)
    mask = _canvas_realness_mask().expand(4, 1, 25, 25).contiguous()
    out = pc(x, mask)
    assert out.shape == (4, 128, 25, 25)
    assert torch.isfinite(out).all()


def test_partial_conv_off_canvas_output_zero() -> None:
    """Off-canvas cells (mask=0) must produce zero output."""
    pc = PartialConv2d(11, 64, 3, 1, bias=False)
    # Single-cell canvas at the centre — everything else off-canvas.
    mask = torch.zeros(1, 1, 25, 25)
    mask[0, 0, 12, 12] = 1.0
    x = torch.randn(1, 11, 25, 25)
    out = pc(x, mask)
    # Far off-canvas cells (5+ away from the single canvas cell) must be 0.
    far = out[0, :, 0:6, 0:6]
    assert far.abs().max() < 1e-6


def test_partial_conv_grad_flow_to_all_params() -> None:
    """Backward must reach every PartialConv2d parameter with finite grads."""
    pc = PartialConv2d(11, 64, 3, 1, bias=False)
    x = torch.randn(2, 11, 25, 25, requires_grad=False)
    mask = _canvas_realness_mask().expand(2, 1, 25, 25).contiguous()
    target = torch.randn(2, 64, 25, 25)
    loss = (pc(x, mask) - target).pow(2).mean()
    loss.backward()
    for name, p in pc.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad has non-finite"


def test_partial_conv_renormalisation_interior_cell_matches_vanilla() -> None:
    """Interior cells (away from boundary) where the full 3×3 receptive
    field is inside the mask must produce the same value as a vanilla
    Conv2d with identical weights.

    This pins the renormalisation: at an interior cell ``count == k²``
    so ``scale == k² / count == 1`` — partial conv reduces to vanilla
    conv on an interior receptive field.
    """
    torch.manual_seed(42)
    pc = PartialConv2d(3, 8, 3, 1, bias=False)
    vanilla = torch.nn.Conv2d(3, 8, 3, padding=1, bias=False)
    vanilla.weight.data.copy_(pc.conv.weight.data)
    # Mask = all 1s (no boundary). Interior cells must match exactly.
    x = torch.randn(1, 3, 25, 25)
    mask = torch.ones(1, 1, 25, 25)
    out_pc = pc(x, mask)
    out_v = vanilla(x)
    # Interior cells (away from the mask boundary) must match within fp32 noise.
    interior_pc = out_pc[:, :, 2:23, 2:23]
    interior_v = out_v[:, :, 2:23, 2:23]
    torch.testing.assert_close(interior_pc, interior_v, atol=1e-5, rtol=1e-5)


def test_hextactoenet_canvas_realness_constructor_v8_only() -> None:
    """canvas_realness=True must reject v6 / v6w25 encodings."""
    import pytest
    with pytest.raises(ValueError, match="canvas_realness"):
        HexTacToeNet(board_size=19, in_channels=8, encoding="v6", canvas_realness=True)
    with pytest.raises(ValueError, match="canvas_realness"):
        HexTacToeNet(board_size=25, in_channels=8, encoding="v6", canvas_realness=True)


def test_hextactoenet_canvas_realness_v8_swaps_input_conv() -> None:
    """Under canvas_realness=True the trunk input_conv is a PartialConv2d."""
    m = HexTacToeNet(
        board_size=25, in_channels=11, filters=128, res_blocks=12,
        encoding="v8", gpool_indices=[6, 10], canvas_realness=True,
    )
    assert m.canvas_realness is True
    assert m.trunk.canvas_realness is True
    assert isinstance(m.trunk.input_conv, PartialConv2d)


def test_hextactoenet_canvas_realness_default_false_unchanged() -> None:
    """canvas_realness defaults to False — v8/B0-B4 wire-format byte-exact."""
    m = HexTacToeNet(
        board_size=25, in_channels=11, filters=128, res_blocks=12,
        encoding="v8", gpool_indices=[6, 10],
    )
    assert m.canvas_realness is False
    assert m.trunk.canvas_realness is False
    assert isinstance(m.trunk.input_conv, torch.nn.Conv2d)


def test_hextactoenet_canvas_realness_forward_and_backward() -> None:
    """Full forward/backward through HexTacToeNet with canvas_realness=True."""
    m = HexTacToeNet(
        board_size=25, in_channels=11, filters=64, res_blocks=4,
        encoding="v8", gpool_indices=[2], canvas_realness=True,
    )
    x = torch.randn(2, 11, 25, 25)
    # Set plane 8 to canvas_realness mask (1 inside).
    mask = _canvas_realness_mask().squeeze(0).squeeze(0)
    x[:, 8] = mask
    # Forward all heads exercised by training (aux=True so opp_reply_head reached).
    log_p, value, v_logit, opp_reply = m(x, aux=True)
    assert log_p.shape == (2, 625)
    assert value.shape == (2, 1)
    assert opp_reply.shape == (2, 625)
    assert torch.isfinite(log_p).all()
    assert torch.isfinite(value).all()
    loss = log_p.sum() + value.sum() + opp_reply.sum()
    loss.backward()
    # Skip heads that aren't part of the (log_p, value, opp_reply) path
    # (uncertainty, ownership, threat, chain — all gated off here).
    skip_prefixes = ("value_var.", "ownership_head.", "threat_head.", "chain_head.")
    for name, p in m.named_parameters():
        if any(name.startswith(s) for s in skip_prefixes):
            continue
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad non-finite"


def test_hextactoenet_canvas_realness_state_dict_key_shift() -> None:
    """The canvas_realness flag wraps input_conv, so the state-dict key
    shifts from ``trunk.input_conv.weight`` to ``trunk.input_conv.conv.weight``.
    The checkpoint loader uses this for detection (§169 A4).
    """
    m = HexTacToeNet(
        board_size=25, in_channels=11, filters=64, res_blocks=4,
        encoding="v8", gpool_indices=[2], canvas_realness=True,
    )
    sd = m.state_dict()
    assert "trunk.input_conv.conv.weight" in sd
    assert "trunk.input_conv.weight" not in sd

    m_off = HexTacToeNet(
        board_size=25, in_channels=11, filters=64, res_blocks=4,
        encoding="v8", gpool_indices=[2], canvas_realness=False,
    )
    sd_off = m_off.state_dict()
    assert "trunk.input_conv.weight" in sd_off
    assert "trunk.input_conv.conv.weight" not in sd_off


def test_checkpoint_loader_detects_canvas_realness() -> None:
    """The eval checkpoint loader must rebuild a canvas_realness model from
    a saved state-dict by the wrapped-key signature alone.
    """
    import tempfile
    from pathlib import Path
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    m = HexTacToeNet(
        board_size=25, in_channels=11, filters=64, res_blocks=4,
        encoding="v8", gpool_indices=[2], canvas_realness=True,
    )
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "A4_canvas_realness.pt"
        torch.save(m.state_dict(), ckpt)
        loaded, spec, label = load_model_with_encoding(ckpt, torch.device("cpu"))
    assert label == "v8"
    assert loaded.canvas_realness is True
    assert isinstance(loaded.trunk.input_conv, PartialConv2d)
