"""§169 A3 — GlobalTokenEncoder unit tests.

Covers:
  (a) shape — (B, 3, 32, 32) input → (B, out_dim) output.
  (b) all-zero canvas (mask_sum_hw=0) is NaN-safe via the clamp.
  (c) gradient sanity — every param touched by backprop.
  (d) state-dict round-trip.
  (e) padding-leak sensitivity — pure padding (mask=0 everywhere) must
      produce a different output than pure mask=1 with identical stones,
      so the canvas-realness mask is actually consumed.
"""
from __future__ import annotations

import torch

from hexo_rl.model.global_token import GlobalTokenEncoder


def test_output_shape():
    enc = GlobalTokenEncoder(in_channels=3, conv_channels=32, out_dim=128).eval()
    x = torch.randn(4, 3, 32, 32)
    out = enc(x)
    assert out.shape == (4, 128)


def test_empty_canvas_does_not_nan():
    """All-zero input (no stones, no canvas mask) must NOT produce NaN.
    The masked gpool divides by mask_sum_hw which is 0 → must clamp."""
    enc = GlobalTokenEncoder(in_channels=3, conv_channels=32, out_dim=64).eval()
    x = torch.zeros(2, 3, 32, 32)
    out = enc(x)
    assert torch.isfinite(out).all(), f"NaN in output for empty canvas: {out}"


def test_gradient_reaches_all_params():
    torch.manual_seed(0)
    enc = GlobalTokenEncoder(in_channels=3, conv_channels=16, out_dim=32)
    x = torch.randn(2, 3, 32, 32, requires_grad=False)
    out = enc(x).sum()
    out.backward()
    missing = []
    for name, p in enc.named_parameters():
        if p.grad is None or torch.all(p.grad == 0):
            missing.append(name)
    assert not missing, f"params with no gradient: {missing}"


def test_state_dict_round_trip():
    torch.manual_seed(0)
    a = GlobalTokenEncoder(in_channels=3, conv_channels=16, out_dim=32).eval()
    b = GlobalTokenEncoder(in_channels=3, conv_channels=16, out_dim=32).eval()
    x = torch.randn(1, 3, 32, 32)
    out_a = a(x)
    b.load_state_dict(a.state_dict())
    out_b = b(x)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_canvas_mask_is_consumed():
    """T2 §E.1 pitfall 2 — the canvas mask plane must influence the gpool
    output. Two inputs with identical stones but different mask coverage
    should produce different tokens."""
    torch.manual_seed(0)
    enc = GlobalTokenEncoder(in_channels=3, conv_channels=16, out_dim=32).eval()
    # Stones in a small region; vary only the mask plane.
    base = torch.zeros(1, 3, 32, 32)
    base[0, 0, 15, 15] = 1.0      # cur stone
    base[0, 1, 16, 16] = 1.0      # opp stone
    # Variant A: mask only over the stones' bbox.
    a = base.clone()
    a[0, 2, 15:17, 15:17] = 1.0
    # Variant B: mask over the entire canvas (padding leak — stones look
    # tiny, canvas average would be nearly zero).
    b = base.clone()
    b[0, 2, :, :] = 1.0
    out_a = enc(a)
    out_b = enc(b)
    diff = (out_a - out_b).abs().max().item()
    assert diff > 1e-3, f"canvas mask does not affect output: max |Δ|={diff}"


def test_mask_plane_index_validation():
    import pytest
    with pytest.raises(ValueError, match="canvas_mask_plane"):
        GlobalTokenEncoder(in_channels=3, canvas_mask_plane=5)


def test_gn_groups_validation():
    import pytest
    with pytest.raises(ValueError, match="conv_channels"):
        GlobalTokenEncoder(conv_channels=10, gn_groups=8)
