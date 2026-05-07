"""Unit tests for KataGo-style global-pool operators.

Covers:
- KataGPool returns (B, 3·C, 1, 1) with masked mean / size-mean / masked max.
- Off-board cells (mask = 0) do not contribute to pooled mean.
- KataConvAndGPool preserves (B, c_out, H, W) shape and broadcast-adds the
  pooled contribution.
- compute_v8_mask flips the off_window plane convention and produces a
  per-sample valid-cell count.
"""
from __future__ import annotations

import math

import pytest
import torch

from hexo_rl.model.gpool import (
    KataConvAndGPool,
    KataGoPolicyHead,
    KataGPool,
    compute_v8_mask,
)


def _identity_mask(b: int, h: int, w: int) -> torch.Tensor:
    return torch.ones(b, 1, h, w)


def test_katagpool_output_shape() -> None:
    pool = KataGPool()
    x = torch.randn(2, 16, 25, 25)
    mask = _identity_mask(2, 25, 25)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)
    out = pool(x, mask, mask_sum)
    assert out.shape == (2, 3 * 16, 1, 1)


def test_katagpool_masked_mean_excludes_offboard() -> None:
    """Off-board cells (mask=0) must NOT contribute to the mean."""
    pool = KataGPool()
    # Ones inside, large constants outside — mask should suppress outside.
    x = torch.ones(1, 1, 4, 4)
    x[0, 0, 0, 0] = 1000.0   # off-board cell with extreme value
    mask = torch.ones(1, 1, 4, 4)
    mask[0, 0, 0, 0] = 0.0
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)  # 15
    out = pool(x, mask, mask_sum)
    # Channel 0 is masked mean. Should be (Σ inside ones) / 15 = 1.0.
    masked_mean = out[0, 0, 0, 0].item()
    assert math.isclose(masked_mean, 1.0, rel_tol=1e-5), \
        f"masked mean leaked off-board cell: got {masked_mean}, expected 1.0"


def test_katagpool_masked_max_excludes_offboard() -> None:
    """Off-board cells must lose the max competition (they get −1 added)."""
    pool = KataGPool()
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, 0] = 1000.0   # off-board
    x[0, 0, 1, 1] = 0.5      # in-board winner
    mask = torch.ones(1, 1, 4, 4)
    mask[0, 0, 0, 0] = 0.0
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)
    out = pool(x, mask, mask_sum)
    # Channel index 2 (third pool slot) is the masked max.
    max_val = out[0, 2, 0, 0].item()
    assert math.isclose(max_val, 0.5, rel_tol=1e-5), \
        f"masked max picked off-board cell: got {max_val}, expected 0.5"


def test_katagpool_three_pools_concatenated() -> None:
    """First C channels = mean; middle C = mean·size_offset/10; last C = max."""
    pool = KataGPool()
    x = torch.full((1, 4, 25, 25), 0.5)  # constant 0.5 everywhere
    mask = _identity_mask(1, 25, 25)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)  # 625
    out = pool(x, mask, mask_sum)
    # Mean: 0.5 across all C channels.
    assert torch.allclose(out[0, :4, 0, 0], torch.full((4,), 0.5), atol=1e-5)
    # Size-aware mean: 0.5 × (sqrt(625) − 14) / 10 = 0.5 × 1.1 = 0.55.
    expected_size = 0.5 * (math.sqrt(625) - 14.0) / 10.0
    assert torch.allclose(out[0, 4:8, 0, 0], torch.full((4,), expected_size), atol=1e-5)
    # Max: 0.5.
    assert torch.allclose(out[0, 8:12, 0, 0], torch.full((4,), 0.5), atol=1e-5)


def test_kataconvandgpool_preserves_nchw_shape() -> None:
    block = KataConvAndGPool(c_in=64, c_out=96, c_gpool=32)
    x = torch.randn(2, 64, 25, 25)
    mask = _identity_mask(2, 25, 25)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)
    out = block(x, mask, mask_sum)
    assert out.shape == (2, 96, 25, 25), f"unexpected shape {out.shape}"


def test_kataconvandgpool_param_count() -> None:
    """Spot-check the param delta is in the +9k/site ballpark per S2 §3."""
    block = KataConvAndGPool(c_in=128, c_out=96, c_gpool=32)
    n_params = sum(p.numel() for p in block.parameters())
    # conv1r 128×96×3×3=110592; conv1g 128×32×3×3=36864; normg 32+32=64;
    # linear_g 96×96=9216 (3·c_gpool=96 → c_out=96). Total ≈ 156736.
    # S2 §3 quoted "+9,280 vs ordinary conv1" (delta against the conv1 the
    # block replaces); absolute count is ordinary_conv1+gpool_branch.
    assert n_params > 100_000, f"suspicious param count {n_params}"
    assert n_params < 200_000, f"suspicious param count {n_params}"


def test_kataconvandgpool_broadcast_add_only() -> None:
    """When pooled vector is identically 0, output equals conv1r alone."""
    block = KataConvAndGPool(c_in=8, c_out=16, c_gpool=8)
    # Force the gpool branch's linear_g weight to 0 so the broadcast add is a no-op.
    with torch.no_grad():
        block.linear_g.weight.zero_()
    x = torch.randn(1, 8, 9, 9)
    mask = _identity_mask(1, 9, 9)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)
    out = block(x, mask, mask_sum)
    expected = block.conv1r(x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_compute_v8_mask_flip_and_sum() -> None:
    # Off_window plane: 1.0 outside, 0.0 inside. compute_v8_mask returns the
    # KataGo convention (1 inside, 0 outside) plus per-sample mask sum.
    x = torch.zeros(1, 11, 5, 5)
    # Mark plane 8 (off_window) as 1.0 in 4 outer cells, 0.0 in 21 inner cells.
    x[0, 8, 0, 0] = 1.0
    x[0, 8, 0, 4] = 1.0
    x[0, 8, 4, 0] = 1.0
    x[0, 8, 4, 4] = 1.0
    mask, mask_sum_hw = compute_v8_mask(x, off_window_plane_idx=8)
    assert mask.shape == (1, 1, 5, 5)
    assert mask_sum_hw.shape == (1, 1, 1, 1)
    # 25 total cells − 4 off-board = 21 valid.
    assert mask_sum_hw.item() == 21.0
    # Corner cells should be 0 (off-board); centre cell should be 1 (valid).
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 2, 2].item() == 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_kataconvandgpool_dtype(dtype: torch.dtype) -> None:
    block = KataConvAndGPool(c_in=8, c_out=16, c_gpool=8).to(dtype)
    x = torch.randn(1, 8, 9, 9, dtype=dtype)
    mask = torch.ones(1, 1, 9, 9, dtype=dtype)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)
    out = block(x, mask, mask_sum)
    assert out.dtype == dtype


def test_katagopolicyhead_with_gpool_log_softmax_shape() -> None:
    """Output is log-softmax over H*W flattened spatial logits."""
    head = KataGoPolicyHead(c_in=128, spatial=625, use_gpool=True, c_p1=32, c_g1=32)
    x = torch.randn(2, 128, 25, 25)
    mask = _identity_mask(2, 25, 25)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)
    out = head(x, mask, mask_sum)
    assert out.shape == (2, 625)
    # log_softmax → exp sum to 1 per row.
    probs = out.exp().sum(dim=1)
    assert torch.allclose(probs, torch.ones(2), atol=1e-3)


def test_katagopolicyhead_no_gpool_branch() -> None:
    """B0 control arm: degrades to conv1p → bias → ReLU → conv2p (no G branch)."""
    head = KataGoPolicyHead(c_in=128, spatial=625, use_gpool=False, c_p1=32)
    # Verify no gpool components were instantiated.
    assert not hasattr(head, "conv1g")
    assert not hasattr(head, "linear_g")
    assert not hasattr(head, "normg")
    assert not hasattr(head, "gpool")
    x = torch.randn(1, 128, 25, 25)
    mask = _identity_mask(1, 25, 25)
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)
    out = head(x, mask, mask_sum)
    assert out.shape == (1, 625)


def test_katagopolicyhead_offboard_logits_near_negative_inf() -> None:
    """Off-board cells should receive ≈0 probability after log_softmax."""
    head = KataGoPolicyHead(c_in=8, spatial=25, use_gpool=False, c_p1=8)
    x = torch.randn(1, 8, 5, 5)
    # Mask: only the centre 3×3 cells are valid.
    mask = torch.zeros(1, 1, 5, 5)
    mask[0, 0, 1:4, 1:4] = 1.0
    mask_sum = mask.sum(dim=(2, 3), keepdim=True)
    log_probs = head(x, mask, mask_sum)
    probs = log_probs.exp()
    # Off-board cells: < 1e-6 probability.
    mask_flat = mask.flatten(1)
    off_board_probs = probs * (1.0 - mask_flat)
    assert off_board_probs.max().item() < 1e-6, \
        f"off-board cell got non-trivial probability {off_board_probs.max().item()}"
    # On-board probabilities sum to ≈1.
    on_board_total = (probs * mask_flat).sum(dim=1)
    assert torch.allclose(on_board_total, torch.ones(1), atol=1e-3)


def test_katagopolicyhead_param_count_v8_savings() -> None:
    """Verify KataGo head is much smaller than v6's FC head at v8 dims.

    v6 FC head: 2·625 → 626 = ~782k params at 25×25 + pass slot.
    KataGo head no-gpool: ~6k params (conv1p 128·32 + conv2p 32·1 + bias2 32).
    KataGo head with gpool: + conv1g 128·32 + linear_g 96·32 + normg = ~10k.
    """
    head_no_g = KataGoPolicyHead(c_in=128, spatial=625, use_gpool=False, c_p1=32)
    n_no_g = sum(p.numel() for p in head_no_g.parameters())
    head_g = KataGoPolicyHead(c_in=128, spatial=625, use_gpool=True, c_p1=32, c_g1=32)
    n_g = sum(p.numel() for p in head_g.parameters())
    # Both should be far below the 782k FC baseline.
    assert n_no_g < 20_000, f"no-gpool head suspiciously large: {n_no_g}"
    assert n_g < 25_000, f"gpool head suspiciously large: {n_g}"
    # Gpool branch should add ~6-10k params.
    assert n_g > n_no_g
    assert n_g - n_no_g < 15_000
