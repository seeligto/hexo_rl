"""KataGo-style global-pool operators for v8 encoding (Path β).

Direct port of KataGo's `KataGPool` and `KataConvAndGPool` from
`python/katago/train/model_pytorch.py:468-583` (lightvector/KataGo @ master,
fetched 2026-05-07). Spec: `audit/encoding_spikes/s2_global_pooling.md` §1.

The v8 encoding wires off-board cells with a mask plane (plane 8 =
`off_window`, 1.0 outside the legal hex, 0.0 inside). Policy/value math
must ignore those cells. KataGo's recipe:

  mean  = Σ(x · mask) / Σ(mask)            — masked mean
  meanS = mean · (sqrt(N) − 14) / 10        — board-size-aware scale
  max   = max(x + (mask − 1))               — off-board cells biased by −1

The 3 pooled scalars are concatenated (3·C channels) and then projected
back to the trunk channel count by a `linear_g`. KataGo broadcasts-ADD
the projected vector into the spatial branch, preserving channel count
(so residual connections stay byte-shape-compatible). See
`audit/encoding_spikes/s2_global_pooling.md` §1.3.

The policy head additionally projects the pooled vector to the spatial
P-branch channels (broadcast-add) and historically to a `linear_pass`
slot. Per P1 close-out the pass slot is dead in HTTT, so the head emits
only spatial logits — `KataGoPolicyHead` with `use_gpool=False` further
drops the G branch entirely (B0 control arm).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default board-size offset for KataGo's size-aware mean scaling.
# KataGo uses 14.0 = sqrt(196), the midpoint of typical training board sizes
# (9² to 19²). For HeXO v8 the only training spatial extent is 25×25 (N=625,
# sqrt≈25), so the offset is fixed at the KataGo default to keep the operator
# verbatim; the `linear_g` projection learns the resulting bias contribution.
_KATAGO_SIZE_OFFSET: float = 14.0


class KataGPool(nn.Module):
    """KataGo global pool: 3 masked pooled scalars per channel.

    Forward:
        x:           (B, C, H, W)     — pre-pool features (typically post norm+ReLU).
        mask:        (B, 1, H, W)     — 1.0 inside valid cells, 0.0 outside.
        mask_sum_hw: (B, 1, 1, 1)     — Σ mask spatial (valid cell count per sample).

    Returns:
        (B, 3·C, 1, 1) tensor: cat([masked_mean, masked_mean·size_offset/10,
        masked_max]) along the channel axis.

    Note: the f32 cast on the running sum mirrors KataGo's numerical-stability
    choice (`model_pytorch.py:483`); the divide stays in f32 and the result is
    cast back to the input dtype before returning.
    """

    def __init__(self, size_offset: float = _KATAGO_SIZE_OFFSET) -> None:
        super().__init__()
        self.size_offset = size_offset

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_sum_hw: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        out_dtype = x.dtype

        # Masked mean — KataGo computes the sum in f32 then divides.
        masked_x = x * mask
        layer_mean = masked_x.sum(dim=(2, 3), keepdim=True, dtype=torch.float32) \
            / mask_sum_hw.float()
        # Size-aware scale (board_size sensitivity bias).
        size_offset = torch.sqrt(mask_sum_hw.float()) - self.size_offset
        layer_mean_size = layer_mean * (size_offset / 10.0)

        # Masked max — off-board cells get −1.0 added so post-ReLU on-board
        # values (≥ 0) win the argmax. Equivalent to `x + (mask − 1)`.
        x_for_max = (masked_x + (mask - 1.0)).reshape(b, c, h * w)
        layer_max = x_for_max.amax(dim=2, keepdim=False).view(b, c, 1, 1)

        out = torch.cat(
            [
                layer_mean.to(out_dtype),
                layer_mean_size.to(out_dtype),
                layer_max,
            ],
            dim=1,
        )
        return out


class KataConvAndGPool(nn.Module):
    """KataGo's gpool-aware conv block.

    Replaces the FIRST conv of a residual block when gpool is enabled at that
    site. Two parallel branches:
      - conv1r (3×3, c_in → c_out)         — regular spatial conv.
      - conv1g (3×3, c_in → c_gpool) → norm → ReLU → KataGPool → linear_g
        (3·c_gpool → c_out)                — gpool branch projected back.
    Output = conv1r + linear_g(pool(...)).unsqueeze(spatial) — broadcast-add.

    Channel count is preserved (output = c_out), so the surrounding residual
    block sees the same NCHW shape it would without the gpool branch.

    Forward:
        x:           (B, c_in, H, W)
        mask:        (B, 1, H, W)
        mask_sum_hw: (B, 1, 1, 1)
    Returns:
        (B, c_out, H, W)
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        c_gpool: int,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        if c_gpool % gn_groups != 0:
            raise ValueError(
                f"c_gpool={c_gpool} must be divisible by gn_groups={gn_groups}; "
                f"adjust c_gpool or norm group count."
            )
        self.c_out = c_out
        self.c_gpool = c_gpool
        self.conv1r = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.conv1g = nn.Conv2d(c_in, c_gpool, 3, padding=1, bias=False)
        self.normg = nn.GroupNorm(gn_groups, c_gpool)
        self.gpool = KataGPool()
        self.linear_g = nn.Linear(3 * c_gpool, c_out, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_sum_hw: torch.Tensor,
    ) -> torch.Tensor:
        outr = self.conv1r(x)
        outg = self.conv1g(x)
        outg = self.normg(outg)
        outg = F.relu(outg)
        outg = self.gpool(outg, mask, mask_sum_hw)        # (B, 3·c_gpool, 1, 1)
        outg = outg.flatten(1)                             # (B, 3·c_gpool)
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)  # (B, c_out, 1, 1)
        return outr + outg                                 # broadcast-add to NCHW


class KataGoPolicyHead(nn.Module):
    """KataGo-style policy head for v8 (no pass slot).

    Two parallel branches:
      - P branch: 1×1 conv (c_in → c_p1)                              — spatial.
      - G branch (optional, when `use_gpool=True`):
            1×1 conv (c_in → c_g1) → norm → ReLU → KataGPool
            → linear_g (3·c_g1 → c_p1)
        broadcast-add'd into the P branch output.
    Followed by:
      - per-channel bias on c_p1, ReLU, 1×1 conv (c_p1 → 1) → spatial logits.
    Off-board cells are biased −5000.0 before flatten so softmax assigns
    them ≈0 probability (KataGo `model_pytorch.py:2479`).

    Per P1 close-out the pass slot is dead in HTTT, so this head emits ONLY
    the H×W spatial logits (n_actions = H*W). The B0 control arm sets
    `use_gpool=False`, degrading the head to conv1p → bias → ReLU → conv2p
    (no global context).

    Returns log-softmax over the H*W flattened logits to match the existing
    HexTacToeNet contract (loss is NLL on `target_log_policy`).
    """

    def __init__(
        self,
        c_in: int,
        spatial: int,
        use_gpool: bool = True,
        c_p1: int = 32,
        c_g1: int = 32,
        gn_groups: int = 8,
        offboard_logit_bias: float = -5000.0,
    ) -> None:
        super().__init__()
        self.spatial = spatial
        self.use_gpool = use_gpool
        self.offboard_logit_bias = offboard_logit_bias

        self.conv1p = nn.Conv2d(c_in, c_p1, 1, bias=False)
        if use_gpool:
            if c_g1 % gn_groups != 0:
                raise ValueError(
                    f"c_g1={c_g1} must be divisible by gn_groups={gn_groups}; "
                    f"adjust c_g1 or norm group count."
                )
            self.conv1g = nn.Conv2d(c_in, c_g1, 1, bias=False)
            self.normg = nn.GroupNorm(gn_groups, c_g1)
            self.gpool = KataGPool()
            self.linear_g = nn.Linear(3 * c_g1, c_p1, bias=False)
        # Per-channel bias on the P branch (KataGo BiasMask reduced to a Param).
        self.bias2 = nn.Parameter(torch.zeros(c_p1))
        # Final 1×1 conv → 1 spatial channel (no pass slot under v8).
        self.conv2p = nn.Conv2d(c_p1, 1, 1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_sum_hw: torch.Tensor,
    ) -> torch.Tensor:
        outp = self.conv1p(x)
        if self.use_gpool:
            outg = self.conv1g(x)
            outg = self.normg(outg)
            outg = F.relu(outg)
            outg = self.gpool(outg, mask, mask_sum_hw).flatten(1)
            outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)
            outp = outp + outg                                   # NCHW + NC11
        outp = outp + self.bias2.view(1, -1, 1, 1)
        outp = F.relu(outp)
        outp = self.conv2p(outp).squeeze(1)                       # (B, H, W)
        # Off-board logit bias — softmax sends those entries to ≈0.
        outp = outp + (1.0 - mask.squeeze(1)) * self.offboard_logit_bias
        return F.log_softmax(outp.flatten(1), dim=1)              # (B, H*W)


def compute_v8_mask(
    x: torch.Tensor,
    off_window_plane_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (mask, mask_sum_hw) once at the trunk forward entry.

    v8 encoding's plane 8 (`off_window`) stores 1.0 OUTSIDE the legal hex
    (padding cell) and 0.0 INSIDE. KataGo's convention is the inverse:
    `mask = 1.0` inside, `0.0` outside. The flip is computed once and
    reused at every gpool site.

    Args:
        x: (B, C, H, W) — v8-encoded input tensor.
        off_window_plane_idx: index of the off_window plane (8 for v8).

    Returns:
        (mask, mask_sum_hw):
        - mask: (B, 1, H, W) float, 1 inside valid cells, 0 off-board.
        - mask_sum_hw: (B, 1, 1, 1) — Σ mask spatial per sample.
    """
    off = x[:, off_window_plane_idx:off_window_plane_idx + 1, :, :]
    mask = (1.0 - off).to(x.dtype)
    mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
    return mask, mask_sum_hw
