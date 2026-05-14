"""§169 A3 — global summary token encoder for K-cluster + PMA + global pool.

Encodes a fixed-size 32×32 global crop of the board (3 planes — cur stones /
opp stones / canvas-realness mask) into a single ``(B, dim)`` token that
joins the K cluster tokens as a (K+1)st element of the SAB input set.

Architecture:
  conv (3 → C, 3×3) → GN → ReLU
  conv (C → C, 3×3) → GN → ReLU
  KataGPool (mean ⊕ size-aware mean ⊕ max, masked by canvas_mask plane)
  Linear (3·C → dim)

KataGo-style gpool is reused verbatim from ``hexo_rl/model/gpool.py`` so the
operator is identical to the v8 trunk's gpool sites — same numerical-stability
choices (f32 sum-reduce / divide), same size-aware mean offset (14.0).

T2 §E.1 pitfall 2 (padding leak): the canvas_mask channel (input plane 2)
is read out as the gpool mask so the conv output is averaged ONLY over
real-canvas cells. Without this, the encoder would treat zero-padding
outside the active bbox as real-but-empty board cells and learn off-board
features as signal.

The encoder's output is the global token ``g``; the consumer
``PMAGlobalPool`` applies a learned scalar gate (init 0.1) before
concatenating ``g`` to the K cluster tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexo_rl.model._constants import MODEL_GN_GROUPS
from hexo_rl.model.gpool import KataGPool


# Default config for the encoder. C=64 conv channels keeps the branch cheap
# (~3·C·dim = ~25k parameters in the linear projection at dim=128) and
# matches the spec's "2-3 conv blocks at ~64ch".
DEFAULT_CONV_CHANNELS: int = 64
DEFAULT_GN_GROUPS: int = MODEL_GN_GROUPS
DEFAULT_CANVAS_MASK_PLANE: int = 2  # input plane 2 = canvas-realness mask


class GlobalTokenEncoder(nn.Module):
    """3-channel global crop → d-dim summary token via masked gpool.

    Args:
        in_channels:        number of input planes. Default 3 (cur / opp /
                            canvas_mask). Pinned at 3 because the consumer
                            assumes plane index 2 is the canvas mask.
        conv_channels:      hidden channel count for the two conv blocks.
        out_dim:            output token dimension; must match the cluster
                            pool's ``dim`` (128 for the v6w25 trunk).
        gn_groups:          GroupNorm group count; conv_channels must be
                            divisible by this.
        canvas_mask_plane:  plane index treated as the gpool mask.
    """

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: int = DEFAULT_CONV_CHANNELS,
        out_dim: int = 128,
        gn_groups: int = DEFAULT_GN_GROUPS,
        canvas_mask_plane: int = DEFAULT_CANVAS_MASK_PLANE,
    ) -> None:
        super().__init__()
        if conv_channels % gn_groups != 0:
            raise ValueError(
                f"conv_channels={conv_channels} must be divisible by "
                f"gn_groups={gn_groups}"
            )
        if not (0 <= canvas_mask_plane < in_channels):
            raise ValueError(
                f"canvas_mask_plane={canvas_mask_plane} out of "
                f"[0, in_channels={in_channels})"
            )
        self.in_channels = int(in_channels)
        self.conv_channels = int(conv_channels)
        self.out_dim = int(out_dim)
        self.canvas_mask_plane = int(canvas_mask_plane)

        self.conv1 = nn.Conv2d(in_channels, conv_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, conv_channels)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, conv_channels)
        self.gpool = KataGPool()
        # 3·C → out_dim — KataGPool emits (mean, size-aware mean, max) → 3·C.
        self.proj = nn.Linear(3 * conv_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a (B, 3, H, W) global crop into a (B, out_dim) token.

        ``x[:, canvas_mask_plane]`` is interpreted as the canvas-realness
        mask: 1.0 in cells inside the active bbox region, 0.0 in padding
        cells. The mask is reused as the gpool mask so off-canvas cells
        don't pollute the masked-mean output.
        """
        if x.dim() != 4 or x.size(1) != self.in_channels:
            raise ValueError(
                f"GlobalTokenEncoder expects (B, {self.in_channels}, H, W); "
                f"got {tuple(x.shape)}"
            )
        # Build the canvas mask in (B, 1, H, W) form, plus its spatial sum.
        # mask_sum_hw is clamped to >=1.0 so the gpool divide is NaN-safe
        # for all-padding inputs (empty-board crop). KataGPool's masked
        # mean degenerates to 0 in that case (sum is 0), which the linear
        # projection then maps through.
        mask = x[:, self.canvas_mask_plane:self.canvas_mask_plane + 1, :, :].to(x.dtype)
        mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)

        h = F.relu(self.gn1(self.conv1(x)))
        h = F.relu(self.gn2(self.conv2(h)))
        pooled = self.gpool(h, mask, mask_sum_hw)        # (B, 3·C, 1, 1)
        pooled = pooled.flatten(1)                        # (B, 3·C)
        return self.proj(pooled)                          # (B, out_dim)
