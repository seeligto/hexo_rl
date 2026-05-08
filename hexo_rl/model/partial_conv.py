"""§169 A4 — PartialConv2d at v8 trunk entry.

Innamorati et al. 2018 ("Learning on the Edge: Investigating Boundary
Filters in CNNs") partial-conv-padding. Liu et al. 2018 (NVIDIA, image
inpainting) is the lineage; Innamorati's variant uses a fixed mask and
is the precedent we follow for the canvas_realness arm.

The hypothesis under §169 A4: v8's bbox encoding pads off-canvas cells
with zeros and labels them via plane 8 (off_window=1), but the trunk's
first 3×3 conv treats those zeros as ordinary "no stone here" cells,
conflating the canvas BOUNDARY with interior emptiness. Partial conv
at the trunk entry zeroes off-canvas contributions on the input and
renormalises the output by the count of valid neighbours per location,
giving the boundary cells the same per-receptive-field statistics as
the interior. This is the architectural intervention that distinguishes
A4 from B1 (B1 has the off_window plane but no boundary-aware conv).

Diagnostic value: if A4 closes most of the K-cluster vs bbox gap
(B1 v8 SealBot WR ≪ A1 v6w25), the bbox arm's loss was bad padding
semantics — bbox direction lives. If A4 closes little, the loss is
structural (commit to K-cluster line).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """Innamorati 2018 partial-conv-padding with fixed mask.

    forward(x, mask):
        x:    (B, C_in, H, W)  input feature tensor.
        mask: (B, 1, H, W)     1.0 inside canvas, 0.0 outside.

    Returns (B, C_out, H, W). Off-canvas output cells are zeroed.

    Math (per output spatial location o):

        ``out[o] = conv(x ⊙ mask)[o] · (k² / count[o]) · 1[count[o] > 0]``

    where ``count = conv2d(mask, ones_kernel)`` is the per-location
    valid-neighbour count and ``k²`` is the kernel area. The
    renormalisation matches Innamorati's "boundary-handling" scaling
    (sum over valid receptive-field cells, not a count of zeros).
    Final ``· mask`` ensures off-canvas cells output exactly 0 — the
    downstream norm + ReLU stack treats them like padding.

    The mask is fixed (canvas geometry only depends on bbox centroid
    + R=8 radius, not stone positions or ply), so the renormalisation
    constants can be reused across batch rows from the same encoding.
    Liu et al.'s learnable-mask variant is unnecessary for this use
    case; the fixed-mask path is cheaper and closer to KataGo gpool
    semantics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.kernel_area = float(kernel_size * kernel_size)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=padding, bias=bias,
        )
        self.register_buffer(
            "ones_kernel",
            torch.ones(1, 1, kernel_size, kernel_size),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        x_in = x * mask
        out = self.conv(x_in)
        with torch.no_grad():
            count = F.conv2d(mask, self.ones_kernel, padding=self.padding)
            valid = (count > 0).to(out.dtype)
            scale = (self.kernel_area / count.clamp_min(1.0)).to(out.dtype)
            scale = scale * valid
        return out * scale * mask.to(out.dtype)
