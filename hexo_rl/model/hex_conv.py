"""
HexConv2d — Conv2d restricted to the hexagonal 7-cell neighborhood.

A standard 3×3 Conv2d on the 19×19 axial-parallelogram board treats all 8
non-center positions equally. Two of those positions — the long-diagonal
offsets (Δq, Δr) ∈ {(-1, -1), (+1, +1)} — are at axial hex_dist = 2, not
hex_dist = 1, so a Cartesian 3×3 conv reaches twice as far along the
parallelogram's long diagonal as it does along any other hex axis.

HexConv2d zeroes those two long-diagonal kernel positions via a fixed
binary mask, leaving 7 active cells (center + 6 hex neighbors). The mask
is applied:

  1. Before each forward via `register_forward_pre_hook` (weight at masked
     positions is zeroed in-place — defends against optimizer.step()
     having just lifted them).
  2. To `weight.grad` via a hook on the parameter (gradient at masked
     positions is zeroed, so optimizer.step() cannot move them away
     from zero in the first place).

Both safeguards are kept so checkpoint round-trips and gradient-based
optimisers see consistent zeros at masked positions.

Phase B' v9 §153 T1 — see /tmp/phase_b_prime_targets.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _hex_3x3_keep_mask() -> torch.Tensor:
    """Return the (1, 1, 3, 3) {0., 1.} mask for the hex 7-cell kernel.

    Mapping: tensor (H, W) ↔ axial (q, r). Kernel indices (kH, kW) ∈
    {0, 1, 2}² correspond to offsets (ΔH, ΔW) = (kH - 1, kW - 1) =
    (Δq, Δr). The two long-diagonal offsets at axial hex_dist = 2 are
    (Δq, Δr) ∈ {(-1, -1), (+1, +1)} → kernel indices (0, 0) and (2, 2).
    """
    m = torch.ones(1, 1, 3, 3, dtype=torch.float32)
    m[0, 0, 0, 0] = 0.0  # offset (-1, -1) — long diagonal
    m[0, 0, 2, 2] = 0.0  # offset (+1, +1) — long diagonal
    return m


class HexConv2d(nn.Conv2d):
    """nn.Conv2d with a fixed hex-neighborhood mask on the 3×3 kernel.

    Behaves as a drop-in for `nn.Conv2d(in, out, 3, padding=1, ...)`.
    `kernel_size` is fixed at 3; passing any other size raises.

    The mask is registered as a non-trainable buffer (so `.to(device)`,
    `state_dict`, and `load_state_dict` all carry it). Two hooks keep
    masked weights and gradients pinned at zero across training:

      * `_apply_mask_pre_forward`: zeros `self.weight.data` at masked
        positions before each forward.
      * `_zero_grad_at_mask`: zeros `weight.grad` at masked positions
        before each optimizer step.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = False,
        **kwargs,
    ) -> None:
        if kernel_size != 3:
            raise ValueError(
                f"HexConv2d only supports kernel_size=3 (got {kernel_size}); "
                f"the hex 7-cell mask is defined on a 3×3 kernel only."
            )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            bias=bias,
            **kwargs,
        )
        # `persistent=False`: hex_mask is reconstructed at __init__ from a
        # fixed lookup, so it never needs to round-trip through state_dict.
        # This keeps checkpoints architecture-agnostic — the same .pt file
        # can be loaded into either a Conv2d HexTacToeNet or a HexConv2d
        # one (the trunk type is selected by the model config knob, not
        # by the saved buffer).
        self.register_buffer(
            "hex_mask", _hex_3x3_keep_mask(), persistent=False
        )
        # Zero masked weights at construction so the model's initial state
        # respects the mask before any forward runs.
        with torch.no_grad():
            self.weight.data.mul_(self.hex_mask)

        self.register_forward_pre_hook(HexConv2d._apply_mask_pre_forward)
        self.weight.register_hook(self._zero_grad_at_mask)

    @staticmethod
    def _apply_mask_pre_forward(module: "HexConv2d", _inputs):
        # Pre-forward: pin weight at masked positions to zero. Defends
        # against optimizer.step() having just nudged them.
        module.weight.data.mul_(module.hex_mask)

    def _zero_grad_at_mask(self, grad: torch.Tensor) -> torch.Tensor:
        # Backward: zero gradient at masked positions so optimizer.step()
        # cannot move them.
        return grad * self.hex_mask
