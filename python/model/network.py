"""
HexTacToeNet — ResNet backbone with policy and value heads.

Architecture (from docs/01_architecture.md §2):
  Input:  (18, 19, 19) float16 tensor

  Backbone:
    Conv2d(18 → filters, 3×3, padding=1) → BN → ReLU
    × res_blocks residual blocks:
      Conv2d(filters → filters, 3×3, padding=1) → BN → ReLU
      Conv2d(filters → filters, 3×3, padding=1) → BN
      + skip → ReLU

  Policy head:
    Conv2d(filters → 2, 1×1) → BN → ReLU → Flatten
    → Linear(2·H·W → H·W + 1)   (last logit = pass move)
    → log_softmax

  Value head:
    Conv2d(filters → 1, 1×1) → BN → ReLU → Flatten
    → Linear(H·W → 256) → ReLU → Linear(256 → 1) → Tanh
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

_log = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class HexTacToeNet(nn.Module):
    """
    ResNet dual-head network for Hex Tac Toe.

    Args:
        board_size:  spatial dimension of the board (default 19)
        in_channels: number of input tensor channels (default 18)
        filters:     number of conv filters in the backbone (default 128)
        res_blocks:  number of residual blocks (default 10)
    """

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 18,
        filters: int = 128,
        res_blocks: int = 10,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        spatial = board_size * board_size

        # Backbone
        self.input_conv = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(filters)
        self.tower      = nn.Sequential(
            *[ResidualBlock(filters) for _ in range(res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(filters, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * spatial, spatial + 1)

        # Value head
        self.value_conv  = nn.Conv2d(filters, 1, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(spatial, 256)
        self.value_fc2   = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 18, H, W) float16 input tensor

        Returns:
            log_policy: (B, H*W + 1) log-softmax probabilities
            value:      (B, 1)       tanh scalar in [-1, 1]
        """
        # Backbone
        out = F.relu(self.input_bn(self.input_conv(x)))
        out = self.tower(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        log_policy = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return log_policy, value


def compile_model(model: HexTacToeNet, mode: str = "default") -> HexTacToeNet:
    """Apply torch.compile() to `model` with graceful fallback.

    If compilation fails (unsupported op, wrong PyTorch version, no CUDA),
    logs a warning and returns the original uncompiled model so training
    can continue.

    Args:
        model: The network to compile.
        mode:  torch.compile mode. Default "default" is stable across
               PyTorch 2.x. Avoid "reduce-overhead" / "max-autotune" until
               the CUDA graph regression in PyTorch 2.5+ is resolved.

    Returns:
        Compiled model (or original if compilation failed).
    """
    try:
        compiled = torch.compile(model, mode=mode)
        _log.info("torch.compile applied successfully (mode=%s)", mode)
        return compiled  # type: ignore[return-value]
    except Exception as exc:
        _log.warning(
            "torch.compile failed, continuing without compilation: %s", exc
        )
        return model
