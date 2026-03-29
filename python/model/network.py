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


class Trunk(nn.Module):
    def __init__(self, in_channels: int, filters: int, res_blocks: int):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(filters)
        self.tower      = nn.Sequential(
            *[ResidualBlock(filters) for _ in range(res_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_bn(self.input_conv(x)))
        return self.tower(out)


class HexTacToeNet(nn.Module):
    """
    Dual-Resolution Foveated Vision ResNet for Hex Tac Toe.
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

        # Dual Trunks
        self.local_trunk = Trunk(in_channels, filters, res_blocks)
        self.global_trunk = Trunk(in_channels, filters, res_blocks)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(filters * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Heads
        self.policy_fc = nn.Linear(256, spatial + 1)
        self.value_fc = nn.Linear(256, 1)

    def forward(self, local_x: torch.Tensor, global_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            local_x: (B, 18, H, W) float16 tensor, 1:1 scale local view.
            global_x: (B, 18, H, W) float16 tensor, macro-grid view.

        Returns:
            log_policy: (B, H*W + 1) log-softmax probabilities
            value:      (B, 1)       tanh scalar in [-1, 1]
        """
        local_out = self.local_trunk(local_x)
        global_out = self.global_trunk(global_x)

        # Global Average Pooling (GAP)
        local_gap = F.adaptive_avg_pool2d(local_out, (1, 1)).flatten(1)
        global_gap = F.adaptive_avg_pool2d(global_out, (1, 1)).flatten(1)

        # Feature Fusion
        fused = torch.cat([local_gap, global_gap], dim=1)
        fused = self.fusion_mlp(fused)

        # Output Heads
        log_policy = F.log_softmax(self.policy_fc(fused), dim=1)
        value = torch.tanh(self.value_fc(fused))

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
