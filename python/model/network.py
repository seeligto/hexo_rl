"""
HexTacToeNet — ResNet backbone with policy and value heads.

Architecture (Multi-Window Cluster-Based Approach):
  Input:  (B, 18, 19, 19) float16 tensor

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
import os
from pathlib import Path
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
    Multi-Window Cluster-Based ResNet for Hex Tac Toe.
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
        self.filters = filters
        self.res_blocks = res_blocks
        spatial = board_size * board_size

        self.trunk = Trunk(in_channels, filters, res_blocks)
        
        self.policy_conv = nn.Conv2d(filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * spatial, spatial + 1)

        self.value_conv = nn.Conv2d(filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(spatial, 256)
        self.value_fc2 = nn.Linear(256, 1)

    @property
    def tower(self) -> nn.Sequential:
        """Backward-compatible alias for the trunk tower."""
        return self.trunk.tower

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 18, H, W) float16 tensor, 1:1 scale local view. B can be K clusters.

        Returns:
            log_policy: (B, H*W + 1) log-softmax probabilities
            value:      (B, 1)       tanh scalar in [-1, 1]
        """
        out = self.trunk(x)

        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        log_policy = F.log_softmax(self.policy_fc(p), dim=1)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return log_policy, value


def compile_model(model: HexTacToeNet, mode: str = "default") -> HexTacToeNet:
    try:
        if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
            cache_dir = Path(".torchinductor-cache").resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        compiled = torch.compile(model, mode=mode)
        _log.info("torch.compile applied successfully (mode=%s)", mode)
        return compiled  # type: ignore[return-value]
    except Exception as exc:
        _log.warning(
            "torch.compile failed, continuing without compilation: %s", exc
        )
        return model
