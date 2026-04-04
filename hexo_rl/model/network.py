"""
HexTacToeNet — ResNet backbone with SE blocks, dual-pooling value head,
policy head, and opponent-reply auxiliary head.

Architecture (Multi-Window Cluster-Based Approach):
  Input:  (B, 18, 19, 19) float16 tensor

  Backbone:
    Conv2d(18 → filters, 3×3, padding=1) → BN → ReLU
    × res_blocks residual blocks:
      Conv2d(filters → filters, 3×3, padding=1) → BN → ReLU
      Conv2d(filters → filters, 3×3, padding=1) → BN
      SE block: GAP → FC(C → C//r) → ReLU → FC(C//r → C) → Sigmoid → scale
      + skip → ReLU

  Policy head:
    Conv2d(filters → 2, 1×1) → BN → ReLU → Flatten
    → Linear(2·H·W → H·W + 1)   (last logit = pass move)
    → log_softmax

  Value head:
    Global avg pool → (C,)  |  Global max pool → (C,)
    Concat → (2C,) → FC(2C → 256) → ReLU → FC(256 → 1) → Tanh

  Opponent reply head (training only):
    Conv2d(filters → 2, 1×1) → BN → ReLU → Flatten
    → Linear(2·H·W → H·W + 1)
    → log_softmax
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction_ratio, 1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))              # (B, C) — squeeze
        s = F.relu(self.fc1(s))              # (B, C//r)
        s = torch.sigmoid(self.fc2(s))       # (B, C)
        return x * s.view(b, c, 1, 1)       # scale


class ResidualBlock(nn.Module):
    def __init__(self, filters: int, se_reduction_ratio: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)
        self.se    = SEBlock(filters, se_reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class Trunk(nn.Module):
    def __init__(self, in_channels: int, filters: int, res_blocks: int,
                 se_reduction_ratio: int = 4):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(filters)
        self.tower      = nn.Sequential(
            *[ResidualBlock(filters, se_reduction_ratio) for _ in range(res_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_bn(self.input_conv(x)))
        return self.tower(out)


class HexTacToeNet(nn.Module):
    """Multi-Window Cluster-Based ResNet for Hex Tac Toe."""

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 18,
        filters: int = 128,
        res_blocks: int = 12,
        se_reduction_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.filters = filters
        self.res_blocks = res_blocks
        spatial = board_size * board_size

        self.trunk = Trunk(in_channels, filters, res_blocks, se_reduction_ratio)

        # Policy head
        self.policy_conv = nn.Conv2d(filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * spatial, spatial + 1)

        # Value head — global avg+max pooling
        self.value_fc1 = nn.Linear(2 * filters, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Opponent reply auxiliary head (training only)
        self.opp_reply_conv = nn.Conv2d(filters, 2, 1)
        self.opp_reply_bn = nn.BatchNorm2d(2)
        self.opp_reply_fc = nn.Linear(2 * spatial, spatial + 1)

        # Value uncertainty head (training only — diagnostic σ², never used in MCTS)
        # Reads from the same trunk features as the value head.
        # Softplus ensures σ² > 0.
        self.value_var = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters, 1),
            nn.Softplus(),
        )

    @property
    def tower(self) -> nn.Sequential:
        """Backward-compatible alias for the trunk tower."""
        return self.trunk.tower

    def forward(
        self, x: torch.Tensor, aux: bool = False, uncertainty: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:           (B, 18, H, W) float16 tensor.
            aux:         If True, also return opponent-reply log-policy (training only).
            uncertainty: If True, also return value variance σ² (training only).
                         Never pass True from InferenceServer, evaluator, or MCTS paths.

        Returns (aux=False, uncertainty=False):   — 3-tuple, unchanged contract
            log_policy:   (B, H*W + 1)  log-softmax probabilities
            value:        (B, 1)        tanh scalar in [-1, 1]  (for MCTS)
            value_logit:  (B, 1)        pre-tanh logit          (for BCE loss)
        Returns (aux=True, uncertainty=False):    — 4-tuple
            log_policy, value, value_logit, opp_reply
        Returns (aux=False, uncertainty=True):    — 4-tuple
            log_policy, value, value_logit, sigma2   (σ² > 0 via Softplus)
        Returns (aux=True, uncertainty=True):     — 5-tuple
            log_policy, value, value_logit, opp_reply, sigma2
        """
        out = self.trunk(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        log_policy = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head — global avg + max pooling
        v_avg = out.mean(dim=(2, 3))           # (B, C)
        v_max = out.amax(dim=(2, 3))           # (B, C)
        v = torch.cat([v_avg, v_max], dim=1)   # (B, 2C)
        v = F.relu(self.value_fc1(v))
        v_logit = self.value_fc2(v)            # (B, 1) raw logit
        value = torch.tanh(v_logit)

        if aux and uncertainty:
            o = F.relu(self.opp_reply_bn(self.opp_reply_conv(out)))
            o = o.flatten(1)
            opp_reply = F.log_softmax(self.opp_reply_fc(o), dim=1)
            sigma2 = self.value_var(out)       # (B, 1), σ² > 0
            return log_policy, value, v_logit, opp_reply, sigma2

        if aux:
            o = F.relu(self.opp_reply_bn(self.opp_reply_conv(out)))
            o = o.flatten(1)
            opp_reply = F.log_softmax(self.opp_reply_fc(o), dim=1)
            return log_policy, value, v_logit, opp_reply

        if uncertainty:
            sigma2 = self.value_var(out)       # (B, 1), σ² > 0
            return log_policy, value, v_logit, sigma2

        return log_policy, value, v_logit


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
