"""
HexTacToeNet — ResNet backbone with SE blocks, dual-pooling value head,
policy head, and opponent-reply auxiliary head.

Architecture (Multi-Window Cluster-Based Approach):
  Input:  (B, 18, 19, 19) float16 tensor
          (18 AlphaZero history/scalar planes; chain-length planes in aux sub-buffer post-§97)

  Backbone:
    Conv2d(18 → filters, 3×3, padding=1) → GN(8) → ReLU
    × res_blocks residual blocks:
      Conv2d(filters → filters, 3×3, padding=1) → GN(8) → ReLU
      Conv2d(filters → filters, 3×3, padding=1) → GN(8)
      SE block: GAP → FC(C → C//r) → ReLU → FC(C//r → C) → Sigmoid → scale
      + skip → ReLU

  Policy head:
    Conv2d(filters → 2, 1×1) → ReLU → Flatten
    → Linear(2·H·W → H·W + 1)   (last logit = pass move)
    → log_softmax

  Value head:
    Global avg pool → (C,)  |  Global max pool → (C,)
    Concat → (2C,) → FC(2C → 256) → ReLU → FC(256 → 1) → Tanh

  Opponent reply head (training only):
    Conv2d(filters → 2, 1×1) → ReLU → Flatten
    → Linear(2·H·W → H·W + 1)
    → log_softmax
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_log = logging.getLogger(__name__)


# Buffer wire-format plane count. The replay buffer, symmetry kernels, and
# inference batch tensors are all hardcoded to 18 planes (engine/src/replay_buffer/
# sym_tables.rs:N_PLANES). Sweep variants reduce model in_channels by selecting
# a subset of these 18 wire planes via the `input_channels` constructor arg —
# the Rust storage format is unchanged.
WIRE_CHANNELS: int = 18

# Required wire planes — every variant must include at least these or the model
# has no stone information. Plane 0 = current-player stones, plane 8 = opponent
# stones (engine/src/board/state.rs:401–408). All other planes are optional
# scalars or history slots.
REQUIRED_INPUT_CHANNELS: tuple = (0, 8)


def validate_input_channels(channels) -> List[int]:
    """Validate a variant's `input_channels` list. Fail loudly on misconfig.

    Returns the canonicalised list (ints, in the order given). Raises
    ValueError with a clear pointer to the config key if invalid. Called at
    HexTacToeNet construction so YAML typos surface at load-time, not later
    inside a forward pass.
    """
    if not isinstance(channels, (list, tuple)):
        raise ValueError(
            f"input_channels must be a list/tuple, got {type(channels).__name__}. "
            f"Fix the variant YAML's `input_channels` field."
        )
    canon: List[int] = []
    for i, c in enumerate(channels):
        try:
            ci = int(c)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"input_channels[{i}] is not an integer: {c!r}. {exc}"
            ) from exc
        if ci < 0 or ci >= WIRE_CHANNELS:
            raise ValueError(
                f"input_channels[{i}]={ci} out of range [0, {WIRE_CHANNELS}). "
                f"Wire format has {WIRE_CHANNELS} planes; see "
                f"hexo_rl/model/network.py:WIRE_CHANNELS."
            )
        if ci in canon:
            raise ValueError(
                f"input_channels has duplicate index {ci}; each plane must "
                f"appear at most once."
            )
        canon.append(ci)
    for required in REQUIRED_INPUT_CHANNELS:
        if required not in canon:
            raise ValueError(
                f"input_channels missing required plane {required} "
                f"(plane 0 = current stones, plane 8 = opponent stones). "
                f"Configured: {canon}. Edit the variant YAML's "
                f"`input_channels` field."
            )
    return canon


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


_GN_GROUPS = 8  # GroupNorm group count; filters must be divisible by this


class ResidualBlock(nn.Module):
    def __init__(self, filters: int, se_reduction_ratio: int = 4) -> None:
        super().__init__()
        assert filters % _GN_GROUPS == 0, (
            f"filters={filters} must be divisible by num_groups={_GN_GROUPS}"
        )
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(_GN_GROUPS, filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(_GN_GROUPS, filters)
        self.se    = SEBlock(filters, se_reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class Trunk(nn.Module):
    def __init__(self, in_channels: int, filters: int, res_blocks: int,
                 se_reduction_ratio: int = 4):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.input_gn   = nn.GroupNorm(_GN_GROUPS, filters)
        self.tower      = nn.Sequential(
            *[ResidualBlock(filters, se_reduction_ratio) for _ in range(res_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.input_gn(self.input_conv(x)))
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
        input_channels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.filters = filters
        self.res_blocks = res_blocks
        spatial = board_size * board_size

        # Sweep variant support: when `input_channels` is provided, the trunk
        # input conv accepts only the selected wire planes; forward() slices
        # x[:, input_channels, :, :] before the trunk. Buffer/sym kernels stay
        # 18-plane — slicing happens entirely model-side.
        if input_channels is not None:
            channels = validate_input_channels(input_channels)
            if int(in_channels) != len(channels):
                raise ValueError(
                    f"in_channels={in_channels} disagrees with "
                    f"len(input_channels)={len(channels)}; pass them consistently "
                    f"or set in_channels=len(input_channels)."
                )
            self._input_channels: Optional[List[int]] = list(channels)
            # Persist as a non-trainable buffer so the indices follow the model
            # across .to(device) and survive state_dict round-trips for audit.
            self.register_buffer(
                "input_channel_index",
                torch.tensor(channels, dtype=torch.long),
                persistent=True,
            )
            self.in_channels = len(channels)
        else:
            self._input_channels = None
            self.input_channel_index = None  # type: ignore[assignment]
            self.in_channels = int(in_channels)

        self.trunk = Trunk(self.in_channels, filters, res_blocks, se_reduction_ratio)

        # Policy head — no normalization: 2 output channels, GN(8, 2) would fail (groups > channels)
        self.policy_conv = nn.Conv2d(filters, 2, 1)
        self.policy_fc = nn.Linear(2 * spatial, spatial + 1)

        # Value head — global avg+max pooling
        self.value_fc1 = nn.Linear(2 * filters, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Opponent reply auxiliary head (training only) — no normalization: same reason as policy head
        self.opp_reply_conv = nn.Conv2d(filters, 2, 1)
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

        # Ownership head (training only — never called from InferenceServer, evaluator, or MCTS).
        # Predicts per-cell stone affiliation: +1 = P1, -1 = P2, 0 = empty.
        self.ownership_head = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size=1),
            nn.Tanh(),
        )

        # Threat head (training only — never called from InferenceServer, evaluator, or MCTS).
        # Predicts per-cell binary winning-line membership. Returns raw logits for BCE.
        self.threat_head = nn.Conv2d(filters, 1, kernel_size=1)

        # Q13-aux chain-length prediction head (training only).
        # Predicts the 6 Q13 chain-length planes from trunk features via smooth-L1
        # (Huber) regression. Rationale (literature review §"Recommended encoding
        # specification"): dual-benefit with chain input planes — forces the trunk
        # to build internal chain-counting circuits even as explicit inputs saturate.
        # KataGo ablation Wu 2019 Table 2 showed auxiliary targets gave 1.65×
        # speedup — the largest single factor in their feature study. Note: our
        # target is an input slice (not future information like KataGo's ownership),
        # so realistic uplift is smaller (~1.1–1.3× tactical sharpening).
        self.chain_head = nn.Conv2d(filters, 6, kernel_size=1)

    @property
    def tower(self) -> nn.Sequential:
        """Backward-compatible alias for the trunk tower."""
        return self.trunk.tower

    def forward(
        self,
        x: torch.Tensor,
        aux: bool = False,
        uncertainty: bool = False,
        ownership: bool = False,
        threat: bool = False,
        chain: bool = False,
    ) -> tuple:
        """
        Args:
            x:           (B, 24, H, W) float16 tensor.
            aux:         If True, also return opponent-reply log-policy (training only).
            uncertainty: If True, also return value variance σ² (training only).
            ownership:   If True, also return ownership prediction (B, 1, H, W) ∈ (-1, 1).
            threat:      If True, also return threat logits (B, 1, H, W) raw (training only).
            chain:       If True, also return Q13 chain-length predictions
                         (B, 6, H, W) raw regression outputs (training only).
            Never pass any of these flags from InferenceServer, evaluator, or MCTS paths.

        Base return (all flags False) — 3-tuple, unchanged inference contract:
            log_policy:   (B, H*W + 1)  log-softmax probabilities
            value:        (B, 1)        tanh scalar in [-1, 1]  (for MCTS)
            value_logit:  (B, 1)        pre-tanh logit          (for BCE loss)
        Additional outputs appended in order:
            opp_reply, sigma2, ownership_pred, threat_pred, chain_pred.
        """
        if self._input_channels is not None:
            x = x.index_select(1, self.input_channel_index)
        out = self.trunk(x)

        # Policy head
        p = F.relu(self.policy_conv(out))
        p = p.flatten(1)
        log_policy = F.log_softmax(self.policy_fc(p), dim=1)

        # Value head — global avg + max pooling
        v_avg = out.mean(dim=(2, 3))           # (B, C)
        v_max = out.amax(dim=(2, 3))           # (B, C)
        v = torch.cat([v_avg, v_max], dim=1)   # (B, 2C)
        v = F.relu(self.value_fc1(v))
        v_logit = self.value_fc2(v)            # (B, 1) raw logit
        value = torch.tanh(v_logit)

        # Build the base 3-tuple; optional heads are appended in order.
        extras: list = []

        if aux:
            o = F.relu(self.opp_reply_conv(out))
            o = o.flatten(1)
            extras.append(F.log_softmax(self.opp_reply_fc(o), dim=1))

        if uncertainty:
            extras.append(self.value_var(out))   # (B, 1), σ² > 0

        if ownership:
            extras.append(self.ownership_head(out))  # (B, 1, H, W) ∈ (-1, 1)

        if threat:
            extras.append(self.threat_head(out))     # (B, 1, H, W) raw logits

        if chain:
            extras.append(self.chain_head(out))      # (B, 6, H, W) raw regression

        if not extras:
            return log_policy, value, v_logit
        return (log_policy, value, v_logit, *extras)


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
