"""
HexTacToeNet — ResNet backbone with SE blocks, dual-pooling value head,
policy head, and opponent-reply auxiliary head.

v6 architecture (Multi-Window Cluster-Based, 8-plane × 19×19):
  Input:  (B, 8, 19, 19) float16 tensor.
  Trunk:  Conv → GN → ReLU → 12 × ResidualBlock(SE).
  Policy: Conv2d(filters→2, 1×1) → ReLU → FC(2·H·W → H·W+1) → log_softmax.
  Value:  GAP+max → FC(2C→256) → ReLU → FC(256→1) → Tanh.
  Opp_reply (aux): mirror of policy head.

v8 architecture (Path β, 11-plane × 25×25, no pass slot):
  Input:  (B, 11, 25, 25) float16. Plane 8 = off_window mask.
  Mask:   computed once at trunk forward entry per `compute_v8_mask`;
          KataGo convention (1.0 inside, 0.0 outside).
  Trunk:  Conv → GN → ReLU → res_blocks ResidualBlocks; blocks at
          `gpool_indices` replace their conv1 with KataConvAndGPool
          (3 pooled scalars per channel · linear_g → broadcast-add).
  Policy: KataGoPolicyHead — 1×1 P branch + (optional) 1×1 G branch with
          KataGPool → linear_g → broadcast-add → bias → ReLU → 1×1 → mask
          off-board → log_softmax. n_actions = H*W = 625 (no pass).
  Value:  unchanged (KataGo's value-head GPool is multi-board-size-only).
  Opp_reply (aux): mirror of policy head.

Spec sources: docs/designs/encoding_v8_contract.md;
audit/encoding_spikes/SPIKE_SUMMARY.md §3.1; audit/encoding_spikes/s2_global_pooling.md.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexo_rl.model.gpool import (
    KataConvAndGPool,
    KataGoPolicyHead,
    compute_v8_mask,
)
from hexo_rl.model.pooling import (
    SUPPORTED_POOL_TYPES,
    PMAPool,
    build_pool,
)

_log = logging.getLogger(__name__)


# Buffer wire-format plane count. Matches engine/src/replay_buffer/sym_tables.rs:N_PLANES.
# Sweep variants reduce model in_channels by selecting a subset of these 8 wire planes
# via the `input_channels` constructor arg — the Rust storage format is unchanged.
WIRE_CHANNELS: int = 8

# Required wire planes — every variant must include at least these or the model
# has no stone information. Plane 0 = cur ply-0, plane 4 = opp ply-0 (8-plane HEXB v6).
REQUIRED_INPUT_CHANNELS: tuple = (0, 4)


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
                f"(plane 0 = cur ply-0, plane 4 = opp ply-0 in 8-plane HEXB v6). "
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


class GPoolResidualBlock(nn.Module):
    """Residual block whose first conv is `KataConvAndGPool`.

    Mirrors KataGo's `ResBlock` with `c_gpool != None`:
    `model_pytorch.py:688-720`. Conv2 stays a regular 3×3 conv preserving
    channel count. SE block sits after conv2 as in our v6 path. The block
    needs `(mask, mask_sum_hw)` plumbed in; non-gpool blocks ignore them.
    """

    def __init__(
        self,
        filters: int,
        c_gpool: int,
        se_reduction_ratio: int = 4,
    ) -> None:
        super().__init__()
        assert filters % _GN_GROUPS == 0, (
            f"filters={filters} must be divisible by num_groups={_GN_GROUPS}"
        )
        self.conv1 = KataConvAndGPool(filters, filters, c_gpool, gn_groups=_GN_GROUPS)
        self.gn1   = nn.GroupNorm(_GN_GROUPS, filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(_GN_GROUPS, filters)
        self.se    = SEBlock(filters, se_reduction_ratio)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_sum_hw: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        out = F.relu(self.gn1(self.conv1(x, mask, mask_sum_hw)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class Trunk(nn.Module):
    """ResNet trunk; supports gpool splices at arbitrary block indices.

    When `gpool_indices` is empty the trunk degrades to v6's stack of plain
    `ResidualBlock`s wrapped in `nn.Sequential` — backward-compatible for
    v6 callers (board_size=19, in_channels=8). When `gpool_indices` is
    non-empty the trunk uses `nn.ModuleList` and threads `(mask, mask_sum_hw)`
    through gpool blocks at the specified positions.
    """

    def __init__(
        self,
        in_channels: int,
        filters: int,
        res_blocks: int,
        se_reduction_ratio: int = 4,
        gpool_indices: Optional[List[int]] = None,
        gpool_channels: int = 32,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, filters, 3, padding=1, bias=False)
        self.input_gn   = nn.GroupNorm(_GN_GROUPS, filters)
        gpool_indices = sorted(set(gpool_indices)) if gpool_indices else []
        self.gpool_indices: List[int] = list(gpool_indices)
        for idx in self.gpool_indices:
            if not (0 <= idx < res_blocks):
                raise ValueError(
                    f"gpool_indices entry {idx} out of [0, {res_blocks}); "
                    f"check variant config."
                )

        # When the trunk has no gpool sites, retain the v6 nn.Sequential layout
        # so existing state_dicts (v6 checkpoints) load byte-exact.
        if not self.gpool_indices:
            self.tower = nn.Sequential(
                *[ResidualBlock(filters, se_reduction_ratio) for _ in range(res_blocks)]
            )
            self._has_gpool = False
        else:
            blocks: List[nn.Module] = []
            for i in range(res_blocks):
                if i in self.gpool_indices:
                    blocks.append(GPoolResidualBlock(filters, gpool_channels,
                                                     se_reduction_ratio))
                else:
                    blocks.append(ResidualBlock(filters, se_reduction_ratio))
            self.tower = nn.ModuleList(blocks)
            self._has_gpool = True

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_sum_hw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = F.relu(self.input_gn(self.input_conv(x)))
        if not self._has_gpool:
            return self.tower(out)
        # Gpool blocks need (mask, mask_sum_hw); plain blocks ignore them.
        if mask is None or mask_sum_hw is None:
            raise RuntimeError(
                "Trunk has gpool blocks but forward() was called without mask. "
                "Provide v8 inputs with off_window plane and ensure "
                "HexTacToeNet.forward computes the mask before calling trunk."
            )
        for block in self.tower:
            if isinstance(block, GPoolResidualBlock):
                out = block(out, mask, mask_sum_hw)
            else:
                out = block(out)
        return out


_V8_OFF_WINDOW_PLANE_DEFAULT: int = 8
_V8_HEAD_P_CHANNELS_DEFAULT: int = 32
_V8_HEAD_G_CHANNELS_DEFAULT: int = 32
_V8_TRUNK_GPOOL_CHANNELS_DEFAULT: int = 32


class HexTacToeNet(nn.Module):
    """ResNet trunk with two encoding paths: v6 (default) and v8 (Path β).

    v8 changes vs. v6:
      - 11-plane × 25×25 input (gated by `encoding="v8"`).
      - Trunk gpool sites at `gpool_indices` (typically `[6, 10]` of 12 blocks).
      - Policy / opp_reply heads → KataGoPolicyHead (1×1 conv + optional G
        branch + off-board logit bias). No pass slot under v8 (P1 close-out:
        pass dead in HTTT) → output dim = H*W = 625 instead of H*W+1 = 362.
      - `off_window_plane_idx` (default 8) sourced for mask plumbing.
    Value head is unchanged — KataGo's value-head GPool is multi-board-size-only.

    The B0 control variant uses `encoding="v8"`, `gpool_indices=None`,
    `head_use_gpool=False` — encoding-shape change only with the KataGo head
    degraded to conv1p → bias → ReLU → conv2p. B1-B4 add gpool to the trunk
    and re-enable the policy head's G branch.
    """

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 8,
        filters: int = 128,
        res_blocks: int = 12,
        se_reduction_ratio: int = 4,
        input_channels: Optional[List[int]] = None,
        encoding: str = "v6",
        gpool_indices: Optional[List[int]] = None,
        gpool_channels: int = _V8_TRUNK_GPOOL_CHANNELS_DEFAULT,
        head_p_channels: int = _V8_HEAD_P_CHANNELS_DEFAULT,
        head_g_channels: int = _V8_HEAD_G_CHANNELS_DEFAULT,
        head_use_gpool: bool = True,
        off_window_plane_idx: int = _V8_OFF_WINDOW_PLANE_DEFAULT,
        pool_type: str = "min_max",
        pool_attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoding not in ("v6", "v6w25", "v8"):
            raise ValueError(
                f"encoding={encoding!r} must be 'v6', 'v6w25' or 'v8'"
            )
        if pool_type not in SUPPORTED_POOL_TYPES:
            raise ValueError(
                f"pool_type={pool_type!r} must be one of {SUPPORTED_POOL_TYPES}"
            )
        # PMA cluster pool replaces the per-cluster value/policy heads with a
        # learned attention pool over the K cluster tokens. Only meaningful for
        # K-cluster encodings (v6 / v6w25). Under v8 (single bbox) there is no
        # K dimension to aggregate, so PMA is gated off.
        if pool_type == "pma" and encoding == "v8":
            raise ValueError(
                "pool_type='pma' is only valid for v6/v6w25 K-cluster "
                "encodings; v8 has a single bounding-box window (no K)."
            )
        self.pool_type: str = pool_type
        # v6w25 = v6 wire format (8 planes + pass slot) at 25×25 cluster
        # window. Model construction is identical to v6 — only board_size
        # differs. Persist the original label so eval/dispatch can detect it.
        self.encoding = encoding
        self.board_size = board_size
        self.filters = filters
        self.res_blocks = res_blocks
        spatial = board_size * board_size

        # Sweep variant support: when `input_channels` is provided, the trunk
        # input conv accepts only the selected wire planes; forward() slices
        # x[:, input_channels, :, :] before the trunk. Buffer/sym kernels stay
        # 8-plane (HEXB v6) — slicing happens entirely model-side. Disabled
        # under v8 since the v8 wire format is already a curated 11-plane set.
        if input_channels is not None:
            if encoding == "v8":
                raise ValueError(
                    "input_channels is a v6-only knob (slices the 8-plane "
                    "wire format); v8 wire format is already a curated 11-plane "
                    "set — drop input_channels under encoding='v8'."
                )
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

        # n_actions: v6 / v6w25 keep the legacy pass slot (HEXB shape); v8 drops it.
        self.has_pass: bool = encoding in ("v6", "v6w25")
        self.n_actions: int = spatial + (1 if self.has_pass else 0)
        self.off_window_plane_idx: int = int(off_window_plane_idx)

        # Trunk — gpool indices threaded through under v8; v6 keeps Sequential.
        self.trunk = Trunk(
            in_channels=self.in_channels,
            filters=filters,
            res_blocks=res_blocks,
            se_reduction_ratio=se_reduction_ratio,
            gpool_indices=gpool_indices if encoding == "v8" else None,
            gpool_channels=gpool_channels,
        )

        # Policy / opp_reply heads — branch on encoding. v6 keeps the FC head
        # (loads existing v6 checkpoints byte-exact); v8 swaps in KataGoPolicyHead.
        if encoding == "v8":
            self.policy_head = KataGoPolicyHead(
                c_in=filters,
                spatial=spatial,
                use_gpool=head_use_gpool,
                c_p1=head_p_channels,
                c_g1=head_g_channels,
                gn_groups=_GN_GROUPS,
            )
            self.opp_reply_head = KataGoPolicyHead(
                c_in=filters,
                spatial=spatial,
                use_gpool=head_use_gpool,
                c_p1=head_p_channels,
                c_g1=head_g_channels,
                gn_groups=_GN_GROUPS,
            )
            # v6 FC head modules left absent under v8 — no fallback path
            # exists, so omitting them surfaces typos at construction time.
            self.policy_conv = None  # type: ignore[assignment]
            self.policy_fc = None    # type: ignore[assignment]
            self.opp_reply_conv = None  # type: ignore[assignment]
            self.opp_reply_fc = None    # type: ignore[assignment]
        else:
            self.policy_head = None  # type: ignore[assignment]
            self.opp_reply_head = None  # type: ignore[assignment]
            # Policy head — no normalization: 2 output channels, GN(8, 2) would fail (groups > channels)
            self.policy_conv = nn.Conv2d(filters, 2, 1)
            self.policy_fc = nn.Linear(2 * spatial, spatial + 1)
            # Opponent reply auxiliary head (training only) — no normalization: same reason as policy head
            self.opp_reply_conv = nn.Conv2d(filters, 2, 1)
            self.opp_reply_fc = nn.Linear(2 * spatial, spatial + 1)

        # Value head — global avg+max pooling
        self.value_fc1 = nn.Linear(2 * filters, 256)
        self.value_fc2 = nn.Linear(256, 1)

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

        # K-cluster pool — §169 A2. When pool_type='min_max' the pool is unused
        # at the model level (the bot does scatter-max). When pool_type='pma'
        # the pool replaces the value/policy heads; forward() routes per-cluster
        # GAP'd trunk features through SAB + PMA seeds. At pretrain time K=1
        # (one cluster window per training sample), so PMA's cross-cluster
        # attention is exercised only via the duplicate-of-itself path; at
        # inference time KClusterMCTSBot drives K>1 via aggregated_forward_K.
        if pool_type == "pma":
            self.cluster_pool: Optional[PMAPool] = build_pool(
                "pma",
                dim=filters,
                n_actions=self.n_actions,
                n_heads=4,
                attn_dropout=pool_attn_dropout,
            )
        else:
            self.cluster_pool = None

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
            x:           (B, in_channels, H, W) float16 tensor. Under v8, plane
                         `off_window_plane_idx` (8) is the off-window mask
                         (1 outside, 0 inside).
            aux:         If True, also return opponent-reply log-policy (training only).
            uncertainty: If True, also return value variance σ² (training only).
            ownership:   If True, also return ownership prediction (B, 1, H, W) ∈ (-1, 1).
            threat:      If True, also return threat logits (B, 1, H, W) raw (training only).
            chain:       If True, also return Q13 chain-length predictions
                         (B, 6, H, W) raw regression outputs (training only).
            Never pass any of these flags from InferenceServer, evaluator, or MCTS paths.

        Base return (all flags False) — 3-tuple, unchanged inference contract:
            log_policy:   (B, n_actions)  log-softmax probabilities
                          (v6: H*W + 1; v8: H*W)
            value:        (B, 1)          tanh scalar in [-1, 1]  (for MCTS)
            value_logit:  (B, 1)          pre-tanh logit          (for BCE loss)
        Additional outputs appended in order:
            opp_reply, sigma2, ownership_pred, threat_pred, chain_pred.
        """
        if self._input_channels is not None:
            x = x.index_select(1, self.input_channel_index)

        # v8 mask plumbing — computed once per forward pass and reused at every
        # gpool site (trunk + policy / opp_reply heads).
        if self.encoding == "v8":
            mask, mask_sum_hw = compute_v8_mask(x, self.off_window_plane_idx)
        else:
            mask = None
            mask_sum_hw = None

        out = self.trunk(x, mask=mask, mask_sum_hw=mask_sum_hw)

        # PMA pool path — replace value/policy heads with the K-cluster pool.
        # At pretrain time x is (B, C, H, W) (K=1 per sample); the pool sees a
        # single token per board. Cross-cluster attention is trained mostly
        # through the K=1 collapse fallback — PMA-collapse risk is the §169
        # surfacing condition. ``aggregated_forward_K`` is the K>1 inference
        # entry point used by ``KClusterMCTSBot`` when ``pool_type='pma'``.
        if self.pool_type == "pma":
            assert self.cluster_pool is not None
            cluster_tokens = out.mean(dim=(2, 3)).unsqueeze(1)   # (B, K=1, C)
            log_policy, value, v_logit = self.cluster_pool(
                cluster_tokens,
                per_cluster_logits=None,
                per_cluster_values=None,
            )
        else:
            # Policy head — branch on encoding.
            if self.encoding == "v8":
                log_policy = self.policy_head(out, mask, mask_sum_hw)
            else:
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
            if self.encoding == "v8":
                extras.append(self.opp_reply_head(out, mask, mask_sum_hw))
            else:
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

    @torch.no_grad()
    def aggregated_forward_K(
        self,
        x_K: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """K>1 inference entry point for K-cluster encodings.

        Args:
            x_K: ``(K, in_channels, H, W)`` — the K cluster windows for a
                 single board. The model treats them as a single sample with
                 K cluster tokens; the cluster pool aggregates them into one
                 ``(log_policy, value, v_logit)`` output.

        Returns the standard 3-tuple but with leading dim 1 (the aggregated
        sample). Callers (``KClusterMCTSBot``) reshape as needed.

        For ``pool_type='min_max'`` this method is a thin wrapper around the
        per-cluster forward followed by a model-internal min/max reduction,
        kept for interface symmetry — most callers stay on the existing
        bot-side scatter-max path.
        """
        if self._input_channels is not None:
            x_K = x_K.index_select(1, self.input_channel_index)

        if self.encoding == "v8":
            raise RuntimeError(
                "aggregated_forward_K is K-cluster only; v8 has no K dim."
            )

        out = self.trunk(x_K, mask=None, mask_sum_hw=None)        # (K, C, H, W)

        if self.pool_type == "pma":
            assert self.cluster_pool is not None
            cluster_tokens = out.mean(dim=(2, 3)).unsqueeze(0)    # (1, K, C)
            log_policy, value, v_logit = self.cluster_pool(
                cluster_tokens,
                per_cluster_logits=None,
                per_cluster_values=None,
            )
            return log_policy, value, v_logit

        # min_max path — run per-cluster heads, reduce on the model side via
        # the same scatter-max-in-prob-space rule as the bot. Lifted into the
        # model so callers can switch pool_type without reaching into bot
        # internals.
        from hexo_rl.model.pooling import MinMaxPool
        per_p = F.relu(self.policy_conv(out)).flatten(1)
        per_logp = F.log_softmax(self.policy_fc(per_p), dim=1)    # (K, A)
        v_avg = out.mean(dim=(2, 3))
        v_max = out.amax(dim=(2, 3))
        v_cat = torch.cat([v_avg, v_max], dim=1)
        per_vlogit = self.value_fc2(F.relu(self.value_fc1(v_cat)))
        per_val = torch.tanh(per_vlogit)                          # (K, 1)

        pool = MinMaxPool()
        return pool(
            cluster_tokens=out.mean(dim=(2, 3)).unsqueeze(0),    # (1, K, C) — unused
            per_cluster_logits=per_logp.unsqueeze(0),             # (1, K, A)
            per_cluster_values=per_val.unsqueeze(0),              # (1, K, 1)
        )


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
