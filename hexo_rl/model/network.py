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
from hexo_rl.model.global_token import GlobalTokenEncoder
from hexo_rl.model.gpool_bias import GpoolBiasBranch
from hexo_rl.model.network_min_max_head import min_max_window_head
from hexo_rl.model.partial_conv import PartialConv2d
from hexo_rl.model.pooling import (
    SUPPORTED_POOL_TYPES,
    MinMaxPool,
    PMAGlobalPool,
    PMAPool,
    build_pool,
)
from hexo_rl.encoding import all_specs, lookup
from hexo_rl.model._constants import MODEL_GN_GROUPS

BUFFER_CHANNELS: int = lookup("v6").n_planes

_log = logging.getLogger(__name__)


# Buffer wire-format plane count. Matches engine/src/replay_buffer/sym_tables.rs:N_PLANES.
# Sweep variants reduce model in_channels by selecting a subset of these 8 wire planes
# via the `input_channels` constructor arg — the Rust storage format is unchanged.

WIRE_CHANNELS: int = BUFFER_CHANNELS

# §176 P1 — registry-derived encoding whitelist replaces hardcoded
# ("v6", "v6w25", "v8") tuple at the HexTacToeNet ctor. Built once at
# import; the §172 registry is the single source of truth for which
# names are accepted. Adding a registry entry auto-extends acceptance.
_VALID_ENCODINGS: frozenset = frozenset(s.name for s in all_specs())

# Required wire planes — every variant must include at least these or the model
# has no stone information. Plane 0 = cur ply-0, plane 4 = opp ply-0 (8-plane HEXB v6).
_REQUIRED_INPUT_CHANNELS: tuple = (0, 4)


def _validate_input_channels(channels) -> List[int]:
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
    for required in _REQUIRED_INPUT_CHANNELS:
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


_GN_GROUPS = MODEL_GN_GROUPS  # GroupNorm group count; filters must be divisible by this


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
        canvas_realness: bool = False,
    ) -> None:
        super().__init__()
        # §169 A4 — under canvas_realness, the trunk-entry conv is a
        # PartialConv2d that masks off-canvas contributions and rescales
        # by the per-location valid-neighbour count. The downstream
        # blocks remain vanilla padded conv (Innamorati's intervention is
        # localised to the input layer; deeper layers see post-norm
        # features with off-canvas cells already at the GroupNorm bias).
        self.canvas_realness: bool = bool(canvas_realness)
        if self.canvas_realness:
            self.input_conv: nn.Module = PartialConv2d(
                in_channels, filters, 3, padding=1, bias=False,
            )
        else:
            self.input_conv = nn.Conv2d(
                in_channels, filters, 3, padding=1, bias=False,
            )
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
        if self.canvas_realness:
            if mask is None:
                raise RuntimeError(
                    "Trunk(canvas_realness=True) requires mask at forward; "
                    "ensure HexTacToeNet.forward computes the mask before "
                    "calling the trunk."
                )
            out = F.relu(self.input_gn(self.input_conv(x, mask)))
        else:
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

    @staticmethod
    def _validate_ctor_args(
        *,
        encoding: str,
        pool_type: str,
        canvas_realness: bool,
        gpool_bias_active: bool,
        gpool_indices: Optional[List[int]],
        policy_only_bias: bool,
        spec,
    ) -> None:
        """Validate incompatible flag combinations passed to ``__init__``.

        §176 P25 — extracted verbatim from ``HexTacToeNet.__init__``. Pure
        move: each raise block here is byte-identical to the pre-refactor
        inline form (error message strings are substring-pinned by tests).
        The ``encoding`` registry check and ``_spec`` resolution remain in
        ``__init__`` (this helper assumes ``spec`` is already resolved).
        """
        if pool_type not in SUPPORTED_POOL_TYPES:
            raise ValueError(
                f"pool_type={pool_type!r} must be one of {SUPPORTED_POOL_TYPES}"
            )
        # PMA / pma_global cluster pools replace the per-cluster value/policy
        # heads with a learned attention pool over the K cluster tokens. Only
        # meaningful for K-cluster encodings (v6 / v6w25). Under v8 (single
        # bbox) there is no K dimension to aggregate, so the PMA family is
        # gated off.
        if pool_type in ("pma", "pma_global") and not spec.has_pass_slot:
            raise ValueError(
                f"pool_type={pool_type!r} is only valid for v6/v6w25 K-cluster "
                "encodings; v8 has a single bounding-box window (no K)."
            )
        # §169 A4 — canvas_realness gates the partial-conv-padding trunk
        # entry. v8-only (the polarity inversion + partial conv only make
        # sense under bbox encoding); v6/v6w25 raise loudly.
        if canvas_realness and encoding != "v8":
            raise ValueError(
                f"canvas_realness=True requires encoding='v8'; got {encoding!r}. "
                "The §169 A4 arm pairs the inverted plane-8 polarity with "
                "PartialConv2d at trunk entry — both are v8/bbox-only."
            )
        # §170 P3 — gpool-bias side-branch is A1-only (K-cluster + min/max).
        # Additive K-invariant bias to value_fc1 hidden + policy_fc logits;
        # gate=0 init means byte-exact A1 at construction. Reject the cross
        # product with v8 / pma / canvas_realness / trunk gpool sites loudly
        # so YAML typos fail at construction, not silently in training.
        if gpool_bias_active:
            if canvas_realness:
                raise ValueError(
                    "gpool_bias_active=True is incompatible with "
                    "canvas_realness=True (canvas_realness is v8-only and "
                    "gpool_bias is K-cluster-only)."
                )
            if gpool_indices:
                raise ValueError(
                    "gpool_bias_active=True is incompatible with non-empty "
                    "gpool_indices. Trunk gpool sites are a different "
                    "intervention (in-trunk feature mixing); the gpool-bias "
                    "branch is the additive head-level analog."
                )
            if not spec.has_pass_slot:
                raise ValueError(
                    "gpool_bias_active=True is K-cluster-only (additive over "
                    "the K-cluster heads). v8 has a single bbox window and no "
                    "K dim — drop gpool_bias_active under encoding='v8'."
                )
            if pool_type != "min_max":
                raise ValueError(
                    f"gpool_bias_active=True requires pool_type='min_max'; "
                    f"got {pool_type!r}. The pma / pma_global pools already "
                    "carry a global-token branch (PMAGlobalPool); the "
                    "gpool-bias side-branch is the A1-only additive analog."
                )
        # §170 P4 — policy_only_bias requires gpool_bias_active=True; on its
        # own it has no consumer. Surface as a loud construction error so
        # YAML / CLI typos fail at load-time, not silently mid-training.
        if policy_only_bias and not gpool_bias_active:
            raise ValueError(
                "policy_only_bias=True requires gpool_bias_active=True; the "
                "policy-only knob configures the GpoolBiasBranch and has no "
                "effect without the branch being active."
            )

    def __init__(
        self,
        board_size: Optional[int] = None,
        in_channels: Optional[int] = None,
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
        canvas_realness: bool = False,
        gpool_bias_active: bool = False,
        policy_only_bias: bool = False,
    ) -> None:
        super().__init__()
        if encoding not in _VALID_ENCODINGS:
            raise ValueError(
                f"encoding={encoding!r} not in registry. "
                f"Registered: {sorted(_VALID_ENCODINGS)}"
            )
        # §176 P1 — cache the registry spec on the instance and as a local
        # alias for the ctor body. Routing decisions read attributes off
        # `spec` (e.g. `has_pass_slot`) instead of comparing the encoding
        # string. Avoids a per-forward registry lookup in the hot path.
        self._spec = lookup(encoding)
        spec = self._spec
        # §176 P26 — defaults for board_size / in_channels now resolve from
        # the registry spec rather than v6-only constants. HexTacToeNet()
        # still returns (19, 8) via v6 spec; HexTacToeNet(encoding="v6w25")
        # returns (25, 8). Explicit kwargs are honored unchanged.
        if board_size is None:
            board_size = spec.board_size
        if in_channels is None:
            in_channels = spec.n_planes
        # §176 P25 — flag-combination validation extracted to a private
        # helper. Pure-move: each raise block below is byte-identical to the
        # pre-refactor inline form (tests substring-pin the messages).
        self._validate_ctor_args(
            encoding=encoding,
            pool_type=pool_type,
            canvas_realness=canvas_realness,
            gpool_bias_active=gpool_bias_active,
            gpool_indices=gpool_indices,
            policy_only_bias=policy_only_bias,
            spec=spec,
        )
        self.gpool_bias_active: bool = bool(gpool_bias_active)
        self.policy_only_bias: bool = bool(policy_only_bias)
        self.canvas_realness: bool = bool(canvas_realness)
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
            if not spec.has_pass_slot:
                raise ValueError(
                    "input_channels is a v6-only knob (slices the 8-plane "
                    "wire format); v8 wire format is already a curated 11-plane "
                    "set — drop input_channels under encoding='v8'."
                )
            channels = _validate_input_channels(input_channels)
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

        # n_actions: v6 / v6w25 / v7-family keep the legacy pass slot
        # (HEXB shape); v8 / v8_canvas_realness drop it. Read straight off
        # the registry spec — single source of truth (§172).
        self.has_pass: bool = spec.has_pass_slot
        self.n_actions: int = spatial + (1 if self.has_pass else 0)
        self.off_window_plane_idx: int = int(off_window_plane_idx)

        # Trunk — gpool indices threaded through under v8; v6 keeps Sequential.
        self.trunk = Trunk(
            in_channels=self.in_channels,
            filters=filters,
            res_blocks=res_blocks,
            se_reduction_ratio=se_reduction_ratio,
            gpool_indices=gpool_indices if not spec.has_pass_slot else None,
            gpool_channels=gpool_channels,
            canvas_realness=self.canvas_realness,
        )

        # Policy / opp_reply heads — branch on encoding. v6 keeps the FC head
        # (loads existing v6 checkpoints byte-exact); v8 swaps in KataGoPolicyHead.
        if not spec.has_pass_slot:
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

        # Value uncertainty head (training only — never used in MCTS).
        # Reads from the same trunk features as the value head. Softplus
        # ensures positive output. §S181-AUDIT Wave 4 4B-impl-5: trained
        # with Huber-on-squared-error (compute_uncertainty_loss), so the
        # output is "predicted squared error of value" rather than variance.
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

        # K-cluster pool — §169 A2 / A3.
        #   - pool_type='min_max': pool is unused at the model level (the bot
        #     does scatter-max).
        #   - pool_type='pma': pool replaces the value/policy heads; forward()
        #     routes per-cluster GAP'd trunk features through SAB + PMA seeds.
        #     At pretrain time K=1 (one cluster window per training sample),
        #     so PMA's cross-cluster attention is exercised only via the
        #     duplicate-of-itself path; at inference time KClusterMCTSBot
        #     drives K>1 via aggregated_forward_K.
        #   - pool_type='pma_global': adds a global-summary token g (built by
        #     GlobalTokenEncoder from a 32×32 cur/opp/canvas-mask crop) as a
        #     (K+1)st SAB element, gated by a learnable scalar (init 0.1).
        # §172 A4.4 — audit confirmed: cluster_pool gate is `pool_type in
        # ("pma", "pma_global")`, encoding-blind; v6 and v6w25 both build it
        # when pool_type requires it (the K-cluster wire format is identical).
        if pool_type in ("pma", "pma_global"):
            extra_kwargs: dict = dict(
                n_heads=4,
                attn_dropout=pool_attn_dropout,
            )
            if pool_type == "pma_global":
                extra_kwargs["gate_init"] = 0.1
            self.cluster_pool: Optional[nn.Module] = build_pool(
                pool_type,
                dim=filters,
                n_actions=self.n_actions,
                **extra_kwargs,
            )
        else:
            self.cluster_pool = None

        # Global summary token branch — §169 A3 only. 3-channel 32×32 crop
        # (cur stones / opp stones / canvas-realness mask) → d=filters token
        # via 2 conv blocks + KataGo gpool + linear projection. Lives only
        # under pool_type='pma_global'; forward(global_crop=...) routes it.
        if pool_type == "pma_global":
            self.global_encoder: Optional[GlobalTokenEncoder] = GlobalTokenEncoder(
                in_channels=3,
                out_dim=filters,
            )
        else:
            self.global_encoder = None

        # §170 P3 — gpool-bias side-branch. Allocated only when explicitly
        # requested AND the (validated) A1 invariants hold (encoding=v6/v6w25,
        # pool_type=min_max, no canvas_realness, no trunk gpool sites). Gate
        # init=0.0 makes forward() output byte-exact A1 at construction.
        # §170 P4 — policy_only_bias forwards into the branch so the value
        # head's bias is structurally zero (state-dict shape is unchanged;
        # value_proj remains for round-trip with §170 P3 checkpoints).
        if self.gpool_bias_active:
            self.gpool_bias_branch: Optional[GpoolBiasBranch] = GpoolBiasBranch(
                filters=filters,
                n_actions=self.n_actions,
                value_hidden=256,
                policy_only=self.policy_only_bias,
            )
        else:
            self.gpool_bias_branch = None

    @property
    def tower(self) -> nn.Sequential:
        """Backward-compatible alias for the trunk tower."""
        return self.trunk.tower

    def gpool_bias_gate_value(self) -> Optional[float]:
        """Return the §170 P3 bias gate scalar, or None if branch inactive.

        For training-loop logging (mirrors the gate convention on
        ``PMAGlobalPool.gate_value()``).
        """
        if self.gpool_bias_branch is None:
            return None
        return self.gpool_bias_branch.gate_value()

    def forward(
        self,
        x: torch.Tensor,
        aux: bool = False,
        uncertainty: bool = False,
        ownership: bool = False,
        threat: bool = False,
        chain: bool = False,
        global_crop: Optional[torch.Tensor] = None,
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
        # gpool site (trunk + policy / opp_reply heads). Under canvas_realness
        # the plane-8 polarity is already (1 inside, 0 outside), so we read the
        # plane directly without the off→mask inversion.
        if not self._spec.has_pass_slot:
            if self.canvas_realness:
                mask = x[
                    :,
                    self.off_window_plane_idx:self.off_window_plane_idx + 1,
                    :,
                    :,
                ].to(x.dtype)
                mask_sum_hw = mask.sum(dim=(2, 3), keepdim=True)
            else:
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
        if self.pool_type in ("pma", "pma_global"):
            assert self.cluster_pool is not None
            cluster_tokens = out.mean(dim=(2, 3)).unsqueeze(1)   # (B, K=1, C)
            g_token: Optional[torch.Tensor] = None
            if self.pool_type == "pma_global":
                if global_crop is None:
                    raise ValueError(
                        "pool_type='pma_global' requires global_crop=...; got None. "
                        "Pass the (B, 3, 32, 32) global summary tensor through the "
                        "dataloader / inference path."
                    )
                assert self.global_encoder is not None
                g_token = self.global_encoder(global_crop)        # (B, dim)
            log_policy, value, v_logit = self.cluster_pool(
                cluster_tokens,
                per_cluster_logits=None,
                per_cluster_values=None,
                global_token=g_token,
            )
        else:
            # §170 P3 — encode the global crop ONCE per forward when the
            # bias branch is active. Same bias is broadcast to every cluster
            # window; the bot-side scatter-max-on-prob still produces the
            # canonical A1 output when gate=0 (additive 0).
            value_bias: Optional[torch.Tensor] = None
            policy_bias: Optional[torch.Tensor] = None
            if self.gpool_bias_active:
                if global_crop is None:
                    raise ValueError(
                        "gpool_bias_active=True requires global_crop=...; "
                        "got None. Pass the (B, 3, 32, 32) global summary "
                        "tensor through the dataloader / inference path."
                    )
                assert self.gpool_bias_branch is not None
                v_bias_raw, p_bias_raw = self.gpool_bias_branch(global_crop)
                # Broadcast (G_B, ...) bias to x's batch dim. G_B==1 expands;
                # G_B==B uses as-is; mismatches raise loudly.
                xb = x.size(0)
                gb = v_bias_raw.size(0)
                if gb == 1 and xb > 1:
                    v_bias_raw = v_bias_raw.expand(xb, -1)
                    p_bias_raw = p_bias_raw.expand(xb, -1)
                elif gb != xb:
                    raise ValueError(
                        f"global_crop batch={gb} disagrees with x batch={xb}; "
                        "expected 1 (broadcast) or matching."
                    )
                value_bias = v_bias_raw
                policy_bias = p_bias_raw

            # Policy + value heads — branch on encoding. has_pass_slot=true
            # encodings (v6 / v6w25 / v7full / v7 / v7e30 / v7mw) route through
            # the shared ``min_max_window_head`` helper (single-sourced with
            # ``aggregated_forward_K``'s per-cluster math — §176 P24).
            # has_pass_slot=false (v8 / v8_canvas_realness) keeps its
            # KataGoPolicyHead for the policy branch but reuses the same value
            # head (avg+max pool → fc1 → fc2 → tanh).
            if not self._spec.has_pass_slot:
                log_policy = self.policy_head(out, mask, mask_sum_hw)
                v_avg = out.mean(dim=(2, 3))           # (B, C)
                v_max = out.amax(dim=(2, 3))           # (B, C)
                v = torch.cat([v_avg, v_max], dim=1)   # (B, 2C)
                v = F.relu(self.value_fc1(v))
                if value_bias is not None:
                    v = v + value_bias.to(v.dtype)
                v_logit = self.value_fc2(v)            # (B, 1) raw logit
                value = torch.tanh(v_logit)
            else:
                log_policy, value, v_logit = min_max_window_head(
                    out,
                    policy_conv=self.policy_conv,
                    policy_fc=self.policy_fc,
                    value_fc1=self.value_fc1,
                    value_fc2=self.value_fc2,
                    policy_bias=policy_bias,
                    value_bias=value_bias,
                )

        # Build the base 3-tuple; optional heads are appended in order.
        extras: list = []

        if aux:
            if not self._spec.has_pass_slot:
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
        global_crop: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """K>1 inference entry point for K-cluster encodings.

        Args:
            x_K:        ``(K, in_channels, H, W)`` — the K cluster windows for
                        a single board. The model treats them as a single
                        sample with K cluster tokens; the cluster pool
                        aggregates them into one ``(log_policy, value,
                        v_logit)`` output.
            global_crop: ``(3, 32, 32)`` or ``(1, 3, 32, 32)`` global summary
                        crop. Required iff ``pool_type='pma_global'``.

        Returns the standard 3-tuple but with leading dim 1 (the aggregated
        sample). Callers (``KClusterMCTSBot``) reshape as needed.

        For ``pool_type='min_max'`` this method is a thin wrapper around the
        per-cluster forward followed by a model-internal min/max reduction,
        kept for interface symmetry — most callers stay on the existing
        bot-side scatter-max path.
        """
        if self._input_channels is not None:
            x_K = x_K.index_select(1, self.input_channel_index)

        if not self._spec.has_pass_slot:
            raise RuntimeError(
                "aggregated_forward_K is K-cluster only; v8 has no K dim."
            )

        out = self.trunk(x_K, mask=None, mask_sum_hw=None)        # (K, C, H, W)

        if self.pool_type in ("pma", "pma_global"):
            assert self.cluster_pool is not None
            cluster_tokens = out.mean(dim=(2, 3)).unsqueeze(0)    # (1, K, C)
            g_token: Optional[torch.Tensor] = None
            if self.pool_type == "pma_global":
                if global_crop is None:
                    raise ValueError(
                        "aggregated_forward_K needs global_crop=... when "
                        "pool_type='pma_global'. Pass the (3, 32, 32) crop "
                        "computed by hexo_rl.utils.global_crop."
                    )
                gc = global_crop
                if gc.dim() == 3:
                    gc = gc.unsqueeze(0)                          # (1, 3, 32, 32)
                if gc.size(0) != 1:
                    raise ValueError(
                        f"aggregated_forward_K expects a single board's global "
                        f"crop (batch=1); got batch={gc.size(0)}"
                    )
                assert self.global_encoder is not None
                g_token = self.global_encoder(gc)                 # (1, dim)
            log_policy, value, v_logit = self.cluster_pool(
                cluster_tokens,
                per_cluster_logits=None,
                per_cluster_values=None,
                global_token=g_token,
            )
            return log_policy, value, v_logit

        # min_max path — run per-cluster heads, reduce on the model side via
        # the same scatter-max-in-prob-space rule as the bot. Lifted into the
        # model so callers can switch pool_type without reaching into bot
        # internals.

        # §170 P3 — encode global crop ONCE for this single board, then
        # broadcast the (1, ...) biases to every cluster window before the
        # per-cluster log_softmax / value_fc2.
        value_bias_K: Optional[torch.Tensor] = None
        policy_bias_K: Optional[torch.Tensor] = None
        if self.gpool_bias_active:
            if global_crop is None:
                raise ValueError(
                    "aggregated_forward_K needs global_crop=... when "
                    "gpool_bias_active=True. Pass the (3, 32, 32) crop "
                    "computed by hexo_rl.utils.global_crop."
                )
            gc = global_crop
            if gc.dim() == 3:
                gc = gc.unsqueeze(0)                               # (1, 3, 32, 32)
            if gc.size(0) != 1:
                raise ValueError(
                    f"aggregated_forward_K expects a single board's global "
                    f"crop (batch=1); got batch={gc.size(0)}"
                )
            assert self.gpool_bias_branch is not None
            v_bias_raw, p_bias_raw = self.gpool_bias_branch(gc)
            # Broadcast (1, ...) → (K, ...) — same bias to every cluster.
            value_bias_K = v_bias_raw.expand(out.size(0), -1)
            policy_bias_K = p_bias_raw.expand(out.size(0), -1)

        # §176 P24 — per-cluster (K, *) head shares math with the
        # single-window forward via ``min_max_window_head``. The pooled output
        # is unsqueezed to ``(1, K, *)`` for ``MinMaxPool``'s scatter-pool
        # contract; bias broadcasting (1→K) happens above before the call.
        per_logp, per_val, _per_vlogit = min_max_window_head(
            out,
            policy_conv=self.policy_conv,
            policy_fc=self.policy_fc,
            value_fc1=self.value_fc1,
            value_fc2=self.value_fc2,
            policy_bias=policy_bias_K,
            value_bias=value_bias_K,
        )

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
