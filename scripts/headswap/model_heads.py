"""D-F HEADSWAP — model surface: frozen-trunk load + swappable value heads.

Binds to scripts/headswap/RECIPE.md §"Model surface", §"KEY TRAINING FACT",
§"Arms". The trunk is a run2-lineage HexTacToeNet (v6_live2_ls); the ORIGINAL
value_fc1/value_fc2 load from the checkpoint but are UNUSED — every arm routes
the 256-d value feature through a FRESH-INIT head (ScalarHead / BinHead).

INV-D1 (STANDING): value targets are game OUTCOME z only. This module holds no
target logic — it only builds the head that maps trunk features -> logits.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from hexo_rl.encoding import lookup
from hexo_rl.env.game_state import GameState
from hexo_rl.model.network import HexTacToeNet
from scripts.headswap.targets import VALUE_BINS

# ── trunk construction (SHA/lineage per RECIPE §"Trunk init") ────────────────

_TRUNK_KW = dict(
    board_size=19,
    in_channels=4,
    filters=128,
    res_blocks=12,
    encoding="v6_live2_ls",
    pool_type="min_max",
)

# The last trunk block that C/D unfreeze. tower is nn.Sequential(len 12); index
# 11 is the final ResidualBlock (verified).
LAST_BLOCK_IDX = 11

# Kept-plane indices for this trunk's encoding (v6_live2_ls -> [0, 8, 16, 17]).
# The 18-plane wire tensor from GameState.to_tensor() is sliced to these before
# the trunk — single-sourced from the registry, matching inference.py's slice.
KEPT_PLANE_INDICES: List[int] = list(lookup(_TRUNK_KW["encoding"]).kept_plane_indices)


def load_trunk(ckpt_path: str, device: torch.device) -> HexTacToeNet:
    """Build a v6_live2_ls HexTacToeNet and load the run2 trunk (strict=False).

    weights_only=False (the checkpoint carries config/optimizer state). The
    value_fc1/value_fc2 in the checkpoint load into the base model but are NOT
    used by the swappable head; the arm reads trunk features via value_feature().
    """
    model = HexTacToeNet(**_TRUNK_KW)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The base value head IS in the checkpoint and IS in the model (same names),
    # so it loads. The only "missing" keys should be aux heads absent from the
    # ckpt (never); "unexpected" should be empty. Surfacing kept for the test.
    model._headswap_missing = list(missing)  # type: ignore[attr-defined]
    model._headswap_unexpected = list(unexpected)  # type: ignore[attr-defined]
    model.to(device)
    return model


# ── swappable value heads (fresh-init both layers) ───────────────────────────


class ScalarHead(nn.Module):
    """256 -> relu(fc1 256->256) -> fc2 256->1. Fresh-init; arm A / D."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        v = torch.relu(self.fc1(feat))
        return self.fc2(v)  # (B, 1) raw pre-tanh logit -> compute_value_loss


class BinHead(nn.Module):
    """256 -> relu(fc1 256->256) -> fc2 256->65. Fresh-init; arm B / C."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, VALUE_BINS)  # 65

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        v = torch.relu(self.fc1(feat))
        return self.fc2(v)  # (B, 65) logits -> two_hot_ce_loss


def build_head(arm: str) -> nn.Module:
    """ScalarHead for A/D (scalar shape); BinHead for B/C (65-bin shape)."""
    if arm in ("A", "D"):
        return ScalarHead()
    if arm in ("B", "C"):
        return BinHead()
    raise ValueError(f"unknown arm {arm!r}; expected one of A/B/C/D")


# ── value feature (replicate production value branch, NO min_max_window_head) ──


def value_feature(model: HexTacToeNet, states: torch.Tensor) -> torch.Tensor:
    """Trunk -> (B, 128, 19, 19) -> cat([mean, amax]) -> (B, 256).

    Replicates network.py:808-810's has_pass value branch exactly (mean+amax
    over spatial dims), which is the SAME feature the production value_fc1
    consumes. mask=None for v6_live2_ls (has_pass_slot=True, no gpool sites).
    """
    out = model.trunk(states)  # (B, 128, 19, 19)
    return torch.cat([out.mean(dim=(2, 3)), out.amax(dim=(2, 3))], dim=1)  # (B, 256)


def board_to_cluster_tensor(model: HexTacToeNet, board) -> Tuple[torch.Tensor, list]:
    """Board -> (K, in_channels, 19, 19) float tensor (on model device) + centers.

    The shared front-half of the multi-window inference path: GameState.to_tensor
    yields the 18-plane wire tensor for K legal-set cluster windows; slice it to
    the model's kept planes (registry-driven, identical to
    ``LocalInferenceEngine.infer_batch``) and move to the model's device as float.
    Kept LOCAL to the headswap harness (see module note) — production callers use
    their own hot-path variant.
    """
    state = GameState.from_board(board)
    tensor, centers = state.to_tensor()          # (K, 18, 19, 19), K centers
    if tensor.shape[1] != model.in_channels:
        tensor = tensor[:, KEPT_PLANE_INDICES]   # (K, in_channels, 19, 19)
    x = torch.from_numpy(tensor).to(next(model.parameters()).device).float()
    return x, centers


# ── freeze / param groups per arm ────────────────────────────────────────────


def freeze_for_arm(
    model: HexTacToeNet,
    head: nn.Module,
    arm: str,
    head_lr: float,
) -> List[dict]:
    """Freeze the trunk; unfreeze the head (all arms) and tower[11] (C/D only).

    Returns optimizer param groups: head at head_lr; tower[11] at 0.1*head_lr
    for C/D. Only requires_grad params are handed to the optimizer.
    """
    if arm not in ("A", "B", "C", "D"):
        raise ValueError(f"unknown arm {arm!r}")

    # Base model fully frozen.
    for p in model.parameters():
        p.requires_grad_(False)
    # Head fully trainable.
    for p in head.parameters():
        p.requires_grad_(True)

    param_groups: List[dict] = [
        {"params": [p for p in head.parameters() if p.requires_grad], "lr": head_lr}
    ]

    if arm in ("C", "D"):
        last_block = model.trunk.tower[LAST_BLOCK_IDX]
        for p in last_block.parameters():
            p.requires_grad_(True)
        param_groups.append(
            {
                "params": [p for p in last_block.parameters() if p.requires_grad],
                "lr": 0.1 * head_lr,
            }
        )

    return param_groups
