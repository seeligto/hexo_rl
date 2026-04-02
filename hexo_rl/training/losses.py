"""Shared loss computation for Trainer and BootstrapTrainer.

Architecture spec (docs/01_architecture.md §2):
    L = L_policy + L_value + w_aux · L_opp_reply
    L_policy     = -sum(π_target · log π_net)   (cross-entropy, masked)
    L_value      = BCE(sigmoid(v_logit), (z+1)/2)
    L_opp_reply  = -sum(π_target · log π_opp_net)  (auxiliary)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LossResult:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    aux: Optional[torch.Tensor] = None


def compute_policy_loss(
    log_policy: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Cross-entropy policy loss, masked on zero-policy rows."""
    if valid_mask.any():
        return -(target_policy[valid_mask] * log_policy[valid_mask]).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


def compute_value_loss(
    value_logit: torch.Tensor,
    outcome: torch.Tensor,
) -> torch.Tensor:
    """BCE value loss on pre-tanh logit. Outcomes mapped {-1,+1} -> {0,1}."""
    value_target = (outcome + 1.0) / 2.0
    return nn.functional.binary_cross_entropy_with_logits(
        value_logit.squeeze(1), value_target
    )


def compute_aux_loss(
    aux_logit: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Opponent reply auxiliary loss (same structure as policy loss)."""
    if valid_mask.any():
        return -(target_policy[valid_mask] * aux_logit[valid_mask]).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


def compute_total_loss(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    aux_loss: Optional[torch.Tensor] = None,
    aux_weight: float = 0.0,
) -> torch.Tensor:
    """Combine policy, value, and optional auxiliary losses."""
    total = policy_loss + value_loss
    if aux_loss is not None and aux_weight > 0.0:
        total = total + aux_weight * aux_loss
    return total


def fp16_backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model: nn.Module,
    fp16: bool,
    max_grad_norm: float = 1.0,
) -> None:
    """Backward pass with optional FP16 gradient scaling and clipping."""
    if fp16:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
