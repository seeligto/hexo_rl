"""Shared loss computation for Trainer and BootstrapTrainer.

Architecture spec (docs/01_architecture.md §2):
    L = L_policy + L_value + w_aux · L_opp_reply + w_unc · L_uncertainty
    L_policy     = -sum(π_target · log π_net)   (cross-entropy, masked)
    L_value      = BCE(sigmoid(v_logit), (z+1)/2)
    L_opp_reply  = -sum(π_target · log π_opp_net)  (auxiliary)
    L_uncertainty = 0.5 * (log σ² + (z - value_detached)² / σ²)  (Gaussian NLL)
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
        valid_targets = target_policy[valid_mask]   # [N, A]
        valid_logits  = aux_logit[valid_mask]        # [N, A]  (log-softmax)
        safe_log = valid_logits.clamp(min=-100.0)    # -inf → 0 contribution; exp(-100)≈0
        return -(valid_targets * safe_log).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


def compute_uncertainty_loss(
    sigma2: torch.Tensor,
    z_targets: torch.Tensor,
    value_detached: torch.Tensor,
) -> torch.Tensor:
    """Gaussian NLL loss for the value uncertainty head.

    Gradient flows only through σ² — ``value_detached`` must be .detach()'d by
    the caller so that this head does not influence the value head's gradients.

    Args:
        sigma2:         (B, 1) predicted variance from value_var head (> 0).
        z_targets:      (B,) game outcomes in {-1, 0, +1}.
        value_detached: (B, 1) value head output with gradients stopped.

    Returns:
        Scalar mean Gaussian NLL.
    """
    z = z_targets.unsqueeze(1)          # (B, 1)
    return 0.5 * (sigma2.log() + (z - value_detached) ** 2 / sigma2).mean()


def compute_total_loss(
    policy_loss: torch.Tensor,
    value_loss: torch.Tensor,
    aux_loss: Optional[torch.Tensor] = None,
    aux_weight: float = 0.0,
    entropy_bonus: Optional[torch.Tensor] = None,
    entropy_weight: float = 0.0,
    uncertainty_loss: Optional[torch.Tensor] = None,
    uncertainty_weight: float = 0.0,
    ownership_loss: Optional[torch.Tensor] = None,
    ownership_weight: float = 0.0,
    threat_loss: Optional[torch.Tensor] = None,
    threat_weight: float = 0.0,
) -> torch.Tensor:
    """Combine policy, value, auxiliary, entropy, uncertainty, ownership, and threat losses."""
    total = policy_loss + value_loss
    if aux_loss is not None and aux_weight > 0.0:
        total = total + aux_weight * aux_loss
    if entropy_bonus is not None and entropy_weight > 0.0:
        total = total - entropy_weight * entropy_bonus
    if uncertainty_loss is not None and uncertainty_weight > 0.0:
        total = total + uncertainty_weight * uncertainty_loss
    if ownership_loss is not None and ownership_weight > 0.0:
        total = total + ownership_weight * ownership_loss
    if threat_loss is not None and threat_weight > 0.0:
        total = total + threat_weight * threat_loss
    return total


def fp16_backward_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    model: nn.Module,
    fp16: bool,
    max_grad_norm: float = 1.0,
) -> float:
    """Backward pass with optional FP16 gradient scaling and clipping.

    Returns the pre-clip gradient norm (the diagnostic signal).
    """
    if fp16:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
        optimizer.step()
    return grad_norm
