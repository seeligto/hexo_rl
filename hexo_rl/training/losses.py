"""Shared loss computation for Trainer and BootstrapTrainer.

Architecture spec (docs/01_architecture.md §2):
    L = L_policy + L_value + w_aux · L_opp_reply + w_unc · L_uncertainty
    L_policy     = -sum(π_target · log π_net)   (cross-entropy, masked)
    L_value      = binary_cross_entropy_with_logits(v_logit, (z+1)/2)
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
    full_search_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy policy loss, masked on zero-policy rows and quick-search positions.

    Args:
        log_policy:        (B, A) log-softmax policy from model.
        target_policy:     (B, A) MCTS visit-count targets.
        valid_mask:        (B,) bool — excludes zero-policy rows (fast-game zeroes).
        device:            Target device for the zero scalar.
        full_search_mask:  Optional (B,) bool/uint8 — 1 = full-search position (apply
                           policy loss), 0 = quick-search (skip). When None, all valid
                           positions contribute (legacy behaviour, full_search_prob=0.0).
    """
    combined = valid_mask
    if full_search_mask is not None:
        combined = valid_mask & full_search_mask.bool()
    if combined.any():
        return -(target_policy[combined] * log_policy[combined]).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


def compute_kl_policy_loss(
    log_policy: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    full_search_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL(target || model) policy loss for completed Q-value targets.

    KL and CE produce identical gradients (differ by the constant entropy of
    the target distribution). KL gives more interpretable loss values: 0 when
    the model perfectly matches the target.

    Args:
        full_search_mask:  Optional (B,) bool/uint8 — gates policy loss same as
                           ``compute_policy_loss``.
    """
    combined = valid_mask
    if full_search_mask is not None:
        combined = valid_mask & full_search_mask.bool()
    if combined.any():
        tgt = target_policy[combined]           # (N, A)
        log_model = log_policy[combined]         # (N, A)
        # FP16 safety: clamp log(target) to prevent -inf propagation.
        # Same pattern as compute_aux_loss (§47 fix).
        log_tgt = torch.log(tgt.clamp(min=1e-8)).clamp(min=-100.0)
        return (tgt * (log_tgt - log_model)).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


def compute_value_loss(
    value_logit: torch.Tensor,
    outcome: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable BCE via binary_cross_entropy_with_logits. Outcomes mapped {-1,+1} -> {0,1}."""
    value_target = (outcome + 1.0) / 2.0
    return nn.functional.binary_cross_entropy_with_logits(
        value_logit.squeeze(1), value_target
    )


def compute_aux_loss(
    aux_logit: torch.Tensor,
    target_policy: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
    full_search_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Opponent reply auxiliary loss (same structure as policy loss).

    opp_reply is a policy-like head trained on the same noisy MCTS visit targets
    that drive ``compute_policy_loss``. Gated with ``full_search_mask`` for the
    same reason: quick-search positions provide low-quality policy-shaped targets.
    """
    combined = valid_mask
    if full_search_mask is not None:
        combined = valid_mask & full_search_mask.bool()
    if combined.any():
        valid_targets = target_policy[combined]   # [N, A]
        valid_logits  = aux_logit[combined]        # [N, A]  (log-softmax)
        safe_log = valid_logits.clamp(min=-100.0)    # -inf → 0 contribution; exp(-100)≈0
        return -(valid_targets * safe_log).sum(dim=1).mean()
    return torch.zeros(1, device=device, dtype=torch.float32).squeeze()


def compute_chain_loss(
    chain_pred: torch.Tensor,
    chain_target: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Q13-aux smooth-L1 (Huber) loss on 6 chain-length planes.

    Args:
        chain_pred:   (B, 6, H, W) raw regression outputs from chain_head.
        chain_target: (B, 6, H, W) target chain planes. Passed as a separate
                      buffer (computed from game positions) — chain planes are
                      no longer part of the 18-channel input tensor.
        legal_mask:   Optional float mask broadcastable to (B, 6, H, W) with
                      1.0 where the cell is on the board and its chain value
                      is meaningful, 0.0 where the loss should be ignored.
                      Shape (B, 1, H, W) is fine — it will broadcast across the
                      6 chain planes. When `None`, every cell contributes
                      equally (reduction="mean"). W2 from the Q13 review: the
                      original brief described a masked loss; the pre-C13
                      implementation was unconditional mean.
        huber_delta:  Transition point between L1 and L2 regions of Huber loss.
                      Defaults to 1.0, matching torch's default smooth_l1_loss.

    Returns:
        Scalar mean loss over the masked cells (or all cells if `legal_mask`
        is None).
    """
    # Targets live in [0, 1] after /6.0 normalization, so values are well
    # within the Huber L2 region for typical predictions — behaves like MSE
    # near small errors and L1 for outliers.
    if legal_mask is None:
        return torch.nn.functional.smooth_l1_loss(
            chain_pred.float(),
            chain_target.float(),
            beta=huber_delta,
            reduction="mean",
        )

    per_cell = torch.nn.functional.smooth_l1_loss(
        chain_pred.float(),
        chain_target.float(),
        beta=huber_delta,
        reduction="none",
    )
    mask = legal_mask.float()
    # Normalise (B, H, W) → (B, 1, H, W) before expand so all three shapes
    # ((B,H,W), (B,1,H,W), (B,6,H,W)) broadcast identically to per_cell.
    if mask.dim() == per_cell.dim() - 1:
        mask = mask.unsqueeze(1)
    mask_b = mask.expand_as(per_cell)
    return (per_cell * mask_b).sum() / mask_b.sum().clamp_min(1.0)


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
    # Promote to FP32 and clamp before log/divide to prevent FP16 subnormal
    # blow-up (sigma2 near 6e-8 → inf in division → NaN in loss).
    # clamp(min=1e-6): log(1e-6) ≈ -13.8, well within representable range;
    # σ ≈ 0.001 is a physically meaningful minimum uncertainty.
    sigma2_fp32 = sigma2.float().clamp(min=1e-6)
    log_sigma2_fp32 = torch.log(sigma2_fp32)
    z_fp32 = z.float()
    v_det_fp32 = value_detached.float()
    return 0.5 * (log_sigma2_fp32 + (z_fp32 - v_det_fp32).pow(2) / sigma2_fp32).mean()


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
    chain_loss: Optional[torch.Tensor] = None,
    chain_weight: float = 0.0,
) -> torch.Tensor:
    """Combine policy, value, auxiliary, entropy, uncertainty, ownership, threat, and chain losses."""
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
    if chain_loss is not None and chain_weight > 0.0:
        total = total + chain_weight * chain_loss
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
