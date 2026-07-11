"""65-bin distributional value primitives (INV-D1: outcome-z only, fp32).

Ported and hardened from scripts/headswap/targets.py so the production trainer
never imports from scripts/. Support is a scalar outcome support over [-1, 1];
NOT moves-to-end / margin / discounted return. fp16 two-hot mis-splits adjacent
bins by ~3% of a bin — keep targets fp32.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

N_VALUE_BINS = 65
VALUE_SUPPORT = torch.linspace(-1.0, 1.0, N_VALUE_BINS)   # (65,), fp32


def scalar_to_two_hot(z: torch.Tensor, n_bins: int = N_VALUE_BINS) -> torch.Tensor:
    """z (N,) in [-1,1] → (N, n_bins) fp32 two-hot (MuZero/KataGo style)."""
    z = z.reshape(-1).detach().to(torch.float32).clamp(-1.0, 1.0)
    scale = (n_bins - 1) / 2.0                      # 32 for 65 bins
    pos = (z + 1.0) * scale                         # [0, n_bins-1]
    lo = torch.floor(pos).to(torch.long).clamp(0, n_bins - 1)
    hi = (lo + 1).clamp(0, n_bins - 1)
    frac = (pos - lo.to(pos.dtype))
    out = torch.zeros(z.shape[0], n_bins, dtype=torch.float32, device=z.device)
    out.scatter_(1, lo.unsqueeze(1), (1.0 - frac).unsqueeze(1))
    out.scatter_add_(1, hi.unsqueeze(1), frac.unsqueeze(1))
    return out


def decode_binned_value(bin_logits: torch.Tensor) -> torch.Tensor:
    """(N, n_bins) logits → (N, 1) E[softmax·support], clamped [-1,1]."""
    probs = F.softmax(bin_logits.to(torch.float32), dim=-1)   # fp32 end-to-end (INV-D1): search-side decode must not run in autocast fp16
    support = VALUE_SUPPORT.to(bin_logits.device, torch.float32)
    v = (probs * support).sum(dim=-1, keepdim=True)
    return v.clamp(-1.0, 1.0)


def binned_value_loss(
    bin_logits: torch.Tensor,
    outcome: torch.Tensor,
    value_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked cross-entropy of head softmax vs two-hot(outcome). fp32 targets.

    Mask semantics identical to compute_value_loss (losses.py:75-102):
    value_mask==0 rows excluded from numerator AND denominator; empty→zeros(())."""
    target = scalar_to_two_hot(outcome.reshape(-1))          # (N, 65) fp32
    logp = F.log_softmax(bin_logits.to(torch.float32), dim=-1)
    per_row = -(target * logp).sum(dim=-1)                   # (N,)
    if value_mask is None:
        return per_row.mean()
    mask = value_mask.reshape(-1).bool()
    kept = per_row[mask]
    if kept.numel() == 0:
        return torch.zeros((), device=per_row.device, dtype=per_row.dtype)
    return kept.mean()
