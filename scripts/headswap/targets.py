"""Two-hot 65-bin distributional value target + scoring primitives.

D-F HEADSWAP (run3 card #1 discriminator). Semantics pinned by
docs/designs/run3_d1_distributional_head.md §D1 (verified from shrimp source:
packages/shrimp/python/shrimp/{constants,losses}.py):

  VALUE_BINS = 65
  support    = linspace(-1, 1, 65)         # scalar OUTCOME support, NOT moves-to-end
  bin width  = 2/64 = 1/32 = 0.03125
  two-hot    = scalar_to_binned_target: pos = (z+1)*32; mass split linearly
               between floor(pos) and floor(pos)+1  (MuZero/KataGo style)
  decode     = E[softmax . support], clamped to [-1, 1]

INV-D1 (STANDING): the target derives ONLY from the game outcome z. No teacher
net, no SealBot/solver score, no TD bootstrap anywhere in the loss path. This
module takes a raw scalar z in [-1, 1] and nothing else.

fp32 end-to-end for the two-hot placement (shrimp losses.py:166-171: fp16
mis-splits adjacent bins by ~3% of a bin).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

VALUE_BINS = 65
BIN_WIDTH = 2.0 / (VALUE_BINS - 1)  # 0.03125 == 1/32

# Loss-tail readout threshold P(v <= -0.5). support[16] == -0.5 EXACTLY
# (-1 + 16/32 = -0.5; 1/32 is a power of two so this is exact in fp32).
LOSS_TAIL_THRESHOLD = -0.5
LOSS_TAIL_BIN = 16  # inclusive upper bin index for the tail sum


def support(device=None, dtype=torch.float32) -> torch.Tensor:
    """The 65-atom scalar support, linspace(-1, 1, 65)."""
    return torch.linspace(-1.0, 1.0, VALUE_BINS, device=device, dtype=dtype)


def scalar_to_two_hot(z: torch.Tensor) -> torch.Tensor:
    """z (B,) in [-1, 1] -> two-hot soft target (B, 65), fp32, rows sum to 1.

    pos = (z+1)*32 in [0, 64]; mass 1-frac at floor(pos), frac at floor(pos)+1.
    z = +1 lands entirely in bin 64; z = -1 entirely in bin 0.
    """
    z = z.detach().to(torch.float32).clamp(-1.0, 1.0).reshape(-1)
    b = z.shape[0]
    pos = (z + 1.0) * 32.0  # [0, 64]
    lo = pos.floor().long().clamp(0, VALUE_BINS - 1)
    hi = (lo + 1).clamp(max=VALUE_BINS - 1)
    frac = (pos - lo.to(torch.float32))  # [0, 1); exactly 0 at pos == 64
    target = torch.zeros(b, VALUE_BINS, dtype=torch.float32, device=z.device)
    target.scatter_(1, lo.unsqueeze(1), (1.0 - frac).unsqueeze(1))
    target.scatter_add_(1, hi.unsqueeze(1), frac.unsqueeze(1))
    return target


def two_hot_ce_loss(
    logits65: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy of head softmax against two-hot(z), masked by
    value_target_valid. Mirrors the reduction of production compute_value_loss
    (per-row, index by mask, .mean(); empty-mask -> differentiable zero)."""
    target = scalar_to_two_hot(z)  # (B, 65) fp32
    logp = F.log_softmax(logits65.to(torch.float32), dim=1)
    ce = -(target * logp).sum(dim=1)  # (B,)
    if mask is not None:
        m = mask.reshape(-1).bool()
        if int(m.sum()) == 0:
            return logits65.sum() * 0.0
        return ce[m].mean()
    return ce.mean()


def decode_expected_value(logits65: torch.Tensor) -> torch.Tensor:
    """Deploy/eval decode: E[softmax . support], clamped [-1, 1]. (B,) -> (B,).

    This is the FAIR comparate to the scalar head's tanh value: same summary
    (a point estimate in [-1, 1]). Used as the design-doc-primary AUC score.
    """
    probs = F.softmax(logits65.to(torch.float32), dim=1)
    v = probs @ support(device=logits65.device)
    return v.clamp(-1.0, 1.0)


def loss_tail_mass(logits65: torch.Tensor) -> torch.Tensor:
    """P(v <= -0.5) = sum of softmax mass over bins 0..16 inclusive. (B,) -> (B,).

    The distributional-specific readout the run3 card predicts. A HIGHER value
    means MORE pessimism, so as an AUC score for lost(=positive)-vs-safe it is
    used directly (higher score -> more likely lost). Pre-committed threshold;
    monotone-transform invariance and threshold sensitivity are red-team checks.
    """
    probs = F.softmax(logits65.to(torch.float32), dim=1)
    return probs[:, : LOSS_TAIL_BIN + 1].sum(dim=1)
