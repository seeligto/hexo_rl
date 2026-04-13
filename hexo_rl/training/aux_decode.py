"""Auxiliary target decoding: u8 buffer columns → float32 training tensors.

Canonical home for ownership + winning_line decode so ``trainer.py``
stays focused on forward / backward / optim / scheduler / save / load.
"""

from __future__ import annotations

import numpy as np
import torch


def decode_ownership(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Decode u8 ownership {0=P2, 1=empty, 2=P1} → float32 {-1, 0, +1}.

    Ships u8 to device first (4× smaller H2D transfer vs fp32) then converts
    and shifts in-place on the GPU.

    Args:
        arr:    Uint8 array of shape (B, 19, 19) or (B, 361).
        device: Target torch device.

    Returns:
        Float32 tensor of shape (B, 19, 19) on ``device``.
    """
    c = np.ascontiguousarray(arr)
    if c.ndim == 2:
        c = c.reshape(-1, 19, 19)
    return torch.from_numpy(c).to(device, non_blocking=True).float().sub_(1.0)


def decode_winning_line(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Decode u8 winning_line → float32 {0.0, 1.0}, moved to ``device``.

    Args:
        arr:    Uint8 array of shape (B, 19, 19) or (B, 361).
        device: Target torch device.

    Returns:
        Float32 tensor of shape (B, 19, 19) on ``device``.
    """
    c = np.ascontiguousarray(arr)
    if c.ndim == 2:
        c = c.reshape(-1, 19, 19)
    return torch.from_numpy(c).to(device, non_blocking=True).float()


def mask_aux_rows(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_pretrain: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice ``[n_pretrain:]`` to exclude corpus rows from aux loss.

    Corpus rows carry dummy aux (ownership=1→0.0, winning_line=0) that must
    not contribute to spatial head losses.  When n_pretrain==0, returns the
    tensors unchanged (no copy).

    Args:
        pred:       Model prediction tensor (B, …).
        target:     Ground-truth target tensor (B, …).
        n_pretrain: Number of corpus rows at the *start* of the batch.

    Returns:
        (pred[n_pretrain:], target[n_pretrain:]) slice pair.
    """
    if n_pretrain == 0:
        return pred, target
    return pred[n_pretrain:], target[n_pretrain:]
