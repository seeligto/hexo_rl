"""§S181 FU-2 — LR warmup ramp test.

Verifies that `lr_warmup_steps` wraps the cosine scheduler with a
LinearLR warmup via SequentialLR, ramping from `lr * start_factor` to
`lr` over the requested step count, then handing off to CosineAnnealing
for the remainder.

Motivation: A2 sustained-from-anchor needs a warmup transient or the
flat-policy → random-selfplay → zero-value-target loop pins value loss
at ln(2). See audit/structural/07_fu2_a2_pretrain_quality.md §6.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hexo_rl.training.trainer import Trainer


def _toy_trainer(config: dict) -> Trainer:
    """Minimal Trainer for scheduler-only tests (no engine/board needed)."""
    model = nn.Linear(8, 8)
    full_config = {
        "lr": 2e-3,
        "lr_schedule": "cosine",
        "total_steps": 100,
        "eta_min": 2e-4,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "aux_opp_reply_weight": 0.0,
        "value_loss_weight": 1.0,
        "torch_compile_enabled": False,
        "torch_compile_inf_enabled": False,
        "amp_enabled": False,
        **config,
    }
    # Trainer requires a model + config; bypass full init by directly
    # building optimizer + scheduler the way Trainer does.
    from torch.optim import AdamW
    t = Trainer.__new__(Trainer)
    t.optimizer = AdamW(model.parameters(), lr=full_config["lr"])
    t.scheduler = t._build_scheduler(full_config)
    return t


def test_no_warmup_falls_back_to_plain_cosine():
    """lr_warmup_steps=0 (default) → plain CosineAnnealingLR (no SequentialLR)."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

    t = _toy_trainer({"lr_warmup_steps": 0})
    assert isinstance(t.scheduler, CosineAnnealingLR)
    assert not isinstance(t.scheduler, SequentialLR)


def test_warmup_wraps_in_sequential_lr():
    """lr_warmup_steps>0 → SequentialLR(LinearLR, CosineAnnealingLR)."""
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    t = _toy_trainer({"lr_warmup_steps": 10})
    assert isinstance(t.scheduler, SequentialLR)
    assert len(t.scheduler._schedulers) == 2
    assert isinstance(t.scheduler._schedulers[0], LinearLR)
    assert isinstance(t.scheduler._schedulers[1], CosineAnnealingLR)
    # cosine T_max = total_steps - warmup_steps
    assert t.scheduler._schedulers[1].T_max == 100 - 10


def test_warmup_ramps_from_start_factor_to_full_lr():
    """LR at step 0 = lr*start_factor; LR at end of warmup ≈ lr."""
    t = _toy_trainer({"lr_warmup_steps": 10, "lr_warmup_start_factor": 0.01})
    # step 0 lr (PyTorch reports the lr BEFORE the first step() call).
    initial_lr = t.optimizer.param_groups[0]["lr"]
    assert abs(initial_lr - 2e-3 * 0.01) < 1e-9, (
        f"step 0 lr={initial_lr} expected {2e-3 * 0.01}"
    )
    # Step through warmup; lr should rise linearly.
    for _ in range(10):
        t.optimizer.step()
        t.scheduler.step()
    final_warmup_lr = t.optimizer.param_groups[0]["lr"]
    # After 10 warmup steps + handoff, lr should be near the peak 2e-3.
    # Cosine starts AT 2e-3 and immediately begins decaying; allow ~5% tolerance.
    assert 0.95 * 2e-3 < final_warmup_lr <= 2e-3, (
        f"end-of-warmup lr={final_warmup_lr} expected ≈ {2e-3}"
    )


def test_warmup_then_cosine_decays_toward_eta_min():
    """Run the full schedule; final lr is at/near eta_min."""
    t = _toy_trainer(
        {"lr_warmup_steps": 10, "lr_warmup_start_factor": 0.01,
         "total_steps": 100, "eta_min": 2e-4},
    )
    for _ in range(100):
        t.optimizer.step()
        t.scheduler.step()
    final_lr = t.optimizer.param_groups[0]["lr"]
    assert abs(final_lr - 2e-4) < 1e-5, (
        f"final lr={final_lr} expected ≈ eta_min=2e-4"
    )


def test_start_factor_default_is_001():
    """Default lr_warmup_start_factor=0.01 when key missing."""
    t = _toy_trainer({"lr_warmup_steps": 10})
    assert abs(t.optimizer.param_groups[0]["lr"] - 2e-3 * 0.01) < 1e-9
