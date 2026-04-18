"""C-004 regression — --steps N must be a hard cap on the trainer step counter.

Before the fix, `trainer.step` was initialised to `-total_pretrain_steps` and
the termination check compared `self.step >= step_budget`, so `--steps N`
silently ran 2*N iterations (overrun to `self.step == N`).
"""
from __future__ import annotations

from pathlib import Path

import torch

from hexo_rl.bootstrap.pretrain import BootstrapTrainer
from hexo_rl.model.network import HexTacToeNet


def test_steps_flag_is_hard_cap(tmp_path: Path) -> None:
    budget = 10
    model = HexTacToeNet(
        board_size=19, in_channels=18, filters=8, res_blocks=1,
        se_reduction_ratio=4,
    )
    config = {"lr": 0.01, "pretrain_total_steps": budget, "batch_size": 2}
    trainer = BootstrapTrainer(model, config, torch.device("cpu"), tmp_path)

    # Reproduce the main() initialisation exactly.
    trainer.step = -budget
    start_step = trainer.step

    # 4-tuple loader matching BootstrapTrainer.train_epoch's expected layout.
    states       = torch.zeros(2, 18, 19, 19, dtype=torch.float32)
    chain_planes = torch.zeros(2, 6, 19, 19, dtype=torch.float16)
    policies     = torch.zeros(2, 19 * 19 + 1, dtype=torch.float32)
    policies[:, 0] = 1.0
    outcomes = torch.zeros(2, dtype=torch.float32)

    # Loader yields *way* more batches than the budget so the cap is the only
    # thing that can terminate the loop.
    loader = [(states, chain_planes, policies, outcomes)] * (budget * 3)

    trainer.train_epoch(
        loader,
        step_budget=budget,
        start_step=start_step,
        log_interval=0,
    )

    # Exactly `budget` iterations must have happened — no more, no less.
    assert trainer.step - start_step == budget, (
        f"expected exactly {budget} iterations, got {trainer.step - start_step} "
        f"(trainer.step={trainer.step}, start_step={start_step})"
    )
