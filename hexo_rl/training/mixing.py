"""Pretrain/selfplay mixing helpers (§176 P13).

Pure functions for AlphaZero-style pretrain weight decay and per-round
selfplay → training step budgeting. Extracted from training/loop.py for
reuse by step_coordinator + tests without dragging the full loop module.
"""
from __future__ import annotations

import math


def _compute_pretrained_weight(step: int, initial_w: float, min_w: float, decay_steps: float) -> float:
    return max(min_w, initial_w * math.exp(-step / decay_steps))


def _steps_budget(new_games: int, training_steps_per_game: float, max_train_burst: int) -> int:
    return min(max(1, round(new_games * training_steps_per_game)), max_train_burst)
