"""Runtime structural-compatibility checks for StepCoordinator protocols (§159a M2)."""
from __future__ import annotations

import time
import tracemalloc
from typing import Any

import pytest
import torch

from engine import ReplayBuffer
from hexo_rl.eval.eval_pipeline import EvalPipeline
from hexo_rl.eval.pipeline_setup import build_eval_pipeline
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.pool import WorkerPool
from hexo_rl.training.step_coordinator import (
    ClockLike,
    EvalPipelineLike,
    GpuMonitorLike,
    RealClock,
    RealTracemalloc,
    ReplayBufferLike,
    StepCoordinatorConfig,
    StepOutcome,
    TracemallocLike,
    TrainerLike,
    WorkerPoolLike,
)
from hexo_rl.training.trainer import Trainer


def test_real_pool_satisfies_worker_pool_like():
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32).to(device)
    buf = ReplayBuffer(capacity=10)
    config = {
        "n_workers": 1,
        "playout_cap": {"fast_sims": 50, "standard_sims": 400},
        "mcts": {"interior_selector": "puct", "n_simulations": 50},
    }
    pool = WorkerPool(model, config, device, buf)
    assert isinstance(pool, WorkerPoolLike)


def test_real_buffer_satisfies_replay_buffer_like():
    buf = ReplayBuffer(capacity=10)
    assert isinstance(buf, ReplayBufferLike)


def test_real_trainer_satisfies_trainer_like():
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32).to(device)
    config = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "checkpoint_interval": 999,
        "log_interval": 5,
        "aux_opp_reply_weight": 0.0,
        "lr_schedule": "none",
    }
    trainer = Trainer(model, config, checkpoint_dir="/tmp/test_trainer_proto", device=device)
    assert isinstance(trainer, TrainerLike)


def test_real_eval_pipeline_satisfies_eval_pipeline_like():
    device = torch.device("cpu")
    config = {
        "eval": {"n_games": 2, "n_simulations": 10},
        "mcts": {"interior_selector": "puct", "n_simulations": 10},
    }
    ep, _, _ = build_eval_pipeline(config, device, "test-run", {})
    assert isinstance(ep, EvalPipelineLike)


def test_real_clock_satisfies_clock_like():
    clock = RealClock()
    assert isinstance(clock, ClockLike)


def test_real_tracemalloc_satisfies_tracemalloc_like():
    tm = RealTracemalloc()
    assert isinstance(tm, TracemallocLike)


def test_step_coordinator_config_frozen():
    cfg = StepCoordinatorConfig(
        eval_interval=100,
        log_interval=10,
        checkpoint_interval=500,
        composition_interval=100,
        value_probe_interval=100,
        min_buf_size=100,
        capacity=1000,
        buffer_schedule=(),
        training_steps_per_game=1.0,
        max_train_burst=8,
        batch_size=256,
        augment=True,
        recency_weight=0.0,
        mixing_initial_w=0.8,
        mixing_min_w=0.1,
        mixing_decay_steps=1_000_000.0,
        soft_ew_threshold=0.0,
        soft_ew_min_pts=0,
        hard_gn_threshold=3.0,
        hard_gn_min_steps=5,
        instrumentation_enabled=False,
        stop_step=None,
        final_eval_drain_timeout_sec=0.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        cfg.eval_interval = 200


def test_step_outcome_frozen():
    out = StepOutcome(
        train_step=0,
        games_played=0,
        in_warmup=True,
        waiting_for_games=False,
        steps_run=0,
        last_loss_info=None,
        buffer_resized=None,
        checkpoint_saved=False,
        axis_emitted=False,
        eval_kicked_off=False,
        eval_skipped_busy=False,
        eval_drained=False,
        promoted_step=None,
        soft_abort_fired=False,
        hard_abort_fired=False,
        consec_high_gn=0,
        instrumentation_emitted=[],
        pool_overflow_delta=0,
        games_per_hour=0.0,
    )
    with pytest.raises((AttributeError, TypeError)):
        out.train_step = 1
