"""Init assertions for StepCoordinator — catches binding bugs before M4 (§159a M3)."""
from __future__ import annotations

from collections import deque
from typing import Any
from unittest.mock import Mock

import pytest

from hexo_rl.training.step_coordinator import (
    RealClock,
    StepCoordinator,
    StepCoordinatorConfig,
)


def _make_config(**overrides: Any) -> StepCoordinatorConfig:
    defaults = {
        "eval_interval": 100,
        "log_interval": 10,
        "checkpoint_interval": 500,
        "composition_interval": 100,
        "value_probe_interval": 100,
        "min_buf_size": 100,
        "capacity": 1000,
        "buffer_schedule": (),
        "training_steps_per_game": 1.0,
        "max_train_burst": 8,
        "batch_size": 256,
        "augment": True,
        "recency_weight": 0.0,
        "mixing_initial_w": 0.8,
        "mixing_min_w": 0.1,
        "mixing_decay_steps": 1_000_000.0,
        "soft_ew_threshold": 0.0,
        "soft_ew_min_pts": 5,
        "hard_gn_threshold": 3.0,
        "hard_gn_min_steps": 5,
        "instrumentation_enabled": False,
        "stop_step": None,
        "final_eval_drain_timeout_sec": 0.0,
    }
    defaults.update(overrides)
    return StepCoordinatorConfig(**defaults)


def test_init_populates_all_instance_fields_from_anchor_state():
    anchor = Mock(best_model=Mock(), best_model_step=42)
    trainer = Mock(step=7)
    pool = Mock(games_completed=3)
    coord = StepCoordinator(
        trainer=trainer,
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=pool,
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=anchor,
        shutdown=Mock(),
        eval_model=Mock(),
        bufs=Mock(),
        config=_make_config(),
    )
    assert coord.best_model_step == 42
    assert coord.best_model is anchor.best_model


def test_init_creates_fresh_eval_result_cell():
    coord = StepCoordinator(
        trainer=Mock(step=0),
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=Mock(games_completed=0),
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=Mock(best_model=None, best_model_step=None),
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=_make_config(),
    )
    assert coord._eval_result == [None]
    assert coord._eval_result is not [None]  # its own list object


def test_init_creates_fresh_ew_history_deque():
    coord = StepCoordinator(
        trainer=Mock(step=0),
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=Mock(games_completed=0),
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=Mock(best_model=None, best_model_step=None),
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=_make_config(soft_ew_min_pts=5),
    )
    assert isinstance(coord._ew_history, deque)
    assert coord._ew_history.maxlen == 5


def test_init_train_step_mirrors_trainer_step():
    trainer = Mock(step=123)
    pool = Mock(games_completed=0)
    coord = StepCoordinator(
        trainer=trainer,
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=pool,
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=Mock(best_model=None, best_model_step=None),
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=_make_config(),
    )
    assert coord.train_step == 123


def test_init_schedule_idx_starts_at_one():
    coord = StepCoordinator(
        trainer=Mock(step=0),
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=Mock(games_completed=0),
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=Mock(best_model=None, best_model_step=None),
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=_make_config(),
    )
    assert coord.schedule_idx == 1


def test_init_consec_high_gn_starts_at_zero():
    coord = StepCoordinator(
        trainer=Mock(step=0),
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=Mock(games_completed=0),
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=Mock(best_model=None, best_model_step=None),
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=_make_config(),
    )
    assert coord.consec_high_gn == 0


def test_init_last_train_game_count_mirrors_pool():
    pool = Mock(games_completed=99)
    coord = StepCoordinator(
        trainer=Mock(step=0),
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=pool,
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=Mock(best_model=None, best_model_step=None),
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=_make_config(),
    )
    assert coord.last_train_game_count == 99


def test_init_rolling_games_per_hour_attached():
    coord = StepCoordinator(
        trainer=Mock(step=0),
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=Mock(games_completed=0),
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=Mock(best_model=None, best_model_step=None),
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=_make_config(),
    )
    # games_per_hour should be a bound method (not recreated each call)
    assert coord.games_per_hour.__self__ is coord


def test_init_compute_pretrained_weight_attached():
    coord = StepCoordinator(
        trainer=Mock(step=0),
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=Mock(games_completed=0),
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=Mock(best_model=None, best_model_step=None),
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=_make_config(mixing_initial_w=0.8, mixing_min_w=0.1, mixing_decay_steps=1_000_000.0),
    )
    assert coord.compute_pretrained_weight(0) == pytest.approx(0.8, abs=1e-9)


def test_init_clock_default_is_real_clock():
    coord = StepCoordinator(
        trainer=Mock(step=0),
        buffer=Mock(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=Mock(games_completed=0),
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=Mock(best_model=None, best_model_step=None),
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=_make_config(),
    )
    assert isinstance(coord._clock, RealClock)


def test_init_with_stub_collaborators():
    """Full init with protocol-typed stubs — rehearsal for M4 test infra."""
    anchor = Mock(best_model=Mock(), best_model_step=0)
    trainer = Mock(step=0)
    pool = Mock(games_completed=0)
    buffer = Mock(spec=["size", "capacity", "resize", "save_to_path"])
    config = _make_config()

    coord = StepCoordinator(
        trainer=trainer,
        buffer=buffer,
        pretrained_buffer=None,
        recent_buffer=None,
        pool=pool,
        eval_pipeline=None,
        subsystems=Mock(),
        anchor_state=anchor,
        shutdown=Mock(),
        eval_model=None,
        bufs=Mock(),
        config=config,
    )
    # Sanity: all expected attributes exist and have the right shape
    assert coord.train_step == 0
    assert coord.games_played == 0
    assert coord.schedule_idx == 1
    assert coord.consec_high_gn == 0
    assert coord._eval_result == [None]
    assert coord._ew_history.maxlen == config.soft_ew_min_pts
    assert coord.last_train_game_count == 0
    assert coord.last_warmup_log == 0.0
    assert coord.last_iter_games == 0
    assert coord._last_quiescence_fires == 0
    assert coord._last_pool_overflows == 0
    assert not coord.is_eval_in_flight
    assert coord.best_model_step == 0
    assert coord.initial_policy_loss is None
    assert coord.last_loss_info == {}
