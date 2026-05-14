"""INV9 (§176 §E) — buffer growth schedule pin.

Pinned at hexo_rl/training/step_coordinator.py:517-524 D1 path.
Canonical values: configs/training.yaml:126-129 buffer_schedule.

NOTE TO REVIEWER: master plan §E text says 250K→500K@500K→1M@1.5M;
training.yaml HEAD says 250K→500K@300K→1M@1M. The yaml is the SoT.
This test pins what is at HEAD, not the narrative.

Also note: master plan refers to key "buffer_growth_schedule"; the actual
yaml key is "buffer_schedule". Pinning actual key.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import yaml
import pytest

from hexo_rl.training.step_coordinator import (
    StepCoordinator,
    StepCoordinatorConfig,
)


_TRAINING_YAML = Path(__file__).parent.parent.parent / "configs" / "training.yaml"


# ── Test 1: canonical yaml pin ───────────────────────────────────────────────

def test_canonical_schedule_in_training_yaml() -> None:
    """Pins the buffer_schedule entries in configs/training.yaml at HEAD.

    Key is 'buffer_schedule' (not 'buffer_growth_schedule').
    Boundaries: 300K → 500K, 1M → 1M (not 500K/1.5M as CLAUDE.md narrative says).
    """
    with _TRAINING_YAML.open() as f:
        cfg = yaml.safe_load(f)

    assert "buffer_schedule" in cfg, (
        "buffer_schedule missing from training.yaml — key may have been renamed"
    )
    schedule = cfg["buffer_schedule"]
    assert schedule, "buffer_schedule is empty"

    # PINNED LITERAL VALUES — read from HEAD 2026-05-13.
    # Each entry is a dict {step: int, capacity: int}.
    assert schedule == [
        {"step": 0,           "capacity": 250_000},
        {"step": 300_000,     "capacity": 500_000},
        {"step": 1_000_000,   "capacity": 1_000_000},
    ], f"buffer_schedule drift from pinned values: {schedule}"


# ── Helpers (mirrors test_step_coordinator.py fakes) ────────────────────────

class _FakeClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class _FakeShutdown:
    running: bool = True
    shutdown_save: bool = False


def _make_config(**overrides: Any) -> StepCoordinatorConfig:
    defaults: dict[str, Any] = {
        "eval_interval": 999_999,
        "log_interval": 999_999,
        "checkpoint_interval": 999_999,
        "composition_interval": 999_999,
        "value_probe_interval": 999_999,
        "min_buf_size": 1,
        "capacity": 250_000,
        "buffer_schedule": (),
        "training_steps_per_game": 1.0,
        "max_train_burst": 1,
        "batch_size": 16,
        "augment": False,
        "recency_weight": 0.0,
        "mixing_initial_w": 0.8,
        "mixing_min_w": 0.1,
        "mixing_decay_steps": 1_000_000.0,
        "soft_ew_threshold": 0.0,
        "soft_ew_min_pts": 0,
        "hard_gn_threshold": 1e9,
        "hard_gn_min_steps": 999,
        "instrumentation_enabled": False,
        "stop_step": None,
        "final_eval_drain_timeout_sec": 0.0,
    }
    defaults.update(overrides)
    return StepCoordinatorConfig(**defaults)


def _make_trainer(start_step: int = 0) -> Mock:
    trainer = Mock()
    trainer.step = start_step

    def _train_step(*_a: Any, **_kw: Any) -> dict[str, float]:
        trainer.step += 1
        return {"loss": 0.1, "policy_loss": 0.5, "value_loss": 0.1, "grad_norm": 0.1}

    trainer.train_step = Mock(side_effect=_train_step)
    trainer.train_step_from_tensors = Mock(side_effect=_train_step)
    trainer.save_checkpoint = Mock(return_value="/tmp/ckpt")
    trainer.model = Mock()
    trainer.model._orig_mod = trainer.model
    return trainer


def _make_buffer(size: int = 1000, capacity: int = 250_000) -> Mock:
    buf = Mock()
    buf.size = size
    buf.capacity = capacity
    buf.resize = Mock()
    buf.save_to_path = Mock()
    return buf


def _make_coordinator(
    *,
    trainer: Mock | None = None,
    buffer: Mock | None = None,
    config_overrides: dict[str, Any] | None = None,
    start_step: int = 0,
) -> StepCoordinator:
    anchor = Mock()
    anchor.best_model = Mock()
    anchor.best_model._orig_mod = anchor.best_model
    anchor.best_model_step = 0
    anchor.best_model_path = "/tmp/best.pt"

    pool = Mock()
    pool.games_completed = 1
    pool.n_workers = 1
    # §176 P9 — pool now exposes typed snapshots; provide them on the mock.
    from hexo_rl.selfplay.pool import RunnerStats, InferenceStats
    pool._runner = Mock(mcts_quiescence_fires=0, model_version=0)
    pool._inference_server = Mock()
    _rstats = RunnerStats(
        games_completed=1, positions_generated=0,
        x_wins=0, o_wins=0, draws=0, model_version=0,
        mcts_quiescence_fires=0, mcts_mean_depth=0.0,
        mcts_mean_root_concentration=0.0, cluster_value_std_mean=0.0,
        cluster_policy_disagreement_mean=0.0, cluster_variance_sample_count=0,
        runner_encoding=None,
    )
    _istats = InferenceStats(
        forward_count=0, total_requests=0, encoding_spec=None,
    )
    pool.runner_stats = Mock(return_value=_rstats)
    pool.inference_stats = Mock(return_value=_istats)
    pool.sync_inference_weights = Mock()
    pool.recent_buffer = None

    cfg = _make_config(**(config_overrides or {}))
    _subsystems = Mock(
        gpu_monitor=Mock(gpu_util_pct=10.0),
        early_game_probe=None,
        value_probe=None,
        axis_baseline={},
        tb_writer=None,
    )
    coord = StepCoordinator(
        trainer=trainer or _make_trainer(start_step),
        buffer=buffer or _make_buffer(),
        pretrained_buffer=None,
        recent_buffer=None,
        pool=pool,
        eval_pipeline=None,
        subsystems=_subsystems,
        anchor_state=anchor,
        shutdown=_FakeShutdown(),
        eval_model=Mock(_orig_mod=Mock()),
        bufs=Mock(),
        config=cfg,
        full_config={"monitors": {}},
        train_cfg={"batch_size": 16},
        mcts_config={},
        mixing_cfg={"buffer_persist": False},
        batch_size_cfg=16,
        iterations=None,
        clock=_FakeClock(),
        tracemalloc_provider=Mock(),
        event_emitter=Mock(),
    )
    # Reset so pool.games_completed=1 drives new_games > 0 on first step().
    coord.last_train_game_count = 0
    return coord


# ── Test 2: D1 path calls Buffer.resize at each schedule boundary ─────────────

def test_step_coordinator_consumes_schedule_at_d1() -> None:
    """D1 path at step_coordinator.py:517-524 calls Buffer.resize with the
    expected capacity at each schedule boundary.

    Schedule pinned from training.yaml HEAD (same values as
    test_canonical_schedule_in_training_yaml):
      step=0       → capacity=250_000   (entry 0, applied before coordinator)
      step=300_000 → capacity=500_000   (entry 1)
      step=1M      → capacity=1_000_000 (entry 2)

    _schedule_idx starts at 1 (entry 0 already applied at buffer construction).
    Test drives two single-step() calls, one for each remaining boundary.
    """
    schedule: tuple[dict[str, Any], ...] = (
        {"step": 0,           "capacity": 250_000},
        {"step": 300_000,     "capacity": 500_000},
        {"step": 1_000_000,   "capacity": 1_000_000},
    )

    # --- boundary 1: step=300_000 triggers resize to 500_000 ---
    buf1 = _make_buffer(size=1000, capacity=250_000)
    trainer1 = _make_trainer(start_step=300_000)
    coord1 = _make_coordinator(
        trainer=trainer1,
        buffer=buf1,
        config_overrides={
            "buffer_schedule": schedule,
            "capacity": 250_000,
        },
        start_step=300_000,
    )
    # _train_step is set from trainer.step at init; should be 300_000.
    assert coord1._train_step == 300_000

    with patch("hexo_rl.training.step_coordinator._emit_axis_distribution") as _ax, \
         patch("hexo_rl.training.step_coordinator._emit_training_events"), \
         patch("hexo_rl.training.step_coordinator._try_save_buffer"), \
         patch("hexo_rl.training.step_coordinator._drain_pending_eval") as _drain, \
         patch("hexo_rl.training.step_coordinator.assemble_mixed_batch"):
        _ax.return_value = None
        _drain.return_value = (None, None)
        outcome1 = coord1.step()

    assert outcome1.buffer_resized == 500_000, (
        f"Expected buffer_resized=500_000 at step 300K; got {outcome1.buffer_resized}"
    )
    buf1.resize.assert_called_once_with(500_000)

    # --- boundary 2: step=1_000_000 triggers resize to 1_000_000 ---
    buf2 = _make_buffer(size=1000, capacity=500_000)
    trainer2 = _make_trainer(start_step=1_000_000)
    coord2 = _make_coordinator(
        trainer=trainer2,
        buffer=buf2,
        config_overrides={
            "buffer_schedule": schedule,
            "capacity": 500_000,
        },
        start_step=1_000_000,
    )
    # Fast-forward _schedule_idx to 2 (entries 0+1 already consumed).
    coord2._schedule_idx = 2
    assert coord2._train_step == 1_000_000

    with patch("hexo_rl.training.step_coordinator._emit_axis_distribution") as _ax2, \
         patch("hexo_rl.training.step_coordinator._emit_training_events"), \
         patch("hexo_rl.training.step_coordinator._try_save_buffer"), \
         patch("hexo_rl.training.step_coordinator._drain_pending_eval") as _drain2, \
         patch("hexo_rl.training.step_coordinator.assemble_mixed_batch"):
        _ax2.return_value = None
        _drain2.return_value = (None, None)
        outcome2 = coord2.step()

    assert outcome2.buffer_resized == 1_000_000, (
        f"Expected buffer_resized=1_000_000 at step 1M; got {outcome2.buffer_resized}"
    )
    buf2.resize.assert_called_once_with(1_000_000)
