"""StepCoordinator — per-step state machine extracted from training loop (§159a).

Protocols + config + outcome dataclass (M2).  Skeleton class (M3).  Step logic (M4).
"""
from __future__ import annotations

import threading
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import structlog

from hexo_rl.training.loop import (
    RollingGamesPerHour,
    _compute_pretrained_weight,
    _steps_budget,
)


# ── Protocols (no torch import) ──────────────────────────────────────────────

@runtime_checkable
class TrainerLike(Protocol):
    step: int
    model: Any
    def train_step(self, buffer: Any, *, augment: bool = True, recent_buffer: Any | None = None) -> dict[str, float]: ...
    def train_step_from_tensors(self, *args: Any, **kwargs: Any) -> dict[str, float]: ...
    def save_checkpoint(self, loss_info: dict[str, float] | None) -> Any: ...


@runtime_checkable
class ReplayBufferLike(Protocol):
    size: int
    capacity: int
    def resize(self, new_capacity: int) -> None: ...
    def save_to_path(self, path: str) -> None: ...


@runtime_checkable
class RecentBufferLike(Protocol):
    def add(self, *args: Any, **kwargs: Any) -> None: ...
    def sample(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class WorkerPoolLike(Protocol):
    games_completed: int
    n_workers: int
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def buffer_composition(self) -> dict[str, Any]: ...
    def model_version_summary(self) -> dict[str, Any]: ...
    def per_worker_draw_rates(self) -> dict[int, float]: ...


@runtime_checkable
class EvalPipelineLike(Protocol):
    def run_evaluation(
        self,
        model: Any,
        step: int,
        best: Any | None,
        *,
        full_config: dict[str, Any],
        best_model_step: int | None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class GpuMonitorLike(Protocol):
    gpu_util_pct: float


@runtime_checkable
class ClockLike(Protocol):
    def now(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


@runtime_checkable
class TracemallocLike(Protocol):
    def start(self, max_frames: int = 25) -> None: ...
    def stop(self) -> None: ...
    def get_traced_memory(self) -> tuple[int, int]: ...
    def take_snapshot(self) -> Any: ...
    def reset_peak(self) -> None: ...


# ── Default implementations ──────────────────────────────────────────────────

class RealClock:
    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class RealTracemalloc:
    def start(self, max_frames: int = 25) -> None:
        tracemalloc.start(max_frames)

    def stop(self) -> None:
        tracemalloc.stop()

    def get_traced_memory(self) -> tuple[int, int]:
        return tracemalloc.get_traced_memory()

    def take_snapshot(self) -> Any:
        return tracemalloc.take_snapshot()

    def reset_peak(self) -> None:
        tracemalloc.reset_peak()


# ── Config dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StepCoordinatorConfig:
    eval_interval: int
    log_interval: int
    checkpoint_interval: int
    composition_interval: int
    value_probe_interval: int
    min_buf_size: int
    capacity: int
    buffer_schedule: tuple[dict[str, Any], ...]
    training_steps_per_game: float
    max_train_burst: int
    batch_size: int
    augment: bool
    recency_weight: float
    mixing_initial_w: float
    mixing_min_w: float
    mixing_decay_steps: float
    soft_ew_threshold: float
    soft_ew_min_pts: int
    hard_gn_threshold: float
    hard_gn_min_steps: int
    instrumentation_enabled: bool
    stop_step: int | None
    final_eval_drain_timeout_sec: float


# ── Step outcome dataclass ───────────────────────────────────────────────────

@dataclass(frozen=True)
class StepOutcome:
    train_step: int
    games_played: int
    in_warmup: bool
    waiting_for_games: bool
    steps_run: int
    last_loss_info: dict[str, float] | None
    buffer_resized: int | None
    checkpoint_saved: bool
    axis_emitted: bool
    eval_kicked_off: bool
    eval_skipped_busy: bool
    eval_drained: bool
    promoted_step: int | None
    soft_abort_fired: bool
    hard_abort_fired: bool
    consec_high_gn: int
    instrumentation_emitted: list[str]
    pool_overflow_delta: int
    games_per_hour: float


# ── StepCoordinator skeleton (M3) ────────────────────────────────────────────

class StepCoordinator:
    """Owns per-step mutable state extracted from ``loop.py::_run_loop``.

    M3: pure assignment in ``__init__``; all action methods stubbed.
    M4: ``step()`` body migrated from closure.
    """

    def __init__(
        self,
        *,
        trainer: TrainerLike,
        buffer: ReplayBufferLike,
        pretrained_buffer: ReplayBufferLike | None,
        recent_buffer: RecentBufferLike | None,
        pool: WorkerPoolLike,
        eval_pipeline: EvalPipelineLike | None,
        gpu_monitor: GpuMonitorLike,
        subsystems: Any,
        anchor_state: Any,
        shutdown: Any,
        eval_model: Any,
        bufs: Any,
        early_game_probe: Any,
        value_probe: Any,
        axis_baseline: dict[str, Any] | None,
        tb_writer: Any,
        config: StepCoordinatorConfig,
        clock: ClockLike = RealClock(),
        tracemalloc_provider: TracemallocLike = RealTracemalloc(),
        event_emitter: Callable[[dict[str, Any]], None] = structlog.get_logger,
        logger: structlog.BoundLogger | None = None,
    ) -> None:
        # Collaborators
        self.trainer = trainer
        self.buffer = buffer
        self.pretrained_buffer = pretrained_buffer
        self.recent_buffer = recent_buffer
        self.pool = pool
        self.eval_pipeline = eval_pipeline
        self.gpu_monitor = gpu_monitor
        self.subsystems = subsystems
        self.anchor_state = anchor_state
        self.shutdown = shutdown
        self.best_model = anchor_state.best_model
        self.eval_model = eval_model
        self.bufs = bufs
        self.early_game_probe = early_game_probe
        self.value_probe = value_probe
        self.axis_baseline = axis_baseline
        self.tb_writer = tb_writer
        self.config = config
        self._clock = clock
        self._tracemalloc = tracemalloc_provider
        self._event_emitter = event_emitter
        self._logger = logger or structlog.get_logger("hexo_rl.training.loop")

        # Mutable per-run state (§1.1)
        self._train_step = trainer.step
        self._games_played = pool.games_completed
        self._initial_policy_loss: float | None = None
        self._last_loss_info: dict[str, float] = {}
        self._eval_thread: threading.Thread | None = None
        self._best_model_step: int | None = anchor_state.best_model_step
        self._schedule_idx = 1  # first schedule entry already applied at buffer construction
        self._consec_high_gn = 0
        self._eval_result: list[dict[str, Any] | None] = [None]
        self._ew_history = deque(maxlen=max(config.soft_ew_min_pts, 1))
        self.last_train_game_count = pool.games_completed
        self.last_warmup_log = 0.0
        self.last_iter_games = self._games_played
        self._last_quiescence_fires = 0
        self._last_pool_overflows = 0
        self._rolling_gph = RollingGamesPerHour(t_start=clock.now())

    # ── Read-only properties ─────────────────────────────────────────────────

    @property
    def train_step(self) -> int:
        return self._train_step

    @property
    def games_played(self) -> int:
        return self._games_played

    @property
    def consec_high_gn(self) -> int:
        return self._consec_high_gn

    @property
    def ew_history(self) -> tuple[float, ...]:
        return tuple(self._ew_history)

    @property
    def schedule_idx(self) -> int:
        return self._schedule_idx

    @property
    def is_eval_in_flight(self) -> bool:
        return self._eval_thread is not None and self._eval_thread.is_alive()

    @property
    def best_model_step(self) -> int | None:
        return self._best_model_step

    @property
    def last_loss_info(self) -> dict[str, float]:
        return self._last_loss_info

    @property
    def initial_policy_loss(self) -> float | None:
        return self._initial_policy_loss

    # ── Delegated helpers (R7, R20) ──────────────────────────────────────────

    def compute_pretrained_weight(self, step: int) -> float:
        return _compute_pretrained_weight(
            step,
            self.config.mixing_initial_w,
            self.config.mixing_min_w,
            self.config.mixing_decay_steps,
        )

    def games_per_hour(self) -> float:
        return self._rolling_gph.update(self._clock.now(), self._games_played)

    def _steps_budget(self, new_games: int) -> int:
        return _steps_budget(
            new_games,
            self.config.training_steps_per_game,
            self.config.max_train_burst,
        )

    # ── Action stubs (M4) ────────────────────────────────────────────────────

    def step(self) -> StepOutcome:
        raise NotImplementedError("§159a M4")

    def flush_pending_eval(self) -> StepOutcome | None:
        raise NotImplementedError("§159a M4")

    def request_stop(self, reason: str) -> None:
        raise NotImplementedError("§159a M4")

    def run_until_stopped(self) -> None:
        raise NotImplementedError("§159a M4")
