"""StepCoordinator — per-step state machine extracted from training loop (§159a).

Owns the closure state previously trapped inside ``loop.py::_run_loop`` and
exposes one outer iteration as ``step()``.  ``run_until_stopped()`` drives
production; tests call ``step()`` directly with stub collaborators.

Protocols + config + outcome dataclass landed in M2.  Skeleton + init in M3.
``step()`` body, ``flush_pending_eval()``, ``request_stop()``, and
``run_until_stopped()`` land here in M4.

R18: this module's logger uses the literal string
``"hexo_rl.training.loop"`` (not ``__name__``) so dashboard/log filters that
key on the wire-name keep firing on safety-critical events
(``hard_abort_grad_norm``, ``soft_abort_ew_flat``, ``iteration_limit_reached``,
``evaluation_start``, ``mcts_pool_overflow``).
"""
from __future__ import annotations

import math
import threading
import time
import tracemalloc
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import structlog

from hexo_rl.eval.result_types import EvalRoundResult
from hexo_rl.monitoring.events import emit_event
from hexo_rl.training.batch_assembly import assemble_mixed_batch
from hexo_rl.training.loop import RollingGamesPerHour
from hexo_rl.training.mixing import (
    _compute_pretrained_weight,
    _steps_budget,
)
from hexo_rl.training.buffer_persist import try_save_buffer as _try_save_buffer
from hexo_rl.training.eval_drain import drain_pending_eval as _drain_pending_eval
from hexo_rl.training.events import (
    emit_axis_distribution as _emit_axis_distribution,
    emit_training_events as _emit_training_events,
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
    def set_radius_override(self, radius: int | None) -> None: ...


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
    # §178 bot-corpus slot
    bot_batch_share: float = 0.0
    # §178 refresh hook (DISABLED for §178; §179 will flip enabled=True)
    bot_corpus_refresh_enabled: bool = False
    bot_corpus_refresh_cooldown: int = 25_000


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


# ── StepCoordinator ──────────────────────────────────────────────────────────

class StepCoordinator:
    """Owns per-step mutable state extracted from ``loop.py::_run_loop``.

    One ``step()`` call equals one outer iteration of the original closure
    (warmup tick, waiting-for-games tick, or full training burst).  Returns a
    :class:`StepOutcome` describing every decision made on that iter.

    The caller (`run_training_loop`) is responsible for the surrounding
    ``tracemalloc.start``/``.stop`` wrap, ``pool.recent_buffer`` injection,
    pretrain-event replay, ``pool.start()``, and final teardown
    (``pool.stop()`` + ``subsys.teardown()``).  ``flush_pending_eval()`` is
    called from the caller's ``finally:`` block to drain a possibly-promoted
    final eval before the inference server is torn down (D-012).

    Single-process invariant: signal handlers are installed by the caller
    against the same ``ShutdownState`` instance the coordinator holds.
    Constructing two coordinators in the same process re-installs handlers
    and leaves the first coordinator's shutdown state stale.
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
        subsystems: Any,
        anchor_state: Any,
        shutdown: Any,
        eval_model: Any,
        bufs: Any,
        config: StepCoordinatorConfig,
        full_config: dict[str, Any] | None = None,
        train_cfg: dict[str, Any] | None = None,
        mcts_config: dict[str, Any] | None = None,
        mixing_cfg: dict[str, Any] | None = None,
        batch_size_cfg: int | None = None,
        iterations: int | None = None,
        clock: ClockLike = RealClock(),
        tracemalloc_provider: TracemallocLike = RealTracemalloc(),
        event_emitter: Callable[[dict[str, Any]], None] = emit_event,
        logger: Any = None,
        bot_buffer: ReplayBufferLike | None = None,
    ) -> None:
        # Collaborators
        self.trainer = trainer
        self.buffer = buffer
        self.pretrained_buffer = pretrained_buffer
        self.bot_buffer = bot_buffer  # §178 bot-corpus slot
        self.recent_buffer = recent_buffer
        self.pool = pool
        self.eval_pipeline = eval_pipeline
        # gpu_monitor / early_game_probe / value_probe / axis_baseline / tb_writer
        # accessed via self.subsystems.<name> (§176 P11 — collapsed redundant kwargs)
        self.subsystems = subsystems
        self.anchor_state = anchor_state
        self.shutdown = shutdown
        self.best_model = anchor_state.best_model
        self.eval_model = eval_model
        self.bufs = bufs
        self.config = config
        self.full_config = full_config if full_config is not None else {}
        self.train_cfg = train_cfg if train_cfg is not None else {}
        self.mcts_config = mcts_config if mcts_config is not None else {}
        self.mixing_cfg = mixing_cfg if mixing_cfg is not None else {}
        self.batch_size_cfg = batch_size_cfg if batch_size_cfg is not None else config.batch_size
        self.iterations = iterations
        self._clock = clock
        self._tracemalloc = tracemalloc_provider
        self._event_emitter = event_emitter
        self._logger = logger or structlog.get_logger("hexo_rl.training.loop")

        # Mutable per-run state (§1.1)
        self._train_step = trainer.step
        self._games_played = pool.games_completed
        self._last_bot_refresh_step: int = 0  # §178 T7 refresh-hook cooldown
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

        # §174 — radius curriculum state
        self._current_radius: int | None = self._resolve_radius(self._train_step)
        if self._current_radius is not None and hasattr(pool, "set_radius_override"):
            pool.set_radius_override(self._current_radius)

    # ── Radius curriculum (§174) ─────────────────────────────────────────────

    def _resolve_radius(self, step: int) -> int | None:
        """Resolve current legal_move_radius from schedule.

        Returns ``None`` if no schedule configured (use encoding default).
        """
        schedule = self.full_config.get("selfplay", {}).get("legal_move_radius_schedule")
        if not schedule:
            return None
        current_radius = None
        for entry in schedule:
            if step >= entry["step"]:
                current_radius = entry["radius"]
        return current_radius

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

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def request_stop(self, reason: str) -> None:
        """Flip shutdown.running = False + log a single ``stop_requested`` event.

        Used by callers / tests that want a structured stop trigger; the
        in-step abort gates (hard-GN, soft-EW) still write
        ``shutdown.running = False`` directly because they emit their own
        domain-specific log right before the flip.
        """
        self.shutdown.running = False
        self._logger.info("stop_requested", reason=reason)

    def run_until_stopped(self) -> None:
        """Production entry point — drives ``step()`` until shutdown.

        Mirrors the ``while _shutdown.running`` loop of the original closure.
        Tests should call ``step()`` directly instead of going through here.
        """
        while self.shutdown.running:
            self.step()

    # ── One outer iteration ──────────────────────────────────────────────────

    def step(self) -> StepOutcome:
        """Run exactly one outer iteration.

        Returns a :class:`StepOutcome` describing every decision fired on
        this iter.  Mutates instance state (``train_step``, ``games_played``,
        eval thread, deques, counters).  Sets ``shutdown.running = False`` on
        iter-limit / shutdown-save / hard-GN / soft-EW.
        """
        cfg = self.config

        # Outcome scaffolding
        in_warmup = False
        waiting_for_games = False
        steps_run = 0
        buffer_resized: int | None = None
        checkpoint_saved = False
        axis_emitted = False
        eval_kicked_off = False
        eval_skipped_busy = False
        eval_drained = False
        promoted_step: int | None = None
        soft_abort_fired = False
        hard_abort_fired = False
        instrumentation_emitted: list[str] = []
        pool_overflow_delta = 0

        # ── O2: iteration-limit reached ─────────────────────────────────────
        if cfg.stop_step is not None and self._train_step >= cfg.stop_step:
            self._logger.info("iteration_limit_reached", iterations=self.iterations)
            self.shutdown.running = False
            return self._build_outcome(
                steps_run=0, in_warmup=False, waiting_for_games=False,
                buffer_resized=None, checkpoint_saved=False, axis_emitted=False,
                eval_kicked_off=False, eval_skipped_busy=False, eval_drained=False,
                promoted_step=None, soft_abort_fired=False, hard_abort_fired=False,
                instrumentation_emitted=[], pool_overflow_delta=0,
            )

        # ── O3: shutdown-save (signal-handler flag) ─────────────────────────
        if self.shutdown.shutdown_save:
            self._logger.info(
                "shutdown_signal_checkpoint",
                msg="Shutdown signal received — saving checkpoint before exit",
                step=self._train_step,
            )
            self.trainer.save_checkpoint(self._last_loss_info if self._last_loss_info else None)
            _try_save_buffer(self.buffer, self.mixing_cfg, "shutdown_signal", self.recent_buffer)
            self.shutdown.running = False
            return self._build_outcome(
                steps_run=0, in_warmup=False, waiting_for_games=False,
                buffer_resized=None, checkpoint_saved=True, axis_emitted=False,
                eval_kicked_off=False, eval_skipped_busy=False, eval_drained=False,
                promoted_step=None, soft_abort_fired=False, hard_abort_fired=False,
                instrumentation_emitted=[], pool_overflow_delta=0,
            )

        self._games_played = self.pool.games_completed

        # ── O4: warmup ──────────────────────────────────────────────────────
        if self.buffer.size < cfg.min_buf_size:
            if (self._clock.now() - self.last_warmup_log) >= 5.0:
                structlog.get_logger().info(
                    "warmup",
                    buffer=self.buffer.size,
                    target=cfg.min_buf_size,
                    games=self._games_played,
                    gpu_pct=round(self.subsystems.gpu_monitor.gpu_util_pct, 0),
                )
                self._event_emitter({
                    "event": "system_stats",
                    "buffer_size": self.buffer.size,
                    "buffer_capacity": cfg.capacity,
                })
                self.last_warmup_log = self._clock.now()
            self._clock.sleep(0.5)
            in_warmup = True
            return self._build_outcome(
                steps_run=0, in_warmup=True, waiting_for_games=False,
                buffer_resized=None, checkpoint_saved=False, axis_emitted=False,
                eval_kicked_off=False, eval_skipped_busy=False, eval_drained=False,
                promoted_step=None, soft_abort_fired=False, hard_abort_fired=False,
                instrumentation_emitted=[], pool_overflow_delta=0,
            )

        # ── O5: no new games ────────────────────────────────────────────────
        new_games = self._games_played - self.last_train_game_count
        if new_games <= 0:
            if (self._clock.now() - self.last_warmup_log) >= 5.0:
                structlog.get_logger().info(
                    "waiting_for_games",
                    games=self._games_played,
                    trained_games=self.last_train_game_count,
                    buffer=self.buffer.size,
                )
                self.last_warmup_log = self._clock.now()
            self._clock.sleep(0.1)
            return self._build_outcome(
                steps_run=0, in_warmup=False, waiting_for_games=True,
                buffer_resized=None, checkpoint_saved=False, axis_emitted=False,
                eval_kicked_off=False, eval_skipped_busy=False, eval_drained=False,
                promoted_step=None, soft_abort_fired=False, hard_abort_fired=False,
                instrumentation_emitted=[], pool_overflow_delta=0,
            )

        # ── O6: compute steps budget + advance bookkeeping ──────────────────
        steps_budget = self._steps_budget(new_games)
        self.last_train_game_count = self._games_played

        # ── Inner training-step burst ───────────────────────────────────────
        loss_info: dict[str, float] = {}
        w_pre = 0.0
        for _ in range(steps_budget):
            # D0: stop_step inside burst
            if cfg.stop_step is not None and self._train_step >= cfg.stop_step:
                break

            # D1: buffer growth schedule
            while (self._schedule_idx < len(cfg.buffer_schedule)
                   and self._train_step >= cfg.buffer_schedule[self._schedule_idx]["step"]):
                new_cap = cfg.buffer_schedule[self._schedule_idx]["capacity"]
                if new_cap > self.buffer.capacity:
                    self.buffer.resize(new_cap)
                    self._logger.info("buffer_resized", step=self._train_step, new_capacity=new_cap)
                    buffer_resized = new_cap
                self._schedule_idx += 1

            # D2: training step path select
            batch_size = int(self.train_cfg.get("batch_size", self.full_config.get("batch_size", 256)))
            if (self.pretrained_buffer is not None
                    and self.pretrained_buffer.size > 0
                    and self.buffer.size > 0):
                w_pre = _compute_pretrained_weight(
                    self._train_step,
                    cfg.mixing_initial_w,
                    cfg.mixing_min_w,
                    cfg.mixing_decay_steps,
                )
                # §178 — top-level bot slot, parallel to corpus + selfplay.
                # n_bot fixed by config (NOT decaying with pretrain_weight).
                n_bot = (
                    round(cfg.bot_batch_share * batch_size)
                    if (self.bot_buffer is not None and self.bot_buffer.size > 0)
                    else 0
                )
                # corpus weight applied to non-bot remainder (batch_size − n_bot).
                n_pre = max(1, int(math.ceil(w_pre * (batch_size - n_bot))))
                n_self = batch_size - n_pre - n_bot
                batch = assemble_mixed_batch(
                    self.pretrained_buffer, self.buffer, self.recent_buffer,
                    n_pre, n_self, batch_size, self.batch_size_cfg,
                    cfg.recency_weight, self.bufs, self._train_step,
                    augment=cfg.augment,
                    bot_buffer=self.bot_buffer,
                    n_bot=n_bot,
                )
                loss_info = self.trainer.train_step_from_tensors(
                    batch.states, batch.policies, batch.outcomes,
                    chain_planes=batch.chain_planes,
                    ownership_targets=batch.ownership,
                    threat_targets=batch.winning_line,
                    is_full_search=batch.is_full_search,
                    # §178 critical pin: n_pretrain extends through bot rows so
                    # aux_decode.mask_aux_rows excludes them from aux losses
                    # (bot rows have neutral aux pad — ownership=1 / wl=0).
                    n_pretrain=n_pre + n_bot,
                    n_recent=batch.n_recent_actual,
                )
            else:
                w_pre = 0.0
                loss_info = self.trainer.train_step(
                    self.buffer,
                    augment=cfg.augment,
                    recent_buffer=self.recent_buffer,
                )

            self._train_step = self.trainer.step
            if self._initial_policy_loss is None:
                self._initial_policy_loss = float(loss_info["policy_loss"])
            self._last_loss_info = loss_info

            # D3: hard-abort on sustained gradient norm
            _step_gn = float(loss_info.get("grad_norm", 0.0))
            if math.isfinite(_step_gn) and _step_gn > cfg.hard_gn_threshold:
                self._consec_high_gn += 1
                if self._consec_high_gn >= cfg.hard_gn_min_steps:
                    self._logger.error(
                        "hard_abort_grad_norm",
                        step=self._train_step,
                        consec_steps=self._consec_high_gn,
                        grad_norm=round(_step_gn, 4),
                        threshold=cfg.hard_gn_threshold,
                        msg="Sustained high gradient norm — halting run. Roll back to ckpt_12190 before re-resuming.",
                    )
                    self.shutdown.running = False
                    hard_abort_fired = True
            else:
                self._consec_high_gn = 0

            # D4: ckpt-cadence buffer save (trainer's own ckpt is inside trainer.train_step)
            if self._train_step % cfg.checkpoint_interval == 0:
                if self._train_step > 0:
                    _try_save_buffer(self.buffer, self.mixing_cfg, "checkpoint_interval", self.recent_buffer)
                    checkpoint_saved = True

            # D5: axis-distribution emit + D5b soft-abort
            if self._train_step > 0 and self._train_step % cfg.eval_interval == 0:
                _axis_q_val = _emit_axis_distribution(
                    self._train_step, self.pool, self.full_config,
                    self.subsystems.axis_baseline, self.subsystems.tb_writer,
                )
                axis_emitted = True
                if (cfg.soft_ew_threshold > 0.0 and cfg.soft_ew_min_pts > 0
                        and _axis_q_val is not None):
                    self._ew_history.append(_axis_q_val)
                    if (len(self._ew_history) >= cfg.soft_ew_min_pts
                            and all(v > cfg.soft_ew_threshold for v in self._ew_history)):
                        self._logger.warning(
                            "soft_abort_ew_flat",
                            step=self._train_step,
                            ew_history=[round(v, 4) for v in self._ew_history],
                            threshold=cfg.soft_ew_threshold,
                            msg=(
                                "E-W fraction flat above threshold — soft-abort. "
                                "Commit checkpoint and open §120 investigation."
                            ),
                        )
                        self.shutdown.running = False
                        soft_abort_fired = True

            # D6: eval (drain previous, then maybe kick off new)
            if (self.eval_pipeline is not None
                    and self._train_step > 0
                    and self._train_step % cfg.eval_interval == 0):
                prev_thread = self._eval_thread
                prev_best_step = self._best_model_step
                self._eval_thread, self._best_model_step = _drain_pending_eval(
                    self._eval_thread, self._eval_result, self.eval_model, self.best_model,
                    self.anchor_state.best_model_path, self._best_model_step, self.pool, self._train_step,
                )
                if prev_thread is not None and self._eval_thread is None:
                    eval_drained = True
                    if self._best_model_step != prev_best_step:
                        promoted_step = self._best_model_step

                if self._eval_thread is None or not self._eval_thread.is_alive():
                    base_model = getattr(self.trainer.model, "_orig_mod", self.trainer.model)
                    assert self.eval_model is not None
                    _eval_base = getattr(self.eval_model, "_orig_mod", self.eval_model)
                    _eval_base.load_state_dict(base_model.state_dict())
                    step_snapshot = self._train_step
                    self._logger.info("evaluation_start", step=step_snapshot)

                    # R3 / F-016: default-arg snapshot is load-bearing — freezes the
                    # references the daemon thread sees AT KICKOFF, so subsequent
                    # mutations of self.eval_model / self.best_model / self._best_model_step
                    # do not leak into the in-flight eval.  KEEP AS NESTED FUNCTION.
                    eval_pipeline = self.eval_pipeline
                    eval_result = self._eval_result
                    logger = self._logger

                    def _run_eval(
                        _model: Any = self.eval_model,
                        _step: int = step_snapshot,
                        _best: Any = self.best_model,
                        _cfg: dict[str, Any] = self.full_config,
                        _best_step: int | None = self._best_model_step,
                    ) -> None:
                        try:
                            # L4: run_evaluation sets result["step"] itself; the
                            # post-hoc assignment here was dead code.
                            result: EvalRoundResult = eval_pipeline.run_evaluation(
                                _model, _step, _best, full_config=_cfg,
                                best_model_step=_best_step,
                            )
                            eval_result[0] = result
                        except Exception:
                            import traceback
                            logger.info("evaluation_error", step=_step, tb=traceback.format_exc())
                            eval_result[0] = {"promoted": False, "error": True, "step": _step}

                    self._eval_thread = threading.Thread(target=_run_eval, daemon=True)
                    self._eval_thread.start()
                    eval_kicked_off = True
                else:
                    self._logger.info("eval_skipped_still_running", step=self._train_step)
                    eval_skipped_busy = True

            # D7: tracemalloc snapshot every 500 steps (try/except swallow)
            if self._train_step > 0 and self._train_step % 500 == 0:
                try:
                    snapshot = self._tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics("lineno")[:10]
                    self._logger.info(
                        "tracemalloc_top10",
                        step=self._train_step,
                        allocators=[
                            {
                                "file": str(s.traceback[0]),
                                "size_mb": round(s.size / 1024**2, 3),
                                "count": s.count,
                            }
                            for s in top_stats
                        ],
                    )
                except Exception as _tm_err:
                    self._logger.warning("tracemalloc_failed", error=str(_tm_err))

            # D7b: §174 radius curriculum — check schedule at log cadence
            if self._train_step % cfg.log_interval == 0:
                new_radius = self._resolve_radius(self._train_step)
                if new_radius != self._current_radius:
                    self._current_radius = new_radius
                    if hasattr(self.pool, "set_radius_override"):
                        self.pool.set_radius_override(new_radius)
                    self._logger.info(
                        "radius_curriculum",
                        step=self._train_step,
                        radius=new_radius,
                    )

            # D8: emit training events + pool-overflow soft warning
            if self._train_step % cfg.log_interval == 0:
                # §176 P9 — typed snapshot replaces direct ``_runner`` reach.
                _cur_qfire = self.pool.runner_stats().mcts_quiescence_fires
                _qfire_delta = _cur_qfire - self._last_quiescence_fires
                self._last_quiescence_fires = _cur_qfire
                # Pool-overflow surface — soft warning only, not hard-fail.  We
                # don't know production frequency yet; aborting on first
                # occurrence would risk killing healthy runs over a benign
                # tick.  Promote to hard-fail once a threshold is calibrated.
                try:
                    from engine import mcts_pool_overflow_count  # DO NOT HOIST — pre-Tier-1.A engine wheels lack this symbol
                    _cur_pool_overflows = int(mcts_pool_overflow_count())
                except (ImportError, AttributeError):
                    # Engine wheel pre-dates Tier-1.A counter — silent skip.
                    _cur_pool_overflows = 0
                _pool_overflow_delta = _cur_pool_overflows - self._last_pool_overflows
                self._last_pool_overflows = _cur_pool_overflows
                pool_overflow_delta = _pool_overflow_delta
                if _pool_overflow_delta > 0:
                    self._logger.warning(
                        "mcts_pool_overflow",
                        step=self._train_step,
                        delta=_pool_overflow_delta,
                        cumulative=_cur_pool_overflows,
                        msg=(
                            "MCTS pool overflow — engine fabricated terminal "
                            "values, biasing training targets. Math says this "
                            "should never happen; investigate board state at "
                            "next occurrence."
                        ),
                    )
                    self._event_emitter({
                        "event": "mcts_pool_overflow",
                        "step": self._train_step,
                        "delta": _pool_overflow_delta,
                        "cumulative": _cur_pool_overflows,
                    })
                _emit_training_events(
                    self._train_step, loss_info, w_pre, self._games_played,
                    self.last_iter_games, self.pool, self.buffer, self.subsystems.gpu_monitor,
                    self.full_config, self.mcts_config, cfg.capacity,
                    self.games_per_hour, _qfire_delta,
                    early_game_probe=self.subsystems.early_game_probe,
                    trainer_model=self.trainer.model,
                )
                self.last_iter_games = self._games_played

            # D9 + D10: instrumentation cadence (gated)
            if cfg.instrumentation_enabled:
                if (self.subsystems.value_probe is not None
                        and self._train_step > 0
                        and self._train_step % cfg.value_probe_interval == 0):
                    try:
                        vp = self.subsystems.value_probe.compute(self.trainer.model)
                        self._event_emitter({
                            "event": "value_probe_drift",
                            "step": self._train_step,
                            "decisive_mean": vp["decisive_mean"],
                            "decisive_std":  vp["decisive_std"],
                            "draw_mean":     vp["draw_mean"],
                            "draw_std":      vp["draw_std"],
                            "n_decisive":    vp["decisive_n"],
                            "n_draw":        vp["draw_n"],
                            "fixture":       self.subsystems.value_probe.fixture_path,
                        })
                        self._logger.info(
                            "value_probe_drift",
                            step=self._train_step,
                            decisive_mean=round(vp["decisive_mean"], 4),
                            draw_mean=round(vp["draw_mean"], 4),
                        )
                        instrumentation_emitted.append("value_probe_drift")
                    except Exception as _vp_err:
                        self._logger.warning(
                            "value_probe_failed", step=self._train_step,
                            error=str(_vp_err),
                        )

                if self._train_step > 0 and self._train_step % cfg.composition_interval == 0:
                    try:
                        comp = self.pool.buffer_composition()
                        self._event_emitter({
                            "event": "buffer_composition",
                            "step": self._train_step,
                            **comp,
                        })
                        mvs = self.pool.model_version_summary()
                        wdr = self.pool.per_worker_draw_rates()
                        self._event_emitter({
                            "event": "worker_draw_rate",
                            "step": self._train_step,
                            "per_worker": {str(k): round(v, 4) for k, v in wdr.items()},
                            "n_workers_observed": len(wdr),
                        })
                        self._event_emitter({
                            "event": "model_version_summary",
                            "step": self._train_step,
                            **mvs,
                            "current_version": self.pool.runner_stats().model_version,
                        })
                        self._logger.info(
                            "instrumentation_periodic",
                            step=self._train_step,
                            draw_target_fraction=comp.get("draw_target_fraction"),
                            colony_terminal_fraction=comp.get("colony_terminal_fraction"),
                            six_terminal_fraction=comp.get("six_terminal_fraction"),
                            cap_terminal_fraction=comp.get("cap_terminal_fraction"),
                            mv_median_range=mvs.get("median_range"),
                            mv_p90_range=mvs.get("p90_range"),
                            mv_spearman_rho=mvs.get("spearman_rho_range_vs_draw"),
                            n_workers_observed=len(wdr),
                        )
                        instrumentation_emitted.append("instrumentation_periodic")
                    except Exception as _instr_err:
                        self._logger.warning(
                            "instrumentation_emit_failed",
                            step=self._train_step,
                            error=str(_instr_err),
                        )

            steps_run += 1

        return self._build_outcome(
            steps_run=steps_run,
            in_warmup=in_warmup,
            waiting_for_games=waiting_for_games,
            buffer_resized=buffer_resized,
            checkpoint_saved=checkpoint_saved,
            axis_emitted=axis_emitted,
            eval_kicked_off=eval_kicked_off,
            eval_skipped_busy=eval_skipped_busy,
            eval_drained=eval_drained,
            promoted_step=promoted_step,
            soft_abort_fired=soft_abort_fired,
            hard_abort_fired=hard_abort_fired,
            instrumentation_emitted=instrumentation_emitted,
            pool_overflow_delta=pool_overflow_delta,
        )

    # ── Final-drain (called from caller's finally:) ──────────────────────────

    def flush_pending_eval(self) -> StepOutcome | None:
        """Drain any in-flight eval before the inference server tears down (D-012).

        Mirrors the closure's ``finally:`` block:
          1. If ``final_eval_drain_timeout_sec > 0`` AND eval still alive AND
             not a SIGINT save, join the eval thread up to the timeout.  Log
             a warning on timeout.
          2. Always call ``drain_pending_eval`` to consume the eval result
             cell — promotes the anchor if the eval gated, no-op otherwise.
             Swallow exceptions and log a warning.

        Returns ``None`` (callers don't consume the outcome — they read
        ``coordinator.best_model_step`` post-call).
        """
        cfg = self.config
        if (cfg.final_eval_drain_timeout_sec > 0.0
                and self._eval_thread is not None
                and self._eval_thread.is_alive()
                and not self.shutdown.shutdown_save):
            self._logger.info(
                "final_eval_drain_waiting",
                timeout_sec=cfg.final_eval_drain_timeout_sec,
            )
            self._eval_thread.join(timeout=cfg.final_eval_drain_timeout_sec)
            if self._eval_thread.is_alive():
                self._logger.warning(
                    "final_eval_drain_timeout",
                    timeout_sec=cfg.final_eval_drain_timeout_sec,
                    msg="eval still running past timeout — proceeding to shutdown",
                )
        try:
            self._eval_thread, self._best_model_step = _drain_pending_eval(
                self._eval_thread, self._eval_result, self.eval_model, self.best_model,
                self.anchor_state.best_model_path, self._best_model_step, self.pool, self._train_step,
            )
        except Exception:
            self._logger.warning("final_eval_drain_failed", exc_info=True)
        return None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_outcome(
        self,
        *,
        steps_run: int,
        in_warmup: bool,
        waiting_for_games: bool,
        buffer_resized: int | None,
        checkpoint_saved: bool,
        axis_emitted: bool,
        eval_kicked_off: bool,
        eval_skipped_busy: bool,
        eval_drained: bool,
        promoted_step: int | None,
        soft_abort_fired: bool,
        hard_abort_fired: bool,
        instrumentation_emitted: list[str],
        pool_overflow_delta: int,
    ) -> StepOutcome:
        return StepOutcome(
            train_step=self._train_step,
            games_played=self._games_played,
            in_warmup=in_warmup,
            waiting_for_games=waiting_for_games,
            steps_run=steps_run,
            last_loss_info=self._last_loss_info if self._last_loss_info else None,
            buffer_resized=buffer_resized,
            checkpoint_saved=checkpoint_saved,
            axis_emitted=axis_emitted,
            eval_kicked_off=eval_kicked_off,
            eval_skipped_busy=eval_skipped_busy,
            eval_drained=eval_drained,
            promoted_step=promoted_step,
            soft_abort_fired=soft_abort_fired,
            hard_abort_fired=hard_abort_fired,
            consec_high_gn=self._consec_high_gn,
            instrumentation_emitted=instrumentation_emitted,
            pool_overflow_delta=pool_overflow_delta,
            games_per_hour=self.games_per_hour(),
        )
