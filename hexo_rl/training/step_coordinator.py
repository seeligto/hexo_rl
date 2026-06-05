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
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from hexo_rl.diagnostics.forced_win_detector import ForcedWinTrend

from hexo_rl.eval.result_types import EvalRoundResult
from hexo_rl.monitoring.alert_rules import check_sealbot_wr_hard_abort
from hexo_rl.monitoring.config import MonitoringConfig
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
    def current_stride5_p90(self) -> int: ...
    def set_radius_override(self, radius: int | None) -> None: ...
    def check_producer_health(self) -> None: ...
    def latest_replay_path(self) -> Any: ...
    def update_checkpoint_step(self, step: int) -> None: ...


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

# Default wall-clock cap for joining the final in-flight eval at shutdown
# (`flush_pending_eval`).  The scheduled eval is double-buffered — kicked off at
# boundary N, drained at N+1 — so when a run stops AT a boundary (`stop_step` ==
# a multiple of `eval_interval`) the final eval has no next boundary to drain it.
# A positive default makes `flush_pending_eval` JOIN that daemon eval so its
# result (and any promotion) is consumed instead of dropped when the process
# tears down.  Scoped large (one full eval wall-clock); shutdown is already
# terminal so the wait is acceptable.
DEFAULT_FINAL_EVAL_DRAIN_TIMEOUT_SEC: float = 900.0


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
    # §CANARY-VAL stride-5 spam hard-abort (validated 2026-05-31: benign rolling
    # P90 ≤4 on all radius-5 runs; cosine-temp draw-collapse spam P90 86-133).
    # Gate fires when the pool's rolling-50 stride5 P90 stays ≥ threshold for
    # ``stride5_p90_consec`` consecutive eval points. threshold ≤ 0 disables.
    stride5_p90_threshold: float = 30.0
    stride5_p90_consec: int = 3
    # §178 bot-corpus slot
    bot_batch_share: float = 0.0
    # §178 refresh hook surface (DISABLED-by-default per master baseline; Wave 3
    # variant flips enabled=True). When enabled, fires async regen via
    # subprocess.Popen with hot-reload post-completion. See
    # docs/designs/s179c_bot_refresh_hook.md for the full design + INV pins.
    bot_corpus_refresh_enabled: bool = False
    bot_corpus_refresh_cooldown: int = 25_000  # legacy key — retained for back-compat
    # §S181-AUDIT Wave 3 Stage 2A — fixed-interval trigger (replaces s179c
    # promotion-delta trigger; L51 corpus staleness is time-based not
    # promotion-based). Defaults match Wave 3 dispatcher §2A spec.
    bot_corpus_refresh_interval_steps: int = 5000
    bot_corpus_refresh_n_games: int = 200
    bot_corpus_refresh_opponent_model: str = "ema"   # "ema" | "raw"
    bot_corpus_refresh_replace_strategy: str = "rolling_window"
    bot_corpus_refresh_max_regens: int = 20
    bot_corpus_refresh_min_wr_delta: float = 0.0
    # Subprocess args — passed through unchanged to scripts/generate_bot_corpus.py
    bot_corpus_refresh_max_plies: int = 150
    bot_corpus_refresh_random_opening_plies: int = 4
    bot_corpus_refresh_think_seconds: float = 0.5
    bot_corpus_refresh_anchor_n_sims: int = 200
    bot_corpus_refresh_anchor_temperature: float = 0.5
    # Canonical NPZ path (so the coordinator owns swap-target resolution
    # symmetric with batch_assembly's load-target). Empty string means
    # "fall back to mixing_cfg['bot_corpus_path']" at runtime.
    bot_corpus_path: str = ""


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
        # Seed the recorder's checkpoint_step with the startup train step: the
        # self-play inference model is built from the trainer at startup
        # (loop.build_inference_model), so its weights correspond to this step.
        # It is thereafter refreshed only on promotion (eval_drain) — matching
        # the inference weight-sync cadence — NOT per train step.  Without this
        # the recorder defaults checkpoint_step=0 forever (nothing called it,
        # the §OFFWINDOW data gap).  hasattr-guarded like other optional pool
        # hooks (e.g. set_radius_override) for partial test doubles.
        if hasattr(pool, "update_checkpoint_step"):
            pool.update_checkpoint_step(self._train_step)
        self._games_played = pool.games_completed
        # §178 T7 refresh-hook cooldown bookkeeping (legacy field; retained for
        # the inert disabled-path that still emits the warning log).
        self._last_bot_refresh_step: int = 0
        # §S181-AUDIT Wave 3 Stage 2A — refresh-hook state per s179c §2.2.
        # All None / 0 until first fire; gated entirely behind
        # ``cfg.bot_corpus_refresh_enabled`` so disabled runs are bitwise
        # identical to master baseline (INV-S179c-4).
        self._refresh_proc: Any = None  # subprocess.Popen | None
        self._refresh_started_step: int = 0
        self._refresh_target_anchor_sha: str | None = None
        self._refresh_ema_snapshot_path: Path | None = None
        self._refresh_tmp_npz_path: Path | None = None
        self._n_refreshes_so_far: int = 0
        self._force_bot_refresh: bool = False
        # INV-S179c-2 deferred check — performed on first activation in
        # ``_resolve_canonical_bot_path`` (config-load time can't see runtime FS).

        # §S181-AUDIT Wave 3 Stage 2B — sliding-window SealBot WR hard-abort gate (L50).
        # Wave 2 evidence: alt V_spread held +0.18-+0.30 across 46k steps while
        # wr_sealbot collapsed 33% → 5%; the held-out V_spread canary failed to
        # track actual eval performance. L50 mandates a SealBot WR sliding-window
        # gate as the PRIMARY abort trigger (alt V_spread downgraded to
        # informational per audit/structural/REAL_RUN_RECIPE.md §3).
        # Ring keeps last 5 eval rounds; pure-function trigger logic lives in
        # ``hexo_rl/monitoring/alert_rules.py:check_sealbot_wr_hard_abort``.
        self._wr_history: list[tuple[int, float]] = []
        self._monitoring_cfg = MonitoringConfig.from_dict(self.full_config)
        # INV-S179c-2 enforced at first fire (deferred to runtime — config-load
        # site does not own filesystem state).
        self._initial_policy_loss: float | None = None
        self._last_loss_info: dict[str, float] = {}
        self._eval_thread: threading.Thread | None = None
        self._best_model_step: int | None = anchor_state.best_model_step
        self._schedule_idx = 1  # first schedule entry already applied at buffer construction
        self._consec_high_gn = 0
        # §CANARY-VAL — consecutive eval points with rolling stride5 P90 ≥ threshold.
        self._consec_high_stride5 = 0
        # F03 — set when a drained eval result carries the crash sentinel
        # ({"error": True}); a clean result clears it.  Surfaced LOUD so a broken
        # eval (which silently disables ALL promotions) is never mistaken for a
        # routine "ran, no promotion".
        self._eval_broken = False
        # WIRE (§EVALGATE-B) incremental-replay state.  The eval-boundary forced-win
        # trend reads only games appended since the previous boundary (O(new), not
        # O(whole jsonl)) so the readout cannot creep on the MAIN thread across 300k.
        # ``_fw_trend`` persists the EMA between boundaries; reset together with the
        # offset when ``latest_replay_path`` rotates (UTC-daily), which keeps the EMA
        # numerically identical to a from-scratch replay of the active file.
        self._fw_trend: "ForcedWinTrend | None" = None
        self._fw_replay_path: str | None = None
        self._fw_replay_offset: int = 0
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
    def consec_high_stride5(self) -> int:
        return self._consec_high_stride5

    @property
    def eval_broken(self) -> bool:
        """F03 — True once a drained eval carried the crash sentinel; cleared by
        the next clean eval.  Lets monitoring distinguish a broken promotion gate
        from a healthy gate that simply did not promote."""
        return self._eval_broken

    def _emit_forced_win_trend(self) -> None:
        """WIRE (§EVALGATE-B): offline forced-win trend over recorded self-play.

        Replays the latest ``GameRecorder`` jsonl through the shared detector
        (no NN, no MCTS — zero hot-path), labels it ``single-window`` (A0), and
        emits the smoothed EMA via ``structlog`` (survives ``--no-web-dashboard``;
        the web dashboard is only a renderer).  Best-effort — an advisory readout
        must never break the eval boundary, so any failure is logged, not raised.

        Incremental (nit #2): only the records appended since the previous boundary
        are folded into a persistent EMA (``_fw_trend``).  Per-boundary cost is
        O(new), not O(whole jsonl) — so the MAIN-thread readout cannot creep as the
        file grows across 300k.  On daily rotation (``latest_replay_path`` returns a
        new file) the trend + offset reset, which keeps the EMA numerically identical
        to the prior from-scratch replay of the active file.
        """
        full_cfg = self.full_config if isinstance(self.full_config, dict) else {}
        cfg = full_cfg.get("forced_win_trend", {}) or {}
        if not cfg.get("enabled", True):
            return
        try:
            path = self.pool.latest_replay_path()
            if path is None or not Path(str(path)).is_file():
                return  # nothing recorded yet — no advisory to emit
            path_str = str(path)
            from hexo_rl.diagnostics.forced_win_detector import (
                ForcedWinTrend,
                emit_forced_win_trend,
                engine_player_sides,
                update_trend_from_file_incremental,
            )
            from hexo_rl.encoding import normalize_encoding_name
            # (Re)initialise on first call OR when the daily file rotated: a new file
            # starts at offset 0 with a fresh EMA (matches the old whole-file replay,
            # which only ever read the active file), and never re-reads the old one.
            if self._fw_trend is None or self._fw_replay_path != path_str:
                self._fw_trend = ForcedWinTrend(
                    path="single-window",  # A0: self-play / in-loop-gate / deploy path
                    smoothing=float(cfg.get("smoothing", 0.2)),
                )
                self._fw_replay_offset = 0
                self._fw_replay_path = path_str
            enc = normalize_encoding_name(full_cfg.get("encoding"))
            self._fw_replay_offset = update_trend_from_file_incremental(
                self._fw_trend,
                path_str,
                self._fw_replay_offset,
                encoding=enc,
                # Both engine movers {1, −1} — the wall is symmetric, and a single
                # default (the old ``mover_side=0``) never matches a player → n=0
                # (§OFFWINDOW §7 inert tripwire).  Derived, not hardcoded.
                mover_side=engine_player_sides(enc),
                max_plies=cfg.get("max_plies"),
            )
            emit_forced_win_trend(self._fw_trend, logger=self._logger)
        except Exception:
            self._logger.warning("forced_win_trend_failed", exc_info=True)

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

        # F02 fail-fast: the self-play buffer feeder is the sole producer; if it
        # died on an exception, abort loudly NOW rather than spin in warmup-wait
        # or train on a stale buffer.  No-op when the feeder is healthy.
        self.pool.check_producer_health()

        # §S181-AUDIT Wave 3 Stage 2A — force-trigger sentinel (s179c §1.4).
        # Operator drops /tmp/hexo_rl_force_bot_refresh to bypass the interval
        # gate on the next eval-boundary fire path. Inert when refresh disabled.
        if cfg.bot_corpus_refresh_enabled:
            self._poll_force_refresh_sentinel()

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
                    # §S181-AUDIT Wave 4 4B-impl-3 — ply-index aux head feed.
                    position_indices=batch.position_indices,
                    # DRAW-MASK (Phase 6) — per-row value-supervision mask (capped
                    # self-play rows masked out of the value loss).
                    value_target_valid=batch.value_target_valid,
                )
            else:
                w_pre = 0.0
                loss_info = self.trainer.train_step(
                    self.buffer,
                    augment=cfg.augment,
                    recent_buffer=self.recent_buffer,
                )

            self._train_step = self.trainer.step
            # NB: the recorder's checkpoint_step is NOT bumped per train step —
            # self-play workers run the *inference* model, which only changes on
            # promotion (eval_drain.sync_inference_weights).  Tagging here would
            # over-advance the step past the weights actually generating the games.
            # The tag is refreshed at the promotion site instead (eval_drain.py).
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
                    # §S181-AUDIT Track B — buffer position-class snapshot.
                    # Gated on trainer._track_b_buffer_snapshot; fire-and-forget.
                    if getattr(self.trainer, "_track_b_buffer_snapshot", False):
                        try:
                            from hexo_rl.training.track_b_buffer_snapshot import (
                                snapshot_buffer_position_classes,
                            )
                            snapshot_buffer_position_classes(
                                self.buffer, self._train_step,
                                n_sample=int(getattr(
                                    self.trainer,
                                    "_track_b_buffer_snapshot_n", 5000,
                                )),
                            )
                        except Exception as exc:  # noqa: BLE001
                            structlog.get_logger().warning(
                                "track_b_buffer_snapshot_dispatch_failed",
                                step=self._train_step, error=str(exc),
                            )

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

                # D5c: hard-abort on sustained stride-5 spam (§CANARY-VAL).
                # Rolling-50 stride5 P90 ≥ threshold for N consecutive eval
                # points → halt. Validated separator: radius-5 benign P90 ≤4;
                # cosine-temp draw-collapse spam P90 86-133. threshold ≤ 0 = off.
                if cfg.stride5_p90_threshold > 0.0:
                    _s5_p90 = int(self.pool.current_stride5_p90())
                    if _s5_p90 >= cfg.stride5_p90_threshold:
                        self._consec_high_stride5 += 1
                        if self._consec_high_stride5 >= cfg.stride5_p90_consec:
                            self._logger.error(
                                "hard_abort_stride5_spam",
                                step=self._train_step,
                                consec_evals=self._consec_high_stride5,
                                stride5_p90=_s5_p90,
                                threshold=cfg.stride5_p90_threshold,
                                msg=("Sustained stride-5 spam (rolling-50 P90 ≥ "
                                     "threshold) — halting run. Inspect for "
                                     "draw-collapse / degenerate lattice fill."),
                            )
                            self.shutdown.running = False
                            hard_abort_fired = True
                    else:
                        self._consec_high_stride5 = 0

            # D6: eval (drain previous, then maybe kick off new)
            if (self.eval_pipeline is not None
                    and self._train_step > 0
                    and self._train_step % cfg.eval_interval == 0):
                prev_thread = self._eval_thread
                prev_best_step = self._best_model_step
                # §S181-AUDIT Wave 3 Stage 2B — snapshot eval result before drain
                # clears `eval_result[0]`. Needed for the sliding-window SealBot
                # WR hard-abort gate check that fires AFTER drain.
                _pending_eval_result: dict[str, Any] | None = (
                    self._eval_result[0]
                    if (prev_thread is not None and not prev_thread.is_alive())
                    else None
                )
                self._eval_thread, self._best_model_step = _drain_pending_eval(
                    self._eval_thread, self._eval_result, self.eval_model, self.best_model,
                    self.anchor_state.best_model_path, self._best_model_step, self.pool, self._train_step,
                )
                if prev_thread is not None and self._eval_thread is None:
                    eval_drained = True
                    if self._best_model_step != prev_best_step:
                        promoted_step = self._best_model_step

                    # F03: a crashed eval sets the {"error": True} sentinel and
                    # CANNOT promote — indistinguishable from a clean "ran, no
                    # promotion" unless surfaced.  A persistently broken eval
                    # (anchor-load / OOM / encoding mismatch) silently disables
                    # ALL promotions.  Flag LOUD + emit a distinct event so the
                    # stoppage is never silent; a clean result clears the flag.
                    if _pending_eval_result is not None and _pending_eval_result.get("error"):
                        self._eval_broken = True
                        self._logger.error(
                            "eval_broken",
                            step=self._train_step,
                            eval_step=_pending_eval_result.get("step"),
                            msg="evaluation thread crashed; promotions are disabled "
                                "until eval recovers (see the evaluation_error log)",
                        )
                        self._event_emitter({
                            "event": "eval_broken",
                            "step": self._train_step,
                            "eval_step": _pending_eval_result.get("step"),
                        })
                    elif _pending_eval_result is not None:
                        self._eval_broken = False

                    # §S181-AUDIT Wave 3 Stage 2B — L50 sliding-window SealBot WR
                    # HARD-ABORT gate. Pure-function trigger logic in
                    # ``alert_rules.check_sealbot_wr_hard_abort``; this site owns
                    # the ring + the abort wiring. Wave 2 alt V_spread canary
                    # PASSED throughout the run while wr_sealbot collapsed 33→5%
                    # — this gate is the replacement primary trigger.
                    if (
                        _pending_eval_result is not None
                        and "wr_sealbot" in _pending_eval_result
                        and _pending_eval_result.get("wr_sealbot") is not None
                    ):
                        _wr_sb = float(_pending_eval_result["wr_sealbot"])
                        _eval_step = int(_pending_eval_result.get("step", self._train_step))
                        self._wr_history.append((_eval_step, _wr_sb))
                        # Keep last 5 eval rounds (enough for rolling + peak window).
                        if len(self._wr_history) > 5:
                            self._wr_history.pop(0)
                        _abort_msg = check_sealbot_wr_hard_abort(
                            self._wr_history,
                            self._train_step,
                            self._monitoring_cfg,
                        )
                        if _abort_msg is not None:
                            self._logger.warning(
                                "wave3_wr_hard_abort_triggered",
                                step=self._train_step,
                                wr_history=[(s, round(w, 4)) for s, w in self._wr_history],
                                msg=_abort_msg,
                            )
                            self._event_emitter({
                                "event": "wave3_wr_hard_abort",
                                "step": self._train_step,
                                "wr_history": list(self._wr_history),
                                "message": _abort_msg,
                            })
                            self.shutdown.running = False
                            hard_abort_fired = True

                # WIRE (§EVALGATE-B): offline forced-win trend over recorded
                # self-play games, emitted at eval cadence in the MAIN loop (NOT
                # the eval daemon — so a broken eval thread can't suppress it,
                # which is why this lands after F03).  Single-window path (A0);
                # structlog so it survives --no-web-dashboard.
                self._emit_forced_win_trend()

                # §S181-AUDIT Wave 3 Stage 2A — bot-corpus refresh hook (active).
                # When ``cfg.bot_corpus_refresh_enabled`` is False the hook is
                # entirely inert (no state init, no Popen, no log). When True,
                # we use a FIXED-INTERVAL trigger (NOT promotion-delta — L51:
                # corpus staleness is time-based not promotion-based).
                #
                # Trigger predicate (replaces s179c §1.1 V179c-1):
                #     enabled AND bot_buffer is not None
                #     AND (train_step - last_refresh) >= interval_steps
                #     AND n_refreshes_so_far < max_regens
                #     AND no refresh subprocess already in-flight
                #
                # Force-trigger sentinel (per s179c §1.4) is polled at top of
                # ``step()`` and overrides the interval gate via
                # ``self._force_bot_refresh`` instance flag.
                if cfg.bot_corpus_refresh_enabled:
                    self._tick_bot_refresh()

                if self._eval_thread is None or not self._eval_thread.is_alive():
                    # §S181-AUDIT Wave 2 — eval consumes EMA weights when
                    # enabled (Trainer.inference_state_dict centralises the
                    # raw-vs-EMA routing).
                    assert self.eval_model is not None
                    _eval_base = getattr(self.eval_model, "_orig_mod", self.eval_model)
                    _eval_base.load_state_dict(self.trainer.inference_state_dict())
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
                            # F03: LOUD (.error) — a crashed eval disables promotions;
                            # at .info it was invisible under a WARNING+ filter.
                            logger.error("evaluation_error", step=_step, tb=traceback.format_exc())
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

    # ── §S181-AUDIT Wave 3 Stage 2A — bot-corpus refresh hook ────────────────

    _FORCE_REFRESH_SENTINEL = Path("/tmp/hexo_rl_force_bot_refresh")

    def _poll_force_refresh_sentinel(self) -> None:
        """Check + consume the operator-drop force-refresh sentinel (s179c §1.4).

        Reading + unlinking the sentinel sets ``self._force_bot_refresh`` so the
        next ``_tick_bot_refresh`` call bypasses the interval gate. Inert when
        ``cfg.bot_corpus_refresh_enabled`` is False (caller-gated).
        """
        try:
            if self._FORCE_REFRESH_SENTINEL.is_file():
                self._FORCE_REFRESH_SENTINEL.unlink(missing_ok=True)
                self._force_bot_refresh = True
                self._logger.info(
                    "bot_corpus_refresh_force_sentinel_consumed",
                    step=self._train_step,
                    msg="operator-force-refresh sentinel observed",
                )
        except OSError as exc:
            # Sentinel access errors are non-fatal; log + continue.
            self._logger.warning(
                "bot_corpus_refresh_sentinel_access_failed",
                error=str(exc),
            )

    def _resolve_canonical_bot_path(self) -> Path | None:
        """Resolve canonical NPZ path, validating INV-S179c-2 same-FS invariant.

        Returns None if the configured path is empty (refresh has nothing to
        swap into). Raises RuntimeError on cross-FS violation (refresh tmp
        must share st_dev with canonical for atomic rename per POSIX).
        """
        cfg = self.config
        raw = cfg.bot_corpus_path or self.mixing_cfg.get("bot_corpus_path", "")
        if not raw:
            return None
        canonical = Path(raw).resolve()
        # INV-S179c-2: tmp lives next to canonical (same parent dir → same FS
        # by construction); the guard fires if the parent does not exist.
        if not canonical.parent.exists():
            raise RuntimeError(
                f"bot_corpus_path parent dir missing: {canonical.parent} — "
                f"cannot host refresh tmp file (INV-S179c-2)"
            )
        return canonical

    def _build_refresh_subprocess_command(
        self,
        canonical_path: Path,
        tmp_path: Path,
        anchor_path: Path,
    ) -> list[str]:
        """Compose argv for the regen subprocess (TC4 contract).

        Reuses ``scripts/generate_bot_corpus.py``'s existing CLI surface
        (--anchor, --n-games, --max-plies, etc.). The anchor passed here is
        the EMA snapshot, NOT bootstrap_model_v6.pt, per Wave 3 §opponent_model.
        """
        import sys as _sys
        cfg = self.config
        script = Path(__file__).resolve().parents[1].parent / "scripts" / "generate_bot_corpus.py"
        return [
            _sys.executable,
            str(script),
            "--anchor", str(anchor_path),
            "--n-games", str(cfg.bot_corpus_refresh_n_games),
            "--out", str(tmp_path),
            "--max-plies", str(cfg.bot_corpus_refresh_max_plies),
            "--random-opening-plies", str(cfg.bot_corpus_refresh_random_opening_plies),
            "--think-seconds", str(cfg.bot_corpus_refresh_think_seconds),
            "--anchor-n-sims", str(cfg.bot_corpus_refresh_anchor_n_sims),
            "--anchor-temperature", str(cfg.bot_corpus_refresh_anchor_temperature),
        ]

    def _save_refresh_anchor_snapshot(
        self,
        canonical: Path,
    ) -> tuple[Path, str]:
        """Snapshot the current opponent-model weights to a transient .pt path.

        For ``opponent_model: ema`` we extract EMA via
        ``trainer.inference_state_dict()`` (which routes through EMA when
        enabled). For ``opponent_model: raw`` we extract raw
        ``trainer.model.state_dict()``. The snapshot lives in
        ``checkpoints/refresh_ema_snapshot.pt`` and is unlinked after the
        subprocess completes.

        Returns ``(snapshot_path, sha256)``.
        """
        import hashlib as _hashlib
        import torch as _torch
        cfg = self.config
        anchor_dir = canonical.parent.parent / "checkpoints"
        if not anchor_dir.exists():
            anchor_dir = canonical.parent
        snapshot_path = anchor_dir / "refresh_ema_snapshot.pt"

        if cfg.bot_corpus_refresh_opponent_model == "ema":
            sd = self.trainer.inference_state_dict()
        else:
            base = getattr(self.trainer.model, "_orig_mod", self.trainer.model)
            sd = base.state_dict()
        # Wrap in load_inference_model-compatible format: bare state_dict accepted
        # by load_inference_model (corpus generator loads via this entry).
        _torch.save(sd, snapshot_path)
        # Compute sha for forensic log.
        h = _hashlib.sha256()
        with snapshot_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return snapshot_path, h.hexdigest()

    def _launch_refresh_subprocess(
        self,
        canonical: Path,
    ) -> None:
        """Launch the async regen subprocess + record state for poll().

        Trainer pause budget: ≤ 200ms per V179c-1 verdict. Heavy work
        (game generation) happens in the subprocess — this function only
        snapshots weights + spawns. Emits ``bot_corpus_regen_requested``.
        """
        import subprocess as _subprocess
        cfg = self.config
        # Snapshot opponent-model weights to a transient .pt; subprocess
        # consumes this as --anchor.
        anchor_snapshot, anchor_sha = self._save_refresh_anchor_snapshot(canonical)
        tmp_npz = canonical.with_name(canonical.name + ".NEW.tmp.npz")
        # Clean any stale tmp from a prior crash (s179c §8 risk #6).
        if tmp_npz.exists():
            tmp_npz.unlink()
        cmd = self._build_refresh_subprocess_command(canonical, tmp_npz, anchor_snapshot)
        proc = _subprocess.Popen(  # noqa: S603 — argv list, paths validated
            cmd,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.PIPE,
            text=True,
        )
        self._refresh_proc = proc
        self._refresh_started_step = self._train_step
        self._refresh_target_anchor_sha = anchor_sha
        self._refresh_ema_snapshot_path = anchor_snapshot
        self._refresh_tmp_npz_path = tmp_npz
        self._event_emitter({
            "event": "bot_corpus_regen_requested",
            "step": self._train_step,
            "trigger": "force" if self._force_bot_refresh else "interval",
            "anchor_sha": anchor_sha,
            "subprocess_pid": proc.pid,
            "interval_steps": cfg.bot_corpus_refresh_interval_steps,
            "opponent_model": cfg.bot_corpus_refresh_opponent_model,
            "n_games": cfg.bot_corpus_refresh_n_games,
        })
        self._logger.info(
            "bot_corpus_regen_requested",
            step=self._train_step,
            anchor_sha=anchor_sha[:12],
            subprocess_pid=proc.pid,
            n_games=cfg.bot_corpus_refresh_n_games,
            opponent_model=cfg.bot_corpus_refresh_opponent_model,
        )

    def _drop_refresh_state(self) -> None:
        """Clear in-flight refresh state + cleanup transient files."""
        # Drop EMA snapshot (transient — only the subprocess needed it).
        if self._refresh_ema_snapshot_path is not None:
            try:
                self._refresh_ema_snapshot_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._refresh_proc = None
        self._refresh_target_anchor_sha = None
        self._refresh_ema_snapshot_path = None
        self._refresh_tmp_npz_path = None

    def _swap_and_hot_reload_bot_corpus(self, canonical: Path) -> None:
        """Atomic NPZ swap + hot-reload of self.bot_buffer (s179c §3 + §4).

        Called only when subprocess returncode == 0.
        """
        from hexo_rl.training.batch_assembly import (
            BotCorpusSwapError,
            load_bot_corpus_buffer,
            swap_bot_corpus_atomic,
        )
        import time as _time

        assert self._refresh_tmp_npz_path is not None
        # Step 1 — atomic swap (sha-verified).
        try:
            old_sha, new_sha = swap_bot_corpus_atomic(
                canonical_path=canonical,
                tmp_path=self._refresh_tmp_npz_path,
            )
        except BotCorpusSwapError as exc:
            self._logger.warning(
                "bot_corpus_regen_failed",
                step=self._train_step,
                returncode=0,
                reason=f"swap_failed: {exc}",
            )
            self._event_emitter({
                "event": "bot_corpus_regen_failed",
                "step": self._train_step,
                "returncode": 0,
                "reason": f"swap_failed: {exc}",
            })
            return
        self._event_emitter({
            "event": "bot_corpus_swap_committed",
            "step": self._train_step,
            "old_npz_sha": old_sha,
            "new_npz_sha": new_sha,
        })
        self._logger.info(
            "bot_corpus_swap_committed",
            step=self._train_step,
            old_sha=old_sha[:12] if old_sha else "(none)",
            new_sha=new_sha[:12],
        )

        # Step 2 — hot-reload (s179c §4.2). Drop ref + reload.
        old_size = getattr(self.bot_buffer, "size", 0)
        self.bot_buffer = None
        import gc as _gc
        _gc.collect()
        _t0 = _time.time()
        self.bot_buffer = load_bot_corpus_buffer(
            self.mixing_cfg, self.full_config, self._event_emitter,
            self.buffer.size, self.buffer.capacity,
        )
        reload_sec = _time.time() - _t0
        new_size = getattr(self.bot_buffer, "size", 0)
        self._event_emitter({
            "event": "bot_corpus_hot_reload",
            "step": self._train_step,
            "old_n_positions": old_size,
            "new_n_positions": new_size,
            "reload_sec": round(reload_sec, 3),
        })
        self._logger.info(
            "bot_corpus_hot_reload",
            step=self._train_step,
            old_n_positions=old_size,
            new_n_positions=new_size,
            reload_sec=round(reload_sec, 3),
        )

    def _tick_bot_refresh(self) -> None:
        """One refresh-hook tick at the eval boundary (s179c §2.2 + §3 + §4).

        Three sub-paths:
          (a) An in-flight subprocess exists → poll(); on terminal handle
              completion (sha + atomic swap + hot-reload) or failure (log).
          (b) Trigger predicate fires → snapshot anchor + Popen the subprocess.
          (c) Neither → no-op.

        All paths preserve canonical NPZ on any failure (s179c §2.3).
        """
        cfg = self.config

        # Sub-path (a) — handle an in-flight subprocess.
        if self._refresh_proc is not None:
            import time as _time
            rc = self._refresh_proc.poll()
            if rc is None:
                return  # still running; check again next eval boundary
            elapsed_sec = max(self._train_step - self._refresh_started_step, 0)
            if rc == 0:
                # Subprocess succeeded — perform swap + hot-reload.
                try:
                    canonical = self._resolve_canonical_bot_path()
                    if canonical is None:
                        # Configuration changed mid-run; bail clean.
                        self._logger.warning(
                            "bot_corpus_regen_no_canonical_path",
                            step=self._train_step,
                        )
                    else:
                        # Wrapped swap + hot-reload. Emit completion event
                        # AFTER the swap commits successfully.
                        n_pos_before = getattr(self.bot_buffer, "size", 0)
                        self._swap_and_hot_reload_bot_corpus(canonical)
                        n_pos_after = getattr(self.bot_buffer, "size", 0)
                        self._event_emitter({
                            "event": "bot_corpus_regen_complete",
                            "step": self._train_step,
                            "returncode": 0,
                            "elapsed_steps": elapsed_sec,
                            "n_positions": n_pos_after,
                        })
                        self._logger.info(
                            "bot_corpus_regen_complete",
                            step=self._train_step,
                            returncode=0,
                            elapsed_steps=elapsed_sec,
                            n_positions_before=n_pos_before,
                            n_positions_after=n_pos_after,
                        )
                        self._n_refreshes_so_far += 1
                        self._last_bot_refresh_step = self._train_step
                except Exception as exc:  # noqa: BLE001
                    # Defensive: never crash the trainer over a refresh failure.
                    self._logger.warning(
                        "bot_corpus_regen_completion_failed",
                        step=self._train_step,
                        error=str(exc),
                    )
            else:
                # Subprocess non-zero. Per s179c §2.3 retain canonical NPZ;
                # do NOT bump n_refreshes_so_far (retry possible at next cadence).
                stderr_excerpt = ""
                try:
                    if self._refresh_proc.stderr is not None:
                        stderr_excerpt = self._refresh_proc.stderr.read()[-512:]
                except Exception:
                    pass
                self._logger.warning(
                    "bot_corpus_regen_failed",
                    step=self._train_step,
                    returncode=rc,
                    reason="subprocess_nonzero_exit",
                    stderr_tail=stderr_excerpt,
                )
                self._event_emitter({
                    "event": "bot_corpus_regen_failed",
                    "step": self._train_step,
                    "returncode": rc,
                    "reason": "subprocess_nonzero_exit",
                })
            # Whatever the outcome (0 or non-zero), tear down refresh state.
            self._drop_refresh_state()
            self._force_bot_refresh = False
            return

        # Sub-path (b) — no in-flight; check trigger.
        if self.bot_buffer is None:
            return  # nothing to refresh into
        if self._n_refreshes_so_far >= cfg.bot_corpus_refresh_max_regens:
            # Hard cap reached; silent skip (rate-limited log to avoid spam).
            return
        interval_ready = (
            self._train_step - self._last_bot_refresh_step
        ) >= cfg.bot_corpus_refresh_interval_steps
        if not (interval_ready or self._force_bot_refresh):
            return

        # Fire — snapshot anchor + Popen subprocess.
        try:
            canonical = self._resolve_canonical_bot_path()
            if canonical is None:
                self._logger.warning(
                    "bot_corpus_refresh_no_canonical_path_at_fire",
                    step=self._train_step,
                )
                self._force_bot_refresh = False
                return
            self._launch_refresh_subprocess(canonical)
        except Exception as exc:  # noqa: BLE001
            # Defensive: never crash the trainer over refresh launch.
            self._logger.warning(
                "bot_corpus_refresh_launch_failed",
                step=self._train_step,
                error=str(exc),
            )
            self._event_emitter({
                "event": "bot_corpus_regen_failed",
                "step": self._train_step,
                "returncode": -1,
                "reason": f"launch_failed: {exc}",
            })
            self._drop_refresh_state()
            self._force_bot_refresh = False

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
