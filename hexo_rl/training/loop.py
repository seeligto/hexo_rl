"""Main AlphaZero self-play training loop.

Canonical home for ``run_training_loop``, which owns everything from
inference-model construction through the post-loop checkpoint / buffer save.

Callers (``scripts/train.py``) build the core objects (Trainer, buffer,
pretrained_buffer, recent_buffer, BatchBuffers) and then hand control here.
Inference model, WorkerPool, dashboards, GPU monitor, and eval pipeline are
all created and torn down inside this module.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.events import emit_event
from hexo_rl.training.anchor import resolve_anchor
from hexo_rl.training.batch_assembly import BatchBuffers, assemble_mixed_batch
from hexo_rl.training.lifecycle import (
    build_eval_model,
    build_inference_model,
    build_subsystems,
    cuda_stream_audit,
    cuda_warmup,
)
# Orchestrator hooks — public in `hexo_rl.training.orchestrator`, aliased here
# so the existing call sites in `run_training_loop` keep their underscore-prefixed
# local names. Pure-move §159 left this layer untouched; refactoring the inner
# call sites to drop the underscores is a follow-up.
from hexo_rl.training.orchestrator import (
    drain_pending_eval as _drain_pending_eval,
    emit_axis_distribution as _emit_axis_distribution,
    emit_training_events as _emit_training_events,
    replay_pretrain_events as _replay_pretrain_events,
    try_save_buffer as _try_save_buffer,
)
from hexo_rl.training.signals import ShutdownState, install_signal_handlers
from hexo_rl.training.trainer import Trainer

log = structlog.get_logger(__name__)


def _compute_pretrained_weight(step: int, initial_w: float, min_w: float, decay_steps: float) -> float:
    return max(min_w, initial_w * math.exp(-step / decay_steps))


def _steps_budget(new_games: int, training_steps_per_game: float, max_train_burst: int) -> int:
    return min(max(1, round(new_games * training_steps_per_game)), max_train_burst)


class RollingGamesPerHour:
    def __init__(self, t_start: float, window_seconds: float = 60.0) -> None:
        self.t_start = t_start
        self.window_seconds = window_seconds
        self._window: list[tuple[float, int]] = []

    def update(self, now: float, games_played: int) -> float:
        self._window.append((now, games_played))
        cutoff = now - self.window_seconds
        while self._window and self._window[0][0] < cutoff:
            self._window.pop(0)
        if len(self._window) < 2:
            elapsed = max(now - self.t_start, 1e-6)
            return games_played / elapsed * 3600
        dt = self._window[-1][0] - self._window[0][0]
        dg = self._window[-1][1] - self._window[0][1]
        return (dg / max(dt, 1e-6)) * 3600


def run_training_loop(
    trainer: Trainer,
    buffer: Any,                          # ReplayBuffer (self-play)
    pretrained_buffer: Optional[Any],     # ReplayBuffer | None
    recent_buffer: Optional[Any],         # RecentBuffer | None
    bufs: BatchBuffers,
    config: dict[str, Any],              # fully merged combined_config
    train_cfg: dict[str, Any],           # config["training"] (or config)
    mcts_config: dict[str, Any],         # config["mcts"] (or {})
    args: argparse.Namespace,
    device: torch.device,
    run_id: str,
    capacity: int,
    min_buf_size: int,
    buffer_schedule: list[dict[str, int]],
    recency_weight: float,
    batch_size_cfg: int,
    mixing_cfg: dict[str, Any],
    mixing_initial_w: float,
    mixing_min_w: float,
    mixing_decay_steps: float,
) -> None:
    """Drive the self-play / train / eval / checkpoint cycle until stopped.

    Responsibilities:
      - Build the inference model (weights copied from trainer.model).
      - Build WorkerPool (Rust workers + InferenceServer).
      - Start GPU monitor, dashboard renderers, and eval pipeline.
      - Start the worker pool.
      - Run the main loop (warmup → mixed-batch sample → train step →
        emit events → eval → checkpoint).
      - Save final checkpoint and buffer on exit.

    Args:
        trainer:             Trainer instance with model, optimizer, scheduler.
        buffer:              Main self-play ReplayBuffer.
        pretrained_buffer:   Corpus ReplayBuffer (None if no corpus).
        recent_buffer:       Optional RecentBuffer for recency weighting.
        bufs:                Pre-allocated batch arrays (see BatchBuffers).
        config:              Fully merged combined_config dict.
        train_cfg:           ``config["training"]`` sub-dict (or config).
        mcts_config:         ``config["mcts"]`` sub-dict (or {}).
        args:                Parsed CLI args (checkpoint_dir, no_dashboard, …).
        device:              Torch device.
        run_id:              UUID hex string for this run.
        capacity:            Initial buffer capacity (for events).
        min_buf_size:        Minimum buffer fill before training begins.
        buffer_schedule:     Sorted list of {step, capacity} growth entries.
        recency_weight:      Fraction of self-play batch drawn from recent_buffer.
        batch_size_cfg:      Pre-allocated batch size (from training config).
        mixing_cfg:          ``train_cfg["mixing"]`` sub-dict.
        mixing_initial_w:    Starting corpus mixing weight.
        mixing_min_w:        Floor corpus mixing weight.
        mixing_decay_steps:  Exponential decay constant for corpus weight.
    """

    # ── Inference model + arch ────────────────────────────────────────────────
    inf_model, _arch = build_inference_model(trainer, device)
    board_size = _arch.board_size
    res_blocks = _arch.res_blocks
    filters = _arch.filters
    in_channels = _arch.in_channels
    se_reduction_ratio = _arch.se_reduction_ratio
    input_channels = _arch.input_channels
    _ckpt_interval = int(trainer.config.get("checkpoint_interval", 500))

    cuda_warmup(inf_model, device, board_size)
    cuda_stream_audit(config, device)

    # ── Worker pool ───────────────────────────────────────────────────────────
    from hexo_rl.selfplay.pool import WorkerPool
    pool = WorkerPool(inf_model, config, device, buffer)
    if recent_buffer is not None:
        pool.recent_buffer = recent_buffer

    # ── Eval pipeline ─────────────────────────────────────────────────────────
    from hexo_rl.eval.pipeline_setup import build_eval_pipeline
    eval_pipeline, eval_ext_config, _eval_interval_cfg = build_eval_pipeline(
        config, device, run_id, train_cfg,
    )

    # ── Anchor resolution ─────────────────────────────────────────────────────
    _anchor = resolve_anchor(
        eval_pipeline=eval_pipeline,
        eval_ext_config=eval_ext_config,
        inf_model=inf_model,
        trainer=trainer,
        args=args,
        config=config,
        device=device,
        board_size=board_size,
        res_blocks=res_blocks,
        filters=filters,
        in_channels=in_channels,
        input_channels=input_channels,
        se_reduction_ratio=se_reduction_ratio,
    )
    best_model = _anchor.best_model
    best_model_step = _anchor.best_model_step
    best_model_path = _anchor.best_model_path

    _eval_thread: threading.Thread | None = None
    _eval_result: list[dict | None] = [None]

    # L1: allocate the eval-side model once and reuse across rounds. Reallocating
    # every 2500 steps churned ~30 MB of CUDA activations per round on 12-block
    # × 128-ch and leaned on the allocator for no benefit.
    # C1: this model is also the source of truth for what weights get promoted.
    # We drain the previous eval's result BEFORE overwriting eval_model for the
    # next round, so holding a reference here is safe — the weights still in
    # eval_model when `promoted=True` is drained are the weights that actually
    # passed the gate, not whatever trainer.model has drifted to since.
    eval_model: HexTacToeNet | None = None
    if eval_pipeline is not None:
        eval_model = build_eval_model(_arch, device)

    # ── Subsystems ───────────────────────────────────────────────────────────
    subsys = build_subsystems(args, config, device, run_id)
    gpu_monitor = subsys.gpu_monitor
    disk_guard = subsys.disk_guard
    early_game_probe = subsys.early_game_probe
    value_probe = subsys.value_probe
    value_probe_interval = subsys.value_probe_interval
    composition_interval = subsys.composition_interval
    instrumentation_enabled = subsys.instrumentation_enabled
    _axis_baseline = subsys.axis_baseline
    _tb_writer = subsys.tb_writer
    dashboards = subsys.dashboards

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    _shutdown = ShutdownState()
    install_signal_handlers(_shutdown)

    # ── Loop setup ────────────────────────────────────────────────────────────
    log_interval = int(config.get("log_interval", 10))
    eval_interval = int(train_cfg.get("eval_interval", config.get("eval_interval", _eval_interval_cfg)))
    training_steps_per_game = float(train_cfg.get("training_steps_per_game", 1.0))
    max_train_burst          = int(train_cfg.get("max_train_burst", 8))
    if "augment" not in train_cfg and "augment" not in config:
        raise ValueError(
            "training.augment missing from merged config — required key per "
            "configs/training.yaml. Set `augment: true` for production (preserves "
            "12-fold hex symmetry augmentation) or `augment: false` for diagnostic runs."
        )
    augment_cfg = bool(train_cfg.get("augment", config.get("augment")))
    train_step = trainer.step
    stop_step  = (trainer.step + args.iterations) if args.iterations else None
    games_played  = 0
    t_start       = time.time()
    initial_policy_loss: float | None = None
    last_loss_info: dict[str, float] = {}

    # ── Abort sentinels ───────────────────────────────────────────────────────
    import collections as _collections
    _mon_cfg = config.get("monitors", {})
    # Soft-abort: E-W axis (axis_q) flat above threshold for N consecutive evals.
    _soft_ew_threshold = float(_mon_cfg.get("soft_abort_ew_threshold", 0.0))
    _soft_ew_min_pts   = int(_mon_cfg.get("soft_abort_ew_min_points", 0))
    _ew_history: collections.deque = _collections.deque(maxlen=max(_soft_ew_min_pts, 1))
    # Hard-abort: gradient norm > threshold for N consecutive training steps.
    _hard_gn_threshold  = float(_mon_cfg.get("hard_abort_grad_norm", 3.0))
    _hard_gn_min_steps  = int(_mon_cfg.get("hard_abort_grad_norm_steps", 5))
    _consec_high_gn     = 0

    # ── Emit run_start ─────────────────────────────────────────────────────────
    emit_event({
        "event": "run_start",
        "step": train_step,
        "run_id": run_id,
        "worker_count": pool.n_workers,
        "config_summary": {
            "n_blocks": int(config.get("res_blocks", 12)),
            "channels": int(config.get("filters", 128)),
            "n_sims": int(mcts_config.get("n_simulations", 800)),
            "buffer_capacity": capacity,
        },
    })

    if args.checkpoint and "pretrain" in Path(args.checkpoint).name:
        _replay_pretrain_events(args)

    # ── Start self-play pool ───────────────────────────────────────────────────
    pool.start()
    log.info("selfplay_pool_started", n_workers=pool.n_workers)

    # ── Rolling games/hour accumulator ────────────────────────────────────────
    _rolling_gph = RollingGamesPerHour(t_start)

    def _games_per_hour_rolling() -> float:
        return _rolling_gph.update(time.time(), games_played)

    # ── Main iteration ─────────────────────────────────────────────────────────
    schedule_idx = 1  # first schedule entry already applied at buffer construction

    def _run_loop() -> None:
        nonlocal train_step, games_played, initial_policy_loss, last_loss_info
        nonlocal _eval_thread, best_model, best_model_step, schedule_idx

        last_train_game_count  = 0
        last_warmup_log        = 0.0
        last_iter_games        = 0
        _last_quiescence_fires = 0
        # MCTS pool-overflow counter snapshot (Tier-1.A from 2026-04-28
        # 5090 sweep analyst v2). Each overflow fabricates a terminal
        # value at the leaf and propagates it through backup() — biases
        # visit counts and policy/value training targets. Counter is
        # global across all worker threads; sample at log cadence and
        # warn loudly on growth so silent training-data corruption gets
        # caught early. Math says overflow shouldn't happen with 400
        # sims × ~80 legal moves; if delta is non-zero, that's a real
        # engine bug (or a pathological board state) worth investigating.
        _last_pool_overflows = 0
        nonlocal _consec_high_gn

        while _shutdown.running:
            if stop_step is not None and train_step >= stop_step:
                log.info("iteration_limit_reached", iterations=args.iterations)
                break

            if _shutdown.shutdown_save:
                log.info(
                    "shutdown_signal_checkpoint",
                    msg="Shutdown signal received — saving checkpoint before exit",
                    step=train_step,
                )
                trainer.save_checkpoint(last_loss_info if last_loss_info else None)
                _try_save_buffer(buffer, mixing_cfg, "shutdown_signal", recent_buffer)
                break

            games_played = pool.games_completed

            if buffer.size < min_buf_size:
                if (time.time() - last_warmup_log) >= 5.0:
                    structlog.get_logger().info(
                        "warmup",
                        buffer=buffer.size,
                        target=min_buf_size,
                        games=games_played,
                        gpu_pct=round(gpu_monitor.gpu_util_pct, 0),
                    )
                    emit_event({
                        "event": "system_stats",
                        "buffer_size": buffer.size,
                        "buffer_capacity": capacity,
                    })
                    last_warmup_log = time.time()
                time.sleep(0.5)
                continue

            new_games = games_played - last_train_game_count
            if new_games <= 0:
                if (time.time() - last_warmup_log) >= 5.0:
                    structlog.get_logger().info(
                        "waiting_for_games",
                        games=games_played,
                        trained_games=last_train_game_count,
                        buffer=buffer.size,
                    )
                    last_warmup_log = time.time()
                time.sleep(0.1)
                continue

            steps_budget = _steps_budget(new_games, training_steps_per_game, max_train_burst)
            last_train_game_count = games_played

            for _ in range(steps_budget):
                if stop_step is not None and train_step >= stop_step:
                    break

                # Buffer growth schedule.
                while (schedule_idx < len(buffer_schedule)
                       and train_step >= buffer_schedule[schedule_idx]["step"]):
                    new_cap = buffer_schedule[schedule_idx]["capacity"]
                    if new_cap > buffer.capacity:
                        buffer.resize(new_cap)
                        log.info("buffer_resized", step=train_step, new_capacity=new_cap)
                    schedule_idx += 1

                # ── Training step ─────────────────────────────────────────────
                batch_size = int(train_cfg.get("batch_size", config.get("batch_size", 256)))
                if pretrained_buffer is not None and pretrained_buffer.size > 0 and buffer.size > 0:
                    w_pre  = _compute_pretrained_weight(train_step, mixing_initial_w, mixing_min_w, mixing_decay_steps)
                    n_pre  = max(1, int(math.ceil(batch_size * w_pre)))
                    n_self = batch_size - n_pre
                    (states, chain_planes, policies, outcomes,
                     ownership, winning_line, is_full_search,
                     n_recent_batch) = assemble_mixed_batch(
                        pretrained_buffer, buffer, recent_buffer,
                        n_pre, n_self, batch_size, batch_size_cfg,
                        recency_weight, bufs, train_step,
                        augment=augment_cfg,
                    )
                    loss_info = trainer.train_step_from_tensors(
                        states, policies, outcomes,
                        chain_planes=chain_planes,
                        ownership_targets=ownership, threat_targets=winning_line,
                        is_full_search=is_full_search,
                        n_pretrain=n_pre,
                        n_recent=n_recent_batch,
                    )
                else:
                    w_pre = 0.0
                    loss_info = trainer.train_step(
                        buffer,
                        augment=augment_cfg,
                        recent_buffer=recent_buffer,
                    )

                train_step = trainer.step
                if initial_policy_loss is None:
                    initial_policy_loss = float(loss_info["policy_loss"])
                last_loss_info = loss_info

                # ── Hard-abort: sustained gradient norm ───────────────────────
                _step_gn = float(loss_info.get("grad_norm", 0.0))
                if math.isfinite(_step_gn) and _step_gn > _hard_gn_threshold:
                    _consec_high_gn += 1
                    if _consec_high_gn >= _hard_gn_min_steps:
                        log.error(
                            "hard_abort_grad_norm",
                            step=train_step,
                            consec_steps=_consec_high_gn,
                            grad_norm=round(_step_gn, 4),
                            threshold=_hard_gn_threshold,
                            msg="Sustained high gradient norm — halting run. Roll back to ckpt_12190 before re-resuming.",
                        )
                        _shutdown.running = False
                else:
                    _consec_high_gn = 0

                if train_step % _ckpt_interval == 0:
                    # Graduation gate: inf_model is the anchor, not a mirror of
                    # trainer.model. Do NOT sync on checkpoint cadence — sync only
                    # when a new model beats the anchor (promotion branch below).
                    if train_step > 0:
                        _try_save_buffer(buffer, mixing_cfg, "checkpoint_interval", recent_buffer)

                # ── Selfplay axis-distribution (§axis_dist) ───────────────────
                if train_step > 0 and train_step % eval_interval == 0:
                    _axis_q_val = _emit_axis_distribution(
                        train_step, pool, config, _axis_baseline, _tb_writer,
                    )
                    # Soft-abort: E-W fraction flat above threshold for N evals.
                    if (_soft_ew_threshold > 0.0 and _soft_ew_min_pts > 0
                            and _axis_q_val is not None):
                        _ew_history.append(_axis_q_val)
                        if (len(_ew_history) >= _soft_ew_min_pts
                                and all(v > _soft_ew_threshold for v in _ew_history)):
                            log.warning(
                                "soft_abort_ew_flat",
                                step=train_step,
                                ew_history=[round(v, 4) for v in _ew_history],
                                threshold=_soft_ew_threshold,
                                msg=(
                                    "E-W fraction flat above threshold — soft-abort. "
                                    "Commit checkpoint and open §120 investigation."
                                ),
                            )
                            _shutdown.running = False

                # ── Eval (non-blocking background thread) ─────────────────────
                if eval_pipeline is not None and train_step > 0 and train_step % eval_interval == 0:
                    _eval_thread, best_model_step = _drain_pending_eval(
                        _eval_thread, _eval_result, eval_model, best_model,
                        best_model_path, best_model_step, pool, train_step,
                    )

                    if _eval_thread is None or not _eval_thread.is_alive():
                        base_model = getattr(trainer.model, "_orig_mod", trainer.model)
                        assert eval_model is not None
                        _eval_base = getattr(eval_model, "_orig_mod", eval_model)
                        _eval_base.load_state_dict(base_model.state_dict())
                        step_snapshot = train_step
                        log.info("evaluation_start", step=step_snapshot)

                        def _run_eval(
                            _model: HexTacToeNet = eval_model,
                            _step: int = step_snapshot,
                            _best: HexTacToeNet | None = best_model,
                            _cfg: dict = config,
                            _best_step: int | None = best_model_step,
                        ) -> None:
                            try:
                                # L4: run_evaluation sets result["step"] itself;
                                # the post-hoc assignment here was dead code.
                                result = eval_pipeline.run_evaluation(
                                    _model, _step, _best, full_config=_cfg,
                                    best_model_step=_best_step,
                                )
                                _eval_result[0] = result
                            except Exception:
                                import traceback
                                log.info("evaluation_error", step=_step, tb=traceback.format_exc())
                                _eval_result[0] = {"promoted": False, "error": True, "step": _step}

                        _eval_thread = threading.Thread(target=_run_eval, daemon=True)
                        _eval_thread.start()
                    else:
                        log.info("eval_skipped_still_running", step=train_step)

                # ── tracemalloc snapshot every 500 steps ──────────────────────
                if train_step > 0 and train_step % 500 == 0:
                    try:
                        snapshot  = tracemalloc.take_snapshot()
                        top_stats = snapshot.statistics("lineno")[:10]
                        log.info(
                            "tracemalloc_top10",
                            step=train_step,
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
                        log.warning("tracemalloc_failed", error=str(_tm_err))

                # ── Emit training events ───────────────────────────────────────
                if train_step % log_interval == 0:
                    _cur_qfire    = int(getattr(pool._runner, "mcts_quiescence_fires", 0))
                    _qfire_delta  = _cur_qfire - _last_quiescence_fires
                    _last_quiescence_fires = _cur_qfire
                    # Pool-overflow surface — soft warning only, not hard-fail.
                    # We don't know the production frequency yet; aborting
                    # training on the first occurrence would risk killing
                    # otherwise-healthy runs over a benign tick. Promote to
                    # hard-fail once a threshold has been calibrated.
                    try:
                        from engine import mcts_pool_overflow_count
                        _cur_pool_overflows = int(mcts_pool_overflow_count())
                    except (ImportError, AttributeError):
                        # Engine wheel pre-dates Tier-1.A counter — silent skip.
                        _cur_pool_overflows = 0
                    _pool_overflow_delta = _cur_pool_overflows - _last_pool_overflows
                    _last_pool_overflows = _cur_pool_overflows
                    if _pool_overflow_delta > 0:
                        log.warning(
                            "mcts_pool_overflow",
                            step=train_step,
                            delta=_pool_overflow_delta,
                            cumulative=_cur_pool_overflows,
                            msg=(
                                "MCTS pool overflow — engine fabricated terminal "
                                "values, biasing training targets. Math says this "
                                "should never happen; investigate board state at "
                                "next occurrence."
                            ),
                        )
                        emit_event({
                            "event": "mcts_pool_overflow",
                            "step": train_step,
                            "delta": _pool_overflow_delta,
                            "cumulative": _cur_pool_overflows,
                        })
                    _emit_training_events(
                        train_step, loss_info, w_pre, games_played,
                        last_iter_games, pool, buffer, gpu_monitor,
                        config, mcts_config, capacity,
                        _games_per_hour_rolling, _qfire_delta,
                        early_game_probe=early_game_probe,
                        trainer_model=trainer.model,
                    )
                    last_iter_games = games_played

                # ── Phase B' instrumentation events ──────────────────────────
                # Cadenced separately from log_interval so the user can choose
                # finer-grained probes without flooding the dashboard.
                if instrumentation_enabled:
                    if value_probe is not None and (train_step > 0
                            and train_step % value_probe_interval == 0):
                        try:
                            vp = value_probe.compute(trainer.model)
                            emit_event({
                                "event": "value_probe_drift",
                                "step": train_step,
                                "decisive_mean": vp["decisive_mean"],
                                "decisive_std":  vp["decisive_std"],
                                "draw_mean":     vp["draw_mean"],
                                "draw_std":      vp["draw_std"],
                                "n_decisive":    vp["decisive_n"],
                                "n_draw":        vp["draw_n"],
                                "fixture":       value_probe.fixture_path,
                            })
                            log.info(
                                "value_probe_drift",
                                step=train_step,
                                decisive_mean=round(vp["decisive_mean"], 4),
                                draw_mean=round(vp["draw_mean"], 4),
                            )
                        except Exception as _vp_err:
                            log.warning(
                                "value_probe_failed", step=train_step,
                                error=str(_vp_err),
                            )

                    if train_step > 0 and train_step % composition_interval == 0:
                        try:
                            comp = pool.buffer_composition()
                            emit_event({
                                "event": "buffer_composition",
                                "step": train_step,
                                **comp,
                            })
                            mvs = pool.model_version_summary()
                            wdr = pool.per_worker_draw_rates()
                            emit_event({
                                "event": "worker_draw_rate",
                                "step": train_step,
                                "per_worker": {str(k): round(v, 4)
                                               for k, v in wdr.items()},
                                "n_workers_observed": len(wdr),
                            })
                            emit_event({
                                "event": "model_version_summary",
                                "step": train_step,
                                **mvs,
                                "current_version": int(
                                    getattr(pool._runner, "model_version", 0),
                                ),
                            })
                            log.info(
                                "instrumentation_periodic",
                                step=train_step,
                                draw_target_fraction=comp.get("draw_target_fraction"),
                                colony_terminal_fraction=comp.get("colony_terminal_fraction"),
                                six_terminal_fraction=comp.get("six_terminal_fraction"),
                                cap_terminal_fraction=comp.get("cap_terminal_fraction"),
                                mv_median_range=mvs.get("median_range"),
                                mv_p90_range=mvs.get("p90_range"),
                                mv_spearman_rho=mvs.get("spearman_rho_range_vs_draw"),
                                n_workers_observed=len(wdr),
                            )
                        except Exception as _instr_err:
                            log.warning(
                                "instrumentation_emit_failed",
                                step=train_step,
                                error=str(_instr_err),
                            )

    # ── Run and teardown ──────────────────────────────────────────────────────
    tracemalloc.start(3)
    try:
        _run_loop()
    finally:
        tracemalloc.stop()
        # When stop_step is reached and an eval was kicked off at the same
        # tick (eval_interval == stop_step, common in sweep mode), the
        # eval thread is still running here. Without this join the daemon
        # thread is killed at process exit and wr_best never persists,
        # which silently breaks downstream selection. Wait up to
        # `eval_final_drain_timeout_sec` (variant override; default 0 = no
        # wait, preserving production Ctrl-C semantics).
        _final_drain_timeout = float(
            train_cfg.get("eval_final_drain_timeout_sec",
                          config.get("eval_final_drain_timeout_sec", 0.0))
        )
        if (_final_drain_timeout > 0.0
                and _eval_thread is not None and _eval_thread.is_alive()
                and not _shutdown.shutdown_save):
            log.info("final_eval_drain_waiting", timeout_sec=_final_drain_timeout)
            _eval_thread.join(timeout=_final_drain_timeout)
            if _eval_thread.is_alive():
                log.warning("final_eval_drain_timeout",
                            timeout_sec=_final_drain_timeout,
                            msg="eval still running past timeout — proceeding to shutdown")
        # D-012: if the last eval completed but training ended before the
        # next tick would drain it, persist the promotion now so graduations
        # aren't silently dropped on shutdown. pool is still up — the
        # inference-server weight sync inside the drain needs it.
        try:
            _eval_thread, best_model_step = _drain_pending_eval(
                _eval_thread, _eval_result, eval_model, best_model,
                best_model_path, best_model_step, pool, train_step,
            )
        except Exception:
            log.warning("final_eval_drain_failed", exc_info=True)
        emit_event({"event": "run_end", "step": train_step})
        pool.stop()
        subsys.teardown()

    # ── Final checkpoint + buffer save ────────────────────────────────────────
    final_ckpt = trainer.save_checkpoint(last_loss_info if last_loss_info else None)
    _try_save_buffer(buffer, mixing_cfg, "session_end", recent_buffer)

    log.info(
        "session_end",
        final_step=trainer.step,
        games_played=games_played,
        buffer_size=buffer.size,
        initial_policy_loss=initial_policy_loss,
        final_policy_loss=last_loss_info.get("policy_loss") if last_loss_info else None,
        elapsed_sec=round(time.time() - t_start, 1),
        final_checkpoint=str(final_ckpt),
    )
    print(f"\nSaved checkpoint: {final_ckpt}")
    print(f"Games: {games_played}  Steps: {trainer.step}  Buffer: {buffer.size}")
    if initial_policy_loss is not None and last_loss_info:
        decrease = initial_policy_loss - last_loss_info["policy_loss"]
        print(
            f"Policy loss: {initial_policy_loss:.4f} → {last_loss_info['policy_loss']:.4f} "
            f"({'↓' if decrease > 0 else '↑'}{abs(decrease):.4f})"
        )

    _suppress_semaphore_leak_warning()


def _suppress_semaphore_leak_warning() -> None:
    """Silence spurious 'leaked semaphore' warning on exit from PyO3 condvars."""
    try:
        import multiprocessing.resource_tracker
        multiprocessing.resource_tracker._resource_tracker._stop()
    except Exception:
        pass
