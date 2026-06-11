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
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.events import emit_event
from hexo_rl.training.anchor import resolve_anchor
from hexo_rl.training.batch_assembly import BatchBuffers
from hexo_rl.training.lifecycle import (
    build_eval_model,
    build_inference_model,
    build_subsystems,
    cuda_stream_audit,
    cuda_warmup,
)
from hexo_rl.training.model_defaults import MODEL_HPARAM_DEFAULTS
from hexo_rl.training.buffer_persist import try_save_buffer as _try_save_buffer
from hexo_rl.training.events import replay_pretrain_events as _replay_pretrain_events
from hexo_rl.training.signals import ShutdownState, install_signal_handlers
from hexo_rl.training.trainer import Trainer

log = structlog.get_logger(__name__)


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
    bot_buffer: Optional[Any] = None,     # §178 ReplayBuffer | None (bot-corpus slot)
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
    stop_step  = (trainer.step + args.iterations) if args.iterations else None
    t_start    = time.time()

    # ── Abort sentinels ───────────────────────────────────────────────────────
    _mon_cfg = config.get("monitors", {})
    # Soft-abort: E-W axis (axis_q) flat above threshold for N consecutive evals.
    _soft_ew_threshold = float(_mon_cfg.get("soft_abort_ew_threshold", 0.0))
    _soft_ew_min_pts   = int(_mon_cfg.get("soft_abort_ew_min_points", 0))
    # Hard-abort: gradient norm > threshold for N consecutive training steps.
    _hard_gn_threshold = float(_mon_cfg.get("hard_abort_grad_norm", 3.0))
    _hard_gn_min_steps = int(_mon_cfg.get("hard_abort_grad_norm_steps", 5))
    # §CANARY-VAL — stride-5 spam hard-abort (default-on; validated FPR=0 on all
    # radius-5 runs). Set hard_abort_stride5_p90 ≤ 0 to disable.
    _stride5_p90_threshold = float(_mon_cfg.get("hard_abort_stride5_p90", 30.0))
    _stride5_p90_consec    = int(_mon_cfg.get("hard_abort_stride5_p90_consec", 3))
    # §D-GOLONG — self-play draw-rate hard-abort (default OFF; operator arms via
    # monitors.hard_abort_draw_rate). threshold ≤ 0 disables. Complements the
    # stride5 spam gate (which catches only lattice-spam-shaped draw collapses).
    _draw_rate_threshold = float(_mon_cfg.get("hard_abort_draw_rate", 0.0))
    _draw_rate_consec    = int(_mon_cfg.get("hard_abort_draw_rate_consec", 3))
    _draw_rate_min_step  = int(_mon_cfg.get("hard_abort_draw_rate_min_step", 0))

    # ── StepCoordinatorConfig ────────────────────────────────────────────────
    from hexo_rl.training.step_coordinator import (
        DEFAULT_FINAL_EVAL_DRAIN_TIMEOUT_SEC,
        StepCoordinator,
        StepCoordinatorConfig,
    )
    _step_cfg = StepCoordinatorConfig(
        eval_interval=eval_interval,
        log_interval=log_interval,
        checkpoint_interval=_ckpt_interval,
        composition_interval=composition_interval,
        value_probe_interval=value_probe_interval,
        min_buf_size=min_buf_size,
        capacity=capacity,
        buffer_schedule=tuple(buffer_schedule),
        training_steps_per_game=training_steps_per_game,
        max_train_burst=max_train_burst,
        batch_size=int(train_cfg.get("batch_size", config.get("batch_size", 256))),
        augment=augment_cfg,
        recency_weight=recency_weight,
        mixing_initial_w=mixing_initial_w,
        mixing_min_w=mixing_min_w,
        mixing_decay_steps=mixing_decay_steps,
        soft_ew_threshold=_soft_ew_threshold,
        soft_ew_min_pts=_soft_ew_min_pts,
        hard_gn_threshold=_hard_gn_threshold,
        hard_gn_min_steps=_hard_gn_min_steps,
        stride5_p90_threshold=_stride5_p90_threshold,
        stride5_p90_consec=_stride5_p90_consec,
        draw_rate_threshold=_draw_rate_threshold,
        draw_rate_consec=_draw_rate_consec,
        draw_rate_min_step=_draw_rate_min_step,
        instrumentation_enabled=instrumentation_enabled,
        stop_step=stop_step,
        final_eval_drain_timeout_sec=float(
            train_cfg.get("eval_final_drain_timeout_sec",
                          config.get("eval_final_drain_timeout_sec",
                                     DEFAULT_FINAL_EVAL_DRAIN_TIMEOUT_SEC))
        ),
        # §178 — bot-corpus slot share + refresh hook (DISABLED-by-default;
        # Wave 3 variant flips `enabled: true`).
        bot_batch_share=float(mixing_cfg.get("bot_batch_share", 0.0)),
        bot_corpus_refresh_enabled=bool(
            mixing_cfg.get("bot_corpus_refresh", {}).get("enabled", False)
        ),
        bot_corpus_refresh_cooldown=int(
            mixing_cfg.get("bot_corpus_refresh", {}).get("cooldown_steps", 25_000)
        ),
        # §S181-AUDIT Wave 3 Stage 2A refresh-hook activation knobs.
        bot_corpus_refresh_interval_steps=int(
            mixing_cfg.get("bot_corpus_refresh", {}).get("interval_steps", 5_000)
        ),
        bot_corpus_refresh_n_games=int(
            mixing_cfg.get("bot_corpus_refresh", {}).get("n_games", 200)
        ),
        bot_corpus_refresh_opponent_model=str(
            mixing_cfg.get("bot_corpus_refresh", {}).get("opponent_model", "ema")
        ),
        bot_corpus_refresh_replace_strategy=str(
            mixing_cfg.get("bot_corpus_refresh", {}).get("replace_strategy", "rolling_window")
        ),
        bot_corpus_refresh_max_regens=int(
            mixing_cfg.get("bot_corpus_refresh", {}).get("max_regens", 20)
        ),
        bot_corpus_refresh_min_wr_delta=float(
            mixing_cfg.get("bot_corpus_refresh", {}).get("min_wr_delta", 0.0)
        ),
        bot_corpus_refresh_max_plies=int(
            mixing_cfg.get("bot_corpus_refresh", {})
            .get("regen_command", {}).get("args", {}).get("max_plies", 150)
        ),
        bot_corpus_refresh_random_opening_plies=int(
            mixing_cfg.get("bot_corpus_refresh", {})
            .get("regen_command", {}).get("args", {}).get("random_opening_plies", 4)
        ),
        bot_corpus_refresh_think_seconds=float(
            mixing_cfg.get("bot_corpus_refresh", {})
            .get("regen_command", {}).get("args", {}).get("think_seconds", 0.5)
        ),
        bot_corpus_refresh_anchor_n_sims=int(
            mixing_cfg.get("bot_corpus_refresh", {})
            .get("regen_command", {}).get("args", {}).get("anchor_n_sims", 200)
        ),
        bot_corpus_refresh_anchor_temperature=float(
            mixing_cfg.get("bot_corpus_refresh", {})
            .get("regen_command", {}).get("args", {}).get("anchor_temperature", 0.5)
        ),
        bot_corpus_path=str(mixing_cfg.get("bot_corpus_path", "") or ""),
    )

    # ── Emit run_start ─────────────────────────────────────────────────────────
    emit_event({
        "event": "run_start",
        "step": trainer.step,
        "run_id": run_id,
        "worker_count": pool.n_workers,
        "config_summary": {
            "n_blocks": int(config.get("res_blocks", MODEL_HPARAM_DEFAULTS["res_blocks"])),
            "channels": int(config.get("filters", MODEL_HPARAM_DEFAULTS["filters"])),
            "n_sims": int(mcts_config.get("n_simulations", 800)),
            "buffer_capacity": capacity,
        },
    })

    # R21: replay pretrain dashboard events BEFORE pool.start() so the
    # dashboard renders setup history first, not an in-progress live run.
    if args.checkpoint and "pretrain" in Path(args.checkpoint).name:
        _replay_pretrain_events(args)

    # ── Start self-play pool ───────────────────────────────────────────────────
    pool.start()
    log.info("selfplay_pool_started", n_workers=pool.n_workers)

    # ── Build coordinator ────────────────────────────────────────────────────
    coordinator = StepCoordinator(
        trainer=trainer,
        buffer=buffer,
        pretrained_buffer=pretrained_buffer,
        bot_buffer=bot_buffer,
        recent_buffer=recent_buffer,
        pool=pool,
        eval_pipeline=eval_pipeline,
        subsystems=subsys,
        anchor_state=_anchor,
        shutdown=_shutdown,
        eval_model=eval_model,
        bufs=bufs,
        config=_step_cfg,
        full_config=config,
        train_cfg=train_cfg,
        mcts_config=mcts_config,
        mixing_cfg=mixing_cfg,
        batch_size_cfg=batch_size_cfg,
        iterations=args.iterations,
        run_id=run_id,
    )

    # ── Run and teardown ──────────────────────────────────────────────────────
    # R10: tracemalloc.start/stop wraps the run loop so the periodic snapshot
    # at log_interval cadence has data to read.  Stays here, not in coordinator.
    tracemalloc.start(3)
    try:
        coordinator.run_until_stopped()
    finally:
        tracemalloc.stop()
        # R22 + D-012: drain a possibly-completed eval before teardown so a
        # promotion at the final tick isn't silently lost.  Pool is still up
        # here — the inference-server sync inside the drain needs it.
        coordinator.flush_pending_eval()
        emit_event({"event": "run_end", "step": coordinator.train_step})
        # R4 + R25: pool.stop() then subsys.teardown() — preserves the
        # gpu_util_pct read window during the last train_step_summary log.
        pool.stop()
        subsys.teardown()

    # ── Final checkpoint + buffer save ────────────────────────────────────────
    final_ckpt = trainer.save_checkpoint(
        coordinator.last_loss_info if coordinator.last_loss_info else None
    )
    _try_save_buffer(buffer, mixing_cfg, "session_end", recent_buffer)

    log.info(
        "session_end",
        final_step=trainer.step,
        games_played=coordinator.games_played,
        buffer_size=buffer.size,
        initial_policy_loss=coordinator.initial_policy_loss,
        final_policy_loss=(coordinator.last_loss_info.get("policy_loss")
                           if coordinator.last_loss_info else None),
        elapsed_sec=round(time.time() - t_start, 1),
        final_checkpoint=str(final_ckpt),
    )
    print(f"\nSaved checkpoint: {final_ckpt}")
    print(f"Games: {coordinator.games_played}  Steps: {trainer.step}  Buffer: {buffer.size}")
    if coordinator.initial_policy_loss is not None and coordinator.last_loss_info:
        decrease = coordinator.initial_policy_loss - coordinator.last_loss_info["policy_loss"]
        print(
            f"Policy loss: {coordinator.initial_policy_loss:.4f} → "
            f"{coordinator.last_loss_info['policy_loss']:.4f} "
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
