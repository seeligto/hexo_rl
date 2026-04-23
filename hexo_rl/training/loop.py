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
import signal
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional

import numpy as np
import structlog
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.early_game_probe import (
    EARLY_GAME_ENTROPY_WARN_THRESHOLD,
    EarlyGameProbe,
)
from hexo_rl.monitoring.events import emit_event, register_renderer
from hexo_rl.monitoring.gpu_monitor import GPUMonitor
from hexo_rl.training.batch_assembly import BatchBuffers, assemble_mixed_batch
from hexo_rl.training.trainer import Trainer

log = structlog.get_logger(__name__)


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

    # ── Inference model — separate instance owned by InferenceServer ──────────
    board_size         = int(trainer.config.get("board_size",         19))
    res_blocks         = int(trainer.config.get("res_blocks",         12))
    filters            = int(trainer.config.get("filters",            128))
    in_channels        = int(trainer.config.get("in_channels",        18))
    se_reduction_ratio = int(trainer.config.get("se_reduction_ratio", 4))

    _torch_compile_enabled = (
        trainer.config.get("torch_compile", False) and device.type == "cuda"
    )
    inf_model = HexTacToeNet(
        board_size=board_size,
        in_channels=in_channels,
        res_blocks=res_blocks,
        filters=filters,
        se_reduction_ratio=se_reduction_ratio,
    ).to(device)
    _train_base = getattr(trainer.model, "_orig_mod", trainer.model)
    inf_model.load_state_dict(_train_base.state_dict())
    inf_model.eval()
    if _torch_compile_enabled:
        try:
            inf_model = torch.compile(inf_model, mode="default", fullgraph=False)
            log.info("torch_compile_inf_enabled", mode="default")
        except Exception as exc:
            log.warning("torch_compile_inf_failed", error=str(exc))

    _ckpt_interval = int(trainer.config.get("checkpoint_interval", 500))

    # ── CUDA warm-up ─────────────────────────────────────────────────────────
    # Force CUDA kernel compilation now (before workers start) so the first
    # inference call from a worker returns immediately instead of blocking for
    # 90-120s while PyTorch JIT-compiles kernels. Without this, the warmup
    # phase shows "games=0" for ~2 minutes on a cold start, which looks broken.
    if device.type == "cuda":
        log.info("cuda_warmup_start")
        _t_warmup = time.time()
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                _dummy = torch.zeros(1, in_channels, board_size, board_size, device=device)
                inf_model(_dummy)
        torch.cuda.synchronize()
        log.info("cuda_warmup_done", elapsed_sec=round(time.time() - _t_warmup, 1))

    # ── CUDA stream audit (B4 perf probe) ─────────────────────────────────────
    # Logged from the main (training) thread context. InferenceServer logs its
    # own stream in run(). If both are on the same default stream pointer, no
    # copy/compute overlap is possible — the Q18 hypothesis.
    _diag = config.get("diagnostics") if isinstance(config.get("diagnostics"), dict) else {}
    if bool(_diag.get("perf_timing", False)) and device.type == "cuda":
        try:
            _cur = torch.cuda.current_stream(device)
            _def = torch.cuda.default_stream(device)
            log.info(
                "cuda_stream_audit",
                context="training_thread",
                current_stream_ptr=int(_cur.cuda_stream),
                default_stream_ptr=int(_def.cuda_stream),
                on_default_stream=_cur.cuda_stream == _def.cuda_stream,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("cuda_stream_audit_failed", context="training_thread", error=str(exc))

    # ── Worker pool ───────────────────────────────────────────────────────────
    from hexo_rl.selfplay.pool import WorkerPool
    pool = WorkerPool(inf_model, config, device, buffer)
    if recent_buffer is not None:
        pool.recent_buffer = recent_buffer

    # ── Eval pipeline ─────────────────────────────────────────────────────────
    from hexo_rl.eval.eval_pipeline import EvalPipeline

    eval_yaml_path = Path("configs/eval.yaml")
    eval_pipeline: EvalPipeline | None = None
    eval_ext_config: dict[str, Any] = {}
    _eval_interval_cfg: int = int(train_cfg.get("eval_interval", config.get("eval_interval", 100)))
    if eval_yaml_path.exists():
        from hexo_rl.utils.config import load_config as _load_config
        eval_ext_config = _load_config(str(eval_yaml_path))
        ep_cfg = eval_ext_config.get("eval_pipeline", {})
        # Allow the main training config to override eval.yaml's enabled flag
        # (test configs disable the pipeline to skip best_model.pt load).
        main_ep_override = config.get("eval_pipeline", {})
        if "enabled" in main_ep_override:
            ep_cfg = {**ep_cfg, "enabled": bool(main_ep_override["enabled"])}
        if ep_cfg.get("enabled", False):
            eval_pipeline = EvalPipeline(eval_ext_config, device, run_id=run_id)
            _eval_interval_cfg = int(ep_cfg.get("eval_interval", 1000))
            log.info("eval_pipeline_enabled", interval=_eval_interval_cfg)

    best_model_path = Path(
        eval_ext_config.get("eval_pipeline", {}).get("gating", {}).get(
            "best_model_path", "checkpoints/best_model.pt"
        )
    )
    best_model: HexTacToeNet | None = None
    best_model_step: int | None = None
    if eval_pipeline is not None:
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        if best_model_path.exists():
            best_ref = Trainer.load_checkpoint(
                best_model_path,
                checkpoint_dir=args.checkpoint_dir,
                device=device,
                fallback_config=config,
            )
            best_model = best_ref.model
            best_model.eval()
            best_model_step = best_ref.step
            # Graduation gate: self-play consumes anchor weights, not trainer.model.
            # Sync inf_model to the loaded anchor before workers start.
            _inf_base = getattr(inf_model, "_orig_mod", inf_model)
            _inf_base.load_state_dict(best_model.state_dict())
            log.info("best_model_loaded", path=str(best_model_path), step=best_model_step)
            # M2: warn if resumed trainer.model and loaded anchor diverge on step.
            # Either side may legitimately be ahead (anchor rollback, or training
            # continued past last promotion) but a silent mismatch can produce a
            # trivially-promoted first eval that wipes a hand-picked anchor.
            if best_model_step is not None and trainer.step != best_model_step:
                log.warning(
                    "resume_anchor_step_mismatch",
                    trainer_step=trainer.step,
                    best_model_step=best_model_step,
                    msg=(
                        "trainer.model and best_model.pt were loaded from different "
                        "training steps. First eval will compare the current trainer "
                        "weights against this anchor; confirm this is intended."
                    ),
                )
        else:
            base_model = getattr(trainer.model, "_orig_mod", trainer.model)
            best_model = HexTacToeNet(
                board_size=board_size, res_blocks=res_blocks, filters=filters,
                in_channels=in_channels, se_reduction_ratio=se_reduction_ratio,
            ).to(device)
            best_model.load_state_dict(base_model.state_dict())
            best_model.eval()
            torch.save(best_model.state_dict(), best_model_path)
            best_model_step = trainer.step
            log.info("best_model_initialized", path=str(best_model_path), step=best_model_step)

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
        eval_model = HexTacToeNet(
            board_size=board_size, res_blocks=res_blocks, filters=filters,
            in_channels=in_channels, se_reduction_ratio=se_reduction_ratio,
        ).to(device)
        eval_model.eval()

    # ── GPU monitor ───────────────────────────────────────────────────────────
    gpu_monitor = GPUMonitor(interval_sec=5)
    gpu_monitor.start()

    # ── Early-game policy-entropy probe (§115 monitoring signal) ──────────────
    # Fixed 10-position fixture. One forward pass per log_interval — rides on
    # the existing _emit_training_events cadence so probe cost is amortised.
    early_game_probe: Optional[EarlyGameProbe]
    try:
        early_game_probe = EarlyGameProbe(device=device)
        log.info(
            "early_game_probe_init",
            n_positions=early_game_probe.n_positions,
            plies=early_game_probe.plies,
        )
    except Exception as _egp_err:
        log.warning("early_game_probe_unavailable", error=str(_egp_err))
        early_game_probe = None

    # ── Dashboard renderers ───────────────────────────────────────────────────
    dashboards: list = []
    mon_cfg = config.get("monitoring", {})
    if mon_cfg.get("enabled", True) and not args.no_dashboard:
        if mon_cfg.get("terminal_dashboard", True):
            from hexo_rl.monitoring.terminal_dashboard import TerminalDashboard
            td = TerminalDashboard(config)
            td.start()
            register_renderer(td)
            dashboards.append(td)
        if mon_cfg.get("web_dashboard", True):
            from hexo_rl.monitoring.web_dashboard import WebDashboard
            wd = WebDashboard(config)
            wd.start()
            register_renderer(wd)
            dashboards.append(wd)

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    _running = [True]
    _stop_count = [0]
    _shutdown_save = [False]

    def _stop(sig: int, frame: Any) -> None:
        _stop_count[0] += 1
        if _stop_count[0] >= 2:
            sys.exit(1)
        log.info("shutdown_requested", msg="finishing current step… press Ctrl+C again to force")
        _shutdown_save[0] = True
        _running[0] = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

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

    def compute_pretrained_weight(step: int) -> float:
        return max(mixing_min_w, mixing_initial_w * math.exp(-step / mixing_decay_steps))

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
    _iter_games_window: list[tuple[float, int]] = []
    _iter_window_sec = 60.0

    def _games_per_hour_rolling() -> float:
        now = time.time()
        _iter_games_window.append((now, games_played))
        cutoff = now - _iter_window_sec
        while _iter_games_window and _iter_games_window[0][0] < cutoff:
            _iter_games_window.pop(0)
        if len(_iter_games_window) < 2:
            elapsed = max(now - t_start, 1e-6)
            return games_played / elapsed * 3600
        dt = _iter_games_window[-1][0] - _iter_games_window[0][0]
        dg = _iter_games_window[-1][1] - _iter_games_window[0][1]
        return (dg / max(dt, 1e-6)) * 3600

    # ── Main iteration ─────────────────────────────────────────────────────────
    schedule_idx = 1  # first schedule entry already applied at buffer construction

    def _run_loop() -> None:
        nonlocal train_step, games_played, initial_policy_loss, last_loss_info
        nonlocal _eval_thread, best_model, best_model_step, schedule_idx

        last_train_game_count  = 0
        last_warmup_log        = 0.0
        last_iter_games        = 0
        _last_quiescence_fires = 0

        while _running[0]:
            if stop_step is not None and train_step >= stop_step:
                log.info("iteration_limit_reached", iterations=args.iterations)
                break

            if _shutdown_save[0]:
                log.info(
                    "shutdown_signal_checkpoint",
                    msg="Shutdown signal received — saving checkpoint before exit",
                    step=train_step,
                )
                trainer.save_checkpoint(last_loss_info if last_loss_info else None)
                _try_save_buffer(buffer, mixing_cfg, "shutdown_signal")
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

            steps_budget = min(
                max(1, int(round(new_games * training_steps_per_game))),
                max_train_burst,
            )
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
                    w_pre  = compute_pretrained_weight(train_step)
                    n_pre  = max(1, int(math.ceil(batch_size * w_pre)))
                    n_self = batch_size - n_pre
                    (states, chain_planes, policies, outcomes,
                     ownership, winning_line, is_full_search) = assemble_mixed_batch(
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

                if train_step % _ckpt_interval == 0:
                    # Graduation gate: inf_model is the anchor, not a mirror of
                    # trainer.model. Do NOT sync on checkpoint cadence — sync only
                    # when a new model beats the anchor (promotion branch below).
                    if train_step > 0:
                        _try_save_buffer(buffer, mixing_cfg, "checkpoint_interval")

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
                    _emit_training_events(
                        train_step, loss_info, w_pre, games_played,
                        last_iter_games, pool, buffer, gpu_monitor,
                        config, mcts_config, capacity,
                        _games_per_hour_rolling, _qfire_delta,
                        early_game_probe=early_game_probe,
                        trainer_model=trainer.model,
                    )
                    last_iter_games = games_played

    # ── Run and teardown ──────────────────────────────────────────────────────
    tracemalloc.start(3)
    try:
        _run_loop()
    finally:
        tracemalloc.stop()
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
        gpu_monitor.stop()
        gpu_monitor.join(timeout=2.0)
        for d in dashboards:
            try:
                d.stop()
            except Exception:
                pass

    # ── Final checkpoint + buffer save ────────────────────────────────────────
    final_ckpt = trainer.save_checkpoint(last_loss_info if last_loss_info else None)
    _try_save_buffer(buffer, mixing_cfg, "session_end")

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


# ── Private helpers ────────────────────────────────────────────────────────────

def _drain_pending_eval(
    eval_thread: Optional[threading.Thread],
    eval_result: list[Optional[dict[str, Any]]],
    eval_model: Optional[HexTacToeNet],
    best_model: Optional[HexTacToeNet],
    best_model_path: Path,
    best_model_step: Optional[int],
    pool: Any,
    train_step: int,
) -> tuple[Optional[threading.Thread], Optional[int]]:
    """Drain the most recent completed eval: emit event + promote if gated.

    Safe at normal eval ticks and at shutdown — the latter is the whole
    point: without a post-``_run_loop`` drain, a promotion from the final
    eval before Ctrl-C / ``stop_step`` never hits ``best_model.pt`` and
    is silently lost on next restart (D-012).

    Returns ``(new_eval_thread, new_best_model_step)``; callers should
    rebind both. No-op when the thread is still running.
    """
    if eval_thread is None or eval_thread.is_alive():
        return eval_thread, best_model_step
    prev = eval_result[0]
    if prev is None:
        return None, best_model_step
    emit_event({
        "event": "eval_complete",
        "step": prev.get("step", train_step),
        "elo_estimate": prev.get("elo_estimate"),
        "win_rate_vs_sealbot": prev.get("wr_sealbot"),
        "eval_games": prev.get("eval_games", 0),
        "anchor_promoted": prev.get("promoted", False),
        "sealbot_gate_passed": prev.get("sealbot_gate_passed"),
    })
    new_best_step = best_model_step
    if prev.get("promoted"):
        assert eval_model is not None
        assert best_model is not None
        # C1: promote the snapshot that actually passed the gate.
        eval_base = getattr(eval_model, "_orig_mod", eval_model)
        best_model.load_state_dict(eval_base.state_dict())
        best_model.eval()
        torch.save(best_model.state_dict(), best_model_path)
        new_best_step = prev.get("step", train_step)
        pool._inference_server.load_state_dict_safe(eval_base.state_dict())
        log.info(
            "best_model_promoted",
            step=train_step,
            eval_step=new_best_step,
            path=str(best_model_path),
            graduated=True,
            wr_best=prev.get("wr_best"),
        )
    eval_result[0] = None
    return None, new_best_step


def _try_save_buffer(
    buffer: Any,
    mixing_cfg: dict[str, Any],
    trigger: str,
) -> None:
    """Save replay buffer to disk if ``buffer_persist`` is enabled in config."""
    if not mixing_cfg.get("buffer_persist", False):
        return
    bp = Path(mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
    try:
        buffer.save_to_path(str(bp))
        log.info("buffer_saved", path=str(bp), positions=buffer.size, trigger=trigger)
    except Exception as exc:
        log.warning("buffer_save_failed", path=str(bp), error=str(exc))


def _replay_pretrain_events(args: argparse.Namespace) -> None:
    """Replay up to 500 pretrain ``training_step`` events into the dashboard on resume."""
    import json
    pretrain_log = Path(args.log_dir) / "pretrain.jsonl"
    if not pretrain_log.exists():
        return
    replay_evs: list[dict] = []
    try:
        with open(pretrain_log) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("event") == "train_step" and d.get("phase") == "pretrain":
                        replay_evs.append({
                            "event": "training_step",
                            "step": d.get("step"),
                            "loss_total": d.get("loss"),
                            "loss_policy": d.get("policy_loss"),
                            "loss_value": d.get("value_loss"),
                            "loss_aux": d.get("aux_opp_reply_loss"),
                            "policy_entropy": d.get("policy_entropy"),
                            "value_accuracy": d.get("value_accuracy"),
                            "lr": d.get("lr"),
                            "grad_norm": d.get("grad_norm"),
                            "corpus_mix": d.get("corpus_mix", {"pretrain": 1.0, "self_play": 0.0}),
                            "phase": "pretrain",
                        })
                except Exception:
                    pass
    except Exception as e:
        log.warning("pretrain_replay_failed", error=str(e))
        return
    if replay_evs:
        log.info("replaying_pretrain_events", count=len(replay_evs[-500:]))
        for ev in replay_evs[-500:]:
            emit_event(ev)


def _emit_training_events(
    train_step: int,
    loss_info: dict[str, float],
    w_pre: float,
    games_played: int,
    last_iter_games: int,
    pool: Any,
    buffer: Any,
    gpu_monitor: GPUMonitor,
    config: dict[str, Any],
    mcts_config: dict[str, Any],
    capacity: int,
    games_per_hour_fn: Any,
    qfire_delta: int,
    early_game_probe: Optional[EarlyGameProbe] = None,
    trainer_model: Optional[Any] = None,
) -> None:
    """Emit ``training_step`` + ``iteration_complete`` events and structlog entry."""
    policy_entropy = float(loss_info.get("policy_entropy", 0.0))
    value_accuracy = float(loss_info.get("value_accuracy", 0.0))
    grad_norm      = float(loss_info.get("grad_norm", float("nan")))
    lr             = float(loss_info.get("lr", 0.0))

    # §115 early-game entropy probe — one forward pass on a fixed 10-position
    # fixture. Rides on log_interval cadence.
    probe_metrics: dict[str, Any] = {}
    if early_game_probe is not None and trainer_model is not None:
        try:
            probe_metrics = early_game_probe.compute(trainer_model)
            if probe_metrics["early_game_entropy_mean"] > EARLY_GAME_ENTROPY_WARN_THRESHOLD:
                log.warning(
                    "early_game_entropy_high",
                    step=train_step,
                    entropy_mean=round(probe_metrics["early_game_entropy_mean"], 4),
                    threshold=EARLY_GAME_ENTROPY_WARN_THRESHOLD,
                    entropy_by_ply=[round(x, 3) for x in probe_metrics["early_game_entropy_by_ply"]],
                )
        except Exception as _egp_err:
            log.warning("early_game_probe_failed", step=train_step, error=str(_egp_err))
            probe_metrics = {}

    training_step_event: dict[str, Any] = {
        "event": "training_step",
        "step": train_step,
        "loss_total":              float(loss_info["loss"]),
        "loss_policy":             float(loss_info["policy_loss"]),
        "loss_value":              float(loss_info["value_loss"]),
        "loss_aux":                float(loss_info.get("opp_reply_loss", 0.0)),
        "loss_ownership":          float(loss_info.get("ownership_loss", 0.0)),
        "loss_threat":             float(loss_info.get("threat_loss", 0.0)),
        "loss_chain":              float(loss_info.get("chain_loss", 0.0)),
        "aux_loss_rows":           int(loss_info.get("aux_loss_rows", 0)),
        "avg_sigma":               float(loss_info.get("avg_sigma", 0.0)),
        "policy_entropy":          policy_entropy,
        "policy_entropy_pretrain": float(loss_info.get("policy_entropy_pretrain", float("nan"))),
        "policy_entropy_selfplay": float(loss_info.get("policy_entropy_selfplay", float("nan"))),
        "policy_target_entropy":   float(loss_info.get("policy_target_entropy", 0.0)),
        # §101 — D-Gumbel / D-Zeroloss split metrics. NaN when the respective
        # subset is empty; renderers must handle NaN + missing keys gracefully.
        "policy_target_entropy_fullsearch":    float(loss_info.get("policy_target_entropy_fullsearch",    float("nan"))),
        "policy_target_entropy_fastsearch":    float(loss_info.get("policy_target_entropy_fastsearch",    float("nan"))),
        "policy_target_kl_uniform_fullsearch": float(loss_info.get("policy_target_kl_uniform_fullsearch", float("nan"))),
        "policy_target_kl_uniform_fastsearch": float(loss_info.get("policy_target_kl_uniform_fastsearch", float("nan"))),
        "frac_fullsearch_in_batch":            float(loss_info.get("frac_fullsearch_in_batch", 0.0)),
        "n_rows_policy_loss":                  int(loss_info.get("n_rows_policy_loss", 0)),
        "n_rows_total":                        int(loss_info.get("n_rows_total", 0)),
        "value_accuracy":          value_accuracy,
        "lr":                      lr,
        "grad_norm":               grad_norm,
        "quiescence_fires_per_step": qfire_delta,
    }
    if probe_metrics:
        training_step_event.update(probe_metrics)
    emit_event(training_step_event)

    gph    = games_per_hour_fn()
    avg_gl = pool.avg_game_length if hasattr(pool, "avg_game_length") else 0.0
    pph    = gph * avg_gl if avg_gl > 0 else 0.0
    _runner = pool._runner

    _buf_sp_pct = round(min(pool.self_play_positions_pushed / max(buffer.size, 1), 1.0), 4)

    emit_event({
        "event": "iteration_complete",
        "step": train_step,
        "games_total":        games_played,
        "games_this_iter":    games_played - last_iter_games,
        "games_per_hour":     round(gph, 1),
        "positions_per_hour": round(pph, 1),
        "avg_game_length":    round(avg_gl, 1),
        "win_rate_p0":        round(float(pool.x_winrate), 4),
        "win_rate_p1":        round(float(pool.o_winrate), 4),
        "draw_rate":          round(float(pool.draws / games_played), 4) if games_played > 0 else 0.0,
        "sims_per_sec":       pool.sims_per_sec or 0.0,
        "buffer_size":        buffer.size,
        "buffer_capacity":    buffer.capacity,
        "corpus_selfplay_frac": round(1.0 - w_pre, 4),
        "batch_fill_pct":     pool.batch_fill_pct,
        "mcts_mean_depth":    float(getattr(_runner, "mcts_mean_depth", 0.0)),
        "mcts_root_concentration": float(getattr(_runner, "mcts_mean_root_concentration", 0.0)),
        # §107 I2 investigation metrics: lifetime-mean per-cluster std-dev of
        # values and top-1 policy disagreement (K≥2 positions only).
        "cluster_value_std_mean":      float(getattr(_runner, "cluster_value_std_mean", 0.0)),
        "cluster_policy_disagreement_mean": float(getattr(_runner, "cluster_policy_disagreement_mean", 0.0)),
        "cluster_variance_sample_count":    int(getattr(_runner, "cluster_variance_sample_count", 0)),
    })

    # Richer summary structlog entry — fires at log_interval cadence alongside
    # the trainer's per-step ``train_step`` log. Kept under a distinct event
    # name to preserve the 1:1 step-to-``train_step`` invariant (Q27 smoke
    # 2026-04-19 root cause: this entry previously emitted under the same
    # ``train_step`` name and duplicated the trainer's per-step emission).
    log.info(
        "train_step_summary",
        step=train_step,
        policy_loss=round(float(loss_info["policy_loss"]), 4),
        value_loss=round(float(loss_info["value_loss"]), 4),
        total_loss=round(float(loss_info["loss"]), 4),
        aux_opp_reply_loss=round(float(loss_info.get("opp_reply_loss", 0.0)), 4),
        avg_sigma=round(float(loss_info.get("avg_sigma", 0.0)), 4),
        policy_entropy=round(policy_entropy, 4),
        policy_entropy_pretrain=round(float(loss_info.get("policy_entropy_pretrain", float("nan"))), 4),
        policy_entropy_selfplay=round(float(loss_info.get("policy_entropy_selfplay", float("nan"))), 4),
        buffer_size=buffer.size,
        buffer_capacity=buffer.capacity,
        pretrained_weight=round(w_pre, 4),
        selfplay_weight=round(1.0 - w_pre, 4),
        buffer_self_play_pct=_buf_sp_pct,
        games_played=games_played,
        games_per_hour=round(gph, 1),
        sims_per_sec=pool.sims_per_sec,
        x_wins=pool.x_wins,
        o_wins=pool.o_wins,
        draws=pool.draws,
        x_winrate=round(float(pool.x_winrate), 3),
        o_winrate=round(float(pool.o_winrate), 3),
        draw_rate=round(float(pool.draws / games_played), 3) if games_played > 0 else 0.0,
        gpu_util=round(float(gpu_monitor.gpu_util_pct), 1),
        vram_gb=round(float(gpu_monitor.vram_used_gb), 2),
        ownership_loss=round(float(loss_info["ownership_loss"]), 4) if loss_info.get("ownership_loss") is not None else None,
        threat_loss=round(float(loss_info["threat_loss"]), 4) if loss_info.get("threat_loss") is not None else None,
        aux_loss_rows=int(loss_info.get("aux_loss_rows", 0)),
        batch_fill_pct=round(pool.batch_fill_pct, 1),
        inf_forward_count=pool._inference_server._forward_count,
        inf_total_requests=pool._inference_server._total_requests,
        mcts_mean_depth=round(float(getattr(_runner, "mcts_mean_depth", 0.0)), 3),
        mcts_root_concentration=round(float(getattr(_runner, "mcts_mean_root_concentration", 0.0)), 3),
        policy_target_entropy_fullsearch=float(loss_info.get("policy_target_entropy_fullsearch", float("nan"))),
        policy_target_entropy_fastsearch=float(loss_info.get("policy_target_entropy_fastsearch", float("nan"))),
        policy_target_kl_uniform_fullsearch=float(loss_info.get("policy_target_kl_uniform_fullsearch", float("nan"))),
        policy_target_kl_uniform_fastsearch=float(loss_info.get("policy_target_kl_uniform_fastsearch", float("nan"))),
        frac_fullsearch_in_batch=float(loss_info.get("frac_fullsearch_in_batch", 0.0)),
        n_rows_policy_loss=int(loss_info.get("n_rows_policy_loss", 0)),
        n_rows_total=int(loss_info.get("n_rows_total", 0)),
        cluster_value_std_mean=float(getattr(_runner, "cluster_value_std_mean", 0.0)),
        cluster_policy_disagreement_mean=float(getattr(_runner, "cluster_policy_disagreement_mean", 0.0)),
        cluster_variance_sample_count=int(getattr(_runner, "cluster_variance_sample_count", 0)),
        early_game_entropy_mean=round(float(probe_metrics.get("early_game_entropy_mean", float("nan"))), 4)
            if probe_metrics else None,
        early_game_top1_mass_mean=round(float(probe_metrics.get("early_game_top1_mass_mean", float("nan"))), 4)
            if probe_metrics else None,
    )


def _suppress_semaphore_leak_warning() -> None:
    """Silence spurious 'leaked semaphore' warning on exit from PyO3 condvars."""
    try:
        import multiprocessing.resource_tracker
        multiprocessing.resource_tracker._resource_tracker._stop()
    except Exception:
        pass
