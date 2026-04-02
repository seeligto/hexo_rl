#!/usr/bin/env python3
"""
End-to-end training loop: self-play → buffer → train → repeat.

Phase 2 design:
  1. Play self-play games until the buffer has enough data.
  2. Run one training step per self-play game (interleaved).
  3. Log structured JSON to file (structlog) every event.
  4. Display rich live dashboard in terminal.
  5. Monitor GPU stats every 5 s via pynvml daemon thread.
  6. Save checkpoints every checkpoint_interval steps.
  7. Stop after --iterations steps (default: runs until Ctrl-C).

Usage:
    .venv/bin/python scripts/train.py --config configs/fast_debug.yaml
    .venv/bin/python scripts/train.py --config configs/fast_debug.yaml \\
        --checkpoint checkpoints/checkpoint_00000100.pt
    .venv/bin/python scripts/train.py --config configs/fast_debug.yaml \\
        --iterations 500 --no-dashboard
"""

from __future__ import annotations

import argparse
import math
import random
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Ensure project root is on sys.path when run as a script ──────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.logging.dashboard import TrainingDashboard
from python.logging.gpu_monitor import GPUMonitor
from python.logging.setup import configure_logging
from python.model.network import HexTacToeNet
from native_core import RustReplayBuffer
from python.training.trainer import Trainer
from python.training.dashboard_utils import DashboardClient


# ── Seeding ───────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 training loop")
    p.add_argument(
        "--config", default="configs/fast_debug.yaml",
        help="Path to YAML config (default: configs/fast_debug.yaml)"
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Resume from this checkpoint file (optional)"
    )
    p.add_argument(
        "--iterations", type=int, default=None,
        help="Stop after this many training steps (default: run until Ctrl-C)"
    )
    p.add_argument(
        "--override-scheduler-horizon", action="store_true",
        help=(
            "When resuming from a full checkpoint with --iterations, override "
            "checkpoint total_steps so the LR scheduler horizon follows the CLI value"
        ),
    )
    p.add_argument(
        "--log-dir", default="logs",
        help="Directory for structlog JSON files (default: logs/)"
    )
    p.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints/)"
    )
    p.add_argument(
        "--run-name", default=None,
        help="Run identifier for log file name (default: timestamp)"
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable rich live dashboard (useful in CI or non-interactive mode)"
    )
    p.add_argument(
        "--web-dashboard", action="store_true",
        help="Push game/metric data to a running dashboard.py server",
    )
    p.add_argument(
        "--web-dashboard-url", default="http://localhost:5001",
        help="Base URL of the dashboard server (default: http://localhost:5001)",
    )
    p.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile (useful for debugging)"
    )
    p.add_argument(
        "--min-buffer-size", type=int, default=None,
        help="Override replay warmup size before first training step"
    )
    return p.parse_args()


# ── Main training loop ────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(signal, "SIGHUP"):
        # Detached runs should survive terminal hangups.
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    # Optimization: Enable TensorFloat32 (TF32) for better performance on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    args = parse_args()

    # ── Load config ──
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Logging ──
    log = configure_logging(log_dir=args.log_dir, run_name=args.run_name)
    log.info("startup", config=config, pid=__import__("os").getpid())

    # ── Seed ──
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    log.info("seeded", seed=seed)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device", device=str(device))

    # ── Model + Trainer ──
    # Handle both nested (default.yaml) and flat (fast_debug.yaml) configs
    model_config = config.get("model", {})
    train_config = config.get("training", {})
    mcts_config  = config.get("mcts", {})
    self_config  = config.get("selfplay", {})
    
    # Merge all into one flat dict for Trainer/MCTS/SelfPlay usage
    # If the config is flat (like fast_debug.yaml), 'config' itself has the keys.
    combined_config = {
        **config,        # Base level (for flat configs)
        **model_config,  # Overrides from nested model section
        **train_config,  # Overrides from nested training section
        **mcts_config,   # Overrides from nested mcts section
        **self_config,   # Overrides from nested selfplay section
    }
    train_cfg = config.get("training", config)

    # Tie LR schedule horizon to the requested run length when provided.
    if args.iterations is not None:
        combined_config["total_steps"] = int(args.iterations)
        train_cfg["total_steps"] = int(args.iterations)

    # torch.compile is now handled inside Trainer.__init__ via config.
    # The --no-compile flag overrides the config key before construction.
    if args.no_compile:
        combined_config["torch_compile"] = False
        train_cfg["torch_compile"] = False

    board_size  = int(combined_config.get("board_size",  19))
    res_blocks  = int(combined_config.get("res_blocks",  10))
    filters     = int(combined_config.get("filters",     128))

    if args.checkpoint:
        config_overrides = None
        if args.iterations is not None and args.override_scheduler_horizon:
            config_overrides = {"total_steps": int(args.iterations)}

        trainer = Trainer.load_checkpoint(
            args.checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            fallback_config=combined_config,
            config_overrides=config_overrides,
        )
        log.info(
            "resumed",
            checkpoint=args.checkpoint,
            step=trainer.step,
            scheduler_horizon_overridden=bool(config_overrides),
            configured_total_steps=trainer.config.get("total_steps"),
        )
    else:
        model   = HexTacToeNet(
            board_size=board_size,
            res_blocks=res_blocks,
            filters=filters,
        )
        trainer = Trainer(
            model, combined_config,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
        )
        log.info("new_run", model_params=sum(p.numel() for p in model.parameters()))

    # ── Replay buffer with growth schedule ──
    buffer_schedule_raw = train_cfg.get("buffer_schedule", config.get("buffer_schedule", []))
    buffer_schedule = sorted(
        [{"step": int(e["step"]), "capacity": int(e["capacity"])} for e in buffer_schedule_raw],
        key=lambda x: x["step"],
    )
    if buffer_schedule:
        capacity = buffer_schedule[0]["capacity"]
    else:
        capacity = int(config.get("buffer_capacity", train_cfg.get("buffer_capacity", 500_000)))

    configured_min_buf = train_cfg.get("min_buffer_size", config.get("min_buffer_size"))
    if args.min_buffer_size is not None:
        min_buf_size = int(args.min_buffer_size)
    elif configured_min_buf is not None:
        min_buf_size = int(configured_min_buf)
    else:
        min_buf_size = max(128, min(512, int(train_cfg.get("batch_size", config.get("batch_size", 256)))))
    buffer = RustReplayBuffer(capacity=capacity)
    schedule_idx = 1  # first entry already applied at construction
    log.info("buffer_init", capacity=capacity, min_buffer_size=min_buf_size,
             schedule_entries=len(buffer_schedule))

    # ── Pretrained buffer (mixed data streams) ──
    mixing_cfg = train_cfg.get("mixing", config.get("mixing", {}))
    pretrained_buffer = None
    pretrained_path = mixing_cfg.get("pretrained_buffer_path")
    if pretrained_path and Path(pretrained_path).exists():
        data = np.load(pretrained_path)
        pre_states = data["states"]       # (T, 18, 19, 19) float16
        pre_policies = data["policies"]   # (T, 362) float32
        pre_outcomes = data["outcomes"]   # (T,) float32
        pretrained_buffer = RustReplayBuffer(capacity=len(pre_outcomes))
        pretrained_buffer.push_game(pre_states, pre_policies, pre_outcomes)
        log.info("pretrained_buffer_loaded", path=pretrained_path,
                 size=pretrained_buffer.size)
    elif pretrained_path:
        log.warning("pretrained_buffer_missing", path=pretrained_path)

    mixing_decay_steps = float(mixing_cfg.get("decay_steps", 1_000_000))
    mixing_min_w = float(mixing_cfg.get("min_pretrained_weight", 0.1))
    mixing_initial_w = float(mixing_cfg.get("initial_pretrained_weight", 0.8))

    def compute_pretrained_weight(step: int) -> float:
        return max(mixing_min_w, mixing_initial_w * math.exp(-step / mixing_decay_steps))

    # ── Self-play pool ──
    from python.selfplay.pool import WorkerPool
    from python.eval.eval_pipeline import EvalPipeline

    pool = WorkerPool(trainer.model, config, device, buffer)

    # ── Evaluation pipeline (Phase 4.0) ──
    eval_yaml_path = Path("configs/eval.yaml")
    eval_pipeline: EvalPipeline | None = None
    if eval_yaml_path.exists():
        with open(eval_yaml_path) as f:
            eval_ext_config = yaml.safe_load(f)
        ep_cfg = eval_ext_config.get("eval_pipeline", {})
        if ep_cfg.get("enabled", False):
            eval_pipeline = EvalPipeline(eval_ext_config, device)
            eval_interval = int(ep_cfg.get("eval_interval", 1000))
            log.info("eval_pipeline_enabled", interval=eval_interval)
    if eval_pipeline is None:
        # Fall back to default.yaml eval_interval
        eval_interval = int(train_cfg.get("eval_interval", config.get("eval_interval", 100)))

    # Best model for gating
    best_model_path = Path(
        eval_ext_config["eval_pipeline"]["gating"]["best_model_path"]
        if eval_pipeline else "checkpoints/best_model.pt"
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
                fallback_config=combined_config,
            )
            best_model = best_ref.model
            best_model.eval()
            best_model_step = best_ref.step
            log.info("best_model_loaded", path=str(best_model_path), step=best_model_step)
        else:
            base_model = getattr(trainer.model, "_orig_mod", trainer.model)
            best_model = HexTacToeNet(
                board_size=board_size,
                res_blocks=res_blocks,
                filters=filters,
            ).to(device)
            best_model.load_state_dict(base_model.state_dict())
            best_model.eval()
            torch.save(best_model.state_dict(), best_model_path)
            best_model_step = trainer.step
            log.info("best_model_initialized", path=str(best_model_path), step=best_model_step)

    # Non-blocking eval state
    _eval_thread: threading.Thread | None = None
    _eval_result: list[dict | None] = [None]  # mutable holder for thread result
    
    # ── GPU monitor ──
    gpu_monitor = GPUMonitor(interval_sec=5)
    gpu_monitor.start()

    # ── Web dashboard client (fire-and-forget) ──
    web_dash = DashboardClient(base_url=args.web_dashboard_url) if args.web_dashboard else None

    # ── Graceful shutdown ──
    _running = [True]
    def _stop(sig, frame):
        log.info("shutdown_requested")
        _running[0] = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── Loop ──
    log_interval  = int(config.get("log_interval",  10))
    eval_interval = int(train_cfg.get("eval_interval", config.get("eval_interval", 100)))
    training_steps_per_game = float(train_cfg.get("training_steps_per_game", 1.0))
    max_train_burst = int(train_cfg.get("max_train_burst", 8))
    train_step    = trainer.step
    games_played  = 0
    t_start       = time.time()
    initial_policy_loss: float | None = None
    last_loss_info: dict[str, float] = {}

    dashboard = TrainingDashboard() if not args.no_dashboard else None
    total_steps = args.iterations or 0
    
    # Start multiprocess self-play
    pool.start()
    log.info("selfplay_pool_started", n_workers=pool.n_workers)

    def _run_loop() -> None:
        nonlocal train_step, games_played, initial_policy_loss, last_loss_info
        nonlocal _eval_thread, best_model, best_model_step

        last_train_game_count = 0
        last_ui_refresh = 0.0
        last_web_games = 0
        last_x_wins = 0
        last_o_wins = 0
        last_draws = 0
        ui_refresh_sec = 1.0
        last_warmup_log = 0.0

        def _refresh_dashboard(eval_metrics: dict | None = None) -> None:
            if dashboard is None:
                return
            nonlocal last_ui_refresh
            now = time.time()
            if now - last_ui_refresh < ui_refresh_sec:
                return
            elapsed = max(now - t_start, 1e-6)
            metrics = {
                "iteration": train_step,
                "policy_loss": last_loss_info.get("policy_loss") if last_loss_info else None,
                "value_loss": last_loss_info.get("value_loss") if last_loss_info else None,
                "buffer_size": buffer.size,
                "games_total": games_played,
                "games_per_hour": games_played / elapsed * 3600,
                "gpu_util": gpu_monitor.gpu_util_pct,
                "vram_gb": gpu_monitor.vram_used_gb,
                "x_winrate": pool.x_winrate,
                "o_winrate": pool.o_winrate,
                "draw_rate": (pool.draws / games_played) if games_played > 0 else 0.0,
                "x_wins": pool.x_wins,
                "o_wins": pool.o_wins,
                "draws": pool.draws,
            }
            if eval_metrics:
                metrics.update(eval_metrics)
            dashboard.update(train_step, total_steps, metrics)
            last_ui_refresh = now
        
        while _running[0]:
            if args.iterations and train_step >= args.iterations:
                log.info("iteration_limit_reached", iterations=args.iterations)
                break

            # ── Training Throttling ──
            # Only train if we have enough data and we haven't already trained 
            # for the current batch of games.
            games_played = pool.games_completed
            _refresh_dashboard()
            
            if buffer.size < min_buf_size:
                if dashboard is None and (time.time() - last_warmup_log) >= 5.0:
                    print(
                        f"[warmup] buffer={buffer.size}/{min_buf_size} games={games_played} "
                        f"gpu={gpu_monitor.gpu_util_pct:.0f}%",
                        flush=True,
                    )
                    last_warmup_log = time.time()
                time.sleep(1.0)
                continue
                
            new_games = games_played - last_train_game_count
            if new_games <= 0:
                if dashboard is None and (time.time() - last_warmup_log) >= 5.0:
                    print(
                        f"[waiting] games={games_played} trained_games={last_train_game_count} "
                        f"buffer={buffer.size}",
                        flush=True,
                    )
                    last_warmup_log = time.time()
                time.sleep(0.1) # Wait for workers
                continue

            steps_budget = max(1, int(round(new_games * training_steps_per_game)))
            steps_budget = min(steps_budget, max_train_burst)
            last_train_game_count = games_played

            for _ in range(steps_budget):
                if args.iterations and train_step >= args.iterations:
                    break

                # ── Buffer growth schedule ──
                nonlocal schedule_idx
                while schedule_idx < len(buffer_schedule) and train_step >= buffer_schedule[schedule_idx]["step"]:
                    new_cap = buffer_schedule[schedule_idx]["capacity"]
                    if new_cap > buffer.capacity:
                        buffer.resize(new_cap)
                        log.info("buffer_resized", step=train_step, new_capacity=new_cap)
                    schedule_idx += 1

                # ── Training step (mixed or single buffer) ──
                batch_size = int(train_cfg.get("batch_size", config.get("batch_size", 256)))
                if pretrained_buffer is not None and pretrained_buffer.size > 0 and buffer.size > 0:
                    w_pre = compute_pretrained_weight(train_step)
                    n_pre = max(1, int(math.ceil(batch_size * w_pre)))
                    n_self = batch_size - n_pre
                    s_pre, p_pre, o_pre = pretrained_buffer.sample_batch(n_pre, True)
                    s_self, p_self, o_self = buffer.sample_batch(max(1, n_self), True)
                    states = np.concatenate([s_pre, s_self], axis=0)
                    policies = np.concatenate([p_pre, p_self], axis=0)
                    outcomes = np.concatenate([o_pre, o_self], axis=0)
                    loss_info = trainer.train_step_from_tensors(states, policies, outcomes)
                else:
                    w_pre = 0.0
                    loss_info = trainer.train_step(buffer)
                train_step = trainer.step
                if initial_policy_loss is None:
                    initial_policy_loss = float(loss_info["policy_loss"])
                last_loss_info = loss_info

                # ── Evaluation (non-blocking via background thread) ──
                eval_metrics = {}
                if eval_pipeline is not None and train_step > 0 and train_step % eval_interval == 0:
                    # Harvest previous eval result if ready
                    if _eval_thread is not None and not _eval_thread.is_alive():
                        prev = _eval_result[0]
                        if prev is not None and prev.get("promoted"):
                            base_model = getattr(trainer.model, "_orig_mod", trainer.model)
                            best_model.load_state_dict(base_model.state_dict())
                            best_model.eval()
                            torch.save(best_model.state_dict(), best_model_path)
                            best_model_step = train_step
                            log.info("best_model_promoted", step=train_step, path=str(best_model_path))
                        _eval_result[0] = None
                        _eval_thread = None

                    if _eval_thread is None or not _eval_thread.is_alive():
                        # Clone model for eval (avoids sharing torch.compiled model)
                        base_model = getattr(trainer.model, "_orig_mod", trainer.model)
                        eval_model = HexTacToeNet(
                            board_size=board_size,
                            res_blocks=res_blocks,
                            filters=filters,
                        ).to(device)
                        eval_model.load_state_dict(base_model.state_dict())
                        eval_model.eval()

                        step_snapshot = train_step
                        log.info("evaluation_start", step=step_snapshot)

                        def _run_eval(
                            _model: HexTacToeNet = eval_model,
                            _step: int = step_snapshot,
                            _best: HexTacToeNet | None = best_model,
                            _cfg: dict = config,
                        ) -> None:
                            try:
                                _eval_result[0] = eval_pipeline.run_evaluation(
                                    _model, _step, _best, full_config=_cfg,
                                )
                            except Exception:
                                import traceback
                                log.info("evaluation_error", step=_step, tb=traceback.format_exc())
                                _eval_result[0] = {"promoted": False, "error": True}

                        _eval_thread = threading.Thread(target=_run_eval, daemon=True)
                        _eval_thread.start()
                    else:
                        log.info("eval_skipped_still_running", step=train_step)

                if train_step % log_interval == 0:
                    elapsed = time.time() - t_start
                    games_per_hour = games_played / elapsed * 3600 if elapsed > 0 else 0.0

                    metrics = {
                        "iteration":     train_step,
                        "policy_loss":   loss_info["policy_loss"],
                        "value_loss":    loss_info["value_loss"],
                        "buffer_size":   buffer.size,
                        "buffer_capacity": buffer.capacity,
                        "pretrained_weight": round(w_pre, 4),
                        "selfplay_weight": round(1.0 - w_pre, 4),
                        "games_total":   games_played,
                        "games_per_hour": games_per_hour,
                        "gpu_util":      gpu_monitor.gpu_util_pct,
                        "vram_gb":       gpu_monitor.vram_used_gb,
                        "x_winrate":     pool.x_winrate,
                        "o_winrate":     pool.o_winrate,
                        **eval_metrics
                    }

                    log.info(
                        "train_step",
                        step=train_step,
                        policy_loss=round(float(loss_info["policy_loss"]), 4),
                        value_loss=round(float(loss_info["value_loss"]), 4),
                        total_loss=round(float(loss_info["loss"]), 4),
                        aux_opp_reply_loss=round(float(loss_info.get("opp_reply_loss", 0.0)), 4),
                        policy_entropy=round(float(loss_info.get("policy_entropy", 0.0)), 4),
                        buffer_size=buffer.size,
                        buffer_capacity=buffer.capacity,
                        pretrained_weight=round(w_pre, 4),
                        selfplay_weight=round(1.0 - w_pre, 4),
                        games_played=games_played,
                        games_per_hour=round(float(games_per_hour), 1),
                        x_wins=pool.x_wins,
                        o_wins=pool.o_wins,
                        draws=pool.draws,
                        x_winrate=round(float(pool.x_winrate), 3),
                        o_winrate=round(float(pool.o_winrate), 3),
                        draw_rate=round(float(pool.draws / games_played), 3) if games_played > 0 else 0.0,
                        gpu_util=round(float(gpu_monitor.gpu_util_pct), 1),
                        vram_gb=round(float(gpu_monitor.vram_used_gb), 2),
                        **eval_metrics,
                    )

                    if web_dash is not None:
                        new_web_games = games_played - last_web_games
                        if new_web_games > 0:
                            dx  = pool.x_wins - last_x_wins
                            do_ = pool.o_wins - last_o_wins
                            dd  = pool.draws  - last_draws
                            for _ in range(dx):
                                web_dash.send_game(moves=[], result=1.0)
                            for _ in range(do_):
                                web_dash.send_game(moves=[], result=-1.0)
                            for _ in range(dd):
                                web_dash.send_game(moves=[], result=0.0)
                            last_web_games = games_played
                            last_x_wins = pool.x_wins
                            last_o_wins = pool.o_wins
                            last_draws  = pool.draws
                        web_dash.send_metrics(
                            iteration=train_step,
                            loss=float(loss_info["loss"]),
                            elo=eval_metrics.get("wr_sealbot"),
                            policy_loss=float(loss_info["policy_loss"]),
                            value_loss=float(loss_info["value_loss"]),
                            games_total=games_played,
                            gpu_util=gpu_monitor.gpu_util_pct,
                            x_winrate=round(float(pool.x_winrate), 3),
                            o_winrate=round(float(pool.o_winrate), 3),
                        )

                    if dashboard is not None:
                        dashboard.update(train_step, total_steps, metrics)
                        last_ui_refresh = time.time()

    try:
        if dashboard is not None:
            with dashboard.live():
                _run_loop()
        else:
            _run_loop()
    finally:
        pool.stop()
        if web_dash is not None:
            web_dash.stop()
        gpu_monitor.stop()
        gpu_monitor.join(timeout=2.0)

    # ── Session end: save final checkpoint and log summary ──
    final_ckpt = trainer.save_checkpoint(last_loss_info if last_loss_info else None)
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
        print(f"Policy loss: {initial_policy_loss:.4f} → {last_loss_info['policy_loss']:.4f} "
              f"({'↓' if decrease > 0 else '↑'}{abs(decrease):.4f})")


if __name__ == "__main__":
    main()
