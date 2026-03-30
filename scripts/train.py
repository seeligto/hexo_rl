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
import random
import signal
import sys
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
from python.model.network import HexTacToeNet, compile_model
from python.training.replay_buffer import ReplayBuffer
from python.training.trainer import Trainer


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

    board_size  = int(combined_config.get("board_size",  19))
    res_blocks  = int(combined_config.get("res_blocks",  10))
    filters     = int(combined_config.get("filters",     128))

    if args.checkpoint:
        trainer = Trainer.load_checkpoint(
            args.checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            fallback_config=combined_config,
        )
        log.info("resumed", checkpoint=args.checkpoint, step=trainer.step)
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

    # Apply torch.compile (Phase 2) unless disabled.
    if not args.no_compile:
        trainer.model = compile_model(trainer.model)
        log.info("torch_compile", applied=True)

    # ── Replay buffer ──
    capacity = int(config.get("buffer_capacity", train_cfg.get("buffer_capacity", 500_000)))
    configured_min_buf = train_cfg.get("min_buffer_size", config.get("min_buffer_size"))
    if args.min_buffer_size is not None:
        min_buf_size = int(args.min_buffer_size)
    elif configured_min_buf is not None:
        min_buf_size = int(configured_min_buf)
    else:
        # Default warmup heuristic: enough samples for a few batches, but not so
        # high that training appears stalled on first launch.
        min_buf_size = max(128, min(512, int(train_cfg.get("batch_size", config.get("batch_size", 256)))))
    buffer = ReplayBuffer(
        capacity=capacity,
        board_channels=18,
        board_size=board_size,
    )
    log.info("buffer_init", capacity=capacity, min_buffer_size=min_buf_size)

    # ── Self-play pool ──
    from python.selfplay.pool import WorkerPool
    from python.eval.evaluator import Evaluator
    
    pool = WorkerPool(trainer.model, config, device, buffer)
    enable_periodic_eval = bool(train_cfg.get("enable_periodic_eval", config.get("enable_periodic_eval", False)))
    evaluator = Evaluator(trainer.model, device, config) if enable_periodic_eval else None
    
    # ── GPU monitor ──
    gpu_monitor = GPUMonitor(interval_sec=5)
    gpu_monitor.start()

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

        last_train_game_count = 0
        last_ui_refresh = 0.0
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
                "x_winrate": 0.0,
                "o_winrate": 0.0,
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

                # Perform one training step
                loss_info = trainer.train_step(buffer)
                train_step = trainer.step
                if initial_policy_loss is None:
                    initial_policy_loss = float(loss_info["policy_loss"])
                last_loss_info = loss_info

                # ── Evaluation ──
                eval_metrics = {}
                if evaluator is not None and train_step > 0 and train_step % eval_interval == 0:
                    log.info("evaluation_start", step=train_step)
                    wr_random = evaluator.evaluate_vs_random(n_games=10)
                    wr_ramora = evaluator.evaluate_vs_ramora(n_games=5)
                    eval_metrics = {"wr_random": wr_random, "wr_ramora": wr_ramora}
                    log.info("evaluation_complete", step=train_step, **eval_metrics)

                if train_step % log_interval == 0:
                    elapsed = time.time() - t_start
                    games_per_hour = games_played / elapsed * 3600 if elapsed > 0 else 0.0

                    metrics = {
                        "iteration":     train_step,
                        "policy_loss":   loss_info["policy_loss"],
                        "value_loss":    loss_info["value_loss"],
                        "buffer_size":   buffer.size,
                        "games_total":   games_played,
                        "games_per_hour": games_per_hour,
                        "gpu_util":      gpu_monitor.gpu_util_pct,
                        "vram_gb":       gpu_monitor.vram_used_gb,
                        "x_winrate":     0.0, # WorkerPool needs to track this separately if desired
                        "o_winrate":     0.0,
                        **eval_metrics
                    }

                    log.info(
                        "train_step",
                        step=train_step,
                        policy_loss=round(float(loss_info["policy_loss"]), 4),
                        value_loss=round(float(loss_info["value_loss"]), 4),
                        total_loss=round(float(loss_info["loss"]), 4),
                        buffer_size=buffer.size,
                        games_played=games_played,
                        games_per_hour=round(float(games_per_hour), 1),
                        gpu_util=round(float(gpu_monitor.gpu_util_pct), 1),
                        vram_gb=round(float(gpu_monitor.vram_used_gb), 2),
                        **eval_metrics,
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
