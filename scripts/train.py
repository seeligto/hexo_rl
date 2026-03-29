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
from python.selfplay.worker import SelfPlayWorker
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
    board_size  = int(config.get("board_size",  19))
    res_blocks  = int(config.get("res_blocks",  10))
    filters     = int(config.get("filters",     128))

    if args.checkpoint:
        trainer = Trainer.load_checkpoint(
            args.checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
        )
        log.info("resumed", checkpoint=args.checkpoint, step=trainer.step)
    else:
        model   = HexTacToeNet(
            board_size=board_size,
            res_blocks=res_blocks,
            filters=filters,
        )
        trainer = Trainer(
            model, config,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
        )
        log.info("new_run", model_params=sum(p.numel() for p in model.parameters()))

    # Apply torch.compile (Phase 2) unless disabled.
    if not args.no_compile:
        trainer.model = compile_model(trainer.model)
        log.info("torch_compile", applied=True)

    # ── Replay buffer ──
    capacity     = int(config.get("buffer_capacity", 500_000))
    min_buf_size = int(config.get("min_buffer_size", 512))
    buffer = ReplayBuffer(
        capacity=capacity,
        board_channels=18,
        board_size=board_size,
    )
    log.info("buffer_init", capacity=capacity, min_buffer_size=min_buf_size)

    # ── Self-play worker ──
    worker = SelfPlayWorker(trainer.model, config, device)

    # ── GPU monitor (daemon thread, polls every 5 s) ──
    gpu_monitor = GPUMonitor(interval_sec=5)
    gpu_monitor.start()

    # ── Graceful shutdown on Ctrl-C ──
    _running = [True]
    def _stop(sig, frame):  # noqa: ANN001
        log.info("shutdown_requested")
        _running[0] = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── Loop ──
    log_interval  = int(config.get("log_interval",  10))
    train_step    = trainer.step
    games_played  = 0
    x_wins        = 0
    o_wins        = 0
    draws         = 0
    t_start       = time.time()
    t_games_window_start = t_start

    # Track initial policy loss for exit-criteria logging
    initial_policy_loss: float | None = None
    last_loss_info: dict = {}

    dashboard = TrainingDashboard() if not args.no_dashboard else None
    total_steps = args.iterations or 0

    def _run_loop() -> None:
        nonlocal train_step, games_played, initial_policy_loss, last_loss_info
        nonlocal t_games_window_start, x_wins, o_wins, draws

        while _running[0]:
            if args.iterations and train_step >= args.iterations:
                log.info("iteration_limit_reached", iterations=args.iterations)
                break

            # ── Self-play: play one game ──
            t_game = time.time()
            n_positions, winner = worker.play_game(buffer)
            game_elapsed = time.time() - t_game
            games_played += 1
            
            if winner == 1:
                x_wins += 1
            elif winner == -1:
                o_wins += 1
            else:
                draws += 1

            log.info(
                "game_complete",
                game_id=games_played,
                plies=n_positions,
                duration_sec=round(game_elapsed, 2),
                buffer_size=buffer.size,
                winner=winner,
            )

            # ── Training: skip until buffer is warm ──
            if buffer.size < min_buf_size:
                continue

            # ── One training step ──
            loss_info = trainer.train_step(buffer)
            train_step = trainer.step

            if initial_policy_loss is None:
                initial_policy_loss = loss_info["policy_loss"]

            last_loss_info = loss_info

            if train_step % log_interval == 0:
                elapsed    = time.time() - t_start
                window_sec = max(time.time() - t_games_window_start, 1e-6)
                games_per_hour = games_played / elapsed * 3600 if elapsed > 0 else 0.0
                
                x_winrate = x_wins / games_played if games_played > 0 else 0.0
                o_winrate = o_wins / games_played if games_played > 0 else 0.0

                log.info(
                    "train_step",
                    step=train_step,
                    policy_loss=round(loss_info["policy_loss"], 4),
                    value_loss=round(loss_info["value_loss"],   4),
                    total_loss=round(loss_info["loss"],         4),
                    buffer_size=buffer.size,
                    games_played=games_played,
                    elapsed_sec=round(elapsed, 1),
                    x_winrate=round(x_winrate, 3),
                    o_winrate=round(o_winrate, 3),
                )

                metrics = {
                    "iteration":     train_step,
                    "policy_loss":   loss_info["policy_loss"],
                    "value_loss":    loss_info["value_loss"],
                    "buffer_size":   buffer.size,
                    "games_total":   games_played,
                    "games_per_hour": games_per_hour,
                    "gpu_util":      gpu_monitor.gpu_util_pct,
                    "vram_gb":       gpu_monitor.vram_used_gb,
                    "x_winrate":     x_winrate,
                    "o_winrate":     o_winrate,
                }

                if dashboard is not None:
                    dashboard.update(train_step, total_steps, metrics)

    if dashboard is not None:
        with dashboard.live():
            _run_loop()
    else:
        _run_loop()

    # ── Cleanup ──
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
