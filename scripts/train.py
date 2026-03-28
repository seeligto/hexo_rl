#!/usr/bin/env python3
"""
End-to-end training loop: self-play → buffer → train → repeat.

Phase 1 design (single-process, no worker pool, no rich dashboard):
  1. Play self-play games until the buffer has enough data.
  2. Run one training step per self-play game (interleaved).
  3. Log to structlog JSON file every log_interval steps.
  4. Save checkpoints every checkpoint_interval steps.
  5. Stop after --iterations steps (default: runs until Ctrl-C).

Usage:
    .venv/bin/python scripts/train.py --config configs/fast_debug.yaml
    .venv/bin/python scripts/train.py --config configs/fast_debug.yaml \\
        --checkpoint checkpoints/checkpoint_00000100.pt
    .venv/bin/python scripts/train.py --config configs/fast_debug.yaml \\
        --iterations 500

Phase 1 exit criteria (from docs/02_roadmap.md):
  - Runs 1 hour without crashing.
  - Policy loss decreases from its initial value.
  - Value loss converges below 0.5.
  - Checkpoint save/load round-trips correctly.
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

from python.logging.setup import configure_logging
from python.model.network import HexTacToeNet
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
    t_start       = time.time()

    # Track initial policy loss for exit-criteria logging
    initial_policy_loss: float | None = None
    last_loss_info = {}

    print(f"Training on {device}. Press Ctrl-C to stop cleanly.")

    while _running[0]:
        if args.iterations and train_step >= args.iterations:
            log.info("iteration_limit_reached", iterations=args.iterations)
            break

        # ── Self-play: play one game ──
        t_game = time.time()
        n_positions = worker.play_game(buffer)
        game_elapsed = time.time() - t_game
        games_played += 1

        log.info(
            "game_complete",
            game_id=games_played,
            plies=n_positions,
            duration_sec=round(game_elapsed, 2),
            buffer_size=buffer.size,
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
            elapsed = time.time() - t_start
            log.info(
                "train_step",
                step=train_step,
                policy_loss=round(loss_info["policy_loss"], 4),
                value_loss=round(loss_info["value_loss"],   4),
                total_loss=round(loss_info["loss"],         4),
                buffer_size=buffer.size,
                games_played=games_played,
                elapsed_sec=round(elapsed, 1),
            )

            # Console summary (Phase 1: plain print, no rich dashboard)
            print(
                f"step={train_step:6d}  "
                f"loss={loss_info['loss']:.4f}  "
                f"policy={loss_info['policy_loss']:.4f}  "
                f"value={loss_info['value_loss']:.4f}  "
                f"buf={buffer.size}  "
                f"games={games_played}  "
                f"t={elapsed:.0f}s"
            )

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
