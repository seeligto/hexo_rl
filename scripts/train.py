#!/usr/bin/env python3
"""
Entry point: parse CLI, merge configs, build core objects, start training loop.

  Training loop logic lives in  hexo_rl/training/loop.py
  Mixed-batch assembly lives in hexo_rl/training/batch_assembly.py
  Aux target decoding lives in  hexo_rl/training/aux_decode.py

Usage:
    # Standard run — loads base configs automatically
    .venv/bin/python scripts/train.py
    # Override for smoke testing
    .venv/bin/python scripts/train.py --config configs/fast_debug.yaml
    # Resume from checkpoint with override
    .venv/bin/python scripts/train.py --config configs/fast_debug.yaml \\
        --checkpoint checkpoints/pretrain/pretrain_00016980.pt --iterations 1
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import uuid
from pathlib import Path

# ── sys.path + auto-thread-budget BEFORE numpy / torch import ───────────────
# Project root must be importable so we can pull in hexo_rl.utils.cpu_budget
# (stdlib-only, no torch). cpu_budget sets OMP / MKL / OPENBLAS / NUMEXPR /
# TORCH_INTEROP_THREADS to match the cgroup allocation; without this, on a
# rented container with N threads carved out of an M-thread host every BLAS
# op grabs M threads against the N-slot cgroup → CPU saturation, GPU
# starvation. Same two calls live at the top of scripts/benchmark.py.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hexo_rl.utils.cpu_budget import apply_auto_thread_budget, apply_torch_interop_cap
from hexo_rl.utils.config import load_config as _load_config_early


def build_argparser(*, peek_only: bool = False) -> argparse.ArgumentParser:
    """Single source of truth for the train.py CLI.

    ``peek_only=True`` returns an ``add_help=False`` parser exposing only
    ``--config`` / ``--variant`` — used by :func:`_peek_n_workers` before
    numpy/torch import to partial-parse the worker count for thread-budget
    sizing. ``peek_only=False`` (default) returns the full parser used by
    :func:`parse_args`.
    """
    if peek_only:
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument("--config", default=None)
        p.add_argument("--variant", default=None)
        return p

    p = argparse.ArgumentParser(description="Phase 4 self-play training loop")
    p.add_argument(
        "--config", default=None,
        help=(
            "Optional override config applied on top of base configs "
            "(configs/model.yaml + configs/training.yaml + configs/selfplay.yaml + "
            "configs/game_replay.yaml). Example: --config configs/fast_debug.yaml"
        ),
    )
    p.add_argument("--checkpoint", "--bootstrap", default=None,
                   help="Resume from this checkpoint file (optional). "
                        "Encoding is auto-detected from checkpoint metadata "
                        "and reconciled against the variant config; a "
                        "version mismatch raises ValueError before training "
                        "starts (Trainer._resolve_checkpoint_encoding).")
    p.add_argument("--iterations", type=int, default=None,
                   help="Stop after this many training steps (default: run until Ctrl-C)")
    p.add_argument(
        "--override-scheduler-horizon", action="store_true",
        help=(
            "When resuming from a full checkpoint with --iterations, override "
            "checkpoint total_steps so the LR scheduler horizon follows the CLI value"
        ),
    )
    p.add_argument(
        "--allow-fresh-scheduler", action="store_true",
        help=(
            "Allow resume from a checkpoint that does not contain scheduler_state. "
            "Default: raise. Use only for legacy checkpoints predating scheduler persistence."
        ),
    )
    p.add_argument("--log-dir", default="logs",
                   help="Directory for structlog JSON files (default: logs/)")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Directory for checkpoint files (default: checkpoints/)")
    p.add_argument("--run-name", default=None,
                   help="Run identifier for log file name (default: timestamp)")
    p.add_argument(
        "--variant", default=None,
        help=(
            "Named variant from configs/variants/ (e.g. gumbel_full, gumbel_targets). "
            "Deep-merged on top of base configs after --config."
        ),
    )
    p.add_argument("--no-dashboard", action="store_true",
                   help="Disable live dashboard renderers (useful in CI or non-interactive mode)")
    p.add_argument("--no-web-dashboard", action="store_true",
                   help="Disable ONLY the Flask-SocketIO web dashboard, keep the terminal "
                        "dashboard. Prolonged-run hygiene: the web-socket teardown raised "
                        "exit-134 (SIGABRT) AFTER the final checkpoint saved on the O1 + "
                        "post-fix 30k runs (benign, but masks the real exit code). "
                        "Use for the 200-300k run.")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile (useful for debugging)")
    p.add_argument("--min-buffer-size", type=int, default=None,
                   help="Override replay warmup size before first training step")
    return p


def _peek_n_workers() -> int | None:
    """Partial-parse ``--config`` / ``--variant`` so the auto-tune can size
    the per-lib slice for the actual self-play worker count. Both load_config
    and the YAML reader are torch-free, so this can run before numpy / torch.
    Failures fall through to None (auto-tune uses the no-workers heuristic).
    """
    try:
        early, _ = build_argparser(peek_only=True).parse_known_args()
        paths = [
            "configs/model.yaml", "configs/training.yaml",
            "configs/selfplay.yaml", "configs/game_replay.yaml",
            "configs/monitoring.yaml", "configs/monitors.yaml",
        ]
        if early.config:
            paths.append(early.config)
        if early.variant:
            vp = f"configs/variants/{early.variant}.yaml"
            if Path(vp).exists():
                paths.append(vp)
        cfg = _load_config_early(*paths)
        sp = cfg.get("selfplay", {})
        n = sp.get("n_workers")
        return int(n) if n is not None else None
    except Exception:
        return None


apply_auto_thread_budget(n_workers=_peek_n_workers(), log_prefix="[hexo_rl train]")


import numpy as np
import structlog
import torch

apply_torch_interop_cap()

from hexo_rl.monitoring.configure import configure_logging
from hexo_rl.training.batch_assembly import load_bot_corpus_buffer, load_pretrained_buffer
from hexo_rl.training.loop import run_training_loop
from hexo_rl.monitoring.events import emit_event  # re-exported for back-compat
from hexo_rl.training import orchestrator as _orchestrator


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    return build_argparser().parse_args()


def main() -> None:
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    args   = parse_args()
    run_id = uuid.uuid4().hex

    # ── Load config ───────────────────────────────────────────────────────────
    config = _orchestrator.load_train_config(args)

    # ── Logging ───────────────────────────────────────────────────────────────
    _log_run_name = args.run_name or f"train_{run_id}"
    log, _log_fh = configure_logging(log_dir=args.log_dir, run_name=_log_run_name)
    log.info("startup", config=config, variant=args.variant, pid=os.getpid())

    # ── Seed + device + TF32 ──────────────────────────────────────────────────
    device = _orchestrator.setup_seed_device_tf32(config, log)

    # ── Config flattening + registry resolve + <auto> path expansion ──────────
    (combined_config, train_cfg, mcts_config, _registry_spec,
     board_size, res_blocks, filters) = _orchestrator.flatten_config_and_resolve_encoding(
        config, args, log,
    )

    # ── Trainer (resume or fresh) ─────────────────────────────────────────────
    trainer, board_size = _orchestrator.init_trainer(
        args, combined_config, device, board_size, res_blocks, filters, log,
    )

    log.info("run_id", run_id=run_id)

    # ── Replay buffer + growth schedule ───────────────────────────────────────
    buffer, buffer_schedule, capacity, min_buf_size = _orchestrator.init_replay_buffer(
        args, config, train_cfg, log,
    )

    # ── Buffer restore (on resume) ────────────────────────────────────────────
    mixing_cfg = train_cfg.get("mixing", config.get("mixing", {}))
    _buffer_restored, _bp = _orchestrator.restore_buffer_from_checkpoint(
        args, buffer, capacity, mixing_cfg, log,
    )

    # ── Buffer contamination guard (§149 task 4d) ─────────────────────────────
    _orchestrator.check_buffer_contamination(args, trainer, buffer, mixing_cfg, log)

    # ── Corpus loading ────────────────────────────────────────────────────────
    pretrained_buffer = load_pretrained_buffer(
        mixing_cfg, config, emit_event, buffer.size, capacity
    )

    # ── §178 bot-corpus loading (None for §177-style runs) ────────────────────
    bot_buffer = load_bot_corpus_buffer(
        mixing_cfg, config, emit_event, buffer.size, capacity
    )

    # ── Recent buffer ─────────────────────────────────────────────────────────
    recent_buffer, _recency_weight = _orchestrator.init_recent_buffer(
        train_cfg, config, capacity, _registry_spec, _bp, _buffer_restored, log,
    )

    # ── Pre-allocated batch arrays ────────────────────────────────────────────
    bufs, _batch_size_cfg = _orchestrator.allocate_batch_buffers_for_config(
        train_cfg, config, combined_config, log,
    )

    # ── Mixing schedule params ────────────────────────────────────────────────
    mixing_decay_steps, mixing_min_w, mixing_initial_w = _orchestrator.read_mixing_params(mixing_cfg)

    # ── Hand off to the training loop ─────────────────────────────────────────
    run_training_loop(
        trainer=trainer,
        buffer=buffer,
        pretrained_buffer=pretrained_buffer,
        bot_buffer=bot_buffer,
        recent_buffer=recent_buffer,
        bufs=bufs,
        config=combined_config,
        train_cfg=train_cfg,
        mcts_config=mcts_config,
        args=args,
        device=device,
        run_id=run_id,
        capacity=capacity,
        min_buf_size=min_buf_size,
        buffer_schedule=buffer_schedule,
        recency_weight=_recency_weight,
        batch_size_cfg=_batch_size_cfg,
        mixing_cfg=mixing_cfg,
        mixing_initial_w=mixing_initial_w,
        mixing_min_w=mixing_min_w,
        mixing_decay_steps=mixing_decay_steps,
    )

    try:
        _log_fh.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
