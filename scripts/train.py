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

import numpy as np
import structlog
import torch

# ── Cap PyTorch inter-op thread count to match the cgroup allocation ─────────
# PyTorch reads `OMP_NUM_THREADS` for intra-op threads but has no env hook for
# inter-op (`torch.get_num_interop_threads()` defaults to host nproc). On a
# rented vast.ai container with N threads allocated out of a 128-thread host,
# the default is 128 — 3-4x oversubscription against the cgroup, manifesting
# as 100% CPU saturation, 60% GPU util, and self-play workers starved of
# inference dispatches. set_num_interop_threads must run BEFORE any parallel
# torch work; doing it here keeps a single import-time call site.
# Honoured precedence: TORCH_INTEROP_THREADS > OMP_NUM_THREADS > leave default.
_interop = int(os.environ.get("TORCH_INTEROP_THREADS", os.environ.get("OMP_NUM_THREADS", "0")))
if _interop > 0:
    try:
        torch.set_num_interop_threads(_interop)
    except RuntimeError:
        # Already set by an earlier import that touched the parallel runtime.
        pass

# ── Ensure project root is on sys.path when run as a script ──────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexo_rl.monitoring.configure import configure_logging
from hexo_rl.model.network import HexTacToeNet
from engine import ReplayBuffer
from hexo_rl.training.trainer import Trainer
from hexo_rl.training.batch_assembly import allocate_batch_buffers, load_pretrained_buffer
from hexo_rl.training.loop import run_training_loop
from hexo_rl.selfplay.utils import N_ACTIONS as _N_ACTIONS
from hexo_rl.monitoring.events import emit_event


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 self-play training loop")
    p.add_argument(
        "--config", default=None,
        help=(
            "Optional override config applied on top of base configs "
            "(configs/model.yaml + configs/training.yaml + configs/selfplay.yaml + "
            "configs/game_replay.yaml). Example: --config configs/fast_debug.yaml"
        ),
    )
    p.add_argument("--checkpoint", default=None,
                   help="Resume from this checkpoint file (optional)")
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
            "Named variant from configs/variants/ (e.g. gumbel_full, gumbel_targets, "
            "baseline_puct). Deep-merged on top of base configs after --config."
        ),
    )
    p.add_argument("--no-dashboard", action="store_true",
                   help="Disable live dashboard renderers (useful in CI or non-interactive mode)")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile (useful for debugging)")
    p.add_argument("--min-buffer-size", type=int, default=None,
                   help="Override replay warmup size before first training step")
    return p.parse_args()


def main() -> None:
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    args   = parse_args()
    run_id = uuid.uuid4().hex

    # ── Load config ───────────────────────────────────────────────────────────
    from hexo_rl.utils.config import load_config
    _BASE_CONFIGS = [
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/selfplay.yaml",
        "configs/game_replay.yaml",
        "configs/monitoring.yaml",
        "configs/monitors.yaml",
    ]
    load_paths: list[str] = list(_BASE_CONFIGS)
    if args.config:
        override = str(Path(args.config).resolve())
        load_paths = [p for p in load_paths if str(Path(p).resolve()) != override]
        load_paths.append(args.config)
    if args.variant:
        variant_path = Path(f"configs/variants/{args.variant}.yaml")
        if not variant_path.exists():
            available = sorted(p.stem for p in Path("configs/variants").glob("*.yaml"))
            raise FileNotFoundError(
                f"Variant '{args.variant}' not found. Available: {available}"
            )
        load_paths.append(str(variant_path))
    config = load_config(*load_paths)

    # ── Logging ───────────────────────────────────────────────────────────────
    _log_run_name = args.run_name or f"train_{run_id}"
    log, _log_fh = configure_logging(log_dir=args.log_dir, run_name=_log_run_name)
    log.info("startup", config=config, variant=args.variant, pid=os.getpid())

    # ── Seed + device ─────────────────────────────────────────────────────────
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    log.info("seeded", seed=seed)

    from hexo_rl.utils.device import best_device
    device = best_device()
    log.info("device", device=str(device))

    # Per-host TF32 configuration (§117). Applies backend flags once; safe
    # no-op on CPU. See hexo_rl/model/tf32.py.
    from hexo_rl.model.tf32 import resolve_and_apply as _tf32_resolve_and_apply
    _tf32_resolved = _tf32_resolve_and_apply(config)
    log.info("tf32_applied", **_tf32_resolved)

    # ── Config flattening ─────────────────────────────────────────────────────
    model_config = config.get("model", {})
    train_config = config.get("training", {})
    mcts_config  = config.get("mcts", {})
    self_config  = config.get("selfplay", {})
    combined_config = {
        **config,
        **model_config,
        **train_config,
        **mcts_config,
        **self_config,
    }
    train_cfg = config.get("training", config)

    if args.iterations is not None:
        combined_config["total_steps"] = int(args.iterations)
        train_cfg["total_steps"]       = int(args.iterations)
    if args.no_compile:
        combined_config["torch_compile"] = False
        train_cfg["torch_compile"]       = False

    board_size = int(combined_config.get("board_size", 19))
    res_blocks = int(combined_config.get("res_blocks", 10))
    filters    = int(combined_config.get("filters",    128))

    # ── Trainer (resume or fresh) ─────────────────────────────────────────────
    if args.checkpoint:
        config_overrides: dict = {"torch_compile": combined_config.get("torch_compile", False)}
        # §116: torch_compile_mode must travel alongside torch_compile on resume.
        # Without this, pre-§116 checkpoints resume at mode="default" regardless of
        # configs/training.yaml's new reduce-overhead setting.
        if combined_config.get("torch_compile_mode") is not None:
            config_overrides["torch_compile_mode"] = combined_config["torch_compile_mode"]
        for _key in (
            "uncertainty_weight", "recency_weight", "ownership_weight", "threat_weight",
            "eta_min", "scheduler_t_max",
        ):
            if combined_config.get(_key) is not None:
                config_overrides[_key] = combined_config[_key]
        if args.override_scheduler_horizon and combined_config.get("total_steps") is not None:
            config_overrides["total_steps"] = int(combined_config["total_steps"])
        if args.allow_fresh_scheduler:
            config_overrides["allow_fresh_scheduler"] = True
        # Perf-investigation: plumb diagnostics section so --config overrides
        # the checkpoint-baked config for probe flags. Without this, probes
        # cannot be enabled on a resumed run (checkpoint config wins by default).
        if isinstance(combined_config.get("diagnostics"), dict):
            config_overrides["diagnostics"] = combined_config["diagnostics"]

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
            configured_total_steps=trainer.config.get("total_steps"),
        )
    else:
        # Sweep-variant support: configs may carry `input_channels: [list]` to
        # pick a subset of the 18 wire planes. When present, in_channels is
        # derived from the list length; otherwise the standard 18-plane model
        # is built.
        input_channels_cfg = combined_config.get("input_channels")
        if input_channels_cfg is None:
            in_channels_arg = int(combined_config.get("in_channels", 18))
        else:
            in_channels_arg = len(input_channels_cfg)
            combined_config["in_channels"] = in_channels_arg
        model = HexTacToeNet(
            board_size=board_size,
            res_blocks=res_blocks,
            filters=filters,
            in_channels=in_channels_arg,
            input_channels=input_channels_cfg,
        )
        trainer = Trainer(model, combined_config, checkpoint_dir=args.checkpoint_dir, device=device)
        log.info(
            "new_run",
            model_params=sum(p.numel() for p in model.parameters()),
            in_channels=in_channels_arg,
            input_channels=list(input_channels_cfg) if input_channels_cfg is not None else None,
        )

    log.info("run_id", run_id=run_id)

    # ── Replay buffer + growth schedule ───────────────────────────────────────
    buffer_schedule_raw = train_cfg.get("buffer_schedule", config.get("buffer_schedule", []))
    buffer_schedule = sorted(
        [{"step": int(e["step"]), "capacity": int(e["capacity"])} for e in buffer_schedule_raw],
        key=lambda x: x["step"],
    )
    capacity = (
        buffer_schedule[0]["capacity"]
        if buffer_schedule
        else int(config.get("buffer_capacity", train_cfg.get("buffer_capacity", 500_000)))
    )

    if args.min_buffer_size is not None:
        min_buf_size = int(args.min_buffer_size)
    elif train_cfg.get("min_buffer_size") is not None:
        min_buf_size = int(train_cfg["min_buffer_size"])
    elif config.get("min_buffer_size") is not None:
        min_buf_size = int(config["min_buffer_size"])
    else:
        min_buf_size = max(128, min(512, int(train_cfg.get("batch_size", config.get("batch_size", 256)))))

    buffer = ReplayBuffer(capacity=capacity)

    glw = train_cfg.get("game_length_weights", config.get("game_length_weights", {}))
    if glw:
        glw_thresholds = [int(t) for t in glw["thresholds"]]
        glw_weights    = [float(w) for w in glw["weights"]]
        glw_default    = float(glw.get("default_weight", 1.0))
        buffer.set_weight_schedule(glw_thresholds, glw_weights, glw_default)
        log.info("replay_buffer.weight_schedule_set",
                 thresholds=glw_thresholds, weights=glw_weights, default_weight=glw_default)

    log.info("buffer_init", capacity=capacity, min_buffer_size=min_buf_size,
             schedule_entries=len(buffer_schedule))

    # ── Buffer restore (on resume) ────────────────────────────────────────────
    mixing_cfg = train_cfg.get("mixing", config.get("mixing", {}))
    _MIN_BUFFER_PREFILL_SKIP = 10_000
    _buffer_restored = False
    if args.checkpoint and mixing_cfg.get("buffer_persist", False):
        _bp = Path(mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
        if _bp.exists():
            try:
                n_loaded = buffer.load_from_path(str(_bp))
                log.info("buffer_restored", path=str(_bp), positions=n_loaded, capacity=buffer.capacity)
                emit_event({"event": "system_stats", "buffer_size": buffer.size, "buffer_capacity": capacity})
                _buffer_restored = n_loaded >= _MIN_BUFFER_PREFILL_SKIP
                if not _buffer_restored:
                    log.info("corpus_prefill_running", restored_positions=n_loaded,
                             reason="buffer_too_small", threshold=_MIN_BUFFER_PREFILL_SKIP)
            except Exception as e:
                log.warning("buffer_restore_failed", error=str(e))
    if _buffer_restored:
        log.info("corpus_prefill_skipped", msg="buffer well-restored from file",
                 buffer_size=buffer.size)

    # ── Corpus loading ────────────────────────────────────────────────────────
    pretrained_buffer = load_pretrained_buffer(
        mixing_cfg, config, emit_event, buffer.size, capacity
    )

    # ── Recent buffer ─────────────────────────────────────────────────────────
    from hexo_rl.training.recency_buffer import RecentBuffer
    _recency_weight = float(train_cfg.get("recency_weight", 0.0))
    recent_buffer: RecentBuffer | None = None
    if _recency_weight > 0.0:
        _recent_cap = max(256, capacity // 2)
        recent_buffer = RecentBuffer(capacity=_recent_cap)
        log.info("recent_buffer_init", capacity=_recent_cap, recency_weight=_recency_weight)
        if _buffer_restored:
            _rbp = Path(str(_bp) + ".recent")
            if _rbp.exists():
                try:
                    _rn = recent_buffer.load_from_path(str(_rbp))
                    log.info("recent_buffer_restored", path=str(_rbp), positions=_rn)
                except Exception as _re:
                    log.warning("recent_buffer_restore_failed", error=str(_re))

    # ── Pre-allocated batch arrays ────────────────────────────────────────────
    _batch_size_cfg = int(train_cfg.get("batch_size", config.get("batch_size", 256)))
    bufs = allocate_batch_buffers(_batch_size_cfg, _N_ACTIONS)

    # ── Mixing schedule params ────────────────────────────────────────────────
    mixing_decay_steps = float(mixing_cfg.get("decay_steps", 1_000_000))
    if mixing_decay_steps <= 0:
        raise ValueError(f"mixing.decay_steps must be > 0, got {mixing_decay_steps}")
    mixing_min_w     = float(mixing_cfg.get("min_pretrained_weight",     0.1))
    mixing_initial_w = float(mixing_cfg.get("initial_pretrained_weight", 0.8))

    # ── Hand off to the training loop ─────────────────────────────────────────
    run_training_loop(
        trainer=trainer,
        buffer=buffer,
        pretrained_buffer=pretrained_buffer,
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
