#!/usr/bin/env python3
"""
End-to-end training loop: self-play → buffer → train → repeat.

  1. Play self-play games until the buffer has enough data.
  2. Run one training step per self-play game (interleaved).
  3. Log structured JSON to file (structlog) every event.
  4. Emit structured events via emit_event() for dashboard renderers.
  5. Monitor GPU stats every 5 s via pynvml daemon thread.
  6. Save checkpoints every checkpoint_interval steps.
  7. Stop after --iterations steps (default: runs until Ctrl-C).

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
import math
import os
import random
import signal
import sys
import threading
import time
import tracemalloc
import uuid
from pathlib import Path

import numpy as np
import structlog
import torch
import yaml

# ── Ensure project root is on sys.path when run as a script ──────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexo_rl.monitoring.events import emit_event, register_renderer
from hexo_rl.selfplay.utils import N_ACTIONS as _N_ACTIONS
from hexo_rl.monitoring.gpu_monitor import GPUMonitor
from hexo_rl.monitoring.configure import configure_logging
from hexo_rl.model.network import HexTacToeNet
from engine import ReplayBuffer
from hexo_rl.training.trainer import Trainer


# ── Seeding ───────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 training loop")
    p.add_argument(
        "--config", default=None,
        help=(
            "Optional override config applied on top of base configs "
            "(configs/model.yaml + configs/training.yaml + configs/selfplay.yaml + "
            "configs/game_replay.yaml). Example: --config configs/fast_debug.yaml"
        ),
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
        "--variant", default=None,
        help=(
            "Named variant from configs/variants/ (e.g. gumbel_full, gumbel_targets, "
            "baseline_puct). Deep-merged on top of base configs after --config."
        ),
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable live dashboard renderers (useful in CI or non-interactive mode)"
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

    # ── Run ID (generated early so the log filename uses it) ──
    run_id = uuid.uuid4().hex

    # ── Load config ──
    # Load order: base configs → --config override → --variant override.
    # --variant deep-merges configs/variants/<name>.yaml on top of everything else.
    from hexo_rl.utils.config import load_config
    _BASE_CONFIGS = [
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/selfplay.yaml",
        "configs/game_replay.yaml",
        "configs/monitoring.yaml",
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
                f"Variant '{args.variant}' not found. "
                f"Available: {available}"
            )
        load_paths.append(str(variant_path))
    config = load_config(*load_paths)

    # ── Logging ──
    _log_run_name = args.run_name or f"train_{run_id}"
    log, _log_fh = configure_logging(log_dir=args.log_dir, run_name=_log_run_name)
    log.info("startup", config=config, variant=args.variant, pid=__import__("os").getpid())

    # ── Seed ──
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    log.info("seeded", seed=seed)

    # ── Device ──
    from hexo_rl.utils.device import best_device
    device = best_device()
    log.info("device", device=str(device))

    # ── Model + Trainer ──
    # Base configs use nested sections (model:, training:, selfplay:, mcts:).
    # Flatten into one dict for Trainer/MCTS/SelfPlay usage, with nested
    # sections taking precedence over any top-level keys.
    model_config = config.get("model", {})
    train_config = config.get("training", {})
    mcts_config  = config.get("mcts", {})
    self_config  = config.get("selfplay", {})

    combined_config = {
        **config,        # top-level keys (seed, etc.)
        **model_config,  # model section overrides
        **train_config,  # training section overrides
        **mcts_config,   # mcts section overrides
        **self_config,   # selfplay section overrides
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
        # Always propagate torch_compile from the merged YAML config so a
        # checkpoint saved with torch_compile=true doesn't re-enable compilation
        # on resume (the embedded checkpoint config would otherwise win).
        config_overrides: dict = {"torch_compile": combined_config.get("torch_compile", False)}
        # Propagate new training keys that may not exist in older checkpoints.
        for _key in ("uncertainty_weight", "recency_weight", "ownership_weight", "threat_weight"):
            if _key in combined_config:
                config_overrides[_key] = combined_config[_key]
        if args.iterations is not None and args.override_scheduler_horizon:
            config_overrides["total_steps"] = int(args.iterations)

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

    # Re-read architecture from the trainer — load_checkpoint infers these from
    # the state dict, so they may differ from the CLI config (e.g. resuming a
    # 12-block/128-filter pretrain checkpoint while fast_debug.yaml says 4/64).
    board_size         = int(trainer.config.get("board_size",         board_size))
    res_blocks         = int(trainer.config.get("res_blocks",         res_blocks))
    filters            = int(trainer.config.get("filters",            filters))
    in_channels        = int(trainer.config.get("in_channels",        18))
    se_reduction_ratio = int(trainer.config.get("se_reduction_ratio", 4))

    # ── Inference model — separate instance owned exclusively by InferenceServer ──
    # train_model (trainer.model) stays on the main thread for gradient updates.
    # inf_model lives on the InferenceServer daemon thread for forward-only passes.
    # torch.compile disabled — Python 3.14 compatibility issues
    # See sprint log §25, §30 for history
    # Re-enable when PyTorch + Python 3.14 CUDA graph support stabilizes
    _torch_compile_enabled = (
        trainer.config.get("torch_compile", False)
        and device.type == "cuda"
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

    def _sync_weights_to_inf() -> None:
        """Copy train_model weights → inf_model.

        Called after every checkpoint save and after every model promotion.
        load_state_dict on _orig_mod updates parameters in-place; weights
        are picked up on the next forward pass.
        """
        train_base = getattr(trainer.model, "_orig_mod", trainer.model)
        inf_base = getattr(inf_model, "_orig_mod", inf_model)
        inf_base.load_state_dict(train_base.state_dict())

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
    buffer = ReplayBuffer(capacity=capacity)
    schedule_idx = 1  # first entry already applied at construction

    # ── Game-length weight schedule ──
    glw = train_cfg.get("game_length_weights", config.get("game_length_weights", {}))
    if glw:
        glw_thresholds = [int(t) for t in glw["thresholds"]]
        glw_weights = [float(w) for w in glw["weights"]]
        glw_default = float(glw.get("default_weight", 1.0))
        buffer.set_weight_schedule(glw_thresholds, glw_weights, glw_default)
        log.info("replay_buffer.weight_schedule_set",
                 thresholds=glw_thresholds, weights=glw_weights,
                 default_weight=glw_default)

    log.info("buffer_init", capacity=capacity, min_buffer_size=min_buf_size,
             schedule_entries=len(buffer_schedule))

    # ── Buffer restore (on resume) ──
    mixing_cfg = train_cfg.get("mixing", config.get("mixing", {}))
    # Minimum restored positions required to consider the buffer "well-restored"
    # and skip corpus prefill.  1770 positions (1.7% full) is not enough.
    _MIN_BUFFER_PREFILL_SKIP = 10_000
    _buffer_restored = False
    _n_buffer_restored = 0
    if args.checkpoint and mixing_cfg.get("buffer_persist", False):
        _bp_resume = Path(mixing_cfg.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
        if _bp_resume.exists():
            try:
                n_loaded = buffer.load_from_path(str(_bp_resume))
                _n_buffer_restored = n_loaded
                log.info("buffer_restored", path=str(_bp_resume), positions=n_loaded,
                         capacity=buffer.capacity)
                emit_event({
                    "event": "system_stats",
                    "buffer_size": buffer.size,
                    "buffer_capacity": buffer.capacity,
                })
                # Only skip corpus prefill if we restored a meaningful number of positions.
                _buffer_restored = n_loaded >= _MIN_BUFFER_PREFILL_SKIP
            except Exception as e:
                log.warning("buffer_restore_failed", error=str(e))

    # ── Pretrained buffer (mixed data streams) ──
    # The pretrained_buffer is ALWAYS initialised when the corpus NPZ exists — even
    # on resume — so that batch mixing (pretrained_weight > 0) works correctly.
    # _buffer_restored only gates the "corpus_prefill_skipped" log message; it no
    # longer prevents pretrained_buffer from being loaded.
    pretrained_buffer = None
    pretrained_path = mixing_cfg.get("pretrained_buffer_path")
    if _buffer_restored:
        log.info("corpus_prefill_skipped",
                 msg="buffer well-restored from file — corpus prefill not required",
                 buffer_size=buffer.size,
                 restored_positions=_n_buffer_restored)
    elif _n_buffer_restored > 0:
        # Buffer was partially restored but below threshold — run prefill on top.
        log.info("corpus_prefill_running",
                 restored_positions=_n_buffer_restored,
                 reason="buffer_too_small",
                 threshold=_MIN_BUFFER_PREFILL_SKIP)

    if pretrained_path and Path(pretrained_path).exists():
        log.info("loading_corpus_npz", path=pretrained_path,
                 msg="copying corpus into Rust pretrained_buffer — may take minutes for large corpora")
        t0 = time.time()
        data = np.load(pretrained_path, mmap_mode='r')
        pre_states = data["states"]       # (T, 18, 19, 19) float16
        pre_policies = data["policies"]   # (T, 362) float32
        pre_outcomes = data["outcomes"]   # (T,) float32
        T = len(pre_outcomes)
        max_pre = int(mixing_cfg.get("pretrain_max_samples", 0))
        if max_pre and T > max_pre:
            _pre_seed = int(config.get("seed", 42))
            _pre_rng = np.random.default_rng(_pre_seed)
            subset_idx = np.sort(_pre_rng.choice(T, size=max_pre, replace=False))
            log.info("corpus_capped", original=T, kept=max_pre)
            pre_states   = pre_states[subset_idx]
            pre_policies = pre_policies[subset_idx]
            pre_outcomes = pre_outcomes[subset_idx]
            T = max_pre
        file_mb = os.path.getsize(pretrained_path) / 1e6
        log.info("corpus_prefill", positions=T, file_mb=round(file_mb, 1))
        if T > 100_000:
            log.warning(
                "corpus_prefill_oversized",
                positions=T,
                msg="NPZ has >100K positions — cold-start only needs 50K. "
                    "Run 'make corpus.npz' to regenerate with the optimized pipeline.",
            )
        # push_game copies all arrays into the Rust buffer (not mmap'd).
        # Estimate RAM: T × (18×19×19×2 + 362×4 + 4) bytes ≈ T × 14.1 KB
        est_ram_gb = T * 14_448 / (1024 ** 3)
        if est_ram_gb > 2.0:
            log.warning(
                "corpus_prefill_high_ram",
                path=pretrained_path,
                n_positions=T,
                estimated_ram_gb=round(est_ram_gb, 1),
                msg="push_game allocates full corpus in RAM — training starts after this completes",
            )
        pretrained_buffer = ReplayBuffer(capacity=T)
        pretrained_buffer.push_game(pre_states, pre_policies, pre_outcomes)
        del pre_states, pre_policies, pre_outcomes   # release mmap; Rust has the data
        del data
        log.info("corpus_loaded", positions=T, seconds=f"{time.time()-t0:.1f}")
        emit_event({
            "event": "system_stats",
            "buffer_size": buffer.size,
            "buffer_capacity": capacity,
        })
    elif pretrained_path:
        log.warning(
            "corpus_npz_not_found",
            path=pretrained_path,
            msg="No corpus NPZ found — skipping pretrained mixing. "
                "Buffer will fill from self-play only. Run 'make corpus.npz' to generate.",
        )

    mixing_decay_steps = float(mixing_cfg.get("decay_steps", 1_000_000))
    if mixing_decay_steps <= 0:
        raise ValueError(f"mixing.decay_steps must be > 0, got {mixing_decay_steps}")
    mixing_min_w = float(mixing_cfg.get("min_pretrained_weight", 0.1))
    mixing_initial_w = float(mixing_cfg.get("initial_pretrained_weight", 0.8))

    def compute_pretrained_weight(step: int) -> float:
        return max(mixing_min_w, mixing_initial_w * math.exp(-step / mixing_decay_steps))

    # ── Recent buffer for recency-weighted sampling ──
    # Holds the newest ~50% of buffer capacity as a Python ring (no Rust changes needed).
    # The pool stats thread will also push positions here as they arrive.
    from hexo_rl.training.recency_buffer import RecentBuffer
    _recency_weight = float(train_cfg.get("recency_weight", 0.0))
    recent_buffer: RecentBuffer | None = None
    if _recency_weight > 0.0:
        _recent_cap = max(256, capacity // 2)
        recent_buffer = RecentBuffer(capacity=_recent_cap)
        log.info("recent_buffer_init", capacity=_recent_cap, recency_weight=_recency_weight)

    # ── Pre-allocated batch arrays — avoids large malloc/free cycling each step ──
    _batch_size_cfg = int(train_cfg.get("batch_size", config.get("batch_size", 256)))
    _states_buf   = np.empty((_batch_size_cfg, 18, 19, 19), dtype=np.float16)
    _policies_buf = np.empty((_batch_size_cfg, _N_ACTIONS), dtype=np.float32)
    _outcomes_buf = np.empty(_batch_size_cfg, dtype=np.float32)
    # Flips to False the first step buffer is full enough to use in-place copy.
    _warmup_fallback_active = True

    # ── Self-play pool ──
    from hexo_rl.selfplay.pool import WorkerPool
    from hexo_rl.eval.eval_pipeline import EvalPipeline

    pool = WorkerPool(inf_model, config, device, buffer)
    if recent_buffer is not None:
        pool.recent_buffer = recent_buffer

    log.info("run_id", run_id=run_id)

    # ── Evaluation pipeline (Phase 4.0) ──
    eval_yaml_path = Path("configs/eval.yaml")
    eval_pipeline: EvalPipeline | None = None
    if eval_yaml_path.exists():
        eval_ext_config = load_config(str(eval_yaml_path))
        ep_cfg = eval_ext_config.get("eval_pipeline", {})
        if ep_cfg.get("enabled", False):
            eval_pipeline = EvalPipeline(eval_ext_config, device, run_id=run_id)
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

    # ── Dashboard renderers ──
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

    # ── Graceful shutdown ──
    _running = [True]
    _stop_count = [0]
    _shutdown_save = [False]  # set by signal handler; checked in loop to save before pool.stop()

    def _stop(sig, frame):
        _stop_count[0] += 1
        if _stop_count[0] >= 2:
            # Second Ctrl+C — force exit immediately (e.g. pool.stop() hung)
            sys.exit(1)
        log.info("shutdown_requested", msg="finishing current step… press Ctrl+C again to force")
        _shutdown_save[0] = True
        _running[0] = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── Loop ──
    log_interval  = int(config.get("log_interval",  10))
    eval_interval = int(train_cfg.get("eval_interval", config.get("eval_interval", 100)))
    training_steps_per_game = float(train_cfg.get("training_steps_per_game", 1.0))
    max_train_burst = int(train_cfg.get("max_train_burst", 8))
    train_step    = trainer.step
    # --iterations is relative to the starting step
    stop_step = (trainer.step + args.iterations) if args.iterations else None
    games_played  = 0
    t_start       = time.time()
    initial_policy_loss: float | None = None
    last_loss_info: dict[str, float] = {}

    # ── Emit run_start ──
    emit_event({
        "event": "run_start",
        "step": train_step,
        "run_id": run_id,
        "worker_count": pool.n_workers,
        "config_summary": {
            "n_blocks": int(combined_config.get("res_blocks", 12)),
            "channels": int(combined_config.get("filters", 128)),
            "n_sims": int(mcts_config.get("n_simulations", 800)),
            "buffer_capacity": capacity,
        },
    })

    if args.checkpoint and "pretrain" in Path(args.checkpoint).name:
        pretrain_log = Path(args.log_dir) / "pretrain.jsonl"
        if pretrain_log.exists():
            import json
            replay_evs = []
            try:
                with open(pretrain_log, "r") as f:
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
                
            if replay_evs:
                log.info("replaying_pretrain_events", count=len(replay_evs[-500:]))
                for ev in replay_evs[-500:]:
                    emit_event(ev)

    # Start multiprocess self-play
    pool.start()
    log.info("selfplay_pool_started", n_workers=pool.n_workers)

    # ── Iteration stats accumulator ──
    _iter_games_window: list[tuple[float, int]] = []  # (timestamp, game_count)
    _iter_window_sec = 60.0

    def _games_per_hour_rolling() -> float:
        """Compute games/hr over a 60s rolling window."""
        now = time.time()
        _iter_games_window.append((now, games_played))
        # Trim old entries
        cutoff = now - _iter_window_sec
        while _iter_games_window and _iter_games_window[0][0] < cutoff:
            _iter_games_window.pop(0)
        if len(_iter_games_window) < 2:
            elapsed = max(now - t_start, 1e-6)
            return games_played / elapsed * 3600
        dt = _iter_games_window[-1][0] - _iter_games_window[0][0]
        dg = _iter_games_window[-1][1] - _iter_games_window[0][1]
        return (dg / max(dt, 1e-6)) * 3600

    def _run_loop() -> None:
        nonlocal train_step, games_played, initial_policy_loss, last_loss_info
        nonlocal _eval_thread, best_model, best_model_step

        last_train_game_count = 0
        last_warmup_log = 0.0
        last_iter_games = 0

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
                # Save buffer immediately — before pool.stop() runs in the finally block.
                # A second Ctrl+C would call sys.exit(1) and skip the post-loop block.
                _mix_sd = train_cfg.get("mixing", config.get("mixing", {}))
                if _mix_sd.get("buffer_persist", False):
                    _bp_sd = Path(_mix_sd.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
                    try:
                        buffer.save_to_path(str(_bp_sd))
                        log.info("buffer_saved", path=str(_bp_sd), positions=buffer.size, trigger="shutdown_signal")
                    except Exception as _bp_exc:
                        log.warning("buffer_save_failed", path=str(_bp_sd), error=str(_bp_exc))
                break

            # ── Training Throttling ──
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

            steps_budget = max(1, int(round(new_games * training_steps_per_game)))
            steps_budget = min(steps_budget, max_train_burst)
            last_train_game_count = games_played

            for _ in range(steps_budget):
                if stop_step is not None and train_step >= stop_step:
                    break

                # ── Buffer growth schedule ──
                nonlocal schedule_idx, _warmup_fallback_active
                while schedule_idx < len(buffer_schedule) and train_step >= buffer_schedule[schedule_idx]["step"]:
                    new_cap = buffer_schedule[schedule_idx]["capacity"]
                    if new_cap > buffer.capacity:
                        buffer.resize(new_cap)
                        log.info("buffer_resized", step=train_step, new_capacity=new_cap)
                    schedule_idx += 1

                # ── Training step (mixed or single buffer) ──
                batch_size = int(train_cfg.get("batch_size", config.get("batch_size", 256)))
                # Sample auxiliary targets from pool's recent-game ring buffer.
                own_targets, thr_targets = pool.get_aux_targets(batch_size)
                if pretrained_buffer is not None and pretrained_buffer.size > 0 and buffer.size > 0:
                    w_pre = compute_pretrained_weight(train_step)
                    n_pre = max(1, int(math.ceil(batch_size * w_pre)))
                    n_self = batch_size - n_pre
                    s_pre, p_pre, o_pre = pretrained_buffer.sample_batch(n_pre, True)
                    if batch_size != _batch_size_cfg:
                        # Edge case: runtime batch size differs from pre-allocated shape.
                        if train_step > 100:
                            log.warning("mixed_batch_size_mismatch",
                                        batch_size=batch_size, expected=_batch_size_cfg)
                        if recent_buffer is not None and recent_buffer.size > 0 and _recency_weight > 0.0 and n_self > 1:
                            n_self_recent = max(1, int(round(n_self * _recency_weight)))
                            n_self_uniform = n_self - n_self_recent
                            s_r, p_r, o_r = recent_buffer.sample(n_self_recent)
                            s_u, p_u, o_u = buffer.sample_batch(max(1, n_self_uniform), True)
                            s_self = np.concatenate([s_r, s_u], axis=0)
                            p_self = np.concatenate([p_r, p_u], axis=0)
                            o_self = np.concatenate([o_r, o_u], axis=0)
                        else:
                            s_self, p_self, o_self = buffer.sample_batch(max(1, n_self), True)
                        states   = np.concatenate([s_pre, s_self], axis=0)
                        policies = np.concatenate([p_pre, p_self], axis=0)
                        outcomes = np.concatenate([o_pre, o_self], axis=0)
                    else:
                        # Fill pre-allocated buffers in-place — no heap churn.
                        if recent_buffer is not None and recent_buffer.size > 0 and _recency_weight > 0.0 and n_self > 1:
                            n_self_recent = max(1, int(round(n_self * _recency_weight)))
                            n_self_uniform = n_self - n_self_recent
                            s_r, p_r, o_r = recent_buffer.sample(n_self_recent)
                            s_u, p_u, o_u = buffer.sample_batch(max(1, n_self_uniform), True)
                            n_r = len(s_r)
                            n_u = len(s_u)
                            n_available = n_pre + n_r + n_u
                            if n_available < batch_size:
                                # Warm-up: buffer returned fewer rows than requested — fall back to
                                # concatenate (allocates, but only happens until buffers fill).
                                states   = np.concatenate([s_pre, s_r, s_u], axis=0)
                                policies = np.concatenate([p_pre, p_r, p_u], axis=0)
                                outcomes = np.concatenate([o_pre, o_r, o_u], axis=0)
                            else:
                                if _warmup_fallback_active:
                                    log.info("buffer_warmup_ended", step=train_step,
                                             n_available=n_available, batch_size=batch_size)
                                    _warmup_fallback_active = False
                                np.copyto(_states_buf[:n_pre], s_pre)
                                np.copyto(_states_buf[n_pre:n_pre + n_r], s_r)
                                np.copyto(_states_buf[n_pre + n_r:], s_u)
                                np.copyto(_policies_buf[:n_pre], p_pre)
                                np.copyto(_policies_buf[n_pre:n_pre + n_r], p_r)
                                np.copyto(_policies_buf[n_pre + n_r:], p_u)
                                np.copyto(_outcomes_buf[:n_pre], o_pre)
                                np.copyto(_outcomes_buf[n_pre:n_pre + n_r], o_r)
                                np.copyto(_outcomes_buf[n_pre + n_r:], o_u)
                                states, policies, outcomes = _states_buf, _policies_buf, _outcomes_buf
                        else:
                            s_self, p_self, o_self = buffer.sample_batch(max(1, n_self), True)
                            n_s = len(s_self)
                            n_available = n_pre + n_s
                            if n_available < batch_size:
                                # Warm-up: buffer returned fewer rows than requested — fall back.
                                states   = np.concatenate([s_pre, s_self], axis=0)
                                policies = np.concatenate([p_pre, p_self], axis=0)
                                outcomes = np.concatenate([o_pre, o_self], axis=0)
                            else:
                                if _warmup_fallback_active:
                                    log.info("buffer_warmup_ended", step=train_step,
                                             n_available=n_available, batch_size=batch_size)
                                    _warmup_fallback_active = False
                                np.copyto(_states_buf[:n_pre], s_pre)
                                np.copyto(_states_buf[n_pre:], s_self)
                                np.copyto(_policies_buf[:n_pre], p_pre)
                                np.copyto(_policies_buf[n_pre:], p_self)
                                np.copyto(_outcomes_buf[:n_pre], o_pre)
                                np.copyto(_outcomes_buf[n_pre:], o_self)
                                states, policies, outcomes = _states_buf, _policies_buf, _outcomes_buf
                    loss_info = trainer.train_step_from_tensors(
                        states, policies, outcomes,
                        ownership_targets=own_targets, threat_targets=thr_targets,
                        n_pretrain=n_pre,
                    )
                else:
                    w_pre = 0.0
                    loss_info = trainer.train_step(
                        buffer, recent_buffer=recent_buffer,
                        ownership_targets=own_targets, threat_targets=thr_targets,
                    )
                train_step = trainer.step
                if initial_policy_loss is None:
                    initial_policy_loss = float(loss_info["policy_loss"])
                last_loss_info = loss_info

                # Sync inference model weights after each checkpoint save.
                if train_step % _ckpt_interval == 0:
                    _sync_weights_to_inf()

                # ── Evaluation (non-blocking via background thread) ──
                if eval_pipeline is not None and train_step > 0 and train_step % eval_interval == 0:
                    # Harvest previous eval result if ready
                    if _eval_thread is not None and not _eval_thread.is_alive():
                        prev = _eval_result[0]
                        if prev is not None:
                            # Emit eval_complete event
                            emit_event({
                                "event": "eval_complete",
                                "step": prev.get("step", train_step),
                                "elo_estimate": prev.get("elo_estimate"),
                                "win_rate_vs_sealbot": prev.get("wr_sealbot", 0.0),
                                "eval_games": prev.get("eval_games", 0),
                                "gate_passed": prev.get("promoted", False),
                            })
                            if prev.get("promoted"):
                                base_model = getattr(trainer.model, "_orig_mod", trainer.model)
                                best_model.load_state_dict(base_model.state_dict())
                                best_model.eval()
                                torch.save(best_model.state_dict(), best_model_path)
                                best_model_step = train_step
                                _sync_weights_to_inf()
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
                                result = eval_pipeline.run_evaluation(
                                    _model, _step, _best, full_config=_cfg,
                                )
                                result["step"] = _step
                                _eval_result[0] = result
                            except Exception:
                                import traceback
                                log.info("evaluation_error", step=_step, tb=traceback.format_exc())
                                _eval_result[0] = {"promoted": False, "error": True, "step": _step}

                        _eval_thread = threading.Thread(target=_run_eval, daemon=True)
                        _eval_thread.start()
                    else:
                        log.info("eval_skipped_still_running", step=train_step)

                # ── tracemalloc heap snapshot every 500 steps ──
                if train_step > 0 and train_step % 500 == 0:
                    try:
                        snapshot = tracemalloc.take_snapshot()
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

                # ── Emit training_step event ──
                if train_step % log_interval == 0:
                    elapsed = time.time() - t_start

                    # Policy entropy: -sum(p * log(p + eps)) averaged over batch
                    policy_entropy = float(loss_info.get("policy_entropy", 0.0))

                    # Value accuracy: fraction where sign(pred) == sign(target)
                    value_accuracy = float(loss_info.get("value_accuracy", 0.0))

                    # Gradient norm (NaN if not computed)
                    grad_norm = float(loss_info.get("grad_norm", float("nan")))

                    # Current LR
                    lr = float(loss_info.get("lr", 0.0))

                    emit_event({
                        "event": "training_step",
                        "step": train_step,
                        "loss_total": float(loss_info["loss"]),
                        "loss_policy": float(loss_info["policy_loss"]),
                        "loss_value": float(loss_info["value_loss"]),
                        "loss_aux": float(loss_info.get("opp_reply_loss", 0.0)),
                        "avg_sigma": float(loss_info.get("avg_sigma", 0.0)),
                        "policy_entropy": policy_entropy,
                        "policy_entropy_pretrain": float(loss_info.get("policy_entropy_pretrain", float("nan"))),
                        "policy_entropy_selfplay": float(loss_info.get("policy_entropy_selfplay", float("nan"))),
                        "policy_target_entropy": float(loss_info.get("policy_target_entropy", 0.0)),
                        "value_accuracy": value_accuracy,
                        "lr": lr,
                        "grad_norm": grad_norm,
                    })

                    # Buffer composition: self-play positions as fraction of total
                    _buf_total = max(buffer.size, 1)
                    _sp_pushed = pool.self_play_positions_pushed
                    _buf_sp_pct = round(min(_sp_pushed / _buf_total, 1.0), 4)
                    # Actual batch mix: fraction of each batch from self-play vs pretrained
                    _batch_selfplay_frac = round(1.0 - w_pre, 4)

                    # Compute iteration stats
                    games_this_iter = games_played - last_iter_games
                    last_iter_games = games_played
                    gph = _games_per_hour_rolling()
                    avg_gl = pool.avg_game_length if hasattr(pool, "avg_game_length") else 0.0
                    pph = gph * avg_gl if avg_gl > 0 else 0.0

                    emit_event({
                        "event": "iteration_complete",
                        "step": train_step,
                        "games_total": games_played,
                        "games_this_iter": games_this_iter,
                        "games_per_hour": round(gph, 1),
                        "positions_per_hour": round(pph, 1),
                        "avg_game_length": round(avg_gl, 1),
                        "win_rate_p0": round(float(pool.x_winrate), 4),
                        "win_rate_p1": round(float(pool.o_winrate), 4),
                        "draw_rate": round(float(pool.draws / games_played), 4) if games_played > 0 else 0.0,
                        "sims_per_sec": pool.sims_per_sec or 0.0,
                        "buffer_size": buffer.size,
                        "buffer_capacity": buffer.capacity,
                        "corpus_selfplay_frac": _batch_selfplay_frac,
                        "batch_fill_pct": pool.batch_fill_pct,
                        "mcts_mean_depth": float(getattr(pool._runner, "mcts_mean_depth", 0.0)),
                        "mcts_root_concentration": float(getattr(pool._runner, "mcts_mean_root_concentration", 0.0)),
                    })

                    log.info(
                        "train_step",
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
                        batch_fill_pct=round(pool.batch_fill_pct, 1),
                        inf_forward_count=pool._inference_server._forward_count,
                        inf_total_requests=pool._inference_server._total_requests,
                    )

    tracemalloc.start(3)
    try:
        _run_loop()
    finally:
        tracemalloc.stop()
        emit_event({"event": "run_end", "step": train_step})
        pool.stop()
        gpu_monitor.stop()
        gpu_monitor.join(timeout=2.0)
        for d in dashboards:
            try:
                d.stop()
            except Exception:
                pass

    # ── Session end: save final checkpoint and buffer, then log summary ──
    final_ckpt = trainer.save_checkpoint(last_loss_info if last_loss_info else None)
    _mix_cfg_end = train_cfg.get("mixing", config.get("mixing", {}))
    if _mix_cfg_end.get("buffer_persist", False):
        _bp_end = Path(_mix_cfg_end.get("buffer_persist_path", "checkpoints/replay_buffer.bin"))
        try:
            buffer.save_to_path(str(_bp_end))
            log.info("buffer_saved", path=str(_bp_end), positions=buffer.size)
        except Exception as _bp_err:
            log.warning("buffer_save_failed", error=str(_bp_err))
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

    try:
        _log_fh.close()
    except Exception:
        pass

    # Suppress spurious "leaked semaphore" warning on exit.
    # The semaphore originates from Rust PyO3 OS primitives (SelfPlayRunner /
    # InferenceBatcher condvars) that Python's resource_tracker cannot own or
    # release. Silencing the tracker prevents a noisy but harmless warning.
    try:
        import multiprocessing.resource_tracker
        multiprocessing.resource_tracker._resource_tracker._stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
