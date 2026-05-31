"""Train.py orchestration helpers (§176 P70).

Extracted from ``scripts/train.py::main`` to reduce its body from
~388 LOC to a thin glue path. Each helper covers one section of the
original ``main`` and was moved verbatim — pure-move discipline per
``docs/refactor-template.md``: no body edits, only signature wrapping.

Helpers:

* :func:`load_train_config` — argparse → loaded config + variant validation.
* :func:`setup_seed_device_tf32` — seed + device + TF32 backend config.
* :func:`flatten_config_and_resolve_encoding` — flatten config sections,
  resolve registry, expand <auto> path literals.
* :func:`init_trainer` — resume-vs-fresh dispatcher (Trainer.load_checkpoint
  vs HexTacToeNet + Trainer ctor).
* :func:`init_replay_buffer` — capacity + min-buffer-size + weight schedule.
* :func:`restore_buffer_from_checkpoint` — load_from_path on resume.
* :func:`check_buffer_contamination` — §149 task 4d guard.
* :func:`init_recent_buffer` — RecentBuffer init + restore.
* :func:`allocate_batch_buffers_for_config` — §172 A4.3 sized arrays.
* :func:`read_mixing_params` — mixing schedule (decay_steps/min_w/initial_w).

INV13 invariant — the ``run_training_loop`` signature must remain
unchanged. These helpers emit the exact local variables that
``scripts/train.py::main`` passes through to ``run_training_loop``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from engine import ReplayBuffer
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.events import emit_event
from hexo_rl.selfplay.utils import N_ACTIONS as _N_ACTIONS
from hexo_rl.training.batch_assembly import allocate_batch_buffers
from hexo_rl.training.recency_buffer import RecentBuffer
from hexo_rl.training.trainer import Trainer


# ── Section 3: load config + variant validation ──────────────────────────────
def load_train_config(args: argparse.Namespace) -> dict:
    """Load base + override + variant configs; abort-on-warning validation.

    Mirrors ``scripts/train.py::main`` lines 181-226 verbatim.
    """
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

        # §176 P68 — abort-on-warning variant validation. Catches silent
        # namespace shadows before merge (e.g. variant declaring
        # ``training: {max_train_burst: 8}`` when base has flat
        # ``max_train_burst``).
        import yaml as _yaml
        from hexo_rl.utils.variant_validator import (
            _load_standard_base_cfgs,
            validate_variant_against_bases,
        )
        with open(variant_path) as _vf:
            _variant_cfg = _yaml.safe_load(_vf) or {}
        _base_cfgs = _load_standard_base_cfgs(Path(".").resolve())
        _warnings = validate_variant_against_bases(_variant_cfg, _base_cfgs)
        if _warnings:
            for _w in _warnings:
                print(f"variant_validator WARNING: {_w}", file=sys.stderr)
            raise RuntimeError(
                f"variant '{args.variant}' has {len(_warnings)} validator warning(s); "
                f"see stderr. Fix the variant or run with --no-validate (not implemented)."
            )

    return load_config(*load_paths)


# ── Section 5: seed + device + TF32 ──────────────────────────────────────────
def setup_seed_device_tf32(
    config: dict, log: "structlog.stdlib.BoundLogger"
) -> torch.device:
    """Seed RNGs, pick best device, apply per-host TF32 backend flags.

    Mirrors ``scripts/train.py::main`` lines 233-246 verbatim. Calls
    :func:`scripts.train.seed_everything` for seed RNGs to keep the
    public surface unchanged.
    """
    # Local import to avoid a circular import (scripts.train imports this
    # module). seed_everything is a thin wrapper around random / np / torch.
    from scripts.train import seed_everything

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
    return device


# ── Section 6: flatten + registry resolve + path expansion ───────────────────
def flatten_config_and_resolve_encoding(
    config: dict,
    args: argparse.Namespace,
    log: "structlog.stdlib.BoundLogger",
) -> tuple[dict, dict, dict, Any, int, int, int]:
    """Flatten config sections, override from CLI, resolve registry spec,
    expand <auto> paths.

    Returns
    -------
    combined_config, train_cfg, mcts_config, registry_spec, board_size,
    res_blocks, filters

    Mirrors ``scripts/train.py::main`` lines 248-289 verbatim.
    """
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

    # §172 A10: registry is the sole source of truth for trunk_size;
    # `combined_config["board_size"]` scalar retired. All downstream readers
    # (eval, model, selfplay) consume `resolve_from_config(cfg).trunk_size`.
    from hexo_rl.encoding import expand_auto_paths, resolve_from_config as _registry_resolve
    _registry_spec = _registry_resolve(combined_config)
    # §172 A10 / T9: expand <auto> corpus/anchor path literals now that the
    # encoding spec is resolved. Mutates combined_config in-place so all
    # downstream readers (batch_assembly, eval_pipeline) see the real paths.
    expand_auto_paths(combined_config, _registry_spec)
    # Trunk size for HexTacToeNet ctor: v6=19 canvas, v6w25=25 cluster
    # window, v8=25 bbox.
    board_size = _registry_spec.trunk_size
    log.info(
        "train_encoding_resolved",
        encoding_name=_registry_spec.name,
        board_size=board_size,
        n_planes=_registry_spec.n_planes,
        is_multi_window=_registry_spec.is_multi_window,
    )
    res_blocks = int(combined_config.get("res_blocks", 10))
    filters    = int(combined_config.get("filters",    128))
    return combined_config, train_cfg, mcts_config, _registry_spec, board_size, res_blocks, filters


def _resolve_fresh_in_channels(combined_config: dict) -> tuple[int, "list | None"]:
    """Resolve (in_channels, input_channels) for a fresh-run model build.

    §P5-CT P0-1 fix: when the variant omits both `input_channels` and
    `in_channels`, fall back to the RESOLVED encoding's plane count via
    `resolve_arch`, NEVER the legacy literal 18 (which is neither the wire
    width nor any registered encoding's n_planes — it built an 18-channel trunk
    against an n-plane wire for any variant lacking an explicit in_channels).

    Sweep-variant support preserved: an `input_channels: [list]` picks a subset
    of the source wire planes and derives in_channels from its length (mutating
    combined_config in-place, as before).
    """
    from hexo_rl.encoding import resolve_arch

    input_channels_cfg = combined_config.get("input_channels")
    if input_channels_cfg is None:
        fallback = resolve_arch(combined_config.get("encoding")).in_channels
        in_channels_arg = int(combined_config.get("in_channels", fallback))
    else:
        in_channels_arg = len(input_channels_cfg)
        combined_config["in_channels"] = in_channels_arg
    return in_channels_arg, input_channels_cfg


# ── Section 7: trainer init (resume vs fresh) ────────────────────────────────
def init_trainer(
    args: argparse.Namespace,
    combined_config: dict,
    device: torch.device,
    board_size: int,
    res_blocks: int,
    filters: int,
    log: "structlog.stdlib.BoundLogger",
) -> tuple[Trainer, int]:
    """Build / resume a Trainer; returns (trainer, possibly-updated board_size).

    On resume, board_size may be re-resolved from the registry after
    propagating ckpt-baked encoding back into combined_config.

    Mirrors ``scripts/train.py::main`` lines 291-379 verbatim.
    """
    from hexo_rl.encoding import resolve_from_config as _registry_resolve

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
        # Propagate the ckpt-resolved encoding back into the local config
        # dicts so downstream selfplay surfaces (pool, worker, lifecycle,
        # eval) read the encoding the model was actually trained under
        # rather than the variant YAML's default. Without this, a v6w25
        # bootstrap loaded against a v6-default variant would route
        # correctly inside the trainer but feed v6 plane geometry into
        # the self-play workers (§171 P3 blocker).
        for _enc_key in (
            "cluster_window_size", "cluster_threshold",
            "legal_move_radius", "encoding",
        ):
            if _enc_key in trainer.config:
                combined_config[_enc_key] = trainer.config[_enc_key]
        # §172 A10: board_size retired — re-resolve trunk_size from registry.
        try:
            _post_load_spec = _registry_resolve(combined_config)
            _registry_name_post = _post_load_spec.name
            board_size = _post_load_spec.trunk_size
        except Exception:
            _registry_name_post = None
        # `encoding` may be a dict ({"version": ...}, post-propagation) or a
        # bare string (variant YAML form, e.g. "v6tp", when the checkpoint
        # lacks encoding metadata and shape inference can't resolve a novel
        # plane count). Log either form; never assume dict (.get on a str
        # raised AttributeError — every other encoding read here already
        # routes through normalize_encoding_name / resolve_from_config).
        _enc_cfg = combined_config.get("encoding")
        _enc_version = _enc_cfg.get("version") if isinstance(_enc_cfg, dict) else _enc_cfg
        log.info(
            "resumed",
            checkpoint=args.checkpoint,
            step=trainer.step,
            configured_total_steps=trainer.config.get("total_steps"),
            encoding_version=_enc_version,
            registry_name=_registry_name_post,
            board_size=board_size,
            cluster_window_size=combined_config.get("cluster_window_size"),
            cluster_threshold=combined_config.get("cluster_threshold"),
        )
    else:
        in_channels_arg, input_channels_cfg = _resolve_fresh_in_channels(combined_config)
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
    return trainer, board_size


# ── Section 8: replay buffer + weight schedule ───────────────────────────────
def init_replay_buffer(
    args: argparse.Namespace,
    config: dict,
    train_cfg: dict,
    log: "structlog.stdlib.BoundLogger",
) -> tuple[ReplayBuffer, list, int, int]:
    """Build the replay buffer and resolve capacity / min_buffer_size /
    weight schedule.

    Returns
    -------
    buffer, buffer_schedule, capacity, min_buf_size

    Mirrors ``scripts/train.py::main`` lines 381-415 verbatim.
    """
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

    from hexo_rl.encoding import normalize_encoding_name as _normalize_encoding_name
    buffer = ReplayBuffer(capacity=capacity, encoding=_normalize_encoding_name(config.get("encoding")))

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
    return buffer, buffer_schedule, capacity, min_buf_size


# ── Section 9: buffer restore on resume ──────────────────────────────────────
def restore_buffer_from_checkpoint(
    args: argparse.Namespace,
    buffer: ReplayBuffer,
    capacity: int,
    mixing_cfg: dict,
    log: "structlog.stdlib.BoundLogger",
) -> tuple[bool, Optional[Path]]:
    """Restore replay buffer from on-disk persist file when resuming.

    Returns
    -------
    _buffer_restored, _bp

    ``_bp`` is the persist Path (used downstream by the recent buffer
    sidecar load); ``None`` when the persist branch never ran.

    Mirrors ``scripts/train.py::main`` lines 417-436 verbatim.
    """
    _MIN_BUFFER_PREFILL_SKIP = 10_000
    _buffer_restored = False
    _bp: Optional[Path] = None
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
    return _buffer_restored, _bp


# ── Section 10: buffer contamination guard (§149 task 4d) ────────────────────
def check_buffer_contamination(
    args: argparse.Namespace,
    trainer: Trainer,
    buffer: ReplayBuffer,
    mixing_cfg: dict,
    log: "structlog.stdlib.BoundLogger",
) -> None:
    """Always emit the pre-corpus buffer size so contamination is greppable.
    When training is being kicked off from a bootstrap-like checkpoint (step
    counter near 0 — the pretrain-end value) but the on-disk buffer was
    populated by a prior, possibly failed, run, log a loud warning. The
    legitimate resume path has a meaningful trainer.step from a self-play
    run, so this gate only fires on fresh-from-bootstrap launches that
    silently inherited stale self-play data.

    Mirrors ``scripts/train.py::main`` lines 438-471 verbatim.
    """
    _ckpt_path_str = str(args.checkpoint) if args.checkpoint else ""
    _looks_like_bootstrap_resume = bool(args.checkpoint) and any(
        marker in _ckpt_path_str for marker in ("bootstrap_model", "pretrain_")
    )
    log.info(
        "buffer_state_at_corpus_load",
        buffer_size_before_corpus_load=buffer.size,
        ckpt_step=trainer.step if args.checkpoint else None,
        ckpt_path=_ckpt_path_str or None,
        buffer_persist_enabled=mixing_cfg.get("buffer_persist", False),
    )
    if _looks_like_bootstrap_resume and buffer.size > 0 and trainer.step <= 0:
        log.warning(
            "buffer_contamination_suspected",
            msg=(
                "Bootstrap-like checkpoint (step <= 0) was loaded but the replay "
                "buffer is non-empty before corpus load. The on-disk "
                "replay_buffer.bin from a prior run was likely auto-restored. "
                "Delete checkpoints/replay_buffer.bin (and *.recent) before a "
                "fresh launch, or set training.mixing.buffer_persist=false on "
                "the variant. See §149 task 4c/4d."
            ),
            buffer_size=buffer.size,
            ckpt=_ckpt_path_str,
            ckpt_step=trainer.step,
        )


# ── Section 12: recent buffer ────────────────────────────────────────────────
def init_recent_buffer(
    train_cfg: dict,
    config: dict,
    capacity: int,
    registry_spec: Any,
    _bp: Optional[Path],
    _buffer_restored: bool,
    log: "structlog.stdlib.BoundLogger",
) -> tuple[Optional[RecentBuffer], float]:
    """Build the optional RecentBuffer + restore from the .recent sidecar.

    Returns
    -------
    recent_buffer, _recency_weight

    Mirrors ``scripts/train.py::main`` lines 478-499 verbatim.
    """
    _recency_weight = float(train_cfg.get("recency_weight", 0.0))
    recent_buffer: RecentBuffer | None = None
    if _recency_weight > 0.0:
        _recent_cap = max(256, capacity // 2)
        recent_buffer = RecentBuffer(
            capacity=_recent_cap,
            state_shape=(registry_spec.n_planes, registry_spec.trunk_size, registry_spec.trunk_size),
            policy_len=registry_spec.policy_logit_count,
            aux_stride=registry_spec.trunk_size * registry_spec.trunk_size,
        )
        log.info("recent_buffer_init", capacity=_recent_cap, recency_weight=_recency_weight,
                 state_shape=recent_buffer._states.shape[1:], policy_len=registry_spec.policy_logit_count)
        if _buffer_restored and _bp is not None:
            _rbp = Path(str(_bp) + ".recent")
            if _rbp.exists():
                try:
                    _rn = recent_buffer.load_from_path(str(_rbp))
                    log.info("recent_buffer_restored", path=str(_rbp), positions=_rn)
                except Exception as _re:
                    log.warning("recent_buffer_restore_failed", error=str(_re))
    return recent_buffer, _recency_weight


# ── Section 13: pre-allocated batch arrays ───────────────────────────────────
def allocate_batch_buffers_for_config(
    train_cfg: dict,
    config: dict,
    combined_config: dict,
    log: "structlog.stdlib.BoundLogger",
) -> tuple[Any, int]:
    """Re-resolve registry post-checkpoint-load and size buffers per
    trunk_size / policy_logit_count.

    Returns
    -------
    bufs, _batch_size_cfg

    Mirrors ``scripts/train.py::main`` lines 501-518 verbatim.
    """
    from hexo_rl.encoding import resolve_from_config as _registry_resolve

    # §172 A4.3: re-resolve registry post-checkpoint-load (resume path may
    # have rewritten encoding via metadata) and size buffers per trunk_size /
    # policy_logit_count. Falls through to v6 defaults on legacy v6 configs.
    _batch_size_cfg = int(train_cfg.get("batch_size", config.get("batch_size", 256)))
    _n_planes_spec = 8  # v6-family default; overridden from the resolved spec.
    try:
        _bufs_spec = _registry_resolve(combined_config)
        _trunk_size = int(_bufs_spec.trunk_size)
        _n_actions_spec = int(_bufs_spec.policy_logit_count)
        _n_planes_spec = int(_bufs_spec.n_planes)
    except Exception as _re_err:
        log.warning("buffer_alloc_registry_resolve_failed", error=str(_re_err)[:120])
        _trunk_size = int(combined_config.get("board_size", 19))  # fallback from config
        _n_actions_spec = _N_ACTIONS
    bufs = allocate_batch_buffers(
        _batch_size_cfg,
        _n_actions_spec,
        trunk_size=_trunk_size,
        n_planes=_n_planes_spec,
    )
    return bufs, _batch_size_cfg


# ── Section 14: mixing schedule params ───────────────────────────────────────
def read_mixing_params(mixing_cfg: dict) -> tuple[float, float, float]:
    """Resolve mixing.decay_steps / min_w / initial_w with the same
    validation as ``scripts/train.py::main`` lines 521-525.
    """
    mixing_decay_steps = float(mixing_cfg.get("decay_steps", 1_000_000))
    if mixing_decay_steps <= 0:
        raise ValueError(f"mixing.decay_steps must be > 0, got {mixing_decay_steps}")
    mixing_min_w     = float(mixing_cfg.get("min_pretrained_weight",     0.1))
    mixing_initial_w = float(mixing_cfg.get("initial_pretrained_weight", 0.8))
    return mixing_decay_steps, mixing_min_w, mixing_initial_w
