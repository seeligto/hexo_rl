"""Bootstrap pretrain CLI entry point (§176 P39 split from pretrain.py).

Contains:
  - pretrain — argparse-driven main + module-level orchestration. Loads
    configs, resolves encoding, builds the model, mmaps the corpus NPZ
    (or falls back to load_corpus), runs train_epoch loops, saves
    checkpoints, and (under v6-family encodings) calls validate.

Usage:
    python -m hexo_rl.bootstrap.pretrain [--epochs N] [--steps N] [--batch-size N]
    make pretrain          # 15 epochs

This module re-exports `pretrain` via the legacy `hexo_rl.bootstrap.pretrain`
shim — Makefile and external callers continue to invoke
`python -m hexo_rl.bootstrap.pretrain`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import structlog
import torch
import torch.optim as optim
from rich.console import Console

from hexo_rl.augment.luts import get_policy_scatters  # noqa: F401  (kept for symmetry w/ legacy module)
from hexo_rl.bootstrap.paths import QUALITY_SCORES_PATH
from hexo_rl.bootstrap.pretrain_dataset import (
    AugmentedBootstrapDataset,
    make_augmented_collate,
)
from hexo_rl.bootstrap.pretrain_freeze import _apply_finetune_freeze
from hexo_rl.bootstrap.pretrain_legacy import load_corpus
from hexo_rl.bootstrap.pretrain_trainer import BootstrapTrainer
from hexo_rl.bootstrap.pretrain_validate import validate
from hexo_rl.encoding import (
    all_specs as _all_specs,
    lookup as _lookup_encoding,
    resolve_corpus_path as _resolve_corpus_path,
    resolve_from_checkpoint as _resolve_encoding_from_ckpt,
    resolve_from_config as _registry_resolve_cfg,
)
from hexo_rl.encoding.registry import EncodingRegistryError as _EncodingRegistryError
from hexo_rl.model.network import HexTacToeNet, compile_model
from hexo_rl.training.checkpoints import get_base_model

log = structlog.get_logger()
console = Console()


# ── Entry point ───────────────────────────────────────────────────────────────

def pretrain() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap pretrain pipeline")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of full passes over the dataset")
    parser.add_argument("--steps", type=int, default=None,
                        help="Hard step budget (overrides epochs; for smoke tests)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile even if config enables it")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a full pretrain checkpoint (model+optimizer+scaler state). "
                             "Cosine LR schedule is restarted across --epochs at --lr-peak (or config default).")
    parser.add_argument("--lr-peak", type=float, default=None,
                        help="Override peak LR for cosine restart (used with --resume; "
                             "lower than original 2e-3 to fine-tune without disturbing learned weights).")
    parser.add_argument("--inference-out", type=str, default=None,
                        help="Override inference-weights output path "
                             "(default: checkpoints/bootstrap_model.pt). "
                             "Use a v7e30-style filename for --resume runs.")
    parser.add_argument("--eta-min", type=float, default=None,
                        help="Override CosineAnnealingLR eta_min "
                             "(default 1e-5). For 30-epoch full retrains, "
                             "5e-5 avoids the §149-observed LR-floor stall "
                             "in the final 3 epochs.")
    # ── v8 / Phase B variant CLI overrides ─────────────────────────────────
    _registered_encodings = tuple(s.name for s in _all_specs())
    parser.add_argument("--encoding", choices=_registered_encodings, default=None,
                        help="Override encoding.version from configs/model.yaml. "
                             "Registered encodings: " + ", ".join(_registered_encodings) + ". "
                             "Routes corpus NPZ and model construction accordingly.")
    parser.add_argument("--filters", type=int, default=None,
                        help="Override trunk channel count (model.yaml: filters).")
    parser.add_argument("--res-blocks", type=int, default=None,
                        help="Override trunk depth (model.yaml: res_blocks).")
    parser.add_argument("--gpool-sites", type=str, default=None,
                        help="Comma-separated trunk gpool indices for v8 "
                             "(e.g. '6,10' for a 12-block trunk). Empty string "
                             "disables trunk gpool (B0 control). v6 ignores this.")
    parser.add_argument("--head-no-gpool", action="store_true",
                        help="Drop the G branch from the v8 policy / opp_reply head. "
                             "B0 control arm uses this. v6 ignores.")
    parser.add_argument("--pool-type", choices=("min_max", "pma", "pma_global"), default=None,
                        help="K-cluster pool type for v6 / v6w25. 'min_max' "
                             "(default) keeps existing per-cluster heads + "
                             "bot-side scatter-max; 'pma' (§169 A2) replaces "
                             "the value/policy heads with a Set-Transformer "
                             "1×SAB + 2 PMA seeds aggregator. 'pma_global' "
                             "(§169 A3) extends 'pma' with a global summary "
                             "token branch (32×32 cur/opp/canvas-mask crop "
                             "+ KataGo gpool + learned scalar gate); requires "
                             "global_crops in the corpus NPZ. v8 ignores.")
    parser.add_argument("--pool-attn-dropout", type=float, default=None,
                        help="Attention dropout for PMA pool (collapse "
                             "mitigation). Default 0.1; raise to 0.2 if "
                             "the §169 PMA-collapse smoke fires.")
    parser.add_argument("--canvas-realness", action="store_true",
                        help="§169 A4 — invert v8 plane 8 polarity to "
                             "canvas_realness (1 inside, 0 outside) and wire "
                             "PartialConv2d (Innamorati 2018 partial-conv-padding) "
                             "at the trunk entry. v8-only. Requires a corpus "
                             "regenerated with --canvas-realness.")
    parser.add_argument(
        "--gpool-bias-active", action="store_true",
        help="§170 P3 — A1 + gpool-bias side-branch (additive K-invariant "
             "global-pool bias to value/policy heads). Requires --pool-type "
             "min_max + global_crops in the corpus NPZ. v6/v6w25 only.",
    )
    parser.add_argument(
        "--policy-only-bias", action="store_true",
        help="§170 P4 — confine the gpool-bias side-branch to the policy "
             "head (value head structurally frozen at A1: value_bias = 0, "
             "no gradient through value_proj). Requires --gpool-bias-active. "
             "Discriminates whether the §170 P3 MCTS regression is caused by "
             "value-head bias drift specifically.",
    )
    parser.add_argument("--corpus-npz", type=str, default=None,
                        help="Override corpus NPZ path. Default resolved from "
                             "encoding registry (resolve_corpus_path).")
    parser.add_argument(
        "--freeze-trunk-entry", action="store_true",
        help="§171 A4 fine-tune — freeze trunk.input_conv (PartialConv2d when "
             "canvas_realness) + trunk.input_gn so the trunk-entry stays at the "
             "weights learned during canvas_realness pretraining.",
    )
    parser.add_argument(
        "--unfreeze-blocks", type=str, default=None,
        help="§171 A4 fine-tune — CSV indices of trunk.tower blocks to keep "
             "trainable (e.g. '8,9,10,11'). All other blocks freeze; heads "
             "(policy/opp_reply/value/value_var) stay trainable. Unset = all "
             "blocks remain trainable.",
    )
    args = parser.parse_args()

    # §174 W1 — when --resume is set and --encoding is unset, auto-detect
    # the encoding from the resume checkpoint's metadata (or shape-inference
    # fallback). Lets `pretrain --resume <ckpt>` work without re-specifying
    # encoding for transfer scenarios.
    if args.resume is not None and args.encoding is None:
        try:
            _resume_spec = _resolve_encoding_from_ckpt(args.resume)
        except _EncodingRegistryError as e:
            raise SystemExit(
                f"--resume {args.resume!r}: could not auto-detect encoding "
                f"(no metadata + shape-inference failed). Pass --encoding "
                f"explicitly. Underlying error: {e}"
            ) from e
        args.encoding = _resume_spec.name
        log.info("auto_detected_encoding_from_resume_ckpt",
                 name=args.encoding, resume=args.resume)

    # Load configs
    from hexo_rl.utils.config import load_config
    config: Dict = load_config("configs/model.yaml", "configs/training.yaml")
    corpus_cfg: Dict = load_config("configs/corpus.yaml")
    if args.batch_size:
        config["batch_size"] = args.batch_size
    batch_size = int(config.get("batch_size", 512))
    label_smoothing = float(corpus_cfg.get("label_smoothing_default", 0.05))
    aux_weight = float(config.get("aux_opp_reply_weight", 0.15))
    source_weights: Dict[str, float] = corpus_cfg.get("source_weights", {})

    # ── v8 / variant overrides — CLI takes precedence over model.yaml ──────
    if args.encoding is not None:
        config["encoding"] = args.encoding
        # §172 A10 T6 — `board_size` / `in_channels` were retired from configs in
        # favor of registry lookup, but `configs/model.yaml` still carries the v6
        # defaults so legacy code paths keep working. CLI override means the
        # registry is canonical; drop scattered keys so resolve_from_config does
        # not raise on the now-stale model.yaml values.
        for stale_key in ("board_size", "in_channels", "n_planes",
                          "cluster_window_size", "cluster_threshold",
                          "legal_move_radius"):
            config.pop(stale_key, None)
    enc_section = config.get("encoding")
    if isinstance(enc_section, str):
        encoding = enc_section
    elif isinstance(enc_section, dict):
        encoding = str(enc_section.get("version", "v6"))
    else:
        encoding = "v6"
    # Validate encoding name against canonical registry.
    _ = _lookup_encoding(encoding)

    # Encoding-specific shape overrides — single source of truth via registry.
    # board_size / in_channels / n_actions scalars retired from configs (§172 A10).
    _enc_spec = _lookup_encoding(encoding)
    config["in_channels"] = _enc_spec.n_planes
    explicit_n_actions = _enc_spec.n_actions

    # Variant overrides — used by Phase B B0..B4 retrains.
    if args.filters is not None:
        config["filters"] = int(args.filters)
    if args.res_blocks is not None:
        config["res_blocks"] = int(args.res_blocks)

    # Parse --gpool-sites into a list of ints (or empty list if explicitly "").
    gpool_indices: Optional[List[int]] = None
    if args.gpool_sites is not None:
        if args.gpool_sites.strip() == "":
            gpool_indices = []
        else:
            gpool_indices = [int(s.strip()) for s in args.gpool_sites.split(",")
                             if s.strip()]
    head_use_gpool: bool = not args.head_no_gpool

    # §169 K-cluster pool selector — defaults to 'min_max' (current behavior).
    pool_type: str = (
        args.pool_type
        if args.pool_type is not None
        else str(config.get("pool_type", "min_max"))
    )
    pool_attn_dropout: float = (
        float(args.pool_attn_dropout)
        if args.pool_attn_dropout is not None
        else float(config.get("pool_attn_dropout", 0.1))
    )

    # §169 A4 — canvas_realness gates inverted plane-8 + PartialConv2d at
    # trunk entry. v8-only; surfaced loudly otherwise.
    canvas_realness: bool = bool(args.canvas_realness or config.get("canvas_realness", False))
    if canvas_realness and encoding != "v8":
        raise ValueError(
            f"--canvas-realness requires --encoding v8; got {encoding!r}"
        )

    # §170 P3 — A1 + gpool-bias. K-cluster-only (v6/v6w25), pool_type='min_max',
    # mutually exclusive w/ canvas_realness + trunk gpool sites. Mirror the
    # model-constructor invariants so YAML typos fail at CLI parse, not silently
    # mid-training. Model double-checks at construction (network.py).
    gpool_bias_active: bool = bool(
        args.gpool_bias_active or config.get("gpool_bias_active", False)
    )
    policy_only_bias: bool = bool(
        args.policy_only_bias or config.get("policy_only_bias", False)
    )
    if policy_only_bias and not gpool_bias_active:
        raise ValueError(
            "--policy-only-bias requires --gpool-bias-active; the policy-only "
            "knob configures the GpoolBiasBranch and has no effect without "
            "the branch being active."
        )
    if gpool_bias_active:
        if encoding == "v8":
            raise ValueError(
                "--gpool-bias-active requires K-cluster encoding (v6/v6w25); "
                f"v8 has no K dim. Got encoding={encoding!r}."
            )
        if pool_type != "min_max":
            raise ValueError(
                "--gpool-bias-active requires --pool-type min_max; got "
                f"pool_type={pool_type!r}. The pma / pma_global pools already "
                "carry a global-token branch; gpool-bias is the A1-only analog."
            )
        if canvas_realness:
            raise ValueError(
                "--gpool-bias-active is incompatible with --canvas-realness "
                "(canvas_realness is v8-only; gpool_bias is K-cluster-only)."
            )
        if gpool_indices:
            raise ValueError(
                "--gpool-bias-active is incompatible with non-empty "
                f"--gpool-sites; got {gpool_indices!r}. Trunk gpool sites are "
                "an in-trunk feature-mixing intervention; gpool-bias is the "
                "additive head-level analog."
            )

    log.info(
        "encoding_resolved",
        encoding=encoding,
        board_size=_registry_resolve_cfg(config).trunk_size,
        in_channels=int(config.get("in_channels", _registry_resolve_cfg(config).n_planes)),
        filters=int(config.get("filters", 128)),
        res_blocks=int(config.get("res_blocks", 12)),
        gpool_indices=gpool_indices,
        head_use_gpool=head_use_gpool,
        pool_type=pool_type,
        pool_attn_dropout=pool_attn_dropout,
        canvas_realness=canvas_realness,
        gpool_bias_active=gpool_bias_active,
        policy_only_bias=policy_only_bias,
    )

    from hexo_rl.utils.device import best_device
    device = best_device()
    console.print(f"[bold]Pretrain — device:[/bold] {device}")

    # Quality scores
    quality_path = QUALITY_SCORES_PATH
    quality_scores: Dict = {}
    if quality_path.exists():
        with open(quality_path) as f:
            quality_scores = json.load(f)
        log.info("loaded_quality_scores", n_games=len(quality_scores))
    else:
        log.warning("no_quality_scores", hint="Run `make corpus.analysis` to generate them")

    # Corpus — prefer mmap'd NPZ to avoid 2× RAM peak from load_corpus()
    console.print("[bold]Loading corpus...[/bold]")
    if args.corpus_npz is not None:
        npz_path = Path(args.corpus_npz)
    else:
        corpus_enc = encoding
        if encoding == "v8" and canvas_realness:
            corpus_enc = "v8_canvas_realness"
        npz_path = _resolve_corpus_path(_lookup_encoding(corpus_enc))
    global_crops_array: Optional[np.ndarray] = None
    if npz_path.exists():
        log.info("loading_corpus_from_npz", path=str(npz_path))
        data = np.load(npz_path, mmap_mode='r')
        states   = data['states']    # memory-mapped, not loaded into RAM
        policies = data['policies']
        outcomes = data['outcomes']
        weights  = data['weights']
        # §169 A3 — opt-in global-crop array. Present iff the NPZ was built
        # with --with-global-crop. Required when pool_type='pma_global'.
        if 'global_crops' in data.files:
            global_crops_array = data['global_crops']
    else:
        log.warning("npz_not_found_falling_back_to_load_corpus", path=str(npz_path))
        states, policies, outcomes, weights = load_corpus(quality_scores, source_weights)

    # §170 P3 — gpool_bias_active also consumes global_crops. Same NPZ as
    # pma_global; the gate=0 init keeps A1 byte-exact at construction.
    needs_global = (pool_type == "pma_global") or gpool_bias_active
    if needs_global and global_crops_array is None:
        raise RuntimeError(
            f"pool_type='{pool_type}' / gpool_bias_active={gpool_bias_active} "
            f"requires a corpus NPZ with global_crops; none found in {npz_path}. "
            f"Regenerate via `python scripts/export_corpus_npz.py --encoding v6w25 "
            f"--with-global-crop --human-only --no-compress`."
        )
    if not needs_global and global_crops_array is not None:
        # Don't waste IO / GPU memory; drop the unused array.
        log.info(
            "ignoring_global_crops_in_npz",
            reason=f"pool_type={pool_type!r} + gpool_bias_active={gpool_bias_active}",
        )
        global_crops_array = None

    if len(outcomes) == 0:
        console.print("[red]No corpus data found. Run `make corpus.fetch` first.[/red]")
        return

    log.info(
        "dataset_built",
        n_positions=int(len(outcomes)),
        quality_scores_loaded=bool(quality_scores),
        source_weights=source_weights,
    )
    console.print(f"Dataset: {len(outcomes):,} positions")

    dataset = AugmentedBootstrapDataset(
        states, policies, outcomes, global_crops=global_crops_array,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=make_augmented_collate(
            augment=True,
            encoding=encoding,
            with_global_crop=(global_crops_array is not None),
        ),
    )

    # One-off timing of the Rust aug path so any throughput regression vs the
    # pre-Q13 Python _apply_hex_sym path is visible in the pretrain console log.
    _t0 = time.perf_counter()
    _probe_batches = min(20, len(loader))
    _it = iter(loader)
    for _ in range(_probe_batches):
        next(_it)
    _dt = time.perf_counter() - _t0
    console.print(
        f"[bold]Aug-path probe:[/bold] {_probe_batches} batches of "
        f"{batch_size} via Rust apply_symmetries_batch = {_dt*1000:.1f} ms "
        f"({(_dt/_probe_batches)*1000:.2f} ms/batch)"
    )

    # Model — encoding-aware. v8 wires gpool sites + KataGo policy head.
    model = HexTacToeNet(
        board_size=_registry_resolve_cfg(config).trunk_size,
        in_channels=int(config["in_channels"]),
        filters=int(config["filters"]),
        res_blocks=int(config["res_blocks"]),
        se_reduction_ratio=int(config.get("se_reduction_ratio", 4)),
        encoding=encoding,
        gpool_indices=gpool_indices,
        head_use_gpool=head_use_gpool,
        pool_type=pool_type,
        pool_attn_dropout=pool_attn_dropout,
        canvas_realness=canvas_realness,
        gpool_bias_active=gpool_bias_active,
        policy_only_bias=policy_only_bias,
    )
    use_compile = (
        config.get("torch_compile", True)
        and device.type == "cuda"
        and not args.no_compile
    )
    if use_compile:
        model = compile_model(model, mode="default")

    checkpoint_dir = Path(args.checkpoint_dir)
    # Compute total steps before creating trainer so the scheduler T_max is exact.
    step_budget = args.steps
    total_pretrain_steps = step_budget if step_budget is not None else args.epochs * len(loader)
    config["pretrain_total_steps"] = total_pretrain_steps
    if args.eta_min is not None:
        config["pretrain_eta_min"] = float(args.eta_min)
    # Persist v8 / variant knobs in the saved checkpoint config so post-hoc
    # consumers (eval pipeline, threat probe, viewer) can reconstruct the model
    # without re-deriving from CLI flags.
    config["gpool_indices"] = gpool_indices
    config["head_use_gpool"] = head_use_gpool
    config["pool_type"] = pool_type
    config["pool_attn_dropout"] = pool_attn_dropout
    config["canvas_realness"] = canvas_realness
    config["gpool_bias_active"] = gpool_bias_active
    config["policy_only_bias"] = policy_only_bias
    config["n_actions"] = explicit_n_actions
    trainer = BootstrapTrainer(model, config, device, checkpoint_dir)

    # Resume mode: load model/optimizer/scaler from full checkpoint, restart
    # cosine schedule across the new --epochs window with the requested peak LR.
    # Step counter is reset so the new cosine completes over the new run length.
    if args.resume:
        resume_path = Path(args.resume)
        log.info("resume_loading", path=str(resume_path))
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        weights_only = isinstance(resume_ckpt, dict) and "model_state" not in resume_ckpt
        if weights_only:
            log.info("resume_weights_only_mode", reason="ckpt has no model_state key — treating as inference state_dict; optimizer/scaler reset")
            get_base_model(trainer.model).load_state_dict(resume_ckpt)
        else:
            get_base_model(trainer.model).load_state_dict(resume_ckpt["model_state"])
            trainer.optimizer.load_state_dict(resume_ckpt["optimizer_state"])
            if resume_ckpt.get("scaler_state") is not None:
                trainer.scaler.load_state_dict(resume_ckpt["scaler_state"])
        new_peak = float(args.lr_peak) if args.lr_peak is not None else float(config.get("lr", 0.002))
        new_eta_min = float(args.eta_min) if args.eta_min is not None else 1e-5
        for g in trainer.optimizer.param_groups:
            g["lr"] = new_peak
            g["initial_lr"] = new_peak
        trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=max(1, total_pretrain_steps), eta_min=new_eta_min,
        )
        log.info("resume_complete", new_peak_lr=new_peak,
                 cosine_t_max=total_pretrain_steps,
                 prev_step=int(resume_ckpt.get("step", 0)) if not weights_only else 0,
                 weights_only=weights_only)

    if args.freeze_trunk_entry or args.unfreeze_blocks is not None:
        unfreeze_set: Optional[set[int]] = None
        if args.unfreeze_blocks is not None:
            unfreeze_set = {int(s) for s in args.unfreeze_blocks.split(",") if s.strip()}
        freeze_report = _apply_finetune_freeze(
            get_base_model(trainer.model),
            freeze_trunk_entry=args.freeze_trunk_entry,
            unfreeze_blocks=unfreeze_set,
        )
        log.info("finetune_freeze_applied", **freeze_report)
        console.print(
            f"[yellow]§171 fine-tune freeze:[/yellow] "
            f"trainable_params={freeze_report['trainable_params']:,} / "
            f"{freeze_report['total_params']:,} "
            f"({100.0 * freeze_report['trainable_params'] / freeze_report['total_params']:.1f}%)"
        )

    # Training loop
    console.print(
        f"[bold]Training:[/bold] epochs={args.epochs} batch={batch_size} "
        f"label_smooth={label_smoothing} aux_weight={aux_weight}"
        + (f" RESUME from {args.resume} peak_lr={trainer.optimizer.param_groups[0]['lr']:.1e}" if args.resume else "")
    )
    trainer.step = -total_pretrain_steps
    start_step = trainer.step

    chain_weight = float(config.get("aux_chain_weight", 0.0))
    prev_loss: Optional[float] = None
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(
            loader,
            label_smoothing=label_smoothing,
            aux_weight=aux_weight,
            chain_weight=chain_weight,
            step_budget=step_budget,
            start_step=start_step,
        )
        log.info("epoch_complete", epoch=epoch, **{k: round(v, 4) for k, v in metrics.items()})
        console.print(
            f"Epoch {epoch}/{args.epochs}  "
            f"loss={metrics['loss']:.4f}  "
            f"policy={metrics['policy_loss']:.4f}  "
            f"value={metrics['value_loss']:.4f}  "
            f"aux={metrics['opp_reply_loss']:.4f}  "
            f"chain={metrics['chain_loss']:.4f}"
        )
        if step_budget is not None and (trainer.step - start_step) >= step_budget:
            break
        prev_loss = metrics["loss"]

    inf_out = Path(args.inference_out) if args.inference_out else None
    ckpt_path = trainer.save_checkpoint(inf_out=inf_out)
    console.print(f"[green]Checkpoint: {ckpt_path}[/green]")

    # validate() walks GameState.to_tensor() / KEPT_PLANE_INDICES — v6 only.
    # v8 validation is deferred to Gate 4 (SealBot WR + threat probe). The
    # v8 retrain harness skips the RandomBot smoke pass with an info log so
    # future v8 self-play (§168 Phase D) can re-introduce a v8-aware probe.
    if encoding == "v8":
        log.info(
            "skipping_validate_v8",
            reason="validate() is v6-only; v8 quality measured via SealBot WR / threat probe",
            ckpt=str(ckpt_path),
        )
    else:
        validate(ckpt_path, device)


if __name__ == "__main__":
    pretrain()
