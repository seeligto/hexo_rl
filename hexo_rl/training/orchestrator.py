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
_BASE_CONFIGS = [
    "configs/model.yaml",
    "configs/training.yaml",
    "configs/selfplay.yaml",
    "configs/game_replay.yaml",
    "configs/monitoring.yaml",
    "configs/monitors.yaml",
]


def _build_load_paths(
    args: argparse.Namespace,
) -> tuple[list[str], Optional[str], Optional[str]]:
    """Single source of truth for the base+config+variant path list (CONFRES F2).

    Returns the ordered ``(base_paths, config_override, variant_path)`` triple so BOTH the
    ``load_config`` merge AND ``capture_config_layers`` build the merge chain from ONE place — no
    duplicated base-configs list, no second list-building site (design §2/§8, F2). ``base_paths``
    already excludes any base file that ``--config`` resolves to (dedup preserved so the
    ``config_key_overlap`` warning does not fire twice). The variant existence check +
    abort-on-warning validation live in :func:`load_train_config`, which owns the launch failure
    modes; this helper is a pure path assembler.
    """
    base_paths: list[str] = list(_BASE_CONFIGS)
    config_override: Optional[str] = None
    if args.config:
        override = str(Path(args.config).resolve())
        base_paths = [p for p in base_paths if str(Path(p).resolve()) != override]
        config_override = args.config
    variant_path: Optional[str] = None
    if args.variant:
        variant_path = str(Path(f"configs/variants/{args.variant}.yaml"))
    return base_paths, config_override, variant_path


def load_train_config(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Load base + override + variant configs; abort-on-warning validation.

    Returns ``(merged_config, layers)`` — the merged config the rest of the launch reads AND the
    kind-tagged raw layer chain (``capture_config_layers`` output) so the CONFRES resolver has
    per-layer provenance without a SECOND path-list-building site (F2). Both are built from the
    single :func:`_build_load_paths` source.

    Mirrors ``scripts/train.py::main`` lines 181-226 verbatim (config merge + variant validation);
    the layers return is CONFRES-additive.
    """
    from hexo_rl.utils.config import load_config
    from hexo_rl.config.resolve.run_config import assert_layers_reconstruct, capture_config_layers

    base_paths, config_override, variant_path = _build_load_paths(args)

    if args.variant:
        variant_p = Path(variant_path)
        if not variant_p.exists():
            available = sorted(p.stem for p in Path("configs/variants").glob("*.yaml"))
            raise FileNotFoundError(
                f"Variant '{args.variant}' not found. Available: {available}"
            )

        # §176 P68 — abort-on-warning variant validation. Catches silent
        # namespace shadows before merge (e.g. variant declaring
        # ``training: {max_train_burst: 8}`` when base has flat
        # ``max_train_burst``).
        import yaml as _yaml
        from hexo_rl.utils.variant_validator import (
            _load_standard_base_cfgs,
            validate_variant_against_bases,
        )
        with open(variant_p) as _vf:
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

    # Flat ordered load-path list for the merge — base(deduped) → --config → variant.
    load_paths: list[str] = list(base_paths)
    if config_override is not None:
        load_paths.append(config_override)
    if variant_path is not None:
        load_paths.append(variant_path)

    config = load_config(*load_paths)
    # CONFRES F2: kind-tagged raw layer chain from the SAME single path source.
    layers = capture_config_layers(base_paths, config_override, variant_path)
    # F2 invariant at LOAD time, where `config` is PRISTINE (== merged_layers(layers) by
    # construction: same files, same order, same deep-merge). Asserted HERE (not at emission)
    # so it cannot false-fail against a `config` that legitimate downstream transforms mutate
    # (mixing/buffer <auto>-path expansion, encoding back-prop). A real dedup/order bug in
    # _build_load_paths raises here, loudly, at launch.
    assert_layers_reconstruct(layers, config)
    return config, layers


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
        # D-FORENSIC F1: this event fires BEFORE checkpoint load — it is the
        # variant's DECLARED intent, not what self-play will run. The
        # post-load truth is `checkpoint_encoding_resolved`. The d1m
        # forensic mis-read this event as ground truth for a week.
        source="variant_declared_pre_checkpoint",
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


# ── Resume config precedence (D-FULLSPEC E0) ─────────────────────────────────
# Keys that MUST come from the CHECKPOINT on resume, never from the launch
# variant. Everything else in ``combined_config`` is the operator's explicit
# launch intent and WINS over the checkpoint-baked config.
#
# Pre-E0 the precedence was inverted: ``init_trainer`` built a tiny whitelist
# (torch_compile[_mode], the four aux *_weight knobs, eta_min, scheduler_t_max,
# total_steps, allow_fresh_scheduler, diagnostics) and the checkpoint config
# WON for every other key (trainer_ckpt_load.load_checkpoint:333 does
# ``config = dict(ckpt["config"])``). A resumed --variant therefore silently
# dropped every runtime knob it did not whitelist — completed_q_values,
# aux_chain_weight, ply_index_weight, per_class_target_temperature,
# policy_prune_frac, entropy_reg_weight, grad_clip, value_distill_* — and
# re-baked the reverted defaults of bot_batch_share / draw_value / ply_cap_value
# into the next checkpoint, corrupting provenance and breaking later
# resume-without-variant. This set inverts the precedence: the variant wins,
# minus the small set below that the checkpoint owns by construction.
RESUME_CHECKPOINT_OWNED_KEYS: frozenset = frozenset({
    # Encoding pins — reconciled by _resolve_checkpoint_encoding /
    # _propagate_encoding_into_config off the checkpoint's trained spec (read
    # from fallback_config, NOT config_overrides). A resumed variant must not
    # silently re-route plane geometry.
    "encoding", "cluster_window_size", "cluster_threshold",
    "legal_move_radius", "board_size",
    # Model architecture — reconciled by _resolve_model_hparams, which RAISES
    # on a config-vs-checkpoint-inferred disagreement. The trained shape is
    # canonical; the variant cannot reshape a loaded model.
    "in_channels", "input_channels", "res_blocks", "filters",
    "se_reduction_ratio", "model",
    # Optimizer / scheduler / step STATE — restored from the checkpoint's
    # optimizer_state / scheduler_state / step blobs. The LR-anneal horizon
    # keys (total_steps / scheduler_t_max) are gated back into the overrides
    # ONLY under --override-scheduler-horizon, preserving the resume LR-anneal
    # semantics that tests/test_scheduler_resume.py pins.
    "total_steps", "scheduler_t_max", "eta_min", "min_lr",
    "lr", "weight_decay", "lr_schedule",
})


def compute_declared_keys(layers: "list[dict] | None") -> frozenset:
    """Top-level keys the OPERATOR declared in a ``--config`` or ``variant`` layer (CONFRES F1/B3).

    A key present in a layer whose ``kind`` is ``"config"`` or ``"variant"`` is an operator
    DECLARATION — including an explicit ``key: null`` (B3: an explicit null is a declaration that
    overrides, to null). A key present only in a ``base`` layer is INHERITED, not a declaration, so
    it DEFERS to a checkpoint-baked value on resume (F1(A)).

    Classification reuses ``run_config._is_declaration_layer`` (the same kind-aware rule the
    provenance resolver uses) so the two surfaces cannot disagree about what "declared" means.
    ``None``/empty layers → empty set (fresh clones + hand-built call sites without layers).
    """
    if not layers:
        return frozenset()
    from hexo_rl.config.resolve.run_config import _is_declaration_layer

    declared: set = set()
    for layer in layers:
        if _is_declaration_layer(layer):
            raw = layer.get("raw") or {}
            if isinstance(raw, dict):
                declared.update(raw.keys())
    return frozenset(declared)


def build_resume_config_overrides(
    combined_config: dict,
    *,
    override_scheduler_horizon: bool = False,
    allow_fresh_scheduler: bool = False,
    declared_keys: "frozenset | set | None" = None,
) -> dict:
    """Build the resume ``config_overrides`` so the launch variant wins.

    D-FULLSPEC E0 fix — inverts the pre-E0 resume precedence. Seeds the
    overrides from the FULL ``combined_config`` minus
    ``RESUME_CHECKPOINT_OWNED_KEYS`` (encoding/arch pins + optimizer/scheduler/
    step state), so every runtime knob and provenance field carried by the
    operator's --variant survives a checkpoint resume instead of reverting to
    the checkpoint-baked value.

    The ``--override-scheduler-horizon`` gate is preserved verbatim: the
    scheduler-horizon keys re-enter the overrides (re-horizoning the LR
    scheduler at load_checkpoint) ONLY when the flag is set. Without the flag
    the checkpoint scheduler_state's T_max is restored untouched
    (tests/test_scheduler_resume.py negative pin).

    B3 null semantics (CONFRES 6b) — the null-skip is now PROVENANCE-AWARE:
    a ``None`` that the operator EXPLICITLY declared (``key in declared_keys``)
    TRAVELS (it is a declaration that overrides the baked value to null); a
    ``None`` merely inherited from a base layer is a default and is SKIPPED so a
    stray null cannot nuke a real checkpoint value. Absent ``declared_keys``
    (hand-built call sites) → every null skipped, preserving the pre-6b
    behaviour byte-for-byte.
    """
    declared: frozenset = frozenset(declared_keys or ())
    overrides: dict = {
        key: val
        for key, val in combined_config.items()
        if key not in RESUME_CHECKPOINT_OWNED_KEYS
        and (val is not None or key in declared)
    }
    # torch_compile always travels with a concrete value (pre-E0 default path:
    # a weights-only resume needs an explicit flag, not an absent key).
    overrides["torch_compile"] = combined_config.get("torch_compile", False)
    # §116: torch_compile_mode must travel alongside torch_compile on resume.
    if combined_config.get("torch_compile_mode") is not None:
        overrides["torch_compile_mode"] = combined_config["torch_compile_mode"]
    # Scheduler-horizon gate: only --override-scheduler-horizon re-horizons the
    # LR anneal. total_steps + scheduler_t_max enter the overrides together so
    # the load_checkpoint re-horizon branch (scheduler_state['T_max']) fires.
    if override_scheduler_horizon:
        if combined_config.get("total_steps") is not None:
            overrides["total_steps"] = int(combined_config["total_steps"])
        if combined_config.get("scheduler_t_max") is not None:
            overrides["scheduler_t_max"] = int(combined_config["scheduler_t_max"])
    if allow_fresh_scheduler:
        overrides["allow_fresh_scheduler"] = True
    return overrides


# ── CONFRES 6a-ii: resolve provenance + emit the resolved_config event ────────
def _checkpoint_stamps_from_trainer(trainer: Trainer) -> dict:
    """Encoding/arch stamps cleanly available on the resumed trainer's baked config.

    Only the encoding stamp is read post-init (``trainer.config['encoding']`` — a dict
    ``{"version": ...}`` post-propagation, or a bare string). Arch stamps (res_blocks/filters) are
    reconciled inside ``load_checkpoint`` and not surfaced as a distinct 'stamp' the encoding
    resolver consumes, so they are omitted here (6b's deeper state-blob threading). Fresh runs have
    no stamp → empty dict.
    """
    stamps: dict = {}
    enc = trainer.config.get("encoding")
    if isinstance(enc, dict):
        enc = enc.get("version")
    if enc is not None:
        stamps["encoding"] = enc
    return stamps


def _checkpoint_state_from_trainer(trainer: Trainer) -> dict:
    """Minimal optimizer-state blob for the LR-effective read (I3/B2).

    Shapes the live optimizer's first param-group lr into the
    ``optimizer_state.param_groups[0].lr`` structure ``_effective_lr_from_state`` reads. Guards an
    empty ``param_groups`` (returns ``{}`` → resolver falls to declared/default). This is the
    MINIMAL, cleanly-available slice; deeper scheduler_state/T_max threading is 6b.
    """
    opt = getattr(trainer, "optimizer", None)
    pgs = getattr(opt, "param_groups", None) if opt is not None else None
    if not pgs:
        return {}
    lr = pgs[0].get("lr")
    if lr is None:
        return {}
    return {"optimizer_state": {"param_groups": [{"lr": lr}]}}


def build_and_emit_resolved_config(
    layers: list[dict],
    merged_config: dict,
    registry_spec: Any,
    trainer: Trainer,
    args: argparse.Namespace,
    log: "structlog.stdlib.BoundLogger",
) -> dict:
    """CONFRES 6a-ii — resolve run provenance + emit the ONE ``resolved_config`` event (design §5).

    STRICTLY ADDITIVE. This does NOT change seeding, device, checkpoint load, or any existing
    event — it EMITS the F1-forensic provenance artifact alongside the already-applied launch
    logic. The emitted value IS what the run used; consumers are NOT migrated here (that is 6c).

    ``merged_config`` is the PRISTINE ``load_config`` merge output (the F2 reconstruct target:
    ``merged_layers(layers) == merged_config``), NOT the flattened/CLI-mutated ``combined_config``
    — the resolver reads its knobs from the raw ``layers``, and asserts the layers reconstruct this
    merged config as a launch-time sanity check (raises loudly on a real bug).

    Phase-A (I7/F3): ``resolve_preload_config`` renders seed/device/tf32, the launch-only knobs
    consumed BEFORE the checkpoint loaded — flagged ``consumed_pre_resolution: true``.
    Phase-B: ``resolve_run_config`` aggregates the post-load knobs. Checkpoint-side inputs are
    gathered MINIMALLY from what is cleanly available post-init (encoding stamp + baked config +
    effective lr); anything not cleanly available is passed None (deeper threading = 6b).

    Returns the emitted payload (for the test harness / callers to inspect).
    """
    from hexo_rl.config.resolve.run_config import (
        ConfigConflictError,
        resolve_preload_config,
        resolve_run_config,
    )

    cli = vars(args)

    # (The F2 invariant is asserted at LOAD time in load_train_config, against the PRISTINE config
    # — NOT here, where `merged_config` may reflect legitimate downstream transforms like
    # mixing/buffer <auto>-path expansion or resume encoding back-prop.)
    # Phase-A: launch-only knobs (no checkpoint ingestion — value emitted IS what the run used).
    preload = resolve_preload_config(layers, cli=cli)

    # Phase-B: post-load knobs. On resume the baked config is the checkpoint's config blob; on a
    # fresh run there is no baked config (init_trainer built the Trainer from combined_config).
    checkpoint_baked = trainer.config if bool(args.checkpoint) else None
    checkpoint_stamps = _checkpoint_stamps_from_trainer(trainer) if bool(args.checkpoint) else {}
    checkpoint_state = _checkpoint_state_from_trainer(trainer)

    # BYTE-PURE guard (6a-ii): resolve+emit ONLY. The encoding resolver's I2 conflict-raise is a
    # DESIGN-intended precedence change (6b), NOT this batch — on the live launch path a
    # variant-vs-stamp encoding mismatch is back-propagated + WARN-logged (ckpt wins,
    # init_trainer), it does NOT abort. So a resolver conflict here must NOT introduce a new launch
    # abort: log + skip the provenance emission instead. The F2 assert above already fired loudly.
    try:
        resolved = resolve_run_config(
            registry_spec,
            layers,
            merged_config,
            checkpoint_stamps=checkpoint_stamps,
            checkpoint_state=checkpoint_state,
            cli=cli,
            checkpoint_baked=checkpoint_baked,
        )
    except ConfigConflictError as exc:
        # Encoding conflict (variant-declared vs checkpoint stamp): 6b-deferred precedence change.
        # Do not abort a launch the live path allows — this is emission-only provenance. 6b owns the
        # precedence enforcement.
        log.warning(
            "resolved_config_skipped",
            reason=str(exc),
            detail="resolver raised during 6a-ii emission; provenance event skipped (byte-pure — "
                   "the live launch path is unaffected). The F1/B3 precedence enforcement is 6b.",
        )
        return {}

    payload = resolved.to_event_payload()
    # Merge the Phase-A preload knobs into the same payload so seed/device/tf32 appear with
    # ``consumed_pre_resolution: true``. Phase-A knobs are launch-only and never overlap the
    # Phase-B knob set.
    preload_payload = preload.to_event_payload()
    payload["knobs"].update(preload_payload["knobs"])

    emit_event(payload)
    log.info(
        "resolved_config_emitted",
        n_knobs=len(payload["knobs"]),
        is_resume=bool(args.checkpoint),
    )
    return payload


# ── Section 7: trainer init (resume vs fresh) ────────────────────────────────
def init_trainer(
    args: argparse.Namespace,
    combined_config: dict,
    device: torch.device,
    board_size: int,
    res_blocks: int,
    filters: int,
    log: "structlog.stdlib.BoundLogger",
    declared_keys: "frozenset | set | None" = None,
) -> tuple[Trainer, int]:
    """Build / resume a Trainer; returns (trainer, possibly-updated board_size).

    On resume, board_size may be re-resolved from the registry after
    propagating ckpt-baked encoding back into combined_config.

    ``declared_keys`` (CONFRES F1(A)) is the operator-declaration set threaded into the resume
    override apply: a base-inherited override that the checkpoint also baked DEFERS to the baked
    value; a declared key still wins. ``None`` preserves the pre-6b behaviour.

    Mirrors ``scripts/train.py::main`` lines 291-379 verbatim (F1 threading is 6b-additive).
    """
    from hexo_rl.encoding import resolve_from_config as _registry_resolve

    if args.checkpoint:
        # CONFRES P3: validate the resolved bootstrap/resume checkpoint exists at LAUNCH
        # (before Trainer.load_checkpoint's torch.load) so a stale Makefile BOOTSTRAP default
        # or a typo'd --checkpoint fails loudly + informatively, not deep in loading.
        from hexo_rl.config.resolve.bootstrap import resolve_bootstrap
        resolve_bootstrap(args.checkpoint)
        # D-FULLSPEC E0: invert the resume precedence — seed config_overrides
        # from the full combined_config (variant = operator intent, wins over
        # the checkpoint-baked config) minus the checkpoint-owned EXCLUDE-set.
        # The diagnostics section, the four aux *_weight knobs, and every other
        # runtime/provenance key now travel automatically (they are not in
        # RESUME_CHECKPOINT_OWNED_KEYS); the --override-scheduler-horizon gate
        # is preserved inside build_resume_config_overrides.
        config_overrides = build_resume_config_overrides(
            combined_config,
            override_scheduler_horizon=bool(args.override_scheduler_horizon),
            allow_fresh_scheduler=bool(args.allow_fresh_scheduler),
            declared_keys=declared_keys,
        )

        trainer = Trainer.load_checkpoint(
            args.checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            fallback_config=combined_config,
            config_overrides=config_overrides,
            declared_keys=declared_keys,
        )
        # Propagate the ckpt-resolved encoding back into the local config
        # dicts so downstream selfplay surfaces (pool, worker, lifecycle,
        # eval) read the encoding the model was actually trained under
        # rather than the variant YAML's default. Without this, a v6w25
        # bootstrap loaded against a v6-default variant would route
        # correctly inside the trainer but feed v6 plane geometry into
        # the self-play workers (§171 P3 blocker).
        #
        # D-WS3V3 FIX1c (2026-07-02): this back-propagation is exactly the
        # mechanism that silently routed the whole v3 build through stale
        # single-window `v6_live2` (the anchor checkpoint's baked
        # metadata['encoding_name']) despite every variant declaring
        # multi-window `v6_live2_ls` — see `scripts/make_ws3v3_warmstart.py`
        # FIX1 docstring. Emit a LOUD warning whenever the pre-propagation
        # variant encoding and the ckpt-resolved encoding disagree, so a
        # future case like this one doesn't burn a GPU-day silently.
        # LOG-ONLY: does NOT change precedence — ckpt-resolved still wins
        # (§171 P3 depends on it).
        _pre_prop_enc = combined_config.get("encoding")
        _pre_prop_enc_name = (
            _pre_prop_enc.get("version") if isinstance(_pre_prop_enc, dict) else _pre_prop_enc
        )
        for _enc_key in (
            "cluster_window_size", "cluster_threshold",
            "legal_move_radius", "encoding",
        ):
            if _enc_key in trainer.config:
                combined_config[_enc_key] = trainer.config[_enc_key]
        # CONFRES F1(A) split-brain fix: back-propagate the FULL set of F1-DEFERRED keys (the exact
        # keys F1's defer-to-baked preserved — from trainer.f1_deferred_keys, NOT a hardcoded list)
        # from trainer.config into combined_config, so the training loop / self-play / inference
        # EXECUTE the values F1 preserved. Without this, run_training_loop(config=combined_config)
        # ran the launch-merge (base-default) value while the resume_base_default_deferred_to_baked
        # WARN, the re-baked checkpoint, AND the resolved_config emission ALL reported the baked value
        # the run did NOT execute (e.g. ply_cap_value; and amp_dtype — a numerical trainer⊥inference
        # autocast-dtype split). The 4 encoding keys above are a subset already covered; this closes
        # every OTHER loop-read knob. E0 is untouched (declared keys are NOT in f1_deferred_keys, so a
        # declaration still wins in combined_config); F2 (load-time assert) + owned-key semantics are
        # unaffected (F1 owns which keys deferred).
        # Plain attribute access (not getattr-with-default): Trainer.__init__ always sets
        # f1_deferred_keys, so a missing attr is a real bug we WANT to surface, not mask. Every
        # deferred key is baked → present in trainer.config, so no membership guard is needed.
        for _f1_key in trainer.f1_deferred_keys:
            combined_config[_f1_key] = trainer.config[_f1_key]
        _post_prop_enc = combined_config.get("encoding")
        _post_prop_enc_name = (
            _post_prop_enc.get("version") if isinstance(_post_prop_enc, dict) else _post_prop_enc
        )
        if (
            _pre_prop_enc_name is not None
            and _post_prop_enc_name is not None
            and _pre_prop_enc_name != _post_prop_enc_name
        ):
            log.warning(
                "checkpoint_encoding_overrides_variant",
                variant_encoding=_pre_prop_enc_name,
                checkpoint_resolved_encoding=_post_prop_enc_name,
                checkpoint=args.checkpoint,
                msg=(
                    "checkpoint-resolved encoding overrides the variant's "
                    "declared encoding; ckpt-resolved wins (precedence "
                    "unchanged) — self-play will route through "
                    f"{_post_prop_enc_name!r}, NOT the variant's "
                    f"{_pre_prop_enc_name!r}. If this is unintended, re-stamp "
                    "the checkpoint's metadata['encoding_name'] before resuming."
                ),
            )
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
        from hexo_rl.training.model_defaults import MODEL_HPARAM_DEFAULTS as _MHPD
        model = HexTacToeNet(
            board_size=board_size,
            res_blocks=res_blocks,
            filters=filters,
            in_channels=in_channels_arg,
            input_channels=input_channels_cfg,
            value_head_type=combined_config.get("value_head_type", _MHPD["value_head_type"]),
            n_value_bins=combined_config.get("n_value_bins", _MHPD["n_value_bins"]),
        )
        trainer = Trainer(model, combined_config, checkpoint_dir=args.checkpoint_dir, device=device)
        log.info(
            "new_run",
            model_params=sum(p.numel() for p in model.parameters()),
            in_channels=in_channels_arg,
            input_channels=list(input_channels_cfg) if input_channels_cfg is not None else None,
            value_head_type=combined_config.get("value_head_type", _MHPD["value_head_type"]),
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

    # CONFRES 6c: size the buffer from the RESOLVED encoding spec (the ONE encoding→spec
    # authority, ``resolve.encoding.window_set``) rather than a raw ``normalize(config.get)``.
    # Byte-pure on no-conflict configs; correct on a metadata-wins resume where the buffer used to
    # size from the pre-checkpoint spec (B5b, design §4 window_set).
    from hexo_rl.config.resolve.encoding import window_set as _window_set
    buffer = ReplayBuffer(capacity=capacity, encoding=_window_set(config.get("encoding")).name)

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
        # §D-1M C1 — HARD-FAIL by default (was warn-only). A stale on-disk
        # replay_buffer.bin can match wire_signature across encodings (e.g.
        # v6_live2 vs v6_live2_ls produce an identical signature), so the
        # shape guard does NOT catch it — 250k prior-run positions would load
        # silently and poison the fresh buffer from step 0, wasting the run.
        # The legitimate resume path has trainer.step > 0, so this only fires
        # on fresh-from-bootstrap launches that inherited stale self-play data.
        _msg = (
            "Bootstrap-like checkpoint (step <= 0) was loaded but the replay "
            "buffer is non-empty before corpus load — the on-disk "
            "replay_buffer.bin from a prior run was auto-restored (its "
            "wire_signature can match across encodings, so the shape guard "
            "does NOT catch it). Delete checkpoints/replay_buffer.bin (and "
            "*.recent) before a fresh launch, or set "
            "training.mixing.buffer_persist=false on the variant. To override "
            "deliberately, set training.mixing.allow_stale_buffer=true. "
            "See §149 task 4c/4d / §D-1M C1."
        )
        if bool(mixing_cfg.get("allow_stale_buffer", False)):
            log.warning(
                "buffer_contamination_suspected_allowed",
                msg=_msg, buffer_size=buffer.size,
                ckpt=_ckpt_path_str, ckpt_step=trainer.step,
            )
        else:
            log.error(
                "buffer_contamination_abort",
                msg=_msg, buffer_size=buffer.size,
                ckpt=_ckpt_path_str, ckpt_step=trainer.step,
            )
            raise RuntimeError(_msg)


# ── Section 12: recent buffer ────────────────────────────────────────────────
def init_recent_buffer(
    train_cfg: dict,
    config: dict,
    capacity: int,
    registry_spec: Any,
    _bp: Optional[Path],
    _buffer_restored: bool,
    log: "structlog.stdlib.BoundLogger",
    combined_config: dict | None = None,
) -> tuple[Optional[RecentBuffer], float]:
    """Build the optional RecentBuffer + restore from the .recent sidecar.

    Returns
    -------
    recent_buffer, _recency_weight

    Mirrors ``scripts/train.py::main`` lines 478-499 verbatim.

    CONFRES 6c: the RecentBuffer state_shape sizes from the RESOLVED encoding spec. The
    ``registry_spec`` argument is the PRE-checkpoint spec (from ``flatten_config_and_resolve_
    encoding``) — STALE after a metadata-wins resume where the checkpoint's baked encoding
    back-propagates a different plane count. When ``combined_config`` (the post-load config) is
    supplied, re-resolve the spec from it via the ONE encoding→spec authority
    (``resolve.encoding.window_set``) — a latent-bug fix, byte-pure on no-conflict configs where
    pre- and post-checkpoint encodings agree (design §4 window_set, handoff 6c).
    """
    _recency_weight = float(train_cfg.get("recency_weight", 0.0))
    recent_buffer: RecentBuffer | None = None
    if _recency_weight > 0.0:
        if combined_config is not None:
            from hexo_rl.config.resolve.encoding import window_set as _window_set
            registry_spec = _window_set(combined_config.get("encoding"))
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
    from hexo_rl.config.resolve.encoding import window_set as _window_set

    # §172 A4.3 / CONFRES 6c: re-resolve the RESOLVED encoding spec post-checkpoint-load (a resume
    # may have rewritten encoding via metadata) and size the batch buffers per trunk_size /
    # policy_logit_count / n_planes. Pre-CONFRES a resolve FAILURE was swallowed into a v6-shaped
    # literal fallback (n_planes=8, trunk from board_size, _N_ACTIONS) + a warn — which would build
    # v6-geometry buffers against a NON-v6 net and corrupt training silently. Under CONFRES this is
    # a HARD-ERROR: the encoding spec MUST resolve (design §4 window_set, handoff 6c).
    _batch_size_cfg = int(train_cfg.get("batch_size", config.get("batch_size", 256)))
    _bufs_spec = _window_set(combined_config.get("encoding"))
    _trunk_size = int(_bufs_spec.trunk_size)
    _n_actions_spec = int(_bufs_spec.policy_logit_count)
    _n_planes_spec = int(_bufs_spec.n_planes)
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
