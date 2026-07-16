"""S7 F1 — run4_gnn.yaml must declare mixing.pretrained_buffer_path: null, not
inherit configs/training.yaml's base "<auto>" default.

Closes S7 Part-1 Finding F1 (reports/probes/gnn_integration/S7_smoke_gate.md):
without an explicit declaration, `configs/variants/run4_gnn.yaml` silently
inherited the base "<auto>" literal, which `expand_auto_paths` resolves to
`data/gnn_corpus_v1.hexg` (registered in `_CORPUS_PATHS`) and stamps
`_pretrained_buffer_path_auto_resolved: true` — `load_pretrained_buffer` then
hard-raises because `gnn_axis_v1` has deliberately NO registered corpus sha
pin yet. That crash fired BEFORE self-play even started (before the
watchdog armed, before a single training step), independent of any smoke
harness isolation.

Also covers S7 round-2 F5a (namespaced `eval_pipeline.gating.best_model_path`,
same §RUN3-STEP0 law as `mixing.buffer_persist_path`) and F6 (the
`run4_gnn_smoke.yaml` laptop gate profile: resolves clean, forbidden weights
all 0, namespaced paths — DISTINCT from run4_gnn.yaml's own, never the bare
shared defaults).

Drives the REAL config-merge entrypoint (`hexo_rl.training.orchestrator`)
against the REAL `configs/variants/run4_gnn*.yaml` files on disk (not a
reimplemented/mocked merge) — same pattern as
`tests/test_run3_corpus_launch_path.py`. Plain (non-integration) unit test:
config resolution + `load_pretrained_buffer`'s early-return branch need no
GPU/model build, so this stays in the default `not slow and not integration`
suite (the launch-path itself IS integration-marked and covered by S7's own
manual run + `tests/test_orchestrator_gnn_build.py` et al. — this test only
pins the config-resolution contract that made S7 Part-1 fail).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

# Both the production launch config and the S7 gate smoke profile must carry
# the SAME namespacing law — parametrize the resolved-config-level assertions
# over both, and additionally cross-check the two variants' namespaced paths
# never collide (a smoke gate run must never share a buffer/anchor file with
# a real production run4_gnn launch).
_VARIANTS = ["run4_gnn", "run4_gnn_smoke"]


def _args(variant: str = "run4_gnn") -> argparse.Namespace:
    """Minimal Namespace covering the attrs `load_train_config` /
    `flatten_config_and_resolve_encoding` read (mirrors
    tests/test_run3_corpus_launch_path.py::_args)."""
    return argparse.Namespace(
        config=None,
        variant=variant,
        iterations=None,
        no_compile=False,
    )


def test_run4_gnn_yaml_declares_null_not_inherited_auto():
    """Checked against the RAW variant file (pre-merge) so this fails if the
    explicit declaration is ever reverted — a missing key would silently
    fall back to the base '<auto>' default again."""
    variant_cfg = yaml.safe_load(Path("configs/variants/run4_gnn.yaml").read_text())
    assert "mixing" in variant_cfg, "run4_gnn.yaml must declare a mixing: block"
    assert "pretrained_buffer_path" in variant_cfg["mixing"], (
        "run4_gnn.yaml must explicitly declare mixing.pretrained_buffer_path "
        "(S7 F1) — omitting the key silently inherits configs/training.yaml's "
        "base '<auto>' default, which hard-raises for gnn_axis_v1 (no "
        "registered corpus sha pin yet)."
    )
    assert variant_cfg["mixing"]["pretrained_buffer_path"] is None, (
        "run4_gnn.yaml's mixing.pretrained_buffer_path must be an explicit "
        "null (real corpus-mix OFF, run4_gnn_design.md Decision 1) — not the "
        "'<auto>' literal, which routes through expand_auto_paths and the "
        "sha-pin gate."
    )
    # bot-mix OFF (design Decision 1 + memory bot-mix-retired-s178-useless) —
    # already declared; re-asserted here so a regression on EITHER of the
    # two "corpus-mix OFF" declarations fails this one test.
    assert variant_cfg.get("bot_batch_share") == 0


def test_run4_gnn_config_resolution_has_no_corpus_prefill():
    """The resolved launch config (base + run4_gnn merge, <auto> expansion
    included) must carry a falsy mixing.pretrained_buffer_path — proving the
    sha-pin gate (`load_pretrained_buffer`'s `<auto>`-resolved branch,
    `hexo_rl/training/batch_assembly.py`) is never reached for a real
    run4_gnn launch."""
    import structlog

    from hexo_rl.training import orchestrator as _orch

    log = structlog.get_logger()
    config, _layers = _orch.load_train_config(_args())
    combined_config, train_cfg, _mcts_cfg, registry_spec, *_ = (
        _orch.flatten_config_and_resolve_encoding(config, _args(), log)
    )

    assert registry_spec.name == "gnn_axis_v1"
    assert registry_spec.representation == "graph"

    mixing_cfg = train_cfg.get("mixing", config.get("mixing", {}))
    assert mixing_cfg.get("pretrained_buffer_path") is None, (
        "expand_auto_paths must NOT have touched this key (only '<auto>' "
        "literals expand) — a non-null value here means run4_gnn.yaml's "
        "declaration regressed back to inheriting the base default."
    )
    assert not mixing_cfg.get("_pretrained_buffer_path_auto_resolved"), (
        "no <auto>-resolve provenance flag should be stamped — that flag is "
        "exactly what makes load_pretrained_buffer require a sha pin."
    )

    # The actual consumer: load_pretrained_buffer must take its cheap
    # `if not pretrained_path: return None` no-op branch — never reach the
    # sha-pin ValueError that crashed S7 Part-1 before self-play started.
    from hexo_rl.training.batch_assembly import load_pretrained_buffer

    result = load_pretrained_buffer(
        mixing_cfg, combined_config, emit_fn=lambda evt: None,
        buffer_size=0, buffer_capacity=0,
    )
    assert result is None


def test_run4_gnn_config_resolution_no_bot_corpus_prefill():
    """bot_batch_share: 0 (already declared) must resolve as 0 post-merge —
    the OTHER half of run4_gnn_design.md Decision 1's corpus-mix OFF ruling."""
    import structlog

    from hexo_rl.training import orchestrator as _orch

    log = structlog.get_logger()
    config, _layers = _orch.load_train_config(_args())
    combined_config, train_cfg, _mcts_cfg, registry_spec, *_ = (
        _orch.flatten_config_and_resolve_encoding(config, _args(), log)
    )
    assert float(combined_config.get("bot_batch_share", -1)) == 0.0


def test_run4_gnn_resolves_every_graph_forbidden_weight_to_zero():
    """S7 follow-up: every key `_train_on_graph_batch`'s §6.3 guard checks
    (GRAPH_FORBIDDEN_NONZERO_WEIGHTS — the SAME module constant the guard
    iterates, so this test can never drift from the guard) must resolve to 0
    in the merged run4_gnn config. Base training.yaml defaults are nonzero
    for most of these; inherited, they would crash the run at the FIRST train
    step (loud-raise) — a known first-step landmine the S7 gate re-run must
    not re-discover. GnnNet ships policy + dist65 only (design §1.3 DROP);
    entropy_reg_weight is guard-listed too (read only by the dense
    _train_on_batch — a nonzero value on graph is the silent-no-op class
    §6.3 forbids)."""
    import structlog

    from hexo_rl.training import orchestrator as _orch
    from hexo_rl.training.trainer import GRAPH_FORBIDDEN_NONZERO_WEIGHTS

    log = structlog.get_logger()
    config, _layers = _orch.load_train_config(_args())
    combined_config, _train_cfg, _mcts_cfg, registry_spec, *_ = (
        _orch.flatten_config_and_resolve_encoding(config, _args(), log)
    )
    assert registry_spec.representation == "graph"

    # The guard must still cover at least the six §6.3 head-aux keys +
    # entropy_reg_weight — a shrink of the constant would silently weaken
    # BOTH the guard and this test (same source of truth cuts both ways).
    assert set(GRAPH_FORBIDDEN_NONZERO_WEIGHTS) >= {
        "aux_opp_reply_weight", "uncertainty_weight", "ownership_weight",
        "threat_weight", "aux_chain_weight", "ply_index_weight",
        "entropy_reg_weight",
    }

    nonzero = {
        key: combined_config.get(key)
        for key in GRAPH_FORBIDDEN_NONZERO_WEIGHTS
        if float(combined_config.get(key, 0.0)) != 0.0
    }
    assert not nonzero, (
        f"run4_gnn resolved config carries nonzero graph-forbidden weights "
        f"{nonzero} — _train_on_graph_batch's §6.3 guard will loud-raise at "
        "the first train step; declare each 0 in configs/variants/run4_gnn.yaml."
    )


# ── S7 round-2 F5a — namespaced eval_pipeline.gating.best_model_path ─────────
#
# §RUN3-STEP0 law: a shared/un-namespaced best_model_path (base configs/
# eval.yaml default checkpoints/best_model.pt) lets one lineage's fresh-init
# anchor leak into another's `resolve_anchor` call — the exact trap the S7
# blocker-fix proof-run planted and the very next run4_gnn launch walked into
# (F5a forensics, S7_smoke_gate.md "Re-run after blocker fixes"). Both
# buffer_persist_path (F1-era) and best_model_path (F5a) must be namespaced —
# never the bare shared defaults.

_SHARED_DEFAULT_BEST_MODEL_PATH = "checkpoints/best_model.pt"  # configs/eval.yaml:146


@pytest.mark.parametrize("variant", _VARIANTS)
def test_variant_yaml_declares_namespaced_best_model_path(variant: str) -> None:
    """Checked against the RAW variant file (pre-merge) — fails if the
    declaration is ever reverted, dropped, or narrowed back to the shared
    default (a missing key silently inherits configs/eval.yaml's
    checkpoints/best_model.pt, F5a's exact bug)."""
    variant_cfg = yaml.safe_load(Path(f"configs/variants/{variant}.yaml").read_text())
    assert "eval_pipeline" in variant_cfg, f"{variant}.yaml must declare an eval_pipeline: block"
    gating = variant_cfg["eval_pipeline"].get("gating", {})
    assert "best_model_path" in gating, (
        f"{variant}.yaml must explicitly declare eval_pipeline.gating.best_model_path "
        "(S7 F5a) — omitting the key silently inherits configs/eval.yaml's shared "
        f"default ({_SHARED_DEFAULT_BEST_MODEL_PATH!r}), which cross-lineage-collides "
        "with every other un-namespaced variant/run (§RUN3-STEP0 law)."
    )
    path = gating["best_model_path"]
    assert path != _SHARED_DEFAULT_BEST_MODEL_PATH, (
        f"{variant}.yaml's best_model_path must be namespaced, not the bare shared "
        f"default {_SHARED_DEFAULT_BEST_MODEL_PATH!r}."
    )
    assert variant in path, (
        f"{variant}.yaml's best_model_path {path!r} should be namespaced with the "
        f"variant's own name (run3_dist65.yaml:172 template) so it is trivially "
        "distinguishable from every other lineage's anchor."
    )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_variant_yaml_declares_namespaced_buffer_persist_path(variant: str) -> None:
    """Same law, the buffer side (pre-existing since S7 F1 for run4_gnn;
    re-asserted here so BOTH variants are covered by one shared regression
    guard and a future edit can't silently narrow either back to a bare
    default)."""
    variant_cfg = yaml.safe_load(Path(f"configs/variants/{variant}.yaml").read_text())
    assert "mixing" in variant_cfg, f"{variant}.yaml must declare a mixing: block"
    path = variant_cfg["mixing"].get("buffer_persist_path")
    assert path, f"{variant}.yaml must declare mixing.buffer_persist_path"
    assert variant in path, (
        f"{variant}.yaml's buffer_persist_path {path!r} should be namespaced with "
        "the variant's own name."
    )


def test_run4_gnn_and_smoke_variant_paths_never_collide() -> None:
    """The smoke gate profile must use paths DISTINCT from run4_gnn.yaml's own
    production paths — a smoke gate run and a real run4_gnn launch sharing a
    buffer/anchor file is the exact §RUN3-STEP0 collision class, just at
    variant-vs-variant granularity instead of run-vs-run."""
    prod = yaml.safe_load(Path("configs/variants/run4_gnn.yaml").read_text())
    smoke = yaml.safe_load(Path("configs/variants/run4_gnn_smoke.yaml").read_text())
    assert (
        prod["mixing"]["buffer_persist_path"] != smoke["mixing"]["buffer_persist_path"]
    )
    assert (
        prod["eval_pipeline"]["gating"]["best_model_path"]
        != smoke["eval_pipeline"]["gating"]["best_model_path"]
    )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_variant_config_resolution_namespaces_best_model_path(variant: str) -> None:
    """Resolved-config-level check: the merged combined_config (base +
    variant, the SAME object `hexo_rl.eval.pipeline_setup.build_eval_pipeline`
    deep-merges onto a freshly-loaded configs/eval.yaml as
    `main_ep_override`) carries the namespaced best_model_path — proving the
    declaration actually survives the full merge chain, not just the raw
    on-disk yaml."""
    import structlog

    from hexo_rl.training import orchestrator as _orch

    log = structlog.get_logger()
    args = _args(variant)
    config, _layers = _orch.load_train_config(args)
    combined_config, *_ = _orch.flatten_config_and_resolve_encoding(config, args, log)

    best_model_path = (
        combined_config.get("eval_pipeline", {}).get("gating", {}).get("best_model_path")
    )
    assert best_model_path is not None
    assert best_model_path != _SHARED_DEFAULT_BEST_MODEL_PATH
    assert variant in best_model_path


# ── S7 F6 — run4_gnn_smoke.yaml resolves clean with the labeled smoke overrides ──


def test_run4_gnn_smoke_resolves_clean_and_carries_smoke_overrides():
    """The launch-path smoke profile (S7 pinned ruling, capacity values
    re-tightened post-F9, then again post-run-4 — see run4_gnn_smoke.yaml's
    own SMOKE OVERRIDE comments): batch_size 16 + minimal buffer pool sizing
    + inference_batch_size 16 (both sized against GENUINE-game graphs,
    ~1494 legal-nodes/graph — S7 run-4 witnessed values), everything else
    identical to run4_gnn.yaml's production semantics (same forbidden-weight
    zeros, same corpus-mix-off, same encoding/outcome levers)."""
    import structlog

    from hexo_rl.training import orchestrator as _orch
    from hexo_rl.training.trainer import GRAPH_FORBIDDEN_NONZERO_WEIGHTS

    log = structlog.get_logger()
    args = _args("run4_gnn_smoke")
    config, _layers = _orch.load_train_config(args)
    combined_config, train_cfg, _mcts_cfg, registry_spec, *_ = (
        _orch.flatten_config_and_resolve_encoding(config, args, log)
    )

    assert registry_spec.name == "gnn_axis_v1"
    assert registry_spec.representation == "graph"

    # Labeled smoke overrides actually resolve (post-run-4 capacity values,
    # sized against genuine-game graphs — S7_smoke_gate.md "Re-run 4").
    assert int(combined_config.get("batch_size")) == 16
    assert int(train_cfg.get("min_buffer_size", combined_config.get("min_buffer_size"))) == 64
    _sp = combined_config.get("selfplay", {})
    assert int(_sp.get("inference_batch_size")) == 16
    assert float(combined_config.get("selfplay_stall_timeout_sec")) == 5400.0

    # Production run4_gnn.yaml's own batch_size stays untouched at the base
    # default (256) — the two files must not have converged.
    prod_config, _ = _orch.load_train_config(_args("run4_gnn"))
    prod_combined, *_ = _orch.flatten_config_and_resolve_encoding(prod_config, _args("run4_gnn"), log)
    assert int(prod_combined.get("batch_size")) == 256

    # Same corpus-mix-off + bot-mix-off Decision 1 semantics as production.
    mixing_cfg = train_cfg.get("mixing", combined_config.get("mixing", {}))
    assert mixing_cfg.get("pretrained_buffer_path") is None
    assert float(combined_config.get("bot_batch_share", -1)) == 0.0

    # Same §6.3 forbidden-weight-zero guard as production.
    nonzero = {
        key: combined_config.get(key)
        for key in GRAPH_FORBIDDEN_NONZERO_WEIGHTS
        if float(combined_config.get(key, 0.0)) != 0.0
    }
    assert not nonzero, (
        f"run4_gnn_smoke resolved config carries nonzero graph-forbidden weights "
        f"{nonzero} — same first-train-step loud-raise risk as run4_gnn.yaml."
    )

    from hexo_rl.training.batch_assembly import load_pretrained_buffer

    result = load_pretrained_buffer(
        mixing_cfg, combined_config, emit_fn=lambda evt: None,
        buffer_size=0, buffer_capacity=0,
    )
    assert result is None


# ── S7 round-2 review S-1 [MEDIUM] — smoke/prod resolved-config parity ───────
#
# run4_gnn_smoke.yaml's own header declares itself "a FULL DUPLICATE of ...
# run4_gnn.yaml ... plus the LABELED smoke overrides below" whose drift "must
# be mirrored here by hand" — but nothing enforced that promise. The suite
# above pins each file resolves clean INDIVIDUALLY; none of it pins that the
# two files agree. A future prod edit to an outcome lever (draw_value/
# ply_cap_value/recency_weight — the exact §178 levers run4 exists to test)
# or a GRAPH_FORBIDDEN_NONZERO_WEIGHTS zero could silently diverge the smoke
# gate from the training semantics it exists to validate, and this file's
# other tests would still pass (each resolves its own variant clean, in
# isolation). This test pins full RESOLVED-config parity except an explicit,
# labeled allowlist of smoke-only overrides.

# Dotted (post-flatten) resolved-config keys allowed to differ between
# run4_gnn.yaml and run4_gnn_smoke.yaml. Each entry is a LABELED smoke
# override (see run4_gnn_smoke.yaml's own "SMOKE OVERRIDE" comments) or a
# §RUN3-STEP0 namespaced path (already covered by
# ``test_run4_gnn_and_smoke_variant_paths_never_collide`` above — re-listed
# here so this test's allowlist is self-contained and doesn't silently grow
# by accident).
_SMOKE_ALLOWED_DIVERGENT_KEYS = frozenset({
    "batch_size",                            # S7 F6 labeled override
    "min_buffer_size",                       # S7 F6 labeled override
    "buffer_capacity",                       # S7 F6 labeled override
    "selfplay_stall_timeout_sec",            # S7 F9 follow-up labeled override
    #   (capacity-class: 4060 first-game wall under bf16's genuinely-played
    #    games exceeds the 1800s prod window sized for the 5080 — see
    #    run4_gnn_smoke.yaml's own comment + S7_f9_bf16_fix.md)
    "inference_batch_size",                  # S7 F9 follow-up labeled override
    "selfplay.inference_batch_size",         # (same key, both flatten forms:
    #    declared under `selfplay:` in the yaml — the resolved combined_config
    #    carries it BOTH nested under `selfplay.` and hoisted bare at the
    #    root, so both dotted paths appear in the flattened diff.
    #    Capacity-class: 64-leaf late-game graph batches spiked 1.35 GiB in
    #    the live inference server concurrent with the train step and OOM'd
    #    the 4060 — smoke halves to 32; per-leaf outputs batch-invariant)
    "mixing.buffer_persist_path",            # §RUN3-STEP0 namespacing
    "eval_pipeline.gating.best_model_path",  # §RUN3-STEP0 namespacing
})


def _flatten_resolved_config(d: dict, prefix: str = "") -> dict:
    """Dotted-path flatten of a nested resolved-config dict (leaves only) —
    mirrors the ``mixing:``/``eval_pipeline:`` nesting the raw yaml (and the
    merged ``combined_config``) actually carries."""
    out: dict = {}
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten_resolved_config(value, path + "."))
        else:
            out[path] = value
    return out


def test_run4_gnn_smoke_resolved_config_parity_with_production():
    """Every resolved key must match between run4_gnn.yaml and
    run4_gnn_smoke.yaml EXCEPT ``_SMOKE_ALLOWED_DIVERGENT_KEYS`` — most
    importantly the §178 outcome levers (draw_value/ply_cap_value/
    recency_weight) and every ``GRAPH_FORBIDDEN_NONZERO_WEIGHTS`` key, so a
    future prod edit to those can never silently diverge the smoke gate from
    the training semantics it exists to validate (S7 round-2 review S-1)."""
    import structlog

    from hexo_rl.training import orchestrator as _orch

    log = structlog.get_logger()

    prod_config, _ = _orch.load_train_config(_args("run4_gnn"))
    prod_combined, *_ = _orch.flatten_config_and_resolve_encoding(
        prod_config, _args("run4_gnn"), log
    )
    smoke_config, _ = _orch.load_train_config(_args("run4_gnn_smoke"))
    smoke_combined, *_ = _orch.flatten_config_and_resolve_encoding(
        smoke_config, _args("run4_gnn_smoke"), log
    )

    prod_flat = _flatten_resolved_config(prod_combined)
    smoke_flat = _flatten_resolved_config(smoke_combined)

    _MISSING = object()
    unexpected_diffs = {}
    for key in sorted(set(prod_flat) | set(smoke_flat)):
        if key in _SMOKE_ALLOWED_DIVERGENT_KEYS:
            continue
        prod_val = prod_flat.get(key, _MISSING)
        smoke_val = smoke_flat.get(key, _MISSING)
        if prod_val != smoke_val:
            unexpected_diffs[key] = (prod_val, smoke_val)

    assert not unexpected_diffs, (
        "run4_gnn_smoke.yaml resolved config diverges from run4_gnn.yaml "
        f"outside the labeled smoke-override allowlist: {unexpected_diffs!r} "
        "— a prod edit to configs/variants/run4_gnn.yaml (e.g. an outcome "
        "lever or a §6.3 forbidden-weight zero) was not mirrored into "
        "run4_gnn_smoke.yaml (or vice versa). Either mirror the change by "
        "hand, or if the divergence is a deliberate NEW labeled smoke "
        "override, add the key to _SMOKE_ALLOWED_DIVERGENT_KEYS above with "
        "a comment explaining why."
    )
