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

Drives the REAL config-merge entrypoint (`hexo_rl.training.orchestrator`)
against the REAL `configs/variants/run4_gnn.yaml` on disk (not a
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

import yaml


def _args() -> argparse.Namespace:
    """Minimal Namespace covering the attrs `load_train_config` /
    `flatten_config_and_resolve_encoding` read (mirrors
    tests/test_run3_corpus_launch_path.py::_args)."""
    return argparse.Namespace(
        config=None,
        variant="run4_gnn",
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
