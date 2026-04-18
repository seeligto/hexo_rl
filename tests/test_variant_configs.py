"""Variant-config regressions — resolved configs after train.py deep-merge.

Each test loads a base+variant config stack through ``hexo_rl.utils.config.load_config``
(the same path ``scripts/train.py`` takes when ``--variant`` is passed) and asserts
on the resolved values. Guards against silent drift when the base ``selfplay.yaml``
grows new keys that variants inherit by accident.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from hexo_rl.utils.config import load_config
from hexo_rl.utils.variant_validator import validate_variant_against_bases

# Match scripts/train.py::main `_BASE_CONFIGS` (minus monitoring/game_replay —
# they do not touch selfplay.playout_cap and keep the test surface minimal).
ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIGS = [
    str(ROOT / "configs" / "model.yaml"),
    str(ROOT / "configs" / "training.yaml"),
    str(ROOT / "configs" / "selfplay.yaml"),
]


def _resolve(variant: str) -> dict:
    variant_path = ROOT / "configs" / "variants" / f"{variant}.yaml"
    return load_config(*BASE_CONFIGS, str(variant_path))


def _base_cfgs_for_validator() -> dict:
    names = ["model", "training", "selfplay"]
    result = {}
    for name in names:
        p = ROOT / "configs" / f"{name}.yaml"
        with open(p) as f:
            result[name] = yaml.safe_load(f) or {}
    return result


def test_baseline_puct_pins_pre_100_semantics() -> None:
    """§102.b — baseline_puct must opt out of the move-level selective policy
    loss (§100) so it reproduces pre-§100 training semantics. Before §102.b
    the variant silently inherited ``full_search_prob: 0.25`` from the base,
    turning the "pre-§67 historical baseline" into a §100-selective run and
    confounding any ablation that used this variant as an unmodified control.
    """
    cfg = _resolve("baseline_puct")
    playout_cap = cfg["selfplay"]["playout_cap"]
    assert playout_cap["full_search_prob"] == 0.0, (
        "baseline_puct inherits selective policy loss from the base — §102.b "
        "pin is missing; see reports/selective_policy_audit_2026-04-18.md §4 B2"
    )
    assert playout_cap["fast_prob"] == 0.0, (
        "baseline_puct has fast_prob != 0.0 — pre-§100 semantics require both "
        "playout caps OFF"
    )


def test_gumbel_full_passes_playout_cap_mutex() -> None:
    """§104 — gumbel_full.yaml must resolve with ``fast_prob == 0`` so it
    survives the WorkerPool mutex (``fast_prob > 0 AND full_search_prob > 0``
    raises at pool init). Pre-§104 the variant set ``fast_prob: 0.25`` and
    then inherited ``full_search_prob: 0.25`` from the base, making the
    desktop Exp E variant un-launchable. Decision justification:
    reports/gumbel_target_quality_2026-04-17.md (D-Gumbel verdict: Option A).
    """
    cfg = _resolve("gumbel_full")
    playout_cap = cfg["selfplay"]["playout_cap"]
    assert playout_cap["fast_prob"] == 0.0, (
        "gumbel_full.fast_prob != 0 — §104 Option A repair missing; the "
        "variant will crash WorkerPool.__init__ on launch"
    )
    assert playout_cap["full_search_prob"] > 0.0, (
        "gumbel_full.full_search_prob == 0 — variant must keep the §100 "
        "move-level cap (inherited from base) so the selective gate actually "
        "fires on quick-search rows"
    )
    # §100 sim counts must survive the merge — noise budget is not a free parameter.
    assert playout_cap["n_sims_quick"] == 100
    assert playout_cap["n_sims_full"] == 600
    # Variant-defining flags must still be set.
    assert cfg["selfplay"]["gumbel_mcts"] is True
    assert cfg["selfplay"]["completed_q_values"] is True


# ---------------------------------------------------------------------------
# E-001 / E-018 / E-019 / Q29 regression tests
# ---------------------------------------------------------------------------


def test_all_variants_have_no_nested_base_namespaces() -> None:
    """E-001/E-018/E-019: every configs/variants/*.yaml must have no nested
    namespace block that shadows a flat base key (silent-drop class of bugs)."""
    base_cfgs = _base_cfgs_for_validator()
    variant_dir = ROOT / "configs" / "variants"
    failures: list[str] = []
    for variant_path in sorted(variant_dir.glob("*.yaml")):
        with open(variant_path) as f:
            variant_cfg = yaml.safe_load(f) or {}
        warnings = validate_variant_against_bases(variant_cfg, base_cfgs)
        for w in warnings:
            failures.append(f"{variant_path.name}: {w}")
    assert not failures, "Nested namespace shadows found:\n" + "\n".join(failures)


def test_validator_catches_nested_training_block() -> None:
    """Q29: validator must flag a variant with training: {...} when training.yaml
    is flat — this was the E-001 class of bug."""
    base_cfgs = _base_cfgs_for_validator()
    synthetic_variant = {"training": {"training_steps_per_game": 4.0}}
    warnings = validate_variant_against_bases(synthetic_variant, base_cfgs)
    assert warnings, (
        "validator did not flag nested 'training:' block — "
        "E-001 class bug would go undetected"
    )
    assert any("training" in w for w in warnings)


def test_training_steps_per_game_resolves_to_variant_value_for_desktop() -> None:
    """E-001: after flattening gumbel_targets_desktop.yaml, max_train_burst must
    resolve to 8 at the top level (not 16 from the base). Pre-fix the nested
    training: block silently dropped the override and desktop ran at burst=16."""
    cfg = _resolve("gumbel_targets_desktop")
    assert cfg["max_train_burst"] == 8, (
        f"max_train_burst={cfg['max_train_burst']} — expected 8 from desktop D3 sweep; "
        "nested training: block fix may be missing or broken"
    )
    # No nested 'training' dict should survive — flat keys only.
    assert not isinstance(cfg.get("training"), dict), (
        "merged config has nested 'training' dict — variant still uses wrong namespace"
    )
