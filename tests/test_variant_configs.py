"""Variant-config regressions — resolved configs after train.py deep-merge.

Each test loads a base+variant config stack through ``hexo_rl.utils.config.load_config``
(the same path ``scripts/train.py`` takes when ``--variant`` is passed) and asserts
on the resolved values. Guards against silent drift when the base ``selfplay.yaml``
grows new keys that variants inherit by accident.
"""
from __future__ import annotations

from pathlib import Path

from hexo_rl.utils.config import load_config

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
