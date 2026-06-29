"""D-ZVALID Z2 — pre-registered verdict logic (pure, no engine/torch).

Locks the TEACHES / DOESN'T-TEACH / INDETERMINATE decision so the GPU-week gate cannot
drift. The standalone-ladder script's heavy paths (model load, self-play, SealBot) need
the engine binding; the `decide()` decision is pure and is what the pre-reg hinges on.
"""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "run_z2_standalone_ladder.py"
_spec = importlib.util.spec_from_file_location("run_z2_standalone_ladder", _SCRIPT)
z2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z2)  # module-level imports are stdlib-only (argparse/json/pathlib)


def _decide(**kw):
    base = dict(
        base_trap=0.50, cand_trap=0.50, base_wr=0.50, cand_wr=0.50, elo_ci_lo=-5.0,
        teach_trap_drop=0.10, teach_wr_rise=0.03, elo_distinct_ok=True,
        z_loss_coverage=0.8, min_z_coverage=0.5,
    )
    base.update(kw)
    return z2.decide(**base)


def test_teaches_requires_trap_drop_and_strength_rise():
    # trap drops 0.50 -> 0.35 (>= 0.10) AND self-play Elo CI-lo > 0.
    assert _decide(cand_trap=0.35, elo_ci_lo=12.0) == "TEACHES"
    # strength can come from a SealBot WR rise >= teach_wr_rise instead of Elo.
    assert _decide(cand_trap=0.35, cand_wr=0.62, elo_ci_lo=-3.0) == "TEACHES"


def test_doesnt_teach_when_flat_on_both_with_measured_recall():
    # flat trap + flat strength, but the fine-tune HAD adequate z-coverage -> earned.
    assert _decide(cand_trap=0.49, cand_wr=0.50, elo_ci_lo=-4.0, z_loss_coverage=0.8) == "DOESNT_TEACH"


def test_wr_arm_needs_min_rise_not_bare_positive():
    # FIX 2: a noise-level WR uptick (+0.005 < teach_wr_rise=0.03) must NOT satisfy the
    # corroboration arm — trap dropped but no real strength rise -> MIXED, not TEACHES.
    assert _decide(cand_trap=0.35, cand_wr=0.505, elo_ci_lo=-3.0) == "INDETERMINATE_MIXED"
    # a real WR rise (+0.05 >= 0.03) does corroborate.
    assert _decide(cand_trap=0.35, cand_wr=0.55, elo_ci_lo=-3.0) == "TEACHES"


def test_starved_recall_withholds_doesnt_teach():
    # FIX 1: flat trap + flat strength but z-coverage BELOW floor -> the lever was starved,
    # not dead; do NOT kill the GPU-week. INDETERMINATE_STARVED_RECALL, not DOESNT_TEACH.
    assert _decide(cand_trap=0.49, cand_wr=0.50, elo_ci_lo=-4.0, z_loss_coverage=0.2) \
        == "INDETERMINATE_STARVED_RECALL"
    # unmeasured coverage (None) is also not enough to EARN DOESNT_TEACH (the safe failure).
    assert _decide(cand_trap=0.49, cand_wr=0.50, elo_ci_lo=-4.0, z_loss_coverage=None) \
        == "INDETERMINATE_STARVED_RECALL"


def test_underpowered_dominates_everything():
    # No distinct-game power -> CI untrusted, regardless of point estimates.
    assert _decide(cand_trap=0.10, cand_wr=0.9, elo_ci_lo=99.0, elo_distinct_ok=False) \
        == "INDETERMINATE_UNDERPOWERED"


def test_no_trapset_is_indeterminate_not_teaches():
    # Strength rose but the primary internalisation signal (trap drop) is unavailable.
    assert _decide(base_trap=None, cand_trap=None, cand_wr=0.62) == "INDETERMINATE_NO_TRAPSET"


def test_mixed_signal_is_indeterminate():
    # Trap dropped but neither strength axis moved -> mixed, not a clean teach.
    assert _decide(cand_trap=0.35, cand_wr=0.50, elo_ci_lo=-2.0) == "INDETERMINATE_MIXED"


def test_trap_drop_below_threshold_is_not_a_teach():
    # 0.50 -> 0.45 is only a 0.05 drop (< 0.10 threshold) -> not taught even with Elo rise.
    assert _decide(cand_trap=0.45, elo_ci_lo=12.0) == "INDETERMINATE_MIXED"
