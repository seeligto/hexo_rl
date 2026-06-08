"""D-EVALFOUND C4 — steer/abort conjunction-matrix pure functions.

Strength-regression abort is CYCLE-AWARE (suppressed when the ladder is a non-transitive
equilibrium); robustness abort is NEVER cycle-suppressed and is operator opt-in (default
PROMOTE+WARN). SealBot-WR is demoted out of the abort path entirely (tested in the
step_coordinator wiring).
"""
from __future__ import annotations

from hexo_rl.monitoring.alert_rules import (
    check_robustness_abort,
    check_robustness_warn,
    check_strength_regression_abort,
)
from hexo_rl.monitoring.config import MonitoringConfig


def _cfg(**kw):
    base = dict(
        strength_abort_enabled=True, strength_abort_floor=0.45,
        strength_abort_consecutive_evals=3, strength_abort_min_step=25000,
        strength_cycle_density_max=0.15,
        robustness_warn_threshold=0.06, robustness_abort_enabled=False,
        robustness_abort_consecutive_evals=3, robustness_abort_min_step=30000,
    )
    base.update(kw)
    return MonitoringConfig(**base)


# ── strength-regression abort (cycle-aware) ──────────────────────────────────

def test_strength_abort_fires_on_sustained_below_floor():
    hist = [(30000, 0.40), (35000, 0.38), (40000, 0.35)]  # 3 consec < 0.45
    msg = check_strength_regression_abort(hist, cycle_density=0.05, current_step=40000, cfg=_cfg())
    assert msg is not None and "STRENGTH" in msg.upper()


def test_strength_abort_suppressed_by_high_cycle_density():
    # same regression but the ladder is a rock-paper-scissors cloud → NOT a regression
    hist = [(30000, 0.40), (35000, 0.38), (40000, 0.35)]
    msg = check_strength_regression_abort(hist, cycle_density=0.30, current_step=40000, cfg=_cfg())
    assert msg is None


def test_strength_abort_needs_consecutive_evals():
    hist = [(30000, 0.50), (35000, 0.38), (40000, 0.35)]  # only 2 consec below
    assert check_strength_regression_abort(hist, 0.05, 40000, _cfg()) is None


def test_strength_abort_respects_min_step():
    hist = [(10000, 0.40), (15000, 0.38), (20000, 0.35)]
    assert check_strength_regression_abort(hist, 0.05, 20000, _cfg()) is None  # before min_step


def test_strength_abort_disabled_returns_none():
    hist = [(30000, 0.40), (35000, 0.38), (40000, 0.35)]
    assert check_strength_regression_abort(hist, 0.05, 40000, _cfg(strength_abort_enabled=False)) is None


def test_strength_abort_healthy_aggregate_no_fire():
    hist = [(30000, 0.55), (35000, 0.58), (40000, 0.60)]  # above floor
    assert check_strength_regression_abort(hist, 0.05, 40000, _cfg()) is None


# ── robustness watch / abort ─────────────────────────────────────────────────

def test_robustness_warn_above_threshold():
    assert check_robustness_warn(exploit_rate=0.235, cfg=_cfg()) is not None


def test_robustness_warn_silent_at_or_below_threshold():
    assert check_robustness_warn(exploit_rate=0.05, cfg=_cfg()) is None


def test_robustness_abort_off_by_default():
    hist = [(30000, 0.30), (35000, 0.28), (40000, 0.25)]  # all breaching 0.06
    assert check_robustness_abort(hist, 40000, _cfg()) is None  # robustness_abort_enabled=False


def test_robustness_abort_fires_when_armed_and_breached():
    hist = [(35000, 0.30), (40000, 0.28), (45000, 0.25)]
    cfg = _cfg(robustness_abort_enabled=True)
    msg = check_robustness_abort(hist, 45000, cfg)
    assert msg is not None and "ROBUST" in msg.upper()


def test_robustness_abort_never_cycle_suppressed():
    # robustness abort takes NO cycle-density arg — it can never be suppressed by a cycle.
    import inspect
    sig = inspect.signature(check_robustness_abort)
    assert "cycle_density" not in sig.parameters


# ── REVIEW follow-ups: single-eval strength WARN + Objective-A coverage pre-flight ──

from hexo_rl.monitoring.alert_rules import (  # noqa: E402
    check_objective_a_coverage,
    check_strength_warn,
)


def test_strength_warn_single_eval_below_floor():
    # spec §1a row 1: WARN on a SINGLE eval below floor (no consecutive requirement)
    assert check_strength_warn(strength_aggregate=0.40, cfg=_cfg()) is not None


def test_strength_warn_silent_above_floor():
    assert check_strength_warn(strength_aggregate=0.50, cfg=_cfg()) is None


def test_strength_warn_silent_when_aggregate_missing():
    assert check_strength_warn(strength_aggregate=None, cfg=_cfg()) is None


def test_objective_a_coverage_warns_when_all_off():
    # SealBot demoted + strength abort off + robustness off + monitor off = NO Obj-A guard
    msg = check_objective_a_coverage(
        strength_abort_enabled=False, robustness_abort_enabled=False,
        offwindow_monitor_enabled=False,
    )
    assert msg is not None and "objective" in msg.lower()


def test_objective_a_coverage_ok_when_robustness_monitor_on():
    # monitor on → off-window rate flows to promote gate + WARN → Objective-A covered
    assert check_objective_a_coverage(
        strength_abort_enabled=False, robustness_abort_enabled=False,
        offwindow_monitor_enabled=True,
    ) is None


def test_objective_a_coverage_ok_when_strength_abort_on():
    assert check_objective_a_coverage(
        strength_abort_enabled=True, robustness_abort_enabled=False,
        offwindow_monitor_enabled=False,
    ) is None
