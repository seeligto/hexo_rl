"""§S181-AUDIT Wave 3 Stage 2B INV pin: sliding-window SealBot WR hard-abort.

Guards the L50 fix — Wave 2 evidence: alt V_spread held +0.25 throughout
while wr_sealbot collapsed 33% → 5%. The held-out V_spread canary
failed to track actual eval performance. L50 mandates a SealBot WR
sliding-window gate as the PRIMARY abort trigger.

The pure-function rule lives at hexo_rl/monitoring/alert_rules.py:
``check_sealbot_wr_hard_abort``. The MonitoringConfig dataclass at
hexo_rl/monitoring/config.py carries the threshold defaults.

If a future refactor drops the pure function, moves the thresholds, or
re-enables the alt V_spread canary as the sole SOFT-ABORT gate, this
test breaks immediately.
"""
from __future__ import annotations

from pathlib import Path

from hexo_rl.monitoring.alert_rules import check_sealbot_wr_hard_abort
from hexo_rl.monitoring.config import MonitoringConfig


def test_check_sealbot_wr_hard_abort_function_exists():
    """The pure-function rule must exist + match the (history, step, cfg) signature."""
    assert callable(check_sealbot_wr_hard_abort)
    # Sanity smoke — disabled gate returns None.
    cfg = MonitoringConfig(wr_hard_abort_enabled=False)
    assert check_sealbot_wr_hard_abort([(10000, 0.05)], 10000, cfg) is None


def test_monitoring_config_carries_wr_thresholds():
    """MonitoringConfig must carry the 8 Wave 3 wr_* fields."""
    cfg = MonitoringConfig()
    for field in (
        "wr_hard_abort_enabled",
        "wr_rolling_consecutive_evals",
        "wr_rolling_threshold",
        "wr_rolling_min_step",
        "wr_collapse_from_peak_ratio",
        "wr_collapse_min_step",
        "wr_collapse_consecutive_evals",
        "wr_early_death_threshold",
        "wr_early_death_min_step",
    ):
        assert hasattr(cfg, field), (
            f"MonitoringConfig.{field} missing — Wave 3 Stage 2B threshold lost"
        )


def test_wr_hard_abort_defaults_match_l50_spec():
    """Wave 3 dispatcher Stage 2B specifies these exact defaults."""
    cfg = MonitoringConfig()
    assert cfg.wr_hard_abort_enabled is True
    assert cfg.wr_rolling_consecutive_evals == 2
    assert cfg.wr_rolling_threshold == 0.10
    assert cfg.wr_rolling_min_step == 20000
    assert cfg.wr_collapse_from_peak_ratio == 0.5
    assert cfg.wr_collapse_min_step == 25000
    assert cfg.wr_collapse_consecutive_evals == 3
    assert cfg.wr_early_death_threshold == 0.05
    assert cfg.wr_early_death_min_step == 15000


def test_l50_canary_downgrade_documented_in_module_docstring():
    """check_value_spread_canary docstring must note L50 downgrade.

    Wave 3 downgrades the dual-bank V_spread canary from SOFT-ABORT to
    INFORMATIONAL per L50. Future refactors must preserve this note so
    the L50 lesson is not lost.
    """
    src_path = Path(__file__).resolve().parents[1] / "hexo_rl" / "monitoring" / "alert_rules.py"
    text = src_path.read_text()
    canary_idx = text.find("def check_value_spread_canary")
    assert canary_idx > 0, "check_value_spread_canary missing — PR-C canary lost"
    docstring_window = text[canary_idx:canary_idx + 2000]
    assert "L50" in docstring_window, (
        "check_value_spread_canary docstring must note the Wave 3 L50 downgrade "
        "(informational only)."
    )


def test_collapse_requires_consecutive_evals_not_single_dip():
    """3-consecutive policy (2026-06-07): Triggers B/C require
    ``wr_collapse_consecutive_evals`` consecutive low evals, NOT a single dip.

    Motivation: the colony attractor causes transient SealBot-WR dips that
    self-correct (§175/L34). A single 5% dip at step 75k fired the OLD
    single-point Trigger B and HARD-ABORTED a RECOVERING golong run at 87.5k
    (the 87.5k re-eval recovered to ~0.23). Load-bearing in BOTH directions: a
    single dip must NOT fire, a SUSTAINED collapse MUST still fire.
    """
    cfg = MonitoringConfig()
    assert cfg.wr_collapse_consecutive_evals == 3

    # The golong incident — peak 0.29 then a SINGLE 0.05 dip → must NOT fire.
    single_dip = [(50000, 0.38), (62500, 0.29), (75000, 0.05)]
    assert check_sealbot_wr_hard_abort(single_dip, 87500, cfg) is None, (
        "A single transient WR dip must NOT hard-abort — it killed a recovering "
        "golong run at 87.5k before this fix."
    )
    # Dip then recovery → must NOT fire.
    recovered = [(62500, 0.29), (75000, 0.05), (87500, 0.23)]
    assert check_sealbot_wr_hard_abort(recovered, 87500, cfg) is None

    # A SUSTAINED Wave-2-style collapse (3 consecutive below peak 0.38 × 0.5 =
    # 0.19) MUST still HARD-ABORT — the gate must not miss a real collapse.
    sustained = [(20000, 0.38), (30000, 0.12), (42500, 0.08), (55000, 0.06)]
    msg = check_sealbot_wr_hard_abort(sustained, 55000, cfg)
    assert msg is not None and "Wave3-B" in msg, (
        "A sustained 3-consecutive collapse must still HARD-ABORT — the gate "
        "must not miss a real Wave-2-style peak-and-collapse."
    )
