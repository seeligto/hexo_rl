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


def test_wave2_trajectory_would_have_fired_at_step_30k():
    """Wave 2 actual trajectory: peak 33% @ 20k → 11% @ 30k. The L50 gate
    would have caught the collapse at step 30k (Trigger B: current 11% <
    peak 33% × 0.5 = 16.5%) — 17k steps earlier than the §S180b 8%
    threshold that actually fired at step 40k.

    This is the load-bearing integration test for the L50 fix.
    """
    cfg = MonitoringConfig()
    wave2_history = [(10000, 0.24), (20000, 0.33), (30000, 0.11)]
    msg = check_sealbot_wr_hard_abort(wave2_history, 30000, cfg)
    assert msg is not None, (
        "Wave 2 collapse trajectory must trigger HARD-ABORT at step 30k. "
        "If this test fails the L50 fix is broken — the gate would miss "
        "Wave-2-style peak-and-collapse."
    )
    assert "Wave3-B" in msg
