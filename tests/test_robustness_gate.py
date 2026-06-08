"""D-EVALFOUND C2 — robustness gate primitive (Objective A: off-window exploitability).

Wraps the deterministic off-window adversary (offwindow_probe.run_adversary_games,
single source) as a tracked, config-keyed gate. Threshold 0.06 = EXT-LINK fix-acceptance;
deployed single-window v6_live2 ≈ 0.235 = the defect. WATCH axis: gates PROMOTE + WARN,
not strength. Tests the pure verdict/gate logic + config.
"""
from __future__ import annotations

import pytest

from hexo_rl.eval.robustness_gate import (
    RobustnessGateConfig,
    classify_verdict,
    gate_passes,
)


# Pre-registered thresholds (exploit_probe.py:124, locked 2026-06-06):
# DEFENDED ≤ 0.05 ; FORCEABLE ≥ 0.15 AND (exploit−control) ≥ 0.10 ; else INCONCLUSIVE.


def test_verdict_defended_when_low():
    assert classify_verdict(exploit_rate=0.03, control_rate=0.02) == "DEFENDED"


def test_verdict_forceable_deployed_defect():
    # deployed single-window v6_live2: exploit 0.235, control 0.062 → FORCEABLE
    assert classify_verdict(exploit_rate=0.235, control_rate=0.062) == "FORCEABLE"


def test_verdict_inconclusive_between():
    # above DEFENDED (0.05) but margin < 0.10 → neither
    assert classify_verdict(exploit_rate=0.12, control_rate=0.10) == "INCONCLUSIVE"


def test_verdict_high_rate_but_thin_margin_is_inconclusive():
    # exploit ≥ 0.15 but margin < 0.10 (control nearly as high) → NOT FORCEABLE
    assert classify_verdict(exploit_rate=0.20, control_rate=0.15) == "INCONCLUSIVE"


def test_gate_passes_at_or_below_threshold():
    assert gate_passes(exploit_rate=0.06, threshold=0.06) is True
    assert gate_passes(exploit_rate=0.04, threshold=0.06) is True


def test_gate_fails_above_threshold():
    assert gate_passes(exploit_rate=0.07, threshold=0.06) is False
    assert gate_passes(exploit_rate=0.235, threshold=0.06) is False


def test_config_defaults_locked_to_prereg():
    cfg = RobustnessGateConfig()
    assert cfg.threshold == 0.06          # fix-acceptance
    assert cfg.n_per_arm == 200
    assert cfg.sims == 128
    assert cfg.opening_plies == 6
    assert cfg.force_spec_mismatch is False


def test_config_from_dict_overrides_and_keeps_defaults():
    cfg = RobustnessGateConfig.from_dict({"enabled": True, "threshold": 0.05})
    assert cfg.enabled is True
    assert cfg.threshold == 0.05
    assert cfg.n_per_arm == 200  # default preserved


def test_config_from_dict_rejects_unknown_key():
    # config-first discipline: a typo'd key must fail loud, not silently no-op.
    with pytest.raises((ValueError, TypeError)):
        RobustnessGateConfig.from_dict({"thresold": 0.05})  # typo of "threshold"
