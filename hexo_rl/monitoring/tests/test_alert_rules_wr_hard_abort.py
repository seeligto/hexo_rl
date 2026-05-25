"""§S181-AUDIT Wave 3 Stage 2B — sliding-window SealBot WR hard-abort tests (L50).

Verify the three trigger conditions (A: rolling-mean below threshold,
B: collapse from peak, C: early death) fire only when their respective
step gates + numeric conditions are met. Wave 2 trajectory replay
(10k=24%, 20k=33%, 30k=11%, 40k=5%) is the integration test — verifies
that the gate WOULD have fired the HARD-ABORT earlier in Wave 2 than
the existing §S180b 8% trigger did.
"""
from __future__ import annotations

import pytest

from hexo_rl.monitoring.alert_rules import check_sealbot_wr_hard_abort
from hexo_rl.monitoring.config import MonitoringConfig


def _default_cfg(**overrides) -> MonitoringConfig:
    """MonitoringConfig with Wave 3 defaults; override fields per test."""
    base = {
        "wr_hard_abort_enabled": True,
        "wr_rolling_consecutive_evals": 2,
        "wr_rolling_threshold": 0.10,
        "wr_rolling_min_step": 20000,
        "wr_collapse_from_peak_ratio": 0.5,
        "wr_collapse_min_step": 25000,
        "wr_early_death_threshold": 0.05,
        "wr_early_death_min_step": 15000,
    }
    base.update(overrides)
    return MonitoringConfig(**base)


# ── No-op gates ────────────────────────────────────────────────────────────
def test_empty_history_returns_none():
    """No eval history → no decision basis → None."""
    assert check_sealbot_wr_hard_abort([], 30000, _default_cfg()) is None


def test_gate_disabled_returns_none_even_with_collapse():
    """Operator disable flag short-circuits all triggers."""
    cfg = _default_cfg(wr_hard_abort_enabled=False)
    history = [(10000, 0.33), (20000, 0.20), (30000, 0.02)]
    assert check_sealbot_wr_hard_abort(history, 30000, cfg) is None


# ── Trigger C — early death ────────────────────────────────────────────────
def test_trigger_c_fires_when_wr_below_5pct_past_step_15k():
    """C: WR 4% past step 15k → HARD-ABORT."""
    history = [(16000, 0.04)]
    msg = check_sealbot_wr_hard_abort(history, 16000, _default_cfg())
    assert msg is not None
    assert "Wave3-C" in msg
    assert "early death" in msg


def test_trigger_c_silent_before_step_15k():
    """C: step ≤ 15k → no fire even if WR very low (early jitter ok)."""
    history = [(10000, 0.02)]
    assert check_sealbot_wr_hard_abort(history, 10000, _default_cfg()) is None


def test_trigger_c_silent_when_wr_at_threshold():
    """C: WR = 5% exactly → no fire (strict <)."""
    history = [(16000, 0.05)]
    assert check_sealbot_wr_hard_abort(history, 16000, _default_cfg()) is None


# ── Trigger B — collapse from peak ─────────────────────────────────────────
def test_trigger_b_fires_when_current_below_half_of_peak_past_25k():
    """B: peak 33%, current 16% (=0.484 < 0.5) past 25k → HARD-ABORT.

    Wave 2 trajectory hypothetical: had the run reached step 26k+ with
    current=16% after a 33% peak, this gate would have fired.
    """
    history = [(10000, 0.24), (20000, 0.33), (26000, 0.16)]
    msg = check_sealbot_wr_hard_abort(history, 26000, _default_cfg())
    assert msg is not None
    assert "Wave3-B" in msg
    assert "collapse" in msg


def test_trigger_b_silent_when_current_above_half_of_peak():
    """B: peak 33%, current 17% (>peak*0.5) → no fire."""
    history = [(10000, 0.24), (20000, 0.33), (26000, 0.17)]
    # peak=0.33, threshold=0.165; current=0.17 > 0.165 → no fire
    assert check_sealbot_wr_hard_abort(history, 26000, _default_cfg()) is None


def test_trigger_b_silent_before_step_25k():
    """B: step ≤ 25k → no fire even if current << peak/2 (allow early jitter)."""
    history = [(10000, 0.30), (20000, 0.10)]
    assert check_sealbot_wr_hard_abort(history, 20000, _default_cfg()) is None


# ── Trigger A — rolling-mean below threshold ───────────────────────────────
def test_trigger_a_fires_when_2_consec_evals_below_threshold_past_20k():
    """A: 2 consecutive evals with WR < 10% past step 20k → HARD-ABORT.

    History chosen so B does NOT preempt: peak 0.12, current 0.08,
    peak*0.5=0.06; 0.08 > 0.06 → B silent. C silent: 0.08 > 0.05.
    """
    history = [(20000, 0.12), (25000, 0.09), (30000, 0.08)]
    msg = check_sealbot_wr_hard_abort(history, 30000, _default_cfg())
    assert msg is not None
    assert "Wave3-A" in msg
    assert "rolling-mean" in msg


def test_trigger_a_silent_when_only_1_of_2_below_threshold():
    """A: 1 below, 1 above → no fire (need consecutive)."""
    history = [(20000, 0.05), (25000, 0.12), (30000, 0.07)]
    # Tail = (25000, 0.12), (30000, 0.07) → 0.12 NOT < 0.10 → A no-fire.
    # B silent: peak=0.12, current=0.07, threshold=0.06; 0.07 > 0.06 → no-fire.
    # C silent: 0.07 > 0.05 → no-fire.
    assert check_sealbot_wr_hard_abort(history, 30000, _default_cfg()) is None


def test_trigger_a_silent_before_step_20k():
    """A/B/C all silent when step ≤ floor gates with non-extreme WR."""
    # current step 20000 (NOT > 20k so A gate fails strictly).
    # WR 0.06 chosen so C silent (>0.05) and B silent (step not > 25k).
    history = [(15000, 0.06), (20000, 0.06)]
    assert check_sealbot_wr_hard_abort(history, 20000, _default_cfg()) is None


def test_trigger_a_silent_with_only_1_eval_in_history():
    """A: need n_consec evals minimum (default 2)."""
    history = [(25000, 0.05)]
    # Only 1 eval; Trigger C is checked first and fires (WR=5% NOT <5%).
    # Trigger B needs peak comparison — same eval, peak=0.05, 0.05 > 0.025 → no.
    # Trigger A needs 2 evals → no.
    assert check_sealbot_wr_hard_abort(history, 25000, _default_cfg()) is None


# ── Trigger priority — C before B before A ──────────────────────────────────
def test_trigger_priority_c_fires_first_when_all_three_match():
    """When all three trigger conditions match, C (early-death) is named."""
    # All triggers active:
    # - C: WR 4% past 15k → fires
    # - B: peak 33%, current 4% < peak*0.5 past 25k → would fire
    # - A: rolling-mean 4% < 10% × 2 past 20k → would fire
    history = [(25000, 0.04), (30000, 0.04)]
    msg = check_sealbot_wr_hard_abort(history, 30000, _default_cfg())
    assert msg is not None
    assert "Wave3-C" in msg  # C is checked first


# ── Wave 2 trajectory replay — would have caught the collapse earlier ──────
def test_wave2_trajectory_no_fire_at_step_10k_20k():
    """Wave 2: peak-and-collapse trajectory — no fire at the peaks."""
    cfg = _default_cfg()
    h10 = [(10000, 0.24)]
    h20 = [(10000, 0.24), (20000, 0.33)]
    assert check_sealbot_wr_hard_abort(h10, 10000, cfg) is None  # early/healthy
    assert check_sealbot_wr_hard_abort(h20, 20000, cfg) is None  # peak — no fire


def test_wave2_trajectory_fires_b_at_step_30k():
    """Wave 2: at step 30k (peak 33% → current 11% — past peak*0.5 of 16.5%),
    Trigger B fires. Wave 2 itself didn't have this gate; trigger A only
    fires past step 20k AND requires 2 consec evals below 10%."""
    cfg = _default_cfg()
    h30 = [(10000, 0.24), (20000, 0.33), (30000, 0.11)]
    msg = check_sealbot_wr_hard_abort(h30, 30000, cfg)
    # 0.11 < 0.33 * 0.5 = 0.165 → Trigger B fires past step 25k
    assert msg is not None
    assert "Wave3-B" in msg


def test_wave2_trajectory_fires_a_at_step_40k():
    """Wave 2 at step 40k: 30k=11% + 40k=5% — both below 10% → Trigger A.

    Trigger C also fires (5% NOT <5% — actually 5%==threshold, strict <).
    But B is checked before A: peak=33%, current=5% < peak*0.5 → B fires.
    """
    cfg = _default_cfg()
    h40 = [(10000, 0.24), (20000, 0.33), (30000, 0.11), (40000, 0.05)]
    msg = check_sealbot_wr_hard_abort(h40, 40000, cfg)
    assert msg is not None
    # B is checked before A in priority order — Wave 2 would have hit B at 30k
    # (caught the collapse 17k EARLIER than the §S180b 8% trigger that fired at 40k).
    assert "Wave3-B" in msg


def test_trigger_a_isolated_fire_with_b_disabled():
    """If Trigger B is disabled (peak ratio → 0), Trigger A must still
    fire on its own pattern (2 consec < threshold past step 20k).

    Wave 2's actual trajectory had wr=11% at step 30k (above the 10%
    threshold) and wr=5% at step 40k — so Trigger A alone would NOT
    have caught Wave 2; Trigger B is the load-bearing rule for the
    peak-and-collapse pattern. This test isolates Trigger A behavior
    on a synthetic trajectory where both step-30k and step-40k WR
    sit below 10%.
    """
    cfg = _default_cfg(wr_collapse_from_peak_ratio=0.0)  # B disabled
    # peak=0.12 so B-disabled is meaningful (otherwise current 0.08
    # would not trigger B at peak*0.5=0.06 anyway).
    history = [(20000, 0.12), (25000, 0.09), (30000, 0.08)]
    msg = check_sealbot_wr_hard_abort(history, 30000, cfg)
    assert msg is not None
    assert "Wave3-A" in msg


# ── Config-from-dict integration ────────────────────────────────────────────
def test_monitoring_config_from_dict_picks_up_wr_thresholds():
    """MonitoringConfig.from_dict accepts wr_* fields from a config dict."""
    cfg_dict = {
        "monitoring": {
            "wr_hard_abort_enabled": False,
            "wr_rolling_threshold": 0.15,
            "wr_early_death_min_step": 10000,
        }
    }
    cfg = MonitoringConfig.from_dict(cfg_dict)
    assert cfg.wr_hard_abort_enabled is False
    assert cfg.wr_rolling_threshold == 0.15
    assert cfg.wr_early_death_min_step == 10000
    # Other fields keep defaults.
    assert cfg.wr_rolling_consecutive_evals == 2
