"""§D-LOOPFIX W1 — the run lifecycle epilogue (close-out).

Training-stop ≠ process-exit. When the run stops at step N the coordinator must:
  DRAIN the in-flight eval (budget sized from the measured round wall-clock ×
  safety factor, hard-capped — NOT a flat 900 s that is 10-16× too small for a
  full round under load) → run a TERMINAL full-battery eval on the FINAL
  checkpoint (all opponents, stride parity IGNORED, step-stamped, recorded as
  TERMINAL not steer-input) → exit.

Pre-fix the terminal promotion-capable round landed on the stop boundary and was
killed at sealbot game 99/100 by the 900 s drain cap → ZERO completed promotion
decision on the final checkpoint, and the stride-4 nnue/offwindow phases got zero
reads all run.
"""
from __future__ import annotations

import time
from unittest.mock import Mock, patch

from hexo_rl.training.step_coordinator import promotion_capable_rounds
from tests.training.test_step_coordinator import _make_coordinator


# ── cadence: surface stride-parity promotion-incapability (was silent) ────────

def test_promotion_capable_rounds_stride2_reproduces_the_w1_two_rounds():
    # The A/B's schedule: stop 50000, interval 12500 → rounds 1..4; best stride 2
    # → only rounds 2 (25000) and 4 (50000) are promotion-capable (W1).
    assert promotion_capable_rounds(50000, 12500, 2) == [2, 4]


def test_promotion_capable_rounds_stride1_every_round():
    assert promotion_capable_rounds(50000, 12500, 1) == [1, 2, 3, 4]


def test_promotion_capable_rounds_empty_when_stride_exceeds_round_count():
    # stride 8 with only 4 rounds → ZERO in-run promotion-capable rounds.
    assert promotion_capable_rounds(50000, 12500, 8) == []


def test_promotion_capable_rounds_guards_bad_inputs():
    assert promotion_capable_rounds(None, 12500, 1) == []
    assert promotion_capable_rounds(50000, 0, 1) == []


# ── drain budget: measured round × safety factor, floored + hard-capped ──────

def test_final_drain_budget_floors_when_no_round_measured():
    coord = _make_coordinator(config_overrides={
        "final_eval_drain_timeout_sec": 900.0,
        "eval_final_drain_safety_factor": 3.0,
        "eval_final_drain_hard_cap_sec": 14400.0,
    })
    coord._last_eval_round_sec = None
    assert coord._final_drain_budget_sec() == 900.0


def test_final_drain_budget_scales_from_measured_round():
    coord = _make_coordinator(config_overrides={
        "final_eval_drain_timeout_sec": 900.0,
        "eval_final_drain_safety_factor": 3.0,
        "eval_final_drain_hard_cap_sec": 14400.0,
    })
    coord._last_eval_round_sec = 1000.0
    assert coord._final_drain_budget_sec() == 3000.0   # 1000 × 3, above the 900 floor


def test_final_drain_budget_hard_caps_a_huge_round():
    coord = _make_coordinator(config_overrides={
        "final_eval_drain_timeout_sec": 900.0,
        "eval_final_drain_safety_factor": 3.0,
        "eval_final_drain_hard_cap_sec": 14400.0,
    })
    coord._last_eval_round_sec = 100_000.0   # 4h round under load
    assert coord._final_drain_budget_sec() == 14400.0  # hard cap, never deadlock


# ── terminal full-battery eval ───────────────────────────────────────────────

def _terminal_coord(run_eval_return, *, hard_cap=60.0, enabled=True, run_eval_side_effect=None):
    eval_pipeline = Mock()
    if run_eval_side_effect is not None:
        eval_pipeline.run_evaluation = Mock(side_effect=run_eval_side_effect)
    else:
        eval_pipeline.run_evaluation = Mock(return_value=run_eval_return)
    eval_model = Mock()
    eval_model._orig_mod = eval_model
    eval_model.state_dict = Mock(return_value={})
    events: list = []
    coord = _make_coordinator(
        eval_pipeline=eval_pipeline,
        eval_model=eval_model,
        event_emitter=events.append,
        config_overrides={
            "terminal_eval_enabled": enabled,
            "terminal_eval_hard_cap_sec": hard_cap,
        },
        run_id="r1",
        full_config={"monitors": {}, "encoding": "v6_live2"},
    )
    coord._train_step = 50000
    return coord, eval_pipeline, events


def test_terminal_eval_runs_full_battery_ignoring_stride():
    coord, eval_pipeline, events = _terminal_coord(
        {"promoted": False, "step": 50000, "wr_best": 0.4}
    )
    coord.run_terminal_eval()
    eval_pipeline.run_evaluation.assert_called_once()
    _args, kwargs = eval_pipeline.run_evaluation.call_args
    assert kwargs["ignore_stride"] is True
    # recorded as TERMINAL — a distinct event, never fed to the steering history.
    terminal_events = [e for e in events if e.get("event") == "terminal_eval_complete"]
    assert len(terminal_events) == 1
    assert terminal_events[0]["terminal"] is True
    assert terminal_events[0]["step"] == 50000


def test_terminal_eval_promotes_when_gated():
    coord, _ep, _events = _terminal_coord(
        {"promoted": True, "step": 50000, "wr_best": 0.7}
    )
    with patch("hexo_rl.training.eval_drain.save_best_model_atomic") as mock_save:
        coord.run_terminal_eval()
    # promotion fired: weights synced to self-play + stamped save at the final step.
    coord.pool.sync_inference_weights.assert_called_once()
    _a, kwargs = mock_save.call_args
    assert kwargs["step"] == 50000
    assert kwargs["run_id"] == "r1"


def test_terminal_eval_noop_when_disabled():
    coord, eval_pipeline, _events = _terminal_coord(
        {"promoted": True, "step": 50000}, enabled=False
    )
    coord.run_terminal_eval()
    eval_pipeline.run_evaluation.assert_not_called()


def test_terminal_eval_noop_when_no_pipeline():
    coord = _make_coordinator(eval_pipeline=None)
    coord._train_step = 50000
    coord.run_terminal_eval()  # must not raise


def test_terminal_eval_skipped_on_sigint():
    """On SIGINT (shutdown_save) the close-out drains the in-flight eval but does
    NOT launch a fresh multi-minute terminal eval — the operator interrupted,
    they want a quick clean exit, not a full battery (red-team: kill -INT)."""
    coord, eval_pipeline, _events = _terminal_coord(
        {"promoted": False, "step": 50000}
    )
    coord.shutdown.shutdown_save = True
    coord.run_terminal_eval()
    eval_pipeline.run_evaluation.assert_not_called()


def test_terminal_eval_hard_cap_warns_on_hung_evaluator():
    """The deadlock backstop: a hung evaluator must not block teardown forever."""
    def _hang(*_a, **_kw):
        time.sleep(0.5)
        return {"promoted": True, "step": 50000}
    coord, _ep, _events = _terminal_coord(
        None, hard_cap=0.05, run_eval_side_effect=_hang
    )
    with patch("hexo_rl.training.eval_drain.save_best_model_atomic"):
        coord.run_terminal_eval()  # returns within ~hard_cap, no promotion, no raise
    coord.pool.sync_inference_weights.assert_not_called()


# ── close-out order: drain THEN terminal ─────────────────────────────────────

def test_close_out_drains_then_runs_terminal():
    coord = _make_coordinator()
    order: list[str] = []
    coord.flush_pending_eval = Mock(side_effect=lambda: order.append("flush"))
    coord.run_terminal_eval = Mock(side_effect=lambda: order.append("terminal"))
    coord.close_out()
    assert order == ["flush", "terminal"]
