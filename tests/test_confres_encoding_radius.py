"""CONFRES 6c/6d — resolve.encoding + resolve.radius authority modules.

Covers the shared raise/normalize encoding rule (delegated to by the launch builder AND the eval
checkpoint loader) and the radius schedule-scan + offline HARD-ERROR gate (B6).
"""
from __future__ import annotations

import pytest

from hexo_rl.config.resolve.encoding import (
    UNSPECIFIED,
    EncodingConflictError,
    EncodingResolution,
    normalize_declared,
    normalize_stamp,
    reconcile_declared_vs_stamp,
    resolve_encoding,
    window_set,
)
from hexo_rl.config.resolve.radius import (
    OfflineRadiusUnresolvableError,
    require_offline_radius,
    resolve_eval_radius,
    resolve_radius_from_schedule,
)


# ── encoding: reconcile_declared_vs_stamp (the shared rule) ───────────────────
def test_declared_and_stamp_agree_declared_wins():
    r = reconcile_declared_vs_stamp("v6_live2_ls", "v6_live2_ls")
    assert r == EncodingResolution("v6_live2_ls", "variant", "v6_live2_ls", "v6_live2_ls")


def test_declared_and_stamp_conflict_raises_naming_both():
    with pytest.raises(EncodingConflictError) as ei:
        reconcile_declared_vs_stamp("v6_live2_ls", "v6_live2")
    assert "v6_live2_ls" in str(ei.value) and "v6_live2" in str(ei.value)
    assert ei.value.declared == "v6_live2_ls" and ei.value.stamp == "v6_live2"


def test_declared_present_no_stamp_declared_wins():
    r = reconcile_declared_vs_stamp("v6w25", None)
    assert r.name == "v6w25" and r.source == "variant"


def test_absent_declared_with_stamp_resolves_to_stamp_no_raise():
    # B5a: no operator declaration + a stamp → stamp wins (metadata-wins), source=checkpoint.
    r = reconcile_declared_vs_stamp(UNSPECIFIED, "v6_live2_ls")
    assert r.name == "v6_live2_ls" and r.source == "checkpoint"


def test_absent_declared_no_stamp_defaults_v6():
    r = reconcile_declared_vs_stamp(UNSPECIFIED, None)
    assert r.name == "v6" and r.source == "default"


def test_normalize_declared_presence_before_normalize():
    # Absent key → UNSPECIFIED (NOT "v6"), so it can't I2-raise against a non-v6 stamp.
    assert normalize_declared(False, None) is UNSPECIFIED
    # Present None (explicit) still normalizes to "v6" — presence is the operator's declaration.
    assert normalize_declared(True, None) == "v6"
    assert normalize_declared(True, {"version": "v6_live2_ls"}) == "v6_live2_ls"


def test_normalize_stamp_dict_and_absent():
    assert normalize_stamp({}) is None
    assert normalize_stamp({"encoding": {"version": "v6tp"}}) == "v6tp"
    assert normalize_stamp({"encoding": "v6"}) == "v6"


def test_resolve_encoding_string_equals_dict_form():
    # M3: string-form ≡ dict-form (F1 dead by construction).
    a = resolve_encoding(True, "v6_live2_ls", {})
    b = resolve_encoding(True, {"version": "v6_live2_ls"}, {})
    assert a.name == b.name == "v6_live2_ls"


def test_window_set_registry_lookup():
    spec = window_set("v6")
    assert spec.name == "v6"
    # dict form resolves too
    assert window_set({"version": "v6"}).name == "v6"


# ── radius: schedule scan + offline hard-error ────────────────────────────────
_SCHED = [{"step": 0, "radius": 4}, {"step": 200_000, "radius": 5}, {"step": 400_000, "radius": 6}]


def test_schedule_scan_picks_last_entry_at_or_below_step():
    assert resolve_radius_from_schedule(_SCHED, 0) == 4
    assert resolve_radius_from_schedule(_SCHED, 199_999) == 4
    assert resolve_radius_from_schedule(_SCHED, 200_000) == 5
    assert resolve_radius_from_schedule(_SCHED, 500_000) == 6


def test_schedule_scan_none_when_no_schedule():
    assert resolve_radius_from_schedule(None, 100) is None
    assert resolve_radius_from_schedule([], 100) is None


def test_resolve_eval_radius_override_and_track():
    # override pins; None tracks the curriculum radius (delegates to eval_board).
    assert resolve_eval_radius(5, override=8) == 8
    assert resolve_eval_radius(5, override=None) == 5
    assert resolve_eval_radius(None, override=None) is None


def test_require_offline_radius_override_wins():
    assert require_offline_radius(None, 4) == 4
    assert require_offline_radius(5, 4) == 4  # explicit override beats resolved


def test_require_offline_radius_resolved_wins_when_no_override():
    assert require_offline_radius(5, None) == 5


def test_require_offline_radius_unresolvable_hard_errors():
    with pytest.raises(OfflineRadiusUnresolvableError) as ei:
        require_offline_radius(None, None, ckpt_label="strip.pt")
    assert "strip.pt" in str(ei.value) and "--radius-stage" in str(ei.value)
