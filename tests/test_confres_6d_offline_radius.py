"""CONFRES 6d — offline radius resolution HARD-ERROR (B6, design §4/§8).

``evalfair.core.radius_from_checkpoint`` delegates to the ONE schedule-scan authority; ``run_arm``
HARD-ERRORS when a per-stage book meets an unresolvable ckpt radius (was: silent skip of the guard,
biasing Series B). Also pins the named grep-gate seams (run_sealbot_eval --legal-radius inert).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.evalfair.core import radius_from_checkpoint


def test_radius_from_checkpoint_delegates_to_schedule_scan():
    sched = [{"step": 0, "radius": 4}, {"step": 200_000, "radius": 5}]
    ck_r4 = {"config": {"selfplay": {"legal_move_radius_schedule": sched}}, "step": 100_000}
    ck_r5 = {"config": {"selfplay": {"legal_move_radius_schedule": sched}}, "step": 300_000}
    assert radius_from_checkpoint(ck_r4) == 4
    assert radius_from_checkpoint(ck_r5) == 5


def test_radius_from_checkpoint_none_when_no_schedule():
    # A weights-only-stripped ckpt with no baked schedule → None (the offline HARD-ERROR is applied
    # at the run_arm consumer, not here).
    ck = {"config": {"selfplay": {}}, "step": 100_000}
    assert radius_from_checkpoint(ck) is None
    assert radius_from_checkpoint({"config": {}, "step": 5}) is None


def test_run_arm_hard_errors_on_unresolvable_radius_with_staged_book(monkeypatch):
    """A per-stage book + an unresolvable (None) ckpt radius → hard-error, not a silent skip."""
    import scripts.evalfair.core as core

    # Stub the ckpt load so we don't need a real .pt: no baked schedule → radius None. Deploy
    # knobs present so extract_deploy_knobs (which runs before the radius guard) doesn't short it.
    _stub_cfg = {
        "selfplay": {"gumbel_m": 16, "c_visit": 50, "c_scale": 1.0,
                     "playout_cap": {"n_sims_full": 150}},
        "mcts": {"c_puct": 1.5},
    }
    monkeypatch.setattr(
        core.torch, "load",
        lambda *a, **k: {"config": _stub_cfg, "step": 100_000},
    )
    monkeypatch.setattr(core, "_ckpt_sha", lambda p: "deadbeef")
    monkeypatch.setattr(core, "sealbot_depth_from_config", lambda: 5)

    book = {"book_id": "evalfair_r4_v2", "radius_stage": 4, "openings": []}
    arm = core.ArmSpec(label="head")
    from hexo_rl.config.resolve.radius import OfflineRadiusUnresolvableError
    # run_arm delegates to the ONE authority resolve.radius.require_offline_radius (design law #1).
    with pytest.raises(OfflineRadiusUnresolvableError) as ei:
        core.run_arm(
            "fake.pt", arm, book, out_dir="/tmp/confres_6d_test",
            book_seed=1, expect_encoding="v6_live2_ls",
        )
    msg = str(ei.value)
    assert "fake.pt" in msg and "--radius-stage" in msg


def test_run_sealbot_legal_radius_flag_is_inert_documented():
    """§173 A6 grep-gate seam: --legal-radius is IGNORED (registry-sourced), not a silent default."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_sealbot_eval.py").read_text()
    assert "--legal-radius is ignored" in src
