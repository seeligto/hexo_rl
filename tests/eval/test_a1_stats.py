"""D-SOLVER A1 — paired-bootstrap + soundness-counting unit tests."""
from __future__ import annotations

from hexo_rl.eval.a1_stats import cand_outcome, paired_delta, soundness_violations


def _g(seed, cand_label, result, *, opp="sealbot", fired_win=0, cand_won=None):
    """Build a game record where `cand_label` is p1 vs `opp`. result in {win,loss,draw}."""
    winner = {"win": "p1", "loss": "p2", "draw": "draw"}[result]
    g = {"seed": seed, "p1": cand_label, "p2": opp, "winner": winner, "fired_win": fired_win}
    if cand_won is None:
        cand_won = result == "win"
    g["cand_won"] = cand_won
    return g


def test_cand_outcome_win_draw_loss():
    assert cand_outcome(_g(0, "x", "win"), "x") == 1.0
    assert cand_outcome(_g(0, "x", "draw"), "x") == 0.5
    assert cand_outcome(_g(0, "x", "loss"), "x") == 0.0
    # label as p2
    g = {"p1": "sealbot", "p2": "x", "winner": "p2"}
    assert cand_outcome(g, "x") == 1.0


def test_paired_delta_clean_lift():
    # 20 shared seeds: 15 fired games backup wins / baseline loses (+1), 5 identical (0).
    base = [_g(s, "baseline", "loss") for s in range(15)] + [_g(s, "baseline", "win") for s in range(15, 20)]
    back = [_g(s, "backup", "win") for s in range(15)] + [_g(s, "backup", "win") for s in range(15, 20)]
    r = paired_delta(base, back, "baseline", "backup", n_boot=2000, seed=1)
    assert r["n_paired"] == 20
    assert r["n_fired"] == 15
    assert abs(r["delta"] - 0.75) < 1e-9
    assert r["ci_lo"] > 0.0, "a consistent 15/20 lift must be CI-clean"


def test_paired_delta_flat_when_balanced():
    # equal wins and losses from the override -> delta ~ 0, CI straddles 0.
    base = [_g(s, "baseline", "loss") for s in range(10)] + [_g(s, "baseline", "win") for s in range(10, 20)]
    back = [_g(s, "backup", "win") for s in range(10)] + [_g(s, "backup", "loss") for s in range(10, 20)]
    r = paired_delta(base, back, "baseline", "backup", n_boot=2000, seed=1)
    assert r["n_fired"] == 20
    assert abs(r["delta"]) < 1e-9
    assert r["ci_lo"] < 0.0 < r["ci_hi"]


def test_paired_delta_empty_intersection():
    base = [_g(0, "baseline", "win")]
    back = [_g(99, "backup", "win")]
    r = paired_delta(base, back, "baseline", "backup", n_boot=100, seed=1)
    assert r == {"delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_gt_0": 0.0, "n_paired": 0, "n_fired": 0}


def test_soundness_violations_counts_loss_and_draw_false_proofs():
    games = [
        _g(1, "backup", "win", fired_win=2),    # fired + won -> OK
        _g(2, "backup", "loss", fired_win=1),   # fired + lost -> VIOLATION
        _g(3, "backup", "draw", fired_win=1),   # fired + drew -> VIOLATION (forced mate can't draw)
        _g(4, "backup", "loss", fired_win=0),   # lost but never fired -> OK (an unsaveable loss)
    ]
    viol = soundness_violations(games, "backup")
    assert sorted(viol) == [2, 3]
