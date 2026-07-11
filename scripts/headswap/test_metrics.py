"""D-F HEADSWAP — metrics + verdict tests (RECIPE §"TESTS").

Run: .venv/bin/python -m pytest scripts/headswap/test_metrics.py -q

All synthetic — does NOT require the real scores tables. Covers:
  - AUC: perfectly separable -> 1.0; random -> ~0.5 within tolerance; tie handling.
  - paired dAUC + cluster bootstrap: known effect -> CI excludes 0; null -> straddles.
  - canary: ply-separated set -> AUC>0.55; balanced set -> passes (<=0.55).
  - matching: exact 1:1 on band + cap; relaxed nearest-band recovery of deep bands.
  - verdict logic fires each of PASS / PASS-JOINT / TRUNK-FORK on constructed inputs.
  - monotone-transform invariance holds.
  - false-pessimism, ECE, sign accuracy sanity.
"""
import numpy as np
import pytest

from scripts.headswap.metrics import (
    auc,
    canary_auc,
    cluster_bootstrap_delta_auc,
    decide_verdict,
    false_pessimism,
    ece_10bin,
    sign_accuracy,
    greedy_match,
    monotone_invariance_check,
    per_arm_primary_auc,
    paired_delta,
    _paired_arrays,
    _score_map,
    tail_threshold_sweep,
    run_metrics,
)


# ── AUC ───────────────────────────────────────────────────────────────────────


def test_auc_perfectly_separable():
    # positives all score higher than negatives -> AUC 1.0
    scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    labels = [1, 1, 1, 0, 0, 0]
    assert auc(scores, labels) == pytest.approx(1.0)


def test_auc_perfectly_inverted():
    scores = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
    labels = [1, 1, 1, 0, 0, 0]
    assert auc(scores, labels) == pytest.approx(0.0)


def test_auc_random_near_half():
    rng = np.random.default_rng(0)
    n = 4000
    scores = rng.normal(size=n)
    labels = rng.integers(0, 2, size=n)
    assert auc(scores, labels) == pytest.approx(0.5, abs=0.03)


def test_auc_ties_midrank():
    # all identical scores -> AUC exactly 0.5 (all midranks equal)
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 1, 0, 0]
    assert auc(scores, labels) == pytest.approx(0.5)


def test_auc_empty_class_nan():
    assert np.isnan(auc([0.1, 0.2], [1, 1]))


# ── cluster bootstrap paired dAUC ─────────────────────────────────────────────


def _paired_synth(effect, n_games=60, per_game=4, seed=0):
    """Build paired (hi, lo) score arrays clustered by game with a known dAUC.

    hi arm separates classes by `sep_hi`; lo by `sep_lo`; effect = sep_hi-sep_lo
    drives the dAUC sign. Each game contributes per_game rows (mixed labels)."""
    rng = np.random.default_rng(seed)
    labels, games, hi, lo = [], [], [], []
    sep_hi = 1.0 + effect
    sep_lo = 1.0
    for g in range(n_games):
        for k in range(per_game):
            y = k % 2
            games.append(f"g{g}")
            labels.append(y)
            base = rng.normal()
            hi.append(base + (sep_hi if y == 1 else -sep_hi) * 0.5)
            lo.append(base + (sep_lo if y == 1 else -sep_lo) * 0.5)
    return (np.asarray(labels), np.asarray(games, dtype=object),
            np.asarray(hi), np.asarray(lo))


def test_cluster_bootstrap_known_effect_ci_excludes_zero():
    labels, games, hi, lo = _paired_synth(effect=1.5, n_games=80, seed=1)
    res = cluster_bootstrap_delta_auc(labels, games, hi, lo, resamples=2000, seed=1)
    assert res["delta_auc"] > 0.02
    assert res["ci_lo"] > 0.0  # CI excludes 0


def test_cluster_bootstrap_null_straddles_zero():
    labels, games, hi, lo = _paired_synth(effect=0.0, n_games=80, seed=2)
    res = cluster_bootstrap_delta_auc(labels, games, hi, lo, resamples=2000, seed=2)
    assert res["ci_lo"] <= 0.0 <= res["ci_hi"]  # straddles 0


def test_cluster_bootstrap_reports_se_and_r():
    labels, games, hi, lo = _paired_synth(effect=0.5, n_games=60, seed=3)
    res = cluster_bootstrap_delta_auc(labels, games, hi, lo, resamples=1500, seed=3)
    assert res["se"] > 0.0
    assert -1.0 <= res["r"] <= 1.0
    assert res["n_resamples_valid"] > 0


# ── canary ────────────────────────────────────────────────────────────────────


def test_canary_detects_ply_separated_set():
    # positives concentrated at high ply_band, negatives at low -> ply predicts label
    pos = [{"ply_band": b} for b in [8, 9, 10, 11, 8, 9, 10, 11]]
    neg = [{"ply_band": b} for b in [0, 1, 2, 3, 0, 1, 2, 3]]
    feats = np.asarray([[r["ply_band"]] for r in pos + neg], dtype=float)
    labels = [1] * len(pos) + [0] * len(neg)
    assert canary_auc(feats, labels) > 0.55


def test_canary_passes_balanced_set():
    # identical ply distributions -> ply carries no label info -> AUC ~0.5
    bands = [0, 1, 2, 3, 4, 5]
    pos = [{"ply_band": b} for b in bands]
    neg = [{"ply_band": b} for b in bands]
    feats = np.asarray([[r["ply_band"]] for r in pos + neg], dtype=float)
    labels = [1] * len(pos) + [0] * len(neg)
    assert canary_auc(feats, labels) <= 0.55


# ── matching ──────────────────────────────────────────────────────────────────


def _mk(pid, band, gid, v=0.0, tail=None, setl="loss"):
    return {"position_id": pid, "ply_band": band, "source_game_id": gid,
            "v": v, "tail_mass": tail, "set": setl, "t": band * 10 + 1}


def test_exact_match_one_to_one_and_band():
    pos = [_mk(f"p{i}", 1, f"pg{i}") for i in range(3)]
    neg = [_mk(f"n{i}", 1, f"ng{i}", setl="safe") for i in range(3)]
    r = greedy_match(pos, neg, relaxed=False)
    assert r["n_pairs"] == 3
    # every matched neg is in the SAME band as its positive
    for p, n in zip(r["matched_pos"], r["matched_neg"]):
        assert p["ply_band"] == n["ply_band"]


def test_exact_match_no_cross_band_leak():
    pos = [_mk("p0", 5, "pg0")]  # deep positive
    neg = [_mk("n0", 0, "ng0", setl="safe")]  # only a shallow negative
    r = greedy_match(pos, neg, relaxed=False)
    assert r["n_pairs"] == 0
    assert r["drops"]["no_band_neg"] == 1


def test_relaxed_match_recovers_deep_band():
    pos = [_mk("p0", 5, "pg0")]
    neg = [_mk("n0", 0, "ng0", setl="safe")]
    r = greedy_match(pos, neg, relaxed=True)
    assert r["n_pairs"] == 1
    assert r["band_dist"] == [5]  # matched a band-5-distant negative


def test_match_cap_two_negs_per_game():
    # 4 positives band 1, but all negatives come from ONE source game -> cap 2
    pos = [_mk(f"p{i}", 1, f"pg{i}") for i in range(4)]
    neg = [_mk(f"n{i}", 1, "SAME_GAME", setl="safe") for i in range(4)]
    r = greedy_match(pos, neg, relaxed=False, max_neg_per_game=2)
    assert r["n_pairs"] == 2  # only 2 negs usable from the one game
    assert r["drops"]["game_cap"] >= 2


# ── verdict logic ─────────────────────────────────────────────────────────────


def _d(delta, lo, hi):
    return {"delta_auc": delta, "ci_lo": lo, "ci_hi": hi, "se": 0.02, "r": 0.5,
            "n_resamples_valid": 1000}


def test_verdict_pass():
    d_ba = _d(0.08, 0.03, 0.13)   # B>A, CI excludes 0
    d_cd = _d(0.01, -0.04, 0.06)  # irrelevant
    v, _ = decide_verdict(d_ba, d_cd, fp_a=0.10, fp_b=0.12)  # fp_B <= fp_A+5pp
    assert v == "PASS"


def test_verdict_pass_blocked_by_false_pessimism():
    d_ba = _d(0.08, 0.03, 0.13)
    d_cd = _d(0.01, -0.04, 0.06)
    # fp_B exceeds fp_A + 5pp -> PASS blocked -> TRUNK-FORK (C~=D too)
    v, _ = decide_verdict(d_ba, d_cd, fp_a=0.10, fp_b=0.20)
    assert v == "TRUNK-FORK"


def test_verdict_pass_joint():
    d_ba = _d(0.01, -0.03, 0.05)  # B~=A (gate not met)
    d_cd = _d(0.07, 0.02, 0.12)   # C>D, CI excludes 0
    v, _ = decide_verdict(d_ba, d_cd, fp_a=0.10, fp_b=0.11)
    assert v == "PASS-JOINT"


def test_verdict_trunk_fork():
    d_ba = _d(0.00, -0.04, 0.04)
    d_cd = _d(0.01, -0.03, 0.05)
    v, _ = decide_verdict(d_ba, d_cd, fp_a=0.10, fp_b=0.11)
    assert v == "TRUNK-FORK"


def test_verdict_ci_straddle_not_pass():
    # positive point estimate but CI includes 0 -> NOT a pass
    d_ba = _d(0.06, -0.01, 0.13)
    d_cd = _d(0.00, -0.05, 0.05)
    v, _ = decide_verdict(d_ba, d_cd, fp_a=0.10, fp_b=0.11)
    assert v == "TRUNK-FORK"


# ── monotone invariance ───────────────────────────────────────────────────────


def test_monotone_invariance_holds():
    rng = np.random.default_rng(4)
    scores = rng.random(200)
    labels = (rng.random(200) < (scores)).astype(int)  # correlated
    res = monotone_invariance_check(scores, labels)
    assert res["invariant"]
    assert res["delta"] < 1e-9


# ── false-pessimism / ECE / sign accuracy ─────────────────────────────────────


def test_false_pessimism_fraction():
    negs = [-0.9, -0.6, -0.4, 0.1, 0.5]  # 2 of 5 <= -0.5
    assert false_pessimism(negs) == pytest.approx(2 / 5)


def test_ece_perfect_calibration_zero():
    # v maps to p=(v+1)/2; perfectly calibrated -> ECE ~ 0
    # p=1 always win, p=0 always loss
    v = [1.0] * 50 + [-1.0] * 50
    y = [1] * 50 + [0] * 50
    assert ece_10bin(v, y) == pytest.approx(0.0, abs=1e-9)


def test_sign_accuracy_perfect():
    v = [0.8, 0.5, -0.7, -0.3]
    y = [1, 1, 0, 0]
    assert sign_accuracy(v, y) == pytest.approx(1.0)


def test_sign_accuracy_excludes_zeros():
    v = [0.0, 0.5, -0.7]
    y = [1, 1, 0]
    # zero row excluded; remaining 2 both correct
    assert sign_accuracy(v, y) == pytest.approx(1.0)


# ── per-arm AUC on tidy tables ────────────────────────────────────────────────


def test_per_arm_primary_auc_scalar_and_bin():
    loss = [_mk("l0", 1, "g0", v=-0.8, tail=0.9),
            _mk("l1", 1, "g1", v=-0.6, tail=0.8)]
    safe = [_mk("s0", 1, "g2", v=0.5, tail=0.1, setl="safe"),
            _mk("s1", 1, "g3", v=0.7, tail=0.05, setl="safe")]
    table = {"loss": loss, "safe": safe}
    out = per_arm_primary_auc(table)
    # decoded v is PESSIMISM-ORIENTED (-v): loss v=-0.8 -> pess 0.8 > safe pess -0.5
    # -> loss(=positive) scores HIGHER -> AUC by v is 1.0 (a discriminating head).
    assert out["auc_v"] == pytest.approx(1.0)
    # tail-mass HIGHER for loss -> AUC by tail is 1.0
    assert out["auc_tail"] == pytest.approx(1.0)


def test_per_arm_primary_auc_scalar_no_tail():
    loss = [_mk("l0", 1, "g0", v=-0.8)]
    safe = [_mk("s0", 1, "g2", v=0.5, setl="safe")]
    out = per_arm_primary_auc({"loss": loss, "safe": safe})
    assert out["auc_tail"] is None


# ── paired_delta wiring on tidy tables ────────────────────────────────────────


def test_paired_delta_wires_end_to_end():
    # arm B separates by v (loss low), arm A does not -> B-A dAUC > 0
    loss_b = [_mk(f"l{i}", 1, f"g{i}", v=-0.8) for i in range(10)]
    safe_b = [_mk(f"s{i}", 1, f"h{i}", v=0.8, setl="safe") for i in range(10)]
    loss_a = [_mk(f"l{i}", 1, f"g{i}", v=0.0) for i in range(10)]
    safe_a = [_mk(f"s{i}", 1, f"h{i}", v=0.0, setl="safe") for i in range(10)]
    tb = {"loss": loss_b, "safe": safe_b}
    ta = {"loss": loss_a, "safe": safe_a}
    # match B's positives to B's negatives (same universe)
    match = greedy_match(loss_b, safe_b, relaxed=False)
    d = paired_delta(tb, ta, match["matched_pos"], match["matched_neg"], "v", seed=0)
    # PESSIMISM-oriented (-v): arm A constant -> AUC 0.5; arm B loss v=-0.8 -> pess
    # 0.8 > safe pess -0.8 -> loss scores HIGHER -> AUC-by-v of B is 1.0.
    # B-A = 1.0-0.5 = +0.5.
    assert d is not None
    assert d["delta_auc"] == pytest.approx(0.5, abs=1e-6)


# ── threshold sweep availability marker ───────────────────────────────────────


def test_threshold_sweep_unavailable_marker():
    r = tail_threshold_sweep(None, None)
    assert r["available"] is False
    assert "per-bin" in r["note"]


def test_threshold_sweep_with_probs():
    rng = np.random.default_rng(5)
    # positives concentrate mass in low bins (losing tail); negs in high bins
    pos = np.zeros((20, 65)); pos[:, 5] = 1.0
    neg = np.zeros((20, 65)); neg[:, 60] = 1.0
    r = tail_threshold_sweep(pos, neg, bins=range(12, 21))
    assert r["available"]
    # every threshold >= bin 5 captures all pos mass, none of neg -> AUC 1.0
    for row in r["curve"]:
        assert row["auc"] == pytest.approx(1.0)


# ── full run_metrics smoke (4 synthetic arms -> a verdict) ────────────────────


def _arm_table(v_loss, v_safe, tail_loss=None, tail_safe=None, n=30):
    rng = np.random.default_rng(hash((v_loss, v_safe)) % (2**31))
    loss, safe = [], []
    for i in range(n):
        loss.append(_mk(f"l{i}", i % 6, f"g{i % 15}",
                        v=float(rng.normal(v_loss, 0.2)),
                        tail=None if tail_loss is None else float(rng.normal(tail_loss, 0.1))))
        safe.append(_mk(f"s{i}", i % 6, f"h{i % 15}",
                       v=float(rng.normal(v_safe, 0.2)),
                       tail=None if tail_safe is None else float(rng.normal(tail_safe, 0.1)),
                       setl="safe"))
    return {"loss": loss, "safe": safe}


def test_run_metrics_full_pipeline_produces_verdict():
    # A: no separation; B: strong separation (loss v very low) -> expect PASS-ish signal
    arms = {
        "A": [_arm_table(0.0, 0.0)],
        "B": [_arm_table(-0.9, 0.9, tail_loss=0.9, tail_safe=0.05)],
        "C": [_arm_table(-0.9, 0.9, tail_loss=0.9, tail_safe=0.05)],
        "D": [_arm_table(0.0, 0.0)],
    }
    m = run_metrics(arms, seed=0)
    assert m["verdict"] in ("PASS", "PASS-JOINT", "TRUNK-FORK")
    # B AUC-by-v: loss has LOWER v -> AUC 0.0; A AUC 0.5. So B-A negative -> NOT pass.
    # But tail-mass (mechanism) separates. Just assert structure is complete.
    assert "primary_auc" in m and "delta_auc" in m
    assert m["primary_auc"]["B"]["auc_tail"] is not None
    assert m["holdout"] == {"note": "N/A (holdout not supplied)"}


def test_run_metrics_pass_when_bin_arm_scores_loss_low():
    # A genuine PASS: arm B is a discriminating pessimistic head -> scores LOSSES
    # pessimistic (v low, pess high) and SAFES optimistic (v high, pess low, so
    # false-pessimism_B ~ 0). Arm A is uninformative (v~0). Pessimism-oriented AUC:
    # B separates (AUC ~1), A ~0.5 -> B-A dAUC > 0.05, CI excl 0, fp_B <= fp_A+5pp.
    arms = {
        "A": [_arm_table(0.0, 0.0)],
        "B": [_arm_table(-0.9, 0.9)],   # loss v low (pess), safe v high (not doom)
        "C": [_arm_table(0.0, 0.0)],
        "D": [_arm_table(0.0, 0.0)],
    }
    m = run_metrics(arms, seed=0)
    d = m["delta_auc"]["B_minus_A_v"]
    assert d["delta_auc"] > 0.05
    assert d["ci_lo"] > 0.0
    # false-pessimism_B ~ 0 (safes scored optimistic) <= fp_A + 5pp -> PASS
    assert m["false_pessimism"]["B"] <= m["false_pessimism"]["A"] + 0.05
    assert m["verdict"] == "PASS"
