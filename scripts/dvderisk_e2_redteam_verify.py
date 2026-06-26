#!/usr/bin/env python3
"""D-FULLSPEC E2-REPROBE RED-TEAM independent verification.

Distinct-adversary checks on the E2 input-feature ablation (4-plane CONTROL vs
8-plane THREAT TREATMENT) that returned ENTANGLED_R. Read-only on npz banks.

Checks:
  1. THREAT-PLANE CORRECTNESS: an INDEPENDENT brute-force threat-plane
     implementation (different code path) vs the reprobe's
     compute_threat_planes_single/_batch, on hand-checkable positions AND on a
     random sample of the real pool. Also hand-derive the expected values.
  2. GAME-DISJOINT: independently recompute train/val/holdout game-id overlap.
  3. MATCHED-STRATUM SIGNAL (false-negative guard): do the 4 threat features
     carry ANY held-out win/loss separation on the turn-phase-matched stratum
     that the from-scratch net failed to extract? A trivial game-disjoint
     logistic on pooled threat-feature summaries. If it ALSO craters, ENTANGLED_R
     is not a net-capacity false negative.
  4. NEAR-DUP LEAKAGE: min Hamming distance (stone planes 0,1) from each matched
     holdout position to any train position (game-disjoint can still leak
     transpositions). Near-dup would INFLATE holdout -> conservative for an
     ENTANGLED verdict.
  5. CONTROL prior-collapse diagnostic: matched stratum class balance.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dvderisk_lighttrunk_probe import build_pool, three_way_game_split, matched_masks
from scripts.dvderisk_e2_reprobe import (
    compute_threat_planes_single, compute_threat_planes_batch,
)

WIN_LEN = 6
HEX_AXES = [(1, 0), (0, 1), (1, -1)]


# --- INDEPENDENT brute-force threat-plane impl (distinct code path) ----------
def independent_threat_planes(cur: np.ndarray, opp: np.ndarray) -> np.ndarray:
    """Re-derive the 4 planes by enumerating EVERY length-6 window via explicit
    bounds per cell-membership, not the reprobe's range() slicing. Same semantics:
      [0] cur_open_line_count (#unblocked windows w/ >=1 cur, 0 opp) /6
      [1] opp_open_line_count                                        /6
      [2] cur_best_window_fill (max cur over unblocked windows)      /6
      [3] opp_best_window_fill                                       /6
    """
    H, W = cur.shape
    colc = np.zeros((H, W), np.float64)
    oolc = np.zeros((H, W), np.float64)
    cbwf = np.zeros((H, W), np.float64)
    obwf = np.zeros((H, W), np.float64)
    for dq, dr in HEX_AXES:
        for sq in range(H):
            for sr in range(W):
                # window cells (sq+t*dq, sr+t*dr) t=0..5
                eq = sq + (WIN_LEN - 1) * dq
                er = sr + (WIN_LEN - 1) * dr
                if not (0 <= eq < H and 0 <= er < W):
                    continue
                cc = 0
                oc = 0
                cells = []
                for t in range(WIN_LEN):
                    q = sq + t * dq
                    r = sr + t * dr
                    cells.append((q, r))
                    if cur[q, r] > 0.5:
                        cc += 1
                    if opp[q, r] > 0.5:
                        oc += 1
                for (q, r) in cells:
                    if oc == 0 and cc >= 1:
                        colc[q, r] += 1
                    if oc == 0 and cc > cbwf[q, r]:
                        cbwf[q, r] = cc
                    if cc == 0 and oc >= 1:
                        oolc[q, r] += 1
                    if cc == 0 and oc > obwf[q, r]:
                        obwf[q, r] = oc
    return np.stack([colc, oolc, cbwf, obwf], 0).astype(np.float32) / float(WIN_LEN)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - s), min(1.0, c + s))


def main():
    t0 = time.time()
    out = {"ran": True, "checks": {}}

    # ---- 1. THREAT-PLANE CORRECTNESS -------------------------------------
    tp = {}
    # 1a empty board
    empty = np.zeros((19, 19), np.float32)
    a = compute_threat_planes_single(empty, empty)
    b = independent_threat_planes(empty, empty)
    tp["empty_all_zero_reprobe"] = bool(np.allclose(a, 0))
    tp["empty_all_zero_indep"] = bool(np.allclose(b, 0))
    # 1b single interior stone -> open_line_count cell = 18 windows/6 = 3.0 (6 per axis x3)
    cur = np.zeros((19, 19), np.float32); cur[9, 9] = 1.0
    a = compute_threat_planes_single(cur, empty)
    b = independent_threat_planes(cur, empty)
    tp["single_stone_olc_reprobe"] = round(float(a[0, 9, 9]), 6)
    tp["single_stone_olc_indep"] = round(float(b[0, 9, 9]), 6)
    tp["single_stone_olc_expected_3.0"] = bool(abs(float(b[0, 9, 9]) - 3.0) < 1e-5)
    tp["single_stone_bwf_reprobe"] = round(float(a[2, 9, 9]), 6)   # best fill = 1/6
    tp["single_stone_bwf_expected_1_6"] = bool(abs(float(b[2, 9, 9]) - 1.0 / 6) < 1e-5)
    # 1c 5-in-a-row completion cell best_fill = 5/6
    cur = np.zeros((19, 19), np.float32)
    for r in range(9, 14):
        cur[r, 9] = 1.0
    a = compute_threat_planes_single(cur, empty)
    b = independent_threat_planes(cur, empty)
    tp["five_row_completion_bwf_reprobe"] = round(float(a[2, 14, 9]), 6)
    tp["five_row_completion_bwf_indep"] = round(float(b[2, 14, 9]), 6)
    tp["five_row_completion_expected_5_6"] = bool(abs(float(b[2, 14, 9]) - 5.0 / 6) < 1e-5)
    # 1d opp-blocked window: a 5-cur line with an opp stone in its completion window
    cur2 = np.zeros((19, 19), np.float32); opp2 = np.zeros((19, 19), np.float32)
    for r in range(9, 14):
        cur2[r, 9] = 1.0
    opp2[14, 9] = 1.0  # block the (9..14) completion window
    a = compute_threat_planes_single(cur2, opp2)
    b = independent_threat_planes(cur2, opp2)
    # window (9..14) now has opp -> blocked; best cur window containing (8,9)?
    # cur_bwf at (8,9): windows (3..8)=0 cur ... (8..13) contains cur 9..13=5 but cell 8 empty cur, opp? none -> 5
    tp["blocked_cur_bwf_at_8_9_reprobe"] = round(float(a[2, 8, 9]), 6)
    tp["blocked_cur_bwf_at_8_9_indep"] = round(float(b[2, 8, 9]), 6)
    # opp_best_window_fill at (13,9): window (13..18) has opp@14 + cur@13? cur present -> not opp-open.
    # cell (14,9) is the opp stone; opp_bwf there: windows w/ 0 cur and >=1 opp.
    tp["opp_bwf_at_14_9_reprobe"] = round(float(a[3, 14, 9]), 6)
    tp["opp_bwf_at_14_9_indep"] = round(float(b[3, 14, 9]), 6)

    out["checks"]["threat_plane_hand"] = tp

    # ---- 1e full-array agreement on random REAL pool sample ----
    pool = build_pool()
    lab = pool["label"]; states4 = pool["states"]
    rng = np.random.default_rng(7)
    samp = rng.choice(len(states4), size=40, replace=False)
    max_abs = 0.0
    n_mismatch = 0
    for i in samp:
        a = compute_threat_planes_single(states4[i, 0], states4[i, 1])
        b = independent_threat_planes(states4[i, 0], states4[i, 1])
        d = float(np.abs(a - b).max())
        max_abs = max(max_abs, d)
        if d > 1e-5:
            n_mismatch += 1
    out["checks"]["threat_plane_pool_sample"] = {
        "n_sampled": int(len(samp)), "max_abs_diff": max_abs,
        "n_mismatch": int(n_mismatch),
        "reprobe_matches_independent": bool(max_abs < 1e-5),
    }

    # batch == single on the same sample
    batch = compute_threat_planes_batch(states4[samp])
    single = np.stack([compute_threat_planes_single(states4[i, 0], states4[i, 1]) for i in samp])
    out["checks"]["batch_equals_single_sample"] = bool(np.allclose(batch, single, atol=1e-6))

    # ---- 2. GAME-DISJOINT independent recompute ----
    gd = []
    for seed in (101, 202, 303):
        tr, val, ho, ng, ntg, nvg, nhg = three_way_game_split(pool["gid"], 0.25, 0.18, seed)
        g_tr = set(pool["gid"][tr].tolist()); g_val = set(pool["gid"][val].tolist()); g_ho = set(pool["gid"][ho].tolist())
        shared = len(g_tr & g_ho) + len(g_tr & g_val) + len(g_val & g_ho)
        gd.append({"seed": seed, "games_tr_val_ho": [ntg, nvg, nhg], "shared": shared,
                   "ho_lw": [int(((lab < 0) & ho).sum()), int(((lab > 0) & ho).sum())]})
    out["checks"]["game_disjoint"] = {"per_seed": gd, "all_disjoint": all(g["shared"] == 0 for g in gd)}

    # ---- 5. matched-stratum class balance (control prior-collapse diagnostic) ----
    mm_full = matched_masks(pool, np.ones(len(lab), bool))
    lm, wm, lo, hi = mm_full
    out["checks"]["matched_stratum_full"] = {
        "n_loss": int(lm.sum()), "n_win": int(wm.sum()),
        "loss_frac": round(float(lm.sum()) / float(lm.sum() + wm.sum()), 4),
        "stone_support": [lo, hi],
        "note": "loss-heavy => an always-predict-loss net gets KILL-A~1 KILL-C~0 (control prior-collapse)",
    }

    # ---- 3. MATCHED-STRATUM THREAT SIGNAL (false-negative guard) ----
    # 8 pooled summary stats per position from the 4 threat planes: max & mean of each.
    threat = compute_threat_planes_batch(states4)   # (N,4,19,19)
    flat = threat.reshape(len(threat), 4, -1)
    feat = np.concatenate([flat.max(axis=2), flat.mean(axis=2)], axis=1)  # (N,8)

    # raw matched-stratum mean per threat plane (idx3 = opp_best_window_fill)
    obwf_max = flat[:, 3].max(axis=1)
    out["checks"]["matched_signal_raw"] = {
        "opp_bwf_max_loss_mean": round(float(obwf_max[lm].mean()), 4),
        "opp_bwf_max_win_mean": round(float(obwf_max[wm].mean()), 4),
        "cur_bwf_max_loss_mean": round(float(flat[:, 2].max(axis=1)[lm].mean()), 4),
        "cur_bwf_max_win_mean": round(float(flat[:, 2].max(axis=1)[wm].mean()), 4),
    }

    # game-disjoint logistic on 8 threat summaries -> held-out matched win/loss AUC + KILL-C
    def fit_logistic(X, y, iters=4000, lr=0.2, l2=1e-3):
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
        w = np.zeros(Xb.shape[1])
        for _ in range(iters):
            z = Xb @ w
            p = 1.0 / (1.0 + np.exp(-z))
            g = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
            w -= lr * g
        return w, X.mean(0), X.std(0)

    def predict(w, mu, sd, X):
        Xn = (X - mu) / (sd + 1e-8)
        Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
        return 1.0 / (1.0 + np.exp(-(Xb @ w)))

    def auc(scores, ytrue):
        pos = scores[ytrue == 1]; neg = scores[ytrue == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        # rank-based AUC
        allv = np.concatenate([pos, neg])
        order = allv.argsort()
        ranks = np.empty_like(order, float)
        ranks[order] = np.arange(1, len(allv) + 1)
        r_pos = ranks[:len(pos)].sum()
        return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    log_seeds = []
    for seed in (101, 202, 303):
        tr, val, ho, *_ = three_way_game_split(pool["gid"], 0.25, 0.18, seed)
        # build matched train + matched holdout
        mtr = matched_masks(pool, tr); mho = matched_masks(pool, ho)
        if mtr is None or mho is None:
            continue
        ltr, wtr, _, _ = mtr; lho, who, _, _ = mho
        Xtr = np.concatenate([feat[ltr], feat[wtr]])
        ytr = np.concatenate([np.zeros(int(ltr.sum())), np.ones(int(wtr.sum()))])
        w, mu, sd = fit_logistic(Xtr, ytr)
        Xho = np.concatenate([feat[lho], feat[who]])
        yho = np.concatenate([np.zeros(int(lho.sum())), np.ones(int(who.sum()))])
        s_ho = predict(w, mu, sd, Xho)
        a_ho = auc(s_ho, yho)
        # KILL-C analog at decision thr 0.5: frac wins scored >0.5; KILL-A frac losses <0.5
        kc = float((s_ho[yho == 1] > 0.5).mean()) if (yho == 1).any() else float("nan")
        ka = float((s_ho[yho == 0] < 0.5).mean()) if (yho == 0).any() else float("nan")
        # train-fit AUC (overfit canary on the logistic itself)
        s_tr = predict(w, mu, sd, Xtr)
        a_tr = auc(s_tr, ytr)
        log_seeds.append({"seed": seed, "ho_auc": round(a_ho, 4), "tr_auc": round(a_tr, 4),
                          "ho_kill_a": round(ka, 4), "ho_kill_c": round(kc, 4),
                          "n_lw": [int(lho.sum()), int(who.sum())]})
    out["checks"]["matched_threat_logistic"] = {
        "per_seed": log_seeds,
        "mean_ho_auc": round(float(np.mean([s["ho_auc"] for s in log_seeds])), 4) if log_seeds else None,
        "mean_ho_kill_c": round(float(np.mean([s["ho_kill_c"] for s in log_seeds])), 4) if log_seeds else None,
        "note": "ho_auc~0.5 and ho_kill_c<<0.85 => threat features lack held-out matched signal => ENTANGLED_R not a net-capacity false-negative",
    }

    # ---- 4. NEAR-DUP LEAKAGE (seed101) ----
    tr, val, ho, *_ = three_way_game_split(pool["gid"], 0.25, 0.18, 101)
    mho = matched_masks(pool, ho); mtr = matched_masks(pool, tr)
    lho, who, _, _ = mho
    ho_mask = lho | who
    stones = (states4[:, 0] + states4[:, 1]).reshape(len(states4), -1)  # (N, 361) binary-ish
    # signed occupancy to also catch color flips: plane0 - plane1
    occ = (states4[:, 0] - states4[:, 1]).reshape(len(states4), -1)
    tr_occ = occ[tr]
    ho_idx = np.where(ho_mask)[0]
    mind = []
    for i in ho_idx:
        d = np.abs(tr_occ - occ[i]).sum(axis=1)  # L1 on signed occ = #differing cells (x1) + color flips (x2)
        mind.append(int(d.min()))
    mind = np.array(mind)
    out["checks"]["near_dup"] = {
        "seed": 101, "n_matched_holdout": int(len(ho_idx)),
        "min_hamming_quantiles": {q: int(np.quantile(mind, q)) for q in (0.0, 0.05, 0.25, 0.5)},
        "frac_within_2_cells": round(float((mind <= 2).mean()), 4),
        "frac_within_6_cells": round(float((mind <= 6).mean()), 4),
        "frac_exact_dup": round(float((mind == 0).mean()), 4),
        "note": "game-disjoint; near-dup would INFLATE holdout (conservative for ENTANGLED). exact_dup>0 would break disjointness claim at the POSITION level",
    }

    out["wall_time_s"] = round(time.time() - t0, 1)
    out_dir = REPO_ROOT / "reports" / "d_fullspec_2026-06-26"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "E2_REDTEAM_VERIFY_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("E2_REDTEAM_VERIFY_JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
