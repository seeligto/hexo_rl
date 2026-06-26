#!/usr/bin/env python3
"""D-FULLSPEC LIGHT-TRUNK RED-TEAM — independent CPU-only audit of the ENTANGLED_LT verdict.

Distinct adversary. Verifies, WITHOUT trusting the probe's asserts:
 1. game-disjoint split: recompute game_id overlap (tr/val/ho) for ALL 3 seeds, both
    pairwise game_id AND byte-identical board overlap.
 2. near-dup leakage: min Hamming on stone planes (planes 0,1) holdout-matched vs train.
 3. turn-phase: plane2==plane3 everywhere; within each seed's tp==0 ∩ stone-support
    MATCHED set, are planes 2,3 a single constant for BOTH wins and losses?
 4. overfit attribution: per-LR train-fit vs holdout KILL-C gap (from probe JSON) — does
    ANY setting produce a holdout separation, or do all settings crater regardless of
    how hard they fit train? (overfit cannot be 'the separation' if no holdout sep exists.)

Read-only on npz banks (np.load only). No model forward, no training. CPU.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# reuse the EXACT probe build_pool + split so we audit the same objects
from scripts.dvderisk_lighttrunk_probe import build_pool, three_way_game_split, matched_masks

HOLDOUT_FRAC = 0.25
VAL_FRAC = 0.18
SEEDS = [101, 202, 303]


def main():
    out = {}
    pool = build_pool()
    lab = pool["label"]; gid = pool["gid"]; tp = pool["tp"]; stones = pool["stones"]
    states = pool["states"]
    n = len(lab)
    print(f"[POOL] n={n} loss={int((lab<0).sum())} win={int((lab>0).sum())} "
          f"distinct_games={len(np.unique(gid))} p23_eq={pool['p23_eq']}")
    out["pool"] = {"n": int(n), "n_loss": int((lab < 0).sum()), "n_win": int((lab > 0).sum()),
                   "distinct_games": int(len(np.unique(gid))), "plane2_eq_plane3": bool(pool["p23_eq"])}

    # ---- flatten stone planes (0,1) once for byte + Hamming checks ----
    stone_planes = (states[:, :2] > 0.5).astype(np.uint8).reshape(n, -1)  # binarized occupancy
    board_sig = [stone_planes[i].tobytes() for i in range(n)]

    # =====================================================================
    # 1. GAME-DISJOINT across all 3 seeds + byte-board overlap
    # =====================================================================
    out["split_audit"] = []
    worst_board_overlap = 0
    for seed in SEEDS:
        tr, val, ho, n_g, n_tr_g, n_val_g, n_ho_g = three_way_game_split(gid, HOLDOUT_FRAC, VAL_FRAC, seed)
        g_tr = set(gid[tr].tolist()); g_val = set(gid[val].tolist()); g_ho = set(gid[ho].tolist())
        sh_tv = len(g_tr & g_val); sh_th = len(g_tr & g_ho); sh_vh = len(g_val & g_ho)
        # byte-identical board across tr vs ho (stronger than game_id — catches transpositions)
        tr_sigs = {board_sig[i] for i in np.where(tr)[0]}
        ho_idx = np.where(ho)[0]
        board_dups = sum(1 for i in ho_idx if board_sig[i] in tr_sigs)
        worst_board_overlap = max(worst_board_overlap, board_dups)
        # coverage: every position assigned to exactly one of tr/val/ho?
        assigned = tr.astype(int) + val.astype(int) + ho.astype(int)
        full_cover = bool((assigned == 1).all())
        rec = {"seed": seed, "n_games": int(n_g), "n_tr_g": int(n_tr_g), "n_val_g": int(n_val_g),
               "n_ho_g": int(n_ho_g), "shared_tr_val": sh_tv, "shared_tr_ho": sh_th,
               "shared_val_ho": sh_vh, "board_byte_dups_ho_in_tr": int(board_dups),
               "full_partition": full_cover,
               "n_ho_loss": int(((lab < 0) & ho).sum()), "n_ho_win": int(((lab > 0) & ho).sum())}
        out["split_audit"].append(rec)
        print(f"[SPLIT seed={seed}] games={n_g} tr/val/ho={n_tr_g}/{n_val_g}/{n_ho_g} | "
              f"shared tr-val={sh_tv} tr-ho={sh_th} val-ho={sh_vh} | "
              f"byte-board dups ho-in-tr={board_dups} | full_partition={full_cover}")
    out["game_disjoint_all_seeds"] = all(
        r["shared_tr_val"] == 0 and r["shared_tr_ho"] == 0 and r["shared_val_ho"] == 0
        for r in out["split_audit"])
    out["zero_board_byte_overlap_all_seeds"] = (worst_board_overlap == 0)

    # =====================================================================
    # 2. NEAR-DUP LEAKAGE — min Hamming (stone planes) holdout-MATCHED vs train-MATCHED
    #    (matched set = the verdict basis). Per seed.
    # =====================================================================
    out["near_dup"] = []
    for seed in SEEDS:
        tr, val, ho, *_ = three_way_game_split(gid, HOLDOUT_FRAC, VAL_FRAC, seed)
        mh = matched_masks(pool, ho)
        mt = matched_masks(pool, tr)
        if mh is None or mt is None:
            out["near_dup"].append({"seed": seed, "error": "empty matched set"})
            continue
        ho_lm, ho_wm, _, _ = mh
        tr_lm, tr_wm, _, _ = mt
        tr_all = np.where(tr_lm | tr_wm)[0]
        tr_mat = stone_planes[tr_all].astype(np.int16)  # (Ntr, F)

        def min_hamming(ho_mask):
            ho_idx = np.where(ho_mask)[0]
            mins = []
            for i in ho_idx:
                d = np.abs(tr_mat - stone_planes[i].astype(np.int16)).sum(1)
                mins.append(int(d.min()))
            return mins

        win_mins = min_hamming(ho_wm)
        loss_mins = min_hamming(ho_lm)
        le2_win = sum(1 for d in win_mins if d <= 2)
        le2_loss = sum(1 for d in loss_mins if d <= 2)
        rec = {"seed": seed,
               "win_min_hamming": int(min(win_mins)) if win_mins else None,
               "win_med_hamming": int(np.median(win_mins)) if win_mins else None,
               "loss_min_hamming": int(min(loss_mins)) if loss_mins else None,
               "loss_med_hamming": int(np.median(loss_mins)) if loss_mins else None,
               "n_win_le2": le2_win, "n_loss_le2": le2_loss,
               "n_ho_win_matched": len(win_mins), "n_ho_loss_matched": len(loss_mins)}
        out["near_dup"].append(rec)
        print(f"[NEAR-DUP seed={seed}] win minH={rec['win_min_hamming']} medH={rec['win_med_hamming']} | "
              f"loss minH={rec['loss_min_hamming']} medH={rec['loss_med_hamming']} | "
              f"<=2: win={le2_win} loss={le2_loss}")
    out["any_near_dup_le2"] = any(
        (r.get("n_win_le2", 0) + r.get("n_loss_le2", 0)) > 0 for r in out["near_dup"])

    # =====================================================================
    # 3. TURN-PHASE — plane2==plane3 everywhere; within matched set, planes 2,3 a single
    #    constant for BOTH wins and losses? (true matching => no turn-phase leak)
    # =====================================================================
    out["turnphase"] = []
    for seed in SEEDS:
        tr, val, ho, *_ = three_way_game_split(gid, HOLDOUT_FRAC, VAL_FRAC, seed)
        mh = matched_masks(pool, ho)
        if mh is None:
            out["turnphase"].append({"seed": seed, "error": "empty matched"})
            continue
        ho_lm, ho_wm, lo, hi = mh
        # per-position scalar value of plane2 / plane3 (broadcast planes => single value/pos)
        p2 = states[:, 2].reshape(n, -1)
        p3 = states[:, 3].reshape(n, -1)
        # are planes 2,3 each spatially-constant (broadcast) per position?
        p2_bcast = bool(np.allclose(p2.min(1), p2.max(1)))
        p3_bcast = bool(np.allclose(p3.min(1), p3.max(1)))
        p2v = p2.mean(1)
        p3v = p3.mean(1)
        win_p2 = np.unique(np.round(p2v[ho_wm], 6))
        loss_p2 = np.unique(np.round(p2v[ho_lm], 6))
        win_p3 = np.unique(np.round(p3v[ho_wm], 6))
        loss_p3 = np.unique(np.round(p3v[ho_lm], 6))
        rec = {"seed": seed, "p2_broadcast": p2_bcast, "p3_broadcast": p3_bcast,
               "matched_win_p2_vals": win_p2.tolist(), "matched_loss_p2_vals": loss_p2.tolist(),
               "matched_win_p3_vals": win_p3.tolist(), "matched_loss_p3_vals": loss_p3.tolist(),
               "p2_constant_across_winloss": bool(set(win_p2.tolist()) == set(loss_p2.tolist()) and len(win_p2) == 1),
               "stone_support": [lo, hi]}
        out["turnphase"].append(rec)
        print(f"[TURNPHASE seed={seed}] p2_bcast={p2_bcast} p3_bcast={p3_bcast} | "
              f"matched win p2={win_p2.tolist()} loss p2={loss_p2.tolist()} | "
              f"win p3={win_p3.tolist()} loss p3={loss_p3.tolist()} | support=[{lo},{hi}]")
    out["turnphase_constant_all_seeds"] = all(
        r.get("p2_constant_across_winloss", False) for r in out["turnphase"])

    # =====================================================================
    # 4. OVERFIT ATTRIBUTION — pull per-LR train-fit vs holdout KC from probe JSON.
    #    If NO setting achieves holdout joint sep even when train-fit is high, overfit
    #    is NOT 'the separation' (there is none to overfit to).
    # =====================================================================
    probe_json = REPO_ROOT / "reports" / "d_fullspec_2026-06-26" / "LIGHTTRUNK_results.json"
    if probe_json.exists():
        pj = json.loads(probe_json.read_text())
        rows = []
        for r in pj["sweep"]:
            m = r["eval"]["matched"]; tf = r["eval"]["train_fit"]
            rows.append({"mode": r["mode"], "trunk_lr": r["trunk_lr"],
                         "trunk_max_delta": r["trunk_max_delta"],
                         "train_fit_ka": round(tf["kill_a"], 4), "train_fit_kc": round(tf["kill_c"], 4),
                         "holdout_ka": round(m["kill_a"], 4), "holdout_kc": round(m["kill_c"], 4),
                         "kc_gap_train_minus_holdout": round(tf["kill_c"] - m["kill_c"], 4),
                         "holdout_joint_pass": bool(m["kill_a"] > 0.35 and m["kill_c"] >= 0.85)})
            print(f"[OVERFIT {r['mode']} lr={r['trunk_lr']:g}] train_fit KC={tf['kill_c']:.3f} "
                  f"holdout KC={m['kill_c']:.3f} gap={tf['kill_c']-m['kill_c']:+.3f} "
                  f"holdout_joint_pass={rows[-1]['holdout_joint_pass']}")
        out["overfit_table"] = rows
        out["any_holdout_joint_pass"] = any(x["holdout_joint_pass"] for x in rows)
        out["max_holdout_kc_any_setting"] = max(x["holdout_kc"] for x in rows)
        # best (lr=1e-5) seed-101 same-split gap (apples-to-apples, not vs multiseed mean)
        lr1e5 = next(x for x in rows if x["mode"] == "full" and abs(x["trunk_lr"] - 1e-5) < 1e-12)
        out["best_same_split_kc_gap"] = lr1e5["kc_gap_train_minus_holdout"]
        print(f"[OVERFIT-SUMMARY] any_holdout_joint_pass={out['any_holdout_joint_pass']} "
              f"max_holdout_kc={out['max_holdout_kc_any_setting']} "
              f"best(lr1e-5) same-split train-vs-holdout KC gap={out['best_same_split_kc_gap']:+.3f}")
        out["value_destruction"] = pj.get("value_destruction")

    rep = REPO_ROOT / "reports" / "d_fullspec_2026-06-26" / "LIGHTTRUNK_REDTEAM_results.json"
    rep.write_text(json.dumps(out, indent=2))
    print(f"\n[WROTE] {rep}")
    print("REDTEAM_VERIFY_JSON " + json.dumps({
        "game_disjoint_all_seeds": out["game_disjoint_all_seeds"],
        "zero_board_byte_overlap_all_seeds": out["zero_board_byte_overlap_all_seeds"],
        "any_near_dup_le2": out["any_near_dup_le2"],
        "turnphase_constant_all_seeds": out["turnphase_constant_all_seeds"],
        "any_holdout_joint_pass": out.get("any_holdout_joint_pass"),
        "max_holdout_kc_any_setting": out.get("max_holdout_kc_any_setting"),
        "best_same_split_kc_gap": out.get("best_same_split_kc_gap"),
    }))


if __name__ == "__main__":
    main()
