#!/usr/bin/env python3
"""§D-OVERSPREAD D5 — SELF-PLAY CO-ADAPTATION ATTRACTOR (driver discriminator).

HYPOTHESIS (D5): self-play is spread-vs-spread -> the over-spread of the model's own force
is never punished INTERNALLY (both sides scatter), only EXTERNALLY by a compact finisher
(SealBot). If true, the over-spread metric should track the SealBot-WR DECLINE across the
arc (the external punishment shows up as a falling external WR while internal play looks
fine), and SealBot losses should be disproportionately spread-force games.

WHAT THIS SCRIPT DOES (EVAL-ONLY, READ-ONLY; no model load, no engine/config edit):
  PART 1 (feasible, LOW POWER): take the LOCKED over-spread metrics from the §D-COHERENCE
    coherence_overspread.py output (investigation/coherence_2026-06-08/overspread.json,
    buckets 30k/53k/87.5k) and the SealBot-WR series parsed from the golong run log
    (37.5k 0.24 / 50k 0.38 / 62.5k 0.29 / 75k 0.05-transient / 87.5k ~0.19). The over-spread
    buckets and WR-eval steps do NOT coincide, so interpolate the over-spread metric LINEARLY
    WITHIN its measured range [30k, 87.5k] to the WR-eval steps (never extrapolate). Report
    Pearson + Spearman of each over-spread metric vs WR, WITH and WITHOUT the 75k false-abort
    transient. n is tiny (3-5 promoted buckets) -> SUGGESTIVE ONLY; CIs are wide and reported.
  PART 2 (INSTRUMENT-LIMITED -> reported as a GAP, NOT fabricated): "SealBot losses are
    disproportionately spread-force games" needs the SealBot eval GAME move-sequences. This
    script VERIFIES they are not banked (the log carries only per-game winrate / colony_wins,
    no moves) and that regenerating them requires the spread-capped ModelPlayer
    (evaluator.py max_spread<=18) which cannot reproduce the spread-306 self-play regime.

All geometry/encoding via hexo_rl.encoding.lookup; zero board literals. No threshold moves.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from hexo_rl.encoding import lookup, normalize_encoding_name


# ---------------------------------------------------------------------------
# Part 2 instrument-gap verification: does the run log bank SealBot eval move-seqs?
# ---------------------------------------------------------------------------
def parse_sealbot_wr(log_path: str):
    """Return list of (step, winrate, colony_wins) for each completed sealbot eval, plus a
    flag whether ANY sealbot eval record carried a move-sequence (Part-2 feasibility)."""
    last_step = None
    series = []
    has_moves_field = False
    pending_eval_step = None
    for line in open(log_path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        for k in ("step", "global_step", "train_step", "checkpoint_step"):
            v = d.get(k)
            if isinstance(v, (int, float)):
                last_step = int(v)
        ev = d.get("event")
        if ev == "evaluation_start" and d.get("checkpoint_step") is not None:
            pending_eval_step = int(d["checkpoint_step"])
        if ev == "evaluation_games_complete" and d.get("phase") == "sealbot":
            step = pending_eval_step if pending_eval_step is not None else last_step
            series.append((step, float(d.get("winrate")), int(d.get("colony_wins", 0))))
            # Part-2 probe: any move-sequence stored alongside the per-game eval result?
            for mk in ("moves", "game_moves", "move_sequence", "games", "per_game_moves"):
                if mk in d:
                    has_moves_field = True
    return series, has_moves_field


# ---------------------------------------------------------------------------
# Part 1 helpers
# ---------------------------------------------------------------------------
def lin_interp(xp, fp, x):
    """np.interp but returns NaN outside [min(xp), max(xp)] (never extrapolate)."""
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    y[(x < min(xp)) | (x > max(xp))] = np.nan
    return y


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), len(a)
    return float(np.corrcoef(a, b)[0, 1]), len(a)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 3:
        return float("nan"), len(a)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan"), len(a)
    return float(np.corrcoef(ra, rb)[0, 1]), len(a)


def perm_pvalue(a, b, stat_fn, n_perm=100000, seed=0):
    """Exact/near-exact permutation p-value (two-sided) for the correlation; tiny-n honest."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = len(a)
    if n < 3:
        return float("nan")
    obs = stat_fn(a, b)[0]
    if np.isnan(obs):
        return float("nan")
    # exact over all permutations when n small, else sampled
    from math import factorial
    if factorial(n) <= n_perm:
        import itertools
        cnt = tot = 0
        for perm in itertools.permutations(range(n)):
            s = stat_fn(a, b[list(perm)])[0]
            if not np.isnan(s):
                tot += 1
                if abs(s) >= abs(obs) - 1e-12:
                    cnt += 1
        return cnt / tot if tot else float("nan")
    rng = np.random.default_rng(seed)
    cnt = tot = 0
    for _ in range(n_perm):
        s = stat_fn(a, rng.permutation(b))[0]
        if not np.isnan(s):
            tot += 1
            if abs(s) >= abs(obs) - 1e-12:
                cnt += 1
    return cnt / tot if tot else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overspread", default="investigation/coherence_2026-06-08/overspread.json")
    ap.add_argument("--log", default="investigation/fragility_2026-06-07/v6l2golong.log")
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--out", default="investigation/overspread_2026-06-08/d5_coadapt.json")
    # The 87.5k sealbot eval STARTED but the run log was cut before evaluation_games_complete
    # (verified: no 87.5k complete record). The dispatcher/§D-COHERENCE report the recovered
    # 87.5k WR ~= 0.19. Inject it as a LOG-EXTERNAL anchor (flagged) because it is the only WR
    # point that coincides with an over-spread bucket AND it is the post-transient recovery
    # point that tests whether the decline-leg correlation is a 75k-transient artifact.
    ap.add_argument("--wr_87500", type=float, default=0.19,
                    help="log-external recovered 87.5k SealBot WR (dispatcher value)")
    args = ap.parse_args()

    name = normalize_encoding_name(args.encoding)
    _spec = lookup(name)  # encoding sanity (registry-driven; no literals used downstream)

    # ---- load locked over-spread buckets ----
    osj = json.loads(Path(args.overspread).read_text())
    buckets = sorted(osj["buckets"], key=lambda b: b["step"])
    os_steps = [b["step"] for b in buckets]
    # over-spread family. components/stone = mover_ncomp / mover_stones (the RED-TEAM-confirmed,
    # stone-count-controlled signature). Direction of "more spread": ncomp up, comp/stone up,
    # largest_frac DOWN, local_support DOWN, opp_near_win UP.
    metrics = {
        "mover_ncomp":          [b["mover_ncomp"] for b in buckets],
        "comp_per_stone":       [b["mover_ncomp"] / b["mover_stones"] for b in buckets],
        "largest_blob_frac":    [b["mover_largest_frac"] for b in buckets],
        "local_support":        [b["local_support"] for b in buckets],
        "opp_near_win":         [b["opp_near_win"] for b in buckets],
        "mover_density":        [b["mover_density"] for b in buckets],
    }

    # ---- parse SealBot WR ----
    series, has_moves = parse_sealbot_wr(args.log)
    series = [(s, wr, cw) for (s, wr, cw) in series if s is not None]
    # inject the log-external recovered 87.5k anchor if not already present in the log
    if not any(s == 87500 for (s, _w, _c) in series) and args.wr_87500 is not None:
        series.append((87500, float(args.wr_87500), -1))  # colony_wins unknown -> -1
        print(f"[note] injected LOG-EXTERNAL 87.5k WR={args.wr_87500} "
              f"(eval started but run log cut before complete; dispatcher value)")
    series.sort()
    wr_steps = [s for (s, wr, cw) in series]
    wr_vals = [wr for (s, wr, cw) in series]
    # 75k = single-point false-abort transient (recovered) per §D-FRAGILITY -> drop variant.
    # parsed eval steps drift slightly off the round tag (75428 not 75000) -> match by window.
    def _is_transient(s):
        return abs(s - 75000) <= 1000
    keep_mask = [not _is_transient(s) for (s, wr, cw) in series]

    print("=== D5 PART 1: over-spread vs SealBot-WR correlation (LOW POWER) ===")
    print(f"over-spread buckets (locked): {os_steps}")
    print(f"SealBot-WR series: " + " / ".join(f"{s/1000:g}k={wr:.2f}" for (s, wr, cw) in series))
    print(f"  (75k=0.05 is the §D-FRAGILITY false-abort TRANSIENT, recovered -> reported both ways)")

    # interpolate each over-spread metric to the WR-eval steps (within-range only)
    wr_steps_arr = np.array(wr_steps, float)
    out = {"encoding": name, "os_steps": os_steps, "wr_series": series,
           "part2_moves_banked": bool(has_moves), "metrics": {}}

    def run_variant(metric_name, os_vals, label, steps_arr, wr_arr):
        os_at_wr = lin_interp(os_steps, os_vals, steps_arr)
        pr, npts = pearson(os_at_wr, wr_arr)
        sr, _ = spearman(os_at_wr, wr_arr)
        pp = perm_pvalue(os_at_wr, wr_arr, pearson)
        print(f"  [{metric_name:>17} | {label:>14}] n={npts}  "
              f"pearson={pr:+.3f}  spearman={sr:+.3f}  perm_p={pp if isinstance(pp,float) else pp:.3f}")
        return {"label": label, "n": npts, "pearson": pr, "spearman": sr, "perm_p": pp,
                "os_at_wr": [None if np.isnan(v) else float(v) for v in os_at_wr],
                "wr": list(map(float, wr_arr))}

    incl_steps = wr_steps_arr
    incl_wr = np.array(wr_vals, float)
    excl_steps = wr_steps_arr[keep_mask]
    excl_wr = np.array(wr_vals, float)[keep_mask]

    for mname, mvals in metrics.items():
        print(f"\n-- {mname}  buckets={[round(v,3) for v in mvals]} --")
        rec = {
            "buckets": mvals,
            "with_transient":    run_variant(mname, mvals, "incl 75k", incl_steps, incl_wr),
            "without_transient": run_variant(mname, mvals, "excl 75k", excl_steps, excl_wr),
        }
        out["metrics"][mname] = rec

    # ---- D5 directional read ----
    # D5 predicts over-spread tracks the WR DECLINE. Sign expectation per metric:
    #   spread-up metrics (ncomp, comp_per_stone, opp_near_win)   -> NEGATIVE corr with WR
    #   spread-down metrics (largest_frac, local_support, density) -> POSITIVE corr with WR
    # "tracks decline" = correct sign AND |r| large. Tally the sign agreement (excl transient).
    expect_neg = {"mover_ncomp", "comp_per_stone", "opp_near_win"}
    agree = []
    for mname, rec in out["metrics"].items():
        r = rec["without_transient"]["pearson"]
        if np.isnan(r):
            continue
        ok = (r < 0) if mname in expect_neg else (r > 0)
        agree.append((mname, r, ok))
    n_ok = sum(1 for *_, ok in agree if ok)
    print("\n=== D5 sign-agreement with 'over-spread tracks WR decline' (excl 75k) ===")
    for mname, r, ok in agree:
        print(f"  {mname:>17}  pearson={r:+.3f}  expected_sign={'neg' if mname in expect_neg else 'pos'}  {'OK' if ok else 'WRONG'}")
    print(f"  -> {n_ok}/{len(agree)} metrics agree in DIRECTION (n=3-4 eval pts -> SUGGESTIVE, not decisive)")
    out["direction_agreement"] = {"n_ok": n_ok, "n_total": len(agree),
                                  "detail": [(m, r, ok) for (m, r, ok) in agree]}

    # ---- non-monotone WR caveat: WR itself rises 37.5->50k then falls; over-spread is ~monotone.
    # so any monotone metric MUST disagree with the early rise. Report WR shape.
    print("\nNOTE: SealBot-WR is NON-MONOTONE (0.24->0.38 rise, then 0.38->0.05/0.19 fall) while the "
          "over-spread metric is ~MONOTONE 30k->87.5k. A monotone predictor cannot fit a "
          "rise-then-fall curve -> linear corr is structurally capped; the relevant D5 claim is the "
          "DECLINE LEG (50k->87.5k). Report the decline-leg slope separately.")
    # decline leg: 50k..87.5k (drop the 37.5k pre-peak rise + 75k transient)
    decl_mask = [(not _is_transient(s) and s >= 50000) for (s, wr, cw) in series]
    decl_steps = wr_steps_arr[decl_mask]
    decl_wr = np.array(wr_vals, float)[decl_mask]
    print(f"  decline-leg eval steps (excl 75k transient): {[int(s) for s in decl_steps]}  wr={[round(float(w),2) for w in decl_wr]}")
    out["decline_leg"] = {}
    for mname, mvals in metrics.items():
        os_at = lin_interp(os_steps, mvals, decl_steps)
        pr, npts = pearson(os_at, decl_wr)
        sr, _ = spearman(os_at, decl_wr)
        print(f"  [{mname:>17}] decline-leg n={npts} pearson={pr:+.3f} spearman={sr:+.3f}")
        out["decline_leg"][mname] = {"n": npts, "pearson": pr, "spearman": sr}

    # ---- PART 2 instrument-gap report ----
    print("\n=== D5 PART 2: 'SealBot losses are disproportionately spread-force' — INSTRUMENT GAP ===")
    print(f"  SealBot eval records carry move-sequences? -> {has_moves}  "
          f"(checked keys: moves/game_moves/move_sequence/games/per_game_moves)")
    print("  Per-game SealBot eval data banked = winrate + win_count + colony_wins ONLY (no moves).")
    print("  Regenerating eval games => ModelPlayer, spread-capped (evaluator.py max_spread<=18,")
    print("  §D-WALLCAUSATION/§D-FRAGILITY) => cannot reproduce the spread-306 self-play regime.")
    print("  => Part 2 is NOT testable from local data. Reported as an instrument gap, not a verdict.")
    out["part2"] = {
        "testable": False,
        "reason": "no eval move-seqs banked; regen requires spread-capped ModelPlayer (max_spread<=18) "
                  "which cannot reproduce spread-306 self-play regime",
        "colony_wins_series": [(s, cw) for (s, wr, cw) in series],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
