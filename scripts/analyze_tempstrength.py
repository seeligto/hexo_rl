#!/usr/bin/env python
"""D-TEMPSTRENGTH verdict analyzer — honest head-to-head from the round_robin per_game data.

PRIMARY = a20@X vs control@X at each rung (s155k–s55k, s160k–s60k, s165k–s65k).
For each pair: WR (a20 perspective, draws=0.5) + distinct-game bootstrap CI (§D-STRENGTHAXIS:
effective-n = DISTINCT games, NOT game count — argmax-from-fixed-opening pseudo-replicates).
Also pools the BT-Elo ladder (ratings.csv) and prints the pre-registered verdict gates.

  python scripts/analyze_tempstrength.py --rr reports/tempstrength_rr [--n-boot 2000]
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402
from hexo_rl.eval.round_robin import distinct_games, load_games  # noqa: E402

# step-key → meaning
ANCHOR = "s50k"
CONTROL = {"s55k": 55000, "s60k": 60000, "s65k": 65000}
A20 = {"s155k": 55000, "s160k": 60000, "s165k": 65000}   # offset +100k
RUNG_PAIRS = [("s155k", "s55k", 55000), ("s160k", "s60k", 60000), ("s165k", "s65k", 65000)]


def pair_games(games, A, B):
    return [g for g in games if {g["p1"], g["p2"]} == {A, B}]


def score_for(g, C):
    """C-perspective score: win=1.0, draw=0.5, loss=0.0."""
    if g["winner"] == "draw":
        return 0.5
    won = (g["p1"] == C and g["winner"] == "p1") or (g["p2"] == C and g["winner"] == "p2")
    return 1.0 if won else 0.0


def wr_with_ci(games, A, B, n_boot=2000, seed=20260613):
    """A-perspective WR + distinct-game bootstrap CI. Returns dict."""
    gs = pair_games(games, A, B)
    n_total = len(gs)
    if n_total == 0:
        return {"n_total": 0}
    reps, _counts = distinct_games(gs)
    n_distinct = len(reps)
    scores = np.array([score_for(g, A) for g in reps], dtype=float)
    draws = sum(1 for g in reps if g["winner"] == "draw")
    a_wins = sum(1 for g in reps if score_for(g, A) == 1.0)
    b_wins = sum(1 for g in reps if score_for(g, A) == 0.0)
    wr_score = float(scores.mean())                       # draws as 0.5
    decisive = a_wins / (a_wins + b_wins) if (a_wins + b_wins) else float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(scores, size=n_distinct, replace=True).mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n_total": n_total, "n_distinct": n_distinct,
        "copy_multiplier": round(n_total / n_distinct, 2) if n_distinct else None,
        "a_wins": a_wins, "b_wins": b_wins, "draws": draws,
        "wr_score": round(wr_score, 4), "decisive_wr": round(decisive, 4),
        "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
        "ci_separated_gt_0.5": bool(lo > 0.5),
        "ci_separated_lt_0.5": bool(hi < 0.5),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rr", default="reports/tempstrength_rr")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rr = Path(args.rr)
    games = load_games([str(rr)])
    print(f"loaded {len(games)} games from {rr}")

    out = {"primary_head_to_head": {}, "drift_vs_anchor": {}}
    print("\n=== PRIMARY: a20 vs control (a20 perspective, draws=0.5) ===")
    for A, B, rung in RUNG_PAIRS:
        r = wr_with_ci(games, A, B, args.n_boot)
        out["primary_head_to_head"][f"rung_{rung}"] = r
        if r.get("n_total"):
            print(f" {rung}: a20 WR={r['wr_score']:.3f} CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
                  f"(decisive {r['decisive_wr']:.3f}; W{r['a_wins']}/L{r['b_wins']}/D{r['draws']}; "
                  f"n_distinct={r['n_distinct']}/{r['n_total']} copy×{r['copy_multiplier']}) "
                  f"{'a20>ctrl ✓' if r['ci_separated_gt_0.5'] else ('ctrl>a20 ✗' if r['ci_separated_lt_0.5'] else 'NULL (CI straddles 0.5)')}")
        else:
            print(f" {rung}: NO GAMES")

    print("\n=== drift: each arm vs golong@50k anchor (arm perspective) ===")
    for lab, rung in {**CONTROL, **A20}.items():
        arm = "a20" if lab in A20 else "ctrl"
        r = wr_with_ci(games, lab, ANCHOR, args.n_boot)
        out["drift_vs_anchor"][lab] = r
        if r.get("n_total"):
            print(f" {arm}@{rung} ({lab}) vs anchor: WR={r['wr_score']:.3f} "
                  f"CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] n_distinct={r['n_distinct']}/{r['n_total']}")

    # BT-Elo ladder (pooled)
    ratings_csv = rr / "ratings.csv"
    if ratings_csv.exists():
        print("\n=== BT-Elo ladder (pooled, distinct-game bootstrap CI) ===")
        rows = list(csv.DictReader(ratings_csv.open()))
        def name(lab): return ("a20@" + str(A20[lab]//1000) + "k") if lab in A20 else \
            (("ctrl@" + str(CONTROL[lab]//1000) + "k") if lab in CONTROL else "anchor@50k")
        for r in sorted(rows, key=lambda x: float(x["elo"]), reverse=True):
            cl = r.get("ci_lo_boot") or r.get("ci_lo"); ch = r.get("ci_hi_boot") or r.get("ci_hi")
            print(f" {name(r['label']):>12} ({r['label']:>6}): Elo {float(r['elo']):+7.1f}  CI[{cl},{ch}]")
        out["elo_ladder"] = rows

    agg = rr / "aggregate.json"
    if agg.exists():
        a = json.loads(agg.read_text())
        warn = a.get("effective_n_warning", {})
        print(f"\neffective-n guard: copy_multiplier={a.get('copy_multiplier')} "
              f"min_distinct/pair={warn.get('distinct_per_pair_min')} "
              f"low_power_warning={warn.get('low_power_warning')}")
        out["effective_n_warning"] = warn

    (rr / "tempstrength_verdict_data.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {rr/'tempstrength_verdict_data.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
