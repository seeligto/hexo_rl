#!/usr/bin/env python3
"""Pre-long-run triage — analysis + routing verdict (offline).

Reads the JSONL produced by prelong_triage_probe.py (HeXO self-play) and
sealbot_selfplay_t1.py (T1b), plus prints the T1a human-corpus stats, and emits:

  T1  intrinsic drawishness: HeXO cap-rate vs SealBot self-play cap-rate vs human.
  T2  MISS-LOCATION (keystone): are missed forced wins off-window / peripheral?
  T3  CAP-RATE TREND: cap-rate vs checkpoint step (falling => H-early).
  ROUTING per the pre-registered tree.

Usage:
  .venv/bin/python scripts/structural_diagnosis/prelong_triage_analyze.py \
     --data-dir reports/investigations/prelong_triage_data
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HALF = 9  # 19x19 window half-width


def load_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def human_corpus_stats(raw_dir="data/corpus/raw_human"):
    files = glob.glob(str(Path(raw_dir) / "*.json"))
    lengths = []
    for fp in files:
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        lengths.append(len(d.get("moves") or []))
    if not lengths:
        return None
    L = np.array(lengths)
    return {
        "n": len(L), "median": float(np.median(L)), "mean": float(L.mean()),
        "p90": float(np.percentile(L, 90)), "p95": float(np.percentile(L, 95)),
        "ge150_frac": float((L >= 150).mean()),
        "note": "PRE-FILTERED to six-in-a-row wins (rated, >=20 mv) — decisive-biased; "
                "measures decisive-game LENGTH only, NOT draw rate.",
    }


def t3_cap_trend(games):
    by_step = defaultdict(list)
    for g in games:
        by_step[g["step"]].append(g)
    rows = []
    for step in sorted(by_step):
        gs = by_step[step]
        n = len(gs)
        ncap = sum(1 for g in gs if g["outcome"] == "cap")
        plies = np.array([g["n_plies"] for g in gs])
        seen = sum(g["forced_seen"] for g in gs)
        missed = sum(g["forced_missed"] for g in gs)
        rows.append({
            "step": step, "n": n, "cap_rate": ncap / n, "n_cap": ncap,
            "mean_ply": float(plies.mean()), "median_ply": float(np.median(plies)),
            "forced_seen": seen, "forced_missed": missed,
            "miss_rate_per_forced_turn": missed / max(1, seen),
        })
    return rows


def t2_miss_location(misses, focus_step=None):
    """Aggregate missed-forced-win cell geometry vs the 19x19 window."""
    if focus_step is not None:
        misses = [m for m in misses if m.get("step") == focus_step]
    # one record per missed turn; each has depth1[] + depth2[] cell geoms.
    # The binding cell = the FURTHEST completing cell needed that turn (perception
    # is limited by the furthest cell; if it's off-window the win is unreachable).
    turn_rows = []
    all_cells = []   # every completing cell geom
    for m in misses:
        cells = list(m.get("depth1", [])) + list(m.get("depth2", []))
        if not cells:
            continue
        all_cells.extend(cells)
        far = max(cells, key=lambda c: c["cheb"])
        near = min(cells, key=lambda c: c["cheb"])
        turn_rows.append({
            "far_cheb": far["cheb"], "far_in_window": far["in_window"],
            "near_cheb": near["cheb"], "near_in_window": near["in_window"],
            "n_clusters": m.get("n_clusters", 1), "bbox_span": m.get("bbox_span", 0),
            "n_stones": m.get("n_stones", 0),
            "offwindow_frac_legal": m.get("offwindow_frac_legal", 0.0),
            "mr_start": m.get("mr_start", 2),
        })
    if not turn_rows:
        return None
    far_cheb = np.array([t["far_cheb"] for t in turn_rows])
    far_off = np.array([not t["far_in_window"] for t in turn_rows])
    near_off = np.array([not t["near_in_window"] for t in turn_rows])
    clusters = np.array([t["n_clusters"] for t in turn_rows])
    spans = np.array([t["bbox_span"] for t in turn_rows])
    base_off = np.array([t["offwindow_frac_legal"] for t in turn_rows])
    cell_cheb = np.array([c["cheb"] for c in all_cells])
    cell_off = np.array([not c["in_window"] for c in all_cells])
    return {
        "n_missed_turns": len(turn_rows),
        "n_missed_cells": len(all_cells),
        # binding (furthest) completing cell per turn
        "far_off_window_frac": float(far_off.mean()),       # win needs an off-window cell
        "near_off_window_frac": float(near_off.mean()),     # EVERY completing cell off-window
        "any_cell_off_window_frac": float(cell_off.mean()),
        "far_cheb_median": float(np.median(far_cheb)),
        "far_cheb_p90": float(np.percentile(far_cheb, 90)),
        "cell_cheb_median": float(np.median(cell_cheb)),
        # baseline: at the SAME miss positions, what frac of random legal moves are off-window?
        "baseline_legal_off_window_mean": float(base_off.mean()),
        "lift_vs_baseline": float(far_off.mean() / max(1e-6, base_off.mean())),
        # board context at miss positions
        "n_clusters_median": float(np.median(clusters)),
        "n_clusters_ge2_frac": float((clusters >= 2).mean()),
        "bbox_span_median": float(np.median(spans)),
        "bbox_span_gt18_frac": float((spans > 18).mean()),  # span>window => can't cover board
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="reports/investigations/prelong_triage_data")
    ap.add_argument("--prefix", default="v6_live2")
    args = ap.parse_args()
    dd = Path(args.data_dir)

    games = load_jsonl(dd / f"{args.prefix}.games.jsonl")
    misses = load_jsonl(dd / f"{args.prefix}.misses.jsonl")
    sb = load_jsonl(dd / "sealbot_selfplay.jsonl")

    print("=" * 70)
    print("T1a — HUMAN CORPUS (decisive-game length; draw rate NOT measurable)")
    print("=" * 70)
    h = human_corpus_stats()
    if h:
        print(f"  n={h['n']}  median_len={h['median']:.0f}  mean={h['mean']:.1f}  "
              f"p90={h['p90']:.0f}  p95={h['p95']:.0f}")
        print(f"  frac >=150 stones (HeXO cap): {h['ge150_frac']*100:.2f}%")
        print(f"  CAVEAT: {h['note']}")

    print("\n" + "=" * 70)
    print("T1b — SEALBOT SELF-PLAY (intrinsic drawishness, unbiased)")
    print("=" * 70)
    if sb:
        ncap = sum(1 for g in sb if g["outcome"] == "cap")
        L = np.array([g["n_plies"] for g in sb])
        seen = sum(g["forced_seen"] for g in sb)
        missed = sum(g["forced_missed"] for g in sb)
        print(f"  n={len(sb)}  cap_rate={ncap/len(sb)*100:.1f}% ({ncap}/{len(sb)})  "
              f"mean_ply={L.mean():.1f}  median={np.median(L):.0f}  p90={np.percentile(L,90):.0f}")
        print(f"  SealBot forced_turns={seen} missed={missed} "
              f"({100*missed/max(1,seen):.1f}%) — sanity: strong bot should ~0")
    else:
        print("  (sealbot_selfplay.jsonl not present yet)")

    print("\n" + "=" * 70)
    print("T3 — HeXO CAP-RATE TREND vs CHECKPOINT (H-early)")
    print("=" * 70)
    if games:
        rows = t3_cap_trend(games)
        print(f"  {'step':>8} {'n':>4} {'cap_rate':>9} {'mean_ply':>9} {'miss/forced':>12}")
        for r in rows:
            print(f"  {r['step']:>8} {r['n']:>4} {r['cap_rate']*100:>8.1f}% "
                  f"{r['mean_ply']:>9.1f} {r['miss_rate_per_forced_turn']*100:>11.1f}%")
        if len(rows) >= 2:
            d = rows[-1]["cap_rate"] - rows[0]["cap_rate"]
            print(f"  trend {rows[0]['step']}->{rows[-1]['step']}: "
                  f"{rows[0]['cap_rate']*100:.1f}% -> {rows[-1]['cap_rate']*100:.1f}%  "
                  f"(Δ {d*100:+.1f}pp) => {'FALLING (H-early)' if d < -0.03 else 'FLAT/RISING (structural)'}")
    else:
        print("  (no games yet)")

    print("\n" + "=" * 70)
    print("T2 — MISS-LOCATION (KEYSTONE): off-window => H-perception")
    print("=" * 70)
    steps = sorted({g["step"] for g in games}) if games else []
    focus = max(steps) if steps else None
    for st in steps:
        r = t2_miss_location(misses, focus_step=st)
        tag = " <= FOCUS (latest)" if st == focus else ""
        if not r:
            print(f"  step {st}: no missed forced wins{tag}")
            continue
        print(f"\n  step {st}{tag}: n_missed_turns={r['n_missed_turns']} n_cells={r['n_missed_cells']}")
        print(f"    off-window (binding cell):   {r['far_off_window_frac']*100:5.1f}%  "
              f"(win needs a cell beyond the 19x19 window)")
        print(f"    off-window (EVERY cell):     {r['near_off_window_frac']*100:5.1f}%")
        print(f"    baseline random-legal off-window: {r['baseline_legal_off_window_mean']*100:5.1f}%  "
              f"=> LIFT {r['lift_vs_baseline']:.2f}x")
        print(f"    far-cell cheb dist: median={r['far_cheb_median']:.0f} p90={r['far_cheb_p90']:.0f} (window edge=9)")
        print(f"    board at miss: n_clusters>=2 {r['n_clusters_ge2_frac']*100:.0f}%  "
              f"span>18 {r['bbox_span_gt18_frac']*100:.0f}%  median_span={r['bbox_span_median']:.0f}")

    print("\n" + "=" * 70)
    print("ROUTING")
    print("=" * 70)
    if focus is not None:
        r = t2_miss_location(misses, focus_step=focus)
        if r:
            peripheral = r["far_off_window_frac"] >= 0.30 or r["lift_vs_baseline"] >= 1.5
            verdict = "PERIPHERAL => H-PERCEPTION => K-cluster restore arm BEFORE long run" if peripheral \
                else "CENTRAL => not perception => lean H-early/H-game"
            print(f"  T2: {verdict}")
            print(f"      (far_off_window={r['far_off_window_frac']*100:.0f}%, "
                  f"lift={r['lift_vs_baseline']:.2f}x, clusters>=2 at miss={r['n_clusters_ge2_frac']*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
