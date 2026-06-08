#!/usr/bin/env python3
"""Analyze the §PRELONG centering-vs-size oracle output → V1-V4 tally + histograms.

Reads oracle.jsonl from prelong_centering_oracle.py and emits:
  - Distribution (1): chebyshev dist bbox-mid -> winning cell (histogram + p50/95/99/max)
  - Distribution (2): winning-LINE bbox span (histogram + p50/95/99/max)
  - Off-window oracle tally: reproduction, recover (won_ii), visit-argmax recovery,
    line_fits_19, recenter-frame containment, both-hit fluke
  - Pre-registered V1-V4 verdict tally
  - RED-TEAM: dense-head index clustering, raw-policy perception, spread representativeness
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else float("nan")


def hist(xs, lo=None, hi=None):
    if not xs:
        return "  (empty)"
    lo = min(xs) if lo is None else lo
    hi = max(xs) if hi is None else hi
    c = Counter(xs)
    lines = []
    for v in range(int(lo), int(hi) + 1):
        n = c.get(v, 0)
        bar = "#" * min(n, 60)
        lines.append(f"   {v:>4}: {n:>4} {bar}")
    return "\n".join(lines)


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "reports/investigations/prelong_centering_data/oracle.jsonl")
    recs = [json.loads(l) for l in path.open()]
    off = [r for r in recs if r.get("off_window")]
    BOARD = 19  # v6_live2 board_size (reported, not used as a literal lever)

    print("=" * 78)
    print(f"§PRELONG CENTERING-VS-SIZE ORACLE — {path}")
    print(f"total miss records: {len(recs)}   off-window oracle records: {len(off)}")
    print(f"in-window misses (no oracle): {len(recs)-len(off)}")
    print("=" * 78)

    # ---- Distribution (1): chebyshev center -> winning cell (ALL misses) ----
    cheb = [r["cheb_center_to_win"] for r in recs]
    print("\n## Distribution (1) — Chebyshev dist  bbox-mid center -> winning cell")
    print(f"   n={len(cheb)}  p50={pct(cheb,50):.1f} p95={pct(cheb,95):.1f} "
          f"p99={pct(cheb,99):.1f} max={max(cheb) if cheb else 'NA'}  (window HALF=9)")
    print(hist(cheb))

    # ---- Distribution (2): winning-LINE bbox span ----
    span = [r["win_line_bbox_span"] for r in recs]
    fits = [r["line_fits_19"] for r in recs]
    print("\n## Distribution (2) — winning-LINE bbox span (the 6-in-a-row the win completes)")
    print(f"   n={len(span)}  p50={pct(span,50):.1f} p95={pct(span,95):.1f} "
          f"p99={pct(span,99):.1f} max={max(span) if span else 'NA'}")
    print(f"   line_fits_19 (span<19): {sum(fits)}/{len(fits)} = {sum(fits)/max(len(fits),1):.3f}")
    print(hist(span))

    if not off:
        print("\n(no off-window oracle records yet)")
        return

    # ---- Off-window oracle tally ----
    won_i = [r["arm_i_current"]["won_turn"] for r in off]
    won_ii = [r["arm_ii_recentered"]["won_turn"] for r in off]
    am_i = [r["arm_i_current"]["visit_argmax_is_win"] for r in off]
    am_ii = [r["arm_ii_recentered"]["visit_argmax_is_win"] for r in off]
    reproduced = [not w for w in won_i]                       # miss persists under current
    n_rep = sum(reproduced)
    fits_off = [r["line_fits_19"] for r in off]
    contains_win = [r["recenter_frame_contains_win_cell"] for r in off]
    contains_line = [r["recenter_frame_contains_full_line"] for r in off]
    both_hit = [won_i[i] and won_ii[i] for i in range(len(off))]

    # recover = won_ii among REPRODUCED (true off-window misses)
    rep_recover = [won_ii[i] for i in range(len(off)) if reproduced[i]]
    rep_am = [am_ii[i] for i in range(len(off)) if reproduced[i]]
    recover_rate = sum(rep_recover) / max(n_rep, 1)
    am_rate = sum(rep_am) / max(n_rep, 1)

    print("\n## OFF-WINDOW ORACLE TALLY (n=%d)" % len(off))
    print(f"   arm_i (current) reproduces miss (won=False): {n_rep}/{len(off)} = {n_rep/len(off):.3f}")
    print(f"   arm_i won the turn anyway (NON-reproduction): {sum(won_i)} "
          f"-> V4-mislabel signal if material")
    print(f"   both-hit fluke (won_i AND won_ii): {sum(both_hit)} (should be ~0)")
    print(f"   line_fits_19 among off-window: {sum(fits_off)}/{len(off)} = {sum(fits_off)/len(off):.3f}")
    print(f"   recenter frame contains WIN CELL: {sum(contains_win)}/{len(off)} = {sum(contains_win)/len(off):.3f}")
    print(f"   recenter frame contains FULL LINE: {sum(contains_line)}/{len(off)} = {sum(contains_line)/len(off):.3f}")
    print(f"\n   RECOVER among reproduced (won_ii): {sum(rep_recover)}/{n_rep} = {recover_rate:.3f}")
    print(f"   visit-argmax==win among reproduced: {sum(rep_am)}/{n_rep} = {am_rate:.3f}")

    # ---- V3 subset: in-window-with-prior but still misses under re-center ----
    # not perception-bound: win move IS a child with prior>0 under re-center, yet not won
    v3 = [r for r in off
          if not r["arm_ii_recentered"]["won_turn"]
          and r["arm_ii_recentered"]["win_in_children"]
          and r["arm_ii_recentered"]["win_prior"] > 0
          and not r["arm_i_current"]["won_turn"]]
    print(f"\n   V3 subset (miss persists re-centered, win is a child w/ prior>0): "
          f"{len(v3)}/{n_rep} = {len(v3)/max(n_rep,1):.3f}")

    # ---- raw policy perception ----
    rp_in = [r["raw_policy"]["win_in_input_window"] for r in off]
    rp_rank = [r["raw_policy"]["best_window_rank"] for r in off if r["raw_policy"]["best_window_rank"]]
    rp_prob = [r["raw_policy"]["best_window_prob"] for r in off]
    print("\n## RED-TEAM — raw per-cluster-window policy (does NN PERCEIVE+SCORE the win?)")
    print(f"   win cell present in some INPUT window: {sum(rp_in)}/{len(off)} = {sum(rp_in)/len(off):.3f}")
    print(f"   win-cell policy RANK in its input window: p50={pct(rp_rank,50):.0f} "
          f"p90={pct(rp_rank,90):.0f} max={max(rp_rank) if rp_rank else 'NA'} (1=top of 362)")
    print(f"   win-cell window prob: p50={pct(rp_prob,50):.4f} p90={pct(rp_prob,90):.4f}")
    # correlation: recovered cases have lower rank?
    rec_rank = [r["raw_policy"]["best_window_rank"] for r in off
                if r["arm_ii_recentered"]["won_turn"] and r["raw_policy"]["best_window_rank"]]
    norec_rank = [r["raw_policy"]["best_window_rank"] for r in off
                  if not r["arm_ii_recentered"]["won_turn"] and r["raw_policy"]["best_window_rank"]]
    print(f"   rank | RECOVERED: p50={pct(rec_rank,50):.0f} (n={len(rec_rank)})   "
          f"NOT-recovered: p50={pct(norec_rank,50):.0f} (n={len(norec_rank)})")

    # ---- dense-head generalization: re-centered win flat-index clustering ----
    flats = [r["arm_ii_recentered"]["win_to_flat"] for r in off
             if r["arm_ii_recentered"]["win_to_flat"] < 362]
    hit_flats = [r["arm_ii_recentered"]["win_to_flat"] for r in off
                 if r["arm_ii_recentered"]["won_turn"] and r["arm_ii_recentered"]["win_to_flat"] < 362]
    print("\n## RED-TEAM — dense-head generalization (index clustering = memorization?)")
    print(f"   distinct re-centered win flat-indices (all off): {len(set(flats))} over {len(flats)}")
    print(f"   distinct re-centered win flat-indices (HITS):    {len(set(hit_flats))} over {len(hit_flats)}")
    top = Counter(hit_flats).most_common(5)
    print(f"   top hit indices: {top}")

    # ---- spread representativeness ----
    nst = [r["n_stones"] for r in off]
    gb = [r["global_bbox_span"] for r in off]
    print("\n## RED-TEAM — spread representativeness (current-best vs 300k Dirichlet?)")
    print(f"   n_stones: p50={pct(nst,50):.0f} p95={pct(nst,95):.0f} max={max(nst)}")
    print(f"   global bbox span: p50={pct(gb,50):.0f} p95={pct(gb,95):.0f} max={max(gb)}")

    # ---- PRE-REGISTERED VERDICTS ----
    cheb_p99 = pct(cheb, 99)
    span_gt19 = sum(1 for s in span if s > 19) / max(len(span), 1)
    fits_rate = sum(fits) / max(len(fits), 1)
    print("\n" + "=" * 78)
    print("PRE-REGISTERED VERDICT TALLY")
    print("=" * 78)
    v1 = recover_rate >= 0.80 and fits_rate >= 0.95
    v2 = (recover_rate < 0.50 or span_gt19 > 0.05) and cheb_p99 <= 12
    print(f"V1 CENTERING  : recover>=0.80 ({recover_rate:.3f}) AND line_fits_19>=0.95 ({fits_rate:.3f})  -> {'FIRES' if v1 else 'no'}")
    print(f"V2 SIZE       : (recover<0.50 OR span>19 frac>5% [{span_gt19:.3f}]) AND cheb_p99<=12 ([{cheb_p99:.1f}])  -> {'FIRES' if v2 else 'no'}")
    print(f"V3 COMPUTE    : in-window-w/prior persist-miss frac = {len(v3)/max(n_rep,1):.3f}  (STOP if dominant)")
    print(f"V4 MISLABEL   : non-reproduction frac = {sum(won_i)/len(off):.3f}  (fix probe if material)")
    print(f"TIEBREAK      : 0.50<=recover<0.80 -> {'YES, run 25x25 eval-only' if 0.50<=recover_rate<0.80 else 'n/a'}")
    if cheb_p99 > 12:
        print(f"   NOTE: cheb_p99={cheb_p99:.1f} > 12 -> per pre-reg, escalate width / reconsider")


if __name__ == "__main__":
    main()
