#!/usr/bin/env python3
"""D-LOCALIZE P3 — search-scaling verdict (standalone, stats-only).

Reads reports/d_localize_p3/<n>/sealbot_games.jsonl for each swept n, computes
per-checkpoint model-WR vs fixed-depth SealBot with a DISTINCT-GAME bootstrap 95%
CI, and applies the pre-registered gate:

  PLATEAU-by-150 : no checkpoint shows CI_low(WR@256) > CI_high(WR@150)  -> keep n=150.
  CLIMBS-past-256: >=1 checkpoint with CI_low(WR@256) > CI_high(WR@150)  -> adopt n=256.

Decision-relevant comparison = 150 -> 256 (128 is the phase-1-starvation control;
512 the climbs-confirm; redteam_d4_n256 the bar-specificity check). Distinct-game
bootstrap per the D-ARGMAX guard (dedupe byte-identical move-seqs). No engine, no GPU.
Writes reports/d_localize_p3/P3_VERDICT.md.
"""
from __future__ import annotations
import glob, json, os, random
from collections import defaultdict

OUT = os.path.join("reports", "d_localize_p3")
CELLS = ["n128", "n150", "n256", "n512", "redteam_d4_n256"]
N_BOOT = 2000
random.seed(20260625)


def model_scores(games, label):
    """Per-game model score (1 win / .5 draw / 0 loss) + the move-seq key, vs sealbot."""
    out = []
    for g in games:
        p1, p2, w = g.get("p1"), g.get("p2"), g.get("winner")
        if "sealbot" not in (p1, p2) or label not in (p1, p2):
            continue
        model_is_p1 = (p1 == label)
        if w == "draw":
            s = 0.5
        elif (w == "p1" and model_is_p1) or (w == "p2" and not model_is_p1):
            s = 1.0
        else:
            s = 0.0
        out.append((s, tuple(tuple(m) for m in g.get("moves", []))))
    return out


def wr_ci(scored):
    """WR + distinct-game bootstrap 95% CI + distinct-n."""
    if not scored:
        return None
    # dedupe by move-seq: keep one score per distinct game (mean if a seq repeats)
    by_key = defaultdict(list)
    for s, k in scored:
        by_key[k].append(s)
    distinct = [sum(v) / len(v) for v in by_key.values()]
    n_raw, n_dist = len(scored), len(distinct)
    wr = sum(s for s, _ in scored) / n_raw
    boots = []
    for _ in range(N_BOOT):
        sample = [distinct[random.randrange(n_dist)] for _ in range(n_dist)]
        boots.append(sum(sample) / n_dist)
    boots.sort()
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]
    return {"wr": wr, "ci_lo": lo, "ci_hi": hi, "n": n_raw, "n_distinct": n_dist,
            "copy_mult": n_raw / n_dist if n_dist else 0}


def load(cell):
    p = os.path.join(OUT, cell, "sealbot_games.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    labels = ["s60k", "s120k", "s150k", "s175k", "s200k", "s226k"]
    table = {}          # cell -> label -> wr_ci
    for cell in CELLS:
        games = load(cell)
        if games is None:
            continue
        table[cell] = {lab: wr_ci(model_scores(games, lab)) for lab in labels}

    lines = ["# D-LOCALIZE P3 — search-scaling verdict\n",
             "Model WR vs fixed-depth SealBot (depth-5 unless noted), distinct-game "
             "bootstrap 95% CI. Decision-relevant comparison = n=150 -> n=256.\n"]
    for cell in CELLS:
        if cell not in table:
            lines.append(f"## {cell}: (no data yet)\n")
            continue
        lines.append(f"## {cell}")
        lines.append("| ckpt | WR | 95% CI | n | distinct | copy_mult |")
        lines.append("|---|---|---|---|---|---|")
        for lab in labels:
            r = table[cell].get(lab)
            if not r:
                lines.append(f"| {lab} | — | — | 0 | 0 | — |")
            else:
                flag = " ⚠low-n" if r["n_distinct"] < 10 else ""
                lines.append(f"| {lab} | {r['wr']:.3f} | [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] | "
                             f"{r['n']} | {r['n_distinct']}{flag} | {r['copy_mult']:.2f} |")
        lines.append("")

    # Pre-registered gate: 150 -> 256
    verdict = "INSUFFICIENT (n150 or n256 missing)"
    climbers = []
    if "n150" in table and "n256" in table:
        for lab in labels:
            a, b = table["n150"].get(lab), table["n256"].get(lab)
            if a and b and b["ci_lo"] > a["ci_hi"]:
                climbers.append(f"{lab}: 256 CI_lo {b['ci_lo']:.3f} > 150 CI_hi {a['ci_hi']:.3f}")
        verdict = (f"CLIMBS-past-256 (adopt n=256): " + "; ".join(climbers)) if climbers \
                  else "PLATEAU-by-150 (keep n=150): no checkpoint has CI_lo(256) > CI_hi(150)"
    lines.append("## PRE-REGISTERED GATE (150 -> 256)")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("Predicted PLATEAU (D-LOCALIZE P2: the gap is VALUE-BLINDNESS, not "
                 "search budget — more sims search deeper but the value head still "
                 "mis-evaluates the leaves). A CLIMBS result would instead implicate "
                 "phase-1 SH pruning at 2 visits/candidate (n=150).")
    # depth-4 red-team note
    if "redteam_d4_n256" in table and "n256" in table:
        lines.append("\n## RED-TEAM (depth-4 vs depth-5 at n=256) — bar-specificity")
        lines.append("Compare s150k/s200k WR at depth-4 (redteam_d4_n256) vs depth-5 (n256) "
                     "above; matching curve shape => bar-independent.")

    out = os.path.join(OUT, "P3_VERDICT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[p3_verdict] -> {out}")


if __name__ == "__main__":
    main()
