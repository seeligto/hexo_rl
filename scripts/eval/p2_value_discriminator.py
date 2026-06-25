#!/usr/bin/env python3
"""D-LOCALIZE P2 — corrected VALUE-vs-SEARCH discriminator (post-hoc, NO re-run).

The RED-TEAM flagged that p2_localize.classify()'s VALUE gate fires at the
*decisive* ply, which is by construction d6-WIN-side — so `net_value >= -0.05`
there merely confirms the position was winning; it does NOT diagnose a value
error. This re-classifies using the per-ply net_value ALREADY logged in
p2_decisions, evaluated at the POST-blunder model ply (the first ply where d6
flips to a terminal-persistent LOSS):

  - VALUE-BLIND : net_value(post-blunder) >= -0.05  -> the value head still thinks
                  the position is OK at a spot deep search calls lost => genuine
                  value-head failure (the value-target lever's target).
  - SAW-LOSS    : net_value(post-blunder) <  -0.05  -> the net DOES see the loss,
                  but the deploy search/policy walked into it (or it was forced)
                  => a search/policy/lookahead issue, NOT a value-head gap.
  - LINES-genuine: ref_mass(decisive) < 0.05 AND refutation_pv_turns >= 3 -> a real
                  forced line the model never put policy mass on (co-signal; these
                  all survived the d7 red-team).

Reads merged p2_decisions.jsonl if present, else auto-pools p2_decisions_s*.jsonl.
Pure analysis of logged fields; no GPU, no SealBot.
"""
from __future__ import annotations
import glob, json, os, sys
from collections import defaultdict

OUT_DIR = os.path.join("reports", "d_localize_2026-06-25")
VALUE_DRAWN = -0.05
MASS_LOW = 0.05
LINES_PV_TURNS = 3


def load_games():
    merged = os.path.join(OUT_DIR, "p2_decisions.jsonl")
    paths = [merged] if os.path.exists(merged) else sorted(
        glob.glob(os.path.join(OUT_DIR, "p2_decisions_s*.jsonl")))
    games, seen = [], set()
    for p in paths:
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            g = json.loads(ln)
            key = (g.get("bucket"), g.get("game_idx"))
            if key in seen:      # de-dupe if both merged + partials present
                continue
            seen.add(key)
            games.append(g)
    return games, paths


def classify_corrected(g):
    """Return (label, detail) for one game using post-blunder net_value."""
    di = g.get("decisive_index")
    decs = g.get("decisions") or []
    if di is None or g.get("decisive_ply") is None:
        return "ALREADY-LOST", {}
    dec = decs[di] if di < len(decs) else {}
    nv_dec = dec.get("net_value")
    # post-blunder model decision = next model ply (first persistent-LOSS ply)
    post = decs[di + 1] if di + 1 < len(decs) else None
    nv_post = post.get("net_value") if post else None
    ref_mass = dec.get("ref_mass", 1.0)
    pv_turns = g.get("refutation_pv_turns", 0) or 0
    lines_genuine = (ref_mass < MASS_LOW) and (pv_turns >= LINES_PV_TURNS)

    if nv_post is None:
        # decisive ply is the model's LAST decision -> loss is immediate/terminal.
        label = "TERMINAL-BLUNDER"
    elif nv_post >= VALUE_DRAWN:
        label = "VALUE-BLIND"
    else:
        label = "SAW-LOSS"
    return label, {
        "nv_decisive": nv_dec, "nv_post": nv_post, "ref_mass": ref_mass,
        "pv_turns": pv_turns, "lines_genuine": lines_genuine,
        "d6_score_post": (post or {}).get("d6_score"),
    }


def main():
    games, paths = load_games()
    by_bucket = defaultdict(lambda: defaultdict(int))
    lines_co = defaultdict(int)
    rows = []
    for g in games:
        b = g.get("bucket", "?")
        label, d = classify_corrected(g)
        by_bucket[b][label] += 1
        if d.get("lines_genuine") and label not in ("ALREADY-LOST",):
            lines_co[b] += 1
        rows.append((b, g.get("game_idx"), label, d))

    labels = ["VALUE-BLIND", "SAW-LOSS", "TERMINAL-BLUNDER", "ALREADY-LOST"]
    lines = []
    lines.append("# P2 corrected VALUE-vs-SEARCH discriminator (post-blunder net_value)\n")
    lines.append(f"Source: {', '.join(os.path.basename(p) for p in paths)}  "
                 f"| games pooled: {len(games)}\n")
    lines.append("Decisive = last d6-WIN ply before terminal-persistent LOSS. "
                 "Label uses net_value at the FIRST post-blunder model ply (d6=LOSS).\n")
    hdr = f"| bucket | n | " + " | ".join(labels) + " | LINES-genuine(co) |"
    sep = "|" + "---|" * (len(labels) + 3)
    lines.append(hdr); lines.append(sep)
    tot = defaultdict(int)
    for b in sorted(by_bucket):
        n = sum(by_bucket[b].values())
        cells = " | ".join(str(by_bucket[b].get(l, 0)) for l in labels)
        lines.append(f"| {b} | {n} | {cells} | {lines_co.get(b,0)} |")
        for l in labels:
            tot[l] += by_bucket[b].get(l, 0)
    n_all = sum(tot.values())
    cells = " | ".join(str(tot.get(l, 0)) for l in labels)
    lines.append(f"| **TOTAL** | {n_all} | {cells} | {sum(lines_co.values())} |")

    decisive_n = n_all - tot["ALREADY-LOST"]
    vb = tot["VALUE-BLIND"]
    sl = tot["SAW-LOSS"] + tot["TERMINAL-BLUNDER"]
    lines.append("")
    lines.append("## Corrected lever read")
    if decisive_n:
        lines.append(f"- VALUE-BLIND (value head fails to see the loss): "
                     f"{vb}/{decisive_n} = {vb/decisive_n:.0%}")
        lines.append(f"- SAW-LOSS / TERMINAL (net saw it; search/policy walked in or forced): "
                     f"{sl}/{decisive_n} = {sl/decisive_n:.0%}")
        lines.append(f"- LINES-genuine co-signal (real missed forced line, ref_mass<0.05, pv>=3): "
                     f"{sum(lines_co.values())}/{decisive_n}")
        verdict = ("VALUE-TARGET (value head blind to SealBot-reachable losses)"
                   if vb >= sl else
                   "SEARCH/POLICY (value sees loss; deploy search walks in)")
        lines.append(f"- **Corrected primary: {verdict}**")
    lines.append("")
    lines.append("## Per-game (decisive only)")
    for b, idx, label, d in rows:
        if label == "ALREADY-LOST":
            continue
        lines.append(f"- {b} idx{idx}: {label}  nv_dec={d.get('nv_decisive')} "
                     f"nv_post={d.get('nv_post')} d6_post={d.get('d6_score_post')} "
                     f"ref_mass={d.get('ref_mass'):.4f} pv_turns={d.get('pv_turns')} "
                     f"lines={d.get('lines_genuine')}")

    out = os.path.join(OUT_DIR, "p2_corrected_summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))
    print(f"\n[p2_value_discriminator] -> {out}  ({len(games)} games)")


if __name__ == "__main__":
    main()
