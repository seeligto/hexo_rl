#!/usr/bin/env python3
"""§D-OVERSPREAD D5 follow-on — is over-spread PUNISHED INTERNALLY in self-play? (Leg A)

D5's core claim: self-play is spread-vs-spread, so over-spread is never punished INTERNALLY
(only externally by a compact finisher). The §D-OVERSPREAD discriminator left D5 INCONCLUSIVE
(Part-2 = "SealBot losses are spread-force" instrument-blocked). This is the FAITHFUL, immediately
feasible half: the banked self-play replays ARE the real spread regime (single-window v6_live2
self-play, both sides the same model). Test directly whether, WITHIN a self-play game, the
MORE-FRAGMENTED side loses (spread punished internally) or fragmentation is outcome-NEUTRAL (spread
not punished internally -> the D5 signature, the precondition for a compact-reference fix).

Method (pure board replay, no NN, reuses coherence_overspread structure metrics): for each
NON-DRAW banked game, replay to a matched cut ply (a FIXED FRACTION of game length, BEFORE the
decisive finish so the winner's compact final line doesn't contaminate the comparison). At the cut,
compute BOTH sides' own-force fragmentation (components-per-stone primary; ncomp, largest-blob-frac
corroborate). Compare loser vs winner: delta = frag(loser) - frag(winner). Bucket by
checkpoint_step; cluster-bootstrap CI over games.

  spread PUNISHED internally  <=> delta > 0 CI-cleared (loser more fragmented) AND
                                  P(more-fragmented side loses) > 0.5 CI-cleared.
  spread NEUTRAL internally    <=> delta ~ 0 (CI straddles) AND P ~ 0.5  -> D5 precondition HOLDS.

EVAL-ONLY, read-only. Zero geometry literals (hex adjacency = HEX_AXES). winner via board.winner()
(robust; does not rely on the o_win/x_win string mapping).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from engine import Board
from hexo_rl.diagnostics.forced_win_detector import HEX_AXES
from hexo_rl.encoding import lookup, normalize_encoding_name

_NB = HEX_AXES + [(-q, -r) for (q, r) in HEX_AXES]


def _components(cells):
    seen, out = set(), []
    for s in cells:
        if s in seen:
            continue
        q = deque([s]); seen.add(s); sz = 0
        while q:
            c = q.popleft(); sz += 1
            for dq, dr in _NB:
                nx = (c[0] + dq, c[1] + dr)
                if nx in cells and nx not in seen:
                    seen.add(nx); q.append(nx)
        out.append(sz)
    return out


def _frag(cells):
    """Return (components_per_stone, ncomp, largest_blob_frac) for a set of own cells."""
    if not cells:
        return None
    comps = _components(cells)
    n = len(cells)
    return (len(comps) / n, len(comps), max(comps) / n)


def _winner_of(mv, name):
    b = Board.with_encoding_name(name)
    for (q, r) in mv:
        try:
            b.apply_move(q, r)
        except Exception:
            return None
        if b.check_win():
            return int(b.winner())
    return None


def _frag_at_cut(mv, name, cut_frac):
    """Replay to floor(cut_frac * len) and return {side: (cps, ncomp, blob)} for both sides."""
    cut = int(cut_frac * len(mv))
    if cut < 6:
        return None
    b = Board.with_encoding_name(name)
    for i in range(cut):
        try:
            b.apply_move(*mv[i])
        except Exception:
            return None
        if b.check_win():           # decisive before the cut -> too short to compare pre-finish
            return None
    stones = [(int(q), int(r), int(p)) for (q, r, p) in b.get_stones()]
    out = {}
    for side in (1, -1):
        mine = {(q, r) for (q, r, p) in stones if p == side}
        f = _frag(mine)
        if f is None:
            return None
        out[side] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--steps", type=int, nargs="+", default=[30000, 53000, 87500])
    ap.add_argument("--cut-fracs", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--out", default="investigation/overspread_2026-06-08/d5_internal_punish.json")
    args = ap.parse_args()
    name = normalize_encoding_name(args.encoding)
    _ = lookup(name)
    want = set(args.steps)
    rng = np.random.default_rng(args.seed)

    # per (step, cut_frac): list of (delta_cps, delta_ncomp, delta_blob, more_frag_side_lost)
    rows = defaultdict(lambda: defaultdict(list))
    for fn in args.files:
        try:
            fh = open(fn)
        except FileNotFoundError:
            print(f"  (skip missing {fn})", file=sys.stderr); continue
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("game_length", 0) <= 0:
                continue
            s = int(d.get("checkpoint_step", 0))
            if s not in want:
                continue
            mv = [(int(q), int(r)) for (q, r) in d["moves"]]
            w = _winner_of(mv, name)
            if w not in (1, -1):       # draw or unparseable -> internal-punishment test needs a decisive game
                continue
            loser = -w
            for cf in args.cut_fracs:
                fr = _frag_at_cut(mv, name, cf)
                if fr is None:
                    continue
                d_cps = fr[loser][0] - fr[w][0]        # loser fragmentation - winner fragmentation
                d_nc = fr[loser][1] - fr[w][1]
                d_bl = fr[loser][2] - fr[w][2]         # blob: winner more compact -> winner blob higher -> d_bl<0
                # "more-fragmented side lost": loser has higher components-per-stone
                more_frag_lost = 1.0 if fr[loser][0] > fr[w][0] else (0.5 if fr[loser][0] == fr[w][0] else 0.0)
                rows[(s, cf)]["d_cps"].append(d_cps)
                rows[(s, cf)]["d_ncomp"].append(d_nc)
                rows[(s, cf)]["d_blob"].append(d_bl)
                rows[(s, cf)]["mfl"].append(more_frag_lost)

    def bci(vals):
        a = np.asarray(vals, float)
        if len(a) == 0:
            return (float("nan"), float("nan"), float("nan"))
        bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(args.nboot)]
        return (float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

    out = {"encoding": name, "cut_fracs": args.cut_fracs, "buckets": []}
    print("§D-OVERSPREAD D5 Leg A — is over-spread PUNISHED INTERNALLY in self-play?")
    print("delta = frag(LOSER) - frag(WINNER) at a matched pre-finish cut ply.  >0 => spread punished; ~0 => NEUTRAL (D5 precondition)")
    hdr = (f"{'step':>7} {'cut':>4} {'n':>5} | {'Δ comp/stone [95% CI]':>30} | "
           f"{'Δ ncomp':>16} | {'P(more-frag side LOST)':>24}")
    print(hdr); print("-" * len(hdr))
    for (s, cf) in sorted(rows):
        r = rows[(s, cf)]
        n = len(r["d_cps"])
        if n < 15:
            continue
        m_cps, lo_cps, hi_cps = bci(r["d_cps"])
        m_nc, lo_nc, hi_nc = bci(r["d_ncomp"])
        m_mfl, lo_mfl, hi_mfl = bci(r["mfl"])
        cleared = "PUNISHED" if lo_cps > 0 else ("NEUTRAL" if (lo_cps <= 0 <= hi_cps) else "INVERTED")
        out["buckets"].append({"step": s, "cut_frac": cf, "n": n,
                               "delta_comp_per_stone": m_cps, "ci_cps": [lo_cps, hi_cps],
                               "delta_ncomp": m_nc, "ci_ncomp": [lo_nc, hi_nc],
                               "p_more_frag_lost": m_mfl, "ci_p": [lo_mfl, hi_mfl],
                               "internal_verdict": cleared})
        print(f"{s:>7} {cf:>4.2f} {n:>5} | {m_cps:>+10.4f} [{lo_cps:>+7.4f},{hi_cps:>+7.4f}] | "
              f"{m_nc:>+7.3f} [{lo_nc:>+5.2f},{hi_nc:>+5.2f}] | "
              f"{m_mfl:>8.3f} [{lo_mfl:.3f},{hi_mfl:.3f}]  {cleared}")
    print("\nINTERPRET: NEUTRAL (Δ~0, P~0.5) across the arc => over-spread is NOT punished internally")
    print("           in spread-vs-spread self-play => D5's 'only a compact finisher punishes it' precondition HOLDS")
    print("           => a compact-reference self-play regularizer (Branch 2) is the indicated internal counter-pressure.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
