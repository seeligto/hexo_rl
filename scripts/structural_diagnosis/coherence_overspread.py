#!/usr/bin/env python3
"""§D-COHERENCE Phase 2b — WHY do the late-arc in-window forced wins get harder to finish?

Phase 2 showed the in-window conversion drop is DISTRIBUTIONAL (the same policy/value
heads, flat across the arc; the later positions are intrinsically harder). This asks WHY
the positions are harder, on the operator's read of the games: NOT tighter coherent lines,
but the model's OWN play is OVER-SPREAD (defending + scattered attacks) so its forced wins
arise as thin, poorly-supported threats amid fragmented force.

Test: at each IN-WINDOW forced-win turn-start snapshot, characterize the MOVER's OWN stone
structure, bucketed by source checkpoint_step. Over-spread predicts: more own connected
components, smaller largest-blob fraction, lower local support around the winning cell,
falling as the arc progresses — independent of (and faster than) the rise in raw stone
count.

EVAL-ONLY, read-only on banked replays. Zero geometry literals (hex adjacency = HEX_AXES;
off-window via is_off_window). No NN, no model load — pure board replay.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from engine import Board
from hexo_rl.diagnostics.forced_win_detector import (
    HEX_AXES, depth1_wins, depth2_wins, is_off_window,
)
from hexo_rl.encoding import lookup, normalize_encoding_name

_NB = HEX_AXES + [(-q, -r) for (q, r) in HEX_AXES]   # 6-neighbour hex adjacency


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


def _bbox_area(cells):
    if not cells:
        return 1
    qs = [c[0] for c in cells]; rs = [c[1] for c in cells]
    return (max(qs) - min(qs) + 1) * (max(rs) - min(rs) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--steps", type=int, nargs="+", default=[30000, 53000, 87500])
    ap.add_argument("--out", default="investigation/coherence_2026-06-08/overspread.json")
    args = ap.parse_args()
    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    want = set(args.steps)

    agg = defaultdict(lambda: defaultdict(list))
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
            b = Board.with_encoding_name(name)
            i, n = 0, len(mv)
            while i < n:
                cp = int(b.current_player)
                snap = b.clone()
                d1 = depth1_wins(snap, cp); d2 = depth2_wins(snap, cp)
                imm = set(tuple(c) for c in d1) | set(tuple(f) for (f, _s) in d2)
                inwin = [c for c in imm if not is_off_window(snap, c, spec)]
                if inwin:
                    stones = [(int(q), int(r), int(p)) for (q, r, p) in snap.get_stones()]
                    mine = {(q, r) for (q, r, p) in stones if p == cp}
                    opp = {(q, r) for (q, r, p) in stones if p == -cp}
                    a = agg[s]
                    a["mover_stones"].append(len(mine))
                    mc = _components(mine)
                    a["mover_ncomp"].append(len(mc))
                    a["mover_largest_frac"].append(max(mc) / len(mine) if mine else 0.0)
                    a["mover_density"].append(len(mine) / _bbox_area(mine))
                    a["nwin"].append(len(inwin))
                    supp = sum(((wc[0] + dq, wc[1] + dr) in mine)
                               for wc in inwin for dq in (-1, 0, 1) for dr in (-1, 0, 1))
                    a["local_support"].append(supp / len(inwin))
                    opp_near = sum(((wc[0] + dq, wc[1] + dr) in opp)
                                   for wc in inwin for dq in (-1, 0, 1) for dr in (-1, 0, 1))
                    a["opp_near_win"].append(opp_near / len(inwin))
                while i < n:
                    q, r = mv[i]
                    try:
                        b.apply_move(q, r)
                    except Exception:
                        i = n; break
                    i += 1
                    if b.check_win():
                        break
                    if int(b.current_player) != cp:
                        break

    cols = ["mover_stones", "mover_ncomp", "mover_largest_frac", "mover_density",
            "nwin", "local_support", "opp_near_win"]
    out = {"encoding": name, "buckets": []}
    print("MOVER-OWN structure at in-window forced-win snapshots (over-spread test):")
    print(f"{'src':>7} {'n':>4} " + " ".join(f"{c:>15}" for c in cols))
    for s in sorted(agg):
        a = agg[s]; nn = len(a["mover_stones"])
        if nn < 10:
            continue
        means = {c: float(np.mean(a[c])) for c in cols}
        out["buckets"].append({"step": s, "n": nn, **means})
        print(f"{s:>7} {nn:>4} " + " ".join(f"{means[c]:>15.3f}" for c in cols))
    print("\nover-spread signature: mover_ncomp RISES + mover_largest_frac FALLS + "
          "local_support FALLS as the arc progresses, faster than mover_stones rises.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
