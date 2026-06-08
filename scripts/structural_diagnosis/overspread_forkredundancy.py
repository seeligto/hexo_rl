#!/usr/bin/env python3
"""§D-OVERSPREAD — TURN-vs-PLY fork-credit gap + redundancy collapse (operator's insight).

Tests two things the §D-COHERENCE over-spread WHAT did not separate, both PURE board-replay
(no NN), bucketed by source checkpoint_step over the golong arc:

(1) REDUNDANCY COLLAPSE. At each IN-WINDOW forced-win turn-start snapshot, the mover's
    turn-level winning-completion redundancy = count_winning_turns (the cells that finish the
    game THIS turn, depth-1 ∪ depth-2-second-stone). Over-spread predicts the redundancy
    distribution shifts toward 1 ("thin, single-threat" finishes) as the arc runs, even as the
    NUMBER of forced wins rises (§D-COHERENCE count 1.49->1.90). i.e. more wins, each thinner.

(2) TURN-vs-PLY CREDIT GAP. count_winning_moves (depth-1, the Rust quiescence primitive in
    engine/src/mcts/backup.rs) vs count_winning_turns (turn-correct). The fraction of
    forced-win snapshots that are turn-forced-wins but have < the quiescence fork threshold (3)
    of single-stone threats = positions where the engine's depth-1 quiescence value override
    UNDER-CREDITS the fork. If this gap is large and rises over the arc, the VALUE TARGET that
    trains the net is systematically blind to within-turn fork concentration -> a candidate
    UPSTREAM mechanism feeding D1 (value can't value concentration) + D3 (target doesn't credit
    fork-building). EVAL-ONLY measurement; does NOT change the engine.

Read-only, zero geometry literals. Reuses depth1_wins/depth2_wins + winning_turn_cells.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from engine import Board
from hexo_rl.diagnostics.forced_win_detector import depth1_wins, is_off_window
from hexo_rl.encoding import lookup, normalize_encoding_name

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turn_wins import FORK_THRESHOLD, winning_turn_cells  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--steps", type=int, nargs="+", default=[30000, 53000, 87500])
    ap.add_argument("--out", default="investigation/overspread_2026-06-08/forkredundancy.json")
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
                d1 = depth1_wins(snap, cp)
                wt = winning_turn_cells(snap, cp)
                inwin_turn = [c for c in wt if not is_off_window(snap, c, spec)]
                if inwin_turn:                                    # in-window forced win exists
                    a = agg[s]
                    n_ply = len(d1)                               # depth-1 (count_winning_moves)
                    n_turn = len(wt)                              # turn-level redundancy
                    n_turn_inwin = len(inwin_turn)
                    a["n_ply"].append(n_ply)
                    a["n_turn"].append(n_turn)
                    a["n_turn_inwin"].append(n_turn_inwin)
                    a["thin"].append(1.0 if n_turn_inwin <= 1 else 0.0)        # single-threat finish
                    a["turn_fork"].append(1.0 if n_turn >= FORK_THRESHOLD else 0.0)
                    # credit gap: a turn-forced win the depth-1 quiescence UNDER-credits
                    # (< FORK_THRESHOLD single-stone threats, so quiescence returns NN value not +1)
                    a["credit_gap"].append(1.0 if n_ply < FORK_THRESHOLD else 0.0)
                    # the strict miss: depth-1 sees NOTHING but a turn win exists (n_ply==0, n_turn>0)
                    a["ply_blind"].append(1.0 if (n_ply == 0 and n_turn > 0) else 0.0)
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

    out = {"encoding": name, "fork_threshold": FORK_THRESHOLD, "buckets": []}
    cols = ["n_ply", "n_turn", "n_turn_inwin", "thin", "turn_fork", "credit_gap", "ply_blind"]
    print("TURN-vs-PLY fork credit + redundancy at in-window forced-win snapshots:")
    hdr = f"{'src':>7} {'n':>5} " + " ".join(f"{c:>13}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for s in sorted(agg):
        a = agg[s]; nn = len(a["n_turn"])
        if nn < 10:
            continue
        means = {c: float(np.mean(a[c])) for c in cols}
        # redundancy histogram of turn-level in-window completions
        hist = np.bincount(np.array(a["n_turn_inwin"], dtype=int), minlength=6)[:6]
        rec = {"step": s, "n": nn, **means, "redundancy_hist_1to5": hist[1:6].tolist()}
        out["buckets"].append(rec)
        print(f"{s:>7} {nn:>5} " + " ".join(f"{means[c]:>13.3f}" for c in cols))
    print("\ncols: n_ply=depth-1 (count_winning_moves)  n_turn=turn-level redundancy  "
          "thin=frac single-threat finishes")
    print("      turn_fork=frac >=3 turn-completions  credit_gap=frac depth-1 quiescence "
          "UNDER-credits (n_ply<3)  ply_blind=frac depth-1 sees nothing but a turn-win exists")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
