#!/usr/bin/env python3
"""Turn-level winning-opportunity primitive for §D-OVERSPREAD.

WHY this exists (operator note, 2026-06-08): `count_winning_moves` (Rust,
`engine/src/board/moves.rs:305`) counts *single-stone* 6-completions — the DEPTH-1
notion. But Hex Tac Toe turns place **2 stones** (after the opening single move), so a win
the side can complete with BOTH stones this turn (`depth2_wins`) is also a win *this turn*.
Counting only depth-1 undercounts the real "can I finish this turn" set. The §D-COHERENCE
scripts are inconsistent about this (one uses the depth-2 pair's first cell `f`, one
flattens the pair, one uses `f`) — this primitive unifies them.

DESIGN — turn-native, not depth-parametrized. The completing cell of a winning turn is the
FINAL stone played:
  - depth-1 win: the single stone completes  -> the cell itself.
  - depth-2 win: the pair (f, s) completes on the second stone  -> `s` (for an immediate
    first-stone win `depth2_wins` returns (f, f), so the completing cell is still pair[1]).
So:
    winning_turn_cells(board, side) = { d1 } ∪ { pair[1] for pair in d2 }
    count_winning_turns(board, side) = |winning_turn_cells(board, side)|
Turn-awareness is automatic: `depth2_wins` is guarded on `moves_remaining >= 2`, so when the
side is on the last stone of its turn (moves_remaining == 1) only depth-1 wins count.

FORK (turn-correct analogue of the Rust quiescence `count_winning_moves >= 3`): the opponent's
next turn places 2 stones and can block at most 2 distinct completing cells, so
`count_winning_turns >= 3` is an (approximately) unstoppable multi-threat. We expose the raw
count + the >=3 fork flag; the independence/shared-block subtlety is a documented refinement,
not modelled here (the §D-OVERSPREAD signal is the fork-affinity TREND, not exact
unstoppability).

PURE / read-only. Builds only on the existing detector primitives — does NOT rename or mutate
them. Recommend promoting into `hexo_rl/diagnostics/forced_win_detector.py` as a deliberate
follow-up (its natural home) once the eval-only phase closes; kept here to keep the Phase-A
git-diff clean.
"""
from __future__ import annotations

# PROMOTED 2026-06-08 (§D-GLOBALCONC Phase 2a): the canonical home of these turn-correct
# primitives is now hexo_rl/diagnostics/forced_win_detector.py. This module is kept as a thin
# re-export shim so the §D-OVERSPREAD investigation scripts that `from turn_wins import ...`
# continue to resolve to the SAME single implementation (no metric can drift between copies).
from hexo_rl.diagnostics.forced_win_detector import (  # noqa: F401
    FORK_THRESHOLD,
    count_winning_turns,
    depth1_wins,
    depth2_wins,
    is_fork_turn,
    winning_turn_cells,
)


if __name__ == "__main__":
    # Self-test: count_winning_turns >= len(depth1_wins) always (depth-2 only ADDS), and the
    # turn-guard holds (moves_remaining==1 -> depth-2 contributes nothing). Replay a few banked
    # games and assert the invariants on every turn-start snapshot.
    import json
    import sys
    from pathlib import Path

    from engine import Board

    files = sys.argv[1:] or sorted(
        str(p) for p in Path("investigation/coherence_2026-06-08/replays").glob("games_2026-06-0*.jsonl")
    )
    name = "v6_live2"
    n_snap = 0
    n_d1_only = 0
    n_turn_more = 0       # turns where depth-2 strictly adds completing cells
    max_wt = 0
    n_fork = 0
    viol = 0
    for fn in files:
        try:
            fh = open(fn)
        except FileNotFoundError:
            continue
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("game_length", 0) <= 0:
                continue
            mv = [(int(q), int(r)) for (q, r) in d["moves"]]
            b = Board.with_encoding_name(name)
            i, n = 0, len(mv)
            while i < n:
                cp = int(b.current_player)
                snap = b.clone()
                d1 = {tuple(c) for c in depth1_wins(snap, cp)}
                wt = winning_turn_cells(snap, cp)
                if d1 or wt:
                    n_snap += 1
                    if not (d1 <= wt):
                        viol += 1                      # INVARIANT: depth1 ⊆ winning_turn_cells
                    if len(wt) == len(d1):
                        n_d1_only += 1
                    if len(wt) > len(d1):
                        n_turn_more += 1
                    max_wt = max(max_wt, len(wt))
                    if len(wt) >= FORK_THRESHOLD:
                        n_fork += 1
                    # turn-guard: on the last stone of a turn, depth-2 must contribute nothing
                    if snap.moves_remaining < 2 and len(wt) != len(d1):
                        viol += 1
                while i < n:
                    q, r = mv[i]
                    try:
                        b.apply_move(q, r)
                    except Exception:
                        i = n
                        break
                    i += 1
                    if b.check_win():
                        break
                    if int(b.current_player) != cp:
                        break
    print(f"snapshots with a win-threat: {n_snap}")
    print(f"  depth-1 alone == turn-count : {n_d1_only}")
    print(f"  depth-2 STRICTLY ADDS cells : {n_turn_more}  "
          f"({100*n_turn_more/max(n_snap,1):.1f}% of threat snapshots undercounted by depth-1)")
    print(f"  max winning-turn cells      : {max_wt}")
    print(f"  fork turns (>= {FORK_THRESHOLD})           : {n_fork}")
    print(f"  INVARIANT violations        : {viol}  (must be 0: depth1 ⊆ winning_turn_cells; turn-guard)")
    sys.exit(1 if viol else 0)
