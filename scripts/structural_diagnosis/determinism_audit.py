#!/usr/bin/env python3
"""§D-RECONVERGE Phase 1b — determinism blast-radius audit (the 65/4068-class sweep).

The §D-GLOBALCONC review found `get_threats()` enumeration order is unstable, making the
completing-cell selection `pair[1]` (hence `winning_turn_cells` and the off-window classification)
non-deterministic run-to-run (65/4068 mismatches), fixed by sorting `depth2_wins`. This audit
REPEAT-CALLS the whole LIVE offline measurement chain on the real golong replay pool and counts any
remaining run-to-run mismatches — to confirm the fix closed the only live-chain leak and nothing else
in the chain jitters.

EVAL-ONLY, read-only, CPU-only. Run:
  cd $REPO_ROOT && PYTHONPATH=. .venv/bin/python \
    scripts/structural_diagnosis/determinism_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board  # noqa: E402
from hexo_rl.diagnostics.forced_win_detector import (  # noqa: E402
    analyze_recorded_game, cheb, depth1_wins, depth2_wins, is_off_window,
    window_center, winning_turn_cells,
)
from hexo_rl.encoding import lookup, normalize_encoding_name  # noqa: E402

NAME = normalize_encoding_name("v6_live2")
SPEC = lookup(NAME)
REPLAYS = sorted((REPO / "investigation/coherence_2026-06-08/replays").glob("games_2026-06-0*.jsonl"))
REPEATS = 8
MAX_GAMES = 1500


def _load_games():
    out = []
    for fn in REPLAYS:
        for line in open(fn):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            mv = [(int(q), int(r)) for (q, r) in d["moves"]]
            out.append((mv, d.get("outcome", ""), int(d.get("checkpoint_step", 0))))
            if len(out) >= MAX_GAMES:
                return out
    return out


def _turn_signature(mv):
    """Per-game list of (winning_turn_cells frozenset, binding cell, off_window flag) at each
    mover turn-start — the exact quantities the live off_window_forced_win_rate binds on."""
    sig = []
    board = Board.with_encoding_name(NAME)
    i, n = 0, len(mv)
    while i < n:
        mover = int(board.current_player)
        snap = board.clone()
        W = winning_turn_cells(snap, mover)
        if W:
            center = window_center([(int(s[0]), int(s[1])) for s in snap.get_stones()])
            Wl = sorted(W)
            binding = max(Wl, key=lambda c: cheb(c, center))
            sig.append((frozenset(W), binding, bool(is_off_window(snap, binding, SPEC))))
        while i < n:
            q, r = mv[i]
            try:
                board.apply_move(q, r)
            except Exception:
                i = n
                break
            i += 1
            if board.check_win():
                break
            if int(board.current_player) != mover:
                break
    return tuple(sig)


def _d1d2_set_signature(mv):
    """depth1_wins (as a SET) and depth2_wins (as a sorted-key SET) at each turn-start — the
    upstream primitives, to confirm the set is stable even where order varies."""
    sig = []
    board = Board.with_encoding_name(NAME)
    i, n = 0, len(mv)
    while i < n:
        mover = int(board.current_player)
        snap = board.clone()
        d1 = frozenset((int(c[0]), int(c[1])) for c in depth1_wins(snap, mover))
        d2 = frozenset(tuple(sorted((tuple(map(int, f)), tuple(map(int, s)))))
                       for (f, s) in depth2_wins(snap, mover))
        if d1 or d2:
            sig.append((d1, d2))
        while i < n:
            q, r = mv[i]
            try:
                board.apply_move(q, r)
            except Exception:
                i = n
                break
            i += 1
            if board.check_win():
                break
            if int(board.current_player) != mover:
                break
    return tuple(sig)


def main():
    games = _load_games()
    print(f"[cfg] encoding={NAME} games={len(games)} repeats={REPEATS}", flush=True)

    # 1) analyze_recorded_game (the LIVE ForcedWinTrend metric) — both engine sides.
    p0 = int(Board.with_encoding_name(NAME).current_player)
    sides = (p0, -p0)
    arg_mismatch = 0
    arg_total = 0
    for (mv, outcome, _src) in games:
        for side in sides:
            ref = None
            for _ in range(REPEATS):
                s = analyze_recorded_game(mv, outcome, encoding=NAME, mover_side=side)
                tup = (s.forced_win_turns, s.off_window_forced_turns, s.converted)
                if ref is None:
                    ref = tup
                else:
                    arg_total += 1
                    if tup != ref:
                        arg_mismatch += 1
    print(f"[1] analyze_recorded_game (forced/off_window/converted): "
          f"{arg_mismatch}/{arg_total} repeat-pair mismatches", flush=True)

    # 2) winning_turn_cells + binding + off-window (the directly-fixed primitive) at scale.
    sig_mismatch = 0
    sig_total = 0
    turn_starts = 0
    for (mv, _o, _s) in games:
        ref = None
        for _ in range(REPEATS):
            sig = _turn_signature(mv)
            if ref is None:
                ref = sig
                turn_starts += len(sig)
            else:
                sig_total += 1
                if sig != ref:
                    sig_mismatch += 1
    print(f"[2] winning_turn_cells+binding+off_window: {sig_mismatch}/{sig_total} repeat-pair "
          f"mismatches over {turn_starts} forced-win turn-starts", flush=True)

    # 3) depth1/depth2 SET stability (order may vary; the SET must not).
    d_mismatch = 0
    d_total = 0
    for (mv, _o, _s) in games:
        ref = None
        for _ in range(REPEATS):
            sig = _d1d2_set_signature(mv)
            if ref is None:
                ref = sig
            else:
                d_total += 1
                if sig != ref:
                    d_mismatch += 1
    print(f"[3] depth1/depth2 win SETs: {d_mismatch}/{d_total} repeat-pair mismatches", flush=True)

    # 4) coherence_decomposition.analyze_game_sides under BOTH units.
    from scripts.structural_diagnosis.coherence_decomposition import analyze_game_sides  # noqa: E402
    coh_mismatch = {"ply": 0, "turn": 0}
    coh_total = {"ply": 0, "turn": 0}
    for unit in ("ply", "turn"):
        for (mv, _o, _s) in games:
            ref = None
            for _ in range(REPEATS):
                gs = analyze_game_sides(mv, SPEC, NAME, unit)
                # canonicalise to a comparable tuple (per-side off-window + forced + converted tallies)
                tup = tuple(sorted(
                    (k, v.get("forced"), v.get("off_window"), v.get("converted"))
                    for k, v in gs.items()))
                if ref is None:
                    ref = tup
                else:
                    coh_total[unit] += 1
                    if tup != ref:
                        coh_mismatch[unit] += 1
        print(f"[4] coherence analyze_game_sides --unit {unit}: "
              f"{coh_mismatch[unit]}/{coh_total[unit]} repeat-pair mismatches", flush=True)

    total_chain = (arg_mismatch + sig_mismatch + d_mismatch
                   + coh_mismatch['ply'] + coh_mismatch['turn'])
    print(f"\n{'='*80}")
    print(f"LIVE-CHAIN DETERMINISM: {total_chain} total mismatches "
          f"({'CLEAN' if total_chain == 0 else 'LEAK FOUND'})")
    print(f"{'='*80}")
    out = {
        "games": len(games), "repeats": REPEATS, "turn_starts": turn_starts,
        "analyze_recorded_game_mismatch": arg_mismatch, "analyze_recorded_game_total": arg_total,
        "winning_turn_sig_mismatch": sig_mismatch, "winning_turn_sig_total": sig_total,
        "depth_set_mismatch": d_mismatch, "depth_set_total": d_total,
        "coherence_mismatch": coh_mismatch, "coherence_total": coh_total,
        "live_chain_total_mismatch": total_chain,
        "verdict": "CLEAN" if total_chain == 0 else "LEAK FOUND",
    }
    outp = REPO / "investigation/reconverge_2026-06-08/determinism_audit.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"[out] {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
