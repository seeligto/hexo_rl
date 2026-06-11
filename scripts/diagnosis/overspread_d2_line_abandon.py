#!/usr/bin/env python3
"""§D-OVERSPREAD D2 — OFF-WINDOW STRUCTURE BIASES PLAY (a SPREAD driver, NOT conversion).

V-OFFWINDOW (the §D-COHERENCE arm) refuted off-window as the CONVERSION driver; it did NOT
test off-window-STRUCTURE as a SPREAD driver. HYPOTHESIS (D2): committing to a directed own
line risks an off-window completion (no policy target / no reward), so the model keeps its
force short / scattered to stay in-window. If that drives the over-spread, then over the
training arc the model should ABANDON its own developing lines MORE, and the abandonment
should CONCENTRATE near the window boundary (lines whose extension would push toward / past
the off-window frontier).

PURE board replay, NO NN, on the banked golong self-play replays, bucketed by checkpoint_step
(30k / 53k / 87.5k). Zero geometry literals: hex axes from forced_win_detector.HEX_AXES,
off-window from is_off_window (spec.policy_logit_count), window center from window_center.

DEFINITIONS (locked by the dispatcher / PREREGISTRATION D2 lighting):
  * "developing line" = a maximal own near-collinear run along a HEX_AXES direction,
    length >= 3, that has >= 1 OPEN LEGAL extension cell (a run with no legal extension is
    already blocked -> not "developing", excluded).
  * For each developing line we look at its two axis extension cells (one off each end).
    The "leading extension cell" = the legal extension furthest from the window center
    (the cell that pushes the line OUTWARD, toward the frontier) -- mirrors the detector's
    max-chebyshev binding-cell convention. If only one end is legal, that end leads.
  * "window-boundary proximity" of a developing line = is the leading extension cell
    OFF-WINDOW (is_off_window) OR within FRONTIER_BAND of the off-window frontier?
    A line is BOUNDARY-adjacent if leading extension is off-window OR its chebyshev distance
    to the nearest off-window legal frontier cell <= FRONTIER_BAND; else INTERIOR.
  * "abandonment" of a developing line this turn = the mover played NONE of that line's legal
    extension cells on its turn. Turn-level abandonment = the mover extended NONE of its
    developing lines while >= 1 legal IN-WINDOW extension existed (i.e. it could have
    extended in-window and chose not to).

MEASURES over the arc:
  (a) developing-line abandonment RATE (per developing line; and per turn) -- does it rise?
  (b) abandonment rate of BOUNDARY-adjacent developing lines vs INTERIOR lines -- and does
      the boundary>interior gap RISE over the arc?

LIT iff developing-line abandonment RISES over the arc AND concentrates near the window
boundary (boundary abandonment rate > interior, rising). Flat OR not boundary-correlated
-> OUT / INCONCLUSIVE. (PREREGISTRATION D2.)

EVAL-ONLY / read-only. Writes ONLY this new script + a JSON under investigation/overspread_*/.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from engine import Board
from hexo_rl.diagnostics.forced_win_detector import (
    HEX_AXES, cheb, is_off_window, window_center,
)
from hexo_rl.encoding import lookup, normalize_encoding_name

MIN_LINE_LEN = 3       # "developing line" minimum own collinear run length (dispatcher)
FRONTIER_BAND = 1      # chebyshev band counted as "near" the off-window frontier (boundary)
N_BOOT = 2000          # bootstrap resamples (game-side cluster unit)


def _maximal_runs(mine: set, axis: tuple) -> list:
    """All maximal collinear own-stone runs along `axis` (length >= MIN_LINE_LEN).

    A run is a maximal sequence of own cells c, c+axis, c+2*axis, ...  We find each run's
    head (a cell whose predecessor c-axis is NOT own) and walk forward. Returns list of
    (run_cells:list, head:cell, tail:cell) where head is the lowest-along-axis end and tail
    the highest.
    """
    dq, dr = axis
    runs = []
    for c in mine:
        prev = (c[0] - dq, c[1] - dr)
        if prev in mine:
            continue  # not a run head
        run = [c]
        nx = (c[0] + dq, c[1] + dr)
        while nx in mine:
            run.append(nx)
            nx = (nx[0] + dq, nx[1] + dr)
        if len(run) >= MIN_LINE_LEN:
            runs.append((run, run[0], run[-1], axis))
    return runs


def _developing_lines(mine: set, legal: set, snap, spec, center):
    """Build the developing-line records for the mover at this snapshot.

    Each record:
      head_ext / tail_ext : the two axis extension cells (head-axis, tail+axis)
      legal_exts          : extension cells that are legal at this snapshot
      leading_ext         : the legal extension furthest from window center (outward)
      boundary            : leading_ext off-window OR within FRONTIER_BAND of an off-window
                            legal frontier cell
      has_inwindow_ext    : >= 1 legal extension that is IN-window
    Only lines with >= 1 legal extension are returned (= "developing", has an open extension).
    """
    # off-window legal frontier cells (for the FRONTIER_BAND proximity test)
    ow_frontier = [c for c in legal if is_off_window(snap, c, spec)]
    lines = []
    for axis in HEX_AXES:
        dq, dr = axis
        for run, head, tail, ax in _maximal_runs(mine, axis):
            head_ext = (head[0] - dq, head[1] - dr)
            tail_ext = (tail[0] + dq, tail[1] + dr)
            legal_exts = [e for e in (head_ext, tail_ext) if e in legal]
            if not legal_exts:
                continue  # blocked both ends -> not developing
            # leading = legal extension furthest from window center (pushes the line outward)
            leading = max(legal_exts, key=lambda e: cheb(e, center))
            lead_off = is_off_window(snap, leading, spec)
            # boundary: leading off-window, OR within FRONTIER_BAND of an off-window frontier cell
            near_frontier = any(cheb(leading, f) <= FRONTIER_BAND for f in ow_frontier)
            boundary = bool(lead_off or near_frontier)
            inwin_exts = [e for e in legal_exts if not is_off_window(snap, e, spec)]
            lines.append({
                "legal_exts": legal_exts,
                "leading": leading,
                "boundary": boundary,
                "lead_off": bool(lead_off),
                "has_inwin_ext": bool(inwin_exts),
                "len": len(run),
            })
    return lines


def _walk_turn(b, mv, i, cp):
    """Apply the mover's turn stones; return (played_cells, new_i, won)."""
    n = len(mv)
    played = []
    won = False
    while i < n:
        q, r = mv[i]
        try:
            b.apply_move(q, r)
        except Exception:
            return played, n, won
        i += 1
        played.append((q, r))
        if b.check_win():
            won = True
            break
        if int(b.current_player) != cp:
            break
    return played, i, won


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--steps", type=int, nargs="+", default=[30000, 53000, 87500])
    ap.add_argument("--out", default="investigation/overspread_2026-06-08/d2_line_abandon.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    want = set(args.steps)

    # Per (step) we collect per-turn records. Cluster bootstrap unit = game-side
    # (one game replayed for one mover side). Each turn contributes line records + a
    # turn-abandonment flag. We key game-side rows for the bootstrap.
    # rows[step] = list of game-side dicts:
    #   {"turns":int, "abandon_turns":int,
    #    "lines":int, "ab_lines":int,
    #    "b_lines":int, "b_ab":int, "i_lines":int, "i_ab":int}
    rows = defaultdict(list)

    for fn in args.files:
        try:
            fh = open(fn)
        except FileNotFoundError:
            print(f"  (skip missing {fn})", file=sys.stderr)
            continue
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
            # one fresh board per (game); both sides walked together but tallied per-side
            # to keep game-side as the cluster unit, accumulate per-side rows.
            b = Board.with_encoding_name(name)
            # determine the two engine sides from the opening player
            side_acc = {}  # cp -> accumulator dict
            i, n = 0, len(mv)
            while i < n:
                cp = int(b.current_player)
                snap = b.clone()
                stones = [(int(q), int(r), int(p)) for (q, r, p) in snap.get_stones()]
                mine = {(q, r) for (q, r, p) in stones if p == cp}
                legal = {(int(q), int(r)) for (q, r) in snap.legal_moves()}
                center = window_center([(q, r) for (q, r, _p) in stones])
                lines = _developing_lines(mine, legal, snap, spec, center)
                played, i, _won = _walk_turn(b, mv, i, cp)
                played_set = set(played)

                acc = side_acc.setdefault(cp, dict(
                    turns=0, abandon_turns=0, eligible_turns=0,
                    lines=0, ab_lines=0,
                    b_lines=0, b_ab=0, i_lines=0, i_ab=0,
                ))
                if not lines:
                    continue  # no developing line this turn -> not an abandonment opportunity
                acc["turns"] += 1
                # per-line abandonment: line abandoned if NONE of its legal exts were played
                turn_extended_any = False
                turn_has_inwin = False
                for ln in lines:
                    acc["lines"] += 1
                    extended = any(e in played_set for e in ln["legal_exts"])
                    if extended:
                        turn_extended_any = True
                    else:
                        acc["ab_lines"] += 1
                    if ln["has_inwin_ext"]:
                        turn_has_inwin = True
                    if ln["boundary"]:
                        acc["b_lines"] += 1
                        if not extended:
                            acc["b_ab"] += 1
                    else:
                        acc["i_lines"] += 1
                        if not extended:
                            acc["i_ab"] += 1
                # turn-level abandonment: extended NONE while a legal IN-WINDOW ext existed
                if turn_has_inwin:
                    acc["eligible_turns"] += 1
                    if not turn_extended_any:
                        acc["abandon_turns"] += 1
            for cp, acc in side_acc.items():
                if acc["lines"] > 0 or acc["turns"] > 0:
                    rows[s].append(acc)

    rng = np.random.default_rng(args.seed)

    def _rate(rs, num_key, den_key):
        num = sum(r[num_key] for r in rs)
        den = sum(r[den_key] for r in rs)
        return (num / den) if den else float("nan"), num, den

    def _boot_rate(rs, num_key, den_key):
        if not rs:
            return (float("nan"), float("nan"))
        idx = np.arange(len(rs))
        vals = []
        for _ in range(N_BOOT):
            samp = rng.choice(idx, size=len(idx), replace=True)
            num = sum(rs[j][num_key] for j in samp)
            den = sum(rs[j][den_key] for j in samp)
            if den:
                vals.append(num / den)
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    def _boot_gap(rs):
        """Bootstrap CI of (boundary_ab_rate - interior_ab_rate)."""
        if not rs:
            return (float("nan"), float("nan"))
        idx = np.arange(len(rs))
        vals = []
        for _ in range(N_BOOT):
            samp = rng.choice(idx, size=len(idx), replace=True)
            bn = sum(rs[j]["b_ab"] for j in samp); bd = sum(rs[j]["b_lines"] for j in samp)
            inn = sum(rs[j]["i_ab"] for j in samp); idn = sum(rs[j]["i_lines"] for j in samp)
            if bd and idn:
                vals.append(bn / bd - inn / idn)
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    out = {"encoding": name, "min_line_len": MIN_LINE_LEN, "frontier_band": FRONTIER_BAND,
           "buckets": []}
    print("D2 developing-line abandonment over the arc "
          "(pure board replay, no NN):\n")
    hdr = (f"{'step':>7} {'gs':>4} {'turns':>6} {'lines':>6} "
           f"{'ab_line_rate':>12} {'turn_ab_rate':>12} "
           f"{'bound_ab':>10} {'inter_ab':>10} {'b-i_gap':>9} {'bound_share':>11}")
    print(hdr)
    for s in sorted(rows):
        rs = rows[s]
        gs = len(rs)
        line_rate, ab_n, ab_d = _rate(rs, "ab_lines", "lines")
        turn_rate, t_n, t_d = _rate(rs, "abandon_turns", "eligible_turns")
        b_rate, bn, bd = _rate(rs, "b_ab", "b_lines")
        i_rate, inn, idn = _rate(rs, "i_ab", "i_lines")
        gap = (b_rate - i_rate) if (bd and idn) else float("nan")
        bound_share = bd / (bd + idn) if (bd + idn) else float("nan")
        lr_ci = _boot_rate(rs, "ab_lines", "lines")
        tr_ci = _boot_rate(rs, "abandon_turns", "eligible_turns")
        gap_ci = _boot_gap(rs)
        out["buckets"].append({
            "step": s, "game_sides": gs,
            "n_turns_with_devline": t_d, "n_devlines": ab_d,
            "ab_line_rate": line_rate, "ab_line_rate_ci": list(lr_ci),
            "turn_ab_rate": turn_rate, "turn_ab_rate_ci": list(tr_ci),
            "eligible_turns": t_d,
            "boundary_ab_rate": b_rate, "boundary_lines": bd,
            "interior_ab_rate": i_rate, "interior_lines": idn,
            "boundary_minus_interior_gap": gap, "gap_ci": list(gap_ci),
            "boundary_share_of_lines": bound_share,
        })
        print(f"{s:>7} {gs:>4} {t_d:>6} {ab_d:>6} "
              f"{line_rate:>12.4f} {turn_rate:>12.4f} "
              f"{b_rate:>10.4f} {i_rate:>10.4f} {gap:>9.4f} {bound_share:>11.4f}")

    # arc trend summary (30k -> 87.5k) on the canonical 3 buckets
    bk = {b["step"]: b for b in out["buckets"]}
    if 30000 in bk and 87500 in bk:
        a, z = bk[30000], bk[87500]
        out["arc"] = {
            "d_ab_line_rate": z["ab_line_rate"] - a["ab_line_rate"],
            "d_turn_ab_rate": z["turn_ab_rate"] - a["turn_ab_rate"],
            "d_boundary_minus_interior_gap": (z["boundary_minus_interior_gap"]
                                              - a["boundary_minus_interior_gap"]),
            "gap_30k": a["boundary_minus_interior_gap"],
            "gap_87k": z["boundary_minus_interior_gap"],
            "gap_30k_ci": a["gap_ci"], "gap_87k_ci": z["gap_ci"],
        }
        print("\nARC 30k -> 87.5k:")
        print(f"  d(ab_line_rate)       = {out['arc']['d_ab_line_rate']:+.4f}")
        print(f"  d(turn_ab_rate)       = {out['arc']['d_turn_ab_rate']:+.4f}")
        print(f"  boundary-interior gap : {a['boundary_minus_interior_gap']:+.4f} "
              f"(CI {a['gap_ci'][0]:+.3f},{a['gap_ci'][1]:+.3f}) -> "
              f"{z['boundary_minus_interior_gap']:+.4f} "
              f"(CI {z['gap_ci'][0]:+.3f},{z['gap_ci'][1]:+.3f})")
        print(f"  d(boundary-interior)  = {out['arc']['d_boundary_minus_interior_gap']:+.4f}")
    print("\nLIT iff: abandonment RISES over the arc AND concentrates near the boundary "
          "(boundary_ab > interior_ab, gap RISING). Flat or not-boundary -> OUT/INCONCLUSIVE.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
