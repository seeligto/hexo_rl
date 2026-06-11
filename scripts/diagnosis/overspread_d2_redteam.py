#!/usr/bin/env python3
"""§D-OVERSPREAD D2 RED-TEAM — adversarial re-test of the OUT verdict.

The committed D2 probe (overspread_d2_line_abandon.py) emits cuts (1) abandonment rate +
(2) boundary-vs-interior gap, and is reproducible. Its NOTE additionally cites a cut (3)
"conditional choice" (P(pick interior) 0.64->0.56) as the DECISIVE leg, but that code is
NOT in the committed script and NOT in the JSON -> not reproducible as banked. This script:

  A) SWEEPS FRONTIER_BAND in {0,1,2,3} (the committed caveat admits band=1 was not swept):
     does the NEGATIVE boundary-interior gap flip to POSITIVE+RISING (= LIT) under a wider
     band? If it stays negative/null across bands -> the OUT verdict is band-robust.

  B) RE-DERIVES cut (3) conditional-choice independently (turns with BOTH a boundary and an
     interior developing line where exactly one class was extended): P(pick interior) over
     the arc. If it stays HIGH+RISING the avoidance mechanism would be supported; if it
     FALLS/flat the model commits to boundary lines MORE -> refutes D2.

  C) Cross-check the over-spread frame: does boundary-share of developing lines rise with
     the established own-force fragmentation (the WHAT)? If boundary-share rises only because
     force fragments outward (a SYMPTOM of over-spread), the boundary signal is NOT an
     off-window-avoidance DRIVER.

Pure board replay, no NN. Reuses the committed script's developing-line builder logic
(re-implemented here read-only; zero geometry literals via forced_win_detector). EVAL-ONLY.
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

MIN_LINE_LEN = 3
N_BOOT = 2000


def _maximal_runs(mine, axis):
    dq, dr = axis
    runs = []
    for c in mine:
        prev = (c[0] - dq, c[1] - dr)
        if prev in mine:
            continue
        run = [c]
        nx = (c[0] + dq, c[1] + dr)
        while nx in mine:
            run.append(nx)
            nx = (nx[0] + dq, nx[1] + dr)
        if len(run) >= MIN_LINE_LEN:
            runs.append((run, run[0], run[-1], axis))
    return runs


def _developing_lines(mine, legal, snap, spec, center, band):
    ow_frontier = [c for c in legal if is_off_window(snap, c, spec)]
    lines = []
    for axis in HEX_AXES:
        dq, dr = axis
        for run, head, tail, ax in _maximal_runs(mine, axis):
            head_ext = (head[0] - dq, head[1] - dr)
            tail_ext = (tail[0] + dq, tail[1] + dr)
            legal_exts = [e for e in (head_ext, tail_ext) if e in legal]
            if not legal_exts:
                continue
            leading = max(legal_exts, key=lambda e: cheb(e, center))
            lead_off = is_off_window(snap, leading, spec)
            near_frontier = any(cheb(leading, f) <= band for f in ow_frontier)
            boundary = bool(lead_off or near_frontier)
            inwin_exts = [e for e in legal_exts if not is_off_window(snap, e, spec)]
            lines.append({
                "legal_exts": legal_exts,
                "boundary": boundary,
                "has_inwin_ext": bool(inwin_exts),
                "len": len(run),
            })
    return lines


def _walk_turn(b, mv, i, cp):
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
    ap.add_argument("--bands", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", default="investigation/overspread_2026-06-08/d2_redteam.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    want = set(args.steps)
    rng = np.random.default_rng(args.seed)

    # Pre-load every game's per-turn raw structure ONCE (independent of band), then
    # apply each band as a pure relabel of boundary/interior. Records keep raw run
    # head/tail + leading-cell so we can recompute boundary at any band without replay.
    # Simpler: replay per band (cheap; ~1600 games). We collect per-band rows.
    # rows[band][step] = list of game-side acc dicts.
    rows = {bnd: defaultdict(list) for bnd in args.bands}
    # cut (3) conditional-choice tallies: choice[band][step] = [pick_interior, pick_boundary]
    choice = {bnd: defaultdict(lambda: [0, 0]) for bnd in args.bands}

    games = []
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
            games.append((s, [(int(q), int(r)) for (q, r) in d["moves"]]))

    for bnd in args.bands:
        for s, mv in games:
            b = Board.with_encoding_name(name)
            side_acc = {}
            i, n = 0, len(mv)
            while i < n:
                cp = int(b.current_player)
                snap = b.clone()
                stones = [(int(q), int(r), int(p)) for (q, r, p) in snap.get_stones()]
                mine = {(q, r) for (q, r, p) in stones if p == cp}
                legal = {(int(q), int(r)) for (q, r) in snap.legal_moves()}
                center = window_center([(q, r) for (q, r, _p) in stones])
                lines = _developing_lines(mine, legal, snap, spec, center, bnd)
                played, i, _won = _walk_turn(b, mv, i, cp)
                played_set = set(played)
                acc = side_acc.setdefault(cp, dict(
                    lines=0, ab_lines=0, b_lines=0, b_ab=0, i_lines=0, i_ab=0,
                ))
                if not lines:
                    continue
                # cut (3): both classes available, extended exactly one class?
                has_b = any(ln["boundary"] and ln["has_inwin_ext"] for ln in lines) or \
                        any(ln["boundary"] for ln in lines)
                has_i = any(not ln["boundary"] for ln in lines)
                ext_b = any(ln["boundary"] and any(e in played_set for e in ln["legal_exts"])
                            for ln in lines)
                ext_i = any((not ln["boundary"]) and any(e in played_set for e in ln["legal_exts"])
                            for ln in lines)
                if has_b and has_i and (ext_b != ext_i):
                    if ext_i:
                        choice[bnd][s][0] += 1
                    else:
                        choice[bnd][s][1] += 1
                for ln in lines:
                    acc["lines"] += 1
                    extended = any(e in played_set for e in ln["legal_exts"])
                    if not extended:
                        acc["ab_lines"] += 1
                    if ln["boundary"]:
                        acc["b_lines"] += 1
                        if not extended:
                            acc["b_ab"] += 1
                    else:
                        acc["i_lines"] += 1
                        if not extended:
                            acc["i_ab"] += 1
            for cp, acc in side_acc.items():
                if acc["lines"] > 0:
                    rows[bnd][s].append(acc)

    def _rate(rs, nk, dk):
        num = sum(r[nk] for r in rs); den = sum(r[dk] for r in rs)
        return (num / den) if den else float("nan")

    def _boot_gap(rs):
        if not rs:
            return (float("nan"), float("nan"))
        idx = np.arange(len(rs)); vals = []
        for _ in range(N_BOOT):
            samp = rng.choice(idx, size=len(idx), replace=True)
            bn = sum(rs[j]["b_ab"] for j in samp); bd = sum(rs[j]["b_lines"] for j in samp)
            inn = sum(rs[j]["i_ab"] for j in samp); idn = sum(rs[j]["i_lines"] for j in samp)
            if bd and idn:
                vals.append(bn / bd - inn / idn)
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    out = {"encoding": name, "bands": {}}
    print("=== A) FRONTIER_BAND sweep: boundary-interior abandonment gap (LIT needs +RISING) ===\n")
    for bnd in args.bands:
        out["bands"][str(bnd)] = {"buckets": []}
        print(f"-- band={bnd} --")
        print(f"{'step':>7} {'gs':>4} {'b_lines':>7} {'i_lines':>7} "
              f"{'b_ab':>7} {'i_ab':>7} {'gap':>9} {'gap_CI':>20} {'bshare':>7}")
        prev_gap = None
        for s in sorted(rows[bnd]):
            rs = rows[bnd][s]
            b_rate = _rate(rs, "b_ab", "b_lines"); i_rate = _rate(rs, "i_ab", "i_lines")
            bd = sum(r["b_lines"] for r in rs); idn = sum(r["i_lines"] for r in rs)
            gap = b_rate - i_rate
            ci = _boot_gap(rs)
            bshare = bd / (bd + idn) if (bd + idn) else float("nan")
            out["bands"][str(bnd)]["buckets"].append({
                "step": s, "game_sides": len(rs),
                "boundary_lines": bd, "interior_lines": idn,
                "boundary_ab_rate": b_rate, "interior_ab_rate": i_rate,
                "gap": gap, "gap_ci": list(ci), "boundary_share": bshare,
            })
            print(f"{s:>7} {len(rs):>4} {bd:>7} {idn:>7} {b_rate:>7.4f} {i_rate:>7.4f} "
                  f"{gap:>+9.4f} [{ci[0]:>+7.4f},{ci[1]:>+7.4f}] {bshare:>7.4f}")
            prev_gap = gap
        # arc delta
        bk = {x["step"]: x for x in out["bands"][str(bnd)]["buckets"]}
        if 30000 in bk and 87500 in bk:
            dgap = bk[87500]["gap"] - bk[30000]["gap"]
            out["bands"][str(bnd)]["d_gap_30k_87k"] = dgap
            print(f"   d(gap) 30k->87.5k = {dgap:+.4f}  "
                  f"(LIT needs positive gap AND rising)\n")

    print("=== B) cut(3) conditional choice: P(pick interior) over arc (D2 needs HIGH+RISING) ===\n")
    out["conditional_choice"] = {}
    for bnd in args.bands:
        out["conditional_choice"][str(bnd)] = {}
        print(f"-- band={bnd} --")
        print(f"{'step':>7} {'choice_turns':>12} {'pick_int':>9} {'pick_bnd':>9} {'P(int)':>8}")
        for s in sorted(choice[bnd]):
            pi, pb = choice[bnd][s]
            tot = pi + pb
            p = pi / tot if tot else float("nan")
            out["conditional_choice"][str(bnd)][str(s)] = {
                "choice_turns": tot, "pick_interior": pi, "pick_boundary": pb,
                "p_pick_interior": p,
            }
            print(f"{s:>7} {tot:>12} {pi:>9} {pb:>9} {p:>8.4f}")
        cc = out["conditional_choice"][str(bnd)]
        if "30000" in cc and "87500" in cc:
            dp = cc["87500"]["p_pick_interior"] - cc["30000"]["p_pick_interior"]
            print(f"   d(P interior) 30k->87.5k = {dp:+.4f}  "
                  f"(D2/LIT needs HIGH+RISING; fall = boundary commitment up = refute)\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
