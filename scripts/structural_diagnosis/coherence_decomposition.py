#!/usr/bin/env python3
"""§D-COHERENCE Phase 1 — decompose forced_win_conversion into IN-WINDOW vs OFF-WINDOW
across the v6_live2 golong arc (checkpoint_step buckets).

PRE-REGISTERED question (reports/investigations/coherence_decomposition_2026-06-08.md):
the global forced_win_conversion decline (0.89->0.66) is mechanically OVER-DETERMINED by
the rising off-window share (off-window wins convert at ~0 by construction: records.rs:62
drops the winning cell from the policy target -> no logit -> unreachable in self-play).
Global conversion CANNOT distinguish "coherence-the-new-phenomenon" from "the known
off-window structural defect". This script splits them.

  V-OFFWINDOW : in-window conversion FLAT (|d| <= 0.05, CI-cleared) + rising off-window
                share accounts for the global decline -> reduces to records.rs:62.
  V-INWINDOW  : in-window conversion ALSO drops materially (d >= 0.10, CI-cleared)
                -> a distinct in-window finishing degradation.
  V-BOTH / V-NULL per the dispatcher.

EVAL-ONLY. Read-only on banked golong self-play game records (Rust GameRecorder,
self-play temperature schedule). ZERO engine/config/Rust/pretrain change. Reuses the
shared detector (forced_win_detector) + golong_game_analysis turn-walk — no metric copies,
zero geometry literals (off-window <=> board.to_flat(cell) >= spec.policy_logit_count,
mirroring records.rs:62 via is_off_window).

Bootstrap CI resamples GAME-SIDES (the independent unit; turns within a game are
correlated) -> the reported eff_n is game-sides with an IN-WINDOW forced win, NOT raw
turn count.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from hexo_rl.diagnostics.forced_win_detector import (
    depth1_wins, depth2_wins, is_off_window, winning_turn_cells,
)
from hexo_rl.encoding import lookup, normalize_encoding_name


def _win_cells(snap, cp, unit):
    """The win-cell set whose off-window classification decides in-window vs off-window-only.

    unit='ply'  : LEGACY flatten of the depth-2 pair {f, s} ∪ depth-1 (what §D-COHERENCE ran).
    unit='turn' : TURN-CORRECT completing cells {pair[1]} ∪ depth-1 (§D-GLOBALCONC Phase 2a) —
                  the cell that LANDS the win, hence the reachability-relevant one.
    Both are non-empty iff there is any win, so `forced`/`converted` (and GLOBAL conversion) are
    invariant; only the IN-WINDOW vs OFF-WINDOW SPLIT differs."""
    if unit == "turn":
        return sorted(winning_turn_cells(snap, cp))
    d1 = depth1_wins(snap, cp); d2 = depth2_wins(snap, cp)
    return [tuple(c) for c in d1] + [tuple(c) for pr in d2 for c in pr]


def analyze_game_sides(moves, spec, name, unit="ply"):
    """Replay one recorded game; per mover side return a dict of forced-win tallies.

    Turn-walk is byte-identical to scripts/golong_game_analysis.analyze_game (validated):
    at each of the mover's turn-starts, run the detector on the turn-start snapshot;
    classify the turn IN-WINDOW (>=1 completing cell convertible) vs OFF-WINDOW-ONLY
    (all completing cells off-window, unconvertible). 'converted' = the side actually
    won on that turn (the game ends, so converted <= 1 per game-side).

    Returns {side: {forced, conv, off_only, in_window, conv_in, conv_off}} for sides
    that reached >=1 of their turn-starts.
    """
    from engine import Board
    board = Board.with_encoding_name(name)
    mv = [(int(q), int(r)) for (q, r) in moves]
    sides = (1, -1)
    # forced, converted(any), off_only_turns, in_window_turns, conv_in_turns, conv_off_turns
    stat = {s: [0, 0, 0, 0, 0, 0] for s in sides}
    seen_side = {s: False for s in sides}

    i, n = 0, len(mv)
    while i < n:
        cp = board.current_player
        if cp not in stat:
            break
        seen_side[cp] = True
        snap = board.clone()
        won_this_turn = False
        while i < n:
            q, r = mv[i]
            try:
                board.apply_move(q, r)
            except Exception:
                i = n
                break
            i += 1
            if board.check_win():
                won_this_turn = board.winner() == cp
                break
            if board.current_player != cp:
                break
        win_cells = _win_cells(snap, cp, unit)
        if not win_cells:
            continue
        s = stat[cp]
        s[0] += 1                                  # forced
        has_in_window = any(not is_off_window(snap, c, spec) for c in win_cells)
        if won_this_turn:
            s[1] += 1                              # converted (any)
        if has_in_window:
            s[3] += 1                              # in-window forced turn
            if won_this_turn:
                s[4] += 1                          # converted in-window turn
        else:
            s[2] += 1                              # off-window-only forced turn
            if won_this_turn:
                s[5] += 1                          # converted off-window turn (~0 by const.)
    out = {}
    for sd in sides:
        if seen_side[sd] and stat[sd][0] > 0:
            f, c, off, inn, ci, co = stat[sd]
            out[sd] = {"forced": f, "conv": c, "off_only": off, "in_window": inn,
                       "conv_in": ci, "conv_off": co}
    return out


def load_games(files, name, spec, min_step, unit="ply"):
    """Return {checkpoint_step: [per-game-side tally dict, ...]} from the replays."""
    buckets = defaultdict(list)
    raw = defaultdict(int)
    for f in files:
        try:
            fh = open(f)
        except FileNotFoundError:
            print(f"  (skip missing {f})", file=sys.stderr)
            continue
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("game_length", 0) <= 0:
                continue
            step = int(d.get("checkpoint_step", 0))
            if step < min_step:
                continue
            raw[step] += 1
            gs = analyze_game_sides(d["moves"], spec, name, unit)
            for rec in gs.values():
                buckets[step].append(rec)
    return buckets, raw


# ── per-bucket estimators (over a list of game-side tally dicts) ──────────────

def _gs_metrics(gss):
    """Compute the headline rates from a list of game-side tallies (no CI)."""
    F = sum(g["forced"] for g in gss)
    C = sum(g["conv"] for g in gss)
    F_in = sum(g["in_window"] for g in gss)
    F_off = sum(g["off_only"] for g in gss)
    C_in = sum(g["conv_in"] for g in gss)
    # per-(game,side) recurrence-deduped (comparable to live EMA)
    fg = sum(1 for g in gss if g["forced"] > 0)
    cg = sum(1 for g in gss if g["conv"] > 0)
    fg_in = sum(1 for g in gss if g["in_window"] > 0)
    cg_in = sum(1 for g in gss if g["in_window"] > 0 and g["conv"] > 0)
    return {
        # turn-level
        "F": F, "C": C, "F_in": F_in, "F_off": F_off, "C_in": C_in,
        "conv_global_turn": (C / F) if F else float("nan"),
        "conv_in_turn": (C_in / F_in) if F_in else float("nan"),
        "offwindow_share_turn": (F_off / F) if F else float("nan"),
        # per-(game,side) deduped (PRIMARY for the verdict)
        "n_gs": len(gss),
        "fg": fg, "cg": cg, "fg_in": fg_in, "cg_in": cg_in,
        "conv_global_gs": (cg / fg) if fg else float("nan"),
        "conv_in_gs": (cg_in / fg_in) if fg_in else float("nan"),
        "offwindow_share_gs": ((fg - fg_in) / fg) if fg else float("nan"),
        # survivorship guard: in-window forced-win COUNT per game-side
        "mean_Fin_per_gs": (F_in / len(gss)) if gss else float("nan"),
        "frac_gs_with_inwindow": (fg_in / len(gss)) if gss else float("nan"),
    }


def _boot(gss, rng, nboot, keys):
    """Cluster bootstrap over game-sides. Returns {key: (lo, hi)} 95% CI + the boot draws."""
    m = len(gss)
    draws = {k: [] for k in keys}
    for _ in range(nboot):
        idx = rng.integers(0, m, m)
        sample = [gss[j] for j in idx]
        mm = _gs_metrics(sample)
        for k in keys:
            draws[k].append(mm[k])
    ci = {}
    for k in keys:
        arr = np.array(draws[k], dtype=np.float64)
        ci[k] = (float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5)))
    return ci, {k: np.array(draws[k], dtype=np.float64) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--min-step", type=int, default=1,
                    help="exclude checkpoint_step < this (drops the legacy step=0 bucket)")
    ap.add_argument("--min-bucket-n", type=int, default=20,
                    help="report only buckets with at least this many game-sides")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--lo-step", type=int, default=30000, help="arc endpoint LOW for the delta")
    ap.add_argument("--hi-step", type=int, default=87500, help="arc endpoint HIGH for the delta")
    ap.add_argument("--unit", choices=["ply", "turn"], default="ply",
                    help="win-cell unit for the in-window/off-window split: 'ply' = legacy "
                         "flatten {f,s} (what §D-COHERENCE ran); 'turn' = turn-correct "
                         "completing cell pair[1] (§D-GLOBALCONC Phase 2a corrected unit).")
    ap.add_argument("--out", default="investigation/coherence_2026-06-08/coherence_decomposition.json")
    args = ap.parse_args()

    name = normalize_encoding_name(args.encoding)
    spec = lookup(name)
    print(f"[cfg] encoding={name} policy_logit_count={spec.policy_logit_count} "
          f"trunk_size={spec.trunk_size} UNIT={args.unit}")

    buckets, raw = load_games(args.files, name, spec, args.min_step, args.unit)
    steps = sorted(buckets)
    rng = np.random.default_rng(args.seed)

    keys = ["conv_in_gs", "conv_global_gs", "offwindow_share_gs", "offwindow_share_turn",
            "conv_in_turn", "mean_Fin_per_gs", "frac_gs_with_inwindow"]

    out = {"encoding": name, "policy_logit_count": int(spec.policy_logit_count),
           "min_step": args.min_step, "nboot": args.nboot, "buckets": []}
    boot_store = {}

    print(f"\n{'='*100}")
    print("§D-COHERENCE — in-window vs off-window forced-win conversion across the golong arc")
    print(f"{'='*100}")
    hdr = (f"{'step':>7} {'games':>6} {'gs':>5} {'eff_n':>6} | "
           f"{'conv_in(gs)':>22} | {'offw_share(gs)':>22} | {'meanFin/gs':>11} | {'conv_glob':>9}")
    print(hdr)
    print("-" * len(hdr))
    for st in steps:
        gss = buckets[st]
        if len(gss) < args.min_bucket_n:
            print(f"{st:>7} {raw[st]:>6} {len(gss):>5}  (skip: < {args.min_bucket_n} game-sides)")
            continue
        m = _gs_metrics(gss)
        ci, draws = _boot(gss, rng, args.nboot, keys)
        boot_store[st] = draws
        eff_n = m["fg_in"]   # game-sides with an in-window forced win = the conv_in_gs denom
        ci_in = ci["conv_in_gs"]; ci_ow = ci["offwindow_share_gs"]
        print(f"{st:>7} {raw[st]:>6} {len(gss):>5} {eff_n:>6} | "
              f"{m['conv_in_gs']:.3f} [{ci_in[0]:.3f},{ci_in[1]:.3f}] | "
              f"{m['offwindow_share_gs']:.3f} [{ci_ow[0]:.3f},{ci_ow[1]:.3f}] | "
              f"{m['mean_Fin_per_gs']:>11.3f} | {m['conv_global_gs']:>9.3f}")
        rec = {"step": st, "n_games": raw[st], "n_game_sides": len(gss), "eff_n_conv_in": eff_n}
        rec.update({k: (None if (isinstance(v, float) and v != v) else v) for k, v in m.items()})
        rec["ci"] = {k: ci[k] for k in keys}
        out["buckets"].append(rec)

    # ── 30k -> 87.5k delta with bootstrapped CI (difference of independent draws) ──
    lo, hi = args.lo_step, args.hi_step
    print(f"\n{'-'*100}")
    print(f"PRE-REGISTERED DELTA  {lo} -> {hi}  (95% CI on the delta via paired bootstrap draws)")
    print(f"{'-'*100}")
    if lo in boot_store and hi in boot_store:
        for k, label in [
            ("conv_in_gs",        "in-window conversion (per-game-side)"),
            ("conv_in_turn",      "in-window conversion (turn-level)"),
            ("offwindow_share_gs","off-window share (per-game-side)"),
            ("conv_global_gs",    "GLOBAL conversion (per-game-side)"),
            ("mean_Fin_per_gs",   "in-window forced-win COUNT / game-side"),
        ]:
            dlo = boot_store[lo][k]; dhi = boot_store[hi][k]
            n = min(len(dlo), len(dhi))
            delta = dhi[:n] - dlo[:n]
            d_pt = float(np.nanmedian(delta))
            d_ci = (float(np.nanpercentile(delta, 2.5)), float(np.nanpercentile(delta, 97.5)))
            v_lo = float(np.nanmedian(dlo)); v_hi = float(np.nanmedian(dhi))
            print(f"  {label:<40} {v_lo:.3f} -> {v_hi:.3f}   d={d_pt:+.3f}  "
                  f"CI[{d_ci[0]:+.3f},{d_ci[1]:+.3f}]")
            out.setdefault("deltas", {})[k] = {
                "from_step": lo, "to_step": hi, "v_from": v_lo, "v_to": v_hi,
                "delta": d_pt, "ci": d_ci}

        # ── shift-share: split the GLOBAL turn-level decline into in-window-drop vs
        #    off-window-share-rise (off-window converts ~0 -> global = conv_in*(1-share)) ──
        blo = next(b for b in out["buckets"] if b["step"] == lo)
        bhi = next(b for b in out["buckets"] if b["step"] == hi)
        ci0, ow0 = blo["conv_in_turn"], blo["offwindow_share_turn"]
        ci1, ow1 = bhi["conv_in_turn"], bhi["offwindow_share_turn"]
        g0, g1 = ci0 * (1 - ow0), ci1 * (1 - ow1)
        dec = g0 - g1
        ci_avg, w_avg = (ci0 + ci1) / 2, ((1 - ow0) + (1 - ow1)) / 2
        inwin_eff = (ci0 - ci1) * w_avg
        offwin_eff = ci_avg * ((1 - ow0) - (1 - ow1))
        inwin_pct = (inwin_eff / dec * 100) if dec else float("nan")
        offwin_pct = (offwin_eff / dec * 100) if dec else float("nan")
        out["decline_decomposition"] = {
            "global_turn_from": g0, "global_turn_to": g1, "total_decline": dec,
            "inwindow_drop_effect": inwin_eff, "inwindow_drop_pct": inwin_pct,
            "offwindow_share_rise_effect": offwin_eff, "offwindow_share_rise_pct": offwin_pct}

        # ── verdict mapping (pre-registered thresholds, NO post-hoc moves) ──
        d_gs = out["deltas"]["conv_in_gs"]      # recurrence-deduped (live-EMA-comparable)
        d_tn = out["deltas"]["conv_in_turn"]    # turn-level (per-turn ratio, EMA proxy)
        dshare = out["deltas"]["offwindow_share_gs"]
        print(f"\n{'='*100}\nVERDICT MAPPING (pre-registered: V-INWINDOW iff in-window drop >=0.10 CI-cleared; "
              f"V-OFFWINDOW iff in-window FLAT |d|<=0.05 AND off-window-rise drives the decline)\n{'='*100}")
        print(f"  in-window conv (turn-level)   d={d_tn['delta']:+.3f} CI[{d_tn['ci'][0]:+.3f},{d_tn['ci'][1]:+.3f}]"
              f"   {'>=0.10 CI-cleared -> MEETS V-INWINDOW' if (-d_tn['delta'])>=0.10 and d_tn['ci'][1]<0 else 'sub-0.10 or CI straddles'}")
        print(f"  in-window conv (game-side)    d={d_gs['delta']:+.3f} CI[{d_gs['ci'][0]:+.3f},{d_gs['ci'][1]:+.3f}]"
              f"   {'CI excludes 0' if d_gs['ci'][1]<0 else 'CI straddles 0'} (|d|={abs(d_gs['delta']):.3f} vs 0.10 bar)")
        print(f"  off-window share (game-side)  d={dshare['delta']:+.3f} CI[{dshare['ci'][0]:+.3f},{dshare['ci'][1]:+.3f}]"
              f"   {'FLAT' if abs(dshare['delta'])<=0.05 else 'moved'}")
        print(f"  GLOBAL decline shift-share: in-window-drop={inwin_pct:.0f}%  off-window-rise={offwin_pct:.0f}%")
        inwin_drops = (-d_tn["delta"]) >= 0.10 and d_tn["ci"][1] < 0      # EMA-proxy clears the bar
        inwin_flat = abs(d_tn["delta"]) <= 0.05 and abs(d_gs["delta"]) <= 0.05
        offwin_drives = offwin_pct >= 50.0
        if inwin_flat and offwin_drives:
            verdict = "V-OFFWINDOW — in-window flat; decline carried by rising off-window share"
        elif inwin_drops and not offwin_drives:
            verdict = ("V-INWINDOW — in-window conversion drops materially (turn-level d>=0.10 CI-cleared; "
                       "game-side d significant) AND in-window-drop dominates the decline; off-window share flat")
        elif inwin_flat and abs(dshare["delta"]) <= 0.05 and abs(out["deltas"]["conv_global_gs"]["delta"]) <= 0.05:
            verdict = "V-NULL — in-window flat AND off-window share flat AND global flat"
        else:
            verdict = "AMBIGUOUS — see both estimators + decomposition"
        out["verdict"] = verdict
        print(f"\n  >>> {verdict}")
    else:
        print(f"  (missing endpoint bucket: have {sorted(boot_store)} — need {lo} and {hi})")
        out["verdict"] = "INCOMPLETE — endpoint bucket missing"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
