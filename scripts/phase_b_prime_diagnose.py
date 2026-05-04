#!/usr/bin/env python3
"""Phase B' diagnosis — final post-run analysis.

Reads reports/phase_b_prime/instrumented/events.jsonl and prints a
self-contained block of numbers used by `diagnosis.md`. Cheap; runs locally.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "reports/phase_b_prime/instrumented/events.jsonl"
)


def _parse_axial(s: str) -> tuple[int, int]:
    s = s.strip("() ")
    a, b = s.split(",")
    return int(a), int(b)


def stride5(game: dict) -> dict | None:
    moves_raw = game.get("moves_list", [])
    if not moves_raw:
        return None
    if isinstance(moves_raw[0], list):
        moves = [tuple(m) for m in moves_raw]
    else:
        moves = [_parse_axial(s) for s in moves_raw]
    if len(moves) < 5:
        return None
    rows: dict[int, list[int]] = {}
    for q, r in moves:
        rows.setdefault(r, []).append(q)
    row_max = max(len(v) for v in rows.values())
    best_run = 1
    for _r, qs in rows.items():
        qss = sorted(set(qs))
        for start in qss:
            run = 1
            q = start
            while (q + 5) in qss:
                run += 1
                q += 5
            best_run = max(best_run, run)
    return {"row_max": row_max, "run5": best_run, "tr": game.get("terminal_reason")}


def axis_distribution(game: dict) -> dict | None:
    """Per-axis same-color-pair fractions (mirror of compute_axis_fractions)."""
    moves_raw = game.get("moves_list", [])
    if not moves_raw:
        return None
    if isinstance(moves_raw[0], list):
        moves = [tuple(m) for m in moves_raw]
    else:
        moves = [_parse_axial(s) for s in moves_raw]
    if len(moves) < 4:
        return None
    # ply 0 = P1; thereafter compound 2.
    p1 = []
    p2 = []
    for ply, m in enumerate(moves):
        own_p1 = (ply == 0) or (((ply - 1) // 2) % 2 == 1)
        (p1 if own_p1 else p2).append(m)
    counts = {"q": 0, "r": 0, "s": 0}
    total = 0
    for stones in (p1, p2):
        S = set(stones)
        for q, r in stones:
            for axis, (dq, dr) in (("q", (1, 0)), ("r", (0, 1)), ("s", (1, -1))):
                if (q + dq, r + dr) in S:
                    counts[axis] += 1
                    total += 1
    if total == 0:
        return None
    return {
        "axis_q": counts["q"] / total,
        "axis_r": counts["r"] / total,
        "axis_s": counts["s"] / total,
    }


def main() -> int:
    events = []
    with open(LOG) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    games = [e for e in events if e.get("event") == "game_complete"]
    steps = [e for e in events if e.get("event") == "training_step"]
    vps = [e for e in events if e.get("event") == "value_probe_drift"]
    bcs = [e for e in events if e.get("event") == "buffer_composition"]
    mvs = [e for e in events if e.get("event") == "model_version_summary"]
    wdrs = [e for e in events if e.get("event") == "worker_draw_rate"]

    last_step = steps[-1]["step"] if steps else 0
    elapsed = events[-1]["ts"] - events[0]["ts"]

    print(f"## Run summary")
    print(f"final_step={last_step}  games={len(games)}  elapsed_min={elapsed/60:.1f}  rate={last_step/(elapsed/60):.1f}/min")
    cap = [g for g in games if g.get("terminal_reason") == "ply_cap"]
    six = [g for g in games if g.get("terminal_reason") == "six_in_a_row"]
    col = [g for g in games if g.get("terminal_reason") == "colony"]
    other = [g for g in games if g.get("terminal_reason") == "other_draw"]
    print(f"terminal: cap={len(cap)} six={len(six)} colony={len(col)} other={len(other)}")
    print(f"draw_rate={len(cap)/max(len(games),1):.3f}")

    # Per-game value-probe drift table
    print(f"\n## Value-probe drift (Class 2)")
    print("| step | decisive_mean | decisive_std | draw_mean | draw_std |")
    print("|---:|---:|---:|---:|---:|")
    for v in vps:
        print(f"| {v['step']} | {v['decisive_mean']:+.3f} | {v.get('decisive_std',0):.3f} | {v['draw_mean']:+.3f} | {v.get('draw_std',0):.3f} |")
    if vps:
        decs = [v['decisive_mean'] for v in vps]
        dws = [v['draw_mean'] for v in vps]
        print(f"\ndecisive: mean={statistics.mean(decs):+.4f} std={statistics.stdev(decs):.4f} range=[{min(decs):+.3f},{max(decs):+.3f}]")
        print(f"draw:     mean={statistics.mean(dws):+.4f} std={statistics.stdev(dws):.4f} range=[{min(dws):+.3f},{max(dws):+.3f}]")
        print(f"v7full baseline (pre-run): decisive=-0.094  draw=+0.146")
        print(f"shift_decisive = {statistics.mean(decs) - (-0.094):+.3f}")

    # Buffer composition table
    print(f"\n## Buffer composition (Class 3)")
    print("| step | corpus_frac | draw_target_frac | six_term | colony_term | cap_term |")
    print("|---:|---:|---:|---:|---:|---:|")
    for b in bcs:
        print(f"| {b['step']} | {b.get('corpus_fraction',0):.3f} | {b.get('draw_target_fraction',0):.3f} | {b.get('six_terminal_fraction',0):.3f} | {b.get('colony_terminal_fraction',0):.3f} | {b.get('cap_terminal_fraction',0):.3f} |")

    # Model-version summary
    print(f"\n## Model-version range (Class 1)")
    print("| step | current | median | P90 | max | rho(range,draw) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for m in mvs:
        rho = m.get('spearman_rho_range_vs_draw')
        rho_s = f"{rho:+.3f}" if isinstance(rho, (int, float)) else "—"
        print(f"| {m['step']} | {m.get('current_version')} | {m.get('median_range')} | {m.get('p90_range')} | {m.get('max_range')} | {rho_s} |")

    # Worker draw rates
    print(f"\n## Per-worker draw rate (last 50 games)")
    print("| step | worker rates | hot(≥0.80) |")
    print("|---:|---|---:|")
    for w in wdrs:
        pw = w.get('per_worker', {})
        rates_str = ", ".join(f"w{k}={float(v):.2f}" for k, v in sorted(pw.items(), key=lambda x: int(x[0])))
        hot = sum(1 for v in pw.values() if v >= 0.80)
        print(f"| {w['step']} | {rates_str} | {hot}/{len(pw)} |")

    # Stride-5 metrics by terminal reason
    print(f"\n## Class-4 — stride-5 / row-density metrics")
    print("| terminal_reason | n | row_max med/P90/max | stride5_run med/P90/max |")
    print("|---|---:|---:|---:|")
    for label, gs in (("ply_cap", cap), ("six_in_a_row", six), ("colony", col), ("other_draw", other)):
        ms = [stride5(g) for g in gs]
        ms = [m for m in ms if m]
        if not ms: continue
        rms = sorted(m['row_max'] for m in ms)
        rns = sorted(m['run5'] for m in ms)
        def s(xs):
            n = len(xs)
            return f"{xs[n//2]} / {xs[max(0,int(0.9*n)-1)]} / {xs[-1]}"
        print(f"| {label} | {len(ms)} | {s(rms)} | {s(rns)} |")

    # Axis distribution computed post-hoc
    print(f"\n## Axis distribution (post-hoc, mirroring axis_distribution event)")
    print("| terminal_reason | n | axis_q | axis_r | axis_s |")
    print("|---|---:|---:|---:|---:|")
    for label, gs in (("ply_cap", cap), ("six_in_a_row", six), ("colony", col)):
        ms = [axis_distribution(g) for g in gs]
        ms = [m for m in ms if m]
        if not ms: continue
        for axis in ('q','r','s'):
            pass
        means = {axis: statistics.mean(m[f'axis_{axis}'] for m in ms) for axis in ('q','r','s')}
        print(f"| {label} | {len(ms)} | {means['q']:.3f} | {means['r']:.3f} | {means['s']:.3f} |")

    # Spearman ρ between row_max and is_cap (class-4 vs draw correlation)
    try:
        from scipy.stats import spearmanr
        feats = []
        labels = []
        for g in games:
            m = stride5(g)
            if not m: continue
            feats.append(m['run5'])
            labels.append(1 if g.get('terminal_reason') == 'ply_cap' else 0)
        if len(feats) > 10:
            rho, p = spearmanr(feats, labels)
            print(f"\n## Class-4 ↔ draw correlation")
            print(f"Spearman ρ(stride5_run, is_cap) = {rho:+.4f}  (p={p:.2e}, n={len(feats)})")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
