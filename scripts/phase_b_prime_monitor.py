#!/usr/bin/env python3
"""Phase B' instrumented smoke monitoring helper — rsync'd to the 5080 host.

Pulls game count, training step rate, instrumentation event counts, latest
value-probe / buffer-composition / model-version readings, stride-5
row-density and chain-length metrics from the latest games.jsonl, and the
ETA to a target step. Prints a compact one-liner block.

Usage on the 5080:
    .venv/bin/python scripts/phase_b_prime_monitor.py [LOG_PATH] [TARGET_STEP]

Defaults: log = newest logs/events_*.jsonl, target = 5000.
"""

from __future__ import annotations

import glob
import json
import math
import statistics
import sys
import time
from typing import Any


def _parse_axial(s: str) -> tuple[int, int]:
    s = s.strip("() ")
    a, b = s.split(",")
    return int(a), int(b)


def _stride5_metrics(game: dict[str, Any]) -> dict[str, Any] | None:
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


def _quartiles(xs: list[float]) -> tuple[float, float, float]:
    if not xs:
        return (0.0, 0.0, 0.0)
    return (statistics.median(xs), max(xs), float(sum(xs) / len(xs)))


def main() -> int:
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    target_step = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    if log_path is None:
        candidates = sorted(glob.glob("logs/events_*.jsonl"), key=lambda p: -((__import__("os").stat(p).st_mtime)))
        if not candidates:
            print("no logs/events_*.jsonl on host")
            return 1
        log_path = candidates[0]
    events: list[dict[str, Any]] = []
    with open(log_path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    if not events:
        print(f"no events in {log_path}")
        return 1

    ts_now = time.time()
    ts_start = events[0]["ts"]
    elapsed_min = (ts_now - ts_start) / 60.0

    games = [e for e in events if e.get("event") == "game_complete"]
    caps = [e for e in games if e.get("terminal_reason") == "ply_cap"]
    sixs = [e for e in games if e.get("terminal_reason") == "six_in_a_row"]
    cols = [e for e in games if e.get("terminal_reason") == "colony"]
    steps = [e for e in events if e.get("event") == "training_step"]
    last_step = steps[-1]["step"] if steps else 0

    # Step rate over last 15 min
    recent = [e for e in steps if (ts_now - e["ts"]) <= 900]
    if len(recent) >= 2:
        span_min = (recent[-1]["ts"] - recent[0]["ts"]) / 60.0
        rate_15 = (recent[-1]["step"] - recent[0]["step"]) / max(span_min, 1e-6)
    else:
        rate_15 = last_step / max(elapsed_min, 1e-6)
    rate_overall = last_step / max(elapsed_min, 1e-6)
    eta_target_min = (target_step - last_step) / max(rate_15, 1e-6) if rate_15 > 0 else float("inf")

    print(
        f"log={log_path}  elapsed_min={elapsed_min:.1f}  step={last_step}  "
        f"rate_overall={rate_overall:.1f}/min  rate_last15={rate_15:.1f}/min  "
        f"ETA_to_{target_step}={eta_target_min:.0f}min"
    )
    dr = len(caps) / max(len(games), 1) * 100
    print(
        f"games={len(games)} cap={len(caps)} six={len(sixs)} colony={len(cols)} "
        f"draw_rate={dr:.1f}%"
    )

    n_vp  = sum(1 for e in events if e.get("event") == "value_probe_drift")
    n_bc  = sum(1 for e in events if e.get("event") == "buffer_composition")
    n_mv  = sum(1 for e in events if e.get("event") == "model_version_summary")
    n_wdr = sum(1 for e in events if e.get("event") == "worker_draw_rate")
    print(f"instr_events: vp={n_vp} bc={n_bc} mv={n_mv} wdr={n_wdr}")

    vps = [e for e in events if e.get("event") == "value_probe_drift"]
    if vps:
        line = " | ".join(
            f"s{e['step']} dec={e['decisive_mean']:+.3f} draw={e['draw_mean']:+.3f}"
            for e in vps
        )
        print(f"VP_trace: {line}")
    bcs = [e for e in events if e.get("event") == "buffer_composition"]
    if bcs:
        last = bcs[-1]
        print(
            f"BC s{last['step']} corp={last.get('corpus_fraction',0):.3f} "
            f"drawT={last.get('draw_target_fraction',0):.3f} "
            f"six={last.get('six_terminal_fraction',0):.3f} "
            f"colony={last.get('colony_terminal_fraction',0):.3f} "
            f"cap={last.get('cap_terminal_fraction',0):.3f}"
        )
    mvs = [e for e in events if e.get("event") == "model_version_summary"]
    if mvs:
        last = mvs[-1]
        print(
            f"MV s{last['step']} cur={last.get('current_version')} "
            f"med={last.get('median_range')} P90={last.get('p90_range')} "
            f"max={last.get('max_range')} rho={last.get('spearman_rho_range_vs_draw')}"
        )
    wdrs = [e for e in events if e.get("event") == "worker_draw_rate"]
    if wdrs:
        last = wdrs[-1]
        pw = last.get("per_worker", {})
        hot = sum(1 for v in pw.values() if v >= 0.80)
        print(f"WDR s{last['step']} workers={len(pw)} hot(>=0.80)={hot} rates={pw}")

    ms = [x for x in (_stride5_metrics(g) for g in games) if x]
    cap_ms = [x for x in ms if x["tr"] == "ply_cap"]
    six_ms = [x for x in ms if x["tr"] == "six_in_a_row"]
    if cap_ms:
        rms = [x["row_max"] for x in cap_ms]
        rns = [x["run5"] for x in cap_ms]
        print(
            f"CAP n={len(cap_ms)} "
            f"row_max med/max={statistics.median(rms):.0f}/{max(rms)} "
            f"stride5_run med/max={statistics.median(rns):.0f}/{max(rns)}"
        )
    if six_ms:
        rms = [x["row_max"] for x in six_ms]
        rns = [x["run5"] for x in six_ms]
        print(
            f"SIX n={len(six_ms)} "
            f"row_max med/max={statistics.median(rms):.0f}/{max(rms)} "
            f"stride5_run med/max={statistics.median(rns):.0f}/{max(rns)}"
        )

    gn_recent = [
        e.get("grad_norm")
        for e in steps[-30:]
        if isinstance(e.get("grad_norm"), (int, float)) and math.isfinite(e["grad_norm"])
    ]
    if gn_recent:
        print(
            f"grad_norm last30: mean={statistics.mean(gn_recent):.2f} "
            f"max={max(gn_recent):.2f}"
        )

    ha = [e for e in events if "hard_abort" in str(e.get("event", ""))]
    sd = [e for e in events if e.get("event") == "shutdown_requested"]
    if ha:
        print(f"!! HARD_ABORT: {ha[-1]}")
    if sd:
        print(f"!! SHUTDOWN: {sd[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
