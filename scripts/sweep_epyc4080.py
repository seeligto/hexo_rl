#!/usr/bin/env python3
"""
Throughput sweep harness for EPYC 7702 + RTX 4080 Super.

Drives scripts/benchmark.py per cell, parses worker_pos_per_hr from the
emitted JSON, and writes a CSV summary. Designed to run unattended on a
rental GPU server.

Usage:
    .venv/bin/python scripts/sweep_epyc4080.py --stage workers
    .venv/bin/python scripts/sweep_epyc4080.py --stage batch_wait \\
        --workers 32
    .venv/bin/python scripts/sweep_epyc4080.py --stage leaf_burst \\
        --workers 32 --batch 128 --wait 3.0
    .venv/bin/python scripts/sweep_epyc4080.py --stage all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
BENCH = str(ROOT / "scripts" / "benchmark.py")
REPORTS = ROOT / "reports" / "benchmarks"
SWEEP_DIR = ROOT / "reports" / "sweeps"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


def write_override(cell: dict) -> Path:
    """Write a YAML override that layers gumbel_targets_epyc4080 + cell knobs."""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="sweep_cell_")
    os.close(fd)
    sp = []
    sp.append("selfplay:")
    sp.append("  completed_q_values: true")
    sp.append("  gumbel_mcts: false")
    sp.append("  max_game_moves: 300")
    # Disable move-level playout cap so the base config's full_search_prob=0.5
    # doesn't survive the load_config merge and confuse any code reading the
    # merged config. The benchmark bench_cfg also hard-codes these to 0, but
    # belt-and-suspenders here prevents silent mismatches.
    sp.append("  playout_cap:")
    sp.append("    fast_prob: 0.0")
    sp.append("    full_search_prob: 0.0")
    sp.append("    n_sims_quick: 0")
    sp.append("    n_sims_full: 0")
    # No random opening plies in the benchmark — skipped rows deflate pos/hr.
    sp.append("  random_opening_plies: 0")
    if "n_workers" in cell:
        sp.append(f"  n_workers: {cell['n_workers']}")
    if "inference_batch_size" in cell:
        sp.append(f"  inference_batch_size: {cell['inference_batch_size']}")
    if "inference_max_wait_ms" in cell:
        sp.append(f"  inference_max_wait_ms: {cell['inference_max_wait_ms']}")
    if "leaf_batch_size" in cell:
        sp.append(f"  leaf_batch_size: {cell['leaf_batch_size']}")
    if "max_train_burst" in cell:
        sp.append(f"max_train_burst: {cell['max_train_burst']}")
    sp.append("training_steps_per_game: 2.0")
    Path(path).write_text("\n".join(sp) + "\n")
    return Path(path)


def latest_json_after(t0: float) -> Path | None:
    """Find newest JSON in reports/benchmarks/ written after t0."""
    if not REPORTS.exists():
        return None
    cands = [p for p in REPORTS.glob("*.json") if p.stat().st_mtime > t0]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def run_cell(cell: dict, pool_duration: int, n_runs: int, no_compile: bool) -> dict:
    """Run benchmark for a single sweep cell, return parsed metrics."""
    override = write_override(cell)
    workers = cell["n_workers"]
    cmd = [
        PY, BENCH,
        "--config", str(override),
        "--pool-workers", str(workers),
        "--pool-duration", str(pool_duration),
        "--n-runs", str(n_runs),
    ]
    if no_compile:
        cmd.append("--no-compile")
    # no_compile defaults to True — compile JIT on a cold cache can take
    # 30-120s, blowing the 90s warmup window → 0 games/hr for entire cell.


    label = " ".join(f"{k}={v}" for k, v in cell.items())
    print(f"\n[cell] {label}", flush=True)
    print(f"[cmd]  {' '.join(cmd)}", flush=True)

    t0 = time.time()
    env = dict(os.environ, MALLOC_ARENA_MAX="2")
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT))
    elapsed = time.time() - t0

    report = latest_json_after(t0 - 5)
    pph = gph = bat = None
    if report and report.exists():
        try:
            data = json.loads(report.read_text())
            metrics = data.get("metrics", {})
            pph = metrics.get("worker_pos_per_hr", {}).get("median")
            gph = metrics.get("worker_games_per_hr", {}).get("median")
            bat = metrics.get("worker_batch_pct", {}).get("median")
        except Exception as e:
            print(f"[warn] failed to parse {report}: {e}", flush=True)

    out = dict(cell)
    out.update({
        "pos_per_hr": pph,
        "games_per_hr": gph,
        "batch_pct": bat,
        "elapsed_s": round(elapsed, 1),
        "exit_code": proc.returncode,
        "report": str(report) if report else "",
    })
    print(f"[result] pos/hr={pph}  games/hr={gph}  batch%={bat}  ({elapsed:.0f}s)",
          flush=True)
    try:
        override.unlink()
    except Exception:
        pass
    return out


def write_csv(stage: str, rows: list[dict]) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out = SWEEP_DIR / f"sweep_{stage}_{ts}.csv"
    if not rows:
        return out
    keys = sorted({k for r in rows for k in r.keys()})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[csv] {out}", flush=True)
    return out


def print_table(rows: list[dict], rank_key: str = "pos_per_hr") -> None:
    if not rows:
        return
    ranked = sorted(
        [r for r in rows if r.get(rank_key) is not None],
        key=lambda r: r[rank_key], reverse=True,
    )
    print("\n=== Ranked by pos/hr ===", flush=True)
    for i, r in enumerate(ranked):
        knobs = ", ".join(f"{k}={v}" for k, v in r.items()
                          if k in ("n_workers", "inference_batch_size",
                                   "inference_max_wait_ms", "leaf_batch_size",
                                   "max_train_burst"))
        print(f"  #{i+1}  pos/hr={r['pos_per_hr']:>9,.0f}  "
              f"games/hr={(r.get('games_per_hr') or 0):>6,.0f}  "
              f"batch%={(r.get('batch_pct') or 0):>5.1f}   {knobs}",
              flush=True)


def stage_workers(args) -> list[dict]:
    cells = []
    for w in args.worker_grid:
        cells.append({
            "n_workers": w,
            "inference_batch_size": 128,
            "inference_max_wait_ms": 3.0,
            "leaf_batch_size": 8,
            "max_train_burst": 16,
        })
    return cells


def stage_batch_wait(args) -> list[dict]:
    cells = []
    for b in args.batch_grid:
        for w in args.wait_grid:
            cells.append({
                "n_workers": args.workers,
                "inference_batch_size": b,
                "inference_max_wait_ms": w,
                "leaf_batch_size": 8,
                "max_train_burst": 16,
            })
    return cells


def stage_leaf_burst(args) -> list[dict]:
    cells = []
    for lb in args.leaf_grid:
        for tb in args.burst_grid:
            cells.append({
                "n_workers": args.workers,
                "inference_batch_size": args.batch,
                "inference_max_wait_ms": args.wait,
                "leaf_batch_size": lb,
                "max_train_burst": tb,
            })
    return cells


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["workers", "batch_wait", "leaf_burst", "all"])
    p.add_argument("--pool-duration", type=int, default=60)
    p.add_argument("--n-runs", type=int, default=2,
                   help="bench reps per cell (2 keeps sweep cells ~3min each)")
    # Default no_compile=True: compile JIT cold-start (30-120s) races the
    # 90s warmup window and produces 0 games/hr on cold machines.
    # Use --compile to opt into compiled mode if the cache is already warm.
    p.add_argument("--no-compile", action="store_true", default=True,
                   help="skip torch.compile per cell (default: True; compile "
                        "cold-start races the warmup window)")
    p.add_argument("--compile", dest="no_compile", action="store_false",
                   help="enable torch.compile per cell (warm .torchinductor-cache required)")

    p.add_argument("--worker-grid", type=int, nargs="+",
                   default=[16, 24, 32, 40])
    p.add_argument("--batch-grid", type=int, nargs="+",
                   default=[64, 128, 192])
    p.add_argument("--wait-grid", type=float, nargs="+",
                   default=[2.0, 4.0, 8.0])
    p.add_argument("--leaf-grid", type=int, nargs="+",
                   default=[8, 16])
    p.add_argument("--burst-grid", type=int, nargs="+",
                   default=[8, 16, 32])

    p.add_argument("--workers", type=int, default=32,
                   help="fixed workers for stages batch_wait / leaf_burst")
    p.add_argument("--batch", type=int, default=128,
                   help="fixed inference batch for stage leaf_burst")
    p.add_argument("--wait", type=float, default=3.0,
                   help="fixed inference wait ms for stage leaf_burst")
    args = p.parse_args()

    stages = []
    if args.stage in ("workers", "all"):
        stages.append(("workers", stage_workers(args)))
    if args.stage in ("batch_wait", "all"):
        stages.append(("batch_wait", stage_batch_wait(args)))
    if args.stage in ("leaf_burst", "all"):
        stages.append(("leaf_burst", stage_leaf_burst(args)))

    for stage_name, cells in stages:
        print(f"\n########## stage: {stage_name} ({len(cells)} cells) ##########",
              flush=True)
        rows: list[dict] = []
        for cell in cells:
            rows.append(run_cell(cell, args.pool_duration, args.n_runs, args.no_compile))
        write_csv(stage_name, rows)
        print_table(rows)

        if args.stage == "all" and stage_name == "workers":
            best = max((r for r in rows if r.get("pos_per_hr") is not None),
                       key=lambda r: r["pos_per_hr"], default=None)
            if best:
                args.workers = best["n_workers"]
                print(f"\n[carry] workers={args.workers} (winner of stage workers)",
                      flush=True)
        if args.stage == "all" and stage_name == "batch_wait":
            best = max((r for r in rows if r.get("pos_per_hr") is not None),
                       key=lambda r: r["pos_per_hr"], default=None)
            if best:
                args.batch = best["inference_batch_size"]
                args.wait = best["inference_max_wait_ms"]
                print(f"\n[carry] batch={args.batch} wait={args.wait} "
                      f"(winner of stage batch_wait)", flush=True)


if __name__ == "__main__":
    main()
