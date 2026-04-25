#!/usr/bin/env python3
"""
Throughput sweep harness for EPYC 7702 + RTX 4080 Super.

Drives scripts/benchmark.py per cell, parses worker-pool sub-stats from
the bench stdout (median + IQR + min/max), and writes a CSV summary.
Designed to run unattended on a rental GPU server.

Defaults are tuned for *stable* measurements (pool_duration=180s, n_runs=5)
because n_runs=2 produced bimodal results on the first sweep. Each cell
takes ~12 minutes at these defaults.

Usage:
    .venv/bin/python scripts/sweep_epyc4080.py --stage workers
    .venv/bin/python scripts/sweep_epyc4080.py --stage batch_wait \\
        --workers 16
    .venv/bin/python scripts/sweep_epyc4080.py --stage leaf_burst \\
        --workers 16 --batch 128 --wait 4.0
    .venv/bin/python scripts/sweep_epyc4080.py --stage all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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

# Cell is flagged bimodal when min run is below this fraction of median —
# indicates a worker-startup race (some runs produced ~0 pos/hr).
BIMODAL_THRESHOLD = 0.30


def write_override(cell: dict) -> Path:
    """Write a YAML override that layers gumbel_targets_epyc4080 + cell knobs."""
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="sweep_cell_")
    os.close(fd)
    sp = []
    sp.append("selfplay:")
    sp.append("  completed_q_values: true")
    sp.append("  gumbel_mcts: false")
    sp.append("  max_game_moves: 300")
    sp.append("  playout_cap:")
    sp.append("    fast_prob: 0.0")
    sp.append("    full_search_prob: 0.0")
    sp.append("    n_sims_quick: 0")
    sp.append("    n_sims_full: 0")
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
    # torch.compile at root level — must come after the selfplay: block.
    # mode=default required: reduce-overhead deadlocks in InferenceServer's
    # background thread (per-thread CUDA-graph TLS uninitialized).
    sp.append("torch_compile: true")
    sp.append("torch_compile_mode: default")
    sp.append("training_steps_per_game: 2.0")
    Path(path).write_text("\n".join(sp) + "\n")
    return Path(path)


# Regex matches lines like:
#   "  Worker pool throughput pos/hr: median=388,426.3  IQR=+/-143,426.2  [266.9k-410.3k]  n=3"
#   "  GPU utilisation util%: median=63.2  IQR=+/-5.7  [58.0-63.7]  n=3"
_NUM = r"([\d,]+\.?\d*)"
_RANGE = r"\[([^\]]+)\]"
_LINE_RE = re.compile(
    rf"^\s*(?P<label>[^:]+?):\s*median={_NUM}\s+IQR=\+/-{_NUM}\s+{_RANGE}\s+n=(\d+)"
)


def _to_float(s: str) -> float:
    s = s.strip()
    mult = 1.0
    if s.endswith("k"):
        mult = 1e3
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1e6
        s = s[:-1]
    return float(s.replace(",", "")) * mult


def parse_bench_stdout(text: str) -> dict:
    """Pull median/min/max/IQR from bench stdout for the metrics we care about."""
    wanted = {
        "Worker pool throughput pos/hr": "pos",
        "Worker pool throughput games/hr": "games",
        "Worker pool throughput batch%": "batch",
        "GPU utilisation util%": "gpu_util",
        "GPU utilisation vram": "vram",
        "NN inference (batch=64)": "nn_inf",
        "MCTS (CPU only, no NN)": "mcts_cpu",
    }
    out: dict = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        label = m.group("label").strip()
        if label not in wanted:
            continue
        prefix = wanted[label]
        median = float(m.group(2).replace(",", ""))
        iqr = float(m.group(3).replace(",", ""))
        rng = m.group(4)
        try:
            lo, hi = rng.split("-", 1) if rng.count("-") == 1 else rng.rsplit("-", 1)
            lo_v, hi_v = _to_float(lo), _to_float(hi)
        except Exception:
            lo_v = hi_v = float("nan")
        out[f"{prefix}_median"] = median
        out[f"{prefix}_iqr"] = iqr
        out[f"{prefix}_min"] = lo_v
        out[f"{prefix}_max"] = hi_v
    return out


def latest_json_after(t0: float) -> Path | None:
    if not REPORTS.exists():
        return None
    cands = [p for p in REPORTS.glob("*.json") if p.stat().st_mtime > t0]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def run_cell(cell: dict, pool_duration: int, n_runs: int, no_compile: bool) -> dict:
    """Run benchmark for one sweep cell. Streams stdout live and parses sub-stats."""
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

    label = " ".join(f"{k}={v}" for k, v in cell.items())
    print(f"\n[cell] {label}", flush=True)
    print(f"[cmd]  {' '.join(cmd)}", flush=True)

    t0 = time.time()
    env = dict(os.environ, MALLOC_ARENA_MAX="2")
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    elapsed = time.time() - t0
    text = "".join(captured)

    parsed = parse_bench_stdout(text)
    pos_med = parsed.get("pos_median")
    pos_min = parsed.get("pos_min")

    # Bimodal detection: at least one run produced near-zero throughput.
    bimodal = False
    if pos_med and pos_min is not None and pos_med > 0:
        bimodal = (pos_min / pos_med) < BIMODAL_THRESHOLD

    out = dict(cell)
    out.update({
        "pos_median": pos_med,
        "pos_iqr": parsed.get("pos_iqr"),
        "pos_min": pos_min,
        "pos_max": parsed.get("pos_max"),
        "games_median": parsed.get("games_median"),
        "batch_median": parsed.get("batch_median"),
        "gpu_util_median": parsed.get("gpu_util_median"),
        "nn_inf_median": parsed.get("nn_inf_median"),
        "mcts_cpu_median": parsed.get("mcts_cpu_median"),
        "bimodal": bimodal,
        "elapsed_s": round(elapsed, 1),
        "exit_code": proc.returncode,
    })
    flag = " ⚠ BIMODAL" if bimodal else ""
    print(f"[result] pos/hr median={pos_med}  min={pos_min}  max={parsed.get('pos_max')}  "
          f"games/hr={parsed.get('games_median')}  batch%={parsed.get('batch_median')}  "
          f"gpu%={parsed.get('gpu_util_median')}  ({elapsed:.0f}s){flag}",
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


_KNOB_KEYS = ("n_workers", "inference_batch_size", "inference_max_wait_ms",
              "leaf_batch_size", "max_train_burst")


def _knob_str(r: dict) -> str:
    return ", ".join(f"{k}={r[k]}" for k in _KNOB_KEYS if k in r)


def print_table(stage: str, rows: list[dict]) -> None:
    if not rows:
        return
    valid = [r for r in rows if r.get("pos_median") is not None]
    if not valid:
        print(f"\n[stage {stage}] no usable results", flush=True)
        return
    ranked = sorted(valid, key=lambda r: r["pos_median"], reverse=True)
    print(f"\n=== Stage {stage} — ranked by pos/hr median ===", flush=True)
    print("  rank │   median │      iqr │      min │      max │ games/hr │ batch% │ gpu% │ flag │ knobs", flush=True)
    print("  ─────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────┼──────┼──────┼─────────", flush=True)
    for i, r in enumerate(ranked):
        flag = "BIMOD" if r.get("bimodal") else "  ok "
        print(f"  #{i+1:>3} │ {r['pos_median']:>8,.0f} │ "
              f"{(r.get('pos_iqr') or 0):>8,.0f} │ "
              f"{(r.get('pos_min') or 0):>8,.0f} │ "
              f"{(r.get('pos_max') or 0):>8,.0f} │ "
              f"{(r.get('games_median') or 0):>8,.0f} │ "
              f"{(r.get('batch_median') or 0):>6.1f} │ "
              f"{(r.get('gpu_util_median') or 0):>4.1f} │ "
              f"{flag} │ {_knob_str(r)}", flush=True)

    bimod = [r for r in ranked if r.get("bimodal")]
    stable = [r for r in ranked if not r.get("bimodal")]
    print(f"\n[stage {stage}] {len(stable)}/{len(ranked)} cells stable, "
          f"{len(bimod)} bimodal (any run < {BIMODAL_THRESHOLD:.0%} of median).",
          flush=True)
    if stable:
        w = stable[0]
        print(f"[stage {stage}] STABLE WINNER  pos/hr={w['pos_median']:,.0f}  {_knob_str(w)}",
              flush=True)
    if bimod:
        b = bimod[0]
        print(f"[stage {stage}] (top BIMODAL  pos/hr={b['pos_median']:,.0f}  "
              f"min={b.get('pos_min'):,.0f} — DO NOT TRUST  {_knob_str(b)})",
              flush=True)


def pick_winner(rows: list[dict]) -> dict | None:
    """Prefer the highest-median cell that is NOT flagged bimodal."""
    valid = [r for r in rows if r.get("pos_median") is not None]
    stable = [r for r in valid if not r.get("bimodal")]
    pool = stable if stable else valid
    if not pool:
        return None
    return max(pool, key=lambda r: r["pos_median"])


def stage_workers(args) -> list[dict]:
    return [{
        "n_workers": w,
        "inference_batch_size": 128,
        "inference_max_wait_ms": 4.0,
        "leaf_batch_size": 8,
        "max_train_burst": 16,
    } for w in args.worker_grid]


def stage_batch_wait(args) -> list[dict]:
    return [{
        "n_workers": args.workers,
        "inference_batch_size": b,
        "inference_max_wait_ms": w,
        "leaf_batch_size": 8,
        "max_train_burst": 16,
    } for b in args.batch_grid for w in args.wait_grid]


def stage_leaf_burst(args) -> list[dict]:
    return [{
        "n_workers": args.workers,
        "inference_batch_size": args.batch,
        "inference_max_wait_ms": args.wait,
        "leaf_batch_size": lb,
        "max_train_burst": tb,
    } for lb in args.leaf_grid for tb in args.burst_grid]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["workers", "batch_wait", "leaf_burst", "all"])
    p.add_argument("--pool-duration", type=int, default=180,
                   help="seconds per bench rep (default 180, was 60 — "
                        "shorter durations let worker-startup races dominate)")
    p.add_argument("--n-runs", type=int, default=5,
                   help="bench reps per cell (default 5, was 2 — "
                        "n=2 produced bimodal cells on the first sweep)")
    # torch.compile defaults to ON now that the threading fix landed.
    # The YAML override forces mode="default" (reduce-overhead deadlocks
    # against InferenceServer's background thread).
    p.add_argument("--no-compile", action="store_true", default=False,
                   help="skip torch.compile per cell (default: compile ON)")

    # Tighter ranges based on first-sweep findings:
    #   workers >= 24 produced bimodal startup races on EPYC 7702;
    #   GPU was at 65% util at workers=16 → little headroom from going wider.
    p.add_argument("--worker-grid", type=int, nargs="+",
                   default=[12, 16, 20, 24])
    p.add_argument("--batch-grid", type=int, nargs="+",
                   default=[64, 128, 192])
    p.add_argument("--wait-grid", type=float, nargs="+",
                   default=[2.0, 4.0, 8.0])
    p.add_argument("--leaf-grid", type=int, nargs="+",
                   default=[8, 16])
    p.add_argument("--burst-grid", type=int, nargs="+",
                   default=[8, 16, 32])

    p.add_argument("--workers", type=int, default=16,
                   help="fixed workers for stages batch_wait / leaf_burst")
    p.add_argument("--batch", type=int, default=128,
                   help="fixed inference batch for stage leaf_burst")
    p.add_argument("--wait", type=float, default=4.0,
                   help="fixed inference wait ms for stage leaf_burst")
    args = p.parse_args()

    stages = []
    if args.stage in ("workers", "all"):
        stages.append(("workers", stage_workers(args)))
    if args.stage in ("batch_wait", "all"):
        stages.append(("batch_wait", stage_batch_wait(args)))
    if args.stage in ("leaf_burst", "all"):
        stages.append(("leaf_burst", stage_leaf_burst(args)))

    final_winners: dict[str, dict] = {}
    for stage_name, cells in stages:
        print(f"\n########## stage: {stage_name} ({len(cells)} cells, "
              f"~{int(cells[0].get('n_workers', 0)) if cells else 0} workers, "
              f"pool_duration={args.pool_duration}s n_runs={args.n_runs}) ##########",
              flush=True)
        rows: list[dict] = []
        for cell in cells:
            rows.append(run_cell(cell, args.pool_duration, args.n_runs, args.no_compile))
        write_csv(stage_name, rows)
        print_table(stage_name, rows)

        winner = pick_winner(rows)
        if winner:
            final_winners[stage_name] = winner
        if args.stage == "all" and stage_name == "workers" and winner:
            args.workers = winner["n_workers"]
            print(f"\n[carry] workers={args.workers} (stable winner of stage workers)",
                  flush=True)
        if args.stage == "all" and stage_name == "batch_wait" and winner:
            args.batch = winner["inference_batch_size"]
            args.wait = winner["inference_max_wait_ms"]
            print(f"\n[carry] batch={args.batch} wait={args.wait} "
                  f"(stable winner of stage batch_wait)", flush=True)

    if final_winners:
        print("\n########## FINAL WINNERS ##########", flush=True)
        for stage, w in final_winners.items():
            tag = " ⚠ bimodal-only" if w.get("bimodal") else ""
            print(f"  {stage:>12} → pos/hr={w['pos_median']:,.0f}{tag}  {_knob_str(w)}",
                  flush=True)
        print("\nUpdate configs/variants/gumbel_targets_epyc4080.yaml with these knobs,",
              flush=True)
        print("then launch training:  make train VARIANT=gumbel_targets_epyc4080",
              flush=True)


if __name__ == "__main__":
    main()
