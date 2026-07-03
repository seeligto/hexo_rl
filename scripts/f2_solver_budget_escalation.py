#!/usr/bin/env python
"""F2 — solver budget-escalation diagnostic (D-SOLVER follow-up).

D-WS3V3's FIX2a measurement (`scripts/measure_native_provable_fraction.py`)
found the native `engine::tactics::TacticalSolver` proves 0/40 post-blunder
trap conversions at node_budget=20000/depth=16 (Wilson CI [0, 0.088]). The
untested knob is solver BUDGET. This script re-runs the IDENTICAL 40-position
fixture (by pos_id, loaded from
`reports/d_ws3v3/native_provable_fraction_sample40_records.jsonl` — the exact
sample the 0/40 number came from) at escalating `node_budget` tiers, holding
`depth` fixed at 16 (the in-loop config) unless `--depth` overrides it.

Solver config is IDENTICAL to the FIX2a measurement: window_half=None,
cand_cap=40, neighbor_dist=2 (quiet-move widening ON — the R3 LOSS-completeness
guard's recall-preserving verify branch is therefore live for every not-in-check
node, see `engine/src/tactics/search.rs` lines ~253-313).

Usage:
  python scripts/f2_solver_budget_escalation.py --tier baseline_1x --node-budget 20000 \\
      --out-prefix reports/investigations/f2_tier_baseline_1x --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RECORDS = "reports/d_ws3v3/native_provable_fraction_sample40_records.jsonl"
DEFAULT_TRAPS = "reports/d_tactical_2026-06-26/heldout_traps_all.jsonl"
DEFAULT_ENCODING = "v6_live2_ls"

# IDENTICAL solver config to scripts/measure_native_provable_fraction.py (FIX2a).
WINDOW_HALF = None
CAND_CAP = 40
NEIGHBOR_DIST = 2
DEFAULT_DEPTH = 16


def load_fixture_pos_ids(records_path: Path) -> List[str]:
    ids = []
    with open(records_path) as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["pos_id"])
    return ids


def load_traps_by_pos_id(traps_path: Path, pos_ids: Sequence[str]) -> List[Dict]:
    wanted = set(pos_ids)
    by_id: Dict[str, Dict] = {}
    with open(traps_path) as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("pos_id") in wanted:
                by_id[t["pos_id"]] = t
    missing = wanted - set(by_id)
    if missing:
        print(f"[f2] FATAL: {len(missing)} fixture pos_ids not found in {traps_path}: "
              f"{sorted(missing)}", file=sys.stderr)
        sys.exit(2)
    return [by_id[pid] for pid in pos_ids]


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _worker_init(niceness: int) -> None:
    try:
        os.nice(niceness)
    except OSError:
        pass


def _prove_one(args: Tuple[Dict, int, int, int]) -> Dict:
    """Runs in a forked worker process — imports engine fresh per-process."""
    trap, depth, node_budget, neighbor_dist = args
    import engine  # noqa: E402  (fork-local import; safe under mp fork context)

    def replay(seq, encoding):
        b = engine.Board.with_encoding_name(encoding)
        for q, r in seq:
            b.apply_move(int(q), int(r))
        return b

    encoding = trap.get("encoding", DEFAULT_ENCODING)
    board = replay(trap["post_move_seq"], encoding)
    solver = engine.TacticalSolver(window_half=WINDOW_HALF, cand_cap=CAND_CAP, neighbor_dist=neighbor_dist)

    t0 = time.time()
    result, line, nodes = solver.prove(board, depth, node_budget)
    dt = time.time() - t0

    return {
        "pos_id": trap["pos_id"],
        "bucket": trap.get("bucket"),
        "mate_distance": trap.get("mate_distance"),
        "proven_depth": trap.get("proven_depth"),
        "in_window": trap.get("in_window"),
        "depth": depth,
        "node_budget": node_budget,
        "solver_result": result,
        "proven_win_for_attacker": bool(result == -1),
        "nodes": nodes,
        "budget_exhausted": bool(nodes >= node_budget),
        "line_len": len(line),
        "wall_s": dt,
    }


def run_tier(traps: Sequence[Dict], depth: int, node_budget: int, workers: int, neighbor_dist: int = NEIGHBOR_DIST) -> List[Dict]:
    jobs = [(t, depth, node_budget, neighbor_dist) for t in traps]
    if workers <= 1:
        _worker_init(0)
        return [_prove_one(j) for j in jobs]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(10,)) as pool:
        return pool.map(_prove_one, jobs, chunksize=1)


def summarize(records: List[Dict], tier_name: str, depth: int, node_budget: int, neighbor_dist: int = NEIGHBOR_DIST) -> Dict:
    n_total = len(records)
    n_proven = sum(1 for r in records if r["proven_win_for_attacker"])
    ci_lo, ci_hi = wilson_interval(n_proven, n_total)
    walls = sorted(r["wall_s"] for r in records)
    nodes = sorted(r["nodes"] for r in records)
    n_exhausted = sum(1 for r in records if r["budget_exhausted"])

    def pct(xs, p):
        if not xs:
            return None
        k = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
        return xs[k]

    by_mate: Dict[str, List[bool]] = {}
    for r in records:
        mb = str(int(round(float(r["mate_distance"])))) if r["mate_distance"] is not None else "unknown"
        by_mate.setdefault(mb, []).append(r["proven_win_for_attacker"])
    breakdown = {mb: {"n": len(v), "proven": sum(v), "fraction": sum(v) / len(v)} for mb, v in by_mate.items()}

    return {
        "tier": tier_name,
        "depth": depth,
        "node_budget": node_budget,
        "solver_config": {"window_half": WINDOW_HALF, "cand_cap": CAND_CAP, "neighbor_dist": neighbor_dist},
        "n_total": n_total,
        "n_proven": n_proven,
        "fraction": n_proven / n_total if n_total else 0.0,
        "wilson95_lo": ci_lo,
        "wilson95_hi": ci_hi,
        "n_budget_exhausted": n_exhausted,
        "frac_budget_exhausted": n_exhausted / n_total if n_total else 0.0,
        "wall_s_median": statistics.median(walls) if walls else None,
        "wall_s_p95": pct(walls, 0.95),
        "wall_s_min": walls[0] if walls else None,
        "wall_s_max": walls[-1] if walls else None,
        "wall_s_total": sum(walls),
        "nodes_median": statistics.median(nodes) if nodes else None,
        "nodes_p95": pct(nodes, 0.95),
        "nodes_min": nodes[0] if nodes else None,
        "nodes_max": nodes[-1] if nodes else None,
        "breakdown_by_mate_distance": breakdown,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True, help="tier name, e.g. baseline_1x / 10x / 50x / sealbot_equiv")
    ap.add_argument("--node-budget", type=int, required=True)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--records", default=DEFAULT_RECORDS)
    ap.add_argument("--traps", default=DEFAULT_TRAPS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--neighbor-dist", type=int, default=NEIGHBOR_DIST,
                     help="quiet-move candidate-widening radius (default matches the FIX2a "
                          "in-loop config, 2); raise this to test candidate-gen width as a "
                          "lever SEPARATE from node_budget")
    ap.add_argument("--out-prefix", required=True, help="writes <prefix>.json + <prefix>_records.jsonl")
    ap.add_argument("--allow-non-40", action="store_true",
                     help="skip the n==40 fixture-integrity assert (SMOKE TESTING ONLY — "
                          "any real tier run must use the full, untouched 40-position fixture)")
    args = ap.parse_args()

    pos_ids = load_fixture_pos_ids(Path(args.records))
    if not args.allow_non_40:
        assert len(pos_ids) == 40, f"expected 40 fixture pos_ids, got {len(pos_ids)}"
    traps = load_traps_by_pos_id(Path(args.traps), pos_ids)
    print(f"[f2] tier={args.tier} depth={args.depth} node_budget={args.node_budget} "
          f"neighbor_dist={args.neighbor_dist} workers={args.workers} n={len(traps)}", flush=True)

    t0 = time.time()
    records = run_tier(traps, args.depth, args.node_budget, args.workers, args.neighbor_dist)
    elapsed = time.time() - t0

    # preserve fixture order in the output records file
    order = {pid: i for i, pid in enumerate(pos_ids)}
    records.sort(key=lambda r: order[r["pos_id"]])

    summary = summarize(records, args.tier, args.depth, args.node_budget, args.neighbor_dist)
    summary["wall_clock_elapsed_s"] = elapsed
    summary["workers"] = args.workers

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_path = Path(str(out_prefix) + ".json")
    records_path = Path(str(out_prefix) + "_records.jsonl")
    summary_path.write_text(json.dumps(summary, indent=2))
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[f2] {args.tier}: {summary['n_proven']}/{summary['n_total']} proven "
          f"({summary['fraction']:.4f}) wilson95=[{summary['wilson95_lo']:.4f}, "
          f"{summary['wilson95_hi']:.4f}] budget_exhausted={summary['n_budget_exhausted']}/{summary['n_total']} "
          f"wall_median={summary['wall_s_median']:.2f}s wall_p95={summary['wall_s_p95']:.2f}s "
          f"elapsed_wall_clock={elapsed:.1f}s", flush=True)
    print(f"[f2] wrote {summary_path} + {records_path}", flush=True)


if __name__ == "__main__":
    main()
