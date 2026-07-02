#!/usr/bin/env python
"""D-WS3V3 FIX2a — MEASURE the native solver's provable fraction on POST-blunder
positions (the honest ceiling for ARM-SEEDED's post-blunder seed class).

Red-team finding this responds to: seed landing positions at cuts k∈{0,2,4}
(before the trap PARENT) prove NO forced win by construction (the parent is a
"saving move exists" position — the model hasn't blundered yet), and even the
POST-blunder position (where the defender IS proven lost, per the SealBot-based
miner) is often NOT provable by the native `engine::tactics::TacticalSolver` at
a realistic self-play node budget — D-TACTICAL A2's "native solver has weak
recall on quiet traps" finding. ARM-SEEDED's naive "fire_rate_seeded ~1.0"
expectation is wrong; this script measures the real ceiling.

For each trap in the 125-trap combined held-out corpus, replay `post_move_seq`
(the board where the DEFENDER — the model — is to move and is proven lost by
SealBot) and ask the native solver to prove the position from the side-to-move
at POST. A proven LOSS for the side-to-move at POST (`result == -1`) is
equivalently a proven forced WIN for the ATTACKER — the frame this script
reports the fraction in ("proven WIN"). `result == -1` and the miner's stored
(and NOT trusted here — recomputed fresh, per the DS1 stale-label lesson)
`native_loss_verified` flag are the same underlying computation.

Usage:
  python scripts/measure_native_provable_fraction.py \\
      --traps reports/d_tactical_2026-06-26/heldout_traps_all.jsonl \\
      --out reports/d_ws3v3/native_provable_fraction.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engine  # noqa: E402

DEFAULT_TRAPS = "reports/d_tactical_2026-06-26/heldout_traps_all.jsonl"
DEFAULT_OUT = "reports/d_ws3v3/native_provable_fraction.json"
DEFAULT_ENCODING = "v6_live2_ls"

# Solver config per the FIX2a spec — neighbor_dist=2 (quiet-move widening,
# matches the v3 selfplay solver_neighbor_dist knob), window_half=None
# (no in-window offense guard — POST is model-to-move deep in the game, not
# necessarily near the encoding's window center), cand_cap=40 (matches the
# mining script's candidate cap).
WINDOW_HALF = None
CAND_CAP = 40
NEIGHBOR_DIST = 2
DEPTH = 16
NODE_BUDGET = 20000


def replay(seq, encoding: str):
    b = engine.Board.with_encoding_name(encoding)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def load_traps(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _mate_bucket(mate_distance) -> str:
    if mate_distance is None:
        return "unknown"
    return str(int(round(float(mate_distance))))


def sample_traps(traps: Sequence[Dict], n: int, seed: int) -> List[Dict]:
    """Deterministic uniform random subset of `n` traps (no replacement),
    order-preserving (selected traps keep their file order). `n <= 0` or
    `n >= len(traps)` returns the full list unchanged."""
    if n <= 0 or n >= len(traps):
        return list(traps)
    idx = sorted(random.Random(seed).sample(range(len(traps)), n))
    return [traps[i] for i in idx]


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traps", default=DEFAULT_TRAPS)
    ap.add_argument("--encoding", default=DEFAULT_ENCODING,
                     help="fallback encoding for traps that don't carry their own 'encoding' field")
    ap.add_argument("--depth", type=int, default=DEPTH)
    ap.add_argument("--node-budget", type=int, default=NODE_BUDGET)
    ap.add_argument("--neighbor-dist", type=int, default=NEIGHBOR_DIST)
    ap.add_argument("--cand-cap", type=int, default=CAND_CAP)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--sample", type=int, default=0,
        help="measure a deterministic uniform random subset of N traps "
             "(0 = full set). Debug-build laptop pace is ~60-80s/trap; the "
             "full 125 is ~2.5h — sample for a CI'd estimate, run the full "
             "set on a vast release build (~2.5x faster).",
    )
    ap.add_argument("--seed", type=int, default=20260702,
                     help="RNG seed for --sample subset selection (deterministic)")
    args = ap.parse_args()

    traps_path = Path(args.traps)
    if not traps_path.exists():
        print(f"[measure] ERROR: traps file not found: {traps_path}", file=sys.stderr)
        sys.exit(2)
    traps_all = load_traps(traps_path)
    traps = sample_traps(traps_all, args.sample, args.seed)
    if len(traps) < len(traps_all):
        print(f"[measure] SAMPLED {len(traps)}/{len(traps_all)} traps "
              f"(uniform, seed={args.seed}, order-preserving)", flush=True)
    print(f"[measure] {len(traps)} traps to scan from {traps_path}", flush=True)

    solver = engine.TacticalSolver(window_half=WINDOW_HALF, cand_cap=args.cand_cap, neighbor_dist=args.neighbor_dist)

    n_proven = 0
    n_total = 0
    n_skipped = 0
    by_mate: Dict[str, List[bool]] = defaultdict(list)
    records: List[Dict] = []
    t0 = time.time()

    for i, t in enumerate(traps):
        post_seq = t.get("post_move_seq")
        if not post_seq:
            n_skipped += 1
            continue
        encoding = t.get("encoding", args.encoding)
        try:
            board = replay(post_seq, encoding)
        except Exception as exc:  # noqa: BLE001 — a malformed row must not crash the whole scan
            print(f"[measure] WARNING: replay failed for {t.get('pos_id', i)}: {exc}", file=sys.stderr)
            n_skipped += 1
            continue

        result, line, nodes = solver.prove(board, args.depth, args.node_budget)
        proven_win_for_attacker = result == -1  # LOSS for defender-to-move at POST

        n_total += 1
        n_proven += int(proven_win_for_attacker)
        mate_bucket = _mate_bucket(t.get("mate_distance"))
        by_mate[mate_bucket].append(proven_win_for_attacker)

        records.append({
            "pos_id": t.get("pos_id"),
            "bucket": t.get("bucket"),
            "mate_distance": t.get("mate_distance"),
            "in_window": t.get("in_window"),
            "solver_result": result,
            "proven_win_for_attacker": bool(proven_win_for_attacker),
            "nodes": nodes,
            "line_len": len(line),
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(traps):
            print(f"[measure] {i+1}/{len(traps)} scanned, {n_proven}/{n_total} proven "
                  f"({time.time()-t0:.0f}s)", flush=True)

    fraction = (n_proven / n_total) if n_total else 0.0
    ci_lo, ci_hi = wilson_interval(n_proven, n_total)

    print("\n=== FIX2a native-provable fraction (POST-blunder, honest ceiling) ===")
    print(f"solver: window_half={WINDOW_HALF} cand_cap={args.cand_cap} "
          f"neighbor_dist={args.neighbor_dist} depth={args.depth} node_budget={args.node_budget}")
    if len(traps) < len(traps_all):
        print(f"SAMPLED estimate: {len(traps)}/{len(traps_all)} traps, seed={args.seed}")
    print(f"n_total={n_total}  n_proven={n_proven}  fraction={fraction:.4f}  "
          f"wilson95=[{ci_lo:.4f}, {ci_hi:.4f}]  n_skipped={n_skipped}")
    print("\nbreakdown by mate_distance (turns):")
    breakdown = {}
    for mb in sorted(by_mate.keys(), key=lambda k: (k == "unknown", float(k) if k != "unknown" else 0)):
        vals = by_mate[mb]
        n = len(vals)
        p = sum(vals)
        breakdown[mb] = {"n": n, "proven": p, "fraction": p / n if n else 0.0}
        print(f"  mate_distance={mb:>8}: {p:>3}/{n:<3} = {p/n if n else 0.0:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "traps_path": str(traps_path),
        "solver_config": {
            "window_half": WINDOW_HALF, "cand_cap": args.cand_cap,
            "neighbor_dist": args.neighbor_dist, "depth": args.depth,
            "node_budget": args.node_budget,
        },
        "sampled": len(traps) < len(traps_all),
        "sample_n": args.sample if len(traps) < len(traps_all) else None,
        "sample_seed": args.seed if len(traps) < len(traps_all) else None,
        "population_n": len(traps_all),
        "n_total": n_total,
        "n_proven": n_proven,
        "fraction": fraction,
        "wilson95_lo": ci_lo,
        "wilson95_hi": ci_hi,
        "n_skipped": n_skipped,
        "breakdown_by_mate_distance": breakdown,
        "elapsed_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    records_path = out_path.with_name(out_path.stem + "_records.jsonl")
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\n[measure] summary -> {out_path}")
    print(f"[measure] per-trap records -> {records_path}")


if __name__ == "__main__":
    main()
