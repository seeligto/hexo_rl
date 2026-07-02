#!/usr/bin/env python
"""D-WS3V3 — MEASURED seed-corpus / eval-corpus disjointness check.

The seed corpus is built with an exclusion set at mine-time
(`build_ws3v3_seed_corpus.py` -> `dpfit_mine_heldout_traps.py
--exclude-trap-sets`), but exclusion-by-construction is a claim, not a proof —
this script MEASURES it against the actual eval JSONLs after the fact:

  1. GAME-LEVEL: {(source_file, source_game_idx)} for the seed corpus must not
     intersect the same tuple set for any eval trap.
  2. POSITION-LEVEL: no seed's `seed_moves` may be a prefix of (or equal to)
     any eval trap's `parent_move_seq` (a seed that walks INTO an eval
     position would train on what the eval later scores) — and no `seed_id`
     may collide with any eval `pos_id` (id-namespace clash) or with another
     `seed_id` (corpus-internal dup).
  3. LEAKAGE DISTANCE (descriptive, not gated): for each seed, the max
     stone-set Jaccard similarity between the seed's TERMINAL board (after
     replaying `seed_moves`) and any eval trap's PARENT board — printed as a
     distribution (max/p95/median) so seed-neighborhood leakage near an eval
     trap is a measured number, not an assumption.

Exit code 0 iff (1) and (2) are both clean; leakage distance never affects
the exit code (it is a magnitude-of-risk report, not a violation).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engine  # noqa: E402

DEFAULT_SEED_CORPUS = "reports/d_ws3v3/seed_corpus.jsonl"
DEFAULT_EVAL_TRAPS = (
    "reports/d_tactical_2026-06-26/heldout_traps.jsonl",
    "reports/d_tactical_2026-06-26/heldout_traps_expanded.jsonl",
    "reports/d_tactical_2026-06-26/heldout_traps_all.jsonl",
)
DEFAULT_EVAL_SOURCE_FILE = "per_game_seald5.jsonl"


def replay(seq, encoding):
    b = engine.Board.with_encoding_name(encoding)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def load_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval_traps(paths: Sequence[str]) -> List[Dict]:
    """Load + concatenate every eval JSONL that exists (missing paths WARN,
    not fatal)."""
    traps: List[Dict] = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            print(f"[disjointness] WARNING eval trap-set not found, skipping: {pp}", file=sys.stderr)
            continue
        traps.extend(load_jsonl(pp))
    return traps


def seed_game_key(seed: Dict) -> Tuple[str, int]:
    return (str(seed["source_file"]), int(seed["source_game_idx"]))


def eval_game_key(trap: Dict, default_source_file: str) -> Tuple[str, int]:
    return (str(trap.get("source_file", default_source_file)), int(trap["game_idx"]))


def check_game_level(
    seeds: Sequence[Dict], eval_traps: Sequence[Dict], eval_source_file: str,
) -> set:
    seed_keys = {seed_game_key(s) for s in seeds}
    eval_keys = {eval_game_key(t, eval_source_file) for t in eval_traps}
    return seed_keys & eval_keys


def _is_prefix_or_equal(shorter: Sequence, longer: Sequence) -> bool:
    a = tuple(tuple(int(x) for x in pair) for pair in shorter)
    b = tuple(tuple(int(x) for x in pair) for pair in longer)
    return len(a) <= len(b) and b[: len(a)] == a


def check_position_level(
    seeds: Sequence[Dict], eval_traps: Sequence[Dict],
) -> Tuple[List[str], set, List[Tuple[str, str]]]:
    """Returns (duplicate_seed_ids, seed_id-vs-pos_id namespace collisions,
    seed-is-prefix-of-eval-parent violations)."""
    seed_ids = [s["seed_id"] for s in seeds]
    dup_seed_ids = sorted(sid for sid, c in Counter(seed_ids).items() if c > 1)

    eval_pos_ids = {t["pos_id"] for t in eval_traps if "pos_id" in t}
    namespace_collisions = set(seed_ids) & eval_pos_ids

    prefix_violations: List[Tuple[str, str]] = []
    for s in seeds:
        seed_moves = s.get("seed_moves") or []
        for t in eval_traps:
            parent_seq = t.get("parent_move_seq") or []
            if not parent_seq:
                continue
            if _is_prefix_or_equal(seed_moves, parent_seq):
                prefix_violations.append((s["seed_id"], t.get("pos_id", "?")))

    return dup_seed_ids, namespace_collisions, prefix_violations


def _stone_set(board) -> frozenset:
    # Occupied-cell shape, irrespective of which player owns each stone —
    # leakage risk is about board-REGION proximity, not stone-identity.
    return frozenset((int(q), int(r)) for q, r, _p in board.get_stones())


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def leakage_distances(
    seeds: Sequence[Dict], eval_traps: Sequence[Dict], encoding: str,
) -> List[float]:
    eval_stone_sets = []
    for t in eval_traps:
        seq = t.get("parent_move_seq")
        if not seq:
            continue
        try:
            b = replay(seq, t.get("encoding", encoding))
        except Exception:  # noqa: BLE001 — a malformed eval row must not crash the leakage scan
            continue
        eval_stone_sets.append(_stone_set(b))

    dists = []
    for s in seeds:
        seq = s.get("seed_moves")
        if not seq:
            continue
        try:
            b = replay(seq, encoding)
        except Exception:  # noqa: BLE001
            continue
        sset = _stone_set(b)
        best = max((_jaccard(sset, es) for es in eval_stone_sets), default=0.0)
        dists.append(best)
    return dists


def distribution_stats(values: Sequence[float]) -> Dict:
    if not values:
        return {"n": 0, "max": None, "p95": None, "median": None, "min": None, "mean": None}
    arr = sorted(values)
    n = len(arr)
    median = arr[n // 2] if n % 2 == 1 else (arr[n // 2 - 1] + arr[n // 2]) / 2.0
    p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
    return {
        "n": n, "max": arr[-1], "p95": arr[p95_idx], "median": median,
        "min": arr[0], "mean": sum(arr) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-corpus", default=DEFAULT_SEED_CORPUS)
    ap.add_argument("--eval-traps", nargs="*", default=list(DEFAULT_EVAL_TRAPS))
    ap.add_argument("--encoding", default="v6_live2_ls")
    ap.add_argument(
        "--eval-source-file", default=DEFAULT_EVAL_SOURCE_FILE,
        help="source_file assumed for eval traps that don't carry one explicitly "
             f"(default: {DEFAULT_EVAL_SOURCE_FILE} — the shared per_game_seald5 origin).",
    )
    args = ap.parse_args()

    seed_path = Path(args.seed_corpus)
    if not seed_path.exists():
        print(f"[disjointness] ERROR: seed corpus not found: {seed_path}", file=sys.stderr)
        sys.exit(2)
    seeds = load_jsonl(seed_path)
    eval_traps = load_eval_traps(args.eval_traps)

    print(f"[disjointness] {len(seeds)} seeds from {seed_path}; {len(eval_traps)} eval traps "
          f"from {sum(1 for p in args.eval_traps if Path(p).exists())}/{len(args.eval_traps)} files", flush=True)

    game_collisions = check_game_level(seeds, eval_traps, args.eval_source_file)
    dup_seed_ids, namespace_collisions, prefix_violations = check_position_level(seeds, eval_traps)
    dists = leakage_distances(seeds, eval_traps, args.encoding)
    stats = distribution_stats(dists)

    print("\n=== (1) game-level: (source_file, game_idx) intersection ===")
    if game_collisions:
        print(f"VIOLATION: {len(game_collisions)} colliding game key(s): {sorted(game_collisions)[:20]}")
    else:
        print("clean: 0 intersecting (source_file, game_idx) pairs")

    print("\n=== (2) position-level ===")
    if dup_seed_ids:
        print(f"VIOLATION: {len(dup_seed_ids)} duplicate seed_id(s) within the corpus: {dup_seed_ids[:20]}")
    else:
        print("clean: 0 duplicate seed_id within the seed corpus")
    if namespace_collisions:
        print(f"VIOLATION: {len(namespace_collisions)} seed_id collides with an eval pos_id: {sorted(namespace_collisions)[:20]}")
    else:
        print("clean: 0 seed_id / eval pos_id namespace collisions")
    if prefix_violations:
        print(f"VIOLATION: {len(prefix_violations)} seed_moves is a prefix of (or equal to) an eval parent_move_seq:")
        for sid, pid in prefix_violations[:20]:
            print(f"    {sid}  ⊆  {pid}")
    else:
        print("clean: 0 seed_moves prefix-of-eval-parent violations")

    print("\n=== (3) leakage distance (MEASURED, not gated) ===")
    print(f"  n={stats['n']}  max={stats['max']}  p95={stats['p95']}  "
          f"median={stats['median']}  min={stats['min']}  mean={stats['mean']}")

    n_violations = (
        len(game_collisions) + len(dup_seed_ids) + len(namespace_collisions) + len(prefix_violations)
    )
    if n_violations:
        print(f"\nFAIL: {n_violations} disjointness violation(s).", file=sys.stderr)
        sys.exit(1)
    print("\nPASS: seed corpus is disjoint from every loaded eval trap-set.")
    sys.exit(0)


if __name__ == "__main__":
    main()
