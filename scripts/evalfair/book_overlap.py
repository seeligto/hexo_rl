"""evalfair/book_overlap.py — book-vs-training-corpus opening overlap check (WP0.5).

Same method as the R2c red-team book-overlap check (`reports/probes/gnn_bc/r2/
R2C_book_overlap.py`, not itself committed): exact-prefix match + a D6 hex-symmetry-folded
match. Generalized here for evalfair book_v2/v3 fixtures (`BOOK_PLIES = 3`, see
`scripts/evalfair/core.py`) and the PINNED run3 training corpus.

Symmetry-folded is NOT merely a stricter upper bound for this repo: `configs/training.yaml`
sets `augment: true` (12-fold hex symmetry augmentation on `sample_batch`) and no active
run3 variant overrides it — the network trains on all 12 symmetric images of every corpus
position. A book opening whose rotated/reflected image matches a training opening genuinely
was seen (in some frame) during training, so BOTH exact and symmetry-folded overlap must be
zero, not just exact.

Training-corpus reconstruction: the pinned run3 NPZ (`data/bootstrap_corpus_v6_live2_ls.npz`,
sha registered in `docs/registers/run3_corpus_manifest.md`) is built by globbing
`data/corpus/raw_human/*.json` at export time — no per-game manifest is retained in the NPZ
itself. `raw_human` is append-only (daily scrape), so the exact 8669-game set that fed the
frozen NPZ is reconstructed as "every raw_human file present by the NPZ sidecar's
`created_at`" (mtime <= cutoff). This reproduces the manifest's documented 8669-included /
29-excluded split exactly (independently verified at WP0.5 time).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
RAW_HUMAN_DIR = REPO / "data/corpus/raw_human"

# NPZ sidecar `created_at: 2026-07-04T07:56:30Z` (docs/registers/run3_corpus_manifest.md §1).
RUN3_CORPUS_CUTOFF_EPOCH = 1783151790.0

BOOK_PLIES = 3  # evalfair book_v2/v3 opening length (scripts/evalfair/core.py BOOK_PLIES)


def _cube(q: int, r: int) -> Tuple[int, int, int]:
    return (q, -q - r, r)


def _axial(x: int, y: int, z: int) -> Tuple[int, int]:
    return (x, z)


def _hex_syms():
    """The 12 D6 hex symmetries about the origin (cube-coord signed permutations)."""
    syms = []
    for perm in permutations(range(3)):
        for sign in (1, -1):
            def make(perm=perm, sign=sign):
                def f(q, r):
                    c = _cube(q, r)
                    p = (c[perm[0]], c[perm[1]], c[perm[2]])
                    p = (sign * p[0], sign * p[1], sign * p[2])
                    return _axial(*p)
                return f
            f = make()
            ok = True
            for (tq, tr) in [(0, 0), (1, 0), (0, 1), (2, -1), (-3, 2)]:
                x, z = f(tq, tr)
                if x + (-x - z) + z != 0:
                    ok = False
                    break
            if ok:
                syms.append(f)
    uniq = {}
    for f in syms:
        key = (f(1, 0), f(0, 1))
        uniq[key] = f
    return list(uniq.values())


SYMS = _hex_syms()
assert len(SYMS) == 12, f"expected 12 hex symmetries, got {len(SYMS)}"


def sym_variants(opening: Tuple[Tuple[int, int], ...]) -> set:
    return {tuple(f(q, r) for (q, r) in opening) for f in SYMS}


def load_training_prefixes(
    raw_human_dir: Path = RAW_HUMAN_DIR,
    cutoff_epoch: float = RUN3_CORPUS_CUTOFF_EPOCH,
    book_plies: int = BOOK_PLIES,
) -> Tuple[Counter, int, int]:
    """Return (Counter of N-ply move-prefixes, n_training_files, n_excluded_files)."""
    all_files = sorted(glob.glob(str(raw_human_dir / "*.json")))
    training_files = [f for f in all_files if os.path.getmtime(f) <= cutoff_epoch]
    excluded_files = [f for f in all_files if os.path.getmtime(f) > cutoff_epoch]
    prefixes: List[Tuple[Tuple[int, int], ...]] = []
    for name in training_files:
        d = json.loads(Path(name).read_text())
        mv = d.get("moves", [])
        if len(mv) < book_plies:
            continue
        prefixes.append(tuple((m["x"], m["y"]) for m in mv[:book_plies]))
    return Counter(prefixes), len(training_files), len(excluded_files)


def check_book(book: Dict, pref_counter: Counter) -> Dict:
    """Exact + D6-symmetry-folded overlap counts for one loaded book_v2/v3 dict."""
    openings = [tuple(tuple(m) for m in o["moves"]) for o in book["openings"]]
    n_book = len(openings)

    exact_per_open = {i: pref_counter.get(op, 0) for i, op in enumerate(openings)}
    exact_hit = sum(1 for c in exact_per_open.values() if c > 0)
    exact_games = sum(exact_per_open.values())

    sym_per_open = {}
    for i, op in enumerate(openings):
        sym_per_open[i] = sum(pref_counter.get(v, 0) for v in sym_variants(op))
    sym_hit = sum(1 for c in sym_per_open.values() if c > 0)
    sym_games = sum(sym_per_open.values())

    return {
        "book_id": book.get("book_id"),
        "n_book_openings": n_book,
        "exact": {
            "openings_with_ge1_match": exact_hit,
            "total_training_games_matching_any_opening": exact_games,
            "per_opening_counts": exact_per_open,
        },
        "symmetry_folded_D6": {
            "openings_with_ge1_match": sym_hit,
            "total_training_games_matching_any_opening": sym_games,
            "per_opening_counts": sym_per_open,
            "note": "12 hex symmetries about origin; load-bearing (not just an upper "
                    "bound) because configs/training.yaml augment=true is live at train "
                    "time for run3.",
        },
        "clean": exact_hit == 0 and sym_hit == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("book_paths", nargs="+", help="Path(s) to book_v2/v3 JSON fixtures")
    ap.add_argument("--out", default=None, help="Optional path to write the full JSON report")
    args = ap.parse_args()

    pref_counter, n_training, n_excluded = load_training_prefixes()
    print(f"[book_overlap] raw_human training(<=cutoff)={n_training} excluded(>cutoff)={n_excluded}")

    results = {}
    any_dirty = False
    for bp in args.book_paths:
        book = json.loads(Path(bp).read_text())
        r = check_book(book, pref_counter)
        results[bp] = r
        status = "CLEAN" if r["clean"] else "OVERLAP DETECTED"
        print(
            f"[book_overlap] {r['book_id']}: EXACT "
            f"{r['exact']['openings_with_ge1_match']}/{r['n_book_openings']} "
            f"(games={r['exact']['total_training_games_matching_any_opening']})  "
            f"SYMFOLD {r['symmetry_folded_D6']['openings_with_ge1_match']}/{r['n_book_openings']} "
            f"(games={r['symmetry_folded_D6']['total_training_games_matching_any_opening']})  "
            f"-> {status}"
        )
        any_dirty = any_dirty or not r["clean"]

    if args.out:
        Path(args.out).write_text(json.dumps({
            "cutoff_epoch": RUN3_CORPUS_CUTOFF_EPOCH,
            "n_training_files": n_training,
            "n_excluded_files": n_excluded,
            "results": results,
        }, indent=2))
        print(f"[book_overlap] wrote {args.out}")

    return 1 if any_dirty else 0


if __name__ == "__main__":
    raise SystemExit(main())
