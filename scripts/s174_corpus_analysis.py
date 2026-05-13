"""§174 v6w25 investigation — corpus opening-position analysis (1.1).

Counts stones-per-position for v6 + v6w25 bootstrap corpora, dumps ply
distributions + summary stats to stdout. Run via .venv python.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


def analyze(corpus_path: Path, name: str, n_planes: int, board_size: int) -> dict:
    """Walk states, count stones per position to estimate ply."""
    print(f"\n=== {name}: {corpus_path}", flush=True)
    t0 = time.time()
    c = np.load(corpus_path, mmap_mode="r")
    states = c["states"]
    n = len(states)
    print(f"  states shape: {states.shape}  dtype: {states.dtype}", flush=True)

    # Planes 0 = current player t0, plane 4 = opponent t0 (kept-plane convention).
    # See registry.toml — kept_plane_indices = [0,1,2,3, 8,9,10,11] sliced from
    # the 18-plane source means: 0=current t0, 4=opponent t0 (after slice).
    own_plane = 0
    opp_plane = 4

    chunk = 5000
    counts = np.zeros(n, dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = np.asarray(states[start:end]).astype(np.float32)
        own = (block[:, own_plane] > 0.5).sum(axis=(1, 2))
        opp = (block[:, opp_plane] > 0.5).sum(axis=(1, 2))
        counts[start:end] = own + opp
        if start % (chunk * 10) == 0:
            elapsed = time.time() - t0
            print(f"    progress: {end}/{n} ({100*end/n:.1f}%) elapsed={elapsed:.1f}s", flush=True)

    dist = {}
    for threshold in [2, 4, 6, 8, 10, 12, 15, 20, 30, 50, 100]:
        cnt = int((counts <= threshold).sum())
        dist[threshold] = (cnt, 100.0 * cnt / n)

    summary = {
        "name": name,
        "n_positions": n,
        "mean_stones": float(counts.mean()),
        "median_stones": float(np.median(counts)),
        "max_stones": int(counts.max()),
        "min_stones": int(counts.min()),
        "distribution": dist,
    }
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.1f}s", flush=True)
    print(f"  n_positions: {n}", flush=True)
    print(f"  mean stones: {summary['mean_stones']:.1f}", flush=True)
    print(f"  median stones: {summary['median_stones']:.1f}", flush=True)
    print(f"  max stones: {summary['max_stones']}", flush=True)
    print(f"  min stones: {summary['min_stones']}", flush=True)
    print("  ply ≤ threshold cumulative:", flush=True)
    for threshold, (cnt, pct) in dist.items():
        print(f"    ply ≤ {threshold:3d}: {cnt:7d} ({pct:5.2f}%)", flush=True)

    return summary


def main() -> int:
    out_path = Path("reports/s174_corpus_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    results["v6w25"] = analyze(
        Path("data/bootstrap_corpus_v6w25.npz"),
        "v6w25", n_planes=8, board_size=25,
    )
    results["v6"] = analyze(
        Path("data/bootstrap_corpus.npz"),
        "v6 (default)", n_planes=8, board_size=19,
    )

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
