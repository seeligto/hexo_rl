"""Compute axis-distribution baseline from the bootstrap corpus.

Reads the corpus NPZ (default: data/bootstrap_corpus.npz or as configured),
computes per-axis same-color stone-pair fractions using the 18-plane state
representation, and writes the result to reports/baselines/corpus_axis_distribution.json.

Usage:
    python scripts/compute_corpus_axis_baseline.py
    python scripts/compute_corpus_axis_baseline.py --corpus data/bootstrap_corpus_v5.npz
    python scripts/compute_corpus_axis_baseline.py --corpus data/bootstrap_corpus.npz --output reports/baselines/corpus_axis_distribution.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from hexo_rl.training.axis_distribution import compute_axis_fractions_from_states  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute corpus axis-distribution baseline")
    p.add_argument(
        "--corpus",
        default=None,
        help="Path to corpus .npz (default: value from configs/training.yaml or data/bootstrap_corpus.npz)",
    )
    p.add_argument(
        "--output",
        default="reports/baselines/corpus_axis_distribution.json",
        help="Output JSON path (default: reports/baselines/corpus_axis_distribution.json)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Subsample at most N positions (0 = no cap, process full corpus)",
    )
    return p.parse_args()


def _resolve_corpus_path(args: argparse.Namespace) -> Path:
    if args.corpus:
        return Path(args.corpus)

    # Try to read from training config
    try:
        from hexo_rl.utils.config import load_config
        cfg = load_config("configs/training.yaml")
        path_str = cfg.get("training", cfg).get("pretrained_buffer_path", "")
        if path_str:
            return Path(path_str)
    except Exception:
        pass

    return Path("data/bootstrap_corpus.npz")


def main() -> None:
    args = _parse_args()
    corpus_path = _resolve_corpus_path(args)

    if not corpus_path.exists():
        print(f"ERROR: corpus not found at {corpus_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading corpus: {corpus_path}")
    t0 = time.monotonic()
    npz = np.load(corpus_path, mmap_mode="r")
    states = npz["states"]  # (N, C, H, W) — C >= 9 required (planes 0, 8)
    print(f"  Shape: {states.shape}  dtype: {states.dtype}")

    if args.max_samples > 0 and len(states) > args.max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(states), size=args.max_samples, replace=False)
        idx.sort()
        states = states[idx]
        print(f"  Subsampled to {len(states)} positions")

    print("Computing axis fractions…")
    # Load into memory for vectorised computation (mmap passes ndarray slices)
    states_arr = np.array(states, dtype=np.float32)
    result = compute_axis_fractions_from_states(states_arr)

    elapsed = time.monotonic() - t0
    print(f"  axis_q = {result['axis_q']:.4f}")
    print(f"  axis_r = {result['axis_r']:.4f}")
    print(f"  axis_s = {result['axis_s']:.4f}")
    print(f"  axis_max = {result['axis_max']}")
    print(f"  Elapsed: {elapsed:.1f}s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "axis_q": result["axis_q"],
        "axis_r": result["axis_r"],
        "axis_s": result["axis_s"],
        "axis_max": result["axis_max"],
        "corpus_path": str(corpus_path),
        "n_positions": len(states_arr),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Baseline written to {output_path}")


if __name__ == "__main__":
    main()
