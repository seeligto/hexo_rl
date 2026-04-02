#!/usr/bin/env python3
"""Export the bootstrap corpus to data/bootstrap_corpus.npz.

Calls load_corpus() from python.bootstrap.pretrain, applies quality-score and
source weighting, and writes a .npz that scripts/train.py can load directly.

Arrays saved:
    states:   float16, (N, 18, 19, 19)
    policies: float32, (N, 362)
    outcomes: float32, (N,)
    weights:  float32, (N,)  — quality × source weight per position

Usage:
    python scripts/export_corpus_npz.py
    make corpus.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.bootstrap.pretrain import load_corpus


def main() -> None:
    out_path = ROOT / "data" / "bootstrap_corpus.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from python.utils.config import load_config
    corpus_cfg = load_config(str(ROOT / "configs" / "corpus_filter.yaml"))
    source_weights: dict = corpus_cfg.get("source_weights", {})

    quality_path = ROOT / "data" / "corpus" / "quality_scores.json"
    quality_scores: dict = {}
    if quality_path.exists():
        with open(quality_path) as f:
            quality_scores = json.load(f)
        print(f"Quality scores loaded: {len(quality_scores)} games")
    else:
        print("Warning: no quality_scores.json — using default weight 0.5 for all games")
        print("  Run `make corpus.analysis` to generate quality scores first")

    print("Loading corpus...")
    states, policies, outcomes, weights = load_corpus(quality_scores, source_weights)

    if len(outcomes) == 0:
        print("ERROR: No corpus data found. Run `make corpus.all` first.")
        sys.exit(1)

    # Source breakdown from quality_scores keys, approximated by position count per source.
    # load_corpus logs counts via structlog; we just print the totals here.
    print(f"\nCorpus loaded:")
    print(f"  Positions : {len(outcomes):,}")
    print(f"  States    : {states.shape}  dtype={states.dtype}")
    print(f"  Policies  : {policies.shape}  dtype={policies.dtype}")
    print(f"  Outcomes  : {outcomes.shape}  dtype={outcomes.dtype}")
    print(f"  Weights   : {weights.shape}  min={weights.min():.4f} max={weights.max():.4f}")

    print(f"\nSaving to {out_path} ...")
    np.savez_compressed(
        out_path,
        states=states,
        policies=policies,
        outcomes=outcomes,
        weights=weights,
    )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
