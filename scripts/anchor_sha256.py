#!/usr/bin/env python
"""Print the deterministic state-dict sha256 of a checkpoint's MODEL weights.

§D-LOOPFIX W2 — use this to compute the value an A/B / sustained run pins into
``eval_pipeline.gating.expected_anchor_sha256``. The launch preflight
(``resolve_anchor``) recomputes the SAME hash over the resolved
``best_model.pt`` weights and refuses to start on mismatch, closing the silent
restore-from-.bak hole that installed golong@50k-PEAK as the A/B incumbent.

The hash is over MODEL WEIGHTS ONLY (sorted keys + raw tensor bytes), so it
matches whether the source is a full training checkpoint (model_state +
optimizer + ...) or a bare/stamped anchor payload — the same number identifies
the same weights across save formats.

Usage:
    python scripts/anchor_sha256.py checkpoints/bootstrap_model_v6_live2.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.training.anchor import checkpoint_state_sha256


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", help="path to a .pt checkpoint / anchor")
    args = ap.parse_args()

    # SAME code path resolve_anchor uses for the W2 pin check — single source of
    # truth, so the pin can never drift from the runtime comparison (§D-RERUNPREP F1).
    print(checkpoint_state_sha256(Path(args.checkpoint)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
