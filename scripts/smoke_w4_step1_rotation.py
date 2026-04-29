#!/usr/bin/env python3
"""W4 Step 1 — laptop smoke for the §130 per-game self-play rotation port.

Spins two WorkerPool runs over `checkpoints/bootstrap_model.pt`: one with
`selfplay.rotation_enabled=False` (canonical baseline) and one with
`selfplay.rotation_enabled=True` (rotation port active). For each run we
collect ~40 game move histories and compute the canonical-frame axis
distribution (`hexo_rl.training.axis_distribution.compute_axis_fractions`).

PASS criterion (W4 Step 1 done-when):
    Rotation-enabled axes are closer to balanced (~1/3 each) than the
    canonical baseline. The exact target is:
      * max(axis_q, axis_r, axis_s) under rotation < max(...) under canonical
        by at least 0.05, OR
      * max under rotation already at or below the §121 D16 corpus baseline
        of ~0.45 per axis.

Usage:
    .venv/bin/python scripts/smoke_w4_step1_rotation.py
        [--n-games 40] [--checkpoint checkpoints/bootstrap_model.pt]
        [--out reports/w4/step1_rotation_smoke.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import ReplayBuffer
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.pool import WorkerPool
from hexo_rl.training.axis_distribution import compute_axis_fractions


def _build_model(checkpoint: Path | None, device: torch.device) -> HexTacToeNet:
    model = HexTacToeNet(board_size=19, in_channels=18).to(device)
    if checkpoint is not None and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = state.get("model_state_dict") if isinstance(state, dict) else None
        sd = sd or (state.get("state_dict") if isinstance(state, dict) else None)
        sd = sd or state
        # Be defensive: allow partial loads (e.g. backbone only) for the smoke.
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError:
            model.load_state_dict(sd, strict=False)
        print(f"loaded checkpoint: {checkpoint}", flush=True)
    else:
        print("WARNING: no checkpoint provided; using random init", flush=True)
    model.eval()
    return model


def _run_pool(
    model: HexTacToeNet,
    device: torch.device,
    rotation: bool,
    n_games: int,
    timeout_s: float,
) -> list[list[tuple[int, int]]]:
    config = {
        "selfplay": {
            "n_workers": 4,
            "max_game_moves": 200,
            "leaf_batch_size": 8,
            "inference_batch_size": 32,
            "inference_max_wait_ms": 4.0,
            "rotation_enabled": rotation,
            "playout_cap": {
                "fast_prob": 0.0,
                "fast_sims": 64,
                "standard_sims": 50,  # smoke speed; matches D16 (200) order-of-magnitude
                "n_sims_quick": 0,
                "n_sims_full": 0,
                "full_search_prob": 0.0,
                "zoi_enabled": True,
                "zoi_lookback": 16,
                "zoi_margin": 5,
            },
            "random_opening_plies": 1,
        },
        "mcts": {
            "n_simulations": 50,
            "c_puct": 1.5,
            "fpu_reduction": 0.25,
            "dirichlet_alpha": 0.05,
            "epsilon": 0.25,
            "dirichlet_enabled": True,
            "quiescence_enabled": True,
            "quiescence_blend_2": 0.3,
        },
        "training": {"draw_value": 0.0},
    }
    buf = ReplayBuffer(capacity=4096)
    pool = WorkerPool(model, config, device, buf)
    pool.start()
    games: list[list[tuple[int, int]]] = []
    try:
        deadline = time.monotonic() + timeout_s
        seen_ids: set[int] = set()
        while time.monotonic() < deadline and len(games) < n_games:
            recent = pool._runner.drain_game_results()
            for plies, _winner, move_history, worker_id in recent:
                # Each entry is uniquely keyed by (worker_id, time-of-arrival).
                # We don't have a clean game_id; just append everything.
                if plies < 4:
                    continue
                games.append(move_history)
            time.sleep(0.2)
    finally:
        pool.stop()
    return games[:n_games]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=40)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/bootstrap_model.pt"))
    p.add_argument("--timeout-s", type=float, default=900.0)
    p.add_argument("--out", type=Path, default=Path("reports/w4/step1_rotation_smoke.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    model = _build_model(args.checkpoint, device)

    print("\n=== Run 1: canonical (rotation_enabled=False) ===", flush=True)
    canonical_games = _run_pool(
        model, device, rotation=False, n_games=args.n_games, timeout_s=args.timeout_s
    )
    print(f"collected {len(canonical_games)} canonical games", flush=True)
    canonical_axes = compute_axis_fractions(canonical_games)
    print(f"  axis_q={canonical_axes['axis_q']:.4f}  "
          f"axis_r={canonical_axes['axis_r']:.4f}  "
          f"axis_s={canonical_axes['axis_s']:.4f}  "
          f"axis_max={canonical_axes['axis_max']}", flush=True)

    print("\n=== Run 2: rotated (rotation_enabled=True) ===", flush=True)
    rotated_games = _run_pool(
        model, device, rotation=True, n_games=args.n_games, timeout_s=args.timeout_s
    )
    print(f"collected {len(rotated_games)} rotated games", flush=True)
    rotated_axes = compute_axis_fractions(rotated_games)
    print(f"  axis_q={rotated_axes['axis_q']:.4f}  "
          f"axis_r={rotated_axes['axis_r']:.4f}  "
          f"axis_s={rotated_axes['axis_s']:.4f}  "
          f"axis_max={rotated_axes['axis_max']}", flush=True)

    canonical_max = max(canonical_axes["axis_q"], canonical_axes["axis_r"], canonical_axes["axis_s"])
    rotated_max = max(rotated_axes["axis_q"], rotated_axes["axis_r"], rotated_axes["axis_s"])
    delta_max = canonical_max - rotated_max
    range_canonical = canonical_max - min(canonical_axes["axis_q"], canonical_axes["axis_r"], canonical_axes["axis_s"])
    range_rotated   = rotated_max   - min(rotated_axes["axis_q"],   rotated_axes["axis_r"],   rotated_axes["axis_s"])

    pass_threshold = 0.05
    pass_corpus = 0.45
    pass_ = (delta_max >= pass_threshold) or (rotated_max <= pass_corpus)

    verdict = {
        "n_games_canonical": len(canonical_games),
        "n_games_rotated": len(rotated_games),
        "canonical": canonical_axes,
        "rotated": rotated_axes,
        "canonical_max": canonical_max,
        "rotated_max": rotated_max,
        "canonical_range": range_canonical,
        "rotated_range": range_rotated,
        "delta_max": delta_max,
        "pass": pass_,
        "criterion": (
            f"PASS if delta_max ≥ {pass_threshold} OR rotated_max ≤ {pass_corpus}"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=2))

    print("\n=== Verdict ===", flush=True)
    print(f"  canonical_max={canonical_max:.4f}", flush=True)
    print(f"  rotated_max  ={rotated_max:.4f}", flush=True)
    print(f"  delta_max    ={delta_max:.4f}", flush=True)
    print(f"  PASS={pass_}", flush=True)
    print(f"  out={args.out}", flush=True)
    return 0 if pass_ else 1


if __name__ == "__main__":
    sys.exit(main())
