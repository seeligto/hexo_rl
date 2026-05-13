#!/usr/bin/env python3
"""Phase B Gate 4 — v8 NN inference latency + parameter count.

Lightweight bench harness limited to model-side metrics that don't go
through the engine's v6-only Rust MCTS. Captures:

  - Total parameter count (millions).
  - NN forward latency at b=1 and b=64 — n=5 runs (median + IQR).
  - Trunk-only FLOPs (rough estimate via thop if available).

Skips MCTS sim/s + worker pos/hr — those require v8-aware self-play
(Phase D §168). Skips bench-gate verdict — Phase B is a variant exploration
sprint, not a regression gate; Gate 5 proposes the v8 bench-gate
recalibration based on canonical-pick measurements.

Output: JSON record per arm, written to
`reports/encoding_phase_b/<arm>_bench_<host>.json`.

Usage:
    python scripts/bench_v8_nn.py \\
        --checkpoint checkpoints/v8_variants/B1_v8full.pt \\
        --host laptop --runs 5 --warmup 20
"""
from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.bootstrap.dataset_v8 import (
    BOARD_SIZE_V8,
    HALF_V8,
    LEGAL_MOVE_RADIUS_V8,
    N_PLANES_V8,
)
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.utils.device import best_device


def _make_v8_input(
    batch: int, device: torch.device, canvas_realness: bool = False,
) -> torch.Tensor:
    """Synthesise a (batch, 11, 25, 25) v8 input with valid plane-8 mask.

    Plane 8 polarity follows ``canvas_realness``: False (default) → 1.0
    OUTSIDE dilated hex (off_window); True → 1.0 INSIDE (§169 A4).
    """
    x = torch.zeros(batch, N_PLANES_V8, BOARD_SIZE_V8, BOARD_SIZE_V8, device=device)
    for q in range(BOARD_SIZE_V8):
        for r in range(BOARD_SIZE_V8):
            lq = q - HALF_V8
            lr = r - HALF_V8
            ls = -(lq + lr)
            inside = max(abs(lq), abs(lr), abs(ls)) <= LEGAL_MOVE_RADIUS_V8
            if canvas_realness:
                x[:, 8, q, r] = 1.0 if inside else 0.0
            else:
                x[:, 8, q, r] = 0.0 if inside else 1.0
    # A few stones inside the hex, broadcast scalars.
    x[:, 0, HALF_V8, HALF_V8] = 1.0
    x[:, 4, HALF_V8 + 1, HALF_V8] = 1.0
    x[:, 9, :, :] = 0.5
    x[:, 10, :, :] = 1.0
    return x


@torch.no_grad()
def _measure_latency(
    model: torch.nn.Module,
    batch: int,
    device: torch.device,
    runs: int,
    warmup: int,
) -> dict:
    x = _make_v8_input(
        batch, device, canvas_realness=getattr(model, "canvas_realness", False),
    )
    # Warmup.
    for _ in range(warmup):
        _ = model(x.float())
    if device.type == "cuda":
        torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x.float())
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return {
        "batch": batch,
        "runs": runs,
        "median_ms": statistics.median(times_ms),
        "p25_ms": times_ms[max(0, len(times_ms) // 4)],
        "p75_ms": times_ms[min(len(times_ms) - 1, 3 * len(times_ms) // 4)],
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "all_ms": [round(t, 3) for t in times_ms],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default=None,
                        help="Host label for output filename (default: hostname)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", default="1,64",
                        help="Comma-separated batch sizes (default 1,64)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        print(f"FATAL: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    arm = ckpt.stem.replace("_v8full", "")
    host = args.host or socket.gethostname().split(".")[0]
    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "reports" / "encoding_phase_b" / f"{arm}_bench_{host}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = best_device()
    model, _spec, label = load_model_with_encoding(str(ckpt), device)
    if label != "v8":
        print(
            f"FATAL: bench_v8_nn requires a v8 checkpoint; got label={label!r}",
            file=sys.stderr,
        )
        return 1
    n_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batches = [int(s.strip()) for s in args.batches.split(",") if s.strip()]
    print(f"[bench-v8] arm={arm}  host={host}  device={device}  "
          f"params={n_params/1e6:.2f}M  batches={batches}", flush=True)

    latency = []
    for b in batches:
        rec = _measure_latency(model, b, device, args.runs, args.warmup)
        latency.append(rec)
        print(f"[bench-v8]  b={b}  median={rec['median_ms']:.2f}ms  "
              f"IQR=[{rec['p25_ms']:.2f}, {rec['p75_ms']:.2f}]  "
              f"min={rec['min_ms']:.2f} max={rec['max_ms']:.2f}", flush=True)

    out = {
        "arm": arm,
        "host": host,
        "checkpoint": str(ckpt),
        "encoding": "v8",
        "canvas_realness": bool(getattr(model, "canvas_realness", False)),
        "filters": int(model.filters),
        "res_blocks": int(model.res_blocks),
        "gpool_indices": list(model.trunk.gpool_indices),
        "params_total": int(n_params),
        "params_trainable": int(n_train_params),
        "latency": latency,
        "device": str(device),
        "torch_version": torch.__version__,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "skipped": [
            "mcts_sims_per_sec — engine MCTS is v6-only; v8 awareness is Phase D §168",
            "worker_pos_per_hr — same reason as MCTS",
        ],
    }
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[bench-v8] DONE — wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
