#!/usr/bin/env python3
"""§169 A2 — NN latency bench for v6w25 (encoding-aware).

Sister to ``scripts/bench_v8_nn.py`` but goes through
``load_model_with_encoding`` so it accepts v6 / v6w25 / v8 checkpoints
uniformly. Reports median + IQR ms latency for the requested batch sizes
and writes a markdown table appendable to bench_per_arm.md.

Usage:
    python scripts/bench_v6w25_nn.py \\
        --checkpoint checkpoints/ablation_169/A2_pma.pt \\
        --batches 1,64 \\
        --runs 5
"""
from __future__ import annotations

import argparse
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.utils.device import best_device


def _bench_one(
    model, board_size: int, in_channels: int, batch: int,
    device: torch.device, n_runs: int = 5, n_warmup: int = 20,
) -> dict:
    x = torch.randn(batch, in_channels, board_size, board_size, device=device)
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times_ms: List[float] = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "batch": batch,
        "median_ms": statistics.median(times_ms),
        "iqr_low_ms": statistics.quantiles(times_ms, n=4)[0] if len(times_ms) > 1 else times_ms[0],
        "iqr_high_ms": statistics.quantiles(times_ms, n=4)[2] if len(times_ms) > 1 else times_ms[0],
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "n_runs": n_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", default="1,64",
                        help="Comma-separated batch sizes (default 1,64).")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--arm-label", default=None,
                        help="Arm label for the report row (e.g. 'A2'). "
                             "Defaults to the checkpoint stem.")
    parser.add_argument("--append-to", type=Path, default=None,
                        help="Append a markdown row to this file.")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        print(f"FATAL: checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    device = best_device()
    model, spec, label = load_model_with_encoding(ckpt, device)
    in_channels = model.in_channels
    board_size = spec.board_size

    n_params = sum(p.numel() for p in model.parameters())
    pool_type = getattr(model, "pool_type", "min_max")
    arm = args.arm_label or ckpt.stem
    host = socket.gethostname().split(".")[0]
    print(
        f"[bench-v6w25] arm={arm} host={host} encoding={label} "
        f"pool_type={pool_type} board={board_size} ch={in_channels} "
        f"params={n_params/1e6:.2f}M"
    )

    batches = [int(s.strip()) for s in args.batches.split(",") if s.strip()]
    rows: List[dict] = []
    for b in batches:
        rec = _bench_one(model, board_size, in_channels, b, device,
                         n_runs=args.runs, n_warmup=args.warmup)
        rows.append(rec)
        print(
            f"[bench-v6w25]  b={b:3d}  median={rec['median_ms']:6.2f}ms  "
            f"IQR=[{rec['iqr_low_ms']:.2f}, {rec['iqr_high_ms']:.2f}]"
        )

    if args.append_to is not None:
        args.append_to.parent.mkdir(parents=True, exist_ok=True)
        if not args.append_to.exists():
            args.append_to.write_text(
                "| arm | host | encoding | pool | params (M) | "
                "b=1 median ms | b=1 IQR | b=64 median ms | b=64 IQR |\n"
                "|---|---|---|---|---:|---:|---:|---:|---:|\n"
            )
        b1 = next((r for r in rows if r["batch"] == 1), None)
        b64 = next((r for r in rows if r["batch"] == 64), None)
        line = (
            f"| {arm} | {host} | {label} | {pool_type} | "
            f"{n_params/1e6:.2f} | "
            f"{b1['median_ms']:.2f} | [{b1['iqr_low_ms']:.2f}, {b1['iqr_high_ms']:.2f}] | "
            f"{b64['median_ms']:.2f} | [{b64['iqr_low_ms']:.2f}, {b64['iqr_high_ms']:.2f}] |\n"
            if b1 and b64 else "| (incomplete benches) |\n"
        )
        with args.append_to.open("a") as f:
            f.write(line)
        print(f"appended row to {args.append_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
