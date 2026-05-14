#!/usr/bin/env python3
"""NN latency bench — encoding-aware (§169 A2 + §176 P66 unify).

Subsumes the former ``scripts/bench_v8_nn.py`` — when the loaded
checkpoint resolves to encoding "v8" (or "v8_canvas_realness"), this
script synthesises a realistic 11-plane v8 input with the proper plane-8
off-window mask (matching the deleted bench_v8_nn._make_v8_input) so the
PartialConv2d-aware branches see the same activation distribution as the
old script. Other encodings (v6 / v6w25 / v7-family) use torch.randn —
shape-only latency is invariant to input content for those models.

Reports median + IQR ms latency for the requested batch sizes. Outputs:

  - Always to stdout (operator-facing).
  - ``--append-to <file>``  → markdown row (v6w25 style; default).
  - ``--out <path>``        → JSON record (v8 style — subsumed feature).

Usage:
    python scripts/bench_v6w25_nn.py \\
        --checkpoint checkpoints/ablation_169/A2_pma.pt \\
        --batches 1,64 --runs 5

    python scripts/bench_v6w25_nn.py \\
        --checkpoint checkpoints/v8_variants/B1_v8full.pt \\
        --out reports/encoding_phase_b/B1_bench_$(hostname).json
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
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.utils.device import best_device


def _make_v8_input(
    batch: int, device: torch.device, canvas_realness: bool = False,
) -> torch.Tensor:
    """Synthesise (batch, 11, 25, 25) v8 input with valid plane-8 mask.

    Plane 8 polarity follows ``canvas_realness``: False (default) → 1.0
    OUTSIDE dilated hex (off_window); True → 1.0 INSIDE (§169 A4).

    Migrated verbatim from the former scripts/bench_v8_nn.py so v8 latency
    measurements stay bit-comparable across the §176 P66 unification.
    """
    from hexo_rl.bootstrap.dataset_v8 import (
        BOARD_SIZE_V8, HALF_V8, LEGAL_MOVE_RADIUS_V8, N_PLANES_V8,
    )

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
    x[:, 0, HALF_V8, HALF_V8] = 1.0
    x[:, 4, HALF_V8 + 1, HALF_V8] = 1.0
    x[:, 9, :, :] = 0.5
    x[:, 10, :, :] = 1.0
    return x


def _bench_one(
    model, board_size: int, in_channels: int, batch: int,
    device: torch.device, encoding_label: str, n_runs: int = 5, n_warmup: int = 20,
    global_crop: torch.Tensor | None = None,
) -> dict:
    # v8 path uses the realistic plane-8-mask input from the old script;
    # all other encodings use torch.randn (shape-only latency).
    if encoding_label.startswith("v8"):
        x = _make_v8_input(
            batch, device,
            canvas_realness=getattr(model, "canvas_realness", False),
        ).float()
    else:
        x = torch.randn(batch, in_channels, board_size, board_size, device=device)
    fwd_kwargs: dict = {}
    if global_crop is not None:
        # Caller provides a single (3, 32, 32) template; broadcast to batch.
        fwd_kwargs["global_crop"] = global_crop.expand(batch, -1, -1, -1).contiguous()
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x, **fwd_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times_ms: List[float] = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x, **fwd_kwargs)
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
    parser.add_argument("--out", type=Path, default=None,
                        help="Write a JSON record to this path "
                             "(subsumes the deleted bench_v8_nn.py output).")
    parser.add_argument("--host", default=None,
                        help="Host label override (default: short hostname).")
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
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pool_type = getattr(model, "pool_type", "min_max")
    gpool_bias_active = bool(getattr(model, "gpool_bias_active", False))
    pool_label = (
        f"{pool_type}+gpool_bias" if gpool_bias_active else pool_type
    )
    arm = args.arm_label or ckpt.stem
    host = args.host or socket.gethostname().split(".")[0]
    print(
        f"[bench-nn] arm={arm} host={host} encoding={label} "
        f"pool_type={pool_label} board={board_size} ch={in_channels} "
        f"params={n_params/1e6:.2f}M"
    )

    batches = [int(s.strip()) for s in args.batches.split(",") if s.strip()]
    # §169 A3 — pma_global needs a (1, 3, 32, 32) global summary template.
    # §170 P3 — gpool_bias_active also needs the same template.
    global_crop_template: torch.Tensor | None = None
    needs_global_crop = pool_type == "pma_global" or gpool_bias_active
    if needs_global_crop:
        from hexo_rl.utils.global_crop import (
            CANVAS_SIZE as _C,
            N_GLOBAL_PLANES as _N,
        )
        gc = torch.zeros(1, _N, _C, _C, device=device, dtype=torch.float32)
        gc[0, 2, 8:24, 8:24] = 1.0  # active canvas mask
        gc[0, 0, 12:16, 12:16] = 1.0  # cur stones (4×4)
        gc[0, 1, 16:20, 16:20] = 1.0  # opp stones (4×4)
        global_crop_template = gc
    rows: List[dict] = []
    for b in batches:
        rec = _bench_one(
            model, board_size, in_channels, b, device,
            encoding_label=label,
            n_runs=args.runs, n_warmup=args.warmup,
            global_crop=global_crop_template,
        )
        rows.append(rec)
        print(
            f"[bench-nn]  b={b:3d}  median={rec['median_ms']:6.2f}ms  "
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
            f"| {arm} | {host} | {label} | {pool_label} | "
            f"{n_params/1e6:.2f} | "
            f"{b1['median_ms']:.2f} | [{b1['iqr_low_ms']:.2f}, {b1['iqr_high_ms']:.2f}] | "
            f"{b64['median_ms']:.2f} | [{b64['iqr_low_ms']:.2f}, {b64['iqr_high_ms']:.2f}] |\n"
            if b1 and b64 else "| (incomplete benches) |\n"
        )
        with args.append_to.open("a") as f:
            f.write(line)
        print(f"appended row to {args.append_to}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_record = {
            "arm": arm,
            "host": host,
            "checkpoint": str(ckpt),
            "encoding": label,
            "canvas_realness": bool(getattr(model, "canvas_realness", False)),
            "filters": int(getattr(model, "filters", 0)),
            "res_blocks": int(getattr(model, "res_blocks", 0)),
            "gpool_indices": list(getattr(getattr(model, "trunk", None),
                                          "gpool_indices", []) or []),
            "params_total": int(n_params),
            "params_trainable": int(n_train_params),
            "pool_type": pool_label,
            "board_size": int(board_size),
            "in_channels": int(in_channels),
            "latency": rows,
            "device": str(device),
            "torch_version": torch.__version__,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        with args.out.open("w") as f:
            json.dump(out_record, f, indent=2)
        print(f"[bench-nn] DONE — wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
