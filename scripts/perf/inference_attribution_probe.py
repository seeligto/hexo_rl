#!/usr/bin/env python3
"""Inference round-trip attribution probe (investigation 2026-05-31).

Drives the REAL WorkerPool self-play path with `diagnostics.perf_timing=true`
+ `perf_sync_cuda=true`, captures `inference_batch_timing` structlog events to a
JSONL, then prints the 5-bucket per-batch attribution + the pre-registered
GPU-bound vs dispatch/GIL-bound verdict.

Buckets (µs, p50 over batches): fetch_wait | h2d | forward | d2h_scatter | submit.
perf_sync_cuda makes forward_us TRUE GPU time (sync after H2D + forward) instead of
async kernel-launch dispatch — mandatory or the attribution mis-reads as dispatch-bound.

Usage:
  .venv/bin/python scripts/perf/inference_attribution_probe.py \
      --checkpoint checkpoints/bootstrap_model_v6_live2.pt \
      --variant v6_live2_smoke_laptop --seconds 45 \
      --out logs/perf/inference_attribution_2026-05-31.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics as S
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_structlog_to_file(path: Path):
    """Route every structlog event to `path` as one JSON line."""
    import structlog

    fh = open(path, "w")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(file=fh),
        cache_logger_on_first_use=True,
    )
    return fh


def pct(xs, p):
    xs = sorted(xs)
    return xs[int(p * (len(xs) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/bootstrap_model_v6_live2.pt")
    ap.add_argument("--variant", default="v6_live2_smoke_laptop")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--warmup-batches", type=int, default=40,
                    help="drop the first N batches (trace warmup / cudnn.benchmark autotune)")
    ap.add_argument("--out", default="logs/perf/inference_attribution.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = _configure_structlog_to_file(out_path)

    from hexo_rl.utils.config import load_config
    from hexo_rl.utils.device import best_device
    from hexo_rl.model.network import HexTacToeNet
    from hexo_rl.selfplay.pool import WorkerPool
    from hexo_rl.encoding import resolve_encoding_for_eval
    from engine import ReplayBuffer

    base = ["configs/model.yaml", "configs/training.yaml", "configs/selfplay.yaml"]
    cfg = load_config(*base, f"configs/variants/{args.variant}.yaml")

    spec = resolve_encoding_for_eval(args.checkpoint, None)
    cfg["encoding"] = spec.name
    for stale in ("board_size", "in_channels", "n_planes", "cluster_window_size",
                  "cluster_threshold", "legal_move_radius"):
        cfg.pop(stale, None)

    # Inject the diagnostic switches (Step 1 instrument).
    cfg["diagnostics"] = {"perf_timing": True, "perf_sync_cuda": True, "vram_probe_interval": 0}

    device = best_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    mcfg = cfg.get("model", {})
    model = HexTacToeNet(
        board_size=spec.trunk_size,
        in_channels=spec.n_planes,
        filters=int(mcfg.get("filters", 128)),
        res_blocks=int(mcfg.get("res_blocks", 12)),
        encoding=spec.name,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    if state is None:
        state = ckpt
    try:
        model.load_state_dict(state, strict=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] partial load: {exc}")
    model.eval()

    sp = cfg.get("selfplay", {})
    print(f"[probe] device={device} encoding={spec.name} n_workers={sp.get('n_workers')} "
          f"batch={sp.get('inference_batch_size')} max_wait={sp.get('inference_max_wait_ms')}ms "
          f"sync_cuda=True seconds={args.seconds}")

    buffer = ReplayBuffer(capacity=200_000, encoding=spec.name)
    pool = WorkerPool(model, cfg, device, buffer, n_workers=None)
    pool.start()
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < args.seconds:
            time.sleep(2.0)
            print(f"  t={time.perf_counter()-t0:4.0f}s games={pool.games_completed}")
    finally:
        pool.stop()
    fh.flush()
    fh.close()

    # ---- parse + attribute ----
    rows = []
    for line in out_path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") == "inference_batch_timing":
            rows.append(ev)

    fields = ["fetch_wait_us", "h2d_us", "forward_us", "d2h_scatter_us", "submit_us"]
    rows = [r for r in rows if all(f in r for f in fields)]
    rows = rows[args.warmup_batches:]
    if not rows:
        print("[probe] NO inference_batch_timing events captured — abort")
        return 1

    by = {f: [r[f] for r in rows] for f in fields}
    batch_n = [r["batch_n"] for r in rows]
    # per-cycle total per batch (sum of the 5 buckets)
    cyc = [sum(r[f] for f in fields) for r in rows]
    rt = pct(cyc, 0.5)

    print(f"\n=== inference attribution  n_batches={len(rows)} (after {args.warmup_batches} warmup) ===")
    print(f"  mean batch_n={S.mean(batch_n):.1f}  (configured batch={sp.get('inference_batch_size')})")
    print(f"  {'bucket':<16}{'p50_us':>10}{'p95_us':>10}{'mean_us':>10}{'p50_share':>11}")
    for f in fields:
        p50 = pct(by[f], 0.5)
        print(f"  {f:<16}{p50:>10.1f}{pct(by[f],0.95):>10.1f}{S.mean(by[f]):>10.1f}{100*p50/rt:>10.1f}%")
    print(f"  {'CYCLE(Σ5)':<16}{rt:>10.1f}{'':>10}{S.mean(cyc):>10.1f}{100.0:>10.1f}%")

    fwd = pct(by["forward_us"], 0.5)
    ffi = pct(by["fetch_wait_us"], 0.5) + pct(by["submit_us"], 0.5)
    h2d = pct(by["h2d_us"], 0.5)
    fwd_share = fwd / rt
    ffi_share = ffi / rt
    stage_share = (ffi + h2d) / rt
    print(f"\n  forward/RT      = {100*fwd_share:5.1f}%   (GPU)")
    print(f"  FFI(fetch+sub)  = {100*ffi_share:5.1f}%   (dispatch/PyO3 — Rust-rewrite target)")
    print(f"  (FFI+h2d)/RT    = {100*stage_share:5.1f}%")

    e1 = fwd_share >= 0.80 and ffi_share < 0.10
    e2 = ffi_share >= 0.10 or (stage_share >= 0.30 and fwd_share < 0.70)
    if e1 and not e2:
        verdict = "E1 GPU-BOUND — Rust inference REJECTED on evidence"
    elif e2 and not e1:
        verdict = "E2 DISPATCH/GIL-BOUND — fix Python server first"
    else:
        verdict = "INCONCLUSIVE"
    print(f"\n  VERDICT: {verdict}")
    print(f"  (E1 gate: forward>=80% AND FFI<10% | E2 gate: FFI>=10% OR (FFI+h2d>=30% & forward<70%))")
    print(f"\n[probe] raw events: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
