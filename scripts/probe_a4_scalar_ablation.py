#!/usr/bin/env python3
"""§170 P0 — A4 scalar-ablation probe.

Discriminates "A4 spatial-dominated-by-scalars" vs "A4 spatial-rich-but-
misdirected." §169a established spatial pathway alive (KL_S=1.53) but a
minority shareholder (KL_S/KL_R=0.27). This probe zeros stone planes 0-7
on Set R and measures per-position policy delta.

If zeroing the stone planes barely changes predictions → scalars dominate
decisively → distribution-shift fine-tune is hopeless (no spatial features
to redirect). If zeroing causes large policy shifts → spatial features are
rich but misdirected → §171 distribution-shift fine-tune is worth trying.

Pre-registered thresholds (locked before run):
  scalar-dominated:   mean(KL_zeroed_vs_original) < 0.30 nats
  spatial-rich:       mean(KL_zeroed_vs_original) > 1.50 nats
  ambig (0.30–1.50):  report + surface, do not commit a verdict.

Discriminates §170 fork:
  scalar-dominated → bbox direction dead, fine-tune hopeless.
  spatial-rich      → distribution-shift fine-tune in §171 worth trying.

Set R: same fixture as §169a (200 positions from data/bootstrap_corpus_v8.npz,
plane 8 inverted off→canvas, numpy seed=20260508).

Usage:
  .venv/bin/python scripts/probe_a4_scalar_ablation.py \\
      --ckpt checkpoints/ablation_169/A4_canvas_realness.pt \\
      --out reports/investigations/a4_scalar_ablation_20260508/probe.json \\
      --n 200 --device cuda

Exit codes:
  0  SCALAR_DOMINATED
  1  SPATIAL_RICH
  2  AMBIGUOUS
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.eval.checkpoint_loader import load_model_with_encoding


def build_set_R(corpus_path: Path, n: int, seed: int) -> np.ndarray:
    """Set R — n real positions from off_window corpus, plane 8 inverted to canvas."""
    rng = np.random.default_rng(seed)
    z = np.load(corpus_path, mmap_mode="r")
    states = z["states"]  # (T, 11, 25, 25)
    idx = rng.choice(states.shape[0], size=n, replace=False)
    sub = np.array(states[idx], dtype=np.float16)
    # Off→canvas polarity flip: plane 8 was 1 outside, becomes 1 inside.
    sub[:, 8, :, :] = (1.0 - sub[:, 8, :, :]).astype(np.float16)
    return sub


@torch.no_grad()
def policy_log_probs(
    model: torch.nn.Module,
    arr: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    n = arr.shape[0]
    out = np.empty((n, 625), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        x = torch.from_numpy(arr[s:e].astype(np.float32)).to(device)
        log_p, _, _ = model(x)
        out[s:e] = log_p.float().cpu().numpy()
    return out


def per_position_symmetric_kl(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """Per-position symmetric KL: (KL(P||Q) + KL(Q||P)) / 2."""
    p = np.exp(log_p)
    q = np.exp(log_q)
    kl_pq = np.sum(p * (log_p - log_q), axis=1)
    kl_qp = np.sum(q * (log_q - log_p), axis=1)
    return 0.5 * (kl_pq + kl_qp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", default="data/bootstrap_corpus_v8.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260508)
    ap.add_argument("--kl_low", type=float, default=0.30,
                    help="scalar-dominated: mean KL < this")
    ap.add_argument("--kl_high", type=float, default=1.50,
                    help="spatial-rich: mean KL > this")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device.startswith("cuda") else "cpu")
    print(f"[probe_a4_scalar_ablation] device={device}", flush=True)

    print("[probe] loading model ...", flush=True)
    t0 = time.time()
    model, spec, label = load_model_with_encoding(args.ckpt, device)
    print(f"  label={label} board_size={spec.board_size} "
          f"in_channels={spec.n_planes} time={time.time()-t0:.1f}s", flush=True)
    if label != "v8":
        raise RuntimeError(f"probe expects v8 checkpoint; got {label}")

    print(f"[probe] building Set R (n={args.n}, seed={args.seed}) ...", flush=True)
    t0 = time.time()
    set_R = build_set_R(Path(args.corpus), args.n, args.seed)
    print(f"  done time={time.time()-t0:.1f}s shape={set_R.shape}", flush=True)

    # Zeroed variant: copy, then wipe stone-history planes 0-7.
    set_R_zeroed = set_R.copy()
    set_R_zeroed[:, 0:8, :, :] = 0.0

    # Fixture audit
    plane8_inside = float(set_R[:, 8, :, :].sum(axis=(1, 2)).mean())
    plane9_var = float(set_R[:, 9, :, :].var())
    plane10_var = float(set_R[:, 10, :, :].var())
    zeroed_sum = float(set_R_zeroed[:, 0:8, :, :].sum())
    print(f"[probe] fixture: plane8_inside_mean={plane8_inside:.1f} "
          f"plane9_var={plane9_var:.4f} plane10_var={plane10_var:.4f} "
          f"zeroed_stones={zeroed_sum:.0f}", flush=True)
    if zeroed_sum != 0.0:
        raise RuntimeError("zeroed stone planes non-zero — ablation bug")
    if plane9_var < 1e-6 or plane10_var < 1e-6:
        print("  WARN: scalar planes appear constant — Set R may not be varying "
              "correctly; check corpus path", flush=True)

    print("[probe] running forward on Set R original ...", flush=True)
    log_orig = policy_log_probs(model, set_R, device)
    print("[probe] running forward on Set R zeroed ...", flush=True)
    log_zero = policy_log_probs(model, set_R_zeroed, device)

    kl_per_pos = per_position_symmetric_kl(log_orig, log_zero)

    kl_mean   = float(np.mean(kl_per_pos))
    kl_median = float(np.median(kl_per_pos))
    kl_p90    = float(np.percentile(kl_per_pos, 90))
    kl_min    = float(np.min(kl_per_pos))
    kl_max    = float(np.max(kl_per_pos))

    argmax_orig  = log_orig.argmax(axis=1)
    argmax_zero  = log_zero.argmax(axis=1)
    argmax_stable     = int(np.sum(argmax_orig == argmax_zero))
    argmax_stable_frac = argmax_stable / args.n

    print(f"[probe] KL zeroed-vs-original: mean={kl_mean:.4f} median={kl_median:.4f} "
          f"p90={kl_p90:.4f} min={kl_min:.4f} max={kl_max:.4f}", flush=True)
    print(f"[probe] argmax stable: {argmax_stable}/{args.n} "
          f"(frac={argmax_stable_frac:.3f})", flush=True)

    if kl_mean < args.kl_low:
        verdict = "SCALAR_DOMINATED"
        exit_code = 0
    elif kl_mean > args.kl_high:
        verdict = "SPATIAL_RICH"
        exit_code = 1
    else:
        verdict = "AMBIGUOUS"
        exit_code = 2

    print(f"[probe] verdict: {verdict}", flush=True)

    result = {
        "ckpt": str(args.ckpt),
        "verdict": verdict,
        "exit_code": exit_code,
        "thresholds": {
            "kl_low": args.kl_low,
            "kl_high": args.kl_high,
        },
        "kl_zeroed_vs_original": {
            "mean": kl_mean,
            "median": kl_median,
            "p90": kl_p90,
            "min": kl_min,
            "max": kl_max,
            "n": args.n,
        },
        "argmax_stable": argmax_stable,
        "argmax_stable_frac": argmax_stable_frac,
        "fixture_audit": {
            "plane8_inside_cells_mean": plane8_inside,
            "plane9_var": plane9_var,
            "plane10_var": plane10_var,
            "zeroed_stone_sum": zeroed_sum,
        },
        "n": args.n,
        "seed": args.seed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[probe] wrote {out_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
