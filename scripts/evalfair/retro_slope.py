"""evalfair/retro_slope.py — WP4 driver: N ckpts x book -> per-ckpt JSON + Theil-Sen slope.

HARD REFUSES --override-n-sims and --solver-backup (deploy-matched only reads).
The full WP4 run is a later dispatch; this builds the driver + a --smoke path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from scripts.evalfair.book import load_book
from scripts.evalfair.core import ArmSpec, run_arm


# ── deploy-matched-only guard ─────────────────────────────────────────────────


def check_deploy_matched_only(
    n_sims_override: Optional[int],
    solver_backup: bool,
) -> None:
    """Raise if any non-deploy-matched flag is set. Called at startup and tested directly."""
    if n_sims_override is not None:
        raise ValueError(
            "retro_slope: --override-n-sims is REFUSED. "
            "Retro slope is deploy-matched-only (per design §4). "
            "Use run_eval.py for WP3 sims-ladder arms."
        )
    if solver_backup:
        raise ValueError(
            "retro_slope: --solver-backup is REFUSED. "
            "Retro slope is deploy-matched-only (per design §4). "
            "Use sims_ladder.py for solver-arm reads."
        )


# ── Theil-Sen slope (per-stage) ───────────────────────────────────────────────


def theil_sen_slope(steps: List[int], wrs: List[float]) -> float:
    """Theil-Sen estimator: median of pairwise slopes."""
    if len(steps) < 2:
        return float("nan")
    slopes = []
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            dx = steps[j] - steps[i]
            if dx != 0:
                slopes.append((wrs[j] - wrs[i]) / dx)
    return float(np.median(slopes)) if slopes else float("nan")


def pair_bootstrap_slope_ci(
    steps: List[int], wrs: List[float], n_boot: int = 2000, seed: int = 42
) -> tuple:
    """Bootstrap CI on Theil-Sen slope over a series of (step, wr) points."""
    if len(steps) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(steps)
    boot_slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s_boot = [steps[i] for i in idx]
        w_boot = [wrs[i] for i in idx]
        boot_slopes.append(theil_sen_slope(s_boot, w_boot))
    valid = [s for s in boot_slopes if not np.isnan(s)]
    if not valid:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="WP4 retro slope: N ckpts x book -> per-ckpt JSON + slope. DEPLOY-MATCHED ONLY."
    )
    ap.add_argument(
        "--ckpts", nargs="+", required=False,
        default=[],
        help="Checkpoint paths (in step order). Use --smoke to run 2 ckpts.",
    )
    ap.add_argument("--book", required=False, help="Path to book_v2 JSON")
    ap.add_argument("--out", default="reports/retro_slope", help="Output directory")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--expect-encoding", default="v6_live2_ls")
    ap.add_argument("--smoke", action="store_true", help="Run on first 2 ckpts, 4 pairs each")
    # These flags are REFUSED — declared so argparse shows them in --help, but immediately
    # rejected via check_deploy_matched_only.
    ap.add_argument("--override-n-sims", type=int, default=None, dest="override_n_sims",
                    help="REFUSED: retro_slope is deploy-matched-only")
    ap.add_argument("--solver-backup", action="store_true", dest="solver_backup",
                    help="REFUSED: retro_slope is deploy-matched-only")
    args = ap.parse_args()

    # Hard refuse non-deploy-matched flags
    check_deploy_matched_only(args.override_n_sims, args.solver_backup)

    if not args.ckpts:
        ap.error("--ckpts required (or --smoke with --ckpts)")
        return

    book_path = args.book
    if book_path is None:
        ap.error("--book required")
        return

    book = load_book(Path(book_path))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpts = args.ckpts
    n_pairs_override = 4 if args.smoke else None
    if args.smoke:
        ckpts = ckpts[:2]

    arm = ArmSpec(label="sims150")  # deploy-matched, no override
    first_knobs = None
    results = []

    for ckpt_path in ckpts:
        ckpt_out = out / Path(ckpt_path).stem
        ckpt_out.mkdir(parents=True, exist_ok=True)
        result = run_arm(
            ckpt_path, arm, book,
            out_dir=str(ckpt_out),
            workers=args.workers,
            n_boot=args.n_boot,
            book_seed=book.get("seed", 20260709),
            first_knobs=first_knobs,
            expect_encoding=args.expect_encoding,
            n_pairs=n_pairs_override,
        )
        if first_knobs is None:
            import torch
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            from scripts.evalfair.core import extract_deploy_knobs
            first_knobs = extract_deploy_knobs(ck.get("config", {}))
        results.append(result)
        print(
            f"[retro] step={result['ckpt_step']} wr={result['wr']:.3f} "
            f"CI=[{result['pair_ci'][0]:.3f},{result['pair_ci'][1]:.3f}]"
        )

    # Per-stage slope
    steps = [r["ckpt_step"] for r in results]
    wrs = [r["wr"] for r in results]
    slope = theil_sen_slope(steps, wrs)
    ci_lo, ci_hi = pair_bootstrap_slope_ci(steps, wrs, args.n_boot)

    summary = {
        "ckpts": ckpts,
        "book_id": book.get("book_id"),
        "steps": steps,
        "wrs": wrs,
        "theil_sen_slope": slope,
        "slope_ci": [ci_lo, ci_hi],
        "n_ckpts": len(results),
        "per_ckpt": results,
    }
    (out / "slope_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[retro] slope={slope:.2e} CI=[{ci_lo:.2e},{ci_hi:.2e}] n_ckpts={len(results)}")


if __name__ == "__main__":
    main()
