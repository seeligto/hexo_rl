"""evalfair/retro_slope.py — WP4 driver: N ckpts x per-stage books -> per-ckpt JSON + Theil-Sen slope.

HARD REFUSES --override-n-sims and --solver-backup (deploy-matched only reads).

Finding #7 fix: accepts --book-r4 and --book-r5; auto-selects the correct book per ckpt
by resolved training radius; aborts if a ckpt's radius has no matching book.

Finding #6 fix: pair_bootstrap_slope_ci is now PAIR-LEVEL — within each ckpt we resample
its per_pair_scores to recompute that ckpt's WR, then recompute Theil-Sen over the series.
Slope is computed PER STAGE (Series A = r4 ckpts; Series B = r5 ckpts) — never spliced
across the 200k boundary (frozen verdicts 1 + 3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from scripts.evalfair.book import load_book
from scripts.evalfair.core import ArmSpec, extract_deploy_knobs, radius_from_checkpoint, run_arm


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


# ── Theil-Sen slope ───────────────────────────────────────────────────────────


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


# ── Finding #6: pair-level bootstrap CI on slope ─────────────────────────────


def pair_bootstrap_slope_ci(
    steps: List[int],
    per_ckpt_pair_scores: List[List[float]],
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Pair-level bootstrap CI on Theil-Sen slope.

    Design §4.5: within each checkpoint resample its per_pair_scores (with replacement),
    recompute that ckpt's WR, recompute Theil-Sen over the series; 2000 reps; percentile CI.

    This is correct pair-level resampling — NOT point-level resampling of (step, WR) tuples.
    The old point-level bootstrap is replaced by this to honour the design mandate.

    Args:
        steps: checkpoint step indices, in order.
        per_ckpt_pair_scores: list (one per ckpt) of pair-score lists (each float in [0,1]).
        n_boot: bootstrap repetitions.
        seed: RNG seed.

    Returns:
        (ci_lo, ci_hi) — 2.5th and 97.5th percentiles of the bootstrap slope distribution.
    """
    if len(steps) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_ckpts = len(steps)
    # Convert each ckpt's pair scores to a numpy array for fast indexing
    scores_arrs = [np.asarray(ps, dtype=float) for ps in per_ckpt_pair_scores]
    n_pairs_per = [len(a) for a in scores_arrs]

    boot_slopes = []
    for _ in range(n_boot):
        resampled_wrs = []
        for k in range(n_ckpts):
            arr = scores_arrs[k]
            n = n_pairs_per[k]
            idx = rng.integers(0, n, size=n)
            resampled_wrs.append(float(arr[idx].mean()))
        boot_slopes.append(theil_sen_slope(steps, resampled_wrs))

    valid = [s for s in boot_slopes if not np.isnan(s)]
    if not valid:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


# ── Finding #7: per-stage book resolution ────────────────────────────────────


def resolve_book_for_radius(
    radius: int,
    books_by_radius: Dict[int, dict],
    ckpt_path: str,
) -> dict:
    """Return the book whose radius_stage matches the ckpt's resolved radius.

    Aborts (ValueError) if no matching book is registered. This enforces the F4 per-stage
    book guard: every ckpt must read on a book sampled at its own training radius.

    Args:
        radius: the ckpt's resolved training radius.
        books_by_radius: mapping radius -> loaded book dict.
        ckpt_path: used in error messages only.
    """
    book = books_by_radius.get(radius)
    if book is None:
        available = sorted(books_by_radius.keys())
        raise ValueError(
            f"No book registered for radius={radius} (ckpt={ckpt_path}). "
            f"Available radii: {available}. "
            f"Pass the matching --book-r{radius} path."
        )
    # Redundant check (run_arm will also enforce), but fail early with a clear message.
    book_stage = book.get("radius_stage")
    if book_stage is not None and int(book_stage) != int(radius):
        raise ValueError(
            f"Book radius_stage={book_stage} does not match ckpt radius={radius} "
            f"(ckpt={ckpt_path}). Internal error in books_by_radius mapping."
        )
    return book


# ── per-stage slope computation ───────────────────────────────────────────────


def compute_stage_slope(
    results: List[dict],
    stage_radius: int,
    n_boot: int = 2000,
    seed: int = 42,
) -> Optional[dict]:
    """Compute Theil-Sen slope + pair-bootstrap CI for one radius stage.

    Returns None if fewer than 2 results in the stage. Returns a "underpowered" flag
    if fewer than 4 points (power too low for a reliable verdict per design §8 MDE note).

    The 200k boundary is treated as a discontinuity — results are only compared within
    a stage, never across it (frozen verdicts 1 + 3).
    """
    stage_results = [r for r in results if r.get("radius") == stage_radius]
    if len(stage_results) < 2:
        return None
    stage_results = sorted(stage_results, key=lambda r: r["ckpt_step"])
    steps = [r["ckpt_step"] for r in stage_results]
    wrs = [r["wr"] for r in stage_results]
    per_ckpt_pairs = [r["per_pair_scores"] for r in stage_results]

    slope = theil_sen_slope(steps, wrs)
    ci_lo, ci_hi = pair_bootstrap_slope_ci(steps, per_ckpt_pairs, n_boot=n_boot, seed=seed)

    n_pts = len(stage_results)
    # MDE from design §8: ~0.13 total ΔWR at 8 pts / 64 pairs; label as underpowered if < 4.
    # Report MDE in units of WR/step (total ΔWR / total span).
    total_span = steps[-1] - steps[0] if len(steps) >= 2 else 0
    mde_wr = 0.13  # total ΔWR over the series (not per-step)
    mde_per_step = mde_wr / total_span if total_span > 0 else float("nan")

    return {
        "radius_stage": stage_radius,
        "n_ckpts": n_pts,
        "steps": steps,
        "wrs": wrs,
        "theil_sen_slope": slope,
        "slope_ci": [ci_lo, ci_hi],
        "ci_excludes_zero": (ci_lo > 0) or (ci_hi < 0),
        "slope_positive_ci_excludes_zero": (slope > 0) and (ci_lo > 0),
        "underpowered": n_pts < 4,
        "mde_total_delta_wr": mde_wr,
        "mde_per_step": mde_per_step,
        "total_span_steps": total_span,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "WP4 retro slope: N ckpts x per-stage books -> per-ckpt JSON + slope. "
            "DEPLOY-MATCHED ONLY. "
            "Series A (r4 ckpts) uses --book-r4; Series B (r5 ckpts) uses --book-r5. "
            "Slope computed PER STAGE — never spliced across the 200k boundary."
        )
    )
    ap.add_argument(
        "--ckpts", nargs="+", required=False,
        default=[],
        help="Checkpoint paths (in step order).",
    )
    # Finding #7: per-stage books instead of one --book
    ap.add_argument("--book-r4", default=None, dest="book_r4",
                    help="Path to evalfair_r4_v2.json (for radius-4 ckpts / Series A)")
    ap.add_argument("--book-r5", default=None, dest="book_r5",
                    help="Path to evalfair_r5_v2.json (for radius-5 ckpts / Series B)")
    # Legacy --book kept for backward compat (used if per-stage books not supplied)
    ap.add_argument("--book", default=None,
                    help="Legacy: single book for all ckpts (ONLY valid if all ckpts are the same stage)")
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
        ap.error("--ckpts required")
        return

    # Build books_by_radius mapping (Finding #7)
    books_by_radius: Dict[int, dict] = {}
    if args.book_r4:
        b = load_book(Path(args.book_r4))
        books_by_radius[int(b["radius_stage"])] = b
    if args.book_r5:
        b = load_book(Path(args.book_r5))
        books_by_radius[int(b["radius_stage"])] = b
    if args.book and not books_by_radius:
        # Legacy path: single book provided; use only if all ckpts are same radius
        b = load_book(Path(args.book))
        books_by_radius[int(b["radius_stage"])] = b
    if not books_by_radius:
        ap.error("At least one book must be supplied via --book-r4, --book-r5, or --book")
        return

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
        # Resume-safe: skip if result.json already exists
        ckpt_out = out / Path(ckpt_path).stem
        result_path = ckpt_out / "result.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text())
            print(
                f"[retro] SKIP (cached) step={existing['ckpt_step']} wr={existing['wr']:.3f} "
                f"CI=[{existing['pair_ci'][0]:.3f},{existing['pair_ci'][1]:.3f}] "
                f"radius={existing.get('radius')} eff_n={existing.get('eff_n')}"
            )
            results.append(existing)
            if first_knobs is None:
                first_knobs = existing.get("knobs")
            continue

        # Resolve ckpt radius for book selection (Finding #7)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        resolved_radius = radius_from_checkpoint(ck)
        if first_knobs is None:
            first_knobs = extract_deploy_knobs(ck.get("config", {}))
        print(f"[retro] ckpt={Path(ckpt_path).name} resolved_radius={resolved_radius}")

        # Series assert: radius must have a matching book
        book = resolve_book_for_radius(resolved_radius, books_by_radius, ckpt_path)

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
        results.append(result)
        print(
            f"[retro] step={result['ckpt_step']} wr={result['wr']:.3f} "
            f"CI=[{result['pair_ci'][0]:.3f},{result['pair_ci'][1]:.3f}] "
            f"radius={result.get('radius')} eff_n={result.get('eff_n')}"
        )

    # Per-stage slopes (Finding #6 + frozen verdicts 1 + 3)
    # Identify which radius stages are present
    all_radii = sorted({r.get("radius") for r in results if r.get("radius") is not None})
    stage_slopes = {}
    for rad in all_radii:
        s = compute_stage_slope(results, stage_radius=rad, n_boot=args.n_boot)
        if s is not None:
            stage_slopes[rad] = s
            print(
                f"\n[retro] Series radius={rad}: slope={s['theil_sen_slope']:.2e} "
                f"CI=[{s['slope_ci'][0]:.2e},{s['slope_ci'][1]:.2e}] "
                f"n_ckpts={s['n_ckpts']} underpowered={s['underpowered']}"
            )
            if not s["ci_excludes_zero"]:
                print(
                    f"  [retro] NULL SLOPE — CI contains 0. "
                    f"MDE(total ΔWR over series)≈{s['mde_total_delta_wr']:.3f}. "
                    f"Verdict: plateau SIGNAL only if true improvement ≥ MDE."
                )
        else:
            print(f"[retro] Series radius={rad}: fewer than 2 ckpts — no slope.")

    summary = {
        "ckpts": ckpts,
        "n_ckpts": len(results),
        "per_ckpt": results,
        "stage_slopes": {str(k): v for k, v in stage_slopes.items()},
        "discontinuity_note": (
            "The 200k boundary (r4->r5) is a structural discontinuity. "
            "Series A and Series B slopes are per-stage only — NEVER spliced (frozen verdict 3)."
        ),
    }
    (out / "slope_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[retro] slope summary written to {out}/slope_summary.json")


if __name__ == "__main__":
    main()
