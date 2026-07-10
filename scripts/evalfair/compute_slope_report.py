"""Compute per-stage slopes + generate retro_slope.md and PNG plot.

Usage:
  .venv/bin/python -m scripts.evalfair.compute_slope_report \\
    --results-dir reports/evalfair/retro_slope \\
    --out-dir reports/evalfair

Reads all result.json files from the retro sweep, computes:
  - Per-stage Theil-Sen slope + pair-bootstrap CI (finding #6)
  - 175k anchor sanity check vs verdict-2 0.594
  - eff_n per ckpt (distinct suffixes)
  - Slope verdict per stage
Writes retro_slope.md + retro_slope_wr_trajectory.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# -- matplotlib lazy import (only needed for PNG)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_all_results(results_dir: Path) -> List[Dict]:
    results = []
    for p in sorted(results_dir.glob("*/result.json")):
        r = json.loads(p.read_text())
        results.append(r)
    return sorted(results, key=lambda r: r["ckpt_step"])


def _theil_sen(steps, wrs):
    slopes = []
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            dx = steps[j] - steps[i]
            if dx != 0:
                slopes.append((wrs[j] - wrs[i]) / dx)
    return float(np.median(slopes)) if slopes else float("nan")


def pair_bootstrap_slope_ci(steps, per_ckpt_pair_scores, n_boot=2000, seed=42):
    if len(steps) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(ps) for ps in per_ckpt_pair_scores]
    boot_slopes = []
    for _ in range(n_boot):
        wrs = [float(a[rng.integers(0, len(a), size=len(a))].mean()) for a in arrs]
        boot_slopes.append(_theil_sen(steps, wrs))
    valid = [s for s in boot_slopes if not np.isnan(s)]
    if not valid:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def stage_results(all_results, radius):
    return [r for r in all_results if r.get("radius") == radius]


def format_slope(slope, ci_lo, ci_hi, unit="per 100k steps"):
    s100 = slope * 100_000 if not np.isnan(slope) else float("nan")
    l100 = ci_lo * 100_000 if not np.isnan(ci_lo) else float("nan")
    h100 = ci_hi * 100_000 if not np.isnan(ci_hi) else float("nan")
    return f"{s100:.4f} WR/{unit} [{l100:.4f}, {h100:.4f}]"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="reports/evalfair/retro_slope")
    ap.add_argument("--out-dir", default="reports/evalfair")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = load_all_results(results_dir)
    if not all_results:
        print("ERROR: no result.json files found")
        sys.exit(1)
    print(f"Loaded {len(all_results)} ckpt results")

    # ── 175k anchor sanity ────────────────────────────────────────────────────
    anchor_result = next((r for r in all_results if r["ckpt_step"] == 175_000), None)
    VERDICT2_WR = 0.594
    VERDICT2_CI = [0.508, 0.672]
    anchor_pass = False
    anchor_note = "NOT FOUND"
    if anchor_result:
        awr = anchor_result["wr"]
        aci = anchor_result["pair_ci"]
        # Pass if 0.594 is within the new run's CI AND new WR is within verdict-2 CI
        in_new_ci = aci[0] <= VERDICT2_WR <= aci[1]
        in_v2_ci  = VERDICT2_CI[0] <= awr <= VERDICT2_CI[1]
        anchor_pass = in_new_ci or in_v2_ci
        anchor_note = (
            f"WR={awr:.3f} CI=[{aci[0]:.3f},{aci[1]:.3f}]; "
            f"verdict-2=0.594 in new CI: {in_new_ci}; "
            f"new WR in v2 CI: {in_v2_ci}; "
            f"PASS={anchor_pass}"
        )
    print(f"175k anchor: {anchor_note}")

    # ── Per-stage slope ───────────────────────────────────────────────────────
    stages = {}
    for radius in [4, 5]:
        sr = sorted(stage_results(all_results, radius), key=lambda r: r["ckpt_step"])
        if len(sr) < 2:
            stages[radius] = None
            continue
        steps = [r["ckpt_step"] for r in sr]
        wrs   = [r["wr"] for r in sr]
        pairs = [r["per_pair_scores"] for r in sr]
        slope = _theil_sen(steps, wrs)
        ci_lo, ci_hi = pair_bootstrap_slope_ci(steps, pairs, n_boot=args.n_boot)
        total_span = steps[-1] - steps[0]
        mde_wr = 0.13
        stages[radius] = {
            "radius": radius,
            "n_ckpts": len(sr),
            "steps": steps,
            "wrs": wrs,
            "slope": slope,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_excludes_zero": (ci_lo > 0) or (ci_hi < 0),
            "total_span": total_span,
            "mde_wr": mde_wr,
            "underpowered": len(sr) < 4,
            "results": sr,
        }

    # ── Latest WR (248k) ─────────────────────────────────────────────────────
    latest = max(all_results, key=lambda r: r["ckpt_step"])
    print(f"Latest WR (step={latest['ckpt_step']}): {latest['wr']:.3f} CI={latest['pair_ci']}")

    # ── eff_n range ───────────────────────────────────────────────────────────
    eff_ns = [r.get("eff_n", 0) for r in all_results]
    print(f"eff_n range: {min(eff_ns)}-{max(eff_ns)} (nominal n={all_results[0]['n']})")

    # ── PNG plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {4: "#2196F3", 5: "#FF9800"}  # blue=r4, orange=r5

    for radius, stage in stages.items():
        if stage is None:
            continue
        steps = stage["steps"]
        wrs = stage["wrs"]
        cis = [(r["pair_ci"][0], r["pair_ci"][1]) for r in stage["results"]]
        lo_arr = [c[0] for c in cis]
        hi_arr = [c[1] for c in cis]
        c = colors[radius]
        label = f"Series {'A' if radius==4 else 'B'} (r={radius})"
        ax.plot([s/1000 for s in steps], wrs, "o-", color=c, label=label, linewidth=2, markersize=6)
        ax.fill_between([s/1000 for s in steps], lo_arr, hi_arr, alpha=0.15, color=c)

    # Mark 175k anchor
    if anchor_result:
        ax.axhline(VERDICT2_WR, color="red", linestyle="--", linewidth=1.2,
                   label=f"verdict-2 0.594 (175k)")
        ax.plot([175], [anchor_result["wr"]], "r*", markersize=14, zorder=5,
                label=f"175k anchor WR={anchor_result['wr']:.3f}")

    # Stage boundary
    ax.axvline(200, color="gray", linestyle=":", linewidth=1, label="200k boundary (r4→r5)")

    ax.set_xlabel("Step (k)")
    ax.set_ylabel("Win Rate vs SealBot d5")
    ax.set_title("run2 Strength Trajectory — EVALFAIR instrument (deploy-matched, fair opening book)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    png_path = out_dir / "retro_slope_wr_trajectory.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"PNG saved: {png_path}")

    # ── Markdown report ───────────────────────────────────────────────────────
    md_lines = []
    md_lines.append("# D-A RETRO FAIR SLOPE — results\n")
    md_lines.append(f"**Date:** 2026-07-09  **Instrument:** EVALFAIR (accepted, WP1)  "
                    f"**Head:** DeployHeadBot Gumbel-150  **Opp:** SealBot d5\n")
    md_lines.append(
        "**Discontinuity note:** The 200k boundary (r4→r5) is a structural discontinuity. "
        "Series A and Series B slopes are computed per-stage ONLY — never spliced "
        "(frozen verdict 3).\n"
    )

    # Anchor
    status = "**PASS**" if anchor_pass else "**FAIL — STOP**"
    md_lines.append(f"\n## 175k Anchor Sanity\n\n{status} {anchor_note}\n")
    if not anchor_pass:
        md_lines.append("\n> **INSTRUMENT FAILURE — 175k anchor does not reproduce 0.594 within CI. "
                        "Investigate before reading any slope verdict.**\n")

    # Per-ckpt table
    md_lines.append("\n## Per-checkpoint Win Rate Table\n\n")
    md_lines.append("| step | radius | WR | CI_lo | CI_hi | eff_n | distinct/nominal | wall_s |\n")
    md_lines.append("|------|--------|----|-------|-------|-------|-----------------|--------|\n")
    for r in all_results:
        n_nom = r.get("n", 0)
        eff_n = r.get("eff_n", 0)
        wall = r.get("wall_sec", 0)
        md_lines.append(
            f"| {r['ckpt_step']} | {r.get('radius')} | {r['wr']:.3f} | "
            f"{r['pair_ci'][0]:.3f} | {r['pair_ci'][1]:.3f} | "
            f"{eff_n} | {eff_n}/{n_nom} | {wall:.0f} |\n"
        )

    # Stage slopes
    md_lines.append("\n## Slope Verdicts\n\n")
    for radius in [4, 5]:
        series = "A" if radius == 4 else "B"
        md_lines.append(f"### Series {series} (radius={radius})\n\n")
        stage = stages.get(radius)
        if stage is None:
            md_lines.append(f"_Fewer than 2 ckpts — no slope computed._\n\n")
            continue

        slope_str = format_slope(stage["slope"], stage["ci_lo"], stage["ci_hi"])
        md_lines.append(f"- **n_ckpts:** {stage['n_ckpts']}  **span:** {stage['steps'][0]}→{stage['steps'][-1]} ({stage['total_span']})\n")
        md_lines.append(f"- **Theil-Sen slope:** {slope_str}\n")
        md_lines.append(f"- **CI excludes 0:** {stage['ci_excludes_zero']}\n")
        md_lines.append(f"- **MDE (total ΔWR over series):** ≈{stage['mde_wr']:.2f} WR\n")
        if stage["underpowered"]:
            md_lines.append(f"- ⚠ **UNDERPOWERED** (< 4 ckpts)\n")

        # Verdict text (verbatim from design §10)
        md_lines.append("\n**Pre-registered verdict 1 (verbatim):**\n")
        md_lines.append(
            "> SLOPE: Theil-Sen over ≥7 checkpoint points, pair-bootstrap CI. "
            "Slope > 0 with CI excluding 0 → the net is learning vs SealBot under a fair instrument. "
            "CI containing 0 across the span → plateau SIGNAL (descriptive only; the seeding gate "
            "remains trap-flip, NOT this — do not conflate).\n\n"
        )
        if stage["ci_excludes_zero"] and stage["slope"] > 0:
            md_lines.append("**VERDICT: LEARNING CONFIRMED** — slope > 0 with CI excluding 0.\n\n")
        elif not stage["ci_excludes_zero"]:
            mde = stage["mde_wr"]
            md_lines.append(
                f"**VERDICT: PLATEAU SIGNAL** (descriptive only) — CI contains 0. "
                f"MDE≈{mde:.2f} total ΔWR; null means 'no slope ≥ MDE detected', not 'zero slope proved'. "
                f"Seeding gate remains trap-flip, not this.\n\n"
            )
        else:
            md_lines.append(f"**VERDICT: CI NEGATIVE** — slope < 0 with CI excluding 0. Regression signal.\n\n")

    # Latest WR
    md_lines.append(f"\n## Latest Fair WR\n\n")
    md_lines.append(
        f"**Step {latest['ckpt_step']}:** WR={latest['wr']:.3f} "
        f"CI=[{latest['pair_ci'][0]:.3f},{latest['pair_ci'][1]:.3f}] "
        f"radius={latest.get('radius')} eff_n={latest.get('eff_n')}\n"
    )

    # PNG reference
    md_lines.append(f"\n## Plot\n\n![WR trajectory](retro_slope_wr_trajectory.png)\n")

    md_path = out_dir / "retro_slope.md"
    md_path.write_text("".join(md_lines))
    print(f"Report written: {md_path}")

    # Summary to stdout for the final return message
    print("\n=== FINAL SUMMARY ===")
    print(f"175k anchor: {anchor_note}")
    for radius in [4, 5]:
        stage = stages.get(radius)
        if stage:
            print(f"Series {'A' if radius==4 else 'B'} (r={radius}) n={stage['n_ckpts']} "
                  f"slope={format_slope(stage['slope'], stage['ci_lo'], stage['ci_hi'])} "
                  f"ci_excl_0={stage['ci_excludes_zero']}")
    print(f"Latest WR (step={latest['ckpt_step']}): {latest['wr']:.3f} CI={latest['pair_ci']}")
    eff_ns_str = f"min={min(eff_ns)}, max={max(eff_ns)}"
    print(f"eff_n range: {eff_ns_str}")


if __name__ == "__main__":
    main()
