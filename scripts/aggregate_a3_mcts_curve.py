#!/usr/bin/env python3
"""Aggregate A3 MCTS-N curve: argmax + MCTS-{32,64,128} vs SealBot.

Outputs reports/ablation_169/A3_mcts_curve.md with:
  - 4-point WR table + Wilson 95% CIs
  - Monotonicity test (Cochran-Armitage trend test)
  - Pre-registered verdict (MONOTONIC-DECLINE / FLAT-NON-MONOTONIC / AMBIGUOUS)

Usage:
    python scripts/aggregate_a3_mcts_curve.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports" / "ablation_169"

# ── Wilson CI ─────────────────────────────────────────────────────────────────
def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return centre - spread, centre + spread


# ── Cochran-Armitage trend test (one-sided: decline with N) ──────────────────
def cochran_armitage(scores: list[float], wins: list[int], ns: list[int]) -> float:
    """
    Return two-sided p-value for linear trend in proportions.

    scores: numeric dose/rank for each group (e.g. MCTS sims count)
    wins, ns: success counts and group sizes

    Uses the standard Cochran-Armitage Z statistic.
    """
    import scipy.stats as st  # type: ignore[import]

    k = len(scores)
    assert k == len(wins) == len(ns)
    N = sum(ns)
    p_bar = sum(wins) / N

    # score-weighted sums
    c_bar = sum(scores[i] * ns[i] for i in range(k)) / N

    num = sum(scores[i] * wins[i] for i in range(k)) - p_bar * sum(
        scores[i] * ns[i] for i in range(k)
    )
    var = p_bar * (1 - p_bar) * sum(ns[i] * (scores[i] - c_bar) ** 2 for i in range(k))

    if var <= 0:
        return 1.0
    Z = num / math.sqrt(var)
    return float(2 * st.norm.sf(abs(Z)))  # two-sided


# ── CI overlap fraction ───────────────────────────────────────────────────────
def ci_overlap_fraction(lo1: float, hi1: float, lo2: float, hi2: float) -> float:
    """Overlap as fraction of the smaller interval width."""
    overlap = max(0.0, min(hi1, hi2) - max(lo1, lo2))
    w1 = hi1 - lo1
    w2 = hi2 - lo2
    smaller = min(w1, w2)
    if smaller <= 0:
        return 0.0
    return overlap / smaller


# ── Load result files ─────────────────────────────────────────────────────────
def load_arm(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: missing {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def main() -> None:
    argmax_d = load_arm(REPORTS / "A3_argmax.json")
    mcts32_d = load_arm(REPORTS / "A3_mcts32.json")
    mcts64_d = load_arm(REPORTS / "A3_mcts64.json")
    mcts128_d = load_arm(REPORTS / "A3_mcts128.json")

    arms = [
        ("argmax", 0, argmax_d),
        ("MCTS-32", 32, mcts32_d),
        ("MCTS-64", 64, mcts64_d),
        ("MCTS-128", 128, mcts128_d),
    ]

    rows: list[dict] = []
    for label, sims, d in arms:
        n = d["wins"] + d["losses"] + d["draws"]
        wins = d["wins"]
        wr = wins / n
        lo, hi = wilson_ci(wins, n)
        rows.append(
            {
                "label": label,
                "sims": sims,
                "wins": wins,
                "losses": d["losses"],
                "draws": d["draws"],
                "n": n,
                "wr": wr,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )

    # ── Cochran-Armitage ──────────────────────────────────────────────────────
    scores = [float(r["sims"]) for r in rows]
    wins_list = [r["wins"] for r in rows]
    ns_list = [r["n"] for r in rows]
    ca_p = cochran_armitage(scores, wins_list, ns_list)

    # ── Monotonicity analysis ─────────────────────────────────────────────────
    wrs = [r["wr"] for r in rows]
    is_monotone_decline = all(wrs[i] > wrs[i + 1] for i in range(len(wrs) - 1))

    # Pairwise CI overlaps between consecutive arms
    overlaps: list[float] = []
    for i in range(len(rows) - 1):
        ov = ci_overlap_fraction(
            rows[i]["ci_lo"], rows[i]["ci_hi"],
            rows[i + 1]["ci_lo"], rows[i + 1]["ci_hi"],
        )
        overlaps.append(ov)

    max_consecutive_overlap = max(overlaps) if overlaps else 1.0

    # Pre-registered verdict logic
    MONO_CA_P_THRESH = 0.10
    MONO_MAX_CI_OVERLAP = 0.50
    FLAT_MIN_CI_OVERLAP = 0.75

    if is_monotone_decline and ca_p < MONO_CA_P_THRESH and max_consecutive_overlap <= MONO_MAX_CI_OVERLAP:
        verdict = "MONOTONIC-DECLINE"
        verdict_detail = (
            "All four points strictly decreasing, max consecutive CI overlap "
            f"{max_consecutive_overlap:.0%} ≤ 50%, Cochran-Armitage p={ca_p:.4f} < 0.10. "
            "PMA-value-semantics degradation hypothesis CONFIRMED."
        )
    elif not is_monotone_decline or all(
        ci_overlap_fraction(rows[i]["ci_lo"], rows[i]["ci_hi"],
                            rows[j]["ci_lo"], rows[j]["ci_hi"]) > FLAT_MIN_CI_OVERLAP
        for i in range(len(rows)) for j in range(i + 1, len(rows))
    ):
        verdict = "FLAT-NON-MONOTONIC"
        verdict_detail = (
            "Ordering inversion present or all CI pairs overlap > 75%. "
            "PMA-value-semantics hypothesis REFUTED. Search-depth-specific mechanism."
        )
    else:
        verdict = "AMBIGUOUS"
        verdict_detail = (
            "Monotone decline present but CI overlaps or Cochran-Armitage p do not "
            "meet full MONOTONIC-DECLINE criterion. Operator judgment required."
        )

    # ── Write markdown ────────────────────────────────────────────────────────
    lines: list[str] = [
        "# A3 PMA+Global — MCTS-N Win-Rate Curve vs SealBot",
        "",
        "**Goal:** discriminate PMA-value-semantics degradation hypothesis.",
        "If WR declines monotonically with MCTS-N, compounding error across",
        "MCTS backups confirms the mechanism (§170 P1).",
        "",
        "## Pre-registered verdict criteria (locked before runs)",
        "",
        "- **MONOTONIC-DECLINE**: argmax > MCTS-32 > MCTS-64 > MCTS-128,",
        "  each consecutive CI overlap ≤ 50%, Cochran-Armitage p < 0.10.",
        "- **FLAT/NON-MONOTONIC**: any inversion OR all CIs overlap > 75%.",
        "",
        "## Results",
        "",
        "| Method   | sims | W   | L   | D | n   | WR     | 95% CI           |",
        "|----------|------|-----|-----|---|-----|--------|------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']:<8} | {r['sims']:>4} | {r['wins']:>3} | "
            f"{r['losses']:>3} | {r['draws']:.0f} | {r['n']:>3} | "
            f"{r['wr']:.1%}  | [{r['ci_lo']:.1%}, {r['ci_hi']:.1%}] |"
        )

    lines += [
        "",
        "## Monotonicity analysis",
        "",
        f"Strictly monotone decline: **{'YES' if is_monotone_decline else 'NO'}**",
        "",
        "Consecutive CI overlap fractions:",
        "",
    ]
    for i, ov in enumerate(overlaps):
        lines.append(
            f"- {rows[i]['label']} → {rows[i+1]['label']}: {ov:.0%}"
        )

    lines += [
        "",
        f"Max consecutive overlap: **{max_consecutive_overlap:.0%}**",
        f"(threshold ≤ 50% for MONOTONIC-DECLINE)",
        "",
        "## Cochran-Armitage trend test",
        "",
        f"p = **{ca_p:.4f}** (two-sided; threshold < 0.10 for MONOTONIC-DECLINE)",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        verdict_detail,
        "",
        "## §170 implication",
        "",
    ]

    if verdict == "MONOTONIC-DECLINE":
        lines += [
            "PMA-value-semantics confirmed. §170 commits to:",
            "- Preserve A1 min-pool for value path (canonical canonical §169 winner).",
            "- Any PMA side-channel must be **policy-only** (no value head influence).",
            "- Bet B / option α: 'preserve A1 pool, add side-channel only'.",
            "",
            "Proceed to §170 P2 + P3 on A1 extension.",
        ]
    elif verdict == "FLAT-NON-MONOTONIC":
        lines += [
            "PMA-value-semantics hypothesis refuted. Search-depth-specific issue.",
            "Surface to operator before §170 P2/P3 — different fix mechanism needed.",
        ]
    else:
        lines += [
            "Ambiguous result. Surface to operator before committing §170 direction.",
        ]

    out = REPORTS / "A3_mcts_curve.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Written: {out}")
    print(f"\nVERDICT: {verdict}")
    for r in rows:
        print(f"  {r['label']:8s}: {r['wr']:.1%} [{r['ci_lo']:.1%}, {r['ci_hi']:.1%}]  ({r['wins']}W/{r['losses']}L/{r['draws']}D)")
    print(f"  Cochran-Armitage p = {ca_p:.4f}")
    print(f"  Max consecutive CI overlap = {max_consecutive_overlap:.0%}")
    print(f"\n{verdict_detail}")


if __name__ == "__main__":
    main()
