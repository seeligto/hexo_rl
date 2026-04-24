#!/usr/bin/env python3
"""§122 sweep — emit final memo to reports/investigations/phase122_sweep/memo.md.

Reads:
  logs/sweep/state.json                                        — driver progress
  logs/sweep/{variant}/*.json                                  — structlog metrics
  reports/investigations/phase122_sweep/tournament.json        — phase 3 result
  configs/variants/sweep_*.yaml                                — channel sets
  checkpoints/sweep/{variant}/checkpoint_NNNNNNNN.pt           — for param count

Emits:
  reports/investigations/phase122_sweep/memo.md
  reports/investigations/phase122_sweep/throughput.json
  reports/investigations/phase122_sweep/per_variant_curves.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGS_ROOT = REPO_ROOT / "logs" / "sweep"
CHECKPOINTS_ROOT = REPO_ROOT / "checkpoints" / "sweep"
VARIANTS_DIR = REPO_ROOT / "configs" / "variants"
OUT_DIR = REPO_ROOT / "reports" / "investigations" / "phase122_sweep"

ALL_VARIANTS = (
    "sweep_2ch", "sweep_3ch", "sweep_4ch",
    "sweep_6ch", "sweep_8ch", "sweep_18ch",
)
PHASE1_STEPS = 2500
PHASE2_STEPS = 10000
TIE_THRESHOLD_PP = 3.0   # WR difference under which two variants are "tied"
DECISIVE_THRESHOLD_PP = 10.0
# §122 augmentation confound: variants without planes 16 / 17 are slightly
# more rotationally invariant than variants that carry them (the augment
# kernel zeroes 90 corner cells in non-bijective rotations, which leaks a
# coordinate-system signal through the scalar planes). Tournament margins
# under this threshold across the plane-16/17 dividing line are not
# decisive enough to overcome the confound — the aggregator refuses to
# auto-recommend across that line at margins < 5pp.
ROTATION_CONFOUND_GUARD_PP = 5.0
PLANES_WITH_ROTATION_LEAK = {16, 17}


def has_scalar_planes(variant: str) -> bool:
    return any(p in PLANES_WITH_ROTATION_LEAK for p in variant_channels(variant))


# ── Per-variant info ────────────────────────────────────────────────────────

def variant_channels(variant: str) -> List[int]:
    p = VARIANTS_DIR / f"{variant}.yaml"
    if not p.exists():
        return []
    with p.open() as f:
        cfg = yaml.safe_load(f) or {}
    return list(cfg.get("input_channels", []))


def parse_variant_metrics(variant: str) -> Dict[int, Dict[str, float]]:
    d = LOGS_ROOT / variant
    if not d.exists():
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for p in d.glob("*.json"):
        try:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    step = rec.get("step")
                    if not isinstance(step, int):
                        continue
                    bucket = out.setdefault(step, {})
                    for k, v in rec.items():
                        if isinstance(v, (int, float)) and k != "step":
                            bucket.setdefault(k, float(v))
        except OSError:
            pass
    return out


def parameter_count(variant: str) -> Optional[int]:
    """Reconstruct a HexTacToeNet for the variant and count params.

    Lazy import of torch + model to avoid forcing the dependency on aggregate
    runs when the user just wants the JSON outputs.
    """
    try:
        import torch  # noqa: F401
        from hexo_rl.model.network import HexTacToeNet
    except ImportError:
        return None
    channels = variant_channels(variant)
    if not channels:
        return None
    model = HexTacToeNet(
        in_channels=len(channels),
        input_channels=channels,
    )
    n = sum(p.numel() for p in model.parameters())
    del model
    return int(n)


def steps_per_second(variant: str) -> Optional[float]:
    """Estimate training throughput from the variant's structlog 'train' events."""
    metrics = parse_variant_metrics(variant)
    if not metrics:
        return None
    steps_seen = sorted(metrics.keys())
    if len(steps_seen) < 2:
        return None
    # Accept either explicit "steps_per_sec" or compute from wall-time deltas.
    sps_vals = [m["steps_per_sec"] for m in metrics.values() if "steps_per_sec" in m]
    if sps_vals:
        return float(sum(sps_vals) / len(sps_vals))
    return None


def metric_at(metrics: Dict[int, Dict[str, float]], target: int, key: str) -> Optional[float]:
    if target in metrics and key in metrics[target]:
        return metrics[target][key]
    earlier = [s for s in metrics if s <= target and key in metrics[s]]
    if not earlier:
        return None
    return metrics[max(earlier)][key]


# ── Recommendation logic ────────────────────────────────────────────────────

def derive_recommendation(
    variants: List[str],
    tournament: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Produce (recommended_variant, reasoning_lines).

    Decision logic looks at: tournament Bradley-Terry ratings (primary),
    axis distribution (constraint), policy CE / value MSE (sanity), WR.
    Auxiliary heads (ownership, threat, chain) are intentionally NOT used
    in the recommendation — D17 established those heads are collapsed
    near zero across all configs. Including them would inject noise.
    """
    bt = tournament.get("bradley_terry", {})
    if not bt:
        return ("(insufficient data)",
                ["No Bradley-Terry ratings available; tournament probably did not complete."])

    # Sort by rating descending.
    ranked = sorted(bt.items(), key=lambda kv: kv[1]["rating"], reverse=True)
    reasoning: List[str] = []

    top_name, top_info = ranked[0]
    top_rating = top_info["rating"]
    top_channels = variant_channels(top_name)

    # Identify the 18-channel baseline if it played.
    baseline = next((n for n, _ in ranked if n == "sweep_18ch"), None)
    baseline_rating = bt[baseline]["rating"] if baseline else None

    # Convert ratings to approximate WR vs top via 400-Elo logistic.
    def wr_diff(a: float, b: float) -> float:
        """Approximate win-rate of A over B given Bradley-Terry ratings (Elo-like)."""
        return 1.0 / (1.0 + 10 ** (-(a - b) / 400.0))

    # Tie detection: if any smaller variant is within TIE_THRESHOLD_PP WR of top.
    smaller_ties: List[str] = []
    for name, info in ranked[1:]:
        ch_count = len(variant_channels(name))
        if ch_count < len(top_channels):
            wr = wr_diff(info["rating"], top_rating)
            if (0.5 - wr) * 100.0 <= TIE_THRESHOLD_PP:
                smaller_ties.append(name)

    # Decisive 18ch baseline?
    if baseline and baseline == top_name and baseline_rating is not None:
        # The baseline IS top — see if it's decisive.
        runner_name, runner_info = ranked[1] if len(ranked) > 1 else (None, None)
        if runner_info is not None:
            runner_wr = wr_diff(runner_info["rating"], baseline_rating)
            margin_pp = (0.5 - runner_wr) * 100.0
            if margin_pp >= DECISIVE_THRESHOLD_PP:
                reasoning.append(
                    f"sweep_18ch wins decisively (runner-up {runner_name} ≈ {runner_wr:.1%} WR, "
                    f"margin {margin_pp:.1f}pp ≥ {DECISIVE_THRESHOLD_PP}pp)."
                )
                reasoning.append(
                    "Recommendation: KEEP 18 channels. Pursue the rest of §122 "
                    "(backbone form, permanent rotation) without channel reduction."
                )
                return ("sweep_18ch", reasoning)

    # Tie-break: prefer the smallest channel count among tied configs.
    if smaller_ties:
        smaller_ties.sort(key=lambda n: len(variant_channels(n)))
        chosen = smaller_ties[0]
        reasoning.append(
            f"Top variant {top_name} ({len(top_channels)} ch) is statistically tied "
            f"with {chosen} ({len(variant_channels(chosen))} ch). Tie-break: prefer "
            f"smaller channel count."
        )
        reasoning.append(f"Recommendation: §122 retrain on {chosen}.")
        return (chosen, reasoning)

    # ── Rotation-confound guard ───────────────────────────────────────────
    # If top variant carries planes 16 / 17 and the runner-up is a smaller
    # variant that does NOT, and the WR gap is under the rotation-confound
    # threshold, we cannot tell signal from artifact. Refuse auto-recommend.
    runner_up = ranked[1] if len(ranked) > 1 else None
    if runner_up is not None:
        ru_name, ru_info = runner_up
        ru_wr = wr_diff(ru_info["rating"], top_rating)
        margin_pp = (0.5 - ru_wr) * 100.0
        top_has_scalar = has_scalar_planes(top_name)
        ru_has_scalar = has_scalar_planes(ru_name)
        if (
            top_has_scalar != ru_has_scalar
            and margin_pp < ROTATION_CONFOUND_GUARD_PP
        ):
            reasoning.append(
                f"INCONCLUSIVE: top variant {top_name} (planes_16_17="
                f"{top_has_scalar}) leads runner-up {ru_name} (planes_16_17="
                f"{ru_has_scalar}) by only {margin_pp:.1f}pp. The "
                f"rotation-augmentation artifact on planes 16/17 (271/90 "
                f"corner zeroing under non-bijective hex symmetries) makes "
                f"variants with scalar planes ~marginally less rotation-"
                f"invariant than variants without — gaps under "
                f"{ROTATION_CONFOUND_GUARD_PP}pp across this dividing line "
                f"cannot be distinguished from the artifact."
            )
            reasoning.append(
                "Recommendation: DO NOT auto-decide. Manually inspect the "
                "axis distribution + value MSE per-step curves; if those "
                "favour the smaller variant or are flat, prefer it. If the "
                "scalar-plane variant has a clear value-head advantage in "
                "addition to the WR lead, prefer it. Otherwise run a "
                "follow-up tournament at 200 games per pair to get under "
                "the noise floor."
            )
            return ("(inconclusive — see notes)", reasoning)

    reasoning.append(
        f"Top variant {top_name} ({len(top_channels)} ch) wins outright "
        f"with no tied smaller variant within {TIE_THRESHOLD_PP}pp."
    )
    reasoning.append(f"Recommendation: §122 retrain on {top_name}.")
    return (top_name, reasoning)


# ── Memo writer ────────────────────────────────────────────────────────────

def write_memo(
    out_path: Path,
    *,
    variants: List[str],
    state: Dict[str, Any],
    tournament: Dict[str, Any],
    per_variant: Dict[str, Dict[str, Any]],
    recommendation: Tuple[str, List[str]],
) -> None:
    rec_name, rec_reasoning = recommendation
    lines: List[str] = []
    lines.append("# §122 sweep — input-channel reduction memo")
    lines.append("")
    lines.append(f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("## Variants under test")
    lines.append("")
    lines.append("| variant | channels | parameters |")
    lines.append("| --- | --- | --- |")
    for v in variants:
        ch = variant_channels(v)
        params = per_variant.get(v, {}).get("parameter_count")
        params_s = f"{params:,}" if isinstance(params, int) else "—"
        lines.append(f"| `{v}` | `{ch}` | {params_s} |")
    lines.append("")

    # Phase 1 ranking
    lines.append("## Phase 1 ranking (step 2,500)")
    lines.append("")
    lines.append("| variant | policy CE | value MSE | WR vs random | WR vs anchor |")
    lines.append("| --- | --- | --- | --- | --- |")
    for v in variants:
        m = per_variant.get(v, {}).get("phase1_metrics", {})
        lines.append(
            f"| `{v}` | "
            f"{_fmt(m.get('policy_ce') or m.get('policy_loss'))} | "
            f"{_fmt(m.get('value_mse') or m.get('value_loss'))} | "
            f"{_fmt_pct(m.get('wr_random'))} | "
            f"{_fmt_pct(m.get('wr_anchor'))} |"
        )
    lines.append("")

    # Phase 2 ranking
    lines.append("## Phase 2 ranking (step 10,000)")
    lines.append("")
    lines.append("| variant | policy CE | value MSE | WR vs random | WR vs anchor | axis worst |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for v in variants:
        m = per_variant.get(v, {}).get("phase2_metrics", {})
        axis_max = max(
            (m.get(k, 0.0) for k in ("axis_q", "axis_r", "axis_s")),
            default=0.0,
        )
        lines.append(
            f"| `{v}` | "
            f"{_fmt(m.get('policy_ce') or m.get('policy_loss'))} | "
            f"{_fmt(m.get('value_mse') or m.get('value_loss'))} | "
            f"{_fmt_pct(m.get('wr_random'))} | "
            f"{_fmt_pct(m.get('wr_anchor'))} | "
            f"{_fmt(axis_max if axis_max else None)} |"
        )
    lines.append("")
    lines.append(
        "_axis worst > 0.55 over 4 consecutive evals → flagged \"learned but biased\"._"
    )
    lines.append("")

    # Tournament
    lines.append("## Tournament matrix (step 10,000)")
    lines.append("")
    pairwise = tournament.get("pairwise", [])
    if not pairwise:
        lines.append("_(tournament results not yet present — run scripts/tournament_sweep.py first)_")
    else:
        sv = list({p["a"] for p in pairwise} | {p["b"] for p in pairwise})
        sv.sort(key=lambda n: variants.index(n) if n in variants else 99)
        matrix = {(p["a"], p["b"]): (p["wins_a"], p["wins_b"], p.get("draws", 0)) for p in pairwise}
        for a, b, _ in list(matrix.values()):
            pass  # silence linter
        header = "| | " + " | ".join(sv) + " |"
        sep = "|" + "---|" * (len(sv) + 1)
        lines.append(header)
        lines.append(sep)
        for row in sv:
            cells: List[str] = []
            for col in sv:
                if row == col:
                    cells.append("—")
                elif (row, col) in matrix:
                    wa, wb, _ = matrix[(row, col)]
                    denom = max(1, wa + wb)
                    cells.append(f"{wa}-{wb} ({wa / denom:.0%})")
                elif (col, row) in matrix:
                    wa, wb, _ = matrix[(col, row)]
                    denom = max(1, wa + wb)
                    cells.append(f"{wb}-{wa} ({wb / denom:.0%})")
                else:
                    cells.append("·")
            lines.append(f"| **{row}** | " + " | ".join(cells) + " |")
    lines.append("")

    # Throughput
    lines.append("## Throughput")
    lines.append("")
    lines.append("| variant | steps/sec |")
    lines.append("| --- | --- |")
    for v in variants:
        sps = per_variant.get(v, {}).get("steps_per_sec")
        lines.append(f"| `{v}` | {_fmt(sps)} |")
    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**Recommended variant for §122 retrain: `{rec_name}`**")
    lines.append("")
    for line in rec_reasoning:
        lines.append(f"- {line}")
    lines.append("")

    # Excluded signals — explicit so reviewers know not to lobby for them.
    lines.append("## Signals NOT used in the recommendation")
    lines.append("")
    lines.append(
        "- **Ownership head MSE**, **threat head BCE**, **chain head smooth-L1**: "
        "D17 established that the auxiliary heads are collapsed near zero "
        "across all production configs and remained so under the §122 input "
        "perturbations. They will report ~uniform near-zero deltas across "
        "every sweep variant; promoting them into the decision logic would "
        "add noise. Decide on policy CE, value MSE, axis distribution, WR, "
        "and tournament result."
    )
    lines.append(
        "- **Phase 1 absolute losses below step 1000 for sweep_2ch**: lower "
        "channel count leads to legitimately slower early-step learning. "
        "The driver loosens the policy-CE-by-1000 gate for sweep_2ch and "
        "sweep_3ch (see scripts/run_sweep.py:PER_VARIANT_POLICY_CE_BY_1000); "
        "weight late-phase-1 + phase-2 metrics, not the first 1000 steps."
    )
    lines.append("")

    # Surprise findings
    lines.append("## Surprise findings")
    lines.append("")
    lines.append(
        "_Edit this section by hand once the data is in. Anything unexpected — a "
        "variant that learned faster than baseline, a pathological axis "
        "distribution, an architecturally smaller variant outperforming a "
        "larger one — belongs here, not in the recommendation._"
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v) < 1e-3 or abs(v) >= 1e3:
        return f"{v:.3e}"
    return f"{v:.3f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if v <= 1.5 else f"{v:.1f}"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS))
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    state_path = LOGS_ROOT / "state.json"
    state: Dict[str, Any] = {}
    if state_path.exists():
        with state_path.open() as f:
            state = json.load(f)

    tournament_path = args.out_dir / "tournament.json"
    tournament: Dict[str, Any] = {}
    if tournament_path.exists():
        with tournament_path.open() as f:
            tournament = json.load(f)

    per_variant: Dict[str, Dict[str, Any]] = {}
    for v in args.variants:
        metrics = parse_variant_metrics(v)
        per_variant[v] = {
            "phase1_metrics": metric_at_step_dict(metrics, PHASE1_STEPS),
            "phase2_metrics": metric_at_step_dict(metrics, PHASE2_STEPS),
            "steps_per_sec": steps_per_second(v),
            "parameter_count": parameter_count(v),
            "channels": variant_channels(v),
        }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_variant_curves.json").open("w") as f:
        json.dump(per_variant, f, indent=2)
    with (out_dir / "throughput.json").open("w") as f:
        json.dump({v: per_variant[v]["steps_per_sec"] for v in args.variants}, f, indent=2)

    rec = derive_recommendation(list(args.variants), tournament)
    write_memo(
        out_dir / "memo.md",
        variants=list(args.variants),
        state=state,
        tournament=tournament,
        per_variant=per_variant,
        recommendation=rec,
    )
    print(f"wrote {out_dir / 'memo.md'}")
    print(f"recommendation: {rec[0]}")
    return 0


def metric_at_step_dict(metrics: Dict[int, Dict[str, float]], target: int) -> Dict[str, float]:
    if target in metrics:
        return dict(metrics[target])
    earlier = [s for s in metrics if s <= target]
    if not earlier:
        return {}
    return dict(metrics[max(earlier)])


if __name__ == "__main__":
    sys.exit(main())
