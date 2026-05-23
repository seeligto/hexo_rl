"""§S181-AUDIT Wave 1 Track B — V-B verdict aggregator.

Reads B1_results.json + B2_results.json + B3_trunk_drift.json and applies
the pre-registered decision tree per
`audit/structural/track_b/B_launch_and_analysis_spec.md` §Aggregation.

LITERAL L13 guard. First match wins. Verdicts may stack — primary is the
first match in the tree, secondaries are any later matches.

Decision tree (verbatim):
  1. Any source share ≥ 60% across steps 500-2000 → V-B-A (source-targeted lever)
  2. Else if all three sources 25-45% across steps 500-2000 AND colony-pushing → V-B-B (multi-source damping)
  3. Else if colony_frac > 50% by step 2000 → V-B-C (feedback loop confirmed)
  4. Else if trunk inter_centroid_dist at step 1000 ≤ 50% of step 0 → V-B-D (trunk co-adaptation)
  5. Else → V-B-E (no clean match, escalate, NO real-run launch)

Output: audit/structural/track_b/B_aggregation.md per the recipe spec.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


# Decision tree thresholds — locked at spec level. Do NOT relax without
# updating REAL_RUN_RECIPE.md + sprint log + an explicit verdict re-issue.
VBA_SINGLE_SOURCE_THRESHOLD = 0.60
VBB_BAND_LOW = 0.25
VBB_BAND_HIGH = 0.45
VBC_COLONY_FRAC_THRESHOLD = 0.50
VBC_DEADLINE_STEP = 2000
VBD_CENTROID_RATIO_THRESHOLD = 0.50  # step-1000 inter ≤ 0.50 × step-0 inter
VBD_REFERENCE_STEP = 1000


def _check_vba(b1: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """V-B-A: any source's *mean* share ≥ 0.60 across steps 500-2000.

    Spec phrasing ("any source share ≥ 60% across steps 500-2000") is
    applied as sustained-share — the mean across the window, not the
    per-step max — so a single noisy batch does not decide the verdict.
    The per-step max is reported as a secondary signal so the operator
    can flag bursty / unstable attribution.
    """
    win = b1.get("windows", {}).get("steps_500_2000", {})
    smean_dict = win.get("shares_mean", {})
    smax_dict = win.get("shares_max", {})
    if not smean_dict:
        return False, {"reason": "no per_source events in window"}
    smean_max = max(smean_dict.values())
    fires = smean_max >= VBA_SINGLE_SOURCE_THRESHOLD
    detail = {
        "max_mean_source_share_in_window": smean_max,
        "max_source_by_mean": max(smean_dict, key=smean_dict.get),
        "mean_shares": smean_dict,
        "per_step_max_shares": smax_dict,
        "threshold": VBA_SINGLE_SOURCE_THRESHOLD,
    }
    # Always record the dominant source (by mean) — used by the routing
    # action even when V-B-A does not strictly fire, since a near-miss
    # mean (e.g. 0.55) still informs which source the secondary lever
    # should hit.
    detail["dominant_source_by_mean"] = max(smean_dict, key=smean_dict.get)
    return fires, detail


def _check_vbb(b1: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """V-B-B: all three sources in [25%, 45%] across steps 500-2000."""
    win = b1.get("windows", {}).get("steps_500_2000", {})
    sm = win.get("shares_mean", {})
    if not sm or len(sm) < 3:
        return False, {"reason": "no per_source events"}
    in_band = all(VBB_BAND_LOW <= sm.get(s, 0.0) <= VBB_BAND_HIGH
                  for s in ("pretrain", "recent", "uniform_self"))
    detail = {
        "shares_mean": sm,
        "band": [VBB_BAND_LOW, VBB_BAND_HIGH],
    }
    return in_band, detail


def _check_vbc(b2: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """V-B-C: colony_frac > 0.50 by step 2000."""
    snaps = b2.get("snapshots", [])
    fires = False
    fire_step = None
    fire_frac = None
    max_observed_frac = 0.0
    max_observed_step = None
    for s in snaps:
        step = int(s.get("step", 0))
        cf = float(s.get("colony_frac", 0.0))
        if cf > max_observed_frac:
            max_observed_frac = cf
            max_observed_step = step
        if step <= VBC_DEADLINE_STEP and cf > VBC_COLONY_FRAC_THRESHOLD:
            fires = True
            fire_step = step
            fire_frac = cf
            break
    return fires, {
        "deadline_step": VBC_DEADLINE_STEP,
        "threshold": VBC_COLONY_FRAC_THRESHOLD,
        "fire_step": fire_step,
        "fire_frac": fire_frac,
        "max_observed_frac": max_observed_frac,
        "max_observed_step": max_observed_step,
        "n_snapshots": len(snaps),
    }


def _check_vbd(b3: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """V-B-D: trunk inter_centroid_dist at step ≈1000 ≤ 50% of step-0 value."""
    ladder = b3.get("ladder", [])
    by_step: Dict[int, Dict[str, Any]] = {int(r.get("step", -1)): r for r in ladder}
    step0 = by_step.get(0)
    # Find the nearest checkpoint to reference step (1000) within ±halo.
    target = VBD_REFERENCE_STEP
    halo = 250
    best = None
    best_delta = halo + 1
    for s, r in by_step.items():
        if s == 0:
            continue
        delta = abs(s - target)
        if delta < best_delta:
            best = r
            best_delta = delta
    if step0 is None or best is None:
        return False, {"reason": "missing step-0 or step-~1000 in trunk ladder",
                       "step0_present": step0 is not None,
                       "step1000_candidate": best,
                       "halo": halo}
    inter0 = float(step0.get("inter_centroid_dist", 0.0))
    inter_t = float(best.get("inter_centroid_dist", 0.0))
    ratio = inter_t / inter0 if inter0 > 1e-9 else float("nan")
    fires = inter0 > 0 and ratio <= VBD_CENTROID_RATIO_THRESHOLD
    return fires, {
        "step0_inter": inter0,
        "step1000_step": int(best.get("step", -1)),
        "step1000_inter": inter_t,
        "ratio": ratio,
        "threshold": VBD_CENTROID_RATIO_THRESHOLD,
        "delta_from_target": best_delta,
    }


def _route(b1: Dict[str, Any], b2: Dict[str, Any], b3: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the decision tree; return primary + secondary verdicts."""
    vba_fires, vba_detail = _check_vba(b1)
    vbb_fires, vbb_detail = _check_vbb(b1)
    vbc_fires, vbc_detail = _check_vbc(b2)
    vbd_fires, vbd_detail = _check_vbd(b3)

    checks = [
        ("V-B-A", vba_fires, vba_detail),
        ("V-B-B", vbb_fires, vbb_detail),
        ("V-B-C", vbc_fires, vbc_detail),
        ("V-B-D", vbd_fires, vbd_detail),
    ]
    primary = next((name for name, fires, _ in checks if fires), "V-B-E")
    secondaries = [name for name, fires, _ in checks if fires and name != primary]

    return {
        "primary": primary,
        "secondaries": secondaries,
        "checks": {name: {"fires": fires, "detail": detail}
                   for name, fires, detail in checks},
    }


def _routing_action(verdict: str, secondary_set: List[str], vba_detail: Dict[str, Any]) -> str:
    """Map verdict → REAL_RUN_RECIPE §3 lever pointer."""
    if verdict == "V-B-A":
        src = vba_detail.get("dominant_source_by_mean") or "(unknown)"
        if src == "pretrain":
            return ("Pretrain class-weighting / per-class target temperature on "
                    "pretrain colony rows; OR reduce pretrain `recency_weight` channel")
        if src in ("bot", "bot_corpus"):
            return ("Static bot corpus → refresh hook "
                    "(`bot_corpus_refresh.enabled=true`, cooldown 25k)")
        if src in ("uniform_self", "recent"):
            return ("PSW (Prioritized Stratified Window) under-sampling colony on "
                    "selfplay slice; OR per-class target temperature on colony positions")
        return f"Source-targeted lever on dominant source: {src}"
    if verdict == "V-B-B":
        return "Multi-source damping: EMA (Wave 2 lands) + per-class target temp + entropy_reg already on"
    if verdict == "V-B-C":
        return "Refresh hook PRIORITY 1 + EMA on selfplay inference model"
    if verdict == "V-B-D":
        return "Aux heads forcing trunk discrimination: 2-stone opponent-reply aux head (~80 LOC) + EMA"
    return "ESCALATE — no real-run launch (V-B-E)"


def _render_markdown(routing: Dict[str, Any], b1_path: Path, b2_path: Path, b3_path: Path) -> str:
    verdict = routing["primary"]
    secondaries = routing["secondaries"]
    vba_detail = routing["checks"]["V-B-A"]["detail"]

    lines = [
        "# §S181-AUDIT Wave 1 — Track B / V-B aggregation",
        "",
        f"**Primary verdict: {verdict}**" + (
            f"  (secondaries: {', '.join(secondaries)})" if secondaries else ""
        ),
        "",
        f"**Routing action**: {_routing_action(verdict, secondaries, vba_detail)}",
        "",
        "Sources:",
        f"- B1 (`{b1_path.name}`) — per-source gradient attribution",
        f"- B2 (`{b2_path.name}`) — buffer position-class snapshots",
        f"- B3 (`{b3_path.name}`) — trunk feature drift",
        "",
        "## Decision-tree (LITERAL L13)",
        "",
        "| check | fires | detail |",
        "|---|:---:|---|",
    ]
    for name, body in routing["checks"].items():
        det = body["detail"]
        lines.append(
            f"| {name} | {'YES' if body['fires'] else 'no'} | "
            f"{json.dumps(det, separators=(', ', '='), default=str)[:240]} |"
        )

    lines += [
        "",
        "## Mechanism narrative",
        "",
        "Per the routing table above, the dominant force on the value-head "
        "discrimination collapse is the channel implied by the primary "
        "verdict. The Wave 2 lever stack pivots on this verdict — see "
        "`audit/structural/REAL_RUN_RECIPE.md` §3 conditional table and "
        "Stage 3 of the Wave 2 dispatcher.",
        "",
        "## Track D cross-reference",
        "",
        "Cross-reference the primary verdict with `audit/structural/"
        "track_d_pipeline_regression.md` §4 smoking-gun candidates. "
        "(Aggregator does not auto-cross; produce manually after reading "
        "the verdict + Track D ranking — see `B_track_d_xref.md`.)",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", type=Path,
                    default=REPO / "audit" / "structural" / "track_b" / "B1_results.json")
    ap.add_argument("--b2", type=Path,
                    default=REPO / "audit" / "structural" / "track_b" / "B2_results.json")
    ap.add_argument("--b3", type=Path,
                    default=REPO / "audit" / "structural" / "track_b" / "B3_trunk_drift.json")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "audit" / "structural" / "track_b")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    def _read_or_default(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
        if not path.exists():
            return default
        try:
            body = path.read_text().strip()
            if not body:
                return default
            return json.loads(body)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
            return default

    b1 = _read_or_default(args.b1, {"windows": {}, "checkpoints": []})
    b2 = _read_or_default(args.b2, {"snapshots": []})
    b3 = _read_or_default(args.b3, {"ladder": []})

    routing = _route(b1, b2, b3)
    out_json = args.output_dir / "B_aggregation.json"
    out_md = args.output_dir / "B_aggregation.md"
    out_json.write_text(json.dumps(routing, indent=2, default=str))
    out_md.write_text(_render_markdown(routing, args.b1, args.b2, args.b3))
    print(f"Primary verdict: {routing['primary']}")
    if routing["secondaries"]:
        print(f"Secondaries: {', '.join(routing['secondaries'])}")
    print(f"wrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()
