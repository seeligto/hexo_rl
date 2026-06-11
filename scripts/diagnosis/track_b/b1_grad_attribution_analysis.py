"""§S181-AUDIT Wave 1 Track B / B1 — per-source gradient attribution analysis.

Consumes `per_source_grad_norm` JSONL events emitted by the instrumented
trainer (`hexo_rl/training/track_b_attribution.py`) and produces the
per-source TOTAL pull + per-source SHARE trajectory per
`audit/structural/track_b/B_launch_and_analysis_spec.md` §"B1 — per-source
gradient attribution".

Outputs:
  - JSON ladder of per-source pull + share at every recorded step
  - Markdown summary (`B1_results.md`) recording shares at step 500 / 1000
    / 2000 / 3000 + per-group (trunk/value/policy) breakdowns
  - Trajectory CSV for downstream plotting

Usage:
    python -m scripts.diagnosis.track_b.b1_grad_attribution_analysis \\
        --log /path/to/s181_b_*.log \\
        --output-dir audit/structural/track_b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

SOURCES = ("pretrain", "recent", "uniform_self")
GROUPS = ("trunk", "value", "policy")
CHECKPOINT_STEPS = (500, 1000, 1500, 2000, 2500, 3000)


def _parse_events(log_path: Path) -> List[Dict[str, Any]]:
    """Extract every `per_source_grad_norm` JSONL line from a structlog file."""
    events: List[Dict[str, Any]] = []
    with log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "per_source_grad_norm":
                continue
            events.append(rec)
    return events


def _per_event_pulls(rec: Dict[str, Any]) -> Dict[str, float]:
    """Sum L2 across (trunk, value, policy) per source for one step.

    Returns a flat dict ``{source: total_pull, source_group: norm, ...}``.
    NaN entries (empty slice) contribute 0 to the source total.
    """
    out: Dict[str, float] = {}
    for src in SOURCES:
        src_total = 0.0
        for grp in GROUPS:
            key = f"{grp}_{src}"
            val = rec.get(key)
            if val is None:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if v != v:  # NaN check
                continue
            out[key] = v
            src_total += v
        out[src] = src_total
    out["total"] = sum(out[s] for s in SOURCES)
    return out


def _shares(pulls: Dict[str, float]) -> Dict[str, float]:
    """Convert per-source totals into proportional shares (sum to 1.0)."""
    total = pulls["total"]
    if total <= 0:
        return {s: 0.0 for s in SOURCES}
    return {s: pulls[s] / total for s in SOURCES}


def _per_group_shares(pulls: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Within each parameter group, what share does each source carry?"""
    per_group: Dict[str, Dict[str, float]] = {}
    for grp in GROUPS:
        denom = sum(pulls.get(f"{grp}_{s}", 0.0) for s in SOURCES)
        if denom <= 0:
            per_group[grp] = {s: 0.0 for s in SOURCES}
            continue
        per_group[grp] = {
            s: pulls.get(f"{grp}_{s}", 0.0) / denom for s in SOURCES
        }
    return per_group


def _window_summary(events: List[Dict[str, Any]], step_lo: int, step_hi: int) -> Dict[str, Any]:
    """Average shares + counts across every per_source event in [step_lo, step_hi]."""
    window = [e for e in events if step_lo <= int(e.get("step", 0)) <= step_hi]
    if not window:
        return {"n_steps": 0, "shares_mean": {}, "shares_max": {}, "per_group_mean": {}}
    n = len(window)
    sum_shares = {s: 0.0 for s in SOURCES}
    max_shares = {s: 0.0 for s in SOURCES}
    sum_per_group = {g: {s: 0.0 for s in SOURCES} for g in GROUPS}
    for ev in window:
        pulls = _per_event_pulls(ev)
        shares = _shares(pulls)
        per_group = _per_group_shares(pulls)
        for s in SOURCES:
            sum_shares[s] += shares[s]
            if shares[s] > max_shares[s]:
                max_shares[s] = shares[s]
            for g in GROUPS:
                sum_per_group[g][s] += per_group[g][s]
    return {
        "n_steps": n,
        "shares_mean": {s: sum_shares[s] / n for s in SOURCES},
        "shares_max": dict(max_shares),
        "per_group_mean": {
            g: {s: sum_per_group[g][s] / n for s in SOURCES} for g in GROUPS
        },
    }


def _checkpoint_snapshot(events: List[Dict[str, Any]], step: int, halo: int = 5) -> Dict[str, Any]:
    """Per-source shares at the event closest to `step` (within ±halo)."""
    nearest = None
    best_delta = halo + 1
    for ev in events:
        delta = abs(int(ev.get("step", 0)) - step)
        if delta < best_delta:
            nearest = ev
            best_delta = delta
    if nearest is None or best_delta > halo:
        return {"step": step, "found": False}
    pulls = _per_event_pulls(nearest)
    return {
        "step": step,
        "found": True,
        "actual_step": int(nearest["step"]),
        "delta": best_delta,
        "shares": _shares(pulls),
        "per_group_shares": _per_group_shares(pulls),
        "totals": {s: pulls[s] for s in SOURCES},
        "total_pull": pulls["total"],
        "n_pretrain": int(nearest.get("n_pretrain", 0)),
        "n_recent": int(nearest.get("n_recent", 0)),
        "n_uniform_self": int(nearest.get("n_uniform_self", 0)),
    }


def _render_markdown(summary: Dict[str, Any], log_path: Path) -> str:
    lines = [
        "# §S181-AUDIT Wave 1 — Track B / B1 — per-source gradient attribution",
        "",
        f"Source log: `{log_path.name}` ({summary['n_events']} per_source_grad_norm events).",
        "",
        "## Decision-relevant windows (per V-B aggregation L13 guard)",
        "",
        "| window | n_steps | mean share pretrain | mean share recent | mean share uniform_self | max single-source share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, win in summary["windows"].items():
        sm = win["shares_mean"]
        smax = max(win["shares_max"].values()) if win["shares_max"] else 0.0
        lines.append(
            f"| {label} | {win['n_steps']} | "
            f"{sm.get('pretrain', 0):.3f} | "
            f"{sm.get('recent', 0):.3f} | "
            f"{sm.get('uniform_self', 0):.3f} | "
            f"{smax:.3f} |"
        )

    lines += [
        "",
        "## Per-group share within decision window (mean steps 500-2000)",
        "",
        "| group | pretrain | recent | uniform_self |",
        "|---|---:|---:|---:|",
    ]
    decision_window = summary["windows"].get("steps_500_2000", {})
    pg = decision_window.get("per_group_mean", {})
    for g in GROUPS:
        gs = pg.get(g, {})
        lines.append(
            f"| {g} | {gs.get('pretrain', 0):.3f} | "
            f"{gs.get('recent', 0):.3f} | {gs.get('uniform_self', 0):.3f} |"
        )

    lines += ["", "## Checkpoint snapshots", "",
              "| ckpt step | actual | shares pretrain | recent | uniform_self | total pull |",
              "|---:|---:|---:|---:|---:|---:|"]
    for snap in summary["checkpoints"]:
        if not snap["found"]:
            lines.append(f"| {snap['step']} | N/A | — | — | — | — |")
            continue
        s = snap["shares"]
        lines.append(
            f"| {snap['step']} | {snap['actual_step']} | "
            f"{s['pretrain']:.3f} | {s['recent']:.3f} | "
            f"{s['uniform_self']:.3f} | {snap['total_pull']:.4f} |"
        )

    lines += [
        "",
        "## V-B-A discrimination guard",
        "",
        "Routing per `audit/structural/REAL_RUN_RECIPE.md` §3 + "
        "`B_launch_and_analysis_spec.md` §Aggregation:",
        "",
        "- V-B-A if any source share ≥ 60% across steps 500-2000",
        "- V-B-B if all three sources land in 25-45% across steps 500-2000",
        "",
    ]
    win = decision_window
    if win.get("n_steps", 0) > 0:
        sm = win["shares_mean"]
        smax = max(sm.values())
        smin = min(sm.values())
        any_above_60 = smax >= 0.60
        all_in_band = (smin >= 0.25 and smax <= 0.45)
        lines += [
            f"- max source share over window: **{smax:.3f}**",
            f"- min source share over window: **{smin:.3f}**",
            f"- V-B-A literal trigger (≥0.60): **{'YES' if any_above_60 else 'NO'}**",
            f"- V-B-B literal trigger (25-45 band): **{'YES' if all_in_band else 'NO'}**",
        ]

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path,
                    help="path to B4 instrumented run log file")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "audit" / "structural" / "track_b")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = _parse_events(args.log)
    if not events:
        print(f"no per_source_grad_norm events found in {args.log}", file=sys.stderr)
        sys.exit(1)

    print(f"parsed {len(events)} per_source_grad_norm events from {args.log.name}")

    windows = {
        "steps_0_500":    _window_summary(events, 0, 500),
        "steps_500_1000": _window_summary(events, 500, 1000),
        "steps_500_2000": _window_summary(events, 500, 2000),
        "steps_1000_3000": _window_summary(events, 1000, 3000),
        "all":            _window_summary(events, 0, 10_000_000),
    }
    checkpoints = [_checkpoint_snapshot(events, s) for s in CHECKPOINT_STEPS]

    summary = {
        "log": str(args.log),
        "n_events": len(events),
        "windows": windows,
        "checkpoints": checkpoints,
    }

    out_json = args.output_dir / "B1_results.json"
    out_md = args.output_dir / "B1_results.md"
    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text(_render_markdown(summary, args.log))
    print(f"wrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()
