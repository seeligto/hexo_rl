"""§S181-AUDIT Wave 1 Track B / B2 — buffer composition analysis.

Consumes `buffer_position_class_snapshot` events from the per-run events
JSONL sink (`logs/events_<run_id>.jsonl`) and produces the
colony/extension/neither class-fraction trajectory per
`audit/structural/track_b/B_launch_and_analysis_spec.md` §"B2 — buffer
composition".

V-B-C verdict: feedback loop confirmed if `colony_frac` crosses 50% by
step 2000.

The events JSONL sink carries the full payload (buffer_size, n_sampled,
colony_n, mean value targets) — the structlog file only has a 4-field
summary. Use `--events-jsonl` for the full picture; `--log` is accepted
as a fallback that reads the structlog stream.

Usage:
    python -m scripts.structural_diagnosis.track_b.b2_buffer_composition_analysis \\
        --events-jsonl /path/to/logs/events_<run_id>.jsonl \\
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


def _parse_snapshots(log_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "buffer_position_class_snapshot":
                continue
            out.append(rec)
    return sorted(out, key=lambda r: int(r.get("step", 0)))


def _render_markdown(snapshots: List[Dict[str, Any]], log_path: Path) -> str:
    lines = [
        "# §S181-AUDIT Wave 1 — Track B / B2 — buffer composition",
        "",
        f"Source log: `{log_path.name}` ({len(snapshots)} snapshots).",
        "",
        "## Position-class trajectory",
        "",
        "| step | buffer size | n sampled | colony frac | extension frac | neither frac | "
        "colony mean v_target | extension mean v_target | neither mean v_target |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in snapshots:
        lines.append(
            f"| {s['step']} | {s.get('buffer_size', 0)} | {s.get('n_sampled', 0)} | "
            f"{s.get('colony_frac', 0.0):.4f} | "
            f"{s.get('extension_frac', 0.0):.4f} | "
            f"{s.get('neither_frac', 0.0):.4f} | "
            f"{_fmt(s.get('colony_mean_value_target'))} | "
            f"{_fmt(s.get('extension_mean_value_target'))} | "
            f"{_fmt(s.get('neither_mean_value_target'))} |"
        )

    lines += [
        "",
        "## V-B-C verdict — feedback loop guard",
        "",
        "From `B_launch_and_analysis_spec.md` §Aggregation: V-B-C fires "
        "if `colony_frac` > 0.50 by step 2000.",
        "",
    ]
    by_step = {int(s["step"]): s for s in snapshots}
    for target_step in (500, 1000, 1500, 2000, 2500, 3000):
        snap = by_step.get(target_step)
        if snap is None:
            lines.append(f"- step {target_step}: no snapshot")
            continue
        cf = snap.get("colony_frac", 0.0)
        marker = "**FIRED**" if cf > 0.50 and target_step <= 2000 else ""
        lines.append(
            f"- step {target_step}: colony_frac = {cf:.4f} "
            f"({snap.get('colony_n', 0)}/{snap.get('n_sampled', 0)}) {marker}"
        )

    # Synthesise a literal yes/no.
    vbc_fires = any(
        (int(s["step"]) <= 2000 and s.get("colony_frac", 0.0) > 0.50)
        for s in snapshots
    )
    lines += [
        "",
        f"**V-B-C trigger (literal):** {'YES' if vbc_fires else 'NO'}",
        "",
    ]
    return "\n".join(lines) + "\n"


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):+.4f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--events-jsonl", type=Path,
                     help="events_<run_id>.jsonl with full snapshot payloads (preferred)")
    src.add_argument("--log", type=Path,
                     help="structlog file with minimal snapshot fields (fallback)")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO / "audit" / "structural" / "track_b")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.events_jsonl if args.events_jsonl is not None else args.log
    snapshots = _parse_snapshots(source_path)
    if not snapshots:
        print(f"no buffer_position_class_snapshot events found in {source_path}",
              file=sys.stderr)
        sys.exit(1)

    print(f"parsed {len(snapshots)} buffer snapshots from {source_path.name}")
    summary = {
        "source": str(source_path),
        "n_snapshots": len(snapshots),
        "snapshots": snapshots,
    }
    out_json = args.output_dir / "B2_results.json"
    out_md = args.output_dir / "B2_results.md"
    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text(_render_markdown(snapshots, source_path))
    print(f"wrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()
