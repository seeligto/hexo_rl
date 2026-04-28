"""Render report.md + final config.yaml from the sweep result dict.

The runner produces a dict with ``host``, ``winners``, ``traces`` and the
recoverable ``cells_csv``. The reporter consumes that dict — never re-runs
bench, never reads stdout, never branches on host. Adding a new knob does
not require changes here unless its trace dicts have new fields you want
rendered (and even then the fallback ``str(d)`` keeps the report readable).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("sweep_harness needs PyYAML — pip install pyyaml") from exc

from .knobs import KNOBS, merge_dicts, param_path_to_yaml


def _fmt_num(x: Any) -> str:
    if x is None or x == "":
        return "-"
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


def render_report(result: dict[str, Any]) -> str:
    host = result["host"]
    winners = result["winners"]
    traces = result["traces"]
    fr = result.get("final_results", {})

    lines: list[str] = []
    lines.append(f"# Sweep Report — {host['host_id']} — "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if result.get("interrupted"):
        lines.append("")
        lines.append("> **PARTIAL RESULTS** — sweep interrupted by user. "
                     "Only completed knobs are shown; cells.csv has the full cell log.")
    lines.append("")
    lines.append("## Host")
    lines.append(f"- CPU threads: {host['cpu_threads']}")
    lines.append(f"- GPU: {host['gpu_name']}, {host['vram_gb']} GB"
                 + (f", sm_{host['sm_arch']}" if host.get("sm_arch") else ""))
    lines.append("")

    lines.append("## Final config")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(_winners_to_yaml(winners), sort_keys=False).rstrip())
    lines.append("```")
    lines.append("")

    lines.append("## Per-knob trace")
    for knob, trace in traces.items():
        lines.append(f"### {knob} — {KNOBS[knob]['strategy']}")
        if not trace:
            lines.append("(no iterations)")
            lines.append("")
            continue

        sample = trace[0]
        if "iter" in sample and "m1" in sample:
            lines.append("| iter | low | high | m1 | f(m1) median | iqr(m1) | m2 | f(m2) median | iqr(m2) | decision |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for t in trace:
                lines.append(
                    f"| {t.get('iter','')} | {t.get('low_in', t.get('low',''))} | "
                    f"{t.get('high_in', t.get('high',''))} | "
                    f"{t.get('m1','')} | {_fmt_num(t.get('f_m1'))} | "
                    f"{_fmt_num(t.get('iqr_m1'))} | "
                    f"{t.get('m2','')} | {_fmt_num(t.get('f_m2'))} | "
                    f"{_fmt_num(t.get('iqr_m2'))} | {t.get('decision','')} |"
                )
        elif "phase" in sample:
            lines.append("| phase | value | median pos/hr | iqr |")
            lines.append("|---|---|---|---|")
            for t in trace:
                lines.append(f"| {t['phase']} | {t['value']} | "
                             f"{_fmt_num(t.get('median'))} | {_fmt_num(t.get('iqr'))} |")
        elif "value" in sample and "median" in sample:
            lines.append("| value | median pos/hr | iqr |")
            lines.append("|---|---|---|")
            for t in trace:
                lines.append(f"| {t['value']} | {_fmt_num(t['median'])} | "
                             f"{_fmt_num(t.get('iqr'))} |")
        elif "mid" in sample:
            lines.append("| iter | mid | f(mid) | mid+1 | f(mid+1) | decision |")
            lines.append("|---|---|---|---|---|---|")
            for t in trace:
                lines.append(f"| {t['iter']} | {t['mid']} | {_fmt_num(t['f_mid'])} | "
                             f"{t['next']} | {_fmt_num(t['f_next'])} | {t['decision']} |")
        elif "fixed" in sample:
            lines.append(f"Fixed at **{sample['fixed']}** — "
                         f"{sample.get('reason') or sample.get('skipped', '')}")
        else:
            for t in trace:
                lines.append(f"- {t}")

        winner = winners.get(knob)
        if winner is not None and knob in fr:
            r = fr[knob]
            lines.append("")
            lines.append(f"**Winner:** {knob}={winner}, "
                         f"pos/hr={_fmt_num(r.median)} ± {_fmt_num(r.iqr)}.")
        elif winner is not None:
            lines.append("")
            lines.append(f"**Winner:** {knob}={winner}.")
        lines.append("")

    lines.append("## Total")
    lines.append(f"- Wall time: {result.get('wall_minutes', 0):.1f} min")
    lines.append(f"- Cells log: `{result['cells_csv']}`")
    lines.append("")
    return "\n".join(lines)


def _winners_to_yaml(winners: dict[str, Any]) -> dict[str, Any]:
    """Convert winners dict → applicable YAML using each knob's param_path."""
    overlays = []
    for k, v in winners.items():
        if k not in KNOBS:
            continue
        overlays.append(param_path_to_yaml(KNOBS[k]["param_path"], v))
    return merge_dicts(*overlays)


def write_report(result: dict[str, Any]) -> tuple[Path, Path]:
    """Write report.md and config.yaml under ``result['out_dir']``."""
    out_dir = Path(result["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    config_path = out_dir / "config.yaml"
    report_path.write_text(render_report(result))
    config_path.write_text(yaml.safe_dump(_winners_to_yaml(result["winners"]),
                                          sort_keys=False))
    return report_path, config_path
