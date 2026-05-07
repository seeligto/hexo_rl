#!/usr/bin/env python3
"""Phase B Gate 5 — generate VARIANT_SUMMARY.md from per-arm JSON outputs.

Aggregates:
  - Per-arm pretrain logs (final epoch loss, wall time)
  - Per-arm bench JSON files (NN latency, params)
  - Per-arm SealBot JSON files (WR, CI, draws)
Computes Bonferroni-corrected pairwise z-tests (B1 vs B0/B2/B3/B4).
Renders the summary table + recommendation surface.

Usage:
    python scripts/phase_b_variant_summary.py \\
        --reports reports/encoding_phase_b \\
        --pretrain-logs reports/encoding_phase_b/pretrain \\
        --out reports/encoding_phase_b/VARIANT_SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ARMS = ("B0", "B1", "B2", "B3", "B4")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def parse_pretrain_log(log_path: Path) -> dict:
    """Extract final epoch loss, wall, and step rate from a pretrain log."""
    if not log_path.exists():
        return {"missing": True, "log_path": str(log_path)}
    text = log_path.read_text()
    epoch_lines = re.findall(r"epoch_complete.*", text)
    final_metrics: dict = {}
    for line in epoch_lines[-1:]:  # last epoch only
        for key in ("loss", "policy_loss", "value_loss", "opp_reply_loss", "chain_loss"):
            # Word boundary so `loss=` doesn't match `value_loss=` /
            # `aux_opp_reply_loss=` substrings.
            m = re.search(rf"(?<![\w_]){re.escape(key)}=([\d.]+)", line)
            if m:
                final_metrics[key] = float(m.group(1))
    # Crude wall time: first → last train_step timestamp
    ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*train_step")
    ts_lines = []
    for line in text.splitlines():
        m = ts_pattern.search(line)
        if m:
            ts_lines.append(m.group(1))
    wall_min = None
    if len(ts_lines) >= 2:
        first = datetime.fromisoformat(ts_lines[0])
        last = datetime.fromisoformat(ts_lines[-1])
        wall_min = (last - first).total_seconds() / 60.0
    return {
        "final_epoch_metrics": final_metrics,
        "wall_min": wall_min,
        "n_train_steps": len(ts_lines) * 50,  # 50 per logged step
    }


def two_proportion_z(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """Two-proportion z-test (uncorrected); returns (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    if pooled <= 0 or pooled >= 1:
        return 0.0, 1.0
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p


def render_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "|" + "|".join(["-" * (w + 2) for w in widths]) + "|"
    head = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body_lines = [
        "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body_lines])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports/encoding_phase_b")
    parser.add_argument("--pretrain-logs", default="reports/encoding_phase_b/pretrain")
    parser.add_argument("--out", default="reports/encoding_phase_b/VARIANT_SUMMARY.md")
    parser.add_argument("--bonferroni-k", type=int, default=4,
                        help="Number of pairwise comparisons (default 4)")
    args = parser.parse_args()

    reports = Path(args.reports)
    logs = Path(args.pretrain_logs)
    out_path = Path(args.out)

    # Per-arm metrics aggregation.
    arm_data: dict[str, dict] = {}
    for arm in ARMS:
        # Pretrain log — try {arm}_5080.log and {arm}_laptop.log
        pretrain = parse_pretrain_log(logs / f"{arm}_5080.log")
        if pretrain.get("missing"):
            pretrain = parse_pretrain_log(logs / f"{arm}_laptop.log")
        # Bench (laptop + 5080) — host_label passed as $1 to phase_b_post_retrain.sh.
        bench_5080 = load_json(reports / f"{arm}_bench_5080.json")
        bench_laptop = load_json(reports / f"{arm}_bench_laptop.json")
        # SealBot WR
        sealbot = load_json(reports / f"{arm}_sealbot.json")
        arm_data[arm] = {
            "pretrain": pretrain,
            "bench_5080": bench_5080,
            "bench_laptop": bench_laptop,
            "sealbot": sealbot,
        }

    # Render table.
    headers = [
        "Arm", "Filters", "Blocks", "GPool",
        "Params (M)", "Final loss", "SealBot WR", "95% CI",
        "Lat b=1 5080 (ms)", "Lat b=1 laptop (ms)",
        "Lat b=64 5080 (ms)", "Pretrain wall (min)",
    ]
    arm_specs = {
        "B0": ("128", "12", "none"),
        "B1": ("128", "12", "{6,10}"),
        "B2": ("96",  "12", "{6,10}"),
        "B3": ("128", "10", "{5,8}"),
        "B4": ("160", "12", "{6,10}"),
    }
    rows = []
    for arm in ARMS:
        spec = arm_specs[arm]
        d = arm_data[arm]
        bench = d["bench_5080"] or d["bench_laptop"]
        params = (
            f"{bench['params_total'] / 1e6:.2f}" if bench else "TBD"
        )
        loss = "TBD"
        wall = "TBD"
        if not d["pretrain"].get("missing"):
            fm = d["pretrain"].get("final_epoch_metrics") or {}
            if "loss" in fm:
                loss = f"{fm['loss']:.4f}"
            if d["pretrain"].get("wall_min"):
                wall = f"{d['pretrain']['wall_min']:.1f}"
        sb = d["sealbot"]
        if sb:
            wr_str = f"{sb['win_rate']:.1%}"
            ci_str = f"[{sb['ci_95_low']:.1%}, {sb['ci_95_high']:.1%}]"
        else:
            wr_str = "TBD"
            ci_str = "TBD"

        def _lat(bench_d, batch):
            if not bench_d:
                return "TBD"
            for r in bench_d.get("latency", []):
                if r["batch"] == batch:
                    return f"{r['median_ms']:.2f}"
            return "TBD"

        rows.append([
            arm, spec[0], spec[1], spec[2], params, loss,
            wr_str, ci_str,
            _lat(d["bench_5080"], 1),
            _lat(d["bench_laptop"], 1),
            _lat(d["bench_5080"], 64),
            wall,
        ])

    table = render_table(rows, headers)

    # Pairwise z-tests (B1 vs others).
    z_rows = []
    bonf_alpha = 0.05 / max(1, args.bonferroni_k)
    for vs_arm in ("B0", "B2", "B3", "B4"):
        sb1 = arm_data["B1"]["sealbot"]
        sbv = arm_data[vs_arm]["sealbot"]
        if sb1 and sbv:
            z, p = two_proportion_z(
                sb1["win_rate"], sb1["n_games"],
                sbv["win_rate"], sbv["n_games"],
            )
            verdict = "REJECT H0" if p < bonf_alpha else "FAIL TO REJECT"
            z_rows.append([
                f"B1 vs {vs_arm}",
                f"WR(B1)={sb1['win_rate']:.1%}, WR({vs_arm})={sbv['win_rate']:.1%}",
                f"{sb1['win_rate'] - sbv['win_rate']:+.1%}",
                f"{z:.2f}", f"{p:.4f}",
                f"{verdict} (α={bonf_alpha:.4f})",
            ])
        else:
            z_rows.append([f"B1 vs {vs_arm}", "TBD", "TBD", "TBD", "TBD", "TBD"])

    z_table = render_table(
        z_rows,
        ["Test", "WRs", "Δ WR", "z", "p", "Verdict"],
    )

    # Render summary doc.
    doc = f"""# Phase B Variant Summary — v8 encoding architecture exploration

**Sprint:** §167  **Branch:** `encoding/phase_b_variants`  **Generated:** {datetime.now(UTC).isoformat()}

**Corpus:** `data/bootstrap_corpus_v8.npz` — 347,142 positions
(347142, 11, 25, 25) fp16, 5.4 GB, 6,259 unique games. Stone-clip rate
~6% per scatter attempt (above S1's 1% Path α-trigger; consistent
across all arms).

## 1. Per-arm metrics

{table}

(`v7full` reference: 3.1573 final total loss, 17.4% SealBot WR n=500
with MCTS sims=128.)

## 2. Pairwise z-tests (Bonferroni α = {bonf_alpha:.4f}, k={args.bonferroni_k})

{z_table}

## 3. Methodological deviations from §167 plan

- **SealBot WR uses policy-argmax, not MCTS sims=128**. Engine Rust MCTS
  hardcodes BOARD_SIZE=19 / feature_len=2888 (`engine/src/lib.rs:632`,
  `replay_buffer/sym_tables.rs:23-26`). v8-aware MCTS is Phase D §168.
  Cross-arm ranking valid; absolute WR will be lower than v7full's 17.4%.
- **Threat probe deferred**. v8 fixture regen + probe v8 awareness is
  ~5 hr work outside Phase B scope. Operator decide whether to fund now
  or defer to §168.
- **MCTS sim/s + worker pos/hr skipped**. Same engine v6-only reason.
- **B3 gpool indices**: re-derived from `{{6, 10}}` to `{{5, 8}}` since
  index 10 is OOB on a 10-block trunk; preserves KataGo b10c128's
  ~50%/~80% depth-fraction pattern.
- **Off-board logit bias**: lowered from KataGo's −5000 to −50 to keep
  label-smoothing cross-entropy bounded (−5000 added ~165 to init loss).

## 4. Recommendation (TBD until metrics in)

(Filled by operator review after data gathered.)

## 5. v8 bench-gate recalibration proposal (TBD)

Old v6 gate calibrated on 8-plane × 19×19 + K-cluster; v8 needs new
targets derived from canonical pick. Will propose specific numbers
once canonical pick is decided.

## 6. Decision surface (Gate 6)

a) Accept canonical variant pick (TBD)?
b) Accept v8 bench-gate recalibration?
c) Threat probe v8 awareness — fund now or defer to §168?
d) Open §168 Phase D self-play encoding sprint, or pause?
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    print(f"wrote {out_path}", file=sys.stderr)
    print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
