#!/usr/bin/env python3
"""Extract graduation-calibration metrics from calib_R[1-4] JSONL logs.

Usage:
    .venv/bin/python scripts/analyze_calibration.py \
        archive/calibration_2026-04-17/calib_R1/calib_R1.jsonl \
        archive/calibration_2026-04-17/calib_R2/calib_R2.jsonl \
        archive/calibration_2026-04-17/calib_R3/calib_R3.jsonl \
        archive/calibration_2026-04-17/calib_R4/calib_R4.jsonl \
        --db reports/eval/results.db \
        --out reports/graduation_calibration_2026-04-17.md

Produces a Markdown report with Phase 2 tables + Phase 3 cross-checks per
prompt-8 §Phase 2/§Phase 3.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Data shapes ──────────────────────────────────────────────────────────────

@dataclass
class TrainStep:
    step: int
    policy_loss: float | None = None
    value_loss: float | None = None
    chain_loss: float | None = None
    ownership_loss: float | None = None
    threat_loss: float | None = None
    policy_entropy_selfplay: float | None = None  # old name; alias for selfplay_model_entropy_batch
    selfplay_model_entropy_batch: float | None = None
    policy_target_entropy_fullsearch: float | None = None
    policy_target_entropy_fastsearch: float | None = None
    policy_target_kl_uniform_fullsearch: float | None = None
    policy_target_kl_uniform_fastsearch: float | None = None
    frac_fullsearch_in_batch: float | None = None
    draw_rate: float | None = None
    games_per_hour: float | None = None
    pretrained_weight: float | None = None


@dataclass
class Promotion:
    step: int
    wr_best: float | None
    eval_step: int | None  # value of best_model_step field at promotion time


@dataclass
class EvalRound:
    step: int
    wr_best: float | None
    wr_sealbot: float | None
    wr_random: float | None
    ci_best: tuple[float, float] | None
    promoted: bool


@dataclass
class RunSummary:
    name: str
    log_path: Path
    final_step: int = 0
    duration_min: float | None = None
    train_steps: list[TrainStep] = field(default_factory=list)
    promotions: list[Promotion] = field(default_factory=list)
    evals: list[EvalRound] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)

    # Derived
    graduation_count: int = 0
    stall_windows_ge_10k: int = 0
    longest_stall_steps: int = 0
    config: dict[str, Any] = field(default_factory=dict)


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_jsonl(path: Path) -> RunSummary:
    summary = RunSummary(name=path.stem, log_path=path)
    t0 = None
    t_last = None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            ev = e.get("event")
            ts = e.get("timestamp")
            if ts:
                if t0 is None:
                    t0 = ts
                t_last = ts

            if ev == "startup":
                summary.config = e.get("config", {}) or {}

            elif ev in ("train_step", "train_step_summary"):
                # Per-step ``train_step`` (trainer) + log_interval-cadence
                # ``train_step_summary`` (loop) split since 2026-04-19;
                # summary fields (games_per_hour, policy_entropy_selfplay,
                # policy_target_*) live on the summary variant.
                ts_step = TrainStep(step=int(e.get("step", 0) or 0))
                for k in (
                    "policy_loss","value_loss","chain_loss","ownership_loss","threat_loss",
                    "policy_entropy_selfplay", "selfplay_model_entropy_batch",
                    "policy_target_entropy_fullsearch",
                    "policy_target_entropy_fastsearch",
                    "policy_target_kl_uniform_fullsearch",
                    "policy_target_kl_uniform_fastsearch",
                    "frac_fullsearch_in_batch",
                    "draw_rate","games_per_hour","pretrained_weight",
                ):
                    if k in e:
                        try:
                            val = float(e[k])
                            if not math.isnan(val):
                                setattr(ts_step, k, val)
                        except Exception:
                            pass
                # backward compat: old JSONL only has old key
                if ts_step.selfplay_model_entropy_batch is None and ts_step.policy_entropy_selfplay is not None:
                    ts_step.selfplay_model_entropy_batch = ts_step.policy_entropy_selfplay
                summary.train_steps.append(ts_step)
                summary.final_step = max(summary.final_step, ts_step.step)

            elif ev == "best_model_promoted":
                summary.promotions.append(Promotion(
                    step=int(e.get("step", 0) or 0),
                    wr_best=e.get("wr_best"),
                    eval_step=e.get("eval_step"),
                ))

            elif ev == "evaluation_round_complete":
                ci = e.get("ci_best")
                summary.evals.append(EvalRound(
                    step=int(e.get("step", 0) or 0),
                    wr_best=e.get("wr_best"),
                    wr_sealbot=e.get("wr_sealbot"),
                    wr_random=e.get("wr_random"),
                    ci_best=tuple(ci) if isinstance(ci, (list,tuple)) and len(ci)==2 else None,
                    promoted=bool(e.get("promoted", False)),
                ))

            elif ev in ("tracemalloc_failed", "evaluation_error"):
                summary.exceptions.append(f"{ev}@step={e.get('step')}")

    # Duration
    if t0 and t_last:
        try:
            from datetime import datetime
            dt0 = datetime.fromisoformat(t0.replace("Z","+00:00"))
            dt1 = datetime.fromisoformat(t_last.replace("Z","+00:00"))
            summary.duration_min = round((dt1-dt0).total_seconds()/60.0, 2)
        except Exception:
            pass

    summary.graduation_count = len(summary.promotions)
    # Stall windows: largest gap between consecutive promotions (or between
    # first-eval step and first promotion, etc.)
    promo_steps = [0] + [p.step for p in summary.promotions]
    if summary.final_step > 0:
        promo_steps.append(summary.final_step)
    gaps = [b - a for a, b in zip(promo_steps, promo_steps[1:])]
    if gaps:
        summary.longest_stall_steps = max(gaps)
        summary.stall_windows_ge_10k = sum(1 for g in gaps if g >= 10000)

    return summary


# ── Aggregate helpers ───────────────────────────────────────────────────────

def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return round(statistics.fmean(xs), 4) if xs else None


def _window(train_steps: list[TrainStep], lo: int, hi: int) -> list[TrainStep]:
    return [t for t in train_steps if lo <= t.step < hi]


def summarize_window(steps: list[TrainStep]) -> dict[str, Any]:
    return {
        "n": len(steps),
        "mean_policy_loss":  _mean([t.policy_loss for t in steps]),
        "mean_value_loss":   _mean([t.value_loss for t in steps]),
        "mean_chain_loss":   _mean([t.chain_loss for t in steps]),
        "mean_entropy_selfplay": _mean([t.selfplay_model_entropy_batch for t in steps]),
        "mean_tgt_entropy_full": _mean([t.policy_target_entropy_fullsearch for t in steps]),
        "mean_tgt_entropy_fast": _mean([t.policy_target_entropy_fastsearch for t in steps]),
        "mean_tgt_kl_full":  _mean([t.policy_target_kl_uniform_fullsearch for t in steps]),
        "mean_tgt_kl_fast":  _mean([t.policy_target_kl_uniform_fastsearch for t in steps]),
        "mean_frac_fullsearch": _mean([t.frac_fullsearch_in_batch for t in steps]),
        "mean_draw_rate":    _mean([t.draw_rate for t in steps]),
        "mean_gph":          _mean([t.games_per_hour for t in steps]),
        "mean_pretrained_w": _mean([t.pretrained_weight for t in steps]),
    }


# ── DB lineage check ────────────────────────────────────────────────────────

def distinct_anchor_ckpt_ids(db_path: Path, run_id: str | None = None) -> list[str]:
    if not db_path.exists():
        return []
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT DISTINCT name FROM players "
                "WHERE name LIKE 'anchor_ckpt_%' "
                + ("AND run_id = ?" if run_id else "")
                , (run_id,) if run_id else ()
            )
            return [row[0] for row in cur.fetchall()]
        except Exception as exc:
            return [f"<query_failed: {exc}>"]


# ── Chain-loss sanity (§102 regression gate) ───────────────────────────────

def chain_loss_sanity(steps: list[TrainStep]) -> tuple[bool, str]:
    """Pass if chain_loss stays > 1e-4 on at least 10 of the last 100 steps."""
    tail = steps[-100:]
    vals = [t.chain_loss for t in tail if t.chain_loss is not None]
    if not vals:
        return False, "no chain_loss samples"
    nonzero = [v for v in vals if v > 1e-4]
    frac = len(nonzero)/len(vals) if vals else 0.0
    msg = f"n={len(vals)} nonzero_frac={frac:.2f} mean={statistics.fmean(vals):.4g}"
    return frac >= 0.1, msg


# ── Report rendering ────────────────────────────────────────────────────────

def render_report(runs: dict[str, RunSummary], db_path: Path) -> str:
    lines = []
    lines.append("# Graduation gate calibration — 2026-04-17")
    lines.append("")
    lines.append("## Run matrix")
    lines.append("")
    lines.append("| Run | threshold | interval | decay | min_games | dur (min) | final_step | evals | promos | stalls ≥10k | longest_gap |")
    lines.append("|-----|-----------|----------|-------|-----------|-----------|------------|-------|--------|-------------|-------------|")
    for name in ("calib_R1","calib_R2","calib_R3","calib_R4"):
        s = runs.get(name)
        if s is None:
            lines.append(f"| {name} | — | — | — | — | _missing_ | — | — | — | — | — |")
            continue
        cfg = s.config
        ei  = cfg.get("eval_interval", "?")
        ds  = cfg.get("mixing",{}).get("decay_steps","?")
        lines.append(
            f"| {name} | _see eval.yaml_ | {ei} | {ds} | _see eval.yaml_ | "
            f"{s.duration_min or '—'} | {s.final_step} | {len(s.evals)} | "
            f"{s.graduation_count} | {s.stall_windows_ge_10k} | {s.longest_stall_steps} |"
        )
    lines.append("")

    # Eval win-rate distribution
    lines.append("## Win-rate distribution at evals (wr_best)")
    lines.append("")
    lines.append("| Run | n_evals | min | median | max | promoted (y/n series) |")
    lines.append("|-----|---------|-----|--------|-----|-----------------------|")
    for name, s in runs.items():
        wrs = [er.wr_best for er in s.evals if er.wr_best is not None]
        if not wrs:
            lines.append(f"| {name} | 0 | — | — | — | — |")
            continue
        promos = "".join("✓" if er.promoted else "·" for er in s.evals)
        lines.append(
            f"| {name} | {len(wrs)} | {min(wrs):.3f} | {statistics.median(wrs):.3f} | "
            f"{max(wrs):.3f} | {promos} |"
        )
    lines.append("")

    # Trajectory windows
    lines.append("## Loss + entropy trajectories (500-step windows)")
    lines.append("")
    windows = [(0,500),(500,1500),(1500,3000),(3000,5000),(5000,8000)]
    for name, s in runs.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| window | n | pol_loss | val_loss | chain | ent_sp | tgt_E_full | tgt_E_fast | tgt_KL_full | draw | pretrn_w |")
        lines.append("|--------|---|----------|----------|-------|--------|------------|------------|-------------|------|----------|")
        for lo, hi in windows:
            win = _window(s.train_steps, lo, hi)
            if not win:
                continue
            w = summarize_window(win)
            lines.append(
                f"| {lo}-{hi} | {w['n']} | {w['mean_policy_loss']} | {w['mean_value_loss']} | "
                f"{w['mean_chain_loss']} | {w['mean_entropy_selfplay']} | "
                f"{w['mean_tgt_entropy_full']} | {w['mean_tgt_entropy_fast']} | "
                f"{w['mean_tgt_kl_full']} | {w['mean_draw_rate']} | {w['mean_pretrained_w']} |"
            )
        lines.append("")

    # Chain loss sanity
    lines.append("## Chain-loss sanity (§102 regression check)")
    lines.append("")
    for name, s in runs.items():
        ok, msg = chain_loss_sanity(s.train_steps)
        flag = "PASS" if ok else "FAIL"
        lines.append(f"- **{name}**: {flag} ({msg})")
    lines.append("")

    # Anchor lineage from DB
    lines.append("## Anchor lineage (distinct `anchor_ckpt_*` in eval DB)")
    lines.append("")
    ids = distinct_anchor_ckpt_ids(db_path)
    lines.append(f"Global DB scan: {len(ids)} distinct names — {ids}")
    lines.append("")

    # Raw promotion list
    lines.append("## Promotion events")
    lines.append("")
    for name, s in runs.items():
        lines.append(f"### {name}")
        if not s.promotions:
            lines.append("_no promotions in this run_")
            lines.append("")
            continue
        lines.append("")
        for p in s.promotions:
            lines.append(f"- step={p.step} wr_best={p.wr_best} eval_step={p.eval_step}")
        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonls", nargs="+", type=Path)
    ap.add_argument("--db", type=Path, default=Path("reports/eval/results.db"))
    ap.add_argument("--out", type=Path, default=Path("reports/graduation_calibration_2026-04-17.md"))
    args = ap.parse_args()

    runs: dict[str, RunSummary] = {}
    for p in args.jsonls:
        if not p.exists():
            print(f"WARN: missing {p}")
            continue
        s = parse_jsonl(p)
        key = s.name  # e.g. calib_R1
        runs[key] = s
        print(
            f"{key}: steps={s.final_step} evals={len(s.evals)} promos={s.graduation_count} "
            f"dur={s.duration_min}min"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_report(runs, args.db))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
