#!/usr/bin/env python3
"""Extract sweep metrics from JSONL logs and produce results.csv + summary.md.

Usage:
    python scripts/sweep_extract.py \
        --log-dir logs/sweep \
        --config-dir /tmp/sweep_configs \
        --output reports/sweep_2026-04-08/results.csv \
        --summary reports/sweep_2026-04-08/summary.md \
        --warmup-sec 90
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Timestamp parsing ────────────────────────────────────────────────────────

def parse_ts(ts_str: str) -> float:
    """Parse ISO timestamp to epoch seconds."""
    # structlog format: "2026-04-06T22:03:22.895567Z"
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return dt.timestamp()


# ── Config parsing ───────────────────────────────────────────────────────────

def load_run_config(config_dir: Path, run_name: str) -> dict[str, Any]:
    """Load the override YAML for a run to get config knobs."""
    import yaml
    config_path = config_dir / f"{run_name}.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def extract_config_knobs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract sweep knobs from override config."""
    sp = cfg.get("selfplay", {})
    return {
        "ratio": cfg.get("training_steps_per_game", "?"),
        "burst": cfg.get("max_train_burst", "?"),
        "game_moves": sp.get("max_game_moves", "?"),
        "wait_ms": sp.get("inference_max_wait_ms", "?"),
        "leaf_bs": sp.get("leaf_batch_size", "?"),
        "inf_bs": sp.get("inference_batch_size", 64),
        "workers": sp.get("n_workers", 14),
        "gumbel_m": sp.get("gumbel_m", ""),
    }


# ── Per-run metric extraction ───────────────────────────────────────────────

def percentile(data: list[float], pct: float) -> float:
    """Compute percentile (0-100) from sorted data."""
    if not data:
        return float("nan")
    s = sorted(data)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def linregress_slope(xs: list[float], ys: list[float]) -> float:
    """Simple linear regression slope (avoids scipy dependency)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if ss_xx == 0:
        return float("nan")
    return ss_xy / ss_xx


def extract_metrics(jsonl_path: Path, warmup_sec: float, max_game_moves: int) -> dict[str, Any] | None:
    """Parse a run's JSONL and compute all 14 metrics (warmup excluded)."""
    events: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Separate event types
    train_steps = [e for e in events if e.get("event") == "train_step"]
    game_completes = [e for e in events if e.get("event") == "game_complete"]
    gpu_stats = [e for e in events if e.get("event") == "gpu_stats"]

    if len(train_steps) < 2:
        return None

    # Find t0 (first train_step timestamp) and apply warmup cutoff
    t0 = parse_ts(train_steps[0]["timestamp"])
    cutoff = t0 + warmup_sec

    train_steps = [e for e in train_steps if parse_ts(e["timestamp"]) >= cutoff]
    game_completes = [e for e in game_completes if parse_ts(e["timestamp"]) >= cutoff]
    gpu_stats = [e for e in gpu_stats if parse_ts(e["timestamp"]) >= cutoff]

    if len(train_steps) < 2:
        return None

    # Time window
    t_start = parse_ts(train_steps[0]["timestamp"])
    t_end = parse_ts(train_steps[-1]["timestamp"])
    window_hrs = max((t_end - t_start) / 3600.0, 1e-6)

    # 1. games_per_hour
    n_games = len(game_completes)
    games_per_hour = n_games / window_hrs

    # 2. steps_per_hour
    first_step = train_steps[0].get("step", 0)
    last_step = train_steps[-1].get("step", 0)
    steps_per_hour = (last_step - first_step) / window_hrs

    # 3. draw_rate
    n_draws = sum(1 for g in game_completes if g.get("winner") == "draw")
    draw_rate = n_draws / max(n_games, 1)

    # 4-5. game_length percentiles
    game_lengths = [g.get("game_length", 0) for g in game_completes]
    game_length_p50 = percentile(game_lengths, 50) if game_lengths else float("nan")
    game_length_p95 = percentile(game_lengths, 95) if game_lengths else float("nan")

    # 6. frac_games_at_cap
    n_at_cap = sum(1 for gl in game_lengths if gl >= max_game_moves)
    frac_at_cap = n_at_cap / max(n_games, 1)

    # 7-8. gpu_util percentiles
    gpu_utils = [g.get("gpu_util_pct", 0.0) for g in gpu_stats]
    gpu_util_median = statistics.median(gpu_utils) if gpu_utils else float("nan")
    gpu_util_p10 = percentile(gpu_utils, 10) if gpu_utils else float("nan")

    # 9. gpu_train_frac — estimate from train_step inter-arrival times
    ts_list = [parse_ts(e["timestamp"]) for e in train_steps]
    gaps = [ts_list[i + 1] - ts_list[i] for i in range(len(ts_list) - 1)]
    # When training is active, consecutive log.info("train_step") events are close together
    # (< 0.5s apart = training; > 0.5s = waiting for games)
    training_time = sum(g for g in gaps if g < 0.5)
    total_time = max(t_end - t_start, 1e-6)
    gpu_train_frac = training_time / total_time

    # 10. batch_fill_pct_median (from new monitoring field)
    bf_vals = [e.get("batch_fill_pct") for e in train_steps if e.get("batch_fill_pct") is not None]
    batch_fill_pct_median = statistics.median(bf_vals) if bf_vals else float("nan")

    # 11. inference_calls_per_move (from new monitoring fields)
    # Only summary events (every log_interval steps) carry inf_forward_count.
    # Use first and last summary events for a wide delta window.
    summary_steps = [e for e in train_steps if e.get("inf_forward_count") is not None]
    if len(summary_steps) >= 2:
        fwd_first = summary_steps[0]["inf_forward_count"]
        fwd_last = summary_steps[-1]["inf_forward_count"]
        games_first = summary_steps[0].get("games_played", 0)
        games_last = summary_steps[-1].get("games_played", 0)
        avg_gl = statistics.mean(game_lengths) if game_lengths else 0
        delta_games = max(games_last - games_first, 1)
        total_moves = delta_games * avg_gl if avg_gl > 0 else 1
        delta_fwd = fwd_last - fwd_first
        inference_calls_per_move = delta_fwd / total_moves if total_moves > 0 else float("nan")
    else:
        inference_calls_per_move = float("nan")

    # 12. value_loss_slope
    steps_x = [float(e.get("step", 0)) for e in train_steps]
    value_losses = [float(e.get("value_loss", 0.0)) for e in train_steps]
    value_loss_slope = linregress_slope(steps_x, value_losses)

    # 13. policy_entropy_mean + slope
    policy_entropies = [float(e.get("policy_entropy", 0.0)) for e in train_steps]
    policy_entropy_mean = statistics.mean(policy_entropies) if policy_entropies else float("nan")
    policy_entropy_slope = linregress_slope(steps_x, policy_entropies)

    # 14. policy_entropy_floor
    policy_entropy_floor = min(policy_entropies) if policy_entropies else float("nan")

    # + total_loss_max (spike detector)
    total_losses = [float(e.get("total_loss", 0.0)) for e in train_steps]
    total_loss_max = max(total_losses) if total_losses else float("nan")
    total_loss_median = statistics.median(total_losses) if total_losses else float("nan")

    # Sanity flags
    flags = []
    if not math.isnan(policy_entropy_floor) and policy_entropy_floor < 1.5:
        flags.append("LOW_ENTROPY")
    if not math.isnan(value_loss_slope) and value_loss_slope > 0:
        flags.append("VALUE_DIVERGING")
    if not math.isnan(total_loss_max) and not math.isnan(total_loss_median) and total_loss_median > 0:
        if total_loss_max > 2 * total_loss_median:
            flags.append("LOSS_SPIKE")
    if frac_at_cap > 0.7:
        flags.append("DRAW_CAP")

    return {
        "games_per_hour": round(games_per_hour, 1),
        "steps_per_hour": round(steps_per_hour, 1),
        "draw_rate": round(draw_rate, 4),
        "game_length_p50": round(game_length_p50, 1),
        "game_length_p95": round(game_length_p95, 1),
        "frac_games_at_cap": round(frac_at_cap, 4),
        "gpu_util_median": round(gpu_util_median, 1),
        "gpu_util_p10": round(gpu_util_p10, 1),
        "gpu_train_frac": round(gpu_train_frac, 4),
        "batch_fill_pct_median": round(batch_fill_pct_median, 1) if not math.isnan(batch_fill_pct_median) else "",
        "inference_calls_per_move": round(inference_calls_per_move, 2) if not math.isnan(inference_calls_per_move) else "",
        "value_loss_slope": f"{value_loss_slope:.6f}" if not math.isnan(value_loss_slope) else "",
        "policy_entropy_mean": round(policy_entropy_mean, 4) if not math.isnan(policy_entropy_mean) else "",
        "policy_entropy_slope": f"{policy_entropy_slope:.6f}" if not math.isnan(policy_entropy_slope) else "",
        "policy_entropy_floor": round(policy_entropy_floor, 4) if not math.isnan(policy_entropy_floor) else "",
        "total_loss_max": round(total_loss_max, 4) if not math.isnan(total_loss_max) else "",
        "sanity_flags": "; ".join(flags) if flags else "OK",
    }


# ── Summary generation ───────────────────────────────────────────────────────

def generate_summary(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write summary.md with per-arm rankings and interpretation."""
    puct_runs = [r for r in results if r["arm"] == "PUCT"]
    gumbel_runs = [r for r in results if r["arm"] == "Gumbel"]

    # Sort by steps_per_hour descending
    puct_runs.sort(key=lambda r: float(r.get("steps_per_hour", 0)), reverse=True)
    gumbel_runs.sort(key=lambda r: float(r.get("steps_per_hour", 0)), reverse=True)

    lines: list[str] = []
    lines.append("# Config Sweep Results — 2026-04-08\n")
    lines.append("## Hardware")
    lines.append("- CPU: Ryzen 7 8845HS")
    lines.append("- GPU: RTX 4060 Laptop (8GB VRAM)")
    lines.append("- Run duration: 20 min wall-clock (90s warm-up excluded)")
    lines.append("- All runs: completed_q_values=true, fresh from bootstrap_model.pt\n")

    # PUCT arm table
    lines.append("## PUCT Arm Rankings (by steps_per_hour)\n")
    lines.append("| Rank | Run | ratio | burst | game_moves | wait_ms | leaf_bs | inf_bs | wkrs | steps/hr | games/hr | gpu_train% | batch_fill% | calls/move | val_slope | ent_floor | flags |")
    lines.append("|------|-----|-------|-------|------------|---------|---------|--------|------|----------|----------|------------|-------------|------------|-----------|-----------|-------|")
    for i, r in enumerate(puct_runs, 1):
        lines.append(
            f"| {i} | {r['run_id']} | {r['ratio']} | {r['burst']} | {r['game_moves']} | "
            f"{r['wait_ms']} | {r['leaf_bs']} | {r['inf_bs']} | {r['workers']} | "
            f"{r['steps_per_hour']} | {r['games_per_hour']} | "
            f"{r['gpu_train_frac']} | {r['batch_fill_pct_median']} | "
            f"{r['inference_calls_per_move']} | {r['value_loss_slope']} | "
            f"{r['policy_entropy_floor']} | {r['sanity_flags']} |"
        )
    lines.append("")

    # Gumbel arm table
    if gumbel_runs:
        lines.append("## Gumbel Arm Rankings (by steps_per_hour)\n")
        lines.append("| Rank | Run | m | ratio | burst | game_moves | wait_ms | leaf_bs | wkrs | steps/hr | games/hr | gpu_train% | batch_fill% | calls/move | val_slope | ent_floor | flags |")
        lines.append("|------|-----|---|-------|-------|------------|---------|---------|------|----------|----------|------------|-------------|------------|-----------|-----------|-------|")
        for i, r in enumerate(gumbel_runs, 1):
            lines.append(
                f"| {i} | {r['run_id']} | {r['gumbel_m']} | {r['ratio']} | {r['burst']} | "
                f"{r['game_moves']} | {r['wait_ms']} | {r['leaf_bs']} | {r['workers']} | "
                f"{r['steps_per_hour']} | {r['games_per_hour']} | "
                f"{r['gpu_train_frac']} | {r['batch_fill_pct_median']} | "
                f"{r['inference_calls_per_move']} | {r['value_loss_slope']} | "
                f"{r['policy_entropy_floor']} | {r['sanity_flags']} |"
            )
        lines.append("")

    # Recommendations placeholder
    lines.append("## Interpretation\n")
    lines.append("_To be filled after reviewing the data above._\n")
    lines.append("Key questions:")
    lines.append("- Did ratio 4→6 improve steps/hr or just increase burst waste?")
    lines.append("- Did leaf_bs 8→16 halve inference_calls_per_move?")
    lines.append("- Did wait_ms 12→4 improve batch_fill without starving workers?")
    lines.append("- Did inf_bs 64→32 (P6) actually fill batches more consistently?")
    lines.append("- Did workers 14→18 (P8) help or just add contention?")
    lines.append("- Did gumbel_m 16→8 (G2) reduce per-game cost enough to boost throughput?\n")

    lines.append("## Recommended Configs for Overnight Run\n")
    if puct_runs:
        best_p = puct_runs[0]
        lines.append(f"**PUCT arm:** {best_p['run_id']} — {best_p['steps_per_hour']} steps/hr, "
                      f"{best_p['games_per_hour']} games/hr")
    if gumbel_runs:
        best_g = gumbel_runs[0]
        lines.append(f"**Gumbel arm:** {best_g['run_id']} — {best_g['steps_per_hour']} steps/hr, "
                      f"{best_g['games_per_hour']} games/hr")
    lines.append("")
    lines.append("**Caveat:** 20-min runs do NOT measure absolute strength. Quality verdict ")
    lines.append("requires overnight Bradley-Terry eval of the winner against Checkpoint_0.")
    lines.append("Do NOT pick an overall winner across arms — PUCT vs Gumbel is a research ")
    lines.append("bet that belongs in a separate overnight comparison.\n")

    output_path.write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sweep metrics from JSONL logs")
    parser.add_argument("--log-dir", required=True, help="Directory containing sweep_*.jsonl logs")
    parser.add_argument("--config-dir", required=True, help="Directory containing per-run override YAMLs")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--summary", required=True, help="Output summary markdown path")
    parser.add_argument("--warmup-sec", type=float, default=90.0, help="Warm-up seconds to exclude")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    config_dir = Path(args.config_dir)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    # Discover runs from JSONL files
    jsonl_files = sorted(log_dir.glob("sweep_*.jsonl"))
    if not jsonl_files:
        print(f"No sweep_*.jsonl files found in {log_dir}")
        return

    # Determine arm from filename: P* = PUCT, G* = Gumbel
    results: list[dict[str, Any]] = []

    for jf in jsonl_files:
        # Extract run name: sweep_P0.jsonl -> P0
        run_name = jf.stem.replace("sweep_", "")
        arm = "PUCT" if run_name.startswith("P") else "Gumbel"

        cfg = load_run_config(config_dir, run_name)
        knobs = extract_config_knobs(cfg)
        max_game_moves = int(cfg.get("selfplay", {}).get("max_game_moves", 200))

        metrics = extract_metrics(jf, args.warmup_sec, max_game_moves)
        if metrics is None:
            print(f"WARNING: {run_name} — insufficient data, skipping")
            continue

        row = {
            "run_id": run_name,
            "arm": arm,
            **knobs,
            **metrics,
        }
        results.append(row)
        print(f"  {run_name}: {metrics['steps_per_hour']} steps/hr, "
              f"{metrics['games_per_hour']} games/hr, flags={metrics['sanity_flags']}")

    if not results:
        print("No valid results to write.")
        return

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV written: {output_path} ({len(results)} runs)")

    # Write summary
    generate_summary(results, summary_path)
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
