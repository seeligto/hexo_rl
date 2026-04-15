"""Phase 2 GPU-util sweep analysis. Reads per-run train.jsonl + dmon.log and
computes the metrics listed in the prompt over the last 15 min of each 20-min
window. Emits results.md."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path("archive/sweep_2026-04-13_gpu_util")
RUNS = [
    ("a", "baseline (inf_bs=64, wait=4ms)"),
    ("b", "H1 (inf_bs=128, wait=8ms)"),
    ("c", "isolate (inf_bs=128, wait=4ms)"),
]
MEASURE_SEC = 15 * 60  # last 15 min of the 20-min window


def parse_ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_train_events(path: Path):
    out = []
    for line in path.open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == "train_step" and d.get("inf_forward_count") is not None:
            d["_ts"] = parse_ts(d["timestamp"])
            out.append(d)
    return out


def load_game_events(path: Path):
    out = []
    for line in path.open():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == "game_complete":
            d["_ts"] = parse_ts(d["timestamp"])
            out.append(d)
    return out


def load_dmon(path: Path):
    """Return list of (sm_util, ts_rel) from nvidia-smi dmon -o T output."""
    samples = []
    first_ts = None
    for line in path.open():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # with -o T: HH:MM:SS gpu sm mem enc dec jpg ofa
        if len(parts) < 3:
            continue
        # first token is time-of-day
        tok = parts[0]
        try:
            hh, mm, ss = tok.split(":")
            secs = int(hh) * 3600 + int(mm) * 60 + int(ss)
        except ValueError:
            continue
        try:
            sm = int(parts[2])
        except (IndexError, ValueError):
            continue
        if first_ts is None:
            first_ts = secs
        samples.append((secs - first_ts, sm))
    return samples


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = max(0, min(len(s) - 1, int(round((q / 100) * (len(s) - 1)))))
    return s[i]


def analyze_run(run_id: str):
    run_dir = ROOT / f"run_{run_id}"
    events = load_train_events(run_dir / "train.jsonl")
    games = load_game_events(run_dir / "train.jsonl")
    if not events:
        return None

    t_first = events[0]["_ts"]
    t_last = events[-1]["_ts"]
    window_start = t_last - MEASURE_SEC
    # Measurement = last 15 min of the run. First 5 min = warm-up.
    win = [e for e in events if e["_ts"] >= window_start]
    if len(win) < 2:
        return None
    win_games = [g for g in games if g["_ts"] >= window_start]

    e0, e1 = win[0], win[-1]
    dt = e1["_ts"] - e0["_ts"]

    d_fwd = e1["inf_forward_count"] - e0["inf_forward_count"]
    d_req = e1["inf_total_requests"] - e0["inf_total_requests"]
    d_buf = e1["buffer_size"] - e0["buffer_size"]
    d_games = e1["games_played"] - e0["games_played"]
    d_step = e1["step"] - e0["step"]

    # Rates
    fwd_per_sec = d_fwd / dt if dt else 0
    pos_per_sec_nn = d_req / dt if dt else 0  # NN positions/sec (forwards × batch)
    pos_per_hr = d_buf * 3600 / dt if dt else 0  # buffer delta (matches §69 convention)
    games_per_hr = d_games * 3600 / dt if dt else 0
    mean_batch = d_req / d_fwd if d_fwd else 0

    fill_vals = [e.get("batch_fill_pct") for e in win if e.get("batch_fill_pct") is not None]
    mean_fill = sum(fill_vals) / len(fill_vals) if fill_vals else 0

    # Policy entropy (selfplay) from last event
    ent_sp_final = e1.get("policy_entropy_selfplay")
    ent_combined_final = e1.get("policy_entropy")
    ent_sp_min = min(
        (e.get("policy_entropy_selfplay", 99) for e in win if e.get("policy_entropy_selfplay") is not None),
        default=None,
    )

    # GPU utilization from dmon — slice to last 15 min of its own timeline.
    dmon_samples = load_dmon(run_dir / "dmon.log")
    if dmon_samples:
        t_max = dmon_samples[-1][0]
        dmon_win = [sm for (t_rel, sm) in dmon_samples if t_rel >= t_max - MEASURE_SEC]
    else:
        dmon_win = []
    gpu_mean = sum(dmon_win) / len(dmon_win) if dmon_win else None
    gpu_p10 = pct(dmon_win, 10) if dmon_win else None
    gpu_p90 = pct(dmon_win, 90) if dmon_win else None

    return {
        "run": run_id,
        "window_sec": dt,
        "events_in_window": len(win),
        "games_in_window": d_games,
        "steps_in_window": d_step,
        "games_per_hr": games_per_hr,
        "pos_per_hr": pos_per_hr,
        "fwd_per_sec": fwd_per_sec,
        "mean_batch": mean_batch,
        "nn_pos_per_sec": pos_per_sec_nn,
        "batch_fill_mean": mean_fill,
        "gpu_util_mean": gpu_mean,
        "gpu_util_p10": gpu_p10,
        "gpu_util_p90": gpu_p90,
        "policy_entropy_selfplay_final": ent_sp_final,
        "policy_entropy_selfplay_min": ent_sp_min,
        "policy_entropy_combined_final": ent_combined_final,
        "game_len_median": median([g.get("game_length", g.get("plies", 0)) for g in win_games]) if win_games else None,
    }


def fmt(v, sp=".1f"):
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:{sp}}"
    return str(v)


def main():
    results = {}
    for run_id, _ in RUNS:
        r = analyze_run(run_id)
        if r:
            results[run_id] = r

    for run_id, label in RUNS:
        r = results.get(run_id)
        if not r:
            print(f"run_{run_id}: no data")
            continue
        print(f"\n=== Run {run_id.upper()} — {label} ===")
        print(f"  window        : {r['window_sec']:.0f}s ({r['events_in_window']} train_step events)")
        print(f"  games_in_win  : {r['games_in_window']}")
        print(f"  steps_in_win  : {r['steps_in_window']}")
        print(f"  games/hr      : {r['games_per_hr']:.0f}")
        print(f"  pos/hr (buf)  : {r['pos_per_hr']:.0f}")
        print(f"  fwd/sec       : {r['fwd_per_sec']:.1f}")
        print(f"  mean_batch    : {r['mean_batch']:.2f}")
        print(f"  nn_pos/sec    : {r['nn_pos_per_sec']:.0f}")
        print(f"  batch_fill%   : {r['batch_fill_mean']:.1f}")
        gpu = r["gpu_util_mean"]
        print(f"  gpu_util mean : {fmt(gpu, '.1f')}")
        print(f"  gpu_util p10  : {fmt(r['gpu_util_p10'], '.0f')}")
        print(f"  gpu_util p90  : {fmt(r['gpu_util_p90'], '.0f')}")
        print(f"  ent_sp final  : {r['policy_entropy_selfplay_final']:.3f}")
        print(f"  ent_sp min    : {r['policy_entropy_selfplay_min']:.3f}")
        print(f"  ent_comb final: {r['policy_entropy_combined_final']:.3f}")
        print(f"  game_len_med  : {r['game_len_median']}")

    # Write results.md
    a = results.get("a")
    b = results.get("b")
    c = results.get("c")
    lines = []
    lines.append("# Phase 2 GPU Util Sweep — 2026-04-13\n")
    lines.append(
        "3-run narrowed sweep testing **H1: raise inference batch coalescing**. "
        "All runs 20 min from `bootstrap_model.pt`, `gumbel_targets` variant, "
        "laptop (Ryzen 7 8845HS + RTX 4060). Measurement window = last 15 min. "
        "See prompt in conversation log and `/tmp/gpu_util_phase1.md` for Phase 1 context.\n"
    )
    lines.append("## Config (held constant)\n")
    lines.append("- variant: `gumbel_targets` (gumbel_mcts=false, completed_q_values=true)")
    lines.append("- `standard_sims: 200`, `training_steps_per_game: 4`, `max_train_burst: 16`")
    lines.append("- `n_workers: 14`, `leaf_batch_size: 8`, `fast_prob: 0.0`")
    lines.append("- `dirichlet_enabled: true`")
    lines.append("- fresh `bootstrap_model.pt`, `mixing.buffer_persist: false`\n")
    lines.append("## Config (varied)\n")
    lines.append("| run | inference_batch_size | inference_max_wait_ms |")
    lines.append("|-----|---------------------:|----------------------:|")
    lines.append("| A   | 64                   | 4.0                   |")
    lines.append("| B   | 128                  | 8.0                   |")
    lines.append("| C   | 128                  | 4.0                   |\n")
    lines.append("## Results\n")
    lines.append(
        "Metrics computed over the **last 15 min** of each 20-min window "
        "(first 5 min discarded as warm-up, beyond the CUDA warmup already done at worker-pool start).\n"
    )
    lines.append(
        "| metric | Run A | Run B | Run C | B vs A | C vs A |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    def row(label, key, fmt_spec=".1f", higher_better=True):
        va = a[key] if a else None
        vb = b[key] if b else None
        vc = c[key] if c else None
        def _f(v):
            return "—" if v is None else f"{v:{fmt_spec}}"
        def _d(v, base):
            if v is None or base is None or base == 0:
                return "—"
            d = (v - base) / base * 100
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.1f}%"
        return f"| {label} | {_f(va)} | {_f(vb)} | {_f(vc)} | {_d(vb, va)} | {_d(vc, va)} |"

    lines.append(row("games/hr", "games_per_hr", ".0f"))
    lines.append(row("pos/hr (buffer delta)", "pos_per_hr", ".0f"))
    lines.append(row("nn_forwards/sec", "fwd_per_sec", ".1f"))
    lines.append(row("nn_mean_batch_size", "mean_batch", ".2f"))
    lines.append(row("nn_pos/sec (fwd × batch)", "nn_pos_per_sec", ".0f"))
    lines.append(row("batch_fill_pct (mean)", "batch_fill_mean", ".1f"))
    lines.append(row("gpu_util_mean (dmon)", "gpu_util_mean", ".1f"))
    lines.append(row("gpu_util_p10 (dmon)", "gpu_util_p10", ".0f"))
    lines.append(row("gpu_util_p90 (dmon)", "gpu_util_p90", ".0f"))
    lines.append(row("policy_entropy_selfplay (final)", "policy_entropy_selfplay_final", ".3f"))
    lines.append(row("policy_entropy_selfplay (min)", "policy_entropy_selfplay_min", ".3f"))
    lines.append(row("policy_entropy combined (final)", "policy_entropy_combined_final", ".3f"))
    lines.append(row("steps in window", "steps_in_window", ".0f"))
    lines.append(row("games in window", "games_in_window", ".0f"))
    lines.append("")

    # Kill criterion
    lines.append("## Kill criterion check\n")
    lines.append(
        "Per Phase 1 correction (`/tmp/gpu_util_phase1.md`): `policy_entropy_selfplay` "
        "must remain ≥ 4.0 nats throughout the window; combined entropy is not a "
        "collapse signal at this training stage.\n"
    )
    for rid in "abc":
        r = results.get(rid)
        if not r:
            continue
        emin = r["policy_entropy_selfplay_min"]
        ok = emin is not None and emin >= 4.0
        status = "PASS" if ok else "**FAIL**"
        lines.append(f"- Run {rid.upper()}: min(policy_entropy_selfplay) = {emin:.3f} — {status}")
    lines.append("")

    # Winner
    lines.append("## Winner\n")
    if a and b and c:
        # pos/hr deltas
        dB = (b["pos_per_hr"] - a["pos_per_hr"]) / a["pos_per_hr"] * 100
        dC = (c["pos_per_hr"] - a["pos_per_hr"]) / a["pos_per_hr"] * 100
        entropy_safe = {rid: (results[rid]["policy_entropy_selfplay_min"] or 0) >= 4.0 for rid in "abc"}
        winner_pos_hr = max(("a", a["pos_per_hr"]), ("b", b["pos_per_hr"]), ("c", c["pos_per_hr"]), key=lambda x: x[1])
        best_non_a = max(("b", dB, entropy_safe["b"]), ("c", dC, entropy_safe["c"]), key=lambda x: (x[2], x[1]))
        lines.append(f"- Run B pos/hr delta vs A: {dB:+.1f}% (entropy safe: {entropy_safe['b']})")
        lines.append(f"- Run C pos/hr delta vs A: {dC:+.1f}% (entropy safe: {entropy_safe['c']})")
        if not all(entropy_safe.values()):
            lines.append("- **Kill criterion eliminated one or more runs.**")
        if best_non_a[1] > 5.0 and best_non_a[2]:
            lines.append(f"- **Winner: Run {best_non_a[0].upper()}** (+{best_non_a[1]:.1f}% pos/hr vs A, entropy safe)")
        else:
            lines.append(
                "- **No winner beats Run A by ≥5% pos/hr with entropy safe.** "
                "Config is already near-optimal on the inf_bs/wait_ms axis."
            )
    lines.append("")

    (ROOT / "results.md").write_text("\n".join(lines) + "\n")
    print("\nwrote", ROOT / "results.md")


if __name__ == "__main__":
    main()
