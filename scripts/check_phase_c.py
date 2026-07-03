#!/usr/bin/env python3
"""Phase C long-run monitor — parse longrun_c2.log over SSH and display stats.

Usage:
    python scripts/check_phase_c.py
    python scripts/check_phase_c.py --tail 5000   # only parse last N lines (faster)
"""

import json
import math
import subprocess
import sys
from datetime import datetime, timezone

SSH_ARGS = ["-p", "13053", "-i", "~/.ssh/vast_hexo", "root@ssh6.vast.ai"]
LOG_PATH = "/workspace/hexo_rl/logs/longrun_c2.log"
TOTAL_STEPS = 200_000

HR = "─" * 60
EQ = "═" * 64


def fetch_log(tail_lines=None):
    if tail_lines:
        cmd = f"tail -n {tail_lines} {LOG_PATH}"
    else:
        cmd = f"cat {LOG_PATH}"
    result = subprocess.run(
        ["ssh"] + SSH_ARGS + [cmd],
        capture_output=True, text=True, timeout=60,
    )
    return result.stdout


def parse_events(raw):
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def fmt_hr(hr):
    if hr < 1:
        return f"{hr*60:.0f}min"
    return f"{hr:.1f}hr"


def avg(lst, key):
    vals = [e[key] for e in lst if key in e and e[key] is not None and not (isinstance(e[key], float) and math.isnan(e[key]))]
    return sum(vals) / len(vals) if vals else float("nan")


def bar(frac, width=20, warn=None, crit=None):
    filled = int(frac * width)
    b = "█" * filled + "░" * (width - filled)
    flag = ""
    if crit and frac >= crit:
        flag = " ⚠️ CRIT"
    elif warn and frac >= warn:
        flag = " ⚠️"
    return f"[{b}] {frac*100:.1f}%{flag}"


def fmt_wr(wr):
    try:
        f = float(wr)
        return "?" if math.isnan(f) else f"{f*100:.1f}%"
    except (TypeError, ValueError):
        return "?"


def main():
    tail = None
    if "--tail" in sys.argv:
        idx = sys.argv.index("--tail")
        tail = int(sys.argv[idx + 1])

    print("Fetching log from vast...", end=" ", flush=True)
    raw = fetch_log(tail)
    events = parse_events(raw)
    print(f"done ({len(events):,} events parsed).")

    # Bucket by event type
    by_type = {}
    for e in events:
        t = e.get("event", "unknown")
        by_type.setdefault(t, []).append(e)

    train_steps = by_type.get("train_step", [])
    games = by_type.get("game_complete", [])
    waiting = by_type.get("waiting_for_games", [])
    checkpoints = by_type.get("checkpoint_saved", [])
    spread_alerts = by_type.get("value_spread_alert", [])
    gpu_stats = by_type.get("gpu_stats", [])

    now = datetime.now(timezone.utc)
    print(f"\n{EQ}")
    print(f"  PHASE C MONITOR  —  {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{EQ}")

    # ── STEP / THROUGHPUT ──────────────────────────────────────────
    cur_step = train_steps[-1]["step"] if train_steps else 0
    run_pct = cur_step / TOTAL_STEPS

    print(f"\n  Step:   {cur_step:>7,} / {TOTAL_STEPS:,}  {bar(run_pct, 30)}")

    if len(train_steps) >= 2:
        w = train_steps[-min(200, len(train_steps)):]
        t0, t1 = ts(w[0]["timestamp"]), ts(w[-1]["timestamp"])
        if t0 and t1:
            dt_hr = (t1 - t0).total_seconds() / 3600
            if dt_hr > 0:
                shr = (w[-1]["step"] - w[0]["step"]) / dt_hr
                remaining_hr = (TOTAL_STEPS - cur_step) / shr if shr > 0 else float("inf")
                print(f"  Speed:  {shr:>7,.0f} steps/hr  (ETA {fmt_hr(remaining_hr)})")

    if len(games) >= 2:
        gw = games[-min(500, len(games)):]
        gt0, gt1 = ts(gw[0]["timestamp"]), ts(gw[-1]["timestamp"])
        if gt0 and gt1:
            dt_hr = (gt1 - gt0).total_seconds() / 3600
            if dt_hr > 0:
                ghr = len(gw) / dt_hr
                print(f"  Games:  {ghr:>7,.0f} games/hr")

    total_games = len(games)
    last_waiting = waiting[-1] if waiting else None
    if last_waiting:
        buf = last_waiting.get("buffer", "?")
        print(f"  Buffer: {buf:>7,} positions  ({total_games:,} games completed)")

    if gpu_stats:
        g = gpu_stats[-1]
        print(f"  GPU:    {g.get('gpu_util_pct',0):.0f}% util  {g.get('vram_used_gb',0):.1f}/{g.get('vram_total_gb',0):.1f} GB VRAM  {g.get('temp_c',0):.0f}°C")

    # ── TRAINING SIGNALS ───────────────────────────────────────────
    print(f"\n  {HR}")
    print(f"  TRAINING SIGNALS  (last {min(50, len(train_steps))} steps)")
    print(f"  {HR}")
    if train_steps:
        rec = train_steps[-50:]
        print(f"  policy_loss:      {avg(rec, 'policy_loss'):.4f}")
        print(f"  value_loss:       {avg(rec, 'value_loss'):.4f}")
        gn = avg(rec, 'grad_norm')
        gn_flag = " ⚠️ HIGH" if gn > 5 else " ✓"
        print(f"  grad_norm:        {gn:.4f}{gn_flag}")
        print(f"  value_accuracy:   {avg(rec, 'value_accuracy'):.4f}")
        fs = avg(rec, "full_search_frac")
        print(f"  full_search_frac: {fs:.3f}  (target ~0.5)")
        print(f"  lr:               {train_steps[-1].get('lr', 0):.6f}")

    # ── GAME STATS ─────────────────────────────────────────────────
    print(f"\n  {HR}")
    n_window = min(500, len(games))
    print(f"  GAME STATS  (last {n_window} games)")
    print(f"  {HR}")
    if games:
        rg = games[-n_window:]
        n = len(rg)
        draws = sum(1 for g in rg if g.get("winner") == "draw")
        x_wins = sum(1 for g in rg if g.get("winner") == "x")
        o_wins = sum(1 for g in rg if g.get("winner") == "o")
        dr = draws / n
        avg_plies = avg(rg, "plies")
        colony_vals = [g["colony_extension_fraction"] for g in rg if "colony_extension_fraction" in g]
        avg_colony = sum(colony_vals) / len(colony_vals) if colony_vals else float("nan")
        avg_sps = avg(rg, "sims_per_sec")

        print(f"  draw_rate:   {bar(dr, 20, warn=0.35, crit=0.50)}  (hard abort ≥0.55×3)")
        print(f"  x_win_rate:  {x_wins/n*100:.1f}%")
        print(f"  o_win_rate:  {o_wins/n*100:.1f}%")
        print(f"  avg_plies:   {avg_plies:.1f}")
        if not math.isnan(avg_colony):
            print(f"  colony_frac: {bar(avg_colony, 20, warn=0.10, crit=0.15)}  (investigate ≥0.15)")
        else:
            print(f"  colony_frac: N/A")
        print(f"  sims/sec:    {avg_sps:.0f}")

    # ── VALUE SPREAD ALERTS ────────────────────────────────────────
    if spread_alerts:
        print(f"\n  {HR}")
        print(f"  VALUE SPREAD ALERTS  ({len(spread_alerts)} total)")
        print(f"  {HR}")
        for e in spread_alerts[-5:]:
            step = e.get("step", "?")
            t3 = e.get("t3_spread", float("nan"))
            alt = e.get("alt_spread", float("nan"))
            both = e.get("both_pass", False)
            alert_txt = e.get("alert", "")
            status = "PASS" if both else "FAIL"
            print(f"  step={step:<6}  T3={t3:.4f}  alt={alt}  [{status}]  {alert_txt[:60]}")

    # ── EVAL ROUNDS ────────────────────────────────────────────────
    print(f"\n  {HR}")
    print(f"  EVAL / PROMOTION EVENTS")
    print(f"  {HR}")

    rounds = sorted(by_type.get("evaluation_round_complete", []), key=lambda e: e.get("step", 0))
    games_complete = by_type.get("evaluation_games_complete", [])

    # group per-phase results by step
    games_by_step = {}
    for e in games_complete:
        s = e.get("step")
        if s is not None:
            games_by_step.setdefault(s, []).append(e)

    if not rounds and not games_complete:
        print(f"  (no eval events yet — first eval at step 15000)")
    else:
        for rc in rounds:
            step = rc.get("step", "?")
            promoted = rc.get("promoted", False)
            promo_tag = "PROMOTED" if promoted else "not-promoted"
            n_games = rc.get("eval_games", "?")
            step_str = f"{step:,}" if isinstance(step, int) else str(step)
            elo = rc.get("elo_estimate")
            elo_str = f"  elo_est={elo:.0f}" if isinstance(elo, (int, float)) else ""
            print(f"\n  step {step_str:<7}  [{promo_tag}]  games={n_games}{elo_str}")
            # round_complete embeds per-opponent winrates directly
            wr_best = rc.get("wr_best")
            ci_best = rc.get("ci_best")
            colony_best = rc.get("colony_wins_best", "?")
            wr_rand = rc.get("wr_random")
            if wr_best is not None:
                ci_str = f"  CI=[{ci_best[0]:.3f},{ci_best[1]:.3f}]" if isinstance(ci_best, list) else ""
                print(f"    best_arena              {fmt_wr(wr_best)}{ci_str}  colony_wins={colony_best}")
            if wr_rand is not None:
                print(f"    random                  {fmt_wr(wr_rand)}")

    # explicit promotion events (best_model_promoted etc.)
    promo_events = sorted(
        [e for k, evts in by_type.items()
         if "promot" in k and "\n" not in k and k != "evaluation_round_complete"
         for e in evts],
        key=lambda e: e.get("step", 0),
    )
    if promo_events:
        print(f"\n  Promotions:")
        for p in promo_events:
            step = p.get("step", "?")
            ckpt = str(p.get("checkpoint_path", p.get("path", "?")))
            step_str = f"{step:,}" if isinstance(step, int) else str(step)
            print(f"    step {step_str}  → {ckpt[-50:]}  wr={fmt_wr(p.get('winrate'))}")

    # ── CHECKPOINTS ───────────────────────────────────────────────
    if checkpoints:
        last_ckpt = checkpoints[-1]
        print(f"\n  Last checkpoint: step {last_ckpt.get('step'):,}  →  {last_ckpt.get('checkpoint_path','?')}")

    # ── ALL EVENT TYPES ────────────────────────────────────────────
    clean_types = sorted(k for k in by_type if "\n" not in k and " " not in k and len(k) < 80)
    print(f"\n  {HR}")
    print(f"  EVENT TYPES SEEN: {', '.join(clean_types)}")
    print(f"{EQ}\n")


if __name__ == "__main__":
    main()
