#!/usr/bin/env bash
# monitor_exp_E.sh — structured snapshot for Experiment E (Gumbel MCTS desktop)
# A/B comparison: desktop gumbel_full vs laptop exp D (PUCT + completed_q)
# Usage: scripts/monitor_exp_E.sh <jsonl_log_path>

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <jsonl_log_path>" >&2
    exit 1
fi

LOG="$1"

if [[ ! -f "$LOG" ]]; then
    echo "Log file not found: $LOG" >&2
    exit 1
fi

python3 - "$LOG" <<'PYEOF'
import sys
import json
import math
from datetime import datetime

LOG_PATH = sys.argv[1]

# Laptop exp D reference table (PUCT + completed_q, gumbel_targets variant)
# Source: exp D JSONL. Extended as exp D progresses.
REF = {
      565: {"draw": 45.7, "policy": 1.719, "x_win": None,  "o_win": None, "games_hr": None},
     1276: {"draw": 32.1, "policy": 1.795, "x_win": None,  "o_win": None, "games_hr": None},
     5860: {"draw": 42.5, "policy": 1.506, "x_win": None,  "o_win": None, "games_hr": None},
     8188: {"draw": 43.6, "policy": 1.184, "x_win": None,  "o_win": None, "games_hr": None},
     9340: {"draw": 41.5, "policy": 1.238, "x_win": None,  "o_win": None, "games_hr": None},
}
REF_STEPS = sorted(REF.keys())

def nearest_ref(step):
    return min(REF_STEPS, key=lambda s: abs(s - step))

def fmt(v, fmt_str=".3f", fallback="N/A"):
    if v is None:
        return fallback
    try:
        return format(float(v), fmt_str)
    except (TypeError, ValueError):
        return fallback

def pct(v, fallback="N/A"):
    if v is None:
        return fallback
    try:
        return f"{float(v)*100:.1f}%"
    except (TypeError, ValueError):
        return fallback

# Parse log
# ``train_step`` (trainer, per-step) carries chain_loss + grad_norm.
# ``train_step_summary`` (loop, log_interval cadence) carries draw_rate + buffer.
# Split under distinct event names since 2026-04-19 (Q27 smoke fix).
latest_train = None    # summary variant (game stats)
latest_losses = None   # per-step variant (loss details)
sims_per_sec_samples = []

with open(LOG_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = d.get("event")
        if ev == "train_step_summary":
            latest_train = d
        elif ev == "train_step":
            latest_losses = d
        elif ev == "game_complete":
            sps = d.get("sims_per_sec") or d.get("sims_per_second")
            if sps is not None:
                sims_per_sec_samples.append(float(sps))

if latest_train is None and latest_losses is None:
    print("No train_step events found yet — training may not have started.")
    sys.exit(0)

step_train  = latest_train.get("step", 0)  if latest_train  else 0
step_losses = latest_losses.get("step", 0) if latest_losses else 0
current_step = max(step_train, step_losses)

def get(rec, key, default=None):
    if rec is None:
        return default
    return rec.get(key, default)

games       = get(latest_train, "games_played")
buf_size    = get(latest_train, "buffer_size")
buf_cap     = get(latest_train, "buffer_capacity")
draw_rate   = get(latest_train, "draw_rate")
x_winrate   = get(latest_train, "x_winrate")
o_winrate   = get(latest_train, "o_winrate")
games_hr    = get(latest_train, "games_per_hour")
pos_hr      = get(latest_train, "positions_per_hour")
gpu_util    = get(latest_train, "gpu_util")
vram_gb     = get(latest_train, "vram_gb")
trunc_rate  = get(latest_train, "truncation_rate")

policy_loss  = get(latest_train, "policy_loss") or get(latest_losses, "policy_loss")
value_loss   = get(latest_train, "value_loss")  or get(latest_losses, "value_loss")
threat_loss  = get(latest_losses, "threat_loss")
chain_loss   = get(latest_losses, "chain_loss")
grad_norm    = get(latest_losses, "grad_norm")
pol_entropy  = get(latest_train, "policy_entropy_selfplay") or get(latest_losses, "policy_entropy")

# Gumbel-specific
sims_per_sec_mean = sum(sims_per_sec_samples[-50:]) / len(sims_per_sec_samples[-50:]) if sims_per_sec_samples else None

ref_step = nearest_ref(current_step)
ref = REF[ref_step]
draw_pct  = float(draw_rate) * 100 if draw_rate is not None else None
delta_draw = (draw_pct - ref["draw"]) if draw_pct is not None else None
delta_pol  = (float(policy_loss) - ref["policy"]) if policy_loss is not None else None

def sgn(v):
    if v is None: return ""
    return "+" if v >= 0 else ""

print()
print(f"=== Exp E (Gumbel MCTS desktop) Snapshot @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
print(f"Step: {current_step}  |  Games: {games if games is not None else 'N/A'}  |  "
      f"Buffer: {buf_size if buf_size is not None else 'N/A'}/{buf_cap if buf_cap is not None else 'N/A'}")

chain_str = fmt(chain_loss, ".5f") if chain_loss is not None else "N/A"
print(f"Losses:  policy={fmt(policy_loss)}  value={fmt(value_loss)}  "
      f"threat={fmt(threat_loss)}  chain={chain_str}  grad_norm={fmt(grad_norm)}")

draw_str  = f"{draw_pct:.1f}%" if draw_pct is not None else "N/A"
x_str     = pct(x_winrate)
o_str     = pct(o_winrate)
x_raw = float(x_winrate) if x_winrate is not None else None
o_raw = float(o_winrate) if o_winrate is not None else None
if x_raw is not None and o_raw is not None and (x_raw + o_raw) > 0:
    decisive_total = x_raw + o_raw
    x_of_decisive = x_raw / decisive_total * 100
    balance_str = f"{x_of_decisive:.1f}% X / {100-x_of_decisive:.1f}% O (ex-draws)"
else:
    balance_str = "N/A"
print(f"Self-play:  draw_rate={draw_str}  x_win={x_str}  o_win={o_str}")
print(f"Balance:    {balance_str}")

games_hr_str = f"{float(games_hr):.0f}" if games_hr is not None else "N/A"
pos_hr_str   = f"{float(pos_hr):.0f}" if pos_hr is not None else "N/A"
gpu_str      = f"{float(gpu_util):.0f}%" if gpu_util is not None else "N/A"
vram_str     = f"{float(vram_gb):.2f} GB" if vram_gb is not None else "N/A"
trunc_str    = f"{float(trunc_rate)*100:.1f}%" if trunc_rate is not None else "N/A"
sps_str      = f"{sims_per_sec_mean:.0f}" if sims_per_sec_mean is not None else "N/A (check game_complete events)"
ent_str      = f"{float(pol_entropy):.3f}" if pol_entropy is not None else "N/A"

print(f"Throughput: games/hr={games_hr_str}  pos/hr={pos_hr_str}  trunc_rate={trunc_str}")
print(f"GPU: util={gpu_str}  VRAM={vram_str}")
print(f"Gumbel:  sims/s/game(last-50)={sps_str}  policy_entropy_selfplay={ent_str}")
print(f"  (Gumbel expected sims/s: ~3400; PUCT baseline: N/A — different search)")

print("--- A/B: vs laptop exp D at nearest reference step ---")
ref_draw_str = f"{ref['draw']:.1f}%"
ref_pol_str  = f"{ref['policy']:.3f}"
cur_draw_str = f"{draw_pct:.1f}%" if draw_pct is not None else "N/A"
cur_pol_str  = fmt(policy_loss)

delta_draw_str = (f"{sgn(delta_draw)}{delta_draw:.1f}pp" if delta_draw is not None else "N/A")
delta_pol_str  = (f"{sgn(delta_pol)}{delta_pol:.3f}"    if delta_pol  is not None else "N/A")

print(f"exp D draw @ step {ref_step}: {ref_draw_str}  |  exp E current: {cur_draw_str}  |  delta: {delta_draw_str}")
print(f"exp D policy @ step {ref_step}: {ref_pol_str}   |  exp E current: {cur_pol_str}   |  delta: {delta_pol_str}")
print()

# Monitoring schedule reminder
if current_step < 1000:
    print("  [SCHEDULE: steps 0-1000 — check every 30 min]")
elif current_step < 5000:
    print("  [SCHEDULE: steps 1000-5000 — check every 1 hr]")
elif current_step == 5000:
    print("  [SCHEDULE: step 5000 — run make probe.latest (LOG result, do NOT kill on FAIL)]")
elif current_step < 15000:
    print("  [SCHEDULE: steps 5000-15000 — check every 2 hr]")
    if current_step >= 10000:
        print("  [SCHEDULE: step 10000 — re-probe]")
elif current_step >= 15000:
    print("  [SCHEDULE: step 15000+ — re-probe + full comparison to laptop exp D]")

# Kill conditions (relaxed per exp D learnings; pos/hr threshold: < 35k)
flags = []
if draw_pct is not None:
    if draw_pct > 70.0:
        flags.append("CRITICAL: draw_rate > 70% — kill if sustained 500+ games and not declining 2k steps")
    elif draw_pct > 55.0:
        flags.append("WARNING: draw_rate > 55% — monitor; kill only if > 70% sustained")
    if delta_draw is not None and delta_draw < -3.0:
        flags.append(f"POSITIVE: draw_rate {abs(delta_draw):.1f}pp BELOW laptop exp D reference")

if pol_entropy is not None and float(pol_entropy) < 1.5:
    flags.append(f"WARNING: policy_entropy_selfplay={pol_entropy:.3f} < 1.5 — watch for collapse")
if grad_norm is not None and float(grad_norm) > 10.0:
    flags.append(f"CRITICAL: grad_norm={grad_norm:.2f} > 10 — kill if sustained 50+ steps")
if pos_hr is not None and float(pos_hr) < 35000:
    flags.append(f"CRITICAL: pos_per_hr={float(pos_hr):.0f} < 35k sustained — Gumbel overhead broke throughput")

if flags:
    for flag in flags:
        print(f"  [{flag}]")
    print()

PYEOF
