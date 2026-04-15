#!/usr/bin/env bash
# monitor_experiment_a.sh — structured snapshot for Experiment A (aux_chain_weight=0)
# Usage: scripts/monitor_experiment_a.sh <jsonl_log_path>
# Compares current run vs smoke_v3b reference table.

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

# smoke_v3b reference table (aux_chain_weight=1.0)
REF = {
       68: {"draw": 25.0, "policy": 1.858, "value": 0.551, "x_win": None,  "o_win": None},
     1920: {"draw": 35.3, "policy": 1.574, "value": 0.554, "x_win": 40.4, "o_win": 24.3},
     2616: {"draw": 36.2, "policy": 1.761, "value": 0.594, "x_win": 38.3, "o_win": 25.5},
     3247: {"draw": 37.7, "policy": 1.625, "value": 0.523, "x_win": 37.8, "o_win": 24.5},
     3804: {"draw": 40.7, "policy": 1.607, "value": 0.525, "x_win": 35.6, "o_win": 23.7},
     4392: {"draw": 43.2, "policy": 1.571, "value": 0.554, "x_win": 33.4, "o_win": 23.4},
     5003: {"draw": 44.7, "policy": 1.528, "value": 0.574, "x_win": 32.1, "o_win": 23.2},
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

# Parse log: keep latest record for each event variant
# train_step: two variants — structlog (has chain_loss) and emit_event (has draw_rate)
latest_train = None    # emit_event variant (game stats)
latest_losses = None   # structlog variant (loss details)

with open(LOG_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") != "train_step":
            continue
        if "draw_rate" in d:
            latest_train = d
        elif "chain_loss" in d:
            latest_losses = d

if latest_train is None and latest_losses is None:
    print("No train_step events found yet — training may not have started.")
    sys.exit(0)

# Merge: use whichever is more recent by step
step_train  = latest_train.get("step", 0)  if latest_train  else 0
step_losses = latest_losses.get("step", 0) if latest_losses else 0
current_step = max(step_train, step_losses)

# Extract fields from the appropriate records
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
gpu_util    = get(latest_train, "gpu_util")
vram_gb     = get(latest_train, "vram_gb")

policy_loss = get(latest_train, "policy_loss") or get(latest_losses, "policy_loss")
value_loss  = get(latest_train, "value_loss")  or get(latest_losses, "value_loss")
threat_loss = get(latest_losses, "threat_loss")
chain_loss  = get(latest_losses, "chain_loss")
grad_norm   = get(latest_losses, "grad_norm")

# --- Comparison row ---
ref_step = nearest_ref(current_step)
ref = REF[ref_step]
draw_pct  = float(draw_rate) * 100 if draw_rate is not None else None
delta_draw = (draw_pct - ref["draw"]) if draw_pct is not None else None
delta_pol  = (float(policy_loss) - ref["policy"]) if policy_loss is not None else None

def sgn(v):
    if v is None: return ""
    return "+" if v >= 0 else ""

# --- Print ---
print()
print(f"=== Experiment A Snapshot @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
print(f"Step: {current_step}  |  Games: {games if games is not None else 'N/A'}  |  "
      f"Buffer: {buf_size if buf_size is not None else 'N/A'}/{buf_cap if buf_cap is not None else 'N/A'}")

chain_str = fmt(chain_loss, ".5f") if chain_loss is not None else "N/A"
print(f"Losses:  policy={fmt(policy_loss)}  value={fmt(value_loss)}  "
      f"threat={fmt(threat_loss)}  chain={chain_str}  grad_norm={fmt(grad_norm)}")

draw_str  = f"{draw_pct:.1f}%" if draw_pct is not None else "N/A"
x_str     = pct(x_winrate)
o_str     = pct(o_winrate)
print(f"Self-play:  draw_rate={draw_str}  x_win={x_str}  o_win={o_str}")

games_hr_str = f"{float(games_hr):.0f}" if games_hr is not None else "N/A"
gpu_str      = f"{float(gpu_util):.0f}%" if gpu_util is not None else "N/A"
vram_str     = f"{float(vram_gb):.2f} GB" if vram_gb is not None else "N/A"
print(f"Games/hr: {games_hr_str}  |  GPU util: {gpu_str}  |  VRAM: {vram_str}")

print("--- Comparison vs smoke_v3b at same step ---")
ref_draw_str = f"{ref['draw']:.1f}%"
ref_pol_str  = f"{ref['policy']:.3f}"
cur_draw_str = f"{draw_pct:.1f}%" if draw_pct is not None else "N/A"
cur_pol_str  = fmt(policy_loss)

delta_draw_str = (f"{sgn(delta_draw)}{delta_draw:.1f}pp" if delta_draw is not None else "N/A")
delta_pol_str  = (f"{sgn(delta_pol)}{delta_pol:.3f}"    if delta_pol  is not None else "N/A")

print(f"smoke draw_rate @ step {ref_step}: {ref_draw_str}  |  current: {cur_draw_str}  |  delta: {delta_draw_str}")
print(f"smoke policy    @ step {ref_step}: {ref_pol_str}   |  current: {cur_pol_str}   |  delta: {delta_pol_str}")
print()

# Flags
flags = []
if draw_pct is not None:
    if draw_pct > 50.0:
        flags.append("WARNING: draw_rate > 50% — approaching kill threshold")
    elif draw_pct < 35.0 and current_step >= 5000:
        flags.append("SUCCESS (primary): draw_rate < 35% at step 5000+")
    if delta_draw is not None and delta_draw < -3.0:
        flags.append(f"POSITIVE: draw_rate is {abs(delta_draw):.1f}pp BELOW smoke_v3b reference")

if flags:
    for f in flags:
        print(f"  [{f}]")
    print()

PYEOF
