#!/usr/bin/env bash
# v9_status_dual.sh — one-shot status reporter for the dual-host
# Phase B' v9 §153 T5 smoke pipeline.
#
# Reads logs from both hosts and emits a compact one-block status:
#
#   * Remote (5080): driver-log tail + active log (bootstrap or smoke)
#     with last train_step / step= line, training metrics, GPU.
#   * Laptop: driver tail + active log JSON-aware extraction of the
#     latest train_step / game_complete fields, GPU.
#
# Designed to be invoked by the persistent monitor every 30 min.

set -uo pipefail

extract_remote() {
    ssh -p 13053 -i ~/.ssh/vast_hexo -o IdentitiesOnly=yes \
        -o UserKnownHostsFile=~/.ssh/known_hosts_vast \
        -o ConnectTimeout=20 \
        root@ssh6.vast.ai bash -s <<'REMOTE' 2>&1 | grep -vE "Welcome to vast|Have fun"
set -uo pipefail
cd /workspace/hexo_rl
DRIVER='logs/v9_smokes_runner.driver.log'
[ -f "$DRIVER" ] && tail -n 2 "$DRIVER"
LATEST="$(ls -1t logs/v9_S*.log 2>/dev/null | head -n 1)"
WARM='logs/v9_bootstrap_v8full_warm.log'
SCRATCH='logs/v9_bootstrap_v8full_scratch.log'
if [ -n "$LATEST" ]; then
    echo "active: $(basename "$LATEST")"
    # KV-format structlog (remote uses default sink). Capture last train_step.
    grep 'train_step\|game_complete\|step_complete' "$LATEST" 2>/dev/null | tail -n 1 | head -c 360
    echo
    grep -oE 'iter_per_min=[0-9.]+|sims_per_sec=[0-9.]+|draw_rate=[0-9.]+|stride5_run_max=[0-9]+' "$LATEST" 2>/dev/null | tail -n 4 | tr '\n' ' '
    echo
elif [ -f "$WARM" ] || [ -f "$SCRATCH" ]; then
    for L in "$SCRATCH" "$WARM"; do
        if [ -f "$L" ]; then
            B=$(basename "$L" .log)
            echo "$B: $(grep epoch_complete "$L" 2>/dev/null | tail -n 1 | head -c 200)"
        fi
    done
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -n 1
REMOTE
}

extract_laptop() {
    cd /home/timmy/Work/hexo_rl
    DRIVER="logs/v9_smokes_runner.driver.log"
    [ -f "$DRIVER" ] && tail -n 2 "$DRIVER"
    LATEST="$(ls -1t logs/v9_S*.log 2>/dev/null | head -n 1)"
    if [ -n "$LATEST" ]; then
        echo "active: $(basename "$LATEST")"
        # Latest train_step (JSON) — extract just the meaningful fields.
        LAST_TS=$(grep -E '"event": "train_step"' "$LATEST" 2>/dev/null | tail -n 1)
        if [ -n "$LAST_TS" ]; then
            echo "$LAST_TS" | python3 -c '
import json, sys
try:
    d = json.loads(sys.stdin.read())
    keys = ["step", "total_loss", "policy_loss", "value_loss", "grad_norm",
            "ownership_loss", "threat_loss", "lr"]
    print(" ".join(f"{k}={d[k]:.4g}" if isinstance(d.get(k), float) else f"{k}={d.get(k)}" for k in keys if k in d))
except Exception as e:
    print(f"(parse err: {e})")
'
        fi
        # Latest game_complete summary (if any).
        grep -E '"event": "game_complete"' "$LATEST" 2>/dev/null | tail -n 1 | python3 -c '
import json, sys
try:
    s = sys.stdin.read().strip()
    if s:
        d = json.loads(s)
        keys = ["game_length", "winner", "draw_rate", "stride5_run_max",
                "row_max_density", "colony_extension_fraction"]
        out = " ".join(f"{k}={d[k]}" for k in keys if k in d)
        if out: print(f"game: {out}")
except Exception as e:
    print(f"(game parse err: {e})")
' 2>/dev/null
        # Recent count of games / iters for pace.
        N_TS=$(grep -cE '"event": "train_step"' "$LATEST")
        N_GC=$(grep -cE '"event": "game_complete"' "$LATEST")
        echo "totals: train_steps=$N_TS games_completed=$N_GC"

        # Stride5 / row_max (Class-4 metric) from the events.jsonl sink.
        EVENTS_FILE="$(ls -1t logs/events_*.jsonl 2>/dev/null | head -n 1)"
        if [ -n "$EVENTS_FILE" ]; then
            python3 - "$EVENTS_FILE" <<'PY'
import json, sys, statistics
path = sys.argv[1]
stride5, rowmax, colony, draws = [], [], [], 0
with open(path) as f:
    for line in f:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") != "game_complete":
            continue
        if "stride5_run_max" in d:
            stride5.append(d["stride5_run_max"])
        if "row_max_density" in d:
            rowmax.append(d["row_max_density"])
        if "colony_extension_fraction" in d:
            colony.append(d["colony_extension_fraction"])
        if d.get("winner") == -1:
            draws += 1
n = len(stride5)
if n:
    p50_s = statistics.median(stride5[-100:])
    p50_r = statistics.median(rowmax[-100:])
    drf = draws / max(1, len(stride5))
    print(f"class-4 (last {min(100,n)}): stride5_p50={p50_s} row_max_p50={p50_r} draw_rate={drf:.3f} colony_p50={statistics.median(colony[-100:]):.3f}")
PY
        fi
    fi
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null | head -n 1
}

echo "=== $(date '+%H:%M:%S') ==="
echo "[REMOTE 5080]"
extract_remote | sed 's/^/  /'
echo
echo "[LAPTOP]"
extract_laptop | sed 's/^/  /'
