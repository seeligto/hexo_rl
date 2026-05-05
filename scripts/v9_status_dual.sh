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
        # JSON-line structlog (laptop). Latest train/game event with key fields.
        grep -E '"event": "(train_step|game_complete|step_complete|iter_complete)"' "$LATEST" 2>/dev/null | tail -n 1 | head -c 360
        echo
        # Extract step / loss / draw / stride5 across last 200 lines.
        tail -n 200 "$LATEST" 2>/dev/null \
          | grep -oE '"(step|loss|draw_rate|stride5_run_max|grad_norm|games_completed|positions_pushed|sims_per_sec)": [0-9.eE+-]+' \
          | sort -u | tail -n 8 | tr '\n' ' '
        echo
    fi
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -n 1
}

echo "=== $(date '+%H:%M:%S') ==="
echo "[REMOTE 5080]"
extract_remote | sed 's/^/  /'
echo
echo "[LAPTOP]"
extract_laptop | sed 's/^/  /'
