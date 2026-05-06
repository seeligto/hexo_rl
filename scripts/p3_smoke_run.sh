#!/usr/bin/env bash
# P3 corner-mask probe: 1k smoke A/B + final-checkpoint SealBot eval n=200.
# Sequential (avoid GPU contention). Bootstrap = bootstrap_model_v7full.pt.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p reports/probes/p3_smoke

PY=.venv/bin/python
BOOTSTRAP=checkpoints/bootstrap_model_v7full.pt
ITERATIONS=1000

run_arm() {
    local arm="$1"          # off | on
    local variant="p3_corner_mask_${arm}"
    local ckpt_dir="checkpoints/p3_corner_mask_${arm}"
    local logdir="reports/probes/p3_smoke/${arm}"
    mkdir -p "$logdir"
    rm -rf "$ckpt_dir"
    mkdir -p "$ckpt_dir"

    echo "=== P3 SMOKE arm=${arm} ($(date -u +%FT%TZ)) ==="
    MALLOC_ARENA_MAX=2 ${PY} scripts/train.py \
        --checkpoint "$BOOTSTRAP" \
        --variant "$variant" \
        --checkpoint-dir "$ckpt_dir" \
        --no-dashboard \
        --iterations "$ITERATIONS" 2>&1 | tee "${logdir}/train.log"

    # Locate final checkpoint (highest checkpoint_*.pt in ckpt_dir).
    local final_ckpt
    final_ckpt=$(ls -1 "${ckpt_dir}"/checkpoint_*.pt 2>/dev/null | tail -n 1 || true)
    if [[ -z "$final_ckpt" ]]; then
        echo "ERROR: no checkpoint produced in $ckpt_dir" | tee -a "${logdir}/train.log"
        return 1
    fi
    echo "final_ckpt=${final_ckpt}" | tee "${logdir}/final_ckpt.txt"

    # SealBot eval n=200, time_limit 0.5, model_sims 128 — matches §157 Gate 4.
    echo "=== P3 SealBot eval arm=${arm} ckpt=${final_ckpt} ($(date -u +%FT%TZ)) ==="
    ${PY} scripts/eval_vs_sealbot.py \
        --checkpoint "$final_ckpt" \
        --n-games 200 \
        --time-limit 0.5 \
        --model-sims 128 \
        --out "${logdir}/sealbot_eval.jsonl" 2>&1 | tee "${logdir}/sealbot_eval.log"

    # Capture last 100 game_complete events for self-play health metrics.
    local events
    events=$(ls -1t logs/events_*.jsonl 2>/dev/null | head -n 1 || true)
    if [[ -n "$events" ]]; then
        cp "$events" "${logdir}/events.jsonl"
    fi
}

run_arm off
run_arm on

echo "=== P3 SMOKE DONE ($(date -u +%FT%TZ)) ==="
