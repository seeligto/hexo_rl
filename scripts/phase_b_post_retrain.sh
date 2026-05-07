#!/bin/bash
# Phase B Gate 4 post-retrain harness — bench + SealBot eval for each arm.
#
# Run once retrains complete; iterates over checkpoints/v8_variants/B*_v8full.pt.
# Bench runs n=5 b=1+b=64; SealBot eval runs n=200 games at time_limit=0.5
# (see scripts/eval_v8_vs_sealbot.py for argmax-only rationale).
#
# Usage:
#   bash scripts/phase_b_post_retrain.sh [host_label] [n_games] [time_limit]
# Defaults: host_label=$(hostname), n_games=200, time_limit=0.5.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST_LABEL="${1:-$(hostname -s)}"
N_GAMES="${2:-200}"
TIME_LIMIT="${3:-0.5}"

REPORTS=reports/encoding_phase_b
mkdir -p "$REPORTS"

echo "=== Phase B post-retrain — host=$HOST_LABEL  n_games=$N_GAMES  time_limit=$TIME_LIMIT ==="

for ckpt in checkpoints/v8_variants/B*_v8full.pt; do
    if [ ! -f "$ckpt" ]; then continue; fi
    arm=$(basename "$ckpt" _v8full.pt)
    echo ""
    echo "=== $arm — bench + eval ==="

    # Bench (NN latency + params)
    .venv/bin/python scripts/bench_v8_nn.py \
        --checkpoint "$ckpt" \
        --host "$HOST_LABEL" \
        --runs 5 --warmup 20 --batches 1,64 \
        --out "$REPORTS/${arm}_bench_${HOST_LABEL}.json"

    # SealBot WR n=N_GAMES
    .venv/bin/python scripts/eval_v8_vs_sealbot.py \
        --checkpoint "$ckpt" \
        --n-games "$N_GAMES" \
        --time-limit "$TIME_LIMIT" \
        --out "$REPORTS/${arm}_sealbot.json"
done

echo ""
echo "=== Phase B post-retrain DONE — outputs in $REPORTS/ ==="
