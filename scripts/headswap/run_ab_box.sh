#!/bin/bash
# D-F HEADSWAP — early read: ARM A (scalar) + B (65-bin), frozen trunk, 3 seeds.
# Runs ON the ephemeral box (cwd /workspace/hexo_rl). First run builds the shared
# batch-stream cache; the other 5 reuse it (identical stream across arms). LR 1x.
set -uo pipefail
cd /workspace/hexo_rl
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
TRUNK=/workspace/headswap_data/checkpoint_00248000.pt
BUF=/workspace/headswap_data/replay_buffer.bin
OUT=reports/headswap/ab
CACHE=reports/headswap/batch_cache_s10000_b256
STEPS=10000
mkdir -p "$OUT"

for ARM in A B; do
  for SEED in 0 1 2; do
    echo "=== ARM $ARM seed $SEED start $(date -u +%H:%M:%S) ==="
    $PY -m scripts.headswap.train_arm --arm "$ARM" --trunk "$TRUNK" --buffer "$BUF" \
      --steps "$STEPS" --seed "$SEED" --lr 2e-3 --batch 256 \
      --out "$OUT/arm_${ARM}_seed${SEED}" --batch-cache "$CACHE" \
      > "$OUT/log_${ARM}_seed${SEED}.txt" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "ARM $ARM seed $SEED FAILED rc=$rc — see $OUT/log_${ARM}_seed${SEED}.txt" | tee "$OUT/AB_FAILED"
      exit $rc
    fi
    echo "=== ARM $ARM seed $SEED done  $(date -u +%H:%M:%S) ==="
  done
done
echo "AB_ALL_DONE $(date -u)" > "$OUT/AB_DONE"
echo "ALL A/B RUNS COMPLETE"
