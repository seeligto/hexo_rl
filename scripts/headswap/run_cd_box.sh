#!/bin/bash
# D-F HEADSWAP — ARM C (65-bin) + D (scalar), LAST BLOCK (tower[11]) UNFROZEN, 3 seeds.
# Runs ON the box. Reuses the SAME shared batch-stream memmap the A/B run built
# (identical stream across ALL arms — the paired-comparison requirement). LR 1x head,
# tower[11] at 0.1x (handled by train_arm freeze_for_arm). Byte-identical to A/B otherwise.
set -uo pipefail
cd /workspace/hexo_rl
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
TRUNK=/workspace/headswap_data/checkpoint_00248000.pt
BUF=/workspace/headswap_data/replay_buffer.bin
OUT=reports/headswap/cd
CACHE=reports/headswap/batch_cache_s10000_b256   # SAME cache as A/B (must already exist)
STEPS=10000
mkdir -p "$OUT"

if [ ! -f "$CACHE/manifest.json" ]; then
  echo "ERROR: shared cache $CACHE missing — run A/B first" | tee "$OUT/CD_FAILED"; exit 2
fi

for ARM in C D; do
  for SEED in 0 1 2; do
    echo "=== ARM $ARM seed $SEED start $(date -u +%H:%M:%S) ==="
    $PY -m scripts.headswap.train_arm --arm "$ARM" --trunk "$TRUNK" --buffer "$BUF" \
      --steps "$STEPS" --seed "$SEED" --lr 2e-3 --batch 256 \
      --out "$OUT/arm_${ARM}_seed${SEED}" --batch-cache "$CACHE" \
      > "$OUT/log_${ARM}_seed${SEED}.txt" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "ARM $ARM seed $SEED FAILED rc=$rc — see $OUT/log_${ARM}_seed${SEED}.txt" | tee "$OUT/CD_FAILED"
      exit $rc
    fi
    echo "=== ARM $ARM seed $SEED done  $(date -u +%H:%M:%S) ==="
  done
done
echo "CD_ALL_DONE $(date -u)" > "$OUT/CD_DONE"
echo "ALL C/D RUNS COMPLETE"
