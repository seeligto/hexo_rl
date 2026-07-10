#!/usr/bin/env bash
# Run all retro-slope ckpts sequentially (resume-safe: skips done ones).
set -e

PYTHON=.venv/bin/python
BOOK_R4=tests/fixtures/opening_books/evalfair_r4_v2.json
BOOK_R5=tests/fixtures/opening_books/evalfair_r5_v2.json
OUT=reports/evalfair/retro_slope
WORKERS=4

run_ckpt() {
  local ckpt="$1"
  echo "=== $(date '+%H:%M:%S') $ckpt ==="
  $PYTHON -m scripts.evalfair.run_retro_ckpt \
    --ckpt "$ckpt" \
    --book-r4 "$BOOK_R4" \
    --book-r5 "$BOOK_R5" \
    --out "$OUT" \
    --workers "$WORKERS"
}

# Series A: radius 4, <200k
run_ckpt checkpoints/run2_retro/checkpoint_00050000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00070000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00090000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00110000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00130000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00150000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00170000.pt
# 175k anchor
run_ckpt scripts/arena/weights/run2_175k.pt
run_ckpt checkpoints/run2_retro/checkpoint_00195000.pt

# Series B: radius 5, >=200k
run_ckpt checkpoints/run2_retro/checkpoint_00200000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00210000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00220000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00230000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00240000.pt
run_ckpt checkpoints/run2_retro/checkpoint_00248000.pt

echo "=== ALL DONE ==="
