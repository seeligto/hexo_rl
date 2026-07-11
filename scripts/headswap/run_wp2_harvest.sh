#!/bin/bash
# D-F HEADSWAP — harvest deep safe negatives from regenerated WP2 won/drew games.
# SealBot-d7 verify (LABEL-ONLY). workers=8 to keep RAM ~16GB (d7 heavier than d5).
# Launch: setsid bash scripts/headswap/run_wp2_harvest.sh </dev/null >/dev/null 2>&1 &
set -uo pipefail
cd /workspace/hexo_rl
export PYTHONUNBUFFERED=1
mkdir -p reports/headswap
.venv/bin/python scripts/headswap/wp2_regen.py harvest_neg \
  --ckpt /workspace/headswap_data/checkpoint_00248000.pt \
  --regen-dir reports/headswap/wp2_regen --expect-encoding v6_live2_ls \
  --sealbot-depth 7 --out reports/valprobe/negatives_v2_wp2.jsonl --workers 8 \
  > reports/headswap/wp2_harvest.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "DONE $(date -u)" > reports/headswap/HARVEST_DONE
else
  echo "FAILED rc=$rc $(date -u)" > reports/headswap/HARVEST_FAILED
fi
