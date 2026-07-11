#!/bin/bash
# D-F HEADSWAP — WP2 game regeneration (5 books, head@248k vs SealBot-d5).
# workers=10 to keep RAM ~14GB (<16GB pref; well under 32GB ceiling) on the box.
# Launch detached: setsid bash scripts/headswap/run_wp2_gen.sh </dev/null >/dev/null 2>&1 &
set -uo pipefail
cd /workspace/hexo_rl
export PYTHONUNBUFFERED=1
mkdir -p reports/headswap
.venv/bin/python scripts/headswap/wp2_regen.py generate \
  --ckpt /workspace/headswap_data/checkpoint_00248000.pt \
  --books-dir reports/valprobe/wp2 --out reports/headswap/wp2_regen \
  --expect-encoding v6_live2_ls --workers 10 \
  > reports/headswap/wp2_gen.log 2>&1
rc=$?
mkdir -p reports/headswap/wp2_regen
if [ $rc -eq 0 ]; then
  echo "DONE $(date -u)" > reports/headswap/wp2_regen/GEN_DONE
else
  echo "FAILED rc=$rc $(date -u)" > reports/headswap/wp2_regen/GEN_FAILED
fi
