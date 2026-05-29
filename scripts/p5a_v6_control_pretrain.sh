#!/usr/bin/env bash
# §P5-CT Phase 5a — CF-1-only v6 CONTROL: corpus export + 30-epoch pretrain.
#
# Mirror of scripts/p5a_v6tp_pretrain.sh with encoding v6 (8-plane) instead of
# v6tp (10-plane). Everything else identical so the v6tp-vs-v6 comparison
# isolates exactly the turn-phase planes (CF-2). Same human games + seed 42.
#
# Produces:
#   data/bootstrap_corpus_v6_p5ctl.npz       — 8-plane control corpus
#   checkpoints/bootstrap_model_v6_p5ctl.pt  — fresh 8-plane bootstrap
#
# (The corpus may already exist from a parallel export; the step is idempotent.)
set -euo pipefail

# 1. Export the 8-plane v6 control corpus (same games/sampling as v6tp).
if [ ! -f data/bootstrap_corpus_v6_p5ctl.npz ]; then
  .venv/bin/python scripts/export_corpus_npz.py --encoding v6 --human-only \
    --no-compress --out data/bootstrap_corpus_v6_p5ctl.npz
fi

# 2. Fresh 30-epoch pretrain on the control corpus (--corpus-npz override so it
#    does NOT pick up the default data/bootstrap_corpus.npz).
MALLOC_ARENA_MAX=2 .venv/bin/python -m hexo_rl.bootstrap.pretrain \
  --encoding v6 \
  --corpus-npz data/bootstrap_corpus_v6_p5ctl.npz \
  --epochs 30 \
  --inference-out checkpoints/bootstrap_model_v6_p5ctl.pt

echo "Done. Launch the control smoke:"
echo "  python scripts/train.py \\"
echo "    --checkpoint checkpoints/bootstrap_model_v6_p5ctl.pt \\"
echo "    --variant v6_p5a_control --iterations 30000"
