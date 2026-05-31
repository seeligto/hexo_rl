#!/usr/bin/env bash
# §P5-CT — v6_live2 H-PLANE-MISMATCH fix: corpus export + 30-epoch pretrain.
#
# Mirror of scripts/p5a_v6_control_pretrain.sh with encoding v6_live2 (4-plane:
# my/opp t0 + turn-phase 16/17) instead of v6 (8-plane). v6_live2 drops the
# history planes 1-3/9-11 that are live in pretrain but zeroed in self-play, so
# pretrain and self-play see the same 4 live planes (no distribution cliff).
# Same human games + seed 42 as the v6tp / v6-control arms so the comparison
# isolates exactly the dropped history planes.
#
# Produces:
#   data/bootstrap_corpus_v6_live2.npz       — 4-plane corpus
#   checkpoints/bootstrap_model_v6_live2.pt  — fresh 4-plane bootstrap
#
# (The corpus export is idempotent — skipped if it already exists.)
set -euo pipefail

# 1. Export the 4-plane v6_live2 corpus (same games/sampling as v6tp/control).
if [ ! -f data/bootstrap_corpus_v6_live2.npz ]; then
  .venv/bin/python scripts/export_corpus_npz.py --encoding v6_live2 --human-only \
    --no-compress --out data/bootstrap_corpus_v6_live2.npz
fi

# 2. Fresh 30-epoch pretrain on the v6_live2 corpus (--corpus-npz override so it
#    does NOT pick up the default data/bootstrap_corpus.npz).
MALLOC_ARENA_MAX=2 .venv/bin/python -m hexo_rl.bootstrap.pretrain \
  --encoding v6_live2 \
  --corpus-npz data/bootstrap_corpus_v6_live2.npz \
  --epochs 30 \
  --inference-out checkpoints/bootstrap_model_v6_live2.pt

echo "Done. Launch the v6_live2 smoke:"
echo "  python scripts/train.py \\"
echo "    --checkpoint checkpoints/bootstrap_model_v6_live2.pt \\"
echo "    --variant v6_live2_smoke --iterations 30000"
