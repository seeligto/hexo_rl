#!/usr/bin/env bash
# §P5-CT Phase 5a — v6tp (CF-2) corpus export + ~30-epoch pretrain.
#
# Produces the two artifacts the Phase 5a smoke (configs/variants/
# v6tp_p5a_smoke.yaml) depends on:
#   data/bootstrap_corpus_v6tp.npz       — 10-plane human corpus (incl. 16/17)
#   checkpoints/bootstrap_model_v6tp.pt  — fresh 10-plane bootstrap (inference)
#
# v6tp is NOT a v6/v7full transfer: in_channels 8 → 10, so the first conv
# shape differs and weights cannot be loaded across. This is a from-scratch
# ~30-epoch pretrain mirroring the §150 v7full recipe.
#
# Operator GPU job (~30-45 min on vast 5080 for the pretrain; the corpus
# export is a CPU job, ~few min). Run from repo root inside the venv.
set -euo pipefail

# 1. Export the 10-plane v6tp corpus from human games (all positions,
#    Elo-weighted, uncompressed for mmap). ~700 MB. Use the repo venv
#    python (non-interactive shells — e.g. vast over ssh — have no bare
#    `python` on PATH; matches the Makefile PY convention).
.venv/bin/python scripts/export_corpus_npz.py --encoding v6tp --human-only --no-compress

# 2. Fresh 30-epoch pretrain. --inference-out writes the smoke's bootstrap
#    checkpoint; training checkpoints land under checkpoints/pretrain/.
make pretrain \
  PRETRAIN_ENCODING=v6tp \
  PRETRAIN_EPOCHS=30 \
  PRETRAIN_OUT=checkpoints/bootstrap_model_v6tp.pt

echo "Done. Launch the smoke:"
echo "  python scripts/train.py \\"
echo "    --checkpoint checkpoints/bootstrap_model_v6tp.pt \\"
echo "    --variant v6tp_p5a_smoke --iterations 30000"
