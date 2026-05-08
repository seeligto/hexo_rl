#!/usr/bin/env bash
# §169 A4 retrain — v8 + canvas_realness mask + PartialConv2d trunk entry.
#
# Same architecture as §167 B1 (128 filters × 12 res_blocks, GPool {6,10},
# KataGo policy head) and same recipe as the v6w25 anchor (30 ep cosine,
# peak 2e-3, eta_min 5e-5, batch 256). Only deltas vs. B1 are:
#   - --canvas-realness flag (inverts plane-8 polarity + wires PartialConv2d
#     at trunk entry). See hexo_rl/model/partial_conv.py + the §169 A4 entry
#     in docs/07_PHASE4_SPRINT_LOG.md (P4).
#   - Corpus NPZ swap: the canvas_realness corpus has plane 8 inverted
#     (1=inside, matches Innamorati 2018 / KataGo gpool convention).
#
# Output checkpoint at checkpoints/ablation_169/A4_canvas_realness.pt; full
# pretrain log captured to reports/ablation_169/A4_pretrain.log.
#
# Hard surface conditions (§169 A4 spec):
#   - Subspike SE × PartialConv2d compatibility — already gated PASS at
#     audit/encoding_spikes/s4_a4_se_partial_conv.py before this script
#     runs. Re-run on the host if architecture changes.
#   - final loss > 5.36 (50% above v6w25 anchor 3.57): STOP, surface, no eval.
#   - NaN-skip rate > 30% even with §167 patch: STOP, retry with bf16.
#   - A4 argmax @ r=8 n=200 > 12% (>80% of B1-vs-A1 gap closure): SURFACE,
#     do not STOP — matched MCTS-N becomes critical, keep the eval running.
#
# Prerequisite — corpus regen (~10 min on 5080):
#   python scripts/export_corpus_npz.py --encoding v8 --canvas-realness \
#       --human-only --no-compress \
#       --out data/bootstrap_corpus_v8_canvas_realness.npz
# capture sha256 from the export-script tail; record in §169 P4 sprint log.
#
# Run from repo root on the 5080 vast.ai host (see .claude/skills/rsync-vast/).
set -euo pipefail

mkdir -p checkpoints/ablation_169 reports/ablation_169

CKPT_DIR="checkpoints/ablation_169/pretrain_a4"
INFERENCE_OUT="checkpoints/ablation_169/A4_canvas_realness.pt"
LOG="reports/ablation_169/A4_pretrain.log"
CORPUS="${CORPUS:-data/bootstrap_corpus_v8_canvas_realness.npz}"

mkdir -p "${CKPT_DIR}"

if [[ ! -f "${CORPUS}" ]]; then
    echo "ERROR: A4 corpus missing: ${CORPUS}" >&2
    echo "       Regenerate via:" >&2
    echo "         python scripts/export_corpus_npz.py --encoding v8 \\" >&2
    echo "             --canvas-realness --human-only --no-compress \\" >&2
    echo "             --out ${CORPUS}" >&2
    exit 1
fi

echo "[$(date -Iseconds)] §169 A4 canvas_realness retrain starting" | tee -a "${LOG}"
echo "  ckpt-dir   : ${CKPT_DIR}"           | tee -a "${LOG}"
echo "  inference  : ${INFERENCE_OUT}"       | tee -a "${LOG}"
echo "  corpus     : ${CORPUS}"              | tee -a "${LOG}"
echo "  log        : ${LOG}"                 | tee -a "${LOG}"
echo "  recipe     : 30 ep cosine, peak 2e-3, eta_min 5e-5, batch 256, v8 + canvas_realness + PC trunk entry" | tee -a "${LOG}"

MALLOC_ARENA_MAX=2 python -m hexo_rl.bootstrap.pretrain \
    --epochs 30 \
    --batch-size 256 \
    --eta-min 5e-5 \
    --encoding v8 \
    --filters 128 \
    --res-blocks 12 \
    --gpool-sites 6,10 \
    --canvas-realness \
    --checkpoint-dir "${CKPT_DIR}" \
    --inference-out "${INFERENCE_OUT}" \
    --corpus-npz "${CORPUS}" \
    2>&1 | tee -a "${LOG}"

echo "[$(date -Iseconds)] §169 A4 canvas_realness retrain complete" | tee -a "${LOG}"
echo "Saved: ${INFERENCE_OUT}"
