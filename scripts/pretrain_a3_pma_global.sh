#!/usr/bin/env bash
# §169 A3 retrain — v6w25 + PMA pool + global summary token branch.
#
# Recipe matches the v6w25 anchor (§168 Gate 5) and A2 PMA, with one
# architectural delta vs A2: pool_type 'pma_global' adds a (3, 32, 32)
# global-crop branch (cur/opp/canvas-mask → conv×2 → KataGo gpool →
# Linear → d=128 token; learned scalar gate init 0.1 multiplies before
# SAB concatenation).
#
# Output checkpoint at checkpoints/ablation_169/A3_pma_global.pt; full
# pretrain log captured to reports/ablation_169/A3_pretrain.log.
#
# Hard surface conditions (§169 P3 spec):
#   - final loss > 5.36 (50% above v6w25 anchor 3.57): STOP, surface.
#   - NaN-skip rate > 30%: STOP even with skip patch.
#   - PMA collapse on synthetic 2-cluster fixture: STOP, retry with
#     pool_attn_dropout=0.2.
#
# Soft surface (warning, not STOP — operator decides):
#   - Collapse onto global token (model ignores all K clusters): surface,
#     recommend attention entropy regularisation (out of §169 scope).
#   - Padding leak (model degrades when canvas-realness mask held out):
#     surface as soft warning unless A3 < A1.
#
# Prerequisite — corpus regen (~10 min on 5080):
#   python scripts/export_corpus_npz.py --encoding v6w25 \
#       --with-global-crop --human-only --no-compress \
#       --out data/bootstrap_corpus_v6w25_with_global.npz
# capture sha256 from the export-script tail; record in §169 P3 sprint log.
#
# Run from repo root on the 5080 vast.ai host (see .claude/skills/rsync-vast/).
set -euo pipefail

mkdir -p checkpoints/ablation_169 reports/ablation_169

CKPT_DIR="checkpoints/ablation_169/pretrain_a3"
INFERENCE_OUT="checkpoints/ablation_169/A3_pma_global.pt"
LOG="reports/ablation_169/A3_pretrain.log"
CORPUS="${CORPUS:-data/bootstrap_corpus_v6w25_with_global.npz}"

mkdir -p "${CKPT_DIR}"

if [[ ! -f "${CORPUS}" ]]; then
    echo "ERROR: A3 corpus missing: ${CORPUS}" >&2
    echo "       Regenerate via:" >&2
    echo "         python scripts/export_corpus_npz.py --encoding v6w25 \\" >&2
    echo "             --with-global-crop --human-only --no-compress \\" >&2
    echo "             --out ${CORPUS}" >&2
    exit 1
fi

echo "[$(date -Iseconds)] §169 A3 PMA+global retrain starting" | tee -a "${LOG}"
echo "  ckpt-dir   : ${CKPT_DIR}"           | tee -a "${LOG}"
echo "  inference  : ${INFERENCE_OUT}"       | tee -a "${LOG}"
echo "  corpus     : ${CORPUS}"              | tee -a "${LOG}"
echo "  log        : ${LOG}"                 | tee -a "${LOG}"
echo "  recipe     : 30 ep cosine, peak 2e-3, eta_min 5e-5, batch 256, pma_global" | tee -a "${LOG}"

MALLOC_ARENA_MAX=2 python -m hexo_rl.bootstrap.pretrain \
    --epochs 30 \
    --batch-size 256 \
    --eta-min 5e-5 \
    --encoding v6w25 \
    --pool-type pma_global \
    --pool-attn-dropout 0.1 \
    --checkpoint-dir "${CKPT_DIR}" \
    --inference-out "${INFERENCE_OUT}" \
    --corpus-npz "${CORPUS}" \
    2>&1 | tee -a "${LOG}"

echo "[$(date -Iseconds)] §169 A3 PMA+global retrain complete" | tee -a "${LOG}"
echo "Saved: ${INFERENCE_OUT}"
