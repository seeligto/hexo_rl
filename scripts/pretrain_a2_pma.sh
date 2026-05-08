#!/usr/bin/env bash
# §169 A2 retrain — v6w25 + PMA pool.
#
# Recipe matches the v6w25 anchor (§168 Gate 5) so the only architectural
# delta vs. A1 is pool_type: 'pma' replaces 'min_max'. Output checkpoint
# at checkpoints/ablation_169/A2_pma.pt; full pretrain log captured to
# reports/ablation_169/A2_pretrain.log.
#
# Hard surface conditions (§169 spec, P2 §A2):
#   - final loss > 5.36 (50% above v6w25 anchor 3.57): STOP, surface, no eval.
#   - NaN-skip rate > 30%: STOP even with skip patch.
#   - PMA collapse (argmax produces identical move regardless of K-th
#     cluster content on synthetic 2-cluster fixture): STOP, retry with
#     pool_attn_dropout=0.2.
#
# Run from repo root on the 5080 vast.ai host (see .claude/skills/rsync-vast/).
# Resume on the laptop is supported via --resume on `pretrain.py`.
set -euo pipefail

mkdir -p checkpoints/ablation_169 reports/ablation_169

CKPT_DIR="checkpoints/ablation_169/pretrain_a2"
INFERENCE_OUT="checkpoints/ablation_169/A2_pma.pt"
LOG="reports/ablation_169/A2_pretrain.log"

mkdir -p "${CKPT_DIR}"

# Match v6w25 anchor recipe: 30 ep cosine, peak 2e-3, eta_min 5e-5, batch 256.
# Pool-type override is the only architectural delta.
echo "[$(date -Iseconds)] §169 A2 PMA retrain starting" | tee -a "${LOG}"
echo "  ckpt-dir   : ${CKPT_DIR}"            | tee -a "${LOG}"
echo "  inference  : ${INFERENCE_OUT}"        | tee -a "${LOG}"
echo "  log        : ${LOG}"                  | tee -a "${LOG}"
echo "  recipe     : 30 ep cosine, peak 2e-3, eta_min 5e-5, batch 256, pma" | tee -a "${LOG}"

# `--inference-out` so the saved checkpoint name is the canonical A2 path.
# `--epochs 30 --eta-min 5e-5` matches the v6w25 anchor schedule.
# `--encoding v6w25` selects the 25×25 K-cluster wire format and corpus.
MALLOC_ARENA_MAX=2 python -m hexo_rl.bootstrap.pretrain \
    --epochs 30 \
    --batch-size 256 \
    --eta-min 5e-5 \
    --encoding v6w25 \
    --pool-type pma \
    --pool-attn-dropout 0.1 \
    --checkpoint-dir "${CKPT_DIR}" \
    --inference-out "${INFERENCE_OUT}" \
    --corpus-npz data/bootstrap_corpus_v6w25.npz \
    2>&1 | tee -a "${LOG}"

echo "[$(date -Iseconds)] §169 A2 PMA retrain complete" | tee -a "${LOG}"
echo "Saved: ${INFERENCE_OUT}"
