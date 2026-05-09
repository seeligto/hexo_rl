#!/usr/bin/env bash
# §170 P4 P1 retrain — A1 v6w25 + gpool-bias policy-only.
#
# Architecture: canonical A1 (K-cluster + min/max pool) + KataGo-style
# additive global-pool bias — POLICY HEAD ONLY. value_bias is hardcoded
# to a fresh zero tensor with no autograd path; value_proj is in the
# state-dict for §170 P3 round-trip but never invoked at forward time
# and receives no gradient. Gate scalar (init 0.0) drives the policy
# path only — gate=0 byte-exact A1 at construction (verified by
# tests/test_gpool_bias_policy_only.py::test_forward_parity_v6w25_anchor).
#
# Recipe identical to §170 P3: 30 ep cosine, peak 2e-3, eta_min 5e-5,
# batch 256, fp16. Corpus = bootstrap_corpus_v6w25_with_global.npz
# (sha256 e2876ae5...) reused verbatim. Only delta vs §170 P3 is the
# --policy-only-bias flag.
#
# Hard surface conditions (§170 P4 spec):
#   - final loss > 5.36 (50% above A1 anchor 3.57): STOP, surface.
#   - NaN-skip rate > 30%: STOP.
#   - policy_gate stuck at ~0 (final < 0.05): bias didn't earn weight,
#     flag in verdict but continue eval — null-result data point.
#
# Soft surface (warning, not STOP):
#   - policy_gate trajectory logged per-step.
#
# Run from repo root on the 5080 vast.ai host.
set -euo pipefail

mkdir -p checkpoints/gpool_bias reports/gpool_bias

CKPT_DIR="checkpoints/gpool_bias/pretrain_policy_only"
INFERENCE_OUT="checkpoints/gpool_bias/A1_gpool_bias_policy_only.pt"
LOG="reports/gpool_bias/policy_only_pretrain.log"
CORPUS="${CORPUS:-data/bootstrap_corpus_v6w25_with_global.npz}"

mkdir -p "${CKPT_DIR}"

if [[ ! -f "${CORPUS}" ]]; then
    echo "ERROR: gpool-bias corpus missing: ${CORPUS}" >&2
    echo "       Regenerate via:" >&2
    echo "         python scripts/export_corpus_npz.py --encoding v6w25 \\" >&2
    echo "             --with-global-crop --human-only --no-compress \\" >&2
    echo "             --out ${CORPUS}" >&2
    echo "       (sha256 e2876ae5... per §169 A3 closeout)" >&2
    exit 1
fi

echo "[$(date -Iseconds)] §170 P4 P1 A1+gpool-bias-policy-only retrain starting" | tee -a "${LOG}"
echo "  ckpt-dir   : ${CKPT_DIR}"           | tee -a "${LOG}"
echo "  inference  : ${INFERENCE_OUT}"       | tee -a "${LOG}"
echo "  corpus     : ${CORPUS}"              | tee -a "${LOG}"
echo "  log        : ${LOG}"                 | tee -a "${LOG}"
echo "  recipe     : 30 ep cosine, peak 2e-3, eta_min 5e-5, batch 256, min_max + gpool_bias_active + policy_only_bias" | tee -a "${LOG}"

MALLOC_ARENA_MAX=2 .venv/bin/python -m hexo_rl.bootstrap.pretrain \
    --epochs 30 \
    --batch-size 256 \
    --eta-min 5e-5 \
    --encoding v6w25 \
    --pool-type min_max \
    --gpool-bias-active \
    --policy-only-bias \
    --checkpoint-dir "${CKPT_DIR}" \
    --inference-out "${INFERENCE_OUT}" \
    --corpus-npz "${CORPUS}" \
    2>&1 | tee -a "${LOG}"

echo "[$(date -Iseconds)] §170 P4 P1 A1+gpool-bias-policy-only retrain complete" | tee -a "${LOG}"
echo "Saved: ${INFERENCE_OUT}"
