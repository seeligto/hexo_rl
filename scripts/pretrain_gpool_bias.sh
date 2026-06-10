#!/usr/bin/env bash
# §170 P3 retrain — A1 v6w25 + gpool-bias side-branch.
#
# Architecture: canonical A1 (K-cluster + min/max pool) + KataGo-style
# additive global-pool bias to value_fc1 hidden + policy_fc logits.
# Gate scalar init 0.0 → byte-exact A1 at construction (commit-1 unit
# test test_gate_zero_byte_exact_a1 enforces this on bootstrap_model_v6w25.pt).
#
# Recipe matches v6w25 anchor (§168 Gate 5) and A3 (§169 P3) — only
# architectural delta vs A1 is gpool_bias_active=true; corpus is the
# A3 v6w25-with-global-crop NPZ (sha256 e2876ae5...) reused verbatim.
#
# Hard surface conditions (§170 P3 spec):
#   - final loss > 5.36 (50% above A1 anchor 3.57): STOP, surface.
#   - NaN-skip rate > 30%: STOP even with §167 patch.
#   - Forward-parity test failing on the post-train checkpoint: STOP
#     (commit-1 unit test verifies gate=0 byte-exact; if a checkpoint
#     fails parity post-train via gate>0, that's expected — but the
#     ARCHITECTURE invariant must hold by re-loading state_dict + zeroing
#     the gate).
#
# Soft surface (warning, not STOP):
#   - Gate scalar < 0.05 at end of training: global branch never earned
#     weight — flag in verdict as null result; eval still proceeds.
#
# Prerequisite — corpus regen (~10 min on 5080):
#   python scripts/export_corpus_npz.py --encoding v6w25 \
#       --with-global-crop --human-only --no-compress \
#       --out data/bootstrap_corpus_v6w25_with_global.npz
# (sha256 from §169 A3 closeout: e2876ae5639958dac3758274b7137faeaff91713fe50df6da04ea43dfd896793)
#
# Run from repo root on the remote 5080 host.
set -euo pipefail

mkdir -p checkpoints/gpool_bias reports/gpool_bias

CKPT_DIR="checkpoints/gpool_bias/pretrain"
INFERENCE_OUT="checkpoints/gpool_bias/A1_gpool_bias.pt"
LOG="reports/gpool_bias/pretrain.log"
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

echo "[$(date -Iseconds)] §170 P3 A1+gpool-bias retrain starting" | tee -a "${LOG}"
echo "  ckpt-dir   : ${CKPT_DIR}"           | tee -a "${LOG}"
echo "  inference  : ${INFERENCE_OUT}"       | tee -a "${LOG}"
echo "  corpus     : ${CORPUS}"              | tee -a "${LOG}"
echo "  log        : ${LOG}"                 | tee -a "${LOG}"
echo "  recipe     : 30 ep cosine, peak 2e-3, eta_min 5e-5, batch 256, min_max + gpool_bias_active" | tee -a "${LOG}"

MALLOC_ARENA_MAX=2 .venv/bin/python -m hexo_rl.bootstrap.pretrain \
    --epochs 30 \
    --batch-size 256 \
    --eta-min 5e-5 \
    --encoding v6w25 \
    --pool-type min_max \
    --gpool-bias-active \
    --checkpoint-dir "${CKPT_DIR}" \
    --inference-out "${INFERENCE_OUT}" \
    --corpus-npz "${CORPUS}" \
    2>&1 | tee -a "${LOG}"

echo "[$(date -Iseconds)] §170 P3 A1+gpool-bias retrain complete" | tee -a "${LOG}"
echo "Saved: ${INFERENCE_OUT}"
