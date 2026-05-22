#!/usr/bin/env bash
# §S181 FU-1.5 — finer-ladder re-run of §S180b. vast 5080 + Ryzen 9 9900X.
#
# Re-runs the §S180b 3-knob config VERBATIM (variant v6_botmix_s180b_2kckpt,
# single delta anchor_every_steps 5000→2000) and STOPS at step 20000 so the
# value-spread ladder has a checkpoint every 2k steps inside the first 20k.
# Resolves FU-1's load-bearing ambiguity: STEP-0-ONSET vs MID-CLIFF vs GRADUAL.
#
# Pre-flight (operator):
#   1. tmux kill-session -t sS180a   (stale §S180a session — closed/FAILED)
#   2. checkpoints/replay_buffer.bin ABSENT (clean start — verified absent)
#   3. data/bootstrap_corpus_v6.npz + data/bot_corpus_s178_sealbot_vs_v6.npz present
#   4. configs/variants/v6_botmix_s180b_2kckpt.yaml present (scp'd from dev)
#   5. --checkpoint MUST be bootstrap_model_v6.pt (NOT bootstrap_model.pt — random)
#
# Run inside tmux:  tmux new -s s181fu15 'bash scripts/s181_fu1_5_launch.sh'
# Wall ~12 h. Stops itself at step 20000 via the checkpoint watchdog.
set -euo pipefail
cd /workspace/hexo_rl
source .venv/bin/activate
export PYTHONPATH=.
mkdir -p logs/s181_fu1_5

STOP_STEP=20000
LOG="logs/s181_fu1_5/train.log"

echo "=== §S181 FU-1.5 finer-ladder launch — $(date -u) ==="
echo "variant: v6_botmix_s180b_2kckpt   anchor: bootstrap_model_v6.pt"
echo "stop step: ${STOP_STEP}   log: ${LOG}"

python scripts/train.py \
  --checkpoint checkpoints/bootstrap_model_v6.pt \
  --variant v6_botmix_s180b_2kckpt \
  --iterations 100000 > "${LOG}" 2>&1 &
TRAIN_PID=$!
echo "trainer PID ${TRAIN_PID}"

# Watchdog — graceful stop once the step-20k checkpoint has landed.
STOP_CKPT="checkpoints/checkpoint_$(printf '%08d' ${STOP_STEP}).pt"
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  if [ -f "${STOP_CKPT}" ]; then
    echo "$(date -u): ${STOP_CKPT} present — graceful stop in 30s"
    sleep 30   # let the checkpoint save + value-spread canary finish
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    break
  fi
  sleep 60
done
wait "${TRAIN_PID}" 2>/dev/null || true

echo "=== FU-1.5 run complete — $(date -u) ==="
echo "preserved checkpoints (anchor_every_steps=2000):"
ls -la checkpoints/checkpoint_000{02,04,06,08,10,12,14,16,18,20}000.pt 2>/dev/null || true
