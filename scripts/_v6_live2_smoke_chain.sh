#!/usr/bin/env bash
# §P5-CT — wait for the in-flight v6_live2 pretrain (running standalone) to
# produce the bootstrap, then launch the 30k v6_live2 smoke on the LAPTOP GPU.
# Guarded: polls for the checkpoint + pretrain process exit before launching.
set -uo pipefail
cd /home/timmy/Work/hexo_rl
CKPT=checkpoints/bootstrap_model_v6_live2.pt
PLOG=logs/v6_live2_pretrain.log
SLOG=logs/v6_live2_smoke.log
echo "[chain] $(date -u) waiting for pretrain to finish ($CKPT) ..."
while true; do
  # done when the bootstrap exists, is size-stable, and no pretrain proc remains
  if [ -f "$CKPT" ] \
     && ! pgrep -f "export_corpus_npz.py --encoding v6_live2" >/dev/null \
     && ! pgrep -f "bootstrap.pretrain --encoding v6_live2" >/dev/null; then
    s1=$(stat -c%s "$CKPT" 2>/dev/null || echo 0); sleep 15
    s2=$(stat -c%s "$CKPT" 2>/dev/null || echo 1)
    [ "$s1" = "$s2" ] && [ "$s1" != "0" ] && break
  fi
  # bail if pretrain clearly failed (no checkpoint and no live process)
  if [ ! -f "$CKPT" ] \
     && ! pgrep -f "export_corpus_npz.py --encoding v6_live2" >/dev/null \
     && ! pgrep -f "bootstrap.pretrain --encoding v6_live2" >/dev/null; then
    if grep -qiE "Traceback|Error|Aborted" "$PLOG" 2>/dev/null; then
      echo "[chain] pretrain appears to have FAILED (no ckpt, dead proc, error in log) — aborting"; exit 1
    fi
  fi
  sleep 30
done
echo "[chain] $(date -u) pretrain done (ckpt=$(stat -c%s "$CKPT") bytes); launching 30k v6_live2 laptop smoke"
.venv/bin/python scripts/train.py \
  --checkpoint "$CKPT" \
  --variant v6_live2_smoke_laptop \
  --iterations 30000 2>&1 | tee -a "$SLOG"
echo "[chain] V6_LIVE2_SMOKE_EXIT=${PIPESTATUS[0]} at $(date -u)" | tee -a "$SLOG"
