#!/usr/bin/env bash
# D-C VALPROBE WP1 re-run — SealBot T_provable (point-of-no-return).
# Run ON VAST via: bash scripts/valprobe/vast_launch_sealbot.sh
# Launches both arms sequentially inside a tmux session (detached from ssh).
# Log: /workspace/hexo_rl/logs/valprobe_sealbot.log
# Monitor: tail -f /workspace/hexo_rl/logs/valprobe_sealbot.log
#          tmux attach -t valprobe_sb
set -euo pipefail

REPO=/workspace/hexo_rl
VENV=$REPO/.venv/bin/python
LOG=$REPO/logs/valprobe_sealbot.log
SESSION=valprobe_sb

mkdir -p "$REPO/logs"
mkdir -p "$REPO/reports/valprobe/248k"
mkdir -p "$REPO/reports/valprobe/175k"

# Kill any stale session
tmux kill-session -t $SESSION 2>/dev/null || true

echo "[launch] Starting tmux session: $SESSION"
echo "[launch] Log: $LOG"

cat > /tmp/valprobe_sb_inner.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
REPO=/workspace/hexo_rl
VENV=$REPO/.venv/bin/python
LOG=$REPO/logs/valprobe_sealbot.log

echo "[$(date -u +%H:%M:%S)] === VALPROBE WP1 SealBot START ===" | tee -a $LOG

# ── ARM 248k ──────────────────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] Starting arm 248k (SealBot d6-d8, window_half=9)" | tee -a $LOG
$VENV $REPO/scripts/valprobe/run_valprobe_sealbot.py \
  --arm 248k \
  --games $REPO/reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl \
  --ckpt $REPO/checkpoints/run2_retro/checkpoint_00248000.pt \
  --expect-encoding v6_live2_ls \
  --out $REPO/reports/valprobe/248k/ \
  --depths 6,7,8 \
  --window-half 9 \
  --workers 20 \
  --game-timeout 600 \
  2>&1 | tee -a $LOG

echo "[$(date -u +%H:%M:%S)] arm 248k DONE" | tee -a $LOG

# ── ARM 175k ──────────────────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] Starting arm 175k (SealBot d6-d8, window_half=9)" | tee -a $LOG
$VENV $REPO/scripts/valprobe/run_valprobe_sealbot.py \
  --arm 175k \
  --games $REPO/reports/evalfair/retro_slope/run2_175k/games.jsonl \
  --ckpt $REPO/scripts/arena/weights/run2_175k.pt \
  --expect-encoding v6_live2_ls \
  --out $REPO/reports/valprobe/175k/ \
  --depths 6,7,8 \
  --window-half 9 \
  --workers 20 \
  --game-timeout 600 \
  2>&1 | tee -a $LOG

echo "[$(date -u +%H:%M:%S)] arm 175k DONE" | tee -a $LOG
echo "[$(date -u +%H:%M:%S)] === VALPROBE WP1 SealBot COMPLETE ===" | tee -a $LOG
INNER
chmod +x /tmp/valprobe_sb_inner.sh

tmux new-session -d -s $SESSION "bash /tmp/valprobe_sb_inner.sh"
echo "[launch] Session '$SESSION' started. Monitor with:"
echo "  tail -f $LOG"
echo "  tmux attach -t $SESSION"
