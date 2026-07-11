#!/usr/bin/env bash
# D-C VALPROBE WP1 — vast launch script.
# Run via: bash scripts/valprobe/vast_launch.sh
# Launches both arms sequentially inside a tmux session (detached from ssh).
# Log: /workspace/hexo_rl/logs/valprobe_run.log
# Monitor: tail -f /workspace/hexo_rl/logs/valprobe_run.log
set -euo pipefail

REPO=/workspace/hexo_rl
VENV=$REPO/.venv/bin/python
LOG=$REPO/logs/valprobe_run.log
SESSION=valprobe

mkdir -p "$REPO/logs"
mkdir -p "$REPO/reports/valprobe/248k"
mkdir -p "$REPO/reports/valprobe/175k"

# Kill any stale session
tmux kill-session -t $SESSION 2>/dev/null || true

echo "[launch] Starting tmux session: $SESSION"
echo "[launch] Log: $LOG"

# Build the command to run inside tmux
CMD=$(cat <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
REPO=/workspace/hexo_rl
VENV=$REPO/.venv/bin/python
LOG=$REPO/logs/valprobe_run.log

echo "[$(date -u +%H:%M:%S)] === VALPROBE WP1 START ===" | tee -a $LOG

# ── ARM 248k ──────────────────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] Starting arm 248k" | tee -a $LOG
$VENV $REPO/scripts/valprobe/run_valprobe_vast.py \
  --arm 248k \
  --games $REPO/reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl \
  --ckpt $REPO/checkpoints/run2_retro/checkpoint_00248000.pt \
  --expect-encoding v6_live2_ls \
  --out $REPO/reports/valprobe/248k/ \
  2>&1 | tee -a $LOG

echo "[$(date -u +%H:%M:%S)] arm 248k DONE" | tee -a $LOG

# ── ARM 175k ──────────────────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] Starting arm 175k" | tee -a $LOG
$VENV $REPO/scripts/valprobe/run_valprobe_vast.py \
  --arm 175k \
  --games $REPO/reports/evalfair/retro_slope/run2_175k/games.jsonl \
  --ckpt $REPO/scripts/arena/weights/run2_175k.pt \
  --expect-encoding v6_live2_ls \
  --out $REPO/reports/valprobe/175k/ \
  2>&1 | tee -a $LOG

echo "[$(date -u +%H:%M:%S)] arm 175k DONE" | tee -a $LOG
echo "[$(date -u +%H:%M:%S)] === VALPROBE WP1 COMPLETE ===" | tee -a $LOG
INNER
)

# Write the inner script
cat > /tmp/valprobe_inner.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
REPO=/workspace/hexo_rl
VENV=$REPO/.venv/bin/python
LOG=$REPO/logs/valprobe_run.log

echo "[$(date -u +%H:%M:%S)] === VALPROBE WP1 START ===" | tee -a $LOG

echo "[$(date -u +%H:%M:%S)] Starting arm 248k" | tee -a $LOG
$VENV $REPO/scripts/valprobe/run_valprobe_vast.py \
  --arm 248k \
  --games $REPO/reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl \
  --ckpt $REPO/checkpoints/run2_retro/checkpoint_00248000.pt \
  --expect-encoding v6_live2_ls \
  --out $REPO/reports/valprobe/248k/ \
  2>&1 | tee -a $LOG
echo "[$(date -u +%H:%M:%S)] arm 248k DONE" | tee -a $LOG

echo "[$(date -u +%H:%M:%S)] Starting arm 175k" | tee -a $LOG
$VENV $REPO/scripts/valprobe/run_valprobe_vast.py \
  --arm 175k \
  --games $REPO/reports/evalfair/retro_slope/run2_175k/games.jsonl \
  --ckpt $REPO/scripts/arena/weights/run2_175k.pt \
  --expect-encoding v6_live2_ls \
  --out $REPO/reports/valprobe/175k/ \
  2>&1 | tee -a $LOG
echo "[$(date -u +%H:%M:%S)] arm 175k DONE" | tee -a $LOG

echo "[$(date -u +%H:%M:%S)] === VALPROBE WP1 COMPLETE ===" | tee -a $LOG
INNER
chmod +x /tmp/valprobe_inner.sh

# Launch detached tmux
tmux new-session -d -s $SESSION "bash /tmp/valprobe_inner.sh"
echo "[launch] Session '$SESSION' started. Monitor with:"
echo "  ssh vast 'tail -f /workspace/hexo_rl/logs/valprobe_run.log'"
echo "  ssh vast 'tmux attach -t valprobe'"
