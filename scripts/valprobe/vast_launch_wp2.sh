#!/usr/bin/env bash
# D-C VALPROBE WP2 — card1 probe-set expansion to ≥200 distinct positions.
# Run ON VAST via: bash scripts/valprobe/vast_launch_wp2.sh
# Log: /workspace/hexo_rl/logs/valprobe_wp2.log
# Monitor: tail -f /workspace/hexo_rl/logs/valprobe_wp2.log
#          tmux attach -t valprobe_wp2
set -euo pipefail

REPO=/workspace/hexo_rl
VENV=$REPO/.venv/bin/python
LOG=$REPO/logs/valprobe_wp2.log
SESSION=valprobe_wp2

mkdir -p "$REPO/logs"
mkdir -p "$REPO/reports/valprobe/wp2"

# Kill any stale WP2 session
tmux kill-session -t $SESSION 2>/dev/null || true

echo "[launch] Starting tmux session: $SESSION"
echo "[launch] Log: $LOG"

cat > /tmp/valprobe_wp2_inner.sh <<'INNER'
#!/usr/bin/env bash
set -euo pipefail
REPO=/workspace/hexo_rl
VENV=$REPO/.venv/bin/python
LOG=$REPO/logs/valprobe_wp2.log

echo "[$(date -u +%H:%M:%S)] === VALPROBE WP2 START ===" | tee -a $LOG

$VENV $REPO/scripts/valprobe/wp2_expand_probe_set.py \
  --ckpt $REPO/checkpoints/run2_retro/checkpoint_00248000.pt \
  --expect-encoding v6_live2_ls \
  --existing $REPO/reports/valprobe/card1_probe_set.jsonl \
  --out $REPO/reports/valprobe/wp2 \
  --final-out $REPO/reports/valprobe/probe_set_v1.jsonl \
  --workers 8 \
  --target 200 \
  --max-batches 3 \
  --batch-openings 64 \
  2>&1 | tee -a $LOG

echo "[$(date -u +%H:%M:%S)] === VALPROBE WP2 COMPLETE ===" | tee -a $LOG
INNER
chmod +x /tmp/valprobe_wp2_inner.sh

tmux new-session -d -s $SESSION "bash /tmp/valprobe_wp2_inner.sh"
echo "[launch] Session '$SESSION' started. Monitor with:"
echo "  tail -f $LOG"
echo "  tmux attach -t $SESSION"
