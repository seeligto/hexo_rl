#!/usr/bin/env bash
# v9_smokes_remote_status.sh — one-shot remote-status reporter for the
# Phase B' v9 §153 T5 smoke runner.  Designed to be invoked locally
# every 30 min by `claude /loop` (or a wrapper) and pulls just enough
# from the vast.ai remote to print:
#
#   * which smoke is running (or "ALL DONE")
#   * current iteration / total
#   * elapsed / projected steps-per-hour
#   * draw_rate so far + grad_norm (hard-abort safety)
#
# Usage:
#   scripts/v9_smokes_remote_status.sh ssh<N>.vast.ai:<PORT>
#
# Required SSH key + known_hosts already configured per .claude/skills/rsync-vast/SKILL.md.

set -euo pipefail

HOSTPORT="${1:-}"
if [[ -z "$HOSTPORT" ]]; then
    echo "Usage: $0 sshN.vast.ai:PORT" >&2
    exit 2
fi
HOST="${HOSTPORT%:*}"
PORT="${HOSTPORT#*:}"
if [[ "$PORT" == "$HOST" ]]; then PORT=22; fi

SSH_OPTS=(-p "$PORT" -i "$HOME/.ssh/vast_hexo" -o IdentitiesOnly=yes -o UserKnownHostsFile="$HOME/.ssh/known_hosts_vast")

# Run a tiny script remotely that emits a one-line summary per smoke.
ssh "${SSH_OPTS[@]}" "root@$HOST" bash -s <<'REMOTE'
set -euo pipefail
cd /workspace/hexo_rl 2>/dev/null || cd ~/hexo_rl

DRIVER_LOG="logs/v9_smokes_runner.driver.log"
if [[ ! -f "$DRIVER_LOG" ]]; then
    echo "[remote] driver log not found ($DRIVER_LOG); runner may not have started"
    exit 0
fi

# Last 3 lines of driver log give context on which smoke is current.
echo "── driver tail ──"
tail -n 5 "$DRIVER_LOG"

# Find the latest active train.py log.
LATEST="$(ls -1t logs/v9_S*.log 2>/dev/null | head -n 1 || true)"
if [[ -z "$LATEST" ]]; then
    echo "[remote] no v9_S*.log files yet"
    exit 0
fi
echo
echo "── current train log: $LATEST ──"

# Parse iteration / steps-per-hour from structlog lines emitted by the
# trainer.  The trainer logs `step_complete` with `train_steps_total` and
# `iter_per_min`; pull the most recent occurrence.
LAST_STEP_LINE="$(tac "$LATEST" 2>/dev/null | grep -m 1 -E 'step_complete|train_step|elapsed_steps' || true)"
if [[ -n "$LAST_STEP_LINE" ]]; then
    echo "$LAST_STEP_LINE"
fi

# Grad-norm hard-abort safety: any line above the gate?
HIGH_GN="$(grep -E 'grad_norm.*[1-9][0-9]+\.|hard_abort|abort_grad_norm' "$LATEST" | tail -n 3 || true)"
if [[ -n "$HIGH_GN" ]]; then
    echo "── grad-norm warnings ──"
    echo "$HIGH_GN"
fi

# Recent draw_rate / colony fraction snapshots.
echo
echo "── recent metrics ──"
tac "$LATEST" 2>/dev/null | grep -m 5 -E 'draw_rate|colony_extension_fraction|stride5|sims_per_sec' || true

# GPU util (nvidia-smi is on every vast.ai box).
echo
echo "── gpu ──"
nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
REMOTE
