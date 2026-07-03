#!/usr/bin/env bash
# armc50k_status.sh — local-only, READ-ONLY progress check for the Arm-C 50k run.
#
# Run it on the laptop: ./scripts/armc50k_status.sh
# SSHes to the vast host (alias `vast` in ~/.ssh/config; override with VAST_HOST=...),
# reads the live log + checkpoints, mutates NOTHING. Safe to run any time / repeatedly.
#
#   watch -n 60 ./scripts/armc50k_status.sh    # refresh every 60s
set -uo pipefail
HOST="${VAST_HOST:-vast}"

ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST" 'bash -s' <<'REMOTE' 2>&1 | grep -avE '^(Welcome to vast|Have fun)'
set -uo pipefail
cd /workspace/hexo_rl 2>/dev/null || { echo "FATAL: /workspace/hexo_rl missing on host"; exit 1; }
STOP=50000
L=$(ls -t armc50k_*.log 2>/dev/null | head -1)
[ -z "${L:-}" ] && { echo "no armc50k_*.log on host"; exit 1; }

pgrep -f "train.py --checkpoint" >/dev/null && ALIVE="RUNNING" || ALIVE="STOPPED"
TERM=$(grep -ac "terminal_eval_complete" "$L"); TERM=${TERM:-0}

first=$(grep -aE '"event": "train_step"' "$L" | head -1)
last=$(grep -aE '"event": "train_step"' "$L" | tail -1)
fs=$(echo "$first" | grep -aoE '"step": [0-9]+' | grep -oE '[0-9]+')
cs=$(echo "$last"  | grep -aoE '"step": [0-9]+' | grep -oE '[0-9]+')
ft=$(echo "$first" | grep -aoE '"timestamp": "[^"]+"' | sed -E 's/.*"([^"]+)".*/\1/')
lt=$(echo "$last"  | grep -aoE '"timestamp": "[^"]+"' | sed -E 's/.*"([^"]+)".*/\1/')
fe=$(date -d "$ft" +%s 2>/dev/null || echo 0)
le=$(date -d "$lt" +%s 2>/dev/null || echo 0)

rate="?"; eta="?"
if [ "${le:-0}" -gt "${fe:-0}" ] && [ "${cs:-0}" -gt "${fs:-0}" ]; then
  rate=$(awk -v a="$fs" -v b="$cs" -v t="$((le-fe))" 'BEGIN{printf "%.0f",(b-a)/t*3600}')
  rmin=$(awk -v s="$cs" -v stop="$STOP" -v r="$rate" 'BEGIN{if(r>0)printf "%d",(stop-s)/r*60; else print 0}')
  eta=$(date -u -d "+${rmin} minutes" "+%Y-%m-%d %H:%M UTC" 2>/dev/null || echo "?")
fi
pct=$(awk -v s="${cs:-0}" -v t="$STOP" 'BEGIN{printf "%.0f",s/t*100}')

dr=$(grep -aoE '"draw_rate": [0-9.]+' "$L" | tail -1 | grep -oE '[0-9.]+')
va=$(grep -aoE '"value_accuracy_masked": [0-9.]+' "$L" | tail -1 | grep -oE '[0-9.]+')
fwc=$(grep -aoE '"forced_win_conversion": [0-9.]+' "$L" | tail -1 | grep -oE '[0-9.]+')

ab=$(grep -acE '"event": "hard_abort' "$L"); ab=${ab:-0}
err=$(grep -acE 'Traceback|panicked|CUDA error|out of memory' "$L"); err=${err:-0}
proms=$(grep -acE '"promoted": true' "$L"); proms=${proms:-0}

er=$(grep -aE '"event": "evaluation_round_complete"' "$L" | tail -1)
estep=$(echo "$er" | grep -aoE '"step": [0-9]+' | grep -oE '[0-9]+')
val(){ echo "$er" | grep -aoE "\"$1\": [0-9eE.+-]+" | tail -1 | sed -E 's/.*: //'; }
eprom=$(echo "$er" | grep -aoE '"promoted": (true|false)' | sed -E 's/.*: //')

CK=$(ls -t checkpoints/checkpoint_0*.pt 2>/dev/null | head -1)
bm=$(ls -l --time-style=+%Y-%m-%dT%H:%M checkpoints/best_model.pt 2>/dev/null | awk '{print $6}')

printf '════════ Arm-C 50k (v6_live2_ls) ════════\n'
printf 'state      : %s' "$ALIVE"
[ "${TERM:-0}" -gt 0 ] && printf '   ✓ terminal_eval_complete' || printf '   (terminal eval: not yet)'
printf '\n'
printf 'step       : %s / %s  (~%s%%)\n' "${cs:-?}" "$STOP" "$pct"
printf 'rate / ETA : %s steps/hr  →  finish ~%s\n' "$rate" "$eta"
printf 'draw_rate  : %s   (HARD-ABORT ≥0.55 ×3)\n' "${dr:-?}"
printf 'value_acc  : %s\n' "${va:-?}"
printf 'forced_win : %s   (coherence frontier; eval-round only)\n' "${fwc:-n/a}"
printf 'health     : hard_abort=%s  errors=%s  promotions=%s\n' "$ab" "$err" "$proms"
printf 'latest ckpt: %s\n' "${CK#checkpoints/}"
printf 'best_model : %s  %s\n' "${bm:-?}" "$([ "${proms:-0}" -gt 0 ] && echo '(PROMOTED)' || echo '(= bootstrap fresh-init / pin holds)')"
printf '─── last eval round (step %s) ───\n' "${estep:-none}"
if [ -n "${er:-}" ]; then
  printf 'promoted=%s  best=%s  sealbot=%s  anchor=%s  random=%s\n' \
    "${eprom:-?}" "$(val wr_best)" "$(val wr_sealbot)" "$(val wr_bootstrap_anchor)" "$(val wr_random)"
else
  printf '(no eval round completed yet — first fires at step 12500)\n'
fi
REMOTE
