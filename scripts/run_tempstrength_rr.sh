#!/usr/bin/env bash
# D-TEMPSTRENGTH post-hoc head-to-head round_robin (run on vast after both arms finish).
#
# Assembles ONE archive dir keyed by distinct step ints (round_robin keys competitors
# by step), runs an all-pairs round_robin at eval temp 0.0 (argmax = the TRAINED model)
# with on-distribution opening-jitter (§D-ARGMAX argmax-at-power → effective-n), then
# aggregates to BT-Elo + win-matrix + distinct-games CI.
#
#   competitor step-key → meaning
#     50000  (s50k)  = golong@50k PEAK anchor (drift reference)
#     55/60/65k      = CONTROL arm rungs
#     155/160/165k   = a20 arm rungs (offset +100000 to avoid step collision)
#
# PRIMARY verdict pair = a20@X vs control@X  (s155k–s55k, s160k–s60k, s165k–s65k).
# Pre-committed budget (fixed before any a20 result existed): n_games=60/pair,
# sims=128, temp=0.0, opening_jitter_plies=8 @ temp 0.5, max_plies=200.
set -euo pipefail
cd /workspace/hexo_rl
source ~/.cargo/env
export VIRTUAL_ENV="$PWD/.venv"
PY=.venv/bin/python

ARCH=rr_archive
OUT=reports/tempstrength_rr
ANCHOR=checkpoints/golong_bank/checkpoint_00050000_PEAK_sb0.38.pt

rm -rf "$ARCH"; mkdir -p "$ARCH" "$OUT"
cp "$ANCHOR" "$ARCH/checkpoint_00050000.pt"
for s in 00055000 00060000 00065000; do
  cp "checkpoints_tstr_control/checkpoint_${s}.pt" "$ARCH/checkpoint_${s}.pt"
done
cp checkpoints_tstr_a20/checkpoint_00055000.pt "$ARCH/checkpoint_00155000.pt"
cp checkpoints_tstr_a20/checkpoint_00060000.pt "$ARCH/checkpoint_00160000.pt"
cp checkpoints_tstr_a20/checkpoint_00065000.pt "$ARCH/checkpoint_00165000.pt"
echo "=== archive assembled ==="; ls -la "$ARCH"

RUNGS=50000,55000,60000,65000,155000,160000,165000

echo "=== round_robin play (all-pairs, n=60, temp 0.0, jitter 8) ==="
$PY scripts/eval_round_robin.py play \
  --archive "$ARCH" --rungs "$RUNGS" \
  --n-games 60 --sims 128 --temp 0.0 \
  --opening-jitter-plies 8 --opening-jitter-temp 0.5 \
  --max-plies 200 --output "$OUT"

echo "=== aggregate ==="
$PY scripts/eval_round_robin.py aggregate --inputs "$OUT" --output "$OUT"
echo "=== RR_EVAL_DONE ==="
