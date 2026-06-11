#!/usr/bin/env bash
# §D-WALLCAUSATION Phase A — regenerate single-window self-play across the
# archived colony-collapse checkpoint trajectories (s180b, s179) + a healthy
# contrast, for per-checkpoint off-window-blind-win measurement.
#
# Sequential (one GPU); each checkpoint writes its own jsonl incrementally so a
# kill leaves partial-but-usable data. Run in background:
#   bash scripts/diagnosis/wallcausation_gen_driver.sh &> logs/wc_gen.log
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
GEN=scripts/diagnosis/wallcausation_selfplay_gen.py
OUT=reports/investigations/wallcausation_data
mkdir -p "$OUT"
N=70
SIMS=128
TEMP=1.0
OP=2

gen() {  # run=$1 step=$2 ckpt=$3
  local run="$1" step="$2" ckpt="$3"
  if [[ ! -f "$ckpt" ]]; then echo "[driver] MISSING $ckpt — skip"; return; fi
  echo "[driver] === $run step=$step ckpt=$ckpt ==="
  $PY "$GEN" --checkpoint "$ckpt" --step "$step" --n-games "$N" --sims "$SIMS" \
      --temp "$TEMP" --opening-plies "$OP" --out "$OUT/${run}_step$(printf '%08d' "$step").jsonl"
}

S180=archive/s180b_3knob_fail/ckpts
gen s180b 10000 "$S180/ckpt_step00010000.pt"
gen s180b 20000 "$S180/ckpt_step00020000.pt"
gen s180b 30000 "$S180/ckpt_step00030000.pt"
gen s180b 40000 "$S180/ckpt_step00040000.pt"
gen s180b 50000 "$S180/ckpt_step00050000.pt"
gen s180b 53500 "$S180/ckpt_step00053500.pt"

S179=archive/s179_recipe_fail/ckpts
gen s179 10000 "$S179/ckpt_step10k.pt"
gen s179 20000 "$S179/ckpt_step20k_peak.pt"
gen s179 30000 "$S179/ckpt_step30k.pt"
gen s179 40000 "$S179/ckpt_step40k.pt"
gen s179 50000 "$S179/ckpt_step50k.pt"
gen s179 60000 "$S179/ckpt_step60k.pt"

# healthy / non-colony contrast (v6_live2 adopted config — colony SUPPRESSED)
gen v6live2 30000 "checkpoints/v6_live2_rl/checkpoint_00030000.pt"

echo "[driver] ALL DONE"
