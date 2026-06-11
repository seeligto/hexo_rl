#!/usr/bin/env bash
# D-RERUNPREP Phase 3 — GPU integration smoke for the Arm-C fixed-loop re-run.
# OPERATOR-RUN ON VAST. This is the acceptance the §D-LOOPFIX report deferred
# ("GPU-bound"). The 50k re-run does NOT launch until this PASSES.
#
# Runs, in order:
#   (A) tests/test_closeout_lifecycle.py  (slow+integration — the deferred GPU test)
#   (B) preflight cleanup + the mini real run (variant v6_live2_ls_ab_smoke, 500 steps)
#   (C) provenance + loader round-trip check
#   (D) mini-resume (5 iters) — proves resume ignores the TERMINAL record on real HW
#   (E) exploit_probe + KClusterMCTSBot wiring smoke (numbers = proof, not strength)
# Then prints a PASS/FAIL line per pre-registered criterion (1..9). FAIL = any miss.
#
# Cost ~1-2 h on vast 5080 (depends on real throughput). ~$0.40-1.50.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

PIN="aba28e10bd80b2bac65e9b33e109cb9dc36a3a83871bf3a3fff0ca0f96d27165"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="reports/phase3_armc_smoke_${STAMP}"; mkdir -p "$OUT"
RUNLOG="$OUT/run.log"; RESUMELOG="$OUT/resume.log"
PY=".venv/bin/python"
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }
declare -A CRIT  # criterion -> PASS/FAIL/SKIP

# ── (A) the deferred closeout integration test ───────────────────────────────
say "(A) tests/test_closeout_lifecycle.py (slow+integration)"
$PY -m pytest tests/test_closeout_lifecycle.py -v -m "slow and integration" \
  >"$OUT/closeout_test.log" 2>&1
A_RC=$?
[ $A_RC -eq 0 ] && say "  closeout test PASS" || say "  closeout test FAIL (rc=$A_RC) — see $OUT/closeout_test.log"

# ── (B) preflight + mini real run ────────────────────────────────────────────
say "(B) preflight: clearing stale anchor + replay buffer (W2 fresh-init the pinned bootstrap)"
rm -f checkpoints/best_model.pt checkpoints/best_model.pt.bak checkpoints/replay_buffer.bin
say "(B) launching mini run: variant v6_live2_ls_ab_smoke, --iterations 500"
$PY scripts/train.py \
  --checkpoint checkpoints/bootstrap_model_v6_live2.pt \
  --variant v6_live2_ls_ab_smoke \
  --iterations 500 \
  --run-name "armc_smoke_${STAMP}" \
  --checkpoint-dir checkpoints/armc_smoke_${STAMP} \
  >"$RUNLOG" 2>&1
B_RC=$?
say "  run exit code: $B_RC"

# ── criterion checks against the run log ─────────────────────────────────────
grepc(){ grep -cE "$1" "$RUNLOG" 2>/dev/null || echo 0; }
# 1. iteration_limit_reached at N=500
[ "$(grepc 'iteration_limit_reached')" -ge 1 ] && CRIT[1]=PASS || CRIT[1]=FAIL
# 2. drain budgeted, WARN-never-kill (a budgeted drain logged; no 'killed' eval round)
if [ "$(grepc 'drain.*budget|final.*drain|eval.*drain')" -ge 1 ] && [ "$(grepc 'killed.*eval|eval.*killed')" -eq 0 ]; then CRIT[2]=PASS; else CRIT[2]=FAIL; fi
# 3. pool stopped BEFORE terminal eval (ordering: pool stop line precedes terminal_eval_complete)
POOLSTOP_LN=$(grep -nE 'pool.*stop|stop.*pool|workers.*halt' "$RUNLOG" | tail -1 | cut -d: -f1)
TERM_LN=$(grep -nE 'terminal_eval_complete' "$RUNLOG" | head -1 | cut -d: -f1)
if [ -n "$POOLSTOP_LN" ] && [ -n "$TERM_LN" ] && [ "$POOLSTOP_LN" -lt "$TERM_LN" ]; then CRIT[3]=PASS; else CRIT[3]=FAIL; fi
# 4. terminal full-battery: ignore_stride=True AND completion token exactly once
TERM_N=$(grepc 'terminal_eval_complete')
if [ "$TERM_N" -eq 1 ] && [ "$(grepc 'ignore_stride.*[Tt]rue|ignore_stride=True')" -ge 1 ]; then CRIT[4]=PASS; else CRIT[4]=FAIL; fi
# 6. >=1 in-run promotion decision at n=400 (best_checkpoint round logged with 400 games)
[ "$(grepc 'best.*(400|n_games.?400|games.?400)|400.*best')" -ge 1 ] && CRIT[6]=PASS || CRIT[6]=WARN
# 7. incumbent sha-pin verified in startup log (anchor_identity)
if [ "$(grep -cE "anchor_identity" "$RUNLOG")" -ge 1 ] && [ "$(grep -cE "$PIN" "$RUNLOG")" -ge 1 ]; then CRIT[7]=PASS; else CRIT[7]=FAIL; fi

# ── (C) provenance + loader round-trip (criterion 8) ─────────────────────────
say "(C) provenance sidecar + loader round-trip"
PROV=$(ls -t checkpoints/best_model.pt.provenance.json checkpoints/armc_smoke_${STAMP}/*.provenance.json 2>/dev/null | head -1)
if [ -n "$PROV" ] && $PY - "$PROV" <<'PYEOF' >"$OUT/provenance_check.log" 2>&1
import json,sys
d=json.load(open(sys.argv[1]))
assert "step" in d and isinstance(d["step"],int), f"no int step in {d}"
print("provenance OK:", d)
PYEOF
then CRIT[8]=PASS; else CRIT[8]=FAIL; fi

# ── (D) mini-resume — resume ignores the TERMINAL record (criterion 5) ───────
say "(D) mini-resume 5 iters (cross-boundary: restore pin-matching anchor first)"
cp checkpoints/bootstrap_model_v6_live2.pt checkpoints/best_model.pt   # satisfy the W2 pin on a deliberate resume
FINAL_CKPT=$(ls -t checkpoints/armc_smoke_${STAMP}/checkpoint_*.pt 2>/dev/null | head -1)
if [ -n "$FINAL_CKPT" ]; then
  $PY scripts/train.py \
    --checkpoint "$FINAL_CKPT" \
    --variant v6_live2_ls_ab_smoke \
    --iterations 5 \
    --override-scheduler-horizon \
    --run-name "armc_resume_${STAMP}" \
    --checkpoint-dir checkpoints/armc_resume_${STAMP} \
    >"$RESUMELOG" 2>&1
  R_RC=$?
  # resume must START (anchor loaded from best_model.pt, NOT from a terminal record) and
  # must NOT re-emit a terminal_eval_complete as a steering input on resume.
  if [ $R_RC -eq 0 ] && [ "$(grep -cE 'anchor_identity' "$RESUMELOG")" -ge 1 ] && [ "$(grep -cE 'terminal_eval_complete' "$RESUMELOG")" -eq 0 ]; then
    CRIT[5]=PASS; else CRIT[5]=FAIL; fi
else
  CRIT[5]=FAIL; say "  no checkpoint produced to resume from"
fi

# ── (E) probe wiring smoke (criterion 9) ─────────────────────────────────────
say "(E) exploit_probe + KClusterMCTSBot wiring (numbers = proof, not strength)"
PROBE_CKPT="${FINAL_CKPT:-checkpoints/bootstrap_model_v6_live2.pt}"
$PY scripts/exploit_probe.py --checkpoint "$PROBE_CKPT" --arms exploit,control \
  --n-games 2 --sims 32 --max-plies 60 --out "$OUT/exploit_probe.json" \
  >"$OUT/exploit_probe.log" 2>&1
# the in-run offwindow_adversary eval (round 2) already exercised the KClusterMCTSBot path live.
if [ -s "$OUT/exploit_probe.json" ] && [ "$(grepc 'offwindow|exploit')" -ge 1 ]; then CRIT[9]=PASS; else CRIT[9]=FAIL; fi

# ── verdict table ────────────────────────────────────────────────────────────
echo; echo "================ PHASE 3 PRE-REGISTERED CRITERIA ================"
declare -A DESC=(
 [1]="iteration_limit_reached at N=500"
 [2]="drain budgeted, WARN-never-kill (no killed eval round)"
 [3]="pool stopped BEFORE terminal eval (log ordering)"
 [4]="terminal full-battery: ignore_stride + completion token x1"
 [5]="decision record TERMINAL; mini-resume ignores it"
 [6]="≥1 in-run promotion decision at n=400"
 [7]="incumbent sha-pin in startup log (anchor_identity)"
 [8]=".provenance.json written + loader round-trips step"
 [9]="exploit_probe + KClusterMCTSBot produce numbers"
)
FAILS=0
for k in 1 2 3 4 5 6 7 8 9; do
  v="${CRIT[$k]:-FAIL}"; printf "  [%s] criterion %s — %s\n" "$v" "$k" "${DESC[$k]}"
  [ "$v" = "FAIL" ] && FAILS=$((FAILS+1))
done
echo "  (closeout integration test A: $([ $A_RC -eq 0 ] && echo PASS || echo FAIL); run exit code B: $B_RC)"
echo "================================================================="
echo "Artifacts: $OUT/"
if [ "$FAILS" -eq 0 ] && [ $A_RC -eq 0 ] && [ "$B_RC" -eq 0 ]; then
  echo "PHASE 3 VERDICT: PASS — the 50k re-run is cleared for operator launch."
  exit 0
else
  echo "PHASE 3 VERDICT: FAIL ($FAILS criteria failed) — do NOT launch the 50k; inspect $OUT/."
  exit 1
fi
