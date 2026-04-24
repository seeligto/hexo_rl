#!/usr/bin/env bash
# Throughput sweep for EPYC 7702 (42 vCPU) + RTX 4080 Super (16 GB).
# Stages: n_workers -> inference_batch x wait -> leaf_batch x train_burst.
# Each stage's winner feeds the next.
#
# Run on the rental box:
#   bash scripts/sweep_epyc4080.sh 2>&1 | tee reports/sweeps/sweep.log
#
# Environment knobs:
#   POOL_DURATION (default 60)  - seconds per bench rep
#   N_RUNS        (default 2)   - bench reps per cell
#   NO_COMPILE    (set to 1)    - skip torch.compile per cell (faster, less realistic)
set -euo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
POOL_DURATION="${POOL_DURATION:-60}"
N_RUNS="${N_RUNS:-2}"
NO_COMPILE_FLAG=""
if [[ "${NO_COMPILE:-0}" == "1" ]]; then
  NO_COMPILE_FLAG="--no-compile"
fi

mkdir -p reports/sweeps

ts() { date +%Y-%m-%dT%H:%M:%S; }

echo "[$(ts)] sweep start  pool_duration=${POOL_DURATION}s  n_runs=${N_RUNS}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo "vCPU:  $(nproc)"
echo "RAM:   $(free -h | awk '/^Mem:/ {print $2 " total, " $7 " avail"}')"
echo

# ── Stage 1: n_workers sweep ────────────────────────────────────────────────
echo "[$(ts)] stage 1 / 3 — n_workers"
$PY scripts/sweep_epyc4080.py \
    --stage workers \
    --worker-grid 16 24 32 40 \
    --pool-duration "$POOL_DURATION" \
    --n-runs "$N_RUNS" \
    $NO_COMPILE_FLAG

# Pick winner from latest stage-1 CSV
WORKERS_WINNER=$(
  $PY - <<'PY'
import csv, glob, os
files = sorted(glob.glob("reports/sweeps/sweep_workers_*.csv"))
assert files, "no stage-1 csv produced"
with open(files[-1]) as f:
    rows = [r for r in csv.DictReader(f) if r.get("pos_per_hr")]
rows.sort(key=lambda r: float(r["pos_per_hr"]), reverse=True)
print(rows[0]["n_workers"])
PY
)
echo "[$(ts)] stage 1 winner: n_workers=${WORKERS_WINNER}"
echo

# ── Stage 2: inference_batch_size x inference_max_wait_ms ───────────────────
echo "[$(ts)] stage 2 / 3 — batch x wait (workers=${WORKERS_WINNER})"
$PY scripts/sweep_epyc4080.py \
    --stage batch_wait \
    --workers "$WORKERS_WINNER" \
    --batch-grid 64 128 192 \
    --wait-grid 2.0 4.0 8.0 \
    --pool-duration "$POOL_DURATION" \
    --n-runs "$N_RUNS" \
    $NO_COMPILE_FLAG

read BATCH_WINNER WAIT_WINNER < <(
  $PY - <<'PY'
import csv, glob
files = sorted(glob.glob("reports/sweeps/sweep_batch_wait_*.csv"))
assert files, "no stage-2 csv produced"
with open(files[-1]) as f:
    rows = [r for r in csv.DictReader(f) if r.get("pos_per_hr")]
rows.sort(key=lambda r: float(r["pos_per_hr"]), reverse=True)
print(rows[0]["inference_batch_size"], rows[0]["inference_max_wait_ms"])
PY
)
echo "[$(ts)] stage 2 winner: batch=${BATCH_WINNER}  wait=${WAIT_WINNER}"
echo

# ── Stage 3: leaf_batch_size x max_train_burst ──────────────────────────────
echo "[$(ts)] stage 3 / 3 — leaf x burst (workers=${WORKERS_WINNER}, batch=${BATCH_WINNER}, wait=${WAIT_WINNER})"
$PY scripts/sweep_epyc4080.py \
    --stage leaf_burst \
    --workers "$WORKERS_WINNER" \
    --batch "$BATCH_WINNER" \
    --wait "$WAIT_WINNER" \
    --leaf-grid 8 16 \
    --burst-grid 8 16 32 \
    --pool-duration "$POOL_DURATION" \
    --n-runs "$N_RUNS" \
    $NO_COMPILE_FLAG

echo
echo "[$(ts)] sweep done."
echo "Stage winners:"
echo "  n_workers              = ${WORKERS_WINNER}"
echo "  inference_batch_size   = ${BATCH_WINNER}"
echo "  inference_max_wait_ms  = ${WAIT_WINNER}"
echo
echo "Per-stage CSVs in reports/sweeps/sweep_{workers,batch_wait,leaf_burst}_*.csv"
echo "Per-cell JSON reports in reports/benchmarks/*.json"
echo
echo "Update configs/variants/gumbel_targets_epyc4080.yaml with the winners,"
echo "then launch training:  make train VARIANT=gumbel_targets_epyc4080"
