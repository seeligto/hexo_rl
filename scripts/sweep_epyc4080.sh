#!/usr/bin/env bash
# Throughput sweep for EPYC 7702 (42 vCPU) + RTX 4080 Super (16 GB).
# Stages: n_workers -> inference_batch x wait -> leaf_batch x train_burst.
# Each stage's stable winner (non-bimodal cells preferred) feeds the next.
#
# Run on the rental box:
#   bash scripts/sweep_epyc4080.sh 2>&1 | tee reports/sweeps/sweep.log
#
# Defaults are calibrated from the first sweep failure (2026-04-25):
#   - n_runs=2 produced bimodal cells; raised to 5.
#   - pool_duration=60s wasn't long enough to wash out worker-startup
#     races on EPYC; raised to 180s.
#   - Worker grid trimmed to {12,16,20,24}: at >=24 workers, individual
#     runs alternated between ~0 and ~600k pos/hr (startup race).
#     GPU was already at ~65% util at workers=16 — wider doesn't help.
#
# Per-cell budget at defaults: ~12 min. Total wall: ~3.5 hr for full sweep.
#
# Environment knobs:
#   POOL_DURATION (default 180) - seconds per bench rep
#   N_RUNS        (default 5)   - bench reps per cell
#   NO_COMPILE    (default 0)   - skip torch.compile per cell. Default OFF
#                                 because the threading fix landed (per memory
#                                 feedback_torch_compile_threading.md). Sweep
#                                 forces mode="default" via YAML override —
#                                 reduce-overhead deadlocks vs InferenceServer.
#                                 Set NO_COMPILE=1 to skip compile entirely.
#   WORKER_GRID   (default "12 16 20 24")
#   BATCH_GRID    (default "64 128 192")
#   WAIT_GRID     (default "2.0 4.0 8.0")
#   LEAF_GRID     (default "8 16")
#   BURST_GRID    (default "8 16 32")
#   MODE          (default full) - "validate" for narrow re-sweep around prior
#                                 winners. Use after a dispatcher change (e.g.
#                                 trace_inference fix on 2026-04-25) when you
#                                 only need to confirm the optimum hasn't moved
#                                 and re-pick if it has. Cells: ~6 (~1.2 hr).
#                                 Validate grids: workers={16,20,24}, batch=
#                                 {128,192}, fixed wait=4.0/leaf=8/burst=16.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
POOL_DURATION="${POOL_DURATION:-180}"
N_RUNS="${N_RUNS:-5}"
MODE="${MODE:-full}"

# Tight grids for the validation re-sweep (post-dispatcher-change). Pinned
# to the dispatch-bound optimum band: workers around the prior winner (16),
# extended to 20/24 because the trace fix moves the bottleneck off Python
# and may absorb more concurrent work; batch from 128 (uncompiled winner)
# up to 192 (compiled winner). Wait/leaf/burst held at prior winners since
# they were robust across the original sweep.
if [[ "$MODE" == "validate" ]]; then
  WORKER_GRID="${WORKER_GRID:-16 20 24}"
  BATCH_GRID="${BATCH_GRID:-128 192}"
  WAIT_GRID="${WAIT_GRID:-4.0}"
  LEAF_GRID="${LEAF_GRID:-8}"
  BURST_GRID="${BURST_GRID:-16}"
else
  WORKER_GRID="${WORKER_GRID:-12 16 20 24}"
  BATCH_GRID="${BATCH_GRID:-64 128 192}"
  WAIT_GRID="${WAIT_GRID:-2.0 4.0 8.0}"
  LEAF_GRID="${LEAF_GRID:-8 16}"
  BURST_GRID="${BURST_GRID:-8 16 32}"
fi

NO_COMPILE_FLAG=""
if [[ "${NO_COMPILE:-0}" != "0" ]]; then
  NO_COMPILE_FLAG="--no-compile"
fi

mkdir -p reports/sweeps

ts() { date +%Y-%m-%dT%H:%M:%S; }

echo "==============================================================="
echo "  HeXO sweep — EPYC 7702 + RTX 4080 Super"
echo "==============================================================="
echo "[$(ts)] start  MODE=${MODE}"
echo "  pool_duration=${POOL_DURATION}s   n_runs=${N_RUNS}   no_compile=${NO_COMPILE_FLAG:-(off)}"
echo "  worker grid:  ${WORKER_GRID}"
echo "  batch  grid:  ${BATCH_GRID}"
echo "  wait   grid:  ${WAIT_GRID}"
echo "  leaf   grid:  ${LEAF_GRID}"
echo "  burst  grid:  ${BURST_GRID}"
echo
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo "vCPU:  $(nproc)"
echo "RAM:   $(free -h | awk '/^Mem:/ {print $2 " total, " $7 " avail"}')"
echo

# Helper: pick stable (non-bimodal) winner from a stage CSV; fall back to
# top-median if every cell is bimodal (and shout about it).
pick_stable_winner() {
  local pattern="$1"
  shift
  local cols="$@"
  $PY - "$pattern" "$cols" <<'PY'
import csv, glob, sys
pattern, cols = sys.argv[1], sys.argv[2].split()
files = sorted(glob.glob(pattern))
assert files, f"no csv matched {pattern}"
with open(files[-1]) as f:
    rows = [r for r in csv.DictReader(f) if r.get("pos_median")]
def keyf(r): return float(r["pos_median"])
stable = [r for r in rows if r.get("bimodal", "False") in ("False", "false", "0", "")]
pool = stable if stable else rows
pool.sort(key=keyf, reverse=True)
w = pool[0]
print(" ".join(w[c] for c in cols))
sys.stderr.write(
    f"[winner] from {files[-1]}  pos/hr={float(w['pos_median']):,.0f}"
    + ("" if stable else "  (NO STABLE CELLS — all bimodal!)")
    + "\n"
)
PY
}

# ── Stage 1: n_workers ─────────────────────────────────────────────────────
echo "[$(ts)] stage 1 / 3 — n_workers"
$PY scripts/sweep_epyc4080.py \
    --stage workers \
    --worker-grid $WORKER_GRID \
    --pool-duration "$POOL_DURATION" \
    --n-runs "$N_RUNS" \
    $NO_COMPILE_FLAG

WORKERS_WINNER=$(pick_stable_winner "reports/sweeps/sweep_workers_*.csv" n_workers)
echo "[$(ts)] stage 1 winner: n_workers=${WORKERS_WINNER}"
echo

# ── Stage 2: inference_batch_size x inference_max_wait_ms ──────────────────
echo "[$(ts)] stage 2 / 3 — batch x wait (workers=${WORKERS_WINNER})"
$PY scripts/sweep_epyc4080.py \
    --stage batch_wait \
    --workers "$WORKERS_WINNER" \
    --batch-grid $BATCH_GRID \
    --wait-grid $WAIT_GRID \
    --pool-duration "$POOL_DURATION" \
    --n-runs "$N_RUNS" \
    $NO_COMPILE_FLAG

read BATCH_WINNER WAIT_WINNER < <(
  pick_stable_winner "reports/sweeps/sweep_batch_wait_*.csv" inference_batch_size inference_max_wait_ms
)
echo "[$(ts)] stage 2 winner: batch=${BATCH_WINNER}  wait=${WAIT_WINNER}"
echo

# ── Stage 3: leaf_batch_size x max_train_burst ─────────────────────────────
echo "[$(ts)] stage 3 / 3 — leaf x burst (workers=${WORKERS_WINNER}, batch=${BATCH_WINNER}, wait=${WAIT_WINNER})"
$PY scripts/sweep_epyc4080.py \
    --stage leaf_burst \
    --workers "$WORKERS_WINNER" \
    --batch "$BATCH_WINNER" \
    --wait "$WAIT_WINNER" \
    --leaf-grid $LEAF_GRID \
    --burst-grid $BURST_GRID \
    --pool-duration "$POOL_DURATION" \
    --n-runs "$N_RUNS" \
    $NO_COMPILE_FLAG

read LEAF_WINNER BURST_WINNER < <(
  pick_stable_winner "reports/sweeps/sweep_leaf_burst_*.csv" leaf_batch_size max_train_burst
)
echo "[$(ts)] stage 3 winner: leaf=${LEAF_WINNER}  burst=${BURST_WINNER}"
echo

# ── Final summary ──────────────────────────────────────────────────────────
echo "==============================================================="
echo "  SWEEP DONE  $(ts)"
echo "==============================================================="
echo "  n_workers              = ${WORKERS_WINNER}"
echo "  inference_batch_size   = ${BATCH_WINNER}"
echo "  inference_max_wait_ms  = ${WAIT_WINNER}"
echo "  leaf_batch_size        = ${LEAF_WINNER}"
echo "  max_train_burst        = ${BURST_WINNER}"
echo
echo "Inspect per-stage tables in this log; raw CSVs:"
ls -1 reports/sweeps/sweep_*.csv | tail -3
echo
echo "Per-cell JSON reports: reports/benchmarks/*.json"
echo
echo "Suggested patch to configs/variants/gumbel_targets_epyc4080.yaml:"
echo "  selfplay:"
echo "    n_workers: ${WORKERS_WINNER}"
echo "    inference_batch_size: ${BATCH_WINNER}"
echo "    inference_max_wait_ms: ${WAIT_WINNER}"
echo "    leaf_batch_size: ${LEAF_WINNER}"
echo "  max_train_burst: ${BURST_WINNER}"
echo
echo "Then launch training:  make train VARIANT=gumbel_targets_epyc4080"
