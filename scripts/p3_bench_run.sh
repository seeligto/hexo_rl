#!/usr/bin/env bash
# P3 corner-mask bench A/B on the host where this lives. Run after engine
# is built. Pool n=3, warmup 60s, duration 90s. n_workers from BENCH_WORKERS
# (default = nproc-2). Output JSONs to reports/probes/.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p reports/probes
WORKERS="${BENCH_WORKERS:-$( ( $(command -v nproc) ) && echo $(( $(nproc) - 2 )) || echo 16)}"
COMMON=(--mcts-sims 50000 --pool-workers "${WORKERS}" --pool-duration 90 \
        --pool-warmup 60 --n-runs 3 --no-compile)

echo "=== P3 BENCH OFF ==="
.venv/bin/python scripts/benchmark.py "${COMMON[@]}" \
    --output reports/probes/p3_bench_off.json 2>&1 | tee reports/probes/p3_bench_off.txt

echo
echo "=== P3 BENCH ON  ==="
.venv/bin/python scripts/benchmark.py "${COMMON[@]}" --corner-mask \
    --output reports/probes/p3_bench_on.json 2>&1 | tee reports/probes/p3_bench_on.txt

echo
echo "=== P3 DONE ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
