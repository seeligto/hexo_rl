#!/usr/bin/env bash
# run_with_dmon.sh — launch a training command with nvidia-smi dmon sidecar.
#
# Captures per-second GPU telemetry (util, mem, SM clock, power, temp)
# alongside any training / benchmark command. Portable: works on any
# NVIDIA GPU + Linux with nvidia-smi.
#
# Usage:
#   scripts/perf/run_with_dmon.sh <label> -- <command> [args...]
# Examples:
#   scripts/perf/run_with_dmon.sh diag_C1 -- .venv/bin/python scripts/train.py --variant gumbel_targets --iterations 2000
#   scripts/perf/run_with_dmon.sh bench -- make bench
#
# Outputs (under reports/perf/<label>_<timestamp>/):
#   dmon.log   — nvidia-smi dmon -s pucvmet -d 1 output
#   cmd.stdout — stdout/stderr of the inner command
#   cmd.status — exit code of the inner command
#
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <label> -- <command> [args...]" >&2
  exit 2
fi
label="$1"; shift
if [[ "$1" != "--" ]]; then
  echo "Expected '--' as separator after label" >&2
  exit 2
fi
shift

ts="$(date +%Y%m%d_%H%M%S)"
outdir="reports/perf/${label}_${ts}"
mkdir -p "$outdir"

# nvidia-smi presence check
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found — running command without GPU telemetry" >&2
  set +e
  "$@" > "$outdir/cmd.stdout" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$outdir/cmd.status"
  exit "$rc"
fi

# dmon fields: p=power, u=util, c=clock, v=violations, m=mem, e=ecc, t=throughput
# -d 1 = 1 second cadence. -f = file output.
nvidia-smi dmon -s pucvmet -d 1 -f "$outdir/dmon.log" &
dmon_pid=$!

cleanup() {
  if kill -0 "$dmon_pid" 2>/dev/null; then
    kill "$dmon_pid" || true
    wait "$dmon_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[run_with_dmon] label=$label outdir=$outdir dmon_pid=$dmon_pid"
set +e
"$@" > "$outdir/cmd.stdout" 2>&1
rc=$?
set -e
echo "$rc" > "$outdir/cmd.status"
echo "[run_with_dmon] command exit=$rc"
exit "$rc"
