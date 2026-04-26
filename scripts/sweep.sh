#!/usr/bin/env bash
# Hardware-agnostic throughput sweep — replaces sweep_epyc4080.sh (§126).
#
# Workflow:
#   bash scripts/sweep.sh detect                       # writes detected_host.json
#   bash scripts/sweep.sh run                          # full registry sweep, ~70 min
#   bash scripts/sweep.sh run --knobs n_workers        # one knob only
#   bash scripts/sweep.sh run --fix n_workers=24       # lock one, search rest
#
# All flags after the subcommand are passed through to the Python harness;
# see scripts/sweep_harness/__main__.py and docs/sweep_harness.md.
set -euo pipefail

cd "$(dirname "$0")/.."

PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
  echo "[sweep.sh] error: $PY missing — create venv first (python -m venv .venv)" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {detect|run} [args...]" >&2
  exit 2
fi

mkdir -p reports/sweeps
exec "$PY" -m scripts.sweep_harness "$@"
