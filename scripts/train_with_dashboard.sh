#!/usr/bin/env bash
# Usage: train_with_dashboard.sh [rich|none] <train_command...>
#
# Modes:
#   rich — Set HEXO_RICH_DASHBOARD=1, training in foreground.
#   none — Training only, no dashboard.
#
# Note: web dashboard is launched by scripts/train.py itself when DASHBOARD=1
# (see Makefile). The old `web` mode referenced a nonexistent dashboard.py
# and has been removed; use `make dashboard` or scripts/serve_dashboard.py
# to run the web dashboard standalone.

set -euo pipefail

MODE="${1:-none}"
shift

TRAIN_CMD=("$@")

if [ "$MODE" = "rich" ]; then
    export HEXO_RICH_DASHBOARD=1
elif [ "$MODE" != "none" ]; then
    echo "Unknown mode: $MODE (expected rich|none)" >&2
    exit 2
fi

"${TRAIN_CMD[@]}"
EXIT_CODE=$?

exit $EXIT_CODE
