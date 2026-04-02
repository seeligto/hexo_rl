#!/usr/bin/env bash
# Usage: train_with_dashboard.sh [web|rich|none] <train_command...>
#
# Modes:
#   web  — Start web dashboard as background process, training in foreground.
#           On exit (Ctrl+C / TERM / natural exit), dashboard is killed.
#   rich — Set HEXO_RICH_DASHBOARD=1, training in foreground.
#   none — Training only, no dashboard.

set -euo pipefail

MODE="${1:-none}"
shift

TRAIN_CMD=("$@")
DASH_PID=""

cleanup() {
    if [ -n "$DASH_PID" ]; then
        kill "$DASH_PID" 2>/dev/null || true
        wait "$DASH_PID" 2>/dev/null || true
    fi
}

if [ "$MODE" = "web" ]; then
    # Start web dashboard in background
    python dashboard.py &
    DASH_PID=$!
    echo "Web dashboard started (PID $DASH_PID) -> http://localhost:5001"
    trap cleanup INT TERM EXIT
elif [ "$MODE" = "rich" ]; then
    export HEXO_RICH_DASHBOARD=1
fi

"${TRAIN_CMD[@]}"
EXIT_CODE=$?

exit $EXIT_CODE
