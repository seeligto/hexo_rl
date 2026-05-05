#!/usr/bin/env bash
# v9_s4_decision.sh — runs locally, polls remote for S2 completion, then
# decides whether to let S4 run (default — do nothing) or kill the runner
# before S4 starts (writes a verdict file).
#
# Decision rule: S4 only runs if S2 long-game (plies>=60) stride5 P50 ≤ 25
# AND P90 ≤ 40 — i.e. hex+jitter is at least matching Q2 jitter alone.
# Otherwise hex is neutral or harmful with jitter; S4 (which adds per-move
# rotation, falsified as null on its own) cannot rescue that.

set -uo pipefail

VERDICT_FILE="/tmp/v9_s4_decision.txt"
P50_GATE=25
P90_GATE=40

while true; do
    sleep 300  # 5-min poll cycle
    payload=$(ssh -p 13053 -i ~/.ssh/vast_hexo -o IdentitiesOnly=yes \
        -o UserKnownHostsFile=~/.ssh/known_hosts_vast -o ConnectTimeout=15 \
        root@ssh6.vast.ai bash -s <<'REMOTE' 2>&1 | grep -vE 'Welcome to vast|Have fun'
set -uo pipefail
cd /workspace/hexo_rl
DRIVER=logs/v9_smokes_runner.driver.log
S2_DONE=$(grep -c 'S2 archive complete' "$DRIVER" 2>/dev/null || echo 0)
S2_STARTED=$(grep -c 'S2  variant=v9_s2' "$DRIVER" 2>/dev/null || echo 0)
if [ "${S2_DONE:-0}" -gt 0 ]; then
    EVENTS=$(ls -1t logs/events_*.jsonl 2>/dev/null | head -n 1)
    .venv/bin/python - "$EVENTS" <<PY
import json, statistics, pathlib, sys
path = sys.argv[1]
games = []
for line in pathlib.Path(path).read_text().splitlines():
    try: d = json.loads(line)
    except: continue
    if d.get('event') != 'game_complete' or 'stride5_run_max' not in d: continue
    games.append((d.get('moves',0), d['stride5_run_max']))
long = [g[1] for g in games if g[0] >= 60]
if not long:
    print('done|p50=NA|p90=NA|n=0')
    sys.exit()
long.sort()
p50 = statistics.median(long)
p90 = long[int(len(long)*0.9)] if len(long) > 9 else max(long)
print(f'done|n={len(long)}|p50={p50}|p90={p90}')
PY
else
    echo "running|S2_started=${S2_STARTED:-0}|S2_done=${S2_DONE:-0}"
fi
REMOTE
)
    echo "[$(date +%H:%M:%S)] $payload" >&2

    if echo "$payload" | grep -q '^done|'; then
        N=$(echo "$payload" | grep -oE 'n=[0-9]+' | head -1 | cut -d= -f2)
        P50=$(echo "$payload" | grep -oE 'p50=[0-9.]+' | cut -d= -f2)
        P90=$(echo "$payload" | grep -oE 'p90=[0-9]+' | cut -d= -f2)
        if [ -z "$P50" ] || [ "$P50" = "NA" ]; then
            echo "S2 finished but no long-games metrics — defaulting to RUN S4 (insufficient data to skip)" | tee "$VERDICT_FILE"
            break
        fi
        run_s4=$(awk -v p50="$P50" -v p90="$P90" -v gp50="$P50_GATE" -v gp90="$P90_GATE" \
            'BEGIN { if (p50 <= gp50 && p90 <= gp90) print 1; else print 0 }')
        if [ "$run_s4" = "1" ]; then
            {
                echo "S2 verdict: HEX+JITTER IS A WIN (n=$N, p50=$P50, p90=$P90)"
                echo "  P50 ≤ $P50_GATE AND P90 ≤ $P90_GATE — letting S4 run"
            } | tee "$VERDICT_FILE"
        else
            {
                echo "S2 verdict: hex+jitter null/regression (n=$N, p50=$P50, p90=$P90)"
                echo "  P50 > $P50_GATE OR P90 > $P90_GATE — SKIPPING S4"
                echo "  killing runner before S4 starts"
            } | tee "$VERDICT_FILE"
            ssh -p 13053 -i ~/.ssh/vast_hexo -o IdentitiesOnly=yes \
                -o UserKnownHostsFile=~/.ssh/known_hosts_vast root@ssh6.vast.ai \
                'pkill -TERM -f "run_v9_smokes.sh"; sleep 5; pkill -TERM -f "scripts/train.py.*v9_s4"; sleep 3; pkill -KILL -f "run_v9_smokes.sh|scripts/train.py.*v9_s4"' 2>/dev/null
        fi
        break
    fi
done
