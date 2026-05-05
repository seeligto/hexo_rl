#!/usr/bin/env bash
# v9_post_smoke_sealbot.sh — strength survey across the v9 §153 T5 candidates.
#
# Runs SealBot eval at n=200 / time_limit=0.1 / model_sims=96 against:
#
#   1. v7full bootstrap         (§150 canonical anchor — strength baseline)
#   2. v8full_warm bootstrap    (the anchor each smoke trained against)
#   3. S1 step-500 promoted     (the model that beat v8full_warm in S1)
#   4. S1 latest checkpoint     (final trainer state in S1, ~step 2500)
#   5. S2 step-500 promoted     (the model that beat v8full_warm in S2)
#   6. S2 latest checkpoint     (final trainer state in S2, ~step 2500)
#
# Each eval ≈ 25-35 min on a 5080 (varies with avg game length). Runs
# strictly sequentially so we never compete with the smoke runner. Total
# wall: ~3h.
#
# Each ckpt's result lands in logs/sealbot_v9_<id>.jsonl + .log; a
# combined summary lands at /tmp/v9_sealbot_summary.md and is also
# echoed to stdout.

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY=".venv/bin/python"
N_GAMES="${N_GAMES:-200}"
TIME_LIMIT="${TIME_LIMIT:-0.1}"
MODEL_SIMS="${MODEL_SIMS:-96}"
LOG_DIR="logs"
SUMMARY="/tmp/v9_sealbot_summary.md"
ARCHIVE_ROOT="archive/v9_smokes_20260505"

mkdir -p "$LOG_DIR"
: > "$SUMMARY"

run_one() {
    local id="$1"
    local label="$2"
    local ckpt="$3"
    local out_jsonl="${LOG_DIR}/sealbot_v9_${id}.jsonl"
    local out_log="${LOG_DIR}/sealbot_v9_${id}.log"

    if [[ ! -f "$ckpt" ]]; then
        echo "[$(date +%H:%M:%S)] $id ($label) MISSING checkpoint $ckpt — skipping"
        echo "| $id | $label | MISSING | — | — | — |" >> "$SUMMARY"
        return
    fi
    echo "[$(date +%H:%M:%S)] $id ($label) — eval vs SealBot, n=$N_GAMES, ckpt=$ckpt"
    set +e
    "$PY" scripts/eval_vs_sealbot.py \
        --checkpoint "$ckpt" \
        --n-games "$N_GAMES" \
        --time-limit "$TIME_LIMIT" \
        --model-sims "$MODEL_SIMS" \
        --out "$out_jsonl" \
        > "$out_log" 2>&1
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "[$(date +%H:%M:%S)] $id RC=$rc — see $out_log"
        echo "| $id | $label | RC=$rc | — | — | — |" >> "$SUMMARY"
        return
    fi

    # Parse the last JSON record from the .jsonl
    "$PY" - "$out_jsonl" "$id" "$label" <<'PY' >> "$SUMMARY"
import json, pathlib, sys
path, ident, label = sys.argv[1], sys.argv[2], sys.argv[3]
recs = []
for line in pathlib.Path(path).read_text().splitlines():
    line = line.strip()
    if not line: continue
    try: recs.append(json.loads(line))
    except: pass
if not recs:
    print(f"| {ident} | {label} | parse_err | — | — | — |")
else:
    r = recs[-1]
    wr = r.get("winrate", 0.0)
    n = r.get("n_games", 0)
    wins = r.get("win_count", 0)
    draws = r.get("draw_count", 0)
    colony = r.get("colony_wins", 0)
    print(f"| {ident} | {label} | {wr:.3f} ({wins}/{n}) | draws {draws} | colony {colony} | n_games {n} |")
PY
    echo "[$(date +%H:%M:%S)] $id done"
}

{
    echo "# v9 §153 T5 — SealBot strength survey"
    echo
    echo "Generated: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "Eval params: n=$N_GAMES games per ckpt, time_limit=${TIME_LIMIT}s, model_sims=$MODEL_SIMS"
    echo
    echo "| id | label | winrate | draws | colony | n |"
    echo "|---|---|---|---|---|---|"
} > "$SUMMARY"

# Run order matches the prompt order: baselines first, then per-smoke.
run_one "v7full" "§150 canonical bootstrap (reference)" \
    "checkpoints/bootstrap_model_v7full.pt"
run_one "v8full_warm" "v9 bootstrap (warm-start, the anchor each smoke trained against)" \
    "checkpoints/bootstrap_model_v8full_warm.pt"

# S1 — hex+corner_mask, no jitter
run_one "S1_step500" "S1 step-500 promoted (hex+corner_mask, no jitter)" \
    "${ARCHIVE_ROOT}/S1/checkpoints/best_model.pt"
S1_LATEST="$(ls -1 "${ARCHIVE_ROOT}/S1/checkpoints"/checkpoint_*.pt 2>/dev/null | tail -n 1 || true)"
if [[ -n "$S1_LATEST" ]]; then
    run_one "S1_latest" "S1 latest trainer ckpt (post-step-500 trainer state)" "$S1_LATEST"
else
    run_one "S1_latest" "S1 latest trainer ckpt" "${ARCHIVE_ROOT}/S1/checkpoints/checkpoint_MISSING.pt"
fi

# S2 — hex+corner_mask + Q2 jitter
run_one "S2_step500" "S2 step-500 promoted (hex+corner_mask + Q2 jitter)" \
    "${ARCHIVE_ROOT}/S2/checkpoints/best_model.pt"
S2_LATEST="$(ls -1 "${ARCHIVE_ROOT}/S2/checkpoints"/checkpoint_*.pt 2>/dev/null | tail -n 1 || true)"
if [[ -n "$S2_LATEST" ]]; then
    run_one "S2_latest" "S2 latest trainer ckpt (post-step-500 trainer state)" "$S2_LATEST"
else
    run_one "S2_latest" "S2 latest trainer ckpt" "${ARCHIVE_ROOT}/S2/checkpoints/checkpoint_MISSING.pt"
fi

echo
echo "============================================================="
echo "SUMMARY"
echo "============================================================="
cat "$SUMMARY"
echo
echo "Detail logs: ${LOG_DIR}/sealbot_v9_*.log + .jsonl"
