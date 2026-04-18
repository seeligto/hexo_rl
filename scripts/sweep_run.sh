#!/usr/bin/env bash
# Config sweep orchestration — 2026-04-08
# Runs 15 training configs sequentially (9 PUCT + 6 Gumbel), 20 min each.
# Pre-sweep gate: P4 for 5 min. Control gates: P0, G0 (full 20 min).
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PROJ_ROOT}/.venv/bin/python"
CONFIG_DIR="/tmp/sweep_configs"
BOOTSTRAP="${PROJ_ROOT}/checkpoints/bootstrap_model.pt"
LOG_DIR="${PROJ_ROOT}/logs/sweep"
CKPT_BASE="${PROJ_ROOT}/tmp/sweep_ckpt"
REPORT_DIR="${PROJ_ROOT}/archive/sweep_2026-04-08"
DURATION=1200   # 20 minutes
GATE_DURATION=300  # 5 minutes

mkdir -p "$LOG_DIR" "$CKPT_BASE" "$REPORT_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────────

kill_training() {
    pkill -f "scripts/train.py" 2>/dev/null || true
    sleep 2
}

run_one() {
    local name="$1"
    local variant="$2"
    local dur="$3"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  RUN: ${name}  (variant=${variant}, ${dur}s)"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════"

    kill_training

    local ckpt_dir="${CKPT_BASE}/${name}"
    mkdir -p "$ckpt_dir"

    timeout --signal=SIGINT --kill-after=60 "$dur" \
        "$PY" "$PROJ_ROOT/scripts/train.py" \
            --config "${CONFIG_DIR}/${name}.yaml" \
            --variant "$variant" \
            --checkpoint "$BOOTSTRAP" \
            --run-name "sweep_${name}" \
            --log-dir "$LOG_DIR" \
            --checkpoint-dir "$ckpt_dir" \
            --no-dashboard \
            --iterations 99999 \
            --override-scheduler-horizon \
        2>&1 | tee "${LOG_DIR}/${name}_stdout.log"
    local rc=${PIPESTATUS[0]}

    if [[ $rc -eq 124 ]]; then
        echo "[${name}] Timeout expired (expected). OK."
    elif [[ $rc -eq 0 ]]; then
        echo "[${name}] Exited cleanly."
    else
        echo "[${name}] WARNING: exit code ${rc}"
    fi

    # Let GPU memory fully release
    sleep 5
    return $rc
}

validate_log() {
    # Quick check that the JSONL has train_step events with the new fields
    local name="$1"
    local logfile="${LOG_DIR}/sweep_${name}.jsonl"
    if [[ ! -f "$logfile" ]]; then
        echo "VALIDATION FAIL: ${logfile} not found"
        return 1
    fi
    local count
    count=$(grep -c '"event": "train_step"' "$logfile" 2>/dev/null || echo 0)
    if [[ $count -lt 2 ]]; then
        echo "VALIDATION FAIL: ${name} has only ${count} train_step events (need >=2)"
        return 1
    fi
    # Check for the new monitoring fields
    if ! grep -q '"batch_fill_pct"' "$logfile"; then
        echo "VALIDATION FAIL: ${name} missing batch_fill_pct field"
        return 1
    fi
    if ! grep -q '"inf_forward_count"' "$logfile"; then
        echo "VALIDATION FAIL: ${name} missing inf_forward_count field"
        return 1
    fi
    echo "[${name}] Validation OK (${count} train_step events, new fields present)"
    return 0
}

# ── Kill any lingering processes ─────────────────────────────────────────────

echo "Killing any lingering training/benchmark processes..."
kill_training
pkill -f "scripts/benchmark.py" 2>/dev/null || true
pgrep -fl "train.py\|benchmark.py" || echo "All clear."

# ── Verify bootstrap checkpoint exists ───────────────────────────────────────

if [[ ! -f "$BOOTSTRAP" ]]; then
    echo "FATAL: bootstrap checkpoint not found at ${BOOTSTRAP}"
    exit 1
fi

# ── PRE-SWEEP GATE: P4 for 5 minutes ────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PRE-SWEEP GATE: P4 (5 min)                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

run_one "P4" "gumbel_targets" "$GATE_DURATION"
gate_rc=$?

if [[ $gate_rc -ne 0 && $gate_rc -ne 124 ]]; then
    echo ""
    echo "GATE FAILED (exit ${gate_rc}). Check ${LOG_DIR}/P4_stdout.log"
    echo "Aborting sweep."
    exit 1
fi

if ! validate_log "P4"; then
    echo "GATE FAILED: metric extraction validation failed."
    echo "Aborting sweep."
    exit 1
fi

# Check for CUDA OOM in stdout log
if grep -qi "out of memory\|CUDA OOM\|OutOfMemoryError" "${LOG_DIR}/P4_stdout.log" 2>/dev/null; then
    echo "GATE FAILED: CUDA OOM detected in P4 stdout."
    echo "Aborting sweep."
    exit 1
fi

echo ""
echo "Pre-sweep gate PASSED."
echo ""

# Remove gate P4 log/ckpt so the full P4 run gets a clean start
rm -f "${LOG_DIR}/sweep_P4.jsonl"
rm -rf "${CKPT_BASE}/P4"

# ── PUCT ARM: P0–P8 ─────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PUCT ARM (variant=gumbel_targets)                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# P0 is the PUCT control gate
run_one "P0" "gumbel_targets" "$DURATION"
p0_rc=$?
if [[ $p0_rc -ne 0 && $p0_rc -ne 124 ]]; then
    echo "P0 CONTROL RUN FAILED (exit ${p0_rc}). Aborting PUCT arm."
    exit 1
fi
if ! validate_log "P0"; then
    echo "P0 CONTROL VALIDATION FAILED. Aborting PUCT arm."
    exit 1
fi
echo "P0 control gate PASSED."

for name in P1 P2 P3 P4 P5 P6 P7 P8; do
    run_one "$name" "gumbel_targets" "$DURATION"
    rc=$?
    if [[ $rc -ne 0 && $rc -ne 124 ]]; then
        echo "WARNING: ${name} crashed (exit ${rc}). Continuing to next run."
    fi
done

# ── GUMBEL ARM: G0–G5 ───────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GUMBEL ARM (variant=gumbel_full)                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# G0 is the Gumbel control gate
run_one "G0" "gumbel_full" "$DURATION"
g0_rc=$?
if [[ $g0_rc -ne 0 && $g0_rc -ne 124 ]]; then
    echo "G0 CONTROL RUN FAILED (exit ${g0_rc}). Aborting Gumbel arm."
    # Don't exit entirely — PUCT results are still valid
    echo "Skipping G1-G5. PUCT results will still be extracted."
else
    if ! validate_log "G0"; then
        echo "G0 CONTROL VALIDATION FAILED. Skipping G1-G5."
    else
        echo "G0 control gate PASSED."
        for name in G1 G2 G3 G4 G5; do
            run_one "$name" "gumbel_full" "$DURATION"
            rc=$?
            if [[ $rc -ne 0 && $rc -ne 124 ]]; then
                echo "WARNING: ${name} crashed (exit ${rc}). Continuing to next run."
            fi
        done
    fi
fi

# ── METRIC EXTRACTION ───────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  METRIC EXTRACTION                                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

"$PY" "$PROJ_ROOT/scripts/sweep_extract.py" \
    --log-dir "$LOG_DIR" \
    --config-dir "$CONFIG_DIR" \
    --output "$REPORT_DIR/results.csv" \
    --summary "$REPORT_DIR/summary.md" \
    --warmup-sec 90

echo ""
echo "Results:  ${REPORT_DIR}/results.csv"
echo "Summary:  ${REPORT_DIR}/summary.md"

# ── POST-SWEEP VERIFICATION ─────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  POST-SWEEP: make bench                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"

kill_training
cd "$PROJ_ROOT"
make bench

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SWEEP COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"
