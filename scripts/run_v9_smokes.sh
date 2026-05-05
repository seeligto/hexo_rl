#!/usr/bin/env bash
# run_v9_smokes.sh — Phase B' v9 §153 T5 sequential smoke runner.
#
# Runs S1, S2, S3, S4 in order on whatever GPU host this script is invoked
# on (designed for the 5080 vast.ai instance).  Each smoke is a fresh
# 2500-iteration run from its bootstrap checkpoint with replay buffer +
# best_model + checkpoint_*.pt cleared.
#
# Per-smoke walltime: ~3 h on a 5080 (matches §152 + v8 smokes).
# Total sequential walltime: ~12-13 h.  Use BG=1 to nohup the runner and
# return immediately; the runner's driver log records progress.
#
# Layout per smoke S_K:
#   logs/v9_S<K>.log                  — train.py stdout
#   logs/v9_S<K>.jsonl                — structured events (configure_logging)
#   reports/phase_b_prime/v9_smokes/S<K>/  — instrumentation jsonl + run.log
#   archive/v9_smokes/S<K>/checkpoints/    — final ckpt + buffer
#
# Usage (on the remote host):
#   scripts/run_v9_smokes.sh                    — run S1..S4 sequentially
#   scripts/run_v9_smokes.sh --only S1 S3       — run only S1 and S3
#   scripts/run_v9_smokes.sh --skip-bootstrap   — skip the v8full bootstrap step
#   ITERATIONS=500 scripts/run_v9_smokes.sh ... — override (e.g. quick local sanity)
#
# Pre-conditions:
#   * `make build` has been run (engine compiled with v9 features)
#   * checkpoints/bootstrap_model_v7full.pt exists (canonical §150)
#   * v8full bootstrap will be produced by step 0 unless it already exists
#     or --skip-bootstrap is passed.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG_DIR="logs"
ARCHIVE_DIR="archive/v9_smokes_$(date +%Y%m%d)"
DRIVER_LOG="${LOG_DIR}/v9_smokes_runner.driver.log"
ITERATIONS="${ITERATIONS:-2500}"
PER_SMOKE_TIMEOUT="${PER_SMOKE_TIMEOUT:-16200}"  # 4.5 h cap; expected ~3 h
PY=".venv/bin/python"

mkdir -p "$LOG_DIR" "$ARCHIVE_DIR" reports/phase_b_prime/v9_smokes

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DRIVER_LOG"; }

# ── Parse flags ──────────────────────────────────────────────────────────────
SKIP_BOOTSTRAP=0
ONLY_RUNS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-bootstrap) SKIP_BOOTSTRAP=1; shift ;;
        --only)           shift; while [[ $# -gt 0 && "$1" != --* ]]; do ONLY_RUNS+=("$1"); shift; done ;;
        *)                echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Smoke registry: (id, variant, bootstrap_path).
SMOKES=(
    "S1|v9_s1_hex_only_5080|checkpoints/bootstrap_model_v8full.pt"
    "S2|v9_s2_hex_q2_5080|checkpoints/bootstrap_model_v8full.pt"
    "S3|v9_s3_per_move_no_hex_5080|checkpoints/bootstrap_model_v7full.pt"
    "S4|v9_s4_full_combined_5080|checkpoints/bootstrap_model_v8full.pt"
)

should_run() {
    local id="$1"
    if [[ ${#ONLY_RUNS[@]} -eq 0 ]]; then
        return 0
    fi
    local r
    for r in "${ONLY_RUNS[@]}"; do
        [[ "$r" == "$id" ]] && return 0
    done
    return 1
}

# ── Step 0: bootstrap v8full if needed ───────────────────────────────────────
if [[ $SKIP_BOOTSTRAP -eq 0 ]] && [[ ! -f checkpoints/bootstrap_model_v8full.pt ]]; then
    log "Step 0 — producing checkpoints/bootstrap_model_v8full.pt via warm-start"
    log "  source: checkpoints/bootstrap_model_v7full.pt (§150 canonical)"
    log "  recipe: --epochs 30 --eta-min 5e-5"
    log "  model overrides: use_hex_kernel=true corner_mask=true"

    if [[ ! -f checkpoints/bootstrap_model_v7full.pt ]]; then
        log "ERROR: checkpoints/bootstrap_model_v7full.pt missing — cannot warm-start"
        exit 1
    fi

    # Patch model.yaml in-place; restore on exit.
    MODEL_YAML="configs/model.yaml"
    BACKUP_MODEL_YAML="${MODEL_YAML}.v9_bootstrap_backup"
    cp "$MODEL_YAML" "$BACKUP_MODEL_YAML"
    restore_model_yaml() {
        if [[ -f "$BACKUP_MODEL_YAML" ]]; then
            mv -f "$BACKUP_MODEL_YAML" "$MODEL_YAML"
            log "  restored $MODEL_YAML from backup"
        fi
    }
    trap restore_model_yaml EXIT

    "$PY" - "$MODEL_YAML" <<'PY'
import sys, yaml, pathlib
p = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(p.read_text()) or {}
m = cfg.setdefault("model", {})
m["use_hex_kernel"] = True
m["corner_mask"] = True
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"patched {sys.argv[1]}: model.use_hex_kernel=True, model.corner_mask=True")
PY

    BOOTSTRAP_LOG="${LOG_DIR}/v9_bootstrap_v8full.log"
    log "  launching pretrain (logs: $BOOTSTRAP_LOG, expected ~30-60 min)"
    # Warm-start path: load v7full's inference weights into a fresh
    # HexConv2d-trunk model with corner_mask on. Optimizer + scheduler
    # start from scratch — that is exactly the §153 T4 option A recipe.
    set +e
    MALLOC_ARENA_MAX=2 "$PY" -m hexo_rl.bootstrap.pretrain \
        --init-from-weights checkpoints/bootstrap_model_v7full.pt \
        --epochs 30 \
        --eta-min 5e-5 \
        --inference-out checkpoints/bootstrap_model_v8full.pt \
        > "$BOOTSTRAP_LOG" 2>&1
    BOOTSTRAP_RC=$?
    set -e

    if [[ $BOOTSTRAP_RC -ne 0 ]]; then
        log "ERROR: bootstrap pretrain failed RC=$BOOTSTRAP_RC — see $BOOTSTRAP_LOG"
        exit $BOOTSTRAP_RC
    fi
    log "  bootstrap done: checkpoints/bootstrap_model_v8full.pt"

    # Restore model.yaml so smokes pick up their own config overrides without
    # being shadowed by the bootstrap-time globals.
    restore_model_yaml
    trap - EXIT

    cp -f "$BOOTSTRAP_LOG" "$ARCHIVE_DIR/" 2>/dev/null || true
    cp -f checkpoints/bootstrap_model_v8full.pt "$ARCHIVE_DIR/" 2>/dev/null || true
fi

if [[ ! -f checkpoints/bootstrap_model_v8full.pt ]] && [[ $SKIP_BOOTSTRAP -eq 0 ]]; then
    # Step 0 didn't run (already exists) — fine. But every S1/S2/S4 needs it.
    :
fi

# ── Step 1..4: run each smoke ────────────────────────────────────────────────
for entry in "${SMOKES[@]}"; do
    IFS='|' read -r SMOKE_ID VARIANT BOOT <<< "$entry"
    if ! should_run "$SMOKE_ID"; then
        log "$SMOKE_ID skipped (--only filter)"
        continue
    fi
    if [[ ! -f "$BOOT" ]]; then
        log "ERROR: $SMOKE_ID requires $BOOT but it does not exist; skipping"
        continue
    fi

    SMOKE_LOG="${LOG_DIR}/v9_${SMOKE_ID}.log"
    SMOKE_ARCHIVE="$ARCHIVE_DIR/$SMOKE_ID"
    REPORT_DIR="reports/phase_b_prime/v9_smokes/$SMOKE_ID"
    mkdir -p "$SMOKE_ARCHIVE/checkpoints" "$REPORT_DIR"

    log "============================================================="
    log "$SMOKE_ID  variant=$VARIANT  bootstrap=$BOOT  iterations=$ITERATIONS"
    log "  log:     $SMOKE_LOG"
    log "  report:  $REPORT_DIR/"
    log "  archive: $SMOKE_ARCHIVE/"
    log "============================================================="

    # Reset per-run ephemeral state.
    log "  reset: best_model.pt, replay_buffer.bin, checkpoint_*.pt"
    rm -f checkpoints/best_model.pt
    rm -f checkpoints/replay_buffer.bin checkpoints/replay_buffer.bin.recent
    rm -f checkpoints/checkpoint_*.pt

    log "  launching $VARIANT for ${PER_SMOKE_TIMEOUT}s ceiling (expect ~3h)"

    set +e
    MALLOC_ARENA_MAX=2 \
        timeout --signal=TERM --kill-after=120s "$PER_SMOKE_TIMEOUT" \
        "$PY" scripts/train.py \
            --checkpoint "$BOOT" \
            --variant "$VARIANT" \
            --checkpoint-dir "checkpoints/v9_${SMOKE_ID,,}" \
            --no-dashboard \
            --run-name "v9_${SMOKE_ID}" \
            --iterations "$ITERATIONS" \
        > "$SMOKE_LOG" 2>&1
    SMOKE_RC=$?
    set -e

    case $SMOKE_RC in
        0)   log "$SMOKE_ID completed RC=0 (full $ITERATIONS iterations)";;
        124) log "$SMOKE_ID hit timeout after ${PER_SMOKE_TIMEOUT}s";;
        *)   log "$SMOKE_ID exited RC=$SMOKE_RC (unexpected — inspect $SMOKE_LOG)";;
    esac

    # Archive logs + final checkpoint + report.
    cp -f "$SMOKE_LOG" "$SMOKE_ARCHIVE/" 2>/dev/null || true
    find "$LOG_DIR" -maxdepth 1 -name "v9_${SMOKE_ID}*.jsonl" -print -exec cp {} "$SMOKE_ARCHIVE/" \; >>"$DRIVER_LOG" 2>&1 || true
    cp -f checkpoints/best_model.pt "$SMOKE_ARCHIVE/checkpoints/" 2>/dev/null || true
    cp -f checkpoints/replay_buffer.bin "$SMOKE_ARCHIVE/checkpoints/" 2>/dev/null || true
    LATEST_CKPT="$(ls -1 "checkpoints/v9_${SMOKE_ID,,}"/checkpoint_*.pt 2>/dev/null | tail -n 1 || true)"
    if [[ -n "$LATEST_CKPT" ]]; then
        cp -f "$LATEST_CKPT" "$SMOKE_ARCHIVE/checkpoints/"
    fi

    # Copy any instrumentation events the variant produced into the report dir.
    if [[ -d "logs/instrumentation/v9_${SMOKE_ID}" ]]; then
        cp -rT "logs/instrumentation/v9_${SMOKE_ID}" "$REPORT_DIR" 2>/dev/null || true
    fi

    log "$SMOKE_ID archive complete"
done

log "============================================================="
log "ALL REQUESTED SMOKES COMPLETE"
log "Archive root: $ARCHIVE_DIR"
log "Reports root: reports/phase_b_prime/v9_smokes/"
log "============================================================="
