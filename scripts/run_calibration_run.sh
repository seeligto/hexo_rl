#!/usr/bin/env bash
# run_calibration_run.sh — graduation-gate calibration driver (Prompt 8 / §calib).
#
# Runs ONE calibration run (R1..R4) for the configured duration, then restores
# configs/eval.yaml. Resets best_model.pt, replay_buffer.bin, and ephemeral
# checkpoint_*.pt between runs so each R starts from bootstrap weights.
#
# D1 (threshold) + D4 (n_games) live in configs/eval.yaml under eval_pipeline
# (read directly by EvalPipeline, not merged via load_config). This script
# patches them in-place around the run. D2 (eval_interval) + D3 (decay_steps)
# are set in configs/variants/calib_R[1-4].yaml.
#
# Usage:
#   scripts/run_calibration_run.sh R1 [--duration 12600]
#
# Defaults: duration=12600 (3.5 h). Launches in foreground with `timeout`
# so SIGTERM triggers the training loop's graceful checkpoint + buffer save.

set -euo pipefail

RUN_ID="${1:-}"
if [[ -z "$RUN_ID" ]]; then
    echo "Usage: $0 R1|R2|R3|R4 [--duration SEC]" >&2
    exit 2
fi

DURATION=12600  # 3.5 h (= 3h30m)
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration) DURATION="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$RUN_ID" in
    R1) THRESHOLD="0.55"; NGAMES="200" ;;
    R2) THRESHOLD="0.52"; NGAMES="200" ;;
    R3) THRESHOLD="0.55"; NGAMES="200" ;;
    R4) THRESHOLD="0.55"; NGAMES="400" ;;
    *) echo "unknown RUN_ID: $RUN_ID (expected R1..R4)" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EVAL_YAML="configs/eval.yaml"
BACKUP="${EVAL_YAML}.calib_backup"
LOG_DIR="logs"
ARCHIVE_DIR="archive/calibration_$(date +%Y-%m-%d)"
RUN_NAME="calib_${RUN_ID}"
DRIVER_LOG="${LOG_DIR}/${RUN_NAME}.driver.log"
TRAIN_LOG="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "$LOG_DIR" "$ARCHIVE_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRIVER_LOG"; }

# ── 1. Kill any lingering train/bench processes ──────────────────────────────
log "killing any lingering train/bench processes"
pkill -f "scripts/train.py"      2>/dev/null || true
pkill -f "scripts/benchmark.py"  2>/dev/null || true
sleep 2
pgrep -fl "train.py|benchmark.py" >>"$DRIVER_LOG" 2>&1 || log "no lingering procs"

# ── 2. Reset per-run ephemeral state so R starts from bootstrap ──────────────
log "resetting best_model.pt, replay_buffer.bin, checkpoint_*.pt"
rm -f checkpoints/best_model.pt
rm -f checkpoints/replay_buffer.bin
rm -f checkpoints/checkpoint_*.pt

# ── 3. Backup + patch eval.yaml for D1 (threshold) and D4 (n_games) ──────────
if [[ -f "$BACKUP" ]]; then
    log "WARNING: stale backup at $BACKUP — removing (previous run likely crashed mid-patch)"
    rm -f "$BACKUP"
fi
cp "$EVAL_YAML" "$BACKUP"
log "patching $EVAL_YAML: threshold=$THRESHOLD  n_games=$NGAMES"

.venv/bin/python - "$EVAL_YAML" "$THRESHOLD" "$NGAMES" <<'PY'
import sys, yaml, pathlib
path = pathlib.Path(sys.argv[1])
threshold = float(sys.argv[2])
n_games = int(sys.argv[3])
cfg = yaml.safe_load(path.read_text())
ep = cfg.setdefault("eval_pipeline", {})
ep.setdefault("gating", {})["promotion_winrate"] = threshold
ep.setdefault("opponents", {}).setdefault("best_checkpoint", {})["n_games"] = n_games
path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"patched: promotion_winrate={threshold} best_checkpoint.n_games={n_games}")
PY

restore_eval_yaml() {
    if [[ -f "$BACKUP" ]]; then
        log "restoring $EVAL_YAML from backup"
        mv -f "$BACKUP" "$EVAL_YAML"
    fi
}
trap restore_eval_yaml EXIT

# ── 4. Launch training with hard timeout ─────────────────────────────────────
log "launching $RUN_NAME for ${DURATION}s (~$((DURATION/60))m)"
log "variant=${RUN_NAME}  checkpoint=bootstrap_model.pt  no-dashboard"

set +e
MALLOC_ARENA_MAX=2 \
    timeout --signal=TERM --kill-after=60s "$DURATION" \
    .venv/bin/python scripts/train.py \
        --checkpoint checkpoints/bootstrap_model.pt \
        --variant "${RUN_NAME}" \
        --no-dashboard \
        --run-name "${RUN_NAME}" \
    >"$TRAIN_LOG" 2>&1
TRAIN_RC=$?
set -e

if [[ $TRAIN_RC -eq 124 ]]; then
    log "training hit duration timeout (expected) — graceful shutdown"
elif [[ $TRAIN_RC -ne 0 ]]; then
    log "training exited RC=$TRAIN_RC (unexpected — inspect $TRAIN_LOG)"
else
    log "training exited RC=0 (completed early)"
fi

# ── 5. Archive logs + copy run-specific artifacts ────────────────────────────
log "archiving logs to $ARCHIVE_DIR/$RUN_NAME"
mkdir -p "$ARCHIVE_DIR/$RUN_NAME"
# JSONL log: configure_logging writes to logs/<run_name>.jsonl when --run-name is set.
find "$LOG_DIR" -maxdepth 1 -name "${RUN_NAME}*.jsonl" -print -exec cp {} "$ARCHIVE_DIR/$RUN_NAME/" \; | tee -a "$DRIVER_LOG"
cp -f "$TRAIN_LOG"   "$ARCHIVE_DIR/$RUN_NAME/" 2>/dev/null || true
cp -f "$DRIVER_LOG"  "$ARCHIVE_DIR/$RUN_NAME/" 2>/dev/null || true
cp -f "$EVAL_YAML.calib_backup" "$ARCHIVE_DIR/$RUN_NAME/eval.yaml.preserved" 2>/dev/null || true
# Snapshot the patched eval.yaml (pre-restore) for provenance
cp -f "$EVAL_YAML" "$ARCHIVE_DIR/$RUN_NAME/eval.yaml.patched"

# ── 6. Copy the final Trainer + best_model + buffer checkpoints ──────────────
mkdir -p "$ARCHIVE_DIR/$RUN_NAME/checkpoints"
cp -f checkpoints/best_model.pt    "$ARCHIVE_DIR/$RUN_NAME/checkpoints/" 2>/dev/null || log "no best_model.pt produced"
cp -f checkpoints/replay_buffer.bin "$ARCHIVE_DIR/$RUN_NAME/checkpoints/" 2>/dev/null || true
# Only keep the final Trainer checkpoint — the periodic 500-step files bloat archive.
LATEST_CKPT="$(ls -1 checkpoints/checkpoint_*.pt 2>/dev/null | tail -n 1 || true)"
if [[ -n "$LATEST_CKPT" ]]; then
    cp -f "$LATEST_CKPT" "$ARCHIVE_DIR/$RUN_NAME/checkpoints/"
fi

log "archive complete: $ARCHIVE_DIR/$RUN_NAME"
log "RUN $RUN_ID DONE"
