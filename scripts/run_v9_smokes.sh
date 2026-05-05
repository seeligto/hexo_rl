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

# ── Step 0: produce the v8full bootstrap (T4) ───────────────────────────────
#
# T4 produces TWO candidate v8full bootstraps and picks the better:
#
#   * v8full_warm    — warm-start from v7full's weights (clean optimizer +
#                      scheduler, hex_kernel + corner_mask on)
#   * v8full_scratch — fresh from-scratch 30-epoch retrain, same recipe
#                      with hex_kernel + corner_mask on
#
# Comparator: threat-probe gates (C1/C2/C3) + final pretrain loss + a
# 100-game SealBot eval (~10 min per candidate at time_limit=0.1s).
# Winner is copied to checkpoints/bootstrap_model_v8full.pt for the
# smoke matrix; both candidates are archived for later forensic comparison.
#
# Skip with --skip-bootstrap if the v8full file already exists from a
# prior run.

if [[ $SKIP_BOOTSTRAP -eq 0 ]] && [[ ! -f checkpoints/bootstrap_model_v8full.pt ]]; then
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

    # ── Step 0a: warm-start ─────────────────────────────────────────────────
    log "Step 0a — warm-start v7full → v8full_warm"
    log "  recipe: --init-from-weights bootstrap_model_v7full.pt --epochs 30 --eta-min 5e-5"
    WARM_LOG="${LOG_DIR}/v9_bootstrap_v8full_warm.log"
    rm -f checkpoints/pretrain/pretrain_*.pt
    set +e
    MALLOC_ARENA_MAX=2 "$PY" -m hexo_rl.bootstrap.pretrain \
        --init-from-weights checkpoints/bootstrap_model_v7full.pt \
        --epochs 30 \
        --eta-min 5e-5 \
        --inference-out checkpoints/bootstrap_model_v8full_warm.pt \
        > "$WARM_LOG" 2>&1
    WARM_RC=$?
    set -e
    if [[ $WARM_RC -ne 0 ]]; then
        log "ERROR: warm-start pretrain failed RC=$WARM_RC — see $WARM_LOG"
        exit $WARM_RC
    fi
    log "  warm-start done: checkpoints/bootstrap_model_v8full_warm.pt"

    # ── Step 0b: from-scratch ───────────────────────────────────────────────
    log "Step 0b — from-scratch hex+corner_mask retrain → v8full_scratch"
    log "  recipe: --epochs 30 --eta-min 5e-5  (matches §150 v7full recipe)"
    SCRATCH_LOG="${LOG_DIR}/v9_bootstrap_v8full_scratch.log"
    rm -f checkpoints/pretrain/pretrain_*.pt
    set +e
    MALLOC_ARENA_MAX=2 "$PY" -m hexo_rl.bootstrap.pretrain \
        --epochs 30 \
        --eta-min 5e-5 \
        --inference-out checkpoints/bootstrap_model_v8full_scratch.pt \
        > "$SCRATCH_LOG" 2>&1
    SCRATCH_RC=$?
    set -e
    if [[ $SCRATCH_RC -ne 0 ]]; then
        log "ERROR: scratch pretrain failed RC=$SCRATCH_RC — see $SCRATCH_LOG"
        exit $SCRATCH_RC
    fi
    log "  scratch done: checkpoints/bootstrap_model_v8full_scratch.pt"

    # Restore model.yaml so smokes pick up their own per-variant overrides
    # without being shadowed by the bootstrap-time globals (smoke variants
    # set model.use_hex_kernel / corner_mask directly in their YAMLs).
    restore_model_yaml
    trap - EXIT

    # ── Step 0c: compare candidates ─────────────────────────────────────────
    log "Step 0c — comparing candidates (threat probe + final loss + SealBot WR)"
    PROBE_WARM="${LOG_DIR}/v9_probe_warm.txt"
    PROBE_SCRATCH="${LOG_DIR}/v9_probe_scratch.txt"
    set +e
    "$PY" scripts/probe_threat_logits.py \
        --checkpoint checkpoints/bootstrap_model_v8full_warm.pt \
        > "$PROBE_WARM" 2>&1
    PROBE_WARM_RC=$?
    "$PY" scripts/probe_threat_logits.py \
        --checkpoint checkpoints/bootstrap_model_v8full_scratch.pt \
        > "$PROBE_SCRATCH" 2>&1
    PROBE_SCRATCH_RC=$?
    set -e
    log "  probe warm    RC=$PROBE_WARM_RC  (PASS=0)"
    log "  probe scratch RC=$PROBE_SCRATCH_RC (PASS=0)"

    SEALBOT_WARM="${LOG_DIR}/v9_sealbot_warm.jsonl"
    SEALBOT_SCRATCH="${LOG_DIR}/v9_sealbot_scratch.jsonl"
    log "  SealBot eval n=100 time_limit=0.1s (~10 min each)"
    set +e
    "$PY" scripts/eval_vs_sealbot.py \
        --checkpoint checkpoints/bootstrap_model_v8full_warm.pt \
        --n-games 100 --time-limit 0.1 --model-sims 96 \
        --out "$SEALBOT_WARM" \
        > "${LOG_DIR}/v9_sealbot_warm.log" 2>&1
    "$PY" scripts/eval_vs_sealbot.py \
        --checkpoint checkpoints/bootstrap_model_v8full_scratch.pt \
        --n-games 100 --time-limit 0.1 --model-sims 96 \
        --out "$SEALBOT_SCRATCH" \
        > "${LOG_DIR}/v9_sealbot_scratch.log" 2>&1
    set -e

    # Pick winner — preference order:
    #   1. Both probes pass: take the higher SealBot WR; tiebreak on lower final loss.
    #   2. Only one probe passes: take that one.
    #   3. Neither passes: surface an error (do NOT silently fall through).
    "$PY" - <<PY
import json, pathlib, re, sys

def final_loss(log_path):
    p = pathlib.Path(log_path)
    if not p.exists():
        return None
    last = None
    for line in p.read_text().splitlines():
        if "epoch_complete" in line:
            m = re.search(r"loss=([0-9.]+)", line)
            if m:
                last = float(m.group(1))
    return last

def sealbot_wr(jsonl_path):
    p = pathlib.Path(jsonl_path)
    if not p.exists():
        return None
    rec = None
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
    if rec is None:
        return None
    # eval_vs_sealbot writes per-checkpoint summary records with a wins/draws/losses
    # tally; convert to WR (X+0.5*draws)/total.
    wins = rec.get("wins", 0); draws = rec.get("draws", 0); losses = rec.get("losses", 0)
    total = wins + draws + losses
    return None if total == 0 else (wins + 0.5 * draws) / total

warm_loss = final_loss("${WARM_LOG}")
scratch_loss = final_loss("${SCRATCH_LOG}")
warm_wr = sealbot_wr("${SEALBOT_WARM}")
scratch_wr = sealbot_wr("${SEALBOT_SCRATCH}")
warm_pass = ${PROBE_WARM_RC} == 0
scratch_pass = ${PROBE_SCRATCH_RC} == 0

print(f"warm    : probe_pass={warm_pass}  sealbot_wr={warm_wr}  final_loss={warm_loss}")
print(f"scratch : probe_pass={scratch_pass}  sealbot_wr={scratch_wr}  final_loss={scratch_loss}")

if warm_pass and not scratch_pass:
    winner = "warm"
elif scratch_pass and not warm_pass:
    winner = "scratch"
elif not warm_pass and not scratch_pass:
    print("FATAL: neither candidate passed threat probe")
    sys.exit(2)
else:
    # Both pass — use SealBot WR; tiebreak (within 1pp) on lower loss.
    a = warm_wr if warm_wr is not None else 0.0
    b = scratch_wr if scratch_wr is not None else 0.0
    if abs(a - b) < 0.01:
        winner = "warm" if (warm_loss or 9e9) <= (scratch_loss or 9e9) else "scratch"
    else:
        winner = "warm" if a >= b else "scratch"

pathlib.Path("${LOG_DIR}/v9_bootstrap_winner").write_text(winner + "\n")
print(f"WINNER: {winner}")
PY
    PICK_RC=$?
    if [[ $PICK_RC -ne 0 ]]; then
        log "ERROR: comparator failed (neither candidate viable)"
        exit $PICK_RC
    fi
    WINNER="$(cat ${LOG_DIR}/v9_bootstrap_winner)"
    log "  comparator picked: $WINNER"

    # ── Step 0d: install the winner ─────────────────────────────────────────
    cp -f "checkpoints/bootstrap_model_v8full_${WINNER}.pt" \
          "checkpoints/bootstrap_model_v8full.pt"
    log "Step 0d — installed bootstrap_model_v8full_${WINNER}.pt → bootstrap_model_v8full.pt"

    cp -f "$WARM_LOG" "$SCRATCH_LOG" "$PROBE_WARM" "$PROBE_SCRATCH" \
          "$SEALBOT_WARM" "$SEALBOT_SCRATCH" \
          "${LOG_DIR}/v9_sealbot_warm.log" "${LOG_DIR}/v9_sealbot_scratch.log" \
          "$ARCHIVE_DIR/" 2>/dev/null || true
    cp -f checkpoints/bootstrap_model_v8full_warm.pt \
          checkpoints/bootstrap_model_v8full_scratch.pt \
          "$ARCHIVE_DIR/" 2>/dev/null || true
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
