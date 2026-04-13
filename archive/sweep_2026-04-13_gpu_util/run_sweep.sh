#!/bin/bash
# Phase 2 GPU util sweep — runs 3 × 20-min windows sequentially.
# Each run: fresh bootstrap_model.pt, dedicated checkpoint dir, nvidia-smi dmon tee.
set -u  # no -e: want to continue on training failure for diagnosis

SWEEP_DIR="archive/sweep_2026-04-13_gpu_util"
DURATION=1200  # 20 minutes

run_one() {
    local name="$1"
    local run_dir="$SWEEP_DIR/run_${name}"
    local override="$run_dir/override.yaml"
    local ckpt_dir="$run_dir/ckpt"
    local run_name="sweep_run_${name}"

    echo "=== Run ${name} starting at $(date -u +%FT%TZ) ==="

    # Fresh checkpoint sandbox (no stale buffer, no stale best_model)
    rm -rf "$ckpt_dir"
    mkdir -p "$ckpt_dir"
    cp checkpoints/bootstrap_model.pt "$ckpt_dir/bootstrap_model.pt"

    # Start nvidia-smi dmon -> archive dir (1 Hz)
    nvidia-smi dmon -s u -d 1 -o T > "$run_dir/dmon.log" 2>&1 &
    local dmon_pid=$!
    echo "dmon pid: $dmon_pid"

    # Launch training in background
    # --no-dashboard to disable Rich+Flask (no X needed, reduced noise)
    # Use --run-name so jsonl path is predictable
    env MALLOC_ARENA_MAX=2 .venv/bin/python scripts/train.py \
        --checkpoint "$ckpt_dir/bootstrap_model.pt" \
        --config "$override" \
        --variant gumbel_targets \
        --checkpoint-dir "$ckpt_dir" \
        --no-dashboard \
        --run-name "$run_name" \
        > "$run_dir/train.log" 2>&1 &
    local train_pid=$!
    echo "train pid: $train_pid"
    echo "$train_pid" > "$run_dir/train.pid"

    # Wait for window
    echo "sleeping ${DURATION}s..."
    sleep "$DURATION"

    echo "=== Run ${name} killing at $(date -u +%FT%TZ) ==="
    # Graceful SIGTERM first so the trainer writes session_end and flushes JSONL
    kill -TERM "$train_pid" 2>/dev/null || true
    # Give it 30s to flush
    for i in $(seq 1 30); do
        if ! kill -0 "$train_pid" 2>/dev/null; then break; fi
        sleep 1
    done
    kill -KILL "$train_pid" 2>/dev/null || true
    wait "$train_pid" 2>/dev/null || true

    kill -TERM "$dmon_pid" 2>/dev/null || true
    wait "$dmon_pid" 2>/dev/null || true

    # Copy JSONL into archive dir
    if [ -f "logs/${run_name}.jsonl" ]; then
        cp "logs/${run_name}.jsonl" "$run_dir/train.jsonl"
    else
        echo "WARN: logs/${run_name}.jsonl not found"
        ls -lh logs/ | tail -5 >> "$run_dir/train.log"
    fi

    echo "=== Run ${name} done at $(date -u +%FT%TZ) ==="
}

# Kill any lingering trainers before starting
pkill -f "scripts/train.py" 2>/dev/null || true
pkill -f "nvidia-smi dmon" 2>/dev/null || true
sleep 2

run_one a
sleep 5
run_one b
sleep 5
run_one c

echo "=== SWEEP COMPLETE $(date -u +%FT%TZ) ==="
