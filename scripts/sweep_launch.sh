#!/usr/bin/env bash
# §122 sweep — one-command launch wrapper for an SSH + tmux box.
#
# Usage:
#   ./scripts/sweep_launch.sh                         # full pipeline, no upload, no shutdown
#   ./scripts/sweep_launch.sh --resume                # skip already-completed work
#   ./scripts/sweep_launch.sh --skip-corpus
#   ./scripts/sweep_launch.sh --skip-phase1
#   ./scripts/sweep_launch.sh --skip-phase2
#   ./scripts/sweep_launch.sh --skip-phase3
#   ./scripts/sweep_launch.sh --skip-aggregate
#
# Archival (any combination of these is fine; default is none):
#   --archive-dir /mnt/persistent/sweep_2026-04-25  # rsync artefacts to a local dir
#   --upload-hf <repo_id>                           # push to Hugging Face Hub (dataset repo)
#                                                   # e.g. --upload-hf user/hexo-sweep-122
#
# Hardware overlay (optional, no default — omit for local desktop):
#   --hw-overlay gumbel_targets_epyc4080   # layer hardware config on each variant
#                                          # sets n_workers/batch_size for the box
#
# Box lifecycle (opt-in, NOT default):
#   --shutdown   # poweroff after success. Requires that EITHER --archive-dir
#                # OR --upload-hf has succeeded — refuses to power off if no
#                # durable copy of checkpoints + reports exists.
#
# Phases are idempotent — rerunning is safe. Each phase records progress to
# logs/sweep/state.json; resume picks up where the last run left off.
#
# Default environment assumption: a long-lived SSH session inside tmux.
# Detach with ctrl-b d, reattach with `tmux attach`. Nothing in this script
# requires AWS, S3, or any cloud-vendor CLI.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_PY="${VENV_PY:-.venv/bin/python}"
SKIP_CORPUS=0
SKIP_PHASE1=0
SKIP_PHASE2=0
SKIP_PHASE3=0
SKIP_AGGREGATE=0
ARCHIVE_DIR=""
UPLOAD_HF=""
SHUTDOWN=0
RESUME=0
HW_OVERLAY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-corpus)    SKIP_CORPUS=1 ;;
    --skip-phase1)    SKIP_PHASE1=1 ;;
    --skip-phase2)    SKIP_PHASE2=1 ;;
    --skip-phase3)    SKIP_PHASE3=1 ;;
    --skip-aggregate) SKIP_AGGREGATE=1 ;;
    --resume)         RESUME=1 ;;
    --shutdown)       SHUTDOWN=1 ;;
    --archive-dir)    shift; ARCHIVE_DIR="$1" ;;
    --upload-hf)      shift; UPLOAD_HF="$1" ;;
    --hw-overlay)     shift; HW_OVERLAY="$1" ;;
    --venv)           shift; VENV_PY="$1" ;;
    -h|--help)
      sed -n '2,32p' "$0"
      exit 0
      ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: $VENV_PY not executable; activate your venv or set VENV_PY=" >&2
  exit 1
fi

# Fail-fast: --shutdown without ANY archival destination is a footgun.
# Catch it now instead of after 8 hours of compute.
if [[ "$SHUTDOWN" -eq 1 ]]; then
  if [[ -z "$ARCHIVE_DIR" && -z "$UPLOAD_HF" ]]; then
    echo "ERROR: --shutdown requires at least one archive target." >&2
    echo "       Pass --archive-dir <path> AND/OR --upload-hf <repo_id>." >&2
    echo "       Refusing to power off without a durable copy of artefacts." >&2
    exit 3
  fi
  if [[ -n "$ARCHIVE_DIR" ]]; then
    if ! mkdir -p "$ARCHIVE_DIR" 2>/dev/null; then
      echo "ERROR: --archive-dir '$ARCHIVE_DIR' is not creatable." >&2
      exit 4
    fi
    if [[ ! -w "$ARCHIVE_DIR" ]]; then
      echo "ERROR: --archive-dir '$ARCHIVE_DIR' is not writable." >&2
      exit 4
    fi
  fi
  if [[ -n "$UPLOAD_HF" ]]; then
    if ! "$VENV_PY" -c "import huggingface_hub" 2>/dev/null; then
      echo "ERROR: --upload-hf set but huggingface_hub is not importable in $VENV_PY." >&2
      echo "       Install with: $VENV_PY -m pip install huggingface_hub" >&2
      exit 5
    fi
    if ! "$VENV_PY" -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
      echo "ERROR: --upload-hf set but Hugging Face credentials are missing/invalid." >&2
      echo "       Authenticate via 'hf auth login' (or set HF_TOKEN) and retry." >&2
      exit 6
    fi
  fi
  echo "preflight: archive target(s) OK; --shutdown will fire after successful archive."
fi

echo "==[ env check ]=="
"$VENV_PY" --version
if ! "$VENV_PY" -c "import engine" 2>/dev/null; then
  echo "engine extension not built; running maturin develop --release"
  "$VENV_PY" -m pip install --quiet maturin || true
  maturin develop --release -m engine/Cargo.toml
fi
"$VENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ ! -f data/bootstrap_corpus.npz ]]; then
  echo "ERROR: data/bootstrap_corpus.npz missing — run 'make corpus.export' before launching the sweep" >&2
  exit 1
fi

# ── Clean prior sweep state when starting fresh ─────────────────────────────
# Without --resume, a previous failed/aborted launch's logs + half-written
# checkpoints + stale state.json will silently mix into the new run
# (run_sweep.py reads state.json and skips variants whose phase1_complete is
# true, even if the operator wanted a clean start). Wipe them up front.
# --resume preserves everything for picking up where the last run left off.
if [[ "$RESUME" -ne 1 ]]; then
  _STALE=()
  for p in logs/sweep/state.json logs/sweep/sweep_2ch logs/sweep/sweep_3ch \
           logs/sweep/sweep_4ch logs/sweep/sweep_6ch logs/sweep/sweep_8ch \
           logs/sweep/sweep_18ch logs/tb/sweep checkpoints/sweep; do
    [[ -e "$p" ]] && _STALE+=("$p")
  done
  if [[ ${#_STALE[@]} -gt 0 ]]; then
    echo "==[ clean prior sweep state (use --resume to keep) ]=="
    for p in "${_STALE[@]}"; do
      echo "  rm -rf $p"
      rm -rf "$p"
    done
  fi
fi

if [[ "$SKIP_CORPUS" -eq 0 ]]; then
  echo "==[ corpus regen — six variants ]=="
  if [[ "$RESUME" -eq 1 ]]; then
    "$VENV_PY" scripts/regen_bootstrap_corpus.py --all
  else
    "$VENV_PY" scripts/regen_bootstrap_corpus.py --all --force
  fi
else
  echo "==[ skip corpus regen ]=="
fi

_HW_OVERLAY_ARG=()
if [[ -n "$HW_OVERLAY" ]]; then
  _HW_OVERLAY_ARG=(--hw-overlay "$HW_OVERLAY")
  echo "==[ hw overlay: $HW_OVERLAY ]=="
fi

if [[ "$SKIP_PHASE1" -eq 0 ]]; then
  echo "==[ phase 1 — 6 variants × 2,500 steps ]=="
  "$VENV_PY" scripts/run_sweep.py --phase 1 "${_HW_OVERLAY_ARG[@]}"
else
  echo "==[ skip phase 1 ]=="
fi

if [[ "$SKIP_PHASE2" -eq 0 ]]; then
  echo "==[ phase 2 — survivors → 10,000 steps ]=="
  "$VENV_PY" scripts/run_sweep.py --phase 2 "${_HW_OVERLAY_ARG[@]}"
else
  echo "==[ skip phase 2 ]=="
fi

if [[ "$SKIP_PHASE3" -eq 0 ]]; then
  echo "==[ phase 3 — round-robin tournament ]=="
  "$VENV_PY" scripts/run_sweep.py --phase 3
else
  echo "==[ skip phase 3 ]=="
fi

if [[ "$SKIP_AGGREGATE" -eq 0 ]]; then
  echo "==[ aggregate — write memo.md ]=="
  "$VENV_PY" scripts/aggregate_sweep.py
else
  echo "==[ skip aggregate ]=="
fi

# ── Archival ──────────────────────────────────────────────────────────────
# Two independent destinations; either, both, or neither can be active.
# A successful archive sets ARCHIVE_OK=1, which is required if --shutdown
# was passed. Pre-flight already verified that any requested destination is
# usable; this block does the actual work and records success per target.

ARCHIVE_OK=0
ARTEFACT_PATHS=(
  "checkpoints/sweep"
  "logs/sweep"
  "logs/tb/sweep"
  "reports/investigations/phase122_sweep"
)

if [[ -n "$ARCHIVE_DIR" ]]; then
  echo "==[ local archive → $ARCHIVE_DIR ]=="
  mkdir -p "$ARCHIVE_DIR"
  set +e
  rsync_ok=1
  for src in "${ARTEFACT_PATHS[@]}"; do
    if [[ ! -e "$src" ]]; then
      echo "  (skip: $src does not exist)"
      continue
    fi
    rsync -a --delete "$src/" "$ARCHIVE_DIR/${src//\//_}/" || rsync_ok=0
  done
  cat > "$ARCHIVE_DIR/README.txt" <<EOF
§122 sweep artefacts — archived $(date -u +%Y-%m-%dT%H:%M:%SZ)
Source host: $(hostname) (git $(git -C "$(pwd)" rev-parse --short HEAD 2>/dev/null || echo unknown))

Layout:
  checkpoints_sweep/        — per-variant checkpoint dirs (every 500 steps)
  logs_sweep/               — driver state.json + per-variant structlog jsonl
  logs_tb_sweep/            — TensorBoard scalar mirrors
  reports_investigations_phase122_sweep/
                            — memo.md, tournament.{json,md}, throughput.json,
                              per_variant_curves.json

To inspect on another box:
  rsync -a <this_dir>/ ./hexo_rl_sweep_replay/
  cat reports_investigations_phase122_sweep/memo.md
EOF
  set -e
  if [[ "$rsync_ok" -eq 1 ]]; then
    ARCHIVE_OK=1
    echo "local archive complete; manifest at $ARCHIVE_DIR/README.txt"
  else
    echo "ERROR: rsync to $ARCHIVE_DIR failed for one or more targets." >&2
    if [[ "$SHUTDOWN" -eq 1 ]]; then
      echo "       --shutdown blocked: archive did not complete." >&2
      exit 7
    fi
  fi
fi

if [[ -n "$UPLOAD_HF" ]]; then
  echo "==[ HF Hub upload → $UPLOAD_HF ]=="
  set +e
  "$VENV_PY" - "$UPLOAD_HF" "${ARTEFACT_PATHS[@]}" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

repo_id = sys.argv[1]
artefact_paths = sys.argv[2:]
api = HfApi()
# Idempotent: create the dataset repo if it doesn't exist; existing repos
# are left alone (exist_ok=True).
create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)

failed = []
for src in artefact_paths:
    p = Path(src)
    if not p.exists():
        print(f"  (skip: {src} does not exist)")
        continue
    # Mirror the local layout under the same path inside the repo.
    print(f"  upload_folder {p} → {repo_id}:{src}")
    try:
        api.upload_folder(
            folder_path=str(p),
            path_in_repo=src,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"sweep update: {src}",
        )
    except Exception as exc:
        print(f"  FAILED: {src}: {exc}", file=sys.stderr)
        failed.append(src)

if failed:
    sys.exit(1)
PY
  hf_rc=$?
  set -e
  if [[ $hf_rc -eq 0 ]]; then
    ARCHIVE_OK=1
    echo "HF Hub upload complete."
  else
    echo "ERROR: HF Hub upload failed." >&2
    if [[ "$SHUTDOWN" -eq 1 ]]; then
      echo "       --shutdown blocked: archive did not complete." >&2
      exit 8
    fi
  fi
fi

if [[ "$SHUTDOWN" -eq 1 ]]; then
  if [[ "$ARCHIVE_OK" -ne 1 ]]; then
    echo "ERROR: --shutdown requires ARCHIVE_OK=1; got ARCHIVE_OK=$ARCHIVE_OK" >&2
    exit 9
  fi
  echo "==[ shutdown ]=="
  if command -v sudo >/dev/null 2>&1; then
    sudo poweroff
  else
    poweroff
  fi
fi

echo "==[ done ]=="
echo "Artefacts on disk:"
for p in "${ARTEFACT_PATHS[@]}"; do
  if [[ -e "$p" ]]; then
    du -sh "$p" 2>/dev/null | sed 's/^/  /'
  fi
done
