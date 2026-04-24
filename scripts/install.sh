#!/usr/bin/env bash
# scripts/install.sh — single-entrypoint setup for hexo_rl
# Usage: bash scripts/install.sh
# Run from the repository root.
set -euo pipefail

# ── Release artifact constants ─────────────────────────────────────────────────
HF_MODEL_REPO="timmyburn/hexo-bootstrap-models"
HF_CORPUS_REPO="timmyburn/hexo-bootstrap-corpus"
MODEL_FILE="bootstrap_model.pt"
CORPUS_FILE="bootstrap_corpus.npz"
# Corpus repo is public; set WITH_CORPUS=0 to skip download.
WITH_CORPUS="${WITH_CORPUS:-1}"

TOTAL_STEPS=10

ok()   { echo "[ok] $*"; }
fail() { echo "[!!] $*" >&2; }
warn() { echo "[--] WARNING: $*"; }
step() { echo; echo "[$1/$TOTAL_STEPS] $2"; }

# ── Download helper (Hugging Face) ─────────────────────────────────────────────
# hf_download <repo_id> <repo_type> <filename> <dest_dir>
hf_download() {
    local repo="$1" repo_type="$2" filename="$3" dest_dir="$4"
    local dest="$dest_dir/$filename"

    if [[ -f "$dest" ]]; then
        ok "$filename [cached]"
        return 0
    fi

    echo "    Downloading $filename from $repo ..."
    mkdir -p "$dest_dir"
    if ! .venv/bin/hf download "$repo" "$filename" \
            --repo-type "$repo_type" \
            --local-dir "$dest_dir" >/dev/null 2>&1; then
        return 1
    fi
    ok "$filename"
}

# ── Ensure we're running from the repo root ────────────────────────────────────
if [[ ! -f "CLAUDE.md" ]]; then
    fail "Run this script from the repository root (the directory containing CLAUDE.md)."
    exit 1
fi
REPO_ROOT="$(pwd)"

echo "========================================================"
echo "  hexo_rl setup"
echo "========================================================"

# ── [1/9] Detect OS ────────────────────────────────────────────────────────────
step 1 "Detecting OS..."
KERNEL="$(uname -s)"
OS=""
if [[ "$KERNEL" == "Linux" ]]; then
    if grep -qi microsoft /proc/version 2>/dev/null; then
        OS="WSL"
    else
        OS="Linux"
    fi
elif [[ "$KERNEL" == "Darwin" ]]; then
    OS="macOS"
else
    fail "Unsupported OS: $KERNEL. Only Linux, macOS, and WSL are supported."
    exit 1
fi
ok "OS: $OS"

# ── [2/9] Check Python >= 3.11 ─────────────────────────────────────────────────
step 2 "Checking Python version..."
if ! command -v python3 &>/dev/null; then
    fail "python3 not found."
    if [[ "$OS" == "macOS" ]]; then
        fail "  Install with: brew install python@3.11"
    else
        fail "  Install with: sudo apt install python3.11 python3.11-venv"
    fi
    exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
    fail "Python 3.11+ required, found $PY_VERSION."
    if [[ "$OS" == "macOS" ]]; then
        fail "  Install with: brew install python@3.11"
    else
        fail "  Install with: sudo apt install python3.11 python3.11-venv"
    fi
    exit 1
fi
ok "Python $PY_VERSION"

# ── [3/9] Check Rust ───────────────────────────────────────────────────────────
step 3 "Checking Rust toolchain..."
MISSING_RUST=0
if ! command -v rustc &>/dev/null; then MISSING_RUST=1; fi
if ! command -v cargo &>/dev/null; then MISSING_RUST=1; fi

if [[ "$MISSING_RUST" -eq 1 ]]; then
    fail "rustc / cargo not found."
    fail "  Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    fail "  Then reload your shell: source \"\$HOME/.cargo/env\""
    exit 1
fi
ok "rustc $(rustc --version | awk '{print $2}'), cargo $(cargo --version | awk '{print $2}')"

# ── [4/10] Create .venv ───────────────────────────────────────────────────────
step 4 "Setting up Python virtual environment..."
if [[ -d ".venv" ]]; then
    ok ".venv [cached]"
else
    python3 -m venv .venv
    ok ".venv created"
fi
echo "    Upgrading pip, maturin, pybind11..."
.venv/bin/pip install --upgrade --quiet pip maturin pybind11
ok "pip / maturin / pybind11 up to date"

# ── [5/10] Detect CUDA and install PyTorch ────────────────────────────────────
step 5 "Installing PyTorch..."
CUDA_MAJOR=""
CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
    ok "GPU: $GPU_NAME"
    CUDA_VERSION="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')" || true
    CUDA_MAJOR="${CUDA_VERSION%%.*}"
fi

if [[ "$OS" == "macOS" ]]; then
    ok "macOS — installing default PyPI torch (gets MPS build)"
    .venv/bin/pip install --quiet torch
elif [[ "$CUDA_MAJOR" == "13" ]]; then
    ok "CUDA $CUDA_VERSION — installing torch (latest PyPI build, cu130)"
    .venv/bin/pip install --quiet torch
elif [[ "$CUDA_MAJOR" == "12" ]]; then
    ok "CUDA $CUDA_VERSION — installing torch cu121 build"
    .venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
elif [[ -n "$CUDA_MAJOR" ]]; then
    warn "CUDA $CUDA_VERSION detected but no matching wheel — installing CPU torch build"
    warn "  For GPU training, use CUDA 12.x or 13.x."
    .venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
else
    warn "No CUDA detected — installing CPU torch build"
    warn "  Self-play training requires a CUDA GPU."
    .venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
ok "torch installed"

# ── [6/10] Install remaining Python dependencies ──────────────────────────────
step 6 "Installing Python dependencies (requirements.txt)..."
.venv/bin/pip install --quiet -r requirements.txt
ok "Python dependencies installed"

# ── [7/10] Git submodules ─────────────────────────────────────────────────────
step 7 "Updating git submodules..."
git submodule update --init --recursive
ok "Submodules up to date"

# ── [8/10] Build engine ───────────────────────────────────────────────────────
step 8 "Building Rust engine extension..."
env -u CONDA_PREFIX VIRTUAL_ENV="$(pwd)/.venv" .venv/bin/maturin develop --release -m engine/Cargo.toml
ok "Engine built"

echo "    Building SealBot C++ extensions..."
if [[ -d "vendor/bots/sealbot/best" ]]; then
    (cd vendor/bots/sealbot/best && CXXFLAGS="-Wno-reorder" "$REPO_ROOT/.venv/bin/python" setup.py build_ext --inplace --quiet 2>&1) \
        && ok "SealBot best built" \
        || warn "SealBot best build failed (non-fatal — only needed for eval)"
fi
if [[ -d "vendor/bots/sealbot/current" ]]; then
    (cd vendor/bots/sealbot/current && CXXFLAGS="-Wno-reorder" "$REPO_ROOT/.venv/bin/python" setup.py build_ext --inplace --quiet 2>&1) \
        && ok "SealBot current built" \
        || warn "SealBot current build failed (non-fatal — only needed for eval)"
fi

# ── [9/10] Download release artifacts from Hugging Face ──────────────────────
step 9 "Downloading release artifacts from Hugging Face..."
mkdir -p checkpoints data

# Bootstrap model — public repo, no auth required.
if ! hf_download "$HF_MODEL_REPO" model "$MODEL_FILE" checkpoints; then
    warn "Model download failed. Check network / HF availability."
fi

# Corpus — public repo; enabled by default, disable via WITH_CORPUS=0.
if [[ "$WITH_CORPUS" == "1" ]]; then
    if ! hf_download "$HF_CORPUS_REPO" dataset "$CORPUS_FILE" data; then
        warn "Corpus download failed. Check network / HF availability."
    fi
else
    ok "Corpus download skipped (disabled via WITH_CORPUS=0)"
fi

if [[ -f fixtures/threat_probe_baseline.json ]]; then
    ok "threat_probe_baseline.json [git-tracked]"
fi

# ── [10/10] Smoke tests ───────────────────────────────────────────────────────
step 10 "Running smoke tests..."
SMOKE_OK=0
if make test 2>&1; then
    ok "Smoke tests passed"
    SMOKE_OK=1
else
    fail "Smoke tests failed — see output above. Setup is otherwise complete."
    fail "You can debug with: make test.rust  /  make test.py"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "========================================================"
if [[ "$SMOKE_OK" -eq 1 ]]; then
    echo "  Setup complete!"
else
    echo "  Setup complete (smoke tests failed — see above)."
fi
echo
echo "  Next steps:"
echo "    make train          # start self-play training loop"
echo "    make bench          # run full benchmark suite"
echo "    make train.resume   # resume from latest checkpoint"
echo "    Dashboard:          http://localhost:5001"
echo "========================================================"
