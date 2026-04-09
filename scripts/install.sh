#!/usr/bin/env bash
# scripts/install.sh — single-entrypoint setup for hexo_rl
# Usage: bash scripts/install.sh
# Run from the repository root.
set -euo pipefail

# ── Release artifact constants ─────────────────────────────────────────────────
RELEASE_BASE_URL="https://github.com/seeligto/hexo_rl/releases/download/v0.1-bootstrap"
BOOTSTRAP_MODEL_SHA256="01862660e4850a517f45f359db5763975341e7dd35c67f593ab21f38a73a9670"
BOOTSTRAP_CORPUS_SHA256="678079c12c65209af1d9ce9969da0455b4c16ccee353ca214c336943758309aa"

TOTAL_STEPS=9

ok()   { echo "[ok] $*"; }
fail() { echo "[!!] $*" >&2; }
warn() { echo "[--] WARNING: $*"; }
step() { echo; echo "[$1/$TOTAL_STEPS] $2"; }

# ── SHA-256 helper ─────────────────────────────────────────────────────────────
if command -v sha256sum &>/dev/null; then
    sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum &>/dev/null; then
    sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    fail "No sha256sum or shasum found. Install coreutils and re-run."
    exit 1
fi

# ── Download + verify helper ───────────────────────────────────────────────────
# download_and_verify <url> <dest_path> <expected_sha256>
download_and_verify() {
    local url="$1" dest="$2" expected="$3"
    local name
    name="$(basename "$dest")"

    if [[ -f "$dest" ]]; then
        local actual
        actual="$(sha256 "$dest")"
        if [[ "$actual" == "$expected" ]]; then
            ok "$name [cached]"
            return 0
        else
            warn "$name exists but hash mismatch — re-downloading."
            rm -f "$dest"
        fi
    fi

    echo "    Downloading $name ..."
    curl -fL --progress-bar "$url" -o "$dest"

    local actual
    actual="$(sha256 "$dest")"
    if [[ "$actual" != "$expected" ]]; then
        fail "Hash mismatch for $name"
        fail "  expected: $expected"
        fail "  got:      $actual"
        rm -f "$dest"
        exit 1
    fi
    ok "$name"
}

# ── Ensure we're running from the repo root ────────────────────────────────────
if [[ ! -f "CLAUDE.md" ]]; then
    fail "Run this script from the repository root (the directory containing CLAUDE.md)."
    exit 1
fi

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

# ── [4/9] Check NVIDIA / CUDA ─────────────────────────────────────────────────
step 4 "Checking NVIDIA / CUDA..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
    ok "GPU: $GPU_NAME"
else
    warn "nvidia-smi not found. Continuing in CPU-only mode."
    warn "  Benchmarks will work but self-play training requires a CUDA GPU."
    warn "  Install the NVIDIA driver + CUDA toolkit for full functionality."
fi

# ── [5/9] Create .venv and install Python dependencies ────────────────────────
step 5 "Setting up Python virtual environment..."
if [[ -d ".venv" ]]; then
    ok ".venv [cached]"
else
    python3 -m venv .venv
    ok ".venv created"
fi

echo "    Upgrading pip, maturin, pybind11..."
.venv/bin/pip install --upgrade --quiet pip maturin pybind11

echo "    Installing requirements.txt..."
.venv/bin/pip install --quiet -r requirements.txt
ok "Python dependencies installed"

# ── [6/9] Git submodules ───────────────────────────────────────────────────────
step 6 "Updating git submodules..."
git submodule update --init --recursive
ok "Submodules up to date"

# ── [7/9] Build engine ────────────────────────────────────────────────────────
step 7 "Building Rust engine extension..."
.venv/bin/maturin develop --release -m engine/Cargo.toml
ok "Engine built"

echo "    Building SealBot C++ extensions..."
if [[ -d "vendor/bots/sealbot/best" ]]; then
    (cd vendor/bots/sealbot/best && .venv/bin/python setup.py build_ext --inplace --quiet 2>&1) \
        && ok "SealBot best built" \
        || warn "SealBot best build failed (non-fatal — only needed for eval)"
fi
if [[ -d "vendor/bots/sealbot/current" ]]; then
    (cd vendor/bots/sealbot/current && .venv/bin/python setup.py build_ext --inplace --quiet 2>&1) \
        && ok "SealBot current built" \
        || warn "SealBot current build failed (non-fatal — only needed for eval)"
fi

# ── [8/9] Download release artifacts ─────────────────────────────────────────
step 8 "Downloading release artifacts..."
mkdir -p checkpoints data

download_and_verify \
    "$RELEASE_BASE_URL/bootstrap_model.pt" \
    "checkpoints/bootstrap_model.pt" \
    "$BOOTSTRAP_MODEL_SHA256"

download_and_verify \
    "$RELEASE_BASE_URL/bootstrap_corpus.npz" \
    "data/bootstrap_corpus.npz" \
    "$BOOTSTRAP_CORPUS_SHA256"

# ── [9/9] Smoke tests ─────────────────────────────────────────────────────────
step 9 "Running smoke tests..."
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
echo "    make bench.full     # run full benchmark suite"
echo "    make train.resume   # resume from latest checkpoint"
echo "    Dashboard:          http://localhost:5001"
echo "========================================================"
