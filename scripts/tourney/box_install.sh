#!/usr/bin/env bash
# box_install.sh — Idempotent base-env provisioning for D-K TOURNEY vast box.
# Re-runnable. Stops before repo rsync / maturin develop (WP3 gate).
#
# USAGE:
#   bash box_install.sh            # run locally (SSHes to box)
#   bash -x box_install.sh         # trace mode
#
# Parameterized at top — edit these to match current rental:
BOX_HOST="ssh7.vast.ai"
BOX_PORT="13523"
BOX_KEY="$HOME/.ssh/vast_hexo"
BOX_USER="root"
WORKSPACE="/workspace/hexo_rl"
VENV="$WORKSPACE/.venv"

set -euo pipefail

SSH="ssh -i $BOX_KEY -p $BOX_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 $BOX_USER@$BOX_HOST"

echo "=== D-K TOURNEY box provisioning ==="
echo "Target: $BOX_USER@$BOX_HOST:$BOX_PORT"
echo ""

$SSH bash -s <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:/.uv/python_bin:$HOME/.cargo/bin:$PATH"

WORKSPACE="/workspace/hexo_rl"
VENV="$WORKSPACE/.venv"

echo "--- [1/6] uv ---"
if command -v uv &>/dev/null; then
    echo "uv already present: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "uv installed: $(uv --version)"
fi

echo ""
echo "--- [2/6] Python 3.14 ---"
if uv python list 2>/dev/null | grep -q "cpython-3\.14"; then
    echo "Python 3.14 already present"
    uv python list 2>/dev/null | grep "cpython-3\.14" | head -2
else
    uv python install 3.14
fi

echo ""
echo "--- [3/6] venv at $VENV ---"
mkdir -p "$WORKSPACE"
if [ -d "$VENV" ]; then
    echo "venv already exists: $("$VENV/bin/python" --version)"
else
    uv venv "$VENV" --python 3.14
    echo "venv created: $("$VENV/bin/python" --version)"
fi

echo ""
echo "--- [4/6] torch cu130 nightly ---"
# Local env uses torch 2.12.0.dev20260329+cu130 (nightly; not on stable PyPI).
# cu130 nightly index minimum available = 2.14.0.dev20260711+cu130 as of 2026-07-12.
# This is a NEWER nightly of the same cu130 series — Blackwell sm_120 support
# is identical (both require cu130; capability (12,0) verified PASS).
# If the exact dev date changes, pin to latest available >=2.14.0.dev,<2.15.
TORCH_VER=$("$VENV/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -n "$TORCH_VER" ]; then
    echo "torch already installed: $TORCH_VER"
else
    uv pip install --python "$VENV/bin/python" \
        "torch==2.14.0.dev20260711+cu130" \
        --index-url https://download.pytorch.org/whl/nightly/cu130
    echo "torch installed: $("$VENV/bin/python" -c "import torch; print(torch.__version__)")"
fi

echo ""
echo "--- [5/6] project deps ---"
# Generated from local .venv pip freeze 2026-07-12.
# Excludes: editable/-e installs, engine extension, torch+nvidia transitive deps.
# torch-transitive (sympy, filelock, fsspec, networkx, mpmath, packaging,
# setuptools, typing_extensions) installed as torch deps — skip to avoid conflicts.
cat > /tmp/hexo_requirements.txt <<'REQS'
aiodns==3.6.1
aiohappyeyeballs==2.6.1
aiohttp==3.13.5
aiosignal==1.4.0
annotated-doc==0.0.4
anyio==4.13.0
argcomplete==3.6.3
attrs==26.1.0
bidict==0.23.1
blinker==1.9.0
borb==2.1.25
certifi==2026.2.25
cffi==2.0.0
charset-normalizer==3.4.6
click==8.4.1
contourpy==1.3.3
cryptography==46.0.4
curlify==3.0.0
cycler==0.12.1
Cython==3.2.8
Flask==3.1.3
Flask-SocketIO==5.6.1
fonttools==4.62.1
frozenlist==1.8.0
gevent==26.4.0
gevent-websocket==0.10.1
greenlet==3.4.0
h11==0.16.0
hf-xet==1.4.3
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.17.0
idna==3.11
iniconfig==2.3.0
itsdangerous==2.2.0
Jinja2==3.1.6
kiwisolver==1.5.0
lxml==6.1.0
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.8
maturin==1.13.1
mdurl==0.1.2
multidict==6.7.1
numpy==2.2.6
pillow==12.1.1
plotext==5.3.2
pluggy==1.6.0
propcache==0.4.1
psutil==6.1.1
pybind11==3.0.4
pycares==4.11.0
pycparser==3.0
pycryptodome==3.23.0
Pygments==2.20.0
pynvml==13.0.1
pyparsing==3.3.2
py-spy==0.4.2
pytest==9.0.2
pytest-timeout==2.4.0
python-barcode==0.16.1
python-dateutil==2.9.0.post0
python-engineio==4.13.1
python-socketio==5.16.1
PyYAML==6.0.3
qrcode==8.2
requests==2.33.0
rich==14.3.3
ruff==0.15.12
scipy==1.17.1
shellingham==1.5.4
simple-websocket==1.1.0
six==1.17.0
structlog==25.5.0
tqdm==4.67.3
typer==0.24.2
urllib3==2.6.3
vastai==1.0.9
Werkzeug==3.1.7
wsproto==1.3.2
xdg==6.0.0
yarl==1.23.0
zope.event==6.1
zope.interface==8.3
REQS

uv pip install --python "$VENV/bin/python" \
    -r /tmp/hexo_requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
    --index-strategy unsafe-best-match
echo "project deps: OK"

echo ""
echo "--- [6/6] Rust toolchain ---"
if command -v cargo &>/dev/null; then
    echo "Rust already installed: $(rustc --version) / $(cargo --version)"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "Rust installed: $(rustc --version)"
fi

echo ""
echo "=== Env summary ==="
echo "  python:    $("$VENV/bin/python" --version)"
echo "  torch:     $("$VENV/bin/python" -c "import torch; print(torch.__version__, 'cuda:', torch.version.cuda)")"
echo "  numpy:     $("$VENV/bin/python" -c "import numpy; print(numpy.__version__)")"
echo "  maturin:   $("$VENV/bin/maturin" --version)"
echo "  rustc:     $(rustc --version)"
echo "  cargo:     $(cargo --version)"
echo "  uv:        $(uv --version)"

echo ""
echo "=== CUDA de-risk ==="
"$VENV/bin/python" -c "
import torch
print('cuda_available:', torch.cuda.is_available())
print('device_capability:', torch.cuda.get_device_capability())
print('device_name:', torch.cuda.get_device_name(0))
a = torch.randn(1024, 1024, device='cuda')
b = torch.randn(1024, 1024, device='cuda')
c = (a @ b).cpu()
print('matmul:', c.shape, '-> PASS')
"

echo ""
echo "=== PROVISIONING COMPLETE ==="
echo ""
echo "TODO (WP3 gate — do NOT run here yet):"
echo "  # rsync repo bundle to box"
echo "  # rsync -avz --delete -e 'ssh -i KEY -p PORT' LOCAL_REPO/ root@HOST:$WORKSPACE/"
echo "  # cd $WORKSPACE && source .venv/bin/activate"
echo "  # cd engine && maturin develop --release"
echo "  # python -c 'import engine; print(engine.__version__)'"
echo "  # python scripts/smoke_selfplay_bootstrap.py --fast"
REMOTE

echo ""
echo "=== LOCAL box_install.sh done ==="
