#!/usr/bin/env bash
# Probe the `make install` jumpstart artifacts end-to-end against the live
# Hugging Face repos, in an isolated temp dir — without running a full install.
#
#   1. download bootstrap_model.pt          (PRIVATE model repo — authenticated)
#   2. download hexo_human_corpus.jsonl      (public dataset repo, curl)
#   3. rebuild data/bootstrap_corpus.npz     (export_corpus_npz.py --from-jsonl)
#   4. load model + corpus, run a forward pass, assert the model's input planes
#      match the rebuilt corpus (encoding consistency) for CORPUS_ENCODING.
#
# Usage:  bash scripts/probe_install_jumpstart.sh
#         CORPUS_ENCODING=v6 bash scripts/probe_install_jumpstart.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"   # override to run from a worktree that has no .venv
ENC="${CORPUS_ENCODING:-v6_live2}"
MAXPOS="${MAXPOS:-5000}"
PROBE="$(mktemp -d "${TMPDIR:-/tmp}/hexo_install_probe.XXXXXX")"
MODEL_REPO="timmyburn/hexo-bootstrap-models"
JSONL_URL="https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus/resolve/main/hexo_human_corpus.jsonl"

echo "probe dir: $PROBE   encoding: $ENC"
mkdir -p "$PROBE/data" "$PROBE/checkpoints"

# Model repo is private; pull via authenticated hf CLI (uses ~/.cache/huggingface
# token). install.sh itself uses unauth curl and cache-skips if the model is
# already present — that path is intentionally not exercised here.
echo "[1/4] download bootstrap_model.pt (private repo, authenticated) ..."
"$VENV/bin/hf" download "$MODEL_REPO" bootstrap_model.pt --repo-type model \
    --local-dir "$PROBE/checkpoints" >/dev/null
echo "      sha256=$(sha256sum "$PROBE/checkpoints/bootstrap_model.pt" | awk '{print $1}')"

echo "[2/4] download hexo_human_corpus.jsonl ..."
curl -fsSL "$JSONL_URL" -o "$PROBE/data/hexo_human_corpus.jsonl"
echo "      lines=$(wc -l < "$PROBE/data/hexo_human_corpus.jsonl")"

# Resolve the encoding-specific corpus filename the TRAINER would read — the
# same call install.sh uses — so this probe catches an install/trainer path drift.
RESOLVED_BN="$( cd "$ROOT" && "$VENV/bin/python" -c \
    "from hexo_rl.encoding import lookup, resolve_corpus_path; print(resolve_corpus_path(lookup('$ENC')).name)" )"
echo "[3/4] rebuild $RESOLVED_BN (encoding=$ENC, max=$MAXPOS) ..."
( cd "$ROOT" && "$VENV/bin/python" scripts/export_corpus_npz.py \
    --from-jsonl "$PROBE/data/hexo_human_corpus.jsonl" \
    --encoding "$ENC" --no-compress --max-positions "$MAXPOS" \
    --out "$PROBE/data/$RESOLVED_BN" ) 2>&1 | grep -E "qualifying|States|Policies|Saved|Encoding" || true

echo "[4/4] load model + corpus, forward pass, assert consistency + resolver agreement ..."
( cd "$ROOT" && "$VENV/bin/python" - "$PROBE" "$ENC" "$RESOLVED_BN" <<'PY'
import sys, numpy as np, torch
probe, enc, resolved_bn = sys.argv[1], sys.argv[2], sys.argv[3]
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.encoding import lookup, resolve_corpus_path
model, spec, label = load_model_with_encoding(f"{probe}/checkpoints/bootstrap_model.pt", torch.device("cpu"))
model.eval()
in_ch = model.trunk.input_conv.weight.shape[1]
z = np.load(f"{probe}/data/{resolved_bn}")
states = z["states"]
print(f"      model: label={label} spec={spec.name} in_planes={in_ch}")
print(f"      corpus: {resolved_bn} states={states.shape} policies={z['policies'].shape}")
# (a) model input planes == corpus planes
assert in_ch == states.shape[1], f"plane mismatch: model {in_ch} vs corpus {states.shape[1]}"
# (b) the corpus the TRAINER resolves for the MODEL's own encoding is the file we built
trainer_path = resolve_corpus_path(lookup(spec.name)).name
assert trainer_path == resolved_bn, f"resolver drift: model encoding -> {trainer_path}, built {resolved_bn}"
with torch.no_grad():
    out = model(torch.from_numpy(states[:4].astype(np.float32)))
heads = out if isinstance(out, (tuple, list)) else (out,)
assert heads[0].shape[0] == 4, heads[0].shape
print(f"      forward OK: {len(heads)} heads; head0={tuple(heads[0].shape)}")
print(f"      resolver agreement OK: model encoding {spec.name} -> {trainer_path}")
print("\nPROBE PASS — model+corpus encoding-consistent, forward runs, install path == trainer path.")
PY
)
echo "(left artifacts in $PROBE — rm -rf when done)"
