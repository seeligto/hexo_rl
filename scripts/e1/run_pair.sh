#!/usr/bin/env bash
# ── T8 D5 — E1 paired scalar-vs-dist65 launch harness ────────────────────────
#
# Launches the two E1 arms SEQUENTIALLY, SCALAR ARM FIRST (pre-registered), both
# warm-started from the SAME weights-only 248k trunk with the SAME shared seed
# (identical data order — one-variable discipline). Before launching, it asserts
# the two variants' resolved configs differ in EXACTLY {value_head_type} and
# REFUSES (exit nonzero) otherwise.
#
# Per-arm isolation is by --checkpoint-dir / --run-name (the yamls are byte-
# identical apart from value_head_type, so isolation cannot come from them).
#
# Usage:
#   scripts/e1/run_pair.sh
# Env overrides (optional):
#   E1_TRUNK   weights-only 248k trunk (default checkpoints/e1/checkpoint_00248000_weights_only.pt)
#   E1_ITERS   iterations per arm      (default 50000)
#   PY         python interpreter      (default .venv/bin/python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-.venv/bin/python}"
E1_TRUNK="${E1_TRUNK:-checkpoints/e1/checkpoint_00248000_weights_only.pt}"
E1_ITERS="${E1_ITERS:-50000}"

SCALAR_VARIANT="e1_scalar"
DIST_VARIANT="e1_dist65"

echo "=== E1 run_pair — one-key-diff gate ==="
# HARD GATE: refuse to launch unless the resolved diff is EXACTLY value_head_type.
"$PY" -m scripts.e1.assert_one_key_diff "$SCALAR_VARIANT" "$DIST_VARIANT"

if [[ ! -f "$E1_TRUNK" ]]; then
  echo "ERROR: weights-only trunk not found: $E1_TRUNK" >&2
  echo "  mint it: $PY scripts/make_ws3v3_warmstart.py \\" >&2
  echo "    --in checkpoints/run2_retro/checkpoint_00248000.pt \\" >&2
  echo "    --out $E1_TRUNK --encoding-name ''" >&2
  exit 2
fi

launch_arm() {
  local variant="$1"
  echo ""
  echo "=== E1 arm: $variant (iters=$E1_ITERS, trunk=$E1_TRUNK) ==="
  "$PY" scripts/train.py \
    --checkpoint "$E1_TRUNK" \
    --variant "$variant" \
    --checkpoint-dir "checkpoints/$variant" \
    --run-name "$variant" \
    --iterations "$E1_ITERS" \
    2>&1 | tee "logs/${variant}.log"
}

# SCALAR ARM FIRST (pre-registered), then DIST65.
launch_arm "$SCALAR_VARIANT"
launch_arm "$DIST_VARIANT"

echo ""
echo "=== E1 pair complete — scalar then dist65, iters=$E1_ITERS each ==="
