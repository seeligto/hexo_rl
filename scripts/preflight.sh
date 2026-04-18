#!/usr/bin/env bash
# scripts/preflight.sh — mandatory pre-flight for any new training run
# Run from repo root. All 6 checks must pass before make train.
set -euo pipefail

PASS=0; FAIL=0
ok()   { echo "[PASS] $*"; ((PASS++)) || true; }
fail() { echo "[FAIL] $*" >&2; ((FAIL++)) || true; }

echo "=========================================================="
echo "PRE-FLIGHT (mandatory, abort if any FAIL)"
echo "=========================================================="

# ── 1. Git position ────────────────────────────────────────────────────────────
echo
echo "[1/6] Git position..."
git log --oneline -3
DESC="$(git describe --tags HEAD 2>/dev/null || echo 'no-tag')"
echo "describe: $DESC"
if [[ "$DESC" == v0.4.0* ]]; then ok "on v0.4.0"; else fail "expected v0.4.0, got $DESC"; fi

# ── 2. bootstrap_model.pt is 24-plane ─────────────────────────────────────────
echo
echo "[2/6] bootstrap_model.pt input planes..."
.venv/bin/python - <<'PYEOF'
import torch, sys
path = "checkpoints/bootstrap_model.pt"
try:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    ic = sd["trunk.input_conv.weight"].shape[1]
    print(f"  input_channels = {ic}")
    assert ic == 24, f"Expected 24, got {ic}"
    print("  PASS")
except FileNotFoundError:
    print(f"  FAIL: {path} not found", file=sys.stderr); sys.exit(1)
PYEOF
ok "bootstrap_model.pt 24-plane"

# ── 3. No stale HEXB v2 buffers ────────────────────────────────────────────────
echo
echo "[3/6] Stale HEXB buffer scan..."
STALE=0
while IFS= read -r -d '' f; do
    VER=$(.venv/bin/python - "$f" <<'PYEOF'
import struct, sys
with open(sys.argv[1], "rb") as fh:
    fh.read(4)
    print(struct.unpack("<I", fh.read(4))[0])
PYEOF
)
    echo "  $f: HEXB v$VER"
    if [[ "$VER" -lt 4 ]]; then echo "  *** STALE — move to archive ***"; STALE=1; fi
done < <(find checkpoints/ data/ -name "*.hexb" -print0 2>/dev/null)
if [[ "$STALE" -eq 0 ]]; then ok "no stale buffers"; else fail "stale HEXB v2 buffer found"; fi

# ── 4. Config values ───────────────────────────────────────────────────────────
echo
echo "[4/6] Config values..."
.venv/bin/python - <<'PYEOF'
import yaml, sys
with open("configs/training.yaml") as f: c = yaml.safe_load(f)
with open("configs/model.yaml")    as f: m = yaml.safe_load(f)
print(f"  aux_chain_weight:  {c.get('aux_chain_weight', 'MISSING')}")
print(f"  threat_pos_weight: {c.get('threat_pos_weight', 'MISSING')}")
print(f"  in_channels:       {m.get('in_channels', 'MISSING')}")
assert m.get("in_channels") == 24, f"in_channels must be 24, got {m.get('in_channels')}"
print("  PASS")
PYEOF
ok "configs correct"

# ── 5. Effective merged config (gumbel_targets variant) ───────────────────────
echo
echo "[5/6] Merged config check (VARIANT=gumbel_targets)..."
.venv/bin/python - <<'PYEOF'
import yaml, sys

def deep_merge(base, override):
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

cfg = {}
for f in ["configs/training.yaml", "configs/selfplay.yaml",
          "configs/model.yaml", "configs/monitoring.yaml"]:
    with open(f) as fh:
        cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
with open("configs/variants/gumbel_targets.yaml") as fh:
    cfg = deep_merge(cfg, yaml.safe_load(fh) or {})

sp = cfg.get("selfplay", {})
pc = sp.get("playout_cap", {})
# NOTE: dirichlet_enabled lives under mcts: not selfplay: — see configs/selfplay.yaml
mc = cfg.get("mcts", {})

vals = {
    "selfplay.gumbel_mcts":        sp.get("gumbel_mcts"),
    "selfplay.completed_q_values": sp.get("completed_q_values"),
    "selfplay.playout_cap.fast_prob": pc.get("fast_prob"),
    "mcts.dirichlet_enabled":      mc.get("dirichlet_enabled"),  # NOT selfplay.dirichlet_enabled
}
for k, v in vals.items():
    print(f"  {k}: {v}")

assert vals["selfplay.gumbel_mcts"]        == False, "gumbel_mcts must be false"
assert vals["selfplay.completed_q_values"] == True,  "completed_q_values must be true"
assert vals["selfplay.playout_cap.fast_prob"] == 0.0,"fast_prob must be 0.0"
assert vals["mcts.dirichlet_enabled"]      == True,  "dirichlet_enabled must be true"
print("  PASS")
PYEOF
ok "merged config PASS"

# ── 6. GPU temperature ─────────────────────────────────────────────────────────
echo
echo "[6/6] GPU temperature..."
TEMP="$(nvidia-smi -q -d TEMPERATURE 2>/dev/null | grep "GPU Current Temp" | awk '{print $5}')"
echo "  GPU Current Temp: ${TEMP:-unavailable} C"
if [[ -z "$TEMP" ]]; then
    fail "nvidia-smi unavailable — check GPU"
elif [[ "$TEMP" -le 50 ]]; then
    ok "GPU ${TEMP}°C ≤ 50°C"
else
    fail "GPU ${TEMP}°C > 50°C — wait to cool"
fi

# ── Summary ────────────────────────────────────────────────────────────────────
echo
echo "=========================================================="
echo "  PASS: $PASS / $((PASS + FAIL))    FAIL: $FAIL"
echo "=========================================================="
if [[ "$FAIL" -gt 0 ]]; then
    echo "  FIX FAILURES BEFORE TRAINING"
    exit 1
else
    echo "  All checks passed — ready for: make train VARIANT=gumbel_targets"
    exit 0
fi
