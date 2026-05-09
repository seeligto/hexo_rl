#!/usr/bin/env bash
# §170 P4 P1 — full eval matrix on A1 + gpool-bias-policy-only checkpoint.
#
# Architecture: canonical A1 v6w25 (K-cluster + min/max pool) + additive
# K-invariant global-pool bias — POLICY HEAD ONLY (value_bias hardcoded
# zero, value_proj never invoked at forward time). Gate scalar drives
# policy bias only; gate=0 byte-exact A1 at construction.
#
# Pre-registered verdict criteria (LOCKED — see §170 P4 P1 sprint
# header):
#   WIN:         argmax > 16% AND MCTS-64 > 27% (Wilson 95% LB > 12% / 24%).
#   PARTIAL-WIN: argmax > 16% AND MCTS-64 in [22%, 27%].
#   NULL:        argmax in [12%, 16%] AND MCTS-64 in [22%, 32%].
#   LOSS:        any axis disjoint-below A1 anchor CI (MCTS-64 < 24% UB).
#
# Runs in order:
#   1. argmax @ r=8, n=200 SealBot.
#   2. MCTS-32, n=200 SealBot.
#   3. MCTS-64, n=200 SealBot (matched A1 anchor baseline depth).
#   4. MCTS-128, n=200 SealBot (rules out search-depth-specific effects
#      given §170 P1 A3 FLAT-NON-MONOTONIC at MCTS-{32,64,128}).
#   5. NN latency bench (b=1, b=64, n=5 each) appended to bench.md.
#   6. Threat probe SKIPPED — no v6w25 fixture (§170 follow-up).
#   7. Combine all 4 evals + bench + threat into eval.json.
#
# Outputs (all under reports/gpool_bias/policy_only/):
#   argmax.json, mcts32.json, mcts64.json, mcts128.json
#   threat.json (status=skipped)
#   eval.json (combined)
# Plus appended row to reports/gpool_bias/bench.md.
#
# Run from repo root on the 5080 vast.ai host.
set -euo pipefail

CKPT="checkpoints/gpool_bias/A1_gpool_bias_policy_only.pt"
REPORTS="reports/gpool_bias/policy_only"
BENCH_MD="reports/gpool_bias/bench.md"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: policy-only checkpoint missing: ${CKPT}" >&2
    echo "       Run scripts/pretrain_gpool_bias_policy_only.sh first." >&2
    exit 1
fi

mkdir -p "${REPORTS}"

# ── 1. argmax @ r=8, n=200 ─────────────────────────────────────────────
echo "[$(date -Iseconds)] §170 P4 P1 — argmax @ r=8, n=200 vs SealBot"
.venv/bin/python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference argmax \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/argmax.json"

# ── 2. MCTS-32, n=200 ──────────────────────────────────────────────────
echo "[$(date -Iseconds)] §170 P4 P1 — MCTS-32 @ r=8, n=200 vs SealBot"
.venv/bin/python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference mcts-32 \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/mcts32.json"

# ── 3. MCTS-64, n=200 ──────────────────────────────────────────────────
echo "[$(date -Iseconds)] §170 P4 P1 — MCTS-64 @ r=8, n=200 vs SealBot"
.venv/bin/python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference mcts-64 \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/mcts64.json"

# ── 4. MCTS-128, n=200 ─────────────────────────────────────────────────
echo "[$(date -Iseconds)] §170 P4 P1 — MCTS-128 @ r=8, n=200 vs SealBot"
.venv/bin/python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference mcts-128 \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/mcts128.json"

# ── 5. NN latency bench (b=1, b=64) ────────────────────────────────────
echo "[$(date -Iseconds)] §170 P4 P1 — NN latency bench"
.venv/bin/python scripts/bench_v6w25_nn.py \
    --checkpoint "${CKPT}" \
    --batches 1,64 \
    --runs 5 \
    --arm-label "A1_gpool_bias_policy_only" \
    --append-to "${BENCH_MD}"

# ── 6. Threat probe — SKIPPED (no v6w25 fixture, §170 follow-up) ───────
cat > "${REPORTS}/threat.json" <<'JSON'
{
  "status": "skipped",
  "reason": "No v6w25 threat-probe fixture exists (same as A2/A3/A1+gpool-bias §170 P3). The shipped v6 fixture (fixtures/threat_probe_baseline.json) is 19x19 — shape-incompatible with v6w25's 25x25 cluster window. Building one requires curated tactical positions on a 25x25 board + a regenerated baseline. Tracked as a §170 follow-up.",
  "C1_contrast_mean": null,
  "C2_ext_in_top5_pct": null,
  "C3_ext_in_top10_pct": null
}
JSON

# ── 7. Combine into eval.json ──────────────────────────────────────────
.venv/bin/python - <<'PY'
import json
from pathlib import Path
out = Path("reports/gpool_bias/policy_only")
combined = {
    "arm": "A1_gpool_bias_policy_only",
    "sprint": "§170 P4 P1",
    "encoding": "v6w25",
    "pool_type": "min_max",
    "gpool_bias_active": True,
    "policy_only_bias": True,
    "argmax": json.loads((out / "argmax.json").read_text()),
    "mcts32": json.loads((out / "mcts32.json").read_text()),
    "mcts64": json.loads((out / "mcts64.json").read_text()),
    "mcts128": json.loads((out / "mcts128.json").read_text()),
    "threat": json.loads((out / "threat.json").read_text()),
}
(out / "eval.json").write_text(json.dumps(combined, indent=2))
print(f"wrote {out / 'eval.json'}")
PY

echo "[$(date -Iseconds)] §170 P4 P1 — eval complete"
echo "Artefacts in ${REPORTS}/"
echo
echo "Pre-registered verdict gates (operator decides):"
echo "  WIN:         argmax > 16% AND MCTS-64 > 27%."
echo "  PARTIAL-WIN: argmax > 16% AND MCTS-64 in [22%, 27%]."
echo "  NULL:        argmax in [12%, 16%] AND MCTS-64 in [22%, 32%]."
echo "  LOSS:        any axis disjoint-below A1 anchor CI (MCTS-64 < 24% UB)."
