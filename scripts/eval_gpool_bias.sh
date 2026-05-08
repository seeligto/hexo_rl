#!/usr/bin/env bash
# §170 P3 — full eval matrix on a trained A1 + gpool-bias checkpoint.
#
# Architecture: canonical A1 v6w25 (K-cluster + min/max pool) +
# additive K-invariant global-pool bias to value_fc1 hidden + policy_fc
# logits. Gate=0 init → byte-exact A1 at construction; only as the gradient
# grows the gate does the global summary earn weight. Commit-1 unit test
# test_gate_zero_byte_exact_a1 enforces this invariant on
# bootstrap_model_v6w25.pt.
#
# Runs in order:
#   1. argmax @ r=8, n=200 SealBot games via V6ArgmaxBot (auto-detects
#      gpool_bias_active and threads global_crop)
#   2. matched MCTS-N (default N=64), n=200 SealBot games via
#      KClusterMCTSBot (pool_type=min_max + global_crop computed per leaf
#      from sim_board)
#   3. NN latency bench (b=1 + b=64, n=5 each)
#   4. Threat probe SKIPPED — no v6w25 fixture; §170 follow-up
#   5. Combine argmax + MCTS into reports/gpool_bias/eval.json
#
# Outputs:
#   reports/gpool_bias/argmax.json
#   reports/gpool_bias/mcts<N>.json
#   reports/gpool_bias/threat.json (status=skipped)
#   reports/gpool_bias/bench.md (appended)
#   reports/gpool_bias/eval.json (combined)
#
# Run from repo root. MCTS_N defaults to 64 (per §170 P3 spec):
#   MCTS_N=128 bash scripts/eval_gpool_bias.sh
set -euo pipefail

CKPT="checkpoints/gpool_bias/A1_gpool_bias.pt"
REPORTS="reports/gpool_bias"
MCTS_N="${MCTS_N:-64}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: gpool-bias checkpoint missing: ${CKPT}" >&2
    echo "       Run scripts/pretrain_gpool_bias.sh first." >&2
    exit 1
fi

mkdir -p "${REPORTS}"

# ── 1. argmax @ r=8, n=200 ─────────────────────────────────────────────
echo "[$(date -Iseconds)] §170 P3 — argmax @ r=8, n=200 vs SealBot"
.venv/bin/python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference argmax \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/argmax.json"

# ── 2. matched MCTS-N, n=200 ───────────────────────────────────────────
echo "[$(date -Iseconds)] §170 P3 — MCTS-${MCTS_N} @ r=8, n=200 vs SealBot"
.venv/bin/python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference "mcts-${MCTS_N}" \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/mcts${MCTS_N}.json"

# ── 3. NN latency bench (b=1, b=64) ────────────────────────────────────
echo "[$(date -Iseconds)] §170 P3 — NN latency bench"
.venv/bin/python scripts/bench_v6w25_nn.py \
    --checkpoint "${CKPT}" \
    --batches 1,64 \
    --runs 5 \
    --arm-label "A1_gpool_bias" \
    --append-to "${REPORTS}/bench.md"

# ── 4. Threat probe — SKIPPED ──────────────────────────────────────────
cat > "${REPORTS}/threat.json" <<'JSON'
{
  "status": "skipped",
  "reason": "No v6w25 threat-probe fixture exists (same as A2/A3). The shipped v6 fixture (fixtures/threat_probe_baseline.json) is 19x19 — shape-incompatible with v6w25's 25x25 cluster window. Building one requires curated tactical positions on a 25x25 board + a regenerated baseline. Tracked as a §170 follow-up.",
  "C1_contrast_mean": null,
  "C2_ext_in_top5_pct": null,
  "C3_ext_in_top10_pct": null
}
JSON

# ── 5. Combine argmax + MCTS into eval.json ────────────────────────────
.venv/bin/python - <<PY
import json
from pathlib import Path
out = Path("${REPORTS}")
combined = {
    "arm": "A1_gpool_bias",
    "sprint": "§170 P3",
    "encoding": "v6w25",
    "pool_type": "min_max",
    "gpool_bias_active": True,
    "mcts_n": ${MCTS_N},
    "argmax": json.loads((out / "argmax.json").read_text()),
    "mcts": json.loads((out / "mcts${MCTS_N}.json").read_text()),
    "threat": json.loads((out / "threat.json").read_text()),
}
(out / "eval.json").write_text(json.dumps(combined, indent=2))
print(f"wrote {out / 'eval.json'}")
PY

echo "[$(date -Iseconds)] §170 P3 — eval complete"
echo "Artefacts in ${REPORTS}/"
echo
echo "Surface-immediately checks (operator decides):"
echo "  - Gate scalar trajectory: read pool_global_gate_value from pretrain.log."
echo "    If final gate < 0.05, branch never earned weight ⇒ null result."
echo "    If gate ≫ 0.05 + argmax > 20%, surface as §171 sustained-run candidate."
echo "  - argmax WR < 12%: gpool-bias didn't help; surface."
echo "  - argmax WR > 20%: BREAKTHROUGH — recommend §171 sustained-run scoping."
