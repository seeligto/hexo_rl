#!/usr/bin/env bash
# §169 A3 — full eval matrix on a trained A3 PMA-global checkpoint.
#
# Runs in order:
#   1. PMA-collapse smoke (gates further eval; A3 mode also tests
#      collapse-onto-global)
#   2. argmax @ r=8, n=200 SealBot games
#   3. matched MCTS-N (default N=128), n=200 SealBot games via
#      KClusterMCTSBot with pool_type=pma_global plumbed through
#      model.pool_type (global crop computed per leaf from sim_board)
#   4. NN latency bench (b=1 + b=64, n=5 each)
#
# Threat probe (C1/C2/C3) requires a v6w25-specific fixture which the
# §169 scope did not deliver — surfaced as a §170 follow-up. The reports/
# block for threat is left as a status=skipped marker (mirrors A2).
#
# A3-only soft-warn: padding-leak check (compare arm with held-out canvas
# mask). Surfacing-only — operator runs the held-out variant manually if
# A3 underperforms A1; STOP only if A3 < A1.
#
# Outputs:
#   reports/ablation_169/A3_collapse.json
#   reports/ablation_169/A3_argmax.json
#   reports/ablation_169/A3_mcts<N>.json
#   reports/ablation_169/bench_per_arm.md (appended)
#
# Run from repo root. `MCTS_N` defaults to 128 (matches A2):
#   MCTS_N=64 bash scripts/eval_a3_pma_global.sh
set -euo pipefail

CKPT="checkpoints/ablation_169/A3_pma_global.pt"
REPORTS="reports/ablation_169"
MCTS_N="${MCTS_N:-128}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: A3 checkpoint missing: ${CKPT}" >&2
    echo "       Run scripts/pretrain_a3_pma_global.sh first." >&2
    exit 1
fi

mkdir -p "${REPORTS}"

# ── 1. PMA-collapse smoke (hard gate) ──────────────────────────────────
echo "[$(date -Iseconds)] §169 A3 — PMA collapse smoke"
if ! python scripts/probe_pma_collapse.py \
        --checkpoint "${CKPT}" \
        --output "${REPORTS}/A3_collapse.json" \
        > "${REPORTS}/A3_collapse.stdout"; then
    echo "STOP: PMA collapse — see ${REPORTS}/A3_collapse.json" >&2
    echo "      A2-style cluster collapse: retry with --pool-attn-dropout 0.2" >&2
    echo "      A3 collapse-onto-global   : recommend attention entropy reg" >&2
    exit 1
fi

# ── 2. argmax @ r=8, n=200 ─────────────────────────────────────────────
echo "[$(date -Iseconds)] §169 A3 — argmax @ r=8, n=200 vs SealBot"
python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference argmax \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/A3_argmax.json"

# ── 3. matched MCTS-N, n=200 ───────────────────────────────────────────
echo "[$(date -Iseconds)] §169 A3 — MCTS-${MCTS_N} @ r=8, n=200 vs SealBot"
python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference "mcts-${MCTS_N}" \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/A3_mcts${MCTS_N}.json"

# ── 4. NN latency bench (b=1, b=64) ────────────────────────────────────
echo "[$(date -Iseconds)] §169 A3 — NN latency bench"
python scripts/bench_v6w25_nn.py \
    --checkpoint "${CKPT}" \
    --batches 1,64 \
    --runs 5 \
    --arm-label A3 \
    --append-to "${REPORTS}/bench_per_arm.md"

# ── 5. Threat probe — SKIPPED (no v6w25 fixture; §170 follow-up) ──────
cat > "${REPORTS}/A3_threat.json" <<'JSON'
{
  "status": "skipped",
  "reason": "No v6w25 threat-probe fixture exists (same as A2). The shipped v6 fixture (fixtures/threat_probe_baseline.json) is 19x19 — shape-incompatible with v6w25's 25x25 cluster window. Building one requires curated tactical positions on a 25x25 board + a regenerated baseline. Tracked as a §170 follow-up.",
  "C1_contrast_mean": null,
  "C2_ext_in_top5_pct": null,
  "C3_ext_in_top10_pct": null
}
JSON

# ── 6. Combine argmax + MCTS into A3_eval.json ─────────────────────────
python - <<PY
import json
from pathlib import Path
out = Path("${REPORTS}")
combined = {
    "arm": "A3",
    "encoding": "v6w25",
    "pool_type": "pma_global",
    "argmax": json.loads((out / "A3_argmax.json").read_text()),
    "mcts": json.loads((out / "A3_mcts${MCTS_N}.json").read_text()),
}
collapse_path = out / "A3_collapse.json"
if collapse_path.exists():
    combined["collapse_smoke"] = json.loads(collapse_path.read_text())
(out / "A3_eval.json").write_text(json.dumps(combined, indent=2))
print(f"wrote {out / 'A3_eval.json'}")
PY

echo "[$(date -Iseconds)] §169 A3 — eval complete"
echo "Artefacts in ${REPORTS}/"
echo
echo "Soft-warn — padding-leak check (operator decides):"
echo "  Compare A3 SealBot WR to A1 anchor (14.5%). If A3 < A1, run a"
echo "  hold-out-mask variant (manually patch GlobalTokenEncoder to drop"
echo "  the canvas_mask plane) and re-eval — significant lift confirms"
echo "  padding leak. Out of §169 scope unless A3 < A1."
