#!/usr/bin/env bash
# §169 A4 — full eval matrix on a trained A4 canvas_realness checkpoint.
#
# Runs in order:
#   1. argmax @ r=8, n=200 SealBot games (V8ArgmaxBot path; checkpoint
#      loader auto-detects canvas_realness from state-dict key signature
#      and reconstructs the model with PartialConv2d at trunk entry).
#   2. matched MCTS-N (default N=128), n=200 SealBot games via V8MCTSBot.
#   3. NN latency bench (b=1 + b=64, n=5 each) via bench_v8_nn.py.
#   4. Threat probe — SKIPPED (v8 fixture regen is a §170 follow-up; same
#      gap as A2/A3). Surfaces the gap as A4_threat.json status=skipped.
#
# Outputs:
#   reports/ablation_169/A4_argmax.json
#   reports/ablation_169/A4_mcts<N>.json
#   reports/ablation_169/A4_threat.json (skipped marker)
#   reports/ablation_169/bench_per_arm.md (appended)
#   reports/ablation_169/A4_eval.json (combined)
#
# Run from repo root. `MCTS_N` defaults to 128 (matches A2/A3):
#   MCTS_N=64 bash scripts/eval_a4_canvas_realness.sh
set -euo pipefail

CKPT="checkpoints/ablation_169/A4_canvas_realness.pt"
REPORTS="reports/ablation_169"
MCTS_N="${MCTS_N:-128}"
ARM_LABEL="A4"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: A4 checkpoint missing: ${CKPT}" >&2
    echo "       Run scripts/pretrain_a4_canvas_realness.sh first." >&2
    exit 1
fi

mkdir -p "${REPORTS}"

# ── 1. argmax @ r=8, n=200 ─────────────────────────────────────────────
echo "[$(date -Iseconds)] §169 A4 — argmax @ r=8, n=200 vs SealBot"
python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference argmax \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/A4_argmax.json"

# ── 2. matched MCTS-N, n=200 ───────────────────────────────────────────
echo "[$(date -Iseconds)] §169 A4 — MCTS-${MCTS_N} @ r=8, n=200 vs SealBot"
python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference "mcts-${MCTS_N}" \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/A4_mcts${MCTS_N}.json"

# ── 3. NN latency bench (b=1, b=64) ────────────────────────────────────
echo "[$(date -Iseconds)] §169 A4 — NN latency bench"
python scripts/bench_v8_nn.py \
    --checkpoint "${CKPT}" \
    --runs 5 --warmup 20 --batches 1,64 \
    --out "${REPORTS}/${ARM_LABEL}_bench.json"

# ── 4. Threat probe — SKIPPED (no v8 fixture; §170 follow-up) ──────────
cat > "${REPORTS}/A4_threat.json" <<'JSON'
{
  "status": "skipped",
  "reason": "No v8 threat-probe fixture exists. The shipped v6 fixture (fixtures/threat_probe_baseline.json) is 19x19 — shape-incompatible with v8's 25x25 bbox window. Building a v8 fixture is the same §170 follow-up tracked under A2/A3.",
  "C1_contrast_mean": null,
  "C2_ext_in_top5_pct": null,
  "C3_ext_in_top10_pct": null
}
JSON

# ── 5. Combine argmax + MCTS into A4_eval.json ─────────────────────────
python - <<PY
import json
from pathlib import Path
out = Path("${REPORTS}")
combined = {
    "arm": "${ARM_LABEL}",
    "encoding": "v8",
    "canvas_realness": True,
    "argmax": json.loads((out / "A4_argmax.json").read_text()),
    "mcts": json.loads((out / "A4_mcts${MCTS_N}.json").read_text()),
}
bench_path = out / "${ARM_LABEL}_bench.json"
if bench_path.exists():
    combined["bench"] = json.loads(bench_path.read_text())
(out / "A4_eval.json").write_text(json.dumps(combined, indent=2))
print(f"wrote {out / 'A4_eval.json'}")
PY

echo "[$(date -Iseconds)] §169 A4 — eval complete"
echo "Artefacts in ${REPORTS}/"
echo
echo "Surface check — argmax > 12% triggers a 'bbox direction lives'"
echo "verdict: matched MCTS-${MCTS_N} becomes critical for the §169 P4"
echo "verdict line. Sprint log P4 entry interprets the result."
