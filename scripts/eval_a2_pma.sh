#!/usr/bin/env bash
# §169 A2 — full eval matrix on a trained A2 PMA checkpoint.
#
# Runs in order:
#   1. PMA-collapse smoke   (hard-stop gate before any further eval)
#   2. argmax @ r=8, n=200 SealBot games
#   3. matched MCTS-N (default N=128), n=200 SealBot games via KClusterMCTSBot
#      with pool_type=pma plumbed through model.pool_type
#   4. NN latency bench (b=1 + b=64, n=5 each)
#
# Threat probe (C1/C2/C3) requires a v6w25-specific fixture which the §169
# scope did not deliver — surfaced as a §170 follow-up. The reports/ block
# for threat is left empty under that gap.
#
# Outputs:
#   reports/ablation_169/A2_collapse.json
#   reports/ablation_169/A2_argmax.json
#   reports/ablation_169/A2_mcts<N>.json
#   reports/ablation_169/bench_per_arm.md (appended)
#
# Run from repo root. `MCTS_N` defaults to 128 (matches §169 P1 default
# operator pick); override on the command line:
#   MCTS_N=64 bash scripts/eval_a2_pma.sh
set -euo pipefail

CKPT="checkpoints/ablation_169/A2_pma.pt"
REPORTS="reports/ablation_169"
MCTS_N="${MCTS_N:-128}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: A2 checkpoint missing: ${CKPT}" >&2
    echo "       Run scripts/pretrain_a2_pma.sh first." >&2
    exit 1
fi

mkdir -p "${REPORTS}"

# ── 1. PMA-collapse smoke (hard gate) ──────────────────────────────────
echo "[$(date -Iseconds)] §169 A2 — PMA collapse smoke"
if ! python scripts/probe_pma_collapse.py \
        --checkpoint "${CKPT}" \
        --output "${REPORTS}/A2_collapse.json" \
        > "${REPORTS}/A2_collapse.stdout"; then
    echo "STOP: PMA collapse — see ${REPORTS}/A2_collapse.json" >&2
    echo "      Retry pretrain with --pool-attn-dropout 0.2"     >&2
    exit 1
fi

# ── 2. argmax @ r=8, n=200 ─────────────────────────────────────────────
echo "[$(date -Iseconds)] §169 A2 — argmax @ r=8, n=200 vs SealBot"
python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference argmax \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/A2_argmax.json"

# ── 3. matched MCTS-N, n=200 ───────────────────────────────────────────
echo "[$(date -Iseconds)] §169 A2 — MCTS-${MCTS_N} @ r=8, n=200 vs SealBot"
python scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference "mcts-${MCTS_N}" \
    --n-games 200 \
    --legal-radius 8 \
    --output "${REPORTS}/A2_mcts${MCTS_N}.json"

# ── 4. NN latency bench (b=1, b=64) ────────────────────────────────────
echo "[$(date -Iseconds)] §169 A2 — NN latency bench"
python scripts/bench_v6w25_nn.py \
    --checkpoint "${CKPT}" \
    --batches 1,64 \
    --runs 5 \
    --arm-label A2 \
    --append-to "${REPORTS}/bench_per_arm.md"

# ── 5. Threat probe — SKIPPED (no v6w25 fixture; §170 follow-up) ──────
# Surface the gap as a JSON artifact so the §169 sprint-log entry can
# point at it. C1/C2/C3 left null; the v6 fixture is shape-incompatible
# with v6w25 (board_size 19 vs 25).
cat > "${REPORTS}/A2_threat.json" <<'JSON'
{
  "status": "skipped",
  "reason": "No v6w25 threat-probe fixture exists. The shipped v6 fixture (fixtures/threat_probe_baseline.json) is 19x19 — shape-incompatible with v6w25's 25x25 cluster window. Building one requires curated tactical positions on a 25x25 board + a regenerated baseline. Tracked as a §170 follow-up.",
  "C1_contrast_mean": null,
  "C2_ext_in_top5_pct": null,
  "C3_ext_in_top10_pct": null
}
JSON

# ── 6. Combine argmax + MCTS into A2_eval.json ─────────────────────────
python - <<PY
import json
from pathlib import Path
out = Path("${REPORTS}")
combined = {
    "arm": "A2",
    "encoding": "v6w25",
    "pool_type": "pma",
    "argmax": json.loads((out / "A2_argmax.json").read_text()),
    "mcts": json.loads((out / "A2_mcts${MCTS_N}.json").read_text()),
}
(out / "A2_eval.json").write_text(json.dumps(combined, indent=2))
print(f"wrote {out / 'A2_eval.json'}")
PY

echo "[$(date -Iseconds)] §169 A2 — eval complete"
echo "Artefacts in ${REPORTS}/"
