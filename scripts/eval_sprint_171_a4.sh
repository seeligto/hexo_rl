#!/usr/bin/env bash
# §171 A4 P2-reopen C — eval the distribution-shift fine-tune checkpoint.
#
# Pre-registered MCTS-64 verdict bins vs SealBot @ r=8, n=200, plies=4:
#   ALIVE:    WR > 8% (Wilson lower > 5%)
#   MARGINAL: WR in [2%, 8%]
#   DEAD:     WR ≤ 2% (Wilson upper < 4%)
#
# Also runs argmax @ r=8 n=200 as a sharpness reference (compares against
# §169 P4 A4 22.0% argmax baseline pre-fine-tune).
#
# Run from repo root. Env overrides:
#   CKPT  — checkpoint path (default: sprint_171_a4 fine-tune output)
#   N_GAMES — sample size (default: 200)
#   MCTS_N — MCTS sims per move (default: 64; ALIVE/MARGINAL/DEAD bins assume 64)
set -euo pipefail

CKPT="${CKPT:-checkpoints/sprint_171_a4/A4_finetune_p2reopen.pt}"
N_GAMES="${N_GAMES:-200}"
MCTS_N="${MCTS_N:-64}"
REPORTS="reports/sprint_171_a4"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: fine-tune checkpoint missing: ${CKPT}" >&2
    exit 1
fi

mkdir -p "${REPORTS}"

echo "[$(date -Iseconds)] §171 A4 P2-reopen — argmax @ r=8, n=${N_GAMES} vs SealBot"
"${PYTHON}" scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference argmax \
    --n-games "${N_GAMES}" \
    --legal-radius 8 \
    --output "${REPORTS}/A4_finetune_argmax.json"

echo "[$(date -Iseconds)] §171 A4 P2-reopen — MCTS-${MCTS_N} @ r=8, n=${N_GAMES} vs SealBot"
"${PYTHON}" scripts/run_sealbot_eval.py \
    --checkpoint "${CKPT}" \
    --inference "mcts-${MCTS_N}" \
    --n-games "${N_GAMES}" \
    --legal-radius 8 \
    --output "${REPORTS}/A4_finetune_mcts${MCTS_N}.json"

# Combine + apply pre-registered verdict bins.
"${PYTHON}" - <<PY
import json
import math
from pathlib import Path

out = Path("${REPORTS}")
argmax = json.loads((out / "A4_finetune_argmax.json").read_text())
mcts = json.loads((out / "A4_finetune_mcts${MCTS_N}.json").read_text())

def wr(d):
    return d.get("win_rate", d.get("wr"))

def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))

mw = wr(mcts)
mn = mcts.get("n_games", $N_GAMES)
mlo, mhi = wilson(mw, mn)

# Pre-registered bins (Wilson 95% CI):
#   ALIVE: wr > 0.08 AND lower > 0.05
#   DEAD:  wr <= 0.02 AND upper < 0.04
#   MARGINAL: otherwise
if mw > 0.08 and mlo > 0.05:
    verdict = "ALIVE"
elif mw <= 0.02 and mhi < 0.04:
    verdict = "DEAD"
else:
    verdict = "MARGINAL"

combined = {
    "sprint": "171_A4_P2_reopen_C",
    "checkpoint": "${CKPT}",
    "argmax": argmax,
    "mcts_${MCTS_N}": mcts,
    "verdict": {
        "bin": verdict,
        "mcts_wr": mw,
        "wilson_lower_95": mlo,
        "wilson_upper_95": mhi,
        "n_games": mn,
        "thresholds": {
            "ALIVE":    "wr > 0.08 AND lower > 0.05",
            "MARGINAL": "wr in [0.02, 0.08]",
            "DEAD":     "wr <= 0.02 AND upper < 0.04",
        },
    },
}
(out / "A4_finetune_eval.json").write_text(json.dumps(combined, indent=2))
print(f"VERDICT: {verdict}  MCTS-${MCTS_N} WR={mw:.4f}  Wilson95=[{mlo:.4f}, {mhi:.4f}]  n={mn}")
print(f"argmax WR = {wr(argmax):.4f}")
PY

echo "[$(date -Iseconds)] §171 A4 P2-reopen — eval complete; see ${REPORTS}/A4_finetune_eval.json"
