#!/usr/bin/env bash
# Quick n=20 argmax checkup of "good" bootstrap models + n=200 spot-check
# on §172 B2 sustained step-30K. Probes whether SealBot @ r=8 vs argmax
# baselines have drifted since their last reported numbers.
set -euo pipefail

REPORTS="reports/bootstrap_checkup_$(date +%Y%m%d)"
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p "${REPORTS}"

run_eval() {
    local label="$1"; local ckpt="$2"; local n="$3"
    local out="${REPORTS}/${label}_argmax_n${n}.json"
    if [[ ! -f "${ckpt}" ]]; then
        echo "[skip] ${label}: ${ckpt} missing"
        return 0
    fi
    echo "[$(date -Iseconds)] === ${label} argmax n=${n} ==="
    "${PYTHON}" scripts/run_sealbot_eval.py \
        --checkpoint "${ckpt}" \
        --inference argmax \
        --n-games "${n}" \
        --legal-radius 8 \
        --output "${out}"
}

# n=20 sweep — fast checkup across known-good baselines
run_eval bootstrap_v6_default   checkpoints/bootstrap_model.pt              20
run_eval bootstrap_v7full       checkpoints/bootstrap_model_v7full.pt       20
run_eval bootstrap_v7e30        checkpoints/bootstrap_model_v7e30.pt        20
run_eval bootstrap_v6w25        checkpoints/bootstrap_model_v6w25.pt        20
run_eval A4_canvas_realness     checkpoints/ablation_169/A4_canvas_realness.pt 20

# n=200 real-info spot — §172 B2 sustained step-30K
run_eval b2_sustained_step_30k  checkpoints/sprint_172_p3_b2_sustained/checkpoint_00030000.pt 200

# Summary table
"${PYTHON}" - <<PY
import json
from pathlib import Path

out_dir = Path("${REPORTS}")
rows = []
for p in sorted(out_dir.glob("*_argmax_n*.json")):
    d = json.loads(p.read_text())
    rows.append({
        "label": p.stem,
        "n": d.get("n_games", 0),
        "wr": d.get("win_rate", 0.0),
        "lo": d.get("ci_95_low", 0.0),
        "hi": d.get("ci_95_high", 0.0),
        "wins": d.get("wins", 0),
        "losses": d.get("losses", 0),
        "draws": d.get("draws", 0),
        "mean_ply": d.get("mean_ply", 0.0),
        "encoding": d.get("encoding", "?"),
    })

print()
print(f"{'label':<35} {'enc':<8} {'n':>4}  {'W/L/D':>10}  {'WR':>7}  {'Wilson95':<18}  {'ply':>5}")
print("-" * 100)
for r in rows:
    wld = f"{r['wins']}/{r['losses']}/{r['draws']}"
    print(f"{r['label']:<35} {r['encoding']:<8} {r['n']:>4}  {wld:>10}  {r['wr']*100:>6.2f}%  [{r['lo']*100:5.2f}%, {r['hi']*100:5.2f}%]  {r['mean_ply']:>5.1f}")

(out_dir / "summary.json").write_text(json.dumps(rows, indent=2))
print(f"\nWrote {out_dir / 'summary.json'}")
PY
