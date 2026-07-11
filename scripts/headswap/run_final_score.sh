#!/bin/bash
# D-F HEADSWAP — final scoring + verdict. All 4 arms x 3 seeds on the FULL 234
# positives (WP2 boards via --wp2-games) + combined negatives (v1 retro + v2 WP2).
# score_all is RAM-bounded (~1.2GB, inference_mode). Launch:
#   setsid bash scripts/headswap/run_final_score.sh </dev/null >/dev/null 2>&1 &
set -uo pipefail
cd /workspace/hexo_rl
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
TRUNK=/workspace/headswap_data/checkpoint_00248000.pt
POS=reports/valprobe/probe_set_v1.jsonl
GAMES=reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl
OUT=reports/headswap/final
mkdir -p "$OUT"

# Combine negatives (retro v1 + WP2 v2), dedup by zobrist.
$PY -c "
import json
seen=set(); out=[]
for f in ['reports/valprobe/negatives_v1.jsonl','reports/valprobe/negatives_v2_wp2.jsonl']:
    try:
        for l in open(f):
            r=json.loads(l); z=str(r['zobrist'])
            if z in seen: continue
            seen.add(z); out.append(r)
    except FileNotFoundError: pass
with open('reports/valprobe/negatives_combined.jsonl','w') as fo:
    fo.write('\n'.join(json.dumps(r) for r in out)+'\n')
print('combined negatives:', len(out))
"
NEG=reports/valprobe/negatives_combined.jsonl

# WP2 games args (book_id=path).
WP2ARGS=""
for b in 0 1 2 3 4; do
  WP2ARGS="$WP2ARGS evalfair_r5_wp2_b${b}=reports/headswap/wp2_regen/evalfair_r5_wp2_b${b}/games.jsonl"
done

score_arm() {  # $1=ARM $2=SEED $3=dir(ab/cd)
  $PY -m scripts.headswap.score_all --head reports/headswap/$3/arm_$1_seed$2/head_$1_seed$2.pt \
    --trunk "$TRUNK" --positives "$POS" --negatives "$NEG" --games "$GAMES" --wp2-games $WP2ARGS \
    --out "$OUT/scores_$1_seed$2.jsonl" > "$OUT/score_$1_$2.log" 2>&1 || echo "SCORE FAIL $1 seed$2"
}

for S in 0 1 2; do score_arm A "$S" ab; score_arm B "$S" ab; done
for S in 0 1 2; do score_arm C "$S" cd; score_arm D "$S" cd; done

$PY -m scripts.headswap.metrics \
  --a "$OUT"/scores_A_seed0.jsonl "$OUT"/scores_A_seed1.jsonl "$OUT"/scores_A_seed2.jsonl \
  --b "$OUT"/scores_B_seed0.jsonl "$OUT"/scores_B_seed1.jsonl "$OUT"/scores_B_seed2.jsonl \
  --c "$OUT"/scores_C_seed0.jsonl "$OUT"/scores_C_seed1.jsonl "$OUT"/scores_C_seed2.jsonl \
  --d "$OUT"/scores_D_seed0.jsonl "$OUT"/scores_D_seed1.jsonl "$OUT"/scores_D_seed2.jsonl \
  --out reports/headswap/VERDICT.md --json-out reports/headswap/metrics_final.json > "$OUT/metrics.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then echo "DONE $(date -u)" > "$OUT/FINAL_DONE"; else echo "FAILED rc=$rc" > "$OUT/FINAL_FAILED"; fi
