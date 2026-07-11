# D-C VALPROBE WP4 — Value-Head Health Trend

**Generated:** 2026-07-10  
**Branch:** phase4.5/valprobe_dc  
**Script:** `scripts/valprobe/value_health.py`  
**Series:** `reports/valprobe/value_health_series.jsonl`

---

## Method

Forward-only (no solver). Laptop RTX 4060, dev=cuda.

**Checkpoints:** 15 banked run2 ckpts loaded via `load_model_with_encoding(..., declared_encoding="v6_live2_ls")`.  
272k skipped — stamped `v6_live2` (not `_ls`), gated loader rejects.

**Games:** Per-ckpt `reports/evalfair/retro_slope/<ckpt>/games.jsonl` — the ckpt's own self-play eval games at the curriculum-correct radius. Draws + censored games excluded. 125–128 decided games per ckpt.

**Positions sampled:** Head-turn-starts only (§4.1 rationale: value head is min-pooled to-move perspective + plane-2 `moves_remaining` confound). ~1800–2400 positions per ckpt.

**Value forward:** `LocalInferenceEngine.infer_batch` — real K-cluster multi-window min-pooled forward (per memory d-percept, NOT a to_flat cripple).

**Outcome:** head-perspective ∈ {+1 win, −1 loss}.

**Metrics:**
- **ECE** — Expected Calibration Error, 10 equal-width bins on P_win = (v+1)/2. Lower = better calibrated.
- **Decided accuracy** — fraction where sign(v_t) == sign(outcome), restricted to |v_t| > 0.05. Excludes near-zero "don't know" positions.
- **Value MAE** — mean|v_t − outcome|. Lower = less bias/spread.
- **Phase split** — early/late halves of each game (ply < midpoint / ply ≥ midpoint).

---

## Series table

| step   | r | ECE    | dec_acc | MAE    | n_pos | wins | losses |
|--------|---|--------|---------|--------|-------|------|--------|
| 50k    | 4 | 0.4567 | 0.4183  | 1.1261 | 2115  | 67   | 60     |
| 70k    | 4 | 0.4891 | 0.4053  | 1.1565 | 2388  | 64   | 64     |
| 90k    | 4 | 0.5972 | 0.3022  | 1.3383 | 2118  | 45   | 82     |
| 110k   | 4 | 0.5226 | 0.3676  | 1.2296 | 2275  | 57   | 68     |
| 130k   | 4 | 0.4017 | 0.4736  | 1.0537 | 2114  | 64   | 63     |
| 150k   | 4 | 0.3530 | 0.5552  | 0.9157 | 2185  | 81   | 46     |
| 170k   | 4 | 0.4218 | 0.4665  | 1.0645 | 2263  | 71   | 57     |
| 175k   | 4 | 0.3611 | 0.5361  | 0.9378 | 2281  | 76   | 52     |
| 195k   | 4 | 0.4724 | 0.4324  | 1.1213 | 2401  | 66   | 62     |
| **200k** | **5** | **0.5406** | **0.3613** | **1.2430** | 2152 | 60 | 67 |
| 210k   | 5 | 0.3039 | 0.5775  | 0.8525 | 1828  | 82   | 45     |
| 220k   | 5 | 0.3714 | 0.5404  | 0.9356 | 2073  | 81   | 45     |
| 230k   | 5 | 0.3847 | 0.4925  | 1.0132 | 2170  | 73   | 55     |
| 240k   | 5 | 0.3639 | 0.5291  | 0.9581 | 2175  | 84   | 44     |
| 248k   | 5 | 0.4723 | 0.4225  | 1.1427 | 2129  | 70   | 57     |

Bold = r4→r5 transition ckpt (first r5 eval book).

---

## What the trend shows

**Overall direction: improving, then plateauing.**

- **50k–90k:** poor calibration (ECE 0.46→0.60) and low decided accuracy (0.42→0.30). The 90k dip is the worst point — corresponds to high loss rate (82/127 games). Value head is confused, reading losses as wins.

- **90k→150k:** sustained improvement. ECE drops 0.60→0.35, decided accuracy rises 0.30→0.56. The net is learning to correctly sign its positions — wins feel positive, losses feel negative.

- **150k→175k:** best performance on r4 eval. ECE ~0.35, accuracy ~0.54–0.56, MAE ~0.91–0.94.

- **195k regression:** brief dip back to ECE=0.47, acc=0.43. Likely noise — 195k falls just before the r4→r5 radius step change at 200k.

- **200k transition (r4→r5):** sharp regression — ECE jumps to 0.54, accuracy drops to 0.36. The net immediately struggles on the wider-radius r5 eval games. This is expected: the net trained under r4 is suddenly evaluated on longer-range positions its value head hasn't calibrated for.

- **200k→210k:** fastest recovery in the series. ECE drops from 0.54 to 0.30 in one 10k-step step — best ECE in the whole series. Value head re-adapts to r5 quickly.

- **210k–240k:** r5 plateau at ECE 0.30–0.37, accuracy 0.49–0.58, MAE 0.85–1.01. Mild improvement.

- **248k terminal dip:** ECE=0.47, acc=0.42, MAE=1.14 — back to 90k-era levels. This is the last banked ckpt (training ended around 248k–272k). Could be a bad checkpoint, final training instability, or a harder eval game set.

**Phase split:** Late-game positions consistently show HIGHER decided accuracy than early-game (e.g. at 150k: early 0.49 vs late 0.61). The value head is better calibrated closer to the end of the game — expected, since outcomes are more certain.

**Key signals for run3:**
1. The r5 transition at 200k causes a reliable calibration cliff (ECE +0.20, acc −0.19). Run3 training should monitor for this at each radius step.
2. Recovery is fast (one eval stride = 10k steps). Run3 value-health monitoring cadence of 10k is sufficient.
3. MAE > 0.90 across the whole series — value head never converges to tight outcome predictions. This is consistent with D-LOCALIZE (value is horizon-blind, not just miscalibrated in direction).
4. The 248k terminal dip warrants a separate investigation to rule out final checkpoint corruption vs training instability.

---

## Caveats

- **Games are self-play, not adversarial.** Win/loss distribution reflects the self-play curriculum, not SealBot performance. A run where the net wins 82 games looks more calibrated than a run where it loses 82 — ECE is win/loss-share-sensitive.
- **Radius per ckpt.** Each ckpt uses its OWN radius-matched eval book. Cross-radius comparisons (r4 vs r5) measure different game distributions. The 200k cliff is mostly radius-transition, not a training regression.
- **272k skipped.** Checkpoint stamped `v6_live2` not `v6_live2_ls` — gated loader rejects. No retro_slope games for 272k anyway.
- **Decided margin = 0.05.** Very small — effectively almost all positions are "decided." A stricter margin (e.g. 0.3) would isolate higher-confidence positions and likely show better accuracy.
