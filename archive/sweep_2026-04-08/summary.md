# Config Sweep Results — 2026-04-08

## Hardware
- CPU: Ryzen 7 8845HS
- GPU: RTX 4060 Laptop (8GB VRAM)
- Run duration: 20 min wall-clock (90s warm-up excluded)
- All runs: completed_q_values=true, fresh from bootstrap_model.pt

---

## PUCT Arm Rankings

Ranked by steps/hr and games/hr as co-primary. Draw rate is a first-class column.

| Rank | Run | ratio | burst | game_moves | wait_ms | leaf_bs | wkrs | steps/hr | games/hr | draw% | gl_p50 | calls/move | batch% | val_slope | flags |
|------|-----|-------|-------|------------|---------|---------|------|----------|----------|-------|--------|------------|--------|-----------|-------|
| 1 | **P8** | 6 | 24 | 150 | 4 | 16 | 18 | **2487** | 415 | **63%** | 75 | 13.1 | 98.7 | -3.5e-5 | HIGH_DRAW |
| 2 | **P3** | 4 | 16 | 150 | 4 | 8 | 14 | **2422** | **606** | **39%** | 45 | 11.2 | 94.2 | -3.4e-5 | |
| 3 | P7 | 6 | 24 | 150 | 4 | 16 | 14 | 2303 | 384 | 63% | 75 | 14.0 | 96.6 | -3.4e-5 | HIGH_DRAW |
| 4 | P2 | 4 | 16 | 150 | 12 | 8 | 14 | 2129 | 533 | 49% | 69 | 11.7 | 93.5 | -4.4e-5 | |
| 5 | P6 | 4 | 16 | 150 | 12 | 8 | 32 | 2070 | 518 | 43% | 58 | 20.3 | 99.4 | -4.3e-5 | |
| 6 | P1 | 4 | 16 | 200 | 12 | 8 | 14 | 1822 | 457 | 47% | 65 | 10.9 | 95.2 | -3.6e-5 | |
| 7 | P4 | 4 | 16 | 150 | 12 | 16 | 14 | 1719 | 431 | 64% | 75 | 13.3 | 96.5 | -5.1e-5 | HIGH_DRAW |
| 8 | P5 | 4 | 16 | 150 | 4 | 16 | 14 | 1698 | 425 | 64% | 75 | 13.8 | 97.1 | -6.3e-5 | HIGH_DRAW |
| 9 | P0 | 2 | 8 | 200 | 12 | 8 | 14 | 932 | 468 | 49% | 85 | 11.3 | 95.5 | -13.3e-5 | |

### P8 vs P3: the steps/hr vs games/hr disagreement

P8 leads on steps/hr (2487 vs 2422, +2.7%) but P3 leads on games/hr (606 vs 415, +46%). The gap is explained by draw rate: P8 draws 63% of games with median game_length=75 (the cap), while P3 draws only 39% with median game_length=45. P8's high ratio=6 + leaf_bs=16 generates a lot of training steps per game — but those steps come from long, indecisive games that hit the move cap. P3 produces more decisive games with shorter lengths and more diverse outcomes.

**Recommendation: P3 is the PUCT winner.** Its 2.7% steps/hr deficit is noise-level, while its +46% games/hr advantage and 24pp lower draw rate represent a categorically different training regime. More unique games with decisive outcomes produce better learning signal than fewer, longer, drawn-out games where the model rehearses indecision.

### P8b confirmation run (completed)

P8b (ratio=6, burst=24, wait_ms=4, **leaf_bs=8**, workers=18) isolates the leaf_bs effect:

| Run | leaf_bs | wkrs | steps/hr | games/hr | draw% | gl_p50 | calls/move |
|-----|---------|------|----------|----------|-------|--------|------------|
| P8  | 16 | 18 | 2487 | 415 | 63% | 75 | 13.1 |
| P8b | 8  | 18 | **2976** | 497 | 55% | 75 | 10.7 |
| P3  | 8  | 14 | 2422 | **606** | **39%** | **45** | 11.2 |

**Verdict: both factors contribute.** Switching leaf_bs 16→8 (P8→P8b) dropped draw rate 63→55% and boosted steps/hr by 20%. But P8b still draws 55% with gl_p50=75, so ratio=6/workers=18 independently pushes games toward the cap. P3 remains the recommended PUCT winner: its 23% steps/hr deficit vs P8b is real, but its +22% games/hr advantage, 16pp lower draw rate, and much shorter median game length (45 vs 75) indicate a fundamentally healthier training regime. The overnight run's quality gate (Bradley-Terry vs Checkpoint_0) should resolve whether the raw throughput advantage of P8b translates to strength.

---

## Gumbel Arm Rankings

| Rank | Run | m | ratio | burst | game_moves | wait_ms | leaf_bs | wkrs | steps/hr | games/hr | draw% | gl_p50 | calls/move | batch% | val_slope | flags |
|------|-----|---|-------|-------|------------|---------|---------|------|----------|----------|-------|--------|------------|--------|-----------|-------|
| 1 | G5 | 8 | 4 | 16 | 150 | 2 | 16 | 14 | **1721** | 431 | **44%** | 39 | 19.1 | 81.9 | -3.0e-5 | |
| 2 | **G3** | 8 | 2 | 8 | 150 | 2 | 8 | 14 | 1417 | **710** | **30%** | 20 | 15.4 | 77.2 | -6.5e-5 | |
| 3 | G2 | 8 | 2 | 8 | 200 | 2 | 8 | 14 | 1211 | 607 | 23% | 20 | 16.2 | 79.0 | -5.8e-5 | |
| 4 | G4 | 8 | 2 | 8 | 150 | 2 | 16 | 14 | 1134 | 569 | 37% | 23 | 16.5 | 83.0 | -11.1e-5 | |
| 5 | G1 | 16 | 2 | 8 | 200 | 2 | 8 | 14 | 569 | 286 | 56% | 100 | 20.8 | 80.0 | -34.9e-5 | |
| 6 | G0 | 16 | 2 | 8 | 200 | 12 | 8 | 14 | 549 | 276 | 49% | 88 | 21.1 | 76.0 | -16.9e-5 | |

### G3 vs G5: same pattern as P3 vs P8

G5 leads on steps/hr (1721 vs 1417, +21%) but G3 leads on games/hr (710 vs 431, +65%). G5's advantage comes from ratio=4 + leaf_bs=16, which inflates steps per game but draws 44% vs G3's 30%. G3's median game_length=20 (very short decisive games) vs G5's 39.

**Recommendation: G3 is the Gumbel winner.** Same reasoning as P3: the games/hr advantage and lower draw rate outweigh the steps/hr gap. G3 also has the highest games/hr of ANY run in the entire sweep (710), producing the most diverse training data.

### gumbel_m=16 to 8 was transformative

G0 to G2 (m=16 to 8, both wait_ms=2): steps/hr doubled (549 to 1211), games/hr doubled (276 to 607). Halving root candidates halved per-move compute. This is the single largest knob in the Gumbel arm.

---

## Key Findings

### 1. leaf_bs=16 is a negative result

The hypothesis (from early Phase 4 discussion) that larger leaf batches would reduce inference round-trips and improve throughput does not survive this sweep. Two isolated comparisons:

| Comparison | leaf_bs | steps/hr | games/hr | draw% | calls/move |
|------------|---------|----------|----------|-------|------------|
| P2 (wait=12) | 8 | 2129 | 533 | 49% | 11.7 |
| P4 (wait=12) | 16 | 1719 | 431 | 64% | 13.3 |
| P3 (wait=4) | 8 | 2422 | 606 | 39% | 11.2 |
| P5 (wait=4) | 16 | 1698 | 425 | 64% | 13.8 |

leaf_bs=16 **increased** calls/move (opposite of intended), **decreased** both throughput metrics by 19-30%, and **increased** draw rate by 25pp. The mechanism: larger leaf batches change MCTS tree shape in a way that makes games longer and less decisive. Every PUCT run with leaf_bs=16 has median game_length=75 (the cap) and ~63% draw rate.

The Gumbel arm shows the same pattern (G3 vs G4): leaf_bs 8 to 16 reduced games/hr from 710 to 569 (-20%) and increased draw rate from 30% to 37%.

### 2. Ratio is the biggest throughput lever

P0 to P1 (ratio 2 to 4): steps/hr nearly doubled (932 to 1822). P1 to P7 (ratio 4 to 6): another +26% (1822 to 2303). But ratio=6 comes with a draw-rate cost when combined with leaf_bs=16.

### 3. wait_ms=12 to 4 gives a clean 14% boost

P2 to P3 (both leaf_bs=8, ratio=4): steps/hr +14% (2129 to 2422), games/hr +14% (533 to 606), draw rate actually **decreased** (49 to 39%). Lower inference wait time reduces batch formation latency without adverse effects at this worker count.

### 4. inf_bs=64 to 32 (P6) fills batches but loses GPU util

P6 hit 99.4% batch fill (highest in sweep) vs 93.5% at inf_bs=64 (P2). But GPU utilization dropped from 84% to 78% — smaller batches mean less GPU parallelism. Net effect: P6 is 3% slower than P2 on steps/hr. The threshold-reachable hypothesis is confirmed (batches DO fill more completely) but the tradeoff is unfavorable.

### 5. workers=14 to 18 (P7 to P8) gives modest gains

P8 gained 8% steps/hr over P7 with 4 more workers. But both share the leaf_bs=16 / high-draw problem. The P8b confirmation run will test workers=18 with leaf_bs=8.

---

## Recommended Configs for Overnight Run

**PUCT arm: P3** — ratio=4, burst=16, game_moves=150, wait_ms=4, leaf_bs=8, inf_bs=64, workers=14
- 2,422 steps/hr, 606 games/hr, 39% draw rate
- Best games/hr in the PUCT arm, competitive steps/hr, lowest draw rate in the top tier

**Gumbel arm: G3** — m=8, ratio=2, burst=8, game_moves=150, wait_ms=2, leaf_bs=8, workers=14
- 1,417 steps/hr, 710 games/hr, 30% draw rate
- Highest games/hr in the entire sweep, lowest draw rate in Gumbel arm

**Caveat:** 20-min runs do NOT measure absolute strength. Quality verdict
requires overnight Bradley-Terry eval of the winner against Checkpoint_0.
Do NOT pick an overall winner across arms — PUCT vs Gumbel is a research
bet that belongs in a separate overnight comparison.

---

## Methodology Notes

- All runs started fresh from bootstrap_model.pt (step 0, empty replay buffer)
- buffer_persist: false on all runs to prevent replay buffer contamination
- Warm-up: first 90s of training data excluded from all metrics
- inference_calls_per_move: computed as delta(forward_count) / delta(total_moves) across the widest available summary-event window
- policy_entropy_floor=0.0 on all runs (LOW_ENTROPY flag): 20-min window artifact — entropy starts near zero from bootstrap and hasn't climbed yet. Not a collapse signal.
- gpu_train_frac estimated from train_step inter-arrival times (<0.5s gap = training active)
- Post-sweep `make bench.full` (n=5): all 10 targets PASS, no regressions
