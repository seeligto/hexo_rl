# Q27 — ZOI reachability audit on real mid/late positions (Probe 1b)

**Date:** 2026-04-19
**Fixture:** `fixtures/threat_probe_positions.npz` regenerated from
`runs/10cc8d56e4394a9ca542740c4bcee069` (500-game self-play run, Apr 16
19:35, median ply 169, 322/500 games reach ply ≥ 49).
**Phase mix:** early=7, mid=7, late=6 (N=20).
**Scope:** §77 follow-up #3, re-run — audit ZOI reachability on real game
positions where the §77 truncation failure modes (ply > `zoi_lookback=16`,
disjoint-cluster threats) can actually fire.
**Supersedes:** the synthetic-only audit in
`reports/q27_zoi_reachability_2026-04-19.md` (caveat §3 "Caveat on
generality"), which concluded 0/20 extensions outside ZOI but on ply=7
positions that could not exercise the §77 failure modes.

---

## 1. Fixture regeneration

### Tooling changes

`scripts/generate_threat_probe_fixtures.py`

- Added `--n-per-phase EARLY MID LATE` CLI flag (e.g. `--n-per-phase 7 7 6`).
- `_phase(ply)` now buckets by compound_move = `(ply + 1) // 2` when
  `ply > 0`, per the user-facing threshold used in Q27 Probe 1b:
  - **early**: compound_move < 10 (ply < 19)
  - **mid**:   10 ≤ compound_move < 25 (19 ≤ ply < 49)
  - **late**:  compound_move ≥ 25 (ply ≥ 49)
  Prior thresholds (`ply < 15`, `ply < 50`) did not match the intended
  bucketing and would have placed ply=15–18 positions in "mid" — treated
  as early under the revised scheme.
- When `--n-per-phase` is set and a phase quota is not met, the script
  exits 2 rather than silently back-filling with synthetic positions.
  Preserves fixture realism.

Backup of the prior synthetic fixture retained at
`fixtures/threat_probe_positions_synthetic_backup.npz` (not committed;
regeneratable via `--synthetic`).

### Resulting fixture

20 positions. Per-phase counts: `{'early': 7, 'mid': 7, 'late': 6}`.
`side_to_move` split: 8×P1 / 12×P2. Ply span: 9 → 150.
Schema unchanged: `states (20, 18, 19, 19) float16`, `side_to_move int8`,
`ext_cell_idx int32`, `control_cell_idx int32`, `game_phase U8`.

---

## 2. Per-position ZOI reachability

Config (`configs/selfplay.yaml:64-66`): `zoi_enabled: true, zoi_lookback: 16,
zoi_margin: 5`.

For each position we reconstruct the move history at extraction time,
define the ZOI anchor set as the **last 16 moves** (or all moves when
ply ≤ 16), and compute `min_dist(ext, anchors)`. `ext_d_any` is the
distance to the nearest stone over the **full history** (parity signal
— shows whether the extension is geometrically anchored even when ZOI
truncation hides it).

|  # | phase | side | ply | n_zoi | ext (q,r) | ext_d_zoi | ext_d_any | ctrl_d_zoi | in_ZOI_ext |
|---:|------:|-----:|----:|------:|----------:|----------:|----------:|-----------:|:----------:|
|  0 | late  | +1 |  64 | 16 | ( 32, -1) |  2 |  2 |  7 | yes |
|  1 | mid   | +1 |  19 | 16 | (  1, -3) |  1 |  1 | 14 | yes |
|  2 | early | -1 |  14 | 14 | (  5, -1) |  2 |  2 | 14 | yes |
|  3 | late  | +1 |  60 | 16 | ( 31, -1) |  1 |  1 | 11 | yes |
|  4 | mid   | +1 |  24 | 16 | (  1, -3) |  2 |  1 | 17 | yes |
|  5 | early | -1 |  18 | 16 | ( -1, -1) |  1 |  1 | 14 | yes |
|  6 | early | -1 |   9 |  9 | (  0,  2) |  2 |  2 | 15 | yes |
|  7 | early | -1 |  17 | 16 | ( -3,  3) |  2 |  2 | 14 | yes |
|  8 | late  | -1 | 149 | 16 | ( 64,  2) |  2 |  2 |  5 | yes |
|  9 | late  | +1 |  91 | 16 | ( 32, -1) | 11 |  2 | 11 | **NO** |
| 10 | late  | +1 |  56 | 16 | ( 32, -1) |  2 |  2 |  7 | yes |
| 11 | late  | -1 | 150 | 16 | ( 64,  2) |  1 |  1 |  4 | yes |
| 12 | mid   | +1 |  20 | 16 | (  2, -3) |  1 |  1 | 14 | yes |
| 13 | mid   | -1 |  21 | 16 | ( -1, -1) |  1 |  1 | 14 | yes |
| 14 | early | -1 |  10 | 10 | (  2,  1) |  2 |  2 | 16 | yes |
| 15 | mid   | -1 |  22 | 16 | ( -1, -1) |  1 |  1 | 14 | yes |
| 16 | mid   | +1 |  23 | 16 | (  1, -3) |  1 |  1 | 14 | yes |
| 17 | mid   | -1 |  25 | 16 | ( -1, -1) |  2 |  1 | 17 | yes |
| 18 | early | +1 |  16 | 16 | (  2, -3) |  1 |  1 | 14 | yes |
| 19 | early | -1 |  10 | 10 | (  0,  3) |  2 |  2 | 14 | yes |

### Distance distribution — ext cell vs nearest ZOI anchor

| min distance | count |
|---:|---:|
| 1  | 9  |
| 2  | 10 |
| 11 | 1  |

Median = 2. 19/20 positions sit at `ext_d_zoi ≤ 2` — two hex steps from the
most recent 16 moves. One outlier (position #9) at distance 11. Compare
`ext_d_any` column: nearest stone over the full history is always within 2
steps — the outlier is purely a **truncation artifact**, not a geometric
isolation.

### Position #9 — §77 truncation failure mode, concrete instance

- ply 91, side-to-move P1, phase=late, cluster center (cq, cr) = (37, 5).
- Extension cell: (q, r) = (32, -1).
- Stones within hex-distance 3 of the extension (over full history):
  `(34,-1), (30,-1), (29,-1), (30,-2), (31,-3), (35,-1), (34,0)` — all
  placed before the lookback window cut-in.
- Last 16 moves: `[(127,2), (139,-6), (42,4), (101,2), (102,1), (50,-4),
  (120,0), (121,-3), (132,1), (111,1), (137,-6), (100,3), (37,5),
  (134,-3), (126,-5), (111,2)]`. Only one move — `(42,4)` — is within
  even hex-distance 11 of the extension, and that is a disjoint cluster
  a full 10 steps away.

The game has grown to span multiple disjoint colonies with q-values
spread from ~(-9,10) to ~(140,-6). The side-to-move's old 3-in-a-row in
the (q≈30, r≈-1) cluster was never "finished" and both sides have moved
on to fighting in remote clusters. Under the live post-search ZOI mask,
this extension cell would be filtered out of `move_candidates` — the
`zoi.legal_fallback_total` counter in `engine/src/game_runner/worker_loop.rs:424`
would trip, and the cell would only be chosen if fewer than 3 candidates
remained in-ZOI.

### Control cells (reference)

| min distance (ctrl to nearest ZOI anchor) | count |
|---:|---:|
| 4 | 1 |
| 5 | 1 |
| 7 | 2 |
| 11 | 2 |
| 14 | 8 |
| 15 | 1 |
| 16 | 1 |
| 17 | 2 |

18/20 control cells are outside ZOI (`d > 5`), as designed —
`find_control_cell` demands min-dist ≥ 4 from all stones and walks the
window in row-major order. The 2 inside are late-game positions with
ply ≥ 149, where the board is dense enough that even a `min_dist=4`
cell sits within 16-move recency of at least one stone.

---

## 3. Headline result

**Count of fixture positions where the extension cell is outside ZOI: 1 / 20 (5%).**

Phase breakdown of outside-ZOI extensions: `{'late': 1, 'mid': 0, 'early': 0}`.
Matches the §77 prediction: the truncation failure mode fires only when
`ply > zoi_lookback` AND the threat cluster has been idle for the full
lookback window. 1/6 late positions (16.7%) exhibit it in this sample;
zero mid or early.

Compared to the prior synthetic audit (0/20 outside ZOI on ply=7
positions), this is the confirmation the caveat in
`q27_zoi_reachability_2026-04-19.md` §3 demanded.

---

## 4. Threat-probe re-measurement on the new fixture

### Baseline v5 → v6

`fixtures/threat_probe_baseline.json` bumped to schema version 6
(`BASELINE_SCHEMA_VERSION = 6` in `scripts/probe_threat_logits.py:78`).
Fixture backing the baseline changed, so v5 numbers are not comparable to
v6. Prior synthetic-fixture baseline preserved at
`fixtures/threat_probe_baseline_v5_synthetic.json.bak` (not committed).

| metric | v5 (synthetic early) | v6 (real mid/late) |
|---|---:|---:|
| ext_logit_mean   |  0.080  |  0.015 |
| ext_logit_std    |  0.093  |  0.399 |
| ctrl_logit_mean  |  0.028  |  0.061 |
| ctrl_logit_std   |  0.012  |  0.030 |
| contrast_mean    | +0.052  | −0.046 |
| contrast_std     |  0.097  |  0.396 |
| ext_in_top5_pct  | 20.0%   | **60.0%** |
| ext_in_top10_pct | 20.0%   | **65.0%** |

The top-5/top-10 ranking metrics (C2/C3) jump from 20% → 60%/65% on the
same `bootstrap_model.pt`. The synthetic-early fixture was distributionally
out-of-sample for the trained policy (3-in-a-row + far stones does not
appear in real self-play games at ply=7), so the probe was asking the
model to rank threats it had never been trained to recognise in that
geometric configuration. On real self-play positions the bootstrap model
routes extensions into top-5 in the majority of cases.

Note `contrast_mean` flips negative: on real positions the scalar threat
head fires *more* on an empty far cell (`ctrl_logit_mean = 0.061`) than on
the extension (`ext_logit_mean = 0.015`). This is a separate diagnostic —
the policy *ranking* of extension cells is healthy (C2/C3 PASS by wide
margin) even when the raw threat-scalar magnitude is not.

### 5K checkpoint re-probe

`checkpoints/checkpoint_00005000.pt` (from the run that ended
2026-04-18 22:38) re-probed against the v6 baseline:

```
PASS  [C1] contrast=+3.317 (≥+0.380) OK
      [C2] top5=50% (≥25%) OK
      [C3] top10=65% (≥40%) OK
[C4] |Δ ext_logit_mean|=0.420 (<5.0) ok
```

All three gating criteria **PASS** on the real-fixture probe, with a
large margin on C1 (contrast +3.317 vs floor +0.38). This is the
inverse of the FAIL verdict logged in
`reports/probes/latest_20260418_223903.md` against the v5 synthetic
fixture.

---

## 5. Verdict

**Outcome A (with caveat).** §77's ZOI-truncation hypothesis is
falsified at the 95% level on realistic fixture positions: 19/20
extensions are well within the live ZOI mask radius. The single
outside-ZOI position (#9) is a concrete instance of the disjoint-cluster
truncation failure mode §77 predicted, but at 1/20 it cannot carry a
population-level C2/C3 miss on its own.

The more consequential finding is that the **Q27 C2/C3 failure itself
dissolves on the real fixture**. C2 goes 20% → 60% and C3 goes 20% → 65%
on the same `bootstrap_model.pt`, and the 5K checkpoint passes all
gates with C1 contrast +3.317. This was a fixture-realism problem, not
a model-training problem: the synthetic early-phase 3-in-a-row-with-far-stones
construction was asking the model to rank threats in a geometric
configuration that does not occur in its training distribution.

This does **not close Q27**. Probes 2 (aux threat weight sweep) and 3
(value-aggregation ablation) still run, independently, per §105 follow-up.
What this closes is:

- Probe 1 (ZOI reachability): cannot explain C2/C3 gap.
- The C2/C3 gap itself, as previously reported, was largely a fixture artifact.
  Future runs of the probe must use the real-fixture baseline as their
  kill criterion; the pre-§105 C2/C3 failure lines in the sprint log
  should be re-read with this in mind.

Follow-ups, for a separate sprint:

- **§77 truncation mode remains real in principle** (position #9 is an
  existence proof). If later sustained runs show late-game collapse
  concentrated on disjoint-cluster threats, the fix is to raise
  `zoi_lookback` or to make the ZOI anchor set colony-aware rather than
  purely recency-based. Not load-bearing for Phase 4.0 graduation.
- **C1 contrast flipped negative on bootstrap** (ctrl logit > ext logit
  on real fixture). Separate from Q27 policy ranking. Tracked as a
  threat-head magnitude diagnostic — does not gate.
- **Baseline v6 should be regenerated** the next time the fixture or
  post-bootstrap stack changes. `probe.fixtures` Make target already
  prints the required warning banner; same discipline applies here.

---

## 6. Tooling touched

- `scripts/generate_threat_probe_fixtures.py` — added `--n-per-phase`,
  compound_move phase thresholds, strict quota enforcement.
- `scripts/probe_threat_logits.py` — `BASELINE_SCHEMA_VERSION` 5 → 6.
- `fixtures/threat_probe_positions.npz` — regenerated from real run,
  20 positions (7/7/6).
- `fixtures/threat_probe_baseline.json` — new v6 baseline.

No changes to `engine/` code. No training touched.

---

*Method:* deterministic replay of `_sample_from_games` with seed=42
against `runs/10cc8d56e4394a9ca542740c4bcee069`; identical position set
to the committed NPZ. Per-position move histories captured from the
replay; hex distances via `hexo_rl.utils.coordinates.axial_distance`
(= `max(|dq|, |dr|, |dq+dr|)`). Ad-hoc analysis script at `/tmp/zoi_audit.py`
(not committed).
