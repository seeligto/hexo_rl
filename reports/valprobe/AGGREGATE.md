# D-C VALPROBE — AGGREGATE (authoritative)

**VERDICT: V-CONFIRM**

**Generated:** 2026-07-10  
**Branch:** phase4.5/valprobe_dc  
**Role:** AGGREGATE step — fresh context, no prior involvement. Frozen verdicts verified against
measured artifacts; no measurements re-run.

---

## The one number run3 needs

**LATE range: 33–54% (robust-core 33–39%, as-frozen 45.6%, censoring-corrected 54.4%)**

Point estimate: 45.6% (26/57). All three numbers exceed the 30% V-CONFIRM threshold. See §4 for
why a range, not a single number, is the correct read.

---

## 1. Frozen-criteria check

Pre-registered thresholds (recognition_lag.md §1):
- **V-CONFIRM fires when:** LATE ≥ 30% AND false-pessimism ≤ 10% on win control

Measured on 248k arm (primary):

| Criterion | Threshold | Measured | Pass? |
|---|---|---|---|
| LATE fraction (n=57) | ≥ 30% (≥18/57) | **45.6% (26/57)** | PASS |
| False-pessimism @ −0.5 | ≤ 10% (≤5/57) | **1.8% (1/57)** | PASS |
| UNMEASURABLE | < 25% (power gate) | 8.8% (5/57) | PASS — not power-degraded |
| V-KILL condition | ≥ 60% EARLY | 35.1% (20/57) | NOT MET — V-KILL rejected |

V-CONFIRM fires. V-KILL does not. Simultaneous-fire (MIXED path) cannot occur: 45.6%+35.1%=80.7%,
split by MID+UNMEASURABLE — arithmetic doesn't produce both-at-threshold here. MIXED adjudication
addendum does not apply.

**Cross-tab integrity check (censoring):** LATE games are 100% uncensored (26/26 have T_provable
fully established). EARLY games are 50% censored (10/20 have T_provable undefined — censoring
makes EARLY the safe class under undefined T_provable, per §4.5). The 26 LATE games rest on
SealBot-proven point-of-no-return with T_provable established before the terminal. No LATE game
gets its classification from a censored proof.

---

## 2. LATE as a range, not a point — why

Red-team produced four alternative LATE estimates; all ≥ 30%:

| Scenario | LATE |
|---|---|
| Maximal-adversarial (reclassify 7 never-crossed→LATE games to UNMEASURABLE) | 33.3% (19/57) |
| Strong-core (sweep at −0.3 threshold) | 38.6% (22/57) |
| As-frozen (−0.5 primary threshold) | **45.6% (26/57)** |
| Censoring-corrected (LATE-uncensored / non-EARLY-uncensored denominator) | 54.4% (26/[57−10]) |
| Guard-off (no window guard on SealBot proofs) | 68.4% — NOT reported; window guard is load-bearing |

Sweep stability: V-CONFIRM at −0.3 (38.6%), −0.5 (45.6%), −0.7 (50.9%) — no flip across the
threshold sweep. Not threshold-fragile.

**The load-bearing caveat:** 7 of the 26 LATE games are classified LATE via the never-crossed path
(value never reaches −0.5 during the entire loss game). Their T_provable is established, but the
terminal-only provability means the point-of-no-return is close to game-end. These 7 games rest on
weaker evidence than the 19 games with an established T_cross. At the 33–39% robust-core range
(which excludes them by worst-case or sweep sensitivity), the clustered bootstrap CI straddles 0.30
(lower bound: clustered bootstrap CI on 248k LATE = [32.3%, 60.0%]; at strong-core 38.6%, the CI
lower bound also straddles 0.30). Per project convention: a verdict whose CI-lower straddles the
threshold is FRAGILE. **V-CONFIRM is therefore FRAGILE at the robust-core boundary — it requires
only 9 reclassifications to flip the point estimate below 30%, and the CI-lower dies at 2.**

The verdict stands, but run3 should not treat the 33–39% floor as a precise measurement — it's an
adversarial lower-bound construction. The as-frozen 45.6% and censoring-corrected 54.4% are the
more representative estimates under sound methodology.

---

## 3. What V-CONFIRM means for run3

**Card #1 (distributional value head) premise CONFIRMED.**

The failure is **value-REPRESENTATION**: the value head stays optimistic (v_t > −0.5) after the
position is irreversibly lost (T_provable established by SealBot d6–d8). In 45.6% of loss games,
the head lags provable loss by ≥ 2 compound turns — sometimes 3–5 turns. The head is not seeing
the loss and being overridden by policy/search (V-KILL, 35.1% EARLY, far below 60%).

**Card #1 probe set exported:** `reports/valprobe/card1_probe_set.jsonl` — 41 positions where the
head has a SealBot-proven loss AND v_t ≥ −0.5 (value optimistic while provably lost). These are
the training-distribution examples that directly motivate the distributional value head: the current
scalar head cannot represent the uncertainty in these positions.

**V-KILL rejected:** EARLY = 35.1%, well below the 60% threshold. Even under maximal-adversarial
reclassification, EARLY cannot reach 60% without also moving games that are unambiguously LATE.
The hypothesis "value head sees the loss, policy/search walks in anyway" is not supported.

**Implication:** the run3 lever is training-time value target improvement, not inference-time
policy/search override. The distributional value head (card #1) addresses the representation gap
directly. A policy-side or search-side fix alone would not close this gap.

---

## 4. Robustness summary

Six red-team attacks, all survived (per REVIEW+RED-TEAM in the dispatch brief):

1. **Cross-host v_t recomputation:** laptop re-ran all 1297 positions (248k arm) →
   max|Δv| = 9.7e-3, ZERO class changes → verdict host-independent.
2. **Censoring is 100% EARLY-sided:** 10 EARLY censored, 0 LATE censored → LATE count is
   structurally sound; censoring cannot inflate LATE.
3. **d6/5s cap is conservative:** deeper d8 doesn't move T_provable (point-of-no-return is
   established in the final unbroken backward streak; deeper probes can only push T_provable
   earlier, making lag larger, never smaller).
4. **Sweep stability:** V-CONFIRM at all three thresholds (−0.3/−0.5/−0.7).
5. **Oscillation streak soundness:** 26 oscillating games have mate-ladder proof structure
   (|score| ≥ 99999000); the point-of-no-return definition correctly ignores pre-streak transient
   proofs from opponent blunders.
6. **Replay-match gate:** aggregate_replay_match_rate = 0.915 (248k). REPLAY_MATCH_MIN was
   lowered to 0.85 (GPU Gumbel-SH non-determinism on vast); v_t forward pass is unaffected (a
   different computation path); q_t-mismatched positions excluded from card1_probe_set only.

**Single load-bearing caveat (repeated):** 7 never-crossed→LATE games rest on terminal-only
provability. At the 33–39% robust-core the clustered CI straddles 0.30. The verdict is FRAGILE
per design convention at this boundary. 9 reclassifications flip the point estimate.

---

## 5. Corroborating lines (independent convergence)

Three independent measurement lines all point to the same failure mode:

**Line 1 — WP1 (this investigation):** Direct trajectory measurement. LATE 45.6% at 248k;
LATE 50.0% at 175k (secondary, different opening book, distributional comparison only).
Lag_srch (searched root value) also lags: median +1 turn, mean +1.82 turns at 248k — the 150-sim
Gumbel search does not recover what the raw value head misses.

**Line 2 — WP4 (value health series):** Solver-independent calibration over 15 checkpoints.
ECE 0.30–0.60 throughout, decided-accuracy near-chance at the 200k radius cliff (0.36) and at
the 248k terminal ckpt (0.42). MAE > 0.90 across the entire run — value head never produces tight
outcome predictions. The 200k r4→r5 radius cliff (ECE +0.20, acc −0.19) is a reliable structural
failure at each curriculum step. [artifact: value_health_series.jsonl; commit c8635d7]

**Line 3 — D-LOCALIZE (prior investigation):** Localized the run2 vs SealBot gap to the VALUE
head: net 0.6–1.0 at d6 forced-loss positions, 56/61=92% value-blind to SealBot-reachable losses.
(memory: d-localize-value-target-verdict)

All three lines converge on: the value head is horizon-blind and optimism-biased in losing
positions, not a policy/search artifact.

---

## 6. 175k secondary result

175k arm: LATE = 50.0% (26/52), FP@−0.5 = 0.0% (0/52). Also meets V-CONFIRM criteria by the
frozen thresholds, but is SECONDARY (distributional, different opening book evalfair_r4_v2 at
radius 4 vs 248k's evalfair_r5_v2 at radius 5; comparison is not opening-matched). Reported for
completeness, no verdict action. 175k verdict in summary.json = "MIXED" — this reflects the
pipeline's pre-revision (earliest-ever T_provable) run that returned MIXED; the current
point-of-no-return revision gives V-CONFIRM by criteria at 175k too, but the secondary arm is
not adjudicated. Primary arm governs.

---

## 7. Methodological arc — why the native solver dead-end happened

WP1 design specified native TacticalSolver as the prover. The solver was found algorithm-bound
(D-FORENSIC F2: 0/40 traps provable at 20k→3M nodes; quiet-move candidate-gen truncation prevents
proof). For WP1, pilot showed the solver proved head_lost only at the terminal-adjacent ply for
100% of loss games, pinning T_provable to game-end and manufacturing spurious V-KILL by construction
(§6.1 red-team risk #1). [artifact: solver_abort_evidence.json]

Pivot: SealBot d6–d8 with window guard ON + point-of-no-return T_provable definition (backward
streak from terminal, not earliest-ever proved-lost). SealBot proves quiet mid-game positions that
the native solver cannot; the point-of-no-return definition correctly handles oscillating games
where the opponent blundered back a won position (26 such games in the 248k arm, median pre-streak
gap = 0 turns).

**Open WP items** (not blocking the V-CONFIRM verdict):
- WP2 (probe-set ≥200 positions): OPEN — the 41-position card1_probe_set is the current export;
  expanding to ≥200 requires a larger evalfair book run.
- WP3-C2 (canary): OPEN — WP3-C1 event-contract test landed (commit a4e7a13); C2 canary not built.

---

## 8. Artifact existence table

| Artifact | Path | Commit | Status |
|---|---|---|---|
| Design doc (frozen verdicts §1, T_provable §4.3, IMPL spec §5) | `reports/valprobe/recognition_lag.md` | ada51c2 | VERIFIED — §7 fully filled, both arms |
| 248k summary.json | `reports/valprobe/248k/summary.json` | ada51c2 | VERIFIED — verdict=V-CONFIRM, class_counts LATE:26/EARLY:20/MID:6/UNMEASURABLE:5, FP@-0.5=1.75%, power_degraded=False, oscillation_count=26 |
| 248k games.jsonl | `reports/valprobe/248k/games.jsonl` | ada51c2 | VERIFIED — 114 lines (57 loss + 57 win control games) |
| 248k positions.jsonl | `reports/valprobe/248k/positions.jsonl` | ada51c2 | VERIFIED — 1854 lines |
| 248k solver_abort_evidence.json | `reports/valprobe/248k/solver_abort_evidence.json` | c23fe23 | VERIFIED — abort=True, algorithm-bound confirmed |
| 175k summary.json | `reports/valprobe/175k/summary.json` | ada51c2 | VERIFIED — verdict=MIXED (pre-revision run), LATE:26/EARLY:15/MID:5/UNMEASURABLE:6 |
| 175k games.jsonl | `reports/valprobe/175k/games.jsonl` | ada51c2 | VERIFIED — 104 lines (52 loss + 52 win control) |
| card1_probe_set.jsonl | `reports/valprobe/card1_probe_set.jsonl` | ada51c2 | VERIFIED — 41 positions |
| value_health_series.jsonl | `reports/valprobe/value_health_series.jsonl` | c8635d7 | VERIFIED — 15 ckpts 50k–248k |
| value_health.md | `reports/valprobe/value_health.md` | c8635d7 | VERIFIED |
| WP1 re-run commit (ada51c2) | — | ada51c2 | VERIFIED present in log |
| threat_moves perf (4692350) | — | 4692350 | VERIFIED present in log |
| WP3-C1 test (a4e7a13) | — | a4e7a13 | VERIFIED present in log |
| WP4 (c8635d7) | — | c8635d7 | VERIFIED present in log |

**Missing / open:**
- WP2 probe-set expansion (≥200 positions): NOT built
- WP3-C2 canary: NOT built
- 272k checkpoint: EXCLUDED by gated loader (stamped v6_live2 not v6_live2_ls) — expected, not missing
