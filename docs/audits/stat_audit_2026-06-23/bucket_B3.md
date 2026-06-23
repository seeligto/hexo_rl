# Bucket B3 — SEARCH stats audit

Sandbox pin `52067631`. Source read-only from `/home/timmy/Work/Hexo/statAudit_wt`.
Empirical claims use ONLY §4 banked logs. My stats are absent from the `train_*` logs but
PRESENT in the two banked `events_*` logs under surface names that DIFFER from the getter
names — recorded below.

Banked-sample distributions (events_cdf… n=76 / events_76cf… n=17):

| surface key (events log) | getter | cdf min/max/mean | 76cf min/max/mean |
|---|---|---|---|
| `mcts_mean_depth` | `mcts_mean_depth` | 3.749 / 3.895 / 3.794 | 3.642 / 3.791 / 3.720 |
| `mcts_root_concentration` | `mcts_mean_root_concentration` | 0.547 / 0.576 / 0.552 | 0.545 / 0.552 / 0.548 |
| `quiescence_fires_per_step` (delta) | `mcts_quiescence_fires` (raw) | 270 / 14435 / 1980 | 299 / 11877 / 3184 |
| `cluster_value_std_mean` | same | 0.282 / 0.437 / 0.423 | 0.353 / 0.438 / 0.410 |
| `cluster_policy_disagreement_mean` | same | 0.595 / 0.698 / 0.689 | 0.587 / 0.686 / 0.664 |
| `cluster_variance_sample_count` | same | 771 / 3.35M / 1.66M | 20360 / 775031 / 382214 |

---

## 1. mcts_mean_depth
- **emit_site:** `engine/src/game_runner/mod.rs:583` (getter); accum `worker_loop/inner.rs:946`,
  per-sim record `mcts/selection.rs:175-176`, formula `mcts/mod.rs:193-198`.
- **A (formula):** mean leaf descent depth = `depth_accum / sim_count`, scaled by 1e6 fixed-point
  in the atomic and divided back at the getter (`mod.rs:588`). `depth_accum += leaf_depth` per
  selected leaf (`selection.rs:175`). Math correct, no invented literal. The "~3.0/3.4" invented-
  literal concern (seed2) is a MONITOR/B5 artifact — NOT in this source; the source computes a real
  per-sim mean (banked ~3.7–3.9). PASS.
- **B (eff-n):** aggregates over every selected leaf since start; n = sim_count (genuine sim count),
  cumulative-since-start, no CI needed (descriptive telemetry, not a comparison gate). PASS.
- **C (planner-semantics):** depth is the INTERIOR-PUCT descent depth. Under Gumbel the root pick is
  SH-forced (`selection.rs:124-126` `forced_root_child`) but everything BELOW root still descends by
  PUCT (`selection.rs:137`), so leaf depth is a real interior-search quantity and meaningful under
  Gumbel-SH. PASS.
- **D (band):** no band; displayed passively (`terminal_dashboard.py:472-481`). PASS.
- **E (goodhart):** descriptive, no lever optimises it. PASS.
- **F (redundancy):** unique. PASS.
- **VERDICT: CORRECT** — keep. (Note: getter `mcts_mean_depth` → surfaces as same key. The
  invented-literal seed lives in the monitor bucket, not here; re-derivation finds source clean.)

## 2. mcts_mean_root_concentration  → seed1
- **emit_site:** `engine/src/game_runner/mod.rs:594` (getter); accum `worker_loop/inner.rs:947`,
  formula `mcts/mod.rs:199-212`. Surfaced as `mcts_root_concentration` (events.py:260).
- **A (formula):** `max_child_visits / root_total_visits`, range [0,1], guarded count==0 →0.0
  (`mod.rs:595-599`). Fixed-point 1e6 round-trip correct. PASS on arithmetic.
- **C (planner-semantics) — FAIL:** This is the seed1 quantity. Under Gumbel-SH the root visit
  distribution is NOT produced by PUCT confidence — it is MECHANICALLY allocated by Sequential
  Halving. `run_mcts_search` (`worker_loop/inner.rs:773-804`) forces each surviving candidate to
  receive `sims_per` visits via `tree.forced_root_child` (`inner.rs:785,794`), and root descent
  bypasses PUCT entirely for the forced child (`selection.rs:124-126`). The winner of the final
  halving phase accumulates roughly half the total budget, so `max/total` is pinned by the SH
  schedule, NOT by how "confident" search is. Banked proof: across 76 + 17 points the value sits in
  0.545–0.576 (σ≈0.006) — essentially flat regardless of position, exactly what an SH-pinned ratio
  predicts. Reading a rise/drop as "search sharpened/collapsed" (the documented D5 reading) is
  Gumbel-meaningless: the lever that moves it is `gumbel_m` / budget split, not play quality.
- **B/D/E/F:** band absent (passive display, `terminal_dashboard.py:474-481`), so the seed1
  "collapse=search-drop GATE" is NOT enforced in this source — it's a passive number; the defect is
  construct-validity, not a live alert. n genuine.
- **VERDICT: WRONG** (Axis C; precedence WRONG>others). The quantity is meaningless as a search-
  health signal under Gumbel-SH and there is no PUCT-style fix that restores meaning (SH owns root
  allocation by construction). **Action: DROP.** Gumbel-meaningless + unfixable. If a root-decision-
  sharpness signal is wanted, derive it from the Gumbel improved-policy / final argmax-gap, not from
  raw root visit concentration.

## 3. mcts_quiescence_fires
- **emit_site:** `engine/src/game_runner/mod.rs:605` (raw cumulative getter); per-search read
  `worker_loop/inner.rs:949-952`; per-move reset via `tree.new_game()` (`inner.rs:880`,
  `mcts/mod.rs:141`); 4 firing branches `mcts/backup.rs:223-244`; delta surface
  `step_coordinator.py:1266-1268` → `quiescence_fires_per_step` (events.py:228).
- **A (formula):** Counts the 4 quiescence override/blend branches (current_wins≥3, opponent_wins≥3,
  current_wins==2, opponent_wins==2), +1 per fire (`backup.rs:217-247`). Tree counter resets each
  MOVE (new_game per move, `inner.rs:880`) and the per-move count is added into the runner atomic
  once per search (`inner.rs:949`) — NO double count (verified reset cadence). Python then diffs the
  cumulative runner getter into a per-step delta (`step_coordinator.py:1267`). Correct. PASS.
- **B (eff-n):** cumulative count, diffed per log_interval; raw count, no CI. PASS.
- **C (planner-semantics):** quiescence is a leaf value-override independent of the root planner —
  fires identically under Gumbel and PUCT. Meaningful under Gumbel. PASS.
- **D (band):** no band. PASS.
- **E (goodhart):** event-count telemetry; not a target. PASS.
- **F (redundancy):** the raw getter is the sole source for the per-step surface; not duplicated. PASS.
- **VERDICT: CORRECT** — keep. (Minor: the surface is a delta of this raw getter; both are needed.)

## 4. cluster_value_std_mean
- **emit_site:** `engine/src/game_runner/mod.rs:612` (getter); compute `worker_loop/inner.rs:651-673`.
- **A (formula) — defect:** Per K≥2 multi-window leaf, population std of the K cluster value
  predictions (`inner.rs:652-656`), accumulated and divided by `cluster_variance_samples` at the
  getter (`mod.rs:612-616`) → a **lifetime cumulative mean since start**, NOT a per-iteration value.
  It is logged RAW every iteration (`events.py:263`), so once sample_count reaches millions
  (banked: 3.35M) a single iteration's samples cannot move the displayed mean. Banked proof: value
  is flat at ~0.42 across 76 points while sample_count climbs 771→3.35M — the metric is structurally
  incapable of showing drift, i.e. it is presented as a live health signal but is a frozen lifetime
  average. The arithmetic per-leaf is correct; the surfacing-as-per-step is the math/intent mismatch.
- **B (eff-n):** n IS the distinct K≥2 sample count (witnessed by stat #6); honest n, but the
  lifetime accumulation means recent-window n is invisible — no windowed mean / no reset. Skew.
- **C:** cluster variance is multi-WINDOW (board-tiling) variance at the leaf, independent of root
  planner — Gumbel-valid. PASS.
- **D:** no band. n/a.
- **E (goodhart):** I2 investigation metric (Q2/Q27); a real construct but, being a lifetime mean,
  cannot serve the "is variance rising NOW" question it is displayed for.
- **F:** distinct from policy-disagreement. PASS.
- **VERDICT: BIASED** (Axis A surfacing defect + Axis E live-signal mismatch; not WRONG — the per-
  leaf std is correct math, and it is Gumbel-meaningful). **Action: FIX** — emit a WINDOWED mean
  (delta accum / delta sample_count between log intervals, mirroring the qfire delta pattern at
  `step_coordinator.py:1267`) instead of the lifetime cumulative mean, OR re-label the surface as
  "lifetime mean" so it is not read as a per-step health number.

## 5. cluster_policy_disagreement_mean
- **emit_site:** `engine/src/game_runner/mod.rs:622` (getter); compute `worker_loop/inner.rs:657-675`.
- **A (formula) — same defect class as #4:** `1 - (max top-1 agreement count / K)` per K≥2 leaf
  (`inner.rs:657-671`), accumulated and divided by the SAME `cluster_variance_samples` lifetime
  counter (`mod.rs:622-626`). Per-leaf disagreement math is correct (top-1 argmax mode count). But
  again a lifetime cumulative mean logged raw per iteration (`events.py:264`) — banked flat at ~0.69
  across 76 points while sample_count grows to millions; cannot show per-step drift.
- **B:** honest distinct-leaf n; lifetime accumulation hides recent n. Skew.
- **C:** multi-window top-1 disagreement, root-planner-independent → Gumbel-valid. PASS.
- **D:** no band. n/a.
- **E:** I2 investigation metric, real construct, but live-signal-incapable as surfaced.
- **F:** distinct from value-std (policy-top1 vs value-magnitude). PASS.
- **VERDICT: BIASED** (Axis A surfacing + Axis E). **Action: FIX** — same windowed-delta fix as #4.

## 6. cluster_variance_sample_count
- **emit_site:** `engine/src/game_runner/mod.rs:631` (getter); incremented `worker_loop/inner.rs:676`.
- **A (formula):** monotone count of K≥2 multi-cluster positions scored since start
  (`inner.rs:676`, `mod.rs:631-632`). Correct. PASS.
- **B (eff-n):** This stat IS the eff-n witness/denominator for stats #4 and #5 — it certifies how
  many distinct K≥2 leaves backed the two cluster means. Honest distinct count. PASS (it is the
  honesty instrument, not a victim of dishonesty).
- **C:** count of multi-window scoring events; planner-independent. PASS.
- **D:** no band; cumulative counter (banked 771→3.35M). PASS.
- **E:** coverage/denominator telemetry; not optimised. PASS.
- **F (redundancy):** It is the shared divisor of #4 and #5, but it carries DISTINCT information the
  two means do not surface: the sample size behind them (and exposes their lifetime-accumulation
  problem). Not a duplicate emit; removing it would strip the only n-witness for the cluster means.
  PASS.
- **VERDICT: CORRECT** — keep. It is the eff-n denominator that makes the (to-be-fixed) cluster means
  auditable; no band to miscalibrate, count is exact.

---

## Seeds re-derived (B3-owned)

### seed1 — root_concentration / depth D5 "collapse = search-drop" reading (Axis C)
**Outcome: CONFIRMED (with refinement).** Re-derived from source: under Gumbel the root visit
distribution is allocated by Sequential Halving, not PUCT. `run_mcts_search`
(`worker_loop/inner.rs:773-804`) forces `sims_per` visits onto each surviving SH candidate via
`tree.forced_root_child`, and `select_one_leaf` bypasses PUCT at root for the forced child
(`mcts/selection.rs:124-126`). Therefore `max_child_visits/total` (`mcts/mod.rs:199-212`) is pinned
by the halving schedule — banked `mcts_root_concentration` is flat at 0.545–0.576 across 93 points.
Reading its level as "search confidence collapsed/sharpened" is meaningless under Gumbel-SH →
seed1 confirms; root_concentration is WRONG/DROP. Refinement vs documented seed: the "collapse=
search-drop" reading is NOT wired as a live GATE in this source — root_concentration and depth are
displayed PASSIVELY (`terminal_dashboard.py:472-481`, no threshold). So the defect is construct-
validity of a passively-surfaced number, not a firing false-alarm. The `depth` half of the seed is
SEPARATELY fine: interior-PUCT descent depth IS meaningful under Gumbel (only the ROOT pick is SH-
forced) — depth = CORRECT/keep. The seed conflated two stats; only root_concentration is the casualty.
