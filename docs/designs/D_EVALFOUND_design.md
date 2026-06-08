# D-EVALFOUND — eval & run foundation: DESIGN spec v2 (pre-registered)

**Status:** DESIGN (Phase 1) — v2, post fresh-context REVIEW (`wf_8eb7e9d8`, PASS_WITH_CHANGES, 11
MAJORs). Pre-registration — locked before IMPL. Eval/infra only; no training run, no encoding
change, no multi-cluster Rust, no Phase-4.5 features.

**Frame.** The §D arc steered/judged a run on vs-SealBot WR — the project's own flagged-wrong
instrument for self-play strength — and misdirected six investigations. Build the two foundations
every downstream fork depends on: (1) steer/abort on the *right* signal (checkpoint-relative
strength + an adversarial/off-distribution robustness gate, NOT SealBot-WR); (2) eval throughput
(serial-eval halves training throughput, GPU ~50%). Validate the strength instrument by USING it to
resolve the two §D-FOUNDING open questions at power. Verdicts FORK the roadmap (operator-owned, NOT
actioned here).

**Anchor convention (REVIEW fix).** TRACKED files carry a bare `file:line`. The round-robin driver is
**`investigation/founding_2026-06-08/rr_driver.py` (UNTRACKED, gitignored)** — every `rr_driver.py:NNN`
below is a *promotion target*, not existing validated code; IMPL §2.1 promotes it to a tracked
primitive. This doc's anchors were checked against the tree this session *except* those marked
(untracked).

**Measurements run this session** (eval-only, local RTX 4060, banked ckpt 50k; probe
`investigation/evalfound_2026-06-08/batch_variance_probe.py`) — they resolve two REVIEW MAJORs and
are cited inline:
- **M-VAR (batch-size FP variance):** 32 mid-game boards, batch=1-alone vs batch=32-together →
  **0/32 argmax flips** under both fp16-autocast (max logit |Δ| 1.7e-2) and float32 (3.7e-3).
  Float32 *shrinks* divergence ~4.7× but does **not** zero it. Repeat-determinism 0/32.
- **M-TP (throughput baseline):** serial eval = **9,252 games/hr, GPU-util 53.3% mean / 55% max**
  (sims=100, n=8). Confirms the GPU-50% bug exactly.
- **M-CYC (banked-ladder non-transitivity):** directed 3-cycle density = **0.073** (16/220 triples,
  12 rungs); pairwise inversion 38% (25/66). Many pair inversions, few full RPS triples.

---

## 0. Architecture: TWO-TIER instrument (load-bearing — read first)

A full all-pairs round-robin every eval round is too expensive and structurally wrong for steering.
Split:

- **Tier A — offline ladder instrument** (`rr_driver` → tracked primitive). All-pairs round-robin
  over a *banked* ladder. The §D-FOUNDING instrument; the Phase-3 tool; post-hoc. Emits win-matrix +
  BT-Elo + non-transitivity index + robust aggregate. Heavy (n×pairs games).
- **Tier B — live steer/abort signal.** At eval cadence, rate the *current* checkpoint against a
  **small FIXED frozen reference set** (~K matches/eval). The Objective-B strength floor replacing
  SealBot-WR in the abort path.

Why fixed-reference, not running-best: §D-FOUNDING §Phase 0 — the running-best arena's "100k lost
0.33 to frozen 75k-best" was inflated by the restart **anchor-reset confound** + single-read noise;
the all-pairs RR did not reproduce it (75k−100k ≈ +40 Elo, CIs overlap). A *moving* champion also
re-introduces a changing-gauge artifact.

**REVIEW correction — "fixed ≠ acyclic" (MAJOR #6).** A fixed gauge removes the *changing-gauge*
artifact; it does NOT remove non-transitivity. M-CYC: 38% of pairs invert, so any 4–5-rung subset
inherits ~2–3 inversions by pigeonhole. The fixed-ref aggregate is therefore still cycle-*contaminated*;
the **cycle-aware abort rule (§1a) is the actual non-transitivity mitigation, not the fixed gauge.**
Mitigations: (a) the ref-set's *own* internal 3-cycle density is computed + logged at lock time, WARN
if > `cycle_density_max`; (b) the operator selects the ref set **after** a preliminary Tier-A
calibration RR (§2.5) to pick a cycle-minimal set; (c) the live aggregate is Copeland/mean-WR
vs the *fixed set* (a within-set cycle perturbs an individual pairwise term but not the
beat-count-vs-fixed-set, which is what gates).

Tiers A and B share the BT/aggregate math (`hexo_rl/eval/bradley_terry.py:21`).

---

## 1a. Strength instrument — checkpoint-relative round-robin (Tier A) + non-transitivity

**Premise RED-TEAM (dispatcher demand).** Is checkpoint-relative Elo a valid steer signal where 38%
of pairs invert (M-CYC; §D-FOUNDING s45k beats s112.5k 15–4)? Partly — three honest limits + fixes:

1. **A single Elo scalar is lossy.** → the primitive reports the **win-matrix + a non-transitivity
   index**, never Elo alone; the gating aggregate is cycle-robust (below), not raw pairwise-Elo.
2. **"Beats running-best" can be a cycle artifact.** → steer (Tier B) rates vs a FIXED set, and the
   abort rule is **cycle-aware** (does not fire on a high-non-transitivity reading).
3. **Self-play-relative ≠ external strength.** A closed RR cannot separate "genuinely stronger" from
   "better at the shared spread game" (§D-FOUNDING caveat 2; flat-external + internal-lean IS the
   degenerate-spiral fingerprint). → the robustness gate (1b) is a *separate, mandatory* axis;
   strength-RR is necessary, not sufficient. (Operator decision §6.5: add a non-compact external rung
   — NNUE — to the steer? Default: no, robustness gate covers the external axis; logged NNUE is a
   cross-check.)

**Non-transitivity index — pick: inversion-fraction + directed-3-cycle-density (both cheap):**
- Inversion fraction = step-ordered pairs where the later rung loses the h2h (`rr_driver.py:317`
  (untracked), 25/66=38%). O(n²).
- **Directed 3-cycle density** = #{i≻j≻k≻i triples}/C(n,3). O(n³), trivial for n≤30 (220 triples at
  n=12). **This is the scalar that gates "skill ladder vs non-transitive equilibrium"** and feeds the
  cycle-aware abort guard. Measured 0.073 on the banked ladder (M-CYC).
- (Logged, not gating) Kendall-τ(Copeland order, BT-Elo order) — divergence = the Elo scalar
  misrepresents the matrix.

**Robust aggregate — pick: Copeland + median-rank, alongside BT-Elo:**
- **Copeland** Cᵢ = Σⱼ 1{wᵢⱼ>wⱼᵢ} — opponents beaten head-to-head. O(n²), immune to sparse pairs,
  margin-blind.
- **BT-Elo** (`compute_ratings`) — magnitude + CI, but NEVER weighted by a CI that can be exactly
  zero (the anchor-CI WLS trap that produced the retracted +1.46 slope; §D-FOUNDING Review). Slope =
  OLS over non-anchor rungs + game-level bootstrap CI (already fixed, `rr_driver.py:258-313`
  (untracked)).
- **Median-rank** over Copeland — the cycle-robust ordering for the steer decision.

**Pre-registered steer/abort — the CONJUNCTION MATRIX (REVIEW MAJOR #7; the two axes are
ORTHOGONAL — §D-FOUNDING Phase 1b: on-dist FLAT-non-transitive, off-dist FELL):**

| decision | rule |
|---|---|
| WARN (strength) | Tier-B aggregate (mean-WR or Copeland vs fixed ref set) below `floor` for 1 eval |
| ABORT (strength) | aggregate below `floor` for `n_consec` (3) consecutive evals past `min_step` **AND** 3-cycle density < `cycle_density_max` *(cycle-suppression: a high-RPS reading is an equilibrium, not a fall — applies to the strength axis ONLY)* |
| ABORT (robustness) | robustness gate (1b) REJECTS for `n_consec_rob` consecutive evals past `min_step_rob` — **fires regardless of cycle density** (cycle-suppression NEVER applies to the robustness axis) |
| PROMOTE | strength aggregate > `promote_floor` **AND** robustness gate PASSES (both axes required) |

All thresholds are config knobs (no literals); `floor` + `cycle_density_max` are TBD until the §2.5
calibration RR (NOT guessed); operator confirms before any real run.

**Cycle-suppression error-cost note (REVIEW lens-3).** The §D-FOUNDING failure was *misdiagnosis*
(a non-halting warning misread for weeks), NOT a bad auto-abort — so a *missed* abort is the cheaper
error than a *false* abort that kills a recovering run (the 87.5k false-abort lesson, §175/L34). The
matrix biases toward not-firing on ambiguous (high-cycle) strength readings, which is the correct
asymmetry — *but only for the strength axis*; robustness rejection is unambiguous and never suppressed.

**Tier-A primitive output (per run):** `win_matrix.csv`, `ratings.csv` (BT-Elo+CI), `aggregate.json`
(Copeland, median-rank, inversion-fraction, 3-cycle-density, Kendall-τ, OLS-slope+bootstrap-CI), and
— critically (REVIEW + Phase-3) — **`per_game.jsonl` with the full move list + checkpoint_step +
the play command (sims/temp)** (current driver records only the GameRecord summary, omits moves —
`rr_driver.py:181` (untracked); closes the docstring-128/run-64 reproducibility gap).

---

## 1b. Robustness gate — adversarial / off-distribution (Objective A)

Mostly pre-built; promote to a tracked primitive. The deterministic off-window adversary
(`hexo_rl/eval/offwindow_probe.py:170 run_adversary_games`, deterministic post-`a7ba110`) measures
`off_window_forced_win_rate`: how often an adversary forces a win on a completing cell the
single-window model has no logit to block.

- **Instrument:** deterministic off-window forced-win-rate (primary). The opening-scatter Elo
  (`investigation/founding_2026-06-08/argmax_discriminator.py`, untracked) is the *orthogonal*
  off-distribution co-instrument (Objective-A *strength*, not exploitability) — optional co-run, NOT
  a substitute (its argmax thresholds cannot be borrowed).
- **Threshold (EXT-LINK basis, locked 2026-06-06 at `exploit_probe.py:124`):** DEFENDED ≤ 0.05;
  FORCEABLE ≥ 0.15 AND (exploit−control) ≥ 0.10. Fix-acceptance target ≤ **0.06**; deployed
  single-window v6_live2 ≈ **0.235** = the defect. Thresholds locked for **n=200/arm, sims=128,
  opening_plies=6** — the gate FIXES these.
- **Role:** WATCH / robustness signal, not a strength meter. By default it gates PROMOTE (matrix
  above) and emits WARN; it fires ABORT only if the operator arms it (§6.3). It is the *only*
  instrument that sees the off-window defect — vs-SealBot false-clears by construction
  (§PRELONG-BRIDGE). The gate MUST reject vs-bot WR as a proxy.
- **Encoding/checkpoint specificity:** the off-window boundary is `policy_logit_count` (362 for
  v6_live2). Results are encoding- and checkpoint-specific. The gate **hard-errors** on a `spec.name`
  mismatch (message: "re-run exploit_probe on this checkpoint"), with an opt-in `--force-spec-mismatch`
  override (REVIEW should-fix).
- **New module** `hexo_rl/eval/robustness_gate.py`: `RobustnessGateConfig` +
  `evaluate_robustness_gate()` → verdict + rate + CI + arm + checkpoint label. **Imports
  `run_adversary_games` from `offwindow_probe.py` (single source — no duplication).** Lives separate
  from `gate_logic.py` (that is the strength/promotion CI gate — orthogonal question).

---

## 1c. SealBot-WR — DEMOTE to logged diagnostic, demotion CONDITIONAL on cross-validation

SealBot-WR is a real Objective-A *canary* (the 0.38→0.07 drop was a true signal, §D-FOUNDING) but
invalid as a *strength* meter (oscillates non-monotonically across adjacent 5k rungs). Disposition:

- Keep measuring `wr_sealbot` (`opponent_runners.py:152`) + **log it** (structlog + dashboard).
- **Remove it from the abort decision.** Today `step_coordinator.py:979-1009` is the live abort site:
  `_wr_history` ring → `check_sealbot_wr_hard_abort` → `shutdown.running = False` (line 1008). After
  the rewire, that site reads the Tier-B strength aggregate + robustness gate (matrix §1a);
  `check_sealbot_wr_hard_abort` → logged-only (the status `check_value_spread_canary` already holds
  per L50, `alert_rules.py:210`). Function + tests stay; never wired to shutdown on a real run.
- **REVIEW MAJOR #8 — demotion is CONDITIONAL.** Before SealBot-WR leaves the abort path on a real
  run, Phase 3 must cross-validate that the robustness gate is a *superset* of the signal SealBot-WR
  correctly caught: run `exploit_probe` on banked ckpts {35,50,75,90,112.5k}, compare its margin to
  historical `wr_sealbot` at the same steps (banked arena `reports/eval/golong_vast_pull_20260608/
  results.db`), confirm exploit_probe sensitivity ≥ SealBot-WR in the post-75k flag region. If
  exploit_probe MISSES a case SealBot-WR caught, keep SealBot-WR as a secondary WARN until the gap is
  explained. (This is the RED-TEAM "lost-signal" check made concrete.)

---

## 1d. Serial-eval fix — cross-game batching (the perf foundation)

**Bug (confirmed M-TP):** `evaluator.py:195` serial `for i in range(n_games)`; each game = one
`MCTSTree`, `batch_size=8` leaves/forward (`evaluator.py:96-106`). 9,252 games/hr, GPU 53.3%.

**Fix:** interleave **N_CONCURRENT** games, each its own `MCTSTree`, single-threaded. Per MCTS step:
`select_leaves(8)` on every active tree → coalesce all leaves into ONE
`LocalInferenceEngine.infer_batch` (accepts `List[Board]`, board-ordered return,
`selfplay/inference.py:61`) → scatter back per-tree by **explicit game-index metadata** (never leaf
arrival order) → `expand_and_backup` each tree. Mirrors selfplay coalescing
(`inference_server.py:402`) but no threads. New `hexo_rl/eval/eval_batcher.py`
(`EvalBatcher`/`Evaluator._run_batched_games`); **keep `ModelPlayer` single-tree unchanged** (the
off-window probe + other callers depend on it). Pool memory = N_CONCURRENT × tree-pool — reported.

**The bit-identity question — three findings, two design decisions (DESIGN-phase premise
correction; REVIEW MAJORs #3/#4/#5):**

1. **Global-RNG concurrency hazard.** `evaluator.py:196-197` seeds the *global* `np.random`/`random`
   per game; interleaving N games races them (opening plies + temperature sampling, `evaluator.py:125`).
   Clean fix = **per-game RNG instances** (`random.Random` + `np.random.Generator`). This CHANGES the
   stream vs the old global-seed path → not byte-identical to *old serial transcripts*. You cannot
   both fix the hazard and keep old-transcript byte-identity. **Decision 1: take per-game RNG.**
2. **GPU FP is not batch-size-invariant — MEASURED (M-VAR), float32 does NOT fix it.** Batching
   changes the forward batch size → autocast + cuBLAS/cuDNN kernel/reduction-order selection → logit
   |Δ| 1.7e-2 (fp16) / 3.7e-3 (float32, ~4.7× smaller but **nonzero**). So byte-identity across batch
   sizes is unachievable even on float32. **But argmax was unaffected (0/32 flips)** → statistical
   equivalence is tight. The review's "try float32 → maybe byte-identical" hypothesis is *falsified*.
3. **Scatter-correctness cannot be proven by FP comparison** (REVIEW MAJOR #4). N=1-vs-itself doesn't
   exercise cross-game scatter; and cross-batch FP noise (finding 2) masks a byte diff anyway. The
   clean proof uses a **deterministic inference stub** (a fixed/hash-based policy+value per board,
   batch-invariant by construction) so the NN-FP axis is removed entirely.

**Pre-registered correctness invariants — explicit GATE HIERARCHY (replaces the dispatcher's flat
"bit-identical"; resolves the REVIEW gate-hierarchy disagreement):**

| gate | role | spec | pass criterion |
|---|---|---|---|
| **G1 orchestration (PRIMARY, the ONLY scatter proof)** | catches mis-scatter / seed / tree-step bugs | **deterministic stub** inference; N=100 games at N_CONCURRENT∈{1,2,4,8,16}, compare **move-by-move keyed on game-index** | byte-identical move sequences across all N; **0** mismatch. Also assert `select_leaves` return order identical across calls + dict/set iteration order pinned (torch determinism does NOT cover these — REVIEW) |
| **G2 NN argmax agreement** | bounds NN-FP decision impact | real model, GPU; ≥300 root decisions, N-batched vs N=1 | ≥99.5% root-move agreement (prior: M-VAR 100% on 32 leaf boards) |
| **G3 repeat-determinism** | catches batch-schedule non-determinism (the get_threats/65-of-4068 class) | same batched GPU run ×5 | **0** mismatch (prior: M-VAR 0/32) |
| **G4 statistical equivalence (diagnostic only)** | sanity, NOT a scatter proof (a symmetric scatter bug survives it) | batched vs serial WR on a fixed ≥200-game set | CI overlap |
| **G5 behavior-neutral reseed** | the per-game-RNG swap is not a silent behavior change | new-RNG serial vs old-global-RNG serial, fixed set | new-RNG CI **CONTAINS** old point estimate **OR** \|ΔWR\| < 2pp (TIGHTENED from "CI overlap" — REVIEW MAJOR #5, opening-driven swings reach 5–10pp); AND opening_plies + τ sample distributions logged + asserted statistically identical |

Old globally-seeded transcripts are **retained as the G4/G5 statistical baseline** (not re-executed;
used only for WR/Elo comparison) — REVIEW should-fix.

**Bench invariants:** the fix touches only `hexo_rl/eval/` — **off** every bench-gate auto-fire glob
(`engine/src/mcts/**`, `replay_buffer`, `game_runner`, `inference_bridge.rs`, NN hot paths). So:
- `make bench` MCTS floor **≥ 73k sim/s** held — a **no-regression sanity** check (the fix is not on
  the benched Rust MCTS path). 10/10 targets unchanged.
- **Real perf evidence = a NEW eval-throughput micro-bench.** Baseline **M-TP: 9,252 games/hr, GPU
  53.3%** (sims=100, n=8, 4060). Target: **≥ 1.5× = ≥ 13,878 games/hr AND GPU-util > 80%** at
  N_CONCURRENT∈{8,16,32}. Re-baseline on the production eval config (sims=128, n=200) in commit 3 —
  the absolute target scales from the prod baseline, recorded in the commit message.

---

## 2. IMPL plan (Phase 2 — built only after this DESIGN passes REVIEW; commit only when operator asks)

One clean commit per feature, conventional prefix, NO Co-Authored-By:

1. **`feat(eval): tracked round-robin primitive`** — promote `rr_driver` →
   **`hexo_rl/eval/round_robin.py` + thin `scripts/eval_round_robin.py` CLI** (REVIEW: pick the
   importable module, not a script-only file). Fix the 4-plane loader **properly** — REVIEW MAJOR #2
   corrected the fix-site: `checkpoint_loader.load_model_with_encoding` ALREADY resolves the spec
   by name (`detect_encoding_label:79` → `detect_encoding_from_state_dict` strict → registry); the
   `encoding="v6"` literal at `_build_min_max_model:217`/`_build_kata_model:259` only mis-stamps
   `model.encoding` (spec is correct upstream) — pass `spec.name` as a cleanup. **The real hardcode
   is `scripts/tournament_validate.py:59 _build_our_model_config` (in_channels=8, encoding='v6')** and
   any loader path that bypasses `load_model_with_encoding`; route the primitive through
   `load_model_with_encoding`, drop the hardcoded config. Validate **both** `inferred_in_channels ==
   spec.n_planes` AND `policy_logit == spec.policy_logit_count` (REVIEW should-fix). Log the play
   command; record full move lists + checkpoint_step; emit win-matrix + non-transitivity index +
   Copeland/median-rank. **Must reproduce §D-FOUNDING's numbers** (REVIEW re-derives one Elo point
   from raw; tolerance ±3 Elo for bootstrap-CI jitter).
2. **`feat(eval): robustness gate primitive`** — `hexo_rl/eval/robustness_gate.py` importing
   `run_adversary_games`; config-keyed; deterministic; encoding/checkpoint hard-error on mismatch.
3. **`fix(eval): cross-game batched evaluator`** — `eval_batcher.py` + per-game RNG; G1–G5 gate suite
   (incl. the deterministic-stub orchestration test) + the eval-throughput micro-bench with the
   recorded prod baseline.
4. **`refactor(monitoring): steer/abort on strength+robustness, demote SealBot-WR`** — Tier-B
   fixed-reference aggregate + the conjunction-matrix pure functions in `alert_rules.py`
   (`check_strength_regression_abort` cycle-aware; robustness-reject path); rewire
   `step_coordinator.py:979-1009`; `check_sealbot_wr_hard_abort` → logged-only; new `MonitoringConfig`
   knobs (`config.py:41`, added to the `@dataclass` so `from_dict` introspection picks them up).
   Promotion side (`eval_pipeline.py:311`) consumes the fixed-reference aggregate per the matrix.

`make test` + `make bench` per commit. Tests follow the existing pure-function + event-emission
pattern (`test_alert_rules_wr_hard_abort.py`).

## 2.5. CALIBRATION (committed gated sub-phase — REVIEW MAJOR #9; BLOCKS Phase 3 + any live run)

`feat(eval): calibrate strength-abort knobs` — a one-off Tier-A calibration RR on the banked ladder
to LOCK `strength_abort.floor` + `cycle_density_max` (both TBD in §5). **Pre-registered method:**
- `floor` = the Tier-B aggregate value s.t. ≥95% of the "healthy" 50k–75k rungs pass AND ≥80% of
  post-peak (90k+) rungs cluster below it (separation test, not a guess).
- `cycle_density_max` = max(0.15, 75th-pct of the ladder's rolling 3-cycle density) — anchored to the
  measured 0.073 (M-CYC) so the live guard suppresses an abort only on a markedly-more-tangled ladder
  (~2× observed).
- Also locks the fixed reference set (cycle-minimal, internal 3-cycle density logged; §0).

---

## 3. VALIDATE plan (Phase 3 — eval-only, banked local ladder; NO training; operator-gated compute)

Banked ladder local: `checkpoints/_archive_golong_kill_20260608T065342Z/` (32 ckpts 5k–112.5k +
replays). **Reframe (REVIEW should-fix):** §D-FOUNDING already *pilot-resolved* on-dist FLAT
(+0.13/1k, CI[-0.25,+0.55]) + off-dist FELL (−1.4..−1.7/1k); Phase 3 is the **at-power CONFIRMATION
re-run**, not open hypotheses. Heavy compute → operator-gated (laptop feasible-slow; vast 5080
preferred). Pre-register:

- **On-distribution plateau, at power.** Tier-A all-pairs RR, n sized to resolve ±50 Elo (4–15× the
  n=40 → **n ≈ 160–600/pair**), argmax + temp. Branches: (i) CI-resolved positive slope 50k→112.5k →
  continued strengthening; (ii) flat → genuine plateau; (iii) 3-cycle density above the §2.5
  threshold → non-transitive equilibrium (not a skill ladder).
- **Off-distribution FELL, at power + MECHANISM.** Re-run the temp×opening 2×2 at power; AND trace
  the off-distribution **losses** on recorded games (needs the move-recording primitive §2.1).
  **Pre-registered metrics (REVIEW MAJOR #11):**
  - *opening-distance* = Chebyshev `max(|q−qc|,|r−rc|)` of the post-opening centroid from board
    centre, bins **[0–2] / [3–5] / [6+]**.
  - *off-window-failure enrichment* = loser-side off-window forced-turn rate on the CORRECTED
    completing-cell unit (`pair[1]` landing stone, `forced_win_detector.py:176` — NOT depth-1; the
    §D-COHERENCE unit lesson), per mover-side, on **cross-model** games (window-centering is
    own-stone-dependent), prioritizing **>90k** games (75k risks a transient mode).
  - **Branch A (off-window-mediated):** loser off-window rate (post-90k) ≥ `0.40` with lower-CI >
    `0.35` AND inverse-correlated with opening distance → **multi-cluster encoding** lever (S0/S1/S3).
  - **Branch B (generic opening overfit):** loser spread collinear with opening distance, off-window
    rate < `0.15` → **opening-diversity** lever (far cheaper).
  - **Discriminator:** regress loser-spread ~ opening-distance + off-window-rate-bin (stratified by
    step bin — opening distance co-varies with step). Opening-distance coefficient survives
    controlling for off-window-rate → B present; vanishes → A dominates. **A and B can CO-OCCUR** —
    report both; do not force a single lever (gate: each branch's CI on its own metric).
- **SealBot-WR cross-validation** (§1c MAJOR #8) — exploit_probe vs historical wr_sealbot on the
  banked ckpts; demotion is conditional on this passing.

**These verdicts FORK the roadmap (operator-owned, NOT actioned here):** plateau-real → strength/
value-ceiling investigation (banked AUC_compact<0.70 candidate); off-window-mediated FELL →
multi-cluster encoding decision; opening-overfit FELL → opening-diversity. Report the fork; do not
pick it.

---

## 4. Out of scope

No multi-cluster Rust (S0), no Phase-4.5 features, no encoding change, no training run, no commit
without operator ask. **Phase-premise defense (RED-TEAM):** every downstream run is blocked on
(1)+(2); this concretizes the arc's lesson. FLAG plainly if the operator would rather gamble a run on
current infra (their call) — see §6.

---

## 5. Pre-registered knob defaults (operator confirms before any real run)

| knob | default | basis |
|---|---|---|
| `eval.n_concurrent_games` | 16 | GPU-util sweep {8,16,32}; pool mem = N×tree-pool |
| `strength_ref_set` | cycle-minimal subset chosen at §2.5 calibration | fixed gauge; internal 3-cycle logged |
| `strength_abort.floor` | **TBD — locked at §2.5** | separation method (§2.5) |
| `strength_abort.n_consec` | 3 | Wave3 consecutive rule (`alert_rules.py:145`) |
| `strength_abort.min_step` | 25000 | matches Wave3-B `wr_collapse_min_step` |
| `strength_abort.cycle_density_max` | **TBD — locked at §2.5** (≈ max(0.15, 75pct); observed 0.073) | M-CYC |
| `robustness_gate.threshold` | 0.06 | EXT-LINK fix-acceptance (`exploit_probe.py:124`) |
| `robustness_gate.{n_per_arm, sims, opening_plies}` | 200, 128, 6 | locked; thresholds tied to these |
| `robustness_gate.arms` | exploit, control | one-switch ablation |
| `eval_throughput.target_games_per_hr` | ≥ 1.5 × prod-baseline | baseline recorded commit 3 (M-TP 9252 @ sims100) |
| `phase3.rr_n_per_pair` | 160–600 | ±50 Elo resolution |
| `phase3.loser_offwindow_threshold` | 0.40 (lower-CI > 0.35) | Branch A gate |
| `phase3.branchB_offwindow_max` | 0.15 | Branch B gate |

**§2.5 calibration OUTCOME** (`hexo_rl/eval/strength_calibration.py`, run on the §D-FOUNDING rr_pull
data; IMPL Phase 2):
- `strength_cycle_density_max` = **0.15** LOCKED (max(0.15, p75); measured density 0.073, M-CYC).
- `strength_abort_floor` = **uncalibratable on this ladder** — the separation method returned None:
  per-rung normalized Copeland for "healthy" (≤peak) vs "post-peak" rungs OVERLAP (post-peak
  `[0.86,0.23,0.46,0.68,0.41]` vs healthy `[0.09…0.91]`) because on-distribution strength is FLAT +
  non-transitive (the founding verdict, Objective B ill-posed). Method working as designed (refuse,
  don't guess). FINDING: on a flat ladder there is no separable strength regression to gate, so the
  strength abort correctly stays DISABLED and the **robustness gate carries the Objective-A signal**.
  A floor is lockable only from a future run with a genuine separable collapse (needs both healthy +
  collapsed samples) — operator-gated at-power. This *confirms* the SealBot demotion: there is no
  on-distribution strength regression for any abort to catch; SealBot-WR was firing on a non-strength
  (off-distribution) signal.

---

## 6. Open decisions for the operator (surfaced at the DESIGN checkpoint)

1. **Bit-identity reframing** — accept the G1–G5 gate hierarchy (deterministic-stub orchestration
   proof + GPU argmax-agreement + statistical equivalence + tightened reseed) in place of literal
   "bit-identical to serial"? M-VAR proves the literal form unachievable even on float32.
2. **Two-tier instrument** — confirm Tier-B fixed-reference for live steer/abort vs a mini all-pairs
   RR each round?
3. **Robustness gate role** — default = gates PROMOTE + WARN; arm as a hard ABORT for a real run?
4. **Promotion-side rewire** — replace `wr_best` (running-best, anchor-reset-confounded) with the
   fixed-reference aggregate, or keep `wr_best` and only rewire abort?
5. **External steer rung** — add a non-compact external opponent (NNUE) to the live steer, or leave
   external strength to the robustness gate + a logged NNUE cross-check?

---

## 7. Real-run config recipe (MANDATORY before any live run — the lost-signal guard)

SealBot-WR is demoted from the abort path. Its replacements are CONFIG-GATED and **OFF by
default**, so a *default* run has NO Objective-A signal at all (SealBot demoted + robustness monitor
off). Before a live run the operator MUST:
1. **Enable the robustness gate** (the operative Objective-A signal): turn on the
   `offwindow_adversary` eval opponent (`opponent_runners` monitor, writes
   `offwindow_forced_win_rate`) so the rate flows to BOTH the promote gate
   (`decide_promotion`, blocks at `gating.robustness_threshold=0.06`) and the step-coordinator WARN.
   Without this, `robustness_rate` is None → "missing = pass" → the gate is inert.
2. **Strength abort:** stays disabled until a future run yields a separable collapse to calibrate
   `strength_abort_floor` (§2.5 finding). The cheap eval-based guard until then is the robustness
   gate + the existing grad-norm / stride-5 / draw-rate aborts.
3. **Promote-side decision 4** (strength aggregate replaces wr_best) activates only once the per-round
   ref-set producer + a configured `strength_ref_set` land (operator-gated follow-up); until then
   promotion uses wr_best (the strength fallback) AND the robustness gate.

Net default behavior is unchanged from before the rewire EXCEPT the SealBot-WR auto-abort is removed —
which is the intended fix. The recipe above re-arms the *correct* Objective-A guard.

## BUILD REVIEW + RED-TEAM disposition (post-IMPL, `wf_f42ce70d`, PASS_WITH_CHANGES)

Fresh-context 5-lens REVIEW+RED-TEAM verified every load-bearing claim (round-robin reproduces
§D-FOUNDING; loader fix wired; robustness single-sources; cycle-suppression asymmetry; SealBot
demoted; conjunction matrix; G1 scatter proof anti-trivial; calibration refuse-don't-guess; 54+ tests
pass; bench MCTS floor 88,989 ≥ 73k). Two lens BLOCKs (strength path inactive) were ADJUDICATED DOWN —
the not-done is honestly disclosed design (ref-set producer = operator-gated follow-up). Genuine
issues were hygiene/enforcement, now resolved:
- **MAJOR (lost-signal enforcement) → FIXED:** `check_objective_a_coverage` pre-flight WARNs loudly at
  run start when SealBot demoted + strength_abort + robustness_abort + offwindow monitor are ALL off
  (`step_coordinator` init); §7 recipe added.
- **MAJOR (missing single-eval strength WARN, spec §1a row 1) → FIXED:** `check_strength_warn` + wired.
- **should-fix → DONE:** `sealbot_wr_revert_to_abort` honesty knob (demotion is code-permanent but
  spec-conditional on Phase-3); G1 parametrized over N∈{8,16,32}; G5 reseed-equivalence DEFERRED to
  Phase-3 banked data (noted in test).
- **BLOCKER (spec untracked) → operator-gated:** committing this spec (and the whole build) is held per
  the dispatcher's "commit only when the operator asks." Surfaced to the operator as the commit decision.

## REVIEW disposition (v1 → v2)

| # | REVIEW MAJOR | disposition in v2 |
|---|---|---|
| 1 | rr_driver cited as tracked anchor | relabeled (untracked) throughout + anchor convention note |
| 2 | checkpoint_loader:217 mis-identified fix-site | corrected §2.1 → real hardcode is `tournament_validate:59`; 217 is a cosmetic mis-stamp |
| 3 | GPU variance asserted not measured; float32 untried | MEASURED (M-VAR); float32 falsified as a byte-identity route; claim refined |
| 4 | CPU primary gate may pass trivially | replaced by G1 deterministic-stub orchestration test (the only scatter proof) |
| 5 | reseed tolerance too loose | G5 tightened: CI contains old point est OR \|ΔWR\|<2pp + distribution assert |
| 6 | fixed-ref not vetted acyclic | §0 reworded "fixed ≠ acyclic"; ref-set 3-cycle logged; cycle-aware rule is the real mitigation |
| 7 | abort/promote conjunction implicit | explicit matrix §1a (robustness abort never cycle-suppressed) |
| 8 | robustness gate not cross-validated vs SealBot | §1c demotion CONDITIONAL on Phase-3 cross-validation |
| 9 | calibration not a committed step | §2.5 committed gated sub-phase + pre-registered method |
| 10 | 1.5× gate no baseline | M-TP baseline 9252 games/hr / GPU 53.3%; absolute target stated |
| 11 | Phase-3 metrics undefined | opening-distance + thresholds defined + pre-registered §3/§5 |

Disagreements adjudicated: ANCHOR-lens §1c/§1b "conflation" downgraded to no-op (v1 text already
separated current/after and stated the single-source dependency — synthesis confirmed). Gate-hierarchy
disagreement resolved by the explicit G1–G5 role table (G1 = sole scatter proof; G4 diagnostic-only).
