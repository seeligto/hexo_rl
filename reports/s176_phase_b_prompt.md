# §176 Phase B — Source A static corpus mixing + colony POC + dual-mode eval ladder
## Opus 4.7 x-high, fresh session

> **This file IS the Phase B launch prompt.** A future Opus session reads this and executes Phase B end-to-end. It is self-contained. It is committed at the pre-launch baseline as the pre-registered done-when artifact (per L13, Gate 1 §145 strengthening Note 5).

---

## Section 1 — Phase B charter

**Goal.** Land Source A static bot-game corpus mixing (S4 design → S4 implementation in a future §177) and the `n_components` colony POC metric (S3) on top of the §176 Phase A anchor `bootstrap_model_v6_step20k.pt`, with a **dual-temperature eval ladder** (S2 with L21 pin) that surfaces sampled-policy regression into the colony attractor — the mechanism L22 pins as the §175 degradation root cause.

**Anchor.** `checkpoints/bootstrap_model_v6_step20k.pt`, SHA256 `297e0ce0e48c8c9c417d923610de6ed0166c7295789ebc7bd029c90bb42bce6a`, weights-only (17 MB, model_state + metadata; optimizer/scaler/scheduler stripped per §34). Baseline anchor `expected_sealbot_winrate_pct_n100: 18.0%`, Wilson95 CI `[11.7%, 26.7%]` transferred via tensor-equality from §175 step_00020000 trainer-ckpt (143/143 keys byte-equal per Gate 3 report). Sidecar at `checkpoints/bootstrap_model_v6_step20k.json`.

**SHOULD-RE-BASELINE flag.** F08 STRENGTHEN deferred to **Wave A1** of this Phase B: re-run n=100 SealBot eval at T=0.5 on the promoted artifact to tighten Phase B forward-comparison CI (~1.5–2 hr vast 5080 wall). If skipped, document anchor transfer via tensor-equality in S1 commit message citing `reports/s176_gate3/smoke_eval.md`. **Recommendation: re-baseline.** Phase B compares Source A deltas against this CI; tighter CI = clearer signal.

**Forensics specimen.** `checkpoints/checkpoint_00070000.pt` (51 MB full trainer ckpt, mtime 2026-05-14 22:41). **Pathology signature:** V70K-4 FAIL strong — every latest_70K win is colony-spam, 100% col>0.3 rate, mean col-frac 63.3%, mean n_components 14.90 (right at the warning threshold). Greedy-mode 25/25 vs SealBot (parity); sampled-mode 4/100 vs SealBot (collapse). L22 pin: this checkpoint is the empirical witness for "sampled-policy distribution flattening into colony attractor." **Retention status:** RETAINED on disk; no sidecar yet; F07 STRENGTHEN flagged this as a hardening opportunity (defer to S6 close-out per Section 8 deferred list).

**Mechanism re-statement (Q11 carry-over).** §174 forward block (sprint log line 1202) framed the §175 degradation as "policy/value head + selfplay-interaction layer." L22 (sprint log line 1374) sharpens this to "**sampled-policy distribution flattening into colony attractor** under T=0.5 sampling, NOT loss of argmax dominance over bootstrap." Phase B S5 design doc (Source B) MUST cite L22 as the mechanism justification — opponent-coupling (V6 PASS, 41pp colony spread) is the empirical lever; sampled-distribution flattening is the failure mode the lever counter-acts. The two are different framings of the same mechanism and Phase B uses them together.

**Why Phase B is reframed from `s176_d_plan.md` (Wave D, pre-Gate-2).** Wave D wrote S4 Source A as the **primary** fix mechanism candidate. Gate 2 + L21/L22/L23 inverted this reading: the L22 mechanism (sampled-policy flattening) is best counter-acted by Source B's opponent-coupled cross-bot games (V6 PASS), not by Source A's static-corpus weighting. Phase B promotes **S5 Source B from "design-only, defer to §177+" to "design-only with explicit primary-fix-mechanism justification, implementation queued immediately after Phase B closes"**. S4 retains its slot but with lowered expected-benefit — it remains the unblocked first-shot at the §175 head drift, and an Elo-weighted static corpus is empirically defensible (Wave C ratings.csv ~500 Elo gap mandates per-source weighting per L1+L15) — but it is no longer the primary mechanism story.

---

## Section 2 — Read-first list

Mandatory reads, in order:

1. **This file** (`reports/s176_phase_b_prompt.md`) — top-to-bottom, fully.
2. `CLAUDE.md` — current phase, encoding registry pin, threat-probe gate.
3. `docs/07_PHASE4_SPRINT_LOG.md` §174 / §175 / §176 Phase A / Gate 1+2+3 close-out (lines 1180–1398), Falsified Hypotheses Register (lines 535–557 + Gate-2-added rows), Mechanism Lessons L18–L23.
4. `reports/s176_review_findings.md` — pre-Phase-B audit; F01–F11 with band classification.
5. `reports/s176_d_plan.md` — Wave D plan, **as updated by F02/F03/F09 in the pre-Phase-B fix wave** (L21 eval-temperature pin, S3 mode-scope qualifier, risk rows 8+9).
6. `reports/s176_gate1_operator_review.md` — operator Gate 1 STRENGTHEN_ONLY verdict, 7-dimension audit.
7. `reports/s176_gate2_tourney/summary.md` + `verdicts_v70k.md` — Gate 2 V70K-1..5 verdicts, methodology divergence note.
8. `reports/s176_gate3/smoke_eval.md` — Gate 3 PROMOTED verdict, byte-equality + audit-clean + n=20 smoke.
9. `reports/s176_a3_selfplay_forensics.md` — §175 attractor REFUTED-single-cluster verdict, n_components POC justification (Cohen's d −0.822).
10. `reports/s176_a4_falsified_scan.md` — 15-item do-not list, L1–L17 mechanism lessons.
11. `reports/s176_c_tourney/summary.md` — Wave C V1–V6 verdicts, BT ladder, opponent-coupled colony evidence (V6 PASS, 41pp spread).
12. `docs/06_OPEN_QUESTIONS.md` Q11 (colony-detection over-inclusion RESOLVED 2026-04-28), Q14 (KrakenBot integration, BT anchor question), Q15 (corpus tactical quality filtering, Source A weight interaction), Q-§176-residual (deferred micro-refactors).
13. `docs/02_roadmap.md` — Phase 4.5+ rows including §175 closed-by-interrupt + §176 Phase A CLOSED + §176 Phase B pending.
14. `docs/designs/encoding_registry_design.md` + `docs/designs/encoding_alpha_multiwindow_selfplay_design.md` — encoding contract, multi-window selfplay design (relevant for S4 encoding pin per Gate 1 §140 Note 2).

Optional reads if a specific sub-task surfaces context need:
- `reports/s176_a1_kraken_smoke.md` (KrakenBot submodule + INTEGRABLE-NOW verdict)
- `reports/s176_a2_eval_arch.md` (CACHING_CLEAN, minimal-diff plan)
- `reports/s176_b_smoke.md` (Wave B harness validation, 3 wrapper tests PASS)
- `reports/s176_e_review.md` (Wave E fresh-context audit)
- `docs/rules/bot-integration.md` (SealBot upstream colony-bug caveat per F09 Note 1)
- `docs/rules/checkpoint-archive-policy.md` (retention semantics for F07 deferred 70K specimen)

---

## Section 3 — Sub-task sequence S1–S6 (refined per Gate 2 + fix wave)

Six commits, ≤10 cap inherited from Wave D. All are cold paths — **zero bench gates**. Two INV pins touched (canonical-order + eval_temperature snapshot per F02 update). Sub-task ordering: S1 → (S2 ‖ S3) → (S4 ‖ S5) → S6.

### S1 — Wrapper promotion verification + Phase B anchor re-baseline

**Goal.** Confirm Phase A wrapper deliverables landed clean on the fresh Phase B branch and re-baseline the anchor at n=100 SealBot T=0.5 (F08 deferred).

**Files in scope.**
- READ-ONLY: `hexo_rl/bots/krakenbot_{bot,random,mcts}.py` (Phase A wrappers), `scripts/tournament_validate.py` (Phase A tourney harness), `tests/test_krakenbot_wrappers.py`.
- WRITE: `reports/s176_phase_b/s1_wrapper_audit.md` (NEW, ~80 LOC; checklist of Phase A wrapper invariants verified on fresh branch).
- WRITE: `reports/s176_phase_b/s1_anchor_rebaseline.md` (NEW, ~50 LOC; n=100 SealBot T=0.5 eval result on promoted artifact + Wilson95 vs original [11.7, 26.7]; or SKIP-WITH-RATIONALE if vast budget not available).

**LOC estimate.** ~130 LOC report + 0 LOC code.

**INV pin risk.** 0.

**Bench gate.** No.

**Commit boundary.** `docs(s176-phaseB): S1 wrapper audit + anchor re-baseline`.

**Empirical source.** Phase A commits 71c551a (wrappers + harness) + Gate 3 report (anchor extraction + audit). Wave C `summary.md` for wrapper stability (1050 games, 0 compound-turn bugs).

**Pre-registered verdict for re-baseline.**
- PASS: n=100 WR overlaps `[11.7, 26.7]` Wilson95 at α=0.10 → anchor preserved by extraction; proceed with original CI.
- WIDEN: n=100 WR falls outside original CI but within `[8.0, 32.0]` (looser α=0.05 expansion) → flag in S6, use new tighter CI for Phase B forward comparisons.
- HALT: n=100 WR < 8.0% point estimate → extraction may have introduced silent corruption beyond tensor-equality scope; investigate before proceeding to S2.

### S2 — Eval pipeline integration with L21 dual-temperature pin + Q14 close

**Goal.** Wire `kraken_minimax_strong` + `kraken_random` into eval ladder with **explicit dual-mode eval schema** per L21. Close Q14.

**Files in scope.**
- WRITE: `configs/eval.yaml` (+20 LOC, two opponent blocks; **`eval_temperature: 0.5` pin at top-level**).
- WRITE: `hexo_rl/eval/opponent_runners.py` (+40 LOC, two runner closures).
- WRITE: `hexo_rl/eval/eval_pipeline.py` (+5 LOC wiring; **expose dual-mode T=0.0 + T=0.5 eval option as a CLI flag**; default remains T=0.5 per L21).
- WRITE: `tests/test_eval_opponent_runners.py` (+8 LOC; canonical-order list + **new INV `test_eval_temperature_pinned` snapshotting `configs/eval.yaml::eval_temperature == 0.5`**).
- WRITE: `docs/06_OPEN_QUESTIONS.md:444-456` (Q14 RESOLVED 2026-?-? close text).

**LOC estimate.** ~73 LOC.

**INV pin risk.** 2 (canonical-order + eval_temperature snapshot).

**Bench gate.** No.

**Commit boundary.** `feat(eval): kraken_minimax_strong + kraken_random ladder + L21 dual-mode pin + Q14 close (§176 S2)`.

**Empirical source.** Phase A Wave A2 CACHING_CLEAN verdict + Wave C 1050-game stability. Gate 2 `verdicts_v70k.md:55-71` methodology divergence note (mode-inversion empirical witness).

**Schema fragment for dual-temperature eval (Phase B S2 ships this):**
```yaml
# configs/eval.yaml (top-level)
eval_temperature: 0.5  # L21 pin — DO NOT change without re-running ALL §175-era continuity comparisons

eval_dual_mode: false  # set true for promotion-gate evals where greedy + sampled both required
# When true, eval_pipeline runs n=100 at T=0.5 AND n=100 at T=0.0 against each opponent;
# both numbers + their delta are reported and gated independently per L21/L22.
```

**Pre-registered verdict for S2 dual-temperature gate (when `eval_dual_mode: true`):**
- **PROMOTION_READY:** Both greedy and sampled WR ≥ anchor lower-CI (11.7%) AND sampled-greedy divergence ≤ 30pp (the divergence ceiling per L23 — Gate 2 saw 50pp inversion at step 70K, which is the catastrophe signature).
- **FLAG_DIVERGENCE:** Either WR ≥ anchor lower-CI but sampled-greedy divergence > 30pp → checkpoint is sliding into the L22 attractor; flag at the trainer-side R&D log, do NOT promote.
- **REJECT:** Either WR < 8.0% lower point or both modes < anchor lower-CI → checkpoint regressed below extraction baseline; reject.

The 30pp divergence ceiling is **pre-registered before any Phase B sustained smoke runs to prevent post-hoc threshold shopping per A4 do-not #9** (L12 moving goalposts).

### S3 — Colony POC metric (`n_components`) at `pool.py game_complete` emit, mode-scope pinned

**Goal.** Land the Python-side `n_components` BFS metric at the selfplay-mode `pool.py game_complete` event with explicit mode-scope comment per F03.

**Files in scope.**
- WRITE: `hexo_rl/selfplay/pool.py` (+25 LOC; BFS helper inline at `game_complete` emit, mode-scope comment block, structlog event field addition).
- WRITE: `tests/test_pool_game_complete_schema.py` (+15 LOC; INV snapshot of `game_complete` event field set, including `n_components_winner`).

**LOC estimate.** ~40 LOC.

**INV pin risk.** 0 (new test; no existing pin on `game_complete` schema per A2 §f).

**Bench gate.** No (cold path, runs per-game at terminal state).

**Commit boundary.** `feat(selfplay): n_components colony POC at game_complete emit (§176 S3)`.

**Empirical source.** `reports/s176_a3_selfplay_forensics.md` §e — Cohen's d −0.822 between 20K and 50K cohorts (largest of 8 candidates). Reuses `hexo_rl/eval/colony_detection.py:31-52` `_connected_components` BFS.

**Pre-registered verdict for S3 G6 colony POC gate.** G6 is the **new mode-qualified** colony gate. The structure is:

| Sub-gate | Metric | Mode | Threshold | Action |
|---|---|---|---|---|
| G6a | `n_components` warning | selfplay (pool.py) | ≥ 15 | warning structlog, no abort |
| G6b | `n_components` divergence | selfplay vs greedy-eval-tourney | Δ ≥ 5 | flag for Phase B Wave C analysis |
| G6c | sampled-greedy WR divergence | dual-mode eval (S2) | Δ ≥ 30pp | flag, do NOT promote |

G6 is a **warning panel**, not a hard abort gate, per L13 + Q12 (no moving goalposts) + L22 (modes carry distinct semantics; one mode crossing a threshold is signal, not catastrophe). The 30pp G6c threshold is pre-registered now and freezes during Phase B execution.

### S4 — Source A static corpus mixing design doc (lowered expected-benefit framing)

**Goal.** Spec Source A static bot-game corpus mixing (sealbot 50 / our_v6_step20k 30 / kraken_minimax_strong 15 / kraken_random 5 of the 15% bot-pool; 75% selfplay; 10% human v7+) with **explicit acknowledgment of L21/L22 lowered expected-benefit**.

**Files in scope.**
- WRITE: `docs/designs/bot_game_mixing_design.md` (NEW, ~220 LOC markdown). Follow format of `docs/designs/encoding_alpha_multiwindow_selfplay_design.md` (Status, Problem statement, References, numbered sections).

**LOC estimate.** ~220 LOC markdown.

**INV pin risk.** 0.

**Bench gate.** No.

**Commit boundary.** `design(corpus): Source A static bot-game mixing design doc (§176 S4)`.

**Empirical source.** Wave C ratings.csv (BT ladder for per-source weights), L1 + L15 (corpus quality floor, v6 corpus retired for contamination). Gate 1 §140 Note 2: doc MUST pin Source A target encoding (avoid cross-encoding drift via persist.rs HEXB v7 header — match anchor's `v6` encoding).

**Lowered expected-benefit framing (Section 4 of the design doc).** The S4 design doc must state, in the "Why this is *not* the primary fix mechanism" section:

> Wave D framed Source A static corpus mixing as the empirically-defensible first-shot at §175 internal head drift. Gate 2 + L21/L22 refined the mechanism understanding: the §175 degradation is **sampled-policy distribution flattening into colony attractor**, witnessed by V70K-4 (100% col>0.3 wins, n_components 14.90) under greedy AND by the §175 eval-pipeline 18%→4% slide under sampled. Static corpus mixing addresses L1 (corpus quality floor) and L15 (v6 corpus contamination), but its mechanism is **weighted-distillation supply-side bias** — it pushes the policy gradient toward bot-game move distributions. It does NOT directly counter-act the sampled-distribution flattening mechanism, which is a self-play-interaction failure mode. Source A is therefore **expected to be necessary-but-not-sufficient**: it stabilizes the corpus quality floor, but the primary fix candidate is Source B (live cross-bot games, opponent-coupled colony rates per V6 PASS — see S5 design doc).

**S4 must include (numbered sections):**
1. Problem statement — citing L1 + L15.
2. Per-source weights table (sealbot 50 / our_v6 30 / kraken_strong 15 / kraken_random 5) per Wave C BT ladder.
3. **Encoding pin** (per Gate 1 §140 Note 2): Source A corpus generated for `v6` encoding only; persist.rs HEXB v7 header reject path for cross-encoding contamination.
4. Lowered expected-benefit framing (above).
5. Forward to Source B (S5) as primary fix mechanism candidate.
6. Validation plan: when Source A lands in §177+, gate is "anchor + Source A vs anchor alone" measured on dual-mode eval (S2) at n=100 per checkpoint, sustained over ≥ 30K steps from the §176 Phase A step-20K anchor.

### S5 — Source B live cross-bot games design doc (elevated to primary fix mechanism candidate)

**Goal.** Spec Source B live bot games + cross-bot G4/G5 protocol with **explicit primary-fix-mechanism justification** per L22.

**Files in scope.**
- WRITE: `docs/designs/bot_live_selfplay_design.md` (NEW, ~280 LOC markdown).

**LOC estimate.** ~280 LOC markdown.

**INV pin risk.** 0.

**Bench gate.** No.

**Commit boundary.** `design(corpus): Source B live cross-bot games + opponent-coupled selfplay design doc (§176 S5)`.

**Empirical source.** Wave C `summary.md:90-96` V6 PASS (sealbot colony rate ranges 0.000 against argmax → 0.412 against kraken_strong — 41pp opponent-coupled spread). L22 (sampled-policy flattening mechanism). A4 do-not #1 (sprint log line 597) — subprocess isolation MANDATORY, §17 c9f39de revert is the empirical witness for 3.3× GIL regression.

**Primary fix mechanism framing (must lead Section 4 of the design doc).** The S5 doc opens its mechanism justification with:

> §175 sustained at v6 + step-20K-onwards exhibits sampled-policy distribution flattening into a colony attractor (L22). The argmax mode of the policy retains bootstrap-like competence (V70K-3 PASS strong: latest_70K dominates own bootstrap 50-0); the sampled mode (T=0.5) collapses (§175 eval 18%→4% across 20K–70K). Wave C V6 PASS demonstrates that **opponent identity changes terminal colony rates by 41pp on the same self-bot** — the colony attractor is opponent-coupled, not bot-intrinsic. Cross-bot games (sealbot-vs-kraken, kraken-vs-our_v6, etc.) inject opponent-induced state-distribution variance directly into the self-play trajectory pool, which is the mechanism the L22 attractor cannot recruit against. Source B is therefore the **primary fix candidate** for the §175 mechanism, with Source A (S4) playing the supply-side stabilization role.

**S5 must include (numbered sections):**
1. Problem statement — citing L22 + V6 PASS opponent-coupling.
2. Architecture — subprocess isolation (per A4 do-not #1 + sprint log line 597), `scripts/tournament_validate.py` (Wave B harness) as the launcher template; one fresh bot per game.
3. Cross-bot protocol — G4 (kraken-vs-sealbot games) + G5 (our_v6-vs-kraken / our_v6-vs-sealbot, used as gradient signal not just eval).
4. Inference-server routing — open question: do cross-bot games share the engine's InferenceServer or run their own? Pin to design-level question; defer to implementation.
5. Corpus-ingest schema — how cross-bot game records enter the replay buffer; per-source weighting (deferred; Source A's table is the starting point).
6. Validation plan: when Source B lands in §177+, gate is dual-mode eval (S2) sampled-greedy divergence < 30pp ceiling (G6c), measured at n=100 per checkpoint, sustained over ≥ 30K steps from anchor. Compare against (Source A alone) and (anchor alone) baselines.

**§17 GIL regression — explicit do-not block.** S5 design doc Section 2 (Architecture) must include:

> S5 MUST NOT propose an in-process Python daemon for opponent mixing. §17 (sprint log line 597, commit c9f39de) showed 1.52M → 464K pos/hr regression (3.3×) when SealBot was integrated as in-process daemon. Subprocess isolation via `scripts/tournament_validate.py`-style single-game-at-a-time process spawning is the validated pattern (Wave B harness, 1050 games / 5111s wall on laptop). If Phase B reviewers find any in-process daemon proposal in this doc, HALT.

### S6 — Sprint log §176 Phase B close-out + Q11 re-statement + deferred-list reconciliation

**Goal.** Close §176 Phase B with sprint log entry, Q11 re-statement landed, deferred STRENGTHEN items reconciled.

**Files in scope.**
- WRITE: `docs/07_PHASE4_SPRINT_LOG.md` (append §176 Phase B close-out block, ~150 LOC).
- WRITE: `docs/06_OPEN_QUESTIONS.md` Q11 re-statement (cite L22 sharpening; ~10 LOC; Q11's RESOLVED 2026-04-28 status preserved, the re-statement is a forward-looking note on how Phase B sustained smokes should report colony detection).
- WRITE: `docs/02_roadmap.md` §176 Phase B row → CLOSED.
- WRITE: `CLAUDE.md:20` Current phase → §177+ active phase pin.

**LOC estimate.** ~180 LOC markdown.

**INV pin risk.** 0.

**Bench gate.** No.

**Commit boundary.** `docs(sprint): §176 Phase B close-out + Q11 re-statement + L24+ if any (§176 S6)`.

**S6 must reconcile (deferred-from-pre-Phase-B list, per Section 8 below):**
- F07 — promote `checkpoint_00070000.pt` as `checkpoints/known_pathological_v6_step70k.pt` weights-only with sidecar JSON, OR add explicit retention note in `docs/rules/checkpoint-archive-policy.md`.
- F10 — acknowledged-only; no action needed.
- F11 — operator preference; defer.

**Empirical source.** All Phase B implementation outputs (S1–S5 reports + design docs).

---

## Section 4 — Constraints (do-not list)

Inherits from Phase A master prompt + Gate 2 / review-derived items. **HALT if any violated.**

1. **No in-process Python daemon for opponent mixing or live cross-bot games.** §17 GIL regression (3.3× pos/hr drop, sprint log line 597, c9f39de revert). Subprocess isolation MANDATORY per A4 do-not #1.
2. **No cosine-temp / Dirichlet / opening_plies / playout_cap / random_opening_plies stacked knob changes during Phase B.** Phase B is corpus + eval-ladder + design doc work; selfplay knobs stay frozen at Phase A anchor settings. L9 cosine-temp + jitter pairing preserved (any selfplay touchpoint that introduces cosine-temp also introduces matched radius jitter; this is N/A for Phase B since no selfplay-knob edits).
3. **Every eval is dual-mode by S2 convention.** Per L21, all sustained-smoke promotion gates evaluate at T=0.5 (default) AND T=0.0 (greedy). Single-mode eval results are interpreted as mode-qualified, not absolute, in any Phase B report or sprint log entry.
4. **No 50/0 result is interpreted as skill signal without a temperature variant.** Per L23, greedy-mode H2H tourneys yield 2 effective games × 25 P1/P2 repetitions; Wilson95 over-confident. Any Phase B H2H comparison MUST include sampled-mode at T=0.5 or T=1.0 as a second axis.
5. **No edits to `engine/src/`, `hexo_rl/eval/eval_pipeline.py` core paths (only the S2 surgical additions), or any MCTS / replay buffer / inference-bridge code.** Phase B is cold-path-only. If S5 design doc surfaces a hot-path change, that change is deferred to §177+, NOT executed in Phase B.
6. **No bench gate skipping for any hot-path edit (N/A — none planned).** If a hot-path edit becomes necessary mid-phase, invoke `make bench` pre-and-post per `docs/refactor-template.md`.
7. **No `bootstrap_model.pt` symlink change (Gate 3 invariant).** Phase A preserves the pre-§176 symlink for §175-era reproducibility; Phase B preserves it identically.
8. **No `bootstrap_model_v6_step20k.pt` weights modification (anchor pin).** The 17 MB weights-only artifact at SHA `297e0ce0…2bce6a` is the Phase B anchor; do NOT re-promote, re-extract, or re-train off it. Phase B compares deltas; the anchor itself is read-only.
9. **No retraining off `checkpoint_00070000.pt` forensics specimen.** This specimen is the empirical witness for L22; any retraining destroys the pathology trail. Read-only.
10. **No re-litigation of Falsified Hypotheses Register rows touching Phase B's scope.** Enumerate at Section 5 risk register. Phase B sub-tasks must not propose:
    - §17 in-process opponent daemon (subprocess only).
    - v6 corpus baseline contamination (Source A uses v7+ Elo-weighted human + sealbot/our_v6/kraken bot games, NOT pre-§148 v6 corpus).
    - e30/e50/transfer-FT bootstrap recipes (§174 falsified all three).
    - kraken-MinimaxBot @ 1.0s > @ 0.1s strength claim (V2 side-finding, Δ −5 Elo).
    - our_v6 bootstrap strictly between Random and weakest Kraken (V5 FAIL).
    - §175 latest_70K regressed past own bootstrap on selfplay axis under greedy (V70K-3 PASS strong; L22 sharpens to sampled mode only).
11. **No moving goalposts on G6 colony POC threshold during Phase B execution.** The 15-component selfplay-mode warning (G6a) and 30pp dual-mode divergence (G6c) are pre-registered at this prompt's commit SHA and freeze for the duration of Phase B. Any threshold revision waits for §177+ fresh-context review per L13 + A4 do-not #9.
12. **No "one-liner" framing for any Phase B sub-task.** L17 magnitude floor: minimum sub-task surface is ~50 LOC including tests. Wave D plan author's existing S1–S5 LOC budgets (~990 LOC total) are the realistic estimate; "trivial fix" framing has been falsified repeatedly (§122 rotation, §163 sweep_harness, §172 plumbing).
13. **No torch.compile mode change for S2 eval-pipeline wiring.** `feedback_torch_compile_threading.md` pins per-thread CUDA-graph TLS; eval threads stay at `mode="default"` per existing convention. Phase B inherits §116 master compile state, no edits.
14. **No corpus regeneration off `bootstrap_model.pt` symlink.** Source A static corpus design (S4) generates new bot-game records; it does NOT regenerate the existing `corpus/` artifacts.
15. **No bot-integration LOC compression below empirical baseline.** Per L17 + Risk row 7, bot integration is ~65 LOC minimum (S2 line item) NOT "one-line config change." Reject any sub-task that proposes such compression.

---

## Section 5 — Risk register

Inherits Phase A 9-row register (7 from Wave D + F09 N1+N3 additions) + Phase B additions.

| # | Risk | Likelihood | Mitigation | Source / Falsified-or-Lesson |
|---|---|---|---|---|
| 1 | KrakenBot MinimaxBot sentinel-fallback fires ~0.42/game distorting strength | OBSERVED | Smart-neighbour-2 fallback preserved in `hexo_rl/bots/krakenbot_bot.py`; document caveat in S2 eval-ladder report | Wave A1 §e; Wave C summary.md:118 |
| 2 | Source A bot-game mixing degrades corpus quality if Kraken games weighted naively | HIGH if uniform | S4 design doc Elo-derived per-source weights table | L1 + L15 |
| 3 | KrakenBot MCTSBot weights never arrive — D2/D3 original hypothesis stays BLOCKED | MEDIUM | S5 design doc treats MCTSBot integration as §177+ scope conditional on weights arrival | A1 §b/g |
| 4 | Colony POC threshold becomes moving goalpost if §177+ sustained underperforms | HIGH if not pre-registered | G6a warning-only + G6c 30pp ceiling pre-registered at this prompt's commit | A4 do-not #9; L12 |
| 5 | Source B live bot-mixing re-introduces §17 GIL regression | CRITICAL if attempted in-process | S5 design Section 2 mandates subprocess isolation; in-process daemon proposals trigger HALT | §17 c9f39de revert; sprint log line 597 |
| 6 | Operator's "one large diffuse cluster" intuition drives §177+ work despite A3 REFUTED | MEDIUM | A3 forensics REFUTED single-cluster (6.3% non-orphan at 50K); §176 Phase B cites multi-island fragmentation as attractor | A3 §d, §verdict |
| 7 | Wave D plan author mis-allocates bot-integration LOC as "one-liner" | HIGH per L17 | S2 budget explicit ~68 LOC; constraint #12 + #15 reject "trivial" framing | L17; sprint log line 641 |
| 8 | SealBot upstream colony-detection bug skews V3 framing for Source A weighting | MEDIUM | S4 design doc cites bot-integration.md:34 caveat; treat sealbot 35.0% colony rate as noisy baseline | F09 N1; bot-integration.md:34 |
| 9 | Vendor submodule SHA drift on `git submodule update --remote` silently breaks Wave A1 INTEGRABLE-NOW | LOW but silent | Documented pin `vendor/bots/krakenbot @ d9c5bfb`; do NOT --remote without re-running smoke | F09 N3; Wave A1 §a |
| 10 | S5 Source B promoted to "primary fix mechanism" creates §177+ implementation pressure that re-attempts in-process daemon for speed | MEDIUM | Constraint #1 hardened; Section 4 of S5 design doc cites §17 as do-not block in-text | L22 + §17 c9f39de revert |
| 11 | S2 dual-temperature eval doubles eval wall-time (~2× per checkpoint) — Phase B sustained smokes blow budget | HIGH at full n=100 | S2 schema's `eval_dual_mode: false` default; promotion gates only set it true (sparse cadence); R&D rounds stay single-mode T=0.5 | L21 mode-divergence cost; Gate 2 6551s tourney wall on n=50/pair |
| 12 | Phase B re-baseline at S1 (n=100 on promoted artifact) returns < 8.0% — extraction had silent corruption beyond tensor-equality scope | LOW | S1 pre-registered HALT rule (Section 3 verdict block) | Gate 3 smoke n=20 gave 5%; binomial expectation supports overlap, but n=20 binomial floor is wide |
| 13 | Phase B Wave C sustained eval contradicts S4/S5 design assumptions — implementation should NOT proceed on stale design | MEDIUM | Section 9 escape hatch: halt + re-plan if empirical contradiction emerges; pre-registered verdicts NULL cleanly, no retroactive softening | A4 §e #15 fresh-context review mandate |
| 14 | F08 SHOULD-RE-BASELINE skipped to save vast budget, then later forward-comparisons under-power the anchor signal | MEDIUM | S1 has SKIP-WITH-RATIONALE path; if skipped, all Phase B forward comparisons explicitly cite original [11.7, 26.7] CI not a tighter post-promotion CI | F08 STRENGTHEN |

---

## Section 6 — Pre-registered Phase B verdicts

Per L13 + Gate 2 precedent (`reports/s176_gate2_verdicts.txt`), Phase B's exit criteria commit up-front. **These verdicts are pre-registered at this prompt's commit SHA.** Phase B execution may not retroactively soften them; verdicts NULL cleanly per L13 if empirical evidence contradicts.

### V-PhaseB-1 — Eval ladder dual-mode integration delivered
**Hypothesis.** S2 lands `kraken_minimax_strong` + `kraken_random` in `configs/eval.yaml` with `eval_temperature=0.5` pin AND a `eval_dual_mode` schema field that, when true, runs T=0.5 + T=0.0 eval against each opponent with both numbers reported.
**Pass:** `tests/test_eval_opponent_runners.py::test_opponents_canonical_order` + `::test_eval_temperature_pinned` both green; Q14 RESOLVED close-text landed.
**Fail action:** halt S3 until S2 dual-mode schema is validated.

### V-PhaseB-2 — Colony POC `n_components` emitted at selfplay-mode game_complete
**Hypothesis.** S3 lands BFS-based `n_components_winner` in `pool.py game_complete` structlog event, with mode-scope comment + INV snapshot.
**Pass:** new test `tests/test_pool_game_complete_schema.py::test_n_components_field_present` green; manual visual on one 30-game smoke shows `n_components_winner` values in `1..25` plausible range.
**Fail action:** halt S4 until S3 emit is validated.

### V-PhaseB-3 — Source A static corpus mixing design doc shipped with lowered expected-benefit framing
**Hypothesis.** `docs/designs/bot_game_mixing_design.md` exists, cites L1/L15/L22, has the lowered-expected-benefit Section 4 verbatim, has per-source weights table matching `s176_d_plan.md` Section 3, has encoding pin per Gate 1 §140 Note 2.
**Pass:** doc landed; design doc audit at S6 confirms all five required Section-4-Section-6 inclusions.
**Fail action:** revise design doc; do NOT proceed to S6 close until S4 passes.

### V-PhaseB-4 — Source B live cross-bot design doc shipped with elevated primary-mechanism framing
**Hypothesis.** `docs/designs/bot_live_selfplay_design.md` exists, cites L22 + V6 PASS as primary-mechanism justification, has explicit §17 GIL do-not block in Section 2 (Architecture), has dual-mode G6c gate (sampled-greedy divergence ≤ 30pp) in Section 6 validation plan.
**Pass:** doc landed; design doc audit at S6 confirms all six required Section-1-Section-6 inclusions.
**Fail action:** revise design doc; do NOT proceed to S6 close until S5 passes.

### V-PhaseB-5 — Phase B end-state SealBot WR at anchor checkpoint, dual-mode (Wave C measurement)
**Hypothesis.** Phase B does NOT run a sustained-training smoke (S4/S5 are design-only; corpus implementation lives in §177+). The anchor `bootstrap_model_v6_step20k.pt` is evaluated in dual mode in S1's re-baseline (or skipped per S1 SKIP-WITH-RATIONALE).

For S1 re-baseline (if executed):
- **Pass:** sampled-mode T=0.5 n=100 WR ∈ [11.7%, 26.7%] (overlap with §175 anchor CI).
- **Pass-with-tightening:** WR ∈ [8.0%, 32.0%] but outside [11.7%, 26.7%] → use new tighter CI for Phase B forward comparisons; flag in S6.
- **Halt:** WR < 8.0% point → extraction silent-corruption hypothesis; investigate before continuing.

For S1 SKIP-WITH-RATIONALE: anchor CI [11.7, 26.7] preserved by tensor-equality; Phase B forward comparisons cite this CI explicitly in any S4/S5 design doc validation-plan Section 6.

### V-PhaseB-6 — Colony fraction ceiling pre-registered
**Hypothesis.** Phase B does not yet measure colony fraction on a sustained run (that happens in §177+). The pre-registration is the **ceiling for §177+ promotion gates**: any §177+ checkpoint promoted off the Phase B anchor must show colony_fraction (winner-side, sampled mode) ≤ 50% on a 100-game sample.
**Pass:** ceiling landed in S5 design doc Section 6 (validation plan).
**Fail action:** revise S5 design until validation plan includes this ceiling.

### V-PhaseB-7 — Sampled-greedy divergence ceiling pre-registered (G6c)
**Hypothesis.** Phase B does not yet measure on a sustained run. The pre-registration is **G6c (S3 sub-gate table): sampled-greedy WR divergence ≤ 30pp** for any §177+ promotion checkpoint.
**Pass:** ceiling landed in S2 dual-mode schema text + S5 design doc Section 6.
**Fail action:** revise until G6c is text-pinned at two locations (configs/eval.yaml comment + design doc).

### V-PhaseB-8 — BT delta vs §175 step-20K anchor reported in Phase B close
**Hypothesis.** Phase B does not run a tourney (no checkpoint to measure against; the anchor is what's being preserved). The pre-registration is **§177+ first-promotion-gate must report BT delta vs `bootstrap_model_v6_step20k.pt` baseline at n ≥ 50 dual-mode**.
**Pass:** S6 close-out forward-pointer to §177+ includes this requirement verbatim.
**Fail action:** S6 close-out revised.

### V-PhaseB-9 — Q11 mechanism re-statement landed
**Hypothesis.** S6 lands Q11 re-statement in `docs/06_OPEN_QUESTIONS.md` citing L22 sharpening (sampled-policy flattening into colony attractor, not argmax-mode regression). Q11's RESOLVED 2026-04-28 status preserved.
**Pass:** Q11 body has both the original RESOLVED note AND the L22 forward-looking re-statement note.
**Fail action:** S6 revised.

---

## Section 7 — Wave structure for Phase B execution

Mirrors Phase A's parallel-subagent + sequential-implementation structure, adapted for Phase B's design-doc-heavy scope.

### Wave A — parallel investigation subagents (READ-ONLY + planning)

Four parallel subagent dispatches:

- **Wave A1 — wrapper audit + anchor re-baseline** (subagent_type: general-purpose). Read-only audit of Phase A wrapper deliverables on the fresh branch + execute n=100 SealBot eval at T=0.5 on the promoted artifact (if vast budget approved). Output: `reports/s176_phase_b/s1_wrapper_audit.md` + `s1_anchor_rebaseline.md`.
- **Wave A2 — eval-schema design** (subagent_type: Plan). Design the dual-mode eval schema (eval_temperature pin + eval_dual_mode field + dual-mode runner closures). Output: design fragment with schema YAML + Python closure sketches for S2.
- **Wave A3 — colony POC metric design** (subagent_type: Plan). Read A3 forensics report + colony_detection.py + pool.py emit site; produce the BFS-helper + game_complete-event-schema delta for S3. Output: design fragment with code sketches.
- **Wave A4 — S4 + S5 design-doc skeletons in parallel** (two subagent dispatches; subagent_type: general-purpose each). Skeletons with Section 1–6 headers + bullet-point content per Section 3 above. Outputs: `docs/designs/bot_game_mixing_design.md` skeleton + `docs/designs/bot_live_selfplay_design.md` skeleton.

Wave A wall budget: ~1.5–2 hr (vast re-baseline dominates if executed).

### Wave B — implementation per Wave A outputs

Sequential single-session execution:
- **B1.** S1 — commit wrapper-audit + re-baseline reports (1 commit).
- **B2.** S2 — wire kraken bots, dual-mode schema, eval_temperature pin, INV snapshot, Q14 close (1 commit).
- **B3.** S3 — `pool.py` n_components emit + mode-scope comment + INV snapshot (1 commit).
- **B4.** S4 — flesh out bot_game_mixing_design.md from Wave A4 skeleton (1 commit).
- **B5.** S5 — flesh out bot_live_selfplay_design.md from Wave A4 skeleton (1 commit).

Wave B wall budget: ~3–4 hr.

### Wave C — empirical validation

S1 re-baseline result (if executed) is the only empirical measurement in Phase B. S4 + S5 are design-only; their validation lives in §177+ sustained smokes. Phase B does NOT run a sustained training smoke.

If S1 re-baseline returned PASS or PASS-with-tightening: proceed to Wave D.
If S1 re-baseline returned HALT: pause Phase B execution; investigate extraction; do NOT proceed to S6 without operator confirmation.

Wave C wall: NIL (validation work absorbed into S1 of Wave B).

### Wave D — synthesis + Phase B close

- **D1.** S6 — sprint log §176 Phase B close-out + Q11 re-statement + CLAUDE.md + roadmap + deferred-list reconciliation (1 commit).

Wave D wall budget: ~1 hr.

### Wave E — fresh-context audit

Dispatch one subagent (subagent_type: general-purpose, fresh-context) to audit Phase B end-to-end against this prompt's Section 6 pre-registered verdicts. Verdict format: per-V-PhaseB row PASS/FAIL/NULL with cited evidence.

Wave E wall budget: ~1 hr.

**Total Phase B wall budget:** ~6–9 hr depending on S1 re-baseline inclusion.

---

## Section 8 — Deferred from pre-Phase-B

STRENGTHEN items absorbed in the pre-Phase-B fix wave commits (1d5b6b5..2014669):
- **F06** — L18 forward pointer to L22 + statistical disclaimer — absorbed in F01 commit (1d5b6b5).
- **F09 N1+N3** — risk register rows 8+9 — absorbed in 2014669.

STRENGTHEN items NOT absorbed (deferred to Phase B S6 close-out):
- **F07** — 70K forensics specimen retention metadata. **Action in S6 close-out:** either promote `checkpoint_00070000.pt` as `checkpoints/known_pathological_v6_step70k.pt` weights-only with sidecar JSON citing V70K-4 + Gate 2 + L22, OR add explicit retention note in `docs/rules/checkpoint-archive-policy.md`. Operator preference between option (a) and option (b) decides at S6 time.
- **F08** — Phase B anchor n=100 re-baseline. **Action in S1:** execute at start of Phase B Wave A1 (Recommended), OR document SKIP-WITH-RATIONALE in S1 commit message citing Gate 3 tensor-equality (Acceptable). See Section 3 S1 + Section 5 risk row 14.
- **F09 N2** — S4 design doc must pin Source A target encoding (`v6`, avoid HEXB v7 cross-encoding drift). **Action in S4:** included in Section 3 S4's "must include numbered sections" item 3 (Encoding pin). Will be naturally satisfied during S4 design-doc authorship.
- **F09 N5** — §176 Phase B pre-registration as committed artifact. **This file IS that artifact.** Once committed (commit message `docs(s176): Phase B prompt artifact pre-launch baseline`), F09 N5 is closed.

MINOR items deferred (no required action):
- **F10** — audit prompt Q11 reference was a question-number typo. F06 covered the substantive issue. No Phase B action.
- **F11** — Gate 2 directory filename consistency (`verdicts_v70k.md` vs `verdicts.txt`). Operator preference; defer.

---

## Section 9 — On uncertainty

Standard escape hatch (mandatory):

If Phase B Wave C (S1 re-baseline) empirical result contradicts S4/S5 design assumptions — for example, if the re-baseline returns < 8.0% triggering S1 HALT, OR if it returns a much wider CI than expected suggesting the extraction was lossy — **halt and re-plan rather than fabricate**.

Pre-registered verdicts (Section 6, V-PhaseB-1..9) **must NULL cleanly, not be retroactively softened**. NULL means: the verdict's empirical basis is invalidated; mark the verdict as `NULL — basis invalidated by [evidence]`; produce a separate report explaining what changed; do NOT retroactively edit the threshold to make the verdict PASS.

This is the L13 + Gate 2 + A4 do-not #9 + A4 do-not #15 combined discipline. The pre-registration commits BEFORE the work, freezes during the work, and either PASS / FAIL / NULL at end with cited evidence — never silently softens.

If a sub-task scope creeps mid-execution (S2 demands a hot-path edit, S4 design doc needs a new corpus regen, etc.), HALT and surface to operator. Do NOT carry scope creep silently per L17 + constraint #12.

If a falsified hypothesis surfaces being re-proposed inside Phase B sub-tasks (e30 bootstrap re-attempt, v6 corpus re-use, in-process daemon for live bots, etc.), HALT. Do not proceed even if the proposing rationale "looks defensible" — Phase A's Falsified Register and A4 do-not list are pre-registered specifically to prevent this.

---

## Done-when

- All five Phase B sub-tasks (S1–S6) committed (6 commits ≤10 cap inherited).
- Pre-registered verdicts V-PhaseB-1..9 all PASS or cleanly NULL with cited evidence.
- Wave E fresh-context audit verdict CLEAR or STRENGTHEN_ONLY.
- Sprint log §176 Phase B close-out landed (S6 commit).
- `CLAUDE.md` Current phase pin updated to §177+ active phase (S6 commit).
- `docs/02_roadmap.md` §176 Phase B row → CLOSED, §177+ row added (S6 commit).
- `make test.py` passes (no regressions from S2/S3 INV pins).
- No uncommitted changes; branch ready for operator merge or PR.

---

## Output

Final summary to terminal at end of Phase B:
- Per-S commit SHAs + LOC delta.
- Pre-registered verdicts V-PhaseB-1..9 status table.
- Wave E audit verdict + dimensions.
- F07 / F08 / F09 deferred-list reconciliation (option chosen).
- Aggregate: ready for §177+ Source A implementation / blocked-on-X.

---

## Provenance

This prompt is committed at the §176 Phase A pre-Phase-B fix wave close (commit SHA pinned at landing time of this artifact). The fix wave preceded this artifact and landed F01–F05 SHOULD-FIX + F06 + F09 N1+N3+N4 STRENGTHEN absorptions in commits `1d5b6b5..2014669` (six commits). F07 + F08 + F09 N2+N5 + F10 + F11 deferred to Phase B per Section 8.

The artifact pre-registers Phase B's S1–S6 sequence, verdicts V-PhaseB-1..9, risk register 14 rows, constraints 15 items, and wave structure A–E. A future Opus session reads this file top-to-bottom and executes Phase B without further upstream-prompt input.
