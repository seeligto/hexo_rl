# Wave A4 Falsified Register & Mechanism Lessons Scan (§176 Phase A)

Enumeration of empirically falsified hypotheses, locked mechanisms, regressions, and mechanism lessons bearing on §176 KrakenBot eval ladder scope. 

**Scope:** Bot mixing / opponent integration, corpus contamination (Source A vs B), colony detection, distribution drift, cosine-temp + jitter load-bearing, subagent prompt discipline.

**Report date:** 2026-05-14
**Report reference:** `docs/07_PHASE4_SPRINT_LOG.md` lines 535–641

---

## (a) Falsified Hypotheses Table

Rows from Falsified Hypotheses Register (sprint log line 539) bearing on §176 scope:

| § | Hypothesis (verbatim) | Mechanism | §176 Constraint |
|---|---|---|---|
| §155 R10 | Super-additive interaction of 5 smoke MCTS+exploration knobs drives 91% draws | Cosine temperature alone is load-bearing (~5% draws → ~91%). Dirichlet / opening_plies / playout_cap are synergy partners, not drivers. | **Do NOT stack knob changes.** Cosine-temp is the singular load-bearing parameter (§156 R11-R14 bisection, sprint log line 542). Source B opponent design must not couple draw-rate tuning with exploration-knob changes. Repro: use R10 (full conjunction) as test fixture, then isolate cosine-temp alone to validate. |
| §169 A4 | Padding semantics (canvas_realness + PartialConv2d) recovers bbox direction | A4 trains to 3.47 loss (below A1 anchor 3.57) but 0% SealBot WR. bbox direction is structural — K-aggregation as cross-cluster contrast, bbox-centroid frame instability, R=8 perception. | **Encoding, not pool tweaks, decides.** Low loss ≠ high WR (L4 origin, sprint log line 628). §176 cannot fix bbox direction via Source B prompt tuning. Must use A1 K-cluster min/max canonical (sprint log line 544). Do NOT attempt PartialConv2d or canvas_realness variants for opponent selfplay. |
| §170 A2/A3 | Learned PMA pool replaces K-cluster min/max | A2 4.5% / A3 7.5% WR vs A1 14.5%. A1 K-cluster min/max canonical. | K-cluster min/max is locked (sprint log line 544). PMA pool empirically inferior. Source B bootstrap must inherit A1 pool, not experiment with learned aggregation. |
| §170 P3 gpool-bias-on-all | Gpool-bias as global lever for both policy + value heads | gpool-bias-policy-only is the load-bearing mechanism; full gpool-bias is NULL on value. | Policy-head bias gate is the load-bearing knob (sprint log line 545). Source B NN variant should inherit gpool-bias-policy-only, not expand to value head. |
| pre-§148 | v6 corpus is human-quality (bot mix at uniform weight does not contaminate) | ~41% bot games at source_weight=1.0; Elo weighting degenerated to uniform via rng.choice on uniform weights. v7 human-only Elo-weighted is canonical. | **v6 corpus retired wholesale** (L15 origin, sprint log line 639). Do NOT use v6 corpus as Source A baseline. v7 human-only Elo-weighted is the canonical reference. Pre-§148 anchors (Q41 51%, Q52 24%) carry contaminated baseline — do not cite. |
| §174 e50 | More pretrain epochs improve self-play | e50 selfplay regressed vs e30 (median 12 vs 17 plies). e50 G4 marginal fail (0.489 vs band 0.462). Value head over-fits to corpus-mode signal that selfplay cannot reproduce. | **More pretrain epochs not strictly better** (L8 origin, sprint log line 632). Source B bootstrap recipe should use e30 (or lower), not e50+. Value head over-fit to corpus-mode signal is a real failure mode — verify selfplay viability at R=8, not just loss curves. |
| §174 radius compression | LEGAL_MOVE_RADIUS 8→5 at bootstrap fixes v6w25 selfplay collapse | Median plies identical across all radii. Radius does not move bootstrap quality. Smokes were already R=8. | Radius knob is inert. Do NOT use radius tuning as a selfplay-collapse mitigation. Collapse is at argmax-degeneracy / selfplay-interaction layer (sprint log line 549), not board sampling. |
| pre-§101 C1 | Promoted weights = evaluated weights | Allocator reuse → every graduation committed unvalidated weights as anchor. Fixed at §101 C1. | **Promotion branch must snapshot eval_model separately.** Do NOT reuse evaluator allocator on promotion path (sprint log line 552). Ensures Source B evaluator is independently scored. |
| §171 A4 P2-reopen | Distribution-shift fine-tune over 5% adversarial corpus (frozen-spine) recovers MCTS signal on A4 | MCTS-64 0/200 Wilson95 [0%, 1.88%] — DEAD bin cleanly met. Falsifies §169 P0 SPATIAL_RICH for frozen-spine class. | Frozen-spine fine-tune does NOT rescue structural bbox failure (sprint log line 556). Do NOT attempt adversarial corpus fine-tuning on Source B as a quick fix for interaction failures. Problem is architectural (K-aggregation, bbox instability), not corpus distribution. |

---

## (b) Mechanism Lessons Table

Lessons L1–L17 (sprint log line 619–641) bearing on §176:

| L# | Lesson (verbatim) | §176 Constraint |
|---|---|---|
| L1 | Corpus quality = model quality floor. Verify corpus completeness BEFORE diagnosing trainer pathology. Silent filter bugs do NOT show in loss curves. | **Source A corpus audit is prerequisite.** Do NOT spin up Source B eval loop until Source A manifest, Elo weighting, and game-filtering logic are validated (§114, §147, sprint log line 625). Silent filter bugs (e.g., rng.choice on uniform weights) bypass loss curves — grep manifest generation code. |
| L2 | Probe gates cannot validate dynamic equivariance. Require MCTS-matched eval, not just argmax probes. | **Threat-logit probes insufficient for Source B eval.**  Use MCTS-matched eval (SealBot n=100) as primary signal, not move-choice or policy-head probes (§154 falsified, sprint log line 626). Probe gates are warning-only. |
| L3 | One sole-load-bearing knob is the default; "super-additive interaction of N knobs" is usually wrong. Bisect within the conjunction. | **Cosine-temp is the singular draw-rate knob.** Do NOT couple it with Dirichlet / opening_plies / playout_cap changes in Source B tuning (§155→§156, sprint log line 627). Isolation via bisection before landing any knob conjunction. |
| L4 | The encoding decides; the pool variant tweaks. Training loss alone is NOT a sufficient signal for downstream WR. | **Encoding is frozen for §176.** A1 K-cluster + min/max is locked. Do NOT trust training loss as a proxy for SealBot WR (sprint log line 628). Pool tweaks (PMA, learned aggregation) are second-order — encoding choice dominates. |
| L8 | More pretrain epochs is not strictly better. Value-head over-fits to corpus-mode signal that selfplay cannot reproduce. | **Pretrain epoch count is a tuning risk.** Source B bootstrap should use e30 baseline; e50+ shows selfplay regression (sprint log line 632). Validate final recipe at R=8 MCTS-128 selfplay, not corpus loss. |
| L9 | Cosine temperature schedule is the load-bearing knob in draw-collapse. Pair with LEGAL_MOVE_RADIUS jitter when active. | **Cosine-temp + LEGAL_MOVE_RADIUS jitter are co-load-bearing** (§156, §157, sprint log line 633). If Source B tunes cosine-temp upward to reduce draws, must pair with radius jitter. Do NOT adjust one without the other. |
| L11 | K-cluster encoding has no board-AI precedent but is structural twin of MVCNN view-pooling, SwAV multi-crop, PointNet++ set-abstraction, deep MIL pooling. 12pp gain at matched MCTS perception is structural inductive bias, not TTA. | **K-cluster is architectural, not empirical tuning.** 12pp gain @ R=8 is structural inductive bias (sprint log line 635). Source B must inherit K-cluster; do NOT attempt to replace with single-window or learned-aggregation alternatives. |
| L12 | Never recalibrate gate thresholds to match failing runs. Never extend smoke runs past stated step limits without explicit go-ahead. | **No moving goalposts on Source B eval.** Pre-register SealBot WR gate (e.g., ≥14.5% to match A1 canonical). Do NOT lower threshold post-hoc because Source B underperforms; do NOT extend smoke beyond pre-registered step limit (sprint log line 636). |
| L13 | Subagent prompts include pre-registered pass criteria; implicit done-when causes scope creep. Independent review subagent at sprint close in fresh context, not implementer's. | **Source B eval prompt must pre-register pass criteria.** Do NOT rely on implicit understanding of "good enough" (sprint log line 637). Wave D plan author must specify SealBot WR target, n_games, confidence interval, horizon (e.g., "≥14.5% WR, n=100, Wilson95, R=8 MCTS-128"). Fresh-context review at §176 close is mandatory. |
| L14 | Pre-flight cold smoke must use canonical sprint bootstrap, not dev defaults. | **Source B cold smoke must use v7full canonical bootstrap, not v6 or dev overrides** (§171 P2, sprint log line 638). Baseline mismatch invalidates comparative claims. |
| L15 | Pre-§148 v6 corpus retired wholesale. All v6-era anchors (Q41 51%, Q52 24%) carry contaminated baseline; do not cite as comparison. | **v6 corpus blacklisted.** Source A corpus must be human-only Elo-weighted (v7 cadence, sprint log line 639). Do NOT use v6 corpus for Source A baseline or cite v6-era anchor performance (Q41, Q52) as comparison. |
| L17 | Always grep receiving code before scheduling a sprint item as "one-line config change". §122 rotation "one-liner" was a ~50-80 line port. | **Underestimate Source B plumbing risk.** BotProtocol wrapper (~30 lines) is minimum; opponent inference loop integration is not a one-liner (sprint log line 641). Wave D plan must budget for full integration audit + bench gate. |

---

## (c) Regressions & Reversions Table

Rows from Regressions & Reversions (sprint log line 593) bearing on §176:

| Feature | When Added | Reverted | Re-implementation Constraint for §176 |
|---|---|---|---|
| SealBot mixed opponent in self-play | §17 (2026-04-02) | Immediately (`c9f39de`) | **CRITICAL**: Python daemon threads caused 3.3× GIL contention regression (1.52M→464K pos/hr, sprint log line 597). Do NOT re-add in-process Python daemon pool for Source B selfplay. **Subprocess-based wrapper required** if Source B parallelism is needed. Reason: GIL blocks NN inference scheduling during opponent move generation. Re-add post-Phase 4.5 baseline with process isolation (fork or subprocess.Popen). This is THE constraint on bot-mixing architecture. |
| torch.compile | §3 | Disabled §32 | **torch.compile disabled** (sprint log line 600) due to Python 3.14 CUDA graph incompatibility. Do NOT re-enable for Source B NN variant without verifying CUDA TLS / Triton stability. Bench gate covers NN latency; compile flags currently off. |
| Chain-length planes in NN input (18→24) | §92 | Reverted to 18 at §97 | **Chain planes OUT of NN input.** Reverted to 18-plane baseline at §97 (sprint log line 602). Chain is now aux sub-buffer target. Source B NN variant must use 18 planes, not 24. Encoding header enforces mismatch rejection. |
| BatchNorm throughout trunk | pre-§99 | Replaced with GroupNorm(8) at §99 | **GroupNorm(8) is canonical.** BatchNorm running stats drift during selfplay (sprint log line 603). Source B NN must inherit GroupNorm(8) per-sample statistics, not BN. MCTS leaf eval @ batch=1 requires batch-independent normalization. |

---

## (d) Open Questions Overlap

Resolved + Active questions from `docs/06_OPEN_QUESTIONS.md` bearing on §176:

| Q# | Status & Bearing on §176 | Source & Line |
|---|---|---|
| Q11 | **RESOLVED 2026-04-28** — Colony detection over-inclusion. `_find_winning_line` locates winning 6-in-a-row; colony check excludes single orphan stones. Colony POC metric for §176 eval must respect "exclude single orphans" rule (docs/06_OPEN_QUESTIONS.md line 390–395). | Resolved, 2026-04-28, commit `fix(eval): Q11 colony detection over-inclusion` |
| Q14 | **WATCH (Phase 4.5 target, blocked on submodule add).** KrakenBot MinimaxBot as eval ladder opponent. Original caveat: "Do NOT use KrakenBot MCTSBot as Bradley-Terry anchor — it is actively training, making it a moving target. **SealBot stays as primary gate**" (docs/06_OPEN_QUESTIONS.md line 455). **§176 corollary**: Source B eval ladder closes with SealBot (or RandomBot), never MCTSBot. MCTSBot was the open question; KrakenBot infrastructure (submodule, BotProtocol) remains pending Phase 4.5 integration. Source B is NOT responsible for landing KrakenBot submodule — that is Wave A1 scope. | Watch item, line 444–456, with caveat at line 455 |
| Q15 | **WATCH (Phase 4.5 target).** Corpus tactical quality filtering — soft-weight sampling by peak threat strength. Could intersect with Source A corpus design if Elo weighting is insufficient. Lower priority (Phase 4.5). Do NOT gate §176 on Q15 implementation (docs/06_OPEN_QUESTIONS.md line 460–475). | Watch item, line 460–475 |
| Q-§176-residual | **OPEN [LOW]** — Two deferred micro-refactor candidates from §176 close-out: (1) HexTacToeNet decomposition (P24b/c, P70); (2) seed_everything shim lift. Both are cold-path or low-risk; hot-path edits in `forward` require bench gate (docs/06_OPEN_QUESTIONS.md line 752–770). Do NOT block §176 Phase A on these micro-refactors. | Deferred, line 752–770 |

---

## (e) Consolidated `do-not` List for §176

Derived empirically from falsified register, mechanism lessons, regressions, and open questions. Each item cites source (§ + sprint-log line OR Q# + open-questions line) and is actionable by Wave D plan author.

1. **Do NOT run an in-process Python daemon pool of opponent bots during Source B selfplay.** GIL contention causes 3.3× throughput regression (1.52M→464K pos/hr, §17, sprint log line 597). If Source B parallelism is required, use subprocess isolation (fork or subprocess.Popen). Re-implementation of bot mixing will be deferred to post-Phase 4.5 baseline.

2. **Do NOT stack knob changes when tuning Source B draw rate.** Cosine-temp is the singular load-bearing parameter; super-additive interaction of Dirichlet / opening_plies / playout_cap is a myth (§155 R10, §156 R11-R14 bisection, sprint log line 542). Isolate cosine-temp tuning via bisection before landing conjunction changes.

3. **Do NOT trust training loss as a proxy for SealBot WR.** A4 achieved lowest loss (3.47 vs A1 3.57) but 0% SealBot WR; L4 falsifies the assumption (sprint log line 628). Encoding decides; pool tweaks are secondary. Source B eval must gate on MCTS-matched SealBot WR (n=100, R=8 MCTS-128), not loss curves or argmax probes.

4. **Do NOT attempt to replace K-cluster min/max pool with learned aggregation (PMA) or alternative variants.** A2 (learned PMA) → 4.5% WR; A3 → 7.5% vs A1 canonical 14.5% (sprint log line 544). K-cluster is architectural, not empirical tuning. Source B must inherit A1 min/max pooling strategy.

5. **Do NOT use more than e30 pretrain epochs for Source B bootstrap.** e50 selfplay regressed (median 12 vs 17 plies), value head over-fitted to corpus-mode signal (§174 e50, sprint log line 632). Validate final recipe at R=8 MCTS-128 selfplay gate, not corpus loss. e30 is the safe baseline; e50+ requires explicit validation.

6. **Do NOT adjust LEGAL_MOVE_RADIUS alone to fix Source B selfplay collapse.** Radius tuning is inert; median plies identical across R=1..8 (§174 radius compression, sprint log line 548). Collapse is at argmax-degeneracy / selfplay-interaction layer, not board sampling. Cosine-temp + radius jitter are co-load-bearing (L9, sprint log line 633).

7. **Do NOT pair cosine-temp upward without radius jitter.** Cosine-temp schedule and LEGAL_MOVE_RADIUS jitter are co-load-bearing in draw-collapse (L9, §156–157, sprint log line 633). If tuning cosine-temp, pair with radius jitter. Standalone cosine-temp changes decouple the load-bearing knob pair.

8. **Do NOT use v6 corpus as Source A baseline.** v6 corpus is human-quality falsified at §147: ~41% bot games at source_weight=1.0 due to Elo-weighting degeneracy (rng.choice on uniform weights, pre-§148, sprint log line 546). v7 human-only Elo-weighted is canonical. Pre-§148 anchors (Q41 51%, Q52 24%) carry contaminated baselines; do NOT cite. Source A must use v7+ corpus.

9. **Do NOT recalibrate SealBot WR gate threshold post-hoc to match Source B underperformance.** Pre-register pass criteria (e.g., ≥14.5% WR, n=100, Wilson95, R=8 MCTS-128) and commit before runs start (L12, sprint log line 636). No moving goalposts. Independent fresh-context review at §176 close validates gate discipline (L13, sprint log line 637).

10. **Do NOT extend Source B smoke runs past pre-registered step limits without explicit go-ahead.** L12 falsifies the practice of extending smokes post-hoc (sprint log line 636). Pre-register step budget (e.g., 5K steps) in eval prompt and enforce boundary.

11. **Do NOT attempt adversarial corpus fine-tuning (frozen-spine) as a quick fix for Source B architecture failures.** §171 A4 P2-reopen falsified frozen-spine fine-tune: MCTS-64 0/200 Wilson95 [0%, 1.88%] cleanly met DEAD bin (sprint log line 556). Problem is architectural (K-aggregation instability, bbox centroid drift), not corpus distribution. Do NOT re-run this experiment.

12. **Do NOT gate Source B eval on threat-logit probes alone.** Probe gates cannot validate dynamic equivariance (L2, §154 falsified, sprint log line 626). Use MCTS-matched eval (SealBot) as primary signal; threat probes are warning-only. Require MCTS-32+ n=20+ sanity gate before advancing.

13. **Do NOT use dev defaults for Source B pre-flight cold smoke.** L14 enforces canonical sprint bootstrap (v7full, sprint log line 638). Cold smoke baseline mismatch (e.g., v6 defaults) invalidates comparative claims. Smoke must use same bootstrap recipe as planned production run.

14. **Do NOT schedule Source B BotProtocol integration as a "one-line config change."** L17 cites §122 rotation "one-liner" which actually required ~50–80 line port (sprint log line 641). Wave D budget must include full opponent inference loop integration + bench gate overhead, not just `~30 lines BotProtocol wrapper`. Wave D plan must allocate realistic complexity for opponent integration plumbing.

15. **Do NOT defer Wave D plan pre-registration to implementation time.** L13 warns implicit done-when causes scope creep (§170–172 A9, sprint log line 637). Source B eval prompt must specify SealBot WR target, n_games, confidence interval, R / MCTS budget, and horizon **before** eval loop launches. Fresh-context review at §176 close is mandatory (not implementer's self-review).

---

## Overlap with Master Prompt `do-not` List

Master prompt specifies 10 implicit `do-not` items in "Constraints — do NOT" block. Cross-check against this empirical scan (15 items):

**Covered by master prompt:**
- In-process GIL daemon regression (item 1)
- Cosine-temp singular knob (item 2)
- MCTS-matched eval (item 12)
- Gate discipline / pre-registration (item 9, 15)

**NOT explicitly covered by master prompt; flagged by empirical record:**
- Freezing K-cluster pool strategy (item 4)
- e30 / e50 pretrain epoch floor (item 5)
- Radius + cosine-temp co-load-bearing pairing (items 6, 7)
- v6 corpus blacklist (item 8)
- Extended smoke boundary (item 10)
- Frozen-spine fine-tune rejection (item 11)
- L17 realistic plumbing budget for bot integration (item 14)

**Recommendation for master prompt author:** Add explicit items 4–8, 10–11, 14 to the "Constraints — do NOT" block. Items 9, 12, 13, 15 are already captured in prompt spirit but could be tightened with line-number cites for operator auditing.

---

## Summary

**Tables completed:**
- (a) Falsified rows table: 9 entries (all §176-scoped from registers)
- (b) Mechanism Lessons table: 11 entries (L1, L2, L3, L4, L8, L9, L11–15, L17)
- (c) Regressions table: 4 entries (SealBot daemon, torch.compile, chain planes, BatchNorm)
- (d) Open Questions overlap: 4 entries (Q11 resolved, Q14 caveat, Q15 watch, Q-§176-residual)
- (e) Consolidated do-not list: **15 items**, each empirically sourced (§ + sprint-log line OR Q# + open-questions line)

**Most critical do-not from empirical record:**
**Item 1 — In-process GIL daemon regression (§17, sprint log line 597).** This is a deal-breaker for bot mixing. 3.3× throughput collapse (1.52M→464K pos/hr) due to Python daemon GIL contention makes Source B opponent selfplay infeasible unless subprocess isolation is enforced. If Wave D plan includes opponent mixing without process isolation, the plan is unsalvageable.

**Master prompt coverage gap:** Master prompt does NOT explicitly cite items 4–8, 10–11, 14 (pool strategy freeze, pretrain epochs, radius+cosine pairing, v6 corpus, extended-smoke boundary, frozen-spine rejection, realistic bot-integration plumbing). These are all empirically falsified or load-bearing and should be added to the master prompt's "Constraints — do NOT" block for operator auditing.

