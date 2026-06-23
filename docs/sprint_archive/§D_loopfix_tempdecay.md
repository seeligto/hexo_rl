# §D_loopfix_tempdecay

_Relocated from `docs/07_PHASE4_SPRINT_LOG.md` (D-DOCS-DEBLOAT split, 2026-06-23). Scope: §D-VALPROBE/§D-VALCEIL/§D-PROMOGATE/§D-LOOPFIX/§D-RERUNPREP + §D-TEMPDECAY. Verbatim; falsified-register rows also consolidated into the sprint-log index register section._

## §D-VALPROBE — probe consolidation + logging decomposition + self-play-distribution ladder — 2026-06-11

Dispatcher: land/clean the value-calibration instrument; decompose trainer `value_loss` logging at source;
close E1's two red-team caveats (C1 train-fit, C2 wrong-regime) on an honest self-play fixture. Branch
`phase4.5/valprobe` (04b11f1..929a35d on master 52911fc), eval/logging-only, full suite 1965 green.

**Phase 1 pre-satisfied:** probe landed tracked in restruct G4a `c485918`; report + raw JSON local-only per
`reports/**`+`audit/` ignore convention. **Phase 2 (04b11f1):** instrument unified fixture-agnostic
(`--fixture corpus|selfplay|<path>`), kernels TDD'd (21 tests), `value_bce` added (trainer-unit), G1/G3
classifier + perspective guard + FIXTURE-VALID bank stats built-in. Corpus-reproduction regression gate
PASS EXACT (banked table's 3 off-cells = old 4dp-console double-rounding, not drift). **Phase 3 (638a729):
PREMISE CORRECTION** — trainer `value_loss` was ALREADY main-only BCE on every surface; banked report's
"flat composite" claim WRONG; flat-curve-vs-falling-MSE = BCE-vs-MSE unit difference (measurement-unit
rule). Decomposition keys (`value_loss_main/_uncertainty/_aux/_composite`) added anyway; logging-only
PROVEN cross-commit (identical loss + post-step param sha256 at pre/post; probe committed
`scripts/diagnosis/bit_identity_probe.py`); caveat: composite = total-minus-pure-policy accounting (aux is
policy-shaped, dominates), NOT value-head signal. **Phase 4 (691de81/1e01ec1/ec52b56/929a35d):** leak-free
generator (WorkerPool at frozen ckpt, K-cluster training-row replay extraction, mover-z from winner spec,
ply-cap drop, perspective+strict-load guards) + FIXTURE-VALID checker (live tail games through SAME
extractor) + occ-matched slicer + K-discriminator.

**FIXTURE-VALID: PARTIAL.** k-axis overlaps (uniform TV 0.098; occ-matching DEGRADED it to 0.172 —
reported); occupancy under-covers at a STANDALONE-GENERATION-PATH support ceiling (fresh ~56 plies vs live
~78 STABLE from run start at near-bootstrap weights; occ==ply → one variable; cause OPEN — ruled out:
load D1 0/0, config deep-merge, version-mixing `model_version_distinct=1`×25103, stop-censoring ~1 ply).
Live-tail bank (run's own last-150 games, same extractor) = in-regime by construction; rungs ≤40k are
LEAK-FREE on it (games postdate their training).

**VERDICTS (pre-registered + red-team-amended):** **G1 FAIL** (fresh leak-free fixture 10k→50k Δsign
+0.011 CI[−0.011,+0.033], Δmse +0.027 CI[−0.015,+0.071]; game-bootstrap, clustering-robust). **G2
re-scoped REGIME-SPLIT** (not train-set-only): livetail 10k→40k leak-free PASSES both G1 thresholds
(+0.053 CI[+0.029,+0.076] / −0.091 CI[−0.140,−0.038]) — early gains TRANSFER to live regime, SATURATE by
~20k (20k→50k +0.007); post-20k corpus gains (+0.065 of +0.105) transferred to NO self-play regime; no
memorization bump at trained-on tip (0.651 ≤ 40k 0.654). **G3: −0.10 deficit EXCLUDED on window-local
axis (≥3 SE @ DEFF≤5)** but axis re-scoped — cluster-centered crops can't see GLOBAL spread, AUC≈0.41
untested by terciles; "most-spread best" demoted (comps↔occ ρ=0.79, collapses under occ control);
**K-axis discriminator (mandated): NO K-deficit any bank/rung, per-row AND min-pool deployment semantics**
(livetail@50k K1 0.571 / K2 0.690 / K3 0.715 / K4+ 0.703). **E1 now claims:** corpus improvement real +
early(≤20k) gains transfer to live regime; post-20k half is corpus-fit-only; live-regime winner-calling
plateaus ~0.65 vs corpus 0.77; mode-collapse refuted everywhere; spread/K-blindness EXCLUDED on measured
axes. STRUCK: implied-mix 0.48 corroboration (value_accuracy unmasked + population mismatch → open
ANOMALY: predicted batch 0.725 vs reported 0.66).

Review (3 fresh) PASS_WITH_NOTES → fixes ec52b56 (strict load, history-plane guard, --ece-bins, hex-adj
docstring, composite caveat). Red-team (4 fresh, default-refute): rt2 z-labels STAND (code-chain +
late-bins 0.63-0.71 orientation evidence, NOT the 0.55 floor); rt1/rt3/rt4 refuted framings → amendments
folded (report `reports/investigations/valprobe_selfplay_ladder_2026-06-11.md`). OPEN: length-gap cause;
value_accuracy anomaly (logging fix candidate: masked/per-source decomposition); AUC≈0.41 needs a
discrimination-AUC instrument. L: pre-register the FIXTURE-VALID gate's FIX path too — "fix the fixture"
without a registered fix recipe invites conditioning-variable surgery that degrades co-registered axes
(occ-matching broke k-TV); always re-gate after the fix. L: a "trained-on" bank still yields LEAK-FREE
reads for rungs that predate its games — date the data vs the rung, don't discard the bank. L: sign_acc
floor guards can't orient near-chance heads — use late-phase bins (decided endgames) for orientation. L:
strength-CI effective-n lesson extends to calibration ladders — CI per-game, not per-row (4000 rows = 150
games here).

## §D-VALCEIL — ceiling-vs-headroom + §D-VALPROBE loose ends — 2026-06-11

Dispatcher: Q1 is the live 0.65 plateau intrinsic-ceiling or headroom; Q2 occ-stratified KSTRAT re-read of
the G3 exclusion; Q3 per-source masked logging + value_accuracy anomaly + mixture mechanism; Q4 78-vs-56
length gap. Eval/logging-only; verdicts pre-registered (dispatcher §2 + pre-read pins: KSTRAT adequacy
floor n_rows≥100∧n_games≥30, occ strata = banked Phase-1 terciles). Branch pushed first (loss-risk);
commits `becc48a` (stratified_tercile_masks kernel + 3 TDD tests — old within-stratum read was 96-100%
degenerate to one bin), `fc46224` (10 additive per-source/masked value keys, 8 TDD tests, bit-identity
EXACT incl. independent reviewer re-run, make test.py 1968 green), `2e4da69` (valceil_analysis.py).
Per-row dumps regenerated on vast from exact ladder slices: ALL 96 cells bit-identical to banked JSONs.

**Q1 CEIL: CEIL-HEADROOM** (registered; CI-robust). Livetail@50k vs corpus@50k per-phase (per-bank occ
terciles, the ladder's method; livetail game-bootstrap CIs): early −0.063 [−0.110,−0.014], mid −0.105
[−0.156,−0.050], late **−0.191 [−0.265,−0.115]** — late-gap CI entirely past both thresholds. Red-team
STRENGTHENED it: matched-ABSOLUTE-occ re-bin widens every gap (−0.13…−0.23, all CI uppers <−0.07);
one-row-per-game control holds (0.754 vs corpus 0.892). Magnitude stamp: decidability-controlled floor
(rows ≤2 plies from termination) deficit −0.082 [−0.132,−0.033] — intrinsic-ceiling alternative KILLED on
objectively-decided rows, but recoverable floor ≈0.08, not the full 0.19 distribution-level gap. Headroom
signatures: 13.4% of late rows never called by ANY rung; 89-row confidently-wrong cluster (19 games,
top-confidence quartile 0.729 < q3 0.786). Saturation is PHASE-UNIFORM post-20k (every post-20k delta CI
straddles 0 in every phase) while corpus-LATE keeps climbing (0.782→0.892 — the post-20k corpus gain
concentrates exactly where live headroom is largest). → value BACK on the live ceiling list (Idea-3
reopened); route: value-investigation design dispatcher.

**Q2 KSTRAT: G3 exclusion BANKED** (occ-stratified, registered rule). In-regime livetail: every adequate
K≥2-vs-K1 gap POSITIVE (+0.04…+0.15; 2 cells significantly; pooled K4+ +0.132 [+0.050,+0.211]); K1 is the
WORST bin — opposite sign of K-blindness; every adequate livetail cell CI-excludes −0.10. Worst adequate
cell anywhere: occmatched mid-K4+ −0.052 [−0.145,+0.034]. Red-team: under unregistered occ-QUARTILES one
adequate fresh cell fires the point rule (−0.108) — refuted by ANCHOR-RUNG control (gap −0.140 at step 0,
pre-exists training, no worsening trend; in-regime livetail same cell +0.103) = bank-composition pocket,
not learned blindness. Floors hide nothing (verdict invariant at 50/15 and 0/0). Open item 4 CLOSED with
the fixed kernel: no within-stratum spread gradient survives occ control. AUC≈0.41 discrimination
instrument stays a separate open question (REOPENED trigger did not fire).

**Q3 SRC: anomaly RESOLVED + mixture CONFIRMED.** The "0.66" was a SPOT batch at step 50000 (trainer emits
per-batch unsmoothed every 10 steps; σ_batch=0.028; 1.2σ below steady mean 0.6941±0.0028 SEM). Real-buffer
reconstruction (250k rows, pyo3 load): 0.625·0.7651 + 0.375·0.5805 = 0.6959 ≈ live 0.6941; the 0.725
prediction over-shot via population gap (−0.016: in-buffer selfplay decided-acc 0.609 ≠ livetail-bank
0.651) + unmasked ply-capped z=0 rows (−0.011; 16% of selfplay rows, model "right" 0.433 there) + corpus
in-batch (−0.003). Masked batch acc 0.7126 ≈ prediction — `value_accuracy_masked` is the
headline-comparable key henceforth. Mixture (registered criteria, same-rows 20k-vs-50k): corpus BCE
0.526→0.422 FALLING × selfplay BCE 0.657→0.653 FLAT × share 0.273→0.375 growing (w_pre=0.8·exp(−t/200k))
= CONFIRMED; share-shift alone costs −0.020 batch acc (masks corpus gains in the aggregate curve).
Buffer facts: win/loss/draw/capped = 43.1/41.3/0.0/15.6%; outcome mix + length stable all run. Premise
correction: THIS run's ply_cap_value=0.0 (the −0.5 override belongs to the §S178 botmix launch). **NEW
BUG flagged (Rust ticket):** .bin persist DROPS value_target_valid (`persist/load.rs:269` "acceptable
shortcut") — any continue-from-ckpt auto-restore (known footgun) would supervise previously-capped rows
at z=ply_cap_value. This run single-session = unaffected.

**Q4 LENGTH: ESCALATE** (registered letter AMBIGUOUS — disclosed cause-attribution decision). Step-1
exploration parity: PASS, configs identical on every axis. Step-2: **run e928c854 had ZERO promotions**
(0 promoted events; 3 evals anchor_promoted=false, wr_best 0.36 only at 25k; best_model.pt mtime = init;
verified 3×) → ALL 25,103 live games were generated by frozen BOOTSTRAP weights; every standalone bank
used the 50k ckpt — the open item's "same weights" premise was FALSE. Matched-weights confirming run
(pre-registered bands 74–84 resolve / 35–56 escalate): bootstrap-weights standalone = 59.7 [57.2,62.2] —
weight identity explains ≤20% of the kept-vs-kept 20.4-ply gap, shift marginal (Welch p≈0.12 vs uniform
bank; zero effect not excludable), k-TV to live got WORSE (0.183 vs 0.098), occ/k moved away → divergence
is in the generation PATH; step-3 trace needs Rust (seed-pluggable worker RNG + per-move trace).
Contamination scope: standalone path exclusive to selfplay_fixture_gen.py (+smoke/perf sibs);
exploit_probe (eval path) + multicluster precheck (KClusterMCTSBot) NOT affected. Standing-record stamps:
livetail bank at EVERY rung = bootstrap-play distribution (G2 REGIME-SPLIT reads "transfer to the
bootstrap-play distribution saturates ~20k"); G1 FAIL certifies non-transfer to a regime no live game
inhabited (verdicts stand, regime labels change); rt1's "~35–56 at same weights" REFUTED on content
(59.7 entirely above band); matched-weights generation does NOT fix FIXTURE-VALID — livetail-style banks
are the only in-regime fresh instrument until the path is traced.

Review: 1 fresh process reviewer + 3 default-refute red-team lenses, all PASS_WITH_NOTES; must-fix
amendments folded (report `reports/investigations/valceil_2026-06-11.md`). L: per-bank terciles can be
the CONSERVATIVE binning — verify with a matched-absolute re-bin before calling a cross-bank phase
comparison apples-oranges. L: anchor-rung (step-0) gaps are a free, decisive control separating
bank-composition pockets from learned deficits — cite them before crediting any training-induced
blindness. L: write confirming-run verdict bands to a timestamped artifact BEFORE launch (in-session
registration was honored but unverifiable post-hoc). L: a zero-promotion run silently turns "live
self-play distribution" into "frozen bootstrap-play distribution" — check promotion count before any
regime claim about live games.

## §D-PROMOGATE — why zero promotions: the dead self-improvement loop (critical path) — 2026-06-11

Dispatcher: diagnose run e928c854's ZERO promotions (gate broken vs eval degenerate vs honest);
pre-registered PG-DEGENERATE/PG-WIRING/PG-HONEST + AB-CONFOUND + PERSIST-FIX; only the XS persist
fix lands. Forensics: 4 parallel agents (Arm C trace / golong trace / wiring / degeneracy repro) +
orchestrator-resolved discrepancy; review = 1 process + 3 default-refute red-team lenses, all
PASS_WITH_NOTES, must-fixes folded. Full: `reports/investigations/promogate_2026-06-11.md`
(+ dispatcher-§2 appendix archived there; registration artifact for the confirming probe written
to vast `tmp/promogate/` BEFORE launch, timestamp-forensics-verified).

**VERDICT — PG-WIRING-PARTIAL + out-of-taxonomy incumbent defect (primary); PG-DEGENERATE REFUTED
(measured); PG-HONEST NOT CLAIMABLE.** The machinery's components are sound (gate arithmetic
correct at the one completed decision; §D-EVALFOUND rewire intact; atomic write; single weight-sync
site eval_drain.py:63; silent promotion not constructible — promoted:true=0, best_model.pt mtime
98.9s pre-run, ALL 25,103 game_complete events model_version (0,0,1)). Three structural defects:
**W1 cadence/truncation** — best stride 2 × interval 12500 × --iterations 50000 → 2 capable rounds
{25k, 50k}; terminal round lands on the stop boundary BY CONSTRUCTION and the 900s final-drain
(D-012 fix PRESENT, FIRED, ~16× undersized: round-2 = 14,604s incl. configured 11,032s sims=1
anchor cell) killed it at sealbot 99/100 → nnue + offwindow stride-4 ALSO lost → zero Objective-A
reads all run → net ONE promotion decision per 50k. W1 fired in golong too (87.5k + 112.5k evals
truncated) — golong lost no DECISIONS only by odd-round parity luck. **W2 stale incumbent+generator**
— best_model.pt ≡ golong checkpoint_00050000_PEAK_sb0.38 (147/147 tensors, diff 0; NOT bootstrap);
restore-resurrect hole fired TWICE pre-AB (a8da 21:07 → s1smoke archive sweep 22:34 → ad0e3535
22:35); via graduation-gate sync (anchor.py:264, in_channels 4==4, no skip event) **ALL 25,103
self-play games were golong@50k-PEAK-generated** — behavioral fingerprint confirms (e928 dist ≡
2ddd 50k-PEAK window: 82.5/77/0.062/0.062/49.5% vs 82.5/77/0.063/0.062/49.8%; ≫σ from 30k-play).
Corollary: a8da S1-smoke selfplay also 50k-PEAK-generated — re-check S1-smoke reads assuming
bootstrap-play. **W3** promotion save drops step (best_model_loaded step=0) — latent in golong legs
2-3 too. **Degeneracy MEASURED ABSENT:** gate cell reproduced exactly (25k ckpt vs best, n=50,
T=0.5 both, seeds 42..91, 0 opening plies): 50/50 distinct games (round_robin dedupe; dup 1.0;
47 distinct 4-ply prefixes) — §D-ARGMAX disease not in this path; effective n ≈ 50. New instrument
hazards: fixed seed schedule (repeat evals not independent); GPU co-tenancy moved same-seed WR
0.36→0.52 (p≈0.11); n=50+CI-guard dead band 0.55-0.62 (promote prob 0.127-0.447). **Power:** de
facto bar 32/50 (=0.64 obs; Wilson-lo>0.5), P(promote|0.55)=0.127, 80% power at true ≈0.685, MDE
+0.185; at design n=400 the guard is benign (bar≈0.55) — the inflation is the variant's n=50
override × guard = power misdesign. PG-HONEST blocked: incumbent wasn't bootstrap (one wrong-config
read vs true bootstrap 0.60 @ 12/20 sims=1 leaves it OPEN) + single attempt + power.

**AB-CONFOUND POSITIVE → A/B-INVALID-AS-RUN** (dispatcher routing): golong promoted 2/4 capable
rounds (50k: wr 0.65 n=100, write deferred to 53432, +305ms after manual Ctrl+C, drain-rescued,
NOT the wave3 abort [that was leg-2 @87.5k]; 75k: wr 0.60 lo 0.5020 EXACTLY at-bar during sealbot
0.05 colony transient, g4 FAIL); generator advanced 30k→50k→75k-era while Arm C froze; arms differ
on launch ckpt (30k-warm vs bootstrap), incumbent, instrument (bar 0.60@n100 vs 0.64@n50 — confound
runs both directions), cadence. GREENLIGHT waits for fixed-loop Arm C or operator re-scope.

**Confirming probe (pre-registered bands reused verbatim):** standalone gen at the TRUE generator
weights (best_model.pt, 450 games) → kept-mean **54.7 [52.8, 56.6]** plies, IN the 35-56 band →
weight identity REFUTED at correct weights; §D-VALCEIL "generation-PATH divergence" STANDS
(+ disclosed strengthening: CI disjoint from bootstrap-run 59.7 [57.2,62.2]; 74-84 band ~20 SE
away; live 78 vs standalone 54.7 at same weights+config). Rust seed-pluggable trace escalation
STANDS. Side-finding: fixture self-check REFUSED the bank — golong@50k-PEAK sign_acc **0.537** on
its OWN games (§S181 value-collapse signature measured directly on the live generator; guard
message "perspective-flipped" is direction-ambiguous). Standalone: ZERO ply-capped games vs live
15.6% capped rows — another path fingerprint. **Regime re-stamp:** §D-VALPROBE/§D-VALCEIL livetail
label "bootstrap-play" → **"golong@50k-PEAK-play"** (verdict letters stand; labels change; value
investigation stays HELD).

**PERSIST-FIX LANDED:** HEXB v9 persists value_target_valid (`cd3da49` fix + `c60e600` hygiene);
TDD (RED verified via poisoned destination; v8-compat test mutation-checked); make test green
(1968 py + full rust, reviewer re-ran fresh); bench gate PASS n=5 pre/post (10/10 targets, no >5%
regression; one spurious batch-fill FAIL did not reproduce — §102 single-run rule);
preflight.sh + compute_threat_pos_weight.py verified v9-safe; no Python row-byte parser.

Routing (operator-gated, design-only): size final-drain from measured round wall-clock or set
--iterations = N·interval+1; preflight MUST pin/assert incumbent sha (auto-restore is correct for
continuation, wrong across experiment boundaries); persist step+encoding in promotion save; for
50k A/Bs consider best stride 1 + per-round seed rotation + banking eval game records. L: "frozen
weights" ≠ "init weights" — pin the generator by tensor identity, the graduation gate syncs
self-play to the ANCHOR; L: schedule shape IS promotion capacity (parity of capable rounds vs stop
step); L: a fixed eval seed schedule makes repeat evals deterministic functionals — co-tenancy
alone moved a 50-game WR by 0.16; L: pre-registered taxonomies need an "experiment-validity defect"
bucket — the biggest finding here fell outside all three registered verdicts.

## §D-LOOPFIX — fix the dead self-improvement loop (W1/W2/W3 + power + co-tenancy + close-out) — 2026-06-11

§D-PROMOGATE proved the loop never worked as designed (cadence structurally promotion-incapable,
incumbent silently wrong, promotions underpowered, terminal eval always killed). This pass FIXES the
loop — code lands — and proves it. Branch `phase4.5/loopfix`, 7 commits over master
(`a4d43fe..` ): W3 `a4d43fe`, W2 `44f4925`, W1 `9a09e18`, POWER `6850ca2`+`6ad97e0`, Phase-6
hardening `52bc24c`, Phase-7 docs (this). Full py suite **1994 passed**; rust unchanged (green
baseline); bench gate N/A (no hot path — close-out is teardown, run_id/encoding threading is per-
promotion, eval-round timing is in the daemon eval thread).

**Per-acceptance verdicts (pre-registered §2, fixed before coding):**

- **A-CLOSEOUT PASS (W1).** training-stop ≠ process-exit. New epilogue: RUN → at N STOP →
  DRAIN(in-flight, BUDGETED) → STOP pool → TERMINAL full-battery eval (all phases, stride IGNORED)
  on the final checkpoint, run UNLOADED → CLOSE → exit 0. Drain budget = measured eval-round
  wall-clock × safety-factor, FLOORED at `eval_final_drain_timeout_sec`, HARD-CAPPED (hung-evaluator
  backstop) — replaces the flat 900 s that was 10-16× undersized (killed the A/B terminal eval at
  sealbot game 99/100). `run_evaluation(ignore_stride=True)` forces every enabled opponent (the
  stride-4 nnue/offwindow got ZERO reads all run). Terminal result is a distinct
  `terminal_eval_complete` event, NEVER fed to steering history (runs outside `step()`); promotes via
  the shared `promote_anchor`; SKIPPED on SIGINT. Cadence: `promotion_capable_rounds()` +
  `eval_schedule_capability` launch log surface stride-parity incapability (WARN on zero). Evidence:
  14 GPU-free unit tests + an integration smoke (real train.py log excerpt: eval_schedule_capability
  → iteration_limit_reached → final_eval_drain_waiting → terminal_eval_start on the final
  checkpoint). Full terminal-eval completion in-subprocess is GPU-bound (19×19 eval games are minutes
  each on CPU) → `tests/test_closeout_lifecycle.py` is slow+integration for vast.

- **A-INCUMBENT PASS (W2).** `resolve_anchor` ALWAYS logs `anchor_identity` (sha256 + path + step +
  run_id) and HARD-FAILS the launch when the LOADED anchor's state-dict sha ≠
  `eval_pipeline.gating.expected_anchor_sha256`. Reads the loaded weights → ANY source (best.pt /
  .bak / bootstrap) yielding non-pinned weights is refused, closing the silent .bak restore that
  installed golong@50k-PEAK as the as-run Arm-C incumbent + generator. `scripts/anchor_sha256.py`
  reproduces the pin; `state_dict_sha256` canonicalises compile prefixes so script & preflight agree.
  Planted-wrong-incumbent test fails loudly. **Pin caveat (red-team Q4):** a pinned run that PROMOTES
  then RESUMES self-deadlocks (advanced best_model.pt ≠ launch pin) — the RuntimeError names both
  causes + eval.yaml documents it; the pin is a per-invocation launch declaration; fixed-loop runs
  run to completion so it rarely bites.

- **A-STAMP PASS (W3).** `save_best_model_atomic(step,run_id,encoding)` wraps the state_dict with
  provenance + writes a `.provenance.json` sidecar; `trainer_ckpt_load` recovers step from the
  partial payload (was 0 → promoted anchors logged step=0, indistinguishable from bootstrap).
  Round-trip tested; bare-payload back-compat pinned.

- **A-POWER PASS.** the n=50 override (× CI guard → bar 0.64, P(promote|0.55)=0.127, dead band
  0.55-0.62) replaced by EXPLICIT n=400 (review caught that DROPPING it inherits eval.yaml's base
  n=100, bar 0.600, P=0.183 — still the golong-era bar). OCs at n=400 (shipped evaluate_gate ×
  binomial): bar 220/400 (0.550), P(promote|0.55)=**0.521** (≥0.5), P(false-promote|0.50)=**0.0255**
  (≤0.05). Dead band GONE (CI-guard bar == nominal 0.55 bar). Quoted in the variant.

- **A-COTENANCY: MECHANISM-NAMED (fix ticketed, not XS).** the 0.36-live/0.52-idle eval swing at
  identical seeds/weights is GPU float-nondeterminism under co-tenant kernel interleaving (fp16
  autocast + non-det cuDNN/cuBLAS flip razor-close MCTS argmax/sampling). NOT a time budget — eval
  MCTS is fixed-sims (`evaluator.ModelPlayer`). NOT `inference_max_wait_ms` — eval uses
  `LocalInferenceEngine` (synchronous per-move forward). Fix not XS (global torch flag; eval runs in
  a daemon thread inside the training process). Structurally mitigated: the decision-critical
  TERMINAL eval now runs UNLOADED (pool stopped first), and n=400 makes the in-run small-n swing
  non-decisive. Detail: `reports/investigations/loopfix_cotenancy_2026-06-11.md` (local).

**A/B re-run (DESIGN-ONLY, un-launched):** `docs/handoffs/loopfix_armc_rerun_runbook.md`. Variant
`v6_live2_ls_ab.yaml` made re-run-ready: incumbent PINNED (aba28e10…), best stride 1 + n 400,
bootstrap_anchor stride 1 / sims 1→128 (off the 11k-s colony cell), nnue/offwindow stride 4→2
(Objective-A in-run), terminal_eval_enabled, robustness gate ON. GREENLIGHT: PRIMARY = absolute
off-window robustness (exploit_probe ≤0.06 + counterfactual via KClusterMCTSBot, NO matched arm);
strength is a non-inferiority guard vs golong50k WITH THE DISCLOSED CONFOUND — the fixed loop biases
Arm C UP, so the guard detects HARM but CANNOT attribute gains to the encoding. DO NOT launch.

**Is the loop now worth 50k GPU?** Yes — promotion machinery proven (atomic save kill-safe, steering
isolation clean, DB double-count blocked by the UNIQUE schema, drain/terminal caps real & bounded),
the four loop defects fixed + the close-out makes the final checkpoint promotion-capable, power
restored, incumbent pinned, terminal eval deterministic (unloaded). The Arm-C re-run is the clean A/B
treatment arm.

L: "remove the override" ≠ "restore the design value" — eval.yaml's base had drifted 200→100; the
POWER fix shipped n=100 until review re-derived OCs against the EFFECTIVE merged config. L: a fix's
own success can trip its own guard — W2's pin self-deadlocks W3's promotion-advanced anchor on
resume; make launch-hygiene gates per-invocation + name the legit case in the error. L: a terminal
"unloaded" eval must actually stop the pool FIRST — close_out before pool.stop() still ran under
self-play load. L: schedule shape IS promotion capacity; surface it at launch, never drop the
decision phase silently.

## §D-RERUNPREP — Arm-C re-run pre-launch sweep + the W2-pin GPU correction — 2026-06-12

Pre-launch sweep for the §D-LOOPFIX Arm-C re-run (DESIGN→IMPL→REVIEW→RED-TEAM, pre-registered).
Phase 0 hygiene PASS (FF merge, 1994 tests, Python-only → bench-skip, hosts reconciled). Phase 1/2
static sweep (6 read-only Explore buckets + aggregate/review/red-team) = 0 blockers. Phase-3 GPU
smoke on vast 5080 (the acceptance §D-LOOPFIX deferred as "GPU-bound") = **FAIL**, and it caught
what static analysis structurally could not:

- **F1 (CORRECTION to W2): the W2 incumbent-pin failed to protect the launch — TWO compounding
  causes, and the first diagnosis was WRONG.** (a) **W2-VACUOUS (code):** the runbook's preflight
  `rm best_model.pt` routes the launch through `anchor_fresh_init_no_bootstrap`, which NEVER ran the
  pin check — so the launch had **zero** incumbent verification. (b) **Host bootstrap DRIFT (data):**
  vast's `bootstrap_model_v6_live2.pt` (`4198d5cb`) ≠ the committed pin (`aba28e10`), which was
  computed on the **laptop's** drifted copy (`ab8d71d`); the bootstrap is gitignored / distributed
  out-of-band, so the dev host and the run host drifted silently (corpus stayed identical — only the
  model). The original smoke's resume hard-failed because it loaded vast's bootstrap (`4198d5cb`) and
  checked it against the laptop-derived pin (`aba28e10`). **The fp16/dtype theory was a MISDIAGNOSIS**
  — a GPU discriminator (run before the re-smoke) showed the anchor is fp32 and OLD==NEW hashing
  (`4198d5cb`); the divergence was the host file mismatch all along. I never reproduced `4198d5cb` on
  the laptop and rationalized the gap instead of treating it as a failed reproduction. Code fix
  (still valid): `checkpoint_state_sha256(path)` hashes the STORED weights (single source of truth
  with `anchor_sha256.py` — prevents pin↔runtime drift); the loader returns the source PATH;
  `verify_launch_anchor_pin` closes the fresh-init vacuum (fail-closed when a pin is set but the seed
  is unverifiable). Data fix: de-facto bootstrap = vast's `4198d5cb` (golong + every vast run used the
  only bootstrap ever on vast); re-pinned the configs/runbook to `4198d5cb` and synced hosts so
  bootstrap+pin agree. TDD, 12 anchor tests green; full suite 1998 passed; the fix now correctly
  PASSES the pin (where the wrong-pin smoke hard-failed).
- **F2 (XS): the `terminal_eval_complete` structlog line dropped `completed`/`terminal`** — the
  fields the JSONL/integration test + watch sheet read. The eval completed fine (wr_best present);
  only the telemetry was incomplete. 2-kwarg fix, mirrors the event_emitter payload.
- **F3: the self-check script was buggy** (`grep -c` "0\n0" → false FAILs; wrong tokens). The
  lifecycle itself PASSED in run.log: iteration_limit_reached, budgeted drain (warn-never-kill),
  terminal full-battery (wr_best 0.605), n=400 gate, exploit_probe wiring. Fixed against real tokens.

L: DON'T ship a fix on an UNREPRODUCED hypothesis — I never reproduced `4198d5cb` on the laptop
(got `aba28e10` on the file, `cae9ae8c` on an fp16 cast — neither matched) and rationalized it as
"fp16, the exact bytes don't matter." They did: a cheap on-target GPU discriminator run BEFORE the
3h re-smoke showed fp32 + OLD==NEW and the real cause (host file drift). Reproduce the exact symptom
or keep digging. L: a gitignored artifact distributed out-of-band DRIFTS silently across hosts —
the pin was computed on the dev host (laptop `aba28e10`) but the runs execute on vast (`4198d5cb`);
compute/verify the pin on the SAME host the run uses, and make the verify catch the drift. L: a
safety GATE the launch-preflight routes AROUND is vacuous; verify the pin on the path the run
actually takes, not only the stale-anchor branch. L: front-loading the integration/GPU smoke before
the expensive run is what turned a 4-day/$67 invalidation risk into a free local catch — of BOTH a
real code hole (W2-VACUOUS) and a misdiagnosis. Artifacts:
`docs/handoffs/phase3_smoke_results.md`, `phase3_finding_terminal_eval_completed.md`,
`armc_rerun_launch_package.md`, `armc_rerun_watchsheet.md`.

## §D-TEMPDECAY — per-game within-game temperature decay: clean-up + probe + smoke — 2026-06-12/13

**Audit:** the within-game cosine (`compute_move_temperature`, quarter-cosine on the
compound-turn clock) is CORRECT-BUT-TOXIC-LEVER, not a bug — the §156/L9 collapse was
floor=0.05, not broken code. 11-agent audit + adversarial re-review: all 6 load-bearing
claims held, 0 refutations. **Cleanup landed (commit `711919d`, pushed):** OFF=constant-0.5
default (de-armed the latent re-arm footgun in `config.rs`/`pool.py`), eval helper
`get_temperature` unified onto one shared `quarter_cosine_temperature` + compound-turn clock,
dead `temperature_threshold_ply` field removed, cross-language Python↔Rust parity golden test.
2006 py + engine + parity green. `gumbel_full.yaml` recreated.

**Probe (Phase-2, golong@50k static, N=400/arm, vast 4080):** all 3 floors PASS draw-safety;
the aggressive floors draw LESS than control (a20=0.020, a30=0.037 vs control 0.048) — the
L9 collapse is a TRAINING dynamic, ABSENT in static generation (pre-registered hypothesis
confirmed). `smoke_recommend=a20`.

**Smoke (Phase-3, control vs a20, 10k steps each):** **TEMP-NEGATIVE.** Clean shared-corpus
metric `value_accuracy_corpus` FLAT (Δ=−0.001, CI straddles 0); the pre-reg `value_accuracy_masked`
"+0.016" was a CONFOUND (selfplay portion — a20's sharper games are easier to label on its OWN
data; shared-corpus subset flat). NOT toxic (0.30 draw gate never fired) but a20 mildly elevates
+ slowly raises TRAINING draws (0.10 vs 0.065, rising) — opposite its static-probe behavior.
Lever PARKED; CEIL-HEADROOM more likely loop-downstream.

**Falsified-register:** re-enabling within-game cosine at floor 0.20 buys a training draw-tax for
no shared-data value-calibration gain. **Meta-lesson:** cross-arm value comparison MUST use the
shared-corpus subset (`value_accuracy_corpus`), never `value_accuracy_selfplay`/`_masked`
(confounded by per-arm self-play distribution). Report: `docs/handoffs/tempdecay_report.md`;
pre-registration: `reports/investigations/tempdecay_phase0_2026-06-12.md`. Branch
`phase4.5/tempdecay`. Total vast run ~14h ≈ $2.9.

