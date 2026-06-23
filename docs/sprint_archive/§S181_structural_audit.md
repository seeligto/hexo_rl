# §S181_structural_audit

_Relocated from `docs/07_PHASE4_SPRINT_LOG.md` (D-DOCS-DEBLOAT split, 2026-06-23). Scope: §S181 structural diagnosis + §S181-AUDIT Waves 1-5. Verbatim; falsified-register rows also consolidated into the sprint-log index register section._

## §S181 — structural diagnosis (research wave)

*DISCRIMINATOR: §S181 = structural-diagnosis research wave (this entry).
INSPECTION-ONLY — no training, no hot-path edit, no config edit. Successor
to §S180b after the config-level anti-colony surface was declared exhausted
(L38).*

**Status:** COMPLETE. 4 parallel inspection tracks, all reviewed and PASSED.
Verdict: the colony attractor is a **training-loop-generated value-head
discrimination collapse** that the architecture PERMITS and the current
metrics are BLIND to. It is NOT bootstrapped and NOT MCTS-driven —
handoff hypotheses #1 and #5 both FALSIFIED.

**Wave identity:**
- Branch: `phase4.5/s181_structural_research`
- Mode: inspection-only (4 standalone probes under
  `scripts/structural_diagnosis/`; no selfplay/training/MCTS core touched)
- Inputs: §S180b archive (`archive/s180b_3knob_fail/`), anchor
  `bootstrap_model_v6.pt` (SHA `7ab77d2c…`), v6 human corpus, eval archives
- Aggregation: `audit/structural/00_aggregation.md`

### Per-track verdicts

| Track | Question | Verdict |
|---|---|---|
| T1 — bootstrap+corpus bias | Does the bootstrap/corpus encode colony bias pre-self-play? | **BIAS-RULED-OUT.** Value head extension-favouring (open-5 +0.978 > 5-blob +0.583; open-4 +0.704 > 4-blob +0.378); Δ(colony−ext) value −0.150, Welch p=0.355 (wrong sign, n.s.). Corpus winning lines 91.3% extension-shaped, rising to 100% at 1400+ Elo. Policy plays the 6th-move win 90%. Falsifies handoff hypothesis #1. Correction: §S178 line uses encoding `v6` (k_max=1), NOT v6w25. |
| T2 — value head + encoding architecture | Does the architecture structurally encourage colony interpretation? | **ARCHITECTURAL-BIAS-CONFIRMED (PERMISSIVE, not FORCED).** Dual-pool value head `v_max` half is a coverage-blind monotone peak detector — `GMP(colony) ≡ GMP(extension)` exactly (max\|diff\|=0.0), no architectural counterweight guaranteed. Permits colony-value-saturation; does not compel it. Density-centred cluster windowing favors compact structure. K-cluster min-pool is v6w25-only — does NOT fire for the §S178 line. Ranked fixes A1–A4; recommends A2 (multi-scale avg-pool ~40 LOC) + A3 (colony-penalty aux ~60 LOC) paired. |
| T3 — MCTS colony dynamics | Does MCTS+PUCT amplify or correct the bias? | **MCTS-NEUTRAL.** c_puct ×0.5/×2.0 and Dirichlet ×4 sweeps move colony-visit fraction <6pp / <3pp — no search-config escape hatch (extends L38). DECISIVE side result: §S180b step-50k value head collapsed FLAT — colony−extension value spread −0.016 vs anchor +0.617. L37 INVERTED: colony positions yield LOWER-entropy, MORE-concentrated visit targets (sharp-and-wrong, strong self-reinforcing CE gradient). |
| T4 — probe + dashboard redesign | Why did the probes miss 4 collapses? | **PROBE-REDESIGN-NEEDED.** C1–C4 static threat-logit probes cleared the gate ~11× at the §S180b 0/100 crash — categorically blind. `colony_a` (anchor-game colony fraction) already in the `evaluation_round_complete` payload at 36/100 by step 10K — 40K before crash — never surfaced first-class. 4 MCTS-in-loop probes designed; retrospective fire-step 20–40K before the §S180b crash. Land order PR-A (`colony_a` ~40 LOC) first. |

### Convergent diagnosis

The colony attractor is **H6 — a training-loop value-head discrimination
collapse**: the value head flattens (loses colony/extension separation)
during self-play, removing the signal MCTS needs to prefer extension;
search then collapses onto the colony-biased policy prior. Measured: value
spread +0.617 (anchor, healthy) → −0.016 (§S180b step-50k, captured). The
architecture **H7 — offers no resistance**: the `v_max` coverage-blind
monotone peak detector has no counterweight. The metrics **H3/H4 — are
blind**: C1–C4 static, `colony_a` never first-class. This IS the
config-invisible capture channel L38 named — and it is directly observable
with a 40-position static value-spread probe, one forward pass per
checkpoint.

### Ranked hypothesis list

| Rank | Hypothesis | Status |
|---|---|---|
| 1 | H6 — training-loop value-head discrimination collapse (value spread +0.617→−0.016) | TOP — MEASURED; primary driver |
| 2 | H7 — architecture permits H6, no resistance (`v_max` coverage-blind, weight-independent) | HIGH — MEASURED; fix surface = T2 A2 |
| 3 | H3/H4 — probes blind, `colony_a` config-invisible | HIGH — CONFIRMED 4×; instrumentation, not a fix |
| 4 | H8 — colony target sharp-and-wrong (L37 inverted; H_frac 0.484 vs 0.805) | MED — explains *why* CE reinforces |
| 5 | H9 — aux opp-reply head lower loss floor in colony regime | MED — mechanism inferred, not measured |
| 6 | H10 — density-centred windowing favors compact structure | LOW for v6 K=1; v6w25 re-entry only |
| — | H1 — bootstrap/corpus encode colony bias | **FALSIFIED** (T1) |
| — | H5 — MCTS dynamics favor colony | **FALSIFIED** (T3); MCTS-search config sub-surface exhausted |
| — | H2 (v6 form) — min-pool architectural asymmetry | **N/A** for v6 anchor (`value_pool="none"`) |

### Next-step decision tree

```
FU-1 — value-spread checkpoint-ladder probe  (~30–60 min, 0 GPU, ~2 dev-hr)
  Probe §S180b ladder ckpt_step{10,20,30,40,50}k for value spread.
  │
  ├─ spread degrades monotonically from step 10K
  │    → loop installs bias early + gradually
  │    → FU-2 architecture A/B must hold the canary from step 0;
  │      PSW/refresh-hook (if used) must act before step 20K
  │
  └─ spread holds healthy then late cliff
       → phase transition exists; target the cliff step directly

FU-2 — value-head re-architecture A/B  (T2 A2+A3; ~3–5 days, ~3–4 GPU-days)
  Requires FU-1 first (pins the canary target). Fresh re-pretrain
  mandatory (A2 breaks value_fc1 shape). Wire value-spread canary +
  hard-abort gate (abort if spread < +0.20).
  │
  ├─ A2+A3 holds spread > +0.20 AND wr_sealbot does not collapse
  │    → architecture was the load-bearing permissive element (H7 = fix)
  │    → adopt A2+A3; close the structural line
  │
  └─ spread still collapses under A2+A3
       → loop installs the bias regardless of architecture
       → escalate to buffer-level levers (PSW / bot-corpus refresh hook),
         success metric = value-spread canary, NOT loss/value-acc (Goodhart)

PR-A — colony_a first-class metric + ALERT-2  (T4; ~40 LOC, ~2 hr)
  Independent of FU-1/FU-2 — land in parallel. Lowest-LOC highest-leverage
  fix in the wave: would have fired 20–40K steps before the §S180b crash.
```

### Process patterns / Mechanism Lessons

§S181 adds L41/L42/L43.

- **L41 (MCTS-search config sub-surface is also exhausted — extends L38).**
  T3 swept the last untested config sub-surface: `c_puct` ×0.5/×2.0 and
  `dirichlet_alpha` ×4. Colony-visit fraction moved <6pp / <3pp — inside
  n=20 noise. Higher c_puct mildly *worsens* colony preference (it
  up-weights an already colony-biased prior). The MCTS-search knobs join
  the YAML config-level levers as exhausted. Do NOT propose c_puct /
  Dirichlet retune as an anti-colony lever — FALSIFIED in
  `audit/structural/03_mcts_colony_dynamics.md` §4/§5.

- **L42 (value-head discrimination collapse is the config-invisible
  capture channel — directly observable).** The L38 "config-invisible
  capture channel" is the value head losing colony/extension separation
  during self-play. MEASURED: colony−extension value spread +0.617
  (anchor) → −0.016 (§S180b step-50k). A flat value head gives MCTS no
  signal to prefer extension over colony, so search collapses onto the
  colony-biased policy prior. It is "config-invisible" only because no
  dashboard metric tracked it — it is fully observable with a 40-position
  static probe, one forward pass per checkpoint. Wire `value_spread`
  (mean V over a colony bank − mean V over an extension bank) as a
  first-class canary + hard-abort gate (abort if spread < +0.20). This
  supersedes diagnosis-by-config-metric.

- **L43 (colony training target is sharp-and-wrong, not diffuse — L37
  inverted).** §S180a L37 hypothesized "colony regime → diffuse MCTS
  visit target → weak CE signal". T3 MEASURED the opposite: colony
  positions yield LOWER-entropy, MORE-concentrated visit targets
  (H_frac 0.484 vs extension 0.805) — the policy prior is sharply peaked
  on blob-adjacent cells. CE against a sharp wrong target is a STRONG
  self-reinforcing gradient pushing the policy further into the colony
  mode — a worse failure mode than a diffuse target. This is the textbook
  attractor mechanic and matches the operator's "structured wrong choice,
  not random collapse" game-read. L37's *empirical finding* (visit-count
  CE weaker than CQV) stands; its *proposed mechanism* (diffuse target)
  is corrected.

### Falsified Hypotheses Register additions

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S181-T1 | `bootstrap_model_v6.pt` + v6 human corpus jointly encode a colony bias in the value head and/or policy head before self-play (handoff hypothesis #1) | §S181-T1 bootstrap+corpus bias audit (`audit/structural/01_bootstrap_corpus_bias.md`) | Value-head Δ(colony−ext)=−0.150, Welch p=0.355 (wrong sign, n.s.); near-win sub-probe rates open-5 extension +0.978 > 5-blob +0.583 (open-4 +0.704 > 4-blob +0.378); policy plays the 6th-move win 90%; corpus winning lines 91.3% extension rising to 100% at 1400+ Elo. No colony bias exists pre-self-play — the attractor is generated by the training loop. |
| §S181-T3 | MCTS-search parameters (`c_puct`, `dirichlet_alpha`/`epsilon`) are a viable anti-colony escape lever (handoff hypothesis #5) | §S181-T3 MCTS colony-dynamics audit (`audit/structural/03_mcts_colony_dynamics.md`) | c_puct ×0.5/×2.0 moves colony-visit fraction <6pp; Dirichlet ×4 moves it <3pp — both inside n=20 noise. Higher c_puct mildly worsens colony preference. MCTS+PUCT neither amplifies nor corrects the bias; it faithfully passes through a colony-biased value/policy head. MCTS-NEUTRAL — extends L38 to the search sub-surface. |

### Files produced

- `audit/structural/00_aggregation.md` — this wave's synthesis
- `audit/structural/0{1,2,3,4}_*.md` — 4 track audits + sidecar JSONs
- `scripts/structural_diagnosis/{probe_value_bias,probe_architecture,mcts_colony_probe,new_probes}.py`
- `reports/s181_next_wave_skeleton.md` — next-wave operations skeleton

### Successor

§S181 does NOT launch a training run. Next wave = `reports/s181_next_wave_skeleton.md`:
FU-1 value-spread ladder probe FIRST (pins the flatten step, ~30 min),
THEN FU-2 value-head re-architecture A/B (T2 A2+A3) gated on FU-1. PR-A
(`colony_a` first-class metric) lands in parallel — independent, ~40 LOC.
PSW / refresh-hook levers are demoted: they reshape the buffer but only
help if the reshaped buffer retrains value-head discrimination — pair
either with the L42 value-spread canary as the success criterion, not
loss/value-acc (Goodhart).

### §S181 FU-1 — value-spread checkpoint-ladder probe

Ran the FU-1 ladder probe (`scripts/structural_diagnosis/fu1_value_spread_ladder.py`,
inspection-only, CPU, ~0 compute). Pins WHEN the value-head discriminator
flattens across the §S180b trajectory. Full report:
`audit/structural/05_fu1_value_spread_ladder.md`.

**Bank.** Reused the T3 40-position canonical bank verbatim (20 colony +
20 extension; `mcts_colony_probe.py` builders, deterministic). Fixture
SHA-256 `934204713620d171…dcc23991`. **Brief correction:** the +0.617
anchor spread is a T3 measurement, NOT T1 — T1's `probe_value_bias.py`
bank is 50+50 and gives a different figure (−0.150). FU-1 used the T3
bank, the only one consistent with +0.617. Anchor reproducibility gate
PASS — FU-1 reproduces T3 `value_head` to 4 dp (+0.6173 vs +0.617).

**Ladder.** V_spread (mean V colony − mean V extension):

| step | 0 | 10k | 20k | 30k | 40k | 50k | 53.5k |
|---|---|---|---|---|---|---|---|
| V_spread | +0.617 | +0.260 | −0.110 | +0.140 | −0.051 | −0.016 | +0.099 |

**Verdict.** Drift classifier returns `OSCILLATION` (non-monotone;
+0.250 up-swing 20k→30k). But the mechanical label is a **noise
artifact** — post-20k swings (±0.25 peak-to-peak) sit within ~1.3×
SE(spread)=0.185 of a flat spread ≈ 0. The substantive signature is
**EARLY / FRONT-LOADED COLLAPSE**: 69 % of spread lost by step 10k,
sub-+0.20 (FU-2 abort gate) at or before 10k, negative by 20k, then
flat-dead. The discriminator dies inside the first 10–20k steps and
never recovers. Ladder limitation: no sub-10k checkpoint — cannot
resolve a step-0 onset from an ~8k cliff.

**colony_a denominator = WINS-ONLY** (`evaluator.py:213-216`,
`display.py:40`: `colony / wins`). The §S180b colony_a series is the
fraction of the model's *wins vs the anchor* that are colony-shaped —
computed over a shrinking small win pool as wr collapses. A colony_a
trigger fires late + noisy; the value-spread static probe (stable
denominator, crosses +0.20 before step 10k) is the better canary.

**Recommendation (operator decides).** Early-collapse read routes to
FU-2 (value-head re-arch A/B, T2 A2+A3) with the value-spread canary
wired as a hard-abort gate **from step 0** — a buffer-level refresh
hook cannot act fast enough for a collapse >half complete by step 10k.
Cheaper de-risk before committing FU-2: a finer 0–20k ladder (new
checkpoints every ~2k from a short §S180b-config re-run) to pin
step-0-onset vs mid-cliff. Not auto-launched.

### §S181 FU-1.5 + PR-A — finer ladder + value_spread canary

Closed 2026-05-23 — Stage 4 merged. 4 commits on master
(`b8ebb1d..9cfadd7`) covering FU-1.5 (probe SHA gate + pre-registered
V-FL classifier + 10-rung 2k-cadence ladder + audit doc), PR-A
(value_spread canary as a first-class colony-capture trigger), the
§S180b config + launch script for the re-run, and the FU-1 audit
artifacts brought along. Independent landed fix `de149e6
fix(mcts): canonical child order — kill FxHashSet-order leak`
addresses a latent `FxHashSet` iteration-order leak found during the
investigation (behaviourally inert per 5 independent tests; not a
regression cause; landed as a hygiene improvement). REVIEW
(Opus 4.7, high-effort, fresh context) returned **PASS 19/19** across
scopes A/B/C.

**Verdict (mechanical, code — L13 guard).** `V-FL-A — STEP-0-ONSET`.

**Ladder (clean-host vast re-run, anchor + bank SHA gates PASS):**

| step | 0 | 2k | 4k | 6k | 8k | 10k | 12k | 14k | 16k | 18k | 20k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V_spread | +0.617 | **+0.175** | **−0.118** | +0.055 | +0.217 | +0.390 | +0.167 | +0.523 | +0.221 | +0.208 | +0.108 |

Single-interval loss 0→2k = **−0.4421 = 86.7 % of the total
trajectory loss**. V_spread crosses the +0.20 FU-2 abort gate
*between step 0 and step 2k*. Step 4k goes negative.

**Key finding — resolves FU-1's open question:** the value-head
collapse is front-loaded to the first 2k steps. The FU-1 10k-cadence
ladder under-sampled this region and could not localize the onset;
FU-1.5's 2k cadence places >86 % of the collapse before step 2k.

**FU-2 routing (pre-registered V-FL-A → A2 arm):** **A2 multi-scale
avg-pool value head load-bearing.** An aux-loss A3 cannot catch a
value-head crash that is >86 % complete by step 2k — there is no
time for an aux gradient to push back before the discriminator is
already through the abort gate.

**Substantive nuance (not the verdict).** §S180b crashed
monotonically to a permanent flat-dead band by step 20k; FU-1.5
crashes fast then **oscillates** post-onset (peak +0.523 at step
14k, range −0.118 to +0.523). Both 20k endpoints sit near zero
(FU-1.5 +0.108, §S180b −0.110, both within ~1 × SE of zero). The
endpoints agree; the paths diverge — consistent with chaotic
post-onset divergence in the colony-attractor neighbourhood.

**§S180b cross-check @ step 10k (brief ±15pp variance gate):**
wr_sealbot 9 % vs §S180b 11 % (−2 pp); elo 421.3 vs 422 (−0.7);
wr_anchor 66 % vs 61 %; wr_best 52 % vs 52 %; colony_wins_anchor
41 vs 36. All within noise — FU-1.5 IS a faithful §S180b
reproduction.

**Investigation context (the host-state regression).** The first
FU-1.5 launch attempt hit an anomalous vast-host regime (`depth ~2.5
/ games ~77-96 plies` vs every pre-05-22 run at depth 3.4-3.8 /
plies 27-57). Five-test code exoneration of §S182/§S183
(deterministic depth probe; deterministic full-game probe; laptop
self-play 100-game smoke A/B; laptop full-training 100-game A/B;
vast 100-game A/B post-reinstall — every test bit-identical old vs
new code) ruled out the perf wave. Root cause: stale vast-host state.
Clean reinstall (`rm -rf hexo_rl` → fresh `git clone` → `make install`:
fresh `.venv` torch 2.11.0+cu128 + fresh engine build + 1555 tests
PASS) restored the §S180b regime. The relaunch on the clean host is
the data this audit reports on.

**PR-A — value_spread canary.** First-class colony-capture trigger
wired as a checkpoint-save callback emitting a `value_spread` event
+ alert rule (WARNING < +0.30, SOFT-ABORT < +0.20 — the FU-1 / FU-2
abort gate). 40-position T3 bank frozen at
`tests/fixtures/value_spread_bank.json` (SHA `9342…23991`); INV pin
`tests/test_inv_value_spread_bank.py` asserts the SHA. Renderer
failure cannot propagate (canary fire-and-forget). `colony_a` stays
in the eval payload — additive, not destructive. No hot-path touch
(canary fires post-save, not in the inner loop). Micro-bench: 15.1
ms → 30.4 ms ckpt-save on RTX 4060 (+15.3 ms, under the 50 ms gate).
Replaces `colony_a` as the *trigger* signal (FU-1 §5: `colony_a` is
wins-only / late / small-denominator; value_spread is a stable
static probe that crosses +0.20 before step 2k).

**Independent landed fix (`de149e6`).** `pick_topk_children` Path A
emitted children in raw `FxHashSet` iteration order (a knowingly-left
"identical to pre-cap behaviour" hash leak); §S182's capacity-reserve
changed that order. The fix unifies both paths to canonical
`(prior desc, flat asc)` — the canonical key Path B already used.
MCTS is now order-stable regardless of hashbrown capacity. Five
independent tests prove behavioural identity (it was never a
regression cause); landed as hygiene. Bench-gated: MCTS sim/s
−3.0 % (under 5 % gate). Regression test
`test_topk_child_order_independent_of_hashset_capacity` pins the
invariant.

**L44 — value-head crashes can be front-loaded.** A 2k-cadence
ladder on §S180b's recipe shows >86 % of the value-spread loss
occurs in the first 2k steps from the bootstrap. The 10k-cadence
ladder of FU-1 under-sampled this region. Future structural-
diagnosis waves that probe value-head onset should default to ≤2k
cadence in the first 10k steps.

**L45 — host-state hygiene is research-load-bearing.** The
05-21→05-22 vast regime change (depth 3.4→2.5 stable; uniform
across all 05-22 vast launches before the reinstall) was *not*
code. It was accumulated host cruft (stale `.venv`, stale compiled
engine, leftover state from prior runs). Five independent code
A/B tests exonerated the perf wave. A clean `git clone` +
`make install` restored the §S180b regime. **Implication:** when a
"first variance since the N-knob baseline" appears, host-state
reinstall belongs in the discriminator set alongside the code
bisect — not as a follow-up.

**L46 — post-onset oscillation ≠ flat-dead.** §S180b reached a
permanent flat-dead V_spread band after step 20k; FU-1.5 (same
recipe, clean host) reaches a similar mean-near-zero by step 20k
but the path oscillates (+0.523 peak at step 14k). Two faithful
reproductions of the same recipe can take different routes through
the chaotic colony-attractor neighbourhood. The 20k endpoint
agrees within ~1 × SE; the trajectory does not.

**Next.** FU-2 NOT auto-launched. **Operator decides FU-2 A2 arm**
(multi-scale avg-pool value-head re-architecture; ~3-4 GPU-days
pretrain cycle) per L34/L42/V-FL-A routing.

---

## §S181-AUDIT — Track A source decomposition + PR-B hygiene

**Wave.** Cheap inspection-only audit + no-regret hygiene patch in
parallel. Track A on `phase4.5/s181_audit_track_a` (6 commits
`c2c9e5e..e717c61`, FF-merged 2026-05-23). PR-B on
`phase4.5/s181_pr_b_hygiene` (3 commits `c2a0f31..03425de` post-rebase,
FF-merged 2026-05-23). REVIEW pass (Opus 4.7 fresh-context, scopes
A/B/C all PASS).

**Provenance.** Designed against L47 (training-loop dominates
architecture by ≥1.0 V_spread per ~1000 steps; banked on the parked
`phase4.5/s181_fu2_a2_arch` branch, audit doc
`audit/structural/08_fu2_a2_sustained_ladder.md`). The FU-2 wave proved
A2's architectural anchor inversion (−0.508 PT-3 baseline) was dragged
to +0.13 within 1000 steps — same end-state as pre-A2 from the
sign-opposite direction. Next surface = the LOOP. Track A localizes
which loop-level source carries the bias; PR-B lands stage-1 hygiene
mechanism-aligned with L47.

### Per-subtask Track A verdicts (LITERAL, L13 guard)

| ID | hypothesis | verdict | quantitative finding |
|---|---|---|---|
| A1 | H-BOT — bot corpus position-level bias | INCONCLUSIVE | colony frac 26.0%, asymmetry +0.078 (8× < anchor +0.617) |
| A2 | H-AUG — augmentation oversamples colony | INCONCLUSIVE + FALSIFIED | unique-variants ratio 1.0 (mechanism dead); feat-var ratio 0.54 is downstream signature |
| A3 | H-BANK — T3 bank confound | **CONFOUND** | Pearson r=0.27, T3 V_spread amplified ~3× vs alt bank |
| A4 | H-CE-STRENGTH — per-class gradient | INCONCLUSIVE | grad L2 ratio 1.21 (L43 entropy confirmed col 0.50 / ext 0.80) |
| A5 | H-PRETRAIN — pretrain position z | INCONCLUSIVE | colony frac 31.2%, asymmetry +0.157 (~25% of anchor) |

No subtask reached STRONG-CONFIRM / ASYMMETRIC-CONFIRMED /
PRETRAIN-COLONY-BIASED. The largest source contributor is H-PRETRAIN
(asymmetry +0.157 × ~30% sample share = +0.057/step upper bound),
followed by H-BOT (+0.014/step), with H-CE-STRENGTH as a 1.21×
multiplier on every colony sample and H-AUG falsified to zero.

### A6 dominant-source identification + routing

**Dominant source: NONE pre-registered-confirmed.** Per routing table:
"None confirmed → escalate to Track B (live training-loop gradient
instrumentation)". The largest unmeasured surface is the self-play
buffer's per-sample gradient signal — Track B is a short instrumented
run that closes the per-step pull accounting gap.

**Composite next-wave lever (operator decides — NOT auto-launched):**

1. Track B (PRIMARY) — short instrumented run logging per-sample
   gradient magnitude bucketed by colony / extension / neither on the
   live self-play buffer.
2. Combined lever IF Track B confirms loop-side imbalance:
   PSW (Prioritized Stratified Window on the buffer) + refresh hook
   (periodic bot-corpus regeneration) + per-class target temperature
   on colony positions.
3. **Dual-bank V_spread canary update.** Per A3, augment PR-A's
   `value_spread_canary.py` to compute V_spread on BOTH T3 and alt
   banks. Alt bank fixture pinned at
   `tests/fixtures/value_spread_bank_alt.json` with SHA in
   `meta.sha256`. Compute on both ≈ 1 sec per ckpt.

**DEFERRED.** Track C (encoding swap, ~1 GPU-day), EMA (changes
self-play inference model), WDL migration (same cost profile as A2).
**DO NOT** re-propose architecture-only fixes (L47 stands).

### PR-B — hygiene patch (Stage-1, landed)

Three trainer-side hygiene changes, ~80 LOC net:

1. **Param-group split for AdamW** (`c2a0f31`). New helper
   `build_param_groups(model, weight_decay)` — 1D params (BN/LN
   scales) OR `.bias`-suffixed → no-decay; 2D+ weights → decay.
   Standard nanoGPT/KataGo pattern; pure hygiene, orthogonal to
   colony attractor.
2. **Cosine `eta_min` floor raised** to 5e-4 (from 2e-4) in
   `configs/training.yaml` (`a43d5eb`). Mechanism alignment with L47:
   prevent loss-of-plasticity at late-run when the colony attractor
   manifests; KataGo precedent (never drop below 0.5× peak).
3. **Policy entropy regularization** set to 0.005 (from 0.01) in
   `configs/training.yaml` (`03425de`). Mechanism alignment with L47:
   counter-pressure on policy collapse via entropy bonus. Wiring
   pre-existing (trainer.py:512 + losses.py:215 — `total = total -
   entropy_weight * entropy_bonus`, correct sign).

**Tests.** `tests/test_optimizer_param_groups.py` (4 unit tests) +
`tests/test_inv_optimizer_param_groups.py` (5 INV pin tests). 9/9 PASS;
broader trainer/optimizer/losses regression 84/84 PASS (excluding
pre-existing minimax_bot ImportError, unrelated).

**Deliberately excluded (cite L47).**
- **EMA.** Smooths gradient flow but changes the self-play inference
  model; contaminates audit baseline + future ladder probes. Defer
  until Track A localizes the dominant lever and we want a clean
  baseline.
- **`beta2 = 0.95`.** L47 says recent gradients are colony-biased;
  shortening AdamW's gradient memory window AMPLIFIES the bias.
  Counter-indicated.
- **WDL migration.** Same cost profile as A2 (re-pretrain, value-head
  re-arch). V-FU2-C falsification of architecture-only fixes makes
  this a low-priority commitment until Track A/B/C surface a
  source-targeted lever.

### L47 (adopted to master from the parked FU-2 branch)

**L47 — value-head architectural inductive-bias fixes are insufficient
without loop-side intervention.** A2's starting-point inversion
(extension-favouring anchor V_spread −0.508) was dragged through zero
to +0.13 colony-favouring within ~1000 training steps under the
§S178/§S180b 3-knob recipe — same end-state as pre-A2 reached
monotonically from +0.617, from the SIGN-OPPOSITE direction. The
training loop's value-target dynamics (selfplay outcome × MCTS
visit-count CE × corpus mix) override the architecture's structural
starting point by a margin of ≥0.5 in V_spread magnitude per 1000
steps (T3-bank metric; see L48 for revised magnitude). The next surface
IS the loop — PSW / refresh hook / value-target perturbation / class-
weighted gradient. **Source:** parked branch
`phase4.5/s181_fu2_a2_arch` audit doc
`audit/structural/08_fu2_a2_sustained_ladder.md`.

### L48 — V_spread metric is partially T3-bank-specific (CONFOUND)

A3 measured Pearson r=0.27 between V_spread(T3) and V_spread(alt
corpus-derived bank) across the FU-1.5 ladder (anchor + 10 ckpts at 2k
cadence). Anchor V_spread on T3 is +0.617; on alt bank +0.212 (~34%).
Alt-bank trajectory range [+0.016, +0.258] vs T3 range [−0.118,
+0.617] — alt magnitudes ~3× smaller throughout the ladder. **L47's
≥1.0/1000-step magnitude is revised downward** by the same factor when
read on real corpus positions: actual per-step pull ≈ +0.33/1000 steps,
not ≥1.0. The mechanism stands; the absolute amplitude was amplified
by T3 bank's specific synthetic structures. **Operational corollary:**
V_spread canary should be augmented to compute on BOTH banks; T3 stays
as historical reference, alt grounds the metric in corpus reality.

### L49 — no single corpus-level source confirmed dominant; multi-source attractor

Across A1 + A4 + A5, no single hypothesis reached its pre-registered
confirmation threshold. H-PRETRAIN is the largest contributor (+0.157
asymmetry, ~25% of anchor V_spread direction), H-BOT contributes
+0.078 (~13% of anchor), H-CE-STRENGTH provides a 1.21× per-colony-
sample gradient multiplier. Sum of source-attributable per-step pull
upper bounds (~+0.07/step) exceeds the observed alt-bank-corrected
movement rate (~+0.0003/step) by ~200× — meaning sources have
**head-room**; the actual per-step rate is gated by optimizer
damping + neither-class counter-currents (61% bot, 63% pretrain) +
unmeasured self-play buffer dynamics (Track B). **Implication:** no
single corpus-side knob fixes the attractor. The next wave should be
multi-lever (PSW + refresh hook + per-class target temperature) +
Track B instrumentation, NOT a single targeted intervention.

---

## §S181-AUDIT Wave 1 — Track B infra + C-LITE-1 + Track D + REAL_RUN_RECIPE

**Wave.** Capstone diagnostic before the next sustained run. Closes the
Track A "no single source" cliff (L49) by landing Track B's
instrumentation infra (operator-mediated run pending), capturing v7full
as a real-run anchor candidate, and synthesizing a parameterized recipe
gated on Track B's V-B verdict. FF-merged to master 2026-05-23 as 4
commits `7cd0dc0..71268ab` across 4 short-lived branches (PR-C / Track
B / Track C-LITE / Track D — all pruned post-merge). REVIEW pass (Opus
4.7 fresh-context, scopes A-F all PASS).

### Landed commits

- `7cd0dc0` PR-C: dual-bank V_spread canary (T3 + alt per L48). 412 LOC
  net, INV pin extended to alt fixture SHA `a68b810f…20a20ff`. 31/31
  PASS (8 INV + 23 unit). Back-compat preserved for legacy single-bank
  payloads. No hot-path touch.
- `3201c39` Track B: B1 per-source gradient-norm attribution
  (`hexo_rl/training/track_b_attribution.py`, ~3× cost via 3
  retain_graph autograd.grad slices) + B2 buffer position-class
  snapshot (`hexo_rl/training/track_b_buffer_snapshot.py`, ckpt-cadence
  hook in StepCoordinator) + B3 post-run trunk feature drift script +
  `v6_botmix_s181_track_b.yaml` variant + launch-and-analysis spec.
  43/43 PASS (12 Track B + 31 PR-C/INV). NOT LAUNCHED — operator-
  mediated vast launch (~6 h, ~$1.50) blocks V-B verdict aggregation.
- `b93d994` Track C-LITE-1: v7full anchor dual-bank V_spread (T3
  +0.2171 / alt +0.4078). Verdict C-LITE-1-A — encoding regression
  candidate CONFIRMED; v7full's alt-bank V_spread is ~2× v6's. T3
  borderline at +0.017 over SOFT-ABORT is L48-explained (T3 calibrated
  on v6's value head; alt is the corpus-grounded reference). C-LITE-2
  v6w25 stock probe DEFERRED — C-LITE-1's answer + REAL_RUN_RECIPE
  conditional path covers the encoding question for the v6 family.
- `47d658e` REAL_RUN_RECIPE: 7-section synthesis with parameterized
  Wave 2 lever stack gated on V-B-{A,B,C,D,E}. Anchor primary
  `bootstrap_model_v7full.pt`; encoding `v7full`; success criteria
  RR-G1..RR-G6 (LITERAL L13); compute budget ~$5 + 2-3 days dev.
- `71268ab` Track D: pipeline regression audit §150→§S178+ (596
  lines). Smoking-gun rank: (1) bot-corpus value-target imprint +
  staleness, (2) pretrain corpus colony pull × recency_weight=0.75,
  (3) ply_cap_value=0.0 × full_search_prob=0.5 cross, (4) bot-corpus
  staleness × outcome feedback, (5) recency_weight ×
  selfplay-buffer compounding (INCONCLUSIVE).

### Track B — B4 run pending (operator-mediated)

Run spec at `audit/structural/track_b/B_launch_and_analysis_spec.md`:
3000-step instrumented run on vast 5080 with `v6_botmix_s181_track_b`
variant, `bootstrap_model_v6.pt` anchor, dual-bank canary firing every
500 steps. Pre-registered V-B-{A..E} decision tree:

| ID | rule | downstream |
|---|---|---|
| V-B-A | one source ≥60% of total grad pull | source-targeted lever |
| V-B-B | all three sources 25-45% | multi-source damping (EMA + 2-stone aux + per-class target temp) |
| V-B-C | buffer colony-heavy ≥50% by step 2k | refresh hook + EMA priority |
| V-B-D | trunk centroids collapse ≥50% by step 1k | aux heads forcing trunk discrimination |
| V-B-E | none match | escalate, no real-run launch |

V-B-{verdict} feeds REAL_RUN_RECIPE §3 conditional lever pick. No L50+
banked this wave — Wave 2 will bank lessons from the actual B4 run +
verdict application.

### Wave 2 — operator decision point

`audit/structural/REAL_RUN_RECIPE.md` §7 captures the Wave 2 sequence:
B4 run → V-B aggregation → conditional lever pick → EMA always; 2-stone
aux iff V-B-D; per-source lever per V-B → pre-launch smoke (~$1.50,
~6 h) → main 100k-step run on v7full anchor (~$3, ~14 h). Total Wave 2
estimated cost ~$5 + 2-3 days dev. Operator decides launch timing; do
NOT auto-launch.

### Branches pruned post-merge

`phase4.5/s181_pr_c_dual_bank`, `phase4.5/s181_track_b_instrumented`,
`phase4.5/s181_track_c_lite`, `phase4.5/s181_track_d` — all FF-merged
and deleted. No push of these branch names; master HEAD `71268ab`
carries every commit.

---

## §S181-AUDIT Wave 2 — refresh-hook-less lever stack peak-and-collapse

**Status.** Lever stack `uniform_self` (v7full anchor + EMA decay 0.999 +
per-class target temperature on selfplay slice `sample_rate=0.20`) hit
project-record SealBot WR **33% peak @ step 20k** (1.9× §150 baseline
17.4%), then monotonic decline 11% @ 30k → 5% @ 40k. HARD-ABORT @ step
47642 on RR-G3 / §S180b 8% threshold breach. Wave 2 PROVES anti-colony
lever stack works short-term but **fails sustained without bot corpus
refresh**. Phase 4.5 remains BLOCKED; Wave 3 with refresh hook +
sliding-window WR gate + per-class temp scope revision is the successor.

### Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (ssh6.vast.ai:13053) |
| branch | `phase4.5/s181_wave2_lever_vba_selfplay` `54bd9da` (origin, NOT master) |
| variant | `v7_real_run_main` (`configs/variants/v7_real_run_main.yaml`) |
| anchor | `bootstrap_model_v7full.pt` SHA `568d8a33…d61e8e98` |
| encoding | v7full (single-window 19×19, 8 planes) |
| iterations target | 100 000; **actual 47 642** (operator-aborted) |
| run_id | `aad5e948c4b94bf395daeddbb57415b9` |
| launch / abort | 2026-05-24 00:09 UTC / 2026-05-25 00:05 UTC (~24h wall) |
| cost | ~$5 (1A B4 $1.30 + 4B smoke $0.70 + 5 main ~$5 = **~$7 total Wave 2**) |
| checkpoints saved | 23 @ 2k cadence + best_model.pt (last promotion step 30k) |
| events captured | 23 dual-bank canary fires; 4 SealBot evals; 3 best_model promotions |
| canonical deliverable | `reports/track_b_main/checkpoints/checkpoint_00020000.pt` (33% peak) |
| key audit doc | `audit/structural/wave2_real_run_analysis.md` |

Lever stack vs §S180b 3-knob recipe:
- encoding v6 → **v7full** (anchor match; C-LITE-1 verdict)
- EMA of weights (decay 0.999, every 10 optimizer steps), dispatched
  through `Trainer.inference_state_dict`
- per-class target temperature on selfplay slice (T_colony=1.5, others
  1.0, pretrain slice untouched, `selfplay_sample_rate=0.20` perf opt
  per smoke L48 throughput recovery)

### Path through V-B verdict (Option D)

B4 LITERAL verdict was **V-B-E** (no clean match): per
`audit/structural/track_b/B_aggregation.md`:
- V-B-A NO — uniform_self mean share 0.563 < 0.60 single-source gate
- V-B-B NO — pretrain mean 0.092 outside [0.25, 0.45] band
- V-B-C NO — buffer colony_frac stable 9-10% (no feedback loop)
- V-B-D NO — trunk inter-centroid 84% of anchor at step 1000

`B_verdict_synthesis.md` surfaced 4 routing options + recommended
**Option D** (partial smoke before full real-run commit) on the
near-miss V-B-A reading: selfplay-family share (uniform_self + recent)
= 90.8% — single-source partition missed by 3.7pp. Operator took
Option D. Smoke S-A PASS authorized main launch. Documented in
B_verdict_synthesis.md as "respects both the literal V-B-E *and* the
near-miss mechanism interpretation".

### SealBot WR trajectory

| step | wr_sealbot | CI95 | wr_anchor | colony_wins_sealbot | promoted |
|---:|---:|---|---:|---:|:---:|
| 10 000 | 24.0 % | [16.7, 33.2] | 66.0 % | 3 | ✓ |
| **20 000** | **33.0 %** | [24.6, 42.7] | 61.0 % | 6 | ✓ |
| 30 000 | 11.0 % | [6.3, 18.6] | 65.0 % | 4 | ✓ |
| **40 000** | **5.0 %** | **[2.2, 11.2]** | **70.0 %** | 5 | ✗ |

Peak step 20k; monotonic decline thereafter. Step-40k 5% below RR-G3
13% gate AND below §S180b 8% HARD-ABORT threshold. promoted=False at
step-40k = bootstrap_floor 0.45 gate failed = training producing
weaker models than current best_model.

### L34 anchor↑/sealbot↓ divergence

| transition | anchor Δ | sealbot Δ | L34 fires? |
|---|---:|---:|:---:|
| 10k → 20k | -5 pp | +9 pp | INVERSE (healthy) |
| 20k → 30k | +4 pp | -22 pp | **YES (1st)** |
| 30k → 40k | +5 pp | -6 pp | **YES (2nd)** |

Classic colony-attractor capture signature: model becomes *relatively
stronger* vs frozen anchor while *absolutely weaker* vs SealBot.
SOFT-ABORT trigger requires 5 consecutive — 2 observed when run was
killed.

### Dual-bank V_spread canary

alt-bank held +0.18–0.30 throughout 46k steps (well above +0.07
sustained gate). T3 started +0.27, oscillated +0.03–+0.09 through
step 28k, then collapsed to -0.26 by step 46k.

**Critical**: alt V_spread stayed above gate the entire run yet
eval-measured wr_sealbot collapsed 33→5%. The held-out V_spread canary
failed to track the actual performance collapse. (L50)

### Pre-registered success criteria — verdict

| ID | criterion | result |
|---|---|:---:|
| RR-G1 | T3 ≥ +0.20 sustained 0→50k | **FAIL** (crossed below at step 1000 in smoke; collapsed deep negative in main) |
| RR-G2 | alt ≥ +0.07 sustained 0→50k | PASS (held +0.18+ throughout) |
| RR-G3 | SealBot WR ≥ 13% @ step 30k (Wilson95 LB ≥ 9%) | **FAIL** (11% / LB 6.3%) |
| RR-G4 | SealBot WR ≥ 18% @ step 50k | **FAIL** (extrapolated from 5% @ 40k) |
| RR-G5 | colony_a < 50/100 in eval rounds | PASS (colony_wins_sealbot 3-6 throughout) |
| RR-G6 | L34 anchor↑/sealbot↓ clean | **FAIL** (2 consecutive instances) |

4 of 6 RR-G* FAIL. RR-G2 + RR-G5 PASS isolate the L50 mechanism:
held-out V_spread + colony-policy gates can both PASS while
eval-measured tactical WR collapses.

### Mechanism diagnosis

Three candidate mechanisms (ranked):

**M1 — Bot corpus opportunistic fit + degeneration (HIGH).**
21,899-position static bot corpus at `bot_batch_share=0.30` exposes
~77 SealBot positions/batch. By step 20k the corpus has been re-
encountered ~70 times (batch 256 × 30 % × 20k steps = 1.54 M bot
positions / 21,899 corpus size); distributional decay vs the
evolving model grows with policy-distance after corpus saturation.
Past step 20k: selfplay drifts model policy off the corpus
distribution → 30% bot-batch becomes off-distribution noise → fit
decays. Track D C4 (bot staleness × outcome feedback) candidate now
CONFIRMED as the dominant Wave 2 failure mechanism.

**M2 — Per-class temp dilution over selfplay slice (MEDIUM).**
T_colony=1.5 softens visit-count CE targets on 20% of selfplay
colony-classified rows. As selfplay buffer accumulates sharper
late-game policies, softening the model's own best moves degrades
tactical learning. Combined with M1 drift, compounds the post-peak
collapse. NOTE: alt V_spread stayed high → M2 doesn't hurt value-head
discrimination; the proxy for M2 damage is policy-sharpness data
not captured in this Wave 2 instrumentation.

**M3 — EMA averaging artifact (LOW, ruled out).** Smoke (Stage 4B,
also EMA-enabled) was clean S-A PASS at step 3000. Collapse appears
at step 20-40k, well past EMA's effective warmup. EMA decay 0.999
is monotonically a smoother (can lag, doesn't actively degrade).

**Most likely: M1 + M2 compound.** Bot-corpus drift is dominant
driver; per-class temp amplifies tactical degradation as selfplay
slice grows past corpus-dominated early window. Wave 3 must address
BOTH: refresh hook (M1) + per-class temp scope revision (M2).

### Wave 1 vs Wave 2 reframing of Track D C4

REVIEW (Stage 1A) surfaced an apparent tension between Wave 1 B4 and
Wave 2 main on the bot-corpus mechanism:

- **Wave 1 B4** (3000 steps, B_track_d_xref.md): C4 ranked "small
  absolute magnitude" — bot corpus gradient-pull share 0.092 mean,
  smallest of three sources.
- **Wave 2 main** (47642 steps, M1 above): C4 confirmed as DOMINANT
  failure mechanism via staleness amplification.

Both findings are correct AT THEIR TIME WINDOW. B4 measured
gradient-pull share at step 0-3000 when the corpus is in-
distribution; the static-staleness mechanism doesn't yet bite. Wave 2
main captured the 20k+ horizon where the model has drifted off the
static distribution and the 30% bot-batch share becomes off-
distribution noise. Same channel, different time windows, two
metrics measuring different aspects (instantaneous pull vs cumulative
distributional decay). The Track D ranking is NOT inverted — it is
extended with a time dimension B4 did not measure.

### Falsified Hypotheses Register additions

- **Static bot corpus alone is a sufficient anti-colony anchor for
  sustained training past peak fit point.** FALSIFIED by Wave 2:
  21,899-position static SealBot-vs-v6 corpus held the colony
  attractor at bay through step 20k (33% SealBot peak); past step 20k
  the model's policy drifted off the corpus distribution and the
  30% bot batch share became off-distribution noise. Future runs need
  dynamic regeneration of the bot corpus against the current model.

- **alt V_spread + dual-bank canary alone is a sufficient gate for
  real-run quality.** FALSIFIED by Wave 2 (L50): alt V_spread stayed
  +0.18–0.30 throughout 46k steps (well above +0.07 sustained gate)
  while wr_sealbot collapsed 33% → 5%. Value-head discrimination on
  fixed held-out banks is not a sufficient proxy for actual
  selfplay/eval performance. Future runs must ALSO gate on sealbot WR
  sliding-window trajectory (Wave 3 hard-abort triggers).

### L50 — alt-bank V_spread is necessary but not sufficient for sustained eval quality

**Rule.** Held-out value-head discrimination metrics (T3 + alt bank
V_spread) can both PASS while training-loop policy quality
deteriorates. The metric measures value-head separation on a fixed
position bank; it does not capture the model's *policy* on actual
self-play / eval games.

**Why.** Wave 2 evidence: alt V_spread sustained +0.18–0.30 across
46k steps (well above the +0.07 sustained gate) yet wr_sealbot
collapsed 33% (step 20k peak) → 5% (step 40k). The value head
remained discriminative on the static bank while the policy
deteriorated tactically on live SealBot games.

**How to apply.** Sustained run gates must include sliding-window
SealBot WR trajectory tracking as a hard-abort lever, not advisory.
alt-bank V_spread remains useful as an early-warning signal but
cannot stand alone. L48 framing (alt is the corpus-grounded
reference vs T3 synthetic) refined: alt-bank is corpus-grounded but
only for the value-head sub-task, not the policy-head sub-task.

### L51 — Bot-corpus staleness predicted as Track D C4 — Wave 2 confirms

**Rule.** A static bot corpus regularization signal has an effective
training-lifetime bounded by the ratio (model drift rate) ÷
(corpus-replay rate). Past that lifetime, the signal becomes
off-distribution noise and contributes negatively rather than as
anti-colony anchor.

**Why.** Wave 2 lever stack used a 21,899-position SealBot-vs-v6
corpus (`bot_corpus_s178_sealbot_vs_v6.npz`) at `bot_batch_share=0.30`.
By step 20k the corpus had been re-encountered ~70 times (batch 256
× 30% × 20k steps = 1.54 M bot positions / 21,899 corpus size); the
model had imprinted the SealBot tactical distribution → peak 33% WR.
Past step 20k the model's selfplay policy drifted off the corpus's
position distribution → the 30% bot slice became increasingly
off-distribution → fit decayed monotonically to 5% by step 40k.

**How to apply.** ANY future sustained run with a static bot corpus
must specify (a) maximum effective training-lifetime, (b) refresh
trigger, (c) refresh cooldown. Wave 3 `bot_corpus_refresh.enabled=true`
is the design-named remedy (refresh against current EMA model on
each best_model_promotion + 5pp WR delta, cooldown 5k steps,
max_regens 19 per 100k run).

### L52 — Per-class target temperature on selfplay slice over-softens late tactical learning when bot corpus drifts off-distribution

**Rule.** Per-class CE-target softening on the SELFPLAY slice
reduces gradient magnitude on the model's strongest moves. This is
benign while selfplay positions are dominated by early-game /
corpus-class shapes, but degrades tactical sharpness once selfplay
buffer accumulates late-game shapes the model is sharpening on
itself. Combined with a stale anti-colony signal (L51), the model
loses tactical strength faster than the static bot corpus can
correct.

**Why.** Wave 2 used T_colony=1.5 with `selfplay_sample_rate=0.20`
on selfplay slice. M2 mechanism: softening visit-count CE on
colony-classified selfplay rows attenuates the model's own best-move
signal. In the bot-corpus-fresh window (step 0-20k) this is balanced
by the 30% bot corpus pulling toward SealBot's tactical distribution.
Past corpus exhaustion (step 20k+ per L51), the per-class temp
softening is the dominant remaining anti-colony pressure on selfplay
rows — and it's tactical-softening, not target-replacement, so it
actively de-sharpens rather than re-pointing.

**How to apply.** Per-class target temperature should apply only to
slices where the model is NOT learning its own play (pretrain + bot
slices). Drop the selfplay slice from per-class temp. Wave 3
implementation adds `apply_to_selfplay: false` flag to the existing
`apply_to_pretrain: true` companion. Combined with Wave 3 refresh
hook (L51), the levers separate: bot corpus stays current via
refresh (anti-colony pressure), per-class temp targets only static-
distribution rows (pretrain + bot), selfplay slice learns its own
play unmodified.

### Wave 2 canonical deliverable preservation

`reports/track_b_main/checkpoints/checkpoint_00020000.pt` is the
project-record SealBot WR 33% snapshot. Preserved as historical
reference even if Wave 3 produces a stronger sustained model. Not
promoted, not anchor candidate — Wave 2 mechanism-evidence ckpt.
Operator decides post-Stage-1 whether to archive to
`reports/canonical_models/wave2_step20k_peak33pct.pt` for long-term
retention.

`best_model.pt` (step-30k promotion, 11% WR degradation point) +
`checkpoint_00010000/30000/40000.pt` retained at
`reports/track_b_main/checkpoints/` as the full trajectory ladder.

### Open handles (Wave 3 carries forward)

- Refresh hook activation per L51 (design at
  `docs/designs/s179c_bot_refresh_hook.md`)
- Sliding-window SealBot WR hard-abort gate per L50
- Per-class temp scope revision per L52 (`apply_to_selfplay: false`)
- REAL_RUN_RECIPE v2 update with new PRIMARY success criterion
  (rolling-mean SealBot WR ≥20% sustained 30k-50k vs current
  RR-G3+RR-G4 single-step gates)
- Compute budget ~$5 + 2-3 days dev (same as Wave 2 allocation)

### Branches

| branch | commit | status |
|---|---|---|
| `phase4.5/s181_wave2_ema` | `6ef4aad` | cherry-picked to master Stage 1C (commit `95624af`) |
| `phase4.5/s181_wave2_b4_analysis` | `814c4ef` | cherry-picked to master Stage 1C (commits `e973d1f`, `6fab8c9`) |
| `phase4.5/s181_wave2_lever_vba_selfplay` | `54bd9da` | KEEP as historical reference (origin push pending operator); audit docs cherry-picked to master Stage 1C (commit `96562fa`) |

Wave 2 lever code (per_class_target_temperature.py + variant configs)
stays on the lever branch — Wave 3 revises scope (L52) so the code
is reference-only.

---

## §S181-AUDIT Wave 3 — refresh hook + per-class temp scope flip plateau-then-collapse

**Status.** Wave 3 lever stack (v7full anchor + EMA + bot corpus refresh
hook per L51 + per-class target temp scope flip per L52 + L50 sliding-
window WR hard-abort gate) ran 50 000 steps across 3 sessions, then
hit L50 Trigger C auto-abort @ step 50k after wr_sealbot collapsed to
**2% @ step 45k**. Wave 3 plateaued the model at 16-25% wr_sb across
steps 10k-30k (5 evals) — a softer trajectory shape than Wave 2's
peak-then-monotonic-decline — but hit the SAME end-state colony
attractor (anchor 70% / sealbot 2% / 68pp divergence). Refresh hook +
per-class scope flip are necessary-not-sufficient against the
colony-attractor capture. Wave 4 design with structural levers (2-stone
aux / WDL / PSW) required. **Phase 4.5 remains BLOCKED.**

### Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (ssh6.vast.ai:13053) |
| branch | `phase4.5/s181_wave3_design` 4435f4d (pushed to origin) |
| variant | `v7_wave3_main` (with mid-run L50 widening @ 4435f4d) |
| anchor | `bootstrap_model_v7full.pt` SHA `568d8a33…d61e8e98` |
| encoding | v7full (single-window 19×19, 8 planes) |
| iterations target | 100 000; **actual 50 000** (L50 Trigger C auto-fire) |
| sessions | 3 (s1: cadence-fix-stop @ 11 746 / s2: L50-B-fire @ 40 000 / s3: L50-C-fire @ 50 000) |
| total wall | ~32 h |
| total cost | ~$8.30 (slight overrun of $8 hard cap on operator "continue to 50k regardless" override) |
| total games | ~25 000 |
| canonical archive | `reports/track_b_main_wave3/` (353 MB) |
| key audit doc | `audit/structural/wave3_real_run_analysis.md` (452 lines) |

Wave 3 lever stack vs Wave 2:
- EMA + v7full anchor (Wave 2 baseline, kept)
- **Bot corpus refresh hook ACTIVATED** per L51 (Wave 2 was disabled)
- **Per-class target temp SCOPE FLIP** per L52 (Wave 2 was selfplay-only;
  Wave 3 is pretrain+bot only, selfplay slice sharp tactical CE preserved)
- **L50 sliding-window WR hard-abort gate** active (Wave 2 didn't have this)

### SealBot WR trajectory (peak-and-collapse)

| step | wr_sb | wr_anc | wr_best | c_sb | promoted | elo |
|---:|---:|---:|---:|---:|:---:|---:|
| 5 000 | 16.0% | 60.0% | 52.0% | 2 | ✗ | 435 |
| **10 000** | **25.0%** | 62.0% | 58.0% | 3 | ✗ | 417 |
| 15 000 | 18.0% | 46.0% | 59.0% | 7 | ✗ | 428 |
| 20 000 | 16.0% | 68.0% | 55.0% | 3 | ✗ | 407 |
| 25 000 | 23.0% | 59.0% | 66.0% | 8 | **✓** | 377 |
| 30 000 | 20.0% | 65.0% | 65.0% | 11 | **✓** | 313 |
| 35 000 | 10.0% | 61.0% | 57.0% | 6 | ✗ | 212 |
| **45 000** | **2.0%** | **70.0%** | **69.0%** | 2 | **✓** | 397 |

Peak step 10k 25% (vs Wave 2 peak step 20k 33%). Plateau 16-25%
steps 10k-30k. Catastrophic collapse 23%→2% over steps 25k→45k. SAME
end-state attractor as Wave 2 (anchor 70% / sb ≤5%).

### L34 anchor↑/sealbot↓ divergence (3 fires)

| transition | anchor Δ | sealbot Δ | L34? |
|---|---:|---:|:---:|
| 15k → 20k | +22 pp | -2 pp | **YES (1st)** |
| 25k → 30k | +6 pp | -3 pp | **YES (2nd)** |
| 35k → 45k | +9 pp | -8 pp | **YES (3rd)** |

3 L34 fires across trajectory. Step 45k = final form: 70%/2% = 68pp
divergence (Wave 2 peak was 65pp at step 40k). Same colony-attractor
end-state.

### L50 hard-abort fires (mechanism validated end-to-end)

- **Fire #1 — Trigger B @ step 40 000** (session 2): "SealBot WR 10.0% <
  peak 23.0% × 50% past step 25,000 — Wave-2-style collapse". Operator
  widened `wr_collapse_from_peak_ratio: 0.5 → 0.25` and resumed from
  ckpt_00040000.pt.
- **Fire #2 — Trigger C @ step 50 000** (session 3): "SealBot WR 2.0% <
  5% past step 15,000 — §S180b-style early death". Clean session_end.

Both fired AS DESIGNED. Stage 2B's L50 gate is validated end-to-end.

### Refresh hook validation (4 cycles end-to-end)

| cycle | requested step | swap step | n_pos_after |
|---:|---:|---:|---:|
| 1 | 5 000 | 10 000 | 6 724 |
| 2 | 15 000 | 20 000 | 7 453 |
| 3 | 30 000 | 35 000 | 8 006 |
| 4 | 45 000 | 50 000 | 9 932 |

Mechanism works perfectly. Atomic NPZ swap + hot-reload + corpus growth
across cycles (6.7k → 9.9k positions as EMA model strengthens + sustains
longer games). Reload_sec ≤ 6s each. The mechanism is NOT defective —
the verdict is that it's INSUFFICIENT against the attractor.

### Mechanism diagnosis (M1 + M2 + M3 compound)

**M1 — Refresh hook insufficient (high).** 30% bot-batch-share + dynamic
EMA-anchored fresh corpus IS overpowered by the 70% selfplay-buffer
share that drives the gradient toward colony shapes. Mechanism works,
share is wrong.

**M2 — Per-class temp scope flip insufficient (high).** Wave 2's M2
mechanism (per-class temp on selfplay slice over-softens tactical CE)
WAS addressed by L52 scope flip (apply_to_selfplay: false). But the
attractor operates ABOVE the per-class CE level. Wave 3 STILL collapsed.
Preserving selfplay CE sharpness ≠ preserving the right policy direction.

**M3 — best_model rotation amplifies the attractor (moderate).** Promotion
at step 25k, 30k, 45k. Step 45k promotion happened WHILE wr_sb crashed
to 2% — the promotion criteria (wr_best ≥ 0.55 + bootstrap_floor 0.45)
reinforce colony exploitation via positive feedback on internal metrics
that don't penalize anchor-exploit.

### L53 — Refresh hook + per-class temp scope flip insufficient against the colony attractor

**Rule.** A 30%-batch-share dynamic-refreshed bot corpus + per-class
CE softening scoped to static (pretrain+bot) rows is INSUFFICIENT to
prevent the colony-attractor capture in v7full sustained training.

**Why.** Wave 3 ran 50k steps with both levers active. wr_sb peaked at
25% (step 10k), oscillated 16-25% across steps 10k-30k, then collapsed
to 2% by step 45k. Same end-state attractor as Wave 2 reached via
different trajectory shape. Refresh hook cycles complete cleanly but
the fresh corpus signal is overpowered by selfplay-dominated gradient.

**How to apply.** Wave 4+ must employ STRUCTURAL interventions, not
RATIO interventions. Candidates: 2-stone opponent-reply aux head
(addresses M3), WDL value-head migration (changes value-target
structure), KL-weighted buffer writes, PSW (s179b parked design),
class-weighted gradient scaling (vs CE softening). The refresh hook +
per-class temp scope flip should REMAIN as defensive substrate but a
fundamental mechanism change is needed atop them.

### L54 — Trigger B peak×0.5 threshold catches collapse late

**Rule.** L50 Trigger B fires AFTER trajectory drops to half its peak —
by then the collapse is already deep.

**Why.** Wave 3 fired Trigger B at step 40k drain (wr_sb 10%, peak 23%,
threshold 11.5%) — but the trajectory crashed 23% (step 25k) to 10%
(step 35k) BEFORE the fire. ~15k steps of collapse training before
auto-abort. Trigger C fired even later (step 50k after wr_sb hit 2%).

**How to apply.** Future runs should add a derivative trigger: "rolling-
mean WR drops ≥ Xpp across 2 consecutive evals past step 20k". X=5pp
would have caught Wave 3 at step 35k (rolling-mean 21.5% → 15% = 6.5pp
drop). Or tighten Trigger B's peak ratio earlier (ratio 0.7 + min_step
20k catches earlier).

### L55 — Wave 3 plateau-then-collapse is SAME L34-attractor with delay, not new mechanism

**Rule.** A long sustained mid-WR phase (plateau) is NOT evidence of
attractor break. It's consistent with the SAME attractor + lever-stack-
induced inertia before the wr_sb-measurable collapse.

**Why.** Wave 2 and Wave 3 end at the SAME L34 signature (anchor ~70%
/ sealbot ≤5% / 65pp divergence). The end-state is identical. Wave 3
lever stack delayed wr_sb-measurable manifestation by ~10-15k steps but
didn't break the attractor.

**How to apply.** Wave 4+ design should NOT chase a different trajectory
shape (longer plateau ≠ better if collapsing at 2%). Design for a
different ATTRACTOR — different value-head / policy-head / value-target
structure with no colony-exploit-of-anchor-bias attractor. WDL, 2-stone
aux, or PSW change WHAT THE MODEL OPTIMIZES, not just the
gradient-share of the existing optimization.

### Falsified Hypotheses Register additions

- **Refresh hook + per-class temp scope flip is sufficient to prevent
  colony-attractor capture in v7full sustained training.** FALSIFIED by
  Wave 3 — wr_sb collapsed to 2% by step 45k despite both levers active.
- **Plateau (long sustained mid-WR phase) is a positive sign of attractor
  break.** FALSIFIED by Wave 3 — model plateaued 16-25% wr_sb across
  10k-30k then catastrophically collapsed to 2%.
- **best_model promotion is a reliable signal of model improvement
  toward Phase 4.5 readiness.** FALSIFIED by Wave 3 — best_model promoted
  AT step 45k (wr_best 69%) WHILE wr_sb crashed to 2%. Promotion gate
  rewards anchor-exploit, not anti-colony improvement.

### Wave 3 canonical deliverable preservation

No single project-record snapshot. Candidates:
- `checkpoint_00010000.pt` — Wave 3 peak 25% wr_sb (CI wide; not promoted)
- `checkpoint_00025000.pt` — first promotion + 23% wr_sb
- `checkpoint_00045000.pt` — **colony-attractor reference** (70%/2% — useful
  for Wave 4 mechanism ablation work)

Wave 2's `wave2_step20k_peak33pct.pt` remains the project-record peak
SealBot WR snapshot. Wave 3 archive at `reports/track_b_main_wave3/`
holds the trajectory for future analysis.

### Wave 4 escalation path (operator decides priority)

Per dispatcher §5C routing (PRIMARY all FAIL → "Mechanism wrong"):

1. **2-stone opponent-reply aux head** (V-B-D conditional, never tested
   in sustained context). Highest priority — forces trunk to discriminate
   on-policy reply patterns, addresses M3.
2. **WDL value-head migration** (parked since §S178; A2 falsified
   arch-only fix, BUT Wave 3 shows loop-side levers ALSO insufficient
   → arch + loop combined hypothesis worth testing).
3. **PSW (Policy Surprise Weighting)** — design `s179b` parked. Penalizes
   high-KL transitions in selfplay buffer writes.
4. **Class-weighted gradient scaling** (different mechanism class than
   per-class CE softening Wave 3 already tested).

L53/L54/L55 + `audit/structural/wave3_real_run_analysis.md` are the
Wave 4 design starting point.

### Cross-references

- `audit/structural/wave3_real_run_analysis.md` — full analysis (452 lines)
- `audit/structural/wave3_smoke.md` — Stage 3 smoke WS-A PASS-WITH-NOTES
- `audit/structural/wave3_launch_readiness.md` — Stage 2 close-out
- `audit/structural/wave2_real_run_analysis.md` — Wave 2 baseline (L50/L51/L52)
- `audit/structural/REAL_RUN_RECIPE.md` — Wave 3 success criteria
- `docs/designs/s179c_bot_refresh_hook.md` — refresh hook design (validated)
- `docs/designs/s179b_policy_surprise_weighting.md` — PSW (Wave 4 candidate)
- `reports/track_b_main_wave3/` — 353 MB local archive (ckpts + logs + events JSONL)

### Branches (Stage 4 → Stage 6 disposition)

| branch | commit | status |
|---|---|---|
| `phase4.5/s181_wave3_design` | `4435f4d` | active; pending Stage 6 REVIEW → master merge |

Wave 3 design branch has 10 commits ahead of master: Stage 2A refresh
hook + Stage 2B WR hard-abort gate + Stage 2C per-class temp scope +
Stage 2D REAL_RUN_RECIPE v2 + Stage 2E launch readiness + Stage 3A
smoke variant + Stage 3C smoke audit + Stage 4A main variant + mid-run
yaml widening + this sprint-log entry.

---

## §S181-AUDIT Wave 4 — subtract-the-variable + multi-aux suite

**Status.** Both Track 4A (subtract bot mix) and Track 4B (multi-aux
density on Wave 3 lever stack) ran sustained sessions on vast 5080
and SIGINTed early when verdict patterns landed. Track 4A peaked
at step 5k 19% and collapsed to 11% by step 10k (W4A-B verdict
LITERAL). Track 4B peaked at step 10k 23% and collapsed to 11% by
step 15k (W4B-B verdict LITERAL). Both reached ~11% wr_sealbot by
step 12-15k via different paths — bot mix removal accelerated the
collapse (Track 4A), multi-aux density delayed it ~5k steps but
didn't prevent it (Track 4B). **Colony-attractor mechanism lives
downstream of all config + density levers tested in Waves 1-4.**
Wave 5 strategic reckoning required: value-target propagation
(TD-λ / n-step), WDL 3-class head, or game-theoretic regularization.
**Phase 4.5 remains BLOCKED.**

### Run identity

| field | Track 4A | Track 4B |
|---|---|---|
| branch | `phase4.5/s181_wave4_subtract` 5b7f85e | `phase4.5/s181_wave4_multiaux` 0523147 |
| variant | `v7full_baseline_minus_bot` | `v7full_wave4_multiaux_w4ac` |
| anchor | `bootstrap_model_v7full.pt` SHA `568d8a33…` | (same) |
| iter target / actual | 60 000 / ~12 000 (SIGINT) | 60 000 / ~15 500 (SIGINT) |
| run_id | `1b8c649a…` | `8e4568c6…` |
| wall | ~7 h | ~9.5 h |
| spend | ~$3 | ~$3 |

Total Wave 4 spend ~$6 (under $7 cap).

### Track 4A subtract-the-variable verdict (W4A-B LITERAL)

Three DELTAs vs Wave 3 main: `bot_batch_share 0.30→0.0`,
`bot_corpus_refresh.enabled true→false`, `per_class_target_temperature.enabled
true→false`. Preserved Wave 2-3 hygiene (EMA + entropy 0.005 +
eta_min 5e-4 + PR-B param-group + L50 hard-abort STRICT thresholds).

| step | wr_sb | wr_anchor | wr_best | col_anchor | promoted | elo |
|---:|---:|---:|---:|---:|:---:|---:|
| 5 000 | **19.0%** | 57.0% | 59.0% | 40 (70%) | ✗ | 448 |
| 10 000 | **11.0%** | 53.0% | 68.0% | 40 (75%) | ✓ | 382 |

Δ 5k→10k: wr_sb -8pp, wr_anchor -4pp, wr_best +9pp, elo -66. BOTH
anchor and sealbot DECLINING (broader value-head degradation, NOT
classic L34 anchor↑/sealbot↓). Colony share 70→75% creeping. Audit
doc `audit/structural/wave4_track_a_subtract.md`.

### Track 4B multi-aux density verdict (W4B-B LITERAL)

Parent v7_wave3_main (KEEPS bot mix + refresh hook + per-class temp).
Multi-aux density bumps in training.yaml: sigma2 Huber 0.1 (NLL
formulation diverged at σ²→0; switched to Huber-on-squared-error per
4B-impl-5 d80de72), ownership 0.1→0.2, threat 0.1→0.2, ply_index
0.0→0.1 (NEW 4B-impl-3 head).

| step | wr_sb | wr_anchor | wr_best | col_sb | col_anchor | promoted | elo |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 5 000 | 20.0% | 64.0% | 62.0% | 4 (20%) | 50 (78%) | ✓ | 465 |
| 10 000 | **23.0%** | 62.0% | 61.0% | 3 (13%) | 41 (66%) | ✓ | 362 |
| 15 000 | **11.0%** | (~62.5 partial) | — | 5 (45%) | — | — | — |

Δ 5k→10k: wr_sb +3pp, col composition decreasing (78→66 anchor,
20→13 sb) — initially looked W4B-A-trending. Δ 10k→15k: wr_sb -12pp,
col_sb composition jumped 13→45% (homogenization confirmed). Audit
doc `audit/structural/wave4_track_b_sustained.md`.

### Cross-run shape comparison

| step | Wave 3 wr_sb | Track 4A wr_sb | Track 4B wr_sb |
|---:|---:|---:|---:|
| 5 000 | 16% | 19% | **20%** |
| 10 000 | **25%** (peak) | 11% (collapsed) | 23% (peak) |
| 15 000 | 18% | (SIGINT step 12k) | 11% (collapsed) |

Track 4B's peak (23%) ≈ Wave 3's peak (25%). Track 4A's peak (19%)
is the only data point where "no bot mix" started ahead before
accelerating its own decline. Multi-aux density configurations
shift WHEN the collapse happens, not WHETHER it happens.

### Lessons (L56-L60 banked)

**L56 (Wave 4 Track 4A).** Bot mix is NOT the load-bearing failure
variable in the colony-attractor mechanism. Removing it produces a
FASTER decline than keeping it (Track 4A step-10k 11% vs Wave 3
step-10k 25%). The §S178+ hypothesis ("bot mix introduces the
colony") is inverted or non-monotonic: bot mix may DELAY or attenuate
the colony pattern; removing it accelerates the mechanism.

**L57 (Wave 4 Track 4A).** Track 4A peaked at step 5k (19%) not step
10k like Wave 3 (25%). Without bot mix, the trainer's
`training_steps_per_game=2.0` rate forces heavy reuse of early
selfplay buffer, causing premature exposure to bootstrap distribution
drift. Recipe sensitivity to selfplay supply rate.

**L58 (Wave 4 Track 4B).** Multi-aux density (sigma2 Huber +
ownership 0.2 + threat 0.2 + ply_index 0.1) DELAYS the colony
attractor by ~5k steps but does NOT prevent it. Peak shifts from
Wave 3 step 10k 25% to Track 4B step 10k 23% (essentially same
magnitude). The KataGo aux-density hypothesis is FALSIFIED for
HeXO's colony attractor.

**L59 (Wave 4 close).** The colony attractor mechanism is INSENSITIVE
to: bot mix presence/absence, refresh hook, per-class target
temperature, multi-aux density, EMA/entropy/PR-B hygiene. The
mechanism lives DOWNSTREAM of all currently-tested levers — in the
training objective itself (value head + target propagation), trainer
step structure, or MCTS+selfplay interaction. Wave 5 must operate
at this boundary.

**L60 (Wave 4 Track 4B).** Colony composition (col_sb%, col_anchor%)
trajectory SHAPE predicts the W4B verdict before raw wr_sb does.
Track 4B step 5→10k showed composition DECREASING (positive signal),
but step 15k composition shot up (sb 13→45%) AS the wr_sb decline was
confirmed. Watch composition deltas as leading indicator of pattern
homogenization. (Per operator's colony-framing memo
`feedback_colony_is_meta_not_kill_signal`: colony presence is not a
kill signal; colony GROWTH + L34 + stride5 spike is.)

### Multi-aux infrastructure SHIPPED (regardless of verdict)

Even though the science hypothesis was falsified, the Wave 4 implementation
work is permanently useful and landed on master:

- `4B-impl-5` d80de72: sigma2 Huber-on-squared-error formulation
  (replaces Gaussian NLL that diverged at σ²→0); density bumps for
  ownership/threat (0.1→0.2); uncertainty re-enabled (0.0→0.1)
- `4B-impl-1` 4fee9d1: `position_index: u16` field in
  `ReplayBuffer`; HEXB v8 wire format with v7 backward-compat load;
  `sample_batch_with_pos` 8-tuple facade (legacy `sample_batch` 7-tuple
  byte-identical preserved); INV20 contract extended; bench gate PASS
  (n=5 on laptop, +25.79% buffer_push_per_s after hot-path cleanup)
- `4B-impl-3` f3509ce: `ply_index_head` 2-layer MLP + `compute_ply_index_loss`
  Huber on `clamp(position_index / 100, 0, 1)` + end-to-end plumbing
  through `BatchAssemblyResult` → `step_coordinator` →
  `train_step_from_tensors`
- `7c80a2d`: `load_state_dict_strict` benign-missing-keys allowlist
  for `ply_index_head.*` (extensible pattern for future aux heads)
- Tests: 7 new uncertainty loss tests + 7 new ply_index loss tests +
  buffer round-trip + INV20 facade pin extension; 1714+ pytest +
  191 cargo tests green

### Wave 5 strategic reckoning (Task 5 pre-write)

Per dispatcher: "if BOTH Track 4A and Track 4B colony, strategic
reckoning is forced." Both colonyed within 15k steps. Surfaces NOT
yet tested:

1. **Value-target propagation**: terminal z → all positions may be
   fundamentally wrong for HeXO. Try n-step bootstrap or TD-λ.
2. **WDL 3-class softmax** value head (replaces tanh scalar).
3. **Game-theoretic regularization**: explicit anti-colony loss term
   penalizing homogeneous-pattern composition.

Wave 5 scope estimate: ~3 weeks dev + ~$20 vast. Major commitment.
Operator decision pending; parked for now per "Stage 6 close + park
for Wave 5 design" route.

### Sprint-log header pre-written

`## §S181-AUDIT Wave 5 — strategic reckoning + structural target rework`

### Falsified Hypotheses Register additions

- "Bot mix is the load-bearing failure variable in the colony
  attractor mechanism" → FALSIFIED 2026-05-27 by Track 4A (W4A-B).
  Removing bot mix produced FASTER decline.
- "Multi-aux density (KataGo-style diverse aux signal) prevents
  single-attractor lock" → FALSIFIED 2026-05-27 by Track 4B (W4B-B).
  Delayed colony attractor ~5k steps but did not prevent it.

### Forensics

- Branches: `phase4.5/s181_wave4_subtract` (Track 4A), `phase4.5/s181_wave4_multiaux` (Track 4B impl + audit docs)
- Audit docs: `audit/structural/wave4_track_a_subtract.md`,
  `audit/structural/wave4_track_b_sustained.md`,
  `audit/structural/wave4_track_b_multiaux_design.md` (pre-impl design + scope reduction history)
- Stage 6 REVIEW: opus 4.7 fresh-context subagent dispatched 2026-05-27, verdict GREEN
- Commits: 5b7f85e (Track 4A variant) + d80de72/4fee9d1/f3509ce/6e75bfd/5fea1ce/3f0d9cd/7c80a2d/37b216d/0523147 (Track 4B impl + audit chain)

---

## §S181-AUDIT Wave 5 entry — compound-turn investigation

**Status:** OPEN — research/audit session landed; sustained-run verdict
operator-pending. (Placeholder: operator fills run details after the
Wave 5 design session.)

**Premise (per L59 boundary).** Wave 4 close established the colony
mechanism is insensitive to every config / hygiene / aux-density lever;
it lives downstream in the training objective + MCTS/selfplay interaction.
Wave 5 entry tests a structural hypothesis: HeXO's 2-stones-per-compound-
turn is modelled as two *sequential single-stone* decisions, which may
favour colony (order-invariant) over extension (needs coordination).

**Phase 5 read-only pipeline audit (this session, 2026-05-28).** Full
static audit of compound-turn handling across all 7 stages:
`audit/structural/compound_turn_pipeline_audit.md`.

Key results (see audit for citations + the five critical questions):
- Hypothesis's STRONG form FALSIFIED — the engine is **not** order-blind.
  Board/zobrist order-invariant; TT merges `{A,B}`≡`{B,A}`; MCTS Q-flips
  per *turn boundary* not per stone, with a correct 2-ply within-turn
  look-ahead. Genuinely sequential facets: greedy per-stone *commitment*
  (2 fresh searches/turn, no subtree reuse) and *per-ply storage* of the
  intermediate position.
- **Bug found — CF-1:** a first-stone win is scored `terminal_value=-1.0`
  (backup.rs:223-228) because `apply_move` does not flip the player after
  stone 1; the value convention is current_player-perspective, so the
  sign is inverted only for stone-1 wins. Distorts the stone-1 policy
  target (visit mass pushed off winning-cell-first). Mechanism-plausible
  minor colony contributor; causal link NOT established. Fix + A/B
  pre-registered in the audit, NOT yet implemented (read-only scope).
- **CF-2 (inferred-strong):** v6/v7full NN input drops the
  `moves_remaining`/`ply_parity` planes (registry.toml:78) — the value
  head gets no explicit turn-phase signal, mechanism-aligned with the
  L47 value-head discrimination collapse.
- **CF-6 (undetermined):** FPU sign at `mr==1` parents may not honour the
  per-turn flip; needs a `puct_score` unit test to resolve.

**Open question opened:** Q-COMPOUND-TURN in `docs/06_OPEN_QUESTIONS.md`.

**Operator next steps (Wave 5 design, pending):** CF-1 sign-fix unit test
+ A/B; CF-2 value-spread probe with planes 16/17 added; the structural
reckoning options pre-written in the Wave 4 entry (value-target
propagation / WDL value head / anti-colony regularization).

---

