# W1-Lens Forensics on Pre-Fix Pathologies

**Date.** 2026-04-19
**Lens.** Re-read pre-fix archives through W1 (commit `e9ebbb9`,
"fix(mcts): negate child Q at intermediate ply in
`get_improved_policy`, Gumbel score, `get_top_visits`"). Classify each
observation: W1-explained / Q17-explained / Q19-explained / Residual.
Archive-only; no retrain, no re-probe.

**W1 mechanism (ground truth).** Commit `e9ebbb9` negates child Q at
three sites when `parent.moves_remaining == 1`, matching behavior
`puct_score` already had. Child `w_value` is stored in child's own
player-to-move perspective via negamax backup
(`engine/src/mcts/backup.rs`). At root `moves_remaining == 1` children
belong to the opponent, so raw Q is opponent-perspective; pre-fix the
three sites consumed raw Q as if it were root-perspective. Affected
sites and paths:

- `engine/src/mcts/mod.rs::get_improved_policy` — completed-Q target
  for KL policy loss (`completed_q_values: true` variants:
  `gumbel_full`, `gumbel_targets`, `gumbel_targets_desktop`).
- `engine/src/game_runner/gumbel_search.rs::score` — Sequential
  Halving candidate selection (`gumbel_mcts: true` variants:
  `gumbel_full`, `gumbel_targets_desktop`).
- `engine/src/mcts/mod.rs::get_top_visits` — top-N reporter, used by
  viewer/analyze.
- `engine/src/lib.rs::get_root_children_info` — viewer/analyze
  (sibling A-010 found by grep audit).

Affected-root fraction: ~50% of compound-turn positions have root at
`moves_remaining == 1` (intermediate ply of a 2-stone compound turn).
Sign-flip inverts training targets on those positions; PUCT selection
during search was unaffected (`puct_score` already correct).

**Q17 mechanism.** No Dirichlet injection on Rust training path.
Static audit `archive/diagnosis_2026-04-10/diag_A_static_audit.md` +
runtime trace `diag_A_trace_training.jsonl` confirmed 30 game_runner
records, 0 `apply_dirichlet_to_root` records. Resolved §73
(`71d7e6e`).

**Q19 mechanism.** Threat head BCE class imbalance
(`threat_pos_weight = 59.0` needed). Resolved §92.

---

## §1 Summary table

| # | Observation | Bucket | Confidence | Evidence pointer |
|---|---|---|---|---|
| O1 | Entropy 1.49–1.70 nats across ckpts 13k–17k, no recovery | **Q17-explained** | mechanism-consistent | `diag_B_sharpness.md:26-36`; `diag_A_static_audit.md:4` |
| O2 | Fixed-point behavior, no downward trend, no recovery | **Q17-explained** | correctness-proof | `diag_C_summary.md:15-17` (Δentropy 0.127 nat); `diag_A_trace_summary.md:74-81` |
| O3 | 100% identical games in ckpt-vs-ckpt round-robin argmax eval | **Not a pathology** (architectural) | correctness-proof | `docs/07_PHASE4_SPRINT_LOG.md:1133-1148` |
| O4 | 94.4% draw rate on fast games (50 sims, τ=1.0) | **Residual (resolved config)** — out of W1/Q17/Q19 scope | correctness-proof | `report.md:106` (draw investigation); §75 |
| O5 | All draws = 150-ply timeouts, zero natural draws | **Residual (resolved config)** — out of W1/Q17/Q19 scope | correctness-proof | `report.md:10-14, 218-227`; §76 |
| O6 | Colony-extension: long sequences far from opponent cluster | **Q17-primary, W1-secondary-possible** | plausible but under-evidenced | `docs/07_PHASE4_SPRINT_LOG.md:1455` (§75); `diag_A_trace_summary.md:66-81` |
| O7 | `fast_prob: 0.0` patched `gumbel_targets`; desktop `gumbel_full` not patched | **Residual (resolved config)** — out of W1/Q17/Q19 scope | correctness-proof | §75; `configs/variants/gumbel_full.yaml` |
| O8 | Colony-win 100% → 60% across n=20 eval rounds | **Residual — insufficient sample** | under-evidenced | `runs/q27_smoke_2eval_20260419_100337/q27_smoke_2eval.jsonl:6028,12012` |
| O9 | Policy loss 2.02 → 1.78 at 5.5K, wr_best 2% → 6% | **Post-W1 anchor** (not a pre-fix pathology) | correctness-proof | same jsonl lines 47, 6370, 12702 |
| O10 | Probe v6: C2 55%, C3 65% at 5K+5.5K, PASSES all gates | **Post-W1 anchor** (not a pre-fix pathology) | correctness-proof | §106; `reports/q27_zoi_reachability_realpositions_2026-04-19.md` |

---

## §2 Per-observation detail

### O1 — Entropy band 1.49–1.70 nats across ckpts 13k–17k

**Pathology.** Raw-policy entropy on 500 fixed positions
(`diag_B_sharpness.md:26-36`) sits in narrow 1.49–1.70 nat band across
ckpt_13000 → ckpt_17428. Bootstrap reference 2.665 nats. No downward
trend, no recovery. Mid-phase (10 ≤ cm < 25) p10 drops to 0.08–0.19
nats per-checkpoint (`diag_B_sharpness.md:52-63`).

**W1 mechanism test.** W1 inverts completed-Q targets at ~50% of
positions (root `moves_remaining==1`). Mechanism predicts training
trajectory *distortion in a directed way* — network learns to prefer
moves the corrupted target said were good. Does not mechanistically
predict "stuck fixed point at band width 0.21 nats." W1-corrupted
targets would produce a biased moving trajectory, not a narrow stable
band. Mechanism-inconsistent with observed stability.

**Q17 mechanism test.** No Dirichlet → every worker starts from the
same empty-board prior (`diag_A_trace_summary.md:46-65`: 30 records,
all cm=0 ply=0, identical `root_priors` vector, identical
`root_visit_counts` vector). Rubber-stamp MCTS
(`diag_C_summary.md:15-17`: mean Δentropy = 0.127 nats, eff support
3.37). Training target ≈ network output → zero gradient signal to
break the equilibrium. This is the self-consistent attractor that
produces "stuck, not drifting." Mechanism directly predicts the band.

**Q19 mechanism test.** Threat-BCE imbalance corrupts threat head
training. Threat head shares trunk with policy; without pos_weight,
threat gradient drives toward all-zero prediction, which adds gradient
noise to trunk but does not systematically sharpen the policy toward a
single mode. Mechanism-inconsistent with the policy-specific fixed
point.

**Verdict.** **Q17-explained** (mechanism-consistent). W1 may have
corrupted training direction during the collapse window but does not
explain the observed stability. Q19 noise does not explain single-mode
sharpening.

**Residual hooks.** None. Band-width interpretation is K=0 caveat
(`diag_B_sharpness.md:4-16`); primary signal is the gap from bootstrap
(0.96 nats) not the absolute band.

---

### O2 — Fixed-point behavior, no downward trend, no recovery

**Pathology.** Across 7 post-bootstrap checkpoints spanning ~4.4K
training steps (13k → 17428), H(π) oscillates within 0.21 nats of a
stable mean. Restart heuristic `diag_B_sharpness.md:82-87` explicitly
warns "Do not use entropy ordering to select the restart point ...
both are stuck at the same fixed point"
(`docs/07_PHASE4_SPRINT_LOG.md:1024-1027`).

**W1 mechanism test.** W1-inverted targets produce *directed* drift,
not stasis. With ~50% of targets flipped in sign, gradient would
consistently push the network toward the wrong solution — but the push
would move the loss, not freeze it. The only way W1 alone produces a
fixed point is if the inverted targets happen to be self-consistent
with the network's output, which requires an additional coincidence
(the empty-board prior + MCTS rubber-stamp equilibrium). That
coincidence is precisely Q17's mechanism. Mechanism-insufficient on
its own.

**Q17 mechanism test.** `diag_C_summary.md:15-17` direct proof: MCTS
is a no-op (H_visits ≈ H_prior, Δ = 0.127 nats). Training targets
match network outputs because visit distribution = prior distribution
(no Dirichlet noise). Gradient signal vanishes at any rubber-stamp
attractor; further training maintains rather than escapes the
equilibrium. Mechanism directly predicts the fixed point.

**Q19 mechanism test.** Threat-BCE miscalibration affects threat head
convergence, not policy-head fixed points. Mechanism-inconsistent.

**Verdict.** **Q17-explained** (correctness-proof —
`diag_C_summary.md:15-17` is a direct measurement of the rubber-stamp,
the defining signature of Q17's attractor).

**Residual hooks.** None.

---

### O3 — 100% identical games in ckpt-vs-ckpt round-robin

**Pathology.** `docs/07_PHASE4_SPRINT_LOG.md:834-838`: ckpt_13000 vs
ckpt_14000 = 100/0 P1, 25 moves, carbon-copy every rollout.

**Sprint log verdict.** `docs/07_PHASE4_SPRINT_LOG.md:1133-1148`
classifies this as "expected behaviour, not a seeding bug":
`ModelPlayer.get_move()` calls `tree.get_policy(temperature=0.0)` →
one-hot argmax → no stochastic element. Any two runs of same matchup
produce identical games by construction. Temperature-sampling check
(`diag_C_summary.md:30-36`) confirmed τ=1.0 produces 13 distinct
lengths across 20 games — sampling code is correct.

**W1 / Q17 / Q19 mechanism test.** None of the three are required.
O3 is a property of the eval harness (argmax τ=0), not a training
pathology. W1 would change *which* deterministic game plays, not that
games are deterministic. Q17 and Q19 are irrelevant to the eval loop.

**Verdict.** **Not a pathology**. Architectural consequence of
argmax-at-τ=0 evaluation. Placed in §4 not-covered.

---

### O4 — 94.4% draw rate on low-sim fast games

**Pathology.** `reports/draw_rate_investigation_2026-04-10/report.md`:
low sims/sec quartile (<3125) draw rate 94.4% (n=1547); high sims/sec
quartile (>3938) 3.7% (n=1547). §75 attributes to `fast_prob=0.25`
games (50 sims, τ=1.0, PUCT) + 150-ply cap.

**W1 mechanism test.** W1 affects completed-Q targets and Gumbel
selection. Fast games ran PUCT (`gumbel_mcts=false`) and had 50 sims
per move — too shallow for completed-Q depth to matter. W1 affects
training signal direction, not in-game visit distribution under
PUCT-τ=1.0. Mechanism-inconsistent with the draw-rate observation.

**Q17 mechanism test.** Q17 affects root noise at self-play. With
τ=1.0 throughout fast games, per-move stochasticity is already high
(visit-count sampling at τ=1.0 spreads across multiple children). Q17
contributes to deterministic game-trajectory collapse in the standard
games, but fast games at τ=1.0 already diverge on their own. Q17 does
not predict the 94.4% draw rate.

**Q19 mechanism test.** Unrelated — threat-head miscalibration does
not produce timeout draws.

**Verdict.** **Residual (resolved by §75/§76)**. Root cause:
150-ply cap × low-sim shallow search × τ=1.0 random play. Not
explained by W1/Q17/Q19. Fixed by `fast_prob: 0.0` (§75) and
`max_game_moves: 200` plies (§76). Out of scope for this forensics
pass; listed in §3 for completeness.

**Discriminator (if reopened).** Rerun pre-§75 config with Dirichlet
ported and W1 applied — if 94.4% holds, confirms cap/low-sim
root-cause. Cost: not worth running; §75 fix already applied.

---

### O5 — All draws are 150-ply timeouts

**Pathology.** `report.md:10-14`: draw game length distribution =
`{75: 2609}`. 100% of 2609 draws occur at exactly game_length=75
compound moves (150 plies / 2 stones per compound turn). Zero natural
draws.

**W1 / Q17 / Q19 mechanism test.** None predict this cap-hitting
behavior. Cap is a config constant; its effect is independent of
correctness, exploration noise, or threat-head class balance.

**Verdict.** **Residual (resolved by §76)**. Pure config pathology —
`max_game_moves: 150` plies was calibrated for `fast_prob=0.25`
throughput; with `fast_prob=0.0` (§75) the cap became the dominant
draw source. Fixed to 200 plies.

**Discriminator.** Not needed; §76 applied.

---

### O6 — Colony-extension behavior in fast-game viewer

**Pathology.** §75 reference (`docs/07_PHASE4_SPRINT_LOG.md:1455`):
"colony-extension behaviour in the viewer" observed in fast games —
long sequences of stones placed far from opponent's cluster (verbal
report from draw investigation). Interacts with 150-ply cap to produce
timeout draws at 94.4% rate in fast regime.

**W1 mechanism test.** W1 inverts completed-Q targets. If bootstrap
had a pre-existing colony bias (stones placed far from opponent), W1
*could* have sustained or shifted that bias via corrupted training
signal, but the mechanism does not directionally predict "extend away
from opponent" — W1 produces directionally unstable training, not a
specific geometric preference. Mechanism-insufficient alone. Also note
fast games ran PUCT (`gumbel_mcts=false`) so `gumbel_search.rs::score`
is inactive; only `get_improved_policy` (KL target) touches these
positions, and fast games with 50 sims at τ=1.0 would produce noisy
targets regardless of W1.

**Q17 mechanism test.** No Dirichlet + deterministic prior + rubber-
stamp MCTS → bootstrap's learned prior gets amplified, not explored
away. Bootstrap pretrain was on SealBot corpus + minimax traces (§92,
v3b pretrain). If that corpus contained "colony-extension" patterns
(plausible: SealBot minimax depth ~4 may favor colony-anchored
lookahead), those get locked in without Dirichlet noise to perturb
the distribution. Mechanism directly predicts "amplify whatever bias
bootstrap had," which is consistent with colony behavior if bootstrap
had it.

**Q19 mechanism test.** Threat-head miscalibration could, in
principle, decouple threat prediction from policy; unclear it produces
specifically colony-extension. §106 reframes the C2/C3 policy-routing
failure as synthetic-fixture artifact, so there is no longer evidence
Q19 was routing the policy away from extensions. Mechanism-insufficient.

**Verdict.** **Q17-primary, W1-secondary-possible. Confidence:
plausible but under-evidenced.** Q17 mechanism (bias amplification via
rubber-stamp attractor) directly predicts colony behavior conditional
on bootstrap bias; W1 corrupts training direction in a way consistent
with drift but not directional toward colony. Archives do not contain
a bootstrap-only colony-bias measurement.

**Residual hooks.** Discriminator: measure bootstrap's per-position
colony-extension frequency (0-shot from `bootstrap_model.pt`) against
post-Q17-fix + post-W1-fix self-play colony frequency at matched
steps. If bootstrap already has colony bias, Q17-primary is confirmed
and W1 contribution is below the discriminator. Cost: moderate
(generate n=500 bootstrap self-play games, classify colony vs
connected wins via `hexo_rl/eval/colony_detection.py`). Would rely on
live post-fix retrain data (which is happening by default); archive
forensics alone cannot discriminate.

---

### O7 — `fast_prob: 0.0` patched laptop; desktop `gumbel_full` never patched

**Pathology.** §75: `fast_prob: 0.0` override landed in
`configs/variants/gumbel_targets.yaml`; `gumbel_full.yaml` unchanged
at inherited `fast_prob=0.25`. Sprint log claim: "Gumbel SH effective
in low-sim regime, §71" — i.e., Gumbel Sequential Halving's
candidate-concentration was argued to provide enough exploration in
fast games to avoid the 94.4% draw mode.

**W1 mechanism test.** Fast games in `gumbel_full` run Gumbel SH with
50 sims. W1 affects `gumbel_search.rs::score` — Sequential Halving
scores were pre-fix computing raw Q when root was at mr==1 (intermediate
ply). Effect: SH picks the wrong candidates at ~50% of move positions
in intermediate-ply roots. At 50 sims with 16 candidates, this would
manifest as degraded search quality during fast games specifically —
but that is a search-quality claim, not a draw-rate one.
Mechanism-inconsistent with the observation (which is a config
difference between variants, not a fix outcome).

**Q17 / Q19 mechanism test.** Q17 is active in both variants pre-§73.
Q19 unrelated.

**Verdict.** **Residual (resolved config oversight)**. The
observation documents a config asymmetry, not a training pathology
per se. If subsequent desktop `gumbel_full` runs regress to fast-game
draw collapse, that would be a new observation needing classification.
Out of scope for W1/Q17/Q19 forensics.

**Discriminator (if reopened).** Run desktop `gumbel_full` post-§73
(Dirichlet) + post-W1 at current config with fast_prob=0.25 left in.
If draw rate stays healthy, §75's Gumbel-SH argument held; if not,
propagate `fast_prob: 0.0` to desktop variant. Cost: low (1 sustained
run already in flight per CLAUDE.md status).

---

### O8 — Colony-win 100% → 60% across n=20 eval rounds

**Pathology.** `runs/q27_smoke_2eval_20260419_100337/q27_smoke_2eval.jsonl:6028,12012`:
vs-random eval at step 2500 reports `winrate=1.0, colony_wins=20`
(20/20 = 100%); step 5000 reports `winrate=1.0, colony_wins=12` (12/20
= 60%). Same n=20 per phase.

**Sample-size analysis.** n=20 binomial: 20/20 = 100% CI95 [0.839,
1.0]; 12/20 = 60% CI95 [0.361, 0.809]. CIs overlap from 0.809 down —
observation is within binomial-noise band at this sample size. Cannot
flag as trend without larger n.

**W1 / Q17 / Q19 mechanism test.** All three are post-fix at this
smoke (commit `a7efa78`, post-e9ebbb9, post-§73, post-§92). None of
the three are viable pre-fix-pathology classifications — this is a
post-fix observation and the smoke is included as an anchor, not as a
pathology to classify.

**Verdict.** **Residual — insufficient sample**. Observation is below
the discriminator at n=20. Would need n≥200 per eval round to flag
drift vs binomial noise.

**Residual hooks.** Discriminator: re-run vs-random eval phase at
n=200 at the 5000-step anchor, or accumulate across sustained-run
eval rounds to get effective n≥200. Cost: low (~15min per eval round
at `model_sims=96`); already happens organically during sustained run.

---

### O9 — Policy loss 2.02 → 1.78 at 5.5K, wr_best 2% → 6%

**Pathology.** `q27_smoke_2eval.jsonl:47` step 1 `policy_loss=2.0246`.
Scan through trajectory to step ~5000 shows band around 1.78–1.80.
`jsonl:6370` (step 2500) `wr_best=0.02, ci_best=[0.004, 0.105]`;
`jsonl:12702` (step 5000) `wr_best=0.06, ci_best=[0.021, 0.162]`.

**Trajectory analysis.** Policy loss drop −0.24 over 5K steps is
consistent with early-training improvement on the bootstrap-anchored
trajectory. wr_best 2%→6% is inside overlapping CIs (CI95 at 2500
includes 0.06; CI95 at 5000 includes 0.02). The graduation gate is
`wr_best ≥ 0.55 AND ci_lo > 0.5` over n=400 — this smoke is far below
the n and far below the wr_best floor, consistent with "early training
from bootstrap anchor, not yet promoting." Not a pathology.

**W1 / Q17 / Q19 mechanism test.** Post-fix anchor; not classifiable
as a pre-fix pathology.

**Verdict.** **Post-W1 anchor** (not a pre-fix pathology). Placed in
§4 not-covered.

---

### O10 — Probe v6: C2 55%, C3 65% at 5K+5.5K, PASSES all gates

**Pathology.** Per §106 and
`reports/q27_zoi_reachability_realpositions_2026-04-19.md`: post-W1
5K checkpoint passes all three gates on the v6 real-positions fixture:
`C1=+3.317, C2=50%, C3=65%` (user's O10 cites 55% / 65% — minor
numerical drift from §106's reported 50%; both PASS C2 ≥ 25% and
C3 ≥ 40%).

**W1 / Q17 / Q19 mechanism test.** All three fixes in tree at the
probe point. The observation is a positive health signal, not a
pathology. Including as an anchor: "post-W1 looks healthy on real
fixture."

**Verdict.** **Post-W1 anchor** (not a pre-fix pathology). Placed in
§4 not-covered. Reinforces that §105's pre-fix 20/20 C2/C3 failure
was a synthetic-fixture OOS artifact (§106), not a pathology needing
W1/Q17/Q19 classification.

---

## §3 Residual list

Ordered by whether archive forensics alone can still discriminate vs
whether live instrumentation on post-fix retrain is required.

### R1 — Colony-extension bias: does Q17 alone explain it, or does W1 also contribute? (O6)

- **What's residual.** Q17 (bias amplification) is mechanism-primary
  for O6. W1 contribution is plausible (corrupted training signal
  during the collapse window) but archive data does not quantify it.
- **Minimum discriminator.** Bootstrap-only colony-extension frequency
  on a fixed N=500 self-play sample. If bootstrap already shows colony
  bias at rate ≥ X observed in pre-fix self-play, Q17-primary is
  confirmed. If bootstrap is colony-free and pre-fix self-play is
  colony-heavy, both W1 and Q17 contributed.
- **Source.** Live instrumentation (run `bootstrap_model.pt` through
  SelfPlayRunner with Dirichlet disabled, W1 applied, PUCT, 50 sims,
  τ=1.0 to isolate bootstrap bias under shallow rubber-stamp).
- **Cost.** ~1 GPU-hour (500 games × 150 plies). Happens organically
  in Phase 4.0 exp D + E trajectories; no dedicated run needed.
- **Priority.** Low. Q17 attribution is sufficient for remediation;
  discriminator is interesting, not decision-blocking.

### R2 — O8 colony-win drift: below discriminator at n=20

- **What's residual.** 100% → 60% across rounds at n=20 is inside
  binomial noise band (CIs overlap).
- **Minimum discriminator.** n≥200 per eval round against random.
- **Source.** Archive-insufficient. Live — accumulate across
  sustained-run eval rounds.
- **Cost.** Zero incremental (organic during sustained run).
- **Priority.** Low.

### R3 — Resolved-config residuals (O4, O5, O7)

- **What's residual.** Not classifiable under W1/Q17/Q19 — they are
  config pathologies resolved by §75 (`fast_prob: 0.0` for
  `gumbel_targets`), §76 (`max_game_moves: 150 → 200`), and a pending
  decision on whether to propagate `fast_prob: 0.0` to `gumbel_full`
  desktop variant (O7).
- **Minimum discriminator.** For O7: observe one sustained desktop
  run at post-fix commit — if fast-game draw rate stays healthy under
  Gumbel SH + Dirichlet + W1, §75's argument holds and no propagation
  needed.
- **Source.** Live (already in flight per CLAUDE.md "exp D + exp E").
- **Cost.** Zero incremental.
- **Priority.** Low; observation-level only.

---

## §4 Not covered (archive items opened but not classified)

### N1 — O3 (identical argmax-eval games): architectural, not a pathology

`docs/07_PHASE4_SPRINT_LOG.md:1133-1148` documents that `ModelPlayer`
calls `tree.get_policy(temperature=0.0)` → one-hot argmax, and eval
loop has no stochastic element. τ=1.0 sampling check (`diag_C_summary.md:30-36`)
confirmed sampling code correct (13 distinct lengths / 20 games).
Not a training pathology. Excluded.

### N2 — O9 and O10: post-W1 anchors

These are anchors for comparison, not pre-fix pathologies. Included
by the user's seed list as the "post-W1 looks healthy" baseline.
Classification framework does not apply. Post-W1 policy loss trajectory
and probe-v6 PASS are normal signals for "training proceeds cleanly
after fixes."

### N3 — §77 depth investigation (reports/mcts_depth_investigation_2026-04-11/)

Opened the directory; depth measurements (mean leaf depth 2.92 plies,
B_eff 6.1, root children 7 visited out of 360 created). §77 concluded
Option A (no change). None of these measurements overlap with W1/Q17/Q19
mechanisms — depth is governed by sim budget × B_eff, independent of
root noise and Q-sign. Not classifiable under the four buckets.

### N4 — §106 Q32 watch item: threat-scalar vs policy-ranking decoupling on bootstrap

`docs/06_OPEN_QUESTIONS.md:121-133`: on v6 real fixture,
`bootstrap_model.pt` has `ctrl_logit_mean` (0.061) > `ext_logit_mean`
(0.015), flipping `contrast_mean` negative, yet policy routes 60%/65%
of extensions into top-5/top-10. Not a pre-fix pathology — observed on
bootstrap which pre-dates W1/Q17/Q19 scope. Watch item per §106.

### N5 — C4 BCE-drift canary (Q19 downstream)

Probe criterion C4 (`abs(ext_logit_mean − bootstrap_ext_logit_mean) <
5.0`) is a warning-only threshold tracking Q19 drift. Archives show C4
within tolerance on both arms of §105 smoke (pre 0.078, post 0.505).
Did not rise to a pathology needing classification. Excluded.

### N6 — Temperature schedule docs-vs-code drift (§70 diag_C.1, diag_C_temp_schedule.md)

Rust code implements quarter-cosine per compound_move with hard floor
at cm 15; sprint log §36 describes half-cosine per ply with
`temp_anneal_moves=60`. `diag_C_temp_schedule.md:54-68` verdict: "not
the cause of mode collapse on its own — no root noise is — but it
narrows the time window in which (1) and (2) could be broken by
chance." Independent of W1/Q17/Q19 mechanism. Remains unresolved at
the time of this forensics pass (CLAUDE.md:"docs-vs-code drift vs §36
flagged in §70 C.1 — unresolved"). Not classified.

---

## Verification

- §1 table: 10 rows (O1–O10), each with bucket + confidence + evidence
  file:line or run:line pointer.
- §2: each O# has pathology statement, three mechanism tests
  (W1/Q17/Q19), verdict with pointer, residual hooks where applicable.
- §3 residuals list discriminator + source + cost for each.
- §4 lists 6 archive items opened and not classified, with reason.
- Confidence flagged explicitly: "correctness-proof" (O2, O3, O4, O5,
  O9, O10), "mechanism-consistent" (O1), "plausible but under-
  evidenced" (O6), "under-evidenced" (O8).
- Report ~350 lines, well under 1500-line cap.
