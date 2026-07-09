# run3 card #1 — distributional 65-bin value head: frozen-trunk discriminator (D-WATCHGUARD WP4)

**Status: DESIGN ONLY — no training run, no GPU sweep executed. Dispatch is a later, separate
decision.** Repo state at design time: master, run2_mw_fresh LIVE on vast @ master fc97cc82,
live step ≈ 242,722; banked run2 checkpoints on vast 5k grid 5000…235000 (+500-stride recency
tail, latest 243000 observed). Nothing here touches the live host beyond read-only ssh.

Card provenance: `reports/d_shrimp_audit.md` headline #4 / run3-card #1 — shrimp
(Cmiller132/hexo-bot, local clone `/home/timmy/Work/Hexo/hexo-bot`) trains a 65-bin
distributional value head; double-corroborated (S1 + S4a) as the highest-value, lowest-cost
run3 lever for the value tail. This doc pins (D1) what the bins actually are, (D2) the probe
set, (D3) the discriminator protocol, (D4) the pre-registered asymmetric verdict, plus a
red-team pass.

---

## 0. Premise chain (established; do not re-derive)

- **D-LOCALIZE**: model⊥SealBot gap is in the VALUE head — blind to SealBot-reachable losses
  (net 0.6–1.0 "winning" at d6 forced-loss positions).
- **D-PERCEPT**: blindness is HORIZON, not perception. Honest proven core 33/61 = 54%; 67% of
  the core is DEEP value-blind (~15% short-lookahead-catchable). Lever = outcome/search-distilled
  value targets, NOT cheap TD bootstrap.
- **D-INJECT**: constant-−1 value-DISTILL is NO-GO (KILL-A ⊥ KILL-C anti-correlated at all
  weights). Teacher-in-the-loss-path is a falsified lever class.
- **D-FULLSPEC**: on the OLD d1m-lineage frozen trunk (ckpt 272357, single-window v6_live2
  self-play per D-FORENSIC F1), NO readout separates win/loss (holdout AUC ceiling ~0.70 pooled,
  pre-pool 0.62, train-fit 0.94–1.0 rules out under-power) → representational entanglement.
  **Trap found there: turn-phase confound** — v6_live2 planes 2,3 differ ~10× between turn
  phases; AUC 0.807 achievable from turn-phase ALONE. Every AUC in this design is computed on a
  turn-phase-MATCHED set (§D3.4).
- **D-VETO**: fire-rate ~0 at deploy-150; arena "missed block" is a double-threat FORMATION
  indicator (value-head territory, "the turn BEFORE"), not a tactical miss. That residual is what
  this card targets.

**Re-validation discipline note (context transfer).** D-FULLSPEC's "frozen features can't
separate" was established on the d1m lineage's single-window-contaminated trunk. run2_mw_fresh
is a FRESH multi-window lineage — the falsification context does NOT auto-transfer. That is
exactly why this discriminator exists: it re-tests the frozen-representation question on run2's
trunk instead of citing the prior to skip the test. It is also why the NULL branch of the
verdict is asymmetric (§D4): a frozen-trunk NULL here would be *consistent with* D-FULLSPEC
transferring, which bounds the readout, not the training lever.

---

## D1 — Bin semantics (VERIFIED from shrimp source)

All claims in this section read directly from `/home/timmy/Work/Hexo/hexo-bot` (single-commit
MIT snapshot; same clone audited by D-SHRIMP). Exact anchors:

- `packages/shrimp/python/shrimp/constants.py:74` — `VALUE_BINS = 65`.
- `packages/shrimp/python/shrimp/losses.py:40-44` — support points =
  `torch.linspace(-1.0, 1.0, 65)`. Bin width = 2/64 = **1/32 = 0.03125**. The bins are a scalar
  support over the game-outcome value range **[-1, +1]**. They are NOT moves-to-end, NOT margin,
  NOT a discounted return.
- `losses.py:63-82` `scalar_to_binned_target` — scalar z ∈ [-1,1] → **two-hot adjacent-bin soft
  target**: `position = (z+1)·32`; probability mass split linearly between `floor(position)` and
  `floor(position)+1` (MuZero/KataGo-style two-hot). fp32 end-to-end (comment at
  `losses.py:166-171`: fp16 two-hot placement mis-splits adjacent bins by ~3% of a bin).
- `losses.py:153-198` `binned_value_loss` — cross-entropy of head softmax against the two-hot
  target; rows with `value_mask==0` (truncated games, no engine winner) excluded from numerator
  and denominator.
- `losses.py:47-51` `decode_binned_value` — deploy/eval decode = softmax **expectation** over
  the support, clamped to [-1,1]. The distribution is auxiliary richness; the scalar read stays
  drop-in compatible with everything that consumed a scalar value.
- Head: `model.py:616-617` — `value_reduction: Linear(3c → c)`, `value_head: Linear(c → 65)`.
  Siblings sharing the same 65-bin support machinery: `moves_left_head` (`model.py:623`, scalar
  affine-mapped from [0, MOVES_LEFT_CAP=209] to [-1,1], `samples.py:347`), `cell_q_head`
  (`model.py:615`), per-horizon `stvalue_<h>` heads.

### D1.1 The value TARGET (the load-bearing fact)

`samples.py:158-210` `finalize_game_samples`:

```
hard_z = _winner_value(winner, player)            # +1 win / -1 loss / 0 no-winner  (samples.py:63-66)
value_target = (1 - soft_z_lambda) * hard_z + soft_z_lambda * root_value   # samples.py:199
```

`soft_z_lambda` **defaults to 0.0 and no caller in the entire repo overrides it** (grep over
py/rs/toml/yaml/sh: the only hits are the definition and its docstring). **Shipped shrimp trains
the 65-bin value head on the pure game outcome z, two-hot encoded.** The optional λ>0 blend
mixes in the game's own MCTS `root_value` (self-search bootstrap — still not an external
teacher, but D-PERCEPT explicitly deprioritizes bootstrap-flavored targets; we pin λ=0).

Truncated games: `value_mask = 0` (`samples.py:174-177`, `batching.py:196-198`) — the value
target is never supervised on rows without a real winner.

### D1.2 Exact target construction from a HeXO replay row

HeXO's replay buffer already stores everything required. Per-row fields
(`engine/src/replay_buffer/mod.rs:215-263`, push signature): `state, chain_planes, policy,
outcome (f32), ownership, winning_line, game_id, game_length, is_full_search, position_index,
value_target_valid (u8)`. The training sample path emits `outcomes` and `value_target_valid`
per drawn row (`engine/src/replay_buffer/sample.rs:306,354-356,395`; the `_with_positions`
variant additionally emits `position_indices`, `sample.rs:462-465,502`).

Target construction for the 65-bin head, per sampled row `i`:

```
z      = outcomes[i]                       # already the row-player-perspective scalar in [-1, 1]
                                           #   (same field the scalar head trains on today:
                                           #    hexo_rl/training/losses.py:89  target=(z+1)/2, BCE)
mask   = value_target_valid[i]             # 1 = supervise, 0 = ply-capped/truncated → excluded
pos    = (z + 1.0) * 32.0                  # ∈ [0, 64]
lo     = floor(pos); hi = min(lo+1, 64)
target = zeros(65); target[lo] = 1-(pos-lo); target[hi] += (pos-lo)   # two-hot
loss   = CE(softmax(value_logits_65), target)  masked by value_target_valid, fp32 targets
```

No new buffer field, no schema change, no regeneration. `outcomes` + `value_target_valid` are
sufficient. (`moves_left`-style aux heads would need `game_length − position_index`; both are
stored, `game_length` is not currently emitted by `sample_batch` — an aux head is explicitly
OUT of the minimal card.)

### D1.3 STANDING INVARIANT — register compatibility (not a passing remark)

> **INV-D1 (outcome-only value targets).** Every target in the 65-bin value-head loss path
> derives from the game's OWN outcome/trajectory as recorded in the replay row: `outcomes`,
> `value_target_valid`, and (for optional aux heads, if ever added) `game_length`/
> `position_index`. **No teacher network output — SealBot score, distilled net value, solver
> score, or any other model's evaluation — may appear anywhere in the loss path.** That would be
> value-distillation, which is FALSIFIED (D-INJECT NO-GO: KILL-A ⊥ KILL-C anti-correlated at
> every weight; D-FULLSPEC closeout: richer continuous SealBot-value target craters KILL-C to
> 0.045). This invariant binds the discriminator below, the run3 production card, and any
> later variant of it. λ (`soft_z_lambda`-style root-value blending) is additionally pinned to
> 0 for this card: root_value is self-search, not a teacher, but it is bootstrap-shaped and
> D-PERCEPT's lever statement ("outcome/search-distilled, NOT cheap TD bootstrap") plus the
> confound cost of a second moving part both argue for pure z. Any relaxation of λ=0 is a NEW
> card, separately registered.

**D1 verdict: the 65 bins do NOT require a teacher.** They are a distributional re-encoding of
the exact scalar outcome target HeXO already trains on (BCE on (z+1)/2,
`hexo_rl/training/losses.py:75-96`). The card's premise stands: same information into the loss,
different loss geometry out — CE over a 65-atom support forces the head to place probability
mass, penalizing confident wrong-side mass in a way scalar BCE does not, and exposes a
loss-tail readout (P(v ≤ −0.5)) that a scalar cannot. SealBot/solver scores appear in this
design ONLY as probe-set LABELS (evaluation instrument, never in any gradient).

---

## D2 — Probe set: trap-FORMATION positions

### D2.1 Definition (pre-registered)

A probe **positive** is a position at the model's move, `N` compound turns before the earliest
provable loss, in a run2-vs-SealBot game the model LOST:

1. Game source: run2 checkpoint (deploy regime: multi-window no-drop, Gumbel SH g=0 m=16,
   150 sims) vs SealBot-d5, seeded paired openings (D-VETO V2 protocol:
   `--opening-plies 4 --opening-radius 4`, both colors per opening — opening DIVERSITY is
   load-bearing, both engines are deterministic; D-VETO got 40/40 distinct games this way).
2. `t*` = the EARLIEST model-decision ply at which SealBot-d7 scoring of the actual game
   position returns a **terminal mate against the model: score ≤ −99999000 at depth ≥ 7**.
   Stored `sealbot_score` labels are NOT acceptable (DS1: ~2.5% sign-flip, ~50% soft heuristic
   — the deep re-score gate is mandatory).
3. The positive is the position at the model's compound turn `t* − N`, for `N ∈ {2, 3}` turns
   (1 turn = 2 stones = 2 model decisions; so 4 and 6 model-decision plies before `t*`).
   **Pre-registered N = {2, 3}.** Justification: N=1 is inside the tactically-decided band
   (D-VETO: at the flagged turn, positions are already provably lost — its 44 no-op events);
   N=2 is the empirically verified walk-in band (§R1 spot-check: 0/46 already mate-proven at
   d6, 78% net-blind); N≥4 dilutes toward generic mid-game (supply drops steeply: only 25/70
   games even have an N=3 sample — see §R1) and the D-VETO mechanism read says the formation
   forms "the turn before", not four turns before.
4. **Not-yet-decided filter** at the sample ply: SealBot-d7 at `t*−N` must NOT prove terminal
   mate (else shift the anchor: that ply becomes the new `t*` candidate — equivalent to
   defining `t*` correctly as the earliest proof), AND the d7 score at `t*−N` must be < 0
   (position already SealBot-worse, i.e. the trap is FORMING). The score<0 clause excludes the
   blunder-in-between class the spot-check surfaced (17% of N=2 candidates had score ≥ 0 —
   including one +99999997 forced WIN two turns before the loss proof; labeling that "trap
   formation" would be a mislabel — §R1).
5. Dedupe: byte-identical board hash (canonical D6 orbit representative) across the whole set.
   eff_n counts DISTINCT positions; source-game clustering handled at CI time (§D3.5).
   Cap: ≤ 2 positives per source game (at most one per N).

A probe **negative** ("safe") is a position from a run2-vs-SealBot game the model WON or DREW,
sampled at the same ply band and same turn-phase as a paired positive, verified by the same
instrument: SealBot-d7 score ≥ 0 and no mate proof against the model. Equal count to positives,
matched 1:1 (§D3.4).

The 8 D-VETO fires are **SEED EXAMPLES ONLY** — existence proofs of the class, never counted.

### D2.2 HARD GATE

> **≥ 200 distinct positive positions (with ≥ 200 matched negatives), or the discriminator is
> NO-TEST.** A NULL banked from an underpowered probe is a fake NULL; do not run the training
> arms at all until the gate is met. eff_n = distinct positions (post-dedupe), reported
> alongside nominal n AND alongside the distinct-source-game count (the clustering unit).

### D2.3 What exists today vs the gate

| source | games | model losses | probe-eligible positives (est.) | eligible? |
|---|---|---|---|---|
| D-VETO V2 arena (`scripts/arena/results/dveto_v2_*.json`, run2@175k, 40 paired + 40 near-dup veto arm) | 40 distinct | 22 | ~20 N=2 + ~11 N=3 ≈ **30** (yield rates from §R1: 66%/36% per loss game) | YES — run2 lineage, full move records verified present |
| D-LOCALIZE p2 decisions (`reports/d_localize_2026-06-25/p2_decisions_s{150k,175k,200k}.jsonl`, 68 games, per-ply d6 annotations) | 68 | 70 loss-side rows→46 N=2 + 25 N=3 | 71 | **NO — d1m lineage** (single-window contaminated trunk generated these games; distribution mismatch). Seed/dev/transfer set only, reported separately, never gate-bearing |
| D-TACTICAL corpora (`reports/d_tactical_2026-06-26/heldout_traps_all.jsonl` 125, `corpus.jsonl` 61) | — | — | — | NO — same lineage caveat + post-blunder tactical traps, not formation-band; seed only |
| In-loop run2 eval sealbot rounds (50 games/50k-round) | ~200+ played | ~110 | 0 recoverable | NO — in-loop eval persists aggregates only (`opponent_runners.py:143-159` → `db.insert_match`; no move lists written). VERIFIED by read. |

**Available today: ~30 eligible positives. Gate shortfall: ~170.**

### D2.4 Cost to reach 200 (generation plan, operator-dispatched later)

- Yield model (measured, §R1): ~1.0 eligible positive per loss game (0.66 N=2 + 0.36 N=3);
  loss rate vs SealBot-d5 ≈ 55% (run2@175k WR 45%). → **~200 loss games → ~370–400 fresh games.**
- Generation: existing arena harness (`scripts/arena/run_arena.py` + hexo-ref-server +
  hexo-bridge; cross-bot-arena memory: wave ≤ 2 on laptop). 400 games at D-VETO-observed pace
  (~40 games/session, games are short — 33–80 plies, no clock) ≈ overnight-to-a-weekend of
  laptop sessions, or spread across 2–3 banked checkpoints (175k / 200k / 235k) which ALSO
  buys checkpoint diversity for the probe distribution. GPU load is inference-only (150 sims);
  this is NOT a training run and NOT a sweep — it is probe-corpus generation, dispatched
  separately after this design is accepted.
- d7 verification: backward scan from game end over model decisions for the earliest mate
  proof. Loss games average ~15 scannable model decisions; a monotone-boundary bisection
  (validated on a 20-game sample first — proof status along the actual line is not guaranteed
  monotone; if non-monotone > 5% of sampled games, fall back to linear backward scan) costs
  ~5 d7 calls/game. 200 games × 5 × O(10–60 s) ≈ 3–17 CPU-hours, embarrassingly parallel over
  the 8-shard pattern (`reports/d_tactical_2026-06-26/seal_shard*.jsonl`) → one laptop night.
- Per-position record (log once, re-classify forever — Workflow-agent lesson): move_seq prefix,
  ply, turn-phase, N, d7 score+depth at sample ply, t*, net_value of the GENERATING checkpoint,
  board hash, source game id, generating checkpoint id.

---

## D3 — Discriminator: frozen-trunk head-swap (laptop RTX 4060)

### D3.1 Arms

One banked run2 checkpoint (pre-registered default: **checkpoint_00235000.pt**, the newest 5k-grid
bank at design time; substitute the newest banked grid point at dispatch, recorded in the run
artifact). Trunk FROZEN (all parameters except the value head; policy/aux heads untouched and
unused). Two value-head arms, trained from fresh init:

| | arm S (scalar, production-shaped) | arm D (65-bin) |
|---|---|---|
| head | `value_fc1: Linear(256→256)` + `value_fc2: Linear(256→1)` (mirrors `network.py:559-560`) | `value_fc1: Linear(256→256)` + `value_fc2_bins: Linear(256→65)` |
| loss | BCE-with-logits vs (z+1)/2, masked by `value_target_valid` (= production `compute_value_loss`, `hexo_rl/training/losses.py:75-96`) | CE vs two-hot(z) per D1.2, same mask |
| decode | sigmoid(logit)·2−1 (scalar v) | E[softmax·linspace(−1,1,65)], clamp |

Both arms re-init BOTH value layers (fc1+fc2) — re-initializing only fc2 would hand arm S a
pretrained fc1 and arm D a mismatched one. SAME frozen trunk instance, SAME replay sample
stream (identical draw seeds), SAME step count, SAME optimizer/LR/batch (production defaults),
3 seeds per arm; report per-seed and pooled.

**K-cluster handling** (v6_live2_ls, k_max=8, value_pool=min, `registry.toml`): production
computes per-cluster value and min-reduces (`network.py:921-961`, `min_max_window_head`).
Arm D preserves the worst-cluster semantics: per-cluster 65-bin logits → per-cluster decoded
scalar → **argmin cluster selection → CE applied to THAT cluster's logits** (gradient to the
argmin cluster only, exactly like min-pool's subgradient today). Logit-space pooling (LSE-min
etc.) changes semantics and is out of scope.

### D3.2 Training sample

Replay source: one-time snapshot of the live buffer
(`/workspace/hexo_rl/checkpoints/replay_buffer.bin`, 2.35 GB observed 2026-07-09) pulled via
rsync-vast at dispatch — a frozen, hash-recorded snapshot; both arms draw from it through the
production `sample_batch` path (weights, dedup, masks — `sample.rs`). The snapshot contains
self-play rows + the human-corpus mixing rows (`pretrained_buffer_path`, real human-game
outcomes — legitimate under INV-D1). Holdout: game-id-disjoint 10% split reserved for metric
(ii)/(iii); never trained on.

### D3.3 Metrics (all pre-registered)

1. **(i) PRIMARY — trap-formation separation**: AUC(lost vs safe) on the D2 probe set,
   scored by each arm's decoded scalar v. **The gate metric is the PAIRED ΔAUC (arm D − arm S)
   on the identical probe set.** Secondary descriptive readout for arm D only (never
   gate-bearing, pre-registered as descriptive): AUC using loss-tail mass P(v ≤ −0.5); if the
   distributional head's win comes from tail mass rather than the mean, that is mechanism
   evidence for the run3 card writeup, not a second chance at the gate.
2. **(ii) decided-row value accuracy**: sign-agreement (decoded v vs z) on holdout replay rows
   with `value_target_valid==1` and z ≠ 0.
3. **(iii) value ECE**: 10-bin expected calibration error of P(win) = (v+1)/2 vs realized
   outcome on the same holdout. (Shrimp's own prefit probe uses value-ECE — `prefit.py`
   per-epoch diagnostic; convergent instrument choice.)

Metrics (ii)/(iii) are sanity/context: an arm that wins (i) while cratering (ii) is a red flag
(D-FULLSPEC KILL-C anti-correlation pattern), and the doc's verdict must quote all three.

### D3.4 Turn-phase-MATCHED evaluation set (the D-FULLSPEC trap, defeated explicitly)

HTTT turns place TWO stones; v6_live2 planes 2,3 encode turn phase and differ ~10× between
phases — a classifier can reach AUC 0.807 from phase alone (D-FULLSPEC). Positives cluster in
specific phases/ply-bands by construction (they sit N turns before a loss proof), so an
unmatched AUC is confounded by construction. Spec:

- Every positive carries (turn_phase ∈ {stone1, stone2}, ply_band = ⌊ply/10⌋, stone_count_band).
- Negatives are sampled to EXACT 1:1 match on (turn_phase, ply_band) — greedy matched pairing;
  unmatched positives are dropped from the AUC (and the drop count reported; if >10% drop, the
  negative-generation pass is extended rather than accepting the bias).
- **Phase canary (NO-TEST tripwire)**: a phase+ply-only logistic classifier (features: turn
  phase, ply, stone count — no board content) is trained on the matched set. If its AUC > 0.55,
  the matching failed and the probe is NO-TEST until fixed. This is the direct anti-0.807 check.
- AUCs additionally reported per-phase (stone1-only, stone2-only) as a consistency read.

### D3.5 CI and clustering

PASS/FAIL CI = **cluster bootstrap by source GAME** (10,000 resamples) over the paired per-position
score differences → CI on ΔAUC. Positions from one game are correlated (≤2 positives/game by
D2.1 cap); game-level resampling is the honest unit (effective-n lesson, §D-ARGMAX: distinct
games, not row count). Report: nominal n, distinct positions, distinct source games, achieved
SE of ΔAUC, and the empirical between-arm score correlation r (see §R2 — r determines whether
+0.05 was detectable; publishing r makes the power claim auditable post-hoc).

### D3.6 Leakage exclusion (head-training sample vs probe set)

- Structural: probe positions come from EVAL games (model-vs-SealBot); the replay buffer holds
  self-play + human-corpus rows. Eval games never enter the buffer (verified: eval path writes
  only `db.insert_match` aggregates). The probe-generation games are played OFFLINE by a
  dispatched harness and never pushed.
- Enforced anyway (belt-and-braces, because "structurally impossible" has failed this project
  before): canonical board-hash intersection between the probe set (all positions) and every
  sampled training row is computed and must be EMPTY; near-duplicate hazard (self-play reaching
  probe-adjacent positions) is quantified by reporting the count of training rows within
  Hamming ≤ 2 stones of any probe position (report-only; exact-hash overlap is the exclusion).

### D3.7 Budget

Frozen-trunk head training = forward passes + 2-layer backward. Batch 256 on RTX 4060,
~20–30k steps/arm × 2 arms × 3 seeds — hours, not days; thermal rule applies (no sustained
LTO-class load; this is well under it). d7 verification is CPU (SealBot pybind). Nothing runs
on vast except the two read-only pulls (1 checkpoint, 1 buffer snapshot) via rsync-vast.

---

## D4 — PRE-REGISTERED ASYMMETRIC VERDICT

- **PASS** = probe AUC improvement ≥ +0.05 with bootstrap CI excluding 0 → green-light card #1
  for run3.
- **NULL** = does **NOT** kill the card (a frozen trunk may simply lack the features).
  Escalation: unfreeze the last trunk block, ONE re-test. A second NULL → card #1 demoted below
  card #4 — **demoted, not deleted** — and the doc must state why (the joint-training confound
  remains unresolved).
- This asymmetry is the point. Any later agent citing a frozen-trunk NULL as a KILL is misusing
  the register. Say so in the doc, in those words.

(Stated here in those words, permanently: **a frozen-trunk NULL is not a KILL.** The
discriminator can only prove the POSITIVE cheaply. The distributional head's hypothesized
mechanism — CE-over-support shaping the TRUNK's representation during joint training — is
structurally invisible to a frozen-trunk test; D-FULLSPEC already demonstrated that frozen
v6-family features can fail every readout while the training-lever question stays open. A
second NULL after the one-block unfreeze still leaves that confound unresolved; it demotes the
card below card #4 (BC-prefit-slot equivalent per the d_shrimp ranking) on cost-of-evidence
grounds only.)

Escalation arm spec (fixed now so the re-test can't be quietly re-tuned): unfreeze the LAST
trunk residual block only, both arms identically, LR for unfrozen block = 0.1× head LR, same
steps/seeds/metrics/CI; every other knob byte-identical to the frozen run.

---

## RED-TEAM (adversarial pass, done as part of this design)

### R1 — "Are proven-loss − N positions actually formation walk-ins, or already-decided noise?"

Spot-checked against the only per-ply solver-annotated model-vs-SealBot corpus in the repo:
`reports/d_localize_2026-06-25/p2_decisions_s{150k,175k,200k}.jsonl` (68 games, 70 model
losses, per-model-decision `d6_score`/`d6_forced_loss`/`net_value`). Construction replicated
exactly (earliest in-game mate-proof anchor `t*`; sample at 2N model decisions before):

- **N=2 (46 candidates): 0/46 already mate-proven at the sample ply.** N=3 (25): 0/25. The
  boundary-anchored construction does NOT sample already-proven positions (at the d6
  instrument; production uses d7 — see R6).
- **But 8/46 (17%) of N=2 candidates had d6_score ≥ 0** — including one at +99999997 (a forced
  WIN two turns before the loss proof: the model blundered in between). These are NOT
  formation walk-ins; counting them as "lost" positives mislabels the probe. → the score<0
  eligibility clause (D2.1.4) exists because of this finding, and the ~1.0/loss-game yield
  estimate already nets it out.
- 36/46 (78%) of N=2 candidates have net_value > 0 (generating net says "winning" at a
  SealBot-worse, soon-provably-lost position) — the probe class exhibits exactly the D-LOCALIZE
  blindness it is meant to measure. The class is real, not noise.
- Supply-side finding: only 46/70 loss games yield an N=2 sample (proof fires too early in the
  rest) and 25/70 an N=3. The 200-gate arithmetic in D2.4 uses these measured yields, not hope.
- Caveat, stated plainly: this spot-check ran on OLD-lineage games (the only annotated corpus
  available without new compute). It validates the position-class GEOMETRY of the
  construction; run2-distribution rates get re-measured on the first 40 fresh games at
  dispatch, and if the already-decided rate at d7 exceeds 10% there, the N choice is revisited
  before generation continues.

### R2 — "Is +0.05 powered at n=200?"

Hanley-McNeil SEs, paired ΔAUC, α=0.05 two-sided, 80% power target; base AUCs 0.65/0.70;
DE = cluster design effect for 2 positions/game at ρ=0.5:

| n/class | DE | eff n | detectable ΔAUC @ r=0.7 | @ r=0.85 | @ r=0.9 | power for +0.05 @ r=0.85 |
|---|---|---|---|---|---|---|
| 100 | 1.0 | 100 | 0.082 | 0.058 | 0.048 | 0.67 |
| 200 | 1.0 | 200 | 0.058 | 0.041 | 0.034 | 0.93 |
| 200 | 1.5 | 133 | 0.071 | 0.050 | 0.041 | 0.79 |
| 150 | 1.5 | 100 | 0.082 | 0.058 | 0.048 | 0.67 |

Findings: (a) at the bare 200-gate with 2 positions/game, +0.05 is at ~0.79 power ONLY if the
between-arm score correlation r ≳ 0.85 — plausible for two heads on one frozen trunk, but an
ASSUMPTION until measured; at r=0.7 the detectable effect is 0.071 and power drops to ~0.50.
(b) Mitigations, adopted into the design: prefer 1 positive/game when supply allows (DE→1.0:
power 0.93 at r=0.85); target 250–300 positives, treating 200 as the NO-TEST floor, not the
plan; publish achieved SE and empirical r with the verdict so an "insufficiently powered
non-PASS" cannot be silently read as evidence of absence. (c) The gate criterion (bootstrap CI
excluding 0 AND point estimate ≥ +0.05) is self-honest under low power: low power makes PASS
harder, never easier — the asymmetric-verdict protection (D4) is what keeps the resulting NULL
from being misread.

### R3 — leakage between head-training sample and probe set

Attack: the buffer might contain positions from the probe games (would let heads memorize probe
positions) or near-duplicates. Resolution in §D3.6: structural disjointness (eval games are
never pushed to the buffer; in-loop eval doesn't even persist move lists) PLUS enforced
canonical-hash empty-intersection check PLUS reported near-duplicate counts. Residual accepted
risk: self-play organically visiting formation-adjacent regions — that is not leakage, that is
the training distribution; it biases BOTH arms equally and is the point of the test.

### R4 — turn-phase confound (the D-FULLSPEC 0.807 shortcut)

Attack: positives sit at loss-proximal plies/phases; any ply- or phase-correlated feature
(planes 2,3; stone count) gives free AUC. Defeated by: exact (turn_phase, ply_band) 1:1
matching of negatives (D3.4), the phase+ply-only classifier canary with a NO-TEST threshold at
AUC 0.55, per-phase AUC reporting, and — critically — the gate metric being the paired ΔAUC
between two arms scored on the IDENTICAL matched set: any residual phase leakage inflates both
arms' absolute AUCs but cancels in the difference unless one head is better at exploiting
phase, which the canary + per-phase breakdown would surface.

### R5 — head fairness (found by this pass; absorbed into D3)

Arm D's final layer has 65× the output parameters (256×65 vs 256×1, ~16.6k extra) and CE vs BCE
have different effective gradient scales. Mitigations: identical fc1 width (the capacity
bottleneck is the shared 256-d input), identical LR with a shared mini-grid {1×, 0.3×} applied
to BOTH arms symmetrically (best-of per arm by holdout metric (ii), chosen BEFORE any probe
scoring — probe AUC is never used for model selection), 3 seeds, and metrics (ii)/(iii)
reported so a raw-capacity win would show as across-the-board improvement rather than
probe-specific separation. A probe-specific gain with flat (ii)/(iii) is the signature the
card predicts.

### R6 — instrument depth mismatch

The spot-check's "0% already-decided" is at d6; production labels use d7. A d7 pass will prove
mates ~1 ply earlier, shifting some t* earlier — which only moves the sampled position earlier
too (construction is anchor-relative, so it degrades gracefully). Deeper-than-d7 provable
positions ("already lost at d9") may still enter as positives; accepted and stated: "formation"
in this design means formed-relative-to-the-d7-instrument, the same instrument class the whole
D-LOCALIZE/D-PERCEPT chain is built on. The alternative (deeper proofs) is the D-RECONFIRM
native-tactics lane, not this card.

### R7 — probe-source lineage

Attack: padding the 200-gate with the 71 old-lineage (d1m) positions would poison the probe
with games generated by a single-window-contaminated policy against which run2's trunk was
never trained. Resolution: hard-excluded from the gate count (D2.3); kept only as a separately
reported transfer read.

---

## Dispatch checklist (LATER — none of this was executed for this design)

1. Generate probe corpus: ~400 paired-opening games across 2–3 banked run2 ckpts (laptop
   arena harness), per-position durable JSONL (audit-artifacts-durable rule: NOT in a /tmp
   worktree — under `reports/run3_d1_probe/`).
2. d7 verification shards → probe set build → dedupe → gate check (≥200 distinct positives,
   phase-matched negatives, canary AUC ≤ 0.55) → publish probe-set manifest + hashes.
3. rsync 1 checkpoint + buffer snapshot from vast (read-only).
4. Train 2 arms × 3 seeds frozen-trunk; compute metrics; cluster-bootstrap ΔAUC CI.
5. Verdict per D4, quoted verbatim, with achieved SE + empirical r attached.

No verdict without its artifact.
