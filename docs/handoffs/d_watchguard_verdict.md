# D-WATCHGUARD — session verdict + handoff

**Date:** 2026-07-09. **Operator:** Tom. **Executed by:** Claude Code session (dispatcher D-WATCHGUARD).
**Run under review:** `run2_mw_fresh`, live on vast.

This document is **self-contained on purpose**. `reports/**` is gitignored (`.gitignore:34`), so the
detailed reports listed in §7 exist on the working laptop's disk only and cannot be fetched from git. Every
load-bearing number is reproduced here with the evidence that produced it.

**Evidence discipline used throughout:** claims are tagged `[V]` = verified first-hand in this session by
running code / reading source / reading the live log, or `[S]` = produced by a subagent and **not**
independently re-verified. Where a subagent's headline claim was consequential, it was re-run first-hand
and is tagged `[V]`.

---

## 0. Executive verdict

**Three of the four work packages found that an *instrument* was defective, not the model.** One work
package produced a substantive result about the net, and it re-ranks the run3 plan.

1. The 400k "emergency" did not exist. The dispatcher named the wrong live branch; the defect had already
   been fixed 19 h earlier. `[V]`
2. The `t3` / `v_spread` deadline instrument reads **backwards**, and its monitor chart has been empty for
   the entire run. Its pre-registered breach criterion is **VOID**. `[V]`
3. The trap-flip seeding gate's `KILL>16%` rule **fires by construction** on any checkpoint ladder. The
   instrument is armed; the decision rule was not usable. Now re-specified. `[V]`
4. **Verdict 2 fired hard:** the deployed deterministic head loses the canonical line 0/2, but scores
   **0.594** over 64 paired fair openings. The SealBot weakness is substantially **opening-line-specific**.
   Card #1 (distributional value head) must be re-ranked, and its probe-set generation is **on hold**. `[V]`

A meta-observation worth carrying forward: the dispatcher's two founding premises (live branch, clocks)
were both stale, and two of its pre-registered instruments failed their own audit. The CLAUDE.md prime
directive — *verify the measurement unit before building a frame on it* — was the operative rule this
session, not any modelling insight.

---

## 1. Corrected premises (both dispatcher assumptions were false)

### 1.1 The live branch is `master`, not `phase4.5/d-solver` `[V]`

vast `git reflog` (read-only):

```
fc97cc82 HEAD@{2026-07-08 15:30:09 +0000}: checkout: moving from phase4.5/eval-radius-fix to master
fc97cc82 HEAD@{2026-07-08 14:32:38 +0000}: checkout: moving from phase4.5/d-solver to phase4.5/eval-radius-fix
53ae01bd HEAD@{2026-07-07 18:49:01 +0000}: merge origin/phase4.5/d-solver: Fast-forward
```

`train.py` (PID 266593) started **15:31:10Z**, 61 s after the final checkout, resuming from
`checkpoint_00190000.pt` with `--variant run2_mw_fresh`. So the running interpreter imported post-`dffd5aa`
code. `hexo_rl/eval/eval_board.py` has been on the live checkout since 14:32.

**Any WP reasoning about `d-solver` as "the live branch" is reasoning about a branch that has not run for
~19 h.** (For the record: on `d-solver`, `eval_board.py` is absent, `_resolve_eval_radius` has 0 hits, and
`evaluator.py:206` is a bare `Board.with_encoding_name(...)` — it *would* have failed at 400k.)

### 1.2 Both clocks were optimistic `[V]`

Measured from 44,388 telemetry points, 2026-07-08T20:21Z → 2026-07-09T10:39Z (steps 202,715 → 242,841):

| window | rate |
|---|---|
| last 2 h | 3,438 steps/h |
| last 6 h | 3,433 steps/h |
| last 12 h | 2,981 steps/h |
| last 24 h | **2,806 steps/h** |

Short windows overstate because eval rounds punctuate training (~38 % of wall after `53ae01b`). On the 24 h
rate: **300k ≈ 20 h out** (dispatcher said 15 h); **400k ≈ 56 h out** (dispatcher said 37 h).

---

## 2. WP3(A) — eval radius follows the curriculum. **PASS.**

### The finding that matters even though the verdict is PASS

`v6_live2_ls` registry `legal_move_radius = 5` (`engine/src/encoding/registry.toml:330`).
run2 curriculum (`configs/variants/run2_mw_fresh.yaml:94-98`):

| step range | curriculum r | registry default | discriminating? |
|---|---|---|---|
| 0 – 200k | 4 | 5 | yes |
| **200k – 400k (now)** | **5** | **5** | **NO — they coincide** |
| 400k – 600k | 6 | 5 | yes |
| 600k+ | 8 | 5 | yes |

**run2 currently sits in the one curriculum stage where the bug is unobservable.** A regression that
re-pinned eval to the registry default would produce identical numbers for the whole 200k–400k span and
reappear silently at 400k. "It looks right today" is not evidence. This is the entire content of the 400k
clock.

### Evidence `[V]`

Driven through the **real** code path — real composed config (the six base yamls + variant, as
`scripts/train.py:125-136` assembles them) → unbound `StepCoordinator._resolve_radius` /
`._resolve_eval_radius` → real engine `Board` — not a re-implementation. (The dispatcher's warning was
well-aimed: `tests/test_radius_curriculum.py:47` contains a hand-written *mirror* of `_resolve_radius`,
which by construction cannot detect divergence from the method the eval loop calls.)

Eval radius equals the curriculum radius at every stage boundary, including 400,000 → 6.
`evaluation.legal_move_radius` is absent from the merged config and from every file under the live host's
`configs/`. Tracked `configs/` on the live host: clean.

Direct live-process confirmation, from the run's own telemetry:

```json
{"step": 200000, "radius": 5, "event": "radius_curriculum", "timestamp": "2026-07-08T18:34:56.514795Z"}
```

### Landed

`1e8ffed` — `tests/test_eval_radius_curriculum.py::TestComposedConfigDrivesEvalRadius`. Mutation-checked:

| injected regression | eval radius @400k | guard |
|---|---|---|
| eval pinned to registry default (the `d-solver` behaviour) | 5 | **catches** |
| `legal_move_radius_schedule` dropped | 5 | **catches** |
| live config (control) | 6 | passes |

**No intervention required at 400k.**

---

## 3. WP1 — the `t3` / `v_spread` deadline instrument is VOID

The frame under audit was: *"t3 = 0.101, below the 0.20 abort threshold ⇒ danger."*

### The frame is inverted `[V]` (reproduced first-hand from `logs/run2_mw_fresh.jsonl`)

- `t3_spread`: **n = 227, eff_n = 227**, steps 130,500 → 242,500. **Mean −0.343**, range [−0.913, +0.348].
- **97.4 % (221/227) of all reads are already below the 0.20 "abort" gate.**
- **214/227 reads are below 0.101.** The observed 0.101 sits at roughly the **94th percentile** — among the
  *highest* values this run has produced, not a dangerous low.
- Last 8 reads: `0.112, −0.055, −0.571, −0.208, −0.475, −0.250, −0.233, −0.286`.

### Why the threshold is wrong `[S]`

This run's own untrained step-0 bootstrap (`run2_bootstrap_v6_live2_ls.pt`) scores **T3 = −0.679** before
any training. The 0.20 / 0.30 / +0.617 thresholds were calibrated against a *different* checkpoint
(`bootstrap_model_v6.pt`) under a default single-window forward that **shape-crashes** on this run's
4-plane multi-window net.

### Two instrument defects `[V]` (event-name census run first-hand)

| event name | count |
|---|---|
| `value_spread_alert` | 225 |
| `value_spread_canary` | 2 |
| `value_spread_alt_skipped_plane_mismatch` | **227** |
| `value_spread` | **0** |

1. The consumer (`hexo_rl/monitoring/run_feed_reader.py:526-527` `[S]`) collects the trend from event name
   `"value_spread"`, which this run **never emits**. `d1m_monitor`'s t3 chart and `vspread_canary_last`
   have been empty/`None` for the **entire run**.
2. The "alt" arm was skipped **227/227** times (plane mismatch). It never computed once.

### Not a cumulative-metric artifact `[S]`

`fire_canary()` (`hexo_rl/monitoring/value_spread_canary.py`, called from `trainer.py:1310-1339`) is
**stateless** — a fresh no-grad forward of current weights over a SHA-pinned frozen 40-position bank at
each checkpoint. It does **not** belong to the `mcts_mean_depth` cumulative class, and shows no
discontinuity at any of run2's three restarts.

### Verdict

Per the dispatcher's own rule — *"if any benign explanation survives, the deadline breach criterion is VOID
until re-based"* — a benign explanation not only survived, it was **confirmed**.

**BREACH CRITERION = VOID-PENDING-REBASE. Not rescored.** Nothing to do at the 300k read.
Re-basing requires: (a) fix the consumer event name; (b) calibrate against *this* run's multi-window
bootstrap; (c) de-noise the statistic; (d) re-derive thresholds. That is a design task, not a watch task.

---

## 4. WP3(B) — seeding gate: instrument ARMED-NOW, rule was inapplicable

### Was it armed? No. `[S]`

Sole producer is `scripts/eval/run_l1_trapflip_smoke.py`. **Cadence: none.** No run2 trap-flip artifact
exists locally or on vast; no cron/timer; zero trapflip references in the launch/resume scripts. The run
spec (`docs/designs/d_run2_multiwindow_run_spec.md:641`) lists it as a manual operator monitoring entry
that was never executed.

### Now armed offline `[S]`

- Sign pinned at source: `run_l1_trapflip_smoke.py:110` — `flip = sum(c == "saving" ...)` ⇒ **higher =
  better**. (This is the sign misread on 2026-07-04 that produced a wrong erosion diagnosis.)
- Fixture trap closed: `threat_probe_positions_v6_live2_ls.npz` does not exist; the guard at
  `probe_threat_logits.py:690-720` correctly hard-fails for `_ls` checkpoints unless the `v6_live2` fixture
  is passed explicitly. The fake `C1 = +10.97` fallback was **not** re-tripped.
- Fresh run2-lineage HOST-MATCH self-baseline minted: 6/31 flip, deploy-disagree 0.0.
- Backfill cost measured: ~164 s/ckpt ⇒ **~2.0 h laptop wall** for the 44 banked checkpoints 50k→240k.

### The real finding: the KILL rule does not transfer `[V]` (verdict logic + statistics re-derived first-hand)

Both validation reads returned `KILL>16%`. That is **not** a defect signal and **not** "base-rate drift".
It is a category error in applying the gate.

`decide()` (`run_l1_trapflip_smoke.py:196-213`) tests KILL **first**:

```python
if kill_rate is not None and kill_rate > kill_thr:   # -> KILL>16%
```

`kill_rate` is `deploy_disagree_rate` from `kill_gate(base_eng, cand_eng, normal_boards)` (`:180-193`): the
fraction of held-out **normal** positions where the candidate's deploy move differs from the **baseline's**.
Its design context is a **same-step two-arm contrast** (solver-injected arm vs control arm), where any
disagreement on non-tactical positions is, in the script's own words, "honest collateral corruption"
(`:138`).

Across a checkpoint **ladder**, baseline and candidate are separated by real training. Step 0 → step 5000
changes >16 % of normal-position deploy moves *because the net learned*. **KILL fires by construction on
every pair**, and the series conveys nothing.

Neither reported erosion is real:

| set | baseline → candidate | relative | Fisher exact p |
|---|---|---|---|
| reg-31 | 6/31 → 3/31 | −50 % | **0.473** |
| comb-125 | 22/125 → 16/125 | −27 % | **0.379** |

At reg-31 a **single position** is a ~17 % relative move — larger than the entire 16 % threshold. The
script's own docstring concedes "thin power (31 traps)" (`:32`).

### OPERATOR DECISION 2026-07-09 — longitudinal rule, frozen

```
series point = flip_rate on comb-125          (NOT reg-31)
  sign  : higher = better (flip counts SAVES)
  GENERALIZES  : flip >= 0.25
  MEMORIZES    : flip <= 0.16   (memorization floor)
  else         : INDETERMINATE
  KILL co-gate : OMITTED — a checkpoint ladder has no control arm
  report eff_n = distinct positions alongside nominal n
```

The ~2.0 h backfill is **unblocked by this decision but was not run.**

---

## 5. WP2 — VERDICT 2 FIRED. This is the session's substantive result.

Harness: `scripts/watchguard/verdict2_opening_line_probe.py` (new, this session).
Checkpoint `run2_175k.pt` (step 175,000, `v6_live2_ls`, commit `53ae01bd`, stamped 2026-07-08T10:42Z —
provenance matches vast's `checkpoint_00175000.pt` mtime exactly).
Radius **resolved from the checkpoint** = 4. `legal_set = true` (matches the real deploy read,
`needs_no_drop_bot(v6_live2_ls) == True`). Opponent: fixed-depth SealBot **d5** read from
`configs/eval.yaml → eval_pipeline.opponents.deploy_strength.sealbot_max_depth`, 600 s ceiling so depth
binds. Head knobs read from the checkpoint config: `gumbel_m=16, n_sims_full=150, c_puct=1.5, c_scale=1.0`.
33 min, 130 games.

### 5.1 Result `[V]`

| arm | n | eff_n | head WR |
|---|---|---|---|
| canonical line (deterministic, empty board) | 2 | 2 | **0.000** |
| fair `book_v1`, 64 openings × colours swapped | 128 | **128 distinct suffixes** | **0.5938** |

Pair-level bootstrap (2,000 replicates, 64 pairs): **WR 0.5938, 95 % CI [0.5078, 0.6719]**.
**Gap = 0.5938, CI excludes 0.**

Integrity gates: `bad_pairs = 0` — both arms of every pair verified **from the move records** (not the
harness log) to share the opening and to have the head actually move; `censored_games = 0` (nothing hit
`max_plies=200`); `eff_n = 128/128` distinct post-opening suffixes ⇒ **no transposition collapse**, the
red-team's `book_v2` escalation is not triggered.

**Frozen criterion:** *Gap ≥ 0.15 with CI excluding 0 → weakness is substantially OPENING-LINE-SPECIFIC →
re-rank run3 cards before committing card #1; the value-formation frame takes a hit.*
Gap 0.594 ≫ 0.15. The pre-registered INDETERMINATE-UNDERPOWERED band [0.12, 0.18] does not apply.

### 5.2 Two artifact hypotheses, both tested and killed `[V]`

**(a) "SealBot degrades on long, fragmented late boards."** SealBot emitted `sealbot_colony_bug_risk`
(clusters ≥ 5) in late positions, so the worry was that the head's wins came from dragging games out.
**Refuted, and inverted:**

| outcome | n | plies mean | median | range |
|---|---|---|---|---|
| head win | 76 | **56.5** | 54 | 29–153 |
| head loss | 52 | **90.0** | 89 | 25–183 |

The head **wins short and loses long**. Both canonical losses ran long (95, 93 plies), consistent with its
loss profile. (Game length is *downstream* of play — a collider — so it cannot be conditioned on to
separate hypotheses; it is reported only to kill this specific artifact.)

**(b) The §174 prior.** `configs/eval.yaml:182` records:
`eval_random_opening_plies: 0  # §174: was 4 — inflated WR by giving free positional advantage`.
Re-validation was mandatory: colour-pairing cancels **colour** skew but **not** a bot-asymmetry — a
scattered opening could simply degrade SealBot. Opening fragmentation is **randomized**, hence an exogenous
treatment. Pair-level (colour cancelled):

| opening max pairwise hex-distance | pairs | pair score |
|---|---|---|
| tight (≤3) | 11 | **0.500** |
| mid (4–5) | 31 | 0.581 |
| spread (≥6) | 22 | 0.659 |

Pearson r(pair score, opening spread) = **+0.162** (n = 64, p ≈ 0.20, **not significant**). All 64 openings
induce exactly 2 clusters, so cluster count has no variance to explain anything.

**Reading:** the §174 effect is real in direction but small — it spans ≈0.16 across the whole fragmentation
range and is not statistically resolved. It cannot produce a 0.594 gap. Decisively, the **tightest**
openings — least fragmented, closest to the canonical contiguous line — still score **0.500** against the
canonical **0.000**. The gap survives exactly where the prior predicts it should collapse.

*(Protocol note: prior cited → context stated (unpaired legacy evaluator, free positional advantage) →
transfer tested (pairing cancels colour but not bot-asymmetry; fragmentation trend measured) → prior
retained as a small real effect, not used to drop the finding.)*

### 5.3 The honest caveat — read this before quoting 0.594

**The canonical arm has eff_n = 2 *structurally*.** The canonical line is a single deterministic trajectory
(g=0 Gumbel, no temperature, empty board), so the head plays it once as P1 and once as P2, and there is
nothing else to sample. Its 0.000 carries no sampling variance because **it is not a sample — it is the
deployed behaviour.** The gap's CI is therefore the fair arm's CI shifted by a constant, exactly as the
frozen criterion specifies ("pair-CI").

So the correct claim is **not** "the head is 0.59 strong and the canonical read was unlucky." It is:

> **The deployed deterministic head walks into a losing line from the empty board, while under any opening
> diversity it scores ≈0.59.**

This is §D-ARGMAX made concrete: a deterministic deployment regime has effective sample size ~2, and any
strength number read off it is a reading of *one line*.

Large first-player advantage, which pairing cancels and which makes any **unpaired** WR read meaningless:
head as P1 = 0.734, head as P2 = 0.453 (spread 0.281).

### 5.4 Magnitude sanity check `[V]`

`scripts/d_decode/stage_d_wr_vs_sealbot.py:8` records a banked baseline: DeployHeadBot alone vs SealBot d5,
`opening_plies=4`, **WR = 0.47**. Our fair-book read at step 175k gives 0.594 under 3 opening plies. Same
order, plausibly higher for a later checkpoint. The instrument is not producing a nonsense number.

Note the in-loop deploy gate already uses `opening_plies: 4` (`configs/eval.yaml:138`), so the in-loop
`wr_sealbot` is **not** the canonical 0.000 read — it already carries some diversity. The canonical 0.000 is
the pure deploy/arena regime.

### 5.5 Consequences

1. **Re-rank run3 cards before committing card #1.** The value-formation frame takes a hit: a substantial
   part of the observed SealBot weakness is opening-line-specific, not a value-head formation defect.
2. **HOLD WP4 / card #1's ~400-game probe-set generation.** This was the sequencing reason to run verdict 2
   first, and it paid: an overnight would have been spent on a card whose premise just weakened.
3. **A cheap inference-time lever now exists.** The deployed head loses a line it walks into
   deterministically. A deploy-side opening book, or root temperature on the **first turn only**, plausibly
   recovers ≈0.59 with **no training run**. This should be costed before any run3 card.
4. **Registered follow-up (not run, pre-registered here not later):** the fragmentation trend (r = +0.162)
   is unresolved at n = 64. Discriminator = tight-openings-only `book_v2` at n ≈ 128 pairs.

### 5.6 Design corrections folded into the harness

- The dispatcher's WP2 spec **contradicted itself**: openings "sampled INSIDE the curriculum radius of the
  checkpoint being evaluated" **and** "same book across all checkpoints" are incompatible across the 200k
  boundary (r=4 → r=5). Resolved by **separating BOOK radius from GAME radius**: radius is
  monotone-permissive (`engine/src/board/moves.rs:103-110` — legal = empty cells within a hex ball of
  radius R, so r=4 ⊂ r=5 ⊂ r=6 `[V]`), so one r=4 book is legal *and* in-distribution at every checkpoint,
  and slope reads stay **within a single curriculum stage**.
- **Seed-derived books are unsafe** `[V]`: `deploy_strength_eval.py:350` derives pair seeds as
  `seed_base + (hash((label_a, label_b)) & 0xFFFF)*1000 + gi` — checkpoint-label-dependent **and**
  `PYTHONHASHSEED`-salted. `_play_pair` / `gumbel_ladder` cannot be reused for CRN. The book is
  **materialized** (`book_v1.json`). (`_play_one_game(seed=…)` itself *is* CRN-safe — the opening is a pure
  function of the seed — but only when an explicit seed is passed.)
- **Turn-vs-ply** `[V]`: an empty board's legal set is a fixed **25-cell** region regardless of radius
  (`moves.rs:95-101`), so a 1-turn book yields ≤25 distinct openings < the 64 required. `book_v1` =
  **2 turns = 3 plies**, turn-boundary asserted (`moves_remaining == 2` after the opening). A HTTT turn
  places two stones; P1's opener places one.

---

## 6. WP4 — card #1 premise stands, but the discriminator is NO-TEST today

### The card-deciding claim, verified at source `[V]`

Read from the local shrimp clone (`/home/timmy/Work/Hexo/hexo-bot/packages/shrimp/python/shrimp/`):

- `VALUE_BINS = 65` (`constants.py:74`); support is `linspace(-1, 1, 65)`.
- Target is a **two-hot adjacent-bin encoding of the game outcome** z ∈ {+1, −1, 0} via `_winner_value()`
  (`samples.py`). Loss = masked CE; decode = softmax expectation. **Not** moves-to-end, **not** margin.
- `value_target = (1 - lam)*hard_z + lam*root_value if lam > 0.0 else hard_z`, with
  `soft_z_lambda: float = 0.0` and **no caller overriding it anywhere in the repo**.

**No teacher net in the loss path ⇒ the card's premise survives the falsified-distillation register
(D-INJECT NO-GO).** Pinning λ = 0 as invariant **INV-D1** is also correct: `root_value` is the blind net's
*own* MCTS root value, i.e. exactly the "cheap TD bootstrap" that D-PERCEPT excludes (D-PERCEPT's endorsed
lever is *search*-distilled targets — deep solver, not shallow self-bootstrap).

HeXO's replay row already carries `outcomes` + `value_target_valid`
(`engine/src/replay_buffer/sample.rs:354-356,395` `[S]`) ⇒ **zero schema change**.

### The hard gate is not met `[S]`

~30 eligible run2-lineage probe positions exist today against the pre-registered **≥200 distinct** floor.
Old-lineage sets (71 positions) are excluded for d1m contamination. Shortfall ~170 ⇒ ~400 fresh
paired-opening games (overnight) + ~3–17 CPU-h of d7 verification.
**Do not run underpowered — that banks a fake NULL**, which the asymmetric verdict exists to prevent.

Red-team `[S]`: 0/46 (N=2) and 0/25 (N=3) candidates were already mate-proven at the sample ply
(construction sound), **but 17 % of N=2 candidates had SealBot score ≥ 0** — one a forced *win* (a blunder
in between, not a formation walk-in). An eligibility clause `score < 0` was added. 78 % show
`net_value > 0`, the exact D-LOCALIZE blindness — the class is real. Caveat: the spot-check corpus is
old-lineage; re-measure on the first 40 fresh games.

Power `[S]`: +0.05 AUC at the bare 200/class floor reaches power ≈ 0.79 **only if** between-arm score
correlation r ≳ 0.85; detectable effect is 0.071 at r = 0.7. Target 250–300; publish achieved SE and
empirical r with the verdict.

**The asymmetric verdict is restated verbatim in the design doc.** A frozen-trunk NULL is **not** a KILL —
any later agent citing one as a kill is misusing the register.

### Status

**BLOCKED by §5.5.2.** Do not generate the probe set until card #1 is re-ranked in light of verdict 2.

---

## 7. Measured costs (all `[V]`, RTX 4060 Laptop / Ryzen 7 8845HS)

| component | measured |
|---|---|
| deploy head, Gumbel `m=16, n_sims=150` | **245 ms / ply** |
| SealBot **d5** @ 4 / 12 / 20 / 28 plies | **1.92 / 1.67 / 1.11 / 0.58 s per turn-search** |
| SealBot d4 | 217 ms |
| SealBot d6 | **26,769 ms** |

d5 is the canonical bar (`configs/eval.yaml:140`, used at `deploy_strength_eval.py:443`). **d6's 26.8 s is
D-LOCALIZE's diagnostic depth and is a red herring for any WR harness.** SealBot searches once per **turn**,
not per ply (`sealbot_bot.py:98-106` caches the turn's second stone). d5 cost *falls* as the board fills
(alpha-beta prunes harder).

Consequences:

- **WP2 Series A** (the full retro slope, 8 ckpts × 64 games): **3.0 h (50-ply games) – 5.4 h (90-ply)**.
  The WP2 design doc's 9–16 h rests on an **inferred** 4–8 s per SealBot search — a 3–7× overstatement.
- **WP3(B) trap-flip backfill**, 44 ckpts: ~2.0 h.
- **Verdict 2** (the decisive read): **33 min**, actual.

---

## 8. Environment notes for whoever runs the next job

- **Laptop NVML is broken** `[V]`: kernel module `595.71.05` vs userspace NVML `610.43.02`. **CUDA compute
  is fine** (`torch.cuda.is_available()` true; matmul verified on the 4060). But `nvidia-smi` and any
  pynvml-based GPU telemetry fail. A reboot fixes it. Any harness reading GPU util via NVML will error/warn.
- **Host mismatch binds** `[V]`: laptop `torch 2.11.0+cu130` / sm_89 (RTX 4060) vs vast `torch 2.11.0+cu128`
  / sm_120 (RTX 5080, driver 570.172.08). Weights are portable (`state_dict`); float results are not
  bit-identical. The **HOST-MATCH** rule applies to every offline number — and happens to agree with WP2
  verdict 3 (never compare the offline instrument to the in-loop vast number).
- **`master` HEAD was red before this session** `[V]`: `dffd5aa` removed `from engine import Board` from
  `hexo_rl/eval/evaluator.py` but left `tests/test_model_player.py:77` patching the removed
  `evaluator.Board`. `make test` had been failing at HEAD since. Fixed test-only in `2dc8410`; the suite is
  now green (**2252 passed**, Rust all ok). Worth noting that the commit which fixed the live eval-radius
  defect shipped with a red suite.
- **Checkpoint cadence anomaly is benign and fully explained** `[V]`: 5k grid to 235000, then 500-stride
  237500…242500. `checkpoint_interval: 500` (`configs/training.yaml:84`) + `anchor_every_steps: 5000`
  (`:220`) + `max_checkpoints_kept: 10` (`:85`). The surviving non-anchor checkpoints number **exactly 10**
  (240000 is a 5k anchor, so it doesn't consume a rolling slot) — the observed set matches the policy
  exactly. Disk 61 % used, 40 G free, `disk_guard` SIGTERMs at 5 G. **No disk risk.**
- **Live-host hygiene** (not correctness) `[V]`: untracked scratch on vast — `build_post.log`, `co_ws3.sh`,
  `co2_ws3.sh`, `deploy_diag.py`, `depth_{analysis,deep,fine}.py`, `eval_{ctrl,ws3}.sh`,
  `configs/variants/z2_solver_in_loop_control.yaml`. Tracked files clean.
- **Never `git pull` or run anything heavy on the live host.** All of this session's compute ran on the
  laptop; vast was touched read-only (`ls` / `tail` / `grep` / `git reflog`) plus one rsync of the log.

---

## 9. Document + artifact index

### Tracked in git (the admin agent can fetch these)

| path | what |
|---|---|
| `docs/handoffs/d_watchguard_verdict.md` | **this document** |
| `docs/designs/run3_d1_distributional_head.md` | WP4 — card #1 design + pre-registration (30,414 B) |
| `scripts/watchguard/verdict2_opening_line_probe.py` | the verdict-2 harness (reproducibility record) |
| `tests/test_eval_radius_curriculum.py` | WP3(A) 400k boundary guard (commit `1e8ffed`) |

### On the laptop's disk only — `reports/**` is gitignored (`.gitignore:34`)

| path | bytes | what |
|---|---|---|
| `reports/watchguard/AGGREGATE.md` | 17,194 | per-WP verdicts against the frozen criteria |
| `reports/watchguard/VERDICT2.md` | 7,235 | verdict-2 result + both artifact tests |
| `reports/watchguard/preflight_400k.md` | 14,717 | WP3 §A + §B |
| `reports/watchguard/t3_provenance.md` | 24,853 | WP1 full trace |
| `reports/watchguard/_partB_seeding_gate.md` | 15,084 | WP3(B) full trace |
| `reports/watchguard/retro_slope_DESIGN.md` | 30,363 | WP2 design + book_v1 spec + red-team |
| `reports/watchguard/verdict2/{result,meta,book_v1}.json`, `games.jsonl` | — | verdict-2 raw + 130 durable move records |
| `reports/watchguard/wp3b_trapflip/*.json` | — | trap-flip validation summaries (rescued from scratchpad) |
| `reports/watchguard/_data/run2_mw_fresh.jsonl` | 169,420,338 | the pulled live telemetry |

### Commits landed this session (on `master`, **unpushed**)

| sha | subject |
|---|---|
| `2dc8410` | `fix(test)`: patch the real eval-board seam in test_model_player |
| `b9e887f` | `feat(deploy)`: one-turn veto — sound insurance, OFF for strength claims |
| `1e8ffed` | `test(eval)`: pin the composed-config eval-radius path at §174 stage boundaries |

WP0 note: the dispatcher's file paths were wrong (`hexo_rl/deploy/`, `hexo_rl/search/`, `tests/`); the real
surface is `hexo_rl/eval/{turn_veto,deploy_strength_eval,gumbel_search_py}.py` + `tests/eval/test_turn_veto.py`.
Verified before committing the "insurance only" claim: `veto=False` is the default, `get_move` is
byte-identical when off, and **all 23** `DeployHeadBot` construction sites — including the `cand`/`best`
bots inside `deploy_strength_eval.py` itself — take the default. No `engine/src/{mcts,replay_buffer,
game_runner}` file touched ⇒ **no bench gate owed**.

---

## 10. Open items, in priority order

1. **Re-rank the run3 cards** in light of verdict 2. Card #1's value-formation premise is weakened.
2. **Cost the inference-time opening lever** (deploy opening book / first-turn root temperature). It is the
   cheapest thing on the board: no training run, plausibly recovers ≈0.59 from 0.000.
3. **HOLD** card #1's ~400-game probe-set generation until (1) resolves.
4. **Re-base the t3 instrument** (fix the consumer event name first — the chart is empty, so nobody has been
   reading it anyway).
5. Trap-flip backfill (~2.0 h) is unblocked by the §4 rule decision, but only worth spending when a seeding
   decision is actually on the table.
6. WP2 Series A slope (3–5 h) — informative, not urgent. Requires a checkpoint rsync off vast.
7. `book_v2` tight-openings discriminator, if the r = +0.162 fragmentation trend ever becomes decision-relevant.
