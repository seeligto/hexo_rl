<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §155 — Phase B' v10: training-mode knob isolation + bootstrap-floor gate — 2026-05-05

**Trigger:** §154 v9 hex-trunk falsified.  Two-class signal opens v10:
(a) which knob in the smoke v6 step-0 self-play causes 92 % draws under
frozen v7full when the same weights produce 3 % draws at T2's
eval-style hyperparams; (b) eval-gate Class-5 colony-attractor coupling
needs a structural guard before any architecture-altering smoke can run
again.

### TL;DR

Two structural pieces shipped on `phase_b_prime_v10_root_cause` (off
master at `28a7892`).  T1 split into two passes — T1 (R0–R5) ruled out
the three named exploration knobs and 18-worker parallelism; T1.1
(R6–R10) located the cause in the **super-additive interaction**
between the smoke MCTS regime (playout_cap + completed_q_values) and
the exploration knobs (Dirichlet + cosine temperature + opening_plies=1).
None of the MCTS sub-knobs alone or together (R6/R7/R8/R9 all ≤ 5.5 %
draws) is the cause; only the conjunction in R10 produces 91 % draws.
**Training updates are not required to reproduce the 92 %** — the
v7full bootstrap policy under the smoke MCTS+exploration regime
already fixes-points on Class-4 stride-5 chains.

* **T1 — knob-isolation harness** (`scripts/v7full_training_knob_isolation.py`).
  R0–R5 first pass; R6–R10 follow-up.  All variants frozen v7full both
  sides, n=200, single code path = `SelfPlayRunner` + `InferenceServer`.
* **T2 — bootstrap-floor multi-anchor gate**
  (`hexo_rl/eval/eval_pipeline.py`, `configs/eval.yaml`).  Default off;
  AND-combines `wr_bootstrap_anchor ≥ floor.min_winrate` with the
  existing `wr_best ≥ 0.55` + `ci_lo > 0.5` gates.  Designed to block
  the v9 Class-5 colony-attractor failure mode in any future sustained
  run.  37 + 5 new tests on `tests/test_eval_pipeline.py` pass; all 54
  eval-stack tests pass.
* **T3 — branch hygiene.**  Master at `28a7892` (v8 plumbing) unchanged.
  v9 branch retained as architecture-research substrate (knobs default
  off; production paths unaffected).
* **T4 — sustained-run pre-flight smoke.**  **BLOCKED.**  T1.1 verdict
  identifies a super-additive interaction, not a single knob, so the
  fix slot in `configs/variants/w4c_smoke_v7_5080.yaml` cannot be
  pinned without a §156 within-R10 bisection.

### T1 — first pass (R0–R5): three exploration knobs + parallelism are NULL

Variant set (n=200 each, frozen v7full both sides, sims=96 held
constant, all MCTS-side knobs at T2-baseline values):

| variant | knob added vs R0 | rationale |
|---|---|---|
| **R0** | T2 baseline (τ=0.5 fixed, no Dirichlet, opening_plies=4) | sanity — must hit ~3 % draws |
| **R1** | + Dirichlet (ε=0.10, α=0.05) | smoke v6 default; §143 γ.2, §115 |
| **R2** | + cosine temp (1.0 → 0.1 over compound_move [0,10)) | smoke v6 default; §143 γ.1 |
| **R3** | + `random_opening_plies=1` | smoke v6 default (T2 used 4) |
| **R4** | R0 + R1 + R2 + R3 (all three) | full exploration regime |
| **R5** | R4 with `n_workers=18` (parallel) | parallel-worker variance test |

Result on 5080 vast.ai (n=200, single-batch infrence per worker count):

| Variant | n | draws | draw_rate (95 % CI) | mean_ply | wall |
|---|---:|---:|---|---:|---:|
| R0 | 200 | 0 | 0.0 % [0.0 %, 1.9 %] | 56 | 923 s |
| R1 | 200 | 4 | 2.0 % [0.8 %, 5.0 %] | 56 | 898 s |
| R2 | 200 | 4 | 2.0 % [0.8 %, 5.0 %] | 54 | 888 s |
| R3 | 200 | 7 | 3.5 % [1.7 %, 7.0 %] | 75 | 1319 s |
| R4 | 200 | 8 | 4.0 % [2.0 %, 7.7 %] | 71 | 1285 s |
| R5 | 200 | 6 | 3.0 % [1.4 %, 6.4 %] | 66 | 260 s |

Sub-verdicts:
* **Dirichlet alone (R1)**: NULL.  +2 draws over baseline.
* **Cosine temp alone (R2)**: NULL.  +2 draws over baseline.
* **opening_plies=1 alone (R3)**: small effect.  +7 draws (3.5 %),
  +19 mean ply.  The longer games are consistent with more search-
  deciding positions but stay well below the 50 % gate.
* **All three combined (R4)**: NULL.  +8 draws (4.0 %).  No super-
  additive interaction at sims=96 sequential.
* **18 workers (R5)**: NULL.  3.0 % within R4's CI; ~5× wall speedup
  is a clean throughput win, not a policy-distribution shift.

### T1.1 — second pass (R6–R10): MCTS-side knob isolation

After R0–R5 came back NULL, the held-constant MCTS-side knobs
(`completed_q_values`, `playout_cap{full_search_prob, n_sims_quick,
n_sims_full}`, `mcts.n_simulations`) became the candidate set.  T1.1
adds five variants on top of R0:

| variant | knob added vs R0 | rationale |
|---|---|---|
| **R6** | + `mcts.n_simulations: 96 → 600` | deep search alone |
| **R7** | + playout_cap{`fsp=0.5`, `q=100`, `f=600`} | move-level cap |
| **R8** | + `completed_q_values: True` | CQV alone |
| **R9** | R7 + R8 | full smoke MCTS regime |
| **R10** | R9 + Dirichlet + cosine temp + `opening_plies=1` | matches smoke v6 step-0 exactly |

All variants n=200, frozen v7full both sides, n_workers=18 (smoke v6
parallel context).  R10 differs from smoke v6 step-0 self-play only by
not running trainer gradient updates (and `legal_move_radius_jitter`
off — the v8 plumbing knob, off in master too).

Result on 5080 vast.ai (n=200, 18 workers, 103.9 min total wall):

| Variant | n | draws | draw_rate (95 % CI) | mean_ply | stride5 P50/P90 | rmax P50/P90 | colony_wins | wall |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| R0 | 200 | 2 | 1.0 % [0.3 %, 3.6 %] | 52 | 2 / 3 | 9 / 13 | 95 | 821 s |
| R6 | 200 | 11 | 5.5 % [3.1 %, 9.6 %] | 62 | 3 / 4 | 9 / 14 | 101 | 1219 s |
| R7 | 200 | 1 | 0.5 % [0.1 %, 2.8 %] | 54 | 3 / 4 | 8 / 13 | 104 | 639 s |
| R8 | 200 | 4 | 2.0 % [0.8 %, 5.0 %] | 52 | 3 / 3 | 8 / 13 | 100 | 202 s |
| R9 | 200 | 5 | 2.5 % [1.1 %, 5.7 %] | 56 | 3 / 4 | 9 / 14 | 109 | 653 s |
| **R10** | **200** | **182** | **91.0 % [86.2 %, 94.2 %]** | **140** | **84 / 97** | **101 / 112** | **9** | **2702 s** |

Terminal-reason breakdown — every R10 draw is `ply_cap`, zero
`other_draw`, zero engine-level colony.  `colony_wins` (the column
counting decisive games where the *winner side* held a colony) drops
from ~100 baseline to 9 in R10 — there are simply very few decisive
games at all, not a colony-rule shift.

### T1.2 — verdict: PROXIMATE_CAUSE_FOUND (super-additive interaction)

R10 reproduces the smoke v6 step-0 92 % draw collapse under frozen
v7full weights — **91.0 % [86.2 %, 94.2 %]** — *without any gradient
updates*.  This rules out:

* training-loop hypotheses (value-head feedback, fresh-buffer bias,
  first-N-step gradient drift) as *required* contributors;
* single-knob hypotheses on the MCTS side (R6/R7/R8 individually all
  ≤ 5.5 %, well below the 50 % gate);
* `playout_cap × CQV` interaction in the MCTS regime alone (R9 = 2.5 %).

The pathology is the **conjunction** of the smoke MCTS regime
(playout_cap + CQV at 18 workers) AND the exploration knobs (Dirichlet
+ cosine temperature + `opening_plies=1`).  T1's R4/R5 (exploration
alone, sims=96) gave 3-4 % draws; T1.1's R9 (MCTS regime alone) gave
2.5 % draws; only R10 (both sets at once) hits 91 %.  The 90-pp gap
from R9 → R10 is far above what any additive model of R4/R5 + R9 would
predict (≈ 6.5 %).

The Class-4 stride-5 chain pattern (§152) is the engine.  R10's
stride5 P90 = 97 against R0's 3 — **32× amplification**, exceeding the
§152 instrumented-smoke 10× amplification (smoke 30 vs T2 baseline 3).
mean_ply 140 (cap = 150) and ply_cap rate 91 % match the §152 cap-rate
86.7 %.  Class-4 is not a property of the trained policy at step 2500;
it is a property of v7full *evaluated under the smoke
MCTS+exploration regime*.  Training updates merely fix-point on what
the regime already produces at step 0.

### T1 — operational lesson

Both passes ran on the 5080 vast.ai (`REMOTE_HOST:REMOTE_PORT`) under
`n_workers=18`, batch=224, wait=8 ms — the §138 5080 sweep verdict.
T1.1 R10 wall was 2702 s vs R9's 653 s under the same regime — the
4× wall increase is consistent with the 91 % cap-rate (games run to
the 150-ply cap instead of resolving by ply 50-60).

### T2 — bootstrap-floor gate: the v9 Class-5 fix

(unchanged from earlier draft)

Class 5 (newly identified in §154): under HexConv2d + corner_mask, 500
self-play steps converge to a `wr_best ≈ 0.86–0.91` local optimum that
wins via colony pattern (16-21 % of head-to-head wins).  The eval gate's
`wr_best ≥ 0.55` criterion is satisfied; promotion fires; the new anchor
is ~5 pp WORSE on SealBot than the bootstrap.

The bootstrap-floor gate is the structural guard.  AND-combines the
trainer's WR vs a frozen reference model (typically the canonical
bootstrap, e.g. v7full) with the existing best-checkpoint gate.  Under
v9's failure mode the trainer's colony-leaner would crush the rotating
v8full_warm anchor 86–91 % (clears `wr_best`) but lose 199/200 vs
v7full (collapses against `bootstrap_floor.min_winrate=0.45`) →
promotion blocked.

Knob surface (`configs/eval.yaml`, default off):

```yaml
eval_pipeline:
  opponents:
    bootstrap_anchor:
      enabled: false                                # opt-in
      stride: 1                                     # match best_checkpoint
      n_games: 100
      model_sims: 128
      opponent_sims: 128
      path: checkpoints/bootstrap_model.pt
  gating:
    bootstrap_floor:
      enabled: false                                # opt-in
      min_winrate: 0.45
```

Anchor model construction is lazy (first eligible eval round) and the
anchor is never reloaded once constructed — disk-side bootstrap
rotation between runs is ignored, so the gate threshold remains a
constant reference.  A persistent `bootstrap_anchor:<filename>`
Bradley-Terry player row is created so rating histories survive
anchor-graduation swaps.

Test coverage on `tests/test_eval_pipeline.py`:
* `wr_anchor < threshold` AND `wr_best ≥ 0.55` → block promotion
* `wr_anchor ≥ threshold` AND `wr_best ≥ 0.55` → allow promotion
* floor disabled, low `wr_anchor` → does NOT block (informational)
* floor enabled, `bootstrap_anchor` opponent disabled (stride mismatch)
  → block promotion (defensive default)
* `eval_games` sum includes the bootstrap-anchor opponent's games

### T3 — branch hygiene

Pre-§155: 4 phase_b_prime branches plus master.

```
master                              28a7892  v8 plumbing (already merged)
phase_b_prime_v8_plumbing           28a7892  identical to master — redundant
phase_b_prime_q3_cluster6           432096c  CLUSTER_THRESHOLD 5→6 (FALSIFIED §148)
phase_b_prime_t5_corner_mask        00fc651  corner-mask encoder (FALSIFIED §148)
phase_b_prime_v9_hex_native         4b4d507  hex-trunk + per-move (FALSIFIED §154)
```

Post-§155:

```
master                              28a7892  unchanged — v8 plumbing live, defaults preserved
phase_b_prime_v9_hex_native         4b4d507  retained — future architecture research substrate
phase_b_prime_v10_root_cause        3ad8cb7  this sprint
```

`origin/phase_b_prime_q3_cluster6` remains on remote (push-delete out
of §155 prompt scope).  `phase_b_prime_v8_plumbing` local kept as a
named marker of the merged commit (no functional weight).

### T4 — sustained-run pre-flight smoke (BLOCKED — surfaces §156 scope)

T1.2 verdict identifies a super-additive interaction, not a single
knob, so the exploration-knob fix slot in
`configs/variants/w4c_smoke_v7_5080.yaml` cannot be pinned without a
within-R10 bisection.  The variant template is committed with the v8
jitter + bootstrap-floor pieces wired in and a `# PENDING T1 verdict`
slot for the exploration-knob fix.  Running it now would inherit smoke
v6 defaults and reproduce the 91 % collapse without mitigation.

**§156 next-step scope:** within-R10 bisection.  R11–R14 each remove
one knob from R10, n=200, frozen v7full, 18 workers:

| variant | knob removed from R10 | discriminator |
|---|---|---|
| R11 | Dirichlet (`dirichlet_enabled=False`) | does ε=0 break the 91 %? |
| R12 | cosine temp (`temp_min=0.5`, `threshold_compound_moves=0`) | does τ=0.5 fixed break the 91 %? |
| R13 | `random_opening_plies: 1 → 4` (T2 baseline value) | does long random opening break the 91 %? |
| R14 | playout_cap (`full_search_prob=0.0`, `n_simulations=600`) | does the move-level cap break the 91 %? |

The variant whose removal collapses the 91 % below the 50 % gate is
the load-bearing knob for the smoke v7 fix.  Conservative interim
option: restore *all three* exploration knobs to T2-style (Dirichlet
ε=0, temp_min=0.5, opening_plies=4) — definitely breaks R10's super-
additivity but loses self-play exploration entirely (§115, §143
re-tuning required).  Not recommended without the bisection.

### Class verdict ranking — post-§155 update

| Class | §154 status | §155 update |
|---|---|---|
| 4 — q-axis stride-5 | CONFIRMED dominant; Q2 jitter is sole confirmed lever | **CONFIRMED (root cause).** R10 reproduces 32× stride-5 amplification under frozen v7full *without training*.  Class 4 is a property of v7full × smoke MCTS+exploration regime, not of trained-policy dynamics. |
| 3 — buffer composition | UNCHANGED | downgraded — buffer signal at smoke step ≥ 250 is *consequence* of Class 4 stride-5 cap-spam at step 0, not cause |
| 2 — value-head drift | UNCHANGED | downgraded — value-head locking at decisive_mean ≈ −0.69 (§152) is downstream of the step-0 stride-5 distribution, not an independent failure mode |
| 1 — stale dispatch | UNCHANGED — eliminated under v7full anchor | UNCHANGED |
| **5 — gate / colony-attractor** | NEW; v10 priority 1 | **STRUCTURAL FIX SHIPPED** — bootstrap-floor gate ready for v10 sustained-run variant.  Default off for backward compat. |

### Artifacts

* `scripts/v7full_training_knob_isolation.py` — T1 + T1.1 harness (R0–R10 driver)
* `reports/phase_b_prime/training_knob_isolation/` —
  `results.md`, `summary.json`, `R{0,6,7,8,9,10}_games.jsonl` (T1.1
  pass), `results_R0R5.md`, `summary_R0R5.json` (T1 pass — preserved),
  `R{1,2,3,4,5}_games.jsonl` (T1 pass JSONLs)
* `hexo_rl/eval/eval_pipeline.py` — bootstrap-floor opponent + gate logic + lazy anchor loader
* `configs/eval.yaml` — `bootstrap_anchor` + `bootstrap_floor` config surface (default off)
* `tests/test_eval_pipeline.py` — 5 new tests for the floor gate paths
* Engineering branch: `phase_b_prime_v10_root_cause`

Commits: `b62a1c0` (T1 + T2 + T3), `2fd6fd6` (master compat fix —
drop v9-only `rotation_cadence` kwarg), `bcb2613` (perf — right-size
inference batch+wait per worker count, 3× speedup at n_workers=1),
`e99663c` (smoke v7 5080 variant template — knob-fix slot pending),
`3ad8cb7` (T1.1 — R6–R10 MCTS-side knob isolation).

### What this sprint DOES NOT do

* Does not change master config defaults (every v10 knob ships opt-in).
* Does not regenerate v7full corpus or retrain bootstrap.
* Does not pre-implement Phase 5+ architectural changes (hex window,
  G-CNN, transformers).
* Does not push-delete `origin/phase_b_prime_q3_cluster6` (out of scope).
* Does not pin a single-knob fix in `w4c_smoke_v7_5080.yaml` — the
  super-additive verdict needs a within-R10 bisection (§156) before
  the smoke v7 launch can be authorised.


## §156 — Phase B' v10: R10 within-bisection + cosine-temp fix + laptop validation — 2026-05-06

### Context

§155 T1.1 closed with PROXIMATE_CAUSE_FOUND: smoke v6 step-0 92% draw
collapse under frozen v7full bootstrap is a super-additive interaction
between the smoke MCTS regime (playout_cap fsp=0.5 + completed_q_values +
18 workers) AND the exploration knobs (Dirichlet ε=0.10 + cosine temp
1.0→0.05 over compound_move [0,10) + opening_plies=1).  R0–R9 each null
(≤5.5% draws); R10 = full conjunction = 91.0% draws [86.2%, 94.2%],
mean_ply 140 (cap 150), stride5 P90 = 97 (32× R0 baseline), 91%
terminal_reason ply_cap.

T4 (smoke v7 launch) was BLOCKED until the load-bearing knob inside R10
was identified.  §156 = within-R10 bisection (R11–R14, each removes ONE
knob), fix authoring, laptop validation, sustained-run authorisation
hand-off to §157.

### Gate 1 — R10 within-bisection (R11–R14)

n=200 each, frozen v7full both sides, 18 workers, 5080.  Per operator
instruction: all four variants run regardless of intermediate verdicts —
full information needed for the fix decision.

| Variant | Knob removed                   | n   | draws | draw_rate (95% CI)        | mean_ply | stride5 P50/P90 | rmax P50/P90 | colony_wins | wall  |
|---------|--------------------------------|----:|------:|---------------------------|---------:|----------------:|-------------:|------------:|------:|
| R10     | (none — full smoke regime)     | 200 | 182   | **91.0%** [86.2%, 94.2%]  | 140      | 84 / 97         | 101 / 112    | 9           | 2702s |
| R11     | Dirichlet ε=0.10 → 0           | 200 | 176   | 88.0% [82.8%, 91.8%]      | 139      | 76 / 86         | 96 / 104     | 15          | 2649s |
| **R12** | **cosine temp → fixed τ=0.5**  | 200 | 10    | **5.0%** [2.7%, 9.0%]     | 63       | **3 / 4**       | **10 / 14**  | 134         | 738s  |
| R13     | opening_plies=1 → 4            | 200 | 170   | 85.0% [79.4%, 89.3%]      | 135      | 82 / 100        | 100 / 112    | 15          | 2620s |
| R14     | playout cap → uniform 600      | 200 | 198   | 99.0% [96.4%, 99.7%]      | 149      | 132 / 133       | 133 / 137    | 0           | 3576s |

**Verdict — LOAD_BEARING = cosine temperature schedule.**

R12 is the only variant whose Wilson upper bound (9.0%) clears the 50%
gate.  R11/R13 stay within R10's 91% baseline noise.  R14 (deeper search
on the same exploration regime) **amplifies** to 99% — confirms playout
cap was partially mitigating the lock; uniform 600 sims with cosine still
on pushes the regime even harder onto the Class-4 stride-5 fixed point.

#### Per-knob sub-verdict

* **R11 — NULL.** ε=0 vs ε=0.10 inert once cosine + cap + CQV active;
  Dirichlet noise dominated by τ→0.05 collapse forcing argmax-on-visits
  at compound_move ≥ 10.
* **R12 — LOAD-BEARING.** Cosine collapse drops draws 91→5%, mean_ply
  140→63, stride5 P90 97→4 (back to R0 baseline 3).
* **R13 — NULL.** Lock-in happens at compound_move ≥ 10, well past the
  random-opening window.
* **R14 — INVERSELY LOAD-BEARING.** Removing cap forces uniform deep
  search → policy uses full budget to build longer Class-4 chains
  (stride5 P90 132 vs R10's 97).

#### Colony caveat for fix design

R12 colony_wins = 134/200 = **67%** — the §147 v5 / §154 v9 colony-
attractor signature.  Fixed τ=0.5 alone breaks the draw lock but lights
up the colony failure mode.  Mitigated by the §156 mandatory pairings
(both already in the variant):

1. `selfplay.legal_move_radius_jitter: true` — Q2 §152 verdict, the only
   confirmed Class-4 lever.
2. `gating.bootstrap_floor.min_winrate: 0.45` — promotion AND-requires
   wr_bootstrap_anchor ≥ 0.45 in addition to the existing wr_best ≥ 0.55,
   ci_lo > 0.5 gates.

Full bisection report:
`reports/phase_b_prime/training_knob_isolation/r10_bisection.md`.

### Gate 2 — Phase B' fix in `configs/variants/w4c_smoke_v7_5080.yaml`

```yaml
selfplay:
  playout_cap:
    fast_prob: 0.0
    temperature_threshold_compound_moves: 0   # §156 R12 fix — disable cosine schedule
    temp_min: 0.5                             # fixed τ=0.5 across the game
  legal_move_radius_jitter: true              # §152 Q2 (mandatory pairing)

eval_pipeline:
  gating:
    bootstrap_floor:
      enabled: true
      min_winrate: 0.45                        # §155 T2 (mandatory pairing)
```

Class-3 buffer surgery (`draw_target_fraction: 0.5` subsample-on-push)
deferred per §156 prompt unless trivial in same diff.  Not applied this
wave.

Commit: `cc4fd4e` (variant fix), `01ebd29` (laptop preflight sibling
variant).

### Gate 3 — Branch hygiene

* `phase_b_prime_v8_plumbing` already at master HEAD — no-op merge
  confirmed (`git rev-list --left-right --count master...origin/phase_b_prime_v8_plumbing`
  = `0  0`).  Local `master` carries the v8 commit (`28a7892`); origin
  master push deferred to §157 close-out commit since master diff is
  zero against v8_plumbing.
* `phase_b_prime_t5_corner_mask` not present locally or on origin — no-op.
* `phase_b_prime_q3_cluster6` (origin only) — **DELETED 2026-05-06** per
  user authorisation in §157 prompt.  q3 unique commit (`432096c`
  CLUSTER_THRESHOLD 5→6) was part of the §154-falsified v9 hex-native
  experiment.
* `phase_b_prime_v9_hex_native` retained — knobs default off, future
  architecture research (per §156 hard constraint).
* `phase_b_prime_v10_root_cause` merge to master DEFERRED to §157 Gate 4
  pending 5k smoke pass on 5080.

### Gate 4 — Laptop validation smoke

`w4c_smoke_v7_laptop_preflight.yaml` (sibling of the 5080 variant with the
§156 fix + Q2 jitter + bootstrap_floor; laptop-tuned n_workers=14 /
batch=64 / wait=4ms / fresh buffer via `mixing.buffer_persist=false`).
Run: 254 games in 1000 train iters on 4060 Max-Q, ~50 min wall.  Healthy
throughout (grad_norm < 1.8 vs 10.0 hard-abort, policy_entropy_selfplay
2.99–3.78).

Per-game aggregates (from the run's `game_complete` events):

| Window     | n   | draws | draw_rate | mean_moves | stride5 P50/P90 | rmax P50/P90 | terminals             |
|------------|----:|------:|----------:|-----------:|----------------:|-------------:|-----------------------|
| ALL games  | 254 | 16    | 6.3%      | 73.0       | 3 / 4           | 10 / 15      | 238 six / 16 ply_cap  |
| LAST 100   | 100 | 4     | 4.0%      | 73.0       | 3 / 4           | 11 / 15      | 96 six / 4 ply_cap    |
| LAST 50    | 50  | 0     | 0.0%      | 66.4       | 3 / 4           | 10 / 14      | 50 six / 0 ply_cap    |

Pass criteria (last 100 games):
* draw_rate < 50% → **PASS** (4.0%)
* stride5_run P90 < 30 → **PASS** (4)
* bootstrap_floor not blocking valid candidates → **PASS** (trivial:
  laptop too slow, single eval still running at process exit; no
  candidates evaluated to block)

Colony wins: 0 / 254 — Q2 jitter mitigation working as predicted from
R12's 67% colony rate (the §147 v5 / §154 v9 colony attractor never
fires).  Player split last 100: 56 player-0 / 40 player-1 (58/42
ex-draws) — within normal first-mover noise per
`feedback_winrate_balance.md` 50/50 baseline.

Note on instrumentation event semantics: `instrumentation_periodic`
reports `draw_target_fraction` (training-side weighted target), not raw
outcome draw rate.  Outcome draw rate is the ply_cap fraction (4/100 last
100).

### Gate 5 — Sustained-run authorisation hand-off

§157 (companion sprint) opened to drive the 5k validation smoke on 5080
with the §156 fix.  The sustained 40k run is gated by §157 Gate 4 verdict
+ user path decision (Path A sustained vs Path B encoding-migration
pivot).  R12's colony caveat means the bootstrap-floor gate is the
primary safety net during sustained training.

### Verdict

§156 work complete:

* **Load-bearing knob identified:** cosine temperature schedule
  (compound_move [0,10) cosine 1.0→0.05 with temp_min=0.05).  Single
  knob, falsifies the v9 / v10 super-additive interaction theory in
  favour of a cosine-schedule single-cause model.
* **Fix shape:** disable cosine (`temperature_threshold_compound_moves: 0`)
  and pin τ to T2 baseline (`temp_min: 0.5`).  Mandatory pairings:
  `legal_move_radius_jitter: true` (Q2 colony mitigation),
  `bootstrap_floor.enabled: true min_winrate: 0.45` (regression catcher).
* **Hard-falsified as load-bearing:** R11 Dirichlet ε removal, R13
  opening-plies extension, R14 playout-cap removal.  All three are
  synergy partners on the cosine collapse, not drivers.
* **Laptop validation passed** (254 games / 50 min, draw_rate 4.0% last
  100, stride5 P90=4, colony 0/254).  Hand-off authorised to §157 for
  5080 5k smoke under load.

Commits in §156: `3ad8cb7` (R6–R10 MCTS-side isolation, §155 follow-up),
`548da64` (R10 within-bisection harness R11–R14), `cc4fd4e` (smoke v7
5080 variant fix), `01ebd29` (laptop preflight sibling variant).

### What this sprint DOES NOT do

* Does not change master-config defaults (top-level config propagation
  is §157 Gate 5 work, separate diff).
* Does not authorise the 40k sustained run (gated on §157 5k verdict).
* Does not implement Class-3 buffer surgery (`draw_target_fraction`
  subsample-on-push) — deferred unless §157 surfaces it as needed.
* Does not push-delete `origin/phase_b_prime_v8_plumbing` (still useful
  as the v8 lineage anchor; revisit in §157 close-out).


## §157 — Phase B' v10: 5k validation smoke + hygiene wave — 2026-05-06

### Context

§156 closed with the cosine temperature schedule identified as the sole
load-bearing knob behind R10's 91% draw lock, fix authored
(`temperature_threshold_compound_moves: 0`, `temp_min: 0.5`) with
Q2 jitter + bootstrap-floor mandatory pairings, laptop preflight
(commit `01ebd29`) PASS at 4% draws / stride5 P90 = 4 / colony 0/254.
§157 = production-scale 5k validation on the 5080 + branch hygiene +
top-level config propagation + sustained-run authorisation hand-off.

### Gate 1 — Pre-flight verification

* Remote synced to `01ebd29`, bootstrap sha256 prefix `29306533…`
  matches §150.  Stale `replay_buffer.bin.recent.npz` archived to
  `archive/replay_buffers/`.
* §156 fix knobs verified in `configs/variants/w4c_smoke_v7_5080.yaml`:
  cosine off + `temp_min: 0.5` + `legal_move_radius_jitter: true` +
  `bootstrap_floor.enabled: true min_winrate: 0.45`.
* `bootstrap_floor` predicate verified in
  `hexo_rl/eval/eval_pipeline.py:401-444`: AND-combines
  `wr_best ≥ promotion_winrate (0.55)` + `ci_lo > 0.5` +
  `wr_bootstrap_anchor ≥ min_winrate (0.45)`.  Missing measurement =
  failure (defensive).
* Variant already self-contained for 5080 throughput
  (`gumbel_targets_5080_24t` knobs baked in: n_workers=18,
  inference_batch_size=224, inference_max_wait_ms=8.0,
  max_train_burst=8) — no overlay needed at launch.

### Gate 2 — 5k smoke launch + completion

Launched 2026-05-06 05:46:37 UTC on vast.ai 5080
(`REMOTE_HOST:REMOTE_PORT`); ran in `tmux hexo_phase_b:smoke5k`.
`--checkpoint bootstrap_model_v7full.pt --variant w4c_smoke_v7_5080
--checkpoint-dir checkpoints/w4c_smoke_v7_5k --no-dashboard
--iterations 5000`.  Completed 09:05:47 UTC; wall 11,916 s = 3h 18m 35s;
1,256 games; 5,000 train steps; cost ~$1.20.

Final ckpt: `checkpoints/w4c_smoke_v7_5k/checkpoint_00005000.pt`.

### Gate 3 — Branch hygiene

* `phase_b_prime_v8_plumbing` already at master HEAD (no-op merge).
  Local master one commit ahead of origin/master pending operator
  authorisation.
* `phase_b_prime_q3_cluster6` deleted from origin (user-authorised
  via §157 prompt).
* `phase_b_prime_t5_corner_mask` not present (no-op).
* `phase_b_prime_v9_hex_native` retained (knobs default off, future
  architecture research).
* `phase_b_prime_v10_root_cause` merge to master deferred to Gate 5
  per user instruction (one bundled landing post-config-propagation).

### Gate 4 — Smoke verdict

**PASS on all live abort signatures, self-play health metrics, AND the
SealBot offline eval ≥17% pass criterion (19.0% WR, n=200).**

Live abort signatures (full-run, dashboard + SSH polls):

| signature | end-of-run value | abort threshold | status |
|---|---|---|---|
| stride-5 P90 (rolling 50 games, dashboard) | 4 | 60 | ✅ |
| row max P90 (rolling 50 games, dashboard) | 13 | 50 | ✅ |
| colony_ext_frac max (per-game, n=1256) | 0.086 | 0.40 | ✅ |
| colony_terminal_fraction (8 measurements) | 0.000 | — | ✅ |
| draw_rate (last 200 games) | 7.5% | 70% (WARN-only) | ✅ |
| grad_norm | 0.98–1.62 | 10.0 hard-abort | ✅ |
| NaN losses | 0 | any | ✅ |

Eval verdicts (3 of ~10 planned rounds completed; see follow-up #1
for the cadence finding):

| step | promoted | wr_best | wr_anchor (v7full) |
|---:|:---:|---:|---:|
| 500  | F | 0.34 | 0.28 |
| 2000 | F | 0.48 | 0.42 |
| 3500 | F | 0.39 | 0.37 |

wr_anchor recovered fast (0.28 → 0.42 in 1500 steps), then sampling-noise
dipped to 0.37 (n=100 ⇒ ±10pp 95% CI; CIs heavily overlap with round 2).
Bootstrap-floor gate operated correctly — refused promotion on a
sub-floor model.

Self-play health (final 200-game window): draw_rate 7.5%,
ex-draws-x/(x+o) 51.4%, plies P50/P90/mean = 65/136/76, sims/sec P50
3,707, colony_ext_frac max 0.086.  Q2 `legal_move_radius_jitter`
mitigation held the §156 R12 67% colony rate at trace levels.

Final-checkpoint SealBot offline eval (n=200, 128 sims, time_limit 0.5):
**winrate 0.19 (38/200, 0 draws, 1 colony win)**.  Beats the 17% pass
gate; matches the §150 v7full baseline (17.4% n=500) within sample noise
(95% CIs overlap, Δ +1.6pp).  Confirms §156 fix did not regress strength
against the external benchmark.

Full live-poll trajectory + per-event tables: see
`reports/phase_b_prime/5k_smoke/results.md`.

### Gate 5 — Top-level config propagation

* **PROPAGATED:** `selfplay.legal_move_radius_jitter: true` added to
  `configs/selfplay.yaml` (commit `83be4d7`).  Q2 §152 verdict, only
  confirmed Class-4 lever, no downside in any phase.  953 py tests
  pass post-edit.
* **Surfaced to operator (Gate 6) — both decisions captured in commit
  `f2e4555`:**
  * **S1 — bootstrap-floor default-on, frozen v7full path:**
    `gating.bootstrap_floor.enabled: true` +
    `opponents.bootstrap_anchor.enabled: true` +
    `opponents.bootstrap_anchor.path: checkpoints/bootstrap_model_v7full.pt`.
    Operator rationale: rotating canonical defeats the gate's regression-
    catching purpose by re-anchoring below v7full on the first sub-floor
    promotion.  Frozen v7full anchors the gate to a known-good baseline
    forever; operators rolling new bootstraps must explicitly re-pin the
    path only after independent SealBot validation.
  * **S2 — cosine-temp disable: NOT propagated (variant-pinned).**
    Comment added at the cosine knobs noting §156 R12 load-bearing
    verdict + warm-start opt-out recommendation pointing at
    `w4c_smoke_v7_5080.yaml`.  Cold-start data unavailable; defaults
    reflect "still under investigation".  Variant-pin preserves both
    options.

### Gate 6 — Decision hand-off

Operator decisions captured 2026-05-06:

* **Path B selected** — skip the sustained 40k run; preserve dev cycles
  for the encoding migration (Phase 5+) that's about to obsolete the
  current v6/v7 8-plane trunk.  5k smoke held the §156 fix and produced
  no drift; sustaining 40k on encoding-about-to-die is sunk cost.
* **S1 = yes + frozen v7full** (propagated, see Gate 5 above).
* **S2 = no, comment-only** (variant-pinned, see Gate 5 above).
* **Bundle merge + push:** `phase_b_prime_v10_root_cause` → `master`
  + `master` push to origin/master in one bundled landing post-Gate 6.
  See commit footers + branch state at sprint close.

### §157 follow-ups (methodology / instrumentation)

#### #1 — `eval_interval` too tight for production hardware

Variant `eval_interval: 500` produced 6 `eval_skipped_still_running`
events; only 3 of ~10 planned rounds actually fired.  Each round
(random + best n=400 + bootstrap_anchor n=100, 128/128 sims) takes
~21 min wall on the 5080 — cadence 500 is a guaranteed backlog.
**Future smokes set `eval_interval ≥ 2500`** (5k smoke → 2 rounds at
steps 2500, 5000); for tighter coverage cut `n_games`, not the
interval.

Saved as memory `feedback_smoke_eval_interval_min_2500.md`.

#### #2 — stride5 / row_max metrics dashboard-only

§156 v8 plumbing wired `stride5_run_max_per_game` and
`row_max_density_per_game` as live dashboard metrics, but they are
absent from `events_*.jsonl` payloads (verified across all 1,256
game_complete events of this run).  SSH-poll abort gates that grep
JSONL cannot see them — only the dashboard can.  **Mirror to
`instrumentation_periodic` payload** (or document the dashboard-only
metric set explicitly).

#### #3 — Final eval round skipped on iteration-limit exit

`--iterations 5000` exited the training process before the step-5000
eval round could run — single `evaluation_game_progress` event then
nothing.  Lost the most informative checkpoint's in-loop measurement.
**Either drain the eval queue before iteration-limit exit, or always
auto-run an offline final-checkpoint eval** as part of the smoke
harness (the SealBot eval invoked manually for Gate 4 is exactly
this pattern — automate it).

#### #4 — `sealbot_colony_bug_risk` startup warning

One emission at startup.  Per `feedback_colony_fraction.md` low
colony in eval is positive; this guard is likely a v5-era legacy.
Worth confirming the predicate is still meaningful or pruning.

#### #5 — User feedback: draw_rate not an abort signal

User inspected actual draw games during this run and confirmed model
plays soundly — draws come from the policy missing some open-4
threats, not pathology.  Demoted `draw_rate > 70%` from abort to
WARN.  Saved as memory `feedback_draw_rate_not_abort_signal.md`.

### Verdict

**§157 PASS.**  §156 cosine-temp fix validated at 5k production scale on
5080.  Class-4 stride-5 fixed point that defined §147 v5 / §155 R10 is
broken under load.  Final-checkpoint SealBot WR 19.0% (n=200) clears the
17% gate and matches the §150 v7full baseline (17.4% n=500) within sample
noise — confirms the fix did not regress strength against the external
benchmark.  wr_anchor trajectory (0.28 → 0.42 → 0.37) is the bootstrap-
floor gate's intended signal: model is closing on v7full strength rather
than diverging; gate operated correctly by refusing all sub-floor
promotions.

Path B selected — proceeding to encoding migration (Phase 5+).  No
sustained 40k run on the v6/v7 8-plane trunk.

Commits in §157:
* `9412a38` — `docs(sprint): §156 R10 bisection + cosine-temp load-bearing verdict`
* `83be4d7` — `chore(configs): propagate §156 legal_move_radius_jitter default to top-level`
* `f2e4555` — `chore(configs): §157 Gate 6 — bootstrap-floor default-on + cosine verdict comment`
* (this entry + bundled v10→master merge to follow as the close-out commits)

### What this sprint DOES NOT do

* Does not run the sustained 40k training run — operator selected Path B
  (encoding migration) over Path A (sustained) at Gate 6.
* Does not modify the v8 plumbing instrumentation event schema
  (follow-up #2 left for §158 or later).
* Does not change the `sealbot_eval` script behaviour to auto-run
  on smoke completion (follow-up #3 left for §158 or later).
* Does not prune the `sealbot_colony_bug_risk` legacy guard
  (follow-up #4 left for §158 or later).
* Does not begin encoding-migration work itself — that opens as Phase 5+
  in a subsequent sprint context.

