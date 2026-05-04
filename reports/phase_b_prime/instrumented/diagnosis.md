# Phase B' — Instrumented smoke diagnosis

**Date:** 2026-05-04
**Run:** `w4c_smoke_v6_instrumented_5080` on RTX 5080 (vast.ai), aborted at
**step 2560** of 5000 (51% complete) after the four-class signal saturated.
**Hardware:** Ryzen 9 9900X + RTX 5080 (vast.ai).
**Bootstrap:** `bootstrap_model_v7full.pt` (sha256 `29306533…`, §150).
**Variant:** n_workers=8, batch=32, max_wait=2 ms, leaf_batch=4, sims=400 base
+ playout_cap full=0.5 (n_sims_quick=100, n_sims_full=600), draw_value=-0.5,
max_game_moves=150, eval_interval=5000.
**Walltime:** 179 min, **rate ≈14.3 steps/min**, 640 games completed.

**Verdict-flag:** **MIXED (Class 3 + Class 4 dominant; Class 2 downstream;
Class 1 inconclusive — methodology gap).**

---

## 1. Run summary

| Metric | Value |
|---|---:|
| final_step | 2560 |
| games_completed | 640 |
| ply_cap (draws) | 555 / 640 = **86.7 %** |
| six_in_a_row | 85 / 640 = 13.3 % |
| colony | 0 / 640 = 0 % |
| other_draw | 0 |
| grad_norm last30 mean / max | 1.09 / 1.43 |
| hard_abort fired | no |
| inference forward count | (8 workers × ~2.4 inf/move × ~50 moves × 640 games) |

The run reproduces the Phase B Stage 1 / Stage 2 plateau (92 % / 94.5 %
draws) without δ.c reverts and without engine constant changes — confirming
the plateau is not unique to W4C v5 colony-collapse but is the dominant
mode of the v7full + smoke + δ.c configuration.

Colony fraction is 0 / 640 — δ.c continues to suppress colony-rule wins
(the §147 W4C v5 abort signature).

---

## 2. Class 2 — Value-head drift probe

The 50-position fixture (25 decisive mid-game from v7full T2, 25 cap-bound
late-game from long colony games) was evaluated every 250 training steps.

| step | decisive_mean | decisive_std | draw_mean | draw_std |
|---:|---:|---:|---:|---:|
| 250  | **−0.726** | 0.175 | +0.192 | 0.634 |
| 500  | −0.723 | 0.156 | +0.104 | 0.570 |
| 750  | −0.678 | 0.292 | +0.205 | 0.611 |
| 1000 | **−0.630** | 0.331 | +0.230 | 0.584 |
| 1250 | −0.684 | 0.183 | +0.040 | 0.530 |
| 1500 | −0.697 | 0.357 | +0.230 | 0.558 |
| 1750 | −0.648 | 0.276 | +0.179 | 0.585 |
| 2000 | −0.713 | 0.256 | +0.233 | 0.629 |
| 2250 | −0.698 | 0.343 | +0.191 | 0.561 |
| 2500 | −0.707 | 0.272 | +0.104 | 0.635 |

**Aggregate (n=10):**

* decisive_mean: **mean = −0.690, std = 0.031, range = [−0.726, −0.630]**
* draw_mean:     mean = +0.171, std = 0.066, range = [+0.040, +0.233]
* v7full pre-run baseline (this fixture, no training): **decisive = −0.094,
  draw = +0.146**
* **decisive shift from v7full: −0.596** in 250 training steps; the value
  head punches **THROUGH** `draw_value=−0.5` to a "side-to-move loses"
  prior. After step 250 the value oscillates around −0.69 with no
  monotonic recovery and no further deterioration.
* draw subset stays ≈ at v7full baseline — the cap-bound positions don't
  drift further into the loss zone.

**Verdict: Class 2 ACTIVE — fixed-point overshoot.** The value head
collapses faster than the original Class 2 hypothesis predicted (250 steps,
−0.6 shift) AND **deeper than the configured draw_value**, indicating
policy/value coupling overgeneralization. The crucial detail: the head
*does not recover* after the initial collapse — it locks. This rules out
"Class 2 is a transient training-loop wobble that self-corrects."

---

## 3. Class 3 — Buffer composition

`draw_target_fraction` = fraction of self-play buffer rows whose value
target lies in [−0.6, −0.4) (i.e. tagged as draw under `draw_value=−0.5`).

| step | corpus_frac | **draw_target_frac** | six_term | colony_term | cap_term |
|---:|---:|---:|---:|---:|---:|
| 500  | 0.000 | **0.979** | 0.152 | 0.000 | 0.848 |
| 1000 | 0.000 | **0.979** | 0.148 | 0.000 | 0.852 |
| 1500 | 0.000 | **0.979** | 0.154 | 0.000 | 0.846 |
| 2000 | 0.000 | **0.979** | 0.146 | 0.000 | 0.854 |
| 2500 | 0.000 | **0.980** | 0.134 | 0.000 | 0.866 |

* `corpus_fraction = 0` is a **measurement caveat** — `pool.buffer_composition()`
  reads only `self.replay_buffer` (self-play buffer); the pretrained
  corpus is held in a separate `pretrained_buffer` instance. The training
  step uses `assemble_mixed_batch` to draw both; the in-buffer draw fraction
  shown here is for the self-play tributary alone.
* `draw_target_fraction` saturates at **0.979 by step 500** (the first
  measurement) and stays there — buffer composition is **locked** within
  the first ~125 games. The buffer is essentially pure draw-coded data
  feeding the value head.
* `cap_terminal_fraction` slowly creeps **0.848 → 0.866** over 2000 steps
  — slightly worsening, never recovering.
* `colony_terminal_fraction = 0` throughout — δ.c still works on its own
  kill criterion.

**Verdict: Class 3 STRONGLY ACTIVE.** The buffer is a self-reinforcing
draw-coded loop from step ≤500. This is the **fuel source** for Class 2:
~98 % draw-coded value targets train the value head into the −0.5
neighbourhood; policy/value coupling overshoots to −0.69. Crossing of
draw_target_fraction > 0.50 happens *before* our first measurement at
step 500, so the brief's "when does it cross 0.50?" answer is **earlier
than step 500** — well before the standard log_interval=10 window of
Class 2 measurement.

---

## 4. Class 4 — q-axis stride-5 policy spam (user-flagged)

**Pattern:** mixed-color stones along a single hex row (`r` = constant)
at stride-5 spacing; e.g. `x_____o_____o_____x_____` along an east-west
row. Distance 5 maps to `LEGAL_MOVE_RADIUS = 5` (§146) and
`CLUSTER_THRESHOLD = 5` (§151 δ.c) — both inclusive at exactly 5.

### 4.1 Quantification on this run (n=640)

| terminal_reason | n | row_max med / P90 / max | stride5_run med / P90 / max |
|---|---:|---:|---:|
| **ply_cap**       | 555 | **42 / 57 / 76** | **30 / 43 / 61** |
| six_in_a_row      |  85 |  9 / 33 / 67 |  5 / 25 / 50 |

* `row_max` = number of stones on the densest east-west row (out of total
  stones in the game).
* `stride5_run` = longest chain of consecutive stones at stride 5 along a
  single east-west row.
* For a typical cap-bound game (150 plies = 75 stones / player = 150
  total): **42 of 150 stones (28 %) sit on a single east-west row, and
  30 of those form a stride-5 chain.** Worst-case 61-stone chain.

### 4.2 Statistical correlation with cap-draw outcome

* **Spearman ρ(stride5_run, is_ply_cap) = +0.5013, p = 5.0e-42, n = 640**

This is the dispositive Class-4 number. Long stride-5 chains and
cap-bound game outcome are tightly correlated; the relationship is *not*
explained by terminal six-in-a-row tactics (which would push stride5_run
high in the SIX bucket — it doesn't, median 5).

### 4.3 Comparison vs v7full T2 baseline (no training)

| metric (cap-bound only) | v7full T2 (n=6) | Smoke (n=555) | ratio |
|---|---:|---:|---:|
| row_max median | 14 | **42** | **3.0×** |
| row_max max | 19 | **76** | 4.0× |
| stride5_run median | 3 | **30** | **10×** |
| stride5_run max | 4 | **61** | 15× |

The pattern *type* is present in v7full's natural play; the smoke
*amplifies* it 10× on the central stride5_run statistic. Amplifying
factors (smoke vs T2): training temperature schedule (1.0 → temp_min vs
fixed 0.5), Dirichlet noise (ε=0.10, α=0.05 vs none), playout_cap
mixing (full_search_prob=0.5, sims_full=600 vs fixed sims=96), 1-ply
random opening (vs 4-ply).

### 4.4 Trajectory across training

stride5_run distribution is **flat** across the 2560-step window — there
is no growth, no decay. The pattern is set within the first ~50 games
and persists. This rules out "Class 4 is a training-induced shift that
eventually self-corrects" — it does not self-correct.

### 4.5 Why pre-existing macro signals miss this

* `colony_extension_fraction` (§107, gate at hex_dist > 6) — under-reports
  by design; stride-5 stones are at exactly the boundary.
* `axis_distribution.axis_q` (adjacency-1 hex pair fraction) — also misses;
  measured live at q=0.326, r=0.338, s=0.336 for cap games (≈ uniform).
  The pattern is at distance 5, not adjacent. The 0.45 / 0.50 axis_q
  thresholds in `monitors.yaml` have **never fired** because the macro
  signal isn't sensitive to stride-5 distributions.

**Verdict: Class 4 STRUCTURAL and DOMINANT.** Pre-existing in v7full,
amplified 3–10× by smoke training conditions. Strong correlation with
cap-draw outcome. Existing macro detectors (colony-extension,
axis-distribution) are blind to it by construction. This is the **base
policy preference** that the other three classes pile on top of.

---

## 5. Class 1 — Stale-model dispatch — INCONCLUSIVE

| step | current_version | median_range | P90_range | max_range | ρ(range, draw) |
|---:|---:|---:|---:|---:|---:|
| 500  | 0 | 0 | 0 | 0 | — |
| 1000 | 0 | 0 | 0 | 0 | — |
| 1500 | 0 | 0 | 0 | 0 | — |
| 2000 | 0 | 0 | 0 | 0 | — |
| 2500 | 0 | 0 | 0 | 0 | — |

The `model_version` atomic increments only on `InferenceServer.load_state_dict_safe()`
calls, which fire only on successful eval-gate promotions. **The variant
sets `eval_interval=5000`, so no eval ran during the 2560-step window —
no weight swap occurred — `current_version` stayed at 0 throughout — no
correlation can be computed.**

This is a **methodology gap** I introduced when I added `eval_interval=5000`
to the variant to avoid the eval cost spike during a 5k-step diagnostic
run. The decision optimised for run completion within budget but disabled
the Class-1 measurement.

**Verdict on Class 1: NOT TESTED.** Cannot attribute or rule out. Flagged
as a follow-up.

---

## 6. Per-worker draw rate

| step | hot (≥ 0.80) | range |
|---:|---:|---|
| 500  | 7/8 | 0.72–0.93 |
| 1000 | 6/8 | 0.79–0.93 |
| 1500 | 7/8 | 0.80–0.93 |
| 2000 | **8/8** | 0.80–0.90 |
| 2500 | **8/8** | 0.80–0.92 |

The plateau is a **population-level effect** — all 8 workers converge to
≥ 80 % draw rate by step 2000. No outlier worker. Combined with the
stride5_run distribution (median 30, near constant variance), this rules
out single-worker pathologies (e.g. a stuck MCTS tree or per-thread RNG
fault). The Class-4 amplification is uniform across workers.

---

## 7. Verdict ranking

| Class | Hypothesis | Verdict | Strength |
|---|---|---|---|
| **4** | q-axis stride-5 policy spam (radius-5 fixed-point) | **DOMINANT (base)** | ρ = +0.50, p = 5e-42; 10× amplification vs T2 |
| **3** | Buffer composition feedback (98 % draw-coded rows) | **STRONGLY ACTIVE (loop)** | drawT = 0.979 from step ≤ 500 |
| **2** | Value-head drift toward draw_value | **ACTIVE (downstream)** | dec = −0.69 ± 0.03; shift −0.60 in 250 steps; no recovery |
| **1** | Stale multi-worker dispatch | **NOT TESTED** | eval_interval=5000 disabled the measurement |

**Causal story (best fit to data):**

1. v7full's policy mildly prefers stride-5 east-west extensions
   (Class 4 base preference; visible at low amplitude in T2 — n=6 cap
   games at row_max=14 / stride5_run=3).
2. Smoke training conditions (γ knobs, Dirichlet noise, playout_cap
   sims=600 full-search, training temperature schedule) amplify this
   3–10× into the dominant policy mode.
3. The amplified Class-4 policy yields cap-bound games at 85 % rate.
4. The 85 % cap rate produces a buffer that's 98 % draw-coded
   (Class 3 fuel).
5. The value head trains on this composition and overshoots the
   −0.5 draw target by ~−0.2 onto a "side-to-move loses" prior
   (Class 2 downstream).
6. The draw-prone value head reinforces the cap-prone policy
   in subsequent searches — feedback loop.

---

## 8. Phase B' priority order — UPDATE (v7 → v8)

The original Phase B' priority order (Tracks A + C synthesis,
`/tmp/phase_b_prime_targets.md` v7) prescribed:

> 1. draw_value −0.5 → −1.0
> 2. max_game_moves 150 → 300
> 3. initial_pretrained_weight 0.8 → 0.4
> 4. (defer Gumbel re-enable)

**None of those four address Class 4** — they are all training-loop
knobs. The instrumented smoke shows Class 4 is the **base policy
preference**, not training-induced. Reordered v8:

### v8 priority order

1. **POLICY-SIDE / WINDOW-SIDE FIX FOR CLASS 4** (highest leverage,
   addresses root cause — must lead).
   *Candidates*:
   * **Asymmetric or jittered legal-move radius** — make `LEGAL_MOVE_RADIUS`
     per-turn-randomised in {4, 5, 6} so the radius-5 fixed point breaks.
   * **Stride-5 anti-spam regulariser** — penalize policy mass on cells
     that would create a stride-5 chain on an existing row (cheap legality
     post-filter; ~O(legal_moves) per move).
   * **`CLUSTER_THRESHOLD` re-test** at 6 or 7 — unwind the §151 δ.c at-
     boundary symmetry. Note: this re-opens the §147 v5 colony-collapse
     question; would require a guarded smoke.
   * **Sample-time policy entropy floor** — minimum dirichlet mixing
     weight on legal moves NOT continuing a stride-5 chain.

2. **CLASS 3 BUFFER SURGERY** (mitigates fuel for Class 2).
   *Candidates*:
   * Cap `draw_target_fraction` in the replay buffer at e.g. 0.5 via
     subsampling on push.
   * `draw_value −0.5 → −1.0` (sharpens signal but does not break the
     fixed point — secondary lever, not first).

3. **`initial_pretrained_weight 0.8 → 0.4`** — shorter corpus-mixing phase
   exits the corpus-distribution overlap sooner. Targets Track C's flip
   inflection mechanism; secondary.

4. **`max_game_moves 150 → 300`** — symptom-only fix; Track A's A5_c
   already showed v7full plays 0/200 caps at 300, but the smoke draws
   are not generated by tactical truncation, they are generated by
   stride-5 stalemates. Cap relax may *increase* the effective game length
   without changing the policy mode.

5. **Defer Gumbel re-enable** until items 1-3 ship.

### Explicitly DO NOT consider in Phase B'

* γ-knob retuning (Track A falsified).
* δ.c reverts on `CLUSTER_THRESHOLD` (5 → 8 etc) **without** a guarded
  smoke — Track C falsifies cluster-threshold as the *cap-binding*
  mechanism but item 1.3 above re-tests it as a Class-4 mechanism.
* sims-budget bumps (Track A A4_s mildly worsens).

---

## 9. Methodology caveats and follow-up runs

### 9.1 Class 1 follow-up (highest priority)

The current run cannot rule Class 1 in or out because `eval_interval=5000`
disabled the weight-swap mechanism. **Required follow-up**: same variant
config but `eval_interval=500` and `iterations=2500`, runs ~3 h. The
Class-1 instrumentation will then produce non-trivial `model_version_range`
distributions and a non-NaN Spearman ρ. Without this run we cannot
prescribe whether the v8 fixes need a fourth ingredient (architectural
model-pinning).

### 9.2 Class 4 metric upgrade

The existing `colony_extension_fraction` (§107) gates at hex_dist > 6
and **misses Class 4 by design**. The `axis_distribution` event
measures distance-1 adjacency and **misses Class 4 by 5×**. Recommended
new dashboard metric:

* `stride5_run_max_per_game` (rolling last 50 games, median + P90)
* `row_max_density_per_game` (same)

These would be live alarms — fire above row_max P90 > 30 to catch
Class 4 onset within 50 games rather than waiting for cap-rate climb.
Implement before the v8 fixes ship so we can A/B them.

### 9.3 Real cap-draw probe positions

The value-probe fixture currently uses 25 long-colony positions as
proxies for cap-bound positions (because v7full produces only 6
ply_cap games per 200). The **instrumented smoke produced 555 real
cap-bound games** — regenerate the fixture with
`--cap-source smoke_jsonl` against `events.jsonl` so future probe
runs measure on actual draw-equilibrium states.

### 9.4 Class 3 measurement gap

`pool.buffer_composition()` reads only the self-play buffer; the
pretrained-buffer mixing weight is not currently exposed. The brief
asked "when does corpus_fraction cross self-play?" — that question is
about the *effective batch composition*, computed by
`compute_pretrained_weight(step)` in the training loop. The instrumented
smoke logs are sufficient to answer this offline; future revisions of
the metric should expose the live mixed-batch weight directly.

---

## 10. Decision

* **Phase B' is unblocked** — proceed with v8 priority order.
* **First v8 lever to ship**: Class-4 fix (item 1). Without it, Class 3
  and Class 2 mitigations alone will treat symptoms, not the cause.
* **Recommended sequencing**:
  1. Implement live stride5_run / row_max metrics (§9.2) — 1-2 h dev.
  2. Implement the Class-4 fix candidates as branched variants
     (asymmetric radius, anti-spam regulariser, threshold re-test) — each
     gets a 2 k-step instrumented smoke.
  3. Whichever combination shows draw_target_fraction < 0.5 by step 1500
     graduates to a sustained run.
  4. Re-run with `eval_interval=500` to close the Class-1 gap.
* **Constraints honored**: 5080 Stage 4 smoke not touched (separate
  instance); v7full untouched; engine constants untouched; no production
  config defaults modified; sprint log untouched.

---

## 11. Artifacts

* `reports/phase_b_prime/instrumented/events.jsonl` — full event stream,
  10 value-probe / 5 BC / 5 MV / 5 WDR readings, 640 game records.
* `reports/phase_b_prime/instrumented/run.log` — `tee` of training
  stdout/stderr.
* `reports/phase_b_prime/instrumented/checkpoints/checkpoint_00002560.pt`
  — final state of trainer.model at abort.
* `reports/phase_b_prime/instrumented/checkpoint_log.json` — trainer
  checkpoint history (500 / 1000 / 1500 / 2000 / 2500 / 2560 ckpts retained
  on remote under `checkpoints/w4c_smoke_v6_instrumented/`).
* `scripts/phase_b_prime_diagnose.py` — analysis script that produced
  every table in §§ 2-6.
* `scripts/phase_b_prime_monitor.py` — live monitor used at 15-min cadence.
* `configs/variants/w4c_smoke_v6_instrumented_5080.yaml` — variant config.
* `fixtures/value_probe_50.npz` — 50-position fixture used by Class 2
  probe.
* `hexo_rl/monitoring/value_probe.py` — probe runner.
* `hexo_rl/selfplay/pool.py` — buffer-composition / model-version /
  worker-draw-rate accessors (`buffer_composition()`,
  `model_version_summary()`, `per_worker_draw_rates()`).
* `engine/src/inference_bridge.rs`, `engine/src/game_runner/mod.rs`,
  `engine/src/replay_buffer/{mod,storage}.rs` — Rust instrumentation
  (model_version atomic, terminal_reason u8, outcome_in_range_count).

## 12. Sign-off

The instrumented smoke discriminated all four hypothesis classes within
2560 steps. The dominant cause of the Phase B draw plateau is **policy-
side, not training-loop-side** (Class 4 base × smoke amplification),
with Class 3 (buffer feedback) and Class 2 (value-head overshoot) as
downstream effects. Phase B' priority order is reordered to lead with
policy / window fixes; previous Track A / Track C synthesis training-loop
knobs become secondary.

Class 1 (stale-dispatch) remains untested — a 2.5 k-step follow-up with
`eval_interval=500` is the minimum to close that gap before any
sustained run.
