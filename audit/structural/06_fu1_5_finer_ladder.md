# §S181 FU-1.5 — finer-ladder value-spread probe

**Wave:** §S181 follow-up FU-1.5. **Date:** 2026-05-23. **Type:** inspection
probe on a re-run, ~0 GPU at probe time. **Branch:**
`phase4.5/s181_fu1_5_finer_ladder`.

**Goal.** Pin WHEN the value-head colony/extension discriminator flattens
across the first 20k steps of the §S180b recipe — at 2k checkpoint
cadence — to discriminate STEP-0-ONSET vs MID-CLIFF vs GRADUAL for the
FU-2 architecture-vs-aux-loss routing (see [[05-fu1-value-spread-ladder]]
§4 limitation: the FU-1 archive holds no sub-10k checkpoint).

**Probe:** `scripts/structural_diagnosis/fu1_value_spread_ladder.py
--ckpt-dir archive/s181_fu1_5/ckpts --out 06_fu1_5_finer_ladder.json`.
**Sidecar:** `audit/structural/06_fu1_5_finer_ladder.json`.

---

## 0. Investigation context (vast host-state regression)

The first attempt to launch FU-1.5 on vast hit an anomalous regime —
all 05-22 vast runs produced `mcts_mean_depth` ~2.5 with games ~77-96
plies, vs every pre-05-22 run at depth 3.4-3.8 / plies 27-57. A
five-test exoneration of the §S182/§S183 perf commits (deterministic
depth probe, deterministic full-game probe, laptop self-play 100-game
A/B, laptop full-training 100-game A/B, and a vast 100-game A/B
post-reinstall — every test bit-identical old vs new code) ruled out
the perf wave. Root cause: stale vast-host state. A clean reinstall
(`rm -rf hexo_rl` → fresh `git clone` → `make install` — fresh `.venv`
with torch 2.11.0+cu128 + fresh engine build) restored the normal
regime. The relaunched FU-1.5 run on the clean host reproduces §S180b
within noise (wr_sealbot 9% vs §S180b 11% at step 10k; elo 421 vs 422;
wr_anchor 66% vs 61%) — confirming this ladder IS comparable to FU-1's
§S180b anchor ladder.

The MCTS child-order non-determinism caught during the investigation
(latent `FxHashSet` iteration-order leak through `pick_topk_children`)
landed independently as `fix(mcts): canonical child order — kill
FxHashSet-order leak` (commit `de149e6` on master); it is bench-gated
(MCTS sim/s −3.0%, under the 5% gate) with a regression test pinning
the invariant. Behaviourally inert (proven five ways) so the FU-1.5
trajectory is unaffected.

---

## 1. Bank fixture

**Source.** T3 40-position canonical bank — reused verbatim from
[[05-fu1-value-spread-ladder]] §1 (20 colony from `build_colony_positions`
+ 20 extension from `build_extension_positions`, both in
`mcts_colony_probe.py`). No bank regeneration.

**SHA-256 (hash over name + class + applied move sequence):**
```
934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991
```

The probe now hard-gates against this value (§S181 finer-ladder
prep — `fu1_value_spread_ladder.py:BANK_SHA256`) — any drift halts
the probe. **Gate: PASS.**

**Anchor reproducibility gate — PASS.** Forwarding
`checkpoints/bootstrap_model_v6.pt` (SHA `7ab77d2c…372103`) on the
40-position bank reproduces the T3 `value_head` numbers to 4 dp:

| metric | T3 JSON | FU-1.5 | Δ |
|---|---|---|---|
| mean V(colony) | 0.1635 | +0.1635 | 0.0000 |
| mean V(extension) | −0.4539 | −0.4539 | 0.0000 |
| V_spread | +0.6174 | +0.6173 | 0.0001 |

Bank + load path are sound; ladder trustworthy.

---

## 2. Re-run config + comparability to §S180b

**Config:** `configs/variants/v6_botmix_s180b_2kckpt.yaml` — verbatim
copy of `v6_botmix_s180b_3knob_escalation.yaml` (the config that
produced §S180b) with one delta: `anchor_every_steps: 5000 → 2000`
(checkpoint preservation only; no gameplay effect; body `diff` ⇒ that
single appended block).

**Host:** vast.ai (5080 + Ryzen 9 9900X), freshly reinstalled
(`git clone https://github.com/seeligto/hexo_rl.git` at master
`de149e6`, fresh `.venv` with torch 2.11.0+cu128, fresh engine build
via `maturin develop --release`, `make install` smoke tests 1555
passed). Anchor SHA-verified `7ab77d2c…`. Corpus SHA-verified
`6ea62afa…`. Clean buffer at start (`buffer_size_before_corpus_load: 0`).

**§S180b cross-check @ step 10k (brief Task-1 variance gate ±15pp):**

| metric | FU-1.5 | §S180b | Δ |
|---|---|---|---|
| wr_sealbot | 0.09 | 0.11 | −2 pp ✓ |
| wr_bootstrap_anchor | 0.66 | 0.61 | +5 pp |
| wr_best | 0.52 | 0.52 | 0 |
| elo_estimate | 421.3 | 422 | −0.7 |
| colony_wins_anchor | 41 | 36 | +5 |
| colony_wins_sealbot | 5 | 7 | −2 |
| sealbot_gate_passed | False | False | match |

All metrics within noise — FU-1.5 IS a faithful §S180b reproduction.

---

## 3. Per-checkpoint V_spread table

`V_spread = mean V(colony bank) − mean V(extension bank)`. Anchor =
step 0. n=20 per class. Per-interval Δ = next − current.

| checkpoint | step | mean V(colony) | mean V(ext) | **V_spread** | Δ from prev |
|---|---:|---:|---:|---:|---:|
| `bootstrap_model_v6.pt` | 0 | +0.1635 | −0.4539 | **+0.6173** | — |
| `checkpoint_00002000` | 2 000 | +0.2245 | +0.0493 | **+0.1752** | **−0.4421** |
| `checkpoint_00004000` | 4 000 | +0.2722 | +0.3902 | **−0.1179** | −0.2931 |
| `checkpoint_00006000` | 6 000 | +0.2802 | +0.2250 | **+0.0553** | +0.1732 |
| `checkpoint_00008000` | 8 000 | +0.2997 | +0.0825 | **+0.2172** | +0.1619 |
| `checkpoint_00010000` | 10 000 | +0.3299 | −0.0601 | **+0.3901** | +0.1729 |
| `checkpoint_00012000` | 12 000 | +0.3295 | +0.1626 | **+0.1669** | −0.2232 |
| `checkpoint_00014000` | 14 000 | +0.3763 | −0.1463 | **+0.5226** | +0.3557 |
| `checkpoint_00016000` | 16 000 | +0.2378 | +0.0170 | **+0.2209** | −0.3017 |
| `checkpoint_00018000` | 18 000 | +0.2583 | +0.0500 | **+0.2083** | −0.0126 |
| `checkpoint_00020000` | 20 000 | +0.1859 | +0.0784 | **+0.1075** | −0.1008 |

**Comparison to FU-1 §S180b ladder (10k cadence):**

| step | §S180b V_spread | FU-1.5 V_spread |
|---|---:|---:|
| 0 | +0.617 | +0.617 |
| 10 000 | +0.260 | **+0.390** |
| 20 000 | **−0.110** | **+0.108** |

FU-1.5's 10k V_spread is +0.13 higher than §S180b's; at 20k FU-1.5
sits at +0.108 (still positive) where §S180b had crashed to −0.110.
Both 20k values are within ~1 × SE(spread)≈0.185 of zero — i.e.
statistically indistinguishable from a flat-dead band — but the
trajectories *between* the two ladders diverge in their post-onset
behaviour (see §5).

**Noise floor.** Post-onset sd(col) ≈ 0.66, sd(ext) ≈ 0.50; SE(spread)
≈ √(0.66²/20 + 0.50²/20) ≈ **0.186**. Any single ladder point within
~±0.2 of zero is consistent with flat-dead.

---

## 4. Plot — V_spread vs training step

```
 V_spread
            -0.15   0.0(═)         +0.20(┊abort)               +0.617
              |      |               |                            |
 step 0       ·······╪···············┊····························● +0.617
 step 2k      ·······╪·············●·┊                            +0.175
 step 4k    ●·········╪···············┊                           -0.118
 step 6k      ·······╪·····●·········┊                            +0.055
 step 8k      ·······╪···············┊·●                          +0.217
 step 10k     ·······╪···············┊·······●                    +0.390
 step 12k     ·······╪···············●┊                           +0.167
 step 14k     ·······╪···············┊·············●              +0.523
 step 16k     ·······╪···············┊·●                          +0.221
 step 18k     ·······╪···············┊·●                          +0.208
 step 20k     ·······╪·············●·┊                            +0.108
              |      |               |
            -0.15   0.0           +0.20

 ═ zero line   ┊ +0.20 FU-2 abort gate (FU-1 / canary threshold)
 anchor +0.617 ─► fast onset (0→2k: −0.442) ─► non-monotone oscillation
```

---

## 5. Drift signature — mechanical + substantive

**Mechanical drift classifier** (`classify_drift`, FU-1 algorithm):
**OSCILLATION** — non-monotone with `max_single_interval_gain = +0.356`
(12k→14k) > 0.10 threshold. Total spread loss anchor→step-20k =
+0.5098 (82.6 % of initial spread). Max single-interval loss =
−0.4421 at 0→2k (86.7 % of total loss).

**Substantive read — STEP-0-ONSET + post-onset noise/oscillation.**
The fine ladder resolves the front-loaded collapse that FU-1 could
not localize:

1. **Step-0 onset is real.** Anchor (+0.617) → step-2k (+0.175) =
   **−0.442 single-interval loss**, or **86.7 % of the total
   trajectory loss in the first 2k steps**. Step-4k goes negative
   (−0.118). The discriminator crosses the +0.20 FU-2 abort gate
   *between step 0 and step 2k*.
2. **Post-onset oscillation, not permanent flat-dead.** Unlike §S180b
   (FU-1 §3: ~0 mean across 20-53.5k), FU-1.5 swings between
   −0.118 and +0.523 across 4-14k, with peak +0.523 at step 14k.
   Multiple data points (10k +0.390, 14k +0.523) are >2 × SE above
   zero — *partial recovery, not collapse*.
3. **By step 20k both runs land near zero.** §S180b: −0.110. FU-1.5:
   +0.108. Both within ±1 SE of zero. The endpoints agree, the paths
   differ.

**Two routes from a clean bootstrap:** §S180b crashed monotonically;
FU-1.5 crashed fast, partially recovered, and oscillated. The *speed*
of the early onset matches; the *late behaviour* diverges. This is
consistent with the system being chaotic past the initial crash
(small RNG/training-RNG differences between runs amplify post-onset
in the unstable colony-attractor regime).

---

## 6. Pre-registered V-FL verdict (mechanical — L13 guard)

The `classify_fl_verdict` predicate runs in code so the verdict
cannot be rewritten to fit the outcome.

| ID | rule | this ladder | match |
|---|---|---|---|
| V-FL-A STEP-0-ONSET | V_spread(2k) ≤ +0.40 AND V_spread(4k) ≤ +0.25 | +0.175 ≤ +0.40 ✓ AND −0.118 ≤ +0.25 ✓ | **TRUE** |
| V-FL-B MID-CLIFF | V_spread(2k) ≥ +0.50 AND any [4k,10k] interval drops ≥ 0.30 | +0.175 < +0.50 ✗ | FALSE |
| V-FL-C GRADUAL | monotone non-increasing AND no 2k interval loses > 0.15 | non-monotone (+0.356 swing 12k→14k) ✗ | FALSE |
| V-FL-D RECOVERY | any >20k interval recovers ≥ +0.15 from sub-+0.20 dip | n/a — 20k-capped run | FALSE |

**VERDICT: `V-FL-A` — STEP-0-ONSET.**

**FU-2 routing per the pre-registered table:** **A2 (multi-scale
avg-pool) architecture arm load-bearing.** Rationale: a value-head
collapse that is >86 % complete by step 2k cannot be caught by a
post-hoc aux-loss gradient (A3) — there is no time for an aux-target
signal to push back before the discriminator is already through the
abort gate. An architectural change that prevents the early-step
collapse (e.g. multi-scale avg-pool removing the
coverage-blind v_max) is load-bearing.

---

## 7. Verdict block

```
FU-1.5 FINER-LADDER VALUE-SPREAD PROBE — VERDICT

  Bank:            T3 40-position canonical bank (20 colony + 20 ext)
  Bank SHA-256:    934204713620d171743820aea6907cf4e117ca97c69e50052b991a3fdcc23991
  Bank SHA gate:   PASS  (hard-asserted in fu1_value_spread_ladder)
  Anchor gate:     PASS  (+0.6173 reproduces T3 +0.617 to 4 dp)

  §S180b @10k cross-check (brief ±15pp gate):
    wr_sealbot 9% vs §S180b 11%  (-2pp) PASS
    elo 421 vs 422  (-0.7)               PASS

  V_spread ladder:
    step:    0   2k    4k   6k    8k   10k   12k   14k   16k   18k   20k
    spread: +0.62 +0.18 -0.12 +0.06 +0.22 +0.39 +0.17 +0.52 +0.22 +0.21 +0.11

  CLASSIFIER VERDICT:  OSCILLATION (mechanical drift signature)
  PRE-REGISTERED V-FL: V-FL-A — STEP-0-ONSET

  Conditions met:
    V-FL-A_step0_onset: TRUE   (V_spread(2k)=+0.175 ≤ +0.40
                                AND V_spread(4k)=-0.118 ≤ +0.25)
    V-FL-B_mid_cliff:   FALSE  (V_spread(2k) < +0.50)
    V-FL-C_gradual:     FALSE  (non-monotone)
    V-FL-D_recovery:    FALSE  (20k-capped run)

  KEY FINDING — RESOLVES FU-1 LIMITATION:
    The collapse is FRONT-LOADED to the first 2k steps.
    Single-interval loss 0→2k = -0.4421 (86.7% of total trajectory loss).
    V_spread crosses the +0.20 FU-2 abort gate between step 0 and 2k.
    Post-onset: oscillation (peak +0.523 at 14k), NOT permanent flat-dead.

  FU-2 ROUTING: A2 architecture arm load-bearing.
    An aux-loss A3 cannot catch a value-head crash that is >86% complete
    by step 2k. Architectural change (e.g. multi-scale avg-pool to remove
    the coverage-blind v_max pool) is the load-bearing intervention.
```

---

## 8. Recommendation for FU-2 (operator decides; not auto-launched)

**Primary arm:** **A2 — multi-scale avg-pool value head**. V-FL-A
verdict literally; the step-0 onset (>86 % of collapse in the first
2k steps) implicates the architecture, not the training loop.

**Secondary consideration.** The post-onset oscillation (peaks
+0.523 at 14k) is a substantive deviation from §S180b's permanent
flat-dead, and suggests the value head retains *some* discriminative
capacity past the initial crash that an A2 architecture might amplify
if the early collapse is prevented. Worth re-measuring V_spread on
the FU-2 A2 candidate over the same 2k ladder.

**Lessons (new — L44+).**

L44. **Value-head crashes can be front-loaded.** A 2k-cadence ladder
on §S180b's recipe shows >86 % of the value-spread loss occurs in the
first 2k steps from the bootstrap. The 10k-cadence ladder of FU-1
under-sampled this region. Future structural-diagnosis waves that
probe value-head onset should default to ≤2k cadence in the first
10k steps.

L45. **Host-state hygiene is a research-load-bearing invariant.** The
05-21→05-22 vast regime change (depth 3.4→2.5 stable; uniform across
all 05-22 vast launches) was *not* code — it was accumulated host
cruft (stale `.venv`, stale compiled engine, leftover state from
prior runs). Five independent code-A/B tests exonerated the perf
wave (§S182/§S183). A clean `git clone` + `make install` restored
the §S180b regime. **Implication:** when a "first variance since the
N-knob baseline" appears, host-state reinstall belongs in the
discriminator set alongside the code bisect — not as a follow-up.

L46. **Post-onset oscillation ≠ flat-dead.** §S180b reached a
permanent flat-dead V_spread band after step 20k; FU-1.5 (same
recipe, clean host) reaches a similar mean-near-zero by step 20k
but the *path* oscillates (+0.523 peak at step 14k). Two faithful
reproductions of the same recipe can take different routes through
the chaotic colony-attractor neighbourhood. The 20k endpoint agrees
within ~1 × SE; the trajectory does not.
