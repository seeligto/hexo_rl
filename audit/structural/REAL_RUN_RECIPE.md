# REAL_RUN_RECIPE — §S181-AUDIT Wave 1 synthesis

§S181-AUDIT Wave 1 close-out. Synthesizes Track A (L47/L48/L49),
PR-B (param-group split + eta_min raise + entropy reg), PR-C (dual-bank
canary), Track C-LITE-1 (v7full anchor reference), Track D (pipeline
regression audit). Track B verdict pending operator run — recipe is
**parameterized on V-B-{A,B,C,D,E}**; the operator picks the matching
branch post-Track-B aggregation.

L13 guard: pre-registered verdict tables apply LITERALLY. Falsified
hypotheses (cosine restoration, beta2=0.95, A2-class arch-only, WDL
without value-target evidence) are NOT proposed below.

---

## §1 Anchor selection

**Recommended primary: `checkpoints/bootstrap_model_v7full.pt`**
(SHA `568d8a33…d61e8e98`, encoding `v7full`, single-window 19×19,
n_planes=8, has_pass_slot=true).

Evidence:

| anchor | V_spread T3 | V_spread alt | SealBot WR (known) | source |
|---|---|---|---|---|
| `bootstrap_model_v6.pt` | +0.6173 | +0.2119 | 15-17% n=500 (§148) | FU-1 / A3 |
| `bootstrap_model_v7full.pt` | +0.2171 | **+0.4078** | 17.4% n=500 (§150) | C-LITE-1 |
| `bootstrap_model_v6_step20k.pt` | n/m | n/m | varies (§176 STRENGTHEN_ONLY) | §176 |

v7full's alt-bank V_spread is **~2× v6's** and held the
single-strongest pre-collapse SealBot WR in the project (§150 17.4%).
The T3 underperformance is L48-explained (T3 was calibrated on v6's
value head; alt is the corpus-grounded reference). C-LITE-1 verdict
C-LITE-1-A — encoding regression candidate CONFIRMED.

Fallback: `bootstrap_model_v6.pt` (SHA `7ab77d2c…`) — proven
discriminator on both banks, slightly weaker on alt but a known
quantity. Use only if Track B verdict + Wave 2 prep block v7full
selection (e.g. if V-B-D + trunk centroid collapse signals require an
architecture change that v7full also exhibits).

**NEVER use:** `bootstrap_model.pt` (random-init, quarantined per
§S178a fix). PR-C dual canary + v6/v7full anchors all live in
`checkpoints/`; random-init has been moved to `archive_quarantine/`.

## §2 Encoding selection

**Recommended: `v7full`** (matches §1 anchor).

v7full + v6 are identical at the plane / spatial level
(`engine/src/encoding/registry.toml`): `board_size=19`, `n_planes=8`,
identical `plane_layout` (X_t0..t-3, O_t0..t-3), identical
`kept_plane_indices=[0,1,2,3,8,9,10,11]`, `has_pass_slot=true`,
`is_multi_window=false`. The difference is the upstream selfplay
corpus / training trajectory, not the encoding spec.

C-LITE-2 (v6w25 stock-encoding probe) — **DEFERRED**. v6w25 is K=8
K-cluster with min-pool aggregation; the structurally extension-
favouring inductive bias is interesting but C-LITE-1's v7full alt
V_spread already answers the encoding question at the single-window
level for the v6 family. Re-open C-LITE-2 only if v7full + V-B-A/B
lever stack fails its 30k-step gate in Wave 2.

**Do NOT** mix encodings within a single training run — the registry
validates encoding consistency at trainer init.

## Wave 2 close — Option D path

§S181-AUDIT Wave 2 executed §7 below (Wave 2 plan, archived below)
under operator-routing Option D (smoke-first per the near-miss V-B-A
synthesis at `audit/structural/track_b/B_verdict_synthesis.md`).

Result: project-record SealBot WR 33% peak @ step 20k → monotonic
collapse to 5% @ step 40k → HARD-ABORT @ step 47642. The Wave 2
canonical deliverable is `reports/track_b_main/checkpoints/checkpoint_00020000.pt`.
Full mechanism analysis: `audit/structural/wave2_real_run_analysis.md`.

L50/L51/L52 banked. Wave 2 falsified two prior assumptions: (a)
static bot corpus alone is sufficient anti-colony anchor past peak
fit point; (b) alt V_spread + dual-bank canary alone is sufficient
gate for real-run quality. Wave 3 below addresses both.

## §3 Wave 3 anti-colony lever stack

Wave 3 stack = Wave 2 stack (LANDED — no regression) + 3 new levers:
**refresh hook** (L51), **per-class temp scope revision** (L52),
**sliding-window SealBot WR hard-abort gate** (L50).

### LANDED on master (no Wave 3 change)

- PR-B param-group split for AdamW (`c2a0f31`)
- PR-B `eta_min: 5e-4` (`a43d5eb`)
- PR-B `entropy_reg_weight: 0.005` (`03425de`)
- PR-C dual-bank V_spread canary (`7cd0dc0`) — kept, now INFORMATIONAL
  only per L50 (was SOFT-ABORT in Wave 2; downgraded because alt
  passed while wr_sealbot crashed)
- EMA-of-weights infrastructure (`95624af`) — held Wave 2's 33% peak
  without infra failure
- v7full anchor + encoding (§1 + §2)

### Wave 3 NEW levers (LANDED on `phase4.5/s181_wave3_design`)

1. **Bot corpus refresh hook** (Stage 2A — addresses M1 / L51).
   `mixing.bot_corpus_refresh.enabled: true` with fixed-interval
   trigger (`interval_steps: 5000`), async subprocess regen via
   `scripts/generate_bot_corpus.py`, anchor against current EMA
   model snapshot, atomic NPZ swap + hot-reload, rolling-window
   replacement, hard cap `max_regens: 20` per 100k run. Implementation
   per `docs/designs/s179c_bot_refresh_hook.md` (Wave 3 deltas:
   fixed-interval trigger replacing the design's promotion-delta
   trigger; EMA opponent; 5k cooldown vs design's 25k).

2. **Sliding-window SealBot WR hard-abort gate** (Stage 2B —
   addresses L50). Rolling-mean tracking of last 3 eval rounds;
   hard-abort on (a) rolling-mean < 10% for 2 consecutive evals
   after step 20k OR (b) current < peak × 0.5 past step 25k OR
   (c) current < 5% past step 15k. Operator-flag override for
   debug runs only.

3. **Per-class target temperature scope revision** (Stage 2C —
   addresses M2 / L52). Add `apply_to_selfplay: false` (default
   true for Wave 2 backward-compat). Wave 3 variant sets
   `apply_to_pretrain: true, apply_to_selfplay: false` — softens
   colony CE on pretrain + bot rows only, leaves selfplay rows
   untouched so the model's own sharp policies are preserved.

### DEFERRED to Wave 4+ (only if Wave 3 fails its primary gate)

- KL-weighted buffer writes (Q9)
- 2-stone opp-reply aux head (V-B-D conditional, never tested)
- Class-weighted gradient scaling (alternative to per-class temp)
- Activation alternatives (SiLU/Mish)
- Muon optimizer
- WDL value-head migration (A2 falsified arch-only fix)
- Gumbel CQV / KrakenBot wrapper / Phase 4.5 feature queue
  (parked behind Wave 3 success)

## §4 Pre-registered success criteria — v2 (rolling-mean PRIMARY per L50)

LITERAL L13 guard. Real run aborts on first PRIMARY failure;
promotion gated on ALL PRIMARY passing.

### PRIMARY (gate)

| ID | criterion | window |
|---|---|---|
| W3-G1 | Rolling-mean SealBot WR ≥ 20% across 3 consecutive eval rounds | sustained over 30k → 50k |
| W3-G2 | No L34 anchor↑/sealbot↓ divergence alarm fires | every eval round |
| W3-G3 | No HARD-ABORT trigger fires (run reaches 100k) | run completion |

### SECONDARY (informational)

| ID | criterion | window |
|---|---|---|
| W3-G4 | Peak SealBot WR ≥ §150 baseline | any single eval; target ≥ 17.4% |
| W3-G5 | Refresh hook fired ≥ floor(100k / 5k) - 1 = 19 times | refresh event count |
| W3-G6 | alt V_spread ≥ +0.10 sustained | informational per L50; not gate |
| W3-G7 | T3 V_spread sign track | informational per L48; not gate |
| W3-G8 | colony_a < 50/100 in eval rounds | informational per Wave 2 (PASS clean in Wave 2 while wr_sealbot crashed) |

Promotion to canonical model: ALL PRIMARY PASS at step 50k AND no
soft-abort/hard-abort between 50k and 100k AND no L34 divergence in
any eval. Failed-PRIMARY ckpts may still be useful reference
(Wave 2 step-20k 33% peak is project record); but not promoted.

## §5 Stop conditions during the run — explicit (per Wave 3 §2B)

HARD-ABORT triggers (run terminates immediately):
- Rolling-mean SealBot WR < 10% for 2 consecutive evals AFTER step 20k.
- Current WR < peak × 0.5 AND past step 25k (catches Wave-2-style
  collapse: 33% → 16% at step 25k+).
- Current WR < 5% past step 15k (catches §S180b-style early death).
- L34 anchor↑/sealbot↓ divergence alarm fires (Wave 3 tightens
  Wave 2's "5 consecutive" to "3 consecutive" per Wave 2 REVIEW
  observation 2; calibration at implementation time).
- GPU NaN/Inf in gradients (`nan_or_inf_loss_skipped` count > 20 in
  any 100-step window).
- Standard §S180b hard-aborts: `grad_norm > 10` for 5 consecutive
  steps, `loss_nan`, `colony_ext_frac > 0.40`, `stride5_p90 > 60`.

SOFT-ABORT (operator decision):
- Dual canary `both_pass: false` for 2 consecutive checkpoints —
  WARNING, NOT gate (per L50). Operator may inspect ckpts but
  trajectory continues unless a HARD trigger fires.
- Wallclock > 24h on a single GPU — operator decision point.

`--ignore-hard-abort` debug flag exists but MUST NOT be used on real
runs.

## §6 Compute budget

| stage | cost | wall |
|---|---|---|
| Wave 3 dev (refresh hook + hard gate + per-class temp scope + REAL_RUN_RECIPE v2) | 1-2 days | 1-2 d |
| Pre-launch smoke (6000 steps to confirm refresh fires + lever stack behaves) | $1.50 | ~6 h |
| Main run (100k steps; ~$3 baseline + ~$0.50 refresh subprocess overhead = ~$3.50) | ~$3.50 | ~14 h |
| Post-run analysis + sprint-log + memory | 0 GPU | 0.5 d |
| **TOTAL** | **~$5 + 2-3 days dev** | **~3 days incl. launch** |

Hard cap unless re-approved: $8 (per dispatcher).

## §7 Wave 3 plan

Dependency order:

1. **Stage 2 design** — branch `phase4.5/s181_wave3_design` off
   post-Wave-2-close master:
   - 2A: Refresh hook activation (~500-800 LOC across step_coordinator,
     batch_assembly, loop, generate_bot_corpus, tests)
   - 2B: Sliding-window SealBot WR hard-abort gate (~50 LOC across
     evaluator, alert_rules, tests)
   - 2C: Per-class temp scope revision (~30 LOC + tests)
   - 2D: REAL_RUN_RECIPE v2 (this update)
   - 2E: REVIEW + operator gate
2. **Stage 3 pre-launch smoke** — `v7_wave3_smoke.yaml` inheriting
   `v7_real_run_smoke.yaml` + Wave 3 deltas; 6000 steps to confirm
   refresh fires at step 5000 + lever stack behaves cleanly. Smoke
   verdict gates per pre-registered WS-A..WS-E table (dispatcher Stage 3C).
3. **Stage 4 main run** — `v7_wave3_main.yaml` inheriting smoke; 100k
   steps from `bootstrap_model_v7full.pt` anchor under v7full encoding
   with full Wave 3 lever stack. Operator-mediated.
4. **Stage 5 analysis** — `audit/structural/wave3_real_run_analysis.md`;
   pre-registered W3-G1..W3-G8 verdict check; Phase 4.5 unblock decision.
5. **Stage 6 REVIEW + close** — REVIEW subagent on chain;
   FF-merge to master.

**Do not auto-launch Wave 3.** Operator-mediated gates at every stage
transition.

---

## Inputs cited

### Wave 1 inputs

- `audit/structural/track_a/A6_aggregation.md` — A1-A5 verdicts, L47/L48/L49
- `audit/structural/track_c/C_LITE_1_v7full_reference.md` — v7full anchor
- `audit/structural/track_d_pipeline_regression.md` — pipeline regression + smoking-gun rank
- `audit/structural/track_b/B_launch_and_analysis_spec.md` — Track B run procedure + V-B verdict logic
- `audit/structural/track_b/B_aggregation.md` — V-B LITERAL verdict (V-B-E)
- `audit/structural/track_b/B_verdict_synthesis.md` — Option D recommendation
- `audit/structural/track_b/B_track_d_xref.md` — Track D candidate confrontation
- `docs/07_PHASE4_SPRINT_LOG.md` §S181-AUDIT — L47/L48/L49 + Track A verdicts + PR-B hygiene
- `engine/src/encoding/registry.toml` — v6 / v7full encoding parity

### Wave 2 inputs

- `audit/structural/wave2_smoke.md` — Stage 4B smoke S-A PASS
- `audit/structural/wave2_real_run_analysis.md` — peak-and-collapse mechanism, L50/L51/L52
- `docs/07_PHASE4_SPRINT_LOG.md` §S181-AUDIT Wave 2 — L50/L51/L52 banked

### Wave 3 inputs

- `docs/designs/s179c_bot_refresh_hook.md` — refresh hook design
- `hexo_rl/training/per_class_target_temperature.py` — per-class temp + Wave 3 apply_to_selfplay flag
- `configs/variants/v7_wave3_smoke.yaml` — Stage 3 smoke variant (NEW)
- `configs/variants/v7_wave3_main.yaml` — Stage 4 main variant (NEW)
