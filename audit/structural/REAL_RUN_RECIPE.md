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

## §3 Anti-colony lever stack

Levers in priority order. Categories: LANDED (already on master /
about to be merged), CONDITIONAL (Wave 2 add iff Track B verdict
matches), DEFERRED (Wave 3+).

### LANDED

- PR-B param-group split for AdamW (`c2a0f31`). nanoGPT/KataGo
  standard; no-decay on 1D / bias params, decay on weights.
- PR-B `eta_min: 5e-4` (`a43d5eb`, raised from 2e-4) — KataGo
  precedent never below 0.5× peak; prevents late-run loss-of-plasticity.
- PR-B `entropy_reg_weight: 0.005` (`03425de`, halved from 0.01) —
  counter-pressure on policy collapse via entropy bonus.
- PR-C dual-bank V_spread canary (`7cd0dc0`) — T3 + alt at every
  checkpoint, SOFT-ABORT if T3<+0.20 OR alt<+0.07.

### CONDITIONAL on Track B verdict (Wave 2 scope)

| If V-B-A (one source ≥60%) | …apply source-targeted lever |
|---|---|
| dominant = pretrain | per-class target temperature on pretrain colony rows; OR class-weighted CE; OR reduce `recency_weight` for pretrain channel |
| dominant = bot | static bot corpus → refresh hook (`bot_corpus_refresh.enabled=true`, cooldown 25k); OR re-sample bot corpus on each best-model promotion |
| dominant = uniform_self | PSW (Prioritized Stratified Window on the replay buffer); under-sample colony class on selfplay slice |

| If V-B-B (multi-source 25-45%) | EMA + 2-stone aux + per-class target temp |
|---|---|

| If V-B-C (buffer colony-heavy ≥50% by step 2k) | refresh hook PRIORITY 1 + EMA on selfplay inference model |
|---|---|

| If V-B-D (trunk centroids collapse ≥50% by step 1k) | aux heads forcing trunk discrimination — 2-stone opponent-reply aux (Wave 2 implement) + maintain existing ownership + threat aux |
|---|---|

| If V-B-E (no clean match) | escalate to operator; NO real run launch |
|---|---|

### DEFERRED

- KL-weighted buffer writes (Q9) — design ready; parked unless V-B-C confirms feedback loop.
- Activation alternatives (SiLU/Mish) — parked until baseline healthy.
- Muon optimizer — parked until baseline established.
- WDL value-head migration — parked; A2 falsified arch-only fix.
- Gumbel CQV / KrakenBot wrapper / Phase 4.5 feature queue — parked behind real-run success.

## §4 Pre-registered success criteria

LITERAL L13 guard. Real run aborts on first failure; promotion gated
on ALL passing.

| ID | criterion | window |
|---|---|---|
| RR-G1 | dual canary T3 ≥ +0.20 sustained | steps 0 → 50k |
| RR-G2 | dual canary alt ≥ +0.07 sustained | steps 0 → 50k |
| RR-G3 | SealBot WR ≥ 13% at step 30k (n ≥ 200 effective games, Wilson95 lower-bound ≥ 9%) | step 30k |
| RR-G4 | SealBot WR ≥ 18% at step 50k (matches §148 v6 baseline lower-bound) | step 50k |
| RR-G5 | colony_a < 50/100 in eval rounds | every eval |
| RR-G6 | L34 anchor↑/sealbot↓ divergence alarm clean (anchor WR not rising while sealbot WR falls) | every eval round |

Promotion to Wave 3+ requires G1-G6 all PASS at step 50k AND the
trajectory does not soft-abort between 50k and 100k.

## §5 Stop conditions during the run

- Dual canary `both_pass: false` for 2 consecutive checkpoints — SOFT-ABORT.
- SealBot WR < 8% at step 30k — HARD-ABORT (matches §S180b stop condition).
- L34 divergence fires (anchor WR climbs ≥ 5pp while sealbot WR drops ≥ 5pp over 5 consecutive evals) — SOFT-ABORT.
- GPU NaN/Inf in gradients (`nan_or_inf_loss_skipped` count > 20 in any 100-step window) — HARD-ABORT.
- Standard §S180b hard-aborts: `grad_norm > 10` for 5 consecutive steps, `loss_nan`, `colony_ext_frac > 0.40`, `stride5_p90 > 60`.
- Wallclock > 24h on a single GPU — operator decision point.

## §6 Compute budget

| stage | cost | wall |
|---|---|---|
| Wave 2 dev (Track B verdict applied → lever stack final) | 1-2 days dev | 1-2 d |
| Pre-launch smoke (3000 steps to confirm levers behave) | $1.50 | ~6 h |
| Main run (100k steps @ ~14 s/step ≈ 14 GPU-h × ~$0.21/h) | ~$3 | ~14 h |
| Post-run analysis + sprint-log + memory | 0 GPU | 0.5 d |
| **TOTAL** | **~$5 + 2-3 days dev** | **~3 days incl. launch** |

C-LITE-2 reopen (only if Wave 2 fails its smoke) — additional $0.75 +
~3 h vast. Track B B4 launch — $1.50 + ~6 h (separate from main run
budget; required before Wave 2 dev kicks off).

## §7 Wave 2 plan

Dependency order:

1. **Operator runs Track B B4** — 3000-step instrumented run on
   `phase4.5/s181_track_b_instrumented` (`3201c39`) with
   `v6_botmix_s181_track_b.yaml`. Emits `per_source_grad_norm` events
   per step + `buffer_position_class_snapshot` events at ckpts
   500/1000/.../3000.
2. **Operator runs B3 trunk drift script** post-run.
3. **Apply V-B aggregator** — `audit/structural/track_b/B_aggregation.md`
   records verdict per the §1.7 decision tree.
4. **Pick V-B-conditional lever from §3** of this doc.
5. **Wave 2 dev:**
   - EMA on selfplay inference model (always — protects Track B baseline
     during Wave 2 sustained probes).
   - 2-stone opp-reply aux head (only if V-B-D).
   - Per-source lever from §3 conditional table.
6. **Wave 2 pre-launch smoke** — 3000-step lever-stack confirmation on
   vast (~6 h, $1.50).
7. **Wave 2 main run launch** — 100k steps from
   `bootstrap_model_v7full.pt` anchor under v7full encoding with the
   final lever stack. Operator-mediated.

**Do not auto-launch Wave 2.** Operator decides post-Track-B-verdict.

---

## Inputs cited

- `audit/structural/track_a/A6_aggregation.md` — A1-A5 verdicts, L47/L48/L49
- `audit/structural/track_c/C_LITE_1_v7full_reference.md` — v7full anchor
- `audit/structural/track_d_pipeline_regression.md` — pipeline regression + smoking-gun rank
- `audit/structural/track_b/B_launch_and_analysis_spec.md` — Track B run procedure + V-B verdict logic
- `docs/07_PHASE4_SPRINT_LOG.md` §S181-AUDIT — L47/L48/L49 + Track A verdicts + PR-B hygiene
- `engine/src/encoding/registry.toml` — v6 / v7full encoding parity
