# RUN4 — GNN production run design (GNN-integration program, R4 ratified b+)

**STATUS: FINAL (2026-07-16) — the 3-part S7 integration smoke gate PASSED**
(`reports/probes/gnn_integration/S7_smoke_gate.md` "S7 GATE — FINAL OVERALL VERDICT: PASS",
five runs, nine blockers closed en route). Every pre-registered launch decision is now RESOLVED
**except the throughput floor** (§0 row 4 / §4.1), which stays OPEN pending the 5080 rider (OQ-2).
Finalization is consolidated in **§9** (INIT adjudication, bf16 numeric LAW, launch-config as-built
pins, eval instrument, gate record, consolidated OPEN list, handoff). Sections §0–§8 are the
pre-registered body, kept verbatim with per-row status stamps in §0 + an OQ disposition table in §9.
Nothing here is launched — box remains off-limits until run3-CNN completes (operator 2026-07-14);
this doc gates the launch, it is not the launch act.

**Date:** 2026-07-14 (draft) · **Finalized:** 2026-07-16 · **Program:** GNN-integration · **Worktree:** `worktree-gnn-integration`

**Inputs consumed (verbatim citations at point of use):**
`reports/probes/gnn_integration/WPA_cuda_bench.md` (throughput, torch-vs-ORT, BUILD-HOT),
`docs/designs/gnn_ragged_contract_v1.md` (WP-B: legacy-v1 contract, HEXG buffer, D6-free-via-coord-
pre-rotation), `docs/designs/lean_d6_adopt_vs_avoid.md` (WP-C: LEGACY-V1-CONFIRMED, lean-D6 = run5
card), `docs/designs/gnn_integration_scope.md` §C6/§C7/§C8 (throughput / bootstrap loader / training
data path), `docs/handoffs/run3_convene_ruling.md` + `_amendment_1.md` (R2 rung, MIXED verdict,
evidence framing), `configs/variants/run3_dist65.yaml` (launch-package template),
`docs/designs/run3_d1_distributional_head.md` (dist65 + 234-probe eval battery),
`docs/designs/evalfair_instrument.md` (EVALFAIR offline strength instrument).

**Standing red-team orders bind (restated, load-bearing):** no search-time incremental deltas
(§S186 — the graph payload is built ONCE per evaluated leaf, never diffed); no value distillation
(INV-D1 — every value target derives from the game's own outcome, no teacher net / SealBot / solver
score in the loss path); no mctx / JAX (native Rust Gumbel already exists, mctx-feasibility-REJECT);
no custom CUDA kernel (§D-STRIX kernel REJECT — see §7 transfer test); one resolver per knob
(WP-B `collate_graph_batch` is the SINGLE wire reader); benches on the REAL self-play distribution
(mean 490/2932 nodes/edges, WP-A frozen `wpa_positions.json`, NOT the human-corpus prior).

---

## 0. Ruling summary (the pre-registered decisions, one table)

| # | Decision | Ruling | Falsifier / gate | Status (2026-07-16) |
|---|---|---|---|---|
| 1 | INIT | **BC-prefit** (`gnn_bc_040000.pt`) as init, **corpus-mix OFF**. Fallback = fresh+mixing (option B). | Divergence signature DS-1/2/3 by ≤50k → restart fresh+mixing (§1.3) | **FINAL** — warm-start seam wired + landed-verified **46/46** tensors (OQ-5 live-fire); fresh-init fallback proven bootable by the gate. §9.1 |
| 2 | Schema | **legacy-v1** axis-graph (WP-C LEGACY-V1-CONFIRMED). Lean-D6 = run5 card. **D6 graph-space aug IN scope** (free via option-c coord pre-rotation, WP-B). | lean-D6 promotion rule = `lean_d6_adopt_vs_avoid.md` §3 | **FINAL** — `gnn_axis_v1` is the only `representation="graph"` encoding; byte-parity oracle green (Part-3). |
| 3 | Net scale | **probe-scale ~284k** (the net that measured +414). | NET-CAPACITY plateau falsifier (§4.2) → scale to prod (run4-v2) | **FINAL** — as-built `GnnNet`+dist65 = **286,082** params (BC 283,970 + fresh dist65 head, §9.3). |
| 4 | Throughput floor | **≥ 1.0k steps/hr at STEP-0** (5080, probe-scale, Rust builder, prod distribution). | below floor = NO LAUNCH → BUILD-HOT perf sub-package first (§4.1) | **OPEN** — 5080 rider (OQ-2); the ONE decision finalize leaves open (§9.7 handoff). |
| 4b | Numeric regime | **bf16 graph autocast** — `amp_dtype_for("graph")` returns bf16 unconditionally in code (§9.2). | fp16 GINE sum-agg overflow on prod-scale graphs (F9) | **FINAL/LAW** — S7 run-3 F9 → run-4/5 bf16 zero-non-finite over ~200 min live self-play. |
| 5 | Eval | EVALFAIR deploy battery + strix-raw external bar + value-health 234-probe; eff_n = distinct games. Graph book = EVALFAIR d5 **r=5** + `--graph-eval-book-radius` override (§9.4). | stop rule §3.3 | **FINAL (instrument)** — Part-2 ran end-to-end vs SealBot d5, eff_n honest (raw=8/deduped=8); in-loop eval loud-skips graph rounds (OQ-8 rider). |
| 6 | Box | run4 takes box when run3-CNN stops per run3's rule; smoke gate NOT PASS by then → box STOPS (§5). | operator order: off-limits until run3 completes | **FINAL** — smoke gate is PASS, so the box-STOPS precondition (§5) is cleared; timing still gated on run3-CNN stop. |
| 7 | Launch artifact | **`configs/variants/run4_gnn.yaml`** (production, bs=256 open) + **`run4_gnn_smoke.yaml`** (S7 gate vehicle, bs=16). Pins in §9.3. | launch-path parity test (`test_run4_gnn_launch_path.py`) | **FINAL** — both landed + committed (`d4c620c`); resolved-config parity + path-disjointness test-pinned. |

---

## 1. INIT — BC-prefit vs fresh+mixing (RULE)

**Pick: BC-prefit init from the banked 40k artifact, corpus-mix OFF.** This is C7's recommendation
(A), taken with the R2 caveat stated and monitored.

**Artifact (banked, pulled locally):** `checkpoints/probes/gnn_bc/gnn_bc_040000.pt`
(283,970 params, state under key `model_state_dict`). NOTE: path is `.../gnn_bc/gnn_bc_040000.pt` —
NOT the `.../gnn_lr1e-3/...` subdir named in the dispatch prompt; the artifact is at the flat path.
It is an **INIT, not a ladder rung** — self-play starts from these weights at step 0.

### 1.1 Evidence on record — BOTH sides, honestly

**FOR prefit:**
- The prefit carries the +414 [+320,+560] BT-Elo inductive-bias signature. WP3 verdict 2:
  Δ(gnn-bc − cnn-bc) = **+418, CI [+318,+580], excl 0**, head-to-head 59-0-5 (`_amendment_1.md`
  RECORDED VERDICT 2). R1 fixed the cnn-bc window-0 decode handicap and MIXED STANDS (Δ=+414
  [+320,+560], `d-l-strixprobe-state`). This is the ENTIRE run4 evidence base — run4's thesis is
  that architecture is the binding constraint, and the 40k prefit is the direct materialization of
  the measured architecture advantage. Fresh init forfeits it at step 0.
- The prefit folds the human-corpus prior into the INIT rather than diluting every self-play batch
  (§C7). This is our structural advantage over strix, which trains from a from-scratch radius
  curriculum with no corpus.
- **Under the throughput penalty the case STRENGTHENS.** WP-A projects ~0.9-1.25k steps/hr vs
  run2's 4.4k — **4-5× slower per GPU-week** (WPA throughput projection). Re-learning the GNN
  representation from scratch through slow RL is the most expensive possible use of the strained
  budget; a strong measured init is worth MORE, not less, when steps are scarce.

**AGAINST prefit:**
- R2 verdict: BC **REGRESSED-VS-40K** at 200k — Δ(200k − 40k) = **−76, CI excl 0, BC-saturated**
  (`d-l-strixprobe-state`; corpus saturates BC at 40k, red-team confirmed). The prefit may anchor
  self-play into BC's imitation basin (human-strength argmax patterns), which self-play must then
  escape.
- fresh+mixing (option B: corpus-mix `initial_pretrained_weight` 0.8 → `min` 0.1, `decay_steps`
  200000, `run3_dist65.yaml:92-96`) is the run2-PROVEN anchor pattern — the corpus prior enters as
  a decaying regularizer, not a fixed basin.

### 1.2 Re-validation of the R2 prior (CLAUDE.md discipline — cite / context / transfer test)

- **Prior:** R2 BC-saturated, Δ(200k−40k) = −76. **Original context:** measures whether MORE BC
  TRAINING helps — a supervised-imitation-length regime. **Transfer test to the INIT question:**
  R2 does NOT measure "40k-prefit + RL vs fresh + RL." It says the 40k checkpoint is the *peak-BC*
  representation (longer BC over-imitates the corpus) — which is precisely why we init from the
  **40k** artifact and not a longer-trained one. It does NOT adjudicate whether prefit-then-selfplay
  locks into a bad basin; that context does not transfer. **Kept, scoped:** R2 pins WHICH prefit
  (40k, the peak), not WHETHER to prefit.
- **Prior:** `bot-mix-retired-s178-useless` (operator 2026-07-10: `bot_batch_share` stays 0).
  **Context:** bot-corpus slot in self-play was useless. **Transfer:** supports corpus-mix OFF as
  the default (the prefit already carries the prior; mixing on top double-counts — §C7). Transfers
  as the mix-OFF default; the F1 preserve-ckpt-baked warning means the run4 variant must DECLARE
  `bot_batch_share: 0` and corpus-mix weights explicitly, not inherit.

**Decisive tradeoff:** prefit buys the strongest measured start and avoids re-learning the
representation through 4-5×-slower RL; its one risk (imitation-basin lock-in) is exactly what the
divergence signature (§1.3) monitors mechanically by 50k — cheaper to detect-and-restart than to
forfeit the +414 up front. Fresh+mixing is the pre-registered fallback, not the default.

### 1.3 DIVERGENCE SIGNATURE — falsifies the prefit pick by ≤50k steps (mechanical fire)

The pick is falsified iff the prefit anchored into the BC imitation basin and self-play cannot
escape (or is actively un-learning the prefit). All thresholds reference `run2_mw_fresh`'s trajectory
at the same step; the run2 reference values are stamped from run2 logs at launch (OQ-1). A monitor
evaluates these at the 25k and 50k eval boundaries.

| ID | Metric (deploy/EVALFAIR unless noted) | Fire threshold | Class |
|---|---|---|---|
| **DS-1 NEGATIVE-TRANSFER** | SealBot WR at 50k | `< gnn_bc_040000.pt`'s OWN frozen SealBot WR measured at step 0 (self-play must not un-learn its init) | KILL |
| **DS-2 BASIN-LOCK** | root policy entropy at 50k | `< 0.7 × run2_mw_fresh` entropy@50k **AND** Δentropy(50k − 5k) `< +5%` (frozen low entropy self-play doesn't reopen) | KILL |
| **DS-3 AXIS-BIAS** | self-play E-W axis share at 50k | outside corpus 38% ± 15pp (i.e. `> 53%` or `< 23%`) — dual-purpose §119 leak canary + lean-D6 §3 gate | KILL |
| **DS-4 VALUE-CALIB** | dist65 value ECE at 50k (holdout) | `> 0.15` (fresh value head; may lag legitimately) | WARN-only |

**Mechanical rule:** ANY of DS-1 / DS-2 / DS-3 fires by 50k → prefit REJECTED → restart from
fresh+mixing (option B: fresh trunk, corpus-mix 0.8→0.1 decay 200k, `bot_batch_share` 0). DS-4 is a
BCE-drift-style canary (mirrors the C4 threat-probe C4-warning): logged, does not abort.

**DS-3 calibration (WP-2 red-team, 2026-07-14, measured on the prefit-40k init):** D6
non-equivariance lives in the POLICY head — policy-logit gap under rotation max 2.196 / mean 0.221
/ p90 0.485 (and BC-without-aug AMPLIFIED it 1.8–5× over random init) — while the VALUE head is
near-equivariant (~0.005). Consequence, binding on instrumentation: the DS-3 canary MUST read a
policy-side signal (move axis share, as specified) — any value-side axis instrument is insensitive
by construction. Baseline gap numbers above are the reference for judging whether training with
aug is SHRINKING the gap (healthy) or growing it (leak).

Rationale per signal: DS-1 is true-north (SealBot WR, deploy regime) — a prefit being un-learned is
the clean number a monitor reads. DS-2 catches the specific basin failure (BC over-imitation =
frozen low entropy). DS-3 reuses the §119 axis-share canary (leak real, aug is the fix — transferred
from `lean_d6_adopt_vs_avoid.md` §5) AND is the lean-D6 §3 pathology gate — one metric, two duties.

---

## 2. Schema — legacy-v1, D6 aug in scope (RULE, restated from WP-C)

**Verdict: LEGACY-V1-CONFIRMED** (`lean_d6_adopt_vs_avoid.md`). run4-v1 ships the **legacy-v1
axis-graph** — 11-dim node (relative-7 + threat-4), GINE `(E,5)` edge_attr, single edge list with
all-zero dummy edges — the exact schema that produced the +414 probe result
(`hexo_rl/bots/strix_v1_graph.py`, port SHA `c381ffbe`). **The single evidence-bearing variable in
run4 = architecture; nothing else.** Any schema change forfeits the +414 evidence base (WP-C Cost 1).

**Lean-D6 disposition: AVOID for v1 → run5 card.** Adopt only on the pre-registered rule
(`lean_d6_adopt_vs_avoid.md` §3): (a) WP-B declares graph-space D6 INFEASIBLE-ON-LEGACY-V1 AND
(b) projected un-augmented loss ≥ X (30% eff-sample loss OR any §119-class axis pathology). WP-B
returned **FEASIBLE-ON-LEGACY-V1** → the rule TERMINATES at AVOID; lean-D6 is a run5 card with
promotion evidence named in WP-C §3 (a committed lean-D6 ≥ legacy measurement — absent even for
strix; OR a WP-B INFEASIBLE + pathology a legacy augmentation port cannot close; OR browser/wasm
made primary AND a native lean-D6 forward built — today it does not exist, WP-C §1e).

**D6 graph-space augmentation IS in scope for v1.** WP-B Part 3: FEASIBLE-ON-LEGACY-V1, realized
**FREE** via option-(c) coord pre-rotation — rotate stored stone coordinates by the D6 element, then
rebuild the graph; the Rust builder emits the correctly-oriented graph natively (correct axis labels
+ signed distances), zero new graph-symmetry code. Dropping aug = ~12× fewer effective distinct
samples (WP-B Part 3, `sym_tables.rs:N_SYMS=12`) — a real hit compounding onto the throughput
penalty, NOT a rounding error. **Aug stays IN.** (This is also why the buffer strategy is option-(c)
store-positions-rebuild-at-sample, §6.)

---

## 3. Eval plan (RULE)

Same EVALFAIR battery as run3, plus the strix-raw external bar and the value-health 234-probe.
Deploy-regime match binds (D-LADDER): the eval must exercise the SAME regime the run deploys —
multi-window no-drop, Gumbel-SH g=0 m=16, 150 sims (`run3_dist65.yaml:64-72`).

### 3.1 The battery

- **EVALFAIR deploy strength** (`scripts/evalfair/`, `evalfair_instrument.md`): the verdict-2 loop,
  SealBot d5 + Gumbel-150, r5, **book_v2 64-opening fair book × color-swap = 128 games**
  (`evalfair_r5_*`). Measured cost 15.2 s/game → a 128-game read ≈ 8 min at 4 workers on the laptop
  (offline; NEVER on the run box during self-play). This is the true-north (SealBot WR) + net-vs-net
  promotion instrument. Replaces the §D-LADDER "triple-miss" temp-0.5 PUCT-128 proxy.
- **strix-raw external bar (the ceiling):** strix ranks **#1 raw-policy at +121 Elo (argmax,
  no search)** and **#1 deploy at +313 (Gumbel-g128)** (`gnn_readjudication.md` §2,
  `argmax-tourney-results`). strix-raw is the same representation-CLASS as run4's GNN — the +414
  predicts run4's GNN should approach/match strix-raw once RL-trained. Deploy-matched net-vs-strix
  WR is the SUCCESS ceiling (§3.3). Caveat (C5 red-team): the strix bar delegates to strix's own
  engine/venv (`strix_g128_child.py`) — it is a cross-process opponent, runnable offline only
  (OQ-9).
- **Value-health 234-probe** (`reports/valprobe/probe_set_v1.md`): 234 distinct card1 positions
  (SealBot head-lost proof AND raw v ≥ −0.5 AND replay_match — value optimistic while provably
  lost), dedup by (zobrist, side_to_move, moves_remaining). Re-scored on run4 GNN checkpoints:
  AUC(lost vs safe) on the decoded dist65 value. This is the D-LOCALIZE value-blindness
  discriminator (memory `d-localize-value-target-verdict`) — the run4 GNN must NOT regress its
  234-probe AUC below run3-CNN's on a matched checkpoint (§3.3 ABORT).

### 3.2 Diversity + eff_n discipline (D-LADDER + §D-ARGMAX)

The deploy regime is deterministic (argmax / Gumbel g=0 from a fixed opening) → it collapses to
~2 distinct games/pair; a BT/Wilson CI over the raw game COUNT is over-confident by √(copies)
(§D-ARGMAX, `d-ladder-verdict-and-eval-mismatch`). Therefore:
- **Diversity is injected by the book, not by temperature:** book_v2's 64 fair openings × both
  colors supply the game diversity (the run3_d1 probe used exactly this: "opening DIVERSITY is
  load-bearing, both engines deterministic; D-VETO got 40/40 distinct games this way"). Fresh seeded
  books per radius stage (`run3_dist65.yaml:124`).
- **eff_n = DISTINCT games:** dedupe byte-identical move sequences (canonical D6-orbit
  representative); bootstrap every strength CI over distinct games, NOT the nominal count. Report
  nominal n, distinct games, and the distinct source-game count alongside every WR (the §D-ARGMAX
  honest-CI protocol). No "CI-resolved" strength claim without the deduped-bootstrap CI.

### 3.3 STOP RULE (pre-registered, with numbers)

Budget frame: at the ≥1.0k steps/hr floor (§4), 300k steps ≈ **300 GPU-hr ≈ 12.5 GPU-days**; the
NULL decision lands inside that. run4 STOPS at the FIRST of:

1. **SUCCESS-STOP.** deploy SealBot WR (EVALFAIR 128-game book, eff_n distinct) **CI-lower ≥
   run2-best-banked SealBot WR** AND net-vs-**strix-raw** deploy WR **CI-lower ≥ 0.50** (parity with
   the external ceiling). → bank + promote; run4 architecture thesis CONFIRMED.
2. **NULL-STOP.** **300k steps** reached AND deploy SealBot WR **CI-upper < run3-CNN's matched-step
   (≤200k) SealBot WR**. → the representation advantage did not materialize at production scale;
   bank + writeup; reconsider net-scale (§4.2) / lean-D6 (run5). (The run2-slope one-variable
   comparison is explicitly given up per `_amendment_1.md` R4 — the comparison baseline is the bar
   ladder + strix-raw ceiling, used here.)
3. **ABORT-STOP.** ANY of: DS-1/2/3 fires ≤50k (§1.3); hard-abort monitor (draw-rate ≥ 0.55 ×3
   consec, or grad-norm > 10, `run3_dist65.yaml:135-139`); 234-probe AUC regresses below run3-CNN's
   on a matched checkpoint (value-blindness NOT fixed — the D-LOCALIZE gate). → halt + diagnose.
4. **HANDOFF-STOP.** run3 needs the box back, or the GPU-day billing bound is hit. → bank the last
   PROMOTED checkpoint; run4 is resume-capable (weights-only restamp).

**INV pins (§S178-style, promotion-gate enforced):**
- **INV-R4-1 (monotone true-north):** a promoted checkpoint's deploy SealBot WR may not fall below
  the previous promoted checkpoint's WR by more than its CI. (The net-vs-net promotion gate is
  CI-lower > 0.5; SealBot WR is the true-north overlay, memory `deploy-strength-inloop-cost`.)
- **INV-R4-2 (external ceiling target):** net-vs-strix-raw deploy WR is the SUCCESS ceiling (clause
  1); a run that plateaus far below it before 300k triggers the NET-CAPACITY read (§4.2).
- **INV-D1 (value targets, inherited verbatim from `run3_d1_distributional_head.md`):** every dist65
  target derives from the game's own outcome (`outcomes` + `value_target_valid`); NO teacher /
  SealBot / solver / distilled-net score anywhere in the loss path; `soft_z_lambda` pinned 0.

**One-line stop rule:** STOP at the first of — SUCCESS (SealBot-WR CI-lower ≥ run2-best AND vs-strix-
raw CI-lower ≥ 0.50), NULL (300k steps with SealBot-WR CI-upper < run3-CNN@≤200k), ABORT
(DS-1/2/3 ≤50k, or draw≥0.55×3 / grad>10, or 234-probe AUC below run3-CNN), or HANDOFF (box reclaimed
→ bank last promoted ckpt).

---

## 4. Throughput floor + net scale (RULE)

### 4.1 Throughput floor at STEP-0

WP-A measured (laptop 4060, prod distribution, ratio-transfer to 5080): per-leaf ≈ builder +
forward → probe-scale 0.539 + 0.335 = **0.874 ms/pos**, prod-scale 0.539 + 0.696 = **1.235 ms/pos**;
against run2's 4.4k steps/hr this projects **~1.25k steps/hr probe-scale / ~0.9k prod-scale** if
inference-bound (WPA throughput projection, below the scoping doc's optimistic 2-3k/h).

**Floor: ≥ 1.0k steps/hr, measured at STEP-0 on the run box (5080), probe-284k net, Rust builder,
prod distribution.** Basis: (a) it is below the probe-scale projection (~1.25k) with margin for the
not-yet-optimized builder; (b) it is the economic floor to reach the 300k NULL decision inside
~12.5 GPU-days (300k / 300 GPU-hr = 1.0k/hr). **Below floor at step-0 = NO LAUNCH** → dispatch the
**BUILD-HOT perf sub-package** first (WP-A BUILD-HOT: the Rust builder is 0.539 ms/pos = **62% of
probe-scale per-leaf cost** — the single largest optimizable term; §S186 binds: optimize the
once-per-leaf build, never a search-time incremental delta). Re-measure the floor from the **5080
rider** (§5) BEFORE launch — absolute ms do not transfer from the 4060; only the GNN-vs-CNN ratio
does, and the 5080's dense-GEMM edge likely widens CNN-vs-GNN slightly (WPA instrument note).

### 4.2 Net scale for run4-v1

**Choice: probe-scale ~284k** (the `gnn_bc` architecture that measured +414). Options and cost
(WP-A headline, bs=64, vs CNN 4.27M fp16 = 1.0×): probe-284k torch-CUDA fp16 = **1.54×**;
prod-1.1M = **3.20×**; strix-4.25M = worse than prod (theirs is wider+deeper — the 3.2× is a LOWER
bound on a true 4.25M forward, WPA caveat).

Rationale:
1. **Evidence discipline (same logic as legacy-v1):** the +414 is a PROBE-scale measurement; shipping
   the measured net keeps the single-evidence-bearing-variable discipline. Bigger nets place the
   architecture bet on a scale never measured on our data.
2. **Throughput:** probe-scale (1.54× CNN → ~1.25k/hr) is the only option comfortably above the 1.0k
   floor. prod-1.1M (~0.9k) sits AT/below the floor; strix-4.25M is disqualifying without a builder
   perf package first.
3. Net scale is a SEPARATE lever (like lean-D6) — scale up in run5/run4-v2 only on evidence.

**NET-CAPACITY falsifier:** if probe-284k self-play PLATEAUS — deploy SealBot WR slope over a 50k
window within CI of 0 — at an absolute WR BELOW run3-CNN's matched WR before 200k, the net is
under-capacity for RL (BC-284k strength ≠ RL-284k ceiling). → escalate to prod-scale (run4-v2) with
the ~0.9k/hr throughput cost accepted, or fold into the NULL-STOP writeup. This is DISTINCT from
DS-2 basin-lock: basin-lock is early (≤50k) entropy collapse; capacity-plateau is a LATE flat slope
at a mediocre level. Both are pre-registered; they do not alias.

---

## 5. Box handoff (RULE)

- **Timing:** run4 takes the box when run3-CNN stops per run3's OWN stop rule. Until then the box is
  **off-limits to this program** (operator order 2026-07-14; supersedes the WP-A "GPU window").
- **Smoke-gate precondition:** if the 3-part integration smoke gate is NOT PASS by the time run3-CNN
  stops, **the box STOPS — no idle billing.** run4 does not launch on an unproven integration; the
  smoke gate is the hard precondition (its exact PASS criteria = OQ-7).
- **5080 WPA rider (handoff window, BEFORE launch):** rerun `scripts/research/gnn_infer_bench.py`
  UNCHANGED on the frozen `reports/probes/gnn_integration/wpa_positions.json`, on the 5080, in the
  handoff window. Re-derive per-leaf ms → re-check the §4.1 floor. If the 5080 step-0 throughput is
  below 1.0k steps/hr → NO LAUNCH → BUILD-HOT first. (The rider is the ONLY on-box GNN work before
  launch; it does not touch a live run.)

---

## 6. Bootstrap loader, corpus-mix, monitoring (build requirements — pointers, not redesign)

### 6.1 Bootstrap loader (C7 red-team — the F1-class silent-partial-load surface)

`c796887` is the WRONG precedent (CNN value-head-only fc1/fc2 copy between shape-identical CNNs).
The run4 transfer is **representation + policy** weights from `GnnBcNet` (keys under
`model_state_dict`) onto the production `gnn_net.py` module names — a real loader, not a lift.
Requirements (WP-B audit nodes 11d/11e + §C7):
- **Trainer loader graph branch:** `scripts/train.py --checkpoint gnn_bc_040000.pt` routes through
  `trainer_ckpt_load.py::_resolve_checkpoint_encoding` → `resolvers.py::detect_encoding_from_state_dict`,
  which RAISES on any state dict lacking `trunk.input_conv(.conv)?.weight` (verified strict path).
  Add a **graph-detect branch in `resolvers.py`, SINGLE-SOURCED with the C4 eval loader** (WP-B node
  11d — both hit the same F1 detection helper; do not duplicate).
- **Landed-verify (F1 guard):** after a `strict=False` load, `torch.allclose` post-load verify
  (the E1 `checkpoint_loader.py:593-603` pattern) MUST cover **representation + policy (+ dist65)**
  tensors — NOT value-only. A silent key-mismatch drop is the exact F1 failure class (wrong
  representation self-played 272k+ steps undetected, `d-forensic-f1-lineage-single-window-cascade`).
- **build_net authority:** the construction-family sites (orchestrator:677, lifecycle:66/172,
  anchor:569 — WP-B nodes 11a-c, all SILENT-CORRUPT: build a CNN unconditionally) are replaced by
  ONE `build_net(spec, state)` dispatch on `spec.representation` (CONFRES `build_player` precedent,
  `69442e5`). This is C4/C7 cross-cutting, single-sourced with the WP-B contract.

### 6.2 Corpus-mix on the unified manifest (C8)

INIT = prefit, corpus-mix OFF (§1) → the corpus-mix path is the **fallback (option B) only**. When
it fires, the GNN cannot consume the dense-plane `.npz` corpus (C8): re-export as HEXG graph-position
records (WP-B Part 2.6 — the probe's `_compact_example` precedent) or replay-and-rebuild at load
(C1 build cost). Guarded by the existing `batch_assembly.py:195` plane-count assert (LOUD-FAIL on a
dense→graph mismatch).

**Unified-manifest claim (dispatcher FRAME): run3 WP0.4 resolved the single corpus manifest that
both CNN and GNN training draw from.** This doc does NOT redo that work — but it could not be located
in the repo (searched docs/ + reports/; no `WP0.4` artifact found; the only "unified manifest" hit is
the temperature-schedule unification in `01_architecture.md:291`, unrelated). → **flagged OQ-3:
verify run3 WP0.4 landed before finalize.** The GNN-specific re-export/parity mechanism (HEXG,
WP-B Part 2.6) is required regardless of the manifest's state.

### 6.3 Monitoring / watchdog reuse (no new machinery)

- **Stall watchdog:** `selfplay_stall_timeout_sec: 1800.0` (`run3_dist65.yaml:141`) — the run2
  eval-boundary CUDA-livelock watchdog (memory `run2-stall-watchdog`). run4 inherits it unchanged.
- **Promotion-gate subprocess isolation:** the run2 livelock ROOT fix is on master, **default-OFF**
  (`step_coordinator.py:318` `promotion_gate_subprocess_isolation: bool = False`; not a re-opened
  hole per `promotion_gate_worker.py:35`). run4 turns it **ON** (`run3_dist65.yaml:55`), as run3
  does. Both watchdog + isolation ride onto run4 verbatim.

---

## 7. Falsified-register transfer tests (priors leaned on — cite / context / transfer)

- **§D-STRIX kernel REJECT** (`07_PHASE4_SPRINT_LOG.md:566`). *Context:* "HeXO's dense CNN has no
  ragged-batching problem" → custom CUDA kernel rejected. *Transfer:* run4 IS a GNN with a ragged
  problem (WPA confirmed), so the kernel-kill's PREMISE no longer holds — but the row is
  perf/kernel-scoped and the standing red-team order (no custom CUDA kernel) is HONORED anyway; the
  BUILD-HOT lever is builder + torch-CUDA optimization, not a kernel. Not relied on to justify a
  kernel; cited only to affirm the no-kernel order still binds.
- **D-LADDER eval-deploy mismatch** (`d-ladder-verdict-and-eval-mismatch`). *Context:* live eval
  (PUCT temp-0.5 64-sim) ≠ deploy (Gumbel-150) → triple-miss. *Transfer:* direct — run4 eval uses the
  deploy regime (EVALFAIR Gumbel-150). Load-bearing in §3.
- **§D-ARGMAX eff_n** (`d-ladder`, CLAUDE.md). *Context:* deterministic argmax collapses to ~2
  distinct games/pair; raw-count CI over-confident by √copies. *Transfer:* direct — run4's deploy
  regime is deterministic; §3.2 dedupes + bootstraps over distinct games.
- **§119 D5/D6 axis bias** (`07_PHASE4_SPRINT_LOG.md:1128`, via `lean_d6_adopt_vs_avoid.md` §5).
  *Context:* CNN un-augmented buffer → E-W axis bias; RESOLUTION = augment, not re-architect.
  *Transfer:* the GNN legacy-v1 has analogous rotation-leaking features; transfers as "leak real AND
  aug is the fix" → DS-3 canary + aug-stays-in (§2). Kept, scoped.
- **D-FULLSPEC entanglement** (`d-fullspec-entangled-feature-problem`). *Context:* frozen v6_live2
  **CNN** features can't separate win/loss. *Transfer:* CNN-feature-specific; says nothing about
  GINE-legacy node/edge schema (WP-C §5). Does NOT transfer to the schema choice; noted only as the
  reason DS-4 value-calibration is a WATCH, not a prediction.
- **run2 corpus-mix decay 200k** (proven anchor pattern, `run3_dist65.yaml:92-96`). *Context:* fresh
  init + decaying corpus prior sustained run2. *Transfer:* the fallback (option B) recipe when the
  divergence signature fires (§1.3). Kept as fallback, not default.

---

## 8. OPEN QUESTIONS — the 3-part integration smoke gate MUST answer these before finalize

**> DISPOSITION (2026-07-16, gate PASS): resolved in §9.6. OQ-3/4/5/6/7 RESOLVED by the gate;
OQ-1/2/8/9 remain OPEN (OQ-2 is the sole LAUNCH-gating open — the throughput floor). Original
text kept below for provenance.**

1. **OQ-1 — run2 reference values for DS-1/2/3.** Stamp `run2_mw_fresh`'s policy-entropy@50k,
   E-W axis-share trajectory, and SealBot-WR trajectory from run2 logs; and measure
   `gnn_bc_040000.pt`'s frozen step-0 SealBot WR (DS-1 anchor). Thresholds are un-fireable until
   stamped.
2. **OQ-2 — 5080 rider throughput.** Rerun `gnn_infer_bench.py` on `wpa_positions.json` on the 5080
   in the handoff window; re-derive per-leaf ms; confirm STEP-0 ≥ 1.0k steps/hr floor (§4.1/§5).
3. **OQ-3 — run3 WP0.4 unified manifest. RESOLVED (WP-5 design, 2026-07-15):** WP0.4 LANDED at
   commits `1d4a206` + `42e4e90` (sha-pinned `_CORPUS_SHA_PINS` canonical-corpus resolver in
   `resolvers.py`) — the earlier "not locatable" flag searched only docs/, not git log. It provides
   the single-resolver seam for the graph corpus branch but NOT a graph-consumable corpus: the
   HEXG re-export ships in WP-5 (~2 pd, BUILD-not-defer per DS-kill-margin ruling,
   `gnn_training_path_design.md` §7.2); only the mixing-batch wiring defers.
4. **OQ-4 — corpus re-export parity (fallback path).** HEXG re-export vs replay-and-rebuild for the
   option-B corpus-mix; byte-parity of the re-exported graph vs `build_axis_graph_raw` on
   `wpa_positions.json` (WP-B Part 4.1 oracle).
5. **OQ-5 — loader landed-verify.** Does `gnn_bc_040000.pt` (`GnnBcNet`, keys under
   `model_state_dict`) map cleanly to production `gnn_net.py` module names? The `torch.allclose`
   landed-verify on representation + policy (+ dist65) MUST pass — a silent drop is F1 (§6.1). The
   smoke gate loads the artifact and asserts the verify fires.
6. **OQ-6 — dist65 head geometry on the GNN.** The run3_d1 K-cluster argmin routing was CNN-specific
   (per-cluster → argmin → CE on that cluster). The GNN is WHOLE-BOARD (no K-cluster, WP-B node 6
   note) → dist loss = plain single-window CE (matching the E1 `e1-cardone-integration-scoped`
   finding: single-window forward → dist loss = plain CE, no argmin routing). Confirm the pooled-head
   geometry + that dist65 warm-starts fresh over the prefit (E1 REVIVE: dist65 warm-starts fine from
   an absent value head).
7. **OQ-7 — 3-part integration smoke gate PASS criteria.** The gate's exact 3 parts + PASS
   thresholds are not pinned in-repo. Define before launch (candidate parts: WP-B contract-sound +
   byte-parity oracle + adversarial-assertion coverage; a step-0 self-play smoke that writes+samples
   HEXG records and round-trips through torch-CUDA collate; and the loader landed-verify OQ-5). The
   box STOPS if this is not PASS by run3-stop (§5).
8. **OQ-8 — promotion gate through the GNN.** The deploy-regime Gumbel promotion gate needs a
   searched-GNN path through our stack (C5 depends on C3). Confirm the promotion gate can run
   net-vs-net on GNN checkpoints in the isolated subprocess before relying on INV-R4-1.
9. **OQ-9 — strix-raw runnable offline.** strix-raw delegates to strix's own engine/venv
   (`strix_g128_child.py`, C5 red-team) — confirm it runs as an EVALFAIR offline opponent for the
   INV-R4-2 external ceiling read.

**Open questions count: 9.** All 9 gate finalize; OQ-2/OQ-5/OQ-7 additionally gate LAUNCH.

---

## 9. FINALIZATION (2026-07-16 — S7 gate PASS)

Per the program endpoint clause ("PASS all three → run4 design doc finalizes: init decision +
everything except the throughput floor, which remains OPEN pending the 5080 rider"). Source of
record: `reports/probes/gnn_integration/S7_smoke_gate.md` (5 runs) + `S7_f9_bf16_fix.md` (F9) +
the committed launch pins (`d4c620c`). **No decision below is invented — each is the draft's own
pre-registered option adjudicated against what LANDED, or a gate finding recorded as LAW.**

### 9.1 INIT — DECISION: BC-prefit warm-start (the pre-registered pick, CONFIRMED)

**Decision:** run4 launches from the **BC-prefit init** (`checkpoints/probes/gnn_bc/gnn_bc_040000.pt`),
transferred via the `gnn_warm_start` YAML seam (`hexo_rl.training.gnn_warmstart`), **NOT** `--checkpoint`
(WP-4's `assert_full_gnn_checkpoint_or_raise` correctly refuses a BC-prefit-only state dict on the
resume path; the seam exists to route around that guard on purpose). Corpus-mix **OFF**. This is
§0-row-1 / §1's pick, unchanged.

**Adjudication vs the pre-registered options (draft §1.1 / §1.2):**
- **Option A (BC-prefit) — what LANDED and is verified.** The warm-start seam is wired and fired live
  in the gate: `gnn_warmstart_loaded loaded_keys=46 verified_tensors=46` — the `torch.allclose`
  landed-verify over representation + policy tensors PASSED (OQ-5 closed, F1-guard live). The banked
  `gnn_bc_040000.pt` artifact carries the +414 architecture signature that is run4's entire evidence
  base; fresh init would forfeit it at step 0, and under the ~4-5× throughput penalty re-learning the
  representation through slow RL is the most expensive possible use of the budget (§1.1 FOR-prefit).
  The dist65 value head stays **FRESH** over the prefit (E1 REVIVE — dist65 warm-starts fine from an
  absent value head; OQ-6 confirmed whole-board GNN → plain single-window dist65 CE, no K-cluster
  argmin routing).
- **Option B (fresh + corpus-mix) — the pre-registered fallback, now proven bootable.** The gate ran
  **fresh-init** for the training smoke throughout (warm-start decoupled from the train-step smoke by
  design) and it launched, self-played, trained, and checkpointed clean — so the fallback recipe is
  a proven-bootable restart target, not a paper option.

**Falsifiable rationale (applied verbatim from §1.3):** the pick is falsified iff the prefit anchored
into the BC imitation basin and self-play cannot escape. The **DS-1/2/3 divergence signature** remains
the mechanical falsifier — ANY of DS-1 (SealBot WR at 50k < the prefit's own frozen step-0 WR) /
DS-2 (entropy basin-lock) / DS-3 (E-W axis bias) fires by ≤50k → prefit REJECTED → restart from
fresh+mixing (option B). **Caveat (OPEN, gates the monitor not the launch):** DS-1/2/3 thresholds are
un-fireable until OQ-1 stamps `run2_mw_fresh`'s entropy@50k / axis-share / SealBot-WR trajectory and
the prefit's frozen step-0 SealBot WR (§9.6). The launch decision is final; the monitor arming is a
launch-window OQ-1 task.

### 9.2 Numeric regime — LAW: bf16 graph autocast (F9)

**LAW for run4:** the GRAPH path trains and infers under **bfloat16** autocast, pinned in CODE via
`hexo_rl/model/build_net.py::amp_dtype_for(representation, config)` — `representation=="graph"`
returns `torch.bfloat16` **unconditionally** (it does NOT consult the `amp_dtype` config key, even if
declared; the dense/`"grid"` path is byte-identical fp16 via the pre-existing knob). GradScaler is
auto-disabled on bf16 (`scaler_enabled = fp16 and amp_dtype == float16` → False).

**Mechanism one-liner:** `_GINEConv` sum-aggregation accumulates one ReLU'd message per incoming edge
onto an un-damped residual stream; on production-scale ply-cap-deep self-play graphs (~500-node
late-game positions, conv-stack absmax reaching 5.56e4 vs fp16's 6.55e4 ceiling) select batches
overflow → `inf` → LayerNorm → NaN through the value/embedding head — bf16's full fp32 exponent range
removes the ceiling at identical 2-byte cost, native on both the dev 4060 (sm_89) and the launch 5080
(sm_120). Evidence: fp16 produced 136 non-finite events in <2 min (S7 run-3); bf16 produced **zero**
across ~200+ min of cumulative live self-play (runs 4-5), throughput parity proven (bf16 701 vs fp16
715 evals/s, live-seam A/B). Why the pin is in code, not YAML: `configs/training.yaml`'s root
`amp_dtype: "fp16"` default is inherited by every non-overriding variant — pinning in code means F9
cannot regress via a dropped/stale variant override (the F1/F5a declared-vs-inherited lesson).

### 9.3 Launch config — `run4_gnn.yaml` (committed `d4c620c`)

The launch artifact is **`configs/variants/run4_gnn.yaml`** (production) with **`run4_gnn_smoke.yaml`**
as the S7 gate vehicle (a full labeled duplicate; no variant-of-variant inheritance exists — parity is
test-pinned). Production pins, all verified resolving clean through the base+variant merge chain:

| Pin | Value | Why |
|---|---|---|
| `encoding` | `gnn_axis_v1` | only `representation="graph"` encoding (registry.toml). |
| `value_head_type` | `dist65` **explicit** | `configs/model.yaml` unconditionally merges `scalar` → the key is NEVER absent, so the representation-aware default never fires; omitting raises `RepresentationMismatch`. Declared (inert/correct — GnnNet ships only `GnnDist65ValueHead`). |
| `in_channels` | `0` **explicit** | base `in_channels: 8` disagrees with `gnn_axis_v1`'s `n_planes=0` → scattered-key consistency gate raises; graph path never reads it. |
| `mixing.buffer_persist_path` | `checkpoints/replay_buffer_run4_gnn.hexg` **namespaced** | §RUN3-STEP0 law — a shared un-namespaced path auto-restored a stale cross-lineage buffer in run3 (STEP0-FAIL). |
| `eval_pipeline.gating.best_model_path` | `checkpoints/best_model_run4_gnn.pt` **namespaced** | §RUN3-STEP0 / S7 F5a — the shared `checkpoints/best_model.pt` planted an anchor-resolve trap; per-lineage path (run3 template). |
| `mixing.pretrained_buffer_path` | `null` **explicit** | corpus-mix OFF (Decision 1). `null` (not `"<auto>"`) skips `expand_auto_paths` → `load_pretrained_buffer` no-ops; `"<auto>"` hard-fails on the unminted `gnn_axis_v1` sha pin (S7 F1). |
| `bot_batch_share` | `0` **explicit** | operator bot-mix retirement; F1 preserve-ckpt-baked (declare, don't inherit). |
| aux loss weights | `aux_opp_reply_weight / uncertainty_weight / ownership_weight / threat_weight / aux_chain_weight / ply_index_weight / entropy_reg_weight` all `0.0` **explicit** | GnnNet ships policy + dist65 only; nonzero inherited defaults trip `_train_on_graph_batch`'s `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` loud-raise at step 1. Guard constant is the enforcement. |
| `draw_value` / `ply_cap_value` | `-0.5` / `0.0` **explicit** | §178 outcome levers, values = current base defaults, pinned per F1 (run4 introduces no new outcome tuning). Feed `finalize_graph_outcome` INV26-verbatim. |
| `recency_weight` | `0.75` **real** | commit B landed the `HexgBuffer.sample_graph_batch(recent_frac=…)` sampler — commit-A's "declare 0 until the sampler lands" flag is CLEARED. |
| `selfplay.random_opening_plies` | `0` **valid** | WP-1 empty-board fix (`8dacf6f`) landed — organic ply-0 graph self-play starts; no longer forced to ≥1. |
| `selfplay_stall_timeout_sec` | `1800.0` | run3 watchdog rides verbatim (5080-sized; the 4060 smoke vehicle raises to 5400 — a labeled capacity override, OQ-2 rider). |
| `promotion_gate_subprocess_isolation` | `true` | run2 livelock ROOT fix, ON as run3 does. |
| `amp_dtype` | **undeclared** | graph path is bf16 in code (§9.2); a key here would be a harmless no-op — left undeclared so a reader doesn't mistake it for a tunable. |
| `gnn_warm_start` | `enabled: true`, `checkpoint: checkpoints/probes/gnn_bc/gnn_bc_040000.pt` | BC-prefit seam (§9.1). |

**As-built net:** the built `GnnNet` with the dist65 head = **286,082 params** (the BC prefit is 283,970;
the fresh dist65 head adds the delta). This is the §3/§4.2 "probe-scale ~284k" net as-shipped —
consistent, not a contradiction (the +414 was measured on the BC rep+policy; the value head is fresh
by construction). **`batch_size` stays inherited (256) in production, OPEN pending OQ-2** — the smoke
vehicle pins bs=16 (F6 capacity ladder 256→64→32→16, final rung sized against GENUINE bf16-game graphs
~1494 legal-nodes/graph; `inference_batch_size: 16`, `min_buffer_size: 64`, `buffer_capacity: 4096`,
`selfplay_stall_timeout_sec: 5400`, all labeled + allowlisted in the parity test). **Do NOT copy the
smoke capacity knobs back into production** — the 5080 rider owns the real capacity knee.

### 9.4 Eval instrument — graph EVALFAIR d5, r=5 book (F3 ruling), in-loop loud-skip

- **Book-radius ruling (S7 F3, pinned controller ruling, operator-overridable):** a whole-board graph
  checkpoint carries no `legal_move_radius_schedule`, so `radius_from_checkpoint` resolved `None` and
  the EVALFAIR d5 book selection raised. Ruling: a graph ckpt maps to the **standard EVALFAIR d5 book
  at r=5** (D-LADDER instrument convention — for a whole-board net the book radius is opening-diversity
  only, not a curriculum stage). Implemented with an explicit logged event
  (`graph_ckpt_evalfair_book_r5 radius=5 overridden=False`, fired at BOTH resolution sites) and a
  validated override knob **`--graph-eval-book-radius`** (threaded `mantis_pull_eval.py` CLI →
  `stage2_d5_eval` → `run_arm` as `graph_eval_book_radius_override`). Verified live: Part-2 ran
  end-to-end 4/4 pairs, 8/8 games vs SealBot d5, **eff_n honest raw=8 / deduped=8, `suffix_collisions=[]`**
  (the §D-ARGMAX dedupe instrument ran and measured distinct games, not assumed).
- **In-loop eval loud-skip semantics (F7):** the in-loop promotion-gate arena is dense-only; for a graph
  candidate it now emits ONE structured `eval_round_skipped_graph_representation` warning and sets
  `EvalRoundResult.eval_opponents_skipped = len(OPPONENTS)` (`evaluation_round_complete eval_games=0
  eval_opponents_skipped=8`) instead of N silent per-opponent crashes. **Consequence, binding:** in-loop
  eval rounds complete with 0 games for graph runs by design — the `eval_opponents_skipped` counter is
  the visibility hook. Promotion + true-north strength come from the OFFLINE EVALFAIR d5 battery
  (Part-2 path), not the in-loop round. This is the **OQ-8 rider**: the deploy-regime Gumbel promotion
  gate through a searched GNN (net-vs-net in the isolated subprocess, INV-R4-1 dependency) is NOT closed
  by the gate — the building blocks exist (offline searched-eval `infer_batch` graph branch landed, F8
  fixed) but the in-loop promotion arena is out of scope (`gnn_integration_scope.md` §C5, needs C3
  mixed-representation in-loop anchor). Kept OPEN (§9.6).

### 9.5 S7 gate record + fp16-artifact re-adjudication

**Gate: PASS** — Part-1 (production entrypoint, as-pinned `run4_gnn_smoke`, run 5): finite bf16 steps,
watchdog armed+live, ckpt@3 write + clean reload (gated loader, `gnn_axis_v1`/schema-1, allclose,
all finite), zero non-finite, zero OOM, clean single-PID shutdown, no env-var dependency. Part-2:
offline EVALFAIR d5 end-to-end, honest eff_n (run 4). Part-3: formal ragged/adversarial/parity suites
250/250 across 14 suites (run 4) + 13/13 launch-path re-run post-amendment (run 5).

**Re-adjudication of RECORD (load-bearing, accepted): fp16-era self-play game-length data was
ARTIFACTUAL.** The fast 26-29-ply games in gate runs 1-3 were fp16-NaN artifacts — saturated/NaN
values terminated games early (organic-draw class, `terminal_reason=3`), NOT genuine play. Under the
bf16 fix genuine games run toward the 150-ply cap (witnessed: 88/107/140/150-ply games at ~30+ min/wave
on the 4060). **Anything downstream that treated those fp16-era buffers / "N games" / eval-strength
reads as representative must be re-read against this** (`S7_f9_bf16_fix.md` "Disposition"). The
practical run4 consequences are the two OQ-2 riders below (memory + wall-time must be measured on
GENUINE bf16 data, not the fp16-artifact distribution every prior sizing used).

### 9.6 OPEN items (explicit — kept OPEN, none block finalize except OQ-2 which gates LAUNCH)

| OQ / item | Status | Blocking reason / what closes it |
|---|---|---|
| **Throughput floor** (§0-4 / §4.1) | **OPEN — LAUNCH gate** | 5080 rider (OQ-2). Absolute ms don't transfer from the 4060; re-derive per-leaf ms → re-check ≥1.0k steps/hr STEP-0 floor. Below floor = NO LAUNCH → BUILD-HOT first. |
| **OQ-2 — 5080 rider** | **OPEN — LAUNCH gate** | (a) rerun `gnn_infer_bench.py` on `wpa_positions.json` → floor; (b) **train-step memory envelope on GENUINE bf16 game data** (the bs=256 production knob is unmeasured for concurrent self-play + backward — F6 ladder shows genuine graphs ~1494 nodes/graph, 3× the fp16-artifact basis); (c) confirm **5080 first-wave wall < the production 1800s watchdog** (the 4060 needs 5400s). |
| **OQ-1 — DS-1/2/3 reference stamps** | **OPEN — monitor-arming** | stamp `run2_mw_fresh` entropy@50k / E-W axis-share / SealBot-WR trajectory + the prefit's frozen step-0 SealBot WR. Thresholds un-fireable until stamped (§1.3); does NOT block the INIT decision. |
| **OQ-8 — promotion gate through the GNN** | **OPEN** | deploy-regime Gumbel net-vs-net through a searched GNN in the isolated subprocess (INV-R4-1 dependency). In-loop eval loud-skips by design (§9.4 F7); offline EVALFAIR d5 is the working strength path. Needs C3 in-loop mixed-representation anchor. |
| **OQ-9 — strix-raw offline opponent** | **OPEN** | strix-raw delegates to strix's own engine/venv (`strix_g128_child.py`); confirm it runs as an EVALFAIR offline opponent for the INV-R4-2 external-ceiling read. Gate used SealBot d5, not strix-raw. |
| **value_spread_canary bare-`forward()`** | **OPEN — flagged, non-fatal** | `hexo_rl/monitoring/value_spread_canary.py` calls a bare dense tensor `forward()` (F2's bug CLASS, not `.in_channels`) → `value_spread_canary_failed` on every graph train step. Caught, warning-only. Future triage; does not block launch. |
| **Corpus games-manifest sha mint** | **OPEN — deferred (fallback path only)** | no canonical `gnn_corpus_v1.hexg` / games-manifest sha exists in-repo; the export takes the sha as a required CLI arg — **mint at FIRST corpus export**. Only needed if the option-B corpus-mix fallback fires (mixing-batch wiring itself stays deferred per `gnn_training_path_design.md` §7.2). |
| **Monitoring panel build-out** | **OPEN — deferred** | no NEW dashboard panel for v1 (the graph `train_step`/`training` events supply the 3 direct-indexed loss keys, existing panels render). Any graph-specific panel MUST land its producer field + extend the contract test FIRST (**producer-manifest law**). |
| OQ-3 unified manifest | **RESOLVED** | WP0.4 landed (`1d4a206`+`42e4e90`, `_CORPUS_SHA_PINS` resolver); graph corpus re-export ships in WP-5. |
| OQ-4 corpus re-export parity | **RESOLVED** | byte-parity oracle green in gate Part-3 (`test_gnn_hexg_corpus_export.py`, `test_hexo_graph_parity.py`). |
| OQ-5 loader landed-verify | **RESOLVED** | 46/46 tensors verified live (§9.1). |
| OQ-6 dist65 head geometry | **RESOLVED** | whole-board GNN → ONE pooled (65) → plain single-window CE, no K-cluster routing; dist65 warm-starts fresh over the prefit (E1 REVIVE). |
| OQ-7 smoke-gate PASS criteria | **RESOLVED** | gate defined + PASSED (5 runs). |

### 9.7 Handoff — the 5080 rider window (BEFORE launch)

The rider is the ONLY on-box GNN work before launch; it does not touch a live run. When run3-CNN stops
per its own rule and the box frees, the rider window MUST, in order:

1. **Throughput-floor freeze (OQ-2a).** Rerun `scripts/research/gnn_infer_bench.py` UNCHANGED on the
   frozen `reports/probes/gnn_integration/wpa_positions.json` on the 5080; re-derive per-leaf ms;
   confirm STEP-0 **≥ 1.0k steps/hr** (§4.1). Below floor → **NO LAUNCH → BUILD-HOT perf sub-package
   first.** Freeze the measured floor number into §0-row-4.
2. **OQ-2b train-step memory envelope on GENUINE bf16 data.** Measure the bs=256 production train step
   CONCURRENT with the live inference server on genuinely-played (bf16, ply-cap-deep ~1494-node) game
   graphs — NOT the fp16-artifact distribution. If bs=256 exceeds the 16 GiB 5080 envelope, pin the
   largest bf16-genuine-data batch that fits (the F6 ladder is the precedent) and record it.
3. **As-pinned bs=256 witness.** Run S7 Part-1 as-pinned (production `run4_gnn.yaml`, bs=256, watchdog
   1800s) on the 5080 — minutes there vs ~70 min on the 4060 — to witness the production capacity knob
   end-to-end and confirm **first-wave wall < 1800s** (OQ-2c). This doubles as the OQ-2 memory rider.
4. **Then, and only then**, arm the OQ-1 DS-1/2/3 monitor (stamp run2 reference values) and launch.
