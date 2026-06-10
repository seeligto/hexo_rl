<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §170 — close-out + Gate 6 operator surface — 2026-05-09

**Branch:** `encoding/gpool_bias_a1` (off `ec6e30b`, post-§169 A4 close-
out + §169a + §170 P0/P1).
**Predecessor:** §169 four-way ablation close-out (A1 anchor canonical;
A2/A3/A4 ablation arms ruled out for canonical pick).
**Successor (operator-decided):** §170 P4 spike (A1 + gpool-bias-policy-
only) and/or §171 (Phase D self-play smoke under canonical pick) — see
Gate 6 operator surface below.
**Status:** P0 + P1 + P3 complete. P2 (A4 MCTS-N curve) NOT RUN — see
§3 below. Awaiting operator go on Gate 6 items (a)–(f).

---

### TL;DR

- **Three discriminator probes ran. None promoted a new canonical;
  one large argmax breakthrough was falsified under MCTS.** A1 anchor
  (v6w25 K-cluster + min/max + no global routing) remains the canonical
  bootstrap path; all global-signal variants tested in §170 (additive
  bias + PMA-merged + scalar-ablation) failed to clear the §171
  sustained-run candidate gate (argmax > 16% AND MCTS > 27%).
- **§170 P0 (A4 scalar-ablation, ~10 s wall): SPATIAL_RICH.** Mean
  symmetric KL between A4 outputs at original Set R and a stones-zeroed
  copy = 4.19 nats; argmax STABLE 0/200 (every position's top-1 cell
  changes). A4's spatial pathway is alive *and load-bearing*. The
  SCALAR_DOMINATED hypothesis is closed; distribution-shift fine-tune
  (§171 candidate) is mechanistically justified.
- **§170 P1 (A3 MCTS-N curve, ~13 min wall): FLAT-NON-MONOTONIC.** A3
  argmax 7.5% → MCTS-{32, 64, 128} = 2.5% / 2.5% / 2.5% (identical
  W/L/D = 5/195/0 across all three depths; CIs identical to four
  decimals). Cliff at argmax→MCTS boundary is a binary switch, NOT
  monotone amplification. The PMA-value-head cliff is structural;
  more sims cannot rescue a broken value floor.
- **§170 P3 (A1 + gpool-bias retrain, 1h 48m pretrain + 28 min eval +
  14 min matched-baseline): FALSIFIED.** Argmax +7.5 pp (14.5% → 22.0%)
  was real and broke the breakthrough threshold (>20%); but the matched
  A1-anchor MCTS-64 baseline (run post-eval to discriminate) revealed
  A1 anchor MCTS-64 = 30.0% [24.1%, 36.7%] vs A1+gpool MCTS-64 = 15.0%
  [10.7%, 20.6%] — a **−15.0 pp regression with disjoint CIs**.
  Mean-ply collapses 33.8 → 29.7 under search. Final loss 2.8963 ≪
  anchor 3.57 confirms (again) **lower training loss does not imply
  better SealBot WR**.
- **Mechanistic synthesis.** P1 + P3 together identify the value
  head's *operating point* (not the routing or pool layer) as the
  controlling factor. PMA-merged routing (A3) corrupts value once
  during training by replacing the head; additive bias (A1+gpool)
  corrupts value once during training by shifting the head's hidden
  activation. Both produce the SAME argmax-up / MCTS-down failure mode
  with different magnitudes. The §170 P4 candidate (policy-only
  injection, value gate frozen at 0) is the surgical fix that follows.

---

### 1. Branch state

```
3f15a42 docs(sprint): §170 P3 FALSIFIED — A1 anchor MCTS-64 baseline reveals -15pp regression
219208b docs(sprint): §170 P3 RESULTS — A1 + gpool-bias breakthrough (+7.5 pp argmax)
bda723d docs(sprint): §170 P3 ENGINEERING-COMPLETE — A1 + gpool-bias retrain
898a1d3 eval(a1-gpool-bias): argmax + matched MCTS evaluation plumbing
b0f0259 feat(retrain): A1 + gpool-bias retrain config + script
641408b feat(dataset,pretrain): v6w25 corpus + 32x32 global crop column for gpool-bias path
cb61a78 feat(model): GpoolBias side-branch + gate scalar (gate=0 byte-exact A1)
3183f60 eval(a3): MCTS-N curve on existing checkpoint              ← §170 P1
3f54152 feat(probe): A4 scalar-ablation — discriminates spatial-feature richness  ← §170 P0
ec6e30b Merge branch 'encoding/four_way_ablation'                  ← §169 close-out base
d9b3c38 chore(169a): spatial deadness probe + gitignore worktrees  ← §169a (§170-pre)
```

Branch pushed to origin at `3f15a42`. This aggregation commit
(`docs(170): aggregation + Gate 6 operator surface`) lands on top of
`3f15a42`; no functional code changes.

---

### 2. Probe-by-probe summary

#### §170 P0 — A4 scalar-ablation (verdict: SPATIAL_RICH)

Question: is A4's residual policy variance scalar-driven or spatial-
driven? Pre-registered thresholds (locked before run): SCALAR_DOMINATED
(KL < 0.30 nats) vs SPATIAL_RICH (KL > 1.50 nats).

Result: **mean KL = 4.19 nats, argmax STABLE 0/200**. SPATIAL_RICH
fires by 2.8× margin over the threshold. Combined with §169a (E2 PASS,
spatial pathway alive at KL_S = 1.53 nats), this fully discriminates
A4's mechanism: spatial features are intact AND load-bearing AND
*decisive* on the top-1 move. The 0% SealBot WR is therefore a
distribution-shift problem, not an architecture problem.

**Implication for §170 scoping (closed in this aggregation).**
Distribution-shift fine-tune (§171 side-arm candidate) is
mechanistically justified. Architectural redesign of the bbox direction
is NOT warranted before §171 evidence. P0 verdict alone sets up
Gate 6 item (c).

Full entry: §170 P0 above (lines 9678–9736).

#### §170 P1 — A3 MCTS-N curve (verdict: FLAT-NON-MONOTONIC)

Question: does A3's argmax→MCTS-128 cliff (7.5% → 2.5%) reflect
**monotone compounding** of PMA-replaced value error across MCTS
backups, or a **binary switch** at the value-backup boundary? Pre-
registered: MONOTONIC-DECLINE (strict argmax > MCTS-32 > MCTS-64 >
MCTS-128 ordering, consecutive CI overlap ≤ 50%, Cochran-Armitage
p < 0.10) vs FLAT/NON-MONOTONIC.

Result: argmax 7.5% → MCTS-32 / MCTS-64 / MCTS-128 = **2.5% / 2.5% /
2.5%** (W/L/D = 5/195/0 IDENTICAL across all three). Consecutive CI
overlap 100% on all MCTS pairs. Cochran-Armitage p = 0.0277 (driven
entirely by argmax→MCTS split, not the trend across N).

**Verdict: FLAT-NON-MONOTONIC.** The compounding hypothesis is
refuted. The cliff is a binary switch already saturated at MCTS-32
(~3 backup levels for a typical median ply 23 game).

**Implication for §170 scoping (closed in this aggregation).** Any A3-
descended variant must fix the value head, not the policy head (which
already works under argmax). The "Add PMA side-channel to A1" framing
(Bet B / option α) is justified ONLY for the policy lift, NOT for
search-depth robustness. The natural minimal fix is policy-only
injection with the value path untouched (low-cost §170 option in §170
P1 §4) — which set up the §170 P3 hypothesis.

Full entry: §170 P1 above (lines 9738–9833).

#### §170 P3 — A1 + gpool-bias retrain (verdict: FALSIFIED)

Question: does an additive K-invariant gpool-bias side-branch lift
both argmax (predicted +2–4 pp) and MCTS (predicted +3–6 pp) over A1
anchor while keeping the load-bearing min/max pool byte-exact at
gate=0? Pre-registered hard-stops: final loss > 5.36, NaN-skip > 30%,
forward parity at gate=0 (architecture invariant).

Result:
- **Final loss 2.8963** (BETTER than A1 anchor 3.57; well below 5.36
  hard-stop). Gate trajectory: 0.000 → 0.038 → **0.0512** (~3× growth
  from absolute zero, just above the 0.05 soft-warn null threshold).
  0% NaN-skips. Forward parity unit test green at gate=0.
- **Argmax @ r=8 n=200: 22.0% [16.8%, 28.2%]** (44 W / 154 L / 2 D,
  mean_ply 47.98). +7.5 pp over A1 anchor 14.5% — broke the >20%
  surface-for-§171 threshold *initially*.
- **MCTS-64 n=200: 15.0% [10.7%, 20.6%]** (30 W / 170 L / 0 D,
  mean_ply 29.7). Matched A1-anchor MCTS-64 baseline run post-eval
  on 5080 (839 s, 60 W / 140 L / 0 D = 30.0% [24.1%, 36.7%], mean_ply
  33.8). **A1+gpool regresses −15.0 pp under MCTS with CIs disjoint
  by 3.5 pp**. The +7.5 pp argmax lift does NOT transfer through
  PUCT search.

**Verdict: FALSIFIED.** Argmax-up / MCTS-down signature reproduces
A2 PMA (4.5%/3.5%) and A3 PMA+global (7.5%/2.5%) at higher absolute
WR but the same shape. The "additive bias preserves load-bearing
pool" invariant holds only structurally at gate=0 — once the gate
gains weight, the value head's hidden activation distribution shifts
vs the gate=0 baseline, and PUCT accumulates the drift across
simulations.

**Implication.** §170 P3 lesson generalises §170 P1's: do NOT modify
the value head's operating point AT ALL — including additive bias.
The §170 P4 candidate (policy-only injection with value gate frozen
at 0) is the surgical follow-up that tests whether limiting injection
to the policy head escapes this trap. Predicted: argmax preserves
~22%, MCTS-64 approaches A1 anchor ~30% (best-of-both).

Full entry: §170 P3 above (lines 9835–10115).

#### §170 P2 — A4 MCTS-N curve at matched perception (NOT RUN)

The §170 P2 slot in the original probe chain (A4 alive vs dead under
MCTS at matched perception) was **not run** in this sprint. Rationale
(folded into the P0 verdict + §169 P4 result + §169a probe):

- §169 P4 already measured A4 MCTS-128 = 0.0% [0.0%, 1.9%] (matching
  argmax-only 0%).
- §169a (E2 PASS): spatial pathway alive at KL_S = 1.53 nats with
  133/200 distinct argmax cells under spatial-only variation.
- §170 P0 (SPATIAL_RICH): spatial features load-bearing, NOT scalar-
  dominated under realistic position diversity (KL = 4.19 nats,
  argmax-stable 0/200 under stones-zeroed).

These three observations together triangulate A4's mechanism without
the curve: spatial features are intact, alive, and load-bearing on
the policy top-1 — but BOTH argmax and MCTS-128 are at the noise
floor (0.0%). MCTS depth cannot rescue a model whose underlying
policy + value at N=1 is already at the floor; there is no signal to
amplify. An A4 MCTS-N curve would therefore produce a flat 0% / 0% /
0% line — adding a third 5080 run to confirm the floor with no new
information.

The mechanistic story stays: A4's failure is corpus-overfitted spatial
representations going OOD against SealBot's adversarial style. The
right next step is **distribution-shift fine-tune (§171 side-arm
candidate)**, not architectural redesign — see Gate 6 item (c).

If the operator wants the formal A4 MCTS-N curve completed for
audit-trail completeness, it is a ~1 hr 5080 run (200 games × 3
depths). The recommendation here is to skip and queue it as a Phase
5+ artefact only if needed for paper / external review.

---

### 3. Comparison matrix (compact)

Full matrix with CIs in `reports/gpool_bias/SUMMARY.md` §1.

| arm                  | encoder        | pool                  | global routing       | argmax        | MCTS-N        | params  |
|----------------------|----------------|-----------------------|----------------------|--------------:|--------------:|--------:|
| **A1 anchor**        | v6w25 K-cluster| min/max               | none                 | **14.5%**     | 25% MCTS-32 (n=20); **30% MCTS-64** (n=200) | 5.29 M |
| A2                   | v6w25 K-cluster| PMA                   | none                 | 4.5%          | 3.5% MCTS-128 | 6.30 M |
| A3                   | v6w25 K-cluster| PMA                   | yes (PMA-merged)     | 7.5%          | 2.5% MCTS-32 / 64 / 128 (FLAT) | 6.37 M |
| A4                   | v8 + canvas    | bbox + PartialConv2d  | n/a (canvas mask)    | 0.0%          | 0.0% MCTS-128 | 3.85 M |
| **A1 + gpool-bias**  | v6w25 K-cluster| min/max + gpool-bias  | yes (additive bias)  | **22.0%**     | **15.0% MCTS-64** | 5.47 M |

Decomposition reading + canonical-pick rationale: see
`reports/gpool_bias/SUMMARY.md` §2–§3.

---

### 4. Gate 6 — operator decision packet

a) **Promote A1+gpool-bias to `bootstrap_model.pt`?**
   **Recommendation: NO, do not promote.** The promotion gate is
   `argmax > 16% AND MCTS > 27%`. A1+gpool argmax = 22.0% (PASS) but
   MCTS-64 = 15.0% (FAIL by 12 pp). Promotion would regress sustained-
   run performance by ~15 pp under PUCT. A1 anchor (v6w25,
   `bootstrap_model_v6w25.pt`, sha256 `571a82f8…`) remains canonical.

b) **Open §171 (Phase D self-play smoke) under canonical pick? Which?**
   **Recommendation: open §171 under A1 anchor (v6w25 K-cluster +
   min/max).** A1 anchor MCTS-64 = 30.0% is the only §169 / §170 arm
   above the §171 sustained-run candidate gate. Two side-arm scope
   decisions are surfaced separately as items (c) and (d) and could
   be folded into the §171 plan or run in parallel.

c) **§171 distribution-shift fine-tune for A4 — open scope or defer?**
   **Recommendation: open as a §171 SIDE-ARM (not the canonical), or
   defer until §170 P4 verdict lands.** §170 P0 verdict SPATIAL_RICH
   means the bbox failure mode is mechanistically tractable: corpus-
   overfitted spatial representations going OOD on SealBot. Augmenting
   with adversarial SealBot-style positions and fine-tuning A4 is
   cheap (~3–4 hr on 5080) and tests the bbox direction without
   architectural redesign. Risk: opening §171 with two parallel
   side-arms (A1 canonical + A1+gpool-policy-only + A4 distribution-
   shift) may dilute operator attention. Operator's call.

d) **Hybrid scope — §170 P4 spike or §172 spike?**
   **Recommendation: open §170 P4 spike (A1 + gpool-bias-policy-
   only) BEFORE §171.** This is the single most informative next
   experiment given the §170 P1 + P3 mechanistic synthesis (value-
   head operating-point drift is the controlling factor). One config
   change (`value_gate_active=False`) + one retrain + one eval ≈
   2h 16m on 5080. If verdict matches prediction (argmax ~22%, MCTS
   ~30%), promote to canonical for §171. If verdict reproduces the
   MCTS regression, the §170 P3 lesson generalises further to
   "shared-trunk gradient flow makes the value head sensitive to ANY
   side-branch training" — a Phase 5+ research question, not a §171
   blocker. Other hybrid scopes (Set-Transformer light-touch,
   PMA-policy + min-value A2′) are alternative routings of the same
   policy-only-side-channel idea; defer to Phase 5+ unless the
   gpool-bias-policy-only spike fails.

e) **v8 bench-gate finalize — replace v6 numbers with current
   canonical?**
   **Recommendation: DEFER.** v8's current canonical (B1 bbox +
   canvas_realness + PartialConv2d) is at 0% SealBot WR; replacing
   v6 bench numbers with a 0%-WR canonical premature. v6w25 / A1
   anchor bench numbers remain authoritative for Phase D until §171
   distribution-shift verdict (or later) clears bbox direction. Track
   as a deferred follow-up; revisit at §171 close-out.

f) **§169 + §170 architecture explorations to defer to Phase 5+?**
   **Recommendation: defer the following list, queue for Phase 5+
   intake review:**
   - Set-Transformer light-touch on policy heads (alternative routing
     to gpool-bias-policy-only).
   - PMA-policy + min-value hybrid (A2′ — §169 P3 close-out option).
   - bbox-direction redesign (K=1 vs K>1 corpus supervision, bbox-
     centroid frame instability, single-window inference-time
     blindness — §169 P4 close-out items).
   - Threat-probe v6w25 fixture build (curated tactical positions on
     a 25×25 board + regenerated baseline). Currently SKIPPED across
     A1 / A2 / A3 / A1+gpool — operator-side curation work; blocks
     C1/C2/C3 column for every v6w25 arm. Track as a §170 hangover.
   - Formal A4 MCTS-N curve (skipped per §2 §170 P2 above) — only
     resurrect for paper / external-review audit trail.

---

### 5. Outstanding for operator at Gate 6

- [ ] Decide (a)–(f). Items (a), (b), (e), (f) recommended verdicts
      above are low-controversy; items (c) and (d) are scope decisions
      that affect the next 2–6 hours of 5080 compute.
- [ ] If (d) is opened: scope §170 P4 (A1 + gpool-bias-policy-only)
      in a separate context. Architecture invariant (gate=0 byte-exact
      A1) already holds; only `value_gate_active=False` knob needs
      exposure on `GpoolBiasBranch`.
- [ ] If (c) is opened standalone or alongside (d): scope §171 A4
      distribution-shift fine-tune in a separate context.
- [ ] Either way, scope §171 (Phase D self-play smoke) under A1 anchor
      canonical pick after §170 P4 spike resolves.

---

### 6. Surface-immediately tracking

None fired during execution. All monitors clear at close:
- §170 P0 + P1 + P3 hard-stops: all PASS or N/A (loss < 5.36 hard-
  stop; NaN-skip 0%; forward parity green).
- §170 P3 surface-for-§171 threshold (argmax > 20%): triggered, then
  cancelled by matched A1-anchor MCTS-64 baseline (30.0% disjoint
  CI vs 15.0%).
- A1 anchor matched-baseline run: completed and archived
  (`reports/gpool_bias/A1_anchor_mcts64.json`).

---

### 7. STOP — awaiting operator go on Gate 6

Branch `encoding/gpool_bias_a1` ready at `3f15a42` + this aggregation
commit. §170 close-out merges into master AFTER operator decision on
items (a)–(f). No checkpoint promotion taken; no gating yaml changed;
no §171 scope opened.

**Key result: A1+gpool-bias falsified at the value-head bias injection
site. A1 anchor remains canonical. §170 P4 (policy-only) is the
recommended next surgical experiment.**

**Next:** §170 P4 spike (operator-decided, item d) → §171 (Phase D
self-play smoke under canonical pick, operator-decided, item b).



---


## §170 P4 P1 — A1 + gpool-bias-policy-only retrain — ENGINEERING COMPLETE — 2026-05-09

**Sprint:** §170 P4 P1 — A1 + gpool-bias-policy-only spike (Gate 6
item d, operator-opened).
**Branch:** `encoding/gpool_bias_a1`.
**Predecessor:** §170 P3 FALSIFIED (`reports/gpool_bias/SUMMARY.md`).
§170 P4 P0 architecture commit `c399a91`.
**Verdict:** **NULL** (pre-registered).
**Key result:** the §170 P3 +7.5 pp argmax lift was bought ENTIRELY
by value-head bias drift. Freezing the value head structurally
(policy-only routing) erases the argmax lift AND the −15 pp MCTS
regression, returning the model to ~A1-anchor parity on every axis.

### 1. Question + pre-registered criteria

Question: does limiting the gpool-bias side-branch's gradient flow
to the policy head (value path frozen at A1 by construction) lift
argmax (predicted preserve ~22 % from §170 P3) while restoring MCTS
to A1 anchor (~30 %)?

Pre-registered criteria (LOCKED at sprint header):
- **WIN**: argmax > 16 % (Wilson 95 % LB > 12 %) AND MCTS-64 > 27 %
  (Wilson 95 % LB > 24 %, non-disjoint with A1 anchor 30 % CI).
- **PARTIAL-WIN**: argmax > 16 % AND MCTS-64 in [22 %, 27 %].
- **NULL**: argmax in [12 %, 16 %] AND MCTS-64 in [22 %, 32 %].
- **LOSS**: any axis disjoint-below A1 anchor CI (MCTS-64 < 24 % UB
  — same fingerprint as P3, value-path freezing failed).

Hard-stops: final loss > 5.36, NaN-skip > 30 %, policy_gate stuck
< 0.05 (soft-warn null), forward parity at gate=0 invariant violated.

### 2. Pretrain results

Recipe identical to §170 P3 (30 ep cosine, peak 2e-3, eta_min 5e-5,
batch 256, fp16) + `--policy-only-bias` flag from §170 P4 P0.
Corpus: `bootstrap_corpus_v6w25_with_global.npz` (sha256 `e2876ae5…`)
reused verbatim. Wall: 1 h 48 m on 5080 (one mid-run SSH drop on the
primary `REMOTE_HOST:REMOTE_PORT` endpoint required relaunch under tmux
detached on the `38118 → REMOTE_IP` alternate; final run
contained the full 30 epochs in one process).

| metric                  | value                                |
|-------------------------|--------------------------------------|
| final loss              | **3.1945** (epoch 30/30)             |
| final policy loss       | 2.3276                               |
| final value loss        | 0.5015                               |
| final aux opp-reply loss| 2.4235                               |
| final chain loss        | 0.0018                               |
| **final gpool_bias_gate** | **0.0718** (peak 0.0775)           |
| NaN-skip rate           | **0 %**                              |
| forward parity gate=0   | PASS (P0 unit test green pre-train)  |

Hard-stops: all PASS (loss 3.1945 << 5.36; NaN 0 %; gate 0.0718 >>
0.05 soft-warn null). Soft-warn: gate trajectory smooth, no
oscillation, ~3 × growth from absolute zero — ~40 % HIGHER than
§170 P3's 0.0512 final, indicating the policy-only path puts MORE
weight on the global signal because it's the only path where the
signal can earn weight at all.

### 3. Post-train value-path-frozen verification

Asserted on the trained checkpoint:

| check | result |
|-------|--------|
| state_dict has `gpool_bias_branch.gate` (single shared gate)        | PASS |
| state_dict has NO `value_gate` key (structurally absent)            | PASS |
| value/value_logit forward INVARIANT under gate change at inference  | PASS (max\|Δ\| = 0) |
| log_policy forward DEPENDS on gate at inference                     | PASS (gate drives policy bias) |
| forward parity at gate=0 vs A1 anchor (atol 1e-6)                   | PASS |
| `value_proj.weight` survived training at random-init footprint       | PASS (\|max\| = 0.0884 matches Kaiming-uniform init bound √(1/128) ≈ 0.0884; std = 0.0510) |
| `policy_proj.weight` was actively trained                            | PASS (\|max\| = 1.0481, ~12 × init bound — gradient flow drove the parameter away from init) |

The structural proof: comparing the trained checkpoint's
`gpool_bias_branch.value_proj.weight` against the constructor random-
init footprint shows zero deviation at maximum-magnitude scale (0.088
≈ 0.0884), confirming no gradient ever flowed through `value_proj`
across the 30-epoch run. By contrast `policy_proj.weight` shows ~12 ×
deviation from init, confirming the policy-bias path was actively
trained.

### 4. Eval matrix (re-run with `--policy-only-bias` forced at inference)

A subtle issue surfaced after the first eval: the inference checkpoint's
state-dict shape is identical between P3 (bilateral) and P4 P1
(policy-only), so `load_model_with_encoding` cannot auto-detect the
flag. The first eval run loaded the trained ckpt with
`policy_only_bias=False` at inference, causing `value_proj` (at
random-init weights) to inject ~5 % noise into the value-head hidden
activation. **argmax was bit-exact between modes** (max\|Δ\| log_policy
= 0; the policy head sees only `policy_bias`); **MCTS shifted by < 1
pp** at MCTS-64 (33.0 % bilateral → 32.5 % policy-only). The final
results below are from the corrected re-run with
`scripts/run_sealbot_eval.py --policy-only-bias` forced at inference;
the bilateral-load JSONs are preserved as `*_bilateral_load.json`
under `reports/gpool_bias/policy_only/` for reference.

vs SealBot, n=200, seed_base=42, legal_radius=8,
random_opening_plies=4, c_puct=1.5, time_limit=0.5 s:

| inference | WR        | 95 % CI         | W / L / D    | mean_ply |
|-----------|----------:|-----------------|--------------|---------:|
| argmax    | **15.0 %** | [10.7 %, 20.6 %] | 30 / 165 / 5 | 52.73    |
| MCTS-32   | 24.5 %    | [19.1 %, 30.9 %] | 49 / 151 / 0 | 35.57    |
| MCTS-64   | **32.5 %** | [26.4 %, 39.3 %] | 65 / 135 / 0 | 37.36    |
| MCTS-128  | **39.5 %** | [33.0 %, 46.4 %] | 79 / 121 / 0 | 38.93    |

Threat probe SKIPPED (no v6w25 fixture; same status as A2 / A3 /
§170 P3) — operator-side curation work, tracked as the §170
threat-fixture follow-up.

NN latency bench (5080 host = 79f24b481d6b, n=5 each):

| arm                          | params | b=1 ms | b=64 ms |
|------------------------------|-------:|-------:|--------:|
| A1 + gpool-bias (P3, full)   | 5.47 M | 1.49   | 11.26   |
| A1 + gpool-bias-policy-only  | 5.47 M | 1.47   | 11.26   |

Inference latency parity confirmed: `policy_only_bias` is a forward-
routing flag only (skips one Linear under True), not a parameter
delta — state-dict shape unchanged, latency unchanged.

### 5. Comparison vs A1 anchor + §170 P3

| metric         | A1 anchor              | §170 P3 (bilateral) | §170 P4 P1 (policy-only) |
|----------------|-----------------------:|--------------------:|-------------------------:|
| argmax WR      | 14.5 %                 | 22.0 %              | **15.0 %**               |
| MCTS-32 WR     | 25 % (n=20)            | not measured        | 24.5 % (n=200)           |
| MCTS-64 WR     | 30.0 %                 | 15.0 %              | **32.5 %**               |
| MCTS-128 WR    | 32.5 % (n=200, +matrix-completion run) | not measured | **39.5 %** (n=200)       |
| MCTS curve     | shallow monotonic (+2.5 pp 64→128) | flat-collapsed | **steep monotonic** (+7.0 pp 64→128) |
| final loss     | 3.57                   | 2.8963              | 3.1945                   |
| final gate     | n/a                    | 0.0512              | 0.0718                   |
| latency b=64   | 10.41 ms               | 11.26 ms            | 11.26 ms                 |

CIs at MCTS-64 — A1 anchor [24.1 %, 36.7 %] vs P4 P1 [26.4 %, 39.3 %]:
overlap of 10.3 pp, point estimates differ by 2.5 pp (P4 P1 > anchor).
Statistical: not distinguishable.

### 6. Pre-registered verdict — NULL

| criterion    | rule                                                            | result |
|--------------|-----------------------------------------------------------------|--------|
| WIN          | argmax > 16 % AND MCTS-64 > 27 %                                | FAIL (argmax = 15.0 %) |
| PARTIAL-WIN  | argmax > 16 % AND MCTS-64 in [22 %, 27 %]                       | FAIL |
| **NULL**     | **argmax in [12 %, 16 %] AND MCTS-64 in [22 %, 32 %]**          | **PASS** |
| LOSS         | MCTS-64 < 24 % UB (disjoint-below A1 anchor CI)                 | FAIL (no LOSS — UB 39.3 %, LB 26.4 %) |

argmax 15.0 % lands cleanly in the NULL band [12 %, 16 %]. MCTS-64
32.5 % point estimate sits 0.5 pp above the strict NULL upper bound
of 32 %, but the CI [26.4 %, 39.3 %] overlaps the A1 anchor CI
[24.1 %, 36.7 %] by 10.3 pp — no statistical distinction. **Verdict
is NULL on argmax with mild positive-leaning point estimate on
MCTS, neither rising to PARTIAL-WIN nor falling to LOSS.**

### 7. Mechanistic reading

§170 P1 (A3 PMA-merged value MCTS-flat at 2.5 %), §170 P3 (A1 + gpool-
bias bilateral argmax-up / MCTS-down), and §170 P4 P1 (A1 + gpool-
bias-policy-only argmax-flat / MCTS-flat) combine into a single
mechanistic story:

1. **Value-head operating-point sensitivity is the controlling factor
   under PUCT.** Any architectural change that lets the value head's
   hidden activation drift during training breaks MCTS by the
   "argmax-up / MCTS-down" fingerprint: argmax escapes via the policy
   head; MCTS amplifies the value drift through PUCT backups.
   Routing matters only in so far as it touches the value head.

2. **The §170 P3 +7.5 pp argmax lift was ENTIRELY bought by value-
   head bias drift, not by the policy-side global signal.** P4 P1
   freezes the value head; the argmax lift evaporates back to A1
   anchor (15.0 % vs 14.5 %, indistinguishable). The gpool-bias
   signal, when delivered ONLY through the policy head, does NOT
   produce measurable improvement on argmax against SealBot.

3. **MCTS depth-scaling is the cleanest signal of value-head
   integrity.** A healthy value head produces monotonic-increasing
   MCTS curves (P4 P1: 24.5 % → 32.5 % → 39.5 %, the standard PUCT-
   search-deepens-improves pattern). A drifted value head produces
   flat curves (§170 P1 A3: 2.5 % / 2.5 % / 2.5 %) or collapsed
   ones (§170 P3 P1: MCTS-64 15.0 %).

4. **Routing is necessary but insufficient for lift on this task.**
   Bilateral bias (P3) and policy-only bias (P4 P1) preserved or
   restored MCTS integrity to different degrees; only bilateral
   lifted argmax (via value-drift artifact); neither routing
   produced statistically distinguishable MCTS lift over A1 anchor.

5. **Implication for the v6w25 corpus + global-crop column.** The
   global-crop column on the corpus does not provide actionable
   signal beyond what the K-cluster + min/max trunk already extracts
   on this distribution. The "policy-side global signal" hypothesis
   is FALSIFIED for SealBot adversarial play on v6w25. Whether the
   global signal would matter on a different distribution (e.g. self-
   play or distribution-shift fine-tune) is open and best deferred
   to Phase 5+.

### 8. Implication for §170 close-out + §171

- **A1 anchor (`bootstrap_model_v6w25.pt`, sha256 `571a82f8…`)
  remains the canonical bootstrap for §171.** No §169 / §170
  architectural variant lifts argmax + MCTS jointly above anchor on
  SealBot adversarial play.
- **Do not promote A1 + gpool-bias-policy-only to `bootstrap_model.pt`.**
  Argmax matches anchor; MCTS-64 statistically indistinguishable;
  no benefit measured.
- **§170 close-out tree resolves cleanly:**
  - Item (a) Promotion: NO (no candidate clears the joint gate).
  - Item (b) §171 canonical pick: A1 anchor (unchanged).
  - Item (d) §170 P4 spike: NULL — closes the gpool-bias-policy-only
    research line. Gpool-bias as a side-channel adds no measurable
    benefit on this distribution under either bilateral or policy-
    only routing.
  - Items (c), (e), (f) unchanged from §170 aggregation
    recommendation.
- **§171 (Phase D self-play smoke under A1 anchor)** is the next
  priority; §170 architecture side-arm queue is now empty.

### 9. Outstanding for operator at §170 P4 P1 close

- [ ] Open §171 scope (Phase D self-play smoke under A1 anchor).
- [ ] Defer gpool-bias / Set-Transformer / PMA-policy-only / bbox-
      direction redesign to Phase 5+ intake review.
- [ ] Threat-probe v6w25 fixture build remains an open §170 follow-
      up (curated tactical positions on a 25 × 25 board + regenerated
      baseline); not a §171 blocker.

### 10. Surface-immediately tracking

None fired. All monitors clear at close:
- Hard-stops (loss < 5.36, NaN < 30 %, gate trajectory healthy): all
  PASS.
- Forward parity gate=0 invariant: PASS (commit-1 unit test green).
- Value-path-frozen post-train invariant (value_proj at random-init
  footprint): PASS (max\|Δ\| 0.088 ≈ Kaiming bound).
- argmax bit-exactness between bilateral-load and policy-only-load
  inference modes: confirmed (max\|Δ\| log_policy = 0); MCTS shifts
  < 1 pp under inference-mode swap; final results use the corrected
  policy-only-load run.

### 11. Test gate

`make test` (local pre-flight): 1168 / 8 / 2 (matches §170 P4 P0
landing — no new tests added in P1; the P0 unit-test bundle gates
the architecture invariant).

### 12. STOP — awaiting operator go on §170 close-out + §171 scope

Branch `encoding/gpool_bias_a1` ready at the §170 P4 P1 commit
bundle:
- `c399a91` feat(model): A1+gpool-bias-policy-only architecture (P0).
- `f2fec2f` feat(retrain): A1+gpool-bias-policy-only retrain config + script.
- `<this commit>` eval(a1-gpool-bias-policy-only): argmax + MCTS-{32, 64, 128}.

§170 close-out merges into master AFTER operator confirms (a) NULL
verdict acceptance + (b) §171 scoping under A1 anchor. No checkpoint
promotion taken; no gating yaml changed; no §171 scope opened in
this sprint.

**Key result: §170 P4 P1 NULL closes the gpool-bias side-channel
line. A1 anchor + min/max pool, no global routing, remains the
canonical v6w25 bootstrap for §171 self-play.**

**Next:** §171 Phase D self-play smoke under A1 anchor (operator-
decided, items (b) + (a) unchanged from §170 aggregation Gate 6).



---


## §170 P4 P2 — adversarial corpus prep for §171 A4 fine-tune — 2026-05-09

**Sprint:** §170 P4 P2 — corpus preparation only, NO retrain.
**Branch:** `encoding/gpool_bias_a1`.
**Predecessor:** §170 P0 verdict SPATIAL_RICH
(`reports/investigations/a4_scalar_ablation_20260508/VERDICT.md`); §170
aggregation Gate 6 item (c) "open §171 distribution-shift fine-tune as a
side-arm" deferred-to-§171-scope.
**Status:** ENGINEERING-COMPLETE. Artefacts on 5080 + laptop. Awaiting
operator go on §171 fine-tune scope (Gate 6 item c).

### 1. Question + scope

§170 P0 established A4's spatial pathway is *load-bearing* (mean KL =
4.19 nats under stones-zeroing, argmax STABLE 0/200). The 0% SealBot WR
is therefore a **distribution-shift problem, not an architecture
problem**. The §171 distribution-shift fine-tune is mechanistically
justified — but §170 P4 P2 generates the data only; the fine-tune is
deferred to §171 to keep the engineering scope tight.

The corpus must mix sources biased toward exactly the positions A4
collapses on (SealBot adversarial play + asymmetric perception fights),
without overfitting to SealBot self-play (since SealBot is the eval
anchor).

### 2. Source mix (operator-tuned weights)

| source | weight | rationale |
|---|---:|---|
| `sealbot_vs_a1` | 0.45 | **PRIMARY.** SealBot vs A1 (v6w25) games — the exact distribution A4 must learn for §171. |
| `scripted_far_line` | 0.13 | §164 P2 catastrophic asymmetric-perception adversary; far-axis stone trains exactly the OOD distribution. |
| `scripted_far_placement` | 0.12 | §164 P2 weaker variant; positional-coverage diversity. |
| `krakenbot_vs_sealbot` | 0.15 | Peer-project minimax style; not SealBot's flat-board pattern matching. |
| `sealbot_vs_sealbot` | 0.15 | LOW WEIGHT — SealBot is the eval anchor; self-play overfits its style. Retained for typical-position coverage. |

### 3. Generator script

`scripts/generate_adversarial_corpus.py` (new, ~600 LoC + inline copies of
`FarLineOpponent` / `FarPlacementOpponent` from `tests/probes/p2_far_placement_opponent.py`
since `tests/probes/` lacks an `__init__.py`).

- BotProtocol-driven game loop (mirror of `scripts/run_sealbot_eval.py:play_game`
  but accepts arbitrary (bot_p1, bot_pm1) pairs and returns the move list).
- Per-source seed offsets reproducible from `--seed` + source-name hash.
- Per-game cluster_threshold / cluster_window_size set only for the
  sealbot_vs_a1 source (A1 v6w25 needs widened cluster geometry; v8 / scripted
  / KrakenBot don't).
- A1 model loaded once via `load_model_with_encoding`, shared across all
  sealbot_vs_a1 games.
- Per-game replay → `replay_game_to_triples_v8(canvas_realness=True)` for v8
  encoding; ply-range filter `[2, 150)`; per-game uniform subsample to
  ≤ 25 positions; per-source down-sample to weight target.
- Output NPZ schema **byte-compatible** with
  `data/bootstrap_corpus_v8_canvas_realness.npz` (states / policies /
  outcomes / weights identical in shape + dtype). One extra column
  `source_labels` (fixed-width bytes) for diagnostics; pretrain loader
  ignores it.
- JSON sidecar (`reports/gpool_bias/adversarial_stats.json`) captures full
  per-source counts, win splits, mean ply, opponent strength bands, sha256,
  bbox-clip telemetry.

KrakenBot wrapper has a known **pair-bug** under tight time budgets where
`MinimaxBot.get_move` returns the same cell twice as a 2-move pair; the
wrapper's pre-cache validity check (`rust_board.get(q2, r2) == 0` _before_
move 1 plays) doesn't catch the move1==move2 case. The generator catches
the resulting `apply_move` exception per-game and skips, with effective
KrakenBot game-yield ≈ 33 % (39/117). Surfaced in the manifest. Out of
scope to fix in this sprint; the position shortfall (~1.3 k under
target) was absorbed by the 1.3× game-count buffer + other sources.

### 4. Run summary (5080 vast.ai, 2026-05-09)

| field | value |
|---|---|
| host | 5080 vast.ai (`REMOTE_HOST:REMOTE_PORT`) |
| seed | 20260509 |
| wall time | ~14 min (459 s sealbot_vs_a1 + 39 s far_line + 25 s far_placement + 214 s krakenbot + 147 s sealbot self-play) |
| games attempted / kept | 781 / 655 |
| **total positions** | **12,781** |
| target positions | 15,000 (12,781 / 15,000 ≈ 85 %; in 10–20 k operator band) |
| size | 198.2 MB uncompressed (mmap-ready) |
| **sha256** | **`e6c1b9b921492d9b23f825cce26e99b818285743fffef8aec3ae47532ef84c2c`** |
| canvas_realness | True (plane 8 inside-mean = 217.0 = R=8 hex cell count, identical to base bootstrap_corpus_v8_canvas_realness.npz, sha `110ea6b2…`) |
| bbox-clip telemetry | 26,674 stones clipped outside 25×25 envelope (informational; matches B1 / A4 base-corpus rate) |

Per-source position counts (post-target-down-sample):

| source | positions | weight (target) | wins P1 / P-1 | mean ply | strength band |
|---|---:|---:|---|---:|---|
| sealbot_vs_a1 | 6,750 | 0.45 (6,750) | 177 / 170 | 59.4 | SealBot (t=0.1 s) ↔ A1 v6w25 argmax |
| scripted_far_line | 1,860 | 0.13 (1,950) | 90 / 0 | 27.4 | FarLineOpponent vs SealBot |
| scripted_far_placement | 958 | 0.12 (1,800) | 62 / 0 | 17.6 | FarPlacementOpponent vs SealBot |
| krakenbot_vs_sealbot | 963 | 0.15 (2,250) | 11 / 28 | 32.0 | KrakenBot (t=0.1 s) vs SealBot |
| sealbot_vs_sealbot | 2,250 | 0.15 (2,250) | 55 / 62 | 39.0 | SealBot self-play |

Scripted sources show 100 % wins-by-SealBot (P1) — expected: §164 P2's
asymmetric-perception adversary is far weaker than SealBot, the
scripted side just spams far-axis stones. The corpus value is in the
*positions encountered* (SealBot's responses to OOD adversarial plays),
not the game outcomes; the outcomes column still uses
`replay_game_to_triples_v8`'s ±1-from-current-player POV computation,
so SealBot's policy targets are correct.

### 5. Schema parity (vs `data/bootstrap_corpus_v8_canvas_realness.npz`)

| key | adversarial | bootstrap (canvas_realness variant) | match |
|---|---|---|---|
| states | (12 781, 11, 25, 25) float16 | (347 142, 11, 25, 25) float16 | ✓ |
| policies | (12 781, 625) float32 | (347 142, 625) float32 | ✓ |
| outcomes | (12 781,) float32 | (347 142,) float32 | ✓ |
| weights | (12 781,) float32 | (347 142,) float32 | ✓ |
| canvas_realness | True (plane 8 inside-mean 217.0) | True (plane 8 inside-mean 217.0) | ✓ |
| extras | + `source_labels` (diagnostic; ignored by pretrain) | n/a | benign |

Verified at the laptop after rsync via
`np.load(path, mmap_mode='r')[k]` for k ∈ states / policies / outcomes /
weights — all four key indexings succeed without `allow_pickle`. The
extra `source_labels` column is currently `dtype=object` (the
generator was patched mid-sprint to emit `S40` fixed-width bytes for
future runs); object-dtype indexing is gated to diagnostic callers
that opt into `allow_pickle=True`, and the pretrain loader doesn't
touch this column.

### 6. §171 entry-point

```
data/bootstrap_corpus_v8_canvas_realness.npz    sha256 110ea6b2…  (347,142 pos, 5,382 MB)
data/adversarial_corpus_v8.npz                  sha256 e6c1b9b9…  ( 12,781 pos,   198 MB)
checkpoints/ablation_169/A4_canvas_realness.pt  21.9 MB; v8; canvas_realness; PartialConv2d trunk-entry; 3.85 M params
```

Mixing ratio (operator-tunable for §171; NOT committed in this sprint):
adversarial ≈ 3.7 % of base by position count — recommended natural-
ratio mix for fine-tune (~2–5 k steps, peak LR ≤ 5 e-5, freeze
PartialConv2d trunk-entry, unfreeze res_blocks 8–11 + heads only).

Full source manifest:
`reports/gpool_bias/adversarial_manifest.md` (this sprint's commit;
force-added under the gitignored `reports/` umbrella following the
§170 P3 / P4 P1 pattern).

### 7. Surface-immediately tracking

None fired. All monitors clear at close:
- Schema-parity check: PASS (4 core keys identical to base corpus).
- canvas_realness polarity: PASS (plane 8 inside-mean 217.0 = R=8 hex cells).
- Position target (10–20 k operator band): PASS (12,781).
- KrakenBot pair-bug: known issue, surfaced in manifest, absorbed by
  game-count buffer; 33 % game-yield is acceptable for a 0.15-weight
  source. Not a §170 P4 P2 fix.

### 8. Done-when

- [x] `data/adversarial_corpus_v8.npz` exists on 5080 + laptop.
- [x] sha256 captured in `reports/gpool_bias/adversarial_manifest.md`.
- [x] Per-source counts + opponent strength bands + position-filter
      criteria documented in the manifest.
- [x] §171 entry-point clear (manifest §"§171 entry-point").
- [x] `scripts/generate_adversarial_corpus.py` committed.
- [x] 1 commit on `encoding/gpool_bias_a1`:
      `feat(corpus): adversarial corpus prep for §171 A4 fine-tune`
      (this sprint's commit).
- [x] Sprint log §170 P4 P2 entry (this section).
- [x] NO retrain performed.

### 9. STOP — awaiting operator go on §171 scope (Gate 6 item c)

§170 P4 P2 prep ends here. The next sprint either opens §171 (Phase D
self-play smoke under A1 anchor canonical pick — Gate 6 items a + b)
or §171 A4 fine-tune side-arm (Gate 6 item c) using
`bootstrap_corpus_v8_canvas_realness.npz` (sha `110ea6b2…`) + this
adversarial corpus + the A4 checkpoint. Operator's call.

**Key result: corpus prep done. 12,781 v8-encoded canvas_realness
positions, 5 sources, 45 % SealBot vs A1 + 25 % scripted adversaries
+ 30 % peer / self — biased toward exactly the distribution A4
collapses on, without contaminating the eval anchor.**



---


## §170 P4 — close-out + Gate 6 operator surface — 2026-05-09

**Branch:** `encoding/gpool_bias_a1` (off `b3f5361`, post-§170 P3
aggregation).
**Predecessor:** §170 close-out (P0 SPATIAL_RICH + P1 FLAT-NON-MONOTONIC +
P3 FALSIFIED; recommended P4 spike under Gate 6 item d).
**Successor (operator-decided):** §171 (Phase D self-play smoke under
canonical pick) ± §171 A4 distribution-shift fine-tune side-arm — see
Gate 6 below.
**Status:** P0 + P1 + P2 complete. Awaiting operator go on Gate 6 items
(a)–(g).

---

### TL;DR

- **§170 P4 closes the gpool-bias side-channel research line at NULL.**
  Three probes ran in P4: P0 (architecture surgery — value-head
  structurally frozen), P1 (retrain + matched eval — pre-registered
  NULL verdict), P2 (§171 A4 corpus prep, no retrain). No checkpoint
  promotion taken.
- **§170 P4 P0 (architecture surgery, ~15 min wall): structurally
  sound.** `GpoolBiasBranch` extended with a `policy_only_bias`
  constructor flag; under `True` the value-bias path is a constant
  zero so no gradient ever flows through `value_proj`. State-dict
  shape unchanged from P3 (no checkpoint break); forward parity at
  gate=0 byte-exact vs A1 anchor preserved; 13 new unit tests under
  `test_gpool_bias_policy_only.py` green; broader `make test` 1168 / 8
  / 2.
- **§170 P4 P1 (retrain + eval, 1 h 48 m + 1 h 11 m wall): NULL.**
  argmax 15.0% [10.7%, 20.6%] (n=200) lands cleanly in the pre-
  registered [12%, 16%] NULL band (A1 anchor 14.5%). MCTS-{32, 64,
  128} = 24.5% / 32.5% / 39.5% — clean monotonic-increasing depth-
  scaling, the standard healthy-value-head signature. MCTS-64 32.5%
  [26.4%, 39.3%] CIs heavily overlap A1 anchor MCTS-64 30.0% [24.1%,
  36.7%] — point estimate +2.5 pp, statistically indistinguishable.
  Final loss 3.1945 (epoch 30/30); final gate 0.0718 (~3× growth from
  zero, ~40 % higher than P3's 0.0512 — policy-only path puts MORE
  weight on the global signal because it's the only path that can
  earn weight); 0 % NaN-skip; forward parity green; post-train
  `value_proj` |max| 0.0884 ≈ Kaiming-uniform init bound √(1/128)
  confirms zero gradient ever flowed through the value path across
  30 epochs.
- **§170 P4 P2 (corpus prep, ~14 min wall, no retrain): ENGINEERING-
  COMPLETE.** `data/adversarial_corpus_v8.npz` (sha256 `e6c1b9b9…`,
  12,781 positions, 198.2 MB, 5 sources at operator-tuned weights)
  ready on 5080 + laptop, schema byte-compatible with
  `bootstrap_corpus_v8_canvas_realness.npz` (sha `110ea6b2…`). §171
  A4 fine-tune entry-point clear; mix and recipe NOT committed
  (operator-side §171 scope decision).
- **Mechanistic synthesis.** P4 P1 confirms the §170 P1 + P3 reading:
  the value head's *operating point* (not the routing or pool layer)
  is the controlling factor under PUCT search. Bilateral bias (P3)
  bought +7.5 pp argmax via value-head drift while collapsing MCTS
  −15 pp; policy-only bias (P4 P1) freezes the value head
  structurally — the argmax lift evaporates back to anchor AND the
  MCTS regression disappears, simultaneously. The K-invariant global
  signal, when delivered ONLY through the policy head, produces NO
  measurable WR lift on this distribution. **The "policy-side global
  signal" hypothesis is FALSIFIED for SealBot adversarial play on
  v6w25 K-cluster.**
- **Operator-decided next steps.** A1 anchor remains the canonical
  bootstrap for §171; gpool-bias as a side-channel adds no measurable
  benefit under either bilateral or policy-only routing on this
  distribution. Gate 6 items (d) and (e) close the architectural
  side-arm queue.

---

### 1. Branch state

```
6e3b433 feat(corpus): adversarial corpus prep for §171 A4 fine-tune       ← §170 P4 P2
0c76b34 eval(a1-gpool-bias-policy-only): argmax + MCTS-{32,64,128}        ← §170 P4 P1 eval
f2fec2f feat(retrain): A1+gpool-bias-policy-only retrain config + script  ← §170 P4 P1 retrain
c399a91 feat(model): A1+gpool-bias-policy-only (value_gate=0, stop-grad)  ← §170 P4 P0
b3f5361 docs(170): aggregation + Gate 6 operator surface                  ← §170 P3 close-out
3f15a42 docs(sprint): §170 P3 FALSIFIED — A1 anchor MCTS-64 baseline reveals -15pp regression
... (§170 P3 chain unchanged below)
```

This aggregation commit (`docs(170-p4): aggregation + Gate 6 operator
surface`) lands on top of `6e3b433`; no functional code changes.

---

### 2. Probe-by-probe summary

#### §170 P4 P0 — A1 + gpool-bias-policy-only architecture (verdict: structurally sound)

Question: can the gpool-bias side-branch be wired with a hard
structural guarantee that NO gradient flows through `value_proj`
during training, while preserving the gate=0 byte-exact A1 forward
invariant?

Result: **PASS.** `GpoolBiasBranch` extended with a `policy_only_bias`
constructor flag. Under `True`: `value_bias = torch.zeros_like(...)`
emitted unconditionally (no `value_proj` invocation in the forward
path) — `value_proj.weight` remains a parameter for state-dict round-
trip with P3 ckpts but never sees a gradient. State-dict shape
unchanged from P3 (no checkpoint break); forward parity at gate=0
byte-exact vs A1 anchor (max\|Δ\| = 0). Architecture invariant tests:
13 new across `test_gpool_bias_policy_only.py`, all green; broader
`make test` 1168 / 8 / 2 (matches §170 P3 landing — no regressions).

**Implication.** Architecture invariant strong enough for P1 to
discriminate the value-head-drift hypothesis: any post-train signal
change must come from policy-side gradient flow only.

Full entry: §170 P4 P1 above (P0 architecture commit `c399a91`).

#### §170 P4 P1 — A1 + gpool-bias-policy-only retrain + eval (verdict: NULL)

Question: does limiting the gpool-bias side-branch's gradient flow to
the policy head (value path frozen at A1 by construction) lift argmax
(predicted preserve ~22 % from §170 P3) while restoring MCTS to A1
anchor (~30 %)?

Pre-registered criteria (LOCKED at sprint header):
- **WIN** = argmax > 16 % AND MCTS-64 > 27 %.
- **PARTIAL-WIN** = argmax > 16 % AND MCTS-64 in [22 %, 27 %].
- **NULL** = argmax in [12 %, 16 %] AND MCTS-64 in [22 %, 32 %].
- **LOSS** = MCTS-64 < 24 % UB (disjoint-below A1 anchor CI).

Result:
- **Pretrain** (recipe identical to §170 P3 + `--policy-only-bias`,
  corpus reused verbatim, sha256 `e2876ae5…`, 30 ep cosine, peak 2e-3,
  eta_min 5e-5, batch 256, fp16): final loss **3.1945** (well below
  5.36 hard-stop), final gate **0.0718** (peak 0.0775; ~3× growth
  from absolute zero, ~40 % higher than P3's 0.0512), 0 % NaN-skip,
  forward parity gate=0 unit test green. Wall: 1 h 48 m on 5080 (one
  mid-run SSH drop on the primary `REMOTE_HOST:REMOTE_PORT` endpoint
  required relaunch under tmux detached on the alternate `38118 →
  REMOTE_IP` endpoint; final run contained the full 30 epochs in
  one process).
- **Post-train value-path-frozen verification** (asserted on the
  trained checkpoint): `value_proj.weight` |max| = 0.0884 ≈ Kaiming-
  uniform init bound √(1/128) and std = 0.0510 — **structural proof
  that zero gradient ever flowed through `value_proj` across 30
  epochs**. By contrast `policy_proj.weight` shows |max| = 1.0481
  (~12× init bound), confirming the policy-bias path was actively
  trained.
- **Eval matrix** vs SealBot (n=200, seed_base=42, legal_radius=8,
  random_opening_plies=4, c_puct=1.5, time_limit=0.5 s, `--policy-
  only-bias` forced at inference; bilateral-load JSONs preserved as
  `*_bilateral_load.json` for reference — argmax bit-exact between
  modes, MCTS shifts < 1 pp under inference-mode swap):

  | inference | WR        | 95 % CI         | W / L / D    | mean_ply |
  |-----------|----------:|-----------------|--------------|---------:|
  | argmax    | **15.0 %** | [10.7 %, 20.6 %] | 30 / 165 / 5 | 52.73    |
  | MCTS-32   | 24.5 %    | [19.1 %, 30.9 %] | 49 / 151 / 0 | 35.57    |
  | MCTS-64   | **32.5 %** | [26.4 %, 39.3 %] | 65 / 135 / 0 | 37.36    |
  | MCTS-128  | **39.5 %** | [33.0 %, 46.4 %] | 79 / 121 / 0 | 38.93    |

  Threat probe SKIPPED (no v6w25 fixture; same status as A2 / A3 /
  §170 P3 — operator-side curation work, tracked as §170 follow-up).

  NN latency bench (5080 host = 79f24b481d6b, n=5 each): A1+gpool-bias
  (P3, full) b=1 1.49 ms / b=64 11.26 ms; A1+gpool-bias-policy-only
  b=1 1.47 ms / b=64 11.26 ms. **State-dict shape unchanged**;
  `policy_only_bias` is a forward-routing flag (skips one Linear
  under `True`), not a parameter delta — latency parity confirmed.

**Verdict: NULL.** argmax 15.0 % lands cleanly in the [12 %, 16 %]
NULL band (point estimate 0.5 pp above A1 anchor 14.5 %, CIs nearly
identical). MCTS-64 32.5 % point estimate sits 0.5 pp above the
strict NULL ceiling (32 %) but its CI [26.4 %, 39.3 %] overlaps A1
anchor MCTS-64 CI [24.1 %, 36.7 %] by 10.3 pp — no statistical
distinction. No axis rises to PARTIAL-WIN; no axis falls to LOSS.

**Implication — discriminator across the three pre-registered
hypothesis branches in the §170 P4 prompt:**
- **policy_only > P3 on MCTS axis: CONFIRMED** (15.0 % → 32.5 % at
  MCTS-64; +17.5 pp). Value-path freezing was the load-bearing fix
  for P3's MCTS regression.
- **policy_only ≈ A1 on MCTS axis: CONFIRMED** (32.5 % vs 30.0 %,
  CIs overlap by 10.3 pp). The K-invariant global signal, delivered
  through policy only, is captured by the SE / min-max trunk OR is
  genuinely uninformative — either way it produces no measurable WR
  lift on this distribution.
- **policy_only < A1 on MCTS axis: REFUTED** (no LOSS; UB 39.3 % ≫
  A1 anchor LB 24.1 %). Shared-trunk gradient flow does NOT make the
  value head sensitive to side-branch policy training; the stronger
  hypothesis flagged in the §170 close-out item (d) risk note is
  refuted.

The canonical claim is therefore: **the +7.5 pp argmax lift in §170
P3 was bought ENTIRELY by value-head bias drift; the global signal
is usable via policy-only routing in the structural sense (no MCTS
regression) but produces NO measurable WR improvement.** Closes the
gpool-bias line at NULL.

Full entry: §170 P4 P1 above (lines 10455–10721).

#### §170 P4 P2 — adversarial corpus prep (verdict: ENGINEERING-COMPLETE; no retrain)

Question (from §170 close-out Gate 6 item c): is the §171 A4
distribution-shift fine-tune corpus ready, given the §170 P0
SPATIAL_RICH verdict mechanistically justifies the fine-tune?

Result: `data/adversarial_corpus_v8.npz` (sha256
`e6c1b9b921492d9b23f825cce26e99b818285743fffef8aec3ae47532ef84c2c`,
12,781 positions, 198.2 MB, 5 sources at operator-tuned weights)
generated on 5080 in ~14 min wall, schema byte-compatible with
`bootstrap_corpus_v8_canvas_realness.npz` (sha `110ea6b2…`) — states /
policies / outcomes / weights identical in shape + dtype; one extra
`source_labels` diagnostic column ignored by the pretrain loader.

Per-source: 6,750 sealbot_vs_a1 (0.45 weight, primary), 1,860
scripted_far_line + 958 scripted_far_placement (0.13 + 0.12 weight,
§164 P2 OOD adversaries), 963 krakenbot_vs_sealbot (0.15 weight,
~33 % game-yield due to KrakenBot pair-bug surfaced in manifest),
2,250 sealbot_vs_sealbot (0.15 weight, lowest — eval-anchor overfit
risk). Total 12,781 / 15,000 target ≈ 85 %, in the 10–20 k operator
band.

**Implication.** Corpus prep done. §171 A4 fine-tune entry-point:
`bootstrap_corpus_v8_canvas_realness.npz` (sha256 `110ea6b2…`) +
`adversarial_corpus_v8.npz` (sha256 `e6c1b9b9…`) +
`checkpoints/ablation_169/A4_canvas_realness.pt`. Recommended natural-
ratio mix per P2 manifest §"§171 entry-point": adversarial ≈ 3.7 % of
base by position count, 2–5 k fine-tune steps, peak LR ≤ 5 e-5,
freeze PartialConv2d trunk-entry, unfreeze res_blocks 8–11 + heads
only — the mix and recipe are NOT committed in this sprint; they are
operator-side §171 scope decisions surfaced under Gate 6 item (c).

Full entry: §170 P4 P2 above (lines 10726–10901).

---

### 3. Comparison matrix (extended with §170 P4 P1 row)

Full matrix with CIs in `reports/gpool_bias/SUMMARY.md` §1.

| arm                                            | encoder         | pool                  | global routing                              |    argmax | MCTS-N WR (depth)                                           | mean_ply MCTS-64 | gate end | params                          |
|------------------------------------------------|-----------------|-----------------------|---------------------------------------------|----------:|-------------------------------------------------------------|-----------------:|---------:|--------------------------------:|
| **A1 anchor**                                  | v6w25 K-cluster | min/max               | none                                        | **14.5 %** | 25 % MCTS-32 (n=20 sanity); **30.0 % MCTS-64**; **32.5 % MCTS-128** (n=200) | 33.8             | n/a      | 5.29 M                          |
| A2                                             | v6w25 K-cluster | PMA                   | none                                        | 4.5 %     | 3.5 % MCTS-128                                              | n/a              | n/a      | 6.30 M                          |
| A3                                             | v6w25 K-cluster | PMA                   | yes (PMA-merged both heads)                 | 7.5 %     | 2.5 % MCTS-32 / 64 / 128 (FLAT-NON-MONOTONIC)               | 22.8             | 0.66     | 6.37 M                          |
| A4                                             | v8 + canvas     | bbox + PartialConv2d  | n/a (canvas-mask trunk)                     | 0.0 %     | 0.0 % MCTS-128                                              | 23.6             | n/a      | 3.85 M                          |
| **A1 + gpool-bias** (§170 P3 — FALSIFIED)      | v6w25 K-cluster | min/max + gpool-bias  | yes (additive bias both heads)              | **22.0 %** | **15.0 % MCTS-64** (regression vs anchor; CIs disjoint)     | 29.7             | 0.0512   | 5.29 M + 0.18 M = 5.47 M        |
| **A1 + gpool-bias-policy-only** (§170 P4 P1 — NULL) | v6w25 K-cluster | min/max + gpool-bias  | yes (additive bias policy only, value frozen) | **15.0 %** | 24.5 % MCTS-32 / **32.5 % MCTS-64** / 39.5 % MCTS-128 (monotonic) | 37.36            | 0.0718   | 5.29 M + 0.18 M = 5.47 M        |

Notes:
- A1+gpool-bias-policy-only state-dict shape is identical to A1+gpool-
  bias bilateral; `policy_only_bias` is a forward-routing flag, not a
  parameter delta. `value_proj.weight` exists in the checkpoint at
  random-init footprint (Kaiming-uniform init bound √(1/128)) but is
  never invoked at inference under the flag.
- mean_ply MCTS-64 axis tracks value-head integrity: A1 anchor 33.8
  (healthy), A3 22.8 (collapsed under PMA-replaced value), §170 P3
  29.7 (collapsed under value-bias drift), §170 P4 P1 37.36
  (recovered — *higher* than anchor, consistent with no regression
  under matched-baseline conditions).
- "monotonic" in MCTS-N column = standard PUCT-search-deepens-improves
  signature; "FLAT-NON-MONOTONIC" = §170 P1 A3 cliff; "regression"
  = §170 P3 collapse.

Decomposition reading + canonical-pick rationale: see
`reports/gpool_bias/SUMMARY.md` §2–§3 (extended with §2e and §3 P4
verdict paragraph in this sprint's commit).

#### Matrix completion — A1 anchor MCTS-128 (operator-flagged 2026-05-09; laptop run)

The original §170 P4 close-out matrix had A1 anchor measured only at
MCTS-64 (30.0 % [24.1 %, 36.7 %]). The §170 P4 P1 cell at MCTS-128
(39.5 % [33.0 %, 46.4 %]) was the highest single MCTS WR in the entire
encoding line, but lacked a matched-depth anchor baseline — the
"indistinguishable from A1 anchor at MCTS-64" reading was depth-
mismatched.

Operator-flagged observation: capture A1 anchor at MCTS-128 to
complete the record. Run on laptop (4060 Max-Q), 27 min wall (1636 s,
8.2 s / game), matched config to all other §170 baselines (n=200,
seed_base=42, legal_radius=8, random_opening_plies=4, c_puct=1.5,
time_limit=0.5 s, temperature=0.0).

| arm                                            | MCTS-32 | MCTS-64               | MCTS-128                | Δ (32→64) | Δ (64→128) |
|------------------------------------------------|--------:|----------------------:|------------------------:|----------:|-----------:|
| A1 anchor                                      | 25 %†   | **30.0 %** [24.1 %, 36.7 %] | **32.5 %** [26.4 %, 39.3 %] | +5.0 pp†  | +2.5 pp    |
| A1 + gpool-bias-policy-only (§170 P4 P1)       | 24.5 %  | 32.5 % [26.4 %, 39.3 %] | **39.5 %** [33.0 %, 46.4 %] | +8.0 pp   | +7.0 pp    |

†A1 anchor MCTS-32 is the n=20 §168 sanity run; not directly comparable to the n=200 cells.

**Reading.**
- A1 anchor's own MCTS depth scaling is **shallow**: +2.5 pp from
  N=64 to N=128. The model does extract some additional signal from
  doubling search, but the lift is modest.
- A1+gpool-bias-policy-only's depth scaling is **steeper**: +7.0 pp
  from N=64 to N=128 (and +15.0 pp end-to-end MCTS-32 → MCTS-128).
- At MCTS-128, A1+gpool-policy-only point estimate is 39.5 % vs A1
  anchor 32.5 % — **+7.0 pp**, the largest matched-depth gap in the
  matrix. CIs overlap by 6.3 pp ([33.0 %, 46.4 %] vs [26.4 %,
  39.3 %]); not statistically distinguishable at n=200.
- A1 anchor MCTS-128 W/L/D = **65 / 135 / 0**, byte-identical W/L
  count to A1+gpool-policy-only MCTS-64 (also 65 / 135 / 0). Neither
  cell distinguishes from the other; per-game outcomes differ but
  aggregate WR is identical to four decimal places. Not load-bearing
  on the matrix — coincidence at n=200.
- mean_ply MCTS-128: A1 anchor 35.35 (median 33.0); A1+gpool-policy-
  only 38.93 (median 35.0). Both are healthy depth-deepens-mean-ply
  curves; gpool-policy-only's longer mean reflects more games where
  it pushes SealBot into deeper play before resolution.

**Implication for verdict.** Does NOT change Gate 6 (a) — A1+gpool-
policy-only's argmax 15.0 % still fails the 16 % gate; joint
promotion gate FAILS regardless of the MCTS-128 cell. **Does refine
the NULL reading**: the §170 P4 P1 NULL verdict held that policy-
only "is captured by the SE / min-max trunk OR is genuinely
uninformative". The matched MCTS-128 baseline weakens the
"genuinely uninformative" branch — at n=200 the point estimate
suggests the global signal *is* doing something at higher search
depth that the trunk does NOT extract. The hypothesis "global signal
becomes measurably informative at higher PUCT depth" is open at
n=200 and would need n=400+ at MCTS-128 to discriminate.

**Implication for §171 / Phase 5+.** Surface as a Phase 5+ question
(operator-confirmed Gate 6 (g) deferral list extended): does the
gpool-bias-policy-only side-branch produce a statistically
distinguishable lift at MCTS-128+ given a corpus or distribution
that exercises the global signal more, OR an n=400+ measurement at
the current corpus? Not a §171 blocker; cleanly deferred. The
operator-promised "doesn't change Gate 6 (a)" reading is preserved;
the matrix is now complete with no orphan cells.

Artefact: `reports/gpool_bias/A1_anchor_mcts128.json` (laptop run);
outer log `reports/gpool_bias/A1_anchor_mcts128.outer.log`.

---

### 4. Gate 6 — operator decision packet

a) **Promote A1+gpool-bias-policy-only to `bootstrap_model.pt`?**
   **Recommendation: NO, do not promote.** Promotion gate is
   `argmax > 16 % AND MCTS-64 > 27 %`. P4 P1 argmax = 15.0 % (FAIL by
   1.0 pp; below the 16 % gate, indistinguishable from A1 anchor
   14.5 %). P4 P1 MCTS-64 = 32.5 % (PASS). Joint gate FAILS on
   argmax. A1 anchor (`bootstrap_model_v6w25.pt`, sha256
   `571a82f8…`) remains canonical. P4 P1 confirms structural
   soundness but no measurable WR lift; promotion would be neutral
   at best on WR while adding the gpool-bias side-branch overhead
   (5.29 M → 5.47 M params, b=64 latency 10.41 ms → 11.26 ms) without
   compensating signal.

b) **Open §171 (Phase D self-play smoke) under canonical pick? Which?**
   **Recommendation: open §171 under A1 anchor (v6w25 K-cluster +
   min/max, no global routing).** A1 anchor is the only §169 / §170
   arm with a measured MCTS-64 above the §171 sustained-run candidate
   gate (~27 %): 30.0 % [24.1 %, 36.7 %]. P4 P1's MCTS-64 32.5 %
   point estimate is statistically indistinguishable from anchor and
   adds a side-branch with no functional benefit.

c) **Open §171 sub-arm: A4 distribution-shift fine-tune using
   `adversarial_corpus_v8.npz` (P2 prepped)?**
   **Recommendation: open as a §171 SIDE-ARM, not the canonical.**
   Mechanistic justification is strong: §170 P0 SPATIAL_RICH verdict
   (mean KL 4.19 nats, argmax stable 0/200 under stones-zeroed) means
   A4's 0 % SealBot WR is a distribution-shift problem, not an
   architecture one; A4's spatial pathway is alive AND load-bearing.
   Corpus prep is done (P2: `e6c1b9b9…`, 12,781 positions, schema-
   parity). Fine-tune cost: ~3–4 hr on 5080 with the natural-ratio
   mix recommended in P2 manifest §"§171 entry-point". Risk: opening
   §171 with a side-arm dilutes operator attention; bench-gate the
   side-arm against A1 anchor's §171 baseline before committing
   wider scope.

d) **Close gpool-bias line entirely (NULL verdict)?**
   **Recommendation: YES, close the gpool-bias research line.** P4 P1
   NULL closes the gpool-bias side-channel under both bilateral (P3
   FALSIFIED, MCTS regression) and policy-only (P4 P1 NULL, no
   measurable lift) routing. The K-invariant global signal does NOT
   add actionable signal beyond what the K-cluster + min/max trunk
   already extracts on the v6w25 + SealBot adversarial-play
   distribution. State-dict shape unchanged from P3 → P4 P1
   (`policy_only_bias` is a forward-routing flag); existing
   checkpoints are not orphaned but should NOT be promoted. The side-
   branch architecture remains in the codebase as documented dead-end
   research; no removal needed.

e) **Hybrid scope: gpool-bias-policy-only + extended training, or
   light-touch attention on policy-only — open as §172 spike?**
   **Recommendation: DEFER to Phase 5+ intake review.** P4 P1 already
   shows policy-only routing produces NO measurable WR lift after 30
   epochs at the §170 P3 recipe; final loss 3.1945 (still > A1 anchor
   3.57 in the §169 reading where loss-vs-WR is decoupled). No data-
   driven evidence that more compute or finer attention (Set-
   Transformer light-touch on policy heads, PMA-policy + min-value
   A2′ hybrid) would clear the 16 % argmax gate. Either spike risks
   ~3–6 hr of 5080 compute against a high-prior-NULL hypothesis; the
   same compute funds §171 (Gate 6 b) or §171 A4 side-arm (Gate 6 c)
   directly. Revisit only if Phase 5+ surfaces a different-distribution
   corpus where the global-crop signal becomes measurably informative.

f) **v8 bench-gate finalize — replace v6 numbers with current
   canonical?**
   **Recommendation: DEFER (unchanged from §170 P3 close-out).** v8's
   current canonical (B1 bbox + canvas_realness + PartialConv2d, 0 %
   SealBot WR) cannot replace v6w25 / A1 anchor bench numbers as
   authoritative for Phase D until §171 A4 fine-tune (Gate 6 item c)
   lifts v8 above the 16 % argmax / 27 % MCTS-64 gate. Revisit at
   §171 close-out if (c) opens and verdict positive.

g) **Architecture explorations to defer to Phase 5+?**
   **Recommendation: defer the following list, queue for Phase 5+
   intake review (extends §170 P3 close-out item (f) with §170 P4
   additions):**
   - **gpool-bias-policy-only with extended training / larger global
     token / different reduction.** P4 P1 NULL with the §170 P3
     recipe; extension is operator-side ablation work, not a Phase 4
     blocker.
   - **Set-Transformer light-touch on policy heads** (alternative
     routing to gpool-bias-policy-only with the same value-head-
     frozen invariant).
   - **PMA-policy + min-value hybrid (A2′)** (§169 P3 close-out
     option; alternative to gpool-bias-policy-only with PMA routing
     on policy only).
   - **bbox-direction redesign** (K=1 vs K>1 corpus supervision,
     bbox-centroid frame instability, single-window inference-time
     blindness — §169 P4 close-out items). Defer until §171 A4 fine-
     tune (Gate 6 c) verdict lands.
   - **Threat-probe v6w25 fixture build** — operator-side curation
     work (curated tactical positions on a 25 × 25 board + regenerated
     baseline). Currently SKIPPED across A1 / A2 / A3 / A1+gpool /
     A1+gpool-policy-only. Track as a §170 hangover not blocking
     §171.
   - **Formal A4 MCTS-N curve** (skipped per §170 close-out §2 §170
     P2) — only resurrect for paper / external-review audit trail;
     not informative under the §170 P0 SPATIAL_RICH + §169 P4 0 %
     MCTS-128 result triangulation.

---

### 5. Outstanding for operator at Gate 6

- [ ] Decide (a)–(g). Items (a), (b), (d), (e), (f), (g) recommended
      verdicts above are low-controversy; item (c) is the active
      scope decision affecting the next 3–4 hours of 5080 compute.
- [ ] If (c) is opened: scope §171 A4 distribution-shift fine-tune in
      a separate context, reusing `adversarial_corpus_v8.npz` (sha256
      `e6c1b9b9…`) +
      `bootstrap_corpus_v8_canvas_realness.npz` (sha256 `110ea6b2…`)
      + `checkpoints/ablation_169/A4_canvas_realness.pt`. Mix and
      recipe in P2 manifest §"§171 entry-point".
- [ ] Either way, scope §171 (Phase D self-play smoke) under A1
      anchor canonical pick.
- [ ] No checkpoint promotion; no gating yaml change; no §171 scope
      opened in this sprint.

---

### 6. Surface-immediately tracking

None fired during execution. All monitors clear at close:
- §170 P4 P0 architecture invariants (forward parity gate=0, state-
  dict round-trip, `value_proj.weight.grad is None` unit test): all
  PASS (13 new tests green).
- §170 P4 P1 hard-stops (loss < 5.36, NaN < 30 %, gate trajectory
  healthy): all PASS (loss 3.1945, NaN 0 %, gate 0.0718 ≫ 0.05 soft-
  warn null).
- Post-train value-path-frozen invariant: PASS (`value_proj` |max|
  0.0884 ≈ Kaiming-uniform bound; `policy_proj` ~12 × init bound).
- argmax bit-exactness between bilateral-load and policy-only-load
  inference modes: confirmed (max\|Δ\| log_policy = 0); MCTS shifts
  < 1 pp under inference-mode swap; final results use the corrected
  policy-only-load run; bilateral-load JSONs preserved as
  `*_bilateral_load.json` for reference.
- §170 P4 P2 schema-parity, canvas_realness polarity, position
  target band: all PASS.

---

### 7. STOP — awaiting operator go on Gate 6

Branch `encoding/gpool_bias_a1` ready at `6e3b433` + this aggregation
commit. §170 P4 close-out merges into master AFTER operator decision
on items (a)–(g). No checkpoint promotion taken; no gating yaml
changed; no §171 scope opened.

**Key result: §170 P4 P1 NULL closes the gpool-bias side-channel
research line. A1 anchor + min/max pool + no global routing remains
the canonical v6w25 bootstrap for §171 self-play. §170 P4 P2 corpus
prep ready as the §171 A4 distribution-shift fine-tune entry-point
if Gate 6 item (c) opens.**

**Next:** §171 (Phase D self-play smoke under A1 anchor, operator-
decided, items b + d) ± §171 A4 fine-tune side-arm (operator-decided,
item c).

---

