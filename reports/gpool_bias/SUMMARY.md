# §170 Aggregation — Comparison matrix + decomposition + canonical pick

**Sprint:** §170 — A1 + gpool-bias retrain (Bet B / option α canonical play).
**Branch:** `encoding/gpool_bias_a1`.
**Date:** 2026-05-09 (artifacts pulled from 5080 vast.ai 2026-05-09 ~02:07 UTC; P4 P1 + P2 artifacts pulled 2026-05-09 ~14:00 UTC).
**Predecessor:** §169 four-way ablation close-out + §169a + §170 P0 + §170 P1.
**Verdict:** §170 P3 **FALSIFIED** + §170 P4 P1 **NULL** → gpool-bias
research line closed. A1 anchor remains canonical. §170 P4 P2 adversarial
corpus prepped as §171 A4 distribution-shift fine-tune entry-point
(operator-side scope decision).

---

## 1. Comparison matrix

All argmax / MCTS evaluated vs SealBot, n=200, seed_base=42, legal_radius=8,
random_opening_plies=4, c_puct=1.5, time_limit=0.5s. Threat-probe rows
SKIPPED across all v6w25 / v8 arms — fixture gap tracked as §170 follow-up.

| arm                                   | encoder        | pool                    | global routing            | argmax WR (CI95)              | MCTS-N WR (CI95)                       | threat C2/C3 | params  | source                           |
|---------------------------------------|----------------|-------------------------|---------------------------|-------------------------------|----------------------------------------|--------------|---------|----------------------------------|
| **A1 anchor** (v6w25)                 | v6w25 K-cluster| min/max                 | none                      | **14.5%** [10.3%, 20.0%]      | 25.0% MCTS-32 (n=20 sanity); **30.0% MCTS-64** [24.1%, 36.7%] (n=200, matched baseline) | SKIPPED      | 5.29 M  | §168 Gate 5 + §170 P3 baseline   |
| A2                                    | v6w25 K-cluster| PMA                     | none                      | 4.5% [2.4%, 8.3%]             | 3.5% MCTS-128                          | SKIPPED      | 6.30 M  | §169 P2                          |
| A3                                    | v6w25 K-cluster| PMA                     | yes (PMA-merged)          | 7.5% [4.6%, 12.0%]            | 2.5% MCTS-32 / 2.5% MCTS-64 / 2.5% MCTS-128 (FLAT) | SKIPPED      | 6.37 M  | §169 P3 + §170 P1                |
| A4                                    | v8 + canvas    | bbox + PartialConv2d    | n/a (canvas-mask trunk)   | 0.0% [0.0%, 1.9%]             | 0.0% MCTS-128 [0.0%, 1.9%]             | SKIPPED      | 3.85 M  | §169 P4 (+ §169a + §170 P0 mech) |
| **A1 + gpool-bias** (§170 P3)         | v6w25 K-cluster| min/max + gpool-bias    | yes (additive bias, gate=0.0512) | **22.0%** [16.8%, 28.2%]      | **15.0% MCTS-64** [10.7%, 20.6%]       | SKIPPED      | 5.47 M  | §170 P3                          |
| **A1 + gpool-bias-policy-only** (§170 P4 P1) | v6w25 K-cluster| min/max + gpool-bias | yes (additive policy only, value frozen, gate=0.0718) | **15.0%** [10.7%, 20.6%]      | 24.5% MCTS-32 [19.1%, 30.9%]; **32.5% MCTS-64** [26.4%, 39.3%]; 39.5% MCTS-128 [33.0%, 46.4%] | SKIPPED      | 5.47 M  | §170 P4 P1                       |

Notes on the matrix:
- `global routing` distinguishes the *mechanism* by which the global
  signal enters the heads. **A3** uses **PMA-merged routing** — a Set-
  Transformer-style learned aggregation that REPLACES min/max in the
  pool layer. **A1+gpool** uses **additive bias routing** — a separate
  side-branch (KataGo gpool encoder → small projector) that injects an
  additive offset INTO each head's hidden activation while leaving the
  pool layer's min/max byte-exact.
- `MCTS-N` is reported at the depth(s) actually measured. A3's flat
  curve (P1) is reported in full; A1 + gpool-bias was measured at
  MCTS-64 only (matched A1 anchor baseline run post-eval, 839 s on 5080).
  A1 anchor was spot-sanity-tested at MCTS-32 (n=20, §168 P1) and
  matched-baseline-tested at MCTS-64 (n=200, §170 P3 follow-up).
- `params` for A1+gpool-bias = A1 anchor (5.29 M) + gpool-bias side-
  branch (~0.18 M = encoder ~4.5 k + value_proj 32.9 k + policy_proj
  ~80.9 k + 2 small Linears).
- A3's 6.37 M is A2's 6.30 M plus the global-token Linear projector;
  A1+gpool-bias's 5.47 M is a SECOND incremental side-branch, larger
  than A3's because it has TWO projectors (value + policy) instead of
  one fused projector.
- **A1+gpool-bias-policy-only state-dict shape is identical to
  A1+gpool-bias bilateral.** `policy_only_bias` is a forward-routing
  flag (skips the `value_proj` Linear under `True`), not a parameter
  delta. `value_proj.weight` exists in the checkpoint at random-init
  footprint (post-train |max| = 0.0884 ≈ Kaiming-uniform init bound
  √(1/128)) but is never invoked at inference. b=64 latency parity
  with §170 P3 confirmed (11.26 ms both, 5080 host = 79f24b481d6b).

---

## 2. Decomposition reading

This section extends `reports/ablation_169/DECOMPOSITION.md` by adding
the §170 axis (additive bias routing) on top of the §169 four-way
ablation (min/max vs PMA × no-global vs global-via-PMA).

### 2a. A1 + gpool-bias vs A1 anchor — does K-invariant global bias help?

| metric        | A1 anchor   | A1 + gpool-bias | Δ                  |
|---------------|------------:|----------------:|-------------------:|
| argmax WR     | 14.5%       | **22.0%**       | **+7.5 pp**        |
| MCTS-64 WR    | 30.0%       | **15.0%**       | **−15.0 pp**       |
| final loss    | 3.57        | **2.8963**      | **−0.67** (better) |
| params        | 5.29 M      | 5.47 M          | +0.18 M (+3.4%)    |
| latency b=1   | 2.64 ms     | 1.49 ms         | (warm-up drift)    |
| latency b=64  | 10.41 ms    | 11.26 ms        | +0.85 ms (+8.2%)   |

**Reading.** The K-invariant gpool-bias signal IS doing real work on
the policy head: argmax lifts +7.5 pp (a ~52% relative improvement)
and final loss undercuts the anchor by 0.67. This is the largest
single-arm argmax gain measured against the v6w25 anchor across §169
and §170 combined.

But the argmax lift does NOT survive PUCT search. At MCTS-64, A1+gpool
regresses 15 pp vs the matched A1-anchor baseline (CIs disjoint by
3.5 pp). Mean-ply collapses in the predicted direction: A1 anchor
MCTS-64 holds at 33.8 (median 33), A1+gpool MCTS-64 collapses to
29.7 (median 29) — SealBot finishes faster against the gpool variant
under search than it does against argmax (where mean-ply is 47.98).

**Mechanism.** §170 P3 bet on the structural invariant "additive bias
on a preserved min/max pool preserves the load-bearing value
semantics." The invariant holds only at gate=0 (verified by the
byte-exact unit test against `bootstrap_model_v6w25.pt`). Once the
gate gains weight during training, the value head adapts to use the
bias signal — `value_fc2(F.relu(value_fc1(...)) + value_bias)` emits
values whose distribution shifts vs the gate=0 baseline. PUCT then
backs up these biased values across many simulations; the cumulative
drift breaks selection in the same way A3's PMA-replaced value head
did. The K-cluster reduction still picks min-of-K, but the per-cluster
scalar value the model emits is no longer A1's value.

### 2b. A1 + gpool-bias vs A3 — does additive bias-injection beat PMA-merged routing?

Both arms add a global signal. They differ in HOW the signal enters
the heads (additive bias on a preserved A1 trunk vs PMA-merged on a
trunk where min/max is replaced by attention).

| metric        | A3 (PMA-merged) | A1 + gpool-bias (additive) | Δ                        |
|---------------|----------------:|---------------------------:|-------------------------:|
| argmax WR     | 7.5%            | **22.0%**                  | **+14.5 pp**             |
| MCTS-N WR     | 2.5% (flat)     | **15.0% MCTS-64**          | **+12.5 pp** (still bad) |
| final loss    | 3.62            | **2.8963**                 | **−0.72** (better)       |
| params        | 6.37 M          | 5.47 M                     | −0.90 M (smaller)        |

**Reading.** Additive-bias routing on a preserved A1 trunk DOMINATES
PMA-merged routing on a replaced trunk along every axis: argmax (+14.5
pp), MCTS (+12.5 pp), final loss (−0.72), and parameter count (−0.9 M).

The trunk preservation matters. A3 replaced the K-cluster pool with
PMA — the policy head still saw a global view (good for argmax), but
the value head's per-cluster scalar was now a learned attention sum
instead of the worst-subgame min, breaking PUCT entirely (2.5% MCTS
floor). A1+gpool kept the min/max pool intact, so the value head's
worst-subgame negamax-conservative semantics survive STRUCTURALLY —
the value head still picks min-of-K. Only the bias offset shifts
those picked values' operating point during training.

The result is a strict improvement over A3 in absolute terms, but the
SAME failure mode (argmax-up / MCTS-down). Bias-injection breaks MCTS
LESS catastrophically than PMA-replacement, but it still breaks it.

**Implication.** Routing matters, but only secondarily — what matters
PRIMARILY is whether the value head's operating point can drift during
training. Both A3 (full replacement) and A1+gpool (additive bias on
hidden activation) allow drift. The §170 P1 prescription "policy-only
side-channel; value gate=0.0 forced" is the surgical fix that follows
from BOTH §170 P1 and §170 P3 verdicts together.

### 2c. A4 P2 curve — does MCTS rescue bbox direction?

§170 P2 (an A4 MCTS-N curve at matched perception) was **not run** in
this sprint. The A4-under-MCTS question is settled by:
- §169 P4: A4 MCTS-128 = 0.0% [0.0%, 1.9%] (0/200), matching argmax-only.
- §169a: spatial pathway is alive (mean KL_S = 1.53 nats; argmax 133/200
  distinct cells under random spatial-only variation).
- §170 P0: spatial-rich, not scalar-dominated (mean KL = 4.19 nats vs
  scalar-zeroed copies; argmax STABLE 0/200).

Together these triangulate that A4 has live, load-bearing spatial
features but emits SealBot-broken policy/value on real positions.
MCTS depth cannot rescue a model whose underlying policy + value at
N=1 is already at the noise floor — there is no signal to amplify.
The mechanistic story is therefore: spatial features are intact; the
SealBot collapse is corpus-overfitted spatial representations going
OOD against SealBot's adversarial style. **Distribution-shift fine-
tune is the right next step**, not architectural redesign — see §170
P0 §1–§4. This is the §171 candidate scope.

### 2d. A3 P1 curve — does PMA-value-semantics confirm with depth?

| MCTS-N    | A3 W/L/D | A3 WR | 95% CI         |
|-----------|----------|-------|----------------|
| argmax    | 15/184/1 | 7.5%  | [4.6%, 12.0%]  |
| MCTS-32   | 5/195/0  | 2.5%  | [1.1%, 5.7%]   |
| MCTS-64   | 5/195/0  | 2.5%  | [1.1%, 5.7%]   |
| MCTS-128  | 5/195/0  | 2.5%  | [1.1%, 5.7%]   |

**FLAT-NON-MONOTONIC** verdict (consecutive CI overlap = 100% across
MCTS pairs; identical W/L/D to four decimal places). Cochran-Armitage
p = 0.0277 reflects ONLY the argmax→MCTS-32 cliff, not a trend across N.

The monotone-compounding hypothesis (more sims → larger PMA error
backup) is **refuted**. The correct reading is a **binary switch**:
PMA corrupts value quality once during training; argmax escapes by
reading the policy head directly; MCTS-32 already routes through the
broken value floor and additional sims cannot recover. The cliff is
already saturated at ~3 backup levels (typical median ply 23 of
SealBot games).

This is the §170 P1 result that motivated §170 P3's bet: keep the
min/max pool intact (so value doesn't get corrupted) and add the
global signal as a side-branch (so the policy head keeps the global
view). §170 P3's falsification refines the prescription further:
preserving the pool *layer* is not enough — the bias must not enter
the value HEAD's hidden activation at all. **Policy-only injection +
value gate frozen at zero** is the §170 P4 candidate.

### 2e. A1 + gpool-bias-policy-only vs A1 anchor + §170 P3 — does value-head freeze restore MCTS without forfeiting the argmax lift?

| metric              | A1 anchor   | §170 P3 (bilateral) | §170 P4 P1 (policy-only) |
|---------------------|------------:|--------------------:|-------------------------:|
| argmax WR           | 14.5 %      | 22.0 %              | **15.0 %**               |
| MCTS-32 WR          | 25 % (n=20) | not measured        | 24.5 % (n=200)           |
| MCTS-64 WR          | 30.0 %      | 15.0 %              | **32.5 %**               |
| MCTS-128 WR         | not measured| not measured        | 39.5 % (n=200)           |
| MCTS curve shape    | n/a         | flat-collapsed      | **monotonic-increasing** |
| mean_ply MCTS-64    | 33.8        | 29.7                | **37.36**                |
| final loss          | 3.57        | 2.8963              | 3.1945                   |
| final gate          | n/a         | 0.0512              | 0.0718                   |
| latency b=64        | 10.41 ms    | 11.26 ms            | 11.26 ms                 |

**Reading.** Two §170 P4 prompt branches fire simultaneously:
- **policy_only > P3 on MCTS axis: CONFIRMED** (15.0 % → 32.5 % at
  MCTS-64; +17.5 pp). Value-path freezing was the load-bearing fix
  for §170 P3's MCTS regression. The §170 P3 −15 pp MCTS regression
  was driven by value-head bias drift, NOT shared-trunk gradient
  flow into the policy head.
- **policy_only ≈ A1 on MCTS axis: CONFIRMED** (32.5 % vs 30.0 %,
  CIs overlap by 10.3 pp; statistically indistinguishable). The K-
  invariant global signal, when delivered ONLY through the policy
  head, produces NO measurable WR lift on this distribution.
- **policy_only < A1 on MCTS axis: REFUTED** (no LOSS; UB 39.3 % ≫
  A1 anchor LB 24.1 %). Shared-trunk gradient flow does NOT drift
  the value head indirectly via side-branch policy training — the
  stronger hypothesis flagged in §170 close-out item (d) risk note
  is refuted.

**The canonical claim.** **The +7.5 pp argmax lift in §170 P3 was
bought ENTIRELY by value-head bias drift; the global signal is usable
via policy-only routing in the structural sense (no MCTS regression)
but produces NO measurable WR improvement.** The argmax-up / MCTS-down
P3 fingerprint and the argmax-flat / MCTS-flat P4 P1 fingerprint
*decompose* the §170 P3 effect cleanly: argmax lift was a value-drift
artifact; MCTS regression was the same value-drift seen through PUCT
backups. Freezing the value head erases both — exactly what the §170
P4 prediction "best-of-both" expected, but with the argmax lift
revealed as illusory rather than signal.

**Mean-ply MCTS-64 corroborates the value-head integrity reading.**
A1 anchor 33.8 (healthy) → P3 29.7 (collapsed under value-bias drift)
→ P4 P1 37.36 (recovered). The mean-ply axis discriminates broken-vs-
healthy value heads independently of WR.

**Closes the gpool-bias line.** Bilateral routing breaks MCTS via
value drift (P3 FALSIFIED). Policy-only routing produces no
measurable signal (P4 P1 NULL). No third routing mode is on the table
within the gpool-bias side-branch design space; the K-invariant global
signal does NOT add actionable signal beyond what the K-cluster +
min/max trunk already extracts on the v6w25 + SealBot adversarial-
play distribution.

---

## 3. Recommended canonical pick

**A1 anchor (v6w25 K-cluster + min/max + no global routing) remains
the canonical bootstrap for §171 self-play / sustained training.**
**§170 P4 P1 NULL closes the gpool-bias side-channel research line.**

Rationale:
- A1 anchor is the only §169 / §170 arm with a measured MCTS WR
  exceeding the §171 sustained-run candidate threshold (~27%): MCTS-64
  = 30.0% [24.1%, 36.7%]. §170 P4 P1 (gpool-bias-policy-only) MCTS-64
  = 32.5% [26.4%, 39.3%] is statistically indistinguishable from
  anchor (CIs overlap 10.3 pp) but argmax 15.0% lands at anchor
  parity; no candidate clears the joint promotion gate.
- All five global-signal variants tested (A2 PMA-only, A3 PMA+global,
  A1+gpool-bias bilateral, A1+gpool-bias-policy-only, A4 v8+canvas)
  failed to lift argmax + MCTS jointly above A1 anchor on SealBot
  adversarial play.
- The §170 P3 verdict reproduces the §169 close-out lesson at the bias
  level: training loss alone is NOT a sufficient signal for SealBot
  WR. Both A4 (loss 3.47, 0% WR) and A1+gpool-bias (loss 2.90, 15%
  MCTS WR) show LOWER training loss than the canonical anchor (3.57)
  paired with WORSE behaviour under search. §170 P4 P1 (loss 3.19,
  WR-anchor-parity) sits between these; loss decoupling persists.
- A1+gpool-bias should NOT be promoted — argmax 22% > 16% but MCTS
  15% < 27% (regression). A1+gpool-bias-policy-only should NOT be
  promoted — MCTS 32.5% > 27% but argmax 15% < 16% (anchor parity);
  promotion would add 0.18 M side-branch params + 0.85 ms b=64
  latency without compensating WR signal.

**§170 P4 P1 verdict (NULL) — closes the gpool-bias line.** Both
routings (bilateral §170 P3 FALSIFIED, policy-only §170 P4 P1 NULL)
fail to clear the joint promotion gate. The K-invariant global signal
does NOT add actionable signal beyond what the K-cluster + min/max
trunk already extracts on the v6w25 + SealBot adversarial-play
distribution. The mechanistic synthesis (P1 + P3 + P4 P1): the value
head's *operating point* under PUCT is the controlling factor;
bilateral routing drifts it (P3); policy-only routing preserves it
but cannot lift WR via policy alone (P4 P1). No third routing mode
exists within the gpool-bias side-branch design space. **Architecture
remains in the codebase as documented dead-end research; no removal
needed; no checkpoint promotion taken.**

**§170 P4 P2 — corpus prep ready as §171 A4 distribution-shift fine-
tune entry-point.** `data/adversarial_corpus_v8.npz` (sha256
`e6c1b9b9…`, 12,781 positions, 5 sources, schema-parity with
`bootstrap_corpus_v8.npz` canvas_realness variant) generated in ~14
min on 5080. Operator-side §171 scope decision (Gate 6 item c); not
opened in this sprint.

**Out of scope — defer to Phase 5+:**

- v8 bbox + canvas_realness redesign (K=1 vs K>1 corpus supervision,
  bbox-centroid frame instability, single-window inference-time
  blindness) — §169 P4 close-out. Defer until §171 A4 fine-tune
  (Gate 6 c) verdict lands.
- gpool-bias-policy-only with extended training / larger global token
  / different reduction (P4 P1 NULL with §170 P3 recipe; extension is
  operator-side ablation work).
- Set-Transformer light-touch on policy-only — alternative to
  gpool-bias-policy-only with the same value-head-frozen invariant.
- PMA-policy + min-value hybrid (A2′) — §169 P3 close-out option,
  alternative routing within the same policy-only-side-channel idea.
- Threat-probe v6w25 fixture build — operator-side curation work
  (curated tactical positions on a 25×25 board + regenerated baseline);
  blocks the C1/C2/C3 column for every v6w25 arm including P4 P1.
- Formal A4 MCTS-N curve — not informative under §170 P0 SPATIAL_RICH
  + §169 P4 0% MCTS-128 triangulation; resurrect only for paper /
  external-review audit trail.

---

## 4. Artefacts

Pulled from 5080 vast.ai (`scp6.vast.ai:13053`) on 2026-05-09:

```
checkpoints/gpool_bias/
  A1_gpool_bias.pt                                   21.9 MB  (§170 P3 bilateral)
  A1_gpool_bias_policy_only.pt                       21.92 MB (§170 P4 P1 policy-only)
data/
  adversarial_corpus_v8.npz                          198.2 MB (§170 P4 P2; sha256 e6c1b9b9…)
reports/gpool_bias/
  argmax.json                                        554 B    (§170 P3)
  mcts64.json                                        564 B    (§170 P3)
  A1_anchor_mcts64.json                              554 B    (§170 P3 matched baseline)
  threat.json                                        434 B    (§170 P3; status=skipped)
  bench.md                                           259 B
  eval.json                                          1.86 KB  (§170 P3 combined)
  eval_outer.log                                     6.11 KB
  pretrain.log                                       276 KB
  pretrain_outer.log                                 276 KB
  policy_only/
    argmax.json                                      588 B    (§170 P4 P1)
    mcts32.json                                      591 B    (§170 P4 P1)
    mcts64.json                                      590 B    (§170 P4 P1)
    mcts128.json                                     592 B    (§170 P4 P1)
    threat.json                                      457 B    (§170 P4 P1; status=skipped)
    eval.json                                        3.26 KB  (§170 P4 P1 combined)
    *_bilateral_load.json                            (§170 P4 P1 reference; pre-fix run
                                                       without --policy-only-bias forced
                                                       at inference; argmax bit-exact,
                                                       MCTS shifts < 1 pp)
    SUMMARY.md                                       (§170 P4 P1 local manifest)
  policy_only_pretrain.log                           ~290 KB
  policy_only_pretrain.outer.log                     ~290 KB
  policy_only_eval.outer.log                         ~7 KB
  adversarial_manifest.md                            (§170 P4 P2 corpus manifest)
  adversarial_stats.json                             (§170 P4 P2 per-source stats)
  adversarial_gen.log                                (§170 P4 P2 generation log)
  SUMMARY.md                                         (this file)
```

Pretrain walls: §170 P3 1h 48m + §170 P4 P1 1h 48m on 5080.
Eval walls: §170 P3 28 min (argmax + MCTS-64 + matched anchor) +
§170 P4 P1 1h 11m (argmax + MCTS-{32, 64, 128} + bench) on 5080.
Corpus-prep wall: §170 P4 P2 ~14 min on 5080.
Bench walls: ~5 s per arm.

---

## 5. Cross-references

- §168 sprint log → eval-harness generalization + v6w25 anchor (Gate 5).
- §169 P2 → A2 results.
- §169 P3 → A3 results.
- §169 P4 → A4 results + §169 close-out matrix.
- §169a → A4 spatial-deadness probe (E2 PASS, alive).
- §170 P0 → A4 scalar-ablation (SPATIAL_RICH).
- §170 P1 → A3 MCTS-N curve (FLAT-NON-MONOTONIC).
- §170 P3 → A1 + gpool-bias retrain (FALSIFIED).
- §170 P4 P0 → A1 + gpool-bias-policy-only architecture (commit `c399a91`).
- §170 P4 P1 → A1 + gpool-bias-policy-only retrain + eval (NULL) →
  `reports/gpool_bias/policy_only/SUMMARY.md`.
- §170 P4 P2 → adversarial corpus prep (no retrain) →
  `reports/gpool_bias/adversarial_manifest.md`.
- `reports/ablation_169/DECOMPOSITION.md` → §169 four-way decomposition.
- `docs/07_PHASE4_SPRINT_LOG.md` §170 P4 close-out → Gate 6 a–g.

---
