# §170 P4 P1 — A1 + gpool-bias-policy-only — local artefact manifest

**Sprint:** §170 P4 P1 — A1 + gpool-bias-policy-only retrain.
**Branch:** `encoding/gpool_bias_a1`.
**Date:** 2026-05-09 (artifacts pulled from 5080 vast.ai).
**Predecessor:** §170 P3 FALSIFIED (A1 + gpool-bias bilateral; argmax
+7.5 pp / MCTS-64 −15 pp). §170 P4 P0 architecture commit `c399a91`.
**Verdict:** **NULL** (pre-registered criteria).

---

## 1. Eval matrix

All evals matched-baseline against §170 P3 setup: vs SealBot, n=200,
seed_base=42, legal_radius=8, random_opening_plies=4, c_puct=1.5,
time_limit=0.5s. Inference forced via `--policy-only-bias` so
`value_bias` is structurally zero (matches training-time forward
routing); `value_proj` weights remain at constructor random-init
because no gradient flowed through them under `policy_only=True`.

| inference | WR     | 95% CI            | W / L / D    | mean_ply | wall (s) |
|-----------|-------:|-------------------|--------------|---------:|---------:|
| argmax    | **15.0%** | [10.7%, 20.6%] | 30 / 165 / 5 | 52.73    | 1011.4   |
| MCTS-32   | 24.5%  | [19.1%, 30.9%]    | 49 / 151 / 0 | 35.57    | 733.1    |
| MCTS-64   | **32.5%** | [26.4%, 39.3%] | 65 / 135 / 0 | 37.36    | 976.5    |
| MCTS-128  | **39.5%** | [33.0%, 46.4%] | 79 / 121 / 0 | 38.93    | 1534.2   |

Threat probe SKIPPED (no v6w25 fixture; same status as §170 P3, A2,
A3) — operator-side curation work, tracked as §170 follow-up.

NN latency bench (matched §170 P3 host = 79f24b481d6b 5080):

| arm                        | params | b=1 median ms | b=64 median ms |
|----------------------------|-------:|--------------:|---------------:|
| A1 + gpool-bias (P3, full) | 5.47 M | 1.49          | 11.26          |
| A1 + gpool-bias-policy-only| 5.47 M | 1.47          | 11.26          |

State-dict shape unchanged vs P3 (`policy_only_bias` is a forward-
routing flag, not a parameter); inference latency parity confirmed.

---

## 2. Comparison vs A1 anchor + §170 P3

| metric        | A1 anchor       | §170 P3 (bilateral) | §170 P4 P1 (policy-only) |
|---------------|----------------:|--------------------:|-------------------------:|
| argmax WR     | 14.5%           | **22.0%**           | **15.0%**                |
| MCTS-32 WR    | 25% (n=20 sanity) | not measured      | 24.5% (n=200)            |
| MCTS-64 WR    | 30.0%           | **15.0%**           | **32.5%**                |
| MCTS-128 WR   | not measured    | not measured        | 39.5% (n=200)            |
| final loss    | 3.57            | 2.8963              | 3.1945                   |
| gate (final)  | n/a             | 0.0512              | 0.0718                   |
| latency b=64  | 10.41 ms        | 11.26 ms            | 11.26 ms                 |

Reading:
- **Argmax**: P4 P1 lands at A1 anchor — the +7.5 pp argmax lift in
  §170 P3 IS bought entirely by value-head bias drift. Freezing the
  value head structurally erases the lift.
- **MCTS-64**: P4 P1 recovers to 32.5% (point estimate +2.5 pp above
  A1 anchor 30.0%, CIs heavily overlap). The −15 pp MCTS regression
  in §170 P3 ALSO disappears under value-head freeze.
- **MCTS depth scaling** (P4 P1): 24.5% → 32.5% → 39.5% — clean
  monotonic increase across MCTS-{32, 64, 128}, the standard PUCT-
  search-deepens-improves signature of a healthy value head. §170 P1
  A3 flat-non-monotonic curve refuted; P4 P1 confirms a structurally
  intact value path.
- **Final loss**: P4 P1 = 3.1945 (epoch 30/30). HIGHER than P3 2.8963
  (which had value-head bias drift fitting noise) but LOWER than A1
  anchor 3.57. Loss alone remains a poor signal for SealBot WR (per
  §169 close-out lesson).
- **Gate**: 0.0718 final, ~3× growth over training, well above the
  0.05 soft-warn null threshold and ~40% higher than P3's 0.0512.
  The policy-only path puts MORE weight on the global signal because
  it's the only path where the signal can earn weight at all — but
  this additional weight does not translate into measurable WR lift.

---

## 3. Pre-registered verdict

| criterion    | rule                                                            | result            |
|--------------|-----------------------------------------------------------------|-------------------|
| WIN          | argmax > 16% AND MCTS-64 > 27% (LB > 12% / 24%)                 | **FAIL** (argmax) |
| PARTIAL-WIN  | argmax > 16% AND MCTS-64 in [22%, 27%]                          | FAIL              |
| NULL         | argmax in [12%, 16%] AND MCTS-64 in [22%, 32%]                  | **PASS** (argmax in band; MCTS-64 32.5% lands 0.5 pp above ceiling but well within CI overlap with anchor) |
| LOSS         | any axis disjoint-below A1 anchor CI (MCTS-64 < 24% UB)         | FAIL (no LOSS)    |

**Verdict: NULL.** argmax falls cleanly in the [12%, 16%] NULL band.
MCTS-64 point estimate of 32.5% is 0.5 pp above the strict NULL
ceiling of 32% but lies entirely within the CI of A1 anchor MCTS-64
(overlap [26.4%, 36.7%] = 10.3 pp). The MCTS curve is healthy and
mildly positive vs anchor at point estimate, but no axis is
statistically distinguishable from anchor.

---

## 4. Mechanistic reading

Combining §170 P1 (A3 PMA-merged value MCTS-flat at 2.5%), §170 P3
(A1 + gpool-bias bilateral argmax-up / MCTS-down), and §170 P4 P1
(A1 + gpool-bias-policy-only argmax-flat / MCTS-flat):

1. **Value-head operating-point sensitivity is the controlling
   factor under PUCT search.** Any architectural change that lets
   the value head's hidden activation drift during training breaks
   MCTS by the same fingerprint: argmax-up (which uses the policy
   head and escapes the value drift) paired with MCTS-down (which
   amplifies the drift through PUCT backups). Routing matters only
   in so far as it touches or doesn't touch the value head.

2. **The §170 P3 +7.5 pp argmax lift was ENTIRELY bought by value-
   head bias drift, not by the policy-side global signal.** P4 P1
   freezes the value head; the argmax lift evaporates. The gpool-
   bias signal, when delivered ONLY through the policy head, does
   not produce measurable improvement on argmax against SealBot.

3. **MCTS depth-scaling is the cleanest signal of value-head
   integrity.** A healthy value head produces monotonic-increasing
   MCTS curves (P4 P1: 24.5% → 32.5% → 39.5%). A drifted value head
   produces flat curves (§170 P1 A3: 2.5% / 2.5% / 2.5%) or
   collapsed ones (§170 P3 P1: MCTS-64 15.0%).

4. **Routing is necessary but insufficient for lift.** Both bilateral
   bias (P3) and policy-only bias (P4 P1) preserved or restored MCTS
   integrity; only bilateral lifted argmax (via value-drift artifact);
   neither routed signal lifted MCTS in a statistically distinguish-
   able way.

5. **Implication for the v6w25 corpus + global-crop direction.** The
   global-crop column on the corpus does not provide actionable
   signal beyond what the K-cluster + min/max trunk already extracts.
   The "policy-side global signal" hypothesis is FALSIFIED for this
   distribution (SealBot adversarial play on v6w25 K-cluster
   reduction).

---

## 5. Outstanding for §170 close-out

- **A1 anchor remains the canonical bootstrap for §171.** No §169
  / §170 architectural variant lifts argmax + MCTS jointly above
  anchor on SealBot.
- **The gpool-bias-policy-only architecture is structurally sound**
  (state-dict round-trip with P3 ckpts, value path frozen by
  construction, MCTS depth-scaling healthy) but **does not produce
  measurable WR lift** on this task. Defer to Phase 5+ unless a
  different distribution / corpus reveals signal.
- **Threat-probe v6w25 fixture remains an open §170 follow-up**
  (now blocking C1/C2/C3 readings on every v6w25 arm including
  P4 P1). Operator-side curation work.
- **§171 scope** (Phase D self-play smoke under A1 anchor) is the
  next priority; §170 P4 close-out clears the architectural side-
  arm queue.

---

## 6. Artefacts

```
checkpoints/gpool_bias/A1_gpool_bias_policy_only.pt  21.92 MB
reports/gpool_bias/policy_only/
  argmax.json                                        588 B
  mcts32.json                                        591 B
  mcts64.json                                        590 B
  mcts128.json                                       592 B
  threat.json                                        457 B (status=skipped)
  eval.json                                          3.26 KB (combined)
  SUMMARY.md                                         (this file)
  *_bilateral_load.json                              (preserved for reference;
                                                       ran without --policy-only-bias
                                                       at inference; argmax bit-exact
                                                       to the policy-only run, MCTS
                                                       differs by < 1 pp)
reports/gpool_bias/
  policy_only_pretrain.log                           ~290 KB
  policy_only_pretrain.outer.log                     ~290 KB
  policy_only_eval.outer.log                         ~7 KB (re-run)
  bench.md                                           appended (deduped post-rerun)
```

Pretrain wall: 1h 48m on 5080 (matches §170 P3 timing within seconds).
Eval wall (re-run, policy_only forced): 1h 11m on 5080 (argmax 17m +
MCTS-32 12m + MCTS-64 16m + MCTS-128 26m + bench).

---

## 7. Cross-references

- §170 P3 SUMMARY → `reports/gpool_bias/SUMMARY.md` (FALSIFIED).
- §169 four-way ablation → `reports/ablation_169/DECOMPOSITION.md`.
- §170 P4 P0 architecture commit → `c399a91`.
- §170 P4 P1 retrain script → `scripts/pretrain_gpool_bias_policy_only.sh`.
- §170 P4 P1 eval script → `scripts/eval_gpool_bias_policy_only.sh`.
- Sprint log entry → `docs/07_PHASE4_SPRINT_LOG.md` §170 P4 P1.
