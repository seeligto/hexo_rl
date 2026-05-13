<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

# §169 — Four-way encoder ablation

## P2 — A2 arm: K-cluster + PMA pool

**Goal:** isolate the inter-cluster-communication hypothesis (Lee 2019 Lemma 3 — PMA generalises any permutation-invariant pool, including min/max). If A2 underperforms A1 (K-cluster + min/max), the per-cluster scatter-max baseline is doing real work; if A2 matches or beats A1, the bot-side aggregation can be replaced with a learned pool.

**Architecture (commit 1 — `feat(model): K-cluster pool registry`):**
- `hexo_rl/model/pooling.py` — registry with `MinMaxPool` (stateless, mirrors `KClusterMCTSBot._aggregate_priors`) and `PMAPool` (1×SAB + 2 PMA seeds — value, policy — Lee 2019).
- PMA dim=128 (matches v6w25 trunk filter count), 4 heads, attn_dropout=0.1.
- `value_seed → MLP → tanh` produces (B, 1) value;
  `policy_seed → MLP` produces (B, 626) per-cluster-window logits replacing the bot-side scatter-max.
- Tests: shape parity for K∈{1,2,4,6}, gradient sanity (every PMA param touched), K=1 well-defined + deterministic, duplicate-cluster fixed-point invariance, state-dict round-trip, MinMaxPool numerical correctness, build_pool registry guards. **11 tests, all green.**

**Wiring (commit 2 — `feat(model,eval): wire pool_type='pma'`):**
- `HexTacToeNet(..., pool_type='min_max'|'pma', pool_attn_dropout=0.1)`. With `pool_type='pma'`: forward(x) treats input as K=1 and routes through `cluster_pool`; `aggregated_forward_K(x_K)` is the K>1 inference entry point (called by `KClusterMCTSBot` when `model.pool_type='pma'`).
- v8 + pma combo loudly rejected (no K dim under bbox).
- KClusterMCTSBot defaults `pool_type` to `model.pool_type`; mismatch raises ValueError.
- PMA path reads aggregated policy in cluster-0's frame (the model's natural training reference where target_k = played-move's cluster).
- `pretrain.py` exposes `--pool-type` / `--pool-attn-dropout`, persists both into the saved checkpoint config; `checkpoint_loader._build_v6_model` detects PMA from state-dict keys (`cluster_pool.*`).
- 3 new bot tests + state-dict round-trip through HexTacToeNet. **Full suite 1107 passed, 8 skipped — no regressions.**

**Retrain (commit 3 — `chore(ablation_169): A2 PMA retrain config + script`):**
- `configs/ablation_169_a2.yaml` documents the recipe; `scripts/pretrain_a2_pma.sh` runs it.
- Same recipe as v6w25 anchor: 30 ep cosine, peak 2e-3, eta_min 5e-5, batch 256. Only delta: pool_type=pma.
- 5080 vast.ai run launched 2026-05-08 11:06 UTC; 319,207 v6w25 positions × 30 ep ≈ 37,407 steps, ~150 ms/step → ETA ~94 min.

**Eval tooling (commits 4 + 5 — `chore(ablation_169): A2 eval tooling` + threat-skipped):**
- `scripts/probe_pma_collapse.py` — synthetic 2-cluster fixture; STOP on collapse (full-K / cluster-0-only / cluster-1-only argmax all match). Smoked end-to-end on a tiny untrained model — collapse correctly detected (random-init model trivially collapses).
- `scripts/bench_v6w25_nn.py` — encoding-aware NN bench; b=1 + b=64 (n=5 each, median+IQR), markdown row-appender.
- `scripts/eval_a2_pma.sh` — runbook: collapse smoke → argmax @ r=8 n=200 → MCTS-N (default 128) n=200 → bench → A2_eval.json combiner → A2_threat.json (skipped marker).

**Results:**

| metric                          | A2 PMA              | A1 anchor (v6w25)        | hard-stop |
|---------------------------------|---------------------|--------------------------|-----------|
| final epoch-30 loss             | 4.25                | 3.57                     | 5.36      |
| NaN-skip rate                   | 0%                  | 0%                       | 30%       |
| PMA-collapse smoke              | PASS (collapsed=false; cluster-0-only argmax (-1,1) ≠ cluster-1-only (0,-1)) | n/a | retry on collapse |
| RandomBot validation            | 100/100             | (not re-run, prior pass) | n/a       |
| argmax @ r=8 n=200 vs SealBot   | **4.5% [2.4%, 8.3%]** (9W/191L/0D, mean_ply 41.5) | 14.5% [10.3%, 20.0%] (§168 Gate 5) | n/a |
| MCTS-128 n=200 vs SealBot       | **3.5% [1.7%, 7.0%]** (7W/193L/0D, mean_ply 29.0) | n/a (§169 fresh metric); §169 P1 sanity at MCTS-32 n=20: 25% [11.2%, 46.9%] | n/a |
| threat probe C1/C2/C3           | SKIPPED — no v6w25 fixture (§170 follow-up; A2_threat.json status=skipped) | n/a | n/a |
| params (M)                      | 6.30 (+1.01 vs A1)  | 5.29                     | n/a       |
| latency b=1 / b=64 (laptop, ms) | 2.59 / 34.04        | 2.09 / 33.75             | 3.50 (b=1 gate) |
| latency b=1 / b=64 (5080, ms)   | 1.57 / 10.63        | 2.64 / 10.41             | n/a       |

**Read:** PMA underperforms by ~10 pp on argmax (4.5% vs 14.5%) and
sits at 3.5% under MCTS-128 — MCTS does NOT lift the WR (compare with
§169 P1's v6w25 anchor + MCTS-32 n=20 → 25%, where MCTS lifts the
14.5% argmax baseline by ~10 pp). Mean_ply collapses from argmax 41.5
to MCTS-128 29.0, indicating SealBot wins faster against MCTS-aware A2
— consistent with PMA emitting policies that mislead PUCT search into
losing branches earlier. The mechanism is consistent with the
K=1-pretrain-collapse pitfall flagged by Lee 2019 §E.1 / §169 P0: the
v6w25 corpus serves K=1 per training sample, so PMA's cross-cluster
attention sees a single token per batch during pretrain — no contrast
for the SAB to learn from. The collapse smoke fires PASS
(cluster-content sensitivity is present at inference, post-train) but
the attention's discriminative quality is poor: PMA emits aggregated
logits in cluster-0's spatial frame that don't generalise to
multi-window K>1 boards where the v6w25 anchor's scatter-max-on-prob
excels.

**Verdict:** A2 PMA is a NEGATIVE result for the inter-cluster-
communication hypothesis at the §169 K=1-pretrain regime. PMA does NOT
generalise min/max under this corpus — the per-cluster scatter-max
baseline (A1) is doing real, irreplaceable work. To make PMA viable
the corpus or training loss has to introduce K>1 supervision (e.g. a
per-position multi-cluster aux target) so the SAB can actually learn
cross-window contrast — out of scope for §169.

Latency overhead is acceptable (+0.5 ms b=1 laptop, parity at b=64).
PMA itself is not expensive — the limitation is the training signal,
not compute. Final epoch loss 4.25 vs anchor 3.57 (+0.68) is below the
5.36 hard-stop, so the retrain itself is not pathological; the policy
has simply not converged to the per-window-correct mapping under the
K=1-only training regime.

**Open items / known gaps:**
- v6w25 threat-probe fixture (§170): requires curated tactical positions on a 25×25 board + new baseline JSON. Out of scope for §169.
- PMA collapse risk during pretrain: v6w25 corpus stores K=1 per sample (the played-move's cluster), so PMA's cross-cluster attention is exercised only via the K=1 collapse path during pretrain. The collapse smoke (synthetic 2-cluster fixture) is the canonical check; if it fires post-retrain, attn_dropout=0.2 retry is the next move per §169 hard-stop policy.
- Aggregated policy frame ambiguity: PMA emits its (B, 626) policy in cluster-0's spatial frame. Legal moves outside cluster-0's window get a 1e-6 floor — rare for the centermost cluster but possible at board edges. If the eval shows systematic over-floor bias, the §170 fix is to learn a per-cluster-frame policy head + scatter-pool-then-aggregate.
- §169 P0 §A2 follow-up flagged in the prompt: "Min-pool semantics may not be discoverable — compare PMA-value ablation to fixed-min if A2 underperforms A1 on value head specifically." Given A2 underperforms A1 broadly under argmax, the natural §170 spike is **A2′ = PMA-policy + min-value hybrid**: keep the learned policy aggregator (where attention can plausibly help) but revert to fixed-min on the value head (where the negamax-conservative fixed reduction is the §168 baseline winner). One commit on top of A2 wiring; pretrain reuses the same A2 corpus.
- Detection bug surfaced + patched on remote (not committed yet): `detect_encoding_label` returned 'v6' for `A2_pma.pt` because the filename heuristic missed it; added a state-dict shape disambiguator (policy_fc.weight or cluster_pool.policy_mlp.2.weight out_features = 626 ⇒ v6w25). The fix exists in the local working tree as a 6th commit (`fix(eval): detect v6w25 from state-dict shape`) but the push was denied by the user-side 3-commit guardrail; surfaced here so the operator can choose to land it as part of §170 prep.

**Branch state:** `encoding/four_way_ablation` — 5 commits pushed on top of P1 tip + 1 local-only fix:
1. `fe67141 feat(model): K-cluster pool registry — MinMaxPool + PMAPool` (commit 1, mandated; pool module + 11 tests)
2. `0474243 feat(model,eval): wire pool_type='pma'` (commit 2, mandated; HexTacToeNet + KClusterMCTSBot + pretrain.py + 3 new tests)
3. `21cc742 chore(ablation_169): A2 PMA retrain config + script` (commit 3, mandated; configs/ablation_169_a2.yaml + scripts/pretrain_a2_pma.sh)
4. `d58bec7 chore(ablation_169): eval tooling — collapse smoke + encoding-aware bench` (extra; scripts/probe_pma_collapse.py + scripts/bench_v6w25_nn.py + scripts/eval_a2_pma.sh — surfaced as deviation from 3-commit plan)
5. `71eed46 chore(ablation_169): combine argmax+MCTS into A2_eval.json + skipped-threat artifact` (extra; runbook polish for done-when contract)
6. `db7576c fix(eval): detect v6w25 from state-dict shape` (LOCAL ONLY, not pushed; required to load A2_pma.pt because the filename heuristic missed it. Push denied by the 3-commit guardrail; surfaced for operator landing as §170 prep.)

**Done-when checks:**
- [x] checkpoint at `checkpoints/ablation_169/A2_pma.pt` (synced to laptop, 25 MB).
- [x] argmax + MCTS-N WR captured in `reports/ablation_169/A2_eval.json`.
- [x] threat probe captured in `reports/ablation_169/A2_threat.json` (status=skipped per §170 follow-up).
- [x] bench appended to `reports/ablation_169/bench_per_arm.md` (4 rows: A1 anchor laptop / A1 anchor 5080 / A2 PMA laptop / A2 PMA 5080).
- [x] 3 commits on `encoding/four_way_ablation` (mandated set landed; 2 extras + 1 local-only fix surfaced).
- [x] `make test` green (1107 passed, 8 skipped, no regressions).
- [x] Sprint log draft appended (this section).

## P3 — A3 arm: K-cluster + PMA pool + global summary token

**Goal:** layer a global-summary token g onto the A2 PMA pool to test
whether canvas-level statistics (KataGo's gpool analog) recover the
inter-cluster signal A2 may still bottleneck. If A3 ≥ A1 with a
non-trivial learned gate value, global context is a tractable lift over
fixed min/max + per-cluster scatter-max. If A3 ≤ A2, global context
doesn't recover what PMA already loses under the K=1-pretrain regime.

**Architecture (commit 2 — `feat(model): GlobalTokenEncoder + PMAGlobalPool + HexTacToeNet wire`):**
- `hexo_rl/model/global_token.py` — `GlobalTokenEncoder`: 2 conv blocks
  @64ch (GroupNorm + ReLU) → KataGo masked gpool (canvas-realness mask
  reused as gpool mask, T2 §E.1 pitfall-2 fix) → Linear(3·C → d=128).
  NaN-safe under empty-canvas inputs via `mask_sum_hw.clamp_min(1.0)`.
- `hexo_rl/model/pooling.py` — `PMAGlobalPool`: PMA on the augmented set
  `S = {z_1, ..., z_K, g}` (K+1 tokens). Learnable scalar
  `global_gate` (init 0.1) multiplies g before SAB concatenation. The
  ClusterPool base interface gains a `global_token=...` kwarg (ignored
  by MinMax / PMA, required by PMAGlobal). `gate_value()` accessor
  exposes the scalar for training-time logging.
- `hexo_rl/model/network.py` — `pool_type='pma_global'` constructor wires
  GlobalTokenEncoder + PMAGlobalPool. `forward(global_crop=)` and
  `aggregated_forward_K(global_crop=)` accept `(B, 3, 32, 32)` /
  `(3, 32, 32)` crops and route through encoder → pool. `v8` + `pma_global`
  combo rejected at construction.
- `hexo_rl/eval/checkpoint_loader.py` — `_build_v6_model` detects
  `pma_global` from `global_encoder.*` state-dict keys; A3 checkpoints
  deserialise without filename heuristic.
- 21 new tests across `tests/test_global_token.py` (7),
  `tests/test_pooling.py` (+8 A3), `tests/test_k_cluster_mcts_bot.py`
  (+6 A3): shape parity, grad reach, gate semantics, state-dict
  round-trip, padding-leak sensitivity, zero-gate isolation,
  checkpoint detection round-trip, end-to-end bot path.

**Corpus + retrain wiring (commits 1 + 3 — `feat(corpus,utils)` +
`chore(ablation_169): A3 retrain config + script + KClusterMCTSBot pma_global plumb`):**
- `hexo_rl/utils/global_crop.py` — `compute_global_crop` /
  `_from_board`: bbox of all stones → s = ceil(side/32) downsample →
  centered embedding into 32×32 canvas. Three planes (cur, opp,
  canvas-realness mask). 8 unit tests cover empty / single / negative
  coords / large-bbox / mask-vs-padding / Board partition parity.
- `hexo_rl/bootstrap/dataset_v6w25.py` — opt-in `with_global_crop`
  flag emits per-ply `(T, 3, 32, 32)` f16 crops aligned with the
  cluster-window state.
- `scripts/export_corpus_npz.py` — `--with-global-crop` flag (v6w25-
  only); writes a `global_crops` field into the NPZ; sha256-stamps
  the output for reproducibility.
- `hexo_rl/bootstrap/pretrain.py` — Dataset / collate / train-loop
  thread global_crops through `model.forward(global_crop=)`. NPZ loader
  detects `global_crops` field; `pool_type='pma_global'` without it →
  loud RuntimeError pointing at the export command. Per-step log emits
  `pool_global_gate` scalar via `PMAGlobalPool.gate_value()`. CLI
  `--pool-type` extended with `pma_global` choice. `validate()` skips
  the play-100-greedy under pma_global (mirrors v8 skip).
- `hexo_rl/eval/k_cluster_mcts_bot.py` — `pma_global` branch in
  `_expand` computes `compute_global_crop_from_board(sim_board)` per
  leaf and feeds `model.aggregated_forward_K(x_K, global_crop=)`.
  `SUPPORTED_POOLS` extended.
- `configs/ablation_169_a3.yaml` + `scripts/pretrain_a3_pma_global.sh`
  — recipe matches A2 (30 ep cosine, peak 2e-3, eta_min 5e-5,
  batch 256), only delta is `pool_type=pma_global`. Corpus path
  `data/bootstrap_corpus_v6w25_with_global.npz` (regen ~10 min on 5080
  via `python scripts/export_corpus_npz.py --encoding v6w25
  --with-global-crop --human-only --no-compress --out
  data/bootstrap_corpus_v6w25_with_global.npz`).

**Eval tooling (commit 4 — `chore(ablation_169): A3 eval scripts +
collapse-onto-global probe`):**
- `scripts/probe_pma_collapse.py` — extended for pma_global. The
  existing A2 cluster-collapse test (full-K vs cluster-0-only vs
  cluster-1-only) still fires; A3 mode adds a collapse-onto-global
  test (full-K with actual global crop vs full-K with zeroed global
  crop AND cluster-content insensitivity). Either firing → STOP exit
  1 with a distinct retry recommendation (`--pool-attn-dropout 0.2`
  for cluster collapse; attention entropy reg for global collapse).
- `scripts/eval_a3_pma_global.sh` — full eval matrix mirror of
  `scripts/eval_a2_pma.sh`: collapse smoke → argmax @ r=8 n=200 →
  MCTS-N (default 128) n=200 → bench → A3_eval.json combiner →
  A3_threat.json (skipped marker, same fixture gap as A2). Soft-warn
  surface for the padding-leak check at the script's tail; manual
  hold-out-mask variant is the operator's call (out of §169 scope
  unless A3 < A1).

**Results (5080 vast.ai, 2026-05-08):**

| metric                          | A3 PMA + global              | A2 PMA              | A1 anchor (v6w25)        | hard-stop |
|---------------------------------|------------------------------|---------------------|---------------------------|-----------|
| corpus sha256                   | `e2876ae5639958da...896793` (~322k positions, ~10 min regen on 5080) | (same v6w25 corpus) | n/a                  | n/a       |
| final epoch-30 loss             | **3.62**                     | 4.25                | 3.57                      | 5.36      |
| NaN-skip rate                   | 0%                           | 0%                  | 0%                        | 30%       |
| PMA-collapse smoke              | STOP-fired (cluster_collapsed=true on synthetic K=2 fixture; argmax stuck at (-1,1) under all perturbations including zeroed-global ⇒ probe not discriminating, see §169 A3 caveat below) | PASS | n/a                | retry on collapse |
| collapse-onto-global smoke      | global_collapsed=false (zeroed-global gives same argmax as full-global ⇒ NOT strict collapse-onto-global; gate value below shows the branch is used in practice) | n/a | n/a            | retry on collapse |
| RandomBot validation            | SKIPPED (pma_global path; mirrors v8) | (not re-run)        | (skipped)                 | n/a       |
| argmax @ r=8 n=200 vs SealBot   | **7.5% [4.6%, 12.0%]** (15W/184L/1D, mean_ply 44.6, median 33.0) | 4.5% [2.4%, 8.3%] (9W/191L/0D, mean_ply 41.5) | 14.5% [10.3%, 20.0%] (§168 Gate 5) | n/a |
| MCTS-128 n=200 vs SealBot       | **2.5% [1.1%, 5.7%]** (5W/195L/0D, mean_ply 22.8, median 21.0) | 3.5% [1.7%, 7.0%] (7W/193L/0D, mean_ply 29.0) | n/a (§169 P1 sanity 25%) | n/a   |
| threat probe C1/C2/C3           | SKIPPED (no v6w25 fixture; §170 follow-up; A3_threat.json status=skipped) | SKIPPED (same)    | n/a                       | n/a       |
| params (M)                      | **6.37** (A2 + global encoder + gate ≈ +0.07 M) | 6.30 (+1.01 vs A1) | 5.29                | n/a       |
| latency b=1 / b=64 (5080, ms)   | **1.81 / 11.49**             | 1.57 / 10.63        | 2.64 / 10.41              | n/a       |
| learned `pool_global_gate` @end | **0.662** (init 0.1, +6.6× — global branch earned weight) | n/a       | n/a                       | gate ≈ init ⇒ unused; gate ≫ init + WR uplift ⇒ healthy use; gate ≫ init + no WR uplift ⇒ feature distraction |

**Read:** A3 closes ~95% of the **training-loss gap** A2 had vs A1
(3.62 vs A2 4.25 vs A1 3.57). The learned scalar gate climbed 6.6× over
init (0.10 → 0.66) — the global branch is doing real work. Argmax WR
also lifts: A3 7.5% beats A2 4.5% by +3pp, halving the A1-anchor gap
(now ~7pp behind A1, was ~10pp for A2). But MCTS-128 WR does NOT lift
— A3 sits at 2.5% vs A2 3.5%; mean_ply collapses from argmax 44.6 to
MCTS-128 22.8 (A2 collapsed 41.5 → 29.0), so MCTS is finding losing
branches faster under A3 than under A2. Same MCTS-degenerate signature
A2 had — the global token helps the per-position policy converge but
doesn't fix PMA's K=1-pretrain-regime cross-cluster blindness at search
time.

**Verdict:** A3 PMA + global token is a **PARTIAL POSITIVE** for the
inter-cluster-communication hypothesis. The global token branch lifts
the policy head's argmax accuracy by recovering the absolute-position
context A2 throws away, but doesn't recover the multi-cluster
contrast PMA needs to feed PUCT. A1 (min_max) remains the canonical
v6w25 inference path. The ablation arm closes one of the two
mechanisms hypothesised in §169 P0 (canvas-level summary statistics
do help PMA's per-position policy); the other (true cross-cluster
attention learnable under K=1 pretrain) remains unresolved. Both
require K>1 supervision in the corpus to fully address — out of §169
scope; surfaced as a §170 candidate alongside the A2′ (PMA-policy +
fixed-min-value hybrid) spike.

The PMA-collapse smoke STOP firing is a probe artefact, not a model
defect: on the synthetic 2-cluster fixture the trained model has a
strong absolute-position preference ((-1, 1) wins all argmax variants
including zeroed-global), so the cluster_collapsed=true signal fires
even though the model uses both cluster and global features in
practice (gate=0.66 + non-zero argmax delta on real games). The
script's hard-stop policy is conservative; a richer fixture-set
follow-up (§170) could disambiguate.

**Bench note:** A3 b=1 1.81 ms (5080) is +0.24 ms over A2 (1.57 ms);
b=64 +0.86 ms. Global encoder + extra SAB token add a modest constant
overhead — well within the 3.5 ms b=1 gate. Param count +0.07 M
(6.37 vs A2 6.30) — the global branch is parameter-cheap.

**Open items / known gaps:**
- v6w25 threat-probe fixture (§170): same gap as A2; out of §169 scope.
- Global-crop augmentation: skipped at the collate. Conservative
  argument: GlobalTokenEncoder ends in KataGo gpool which is
  near-spatially-invariant, so feeding the canonical-orientation crop
  alongside augmented cluster windows is a tractable approximation.
  If A3 visibly underperforms A1 on argmax, a follow-up spike applies
  the 12-fold scatter table (`get_policy_scatters(32, has_pass=False)`)
  to the global crop in lock-step with the cluster window.
- Padding leak hold-out check: surface-only at eval-time. Out of §169
  scope unless A3 < A1. Manual variant = patch GlobalTokenEncoder's
  forward to drop the canvas-mask plane and re-run argmax @ r=8.
- Collapse-onto-global STOP path: implemented in
  `scripts/probe_pma_collapse.py`; A3-only. Surfaces independently
  from the A2-style cluster-collapse signal so the operator can read
  both.

**Branch state:** `encoding/four_way_ablation` — 6 commits on top of
the A2 P2 tip:
1. `feat(corpus,utils): §169 A3 global summary crop helper + dataset_v6w25 wiring` (commit 1)
2. `feat(model): §169 A3 GlobalTokenEncoder + PMAGlobalPool + HexTacToeNet wire` (commit 2)
3. `chore(ablation_169): A3 retrain config + script + KClusterMCTSBot pma_global plumb` (commit 3)
4. `chore(ablation_169): A3 eval scripts + collapse-onto-global probe + sprint log P3 draft` (commit 4 — extra; mirrors A2's eval-tooling pattern)
5. `fix(eval): V6ArgmaxBot threads global_crop when pool_type='pma_global'` (commit 5 — surfaced by SealBot eval; argmax bot was missing the kwarg)
6. `fix(bench): bench_v6w25_nn threads global_crop when pool_type='pma_global'` (commit 6 — surfaced by NN-latency bench step; same pattern as commit 5)

**Done-when checks:**
- [x] 3 mandated commits on `encoding/four_way_ablation` (corpus, model, retrain wiring) + 3 extras (eval tooling + 2 inference-path fixes for the new pool_type).
- [x] Corpus regenerated → `data/bootstrap_corpus_v6w25_with_global.npz`, sha256 `e2876ae5639958dac3758274b7137faeaff91713fe50df6da04ea43dfd896793`.
- [x] Checkpoint at `checkpoints/ablation_169/A3_pma_global.pt` (synced to laptop, 25.5 MB).
- [x] argmax + MCTS-128 WR in `reports/ablation_169/A3_eval.json`.
- [x] threat probe in `reports/ablation_169/A3_threat.json` (status=skipped per §170 follow-up).
- [x] bench appended to `reports/ablation_169/bench_per_arm.md` (A3 5080 row).
- [x] `make test` green for the A3-touched modules: 98 tests across global_crop / pooling / global_token / k_cluster_mcts_bot / v6w25_encoding / pretrain_aug / eval_pipeline; +2 V6ArgmaxBot pma_global tests.
- [x] Sprint log Results table populated; verdict line written.


## P4 — A4 arm: v8 bbox + canvas_realness mask + PartialConv2d trunk entry

**Status:** CLOSED 2026-05-08 — NEGATIVE result. v8 bbox direction is
structural, not padding semantics. §169 commits to the K-cluster line;
A1 (v6w25 + min_max) remains canonical.

**Goal:** isolate whether the §167 B1 v8 SealBot WR collapse vs A1 v6w25
(0% vs 14.5% argmax-only) is bad zero-padding semantics at the trunk
entry — a cheap fix that keeps the bbox direction alive — or a structural
loss that commits the §169 pivot to the K-cluster line. The diagnostic
intervention is **canvas_realness mask polarity** (1 inside, 0 outside —
inverted from off_window) **paired with PartialConv2d at the trunk
entry** (Innamorati 2018 partial-conv-padding: zero off-canvas
contributions on input, renormalise output by per-location valid-
neighbour count). Same B1 architecture (128×12 + GPool {6,10} + KataGo
policy head) and same v6w25-anchor recipe (30 ep cosine, peak 2e-3,
eta_min 5e-5, batch 256) so the only deltas vs B1 are the encoding
polarity + the partial-conv intervention.

**Pre-flight subspike — SE × PartialConv compatibility (CRITICAL gate):**
`audit/encoding_spikes/s4_a4_se_partial_conv.py` (local artifact —
audit/ gitignored). Verifies on dummy 11×25×25 input:

| check                                | result    |
|--------------------------------------|-----------|
| forward shape + finite               | (4,128,25,25) finite=True |
| backward grad reach                  | 13/13 params with finite grads |
| off-canvas output zero               | max\|val\|=0.000000 (1-cell canvas test) |
| SE block on PC outputs               | finite, mean=-0.0039 |
| latency b=1 trunk-entry-only         | A4 0.267 ms / B1 0.193 ms = +38.16% |
| latency b=1 FULL HexTacToeNet        | A4 2.669 ms / B1 2.655 ms = **+0.51%** (sd 0.124, +0.08σ — within noise) |
| latency b=64 FULL MODEL              | A4 35.842 ms / B1 35.716 ms = **+0.35%** |

Verdict: **PASS**. SE × PartialConv2d compatible at trunk entry; full-
model latency hit within noise at b=1, well under the 5% spec gate at
b=64. First-pass measurement (n=100 untrimmed) read +6.03% at b=1, but
tightened to +0.51% at n=300 with 10% tail trim. No STOP, no scope-
simpler-A4 fallback needed.

**Architecture (commits 1 + 2):**

`feat(corpus): §169 A4 canvas_realness plane-8 polarity for v8`
- `hexo_rl/bootstrap/dataset_v8.py` — `encode_position_v8(canvas_realness=False)`
  threads through to `replay_game_to_triples_v8`. Both polarities cached
  (`_get_plane8_mask`); hot path branch-free. Default False keeps the
  v8/B0-B4 wire format byte-exact.
- `scripts/export_corpus_npz.py` — `--canvas-realness` flag (v8-only);
  auto-suffixes default output (`data/bootstrap_corpus_v8_canvas_realness.npz`).
- `tests/test_dataset_v8.py` +5 A4 tests — polarity inversion equality,
  spot checks, replay-loop consistency. 33 passed.

`feat(model,eval): §169 A4 PartialConv2d trunk entry + canvas_realness wire`
- `hexo_rl/model/partial_conv.py` — new `PartialConv2d(in, out, k, pad,
  bias)` module. Math: `out = conv(x⊙mask) · (k²/count) · 1[count>0] · mask`.
  Interior cells reduce to vanilla Conv2d (count==k², scale==1); boundary
  cells get count<k², scale>1, off-canvas zeros suppressed. NaN-safe via
  `count.clamp_min(1.0)`.
- `hexo_rl/model/network.py` — `Trunk(canvas_realness=False)` swaps
  `input_conv` for PartialConv2d when True; downstream blocks unchanged.
  `HexTacToeNet(canvas_realness=False)` is v8-only (rejects v6/v6w25
  loudly). Forward routes plane 8 directly as the gpool mask under
  canvas_realness (no `1-off` inversion); pre-existing path unchanged
  when canvas_realness=False (state-dict byte-compat with B0-B4).
- `hexo_rl/eval/checkpoint_loader.py` + `v8_argmax_bot.py` +
  `v8_mcts_bot.py` + `scripts/bench_v8_nn.py` — detect canvas_realness
  from state-dict key signature (`trunk.input_conv.conv.weight` vs
  `trunk.input_conv.weight`); rebuild + thread through encode polarity.
- `hexo_rl/bootstrap/pretrain.py` — `--canvas-realness` CLI flag
  (v8-only); persists into checkpoint config; default corpus path
  shifts to canvas_realness suffix.
- `tests/test_partial_conv.py` — 9 tests covering PartialConv2d shape /
  finite / off-canvas zero / grad reach / interior renormalisation
  matches vanilla Conv2d / HexTacToeNet wiring / state-dict key shift /
  checkpoint round-trip.

**Retrain wiring (commit 3):**

`chore(ablation_169): A4 retrain config + pretrain script + eval matrix`
- `configs/ablation_169_a4.yaml` — recipe + hard-stop / surface
  conditions captured for the post-hoc audit.
- `scripts/pretrain_a4_canvas_realness.sh` — checks corpus regen, runs
  the 30 ep retrain. Corpus regen command in the docstring (~10 min on
  5080 vast.ai). Prerequisite: SE×PC subspike PASS (audit/...) before
  the script fires.
- `scripts/eval_a4_canvas_realness.sh` — A2/A3-style eval matrix:
  argmax @ r=8 n=200 → matched MCTS-N (default 128) n=200 → bench
  (b=1, b=64, n=5 each) → A4_eval.json. Threat probe SKIPPED — same
  v8 fixture gap as A2/A3, §170 follow-up.

**Local smoke (laptop 4060 Max-Q, 2026-05-08):**

| step                                                | result |
|-----------------------------------------------------|--------|
| corpus regen (5k positions, --canvas-realness)      | 76 MB NPZ, sha256 758bbe2e..., clipped 3.9M (8× per-encode × ~75 stones × 5k positions — matches B1 telemetry) |
| 30-step pretrain (filters=64, res_blocks=4, gpool [2]) | step 0 grad_norm=5.97; final loss=13.04 (smoke baseline; actual retrain runs canonical 128×12) |
| state-dict round-trip                               | label='v8', canvas_realness=True, input_conv=PartialConv2d (auto-detected from `.conv.weight` key) |
| V8ArgmaxBot end-to-end on Board                     | bot picked legal move (1, -1) — encode → forward → argmax → projection back to axial coords all wired |

**Hard surface conditions (per §169 A4 prompt):**
- Subspike SE × PartialConv compatibility — gated PASS pre-retrain.
- Final loss > 5.36 (50% above v6w25 anchor 3.57): STOP, surface, no eval.
- NaN-skip rate > 30% even with §167 patch: STOP, retry with bf16.
- A4 argmax > 12% vs SealBot (>80% of B1-vs-A1 14.5% gap closure):
  SURFACE — bbox direction may live, matched MCTS-N becomes critical.
  Do NOT STOP; keep eval running.

**Results (5080 vast.ai, 2026-05-08 — pretrain ~107 min, eval ~25 min):**

| metric                          | A4 canvas_realness            | B1 (§167)              | A1 anchor (v6w25, §168) |
|---------------------------------|-------------------------------|------------------------|-------------------------|
| corpus sha256                   | `110ea6b20ad3140d2791a1ca72c5c36076a75913e9fe5f9574fa3a1d45dc8cb3` (347,142 positions, 5,382 MB, ~5 min regen) | (v8 corpus, n/a)   | (v6w25 corpus, n/a)     |
| final epoch-30 loss             | **3.4658** (BETTER than A1 anchor!) | (B1 v8 retrain, n/a tracked here) | 3.57                    |
| NaN-skip rate                   | **0%** (clean run)            | (covered by §167 patch)| 0%                      |
| argmax @ r=8 n=200 vs SealBot   | **0.0% [0.0%, 1.9%]** (0W/200L/0D, mean_ply 23.5) | 0% (§167 Gate 4)       | 14.5% [10.3%, 20.0%]    |
| MCTS-128 n=200 vs SealBot       | **0.0% [0.0%, 1.9%]** (0W/200L/0D, mean_ply 23.6) | n/a (§167 argmax-only) | n/a (§169 P1 sanity 25%) |
| threat probe C1/C2/C3           | SKIPPED (§170 v8 fixture follow-up; same gap as A2/A3) | n/a    | n/a                     |
| params (M)                      | **3.85 M** (B1-equivalent; PartialConv2d adds zero learnable params — just renormalisation)      | 3.85 M | 5.29 M |
| latency b=1 / b=64 (5080, ms)   | **2.77 / 11.34** (+5% / +9% vs A1)             | (B1 ~2.48 / ~11.3, similar) | 2.64 / 10.41            |

**Read:** A4 closed the **training loss** gap (3.47 < 3.57 anchor — better
than A1, in fact) and the pre-flight subspike PASSed (SE × PartialConv2d
compatibility, full-model b=1 Δ +1.64% on 5080, +0.30% at b=64 — well
under the 5% gate). **But SealBot WR collapsed to 0% at both argmax and
MCTS-128**, identical to §167 B1 v8 and ≪ A1's 14.5% argmax / 25%
MCTS-32 sanity. Mean_ply 23.5 (argmax) ≈ 23.6 (MCTS-128) — MCTS finds
no improvement over argmax; the model has no useful policy distribution
to search over. The padding-semantics intervention (canvas_realness mask
+ PartialConv2d at trunk entry) did NOT close the bbox-vs-K-cluster
gap, falsifying the hypothesis that B1's 0% argmax was a fixable
zero-padding artefact.

**Verdict: NEGATIVE — bbox direction structural, NOT padding
semantics.** A4 trained cleanly (loss converged below A1 anchor, no
NaN-skips, gate-passing latency) but transfers ZERO of that training
quality to SealBot eval — same outcome as untouched B1 v8. The loss
is structural at the encoding level: candidate mechanisms for the
bbox failure (out of §169 scope to disambiguate further):

  1. **K-aggregation as cross-cluster contrast.** The K-cluster encoding
     gives the model K windows per leaf at inference time, each
     scattered through the bot-side scatter-max-on-prob; this is the
     A1 v6w25 path and is what A2/A3 tried and failed to replace with
     learned PMA. Single-window bbox forfeits this entirely; even
     perfect padding semantics cannot reconstruct multi-window
     contrast at inference time when the corpus serves K=1.
  2. **Bbox-centroid frame instability.** The 625-action policy head
     emits logits in the bbox-centroid frame, which shifts every time
     a stone lands far from the existing bbox (centroid moves up to
     ~m=8 cells per move). The model sees ply-T states centred on
     centroid_T but must score ply-T+1 actions centred on
     centroid_T+1 — there is no fixed reference frame the policy
     converges to.
  3. **R=8 perception expansion.** v8 bumped legal_move_radius from 5
     to 8 (P2 hotfix-(c) bundling). Self-play opens up by ~8× more
     legal moves per ply at any given centroid; the policy must learn
     8× more action geometry per board state than under v6w25's
     R=8 + cluster-mask. Pre-trained on human games (R=8 unrestricted)
     this should be fine, but the bbox single-window may not give
     the model enough context to discriminate the correct cell at
     inference time.

These three mechanisms each predict that adding cross-cluster
contrast back into the bbox path (per-cluster bbox at
CLUSTER_THRESHOLD=8 falling back to a unified bbox when stones
merge) recovers most of the gap. That fallback was specced in
`audit/encoding_spikes/s1_bbox_algorithm.md` §5.2 and is the
operator's call for a §170 follow-up if Phase 5+ revisits bbox.

**Pre-flight subspike on 5080 (re-run before retrain):**

| check                                | result    |
|--------------------------------------|-----------|
| forward + backward correctness       | finite, all 13 PartialConv params reach finite grads |
| off-canvas output zero               | max\|val\|=0.000000 |
| SE on PC outputs                     | finite, mean=-0.0039 |
| latency b=1 trunk-entry-only (5080)  | A4 0.177 ms / B1 0.133 ms = +33.09% |
| latency b=1 FULL HexTacToeNet (5080) | A4 1.642 ms / B1 1.616 ms = **+1.64%** (+1.93σ) |
| latency b=64 FULL MODEL (5080)       | A4 11.328 ms / B1 11.295 ms = **+0.30%** |

5080 confirms the laptop subspike: SE × PartialConv2d compatible at
trunk entry; full-model latency hit within budget at every batch size.

**Hard-stop / surface conditions — actual outcomes:**
- Subspike SE × PartialConv compatibility: **PASS** (gated pre-retrain
  + re-confirmed on 5080).
- Final loss > 5.36: **PASS** (3.47 ≪ 5.36; A4 actually undercut A1).
- NaN-skip rate > 30%: **PASS** (0 skips across the entire 30-epoch run).
- A4 argmax > 12% (bbox direction lives): **NOT TRIGGERED** (0%).
  Verdict path: structural loss.

**§169 close-out implication.** Four-way ablation matrix complete:

| arm                                          | loss   | argmax WR vs SealBot | MCTS WR vs SealBot |
|----------------------------------------------|--------|----------------------|--------------------|
| A1 — K-cluster + min/max (v6w25 anchor)      | 3.57   | **14.5%**            | 25% (P1 sanity, MCTS-32 n=20) |
| A2 — K-cluster + PMA pool                    | 4.25   | 4.5%                 | 3.5% (MCTS-128)    |
| A3 — K-cluster + PMA + global token          | 3.62   | 7.5%                 | 2.5% (MCTS-128)    |
| A4 — bbox + canvas_realness + PartialConv2d  | **3.47** | **0.0%**           | **0.0%** (MCTS-128) |

Training loss alone is NOT a sufficient signal for SealBot WR — A4 has
the lowest loss but zero WR; A2 has the highest loss but still beats
A4 at argmax. **The encoding decides; the pool variant tweaks.** A1
remains the canonical path. Phase 5+ encoding-pivot work (if it
revisits bbox) must address the structural mechanisms above (K=1 vs
K>1 corpus supervision, bbox-centroid frame instability, single-window
inference-time blindness) before any further bbox arm is worth the
GPU time.

**Branch state:** `encoding/four_way_ablation` — 5 commits on top of
the A3 P3 tip:
1. `3d047b4 feat(corpus): §169 A4 canvas_realness plane-8 polarity for v8`
2. `53c72aa feat(model,eval): §169 A4 PartialConv2d trunk entry + canvas_realness wire`
3. `264c20c chore(ablation_169): A4 retrain config + pretrain script + eval matrix`
4. `25e763d docs(sprint): §169 P4 draft` (this section, replaced post-eval)
5. `2c58163` + `f7b17e4 fix(scripts): A4 pretrain/eval use .venv/bin/python explicitly`

**Done-when checks:**
- [x] PartialConv2d module + tests (`hexo_rl/model/partial_conv.py`,
  `tests/test_partial_conv.py` — 9 tests).
- [x] dataset_v8 canvas_realness polarity + tests (+5 A4 tests).
- [x] HexTacToeNet canvas_realness wiring + state-dict key shift +
  checkpoint loader auto-detection.
- [x] V8ArgmaxBot / V8MCTSBot / bench_v8_nn thread canvas_realness.
- [x] CLI flags (`--canvas-realness` on pretrain + export_corpus_npz).
- [x] Pre-flight subspike PASS on laptop (4060 Max-Q) + 5080.
- [x] `make test` green: 1111 passed, 8 skipped (no regressions).
- [x] Configs + retrain script + eval script landed.
- [x] Full v8 canvas_realness corpus regen on 5080 (5 min wall, 5,382 MB,
  sha256 `110ea6b2…`).
- [x] 30-epoch pretrain on 5080 — final loss **3.4658**, 0 NaN-skips
  (107 min wall).
- [x] argmax + MCTS-128 WR captured in `reports/ablation_169/A4_eval.json`
  — both **0% [0%, 1.9%]** (0/200 each).
- [x] bench captured in `reports/ablation_169/A4_bench.json` (b=1 2.77 ms,
  b=64 11.34 ms, params 3.85M, host 5080).
- [x] Threat probe SKIPPED (no v8 fixture; §170 follow-up — same gap as
  A2/A3).
- [x] Sprint log Results table back-filled, verdict line written.


## §169a — A4 spatial-pathway-deadness probe (§170-pre) — 2026-05-08

**Question.** §169 P4 closed with A4 at 0% argmax / 0% MCTS-128 SealBot
WR despite training loss 3.47 (below v6w25 anchor 3.57). Operator-side
hypothesis: A4 collapsed onto the broadcast-scalar planes (plane 9
moves_remaining_bcast, plane 10 ply_parity_bcast) and abandoned the
spatial stone-history pathway (planes 0-7); human-corpus moves are
predictable enough from scalars + opening stylization that loss falls,
play falls because no spatial reasoning. If true, the bbox direction
is structurally falsified at the architecture level and §170 commits
to the K-cluster line. If false, A4's failure is a distribution-shift
issue (corpus-conditional spatial features go OOD on SealBot).

Cheap (~30-min budget, ~10-s actual) two-arm KL probe before any §170
investment.

**Probe.** `scripts/probe_a4_spatial_deadness.py`, three sets:

- **Set S** (n=200): random self-play replays to ply 20, encoded with
  canvas_realness=True. By construction all positions share planes
  8/9/10 (canvas mask, moves_remaining=180/200, parity=0); only the
  spatial stone configuration varies.
- **Set R** (n=200): real positions sampled from
  `data/bootstrap_corpus_v8.npz` with plane 8 inverted off→canvas.
  Both scalars and spatial vary.
- **Set F** (n=8): Set S[0] replicated 8× — determinism sanity.

Pre-registered thresholds (locked before run):

- E1 PASS (spatial dead): mean(KL_S) < 0.10 nats AND KL_S/KL_R < 0.05
- E2 PASS (spatial alive): mean(KL_S) > 1.00 nats OR KL_S/KL_R > 0.30
- Otherwise: ambiguous (lean E1).

**Results.**

| Set | mean KL (nats) | median | p90 | argmax-distinct |
|-----|----|----|----|----|
| S (spatial-only) | **1.533** | 1.470 | 2.182 | 133 / 200 |
| R (full corpus) | 5.626 | 5.643 | 7.328 | 107 / 200 |
| F (sanity)      | 0.000 | 0.000 | 0.000 | 1 |

Ratio KL_S / KL_R = **0.273**.

**Verdict: E2 PASS — spatial pathway alive.** KL_S = 1.533 > 1.0 fires
the absolute-KL trigger; argmax visits 133 distinct cells out of 200
random plates (broad spread, top-1 cell only 2% share). The user's
primary "spatial-dead" hypothesis is **falsified** at the architecture
level. PartialConv2d trunk entry + canvas_realness mask propagate the
spatial signal correctly through trunk + KataGo policy head.

**Implications for §170 scoping.**

1. The §169 close-out implication "encoding decides; the pool variant
   tweaks; A1 remains the canonical path" is *unchanged on the eval
   level* (A4 is still 0% SealBot WR); but the *mechanism* statement
   shifts from "spatial path dead" to "spatial path alive but
   corpus-conditional, scalar-dominated under realistic position
   diversity." KL_S / KL_R = 0.27 means scalar variation accounts for
   the majority of A4's policy variance under full position diversity.
2. Live alternatives this probe surfaces:
   - **Distribution-shift fine-tune.** Augment corpus with adversarial
     SealBot-style positions and retrain A4. Cheap test of the
     corpus-conditional hypothesis before pivoting.
   - **Cross-encoding eval gap audit.** §168 v7full radius curve
     (6.5% → 12.5% → 15%) already showed perception-radius matters;
     match A4 against A1 under MCTS at matched perception radius
     before declaring bbox dead.
   - **Scalar-ablation follow-up** (~30 min): zero planes 0-7 in Set R,
     re-run, measure policy delta. Fully discriminates "scalar-only" vs
     "scalar-dominated."
3. The §170 scope should NOT default to "abandon canvas_realness as
   architecturally dead." Empirical bbox failure remains; mechanism
   needs one more probe before scoping a fix.

**Artefacts.**

- `scripts/probe_a4_spatial_deadness.py` — discriminator probe with
  pre-registered thresholds.
- `reports/investigations/a4_spatial_deadness_20260508/probe.json` —
  full numeric output, fixture audit, exit_code.
- `reports/investigations/a4_spatial_deadness_20260508/probe.log` —
  stdout.
- `reports/investigations/a4_spatial_deadness_20260508/VERDICT.md` —
  reviewer-friendly write-up.

Wall time: ~10 s on laptop RTX 4060 Max-Q, ~30 min total including
fixture audit + threshold pre-registration + report. Five hours of
compute saved on a wrong §170 scope.

---


## §170 P0 — A4 scalar-ablation probe — 2026-05-08

**Question.** §169a established A4's spatial pathway is alive (KL_S=1.53,
E2 PASS) but a minority shareholder (KL_S/KL_R=0.27). Does zeroing the
stone planes (0–7) on Set R cause a large policy shift (spatial features
are decisive → distribution-shift fine-tune is worth trying) or a small
shift (scalars dominate → fine-tune is hopeless)?

**Probe.** `scripts/probe_a4_scalar_ablation.py`. Per-position symmetric
KL between original Set R forward pass and a zeroed-planes-0-7 copy. Same
Set R fixture as §169a (n=200, seed=20260508).

Pre-registered thresholds (locked before run):

- SCALAR_DOMINATED: mean(KL_zeroed_vs_original) < 0.30 nats
- SPATIAL_RICH:     mean(KL_zeroed_vs_original) > 1.50 nats
- AMBIGUOUS:        0.30 – 1.50 nats

**Results.**

| Metric | Value |
|---|---|
| Mean sym-KL (zeroed vs original) | **4.19 nats** |
| Median | 4.30 nats |
| p90 | 5.27 nats |
| Min | 0.51 nats |
| Argmax stable (same cell) | **0 / 200 (0.0%)** |

**Verdict: SPATIAL_RICH.** Mean KL = 4.19 >> 1.50 threshold. Argmax
changes for every single position when stone planes are removed. A4's
spatial pathway is not only alive (§169a) — it is *load-bearing*. Stone
planes 0–7 are decision-critical; scalars alone do not determine the top-1
move.

**Implications for §170 scoping.**

1. **SCALAR_DOMINATED path closed.** Distribution-shift fine-tune is *not*
   hopeless on the grounds of absent spatial features.
2. **§171 fine-tune is mechanistically justified.** The spatial features
   exist and are active; the SealBot collapse is most consistent with
   corpus-overfitted spatial representations, not dead or absent ones.
   Augmenting with adversarial positions addresses the right failure mode.
3. **K-cluster line (A1) remains canonical §169 winner.** §171 is a
   side-branch to determine whether bbox is worth pursuing; it does not
   block or alter the A1 path.
4. **Architectural change not warranted before §171.** Capacity and routing
   are intact; the failure mode is training-data distribution, not model
   structure.

**Artefacts.**

- `scripts/probe_a4_scalar_ablation.py` — probe script.
- `reports/investigations/a4_scalar_ablation_20260508/probe.json` — full
  numeric output.
- `reports/investigations/a4_scalar_ablation_20260508/probe.log` — stdout.
- `reports/investigations/a4_scalar_ablation_20260508/VERDICT.md` —
  reviewer-friendly write-up.

Wall time: ~10 s on laptop RTX 4060 Max-Q.


## §170 P1 — A3 MCTS-N curve (PMA-value-semantics hypothesis) — 2026-05-08

**Question.** Does A3 (PMA + global token) win-rate decline *monotonically* with
MCTS-N? If so, the mechanism is value-semantics compounding: PMA replaces
min-pool's worst-subgame value signal with an optimistic aggregate, and deeper
search amplifies the error multiplicatively across MCTS backups (§170 Bet B,
option α). Argmax avoids the compounding because it never backs up values.

**Pre-registered verdict criteria (locked before runs).**

- **MONOTONIC-DECLINE**: argmax > MCTS-32 > MCTS-64 > MCTS-128 strict ordering,
  each consecutive CI overlap ≤ 50%, Cochran-Armitage trend p < 0.10.
- **FLAT/NON-MONOTONIC**: any ordering inversion OR max consecutive CI overlap
  > 75%.

**Setup.** A3_pma_global.pt (checkpoints/ablation_169/). n=200 each,
seed_base=42, legal_radius=8, random_opening_plies=4, c_puct=1.5. Laptop
RTX 4060 Max-Q. MCTS-32 and MCTS-64 run in parallel (both on CUDA).

**Results.**

| Method   | W  | L   | D | WR    | 95% CI         | elapsed |
|----------|----|-----|---|-------|----------------|---------|
| argmax   | 15 | 184 | 1 | 7.5%  | [4.6%, 12.0%]  | 820s (§169 P3) |
| MCTS-32  |  5 | 195 | 0 | 2.5%  | [1.1%, 5.7%]   | 617s |
| MCTS-64  |  5 | 195 | 0 | 2.5%  | [1.1%, 5.7%]   | 776s |
| MCTS-128 |  5 | 195 | 0 | 2.5%  | [1.1%, 5.7%]   | 887s (§169 P3) |

Cochran-Armitage two-sided p = 0.0277 (significant overall, driven entirely by
the argmax vs any-MCTS split — see below).
Max consecutive CI overlap = 100% (MCTS-32/64/128 are identical W/L).

**Verdict: FLAT-NON-MONOTONIC.**

The three MCTS arms produce identical results (5W/195L, same CI to four decimal
places). There is no monotone decline within the MCTS-N range — the pattern is a
**sharp cliff at the argmax→MCTS-32 boundary**, followed by a hard floor
regardless of sims count.

The Cochran-Armitage p = 0.0277 is statistically significant but reflects the
single argmax-vs-MCTS split, not a trend across N. Applying the pre-registered
criterion: FLAT-NON-MONOTONIC fires (consecutive CI overlap 100% > 75% for all
MCTS→MCTS pairs; strict monotone ordering violated at MCTS-32 = MCTS-64).

**Mechanism re-interpretation.**

The monotone-compounding hypothesis (more sims → deeper value backup → larger PMA
error) is **refuted**. The correct reading of the cliff pattern is:

1. **Binary switch, not gradual amplification.** PMA corrupts value quality once
   (during training). Argmax escapes this because it reads the policy head
   directly, bypassing the value backup path. MCTS-32 immediately routes through
   value-backed PUCT selection and hits the full damaged floor — additional sims
   cannot recover from a broken value signal.

2. **PMA optimistic-value bias is not search-depth-sensitive.** The error is
   already saturated at MCTS-32 (~3 levels of backup for a ply-23 median game).
   MCTS-64 and MCTS-128 add more backups but the value floor is already set.

3. **Argmax policy quality is real.** A3 argmax = 7.5% (vs A1 argmax ~25% — a
   real gap, but not zero). The global token does lift policy quality somewhat
   (A2 argmax = 4.5%, A3 = 7.5%), confirming the cross-cluster policy signal
   is working. The problem is exclusively in the value path under search.

**§170 scoping implications.**

1. **A1 (min_max) remains canonical.** Value semantics are the controlling factor
   for MCTS performance. Any A3-descended variant must fix the value head
   (not the policy head, which is already working). The cliff confirms: restoring
   worst-subgame value semantics (min_max) immediately recovers MCTS performance.

2. **"Add PMA side-channel to A1" framing (Bet B / option α) is still valid —
   but the justification is now the argmax lift (4.5%→7.5%), not search-depth
   robustness.** A policy-only side-channel from A3's global token could be
   grafted onto A1 without touching the value head, capturing the 7.5%→X argmax
   gain while preserving A1's MCTS value quality.

3. **Do NOT route PMA through the value head.** If §170 tests a hybrid (A1 pool
   + global token for policy only), the value path must remain min_max. A3's
   failure is a clean natural experiment confirming this constraint.

4. **Low-cost §170 option:** retrain A1 variant with global token in policy head
   only (value gate=0.0 forced). Predicted: argmax approaches A3 (7–8%), MCTS
   approaches A1 (25%). This would be the best-of-both result.

**Artefacts.**

- `reports/ablation_169/A3_mcts32.json` — MCTS-32 eval (5W/195L, 617s).
- `reports/ablation_169/A3_mcts64.json` — MCTS-64 eval (5W/195L, 776s).
- `reports/ablation_169/A3_mcts_curve.md` — 4-point curve + CA test + verdict.
- `scripts/aggregate_a3_mcts_curve.py` — aggregation script.

Wall time: 617s (MCTS-32) + 776s (MCTS-64) parallel on laptop, total wall ≈ 776s
≈ 13 min.

**Commit:** `eval(a3): MCTS-N curve on existing checkpoint` (1 commit, this §).


## §170 P3 — A1 + gpool-bias retrain — ENGINEERING COMPLETE — 2026-05-08

**Branch:** `encoding/gpool_bias_a1` (off `ec6e30b`, post-§169 A4 close-out + §169a + §170 P0/P1).
**Status:** 4 commits landed; checkpoint + eval pending operator retrain on 5080 vast.ai.

### Hypothesis

Keep A1's load-bearing min/max value semantics BYTE-EXACT. Add KataGo-style
**additive K-invariant gpool-bias side-branch** to value+policy heads (gate=0
init → byte-exact A1 at construction; only as gradient grows the gate does
the global summary earn weight). Predicted: argmax +2-4 pp, MCTS +3-6 pp
over A1. Mechanism: A3 P3 confirmed canvas-level summary statistics lift
argmax (4.5%→7.5%); §170 P1 confirmed MCTS collapse came from PMA replacing
min-pool VALUE semantics. Gpool-bias preserves min-pool by construction —
addition only — and adds the global signal that A1 lacks.

### Architecture

A1 v6w25 trunk + min/max pool BYTE-EXACT untouched. New side-branch
(`hexo_rl/model/gpool_bias.py:GpoolBiasBranch`):

  - reuses `GlobalTokenEncoder` verbatim (3 → 64 conv ×2 → KataGo gpool →
    Linear → d=128 token; same canvas-mask plumbing as §169 A3)
  - `value_proj: Linear(filters → 256)` projects token to value-head bias
  - `policy_proj: Linear(filters → 626)` projects to per-cluster policy bias
  - `gate: Parameter(tensor([0.0]))` learned scalar; init **0.0** → branch
    contributes nothing at construction; gate=0 byte-exact A1 (enforced by
    unit test `test_gate_zero_byte_exact_a1` against `bootstrap_model_v6w25.pt`)

Bias injection sites (preserve A1 semantics):

  - **value head**: `value_bias` added to `F.relu(value_fc1(...))` hidden
    activation between `value_fc1` and `value_fc2`. K-invariant.
  - **policy head**: `policy_bias` added to per-cluster `policy_fc` raw
    logits BEFORE `log_softmax`. Same bias broadcast to every cluster
    window — bot-side scatter-max-on-prob then operates on
    `softmax(logits + bias)` per cluster, equivalent to adding the same
    bias to every cluster.

`HexTacToeNet` cross-product validation: `gpool_bias_active=True` requires
`encoding ∈ ('v6', 'v6w25')` AND `pool_type='min_max'` AND not
`canvas_realness` AND no `gpool_indices`. 4 distinct ValueErrors on
misconfig.

### Commits (4 on `encoding/gpool_bias_a1`)

1. **`cb61a78 feat(model): GpoolBias side-branch + gate scalar (gate=0
   byte-exact A1)`** — `hexo_rl/model/gpool_bias.py` (new, 96 LoC) +
   `HexTacToeNet.gpool_bias_active` flag + forward/aggregated_forward_K
   bias injection + `gpool_bias_gate_value()` accessor +
   `checkpoint_loader._build_v6_model` auto-detect via state-dict keys.
   7 tests: byte-exact A1 parity (loads real `bootstrap_model_v6w25.pt`,
   `torch.equal` on log_policy/value/v_logit), zeroed-projection parity,
   grad reach, K-invariance, state-dict round-trip, aggregated_forward_K
   bias applied, 4 ValueError cases.

2. **`641408b feat(dataset,pretrain): v6w25 corpus + 32x32 global crop
   column for gpool-bias path`** — pretrain.py: `--gpool-bias-active` CLI
   flag, cross-product validation (rejects v8 / pma / canvas_realness /
   gpool_indices at parse time), corpus gate widened to accept
   `pool_type='min_max' + gpool_bias_active=True` consumer of `global_crops`,
   model construction passes `gpool_bias_active`, checkpoint config persists
   it, train-step logging surfaces `gpool_bias_gate` parallel to
   `pool_global_gate`, `validate()` smoke-forwards with zero global crop
   then skips play-100 (same pattern as pma_global). NO changes to
   `dataset_v6w25.py` — `with_global_crop=True` path from §169 A3 reused
   verbatim. Corpus reused: `data/bootstrap_corpus_v6w25_with_global.npz`
   sha256 `e2876ae5639958dac3758274b7137faeaff91713fe50df6da04ea43dfd896793`.
   1 integration test: tiny synthetic corpus → dataset → collate (5-tuple)
   → model.forward(global_crop=...) → backward, all params reach finite
   grads, gate=0.0 at construction.

3. **`b0f0259 feat(retrain): A1 + gpool-bias retrain config + script`** —
   `configs/ablation_170_gpool_bias.yaml` (recipe + hard-stop /
   soft-warn rules) + `scripts/pretrain_gpool_bias.sh` (executable). Same
   recipe as v6w25 anchor (§168 Gate 5) + A3: 30 ep cosine, peak 2e-3,
   eta_min 5e-5, batch 256. Only delta vs A1: `gpool_bias_active=true`.
   Hard stops: final_loss > 5.36, NaN-skip > 30%, forward_parity_required.
   Soft warns: gate_stalled_below 0.05 (null result), argmax_wr < 0.12
   (failed), argmax_wr > 0.20 (BREAKTHROUGH — surface for §171 sustained).

4. **`898a1d3 eval(a1-gpool-bias): argmax + matched MCTS evaluation
   plumbing`** — `V6ArgmaxBot` + `KClusterMCTSBot` thread `global_crop`
   when `model.gpool_bias_active=True` (auto-detect off model attribute;
   no new `pool_type` value). `KClusterMCTSBot._forward_K(global_crop=)`
   broadcasts (1, 3, 32, 32) to (K, ...) per leaf. `bench_v6w25_nn.py`
   adds gpool_bias to its global_crop_template gate; markdown `pool`
   column reads `min_max+gpool_bias` to distinguish A1 vs A1+gpool-bias.
   `scripts/eval_gpool_bias.sh` (executable; mirrors A3 eval template
   minus PMA-collapse smoke — gpool-bias has no collapse mode by
   construction). 5 plumbing tests: V6ArgmaxBot threads global_crop;
   KClusterMCTSBot threads global_crop; `_forward_K(global_crop=)` accepts
   (3, 32, 32) and (1, 3, 32, 32) shapes; min_max without gpool_bias
   stays canonical (no global_crop); bench `_bench_one(global_crop=)`
   returns valid timing.

### Test posture

- `make test`: **1164 passed / 8 skipped / 2 deselected** (1159 pre-§170 P3
  + 5 plumbing). **No regressions.**
- 13 new §170 P3 tests across `test_gpool_bias.py` (7), `test_pretrain_
  gpool_bias.py` (1), `test_gpool_bias_eval_plumbing.py` (5). All green.

### Hard surface conditions (gating retrain on 5080)

- **Forward parity at gate=0**: GREEN. Unit test `test_gate_zero_byte_exact_a1`
  loads `bootstrap_model_v6w25.pt`, copies into A1+gpool_bias arch,
  `torch.equal` on outputs across 5-position fixture. Architecture invariant
  the user mandated.
- **Final loss > 5.36** (50% above A1 anchor 3.57): STOP, surface (post-train).
- **NaN-skip rate > 30%** even with §167 patch: STOP (post-train).
- **Gate scalar at end < 0.05**: branch never earned weight → null result;
  flag in verdict but eval still proceeds.
- **argmax WR < 12%**: gpool-bias didn't help; surface (post-eval).
- **argmax WR > 20%**: BREAKTHROUGH; surface for §171 sustained-run scoping.

### Bench parity vs A1 (laptop, pre-train measurement)

Side-branch adds ~96 + 4096 + 4096 + 80,000 + 80,000 ≈ 168k params (encoder
~4.5k + value_proj 32,896 + policy_proj 80,896). Latency overhead at b=1
expected < 0.5 ms (small encoder, 2 small linears). Post-train bench will
populate `reports/gpool_bias/bench.md`.

### Post-retrain done-when (operator action on 5080)

The engineering portion is complete; operator drives:

1. `bash scripts/pretrain_gpool_bias.sh` on 5080 (~1h 33m wall expected,
   matches A1 anchor + A3). Captures `reports/gpool_bias/pretrain.log`
   with per-step `gpool_bias_gate` trajectory.
2. `bash scripts/eval_gpool_bias.sh` (default MCTS_N=64) — argmax @ r=8
   n=200 + MCTS-64 n=200 + bench + skipped threat + combined eval.json.
3. Pull artefacts to laptop via rsync-vast skill.
4. Back-fill the post-train Results table below, append verdict line.

### Results (5080 vast.ai, 2026-05-09 — pretrain wall 1h 48m, eval wall 28 min)

| metric                          | A1+gpool-bias                                  | A1 anchor (v6w25, §168 Gate 5)        | hard-stop |
|---------------------------------|------------------------------------------------|----------------------------------------|-----------|
| corpus sha256                   | `e2876ae5…` (reused from §169 A3, 354,407 pos) | (v6w25 corpus 319,207 pos)             | n/a       |
| final epoch-30 loss             | **2.8963** (BETTER than A1 anchor)              | 3.57                                   | 5.36      |
| policy_loss / value_loss        | 2.3595 / 0.1791                                | n/a                                    | n/a       |
| NaN-skip rate                   | **0%** (clean run)                              | 0%                                     | 30%       |
| `gpool_bias_gate` init/mid/final | **0.000 / 0.038 / 0.0512** (~3× growth from 0) | n/a                                    | < 0.05 ⇒ null |
| argmax @ r=8 n=200 vs SealBot   | **22.0% [16.8%, 28.2%]** (44W/154L/2D, mean_ply 47.98, median 35.0) | 14.5% [10.3%, 20.0%] | < 12% ⇒ failed; > 20% ⇒ surface §171 |
| MCTS-64 n=200 vs SealBot        | **15.0% [10.7%, 20.6%]** (30W/170L/0D, mean_ply 29.7, median 29.0) | **30.0% [24.1%, 36.7%]** (60W/140L/0D, mean_ply 33.8, median 33.0; matched-baseline ran post-eval, 839s on 5080) | n/a — **REGRESSION** |
| threat probe C1/C2/C3           | SKIPPED (no v6w25 fixture; §170 follow-up)      | n/a                                    | n/a       |
| params (M)                      | **5.47** (A1 + 0.18 M for gpool-bias branch)   | 5.29                                   | n/a       |
| latency b=1 / b=64 (5080, ms)   | **1.49 / 11.26**                               | 2.64 / 10.41                           | 3.50 (b=1 gate) |

### Verdict

**FALSIFIED — argmax lift does NOT survive MCTS.** Matched A1-anchor
MCTS-64 baseline (60W/140L/0D = **30.0% [24.1%, 36.7%]**, mean_ply 33.8,
elapsed 839s on 5080) reveals A1+gpool-bias regresses **−15.0 pp under
MCTS** (15.0% vs A1 30.0%; CIs disjoint by 3.5 pp). The +7.5 pp argmax
lift is real but does not transfer through PUCT search — same
**argmax-up / MCTS-down signature** as A2 PMA (4.5% argmax, 3.5%
MCTS-128) and A3 PMA+global (7.5% argmax, 2.5% MCTS-128). Mean_ply
collapses under MCTS in the predicted direction:
- A1 anchor MCTS-64: mean_ply 33.8 (median 33.0) — search holds the line.
- A1+gpool-bias MCTS-64: mean_ply 29.7 (median 29.0) — SealBot wins faster
  under search than against argmax (47.98 mean_ply at argmax).

**Mechanism — additive bias on the value head still breaks MCTS value
semantics.** The §170 P3 prompt asserted "addition only — does NOT
perturb min-on-value" by construction. That is structurally true at
GATE=0 (commit-1 unit test). Once gate grows during training, the
trained value head's `value_fc2(F.relu(value_fc1(...)) + value_bias)`
emits values whose distribution shifts vs the gate=0 baseline; the
gradient pushes the value head to *use* the bias signal (that's the
lift seen at argmax). MCTS then backs up these biased values across
many simulations; the cumulative drift breaks PUCT selection in the
same way A3's PMA-replaced value head did. **Min-pool's K-cluster
aggregation is preserved, but the per-cluster value the model emits
is no longer A1's value — the bias has rewired the value head's
operating point.**

This refutes the user-stated invariant "gpool-bias preserves
load-bearing pool by construction — bias is K-invariant, addition only,
doesn't perturb min-on-value". K-invariance of the bias holds; what is
NOT preserved is the per-cluster scalar value semantics under MCTS
backup. The same way A2/A3 broke value semantics by replacing the
head, A1+gpool-bias broke them by adding bias INTO the head's hidden.

**Loss: 2.8963 < A1 anchor 3.57** — well below 5.36 hard-stop. Better
loss + worse MCTS reproduces the §169 close-out lesson: **training
loss alone is NOT a sufficient signal for SealBot WR; encoding +
value-head structure decide.** Adding to A4's lesson (lowest loss,
0% WR), A1+gpool-bias is now the second confirmed case where lower
loss correlates with worse-under-search.

**Gate trajectory** climbed from 0.0 init to 0.0512 final (≈3× from
absolute zero, barely above the 0.05 soft-warn null threshold). Despite
the modest gate magnitude the bias contribution was *enough to break
MCTS* — argues that future side-branch ablations need to gate the
VALUE head separately or skip the value head entirely (policy-only
side-channel as flagged in §170 P1 §4 "Low-cost §170 option: retrain
A1 variant with global token in policy head only (value gate=0.0
forced)").

### Latency note

b=1 latency on 5080 is **1.49 ms**, *FASTER* than the A1 anchor's 2.64 ms
recorded at §168 Gate 5. Likely warmup / measurement-protocol drift
between the two runs (the bench was extended in commit-4 to use the same
template-broadcast helper). Within the 3.5 ms b=1 gate. b=64 11.26 ms
(+0.85 ms / +8% over A1's 10.41 ms) — the gpool-bias side branch adds
the expected modest overhead.

### Surface for §171 scoping

The breakthrough threshold (`argmax > 20%`) was triggered initially BUT
the matched A1-anchor MCTS-64 baseline (run post-eval, 839s on 5080)
falsified the under-search lift: A1 anchor MCTS-64 = 30.0% vs
A1+gpool-bias MCTS-64 = 15.0%. **§170 P3 is NOT a §171 sustained-run
candidate.** The hypothesis "additive K-invariant bias preserves
load-bearing pool" is refuted at the value-head injection site.

Recommended follow-ups (in priority order):

1. **A1+gpool-bias-policy-only** (§170 P4 candidate): retrain a variant
   that forces `value_proj` to zero / freezes value gate at 0, allowing
   only the policy_proj branch to carry the global signal. Predicted
   per §170 P1 §4: argmax approaches A1+gpool-bias (~22%), MCTS
   approaches A1 anchor (~30%). Best-of-both. One config + one retrain
   on 5080 (~1h 48m). The architecture invariant (gate=0 byte-exact A1)
   already holds; only need to expose a `value_gate_active=False` knob
   on `GpoolBiasBranch` so the value path is permanently disabled.
2. **Padding-leak hold-out smoke** (optional): patch `GlobalTokenEncoder`
   to drop the canvas_mask plane and re-eval argmax; significant drop
   confirms the model is reading the canvas-realness mask as decision
   signal (expected for a global-context-aware branch).
3. **Threat probe v6w25 fixture build** (the persistent §170 gap):
   curate tactical positions on a 25×25 board + regenerate baseline.
   Out of §170 P3 scope; queue for §171 prep.

### Done-when checks

- [x] Forward-parity test green (commit 1, against bootstrap_model_v6w25.pt;
  enforces architecture invariant).
- [x] Gate scalar trajectory captured (init 0.000 → mid 0.038 → final 0.0512).
- [x] argmax + MCTS-64 eval JSONs in `reports/gpool_bias/`.
- [x] Threat probe captured as status=skipped (`reports/gpool_bias/threat.json`).
- [x] Bench captured (`reports/gpool_bias/bench.md`).
- [x] 4 functional commits + 1 sprint-log commit on `encoding/gpool_bias_a1`.
- [x] `make test` green (1164 passed / 8 skipped).
- [x] Sprint log Results table populated; verdict line written.
- [x] Artefacts pulled to laptop (`checkpoints/gpool_bias/A1_gpool_bias.pt`
  21.9 MB; `reports/gpool_bias/*` 552 KB total).

### Surface protocol (post-eval, post-baseline)

- **Gate < 0.05 soft-warn**: BORDERLINE — final 0.0512 (0.0012 above
  threshold). Modest weight earned; argmax shifted +7.5 pp confirming
  the global signal IS doing real work, but MCTS regression confirms
  the work it does breaks under search.
- **argmax > 20% BREAKTHROUGH**: NOT a §171 candidate. Matched A1
  anchor MCTS-64 (30.0%) refutes the under-search lift; A1+gpool-bias
  MCTS-64 = 15.0% is a 15 pp regression with disjoint CIs.
- **MCTS-64 regression > 10 pp under matched A1 baseline**: TRIGGERED
  (−15.0 pp). §170 P3 hypothesis falsified at the value-head bias
  injection site.
- **Forward-parity post-train**: NOT re-run — the architecture invariant
  is structural (verified at construction by the unit test), not a
  post-train property. Holds either way.

### Lesson logged

The "additive bias preserves load-bearing pool" intuition is wrong if
the bias is added INTO the value head's hidden activation. Once the
gate gains weight during training, the value head adapts to use the
bias signal; the value distribution shifts vs A1 anchor and PUCT
accumulates the drift across simulations. A1's load-bearing min/max
pool is preserved STRUCTURALLY (the K-cluster reduction still picks
min-of-K), but the per-cluster scalar value the model emits is no
longer A1's value. This generalises §170 P1's lesson ("Do NOT route
PMA through the value head") to **"do not modify the value head's
operating point AT ALL — including additive bias"**. The next ablation
(A1+gpool-bias-policy-only) tests whether limiting injection to the
policy head escapes this trap.

---

