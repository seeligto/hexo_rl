# §S181 IMPL-S181T2 — Value-Head + Encoding Architectural Analysis

**Wave:** §S181 structural-diagnosis. **Track:** T2 (value head + encoding).
**Mode:** inspection-only. No training, no hot-path edits, no config edits.
**Branch:** `phase4.5/s181_structural_research`.
**Question:** does the network architecture STRUCTURALLY encourage colony
interpretation independent of the training data?

Helper probe: `scripts/structural_diagnosis/probe_architecture.py`
(standalone, re-implements `network.py` value-head math, no checkpoint load).
Probe output transcript embedded inline below.

---

## 0. Scope correction — which encoding the §S178 line actually runs

Critical framing fact, established first because it narrows the whole
analysis: **the §S178 colony line (§175 / §S179 / §S180a / §S180b) runs
encoding `v6`, NOT `v6w25`.**

- §S178 anchor = `checkpoints/bootstrap_model_v6.pt`, variant
  `configs/variants/v6_botmix_s178.yaml` → encoding `v6`
  (`v6_botmix_s178.yaml:13` "v6.pt = canonical v6 base bootstrap").
- `engine/src/encoding/registry.toml:58-82` — `v6`: `is_multi_window=false`,
  `k_max=1`, `value_pool="none"`, `policy_pool="none"`, single 19×19 window.
- The K-cluster min-pool / scatter-max machinery (`value_pool="min"`,
  `policy_pool="scatter_max"`) is `v6w25` / `v7mw` only
  (`registry.toml:190-191`, `:225-226`).

**Consequence:** for the §S178 line, the K-cluster min-pool value
aggregation (handoff hypothesis #2, "min-pool for losing-cluster
detection") DOES NOT FIRE. There is exactly one window, K=1, so
`min(K)=max(K)=identity`. The colony attractor reproduced on `v6` cannot
be a K-cluster-aggregation artifact. It must live either in (a) the
single-window value head, (b) the bootstrap/corpus, or (c) MCTS dynamics.
This track owns (a). Tracks T1/T3/T4 own (b)/(c)/probes.

K-cluster analysis is still included below (§3) because it is the wave's
assigned question and because v6w25 is a live re-entry target — but it is
explicitly NOT the §S178 capture channel.

---

## 1. Dual-pool value-head asymmetry

### 1.1 Code path

v6 value head, single window (`network.py:787-796`):

```
v_avg = out.mean(dim=(2, 3))           # (B, C)   GLOBAL AVERAGE POOL
v_max = out.amax(dim=(2, 3))           # (B, C)   GLOBAL MAX POOL
v = torch.cat([v_avg, v_max], dim=1)   # (B, 2C)
v = F.relu(self.value_fc1(v))          # (B, 256)
v_logit = self.value_fc2(v)            # (B, 1)
value = torch.tanh(v_logit)
```

Identical math in the shared helper `min_max_window_head`
(`network_min_max_head.py:87-94`) used by every `has_pass_slot=true`
encoding (v6/v6w25/v7full/v7/v7e30/v7mw). Module construction:
`value_fc1 = nn.Linear(2*filters, 256)`, `value_fc2 = nn.Linear(256, 1)`
(`network.py:559-560`). Docstring `network.py:9` confirms:
`Value: GAP+max → FC(2C→256) → ReLU → FC(256→1) → Tanh`.

### 1.2 The asymmetry

The value head has **two pooling halves with opposite spatial selectivity**,
concatenated and fed to a learned `fc1`:

| half | operator | property |
|---|---|---|
| `v_avg` | `mean` over H×W | sensitive to COVERAGE (how many cells active) |
| `v_max` | `amax` over H×W | sensitive to PEAK only; BLIND to coverage |

There is no documented min-pool in the single-window v6 value head — the
"min-pool for losing-cluster detection" in CLAUDE.md refers to the
K-cluster `value_pool="min"` (§3), a different axis. Within one window the
spatial reduction is **avg + max**, and `max` is a one-sided detector:
`v_max[c]` rises monotonically when ANY single cell on channel `c` rises,
and never falls when other cells fall.

**This is the structural "single high-value cluster ⇒ high value" path the
task asks about.** A trained `fc1`/`fc2` that puts positive weight on the
`v_max` block has a monotone route from "one saturated activation cell" to
a high value logit, with `tanh` the only ceiling. The avg half can only
*temper* this if `fc1` learns to subtract it — there is no architectural
guarantee it does.

### 1.3 Probe evidence (random + adversarial weight sets)

`probe_architecture.py` §[1], untrained random value weights, synthetic
trunk maps (`colony_blob` = dense 6×6 patch; `extension_line` = thin 1-wide
5-cell ray; both peak activation 4.0):

```
  colony_blob       value=+0.7257  avg_pool_norm=4.513  max_pool_norm=45.255
  extension_line    value=+0.7796  avg_pool_norm=0.627  max_pool_norm=45.255

  Ablation — which pool half carries the signal?
    colony_blob     logit_full=+0.9195  avg_only=-0.1866  max_only=+1.0693
    extension_line  logit_full=+1.0444  avg_only=-0.0259  max_only=+1.0693

  GMP(colony) == GMP(extension) ?  max|diff|=0.000000   <- max pool BLOB-vs-LINE BLIND
  GAP(colony) / GAP(extension) ratio = 7.20x            <- avg pool separates them
```

Two structural facts, both architectural (hold for ANY weight set):

1. **`GMP(colony) ≡ GMP(extension)` exactly** (max|diff| = 0.000000). The
   max-pool half cannot distinguish a 36-cell colony from a 5-cell
   extension at equal peak activation. Half the value-head input is
   colony-vs-extension blind by construction.
2. **The max half dominates the logit.** `max_only` logit (+1.07) ≈
   `full` logit; `avg_only` logit is near zero / negative. With random
   weights the value head is already mostly reading the coverage-blind
   half.

Adversarial weight set `colony_pos` (`fc1` max-block forced positive,
`fc2` forced positive — models a net that *learned* "high max ⇒ winning"),
probe §[2] colony peak sweep:

```
  peak= 0.5  value=+1.0000  logit=+2237      <-- SATURATED
  peak= 8.0  value=+1.0000  logit=+35793     <-- SATURATED
```

Monotone, no head-internal ceiling — `tanh` saturates the output at +1
for any colony with non-trivial peak activation. A net that learns the
positive-`w2`-on-`v_max` solution **locks colony positions at value ≈ +1
regardless of threat content.** The architecture permits — does not
forbid — exactly the failure the handoff describes ("value-head colony
bias … high-value-everywhere positions saturate value head").

### 1.4 Verdict on dual-pool

The dual-pool value head is **structurally asymmetric and structurally
permits a colony-value-saturation solution.** The `v_max` half is a
coverage-blind monotone peak detector; the `v_avg` half is the only
counterweight and has no architectural guarantee of being used. This is a
PERMISSIVE bias (the architecture allows the bad solution) not a FORCED
bias (the architecture does not *compel* it — a net could learn negative
`v_max` weights). Permissive + a colony-rich gradient (T1's corpus, T3's
MCTS dynamics) = the basin is reachable and has no architectural wall.

---

## 2. K-cluster colony amplification (v6w25 path)

NOTE: not the §S178 capture channel (§0). Analysis retained per wave scope
and for the live v6w25 re-entry target.

### 2.1 Code path

v6w25 K-cluster aggregation, K windows per position:

- **value pool = `min` over K** — `engine/src/game_runner/worker_loop/inner.rs:589-593`:
  ```
  let mut min_v = leaf_values[0];
  for &v in leaf_values { if v < min_v { min_v = v; } }
  aggregated_values.push(min_v);
  ```
  Python mirror `MinMaxPool.forward` `pooling.py:110`:
  `value = per_cluster_values.min(dim=1).values`.
  Bot mirror `k_cluster_mcts_bot.py:371`: `value = float(values_K.min())`.
- **policy pool = scatter-max over K** — `engine/src/game_runner/records.rs:64-75`:
  per legal move, `max_prob` across all clusters covering that move.
  Python mirror `_aggregate_priors` `k_cluster_mcts_bot.py:148-155`.

### 2.2 The K-axis asymmetry

Probe §[3], K=8 (1 colony cluster reading high + 7 sparse clusters):

```
  per-cluster values: colony=+0.7514  7x sparse=+0.2578
  min-pool aggregate = +0.2578  (mean would be +0.3195)

  policy scatter-max: colony move prob in colony cluster = 0.979
  after scatter-max + renorm, colony move prob = 0.499  (argmax cell = 10)
```

**Value min-pool is colony-RESISTANT** — a lone high colony cluster does
NOT pull the aggregate up; `min` picks the weakest cluster. Handoff
hypothesis #2's "single high-value cluster ⇒ high value" is FALSE for the
v6w25 value axis: it is the inverse, "single LOW cluster ⇒ low value".

**Policy scatter-max is colony-AMPLIFYING** — the colony cluster's sharp
high-prob peak survives the max across K and dominates after renorm
(0.979 → 0.499, still the argmax). Sparse not-yet-lost clusters do NOT
suppress it because scatter-max only ever takes the max.

This is the genuine K-cluster structural finding: **value and policy
reduce K with OPPOSITE operators (min vs max).** The value head conserves
the worst case; the policy head promotes the best single cluster's
favorite move. A colony cluster, by construction dense and sharp-peaked,
is exactly the cluster whose policy peak wins scatter-max. So even with a
colony-resistant value aggregate, the v6w25 *policy* would still surface
colony moves. But again — this is v6w25, not the §S178 v6 line.

---

## 3. Encoding channel asymmetry — extension vs colony

### 3.1 Cluster windowing centres on stone density

`get_cluster_views` (`engine/src/board/state/cluster.rs:42-144`):

- `get_clusters` (`engine/src/board/moves.rs:371-413`) — BFS connected
  components under `hex_distance <= cluster_threshold` (5 for v6/v7mw, 8
  for v6w25).
- Small clusters (`span <= window-4`): one window centred on the cluster
  bbox midpoint — `cluster.rs:79` `((min_q+max_q)/2, (min_r+max_r)/2)`.
- Massive clusters: one window per action/threat anchor — `cluster.rs:82-112`.

The window centre is **geometry of existing stone density**, not of the
THREAT line. A compact colony IS a dense connected component → exactly one
tight window owns it. A long thin open-ended extension whose stones are
spread can either (a) span > `window-4` → become a "massive cluster" with
per-anchor windows, or (b) at v6 single-window K=1, fall in one 19×19
window only if it fits.

### 3.2 Diffusion probe

Probe §[4], compact 5-in-row vs spread 5-in-row, best single half-board
window coverage:

```
  compact extension: best single-window coverage = 1.00
  diffuse extension: best single-window coverage = 0.60
```

A diffuse extension fragments — the best window owns only 60% of the
threat activation. Under K-cluster scatter-max the per-cluster policy then
sees a *partial* line; logits diffuse; scatter-max cannot concentrate
visit-driving prior on the single 6th-move cell. A **compact colony never
fragments** — it is dense by definition, always wholly owned by one
window, always gets a sharp per-cluster policy peak.

**Encoding-level asymmetry:** the cluster windowing rewards spatial
compactness. Colony (compact) → sharp owned policy. Extension (potentially
diffuse) → fragmented policy. The encoding makes the colony the
*easier-to-represent* structure. For the §S178 v6 single-window path this
is weaker (K=1, one 19×19 window covers most realistic positions) but the
underlying bias — density-centred cropping favors compact structure — is
present in the windowing rule itself.

---

## 4. Aux-head interaction

### 4.1 Code path

- Aux head = `opp_reply_head` / `opp_reply_conv`+`opp_reply_fc`, a mirror
  of the policy head (`network.py:554-556`, `network.py:9` "Opp_reply
  (aux): mirror of policy head").
- Loss `compute_aux_loss` (`hexo_rl/training/losses.py:86-107`): cross
  entropy of the aux log-policy against **`target_policy`** — the SAME
  noisy MCTS visit target that drives `compute_policy_loss`
  (`losses.py:95` "trained on the same noisy MCTS visit targets").
- Weight `aux_opp_reply_weight: 0.15` (`configs/training.yaml:85`),
  train-only, never called from InferenceServer / MCTS
  (`network.py:690`).

### 4.2 Does it reward colony moves?

The aux head predicts the opponent's reply distribution. In self-play both
sides are the SAME network. In a colony regime both sides play colony
moves — so the opponent's reply IS a colony move, and it is **highly
predictable** (low-entropy, compact, density-centred — exactly the
structure §1/§3 show the net represents sharply).

Mechanism: aux loss is minimized when the aux head predicts the opponent
reply well. Colony positions → predictable opponent reply → low aux loss
*achievable*. Extension/threat positions → opponent must find a specific
threat-defusing or threat-extending cell → higher-entropy, harder-to-
predict reply → higher residual aux loss. The aux head therefore has a
**lower loss floor in the colony regime.** At `w=0.15` this is a real,
if modest, gradient component that is *easier to satisfy* when the game
state is colony-like.

This is NOT a direct "reward colony moves" — the aux head does not feed
MCTS. It is an **indirect shared-trunk pressure**: 15% of the policy-shaped
loss term is systematically cheaper to drive down in the colony regime, so
the trunk features that make opponent-reply predictable (compact,
density-peaked) get extra gradient. Those are the same features that make
the *colony* representable. Co-adaptation, not direct reward.

Severity: SECONDARY. `w=0.15`, train-only, indirect. It mildly reinforces
the basin; it does not create it. Quantifying the corpus-level
colony-vs-extension opp-reply entropy delta needs the actual corpus +
checkpoint (T1's surface, not inspection-only) — flagged for T1 as a
cheap follow-up: bucket `bootstrap_corpus_v6.npz` positions by
colony/extension and measure target_policy entropy of the opponent's move.

---

## 5. Architectural contrast — clustered vs global (thought experiment)

What a non-clustered, KataGo-style single global value head would do
differently. NO implementation — design contrast only.

| axis | current v6 (HexTacToeNet) | KataGo-style global |
|---|---|---|
| spatial reduction | GAP **+ GMP** concat → fc | GAP only, or attention-pooled; KataGo value head has NO max-pool over the board |
| coverage blindness | half the value input (`v_max`) is coverage-blind (§1.3) | pure GAP value head sees only coverage — colony (dense) and extension (sparse) produce *different* pooled vectors |
| board frame | density-centred cluster crop (§3) | whole board, fixed frame, off-board mask plane |
| colony saturation route | monotone `v_max`→logit, no head ceiling (§1.3) | no max half → no single-cell monotone route; saturation requires the WHOLE board to read high, much harder |
| extension representation | can fragment across crops (§3.2) | whole-board conv sees the entire line in one frame; threat is never split |

**The two architectural levers that would change the picture:**

1. **Drop the `v_max` half** (or replace it with a second `v_avg` at a
   different scale). Removing the coverage-blind monotone peak detector
   removes the structural single-cell→value route. The value head would
   then read coverage, and colony (dense) vs extension (sparse) would
   produce distinguishable pooled vectors *before* `fc1` ever sees them.
2. **Whole-board fixed frame** instead of density-centred cropping. The
   cluster crop centres on where the stones ARE (density) — which is the
   colony. A fixed off-board-masked frame (KataGo) never lets the colony
   define the coordinate origin, and never fragments an extension.

KataGo does NOT use a board-spatial max-pool in its value head — its value
head GPool is *multi-board-size* normalization only (`network.py:23-24`
"KataGo's value-head GPool is multi-board-size-only"). Our `v_max` half is
a HeXO-specific addition. That addition is the single most colony-
permissive element identified in this audit.

---

## 6. VERDICT

```
┌────────────────────────────────────────────────────────────────────┐
│ VERDICT: ARCHITECTURAL-BIAS-CONFIRMED  (PERMISSIVE, not FORCED)      │
└────────────────────────────────────────────────────────────────────┘
```

The network architecture **structurally permits and mildly reinforces**
colony interpretation independent of training data. It does NOT *compel*
it (a net could learn negative `v_max` weights), so the bias is
PERMISSIVE — the architecture removes the wall, the data + MCTS pick the
basin.

Confirmed structural elements (all citation-pinned):

1. **Dual-pool `v_max` half is a coverage-blind monotone peak detector.**
   `network.py:790-791`. `GMP(colony) ≡ GMP(extension)` exactly at equal
   peak (probe §[1], max|diff|=0.0). Provides an unobstructed monotone
   route from one saturated cell to value≈+1 (probe §[2]). No
   architectural counterweight is guaranteed. — **PRIMARY**, applies to
   the §S178 v6 line.

2. **Density-centred cluster windowing favors compact structure.**
   `cluster.rs:79`, `moves.rs:402`. Colony (dense) is always wholly owned
   by one window; extensions can fragment (probe §[4], 1.00 vs 0.60
   coverage). — **SECONDARY** for v6 single-window; stronger for v6w25.

3. **K-cluster value/policy reduce K with opposite operators.**
   `inner.rs:589-593` (value `min`) vs `records.rs:64-75` (policy
   scatter-max). Value is colony-resistant; policy is colony-amplifying
   (probe §[3]). — **v6w25 ONLY, NOT the §S178 capture channel.**

4. **Aux head has a lower loss floor in the colony regime.**
   `losses.py:86-107`, `training.yaml:85` (`w=0.15`). Indirect shared-
   trunk co-adaptation pressure. — **SECONDARY**, mild reinforcement.

What this audit does NOT claim: it does not claim the architecture is the
SOLE cause. The handoff's #1 hypothesis (bootstrap+corpus jointly encode
colony bias) is T1's surface and remains the most likely *primary* driver.
This track's finding is that the architecture offers **no resistance** —
specifically the `v_max` half is a colony-permissive element with no
counterpart in KataGo's value head.

---

## 7. Surgical architectural alternatives (ranked by implementation cost)

CONFIRMED ⇒ alternatives ranked. All require a fresh bootstrap pretrain +
sustained run to validate (architecture changes ⇒ checkpoint shape
changes). Cost = code + the unavoidable re-pretrain.

### A1. Re-weight, don't remove — `v_max` channel-gate (LOWEST cost)
Add a learnable per-channel scalar gate on the `v_max` block before
`cat`, init 1.0. ~15 LOC in `network.py` + `network_min_max_head.py`.
Lets the net learn to *suppress* the coverage-blind half if it hurts;
state-dict gains one `(filters,)` parameter. Does NOT force a fix — still
permissive — but gives the optimizer an explicit knob and a loggable
gate-value signal (mirrors `gpool_bias_gate_value`). Cheapest, weakest.
**Cost: ~15 LOC + 1 re-pretrain.**

### A2. Replace `v_max` with multi-scale `v_avg` (MEDIUM cost)
Drop GMP; replace with average pools at 2-3 spatial scales (e.g. global
mean + mean over 2×2 blocks pooled). Removes the coverage-blind monotone
route entirely; value head reads only coverage at multiple resolutions, so
colony (dense) and extension (sparse) are *always* separable pre-`fc1`.
Changes `value_fc1` input dim (`2C` → `3C` or chosen). ~40 LOC + the
`fc1`-shape state-dict break. **Cost: ~40 LOC + 1 re-pretrain. RECOMMENDED
— directly removes the PRIMARY element from §6.1.**

### A3. Add a colony-penalty value-head auxiliary target (MEDIUM cost)
Keep the head; add a small auxiliary value-head loss that penalizes value
> threshold on positions flagged colony by the existing colony detector
(Q11 RESOLVED on detection). Trains the value head AWAY from colony
saturation directly. ~60 LOC in `losses.py` + trainer wiring + needs the
colony flag plumbed to the value-loss site. Does not change architecture
shape (no re-pretrain needed for shape — but needs a sustained run to
take effect). **Cost: ~60 LOC, no shape break.**

### A4. Whole-board fixed-frame encoding for v6 (HIGHEST cost)
Replace density-centred cluster cropping with a KataGo-style fixed
off-board-masked frame (the v8 family already has the mask machinery —
`compute_v8_mask`, `network.py:717`). Removes §6.2 entirely + makes the
colony unable to define the coordinate origin. This is effectively
"adopt the v8 encoding for the sustained line" — a new encoding registry
entry + bootstrap regen + full re-pretrain. Largest blast radius; also the
most complete fix. **Cost: new encoding + corpus regen + re-pretrain;
multi-day. Defer unless A2/A3 fail.**

**Recommended sequence:** A2 (remove the PRIMARY coverage-blind route)
paired with A3 (direct anti-colony value gradient). A2 alone changes the
architecture; A3 alone changes the gradient; together they attack the
permissive-architecture + colony-rich-data combination this audit
identified. A1 is a cheap pre-test if a re-pretrain budget is tight. A4
is the fallback if the value-head fixes are insufficient and §3.2 frame
instability proves load-bearing.

Do NOT re-propose: PMA/global pool (§170 FALSIFIED), gpool-bias-on-value
(§170 P4 NULL), bbox+canvas_realness+frozen-spine (§171 DEAD), any cosine
schedule (L9 banned), config-knob anti-colony levers (§S180b L38
exhausted).

---

## Files created / modified by IMPL-S181T2

```
audit/structural/02_value_head_encoding_architecture.md     (this file, new)
scripts/structural_diagnosis/probe_architecture.py          (new, standalone)
```

No git add / commit performed (orchestrator commits per track).
No hot-path / config / engine files modified.
