# Lean-D6 adopt-vs-avoid — WP-C schema ruling (GNN-integration program)

**Date:** 2026-07-14 · **Program:** GNN-integration (R4 ratified b+) · **WP-C (COND-2)** ·
**Status:** VERDICT RECORDED — **LEGACY-V1-CONFIRMED**

**Verdict in one line:** run4-v1 ships the **legacy-v1 axis-graph schema**; the rotation leak in
legacy-v1 is REAL but MITIGATED by D6 augmentation (which strix ships for its own legacy runs and
which our own §119 finding independently prescribes) — it does **not** make legacy-v1 unshippable.
Lean-D6 is a **run5 card**, adoptable only on the pre-registered rule in §3.

---

## Standing ruling (restated verbatim from R4 — this doc justifies, does not re-litigate)

> **run4-v1 ships the legacy-v1 axis-graph schema — the exact schema that produced the BC probe's
> Δ=+414 [+320,+560] result. The single evidence-bearing variable in run4 = architecture, nothing
> else. Lean-D6 is a SEPARATE lever decided on evidence.**

This document's job: justify that ruling from strix's source, OR escalate it if legacy-v1 is found
actually unshippable (a training-poisoning rotation leak, quantified). It does **not** re-open the
schema choice by vibes. Finding below: legacy-v1 is shippable; escalation not triggered.

Evidence base for the ruling: the BC probe Δ=+414 ran on legacy-v1 — 11-dim relative+threat node
features, GINE `(E,5)` edge_attr, single edge list with unified all-zero dummy edges
(`hexo_rl/bots/strix_v1_graph.py::build_axis_graph_raw`, ported at strix SHA `c381ffbe`). Any
schema change forfeits that measurement (see §2 cost 1).

---

## 1. What strix's lean-D6 actually is

Read from strix current source (`SootyOwl/hexo-strix`, local shallow HEAD `031d309`; the port SHA
`c381ffbe` is not in the shallow history, so the comparison below is current-strix-source vs our
vendored legacy-v1 builder, not a SHA diff).

Lean-D6 is a **from-scratch-only schema + conv change** that makes the network **exactly
D6-invariant by construction** (D6 = the 6 rotations × 2 reflections of the hex lattice, 12
elements). It is gated behind `LeanOpts` flags that **all default to the legacy schema**, so
`LeanOpts::default()` reproduces today's byte-exact legacy output
(`hexo-rs/hexo-mcts/src/axis_graph.rs:24-48`). Motivation, from the config header verbatim
(`configs/gine-mini/4l-128p32v-lean-d6.toml:5-9`):

> Fixes the "symmetric moves rated differently" bug (chan, 2026-07-03) BY CONSTRUCTION: the 3 hex
> axes are an edge-TYPE partition consumed by tied-weight AxisRelationalConv layers (symmetric
> per-axis sum), with an invariant node schema (empty one-hot + norm-q/r dropped) and unsigned edge
> distance. Symmetric positions get bit-identical evaluations.

### 1a. Schema changes vs the legacy graph (cited)

Four flags, each `FROM-SCRATCH ONLY` (`configs/.../config.py:36-42`):

| Flag | Legacy (our v1) | Lean-D6 | What it drops / changes |
|---|---|---|---|
| `axis_relational` | `false` — axis one-hot in `edge_attr[0:3]`, signed dist `[3]`, `src_player [4]`; single `(E,5)` GINE edge list; dummy edges appended to that list | `true` — axis becomes integer `edge_type ∈ {0,1,2}` + **unsigned** `edge_dist`; **`edge_attr` cleared**; dummy edges routed to a **separate** `global_edge_src/global_edge_dst` list (`axis_graph.rs:436-465`) | Drops `src_player` edge feature AND the distance **sign** (both "redundant") |
| `compact_stone_onehot` | `false` — `[own, opp, empty, …]` | `true` — drops the `empty` one-hot | node base −1 |
| `node_coords` | `true` — `norm_q, norm_r` present | `false` — dropped | node base −2; comment (`config.py:41`): "These are a **rotation-symmetry leak** and largely redundant with the axis-edge geometry" |
| `moves_scope` | `"node"` (per-node column) | `"node"` (unchanged in the shipped lean config) | — |

Net node-feature width: **legacy 11** (relative-7 base + 4 threat) → **lean-D6 8** (relative-7 −
empty(1) − coords(2) = 4 base + 4 threat). Confirmed by the config header "node dim 8 vs legacy 11"
and `node_feature_dim()` (`config.py:45-59`).

### 1b. AxisRelationalConv (`hexo-a0/src/hexo_a0/axis_conv.py`)

The conv that consumes the relational edges. Structure (docstring, verbatim):

```
a_k = Conv_shared(x, edges[type==k], dist_embed(dist[type==k]))  k in 0..2
g   = Conv_global(x, global_edges)          # optional, its OWN weights
x'  = node_update([x, a_0 + a_1 + a_2 + g]) # symmetric SUM over axes
```

`Conv_shared` is **ONE** `GINEConv` reused across all 3 axes with **tied weights**; the three
relation outputs are combined by a permutation-symmetric SUM. Because GINE aggregation is additive
and the weights are shared, relabelling the axes only permutes the summands → **exact
axis-permutation invariance, no axis-perm augmentation needed** (`axis_conv.py:141-150`). The
global/dummy relation is a 4th edge type with its OWN untied branch, kept OUT of the axis sum so it
cannot break the symmetry.

### 1c. What lean-D6 buys strix (claims found; measured-vs-asserted)

- **Rotation/axis-relabel leak kill — ASSERTED-BY-CONSTRUCTION, no measured result in repo.** The
  "symmetric moves rated differently" bug is real (config header, dated 2026-07-03). The fix is
  architectural (invariance by construction), not an empirical claim; strix cites a design spec
  `docs/superpowers/specs/2026-07-03-d6-invariant-axis-architecture-design.md` that is **NOT
  present in the shipped tree** (shallow clone; only the configs are committed).
- **12× D6 augmentation — NOT a lean-D6 benefit; it is a legacy REQUIREMENT that lean-D6 makes
  redundant.** `augment_symmetries` defaults `True` for ALL schemas ("Apply D6 symmetry
  augmentation (12x training data via hex rotations/reflections)", `config.py:218`), and the
  lean-D6 config turns it **OFF** precisely because "the model is exactly D6-invariant"
  (`4l-128p32v-lean-d6.toml:123`). So the 12× is what **legacy** relies on; lean-D6's benefit is
  being able to DROP it (no augmentation compute, no 12× data blow-up, and invariance guaranteed
  rather than learned).
- **No committed evidence lean-D6 beats legacy.** No `runs/` results, no Elo/eval artifact, no
  head-to-head is committed. The lean-D6 config is `total_train_steps = 12000` on a `gine-mini`
  (4-layer/128p) net — an experimental scaffold. **Every measured strix strength number that grounds
  our re-open (strix-g128 +313 deploy, strix-raw +121, `gnn_readjudication.md` §2) was produced by
  the LEGACY schema** (the `c381ffbe` port is legacy). Lean-D6 is, as of `031d309`, an unvalidated
  schema even for strix.

### 1d. Divergence: current strix vs our vendored legacy-v1 builder

Our port (`strix_v1_graph.py`, legacy-only) is **byte-parity with current strix's `LeanOpts::default`
legacy path** — spot-verified: the dedup-by-`(src,dst,axis_idx)` keep-first and the bidirectional
all-zero `[0.0;5]` dummy-edge append match (`axis_graph.rs:400-478` vs `strix_v1_graph.py:264-291`);
node layout, threat fill, and edge walk match. **The only divergence is that current strix ADDS the
`LeanOpts` machinery our port does not carry** (relational/compact/coords/moves flags + the
`edge_type`/`edge_dist`/`global_edge_*` output fields). Under the legacy default those fields are
empty and the output is identical. **No drift crept into the legacy default** — strix's own
docstrings pin it ("`LeanOpts::default()` reproduces today's byte-exact output",
`axis_graph.rs:24-25, 540-544`), and the parity is what makes our vendored builder a valid oracle.

### 1e. Load-bearing: lean-D6 has NO native forward, even for strix

`hexo-infer` (strix's pure-Rust GNN forward, the serving fast-path) **rejects any non-GINE conv**:
`if conv_type != "gine" { return Err(UnsupportedConfig) }` (`hexo-infer/src/weights.rs:187-188`),
and its request struct leaves the relational fields empty, "never read by `forward`"
(`hexo-infer/src/server.rs:40, 80-83`). The **wasm/browser** path wraps hexo-infer
(`hexo-wasm/src/lib.rs:3, 14, 48`; `Cargo.toml:15`) — so it is gine-only too. Lean-D6 runs ONLY
through torch/torch_geometric (or `torch.jit.script` → tch-rs via `scriptable_model.py`). Strix's
own optimized inference and browser deploy ride the **legacy GINE schema**. Adopting lean-D6 would
put us on a path strix themselves have not productionized.

---

## 2. Costs of adopting lean-D6 for run4-v1 (top 3)

**Cost 1 — Invalidates the +414 evidence base; run4's architecture bet rides on UNMEASURED
representation.** The BC probe Δ=+414 [+320,+560] measured `{axis-graph representation}` in the
legacy-v1 schema (11-dim node, GINE `(E,5)`). Lean-D6 is a materially different representation
(8-dim node, `edge_type`/`edge_dist`, AxisRelationalConv) — strix marks it "incompatible with
existing checkpoints; do NOT graft; start a clean run" (`4l-128p32v-lean-d6.toml:12-13`). Adopting
it means run4's entire architecture bet is placed on a representation with **zero measurement** on
our data, in combination with nothing. This directly violates the standing ruling's
single-evidence-bearing-variable discipline: run4 would test `{axis-graph AND lean-D6-schema}`
jointly, having measured neither the schema alone nor the combination.

**Cost 2 — Second variable + schema churn on the highest-blast-radius component, mid-program.** WP-B
is designing the ragged-payload contract against legacy-v1 **right now** (11-dim node, single `(E,5)`
edge list, unified dummy edges). Lean-D6 changes the payload shape on every axis WP-B is pinning:
node 11→8, `edge_attr` replaced by `edge_type`+`edge_dist`, and a **new** `global_edge_index` list
added. That is a different contract — it forces a WP-B redesign and moves the C8 target (buffer
`push`/`storage`/`sample`/`persist` + graph symmetry) mid-flight. The scope doc names this seam as
**RISK 1**: "a subtle graph-collate / offset / symmetry bug silently corrupts self-play with no loud
failure — the exact D-FORENSIC F1 failure class" (`gnn_integration_scope.md` §"two biggest risks").
Churning the schema under it maximizes that risk for no measured gain.

**Cost 3 — Worse ONNX/wasm export surface, with NO native forward fallback.** WPA already found the
exported **legacy** GINE materializes E×hidden intermediates (`Expand`+`Slice`) and OOMs ORT-CUDA at
bs=256 (`WPA_cuda_bench.md` verdict 3). Lean-D6's AxisRelationalConv is **strictly harder** to
export: its forward does a per-relation Python loop with a **boolean-masked dynamic gather**
(`edge_index[:, edge_type==k]`) ×3 axes plus a separate global branch (`axis_conv.py:146-159`) —
dynamic-shape scatter/gather that onnxruntime-web handles poorly, at 3–4× the conv invocations. And
per §1e there is **no native Rust forward for lean-D6** — strix's `hexo-infer`/wasm path is
gine-only. So adopting lean-D6 forfeits the ORT/wasm deploy lever WPA scoped (`WPA_cuda_bench.md`
verdict 3: "ORT remains the browser/WASM path only") with no fallback that exists today. Legacy-v1
keeps that lever open.

(Runners-up, not in the top 3 but noted: lean-D6's tied-per-axis masked loop makes block-diagonal
batching MORE ragged, mildly adverse to the WPA BUILD-HOT / batching finding; and it is
unvalidated even by strix, §1c.)

---

## 3. Decision rule (pre-registered — mechanically evaluable once WP-B lands)

Default = **AVOID for run4-v1**. Propose **ADOPT** only if BOTH:

- **(a)** WP-B declares graph-space D6 augmentation **INFEASIBLE-ON-LEGACY-V1**, AND
- **(b)** the projected loss from shipping legacy-v1 **without** augmentation is **≥ X**, where
  **X = 30% effective-sample-size loss, OR any projected §119-class axis-bias pathology** (whichever
  fires first — the pathology clause is the teeth; the 30% is the graceful-degradation backstop).

### Why the gate is augmentation feasibility, not schema adoption

The rotation leak (§4) is the ONLY thing that could make legacy-v1 unshippable, and its standard
remedy is **D6 augmentation**, not re-architecture. Strix mitigates the identical leak in its own
**legacy** runs with `augment_symmetries=True` (§1c). Crucially, strix already ships the exact Rust
machinery to augment the **legacy** schema: `augment_axis_graph_all_opts(game, prune, threat,
relative)` applies the 11 non-identity D6 transforms to the legacy axis-graph **with threat +
relative flags = our legacy-v1** (`axis_graph.rs:618-655`), returning the legal-index permutation for
policy remap. So graph-space D6 augmentation on legacy-v1 is **demonstrably feasible** (strix does
it) — making WP-B's INFEASIBLE verdict a priori unlikely, and the correct first move in the unlikely
INFEASIBLE case is to **port that machinery**, not to adopt lean-D6 (which forfeits the +414
evidence base, Cost 1).

### Justification of X = 30%

- **Pathology clause (primary).** Our §119 D5/D6 finding measured that un-augmented data (67% of the
  batch un-augmented) produced a *qualitative correctness* pathology in the CNN — self-play E-W axis
  share 65% vs corpus 38% (+27pp), a persistent axis bias, not merely slower convergence
  (`07_PHASE4_SPRINT_LOG.md:1128`). A §119-class pathology is an effectively-unbounded correctness
  cost, so its projection escalates regardless of the % number.
- **30% backstop (efficiency-only branch).** D6 has 12 elements, but augmented copies are highly
  correlated, so the effective-sample-size gain is far below the nominal 12×; AlphaZero-family
  practice (8× dihedral in Go) treats symmetry augmentation as a ~2–5× *early-stage* effective-data
  lever tapering to less. A 30% effective-sample loss ≈ ~1.4× more GPU-weeks to matched strength.
  WPA already projects the GNN at ~0.9–1.25k steps/hr vs run2's 4.4k — **~4–5× slower per step**
  (`WPA_cuda_bench.md` throughput projection). A further 1.4× data-inefficiency **compounds** onto an
  already-strained throughput budget; at that compounding point lean-D6's invariance-by-construction
  (which recovers the augmentation benefit with no 12× blow-up and no per-sample augment cost) begins
  to pay for its adoption cost. Below 30% (graceful), the +414 evidence-base preservation + no-second-
  variable + the export/wasm lever dominate → still AVOID.

### Mechanical evaluation table (dispatcher applies once WP-B returns)

| WP-B verdict | (b) loss projection | Ruling |
|---|---|---|
| **FEASIBLE-ON-LEGACY-V1** | — | **AVOID.** Ship legacy-v1 + graph-space D6 augmentation (port `augment_axis_graph_all_opts`). Leak mitigated. Rule terminates; lean-D6 → run5 card. |
| **INFEASIBLE-ON-LEGACY-V1** | loss **< X** (graceful, no pathology projected) | **AVOID (with canary).** Ship legacy-v1 un-augmented; accept the hit; run the §119 axis-share canary (self-play E-W axis share vs corpus) as an in-run gate. lean-D6 → run5 card; escalate mid-run only if the canary fires. |
| **INFEASIBLE-ON-LEGACY-V1** | loss **≥ X** (≥30% or §119-class pathology projected) | **ESCALATE to operator.** legacy-v1 un-augmented is the only measured-schema option but is data-poisoning. Two remedies: (i) make augmentation feasible (port strix's Rust augment) — **preferred, preserves +414 evidence base**; (ii) adopt lean-D6 — forfeits the evidence base (Cost 1), run4 rides unmeasured representation. Operator decides. This doc's recommendation even here: (i) before (ii). |

### Lean-D6 as a run5 card — evidence that would justify it

Promote lean-D6 from run5-card to adoption only on: a committed strix (or our own) measurement of
**lean-D6 ≥ legacy at matched compute** (currently absent, §1c); OR a WP-B INFEASIBLE + §119-class
pathology that a legacy-v1 augmentation port cannot close; OR a decision to make the browser/wasm
deploy path primary AND a native lean-D6 forward being built (today it does not exist, §1e). Absent
all three, legacy-v1 stands.

---

## 4. Rotation-leak risk found in legacy-v1 (the escalation test)

**Finding: the leak is REAL but MITIGATED-BY-AUGMENTATION — legacy-v1 is NOT unshippable.**

Evidence the leak is real in legacy-v1:
- Legacy-v1 node features carry absolute `norm_q, norm_r` coordinates (`strix_v1_graph.py:162-166`),
  which strix labels a "rotation-symmetry leak" (`config.py:41`).
- Legacy-v1 edges carry an **absolute** axis one-hot + **signed** distance (`edge_attr[0:3]`, `[3]`,
  `strix_v1_graph.py:243-255`) — none D6-invariant; the network CAN learn axis-asymmetric /
  rotation-asymmetric features, exactly the "symmetric moves rated differently" bug strix's config
  header names.

Why it does NOT trigger ESCALATE:
- The standard remedy is D6 augmentation, and strix applies it to legacy by default
  (`augment_symmetries=True`, §1c). The leak makes augmentation **mandatory** for a legacy-v1 run;
  it does not make the schema unshippable.
- Strix ships the legacy-schema augmentation machinery ready to port (`axis_graph.rs:618-655`, §3).
- The leak's blast surface on a GNN is *smaller* than on the CNN that §119 diagnosed: the GNN's
  policy is read per-legal-node through relational message passing, not through the CNN's flat
  absolute-position FC head, so it is less absolute-position-bound.

This is precisely why the ruling is LEGACY-V1-CONFIRMED, not ESCALATE: the only unshippability
candidate is a mitigable leak, and its mitigation (augmentation) is the WP-B gate in §3 — not a
schema swap.

---

## 5. Re-validation discipline (CLAUDE.md — priors cited, context tested for transfer)

- **§119 D5/D6 (CNN main-island axis bias).** *Original context:* CNN absolute-position FC policy
  head; un-augmented RecentBuffer (67% of batch) → axis-asymmetric features → E-W self-play bias;
  **resolution = augment the RecentBuffer, not re-architect** (`07_PHASE4_SPRINT_LOG.md:1128`).
  *Transfer test:* the GNN legacy-v1 schema has analogous rotation-leaking features (absolute coords +
  absolute axis one-hot), so the *mechanism* (un-augmented → asymmetric) plausibly transfers.
  *But §119's own RESOLUTION was augmentation* — so it transfers as **"the leak is real AND
  augmentation is the fix,"** NOT as "legacy-v1 is unshippable." Used in §3/§4 to make augmentation
  feasibility the gate and to source the X pathology clause. Kept, scoped.
- **§D-STRIX kernel REJECT (falsified register, `07_PHASE4_SPRINT_LOG.md:566`).** *Original context:*
  ragged-batch forward-throughput perf; "HeXO's dense CNN+attention has no ragged-batching problem."
  *Transfer test:* this program is a GNN, which DOES have a ragged problem (WPA confirmed) — so the
  kernel-kill's *premise* no longer holds here, but the row is CUDA-kernel/perf-scoped and does not
  adjudicate representation (per its own 2026-07-13 scope footnote). It bears on neither lean-vs-legacy
  choice. **Not transferred; not relied on.** (Noted only: lean-D6's masked per-axis loop makes
  batching *more* ragged, a minor perf/export negative, §2 runner-up.)
- **D-FULLSPEC ENTANGLED.** *Original context:* frozen v6_live2 **CNN** features could not separate
  win/loss even class-balanced → a CNN-representational fix was needed. *Transfer test:* it is
  CNN-feature-specific and says nothing about GINE-legacy vs relational-lean edge/node schemas.
  **Does not transfer** to the schema choice; not relied on.
- **NOTE-ONLY axis-graph representation card (`gnn_readjudication.md` §1b).** *Original context:*
  banked, restart-gated, never falsified; RE-OPENED 2026-07-13 pending the BC probe. *Transfer test:*
  the re-open + the +414 probe are the evidence base for shipping the axis-graph AT ALL — and that
  evidence is **legacy-schema** (§1c). Transfers directly as the reason the shipped schema must be
  the one that was measured. Kept; it is the backbone of Cost 1.

---

## Verdict

**LEGACY-V1-CONFIRMED.** run4-v1 ships the legacy-v1 axis-graph schema. The rotation leak in
legacy-v1 is real but mitigable by D6 augmentation (strix ships the legacy-schema machinery; our §119
finding independently prescribes augmentation for the analogous CNN leak) — it does not make
legacy-v1 unshippable, so ESCALATE is not triggered. Adopting lean-D6 for run4-v1 would forfeit the
+414 evidence base, introduce a second variable and mid-program schema churn on the highest-blast-
radius seam, and worsen the ONNX/wasm export surface with no native forward — none justified unless
WP-B returns INFEASIBLE-ON-LEGACY-V1 **and** the §3 loss threshold (X = 30% effective-sample loss or
a §119-class pathology) is met, in which case the preferred remedy is still to make legacy-v1
augmentation feasible before adopting lean-D6. Lean-D6 is a **run5 card** with the promotion evidence
named in §3.

## Provenance

- Standing ruling / evidence base: `docs/designs/gnn_readjudication.md`; `docs/designs/gnn_integration_scope.md`; `reports/probes/gnn_integration/WPA_cuda_bench.md`.
- Legacy-v1 oracle builder: `hexo_rl/bots/strix_v1_graph.py` (ported at strix SHA `c381ffbe`).
- Strix current source (shallow HEAD `031d309`, MIT, `SootyOwl/hexo-strix`, read-only):
  `hexo-rs/hexo-mcts/src/axis_graph.rs` (`LeanOpts` :24-48, relational branch :436-465, legacy dummy edges :466-478, legacy D6 augment :618-655);
  `hexo-a0/src/hexo_a0/axis_conv.py` (AxisRelationalConv);
  `hexo-a0/src/hexo_a0/config.py:36-42, 218` (lean flags + augment default);
  `configs/gine-mini/4l-128p32v-lean-d6.toml` (the lean-D6 config + motivation header);
  `hexo-rs/hexo-infer/src/weights.rs:187-188` + `server.rs:40,80-83` (gine-only native forward);
  `hexo-rs/hexo-wasm/src/lib.rs` (wasm wraps gine-only hexo-infer).
- Prior context: `docs/07_PHASE4_SPRINT_LOG.md` (§119 D5/D6 :1128; §D-STRIX kernel row :566; §GNN-INT :2987).
