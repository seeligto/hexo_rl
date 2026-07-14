# GNN Integration Scoping — R3 of the D-M R-LADDER

**Status:** scoping doc (not a build order). Required input to the R4 operator decision
(`docs/handoffs/run3_convene_ruling_amendment_1.md`, commit `b81ca86`). Sanctioned under
the MIXED mapping as run4 design work regardless of which run3 launches.

**Question this answers:** what does it take to bring a GNN trunk (axis-graph representation,
as in hexo-strix and our `hexo_rl/probes/gnn_bc/` probe) into OUR training / self-play stack,
per-component, from MEASURED scopes — files touched, seams named, precedent cited — not vibes.

**Scope discipline:** estimates below cite the exact functions/traits/configs each component
touches. Where a number was cheaply measurable it was MEASURED (laptop RTX 4060 Max-Q, main
checkout venv, 2026-07-14). Two measurements were run: the probe's Python graph builder on 100
real corpus positions, and the probe GNN forward at self-play batch sizes. Both are recorded in
§C1 / §C6 with hardware caveats.

---

## Measurements (recorded once, referenced throughout)

| Measurement | Value | Method |
|---|---|---|
| `build_axis_graph_raw` (Python, 1 thread) | **5.94 ms / position** | 100 real `raw_human` corpus positions, warm |
| axis-graph size (mean / max) | **290 / 523 nodes, 1294 / 2380 edges** | same 100 positions |
| probe GNN forward, bs=64 | **20.86 ms** (0.326 ms/graph) | disjoint-union batch, fp32, 5-iter warm, cuda-sync |
| probe GNN forward, bs=128 / 256 | **42.6 / 83.1 ms** (~0.325 ms/graph, flat) | scales linearly in edges → edge-scatter bound |
| production CNN forward (4.27M, dist65), bs=64 | **20.07 ms** (autocast fp16) | v6_live2_ls, filters=128, blocks=12, same GPU |
| probe GNN params | 283,970 | `GnnBcNet` hidden=128 / 4 layers |

**Headline finding.** At probe scale the GNN forward (284k params, fp32) is at PARITY with the
production CNN forward (4.27M params, fp16): 20.86 vs 20.07 ms at bs=64. The GNN is 15× smaller in
params yet no faster — its cost is **edge-scatter bound** (158k edges in a 64-graph union), not
FLOP-bound, and it does not benefit from the CNN's autocast. Two consequences drive the whole
scope: (a) a production-SCALE GNN (strix ships 4.25M, hidden ~256) forward will be materially
slower than the CNN, not faster; (b) the GNN carries a mandatory per-leaf **graph-build** cost
(5.94 ms/position in Python) that the CNN's plane-scatter encoder does not. These are the two
throughput risks in the final table.

---

## C1 — Graph builder in our loop

**What it is.** Port axis-graph construction — sorted stone nodes + sorted legal (empty) nodes +
one global/dummy node, 3-axis windowed edges with stop-at-opponent walk + first-wins dedup,
5-dim GINE edge_attr `[axis one-hot(3), signed_dist, src_player]`, per-node features (relative
7-base + 4 threat = 11-dim), bidirectional all-zero dummy edges — into the self-play loop.

**Current state.** A faithful, fidelity-gated Python builder already exists:
`hexo_rl/bots/strix_v1_graph.py::build_axis_graph_raw` (317 lines, ported from strix Rust at SHA
`c381ffbeb2…`, cross-checked by `hexo_rl/probes/gnn_bc/graph_check.py` + `cross_check.py` +
`tests/test_strix_v1_bot.py` against strix's Rust unit vectors). The probe consumes it via
`_compact_example` → `_collate_gnn` (disjoint-union block-diagonal batch) in
`hexo_rl/probes/gnn_bc/train_bc.py`. So the builder is SOLVED for offline BC; the open question
is the in-loop (MCTS-hot-path) build.

**Rust-side vs Python-side — the decision.**

- **Python-side (build in `InferenceServer` from stones shipped over PyO3).** Cheapest to write —
  reuse `build_axis_graph_raw` unchanged, move it behind the inference thread. Files touched:
  `hexo_rl/selfplay/inference_server.py` (new graph-collate branch), plus a new PyO3 surface to
  ship per-leaf stone-lists instead of dense planes. **Fatal for throughput:** MEASURED 5.94
  ms/position × 64 leaves = **~380 ms/batch serial**, dwarfing the 20 ms forward. Even the probe
  needed 8 worker processes + full-dataset materialization to make BC tractable
  (`train_bc.py` "MATERIALIZATION (the perf fix)" note); worker-process fan-out is not available
  on the single-threaded per-move MCTS hot path. Estimate: **1-2 person-days to write, but
  throughput-disqualifying** (see §C6).

- **Rust-side (native builder in the `engine` crate).** Port strix's `axis_graph.rs` /
  `graph.rs::fill_threat_features` / `threat.rs::node_threat_features` (same MIT source the Python
  builder came from) into a new `engine/src/graph/` module, wired to `Board::get_stones()` /
  `legal_moves` (both already native). Precedent for adding a substantial native subsystem:
  `engine/src/tactics/` (the D-ZVALID native minimax) and `engine/src/board/state/encode.rs`
  (`encode_state_to_buffer_channels`, the current plane encoder). **Recommended.** The Rust plane
  encoder runs in <50 µs; the axis-graph walk (290 nodes × 3 axes × window + threat windows) is
  10-30× heavier but still far under the Python 5.94 ms — strix does exactly this build inside
  their 33.8 ms/position deploy figure. Estimate: **5-10 person-days** (native port + PyO3
  exposure + a byte-parity `cargo test` gate against `build_axis_graph_raw`, which is the existing
  oracle).

**Recommendation: Rust-side builder.** The Python builder is disqualified by §C6; the only
viable path is native. The Python builder is retained as the parity ORACLE (a `cargo`/`pytest`
cross-check identical in spirit to `cross_check.py`), which is why the Rust port is cheaper than a
from-scratch build — the reference semantics are already pinned and tested.

---

## C2 — Net + heads (HeXONet-equivalent + dist65-on-GNN value head)

**What ports unchanged.** The net already exists grad-capable:
`hexo_rl/probes/gnn_bc/gnn_bc_net.py::GnnBcNet` wraps
`hexo_rl/bots/strix_v1_net.py` (`RepresentationNetwork` / `_GINEConv` / `PolicyHead` /
`ValueHead`), a self-contained pure-PyTorch HeXONet (no torch_geometric). `forward_batch`
(disjoint-union, grad-capable) is the training path; `policy_logits_for_graph` is deploy. These
move into the training stack essentially as-is.

**dist65-on-GNN value head — VERIFIED reusable (read the E1 code, did not assume).** The E1
distributional primitives live in `hexo_rl/training/binned_value.py`:
- `scalar_to_two_hot(z, n_bins=65)` — operates on `z (N,)`; **pool-agnostic**.
- `decode_binned_value(bin_logits (N,65))` — operates on bin logits; **pool-agnostic**.
- `binned_value_loss(bin_logits, outcome, value_mask)` — masked CE vs two-hot; **pool-agnostic**.

None of these three touch the trunk or the pooling — they consume `(N, 65)` logits and `(N,)`
targets. The production wiring confirms it: `hexo_rl/model/network_min_max_head.py:98-105` decodes
dist65 as `decode_binned_value(value_fc2_bins(v))` where `v` is the CNN's pooled 256-vector, and
`hexo_rl/training/trainer.py:748-749` already calls `binned_value_loss(value_aux, outcomes_t,
value_mask_t)` when `value_head_type=="dist65"`. **The loss and decode primitives port unchanged.**

**What needs a pooled-input variant.** Exactly ONE layer: the linear that produces the 65 bin
logits. In the CNN it is `value_fc2_bins = nn.Linear(256, 65)` fed by the avg+max pooled
`v (B,256)`. On the GNN the value head is **stone-pooled**: `emb[stone_mask].mean(dim=0)` over the
JK-cat `L*H = 512`-wide node embeddings (`strix_v1_net.py::HeXONet.forward` / `GnnBcNet`). So the
GNN dist65 head is `nn.Linear(512, 65)` on the stone-pooled vector, then the SAME
`decode_binned_value`. Concretely: replace `ValueHead`'s `nn.Linear(value_hidden, 1) + Tanh` tail
with a `value_hidden → 65` bin-logit tail and route through `decode_binned_value`. The probe's
`GnnBcNet.value_head` is present but unsupervised — this component supervises it.

**Files touched.** New `hexo_rl/model/gnn_net.py` (production HeXONet + dist65 pooled head; lift
from `gnn_bc_net.py` + `strix_v1_net.py`); `hexo_rl/training/trainer.py` (a forward-input branch:
graph batch vs dense tensor — the `binned_value_loss` call is unchanged); `hexo_rl/training/
binned_value.py` (unchanged — reused). **Precedent:** the whole E1 head-swap (`4e5d7fe`
head-type-switchable value head, `6814c7a` route dist65 loss, `60cf720` T1-T5 merge, `7496b08`
warm-start) did precisely this surgery on the CNN; the GNN repeats it against a different pool.
**Estimate: 3-6 person-days** (net lift is cheap; the trainer forward-input branch + a warm-start
loader for a BC-prefit GNN checkpoint is the bulk — mirrors `c796887` E1 warm-start loader).

**Note (single-window, per E1):** the E1 trainer forwards a SINGLE window → dist loss is plain
per-row CE, NO argmin-cluster routing. A GNN is inherently whole-board (one graph per position, no
K-cluster windows) so this simplifies: no per-cluster value pooling, no `value_pool=min`
aggregation. The multi-window machinery in `v6_live2_ls` is **not needed** for a GNN — the graph
IS the global view.

---

## C3 — MCTS / PyO3 interface (the graph-input inference contract)

**This is the largest and highest-risk component.** The entire self-play inference seam is
architected around a **fixed-width dense feature vector**, and a graph is variable-size.

**The fixed-width contract, named exactly.** `engine/src/inference_bridge.rs::InferenceBatcher`
is constructed with a scalar `feature_len` (`= n_planes * board_size²`, derived from the encoding
spec's `state_stride()`, `inference_bridge.rs:295-306`). Every request is a `Vec<f32>` of exactly
`feature_len` (validated at `submit_batch_and_wait_rust:193` and `submit_and_wait:68`), fused into
a dense `PyArray2<f32>` of shape `[batch, feature_len]` (`next_inference_batch:394`). The Python
side (`hexo_rl/selfplay/inference_server.py`) reshapes that flat buffer to `(n, C, H, W)`
(`run:466`), stages it through a **pinned fixed-shape H2D buffer**
`_h2d_staging (batch_size, C, H, W)` (`__init__:99`), and forwards a **TorchScript-traced** model
captured for that exact shape (`_setup_inference_path:159`). The hot Rust caller is
`engine/src/game_runner/worker_loop/inner.rs::infer_and_expand:735-757` — it fills one fixed
`feature_len` buffer per cluster view (`encode_state_to_buffer_channels:744`), pushes them, and
calls `submit_batch_and_wait_rust`.

**What a graph breaks.** Each of the above assumes fixed width:
- The `Vec<f32>` request → must become a variable-length **ragged graph payload** (per-graph:
  `x (N,11)`, `edge_index (2,E)`, `edge_attr (E,5)`, `legal_mask (N,)`, `stone_mask (N,)`, with N,E
  per-graph). New batcher contract + serialization.
- `next_inference_batch`'s `[batch, feature_len]` fuse → must ship a **disjoint-union** batch
  (concatenated nodes/edges + per-graph offsets), i.e. the Rust equivalent of `_collate_gnn` in
  `train_bc.py`. The `feature_len` validation, the `flat_features` reshape, and the
  `get_feature_buffer`/`return_feature_buffer` fixed-size pool (`inference_bridge.rs:257-269`) all
  need graph-shaped replacements.
- The pinned `_h2d_staging` (fixed `(batch, C, H, W)`) and the **TorchScript trace** (traced for a
  fixed input shape) are both invalidated by variable graph size. Trace is currently ~33% of
  dispatcher wall-time savings (`_setup_inference_path` note) — losing it, or moving to
  `compile(dynamic=True)`, is a real perf hit that must be re-benched.
- Output contract is FINE: the P74/P75 `(policy, value)` return
  (`submit_inference_results:444-458`) is unaffected — a GNN returns a dense per-legal-node policy
  re-projected to the 362 action space + a scalar value, exactly like the CNN. Only the INPUT
  contract changes.

**Nearest precedent (reduces but does not eliminate risk).** The §173 multi-window K-cluster path
ALREADY handles a variable NUMBER of fixed-width views per leaf: `infer_and_expand` loops
`get_cluster_views()` (K views/leaf, K∈[1,8]) and re-segments the flat results by
`leaf_metadata` (`inner.rs:738-788`). And `v6_live2_ls`'s ragged legal-set action policy is
already "Rust-internal, re-projected per-cluster into dense-362 buffer rows"
(registry.toml `v6_live2_ls` note) — precedent that ragged data can live Rust-side behind a dense
external contract. The graph case generalizes "K fixed-width views" to "1 variable-width graph";
the segmentation bookkeeping is analogous, but the payload width becoming variable is genuinely new.

**Livelock-isolation interaction (named).** The promotion-gate CUDA subprocess isolation
(`hexo_rl/eval/promotion_gate_worker.py`, commits `ac6f0fc` + `a129647`, memory `run2-stall-
watchdog`) spawns the eval in its OWN process with its OWN CUDA context and re-loads the
checkpoint via `load_model_with_encoding`. A graph-input contract interacts in two places: (1) the
subprocess `EvalPipeline` must construct the same graph-inference path (its own `InferenceBatcher`
+ graph builder + traced/compiled GNN) — the isolation guarantee only holds if the child builds
the graph contract independently, so the graph collate/build must be import-safe and not depend on
parent process state; (2) the sidecar `EvalRoundResult` JSON contract is representation-agnostic
(scores, not tensors) so it is unaffected. The isolation fix is DEFAULT-OFF behind
`promotion_gate_subprocess_isolation` — a GNN run flipping it ON inherits the same one-time
graph-path construction cost in the child.

**Files touched.** `engine/src/inference_bridge.rs` (new ragged batcher contract — the big one);
`engine/src/game_runner/worker_loop/inner.rs` (`infer_and_expand` graph-build + ragged submit);
new `engine/src/graph/` (from C1); `hexo_rl/selfplay/inference_server.py` (graph collate +
drop/adapt trace + staging); PyO3 glue in `engine/src/pyo3/`. **Estimate: 8-15 person-days.**
This is where the bench-gate (`make bench`, `docs/rules/perf-targets.md`) is mandatory — the seam
sits on the hottest path and the `bench-gate` skill fires on any `inference_bridge.rs` /
`worker_loop` edit.

---

## C4 — CONFRES seams (encoding registry, resolver, gated loader)

**Registry entry for a graph representation — the fields don't map cleanly.**
`engine/src/encoding/registry.toml` is the canonical source (both Rust and Python parse it;
CLAUDE.md "Encoding registry" rule). Its schema is **grid-shaped**: `board_size`, `trunk_size`,
`n_planes`, `plane_layout` (a list of named planes), `kept_plane_indices`, `n_source_planes`,
`policy_logit_count == board_size²+pass`. A graph encoding has NO planes, NO board_size in the
tensor sense, NO `kept_plane_indices`. The parse-time invariants (`len(plane_layout)==n_planes`,
`len(kept_plane_indices)==n_planes`, `policy_logit_count==bs²+pass`) would all need graph-aware
branches. Two options:
- **A named graph family with a `representation = "grid" | "graph"` discriminant** (schema v4
  bump). Graph rows carry graph-specific keys (`node_feat_dim=11`, `edge_feat_dim=5`, `win_axes`,
  `graph_radius`, `win_length`) and the grid invariants are gated on `representation=="grid"`.
  `policy_logit_count` STAYS 362 (the action space is unchanged — a graph still emits a move on the
  same 19×19+pass board), which is why the deploy path re-projects per-legal-node logits to dense
  362 (the strix bot already does this, `strix_v1_bot.py::get_move`). `state_stride()` /
  `policy_stride()` (consumed by `InferenceBatcher::new` and `inference_server`) need a graph
  meaning: `policy_stride` = 362 unchanged; `state_stride` is undefined for a ragged graph → the
  batcher must be constructed on the graph contract (C3), NOT from `state_stride()`.
- Adding the entry itself is 1 TOML table + `python -m hexo_rl.encoding audit` (the documented
  flow), but the audit's invariant checks + the Rust/Python `EncodingSpec` dataclass both need the
  `representation` discriminant first. **Precedent for a schema bump:** schema v2 (§173 A3, added
  `kept_plane_indices`/`n_source_planes`), schema v3 (cycle 3 P55 Wave 9, added `k_max`).

**Resolver rules.** `hexo_rl/encoding/resolvers.py::detect_encoding_from_state_dict` sniffs
`trunk.input_conv.weight` shape — a GNN has no such key (it has `representation.input_proj.weight`
etc.). The detector needs a graph branch keyed on GNN state-dict signatures. The
`InferenceServer` encoding derivation (`inference_server.py:55-81`, `n_planes`/`board_size`/
`policy_logit_count` from spec) needs the graph contract instead of the plane geometry.

**Gated-loader semantics for GNN checkpoints (the C1-eval-guard class MUST cover graphs).**
`hexo_rl/eval/checkpoint_loader.py::load_model_with_encoding` is the D-EVALGATE G1 gate: it
reconciles a declared encoding against the checkpoint's own stamp
(`metadata['encoding_name']`/`config['encoding']`/`raw['encoding']`) and REFUSES silent
shape/filename override (`_resolve_ckpt_stamped_encoding` / `_check_declared_vs_stamped_encoding`).
Two required extensions:
- `_build_model_from_spec` currently branches only `has_pass_slot` → `_build_min_max_model` /
  `_build_kata_model`, both of which read `state["trunk.input_conv.weight"]`. A GNN needs a THIRD
  builder `_build_gnn_model` selected on `spec.representation=="graph"` (or a GNN state-dict
  signature), constructing the `gnn_net.py` HeXONet. The E1 dist65 detection
  (`bins_w = state.get("value_fc2_bins.weight")`, loader:569-575) must be mirrored on the GNN
  builder so a dist65-on-GNN checkpoint doesn't silently load a scalar/random value head — the E1
  post-load `torch.allclose` guard (loader:593-603) ports directly.
- `validate_arch_against_spec` (loader:254) checks `in_channels==spec.n_planes` and
  `policy_logit_count`; for a graph it checks `node_feat_dim`/`edge_feat_dim` instead of
  `n_planes`. The stamp reconciliation itself is representation-agnostic and ports unchanged.
- **Why this matters (the exact hole to close):** v6_live2 vs v6_live2_ls are shape-identical, so
  shape inference could not disambiguate them — the d1m lineage self-played the wrong encoding for
  272k+ steps (D-FORENSIC F1). A GNN stamp (`gnn_axis_v1`) is even MORE opaque to shape inference
  (a from-scratch GNN and a BC-prefit GNN are shape-identical); the declared-vs-stamp gate is the
  ONLY defense, so the GNN builder must go through it, never around it.

**Files touched.** `engine/src/encoding/registry.toml` (+1 graph entry); `engine/src/encoding/
spec/` + `hexo_rl/encoding/` (schema v4 `representation` discriminant, invariant gating);
`hexo_rl/encoding/resolvers.py` (graph detect branch); `hexo_rl/eval/checkpoint_loader.py`
(`_build_gnn_model` + dist65-on-GNN detection + `validate_arch_against_spec` graph branch).
**Estimate: 3-6 person-days.**

---

## C5 — Instrument compatibility (EVALFAIR, mantis, promotion gate, opening books)

**Two consumer classes, both already partly graph-ready.**

- **BotProtocol path (tournament / arena / EVALFAIR external bars) — ALREADY WORKS.**
  `hexo_rl/bots/strix_v1_bot.py::StrixV1Bot` and `hexo_rl/probes/gnn_bc/gnn_bc_bot.py::GnnBcBot`
  are raw-policy `BotProtocol` adapters that build the axis-graph from `rust_board.get_stones()`,
  forward, argmax legal nodes, filter through OUR legal set. These already run in the D-K
  tournament and `scripts/evalfair/head_vs_strix.py` (the strix-g128 bar in
  `scripts/eval/mantis_pull_eval.py` Stage 3b). A run3 GNN promoted checkpoint drops into a new
  `GnnV1Bot` adapter (lift `GnnBcBot`, swap the net + add MCTS search + dist65 value decode) with
  no harness change — the promotion gate, EVALFAIR book eval, and mantis reader all consume a
  `BotProtocol`/checkpoint by path.

- **ModelPlayer path (in-loop promotion gate `EvalPipeline`) — needs C3.** The promotion gate
  (`hexo_rl/eval/eval_pipeline.py`, run via `promotion_gate_worker.py`) constructs `ModelPlayer`s
  that go through the SAME `InferenceServer`/`InferenceBatcher` tensor path as self-play. So the
  in-loop gate inherits the graph-input contract from C3 — it does NOT need separate graph plumbing
  beyond what C3 delivers, but it cannot run until C3 lands. `mantis_pull_eval.py` is laptop-side
  and reuses `run_retro_ckpt.py`'s `run_arm` (checkpoint-path in, BT/Wilson CI out) — representation
  -agnostic given a working loader (C4).

- **Opening books — VERIFIED representation-agnostic (read the book code).**
  `scripts/evalfair/book.py` and `scripts/evalfair/core.py::build_book` emit
  `{book_id, seed, radius_stage, openings:[{id, moves, rng_seed:null}]}` — books are a list of
  **move sequences** (game-level), sampled at a radius stage, with `ENCODING = "v6_live2_ls"` used
  ONLY as the sampler's board-legality context, not baked into the fixture semantics. A GNN plays
  the identical 19×19 action space (same 362 moves), so the existing `evalfair_r4_v2` /
  `evalfair_r5_v2` fixtures apply to a GNN unchanged; at most a `representation`-tagged book id if
  the sampler's radius geometry needs re-pinning (it does not — the moves are the artifact). **No
  book regeneration required.**

**Files touched.** New `hexo_rl/bots/gnn_v1_bot.py` (MCTS + dist65 deploy adapter, lift `GnnBcBot`
+ `StrixV1Bot`); no change to `eval_pipeline.py`, `mantis_pull_eval.py`, `evalfair/*` beyond C3/C4.
**Estimate: 2-4 person-days.**

---

## C6 — Self-play throughput estimate

**Reference points.** Strix cites **33.8 ms/position** (their production 4.25M GNN, build +
forward). run2 sustained **4.4k steps/hr** on vast (5080) with the production CNN.

**Our measured stack (RTX 4060 Max-Q, §Measurements).** Probe GNN forward bs=64 = 20.86 ms, at
PARITY with the production CNN's 20.07 ms — but the probe GNN is 284k params (15× smaller than the
4.25M CNN) and ran fp32. A production-scale GNN forward will be **~1.5-2× the probe** (edge-scatter
scales with hidden width; strix's 33.8 ms is ~1.7× our CNN's 20 ms, corroborating). ON TOP of
forward, the GNN pays the graph-BUILD cost the CNN's plane-scatter does not (§C1): 5.94 ms/position
Python, or an estimated ~0.5-1.5 ms/position Rust (10-30× the <50 µs plane encoder; bounded above
by strix fitting build+forward in 33.8 ms).

**Projected steps/hr vs run2's 4.4k/h.**

| Path | Per-leaf cost model | Projected steps/hr (5080) |
|---|---|---|
| Rust builder, pipelined, prod-scale GNN | forward ~1.7× CNN + ~1 ms build | **~2.0-3.0k/h** (≈0.5-0.7× run2) |
| Python builder (no Rust port) | 5.94 ms build ≫ 20 ms forward, dominates | **~0.5-1.0k/h** (not viable) |

The Rust-builder path buys **materially fewer steps per GPU-week** than run2 (roughly half). The
Python-builder path is disqualifying. This is the direct cost the R4 decision trades against the
representation advantage.

**Hardware caveat.** Measured on the laptop 4060 Max-Q; vast runs the **5080**, ~1.8-2.2× faster on
dense conv. Absolute ms will drop on the 5080, but the GNN-vs-CNN RATIO is what transfers and it
likely WIDENS slightly: dense GEMM (CNN) speeds up more on the bigger GPU than the
memory-bandwidth/atomics-bound edge-scatter (GNN). Treat the ranges as 5080 projections derived
from the measured 4060 ratio, not from the 4060 absolute times.

---

## C7 — Bootstrap option for run3-GNN

**Two options.**

- **(A) Corpus-BC prefit as init.** The R2 BC-scaling artifact (150-200k BC steps, per the
  amendment's R2 rung) IS a candidate GNN init — a `gnn_bc_*.pt` checkpoint whose
  `RepresentationNetwork` + `PolicyHead` weights are directly loadable by the production
  `gnn_net.py` (state-dict keys match strix exactly, `strix_v1_net.py` docstring). Warm-start via
  the C4 gated loader + an E1-style head-swap loader (`c796887` precedent). Value head is fresh
  (BC probe leaves it unsupervised) → dist65 bins start random, warm through self-play (exactly the
  E1 REVIVE finding: dist65 warm-starts fine from a scalar/absent value head).

- **(B) Fresh trunk + corpus mixing.** Initialize the GNN from scratch, mix corpus BC batches into
  self-play (the `bot_batch_share`/corpus-mix machinery), as run2 did.

**Recommendation: (A) BC prefit init.** This is our structural advantage over strix, which trains
from a from-scratch radius curriculum with no human corpus. The project's mixing philosophy
(§114 Elo-band corpus, run2 corpus-mix) already treats human games as a prior; a BC-prefit GNN
folds that prior into the INIT rather than diluting every self-play batch, and the R2 rung produces
the artifact for free. Caveat to state at R4: the R2 verdict gates this — if R2 shows BC saturates
(flat vs 40k), the prefit's marginal value over fresh is small and option (B) with mixing is the
fallback. Either way the amendment's TRANSFERS (E1 dist65 head, EVALFAIR, promotion-gate isolation)
ride on top unchanged.

**Files touched.** `configs/variants/run3_gnn.yaml` (new, mirror `run3_dist65.yaml` with the graph
encoding + `bootstrap` warm-start pointing at the R2 artifact); a GNN warm-start loader in
`hexo_rl/training/` (lift E1's `c796887`). **Estimate: 1-3 person-days.**

---

## Lead-time table

| Component | Files touched (primary) | Estimate (person-days) |
|---|---|---|
| C1 Graph builder in loop (Rust, recommended) | `engine/src/graph/*` (new), `board/state/encode.rs` sibling, PyO3, parity test vs `strix_v1_graph.py` | **5-10** |
| C2 Net + dist65 pooled head | `hexo_rl/model/gnn_net.py` (new), `training/trainer.py` (forward branch), `binned_value.py` (reused) | **3-6** |
| C3 MCTS/PyO3 graph-input contract | `engine/src/inference_bridge.rs`, `worker_loop/inner.rs`, `pyo3/*`, `selfplay/inference_server.py` | **8-15** |
| C4 CONFRES seams (registry schema v4, resolver, gated loader) | `encoding/registry.toml`, `encoding/spec/*`, `hexo_rl/encoding/*`, `eval/checkpoint_loader.py` | **3-6** |
| C5 Instrument compat (BotProtocol + ModelPlayer + books) | `hexo_rl/bots/gnn_v1_bot.py` (new); books verified no-change | **2-4** |
| C6 Throughput (analysis — done in this doc) | — | **0** |
| C7 Bootstrap (BC-prefit init, recommended) | `configs/variants/run3_gnn.yaml` (new), GNN warm-start loader | **1-3** |
| **Total** | | **22-44 person-days** |

Dependency note: C5 (ModelPlayer path) and the in-loop promotion gate are BLOCKED on C3; C4's
gated loader is required before any GNN checkpoint can be eval'd safely; C1 is a hard prerequisite
for C3's Rust submit path. C2 and C7 can proceed in parallel against the probe net.

---

## The two biggest risks (named)

**RISK 1 — the graph-input inference contract (C3).** The `InferenceBatcher` / PyO3 / pinned-H2D /
TorchScript-trace seam is architected end-to-end around a FIXED `feature_len` dense tensor
(`inference_bridge.rs:295-306`, `inner.rs:735-757`, `inference_server.py:99/159/466`). Variable-size
graphs break the fused `[batch, feature_len]` tensor, the fixed-size feature-buffer pool, the pinned
staging buffer, and the shape-pinned trace — all on the hottest path, all bench-gated. The §173
K-cluster segmentation and the `v6_live2_ls` Rust-internal ragged policy are the nearest precedents
(variable count / ragged-behind-dense), but variable PAYLOAD WIDTH is genuinely new. This is the
highest effort (8-15 pd) AND the highest blast radius: a subtle graph-collate or offset bug silently
corrupts self-play with no loud failure — the exact D-FORENSIC F1 failure class (wrong
representation self-played for 272k+ steps undetected). Mitigation: the byte-parity oracle
(`build_axis_graph_raw` + `_collate_gnn` already tested) and a mandatory `make bench` gate.

**RISK 2 — self-play throughput collapse from graph-build cost (C1/C6).** MEASURED: the graph build
is 5.94 ms/position (Python), on the MCTS hot path per-leaf-per-sim, versus a <50 µs plane scatter
for the CNN. Even the Rust port (est. 0.5-1.5 ms) plus a production-scale GNN forward (~1.7× the
CNN, edge-scatter-bound and NOT FLOP-bound — the measured probe forward is already at CNN parity
with 15× fewer params) projects **~2-3k steps/hr vs run2's 4.4k/h** on the 5080 (~half), and **<1k/h
if the builder stays in Python** (disqualifying). A GPU-week of run3-GNN buys materially fewer
steps than a CNN week — the throughput penalty is the concrete price of the representation
advantage, and it must be an explicit input to the R4 (a) vs (b) opportunity-cost statement.
