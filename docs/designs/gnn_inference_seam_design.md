# GNN Inference Seam Design — WP-3 (C3) of the GNN-integration program

**Status:** design (not a build order). Consumes the amended ragged contract
(`docs/designs/gnn_ragged_contract_v1.md`, CONTRACT-SOUND @ `2cd8bb7`), the committed
producer (`engine/hexo-graph`, WP-1 PARITY-EXACT), the CUDA bench
(`reports/probes/gnn_integration/WPA_cuda_bench.md`, TORCH-BEATS-ORT + BUILD-HOT), and the
scope doc §C3 (`docs/designs/gnn_integration_scope.md`). Feeds the run4 launch design
(`docs/designs/run4_gnn_design.md` §OQ-7/OQ-8) and the C8/WP-5 training-data path.

**Question this answers:** how a ragged axis-graph payload flows engine → Python → GnnNet →
engine on the self-play hot path, how eval rides the SAME seam (one resolver), and — the
first-order ruling — what happens to the **43.55% of legal cells WP-1 measured OFF-WINDOW**
when the amended contract's output seam is dense `[B,362]`.

**Standing orders bind:** build the payload ONCE per evaluated leaf, never a search-time delta
(§S186); one resolver per knob (`collate_graph_batch` is the single wire reader); eval reads
self-play's seam; no silent fixed-width fallback anywhere (D-FORENSIC F1); benches on the REAL
self-play distribution (WPA `wpa_positions.json`, mean 490/2932). **No custom CUDA kernel**
(§D-STRIX REJECT); **torch-CUDA only** on this seam (ORT = browser/WASM path, WP-D).

---

## 0. Ruling summary (one table)

| # | Decision | Ruling | Falsifier / gate |
|---|---|---|---|
| 1 | **Off-window policy** | **OPTION (b) — ragged per-legal-node return; window concept EXITS the policy path.** Amends contract §2.4: deploy policy output is NOT dense-`[B,362]`-scatter. | §1.4 off-window-chosen-move probe on `wpa_positions.json`: if <2% of deploy-argmax moves are off-window AND drop-vs-no-drop net-vs-net WR within CI → (a) admissible as a v1 simplification. Else (b) mandatory. |
| 2 | **MCTS consumer** | Reuse the EXISTING §D-MULTICLUSTER-S0 `LegalSetPolicy` / `expand_and_backup_ls` no-drop path. GNN assembles a `LegalSetPolicy` (dense-362 in-window + coord-keyed off-window overflow) from per-legal-node probs. | round-trip parity vs `GnnBcBot` decode (the +414 regime) on the frozen set. |
| 3 | **PyO3 surface** | `engine` crate depends on `hexo-graph` as an **rlib** (Rust→Rust build, no PyO3 on the builder). Ragged payload crosses via NEW methods on the existing `#[pyclass] InferenceBatcher`. maturin builds ONE extension module. | `hexo-graph` stays independently wasm-buildable (WP-D `make check.wasm` GREEN). |
| 4 | **Registry** | `representation = "grid" \| "graph"` discriminant (schema v4); new `gnn_axis_v1` entry; every loader/eval/self-play path RAISES `RepresentationMismatch` on kind mismatch (D-EVALGATE). | `python -m hexo_rl.encoding audit` Rust/Python parity + graph invariants. |
| 5 | **Inference lib** | drop TorchScript trace (shape-pinned, invalid for variable N/E) → `torch.compile(dynamic=True)` or eager; MANDATORY re-bench. | `make bench` 10-metric + new graph cells; STEP-0 ≥1.0k steps/hr floor (run4 §4.1). |
| 6 | **Failure** | `builder_impl==1` handshake + `contract_version==1` pin; any assertion failure mid-self-play → `submit_inference_failure` → worker dies loud. NO dense fallback path exists on the graph batcher (structurally). | 9 adversarial payloads (contract §4.2) fire on the collate resolver. |

pd estimate: **10–18 pd** for the inference seam proper (INPUT+OUTPUT ragged path, batcher
graph variant, collate resolver, registry graph discriminant single-sourced with C4, bench),
**excluding** the HEXG buffer/record path (C8/WP-5, co-designed, §7). File-touch list §8.

---

## 1. The off-window ruling (first-order, evidence-bound)

### 1.1 The three options, weighed

WP-1 (`WP1_builder.md` §4.1) measured **43.55% of legal cells (57567/132174) OFF-WINDOW** on the
320-position self-play set — outside the trunk-19 window, where `window_flat_idx_at_geom` returns
`usize::MAX`. The builder emits `OFF_WINDOW_SLOT = u32::MAX` (harness JSON `-1`) and defers the
downstream meaning to this design (the ambiguity WP-1 refused to silently pick).

- **(a) v1 keeps `[B,362]` drop** — smallest surgery; scatter each in-window legal node's logit to
  `dense[policy_dst_slot]`, drop the off-window 43.55%.
- **(b) ragged per-legal logits** — engine consumes the GNN policy over the TRUE legal set;
  window concept exits the policy path.
- **(c) K-window scatter** `[B,K,362]` riding the §173 K-cluster machinery.

### 1.2 Why the evidence regime FORBIDS (a) as the default

Three independent evidence rows say a dense-`[B,362]`-drop seam diverges from the condition under
which the run4 thesis (`+414`) was measured:

1. **The GNN policy head IS per-legal-node.** Read the net, did not assume:
   `gnn_bc_net.py::forward_batch` computes `legal_emb = emb[legal_mask]; policy_head.mlp(legal_emb)`
   → **one logit per legal node**, in legal-node order, over ALL legal cells (in- AND off-window).
   There is NO 362-slot grid in the GNN policy path. The dense-362 is a WP-B transport contrivance;
   the net never produces it.
2. **The +414 decode was no-drop.** `GnnBcBot.get_move` argsorts the per-legal-node logits and picks
   the top LEGAL coord (`strix_legal[idx] ∈ board.legal_moves()`) — no window, no drop
   (`gnn_bc_bot.py:63-72`). R1 (`8eb4fc2`) FIXED the cnn-bc CONTROL to also no-drop (window-0-only
   decode had dropped 34.2% of legal-move mass across 89.7% of decision points); ONLY after that fix
   did Δ settle at +414 [+320,+560]. **A deploy seam that drops off-window moves reproduces the
   pre-R1 handicap on the treatment side — it measures a weaker thing than what graduated.**
3. **D-DECODE names the mechanism.** Off-window defense is ACTION-SPACE — ~60% geometric exclusion
   was the mechanism behind deploy-head off-window blindness; the lever was multi-window **no-drop**
   (memory `d-decode-offwindow-decoding-reframe`). `records.rs:62` (`aggregate_policy`) SKIPS
   off-window today; that skip is the CNN-inheritance the GNN must NOT copy.

### 1.3 Why (b) is CHEAP here — the engine already has the no-drop consumer

Option (b) reads as "large surgery" only if you assume a new ragged MCTS path. It is NOT new. The
§D-MULTICLUSTER-S0 legal-set treatment (`v6_live2_ls`) already ships a **no-drop MCTS consumer** in
`engine/src/game_runner/records.rs`:

- `LegalSetPolicy { dense: Vec<f32> /*362*/, overflow: FxHashMap<(i32,i32), f32> }` — dense window +
  coord-keyed off-window overflow.
- `MCTSTree::expand_and_backup_ls(&[LegalSetPolicy], &values)` assigns priors to legal children
  reading BY COORD (`sample_policy_ls`, `LegalSetPolicy::get`), retaining off-window cells.
- `LocalInferenceEngine.infer_batch_per_cluster` (the eval twin) returns RAW per-cluster policies for
  exactly this legal-set expand, "NO scatter-max, NO drop, NO min-pool" (`inference.py:141-198`).

The GNN plugs straight into this. **The dense-362 remains the transport for the in-window
majority; off-window legal nodes go to `overflow[coord]` instead of being dropped.** The only new
Rust code is `assemble_ls_from_gnn_probs` (§3.4) — populate a `LegalSetPolicy` from per-legal-node
probs + the builder's own `policy_dst_slot` (in-window → `dense[slot]`) and `node_coords[gather]`
(off-window → `overflow[coord]`). `expand_and_backup_ls` consumes it byte-identically to the CNN
legal-set path. **This is why (b) is low-surgery and (a)'s "smallest surgery" advantage is
illusory** — (a) would ALSO need the dense scatter path wired, for a WORSE evidence outcome.

### 1.4 (c) rejected + the falsifier that keeps (a) honest

**(c) rejected:** the GNN is WHOLE-BOARD — one graph per position, no K-cluster windows (C2 note;
WP-B audit node 6). `[B,K,362]` would require inventing K windows around a graph that has none,
riding no existing GNN machinery and needing multiple slots per off-window cell. Mismatch. Drop.

**Divergence quantification for (a) (pre-registered, required before (a) is ever admissible):**
- Upper bound: 43.55% of legal MOVES unrepresentable at the policy output (measured, `WP1_builder.md`).
- Load-bearing number (per §D-COHERENCE — verify the unit that gates the decision): NOT the fraction
  of legal *cells* off-window, but the fraction of **deploy-argmax CHOSEN moves** off-window. This
  is measurable NOW, eval-only, no training: run the banked `gnn_bc_040000.pt` through `GnnBcBot`
  on `wpa_positions.json`, count `policy_dst_slot[argmax] == OFF_WINDOW_SLOT`. The GNN was trained
  per-legal-node with no window bias, so a naive prior expects the chosen-move off-window fraction
  to track the spread of the position — non-negligible on spread boards.
- **Pre-registered rule:** if that measured fraction is `<2%` AND a net-vs-net drop-vs-no-drop WR on
  a matched checkpoint sits within CI, (a) becomes an admissible v1 simplification (cheap
  eval-discriminator before the seam, §D-COHERENCE corollary). **Otherwise (b) is mandatory.**
  Default is **(b)** — it carries zero evidence risk and reuses tested machinery, so it ships unless
  the probe actively clears (a).

**RULING: option (b).** Deploy policy output is ragged per-legal-node, assembled into the existing
`LegalSetPolicy` no-drop consumer. This reproduces the +414 decode regime. Amends contract §2.4:
the dense-`[B,362]`-scatter is NOT the deploy policy output; `policy_dst_slot` is retained on the
wire ONLY as the in-window fast-key for `LegalSetPolicy` assembly (and as the value/aux window
origin), never as a drop gate.

---

## 2. PyO3 surface decision

**Decision: `engine` depends on `hexo-graph` as a Rust rlib; the ragged payload crosses PyO3
through the EXISTING `#[pyclass] InferenceBatcher`. maturin builds ONE extension module.**

### 2.1 Why the builder is Rust→Rust, never PyO3

`hexo_graph::build_axis_graph` is a **once-per-evaluated-leaf hot-path call** from
`worker_loop/inner.rs::infer_and_expand` (§3.2). It must run Rust-side with no GIL, no PyO3
crossing. `hexo-graph`'s core is already dep-free / PyO3-free / `std::thread`-free precisely for
this (and for the wasm target). The crate's dormant `python` feature (`python_glue` skeleton
marker, `lib.rs:715`) is the WRONG axis — exposing the builder as a Python module would force the
worker to cross into Python per leaf, the throughput analog of the 26× trap. **Delete `python`
from `hexo-graph`'s consideration for this seam; use `native`.**

### 2.2 The wiring

- `engine/Cargo.toml`: add `hexo-graph = { path = "hexo-graph", default-features = true }` (pulls
  `native`; the core `build_axis_graph` is not feature-gated). Root `Cargo.toml` already lists
  `engine/hexo-graph` as a workspace member — this adds the dep EDGE engine→hexo-graph that the
  member comment says lands "once WP-B's contract lands". maturin still builds only
  `engine/Cargo.toml`; the rlib dep folds into that single `.so`. No second extension module.
- The workspace `profile.release` (`panic="abort"`, `lto="fat"`) applies to hexo-graph as engine's
  dep — matches the MCTS hot path. (The hexo-graph criterion bench needs `panic=unwind` override,
  `WP1_builder.md` §2; that is bench-only, unaffected.)
- **wasm invariant preserved:** engine depending on hexo-graph does NOT gate hexo-graph's features
  for the standalone `cargo check -p hexo-graph --features wasm` build (WP-D). The dep edge is
  one-directional; hexo-graph never learns about engine/pyo3.

### 2.3 Zero-copy where the existing seam is

The builder returns owned `Vec`s (`node_feat`, `edge_index.{src,dst}`, `edge_attr`, `node_coords`,
…). The batcher's `next_graph_batch` (§4) block-diagonal-concatenates them into union `Vec`s (one
concat copy — unavoidable, exactly like the CNN's `flat_features.extend`), then moves each union Vec
into numpy via `PyArray1::from_vec` (zero-copy move, the SAME pattern as the current
`inference_bridge.rs:394` `flat_features` path). No per-graph PyArray, no intermediate Python
objects. The return crossing (`submit_graph_inference_results`) reads a numpy view read-only
(`policies.readonly().as_slice()`), same as `submit_inference_results:426`.

---

## 3. The Rust seam — `infer_and_expand` graph branch

### 3.1 What breaks (source-verified, contract Part 1 audit)

`infer_and_expand` (`inner.rs:720-838`) is architected around fixed-width dense:
`get_feature_buffer()` hands a `vec![0.0; feature_len]` (SILENT-CORRUPT for a graph, contract node
3); K views encode into it; `submit_batch_and_wait_rust` validates `features.len()==feature_len`
(node 1); the fuse reshapes `[n, feature_len]` (node 2). A graph has no `feature_len`.

### 3.2 The graph branch (dispatch on `spec.representation`)

Add a sibling `infer_and_expand_graph` selected once at the worker boundary on
`spec.representation == "graph"` (NOT per-sim; the branch is hoisted like `legal_set`):

```
for leaf in leaves:
    stones = leaf.get_stones()                     // already native (C1 §"wired to")
    params = BuildParams {                          // ALL static fields from the spec — no literals
        win_length:   spec.win_length,              // registry (§5)
        radius:       spec.graph_radius,            // registry
        trunk_size:   spec.trunk_size,              // registry (=19)
        current_player: leaf.current_player_pm1(),  // per-position
        moves_remaining: leaf.moves_remaining_u8(), // per-position
    }
    // §130 inference-time aug (default-OFF for v1, §3.5): coord pre-rotate `stones` by sym_idx here
    graph = hexo_graph::build_axis_graph(&stones, &params)   // ONCE per leaf (§S186)
    batcher.push_graph(id, graph)                   // NEW: enqueue PendingGraphRequest
```

Then submit the whole batch, block, and get back per-leaf per-legal-node probs (§4). Assemble
`LegalSetPolicy` per leaf (§3.4), min-pool is a no-op (one graph → one value; keep the scalar), and
call the EXISTING `tree.expand_and_backup_ls(&aggregated_ls, &values)`.

**Symmetry note:** the CNN forward-scatters the input planes (`rotate_state_inplace`) then
inverse-scatters the policy (`rotate_policy_inplace`). The GNN realization is coord pre-rotation:
rotate `stones` by `sym_idx` before `build_axis_graph`; the returned per-legal-node probs map back
to ORIGINAL coords via inverse-rotation of each legal node's coord (Rust holds both the original
legal set and the rotated `node_coords`, so the map is exact — the same coord-key path
`LegalSetPolicy` already uses). **v1: inference-time sym DEFAULT-OFF** — the net trains with D6 aug
(run4 §2), so orientation bias at inference is already reduced; keep the knob, ship it off, avoid a
second rotation surface on the deploy path until measured.

### 3.3 The batcher graph variant (`InferenceBatcher`)

Add a graph request kind and two graph methods; the dense path is byte-identical and untouched:

- `enum` the queue payload: `PendingRequest{Dense{Vec<f32>}, Graph{AxisGraph}}` (or a parallel
  graph queue — parallel queue is cleaner, keeps the dense `Vec<f32>` hot path allocation-identical
  and lets a mixed-rep anchor run BOTH, C5). Recommend **parallel graph queue** on `Inner`.
- `push_graph(id, AxisGraph)` — the Rust-internal enqueue from `infer_and_expand_graph`
  (mirrors the internal enqueue inside `submit_batch_and_wait_rust`).
- `next_graph_batch(py, batch_size, max_wait_ms) -> (ids, GraphWirePyArrays)` — §4.
- `submit_graph_inference_results(ids, legal_probs_flat, legal_offsets, values)` — the ragged
  OUTPUT; wakes waiters with per-leaf `(LegalSetPolicy, value)` via the same DashMap+Condvar infra.
  The P74 `Arc<Vec<f32>>` bulk-share still applies to `legal_probs_flat`.

Waiter payload generalizes: `submit_batch_and_wait_graph_rust(graphs) -> Vec<(LegalSetPolicy, f32)>`
(vs the dense `Vec<(Vec<f32>, f32)>`). The blocking/condvar/close/model_version machinery is reused.

### 3.4 `assemble_ls_from_gnn_probs` — the one new records fn

```
// records.rs — GNN counterpart of aggregate_policy_ls. NO K-cluster scatter-max: the GNN emits
// ONE prob per legal node directly. In-window → dense[slot]; off-window → overflow[coord].
fn assemble_ls_from_gnn_probs(
    n_actions: usize,                 // spec.policy_stride() = 362
    legal_probs: &[f32],              // per-legal-node, softmaxed over THIS leaf's legal set (Python)
    policy_dst_slot: &[u32],          // builder output, OFF_WINDOW_SLOT for off-window
    legal_coords: &[(i32,i32)],       // node_coords[legal_node_gather[i]] (builder output)
) -> LegalSetPolicy {
    let mut dense = vec![0.0; n_actions];
    let mut overflow = FxHashMap::default();
    for i in 0..legal_probs.len() {
        if policy_dst_slot[i] == OFF_WINDOW_SLOT { overflow.insert(legal_coords[i], legal_probs[i]); }
        else { dense[policy_dst_slot[i] as usize] = legal_probs[i]; }
    }
    LegalSetPolicy { dense, overflow }   // already normalized (softmax over legal set); no renorm
}
```

Probs are pre-normalized by the per-graph segmented softmax (§4.3), so no renorm — unlike
`aggregate_policy_ls` which renorms a scatter-max. Guard: `debug_assert` that `dense.iter().sum() +
overflow.values().sum() ≈ 1.0` (loud on a segmentation desync). The pass slot stays `0.0` (no pass
in HTTT; `has_pass_slot=true` for `gnn_axis_v1`).

### 3.5 What this does NOT change

`expand_and_backup_ls`, `sample_policy_ls`, `LegalSetPolicy`, the P74/P75 Arc-share value return,
the promotion-gate `EvalRoundResult` JSON (scores, not tensors, C5-verified), the model_version
bump, the dense CNN path — all untouched. The GNN rides the legal-set MCTS interior verbatim.

---

## 4. Hot-path batching — the ragged fuse (`next_graph_batch`)

### 4.1 The block-diagonal union (contract §2.1/§2.2)

`next_graph_batch` pops up to `batch_size` `PendingGraphRequest`s (the `pop_batch_blocking`
threshold `batch_size/2` saturation transfers) and fuses them PyG-block-diagonal, NO padding:

- concat `node_feat` → `[N_total*11] f32`; `edge_attr` → `[E_total*5] f32`; `node_coords` →
  `[2*N_total] i32`.
- `edge_index`: per-graph local `u32` src/dst + `node_offsets[g]` → **`i64` GLOBAL** (contract §2.2
  ruling: bs=256 union ≈125k nodes wraps u16 at bs≈73; i64 is torch's native index dtype, zero
  device-cast on the GINE scatter).
- offsets: `node_offsets[B+1]`, `edge_offsets[B+1]`, `legal_offsets[B+1]` (all `i64`, `[0]=0`,
  `[B]=total`, non-decreasing).
- gather/scatter: `legal_node_gather[Lg]` (GLOBAL row, `i64`), `policy_dst_slot[Lg]` (`u16`,
  per-graph slot, never batched-global).
- semantic-layer wire (contract F1-F3): `n_stones[B] u16`, `window_center[2B] i32`,
  `current_player[B] i8`, `n_nodes_checksum[B] u32`, `builder_impl` scalar `u8`, `contract_version`
  scalar `u32`.

Each union Vec moves into a `PyArray1` (zero-copy, §2.3); returned as a tuple / a small `#[pyclass]`
`GraphWire` holding the arrays. `edge_index` is emitted ALREADY globally-offset so Python does no
index arithmetic.

### 4.2 Backpressure / batch-size interaction

- No fixed-width feature-buffer pool (the flume pool is dense-only; the union Vec is moved into
  numpy, not recycled). Optional arena: pre-size the union Vecs from `k_max_nodes` reserves
  (`WP1_builder.md` capacity-reserve lesson) — a perf lever, not v1-critical.
- Union size is variable: bs=64 ≈ 31k nodes / 188k edges; bs=256 ≈ 125k / 750k (contract §2.2).
  Forward is edge-scatter-bound (WPA), so larger bs amortizes launch overhead but grows the
  E×hidden intermediate. **bs=64 is the WPA sweet spot** (GPU util 97-100% from bs≥32; the ORT
  E×H materialization that OOM'd bs=256 does NOT bite torch — torch ran bs=256 in 213 ms). Keep the
  run's `inference_batch_size: 64` default; the graph batcher inherits it.

### 4.3 The torch path (no ORT — TORCH-BEATS-ORT)

Self-play/eval inference rides **torch-CUDA** (WPA verdict; ORT loses 1.6-2× on CUDA and OOMs the
E×H Expand at bs=256). The InferenceServer graph branch:

1. `collate_graph_batch(wire, expected_version=1)` → `GraphBatch` torch tensors on device (§6).
2. `GnnNet.forward_batch(x, edge_index, edge_attr, legal_mask)` → `[Lg_total]` per-legal-node
   logits (grad-off, autocast fp16). **The TorchScript trace is DROPPED** — it is shape-pinned
   (`_setup_inference_path:159` traces a fixed `[batch,C,H,W]` example), invalid for variable N/E.
   Replace with `torch.compile(dynamic=True)` (already a config arm, `inference_server.py:182-209`,
   mutually-exclusive with trace) or eager. **Mandatory re-bench** (§5): trace was ~33% of
   dispatcher wall-time on the CNN; the GNN forgoes it — WPA's 0.335 ms/pos probe figure was eager,
   so the floor already assumes no trace.
3. **Segmented softmax over each graph's legal nodes** using `legal_offsets` (torch
   `segment_softmax` / scatter-softmax) → `[Lg_total]` probs summing to 1 per graph. This is the
   normalization `assemble_ls_from_gnn_probs` (§3.4) relies on.
4. `batcher.submit_graph_inference_results(ids, legal_probs_flat, legal_offsets, values)`.

Value: the GNN value head is stone-pooled scalar (dist65 decoded, C2) → `[B]`, unchanged return
shape. Merged D2H (probs+value in one `cat().cpu()`) transfers.

---

## 5. Bench plan for IMPL (mandatory — hottest path)

The seam fires the `bench-gate` skill (`inference_bridge.rs` + `worker_loop` + hot-path edits).
`make bench` (10-metric, `docs/rules/perf-targets.md`) is mandatory before the producer/consumer
lands. Because the dense CNN path is byte-identical (parallel graph queue), the 10-metric CNN gate
must show **no regression** (the graph code is dormant behind `representation=="graph"`).

New graph-path bench cells (bench on the REAL WPA distribution, not human-corpus):
- **per-leaf build** (`hexo-graph` criterion, already exists, `WP1_builder.md`: 0.93 ms/pos warm —
  WITHIN the ≤1.5 ms budget; re-run on the 5080 rider before launch, run4 §5).
- **ragged fuse** (`next_graph_batch` block-diagonal concat + offset build) at bs=64/256.
- **collate + forward + segmented-softmax + submit** end-to-end per-leaf ms, torch-CUDA fp16.
- **assemble_ls_from_gnn_probs** (cheap; assert it's `<<` forward).

Expected numbers (WPA/WP-1, 4060 ratios; 5080 rider re-sets absolutes): builder 0.93 ms + probe
forward 0.335 ms ≈ **1.27 ms/leaf** (5.0× the CNN's ~0.25 ms) → per WP-A's own recompute 0.93×256
≈ 0.24 s/step build alone. **Tie-in: STEP-0 ≥1.0k steps/hr floor** (run4 §4.1) measured on the 5080
before launch; below floor → NO LAUNCH → BUILD-HOT perf sub-package first (optimize the
once-per-leaf build, §S186 — never a search-time delta). The bench gate additionally asserts
`builder_impl==1` on the benched path (contract §4.3): a Python-builder bench is both an invalid
measurement and the exact F7 trap.

---

## 6. The single resolver — `graph_collate.collate_graph_batch`

**One resolver, named** (contract §2.3): `hexo_rl/selfplay/graph_collate.py::collate_graph_batch(
wire, expected_version) -> GraphBatch`. THE single reader of the wire contract. Lives in a module
imported by BOTH:
- `inference_server.py` (self-play hot path), and
- the promotion-gate CUDA subprocess `eval_pipeline.py` / `promotion_gate_worker.py` (eval reads
  self-play's seam — memory `run2-stall-watchdog`; the child builds its OWN inference path in its
  OWN process, so `graph_collate` must be import-safe, no parent-process state).
- (later) the trainer sample path (C8/WP-5).

It: (1) asserts `contract_version==1` (`GraphContractVersionMismatch`); (2) asserts
`builder_impl==1` on any training/self-play path unless `HEXO_ALLOW_ORACLE_BUILDER=1`
(`NonNativeSampleBuilder`, the F7 handshake); (3) runs the assertion set; (4) builds torch-CUDA
tensors (`edge_index` already global — no offset arithmetic Python-side). **No second reader.**

### 6.1 The 18-assertion split (full on trainer, single-source+canary on hot path)

Per contract §2.5, two layers — 13 structural + 4 semantic/geometric + the F7 handshake:
- **Structural (13)** — index in-range/unique/monotonic/typed — run **FULL everywhere, always**
  (vectorized numpy, cheap): `NodeFeatDimMismatch`, `EdgeAttrDimMismatch`, `DtypeMismatch` (i64
  edge_index — the u16-wrap ADV-4 defense), `BatchCountMismatch`, `OffsetsNonMonotonic`,
  `NodeCountChecksum`, `EdgeIndexOutOfBounds`, `EdgeCrossesGraphBoundary`,
  `ScatterGatherCrossesGraph`, `ScatterSlotOutOfBounds`, `ScatterSlotAliasing`, `EmptyLegalSet`.
- **Semantic/geometric (4)** — `EdgeAttrGeometryMismatch`, `GatherNotLegalNode`,
  `ScatterSlotCanonicalMismatch`, `AugRoundTripMismatch` — recompute geometry from `node_coords` +
  `n_stones` + `window_center` + `current_player`. **Trainer path: FULL every batch** (~10-20 ms
  vectorized, small vs the ~138 ms/step rebuild). **Self-play hot path: FIRST batch after every
  process-start / weight-swap + every Nth (canary cadence, config knob, loud log line) + ALWAYS in
  tests/debug builds.** Rationale: the hot-path producer is the same audited single-pass Rust walk
  emitting `edge_attr`, `node_coords`, `policy_dst_slot` together (`hexo-graph::build_axis_graph`,
  single-source — internal desync requires the same bug twice), so the primary F1 defense is
  structural-by-construction; the semantic canary catches a regression, not per-batch drift.

### 6.2 Amendment to the output plan (option-b consequence)

`collate_graph_batch` produces the INPUT `GraphBatch` (incl. `legal_offsets`). The OUTPUT is NOT a
dense-362 scatter (contract §2.4 superseded by §1's ruling): the InferenceServer segmented-softmaxes
per-graph legal nodes (§4.3) and returns flat `[Lg_total]` probs. `policy_dst_slot` travels on the
wire but is consumed Rust-side in `assemble_ls_from_gnn_probs` (§3.4), NOT as a Python dense scatter.
`legal_node_gather` + `node_coords` give the off-window coord for the overflow key. The `EmptyLegalSet`
assertion guarantees every graph produces a policy row.

---

## 7. Failure semantics + build-order

- **builder_impl handshake:** `collate_graph_batch` asserts `builder_impl==1`; the ONLY builder that
  exists is native `hexo-graph` (`BUILDER_IMPL_NATIVE=1`). The Python oracle (`build_axis_graph_raw`,
  14.0 ms/pos = 3.58 s/step 26× trap) is reachable ONLY under `HEXO_ALLOW_ORACLE_BUILDER=1` (parity
  tests/CI). Hard refuse-to-train/serve otherwise (`NonNativeSampleBuilder`).
- **Contract version pinning:** the Rust producer stamps `contract_version=1`; a stale producer/reader
  mismatch dies at assertion 1.
- **Assertion failure mid-self-play → die LOUD:** any resolver assertion raises → the InferenceServer
  `except` calls `submit_inference_failure(ids, error_msg)` (unchanged, `inference_server.py:552`) →
  Rust waiters return `Err` → `infer_and_expand_graph` returns 0 → the worker surfaces the failure.
  **There is NO dense fallback path on the graph batcher** (the graph queue has no `Vec<f32>`
  interpretation), so the D-FORENSIC F1 silent-fixed-width-fallback class is structurally
  impossible — a graph can never be misread as planes (contract audit node 1's len-collision is
  closed because the graph never enters the dense `submit_and_wait`).
- **Build-order (WP-5 gate):** the Rust producer path is LIVE only after C1 (`hexo-graph`, DONE) +
  this seam land. Option-(c) training is gated on C1 (contract F7); there is no legitimate interim
  Python-builder self-play config, and the handshake enforces it.

**IMPL sequencing (what lands behind a default-OFF flag):**
1. `engine` deps on `hexo-graph` (rlib) — inert, no call sites. Land first (zero behavior change).
2. Registry `gnn_axis_v1` + `representation` discriminant (§5-below) + Rust/Python spec fields +
   audit — inert until an encoding is selected. Single-sourced with C4.
3. `InferenceBatcher` graph queue + `next_graph_batch` + `submit_graph_inference_results` + waiter
   generalization — dormant (no producer enqueues yet). Dense path byte-identical → CNN bench green.
4. `assemble_ls_from_gnn_probs` in records.rs — pure fn, unit-tested, uncalled.
5. `graph_collate.collate_graph_batch` + 18 assertions + 9 adversarial tests (contract §4.2).
6. `infer_and_expand_graph` + InferenceServer graph branch — the seam goes LIVE behind
   `representation=="graph"` (a `gnn_axis_v1` variant config). `make bench` gate + graph cells.
7. The step-0 self-play smoke (run4 OQ-7): write nothing to the buffer yet (C8/WP-5), just verify a
   leaf builds → fuses → collates → forwards → assembles LS → expands, round-tripping vs `GnnBcBot`.

Steps 1-5 are inert / test-only and can land ahead of the box. Step 6 is the bench-gated hot-path
commit. The C8/WP-5 HEXG buffer + `record_position` graph branch (store ragged coord-keyed visit
distribution, NO `aggregate_policy_to_local` dense scatter, NO `reproject_game_end_row` — the GNN
value is scalar dist65, no ownership/winning-line aux head) is co-designed with this seam but
OUT of the C3 scope; it rides the same `LegalSetPolicy`/coord-keyed no-drop discipline.

---

## 5-below. Registry — `representation` discriminant + `gnn_axis_v1`

**Discriminant:** add `representation: "grid" | "graph"` (schema v4; the "kind" discriminant). All
existing entries default to `"grid"` (absent → `"grid"` for back-compat). Grid invariants
(`len(plane_layout)==n_planes`, `policy_logit_count==bs²+pass`, `len(kept_plane_indices)==n_planes`,
`trunk_size==cluster_window_size|board_size`) are **gated on `representation=="grid"`**.

**Graph invariants (representation=="graph"):** `node_feat_dim==11`, `edge_feat_dim==5`,
`win_length` present, `graph_radius` present, `win_axes==3`, `contract_version` present,
`policy_logit_count==362` (action space UNCHANGED — a graph plays the identical 19×19+pass board,
C4), `has_pass_slot==true`, `plane_layout` empty / `n_planes==0` (or "none"), multi-window keys
`"none"`.

```toml
[encodings.gnn_axis_v1]
representation      = "graph"
board_size          = 19          # action-space board (19×19+pass)
trunk_size          = 19          # window_flat_idx origin for policy_dst_slot
policy_logit_count  = 362         # UNCHANGED action space
has_pass_slot       = true
node_feat_dim       = 11          # relative-7 + threat-4 (contract §2.1)
edge_feat_dim       = 5           # axis one-hot(3) + signed_dist + src_player
win_length          = 6
graph_radius        = 6
win_axes            = 3
contract_version    = 1           # ragged-payload contract this encoding speaks
builder_impl_required = 1         # native tag the resolver asserts
sym_table_id        = "size_19"   # D6 aug reuses axis_perm (run4 §2 / contract Part 3)
n_planes            = 0
plane_layout        = []
cluster_window_size = "none"
cluster_threshold   = "none"
is_multi_window     = false
value_pool          = "none"
policy_pool         = "none"
legal_move_radius   = 6
schema_version      = 4
notes               = "GNN axis-graph (legacy relative+threat schema). Ragged payload contract v1. Whole-board (no K-cluster); per-legal-node policy, no off-window drop."
```

**No scattered constants** (CLAUDE.md registry rule): the builder's `BuildParams` is populated from
the spec — `win_length=spec.win_length`, `radius=spec.graph_radius`, `trunk_size=spec.trunk_size`
(static) + `current_player`/`moves_remaining` from the board (dynamic). The Rust `RegistrySpec` and
Python `EncodingSpec` dataclass both gain `representation`, `node_feat_dim`, `edge_feat_dim`,
`win_length`, `graph_radius`, `contract_version` (schema v4). `hexo_rl.encoding.lookup("gnn_axis_v1")`
exposes them; the worker reads `spec.win_length` etc. — no `6`/`19`/`11` literal anywhere.

**Encoding gate (D-EVALGATE precedent):** every construction/loader site RAISES on kind mismatch:
- `InferenceBatcher::new` for a graph spec must NOT derive `feature_len` from `state_stride()` (a
  graph has none, contract §"Load-bearing audit finding") — it constructs the graph queue; a grid
  ctor handed a graph spec, or vice-versa, raises `RepresentationMismatch`.
- `build_net(spec, state)` (the one construction authority replacing the SILENT-CORRUPT
  orchestrator:677 / lifecycle:66,172 / anchor:569 sites, contract nodes 11a-c; CONFRES
  `build_player` precedent `69442e5`) dispatches on `spec.representation`; a `graph` config building
  a CNN is the F1 hole this closes.
- `resolvers.py::detect_encoding_from_state_dict` gains a graph-detect branch (GNN sd has
  `representation.input_proj.weight`, no `trunk.input_conv.weight`) SINGLE-SOURCED across the C4 eval
  loader and C7 trainer loader (contract nodes 11d-e; the loader-family already LOUD-FAILs on a GNN
  sd — this makes it RAGGED-OK). `python -m hexo_rl.encoding audit` verifies Rust/Python parity + the
  graph invariants (`hexo_rl.encoding` audit flow).

---

## 8. Person-days + file-touch list

These are the C3 INFERENCE-seam slice (scope doc C3 = 8-15 pd; this refines to 10-18 with the
torch-trace-drop + ragged-OUTPUT option-b as the high-end drivers). NOT additive on the contract's
already-scoped Rust-producer/resolver rows — single-sourced. Excludes C8/WP-5 buffer (§7).

| Component | Files (primary) | pd |
|---|---|---|
| engine→hexo-graph rlib dep; `infer_and_expand_graph` (BuildParams-from-spec, per-leaf build, coord-prerotate aug knob) | `engine/Cargo.toml`, `game_runner/worker_loop/inner.rs` | 2-4 |
| InferenceBatcher graph queue + `next_graph_batch` ragged fuse + `submit_graph_inference_results` + waiter generalize + PyO3 tuple/`GraphWire` | `engine/src/inference_bridge.rs`, `engine/src/pyo3/*` | 3-5 |
| `assemble_ls_from_gnn_probs` (records) + unit tests | `engine/src/game_runner/records.rs` | 1 |
| `graph_collate.collate_graph_batch` (18 assertions, torch tensorize) — single resolver | `hexo_rl/selfplay/graph_collate.py` (new) | 3-5 |
| InferenceServer graph branch (drop trace → compile(dynamic), segmented softmax, submit_graph) + eval subprocess import parity | `hexo_rl/selfplay/inference_server.py`, `hexo_rl/eval/eval_pipeline.py` | 2-4 |
| Registry `gnn_axis_v1` + `representation` discriminant + Rust/Python spec fields + invariant gating + audit (SINGLE-SOURCED with C4) | `engine/src/encoding/registry.toml`, `engine/src/encoding/spec/*`, `hexo_rl/encoding/*` | 2-4 |
| Bench: `make bench` no-regress + graph cells (fuse, collate+forward+submit) | `docs/rules/perf-targets.md` cells, criterion | 1-2 |
| Seam round-trip test vs `GnnBcBot` decode + off-window-chosen-move probe (§1.4 falsifier) | `tests/`, `engine/tests/` | 1-2 |
| **Total (C3 inference seam, option-b)** | | **~10-18 pd** |

**File-touch list (durable):**
`engine/Cargo.toml`, `engine/src/inference_bridge.rs`, `engine/src/game_runner/worker_loop/inner.rs`,
`engine/src/game_runner/records.rs`, `engine/src/encoding/registry.toml`, `engine/src/encoding/spec/*`,
`engine/src/pyo3/*`, `hexo_rl/selfplay/graph_collate.py` (new), `hexo_rl/selfplay/inference_server.py`,
`hexo_rl/eval/eval_pipeline.py`, `hexo_rl/encoding/*` (spec fields + resolver graph branch),
`build_net` construction authority (contract nodes 11a-c).
**Co-designed, OUT of scope (C8/WP-5):** `record_position` graph branch, HEXG buffer, corpus re-export.

Dependencies: builder is DONE (C1, `hexo-graph` PARITY-EXACT). The resolver + `assemble_ls` can be
built + tested against the `_collate_gnn` / `GnnBcBot` oracle in parallel with the batcher. The
registry graph discriminant is C4/C7 cross-cutting, single-sourced here. C8/WP-5 buffer rides the
same coord-keyed no-drop discipline this seam establishes.

---

## Verdict

- **Off-window: OPTION (b)** — ragged per-legal-node, assembled into the EXISTING `LegalSetPolicy` /
  `expand_and_backup_ls` no-drop consumer; reproduces the +414 decode regime; amends contract §2.4.
  Falsifier: the off-window-chosen-move probe (§1.4) — default (b), (a) admissible only if <2% + WR
  within CI. Divergence for (a) quantified: up to 43.55% of legal moves unrepresentable; the
  decision-gating unit is the chosen-move off-window fraction, measurable eval-only NOW.
- **PyO3:** engine deps on `hexo-graph` rlib (builder Rust→Rust, no PyO3); ragged payload crosses the
  existing `InferenceBatcher` pyclass; maturin builds one `.so`; wasm invariant preserved.
- **Registry:** `representation` discriminant (schema v4) + `gnn_axis_v1`; loaders RAISE on kind
  mismatch; BuildParams from spec, zero scattered constants.
- **One resolver:** `collate_graph_batch`, imported by self-play + eval subprocess (+ trainer later);
  18-assertion split (structural-always, semantic-full-on-trainer / canary-on-hot-path).
- **torch-CUDA** hot path (no ORT); TorchScript trace DROPPED → `compile(dynamic=True)`; mandatory
  `make bench` + graph cells; ≥1.0k steps/hr floor tie-in.
- **No silent fallback:** graph batcher has no dense interpretation; assertion failure → loud worker
  death; builder_impl + contract_version handshakes.
- **pd: 10-18** (C3 inference seam, option-b); C8/WP-5 buffer co-designed, excluded.

**SEAM-SOUND — option (b) ruling evidence-bound to the +414 no-drop decode regime.**

---

## §1.4 falsifier — EXECUTED (2026-07-14, dispatcher)

`scripts/research/offwindow_chosen_move_probe.py`, banked `gnn_bc_040000.pt`, 320 real
self-play positions (`wpa_positions.json`), Rust-harness `policy_dst_slot`:

- off-window **CHOSEN-move** fraction (deploy argmax): **64/320 = 20.0%**
- mean off-window policy **mass**: **17.5%**
- off-window CELL fraction sanity: 0.4355 (== WP-1's 43.55%, unit cross-check per §D-COHERENCE)

The option-(a) admissibility bar was <2% chosen-move fraction; measured 20% exceeds it 10×.
**Option (b) is MANDATORY, not preferred** — a dense-362 deploy seam drops the model's actual
chosen move in 1 of 5 positions (the pre-R1 handicap class). Ruling upgraded from
evidence-bound to measurement-confirmed.
