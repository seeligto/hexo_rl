# WP-3 IMPL — inert seam steps 1-5 (C3 of the GNN-integration program)

**Agent:** WP-3 IMPL (steps 1-5 ONLY — dormant/test-only). Worktree
`worktree-gnn-integration`. **No git** (dispatcher commits + runs the bench gate;
`inference_bridge.rs` is a bench-gate trigger, so all graph code here is provably
dormant behind `representation=="graph"`). Consumes the seam design
(`gnn_inference_seam_design.md`, steps 1-5), the amended ragged contract
(`gnn_ragged_contract_v1.md`, `policy_dst_slot` i32/−1), the committed producer
(`engine/hexo-graph`, `verify_contract` always-on), the registry design, and the
WP-1 review + red-team seam obligations.

**Verdict: STEPS 1-5 DONE — all inert / default-OFF, CNN path byte-identical.**

---

## Per-step outcome

| Step | What landed | Status |
|---|---|---|
| 1 | `engine` → `hexo-graph` rlib dep (`default-features = true` → native) | DONE |
| 2 | Registry `representation="grid"\|"graph"` discriminant (schema v4) + `gnn_axis_v1` + Rust/Python spec fields + invariant gating + audit | DONE |
| 3 | Dormant `InferenceBatcher` graph queue: `submit_batch_and_wait_graph_rust`, `next_graph_batch`, `submit_graph_inference_results`, `GraphWire`, seam-obligation asserts | DONE |
| 4 | `assemble_ls_from_gnn_probs` (records.rs) + unit tests incl. the off-window-argmax case | DONE |
| 5 | `graph_collate.collate_graph_batch` (18-assertion split) + ADV-1..9 pytest cases | DONE |

### Step 1 — rlib dep
`engine/Cargo.toml` gains `hexo-graph = { path = "hexo-graph", default-features =
true }`. maturin still builds ONE `.so` (verified: one wheel emitted). The dep
edge is one-directional; `cargo check -p hexo-graph --features wasm` stays
independent (`make check.wasm` GREEN). Enabled single-sourcing the graph
schema: `validate.rs` and `records.rs` now reference `hexo_graph::{NODE_FEAT_DIM,
EDGE_FEAT_DIM, WIN_AXES, BUILDER_IMPL_NATIVE, OFF_WINDOW_SLOT}` directly.

### Step 2 — registry
- `Representation { Grid, Graph }` enum on `RegistrySpec`; TOML key
  `representation` OPTIONAL → defaults `Grid` (design §5-below "absent → grid");
  every pre-v4 grid entry is UNTOUCHED (no key, `schema_version` stays 3).
- New graph-only Option fields on `RegistrySpec` + PyO3 getters:
  `node_feat_dim, edge_feat_dim, win_length, graph_radius, win_axes,
  contract_version, builder_impl_required` + `representation`/`is_graph`.
- `gnn_axis_v1` entry: `representation="graph"`, board/trunk 19, 362 logits,
  `has_pass_slot=true`, `node_feat_dim=11`, `edge_feat_dim=5`, `win_length=6`,
  `graph_radius=6`, `win_axes=3`, `contract_version=1`, `builder_impl_required=1`,
  `n_planes=0`, empty plane/kept, `schema_version=4`.
- Validator: grid invariants (`policy_logit_count==bs²+pass`, plane/kept
  relationships) gated on `Grid`; graph invariants gated on `Graph`
  (dims == builder constants, contract/builder handshake, `n_planes==0`, empty
  plane/kept). Grid entries validate byte-identically.
- Python: `EncodingSpec IS engine.RegistrySpec` (thin shim) → new getters
  auto-exposed; added `gnn_axis_v1` to `_REGISTERED_NAMES`. `lookup("gnn_axis_v1")`
  + `all_specs()` (now 12) work.

### Step 3 — dormant batcher graph queue
PARALLEL graph queue on `Inner` (dense `Vec<f32>` path byte-identical — the
2 diff deletions are only the dense prefill loop, re-added verbatim inside
`if !is_graph`). New surface:
- `submit_batch_and_wait_graph_rust(Vec<AxisGraph>) -> Vec<(LegalSetPolicy,f32)>`
  (worker-facing; NOT wired to `worker_loop`).
- `next_graph_batch(bs, wait) -> (ids, GraphWire)` — PyG block-diagonal fuse,
  NO padding; `edge_index`/gather/offsets emitted ALREADY globally-offset i64;
  retains per-id `policy_dst_slot`+legal-coords for the output assemble.
- `submit_graph_inference_results(ids, legal_probs, legal_offsets, values)` —
  slices per-leaf, calls `assemble_ls_from_gnn_probs`, wakes graph waiters.
- `GraphWire` `#[pyclass]` exposing the full contract §2.1 wire.
- Seam obligations (WP-1 red-team) land in `build_graph_from_request`:
  `current_player∈{−1,+1}` + `moves_remaining∈[0,255]` validated BEFORE the
  narrowing i8/u8 cast (Attack-4); stone `|q|,|r|` bounded < i32::MAX−radius
  (Attack-2 silent-wrap guard); `builder_impl==1` handshake; `contract_version==1`
  asserted from the spec. Grid batchers RAISE `RepresentationMismatch` on any
  graph method; `InferenceBatcher::new` is representation-aware (graph spec →
  graph-capable batcher, no dense prefill).

### Step 4 — `assemble_ls_from_gnn_probs`
records.rs pure fn: in-window `policy_dst_slot != OFF_WINDOW_SLOT` → `dense[slot]`;
off-window (−1) → `overflow[coord]`. Pre-normalized (segmented softmax) → NO
renorm; pass slot 0.0; `debug_assert` sum≈1. Plugs straight into the existing
`LegalSetPolicy` / `expand_and_backup_ls` no-drop consumer (reproduces the +414
decode regime). Tests on REAL `build_axis_graph` fixtures: in/off split, the
**off-window argmax survives into overflow and stays the global argmax** (the
seam §1.4 20%-off-window motivating case), all-in-window → empty overflow.

### Step 5 — `graph_collate.collate_graph_batch`
The SINGLE wire reader, import-safe (deferred `torch`), imported by self-play +
eval subprocess later. Handshakes (version, native-builder w/
`HEXO_ALLOW_ORACLE_BUILDER` escape) → 13 structural (always full) → 4
semantic/geometric (full / canary-cadence / off) → block-diagonal torch tensors.
OUTPUT stays ragged (option (b)): the −1 sentinel is CARRIED into the
`GraphBatch`, never dense-scattered or dropped. The 4 semantic checks recompute
geometry from `node_coords`+`n_stones`+`window_center`+`current_player` — a
genuine Rust-builder↔Python-recompute PARITY gate on real wires.

---

## Contract-test spec made real (ADV-1..9)

Driven against a REAL block-diagonal wire from the Rust producer
(`next_graph_batch` on the mixed spread board, B≥2, off-window nodes present),
copied to a mutable payload, mutated per adversarial construction:

| ADV | construction | dies with |
|---|---|---|
| 1a | `node_offsets[B]=N−1` | `OffsetsNonMonotonic` |
| 1b | interior `node_offsets[g]+=1` | `NodeCountChecksum` |
| 2a | gather → graph g+1 | `ScatterGatherCrossesGraph` |
| 2b | two legal nodes share a slot | `ScatterSlotAliasing` |
| 3 | edge dst → graph g+1 | `EdgeCrossesGraphBoundary` |
| 4 | edge_index as u16 | `DtypeMismatch` |
| 7 | window moved, slot-map not | `ScatterSlotCanonicalMismatch` |
| 8 | flip signed_dist | `EdgeAttrGeometryMismatch` |
| 9 | gather at a stone row | `GatherNotLegalNode` |

Plus: version mismatch, native-builder handshake (+escape), `EmptyLegalSet`,
`BatchCountMismatch`, `AugRoundTripMismatch` (trainer target-desync canary), the
real-wire clean-collate parity gate, the off-window-survives gate, and the
full step-3+5 integration (wire → collate → segmented-softmax stand-in →
`submit_graph_inference_results` → assembled `LegalSetPolicy` wakes the blocked
mock worker → `completed_graph_games==n`).

---

## Dormancy proof (CNN path byte-identical)

- **No call site.** `grep` for the graph seam across `engine/src`,
  `hexo_rl/{selfplay,model,training}` finds only: the `GraphWire` pyclass
  registration in `lib.rs`, two COMMENTS (validate.rs, gnn_net.py docstring). No
  invocation. `worker_loop/` has ZERO graph references (untouched).
- **Guarded.** Every graph pymethod calls `require_graph()` → grid batchers
  raise `RepresentationMismatch` (test-verified). Reachable only from a
  `gnn_axis_v1` config, which no launch path selects.
- **Diff.** `inference_bridge.rs` +788/−2; the 2 deletions are the dense prefill
  loop, re-added verbatim inside `if !is_graph`. Dense hot methods
  (`submit_batch_and_wait_rust`, `next_inference_batch`,
  `submit_inference_results`, `pop_batch_blocking`) unchanged.

---

## Gate results

| Gate | Result |
|---|---|
| `cargo test` (engine + hexo-graph) | **301 lib + all integration/doc PASS, 0 fail** (45 new: registry/spec/records-assemble/graph-seam) |
| `pytest -m "not slow and not integration"` | **2505 passed, 0 failed, 134 skipped, 1 xpassed** |
| collection | **2653 total** (baseline 2626, +27), 2639 collected, **zero errors** |
| `python -m hexo_rl.encoding audit` | §1 lists `gnn_axis_v1`; summary `info=64 warn=3 error=1` **identical to baseline** (exit 2 pre-existing §2 checkpoint finding; NOT mine) |
| `make check.wasm` (`cargo check -p hexo-graph --features wasm`) | **GREEN** |
| clippy (touched crates) | **no NEW warnings** (inference_bridge doc/`map_or` fixed; validate/records/pyo3 warnings all pre-existing on unchanged code) |

New Python test count-fixups (consequences of registering a 12th, graph,
encoding): CNN-only parametrized tests filtered to `representation=="grid"`
(`test_network_encoding_dispatch`, `test_encoding_arch_resolver`,
`test_buffer_roundtrip`); graph early-returns
(`test_n_chain_n_source_subset::kept_indices_subset`,
`test_encoding_round_trip`); allowlist adds
(`test_encoding_resolver_paths` corpus/anchor gap, `test_pool_encoding_resolve`
graph bucket, `test_encoding_registry` expected-set, `graph_collate.py` two
`# plane-literal-ok` tokens + two `spec.` audit annotations).

---

## Divergences from design (source-verified reality)

1. **`policy_dst_slot` i32/−1** already applied in the committed `hexo-graph`
   (contract amended twice); no schema improvisation needed — consumed it as-is.
2. **`representation` OPTIONAL (absent→Grid)**, existing entries left at
   `schema_version=3` untouched; only `gnn_axis_v1` is schema-v4. Minimal-diff,
   design-faithful ("absent → grid"), zero churn to the 11 grid entries. Rust
   count tests bumped 11→12.
3. **Seam obligations home:** the coord/current_player/moves_remaining range
   guards live in `build_graph_from_request` (the build-path helper, testable via
   `check_graph_request`), not scattered — the worker's step-6 BuildParams
   construction should route through the same guard.
4. **`AugRoundTripMismatch`** is trainer-path (needs a target); on the inference
   collate ADV-7 fires via its slot-map leg `ScatterSlotCanonicalMismatch`
   (matches the contract table). The full round-trip byte-parity face is C8/WP-5.
5. **Round-trip vs `GnnBcBot` decode** (design §8 test row) needs the trained
   GnnNet (WP-2) — deferred; the step-5 parity gate here is the Rust-builder↔
   Python-geometry recompute on real wires, which is stronger for the wire
   contract itself.

---

## Step-6 preconditions (for the dispatcher's bench-gated commit)

READY: rlib dep, registry `gnn_axis_v1`, graph batcher queue + `GraphWire`,
`assemble_ls_from_gnn_probs`, `collate_graph_batch` + full ADV suite — all inert.
Step 6 (`infer_and_expand_graph` + InferenceServer graph branch) wires the seam
LIVE behind `representation=="graph"`; it MUST run `make bench` (10-metric CNN
no-regress + graph cells) — that is the bench-gate trigger, deferred to the
dispatcher. C8/WP-5 (HEXG buffer / `record_position` graph branch) remains out of
scope and gated on C1 per contract F7.

**STEPS 1-5 SOUND — dormant, gated, byte-identical CNN path.**
