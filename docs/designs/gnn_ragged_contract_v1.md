# GNN Ragged-Payload Contract v1 — WP-B (COND-1) of the GNN-integration program

**Status:** versioned contract design (not a build order). Consumes the R3 scoping doc
(`docs/designs/gnn_integration_scope.md`, esp. §C3 / §C8 / §Red-team pass) and the WP-A CUDA bench
(`reports/probes/gnn_integration/WPA_cuda_bench.md`, verdict CUDA-WINS + BUILD-HOT +
TORCH-BEATS-ORT). Feeds WP-C (D6-in-graph-space decision rule) and WP-D (wasm op-set).

**Question this answers:** what is the exact, versioned, flat-array PyO3 contract for a ragged
`(nodes, edges)` self-play payload, such that EVERY consumer on both sides of the loop (inference
seam C3 + training-data path C8) either handles ragged natively or asserts-and-dies — no silent
fixed-width fallback anywhere. This is the F1-class silent-corruption surface of the program
(D-FORENSIC F1: the d1m lineage self-played the wrong representation for 272k+ steps because a
string-form encoding key silently fell back — the exact failure class a subtle graph-collate /
offset / symmetry bug reproduces).

**Scope discipline:** every break point below was source-verified by reading the surrounding code
(not grep). "grep is not proof of coverage" — each dispatch site names what it DOES when handed a
graph encoding. Measured graph distribution is WP-A's (self-play, not human corpus): mean 490
nodes / 2932 edges, p90 729 / 4796, max 897 / 5690 per graph; batched union at bs=256 reaches
~74k nodes / ~750k edges.

---

## Part 1 — Full data-path audit, node by node

Every node classified: **SILENT-CORRUPT** (accepts a graph, misinterprets, no error),
**SILENT-FALLBACK** (silently substitutes fixed-width / wrong path), **LOUD-FAIL** (asserts/raises
today), **RAGGED-OK** (already representation-agnostic). "Design's class" is what the v1 contract
changes it to. **The design target is: every node is RAGGED-OK or LOUD-FAIL — zero
SILENT-CORRUPT, zero SILENT-FALLBACK.**

### Audit table (compact)

| # | Node (file:line) | Today's class | Why (source-verified) | Design's class |
|---|---|---|---|---|
| 1 | `inference_bridge.rs:62-74,192-195` — `PendingRequest{features:Vec<f32>}` fixed-len check | LOUD-FAIL / **SILENT-CORRUPT-on-len-collision** | `features.len() != feature_len → Err`; a graph flattened to exactly `feature_len` PASSES and is misread as planes | RAGGED-OK — new `GraphRequest` variant, versioned ragged payload |
| 2 | `inference_bridge.rs:387-396` — single `PyArray2<f32>` fuse `[n, feature_len]` reshape | LOUD-FAIL | `PyArray1::from_vec(flat).reshape([n, feature_len])` raises on a ragged total that isn't `n*feature_len` | RAGGED-OK — new `next_graph_batch` returns the multi-array ragged tuple |
| 3 | `inference_bridge.rs:257-269` — `get/return_feature_buffer` fixed-`feature_len` pool | **SILENT-CORRUPT** | pool hands back a `vec![0.0;feature_len]`; a graph encode into it truncates/misaligns with NO error (§C3 named break point) | LOUD-FAIL on the dense path + separate ragged arena (see Part 2) |
| 4 | `inference_server.py:99,159,466` — pinned `_h2d_staging (batch,C,H,W)` + TorchScript trace | LOUD-FAIL | flat→`(n,C,H,W)` reshape fails; shape-pinned trace rejects variable input | RAGGED-OK — torch-CUDA block-diagonal collate (WP-A TORCH-BEATS-ORT), trace→`compile(dynamic=True)` + re-bench |
| 5 | `inference_bridge.rs:400-459` — fixed-`policy_len` OUTPUT scatter (`start=i*policy_len`) | LOUD-FAIL (output width) / **no input-side node→slot map exists** | output `[batch,362]` seam is fine; the missing piece is per-node-logit → move-slot mapping | RAGGED-OK — `policy_scatter_index` builds dense `[B,362]` Python-side; **OUTPUT seam UNCHANGED** |
| 6 | `worker_loop/inner.rs:1399-1400` — `record_position` dense `feat=vec![0;kept_planes·n_cells]` per cluster | **SILENT-CORRUPT** | records K dense per-cluster plane rows; a whole-board graph has no cluster/plane meaning — unchanged path records garbage | RAGGED-OK — records ONE compact graph-position row (stones + ragged policy + aux); no K-cluster loop (§C2 note: GNN is whole-board) |
| 7 | `replay_buffer/mod.rs:103` + `storage.rs:54-102` — `states:Vec<u16>` flat `[cap·state_stride]`, resize by stride rotation | **SILENT-CORRUPT** | `rotate_left(head*state_stride)` + `resize(cap*state_stride)` assume fixed width; ragged data mis-slices silently | RAGGED-OK — separate HEXG graph-position ring (Part 2 option c); dense ring untouched |
| 8 | `replay_buffer/sample.rs:324-328` — `states[idx*state_stride..(idx+1)*state_stride]` scatter | **SILENT-CORRUPT** | pure stride arithmetic; no ragged path; a graph record read at a stride boundary returns misaligned bytes | RAGGED-OK — graph sampler slices by offsets, rebuilds graph + aligns policy target |
| 9 | `sym_tables.rs` + `sample.rs:222-256` — 12-fold `apply_sym` / `apply_symmetry_state` / `apply_chain_symmetry` | **SILENT-CORRUPT** | dense coordinate scatter over cells; on a graph it scatters node-feature bytes as if they were cells | RAGGED-OK — Part 3: option-(c) coord-pre-rotation makes aug free; graph-space D6 also FEASIBLE |
| 10 | `batch_assembly.py:183-257` — `.npz` corpus `(T,8,bs,bs)` → `push_game` | LOUD-FAIL (guarded) | `pre_states.shape[1] != _spec.n_planes → raise` (:195) — the dense→graph mismatch already dies loud | RAGGED-OK — graph corpus re-export (probe `_compact_example` precedent) or replay-stones-and-rebuild |
| 11a | `orchestrator.py:677` — fresh-run `HexTacToeNet(...)` | **SILENT-CORRUPT** | builds a CNN UNCONDITIONALLY; a `representation=graph` config builds a CNN and self-plays it — no error | LOUD-FAIL/RAGGED-OK — single `build_net(spec,state)` dispatch on `spec.representation` |
| 11b | `lifecycle.py:66` (inf_model) + `:172` (eval_model) — `HexTacToeNet(...)` then `load_state_dict` | SILENT-CORRUPT-then-LOUD | builds CNN, then `load_state_dict(graph_sd)` strict=True → key-mismatch raise; wrong net built first | LOUD-FAIL/RAGGED-OK — same `build_net` seam |
| 11c | `anchor.py:569` — `best_model = HexTacToeNet(...)` (in-loop anchor/eval) | **SILENT-CORRUPT** | builds CNN for the anchor; mixed-rep anchor (CNN bootstrap + GNN candidate, §C5) builds wrong net | LOUD-FAIL/RAGGED-OK — `build_net` seam + mixed-rep anchor path |
| 11d | `trainer_ckpt_load.py:598` (resume) → `resolvers.py:455-462` | LOUD-FAIL | `detect_encoding_from_state_dict` RAISES "no trunk.input_conv(.conv)?.weight; cannot detect encoding" on any GNN sd (red-team finding 3) | RAGGED-OK — graph-detect branch in `resolvers.py`, SHARED with C4 eval loader |
| 11e | `checkpoint_loader.py:576` `_build_min_max` + `:634` `_build_kata` | LOUD-FAIL | both read `state["trunk.input_conv(.conv)?.weight"]` → `None`/KeyError on a GNN sd | RAGGED-OK — `_build_gnn_model` selected on `spec.representation=="graph"` (C4), with the E1 `torch.allclose` landed-verify (`:593-603`) mirrored onto representation+policy+dist65 tensors |

**Coverage note (per-site read, not grep).** The scoping doc / red-team named "5 hardcoded
`HexTacToeNet` dispatch sites" (orchestrator:677, lifecycle:66+172, anchor:569,
trainer_ckpt_load:598). Reading the tree surfaced **two more production builders** the eval loader
uses (`checkpoint_loader.py:576` + `:634`) and two pretrain-only sites
(`pretrain_cli.py:415`, `pretrain_validate.py:65`) plus the viewer (`model_loader.py:56`). The
load-bearing split is **behavior, not count**: the loader-family sites (11d, 11e) read
`trunk.input_conv.weight` and therefore **die loud** on a GNN state dict — safe today. The
**construction-family sites (11a, 11b, 11c) build a CNN unconditionally and are the SILENT-CORRUPT
hole**: a `representation=graph` config at orchestrator:677 builds a CNN and self-plays it with no
error. That is the exact F1 class. The fix is one `build_net(spec, state)` authority (CONFRES
`build_player` single-construction-authority precedent, commit `69442e5`) replacing all
construction-family sites; the loader-family sites get a shared `resolvers.py` graph-detect branch
(single-sourced across C4 eval + C7 trainer, per red-team finding 4).

**Load-bearing audit finding.** The whole INPUT contract is architected end-to-end around a scalar
`feature_len` (`inference_bridge.rs:295-306` derives it from `spec.state_stride()`) and every
training-data node re-derives fixed strides from `spec.state_stride()` /
`spec.policy_stride()` / `spec.aux_stride()` (`mod.rs:186-196`, `storage.rs:54-57`,
`sample.rs:296-299`, `persist/load.rs:172`). A graph has NO `state_stride`. Nodes 3, 6, 7, 8, 9,
11a-c are SILENT-CORRUPT today — they accept a graph-shaped input (or a graph config) and mis-slice
or build the wrong architecture with no loud failure. **The contract's job is to convert every one
of those to RAGGED-OK or LOUD-FAIL.**

---

## Part 2 — Contract v1 spec

Versioned flat-array PyO3 contract, **block-diagonal batching per PyG convention, NO padding**.
The union of B graphs is concatenated node/edge arrays plus offset pointers — exactly the Rust
equivalent of the probe's `_collate_gnn` (`hexo_rl/probes/gnn_bc/train_bc.py`), which is the tested
oracle for the collate semantics.

### 2.1 Payload arrays (the wire contract)

All arrays are **flat 1-D** (or declared-shape numpy) crossing PyO3. `B` = batch size (graphs),
`N` = total nodes, `E` = total edges (directed; the builder emits both i→j and j→i,
`strix_v1_graph.py:240-255`), `Lg` = total legal nodes across the batch.

| Array | Shape (flat) | dtype | Meaning |
|---|---|---|---|
| `contract_version` | scalar | `u32` | = `1`. First field checked. |
| `node_feat` | `[N * 11]` | `f32` | per-node features, F=11 fixed (relative-7 + threat-4, `strix_v1_graph.py:108-124`) |
| `edge_index` | `[2 * E]` | **`i64`** | **globally-offset** src/dst node ids (block-diagonal; graph g's local ids + `node_offsets[g]`) |
| `edge_attr` | `[E * 5]` | `f32` | `[axis_onehot(3), signed_dist, src_player]` (`strix_v1_graph.py:243-255`) |
| `node_offsets` | `[B + 1]` | `i64` | node ptr; `[0]=0`, `[B]=N`, non-decreasing (PyG `ptr`) |
| `edge_offsets` | `[B + 1]` | `i64` | edge ptr; `[0]=0`, `[B]=E`, non-decreasing |
| `legal_offsets` | `[B + 1]` | `i64` | legal-node ptr; segments `policy_scatter_index` per graph |
| `legal_node_gather` | `[Lg]` | `i64` | **global** node-row index of each legal node (into `node_feat`/embeddings) |
| `policy_dst_slot` | `[Lg]` | `u16` | destination action slot (0..361) for each legal node |
| `n_nodes_checksum` | `[B]` | `u32` | per-graph declared node count (stones+legal+1 dummy); off-by-one tripwire (ADV-1) |

Value/output arrays are **unchanged from the CNN contract** (see 2.4).

### 2.2 The u32-vs-u16 ruling (explicit, per WP-A distribution)

**Per-graph node ids fit u16** (max measured 897 nodes << 65535). **Batched flat indices do NOT:**
the block-diagonal union at bs=256 reaches ~74k nodes (WP-A), so any GLOBAL node id
(`edge_index`, `legal_node_gather`) exceeds u16 by >1 order of magnitude. **Ruling: batched flat
node indices are `i64`; per-graph u16 is a false economy** — it cannot survive block-diagonal
batching and would silently wrap at node 65536 (SILENT-CORRUPT, the worst class).

- `edge_index`, `legal_node_gather`, all three `*_offsets` → **`i64`**. Rationale: (a) u16 wraps at
  bs≈45 (65535/1450 nodes) — disqualified; (b) `u32` is numerically safe (74k < 2^32) but forces a
  device-side cast to `torch.long` for every `scatter`/`index_select` in the GINE message step,
  adding an op + a 2× intermediate on the hot path; (c) `i64` is torch's native index dtype →
  zero-cast, and the +6 MB/batch H2D (edge_index i64 vs u32 at ~750k edges) is immaterial against
  the edge-scatter-bound forward (WP-A: forward is memory/atomics-bound, not H2D-bound).
  **Downgrade lever (named, not taken for v1):** if a future PCIe-bound regime appears, `edge_index`
  MAY ship `u32`-on-wire with a SINGLE documented cast at the collate resolver (2.3) — never per-op,
  never per-graph.
- `policy_dst_slot` stays **`u16`** — it is a per-graph action slot (0..361), never batched-global;
  u16 is correct and halves its H2D.
- `node_feat` (F=11) and `edge_attr` (5) dims are **fixed and small** — no index-width question;
  they are `f32` value arrays, not indices.

### 2.3 The single resolver (version check + one reader)

**One resolver, named:** `hexo_rl/selfplay/graph_collate.py::collate_graph_batch(payload,
expected_version) -> GraphBatch`. It is the SINGLE reader of the wire contract. It:
1. asserts `payload.contract_version == expected_version` (else `GraphContractVersionMismatch`);
2. runs the full assertion set (2.5);
3. builds torch-CUDA tensors (block-diagonal, `edge_index` already global) — WP-A hot-path
   consumer is **torch-CUDA collate** (TORCH-BEATS-ORT verdict). ORT/ONNX consumes the SAME flat
   arrays ONLY in the export/browser path (WP-D, `onnxruntime-web`) — never the self-play loop.

**Eval must read self-play's seam.** The promotion-gate CUDA subprocess
(`promotion_gate_worker.py`, `eval_pipeline.py`) constructs its OWN inference path in its OWN
process (memory `run2-stall-watchdog`). The resolver therefore lives in a module imported by BOTH
`inference_server.py` (self-play) AND the subprocess `EvalPipeline` — import-safe, no parent-process
state (§C3 livelock-isolation interaction). The Rust producer stamps `contract_version`; the
Python `collate_graph_batch` is the one consumer. **No second reader.** (One-resolver-per-knob
standing red-team order.)

### 2.4 policy_scatter_index — replacing the fixed-policy_len scatter

The fixed-`policy_len` scatter break point (`inference_bridge.rs:449` `start=i*policy_len`) is an
OUTPUT-side assumption; the design keeps the output seam and moves the ragged mapping to the INPUT
metadata:

1. The GNN forward produces a per-legal-node policy logit (one scalar per legal node, `Lg` total).
2. Python collate scatters those into a dense `[B, 362]` buffer using `policy_dst_slot` (the action
   slot) segmented by `legal_offsets`: `dense[g, policy_dst_slot[i]] = logit[i]` for i in graph g's
   legal range; non-legal slots get `-inf`/0 (masked, exactly as the CNN's legal mask).
3. That dense `[B, 362]` is handed to the EXISTING `submit_inference_results(ids, policies[B,362],
   values[B])` (`inference_bridge.rs:400`) **unchanged** — the P74/P75 `(Arc<Vec<f32>>, range,
   value)` return contract, the `aggregate_policy` K-cluster machinery
   (`game_runner/records.rs`), and the promotion-gate JSON (`EvalRoundResult`) are all untouched.

**This is the load-bearing decoupling:** the ragged payload is INPUT-only; the output stays dense
`[B,362]`. `policy_dst_slot` maps per-node → move slot via the same `window_flat_idx` action space
the strix bot already re-projects through (`strix_v1_bot.py::get_move`). The action space is
unchanged (a graph plays the identical 19×19+pass board — `policy_logit_count` STAYS 362, §C4).

### 2.5 Error semantics — the assertion set (every mismatch asserts-and-dies, named)

All raised by the single resolver (Python) with a mirror in the Rust producer's debug-assert. **No
silent path.**

| Name | Trigger | Catches |
|---|---|---|
| `GraphContractVersionMismatch` | `contract_version != 1` | wrong/stale producer |
| `NodeFeatDimMismatch` | `len(node_feat) % 11 != 0` or declared F≠11 | truncated/padded node features |
| `EdgeAttrDimMismatch` | `len(edge_attr) % 5 != 0` or `len//5 != E` | truncated edge attrs |
| `DtypeMismatch` | any array dtype ≠ declared (node_feat/edge_attr f32, edge_index/offsets i64, dst_slot u16) | a u16 edge_index that would wrap |
| `BatchCountMismatch` | `len(node_offsets)!=len(edge_offsets)!=len(legal_offsets)!=B+1`, or `len(values)!=B`, or `len(n_nodes_checksum)!=B` | dropped/added graph |
| `OffsetsNonMonotonic` | any `*_offsets` not non-decreasing, or `[0]!=0`, or `[B]!=total` (N/E/Lg) | off-by-one / mis-segmentation (ADV-1) |
| `NodeCountChecksum` | for any g: `node_offsets[g+1]-node_offsets[g] != n_nodes_checksum[g]` | INTERNAL off-by-one that keeps endpoints+monotonic (ADV-1) |
| `EdgeIndexOutOfBounds` | any `edge_index` entry `<0` or `>=N` | corrupt/uninitialised edge id |
| `EdgeCrossesGraphBoundary` | for edge e in graph g's `[edge_offsets[g],edge_offsets[g+1])`, either endpoint outside `[node_offsets[g],node_offsets[g+1])` | edge_index crossing a graph boundary (ADV-3) |
| `ScatterGatherCrossesGraph` | for legal node i in graph g's legal range, `legal_node_gather[i]` outside graph g's node range | scatter gather aliasing two graphs (ADV-2) |
| `ScatterSlotOutOfBounds` | any `policy_dst_slot >= 362` | policy logit into a non-existent move |
| `ScatterSlotAliasing` | within one graph, two legal nodes map to the same `policy_dst_slot` | two legal nodes collapsing into one action (ADV-2) |
| `EmptyLegalSet` | any graph with `legal_offsets[g+1]==legal_offsets[g]` | a position with no legal move → no producible policy row |

### 2.6 Replay-buffer strategy for ragged records

Three options, costs stated:

- **(a) Ragged CSR buffer.** Store the built graph directly: flat `node_feat`/`edge_index`/
  `edge_attr` Vecs + per-record `node_offsets`/`edge_offsets`. Sample slices by offsets (not
  stride). **Cost:** largest surgery — `push`/`sample`/`persist`/`resize` all become
  offset-driven; ~63 KB/record (490·11·2 + 2932·(8+10) bytes) → ~31 GB at 500k capacity (matches
  WP-A's ~36 GB corpus materialization). Zero build cost at sample. Needs a graph-native symmetry
  module (Part 3 graph-space D6).
- **(b) Separate graph-native ring.** Same storage as (a) but a parallel ring beside the dense one
  (lets CNN + GNN buffers coexist for mixed-rep). More code, same storage cost.
- **(c) Store-positions-rebuild-at-sample (RECOMMENDED for v1).** Store a COMPACT position record
  — sorted stone list `[(q,r,player)]`, `current_player`, `moves_remaining`, the MCTS
  visit-distribution over legal moves (ragged, keyed by coord), `outcome`, `winning_line`,
  `ply_index`. Rebuild the graph + align the policy target at sample time on the C1 Rust builder.
  **Cost:** smallest storage (~2 KB/record — stones + a small coord→visit map) → ~1 GB at 500k;
  re-imports the build cost into the sample path (WP-A Rust builder 0.539 ms × batch 256 ≈
  0.14 s/step — borderline but bounded, and the BUILD-HOT perf sub-package optimizes it anyway).
  **Decisive advantage: makes D6 augmentation FREE** — rotate the stone coordinates before
  building, and the builder emits a correctly-oriented graph natively; no graph-symmetry module
  needed (Part 3).

**Ruling: v1 = option (c).** Smallest buffer surgery, free augmentation, bounded sample-time build.
Keep (a) specified as the fallback for a future regime where sample-time build becomes the
bottleneck.

**HEXB persist ruling.** The dense HEXB v9 format (`persist/mod.rs`, magic `0x48455842`, stamps
`encoding_name`, validates stride-sig at `load.rs:125-163`) is **NOT extended in place** — a ragged
record has no `state_stride` and would corrupt the dense reader's `entry_bytes` math
(`load.rs:187`). v1 introduces a **separate `HEXG` graph-position format** (own magic + version)
storing the option-(c) compact records. The dense HEXB lineage is untouched, so a stray graph
buffer handed to the dense loader hits the existing "declares unknown encoding" /
"HEXB version not supported" LOUD-FAIL (`load.rs:116,127`) rather than silently mis-parsing.
**Training path stated:** self-play writes HEXG position records → sampler rebuilds graphs +
aligns targets + coord-rotates for augmentation → torch-CUDA block-diagonal collate. The dense
`.npz` corpus is re-exported as HEXG position records (or replayed-and-rebuilt), guarded by the
existing `batch_assembly.py:195` plane-count assert.

---

## Part 3 — Augmentation ruling (D6 in graph space)

**Verdict: FEASIBLE-ON-LEGACY-V1.** (This is the explicit input to WP-C's decision rule.)

The CNN's 12-fold hex augmentation (6 rotations × 2 reflections = D6, `sym_tables.rs:N_SYMS=12`) is
a dense-tensor coordinate scatter. In graph space the SAME D6 is a node-coordinate remap + edge
re-index. Reading the axis-graph schema (`strix_v1_graph.py`) pins exactly what permutes:

**What transforms under a D6 element:**
- **Node `norm_q`, `norm_r`** (features[4], features[5], relative schema
  `strix_v1_graph.py:114,164-166`): coordinate-carrying → remapped by the hex axial rotation
  matrix (the same rotation the CNN's coordinate scatter encodes). These are the ONLY node features
  that move.
- **`edge_attr[0:3]` axis one-hot** (`strix_v1_graph.py:244`): the 3 WIN_AXES
  `[(1,0),(0,1),(1,-1)]` PERMUTE under rotation → apply the axis permutation.
- **`edge_attr[3]` signed_dist** (`strix_v1_graph.py:245,253`): under a reflection or an
  orientation-reversing rotation the along-axis sign flips → multiply by the axis-map SIGN.

**What is INVARIANT (verified, does NOT move):**
- Node `own`/`opp`/`empty` (player identity — geometric rotation doesn't change who owns a stone),
  `moves` (scalar), `inv_dist` (a distance — rotation-invariant).
- The **4 threat features**: `node_threat_features` returns `[own_max/wl, opp_max/wl, own_axes/3,
  opp_axes/3]` (`strix_v1_graph.py:61`) — `own_max`/`opp_max` are MAX over axes, `own_axes`/
  `opp_axes` are COUNTS of axes over threshold. Both aggregate symmetrically across the 3 axes →
  **axis-permutation invariant**. (This is load-bearing: it means threat dims need NO remap.)
- **`edge_attr[4]` src_player** — player identity, invariant.
- **`edge_index` itself — UNCHANGED.** A D6 element is an automorphism of the hex lattice: the same
  node pairs remain adjacent, so the connectivity (and thus `edge_index`) is identical. Only the
  coordinate features and axis LABELS transform. (This is why graph-space D6 is cheap — no
  re-indexing of the topology.)

**The one new bit of machinery.** The axis permutation already exists:
`sym_tables.rs:164` `axis_perm: [[usize;3]; N_SYMS]` — the per-symmetry 3-axis remap for the CNN's
Q13 chain planes, "board-size invariant (purely a function of the 3 hex axes)"
(`sym_tables.rs:189`). It is **direction-UNSIGNED** (`same_axis`, `sym_tables.rs:132` — chain runs
have no direction). The graph edge case needs the **SIGNED** axis map (which axis → which axis, AND
whether the along-axis direction flips) because `edge_attr[3]` signed_dist carries a sign. So the
graph-space D6 module lifts `axis_perm` and adds the per-element sign vector — a small, well-defined
extension of existing, tested tables.

**Two realizations:**
- **(i) Graph-space D6** (needed only if the buffer stores built graphs, option (a)/(b)): permute
  `edge_attr[0:3]` via signed `axis_perm`, flip `edge_attr[3]` sign, remap node `norm_q`/`norm_r`
  via the axial rotation; `edge_index`, threat, player features untouched. Cost: ~2-3 pd (a
  graph-symmetry module + a byte-parity test against a rotate-then-rebuild oracle).
- **(ii) Coord pre-rotation (FREE, recommended with option (c)):** rotate the stored stone
  coordinates by the D6 element, then rebuild the graph — the builder produces the correctly-
  oriented graph natively, including correct axis labels and signed distances. **Zero new
  graph-symmetry code.** This is the decisive reason option (c) is the v1 buffer strategy.

**Honest data-efficiency cost of dropping aug.** run2 used the full 12-fold
(`sym_tables.rs:N_SYMS=12`). strix's lean-D6 gets 12× augmentation from exactly this graph
symmetry. Dropping augmentation for run4-v1 = ~12× fewer effective distinct training samples per
recorded position — a real hit, NOT a rounding error, especially with the throughput penalty
already reducing raw sample volume (WP-A: ~0.9-1.25k steps/hr vs run2's 4.4k). **Do not drop it.**
Because a feasible realization exists (and a FREE one via option (c) coord pre-rotation), the D6
verdict is **FEASIBLE-ON-LEGACY-V1** — augmentation stays in for v1.

---

## Part 4 — Contract test spec

The contract test suite must (1) verify byte-parity of the built payload against the tested oracle,
and (2) prove each assertion in 2.5 fires on a targeted adversarial payload.

### 4.1 Parity gate (necessary-condition)

- **Builder parity:** the Rust producer's payload must byte-match `build_axis_graph_raw` +
  `_collate_gnn` (`train_bc.py`) on the frozen WP-A position set
  (`reports/probes/gnn_integration/wpa_positions.json`). This oracle is already tested
  (`graph_check.py`, `cross_check.py`, `tests/test_strix_v1_bot.py`).
- **Round-trip:** collate → forward → dense `[B,362]` scatter must equal the strix bot's own
  `policy_logits_for_graph` re-projection on the same positions (parity like WP-A's `max|Δ|<6.6e-7`
  prod gate).

### 4.2 Adversarial payloads (each MUST be caught — a naive impl corrupts silently)

Each adversarial payload is constructed to **PARSE** (valid shapes, valid dtypes, monotonic
top-level offsets) but corrupt silently under a naive implementation. The test asserts the NAMED
error fires.

| ID | Adversarial construction | Naive-impl silent corruption | Caught by |
|---|---|---|---|
| **ADV-1a** | `node_offsets[B]` = `N-1` (drops last node), still monotonic | last graph reads one node short; stride-free slice returns misaligned tail | `OffsetsNonMonotonic` (`[B]!=N`) |
| **ADV-1b** | INTERNAL off-by-one: `node_offsets[g] += 1` for one interior g, endpoints + monotonicity intact | graph g and g-1 both mis-segmented; parses clean | `NodeCountChecksum` (per-graph count ≠ `n_nodes_checksum[g]`) |
| **ADV-2a** | `legal_node_gather[i]` for a legal node of graph g points into graph g+1's node range | graph g's policy row gathers graph g+1's embedding — cross-graph leak, no error | `ScatterGatherCrossesGraph` |
| **ADV-2b** | two legal nodes in graph g share one `policy_dst_slot` | two moves collapse into one action logit; the other move silently gets none | `ScatterSlotAliasing` |
| **ADV-3** | an `edge_index` pair with src in graph g, dst in graph g+1 (global ids both in-range) | message-passing routes across the graph boundary — two positions bleed features; passes bounds check | `EdgeCrossesGraphBoundary` |
| **ADV-4** | `edge_index` as `u16` sized so the batched union wraps past 65535 | node 65536 aliases node 0 — the u16 trap | `DtypeMismatch` (edge_index must be i64) |

**ADV-1b, ADV-2a, ADV-3 are the F1-class cases** — they parse cleanly and corrupt self-play with no
loud failure, exactly the D-FORENSIC F1 pattern. The `n_nodes_checksum` array and the per-graph
range checks (`EdgeCrossesGraphBoundary`, `ScatterGatherCrossesGraph`) exist SOLELY to convert them
from SILENT-CORRUPT to LOUD-FAIL.

### 4.3 Bench gate

The contract sits on the hottest path (`inference_bridge.rs` / `worker_loop` / `replay_buffer` all
fire the `bench-gate` skill). `make bench` is mandatory before any producer/consumer lands
(`docs/rules/perf-targets.md`). Bench on the REAL distribution — the frozen WP-A self-play position
set (mean 490/2932), NOT the lighter human-corpus prior. Standing red-team orders bind: NO
search-time incremental deltas (§S186) — build the payload once per evaluated leaf; one resolver
per knob.

---

## Person-day costs (contract-specific, honest)

These are the ragged-payload SLICE of the scope doc's C3 (8-15 pd) + C8 (10-20 pd) — **not additive
on top of them**. The contract single-sources the collate/offset/scatter logic that C3 and C8 both
need.

| Contract component | Files (primary) | pd |
|---|---|---|
| Rust producer: ragged payload from `infer_and_expand`, block-diagonal offsets, scatter index, `n_nodes_checksum` | `inference_bridge.rs`, `worker_loop/inner.rs`, `pyo3/*` | 3-5 |
| Python single resolver: `graph_collate.collate_graph_batch` → torch-CUDA tensors + full assertion set | `selfplay/graph_collate.py` (new), `inference_server.py` | 2-4 |
| `policy_scatter_index` → dense `[B,362]` re-projection (replaces fixed-policy_len scatter; keeps output seam) | `graph_collate.py`, `inference_server.py` | 1-2 |
| Replay buffer option-(c): HEXG graph-position ring, store stones + ragged policy, rebuild-at-sample, coord-rotate aug | `replay_buffer/*` (new HEXG format), `record_position` | 5-9 |
| Contract test suite: byte-parity oracle + 6 adversarial payloads + assertion coverage | `engine/tests/`, `tests/` | 2-3 |
| Graph-space D6 module (only if option (a)/(b); option (c) makes it FREE) | graph-symmetry module (new) | 0 for v1 (c) / +2-3 if (a) |
| **Total (contract slice, option (c))** | | **13-23 pd** |

Dependency note: the Rust producer is BLOCKED on C1 (the native builder). The resolver and scatter
re-projection can proceed against the probe's `_collate_gnn` oracle in parallel. The HEXG buffer is
co-designed with C8. Cross-cutting: the `build_net(spec,state)` dispatch seam (audit nodes 11a-c)
and the shared `resolvers.py` graph-detect branch (nodes 11d-e) are C4/C7 work single-sourced with
this contract.

---

## Verdict: CONTRACT-SOUND

The design closes every enumerated break point:
- Every inference-seam node (1-5) → RAGGED-OK or LOUD-FAIL; the fixed-width buffer pool (node 3,
  the one SILENT-CORRUPT of the inference seam) is replaced by an asserting ragged path.
- Every training-data node (6-10) → RAGGED-OK; the dense-plane record, stride storage, stride
  sample, and plane symmetry (nodes 6-9, all SILENT-CORRUPT today) are converted via the compact
  HEXG position ring + rebuild-at-sample + coord-rotation aug.
- Every `HexTacToeNet` construction site (11a-e) → the construction-family SILENT-CORRUPT hole
  (11a-c) closed by one `build_net(spec,state)` authority; the loader-family (11d-e) already
  LOUD-FAILs and gets a shared graph-detect branch.
- The u16-wrap trap is ruled out by the i64 edge_index decision (ADV-4).
- The three F1-class silent cases (ADV-1b off-by-one, ADV-2a scatter aliasing, ADV-3 boundary-
  crossing edge) are each converted to a NAMED LOUD-FAIL by a targeted assertion + the
  `n_nodes_checksum` tripwire.

**No SILENT-CORRUPT and no SILENT-FALLBACK node remains** in the audited path. The one residual
risk is discipline, not design: the contract test suite (4.2) and the `make bench` gate (4.3) MUST
land WITH the producer/consumer — an unenforced assertion set is an F1 relapse waiting to happen.
That is a build-order requirement, not a contract gap.
