# GNN Ragged-Payload Contract v1 — WP-B (COND-1) of the GNN-integration program

**Status:** versioned contract design (not a build order). Consumes the R3 scoping doc
(`docs/designs/gnn_integration_scope.md`, esp. §C3 / §C8 / §Red-team pass) and the WP-A CUDA bench
(`reports/probes/gnn_integration/WPA_cuda_bench.md`, verdict CUDA-WINS + BUILD-HOT +
TORCH-BEATS-ORT). Feeds WP-C (D6-in-graph-space decision rule) and WP-D (wasm op-set).

**AMENDED 2026-07-14** per the WP-B red-team pass
(`reports/probes/gnn_integration/WPB_redteam.md`, commit `2cd8bb7`, verdict GAPS-FOUND). Red-team
root cause: the first-cut 13-assertion set was a STRUCTURAL/type validator — it never checked that
an index or attribute points at the geometrically-correct thing. Amendments: (F1-F3) four
SEMANTIC/geometric assertions + five wire-format additions (`node_coords`, `n_stones`,
`window_center`, `current_player`, `builder_impl`) + three new adversarial payloads ADV-7..9;
(F4) Part-3 D6 mechanism rewritten — "edge_index UNCHANGED" was FALSE for the shipped option-(c)
path (coordinate-sorted node rows re-index under rotation); (F5/F6) u16-wrap and batch-union
arithmetic corrected to the measured WP-A distribution; (F7) native-builder handshake — the 26×
Python-builder sample-path trap is now a named LOUD-FAIL, with the WP-5 build-order consequence
stated.

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
~125k nodes / ~750k edges (256 × mean 490 / 2932 — the first cut's "~74k nodes" was the stale
human-corpus mean 290, corrected per red-team F6).

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
| 8 | `replay_buffer/sample.rs:324-328` — `states[idx*state_stride..(idx+1)*state_stride]` scatter | **SILENT-CORRUPT** | pure stride arithmetic; no ragged path; a graph record read at a stride boundary returns misaligned bytes | RAGGED-OK — graph sampler slices by offsets, ONE rebuild call emits graph + slot map + aligned target (F1 single-source) |
| 9 | `sym_tables.rs` + `sample.rs:222-256` — 12-fold `apply_sym` / `apply_symmetry_state` / `apply_chain_symmetry` | **SILENT-CORRUPT** | dense coordinate scatter over cells; on a graph it scatters node-feature bytes as if they were cells | RAGGED-OK — Part 3: option-(c) coord+visit-key pre-rotation, guarded by `AugRoundTripMismatch` + ADV-7 (red-team F1: the aug seam itself was an UNGUARDED silent path in the first cut) |
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
| `policy_dst_slot` | `[Lg]` | **`i32`** | destination action slot (0..361) for each in-window legal node; **−1 = off-window sentinel** (WP-1 measured 43.55% of legal cells off-window). AMENDED from u16 per the WP-3 option-(b) ruling (`gnn_inference_seam_design.md` §1): deploy policy is ragged per-legal-node, so this field is training/probe METADATA — u16 cannot carry the sentinel cleanly and the H2D saving no longer buys a hot-path anything |
| `n_nodes_checksum` | `[B]` | `u32` | per-graph declared node count (stones+legal+1 dummy); off-by-one tripwire (ADV-1) |
| `node_coords` | `[2 * N]` | `i32` | raw `(q, r)` per node (the builder's `coords`, `strix_v1_graph.py:133-145`; dummy = (0,0)). **Enables EXACT geometric verification** — edge_attr recompute (F2), canonical action-slot check (F1/F3). ~1 MB/batch at 125k nodes |
| `n_stones` | `[B]` | `u16` | per-graph stone count → splits each graph's node rows into `[stones \| legal \| dummy]` (builder layout `strix_v1_graph.py:131,136-145`). REQUIRED for the legal-subrange check (F3): without it the resolver structurally cannot tell a stone node from a legal node |
| `window_center` | `[2 * B]` | `i32` | per-graph global-window centre `(bcq, bcr)` — the origin of the coord→action-slot map (`window_flat_idx_at_geom`); enables the canonical-slot check (F1/F3) |
| `current_player` | `[B]` | `i8` | +1 / −1; with the node `own`/`opp` columns this makes `edge_attr[4]` (src_player, absolute) exactly recomputable (F2) |
| `builder_impl` | scalar | `u8` | 1 = native Rust builder, 2 = Python oracle builder. Checked in the version handshake (2.3) — closes the F7 silent-throughput trap |

The last five arrays are the red-team amendment (F1-F3, F7): the geometry needed for semantic
verification now travels ON the wire (n_nodes_checksum gives the total; `n_stones` gives the split
point). Combined overhead ≈ 1 MB/batch at bs=256 — noise against the 5.5 MB `node_feat` +
~12 MB `edge_index`.

Value/output arrays are **unchanged from the CNN contract** (see 2.4).

### 2.2 The u32-vs-u16 ruling (explicit, per WP-A distribution)

**Per-graph node ids fit u16** (max measured 897 nodes << 65535). **Batched flat indices do NOT:**
the block-diagonal union at bs=256 reaches ~125k nodes (256 × mean 490, WP-A self-play
distribution — F6 corrected; the stale human-corpus mean gave "~74k"), ~2× past u16 even in
aggregate. **Ruling: batched flat node indices are `i64`; per-graph u16 is a false economy** — it
cannot survive block-diagonal batching and would silently wrap at node 65536 (SILENT-CORRUPT, the
worst class).

- `edge_index`, `legal_node_gather`, all three `*_offsets` → **`i64`**. Rationale: (a) u16 wraps at
  bs≈73 worst-case (65535 / max 897 nodes) and ~134 at the mean (65535 / 490) — both well below the
  trainer's `batch_size: 256` (`configs/training.yaml`) — disqualified (F5 corrected: the first
  cut's "bs≈45 (65535/1450)" used an unsourced node count; the honest numbers land the same
  conclusion); (b) `u32` is numerically safe (125k < 2^32) but forces a
  device-side cast to `torch.long` for every `scatter`/`index_select` in the GINE message step,
  adding an op + a 2× intermediate on the hot path; (c) `i64` is torch's native index dtype →
  zero-cast, and the +6 MB/batch H2D (edge_index i64 vs u32: 2 × 750k entries × 4 B — re-checked
  against the CORRECT self-play edge count per F6, the figure stands) is immaterial against the
  edge-scatter-bound forward (WP-A: forward is memory/atomics-bound, not H2D-bound). Collate arena
  budget re-derived on the corrected node union: `node_feat` = 125k × 11 × 4 B ≈ 5.5 MB (the stale
  74k figure implied 3.3 MB — understated 1.7×).
  **Downgrade lever (named, not taken for v1):** if a future PCIe-bound regime appears, `edge_index`
  MAY ship `u32`-on-wire with a SINGLE documented cast at the collate resolver (2.3) — never per-op,
  never per-graph.
- `policy_dst_slot` is **`i32`** (AMENDED 2026-07-14; was u16) — it is a per-graph action slot
  (0..361), never batched-global, PLUS the −1 off-window sentinel the u16 cannot carry cleanly.
  Ruling source: WP-3 seam design §1 option (b) (`gnn_inference_seam_design.md`) demoted this
  field from deploy-critical to training/probe metadata (deploy policy rides the ragged
  per-legal-node path), so the original "halves its H2D" rationale no longer binds a hot path.
- `node_feat` (F=11) and `edge_attr` (5) dims are **fixed and small** — no index-width question;
  they are `f32` value arrays, not indices.

### 2.3 The single resolver (version check + one reader)

**One resolver, named:** `hexo_rl/selfplay/graph_collate.py::collate_graph_batch(payload,
expected_version) -> GraphBatch`. It is the SINGLE reader of the wire contract. It:
1. asserts `payload.contract_version == expected_version` (else `GraphContractVersionMismatch`);
2. asserts `payload.builder_impl == 1` (native) on any TRAINING or SELF-PLAY path — the Python
   oracle builder (tag 2) is accepted ONLY under the test-only escape hatch
   `HEXO_ALLOW_ORACLE_BUILDER=1` (parity tests, CI); otherwise `NonNativeSampleBuilder` — the F7
   handshake (see 2.6);
3. runs the full assertion set (2.5) — structural AND semantic/geometric layers;
4. builds torch-CUDA tensors (block-diagonal, `edge_index` already global) — WP-A hot-path
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

> **SUPERSEDED for the deploy-policy path (WP-3 ruling, `gnn_inference_seam_design.md` §1,
> commit daaa21b):** the dense-`[B,362]` output scatter DROPS the 43.55% off-window legal
> nodes (WP-1 measured) — reproducing the pre-R1 decode handicap the +414 evidence removed.
> Deploy policy returns RAGGED per-legal-node logits consumed by the existing
> `LegalSetPolicy` legal-set machinery (option (b)); the dense scatter below remains valid
> ONLY for consumers that are structurally window-bound. The wire fields
> (`policy_dst_slot`, `legal_offsets`) are unchanged — `OFF_WINDOW_SLOT = -1` entries are
> carried, not dropped.

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
silent path.** Two layers (red-team amendment): the **STRUCTURAL layer** (13 checks — index
in-range, unique, monotonic, correctly-typed) and the **SEMANTIC/GEOMETRIC layer** (4 checks + the
F7 startup handshake). The red-team demonstrated (F1-F3) that the structural layer alone is a type
validator — three payloads parse structurally clean and corrupt silently. The semantic layer checks
that indices and attributes point at the geometrically-CORRECT thing, using the geometry the
amended wire now carries (`node_coords`, `n_stones`, `window_center`, `current_player`).

**Structural layer (13):**

| Name | Trigger | Catches |
|---|---|---|
| `GraphContractVersionMismatch` | `contract_version != 1` | wrong/stale producer |
| `NodeFeatDimMismatch` | `len(node_feat) % 11 != 0` or declared F≠11, or `len(node_coords) != 2N` | truncated/padded node features/coords |
| `EdgeAttrDimMismatch` | `len(edge_attr) % 5 != 0` or `len//5 != E` | truncated edge attrs |
| `DtypeMismatch` | any array dtype ≠ declared (node_feat/edge_attr f32, edge_index/offsets/gather i64, dst_slot i32 (amended, §2.2), n_stones u16, coords/centers i32, current_player i8) | a u16 edge_index that would wrap |
| `BatchCountMismatch` | `len(node_offsets)!=len(edge_offsets)!=len(legal_offsets)!=B+1`, or `len(values)!=B`, or `len(n_nodes_checksum)!=B`, or `len(n_stones)!=B`, or `len(window_center)!=2B`, or `len(current_player)!=B` | dropped/added graph; missing semantic-layer array |
| `OffsetsNonMonotonic` | any `*_offsets` not non-decreasing, or `[0]!=0`, or `[B]!=total` (N/E/Lg) | off-by-one / mis-segmentation (ADV-1) |
| `NodeCountChecksum` | for any g: `node_offsets[g+1]-node_offsets[g] != n_nodes_checksum[g]`, or `n_stones[g] + 1 > n_nodes_checksum[g]` | INTERNAL off-by-one that keeps endpoints+monotonic (ADV-1) |
| `EdgeIndexOutOfBounds` | any `edge_index` entry `<0` or `>=N` | corrupt/uninitialised edge id |
| `EdgeCrossesGraphBoundary` | for edge e in graph g's `[edge_offsets[g],edge_offsets[g+1])`, either endpoint outside `[node_offsets[g],node_offsets[g+1])` | edge_index crossing a graph boundary (ADV-3) |
| `ScatterGatherCrossesGraph` | for legal node i in graph g's legal range, `legal_node_gather[i]` outside graph g's node range | scatter gather aliasing two graphs (ADV-2) |
| `ScatterSlotOutOfBounds` | any `policy_dst_slot >= 362`, or negative and ≠ −1 (the off-window sentinel, §2.1 i32 amendment) | policy logit into a non-existent move |
| `ScatterSlotAliasing` | within one graph, two legal nodes map to the same `policy_dst_slot` | two legal nodes collapsing into one action (ADV-2) |
| `EmptyLegalSet` | any graph with `legal_offsets[g+1]==legal_offsets[g]` | a position with no legal move → no producible policy row |

**Semantic/geometric layer (4) + startup handshake (1) — red-team F1-F3/F7 fixes:**

| Name | Trigger | Catches |
|---|---|---|
| `EdgeAttrGeometryMismatch` | for edge e=(u,v): recompute expected attrs from wire geometry — Δ=`node_coords[v]−node_coords[u]` must equal `signed_dist × axis_vec` for exactly the axis the one-hot claims; `src_player` must match u's stone identity (from node own/opp cols + `current_player[g]`, 0.0 for empty/dummy); edges touching the dummy row (`node_offsets[g+1]−1`) must be ALL-ZERO attrs (`strix_v1_graph.py` dummy-edge rule) — any mismatch dies | edge_attr rows permuted/misaligned within a graph — the positional-correspondence hole (ADV-8, F2) |
| `GatherNotLegalNode` | `legal_node_gather[i]` for graph g outside the LEGAL subrange `[node_offsets[g]+n_stones[g], node_offsets[g+1]−1)` (rows below = stones, last row = dummy); or `legal_offsets[g+1]−legal_offsets[g] != n_nodes_checksum[g]−n_stones[g]−1` | gather at a stone or dummy node of the SAME graph — in-range but wrong node-kind (ADV-9, F3) |
| `ScatterSlotCanonicalMismatch` | `policy_dst_slot[i] != window_flat_idx_at_geom(node_coords[legal_node_gather[i]], window_center[g], trunk_sz=19)` — the slot must be the CANONICAL action slot of the gathered node's (possibly rotated) coordinate | valid-unique-but-wrong-cell slot permutations (F3 sibling) AND graph-rotated/slot-map-unrotated desync (ADV-7 slot leg, F1) |
| `AugRoundTripMismatch` | augmentation round-trip parity: applying element s then s⁻¹ to (graph, TARGET) must recover the stored record byte-identically (test-time, every ADV/CI run; runtime canary form each sampled batch: the target-argmax cell must map to a legal node whose `policy_dst_slot` equals the canonical slot of that cell's ROTATED coordinate) | graph rotated / target NOT (or rotated by s′≠s) — silent label poisoning, the headline F1 payload (ADV-7) |
| `NonNativeSampleBuilder` | startup handshake (resolver step 2): `builder_impl != 1` on any training/self-play path without `HEXO_ALLOW_ORACLE_BUILDER=1` | the 26× Python-builder sample-path throughput trap (F7) |

**Where the semantic checks run (cost stated honestly).** All checks are vectorizable numpy/torch
array ops over the flat wire arrays. On the TRAINER sample path they run FULL every batch
(~750k-edge recompute ≈ 10-20 ms vectorized, small against the ~138 ms/step native rebuild). On the
self-play INFERENCE hot path the producer is the same audited Rust walk emitting `edge_attr`,
`node_coords`, and `policy_dst_slot` from ONE pass (single-source, so internal desync requires the
same bug twice); the full geometric layer runs on the FIRST batch after every process start /
weight swap and every Nth batch thereafter (canary cadence, N a config knob with a loud log line),
plus ALWAYS-FULL in tests and debug builds. The structural layer runs full everywhere, always.
Primary F1 defense is STRUCTURAL-BY-CONSTRUCTION (2.6: one rebuild call emits graph + slot map +
aligned target — separate rotation of the target is forbidden by design), with `AugRoundTripMismatch`
as the enforcement that the construction stays single-sourced.

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
  re-imports the build cost into the sample path — **0.14 s/step CONDITIONAL on the C1 native
  builder** (WP-A Rust proxy 0.539 ms × batch 256 ≈ 0.138 s/step — borderline but bounded, and the
  BUILD-HOT perf sub-package optimizes it anyway; see the native-builder requirement below for the
  Python-path trap).
  **Decisive advantage: makes D6 augmentation FREE** — rotate the stored stone coordinates AND the
  stored visit-map KEYS (the policy-target coords — rotating stones alone desyncs the label, F1/F4)
  by the same D6 element inside the ONE rebuild call; that single call emits the re-indexed graph,
  the rotated legal set, the recomputed `policy_dst_slot`, and the aligned target row together —
  **single source of orientation, so graph/target desync is structurally impossible** (the F1 fix),
  enforced by `AugRoundTripMismatch` (2.5) + ADV-7 (4.2). No graph-symmetry module needed (Part 3).

**Ruling: v1 = option (c).** Smallest buffer surgery, free augmentation, bounded sample-time build.
Keep (a) specified as the fallback for a future regime where sample-time build becomes the
bottleneck.

**Native-builder requirement (F7 — contract-level, not advisory).** The 0.138 s/step figure
assumes the C1 Rust builder, which is BLOCKED on WP-1 and does not exist yet. The only builder that
exists TODAY is the Python `build_axis_graph_raw` — measured **14.0 ms/pos on the self-play
distribution** (WP-A) → 14.0 ms × 256 = **3.58 s/step, a 26× trap**: the sample path silently
becomes ~90% of a 4 s/step budget instead of ~3.5%. Because the BC oracle path (`_collate_gnn`)
already rides the Python builder, the natural first implementation before C1 lands would inherit
it with no error — the throughput analog of the F1 silent-fallback class. The contract makes this
impossible silently: the producer stamps `builder_impl` (2.1), the resolver's version handshake
asserts `builder_impl == 1` on every training/self-play path (`NonNativeSampleBuilder`, 2.5), and
the Python oracle is reachable ONLY under the test-only `HEXO_ALLOW_ORACLE_BUILDER=1` flag.
**Interim build-order consequence (WP-5):** option-(c) training is NOT viable until the C1 native
builder lands — gate any run4/HEXG training launch on C1; there is no legitimate interim
Python-builder training configuration, and the handshake enforces that (hard refuse-to-train, not
a warning).

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
- **Adjacency PAIRS are preserved; the integer `edge_index` array is NOT** (F4 correction — the
  first cut claimed "edge_index UNCHANGED", which is FALSE for the shipped path). A D6 element is
  an automorphism of the hex lattice, so the same node PAIRS remain adjacent — but the builder
  assigns node ROW indices by COORDINATE SORT (stones `sorted(stone_map.items())`,
  `strix_v1_graph.py:101`; legal `sorted(candidates)`, `:78`), and rotation changes the sort keys
  → the row order → the integers in `edge_index`. Under the shipped option-(c) rebuild the
  re-indexing happens correctly BY CONSTRUCTION (the rebuild re-sorts the rotated coords and emits
  a self-consistent graph). Hard rule that falls out: **`edge_index` MUST NOT be cached or reused
  across augmentations** — the "no re-indexing" optimization the first cut's wording invited would
  silently mis-wire every rotated graph (exactly the SILENT-CORRUPT class this doc exists to kill).

**The one new bit of machinery.** The axis permutation already exists:
`sym_tables.rs:164` `axis_perm: [[usize;3]; N_SYMS]` — the per-symmetry 3-axis remap for the CNN's
Q13 chain planes, "board-size invariant (purely a function of the 3 hex axes)"
(`sym_tables.rs:189`). It is **direction-UNSIGNED** (`same_axis`, `sym_tables.rs:132` — chain runs
have no direction). The graph edge case needs the **SIGNED** axis map (which axis → which axis, AND
whether the along-axis direction flips) because `edge_attr[3]` signed_dist carries a sign. So the
graph-space D6 module lifts `axis_perm` and adds the per-element sign vector — a small, well-defined
extension of existing, tested tables.

**Two realizations (F4-corrected):**
- **(i) Graph-space D6 in-place transform** (only if the buffer stores built graphs, option
  (a)/(b) fallback): rotate all node coordinates, then apply the node permutation induced by
  re-sorting the ROTATED coordinates (stones and legal re-sorted separately, dummy stays last),
  remap `edge_index` THROUGH that permutation, permute `edge_attr[0:3]` via the signed `axis_perm`
  lift, flip `edge_attr[3]` per the element's along-axis sign, remap node `norm_q`/`norm_r`, and
  **RECOMPUTE `policy_dst_slot` from each legal node's rotated coordinate** (the action slot
  follows the cell to its new location — omitted in the first cut, F4); threat + player features
  untouched (invariant, above). Parity test: byte-parity vs a rotate-then-rebuild oracle holds
  ONLY after the canonical re-sort step — the first cut proposed byte-parity while keeping row
  order pinned, which is **unsatisfiable** (the oracle re-sorts, the pinned transform doesn't);
  without the re-sort the honest gate is graph-ISOMORPHISM parity, which is weaker and slower.
  Cost: ~3-4 pd. Not built for v1.
- **(ii) Coord pre-rotation + rebuild (FREE, the SHIPPED realization with option (c)):** rotate the
  stored stone coordinates AND the visit-map keys by the D6 element, rebuild — the builder
  natively produces the correctly re-indexed graph (rotated-coord sort), correct axis labels,
  correct signed distances, the recomputed `policy_dst_slot`, and the aligned target, all from one
  call (2.6). **Zero new graph-symmetry code**, and the single-call structure is itself the F1
  defense. This is the decisive reason option (c) is the v1 buffer strategy. Enforcement:
  `AugRoundTripMismatch` (2.5) + ADV-7 (4.2) — s then s⁻¹ must recover the stored record
  byte-identically, graph AND target.

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
- **Augmented parity (red-team F1 — the first cut tested only the UN-augmented builder):** for
  every D6 element s and every frozen position: rebuild-at-s then rebuild-at-s⁻¹∘s must be
  byte-identical to rebuild-at-identity, for the FULL record — graph arrays AND `policy_dst_slot`
  AND the aligned target row. This is the test-time face of `AugRoundTripMismatch`.

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
| **ADV-7** (red-team F1, the 7th payload — DEFEATED the first-cut assertion set) | augmentation-seam desync: graph rotated by s, policy TARGET left unrotated (or rotated by s′≠s); D6 preserves node/edge counts so ALL 13 structural checks pass | network trained to map an s-rotated board onto the un-rotated move distribution — silent label poisoning, the exact §119/D-FORENSIC-F1 class | slot-map leg: `ScatterSlotCanonicalMismatch` (dst_slot vs canonical slot of the ROTATED coord); target-value leg: `AugRoundTripMismatch` (s∘s⁻¹ byte-identity of graph AND target + argmax-canary) — plus structurally-impossible-by-construction via the single rebuild call (2.6) |
| **ADV-8** (red-team F2) | `edge_attr` rows permuted within one graph (or collate concatenates edge_index in graph order but edge_attr off-by-one-graph); total length 5E, dtypes, offsets all intact | GINE consumes the wrong axis / signed_dist / src_player for every edge — silent relational-feature scramble; no positional-correspondence check existed | `EdgeAttrGeometryMismatch` (recompute attrs from `node_coords` endpoints + `current_player`; dummy edges all-zero) |
| **ADV-9** (red-team F3) | `legal_node_gather[i]` pointed at a STONE row (`< n_stones[g]`) or the DUMMY row (last) of the SAME graph — inside graph g's node range, so the cross-graph check passes | policy logit for a "legal move" gathered from a stone/dummy embedding, scattered to a valid unique slot — silent wrong-source policy row | `GatherNotLegalNode` (legal-subrange membership via the new per-graph `n_stones`) |

**ADV-1b, ADV-2a, ADV-3, and ADV-7..9 are the F1-class cases** — they parse cleanly and corrupt
self-play with no loud failure, exactly the D-FORENSIC F1 pattern. ADV-7..9 additionally DEFEAT a
purely structural validator (the red-team's root-cause finding): every index is in-range, unique,
monotonic, and typed — and still wrong. The `n_nodes_checksum` array, the per-graph range checks
(`EdgeCrossesGraphBoundary`, `ScatterGatherCrossesGraph`), and the semantic/geometric layer
(`EdgeAttrGeometryMismatch`, `GatherNotLegalNode`, `ScatterSlotCanonicalMismatch`,
`AugRoundTripMismatch`) exist SOLELY to convert them from SILENT-CORRUPT to LOUD-FAIL.

### 4.3 Bench gate

The contract sits on the hottest path (`inference_bridge.rs` / `worker_loop` / `replay_buffer` all
fire the `bench-gate` skill). `make bench` is mandatory before any producer/consumer lands
(`docs/rules/perf-targets.md`). Bench on the REAL distribution — the frozen WP-A self-play position
set (mean 490/2932), NOT the lighter human-corpus prior. The bench gate additionally asserts
`builder_impl == 1` on the benched path (F7): a Python-builder bench run is BOTH an invalid
measurement AND the exact trap the handshake exists to catch — it must fail the gate, not skew it.
Standing red-team orders bind: NO search-time incremental deltas (§S186) — build the payload once
per evaluated leaf; one resolver per knob.

---

## Person-day costs (contract-specific, honest)

These are the ragged-payload SLICE of the scope doc's C3 (8-15 pd) + C8 (10-20 pd) — **not additive
on top of them**. The contract single-sources the collate/offset/scatter logic that C3 and C8 both
need.

| Contract component | Files (primary) | pd |
|---|---|---|
| Rust producer: ragged payload from `infer_and_expand`, block-diagonal offsets, scatter index, `n_nodes_checksum` + semantic-layer arrays (`node_coords`, `n_stones`, `window_center`, `current_player`, `builder_impl`) | `inference_bridge.rs`, `worker_loop/inner.rs`, `pyo3/*` | 3-6 |
| Python single resolver: `graph_collate.collate_graph_batch` → torch-CUDA tensors + full assertion set (13 structural + 4 semantic/geometric vectorized + F7 handshake, canary cadence knob) | `selfplay/graph_collate.py` (new), `inference_server.py` | 3-5 |
| `policy_scatter_index` → dense `[B,362]` re-projection (replaces fixed-policy_len scatter; keeps output seam) | `graph_collate.py`, `inference_server.py` | 1-2 |
| Replay buffer option-(c): HEXG graph-position ring, store stones + ragged policy, rebuild-at-sample (single-call graph+target emission, F1), coord+visit-key-rotate aug | `replay_buffer/*` (new HEXG format), `record_position` | 5-9 |
| Contract test suite: byte-parity oracle + augmented round-trip parity + 9 adversarial payloads + assertion coverage | `engine/tests/`, `tests/` | 3-4 |
| Graph-space D6 module (only if option (a)/(b); option (c) ships realization (ii) FREE) | graph-symmetry module (new) | 0 for v1 (c) / +3-4 if (a) |
| **Total (contract slice, option (c))** | | **15-26 pd** (was 13-23 pre-amendment; delta = semantic-layer arrays + vectorized geometry checks + ADV-7..9 tests) |

Dependency note: the Rust producer is BLOCKED on C1 (the native builder). The resolver and scatter
re-projection can proceed against the probe's `_collate_gnn` oracle in parallel. The HEXG buffer is
co-designed with C8. Cross-cutting: the `build_net(spec,state)` dispatch seam (audit nodes 11a-c)
and the shared `resolvers.py` graph-detect branch (nodes 11d-e) are C4/C7 work single-sourced with
this contract.

---

## Verdict

The design closes every enumerated break point, including the three red-team payloads that
defeated the first cut:
- Every inference-seam node (1-5) → RAGGED-OK or LOUD-FAIL; the fixed-width buffer pool (node 3,
  the one SILENT-CORRUPT of the inference seam) is replaced by an asserting ragged path.
- Every training-data node (6-10) → RAGGED-OK; the dense-plane record, stride storage, stride
  sample, and plane symmetry (nodes 6-9, all SILENT-CORRUPT today) are converted via the compact
  HEXG position ring + single-call rebuild-at-sample + coord/visit-key-rotation aug.
- Every `HexTacToeNet` construction site (11a-e) → the construction-family SILENT-CORRUPT hole
  (11a-c) closed by one `build_net(spec,state)` authority; the loader-family (11d-e) already
  LOUD-FAILs and gets a shared graph-detect branch.
- The u16-wrap trap is ruled out by the i64 edge_index decision (ADV-4), on the corrected
  arithmetic (F5: wrap at bs≈73 worst-case / ~134 mean; F6: true union ~125k nodes / ~750k edges —
  the i64 and +6 MB conclusions stand on the honest numbers).
- The six F1-class silent cases are each a NAMED LOUD-FAIL: ADV-1b off-by-one → `NodeCountChecksum`;
  ADV-2a gather aliasing → `ScatterGatherCrossesGraph`; ADV-3 boundary-crossing edge →
  `EdgeCrossesGraphBoundary`; and the red-team's three semantic payloads — ADV-7 augmentation-seam
  desync → `ScatterSlotCanonicalMismatch` + `AugRoundTripMismatch` (plus
  structurally-impossible-by-construction via the single rebuild call); ADV-8 edge_attr permutation
  → `EdgeAttrGeometryMismatch`; ADV-9 gather-at-stone/dummy → `GatherNotLegalNode` (via the new
  per-graph `n_stones` wire field).
- The assertion set is now two-layer (13 structural + 4 semantic/geometric + the F7
  `NonNativeSampleBuilder` startup handshake = 18 named errors): indices are checked not only for
  structure but for pointing at the geometrically-correct thing — the red-team's root-cause gap.
- The 26× Python-builder sample-path trap (F7) is impossible silently: `builder_impl` handshake,
  hard refuse-to-train, WP-5 gated on the C1 native builder.
- The D6 mechanism (Part 3) now matches the shipped realization: `edge_index` re-indexes under
  rotation via the coordinate sort (F4) and MUST NOT be cached across augmentations;
  FEASIBLE-ON-LEGACY-V1 confirmed by the red-team as unbroken.

**No SILENT-CORRUPT and no SILENT-FALLBACK node remains** in the audited path — structural OR
semantic. The one residual risk is discipline, not design: the contract test suite (4.2, all 9
adversarial payloads) and the `make bench` gate (4.3, native-builder-asserted) MUST land WITH the
producer/consumer — an unenforced assertion set is an F1 relapse waiting to happen. That is a
build-order requirement, not a contract gap.

**CONTRACT-SOUND (amended post-red-team, 2cd8bb7 findings closed)**
