# WP-B RED-TEAM — adversarial review of `gnn_ragged_contract_v1.md`

**Date:** 2026-07-14 · **Program:** GNN-integration (R4 ratified b+) · **Role:** WP-B RED-TEAM
(adversarial reviewer, DISTINCT from the contract author) · **Target:** `docs/designs/gnn_ragged_contract_v1.md`
(committed `e318c84`), claim **CONTRACT-SOUND** ("zero SILENT-CORRUPT, zero SILENT-FALLBACK node
remains"). **Method:** source-level read, not grep. **Verdict:** **GAPS-FOUND** (see end).

Root cause running through F1–F3: **the 13-name assertion set (2.5) is a STRUCTURAL/type validator,
not a SEMANTIC/geometric one.** It checks that every index is in-range, unique, monotonic, and
correctly-typed — it NEVER checks that an index or attribute points at the geometrically-correct
thing. Every uncaught payload below exploits exactly that. The payload already carries the geometry
needed to close the gap (node coords live in `node_feat[norm_q,norm_r]`), so the fixes are cheap.

---

## F1 — 7th adversarial payload: augmentation-seam desync (graph rotated, target NOT) — **BLOCKS-CONTRACT-SOUND** (HEADLINE)

**CLAIM (doc).** Verdict: *"No SILENT-CORRUPT and no SILENT-FALLBACK node remains in the audited
path."* Audit node 9 (symmetry, `sample.rs:222-256`) design class **RAGGED-OK** via *"option-(c)
coord-pre-rotation makes aug free"*; node 8 (`sample.rs:324-328`) RAGGED-OK: *"graph sampler slices
by offsets, rebuilds graph + aligns policy target."* Part 3 realization (ii): *"rotate the stored
stone coordinates by the D6 element, then rebuild the graph."* The training-data augmentation path is
explicitly IN the audited path and claimed covered.

**The payload.** Under option-(c) rebuild-at-sample with D6 aug, the sampler (a) rotates stone coords
by symmetry element `s`, rebuilds → rotated graph + rotated `legal_coords` + `policy_dst_slot`; and
SEPARATELY (b) permutes the stored visit-distribution target by `s`. These are two independent code
paths. Construct a batch where the GRAPH is rotated by `s` but the POLICY TARGET is left unrotated
(or rotated by `s' ≠ s`) — a one-line producer bug (forgot to permute the target, or applied the
CNN's `sym_idx` to the graph and identity to the label).

**Why it passes all 13 assertions.** D6 is a lattice automorphism, so rotation preserves node count
(same stones + same legal-cell count → `n_nodes_checksum` matches), edge count, offset monotonicity,
dtypes, edge bounds, edge-within-graph, gather-within-graph, slot bounds, slot uniqueness. Every
array is structurally valid. The corruption is that the LABEL the loss compares against belongs to a
different orientation than the graph — **the policy TARGET is not in any checked array** (2.1 carries
`policy_dst_slot`, the index map — never the target VALUES), and **no invariant ties the graph's
orientation to the target's orientation.**

**What corrupts.** The network is trained to map a board rotated by `s` onto the un-rotated move
distribution. Silent label poisoning — precisely the §119 axis-bias / D-FORENSIC-F1 class the doc
names as the program's raison d'être (Part 1 line 12-14; RISK-1). The seam WP-C's ENTIRE
FEASIBLE-ON-LEGACY-V1 verdict rests on is left UNGUARDED and UNTESTED (4.2 has no augmentation
adversarial; 4.1 byte-parity tests only the UN-augmented builder).

**EVIDENCE.** `gnn_ragged_contract_v1.md` §2.5 (assertion table — no target/graph orientation check),
§4.2 (adversarial table ADV-1..6 — none is an augmentation-desync), §2.1 (target values absent from
wire arrays). Builder confirms rotation preserves counts: `strix_v1_graph.py:64-78`
(`legal_moves_from_stones` = empty cells within radius of any stone → automorphism-invariant count),
`:101-102` (stones/legal both coord-sorted).

**FIX.** Add an augmentation round-trip parity assertion to 4.2 + 2.5: rotate graph AND target by `s`,
then by `s⁻¹`, assert byte-identity recovers the stored record; and a runtime invariant tying the
target's argmax cell to a legal node whose `policy_dst_slot` equals the canonical action-slot of that
cell's ROTATED coordinate. Cheapest form: compute the target-alignment inside the SAME rebuild call
that emits `policy_dst_slot` (single source of the orientation), so desync is structurally
impossible, and add the round-trip test as the enforcement.

---

## F2 — 8th payload: `edge_attr` rows permuted within a graph — **BLOCKS-CONTRACT-SOUND**

**CLAIM (doc).** §2.5 `EdgeAttrDimMismatch` triggers on `len(edge_attr)%5 != 0` or `len//5 != E`;
that is the ONLY `edge_attr` check.

**The payload.** Permute `edge_attr` rows within one graph (or, realistically, a Rust collate that
concatenates per-graph `edge_index` in graph order but `edge_attr` in a different order / off-by-one
graph). Total length stays `5E`; multiset of axis one-hots and signed-dists unchanged.

**Why it passes.** No assertion ties `edge_attr[e]` to `edge_index[:,e]`. Dim check passes, dtype
passes, offsets/bounds all pass. **There is no positional correspondence check.**

**What corrupts.** GINE message-passing consumes `edge_attr[e]` for edge `e`; every edge now carries
the wrong axis / signed-distance / `src_player`. Silent relational-feature scramble. The builder keeps
the three arrays aligned via a shared `keep` list (`strix_v1_graph.py:279-282`), but a buggy first-
party Rust port (WP-1) or a mis-fenced block-diagonal collate reintroduces the misalignment with zero
loud failure.

**EVIDENCE.** `gnn_ragged_contract_v1.md` §2.5 (only dim check on edge_attr); `strix_v1_graph.py:239-255`
(edge_attr[0:3]=axis one-hot, [3]=signed_dist, [4]=src_player are pure functions of endpoint geometry).

**FIX.** Geometric consistency assertion: recompute expected axis one-hot + `signed_dist` from the
edge's endpoint coordinates (available via `node_feat[norm_q,norm_r]` de-normalized, or the stored
stones) and assert match. Converts the whole edge_attr-misalignment class to LOUD-FAIL.

---

## F3 — 9th payload: `legal_node_gather` → stone or dummy node (valid range, wrong node-kind) — **BLOCKS-CONTRACT-SOUND**

**CLAIM (doc).** §2.5 `ScatterGatherCrossesGraph`: *"for legal node i in graph g's legal range,
`legal_node_gather[i]` outside graph g's node range."* Catches only CROSS-graph gather.

**The payload.** Point a legal node's `legal_node_gather[i]` at a STONE node (row `< n_stones`) or the
DUMMY node (row `n_real`) of the SAME graph `g`. Both are inside graph g's node range.

**Why it passes.** The check is graph-RANGE membership only. Stone rows and the dummy row are within
`[node_offsets[g], node_offsets[g+1])`, so the gather passes. The wire contract **carries no per-graph
stone/legal split** — `n_nodes_checksum` is a single count *"stones+legal+1 dummy"* (2.1), and there is
no `stone_mask`/`legal_mask`/`n_stones` array. **The resolver structurally cannot tell a stone node
from a legal node**, so it cannot enforce that `legal_node_gather` indexes an actually-legal (empty)
node.

**What corrupts.** The policy logit for a "legal move" is gathered from a stone's (or the all-zero
dummy's) embedding and scattered to a valid, unique action slot — a silent wrong-source policy row.
Same family, inference side: a `policy_dst_slot` that is a valid unique permutation of the WRONG cells
(right structure, wrong geometry) also passes `ScatterSlotOutOfBounds` + `ScatterSlotAliasing` and
silently permutes the dense `[B,362]` policy.

**EVIDENCE.** `gnn_ragged_contract_v1.md` §2.1 (no stone/legal marker on the wire; `n_nodes_checksum`
is one combined count), §2.5 (`ScatterGatherCrossesGraph` range-only). Builder distinguishes the kinds
but only internally: `strix_v1_graph.py:204-210` (`legal_mask`/`stone_mask`), `:131` (dummy at
`n_real`) — none exported to the contract's array set.

**FIX.** Add per-graph `n_stones` (or export `legal_mask`) so the resolver asserts
`legal_node_gather[i] ∈ [node_offsets[g]+n_stones[g], node_offsets[g]+n_stones[g]+n_legal[g])`, and
assert `policy_dst_slot[i]` equals the canonical action-slot of `legal_coords[i]`.

---

## F4 — Part 3 D6 mechanism: "edge_index UNCHANGED" is wrong for the shipped path — **MUST-FIX-BEFORE-BUILD** (FEASIBLE verdict survives)

**CLAIM (doc, Part 3 :253-256).** *"`edge_index` itself — UNCHANGED. A D6 element is an automorphism
of the hex lattice: the same node pairs remain adjacent, so the connectivity (and thus `edge_index`)
is identical … no re-indexing of the topology."* Realization (i) cost (:271): *"a byte-parity test
against a rotate-then-rebuild oracle."*

**EVIDENCE the mechanism is wrong.** The builder assigns node ROW indices by COORDINATE SORT: stones
`sorted(stone_map.items(), key=lambda kv: kv[0])` (`strix_v1_graph.py:101`), legal `sorted(candidates)`
(`:78`). `edge_index` stores those row indices (`:241-242,249-250`). Under a D6 rotation the sort key
changes, so the sort order — and thus the integer row indices — **change**. The set of adjacent PAIRS
is preserved (automorphism); the integer array `edge_index` is NOT, unless node rows are pinned.

Two consequences, both fatal to the claim as written:
1. **Shipped path (option-c rebuild = realization ii):** rebuild sorts by ROTATED coords → node order
   differs → `edge_index` integers differ from the un-rotated graph. So "edge_index is identical" is
   **FALSE for the path v1 actually ships.** The claim invites the optimization it advertises ("no
   re-indexing of the topology" → cache edge_index across augmentations), which would silently corrupt
   under option-c.
2. **Realization (i) in-place remap:** keeping row order lets `edge_index` stay identical, but then it
   does NOT byte-match the "rotate-then-rebuild oracle" (which re-sorts) — the proposed byte-parity
   test (:271) is **unsatisfiable** by the realization it tests. And the (i) description omits
   recomputing `policy_dst_slot` for the rotated cells (the action slot follows the cell to its new
   location) — a second silent hole.

**Does FEASIBLE-ON-LEGACY-V1 survive? YES.** Option-c coord-pre-rotation + rebuild is genuinely
feasible and correct (the GNN is permutation-equivariant; the rebuild produces a self-consistent
graph+`legal_coords`+`policy_dst_slot`). So **WP-C's LEGACY-V1-CONFIRMED decision input is NOT
broken.** What is broken is the doc's *explanation* — and the realization-(i)/(a)-(b) fallback spec,
which would produce a mis-permuted graph or fail its own oracle if ever built.

**FIX.** (1) Delete/replace the "edge_index UNCHANGED" claim; state that under option-c rebuild
`edge_index` is re-indexed by the rotated-coordinate sort permutation (correctly), and MUST NOT be
cached across augmentations. (2) For the (a)/(b) fallback: either re-sort rows to canonical order (then
`edge_index` IS remapped through the node permutation) OR make the parity test graph-ISOMORPHISM (not
byte) parity; and add `policy_dst_slot` recomputation to realization (i). (3) Note that option-c also
must rotate the stored visit-map KEYS, not only stone coords (:273-274 mentions only stones) — folds
into F1.

---

## F5 — u16-wrap arithmetic "65535/1450 nodes, bs≈45" is unjustified — **NOTE** (i64 conclusion holds)

**CLAIM (doc :113-114).** *"(a) u16 wraps at bs≈45 (65535/1450 nodes) — disqualified."*

**EVIDENCE.** `65535/1450 = 45.2` (self-consistent), but **1450 nodes/graph matches no WP-A statistic**:
measured mean 490, p90 729, max 897 (`WPA_cuda_bench.md:27-28`). 1450 is ~1.6× the worst-case max and
~3× the mean — fabricated. Honest recompute of the u16 union-wrap batch size:

| basis | nodes/graph | u16 wrap bs |
|---|---|---|
| mean | 490 | **134** |
| p90 | 729 | 90 |
| worst-case max | 897 | **73** |

The doc's bs≈45 is EARLIER than even the worst case (73), i.e. it OVERSTATES the danger — so the
conclusion (**i64; u16 disqualified**) still holds at the trainer's real `batch_size: 256`
(`configs/training.yaml:45`), where the union is 125k ≫ 65535 for any per-graph size. But per the
red-team brief, load-bearing arithmetic that is wrong-but-conveniently-conservative still gets flagged:
it invites the next error.

**FIX.** Replace with: *"u16 wraps at bs≈73 (worst-case 65535/897) — well below `batch_size=256`;
disqualified."*

---

## F6 — "~74k nodes bs=256 union" is the STALE human-corpus number — **MUST-FIX-BEFORE-BUILD** (internal inconsistency; +6MB survives)

**CLAIM (doc :19-20, :108-109).** *"batched union at bs=256 reaches ~74k nodes / ~750k edges"* /
*"the block-diagonal union at bs=256 reaches ~74k nodes."*

**EVIDENCE.** With the doc's OWN self-play mean (490 nodes): `256 × 490 = 125,440 ≈ 125k` nodes. The
`74k` = `256 × 290 = 74,240` — the **human-corpus** mean (290) WP-A explicitly superseded
(`WPA_cuda_bench.md:28,49`: *"mean 490/2932 vs 290/1294 nodes/edges"*). The edge figure `750k` =
`256 × 2932` IS the correct self-play number. **So the same sentence mixes human-corpus NODES with
self-play EDGES** — the nodes were not updated when the distribution was corrected. True self-play
union at bs=256 ≈ **125k nodes / 750k edges**.

Blast radius: (a) `node_feat` memory understated 1.7× (`125k×11×4 = 5.5 MB` vs the doc's implied
`74k×11×4 = 3.3 MB`) — matters for the collate arena budget; (b) the **+6 MB/batch i64-vs-u32 claim
SURVIVES** — it was computed from the CORRECT edge count (`2×750k×4 = 6.0 MB`), not the wrong node
count, so that specific number is right despite the node error.

**FIX.** Replace "~74k nodes" with "~125k nodes" everywhere (:19, :108-109); re-derive any memory
budget that consumed the node union.

---

## F7 — option-c "0.14 s/step" assumes the Rust builder; interim path is a 26× silent throughput trap — **MUST-FIX-BEFORE-BUILD**

**CLAIM (doc :204, §2.6).** *"re-imports the build cost into the sample path (WP-A Rust builder
0.539 ms × batch 256 ≈ 0.14 s/step — borderline but bounded)."* Recommended v1 = option (c).

**EVIDENCE.** `0.539 ms × 256 = 0.138 s/step` — correct, but **conditional on the RUST builder**, which
is BLOCKED on C1/WP-1 (the doc admits this at :349: *"the Rust producer is BLOCKED on C1"*). The only
builder that exists TODAY is the Python `build_axis_graph_raw`, measured at **14.0 ms/pos**
(`WPA_cuda_bench.md:49`). Python rebuild-at-sample = `14.0 ms × 256 = 3.58 s/step` — a **26× trap**,
turning the trainer sample path (option-c rebuilds EVERY sampled position every step) from ~3.5% of a
4 s/step budget into ~90% of it. The `_collate_gnn` / BC oracle already uses the Python builder, so the
natural first implementation before C1 lands **silently rides the 14 ms path** — nothing in the
contract asserts the sample-path builder is native. This is the throughput analog of the F1-class
silent fallback.

**FIX.** State the interim explicitly: until the C1 Rust builder lands, option-c rebuild-at-sample is
NOT viable (26× cost) — either gate run4 launch on C1, or hard-FAIL/loud-WARN if the HEXG sampler
resolves to the Python builder. Add a bench-gate assertion (4.3) that the sample-path builder is the
native one. Do not let 0.14 s/step read as an unconditional floor.

---

## Claim-2 audit spot-check — file:line citations VERIFIED ACCURATE

Read the highest-blast-radius rows against source; all match:

- **Buffer pool SILENT-CORRUPT (node 3, `inference_bridge.rs:257-269`):** `get_feature_buffer` returns
  `vec![0.0f32; self.feature_len]` (`:257-261`), `feature_len` derived from `spec.state_stride()`
  (`:297`). Fixed-width pool with no error on a graph encode — **accurate.**
- **Replay resize stride rotation (node 7, `storage.rs:54-102`):** `state_stride =
  self.encoding.state_stride()` (`:54`), `rotate_left(self.head * state_stride)` (`:61-62`),
  `resize(new_capacity * state_stride, 0u16)` (`:87`); `states: Vec<u16>` — **accurate.**
- **Replay sample stride scatter (node 8, `sample.rs:324-328`):** `states[idx*state_stride..
  (idx+1)*state_stride]` — **accurate.**
- **Symmetry coord-scatter (node 9, `sample.rs:222-256`):** `apply_sym` → `apply_symmetry_state` /
  `apply_chain_symmetry` + 361-cell scatter loop (`:242-256`) — **accurate.**
- **`record_position` dense feat (node 6, `worker_loop/inner.rs:1399-1400`):** `fn record_position`
  confirmed at `:1380`; `feat = vec![0.0f32; kept_planes.len()*n_cells]` in the K-cluster loop at
  `:1398-1400` — **accurate.** (The doc correctly cites `record_position` at `worker_loop/inner.rs`,
  and `aggregate_policy` separately at `records.rs:41` — no mis-location despite the brief's compressed
  phrasing.)
- **Construction family (nodes 11a-c):** `orchestrator.py:677 model = HexTacToeNet(`,
  `lifecycle.py:66 inf_model` + `:172 eval_model`, `anchor.py:569 best_model = HexTacToeNet(` — all
  present, single unconditional CNN construction, no `representation` dispatch — **accurate**; the
  SILENT-CORRUPT-on-`representation=graph` characterization holds.

No file:line finding in claim 2.

---

## Verdict: **GAPS-FOUND**

The contract's structural assertion set is sound as far as it goes, but **CONTRACT-SOUND ("zero silent
paths") does not stand**: three distinct payloads corrupt silently under the proposed contract
(F1/F2/F3), all exploiting the same root — the assertions validate index STRUCTURE, never index/attr
GEOMETRY. The headline **F1 augmentation-seam desync** is the exact F1/§119 class the program exists to
kill, and it sits on the seam WP-C's whole verdict depends on, yet is neither asserted nor tested.

**Demotions:**
- **BLOCKS-CONTRACT-SOUND:** F1 (augmentation-desync, 7th payload — DEFEATED the assertion set), F2
  (edge_attr within-graph permutation), F3 (`legal_node_gather` → stone/dummy; no per-graph
  stone/legal split on the wire).
- **MUST-FIX-BEFORE-BUILD:** F4 (Part-3 "edge_index UNCHANGED" wrong for shipped option-c; realization-
  (i) byte-parity oracle unsatisfiable; policy co-rotation omitted), F6 (~74k node union is stale
  human-corpus; true ~125k), F7 (option-c 0.14 s/step is Rust-builder-conditional; interim Python path
  = 26× silent trap).
- **NOTE:** F5 (u16-wrap "1450 nodes/bs≈45" fabricated; honest wrap bs≈73 worst-case / 134 mean; i64
  conclusion holds).

**What still stands:** the i64 ruling (F5 conclusion holds), the +6 MB/batch cost (F6 — right edge
count), TORCH-BEATS-ORT hot-path consumer, and — importantly — **WP-C's FEASIBLE-ON-LEGACY-V1 is NOT
broken** (F1/F4 leave augmentation feasible via option-c; they break the doc's *coverage/mechanism
claims*, not the feasibility verdict). The audit's file:line factual claims (claim 2) are accurate.

**Path to CONTRACT-SOUND:** add the four SEMANTIC/geometric assertions (edge_attr↔endpoint-geometry;
`policy_dst_slot`↔canonical action-slot; `legal_node_gather`↔legal-subrange via per-graph `n_stones`;
augmentation round-trip parity of graph AND target), fix F4/F6/F7 text, and land all with the producer
per the doc's own 4.2/4.3 build-order requirement. These are cheap — the geometry is already in the
payload.
