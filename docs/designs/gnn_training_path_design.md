# GNN Training-Data Path Design — WP-5 (C7 + C8) of the GNN-integration program

**Status:** design (not a build order). IMPL follows this doc. Consumes the amended ragged contract
(`docs/designs/gnn_ragged_contract_v1.md`, CONTRACT-SOUND @ `2cd8bb7`), the seam design
(`docs/designs/gnn_inference_seam_design.md` + addendum, SEAM-SOUND, option-(b) no-drop), the run4
launch design (`docs/designs/run4_gnn_design.md`, INIT=prefit-40k), the scope doc §C7/§C8 + red-team
(`docs/designs/gnn_integration_scope.md`), and the landed WP reports WP-1 (`hexo-graph` builder
PARITY-EXACT + BUILD-HOT 0.93 ms/pos), WP-3 steps-6/7 (live inference seam), WP-4 (build_net + loader
family). Source-verified against `engine/src/replay_buffer/**`, `game_runner/records.rs`,
`worker_loop/inner.rs`, `hexo_rl/training/{trainer,batch_assembly,recency_buffer,checkpoints,
trainer_ckpt_load}.py`, `hexo_rl/selfplay/{pool,graph_collate}.py`, `hexo_rl/model/gnn_net.py`.

**Question this answers:** how a self-play GNN position becomes a training gradient, end to end —
recording → HEXG replay ring → rebuild-at-sample → collate → GnnNet losses → checkpoint — such that
no dense-plane assumption leaks in and no F1 silent-corruption surface opens on the write side of the
loop (the C8 half of the two-sided ragged-payload risk).

**Standing orders bind:** build the payload ONCE per rebuild (§S186, never a search-time delta); one
resolver per knob (`collate_graph_batch` is the single wire reader — the trainer imports it, does not
fork it); NO value distillation (INV-D1 — every value target is the game's own outcome); benches on
the REAL WP-A self-play distribution (mean 490/2932); hard refuse-to-train if `builder_impl != 1`
(F7). **WP-5 training is gated on C1 (native builder, DONE) — there is no legitimate interim
Python-builder training configuration** (contract F7; the 26× / 3.58 s-per-step trap).

---

## 0. Ruling summary (one table)

| # | Decision | Ruling | Basis / falsifier |
|---|---|---|---|
| 1 | Buffer strategy | **option-(c) store-positions-rebuild-at-sample**, self-contained fixed-max-slot compact record | contract §2.6; reuses HEXB stride-ring mechanics verbatim (§2) |
| 2 | Record layout | stones `[(q,r,player)]` + sparse coord-keyed visit target + outcome placeholder + per-game scalars; **NO dense planes, NO aux** | §1; ~2 KB/record fixed → ~1 GB @ 500k |
| 3 | Game-end path | reuse `finalize_game` outcome + `value_valid` (§178 split — representation-agnostic); **DROP** `reproject_game_end_row` (ownership+winning_line aux), chain, ply-index aux | §1; GnnNet has only policy + dist65 value heads |
| 4 | Persist | **new HEXG v1 format** (own magic `0x48455847`, own version); dense HEXB untouched | §3; stray graph file → dense loader LOUD-FAILs on magic/version |
| 5 | Rebuild cost | **0.24 s/step serial (6.6% of the 3.6 s/step floor budget); batch-parallel ≤0.08 s/step** | §4; NOT floor-binding (floor is self-play-inference-bound) |
| 6 | D6 aug | **coord + visit-key pre-rotation before rebuild, single-call graph+target emission** (realization (ii)); uniform per-sample draw (matches run2, not 12-fold enumeration) | §5; ADV-7 TRUE reproduction lands here |
| 7 | Policy target | ragged per-legal-node CE over the FULL legal set incl. off-window (no-drop mirror of option (b)); `records.rs:62` off-window skip NOT inherited | §6 |
| 8 | Value target | dist65 (`binned_value_loss`, E1 primitive) on outcome z; §178 `ply_cap_value`/`draw_reward` + `value_valid` mask apply UNCHANGED to graph rows | §6 |
| 9 | Corpus branch | replay-stones-and-rebuild from the encoding-free move-list corpus (`build_axis_graph` native); **build the export in WP-5** (cannot defer past a DS kill), defer only the mixing-batch wiring | §7 |
| 10 | OQ-3 | **WP0.4 LANDED** (`1d4a206`) — sha-pinned canonical corpus resolver, but for the DENSE `.npz`; gives the seam to hang the graph branch on, does NOT remove the HEXG re-export need | §7.3 |
| 11 | BC warm-start | `load_representation_policy_from_bc` (WP-2, EXISTS in `gnn_net.py`) wired into a new bootstrap seam; value head fresh (E1 REVIVE) | §8 |

**pd: ~15-26** (WP-5 = C7 3-6 + C8 10-20 envelope; C7 pulls low because the WP-2 transfer fn already
exists). Bench-gated commit list + failure budget: §11.

---

## 1. Recording — what the worker records per graph position

### 1.1 The dense path (what it does today, source-verified)

`record_position` (`inner.rs:1469`) loops the **K cluster views** and per cluster writes a dense
`feat = vec![0.0; kept_planes·n_cells]` + separately-encoded `chain = vec![0.0; 6·n_cells]` + a
per-cluster-window-projected `projected_policy` (`aggregate_policy_to_local{,_ls}`), pushing a
`RecordTuple = (feat, chain, projected_policy, player, cq, cr, is_full_search, ply_index)`
(`inner.rs:122`). At game end `finalize_game` (`inner.rs:1541`) assigns each row an `outcome`
(§178: winner → ±1; winner=None → `ply_cap_value` if `terminal_reason==2` else `draw_reward`), a
`value_valid` mask (`u8::from(terminal_reason != 2)`), and reprojects `ownership + winning_line` aux
per row (`reproject_game_end_row`, `inner.rs:1620`), rotating all four under the row's `sym_idx`.
Result: `WorkerResultRow = (feat, chain, pol, outcome, plies, aux_u8, is_full_search, ply_index,
value_valid)` (`mod.rs:44`), drained by `pool.py::_run_stats_loop` → `push_many`.

Audit class (contract node 6): **SILENT-CORRUPT** — a whole-board graph has no cluster/plane meaning;
the unchanged K-loop records garbage.

### 1.2 The graph record — ONE compact position row, no K-loop, no planes

`record_position_graph` (new sibling, dispatched at the worker boundary on
`spec.representation == "graph"`, hoisted like the inference seam per WP-3 step-6). Per **whole-board**
position (NOT per cluster — the GNN is whole-board, contract §C2 / WP-B node 6) it records ONE
`GraphRecordTuple`:

| field | type | source | note |
|---|---|---|---|
| `stones` | `Vec<(i16,i16,i8)>` | `board.stones()` (already native, WP-3 `build_leaf_graph`) | sorted; the rebuild input |
| `visit_target` | `Vec<(i16,i16,f32)>` | the leaf's `LegalSetPolicy` (dense-in-window + coord overflow) → coord→visit map | **sparse** (only visited children); the no-drop policy target (§6) |
| `current_player` | `i8` | `board.current_player_pm1()` | rebuild `BuildParams` |
| `moves_remaining` | `u8` | `board.moves_remaining_u8()` | rebuild `BuildParams` |
| `ply_index` | `u16` | `board.ply` | weight/diagnostics only (no ply-aux head) |
| `is_full_search` | `bool` | move flag | policy-loss gate (kept) |
| `outcome` | `f32` | **placeholder** → filled at `finalize_game` | §178 path, unchanged |
| `value_valid` | `u8` | **placeholder** → filled at `finalize_game` | draw-mask, unchanged |
| `game_length` | `u16` | `finalize_game` | sampling weight (kept) |

The visit target is captured from the SAME `LegalSetPolicy` the graph MCTS already assembled
(WP-3 `assemble_ls_from_gnn_probs`) — dense-362 in-window slots keyed back to coords + `overflow`
coords, giving one coord→visit-prob map over the **full** legal set. **No `aggregate_policy_to_local`
call** — that fn is the per-cluster dense-window projection that DROPS off-window (`records.rs:62`,
the skip §6 refuses to inherit).

### 1.3 Interaction with game-end reprojection — what DROPS

`finalize_game` splits cleanly into a **representation-agnostic per-game outcome path** (KEEP) and a
**grid-specific aux path** (DROP):

- **KEEP verbatim (per-game, no plane math):** the §178 `outcome` computation (winner/`ply_cap_value`/
  `draw_reward` by `terminal_reason`) and the `value_valid = terminal_reason != 2` draw-mask. These
  read `winner`/`plies`/`terminal_reason` only — no cell geometry. The graph `finalize_game_graph`
  assigns them into the compact record's placeholder fields identically. **INV26 / §178 outcome split
  transfers unchanged to graph rows.** The `ply_cap_value`/`draw_reward` machinery is a value-target
  lever, not a representation lever — it stays live.
- **DROP (grid-specific, no consuming head on GnnNet):** `reproject_game_end_row` (ownership +
  winning_line aux) and its `rotate_aux_inplace`; `chain_planes` (Q13 chain — a CNN INPUT the GNN
  encodes natively via node/edge features); the `position_indices`-as-ply-aux target. `GnnNet` ships
  ONLY `policy_head` + `GnnDist65ValueHead` (`gnn_net.py:153-154`) — there is no ownership/
  winning_line/chain/ply head to feed. **Confirmed:** the GNN value is a dist65 scalar; no aux.

The seam design already stated this (`gnn_inference_seam_design.md` §7: "NO `reproject_game_end_row`
— the GNN value is scalar dist65, no ownership/winning-line aux head"); this doc pins the write-side
consequence.

---

## 2. HEXG replay ring — storage, mechanics, sizing

### 2.1 Record layout choice + bytes/record (the load-bearing sizing decision)

Contract §2.6 recommends option-(c) but leaves the **ring-overwrite of variable-length records**
unspecified. Two realizations:

- **(i) True CSR** — flat `stones` / `visits` Vecs + per-record offset ptrs. Memory-optimal
  (~1.1-1.5 KB/typical record → ~0.6 GB @ 500k) but the ring overwrite of a variable-length slot
  needs compaction/free-list logic — real surgery on the hottest F1-risk module.
- **(ii) Fixed-max-slot compact record (RULED for v1)** — each record padded to `MAX_STONES` +
  `MAX_VISITS`, with per-record `n_stones`/`n_visits` counts. **This makes the HEXG ring a byte-for-
  byte parallel of the proven HEXB stride-ring** (`storage.rs::resize_impl` `rotate_left(head·stride)`
  + `resize(cap·stride)`; ring overwrite = `head = (head+1) % cap`). Minimal, bench-safe surgery on
  the module the two-sided risk lives in.

**Ruling: v1 = (ii) fixed-max-slot.** Storage is not a bottleneck (1 GB on the 9900X vs option-(a)'s
36 GB materialization); byte-for-byte reuse of the audited stride-ring is worth ~2× memory on a
non-scarce resource. Keep (i) specified as the memory-optimal fallback if a future regime is RAM-bound.

**bytes/record (fixed slot):**
- `MAX_STONES` = config-derived from the ply cap (a late position holds ≈ one stone per ply;
  size `MAX_STONES = max_moves + headroom`, example **256**) × 5 B `(i16 q, i16 r, i8 player)` = 1280 B
- `MAX_VISITS` = Gumbel `m` + safety (visits are sparse; the deploy regime is `m=16`, size **128**) ×
  8 B `(i16 q, i16 r, f32 prob)` = 1024 B
- scalars (outcome f32, current_player i8, moves_remaining u8, ply_index u16, value_valid u8,
  is_full_search u8, game_length u16, game_id i64, weight u16, n_stones u16, n_visits u16) ≈ 24 B

→ **≈ 2.3 KB/record fixed → ~1.15 GB @ 500k capacity** (matches the contract's ~2 KB estimate; the
biased 0.539-proxy line understated it, the honest full-distribution figure lands here). Mean live
content is ~1.1 KB (mean 76 stones / ~40 visited moves), so a CSR would reach ~0.55 GB — the fallback
lever if RAM binds.

### 2.2 Ring mechanics — separate module, HEXB untouched

New module `engine/src/replay_buffer/hexg/` (mod/storage/push/sample/persist) — a **parallel ring
beside the dense one** (contract row 7: dense ring untouched). It holds fixed-stride Vecs
(`stones: Vec<i16>` flat `[cap·MAX_STONES·2]`, `stone_players: Vec<i8>`, `visit_coords: Vec<i16>`,
`visit_probs: Vec<f32>`, + the per-record scalar Vecs), and reuses the HEXB pattern verbatim:
`resize_impl` rotate_left+resize by the fixed strides; ring overwrite by `head`; the weight-bucket
histogram + `WeightSchedule` (`sym_tables::WeightSchedule`) lift over unchanged (sampling weight is
game-length-driven, representation-agnostic). `size`/`capacity`/`get_buffer_stats` getters mirror
`ReplayBuffer` so `pool.py` reads them representation-blind.

**Not extended in place:** the dense HEXB stride math (`state_stride()` etc.) has NO graph meaning; a
graph record has no `state_stride`. A stray graph buffer handed to the dense loader hits the existing
magic/version LOUD-FAIL (§3), never a silent mis-parse.

### 2.3 Capacity sizing vs run config

`configs/training.yaml` capacity schedule = 250k → 500k → 1M positions; `run3_dist65.yaml`
(run4 template) `min_buffer_size: 25000`. At 2.3 KB/record the graph ring is 0.58 / 1.15 / 2.3 GB
across the schedule — all fit the 9900X. **No capacity change needed;** the graph buffer inherits the
same `capacity_schedule` knob.

---

## 3. Persist — HEXG v1 format

New format, parallel to HEXB v9 (`persist/mod.rs` magic `0x48455842`, version 9, versions 7-9
accepted else "not supported" LOUD-FAIL at `load.rs:115`):

- **magic `0x48455847`** ("HEXG"), **version `1`**. Header stamps `encoding_name` ("gnn_axis_v1"),
  `contract_version` (1), `MAX_STONES`/`MAX_VISITS` (the slot geometry — a load with different slot
  sizing is rejected, the HEXG analog of HEXB's stride-sig check).
- `save_to_path` / `load_from_path` mirror HEXB: write the fixed-stride record blocks + scalar Vecs;
  version-reject any `version != 1` LOUD (the HEXB `load.rs:115` pattern).
- **Cross-format safety (both directions):** a HEXG file handed to `ReplayBuffer::load_from_path`
  fails the HEXB magic check (`load.rs:48` "invalid magic: expected 0x48455842"); a HEXB file handed
  to `HexgBuffer::load` fails the HEXG magic check. Neither silently mis-parses — the exact
  requirement of contract §"HEXB persist ruling".
- `run3_dist65.yaml:178` `buffer_persist_path` → a `.hexg` sibling for graph runs (resume-capable per
  run4 HANDOFF-STOP).

---

## 4. Sample path — rebuild-at-sample via the native builder

### 4.1 The flow

`HexgBuffer::sample_graph_batch(batch_size, augment) -> GraphWirePayload` (Rust, single PyO3 return
of the block-diagonal wire arrays, mirroring `next_graph_batch` from the inference seam):

1. weighted-sample `batch_size` record indices (the HEXB weighted sampler lifts over — weights are
   game-length, representation-agnostic).
2. per sampled record: draw `sym_idx ∈ 0..12` from the buffer RNG (aug, §5); **coord-rotate stones +
   visit-keys by `sym_idx`**; call `hexo_graph::build_axis_graph(&rotated_stones, &params)` — the
   C1 native builder, `builder_impl = 1` — emitting the graph wire slice (node_feat, edge_index,
   edge_attr, node_coords, legal_node_gather, policy_dst_slot, n_stones, window_center, …) AND, in the
   SAME call, align the rotated visit-keys to the built legal nodes → the per-legal-node policy target
   row + the dist65 outcome. **One call emits graph + target together (F1 single-source).**
3. block-diagonal-fuse the `batch_size` per-graph slices into the union wire arrays (the Rust twin of
   `_collate_gnn`), stamping `contract_version=1`, `builder_impl=1`, offsets, checksums.

Python side: `train_step` graph branch calls `collate_graph_batch(wire, expected_version=1,
allow_oracle_builder=False)` — **the SAME single resolver** the inference seam imports
(`graph_collate.py`, WP-3), now on its third consumer (self-play, eval, trainer — contract §2.3 named
this). **Trainer mode runs the FULL 18-assertion set every batch** (structural 13 + semantic/geometric
4 + F7 handshake; ~10-20 ms vectorized, small vs the ~0.24 s/step rebuild — the seam design §6.1
pinned trainer=full-always vs hot-path=canary). Output: `GraphBatch` torch tensors + the aligned
per-legal-node policy target + the dist65 outcome/value_valid, handed to `GnnNet.forward_batch`.

### 4.2 The F7 assert — where "refuse-to-train" lives

`collate_graph_batch` step 2 (`graph_collate.py:284`) already asserts `builder_impl == 1` unless
`HEXO_ALLOW_ORACLE_BUILDER=1` (`NonNativeSampleBuilder`). The trainer NEVER sets that env var (only
parity tests/CI do), so **the refuse-to-train gate is the collate call itself, on the trainer's own
sample path** — a Python-builder record can never enter a gradient step silently. The Rust sampler
stamps `builder_impl=1` because it calls `hexo_graph::build_axis_graph` (the only native builder); a
sampler wired to the Python `build_axis_graph_raw` oracle would stamp 2 and die at the resolver. This
is the contract F7 handshake landing on the C8 side of the loop.

### 4.3 Rebuild-at-sample steps/hr impact (the headline number)

- WP-1 measured **0.93 ms/pos** (native builder, warm, full self-play distribution, always-on
  `verify_contract`). At training `batch_size: 256` (`configs/training.yaml:45`) → **0.24 s/step
  SERIAL** rebuild.
- run4 floor is **≥1.0k steps/hr = 3.6 s/step** (`run4_gnn_design.md` §4.1). 0.24 s = **6.6% of the
  step budget**. The rebuild does NOT lower the floor: the floor is **self-play-inference-bound**
  (WP-A: 0.9-1.25k steps/hr from per-leaf build+forward on the self-play worker path), and the
  train-step (sample+rebuild+forward+backward) overlaps self-play generation asynchronously. A
  probe-284k forward+backward on batch 256 is ~50-100 ms; +0.24 s rebuild = ~0.3 s/step ≪ 3.6 s → the
  train step is not the binding constraint.
- **Does it parallelize?** Yes, at the BATCH level: the sampler builds the 256 graphs across cores via
  rayon (each per-position build stays single-threaded — WP-1: "the caller parallelizes over leaves;
  rayon stays out of the core path"). Ideal ~20 ms on the 9900X's ~12 physical cores; realistic
  ~40-80 ms with self-play CPU contention. **Design the seam with a BOUNDED rayon pool** (sized to
  leave cores for the self-play workers) so the trainer rebuild does not starve self-play inference-
  feeding — the one perf coupling that could indirectly depress steps/hr. Even the serial 0.24 s
  clears the floor, so batch-parallelism is an optimization, not a v1 requirement (default: serial
  for v1 correctness; parallel as the BUILD-HOT follow-on if a 5080-rider measurement shows train-step
  pressure).

**Headline: rebuild-at-sample ≈ 0.24 s/step serial (6.6% of the 3.6 s/step floor budget), ≤0.08 s
batch-parallel; NOT floor-binding.**

---

## 5. D6 augmentation — realization summary

### 5.1 The mechanism (coord + visit-key pre-rotation, realization (ii))

Contract Part 3 realization (ii): the aug is FREE under option-(c). In the §4.1 sampler, per sampled
record:

- **Draw** `sym_idx` uniformly from `0..12` (the 12 D6 elements, `sym_tables.rs:N_SYMS=12`). **Uniform
  per-sample draw** — matches run2's dense `augment` semantics (one random sym per sampled row, NOT
  12-fold enumeration that would 12× the batch). 12× data-efficiency comes from diversity across
  epochs, exactly as run2 (`sym_tables.rs` per-row scatter). Enumeration is rejected: it inflates the
  effective batch and diverges from the run2-proven regime.
- **Rotate stones AND visit-map KEYS** by `sym_idx` (the axial-coord transform). Rotating stones alone
  desyncs the label (F1/F4) — the visit-target coords co-rotate so the policy target follows each cell
  to its new location.
- **Rebuild** via the native builder on the rotated stones → the builder natively emits the correctly
  re-indexed graph (rotated-coord sort → new `edge_index` integers, F4), correct axis one-hots +
  signed distances, the recomputed `policy_dst_slot`, and — matching rotated visit-keys to the built
  legal nodes by coord — the aligned per-legal-node target. **`edge_index` is NEVER cached across
  augmentations** (F4 hard rule: the coord sort re-indexes under rotation).

**The one new primitive:** a shared axial-coord rotation `rotate_axial(q, r, sym_idx) -> (q', r')`.
The CNN's `sym_tables` builds its `(src_cell, dst_cell)` scatter pairs from exactly this coord
transform; **lift it into one function** both the CNN cell-scatter and the graph coord/visit-key
rotation call, so "D6 element `s`" means the identical geometry on both paths (single source of the 12
elements). This is a small extraction, NOT the option-(a) `axis_perm`+signed-map machinery (that is
only needed for an in-place built-graph transform, not built for v1). Zero graph-symmetry module.

### 5.2 ADV-7 TRUE reproduction + the AugRoundTripMismatch check (lands HERE, per WP-3 review S1)

WP-3 deferred the TRUE rotated-graph/rotated-target ADV-7 to WP-5 ("ADV-7 true rotated-graph/unrotated-
target reproduction lands with WP-5 aug"). The trainer aug path is where a graph/target desync can
actually occur, so the enforcement lands here:

- **`AugRoundTripMismatch` (test-time, every ADV/CI run):** rebuild-at-`s` then rebuild-at-`s⁻¹∘s`
  must recover the stored record byte-identically — graph arrays AND the aligned target row AND
  `policy_dst_slot`. Because the single rebuild call emits graph + target together, a desync requires
  the same bug twice; the round-trip test is the enforcement that construction stays single-sourced.
- **Runtime canary (each sampled trainer batch, cheap):** the policy target's argmax cell must map to
  a legal node whose `policy_dst_slot` equals the canonical slot of that cell's ROTATED coordinate
  (the `ScatterSlotCanonicalMismatch` slot-leg + the argmax-canary form). `collate_graph_batch`
  already accepts `target_argmax_cells` for exactly this (`graph_collate.py:261`).
- **The ADV-7 payload proper** (graph rotated by `s`, target left unrotated / rotated by `s'≠s`; D6
  preserves node/edge counts so all 13 structural checks pass) fires `ScatterSlotCanonicalMismatch`
  (slot leg) + `AugRoundTripMismatch` (target leg). This is the write-side twin of the inference
  slot-leg proxy WP-3 shipped.

### 5.3 The data-efficiency stake

Dropping aug = ~12× fewer effective distinct samples (contract Part 3), compounding onto the ~4-5×
throughput penalty. **Aug stays IN** (run4 §2 RULE). Realization (ii) makes it free, so there is no
cost tradeoff — only the discipline of the single-call construction + the ADV-7 enforcement above.

---

## 6. Losses / targets

### 6.1 Policy — ragged per-legal-node CE, no-drop (the `records.rs:62` replacement)

- **Target:** the record's coord-keyed visit distribution over the FULL legal set (in- AND
  off-window). At sample, after rebuild, the target is assembled per-legal-node by coord match:
  in-window nodes via `policy_dst_slot`, off-window nodes via the overflow coord — producing a
  `[Lg]`-vector (over the graph's legal nodes, batch-concatenated) summing to 1 per graph. This is
  the **no-drop training-target mirror of inference option (b)** (`LegalSetPolicy` incl. overflow).
- **The `records.rs:62` skip is NOT inherited.** The dense `aggregate_policy{,_to_local}` SKIPS
  off-window legal cells (`usize::MAX` slot) because a dense `[362]` window has no room for them. The
  graph record stores the raw coord→visit map with NO skip; the ragged per-legal-node target carries
  the off-window mass. Inheriting the skip would reproduce the pre-R1 decode handicap the +414
  evidence removed (seam §1.2). **Replacement stated: store the raw LegalSetPolicy visit map at
  record time; align to legal nodes at sample time; never project-and-drop.**
- **Loss:** ragged per-legal-node CE — a new small loss fn `ragged_policy_ce(policy_logits[Lg],
  target[Lg], legal_offsets[B+1], full_search_mask[B])`: per-graph `log_softmax` over its legal-node
  segment (the `segment_softmax` the seam already added, `graph_collate.py`), then `-Σ target·logp`,
  masked by `is_full_search` (quick-search rows contribute value only, exactly as the CNN
  `full_search_mask_t` gate, `trainer.py:617`). This replaces the CNN's dense-362 `compute_policy_loss`
  on the graph branch; the per-class-target-temperature / policy-prune levers (`trainer.py:639-672`)
  are dense-362 ops → **N/A on the graph branch for v1** (state as a known gap; they are OFF by
  default in run4).

### 6.2 Value — dist65 on outcome z, §178 machinery intact

- **Target:** the game's own outcome z (INV-D1 — no teacher/solver/distill), from the §178 path (§1.3).
  `GnnNet.forward_batch` returns `bin_logits (B, 65)` (`gnn_net.py:187`) → **`binned_value_loss(
  bin_logits, outcomes_t, value_mask=value_mask_t)`** — the identical E1 dist65 primitive the CNN uses
  (`trainer.py:749`, `binned_value.py`). `scalar_to_two_hot` on z, CE over 65 bins, masked. Zero new
  value machinery.
- **§178 `ply_cap_value`/`draw_reward` + `value_valid`:** these are per-game outcome-assignment levers
  computed in `finalize_game` from `terminal_reason` (§1.3) — **representation-agnostic, apply to graph
  rows UNCHANGED.** The `value_valid` draw-mask feeds `binned_value_loss(value_mask=...)` exactly as
  the CNN. The graph run declares them in its yaml (F1 preserve-ckpt-baked — DECLARE, don't inherit).
- **OQ-6 (dist65 geometry on the GNN) resolved:** the run3_d1 K-cluster argmin routing was CNN-specific
  (per-cluster → argmin → CE on that cluster). The GNN is WHOLE-BOARD (one graph, one pooled value) →
  **dist loss = plain single-window CE, no argmin routing** — matching the E1 finding
  (`e1-cardone-integration-scoped`: single-window forward → dist loss = plain CE). `GnnDist65ValueHead`
  pools stone-node embeddings → one `(65)` per graph; `binned_value_loss` consumes it directly. dist65
  warm-starts FRESH over the prefit (E1 REVIVE) — §8.

### 6.3 Aux — dropped

No opp-reply / uncertainty / ownership / threat(winning_line) / chain / ply-index heads on `GnnNet`
→ all aux losses (`trainer.py:601-606` weights) are N/A on the graph branch. The graph run must NOT
set their weights > 0 (a positive `ownership_weight` etc. on a graph config is a config error — the
graph `_train_on_graph_batch` asserts them zero / ignores them with a loud log).

---

## 7. Corpus-mix / bootstrap — the fallback branch (design-level)

### 7.1 Launch state: corpus-mix OFF

run4 INIT = prefit-40k, corpus-mix OFF (run4 §1 RULE). At launch the graph corpus path is **dormant**
— the prefit folds the human prior into the INIT (§8), and mixing on top double-counts. The graph run
yaml DECLARES `bot_batch_share: 0` + no corpus-mix weights (F1 preserve-ckpt-baked: declare, don't
inherit).

### 7.2 The fallback needs the path to EXIST — build the export, defer only the mixing wiring

The fresh+mixing fallback (option B) fires on a DS-1/2/3 kill by **≤50k steps** (run4 §1.3). At the
≥1.0k steps/hr floor, 50k steps ≈ **50 GPU-hr ≈ 2 GPU-days** — there is **NO margin to build AND
parity-test a graph-corpus path post-kill without idling the box** (the box-STOPS-if-not-ready rule,
run4 §5, bites here). Therefore:

- **BUILD in WP-5 (do NOT defer): the graph-corpus replay-and-rebuild export.** Iterate the
  encoding-free move-list corpus (games as move sequences, `data/corpus/raw_human` — the BC probe's
  source, `train_bc.py:453`); replay each game move-by-move; at each recorded decision, call
  `hexo_graph::build_axis_graph` on the position's stones + build the policy target from the stored
  visit distribution → push a HEXG record. This **reuses the C1 native builder + the WP-5 HEXG push
  path** (both built for self-play) + the BC probe's replay precedent (`train_bc.py:272` "board is
  replayed once per game") → **~2 pd**, and it doubles as the OQ-4 byte-parity gate (re-exported graph
  vs `build_axis_graph_raw` on `wpa_positions.json`).
- **DEFER (buildable in <1 day once the export exists, dead until fallback fires): the mixing-batch
  wiring** — the `initial_pretrained_weight` 0.8→0.1 decay reading graph corpus batches into the
  train step (`batch_assembly.py` graph branch + the decay schedule plumbing). This is a config/wiring
  layer on top of the export; pre-register it as the post-kill build.

**Pre-registered failure-budget answer:** the export CANNOT reliably be built post-kill inside 2 days;
the mixing wiring CAN. So the export lands in WP-5, the mixing wiring is the deferred fallback tail.

### 7.3 OQ-3 resolution — WP0.4 LANDED (dispatcher FRAME confirmed, scope corrected)

The run4 doc flagged OQ-3 unresolved ("WP0.4 not locatable"). **Searched git log (not just docs/):
WP0.4 DID land** — commit `1d4a206` "feat(corpus): sha-pinned canonical manifest + single-resolver
corpus path for run3 (WP0.4)" + `42e4e90` (fix wave):
- `docs/registers/run3_corpus_manifest.md` — canonical ruling (laptop 8698 manifest canonical;
  sha-pinned).
- `resolvers.py::_CORPUS_SHA_PINS` + `resolve_corpus_sha_pin()` + the single `resolve_corpus_path`;
  `batch_assembly.load_pretrained_buffer` hard-fails on sha mismatch keyed on the ACTIVE ENCODING.
- pins `data/bootstrap_corpus_v6_live2_ls.npz` (a **DENSE `.npz`**).

**Honest C8 dependency:** WP0.4 is a sha-pinned canonical corpus RESOLVER for the dense `.npz` — it
gives the single-resolver SEAM to hang the graph branch on (add a `gnn_axis_v1` entry to
`_CORPUS_SHA_PINS` / `resolve_corpus_path` pointing at the re-exported graph corpus or the encoding-
free JSONL), but it does NOT make a graph-consumable corpus exist. The GNN still needs the §7.2
replay-and-rebuild export. **OQ-3 status: RESOLVED — WP0.4 present; the graph re-export is required
regardless of the manifest and is scoped in §7.2, hung on the WP0.4 resolver.**

---

## 8. Trainer integration

### 8.1 Forward/loss branch

`train_step` dispatches to `_train_on_graph_batch` when `spec.representation == "graph"` (or the model
is a `GnnNet` / `value_head_type == "dist65"` + graph — the cleanest key is the resolved spec's
representation, single-sourced with `build_net`). The graph branch: `sample_graph_batch` →
`collate_graph_batch` (FULL asserts) → `GnnNet.forward_batch(x, edge_index, edge_attr, legal_mask,
stone_mask, node_offsets)` → `(policy_logits[Lg], value[B], bin_logits[B,65])` → `ragged_policy_ce`
(§6.1) + `binned_value_loss` (§6.2) → backward → optimizer step. The dense `_train_on_batch` is
untouched (byte-identical for grid).

### 8.2 Optimizer / scheduler / compile — unchanged

`build_param_groups(self.model, weight_decay)` (`trainer.py:334`) iterates `model.parameters()` →
works over `GnnNet` verbatim; weight-decay grouping is name-based and tolerant. Optimizer + scheduler
**UNCHANGED**. `torch.compile`: the inference seam DROPS the shape-pinned TorchScript trace
(invalid for variable N/E); for TRAINING, `torch.compile(dynamic=True)` is OPTIONAL and **OFF by
default for v1** (variable N/E → recompilation churn; the WP-A forward figures were eager, so the
floor already assumes no compile).

### 8.3 Checkpoint save / load

- **Load side: DONE by WP-4.** `build_net(spec, config)` builds `GnnNet` on `representation=="graph"`;
  `detect_encoding_from_state_dict` graph-detects via `representation.input_proj.weight`;
  `checkpoint_loader._build_gnn_model` + `trainer_ckpt_load` resume graph branch ground-truth hparams
  from tensor shapes (`infer_gnn_hparams_from_state_dict`) + landed-verify. Resume-capable.
- **Save side: DONE by existing machinery.** `save_checkpoint` stamps `encoding_name =
  resolve_from_config(self.config).name` (`trainer.py:1259`) → "gnn_axis_v1" for a graph config
  (WP-3 registry added `representation` to the resolved spec; `resolve_from_config` returns it). The
  metadata `encoding_name` + the state-dict shape marker are sufficient for the load-side detect. EMA
  sidecars + `inference_state_dict` operate over `GnnNet` params unchanged. **No save-side change
  required** — verify only that `resolve_from_config` on a `gnn_axis_v1` config returns the graph spec
  (a one-line WP-5 assertion / smoke).

### 8.4 BC-prefit warm-start wiring (C7 — the remaining loader work)

The transfer fn `load_representation_policy_from_bc` **already EXISTS** in `gnn_net.py:250` (WP-2:
STRICT on `representation.*` / `policy_head.*`, `strict=False` load + `torch.allclose` landed-verify
over all 46 transferred tensors — the F1 guard). It is **NOT wired**. WP-4 correctly refuses a
BC-prefit-only state dict with `assert_full_gnn_checkpoint_or_raise` (`checkpoints.py:154`) — a named
raise, not silent — precisely so WP-5 can route around it.

**Wire it:** a new bootstrap-warm-start seam (`hexo_rl/training/gnn_warmstart.py`, or a branch in the
trainer's fresh-init path) — when a graph run declares `bootstrap`/init = `gnn_bc_040000.pt` and the
loader detects a BC-prefit-only sd (rep+policy, no `value_head.fc2_bins`), build a fresh `GnnNet` via
`build_net` and call `load_representation_policy_from_bc(net, sd["model_state_dict"])` → value head
stays fresh (E1 REVIVE: dist65 warm-starts fine over an absent value head). This is the C7 "GNN
warm-start loader (new)" — **~1-2 pd (the fn exists; wire + config plumbing + smoke).** OQ-5 gate:
the landed-verify MUST fire (the fn raises on a silent drop).

### 8.5 Watchdog / monitoring — what breaks, what reads N/A

- **Reads clean (representation-agnostic):** `pool.py:714/977` `buffer_size = replay_buffer.size`
  (HEXG exposes `.size`); the weight histogram (`get_buffer_stats`, HEXG maintains it);
  `mcts_mean_depth` (`pool.py:621` — interior-PUCT descent depth; the GNN rides the SAME legal-set
  MCTS interior via `expand_and_backup_ls`, WP-3 — the metric is unchanged and NOT a plane concept);
  the stall watchdog + promotion-gate subprocess isolation (run4 §6.3, ride verbatim).
- **BREAKS — needs a graph branch (dense-shape assumptions):**
  - `pool.py::_run_stats_loop` (`:772`) — `collect_data()` 10-tuple of dense arrays + the
    `_feat_len/_in_ch` reshape (`:776,794`) + `push_many`. Graph needs `collect_graph_data()` →
    `push_graph_positions` (skip the `(N,C,H,W)` reshape). This is the C8 drain twin.
  - `recency_buffer.py` — a dense numpy ring (`_ownership`/`_winning_line`/state, `:59`). run4 keeps
    `recency_weight: 0.75` (`training.yaml:114`, not overridden) → load-bearing. **Ruling: for v1,
    absorb recency into the HEXG ring** (sample a `recent_frac` slice from the ring's newest slots
    via `sample_graph_batch(recent_frac=...)`) rather than building a parallel compact graph recency
    ring — smaller surgery, one sampler. Tradeoff: the HEXG "recent" is the newest ring slots, not a
    separately-sized Python ring; acceptable (the dense recent ring is ~newest-50%-of-buffer, which
    the HEXG newest-slots slice approximates). State the alternative (a compact graph recency ring) as
    the fallback if the approximation matters.
- **Reads N/A (dense-only analysis, not on the run path):** `monitoring/value_probe.py`
  `BUFFER_CHANNELS = _V6.n_planes`; `analyze_api.py` `n_planes == in_channels` match — dense corpus/
  buffer analysis tools, not exercised by a graph run (the run4 value-health read is the 234-probe on
  the decoded dist65 value, encoding-agnostic at eval).

---

## 9. Smoke-gate mapping (OQ-7 parts 1 + 3)

The 3-part integration smoke gate (run4 §5 hard precondition, OQ-7). WP-5 enables **part 3** (the
training-path leg); parts 1-2 are landed (WP-1/WP-B contract + adversarial coverage; WP-3 step-7
inference round-trip). Test files this design enables + pass criteria:

| Test (new) | Marker | Asserts (pass criteria) |
|---|---|---|
| `tests/rust` HEXG round-trip (`engine/tests/` or `#[cfg(test)]`) | cargo | HEXG push → save → load byte-identical; resize (linearise+extend) preserves records; a HEXB file → HEXG load LOUD-FAILs on magic, and vice-versa |
| `tests/training/test_gnn_hexg_buffer.py` | unit | push_graph_positions → sample_graph_batch → collate round-trips; `builder_impl==1` on the sampled wire; ADV stray-buffer → dense loader raises |
| `tests/training/test_gnn_aug_roundtrip.py` | unit | `AugRoundTripMismatch`: rebuild-at-`s` then `s⁻¹∘s` byte-identical for graph AND target for all 12 `s`; **ADV-7 TRUE**: a desynced target (graph rot `s`, target rot `s'≠s`) fires `ScatterSlotCanonicalMismatch` + `AugRoundTripMismatch` |
| `tests/training/test_gnn_train_step.py` | `integration` | 3 train steps on a `gnn_bc_040000.pt`-warm-started `GnnNet`: sample HEXG → rebuild (native) → collate (full asserts) → `forward_batch` → `ragged_policy_ce` + `binned_value_loss` → backward; **all losses finite, all grads finite**, step counter advances |
| `tests/training/test_gnn_bc_warmstart.py` | unit | `load_representation_policy_from_bc` lands + landed-verify fires; save full `GnnNet` ckpt → reload via `checkpoint_loader` → `torch.allclose` (the PyO3→train-step→replay→reload ragged parity leg — OQ-5) |

**Part-3 PASS = a graph position round-trips PyO3 → HEXG replay → rebuild-at-sample → train-step
(finite losses) → checkpoint → reload, with the full ragged assertion set green and the ADV-7 aug
enforcement firing.**

---

## 10. Person-days + file-touch list

WP-5 = the C7 (3-6) + C8 (10-20) slice, single-sourced with the contract's already-scoped resolver
(not additive on it). Honest per-component:

| Component | Files (primary) | pd | Bench-gated? |
|---|---|---|---|
| HEXG buffer module — storage/push/sample(rebuild)/resize, fixed-slot ring | `engine/src/replay_buffer/hexg/**` (new) | **4-7** | YES (replay_buffer/**) |
| record_position graph branch + `GraphRecordTuple` + finalize outcome reuse + `collect_graph_data` drain | `game_runner/worker_loop/inner.rs`, `game_runner/records.rs`, `game_runner/mod.rs` | 2-3 | YES (game_runner/**) |
| HEXG v1 persist (magic/version/save/load, cross-format LOUD-FAIL) | `engine/src/replay_buffer/hexg/persist.rs` (new) | 1-2 | no (io) |
| D6 aug — shared `rotate_axial` lift + visit-key rotation + AugRoundTrip/ADV-7 enforcement | `hexg/sample.rs`, shared sym primitive, `graph_collate.py` (target-argmax leg) | 2-3 | partial (sampler) |
| Trainer graph train-step branch + `ragged_policy_ce` loss + pool.py graph drain + recency-in-ring | `training/trainer.py`, `training/losses`, `selfplay/pool.py` | 2-3 | no (python) |
| Graph-corpus replay-and-rebuild export + OQ-4 parity (mixing wiring DEFERRED) | `training/batch_assembly.py` (graph branch), export script | 2 | no |
| BC-prefit warm-start wiring (fn EXISTS) + config + smoke | `training/gnn_warmstart.py` (new) or `trainer_ckpt_load.py` | 1-2 | no |
| run4 graph variant yaml (launch scope — declare encoding/bootstrap/mix-OFF/§178) | `configs/variants/run4_gnn.yaml` (new) | 0.5-1 | no |
| Smoke-gate part-3 tests (5 files) | `tests/training/**`, `engine/tests/` | folded (~2-3) | no |
| **WP-5 total** | | **~15-26 pd** | |

**File-touch list (durable):**
`engine/src/replay_buffer/hexg/{mod,storage,push,sample,persist}.rs` (new module),
`engine/src/game_runner/worker_loop/inner.rs`, `engine/src/game_runner/records.rs`,
`engine/src/game_runner/mod.rs`, a shared axial-coord rotation primitive (lifted from
`replay_buffer/sym_tables.rs`), `hexo_rl/training/trainer.py`, `hexo_rl/training/batch_assembly.py`,
`hexo_rl/selfplay/pool.py`, `hexo_rl/training/recency_buffer.py` (recency-in-ring note or graph ring),
`hexo_rl/training/gnn_warmstart.py` (new), `hexo_rl/selfplay/graph_collate.py` (trainer-mode target
alignment — extend, not fork), a ragged-policy-CE loss fn, `configs/variants/run4_gnn.yaml` (new),
`tests/training/test_gnn_{hexg_buffer,aux_roundtrip,train_step,bc_warmstart}.py` + `engine/tests/`
HEXG round-trip.

### 10.1 Bench-gated commit list (fires the `bench-gate` skill)

Two commits touch bench-gated paths (`replay_buffer/**`, `game_runner/**`, hot paths):

1. **HEXG buffer module** (`replay_buffer/hexg/**`). `make bench` MUST show **no regression on the
   dense CNN 10-metric gate** (HEXG is a parallel module; the dense ring is untouched) + a NEW graph
   sample-rebuild cell measured on the REAL WPA distribution, asserting `builder_impl==1` (F7 — a
   Python-builder bench is both invalid and the trap).
2. **record_position graph branch + collect_graph_data** (`game_runner/**`, `worker_loop`). Dense path
   byte-identical (graph dormant behind `representation=="graph"`, hoisted single branch like WP-3
   step-6); CNN 10-metric no-regress.

The trainer graph train-step, corpus export, warm-start wiring, and yaml are Python/inert/test — not
bench-gated by the skill's path list, but the rebuild-at-sample cell (commit 1) IS the §4.3 floor
measurement.

### 10.2 Pre-registered failure budget (which component doubles)

**Most likely to blow its estimate: the HEXG buffer module (4-7 pd).** It is the largest genuinely-new
Rust cut and carries the one **unmeasured perf coupling** — batch-level rebuild parallelism competing
with self-play worker cores. The risk is not the storage code (fixed-slot ring reuses HEXB verbatim);
it is that a naive rayon rebuild pool **starves self-play inference-feeding**, silently depressing
steps/hr below the 1.0k floor — a regression that only surfaces on the 5080 under real concurrent
load, invisible to a unit test. If the bounded-pool isolation doesn't cleanly separate trainer-rebuild
CPU from self-play CPU, this component doubles (→ 8-14 pd) and drags the 5080-rider floor
re-measurement (OQ-2). **Escalate per red-team order if it doubles mid-WP.** Mitigation: ship v1 with
SERIAL rebuild (0.24 s clears the floor with margin), add batch-parallelism only as a measured
BUILD-HOT follow-on gated on a 5080 train-step-pressure read — decoupling correctness from the perf
coupling.

---

## 11. Verdict

The design closes the C8 write-side of the ragged-payload loop with the same no-SILENT-CORRUPT /
no-SILENT-FALLBACK discipline the inference seam established:

- **Recording:** ONE compact graph-position row (stones + sparse coord-keyed visit target + outcome
  placeholder), NO dense planes, NO aux; the §178 outcome/value_valid path reused verbatim, the
  ownership/winning_line/chain/ply aux DROPPED (no consuming head).
- **HEXG ring:** fixed-max-slot compact record (~2.3 KB → ~1 GB @ 500k) reusing the HEXB stride-ring
  mechanics; a separate `HEXG` persist format (magic `0x48455847`) so a stray graph file LOUD-FAILs
  on the dense loader.
- **Sample:** rebuild-at-sample via the C1 native builder, single-call graph+target emission (F1
  single-source), full 18-assertion collate on every trainer batch, `builder_impl==1` refuse-to-train.
- **Rebuild cost:** 0.24 s/step serial (6.6% of the 3.6 s floor budget), NOT floor-binding.
- **D6 aug:** coord + visit-key pre-rotation, uniform per-sample draw, ADV-7 TRUE reproduction +
  AugRoundTripMismatch enforcement landing here per WP-3 S1.
- **Losses:** ragged per-legal-node no-drop CE (the `records.rs:62` skip replaced) + dist65
  `binned_value_loss` on outcome z; §178 levers intact.
- **Corpus branch:** replay-and-rebuild export BUILT in WP-5 (cannot defer past a ≤50k-step / 2-day DS
  kill), mixing wiring deferred; OQ-3 resolved (WP0.4 landed, dense-`.npz`-scoped, graph re-export
  still required).
- **Trainer:** graph forward/loss branch; optimizer/scheduler unchanged; load-side DONE (WP-4),
  save-side DONE by existing stamping; BC warm-start wiring the one C7 remainder (fn exists).

**No dense-plane assumption survives on the graph training path. The residual risk is the one
unmeasured perf coupling (batch-rebuild vs self-play CPU), mitigated by shipping serial v1.**

**TRAINING-PATH-SOUND — IMPL follows; gated on C1 (DONE), single-sourced on `collate_graph_batch`.**
