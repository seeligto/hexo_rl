# WP-3 steps 6+7 IMPL — the bench-gated hot-path graph seam (C3)

**Agent:** WP-3 step-6+7 IMPL. Worktree `worktree-gnn-integration`. **No git**
(dispatcher commits + runs `make bench`). Consumes the seam design steps 6-7 +
the Step-6 preconditions addendum (S2 MANDATORY), the WP3 steps-1-5 dormant seam
(`871ea41`, REVIEW-PASS), the WP-2 net (`hexo_rl/model/gnn_net.py`), the amended
ragged contract, and the committed `hexo-graph` builder.

**Verdict: STEP 6 DONE (grid byte-identical, dormant-for-grid) · STEP 7 SMOKE PASS.**

The graph seam is LIVE behind `spec.representation == "graph"`; every grid
encoding takes a single never-taken predicted branch and runs the dense path
unchanged.

---

## Step 6 — what wired where

| Layer | File | What landed |
|---|---|---|
| Worker dispatch | `engine/src/game_runner/worker_loop/inner.rs:738` | `if infer.batcher.is_graph() { return infer_and_expand_graph(...) }` at the TOP of `infer_and_expand` — hoisted once per leaf-batch (NOT per-sim), the graph fn is `#[cold]`/`#[inline(never)]`. |
| Worker graph path | `inner.rs:871-926` (`infer_and_expand_graph`) | `select_leaves` → build ONE `AxisGraph` per leaf from the board's stones (§S186, never a search-time delta) via `batcher.build_leaf_graph` → `submit_batch_and_wait_graph_rust` → `expand_and_backup_ls_at`. Whole-board = one value/leaf (no K-cluster min-pool). v1 inference-time symmetry DEFAULT-OFF (`infer.sym_idx` not applied — coord pre-rotate is WP-5). |
| S2 frame threading | `engine/src/mcts/backup.rs:478-511` (`expand_and_backup_ls_at`) + `:361-441` (`expand_and_backup_single_ls_framed`) | see S2 evidence below. |
| Batcher accessors | `engine/src/inference_bridge.rs:499,506,516` | `is_graph()`, `graph_trunk_size()`, `build_leaf_graph()` (routes the WP-1 seam guards; `None` on a guard trip → worker skips the batch like a dense inference failure). |
| Runner graph batcher | `engine/src/game_runner/mod.rs:421-434` | `SelfPlayRunner::new` constructs a `representation="graph"` batcher (parallel graph queue, no dense pool) for a graph spec; every grid spec keeps the `None` encoding_spec → `Grid` batcher, byte-identical. |
| Python resolver output | `hexo_rl/selfplay/graph_collate.py` | added `segment_softmax(logits, legal_offsets)` (vectorized stable per-graph softmax, `scatter_reduce_`/`scatter_add_`) + `stone_mask_from_batch(batch)` (value-head pool subset). |
| InferenceServer graph branch | `hexo_rl/selfplay/inference_server.py` | `__init__` `_is_graph` split (skips CNN trace/H2D/`(C,H,W)` shape — meaningless at `n_planes=0`); `run()` dispatches to `_run_graph_loop`: `next_graph_batch` → `collate_graph_batch` (canary semantic) → `GnnNet.forward_batch` (fp16 autocast on CUDA) → `segment_softmax` → `submit_graph_inference_results`; die-loud via `submit_graph_inference_failure`. CNN path unchanged. |
| Smoke driver | `inference_bridge.rs:1171` (`submit_graphs_and_wait`) | Python-visible blocking driver: builds graphs from positions, submits through the SAME `submit_batch_and_wait_graph_rust` + `assemble_ls_from_gnn_probs` path the worker rides, releases the GIL, returns per-leaf `(dense[362], overflow[(q,r)→p], value)`. |

**Review NITs folded in:** N1 (`inference_bridge.rs:609-612` `.expect()` for the
graph geometry, dead `6`/`1` literals removed) · N4 (`records.rs` `assemble_ls_from_gnn_probs`
now returns `Result<LegalSetPolicy, String>` — graceful die-loud instead of a
release-`panic=abort` `assert!`) · N5 (`inference_bridge.rs` `submit_batch_and_wait_graph_rust`
pre-validates the handshake before touching the queue; `submit_graph_inference_results`
routes every mid-loop error through `fail_remaining_graph_ids` so no waiter is orphaned).

The GNN `PolicyHead` is per-legal-node and the assemble is option (b) no-drop:
off-window legal nodes (`policy_dst_slot == -1`) land in `LegalSetPolicy.overflow`
by coord, reproducing the +414 decode regime (never the dense-362 drop).

---

## S2 threading evidence (MANDATORY — F1 coord/slot class)

The assembled `LegalSetPolicy.dense` slots are baked against the **builder's**
per-leaf `window_center` (via `policy_dst_slot`). Step 6 threads that exact
centre into the expand read frame rather than relying on a coincident
`board.window_center()` re-derivation:

- **Capture at build:** `inner.rs:894` — `centers.push(g.window_center)` (the
  builder's own centre, per leaf, in `select_leaves`/`pending` order).
- **Thread into expand:** `inner.rs:925` —
  `tree.expand_and_backup_ls_at(&aggregated_ls, &aggregated_values, &centers, agg_trunk_sz)`.
- **Consume with the baked frame:** `backup.rs:495,510` — `let (cq, cr) = centers[i];`
  → `expand_and_backup_single_ls_framed(*leaf_idx, board, ls, value, cq, cr, trunk_sz)`,
  which reads `ls.dense` via `pick_topk_children_ls(..., cq, cr, ..., trunk_sz, ...)`
  — the SAME `window_flat_idx` frame the builder used, not `board.window_center()`.
- **Invariant guard:** `backup.rs:499` + `inner.rs:920` `debug_assert_eq!`
  (builder centre == `Board::window_center()`, and `agg_trunk_sz` ==
  `batcher.graph_trunk_size`) — pins the coincidence the TT-hit re-read path
  (`expand_and_backup_single_ls`, board frame) also relies on.
- **Refactor safety:** `expand_and_backup_single_ls` now delegates to
  `expand_and_backup_single_ls_framed` with `board.window_center()` /
  `board.cluster_window_size()` — byte-identical for the CNN legal-set callers.

Trunk single-source: the builder trunk (`batcher.graph_trunk_size` =
`spec.trunk_size` = 19) IS `agg_trunk_sz` (WorkerGeometry `spec.trunk_size`);
threaded from one field, asserted equal.

A dedicated Rust matched-centre round-trip test
(`records.rs::assemble_read_back_at_builder_center_matches_and_wrong_center_misreads`)
proves reading back at the builder centre recovers every prob AND a wrong centre
misreads in-window slots — the exact F1 class S2 closes.

---

## Grid-dormancy argument (bench gate should pass)

The dense/CNN path is byte-identical; the graph code is dormant for every grid
encoding:

1. **Single hoisted branch.** `infer_and_expand` gains ONE `if
   infer.batcher.is_graph()` at fn entry. `is_graph()` is `matches!(self.representation,
   Graph)`; the 11 grid encodings default `Representation::Grid` → the branch is
   never taken and the instruction stream after it is unchanged. The graph fn is
   `#[cold]`/`#[inline(never)]` so it never bloats the inlined dense hot path.
2. **Parallel graph queue, untouched dense state.** The 4 dense hot methods
   (`submit_batch_and_wait_rust`, `next_inference_batch`, `submit_inference_results`,
   `pop_batch_blocking`) + the feature-buffer pool are NOT in the diff. New
   accessors (`is_graph`/`graph_trunk_size`/`build_leaf_graph`) + `submit_graphs_and_wait`
   + `fail_remaining_graph_ids` are pure additions reachable only from graph
   batchers / the smoke.
3. **Runner batcher gated on the spec.** `mod.rs` constructs the graph batcher
   ONLY under `s.representation.is_graph()`; every grid runner keeps the
   pre-WP3 `InferenceBatcher::new(None, Some(feature_len), Some(policy_len), …)`
   call, unchanged.
4. **backup.rs is pure addition.** `expand_and_backup_ls_at` +
   `expand_and_backup_single_ls_framed` are new; the existing
   `expand_and_backup_single_ls` delegates to the framed helper with the same
   board-derived values (byte-identical for the v6_live2_ls legal-set callers).
5. **InferenceServer `__init__`.** The CNN setup (trace, H2D staging, `(C,H,W)`
   shape) is wrapped in the `else` (not-graph) arm verbatim; `test_inference_server.py`
   + `test_selfplay_encoding_aware.py` (41 tests) pass unchanged.

---

## Step 7 — OQ-7 step-0 smoke (`tests/selfplay/test_gnn_seam_smoke.py`, `-m integration`)

Runs a few dozen leaf evaluations through the LIVE seam on the banked
`checkpoints/probes/gnn_bc/gnn_bc_040000.pt` (46 rep/policy tensors loaded +
landed-verified). Device = **CPU** (fp32 → strict 1e-5 parity + the design's
mandated CPU-only fallback; the identical seam runs on CUDA under fp16 autocast).
Drives the production `SelfPlayRunner`-built graph batcher +
`InferenceServer(_run_graph_loop)`.

**Result: PASS.** Numbers (CPU fp32):

| Metric | Value |
|---|---|
| positions evaluated | **26** (20 spread two-cluster + 6 compact) |
| positions with ≥1 off-window overflow node | **20 / 26** |
| total off-window overflow nodes (option-b no-drop) | **4603** |
| per-position round-trip parity vs `forward_single` | **max|Δ| = 1.3e-8** (< 1e-5 gate) |
| finite priors / distribution (sum≈1) | all 26 |
| legal-only (overflow keys ⊆ legal; argmax move legal) | all 26 |

Parity oracle = `build_axis_graph_raw` (Python, `prune_empty_edges=False` — matches
the native `build_axis_graph`, WP-1 PARITY-EXACT) → `GnnNet.forward_single` → softmax
by legal coord; compared to the LS the engine assembled, read back by coord
(`dense[slot]` in-window / `overflow[coord]` off-window). CUDA fp16 leg gives
max|Δ| ≈ 4.4e-5 (autocast rounding, not a seam defect).

"Moves get played": the argmax over the engine-received priors (restricted to the
legal set) is asserted legal per position — the move MCTS expands/plays from the LS.
Full recorded self-play is intentionally NOT run (buffer/`record_position` graph
branch = C8/WP-5, out of scope; design step 7 writes nothing to the buffer).

---

## Gate status

| Gate | Result |
|---|---|
| `cargo test -j4 -p engine --lib` | **303 passed, 0 failed** (was 301; +1 matched-centre records test, +1 worker-glue end-to-end test) |
| step-7 `-m integration` | **1 passed** |
| `pytest -m "not slow and not integration"` | **2532 passed, 134 skipped, 1 xpassed** + **1 PRE-EXISTING FLAKY fail** (see below) — my +27 passing vs the 2505 baseline; 0 collection errors |
| `make check.wasm` | **GREEN** (hexo-graph untouched) |
| clippy (engine) | **no NEW warnings** — 65-warning pre-existing baseline; the one new `too_many_arguments(8/7)` on `expand_and_backup_single_ls_framed` carries the §173 A5b hot-path `#[allow]` |
| `python -m hexo_rl.encoding audit` | `info=64 warn=3 error=1` — **identical to the WP3 steps-1-5 baseline** (lone ERROR = pre-existing no-metadata probe ckpt) |

---

## The 1 suite failure is a PRE-EXISTING flaky test (NOT mine)

`tests/scripts/test_t10_manifest.py::test_shape_fallback` — `assert 'v7' == 'v6w25'`.
Root-caused and unrelated to WP-3: `hexo_rl.encoding.compat.infer_encoding_from_state_dict`
tries `_filename_match(path_hint)` BEFORE the shape probe, returning any registered
encoding name that appears as a **substring** of the checkpoint path. The test builds
its checkpoint under a random `tempfile.TemporaryDirectory()`; when the random dir
name happens to contain a 2-char encoding name (`v6`/`v7`/`v8`), filename-match wins
over the intended shape resolution. Demonstrated deterministically:
`_filename_match('/tmp/tmpv7ab12/unknown.pt') == 'v7'`. It PASSES in isolation
(`-m ...::test_shape_fallback`), as a module, and across the whole `tests/scripts/`
dir; it only trips when the tempdir draw collides. **Decisive confirmation:** the
full suite re-run with my new test file MOVED ASIDE = **2533 passed, 0 failed**
(`test_shape_fallback` PASSED that run — same code, a different clean tempdir draw).
Adding a test file merely shifts collection order → the tempfile RNG draw → one
run happened to draw a `v#`-containing dir. My diff touches ZERO shape-inference /
registry / resolver / manifest code (`git diff --name-only` = engine graph-seam +
`graph_collate.py` + `inference_server.py`), and the new `-m integration` test
collects into `tests/selfplay/` (after `tests/scripts/`), so it cannot affect this
test's execution. Pre-existing test/resolver flake — out of WP-3 scope.

## What the bench gate should watch (dispatcher `make bench`)

- **10-metric CNN gate: expect no regression.** The graph dispatch is one
  never-taken branch at `infer_and_expand` entry (§ dormancy above); the dense
  hot methods + feature pool + `expand_and_backup`/`expand_and_backup_single_ls`
  (board-frame) are unchanged. WP3 steps-1-5 already measured `worker_pos_per_hr`
  −2.24% (noise band) from the dormant fields; step 6 adds no dense-path alloc/lock.
- **Graph-path cells (new, run4 §4.1 floor ≥1.0k steps/hr):** per-leaf build,
  `next_graph_batch` block-diagonal fuse, `collate + forward_batch +
  segment_softmax + submit` end-to-end. Bench on the REAL WPA distribution
  (`wpa_positions.json`, mean 490/2932); the gate must assert `builder_impl==1`
  (F7). The seam drops the CNN's TorchScript trace on the graph path (invalid for
  variable N/E) — the graph loop is eager forward + fp16 autocast (WPA figure was
  eager, so the floor already assumes no trace).
- **`submit_graphs_and_wait` is a smoke/eval driver, NOT the self-play hot path**
  — workers never cross PyO3 per leaf; they ride `submit_batch_and_wait_graph_rust`
  in-process. Don't bench through the Python driver.

## Files touched
`engine/src/game_runner/worker_loop/inner.rs`, `engine/src/mcts/backup.rs`,
`engine/src/game_runner/records.rs`, `engine/src/inference_bridge.rs`,
`engine/src/game_runner/mod.rs`, `hexo_rl/selfplay/graph_collate.py`,
`hexo_rl/selfplay/inference_server.py`, `tests/selfplay/test_gnn_seam_smoke.py` (new).
