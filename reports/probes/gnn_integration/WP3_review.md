# WP-3 steps 1-5 REVIEW (fresh eyes) — commit `871ea41`

**Reviewer:** WP-3 steps-1-5 review agent. READ-ONLY on source. Target
`871ea41` (2802 ins / 25 files). Authoritative: `gnn_inference_seam_design.md`
(steps 1-5), `gnn_ragged_contract_v1.md` (amended), `WP3_impl_steps1-5.md`
(claims — verified, not trusted).

## VERDICT: REVIEW-PASS

Dormancy is sound, the CNN path is byte-identical, and every named contract
error is present, correctly classified, and fires on execution. All findings are
NITs or forward-flags for step 6; none block the dormant commit. Independent
verification run (not just claim-trust):

- `cargo test -p engine --lib` — inference_bridge **10/10**, records
  `gnn_assemble_tests` **3/3**, encoding registry **18/18** PASS.
- `pytest tests/selfplay/test_graph_collate.py` (venv engine, graph seam built) —
  **19/19** PASS incl. ADV-1..9 + real-wire full-semantic clean collate + the
  full step-3+5 round-trip integration.
- The 8 retargeted grid test files — **206 passed / 1 skipped** (grid coverage
  intact).
- `python -m hexo_rl.encoding audit` — `info=64 warn=3 error=1`; the lone ERROR
  is the pre-existing `gnn_bc_040000.pt` NO-META probe checkpoint, NOT a
  registry/parity error. Rust↔Python spec parity clean on the schema-v4 fields.
- `window_flat_idx` (Rust, `hexo-graph/src/lib.rs:258-267`) vs
  `_canonical_slot_vec` (Python, `graph_collate.py:203-211`) — byte-for-byte
  identical (`half=(trunk-1)/2`, `wq/wr`, inside test, `wq*trunk+wr`).

## DORMANCY CONCLUSION: CONFIRMED DORMANT — CNN PATH BYTE-IDENTICAL

- **Dense prefill re-added verbatim inside `!is_graph`** (inference_bridge.rs
  ~354). For every grid batcher `representation = spec.map_or(Grid, ..)` → all 11
  grid encodings are `Grid` → `is_graph=false` → the prefill loop runs
  identically. The 2 diff deletions are exactly this loop, re-added unchanged.
- **No new alloc / lock widening / branch on the dense hot path.** The 4 dense
  hot methods (`submit_and_wait`, `next_inference_batch`,
  `submit_inference_results`, `pop_batch_blocking`) + the feature-buffer pool are
  NOT in the diff — untouched. The 5 new `Inner` fields (graph_queue Mutex,
  graph_queue_cv Condvar, 2 DashMaps, AtomicUsize) are constructed empty once and
  never read by any dense method. `close_rust` (not hot) adds a graph_queue_cv
  notify + an iterate over graph_waiters that is empty (no-op) for a grid
  batcher. Bench gate corroborates (worker_pos_per_hr −2.24%, noise band).
- **Zero production call sites.** `worker_loop/` has no graph reference (grep
  clean). `collate_graph_batch` is imported/called by NO production module (only
  its own + `gnn_net.py` docstrings + a `registry.py` comment). No config selects
  `gnn_axis_v1`. Every graph pymethod is `require_graph()`-guarded →
  `RepresentationMismatch` on a grid batcher (test-verified,
  `test_grid_batcher_rejects_graph_methods`).
- **`assemble_ls_from_gnn_probs` is a pure addition** — new fn + new test module,
  unreachable from the 11 grid encodings (only called by the graph seam methods,
  themselves require_graph-guarded).

## Findings (severity · file:line · fix)

**S1 (SHOULD-FIX) · tests/selfplay/test_graph_collate.py:192** — `test_adv_7` is a
slot-leg PROXY, not a true rotated-graph/unrotated-target pair. It shifts
`window_center[0]+1` to desync the canonical-slot recompute (fires
`ScatterSlotCanonicalMismatch`); it does NOT rotate `node_coords` by a D6 element
while pinning `policy_dst_slot`. Detection logic exercised is the same, and impl
divergence #4 is honest that the full-rotation face is deferred to C8/WP-5 (aug is
default-OFF at inference). Acceptable for the dormant inference scope; track the
real rotation reproduction (rebuild-rotated graph, splice unrotated slot-map) as a
WP-5 obligation.

**S2 (SHOULD-FIX — step-6 forward-flag, not a steps-1-5 defect) · records.rs:324 /
inference_bridge.rs `assemble`/`InFlightGraph`** — the in-window half of the
assembled `LegalSetPolicy.dense` is baked with the BUILDER's per-leaf
bbox-midpoint `window_center` (via `policy_dst_slot`). The consumer
`LegalSetPolicy::get` reads `dense[window_flat_idx_at_geom(q,r,bcq,bcr,..)]` with
whatever center step 6 passes at expand time. If step 6 does not thread each
leaf's builder `window_center` into `expand_and_backup_ls`, the in-window priors
misread by a slot offset — the exact F1 coord/slot geometry-mismatch class.
`assemble_ls_from_gnn_probs` and `InFlightGraph` do NOT currently retain
`window_center` for the consumer (the wire carries it, but nothing threads it to
expand). Flag for the dispatcher's step-6 wiring: pass builder per-leaf
`window_center` (and add a matched-center round-trip test).

**N1 (NIT) · inference_bridge.rs ~331-335** — `spec.win_length.unwrap_or(6)` /
`spec.graph_radius.unwrap_or(6)` introduce dead `6` literals on the graph ctor
path. `validate()` guarantees these `Some` for a graph spec, so the fallback never
fires — but design §5 mandates "no 6/19/11 literal anywhere." Use
`.expect("validate guarantees win_length for graph")`.

**N2 (NIT) · tests/selfplay/test_graph_collate.py:201** — `test_adv_8` flips one
`signed_dist` value (`edge_attr[3] = -edge_attr[3]`) rather than PERMUTING
edge_attr rows as the contract names ("edge_attr rows permuted within one graph").
Detection (`delta != signed_dist × axis_vec`) is genuine; a within-graph row-roll
is the stronger positional-correspondence reproduction.

**N3 (NIT) · graph_collate.py:427,449** — 2 of the 13 structural named errors have
NO dedicated adversarial test: `EdgeIndexOutOfBounds` and `ScatterSlotOutOfBounds`
(16/18 errors covered). Add two cheap targeted payloads.

**N4 (NIT) · records.rs ~455-460** — the slot-range guard in
`assemble_ls_from_gnn_probs` is a hard `assert!` (release-panic → process ABORT
under the workspace `panic="abort"`) vs the graceful die-loud-via-`submit_graph_
inference_failure` path §7 prescribes. Unreachable in practice (slots come from the
builder's `verify_contract`-guaranteed `meta.policy_dst_slot`, never from Python),
so harmless — but prefer `debug_assert`/`Result` for step-6 consistency.

**N5 (NIT) · inference_bridge.rs ~247-292 / ~588-672** — mid-loop error returns in
`submit_batch_and_wait_graph_rust` / `submit_graph_inference_results` leak
partially-enqueued in_flight/waiter state. Only reachable on the die-loud
handshake/segment-desync paths (worker dies anyway) and practically unreachable
(contract_version is a batcher constant checked on graph 0; builder_impl is always
native from `build_graph_from_request`).

**N6 (NIT — correctly out of scope) · tests/test_network_encoding_dispatch.py** —
the NET-construction loud-fail (a `representation=graph` spec building a
`HexTacToeNet`) is NOT asserted; the test retargets to exclude graph rather than
asserting rejection. This is the `build_net(spec,state)` authority = C4/C7 per
design §8, not WP-3. The INFERENCE-seam loud-fail (`require_graph` →
`RepresentationMismatch`) IS tested. Track net-construction loud-fail for C4/C7.

## Priority-by-priority

1. **Dormancy** — PASS (see conclusion above). Dense control flow provably
   unchanged; records.rs is pure addition; nothing reachable via the 11 grid
   encodings.
2. **Registry schema v4** — PASS. No code depends on `schema_version` bumping
   (only a pyo3 getter + audit print; `corpus_io`/`checkpoints` schema_versions
   are unrelated namespaces). Grid invariants all run for the 11 grid entries
   (default `Grid` → Grid arm); `expected_logits` preserved in BOTH arms;
   universal checks (multi-window/legal_move_radius/k_max/plane-dup) verified safe
   for the graph spec. `parse.rs` `representation.unwrap()` is guarded by the prior
   `if !errs.is_empty() { return Err }` (no panic on bad TOML). Python parity is
   structural — `EncodingSpec IS engine.RegistrySpec`, getters auto-exposed; audit
   clean.
3. **graph_collate vs contract §2.5** — PASS. 18 assertions present + correctly
   split (12 structural in `_check_structural` + version-check in collate = 13
   structural; 4 semantic in `_check_semantic`; `NonNativeSampleBuilder` handshake).
   Full/canary/off split matches design (semantic="full" trainer, "canary" hot
   path first+Nth, structural always full). ADV-1a/1b/2a/2b/3/4/9 are faithful
   reproductions; ADV-7/8 are detection-equivalent proxies (S1/N2). All fire named
   errors on execution.
4. **Seam guards** — PASS. Range-validate-before-narrowing-cast (current_player,
   moves_remaining, coord bound, player) all in `build_graph_from_request`, the
   single build path (spawn_mock + check_graph_request route through it; step-6
   BuildParams should too — impl divergence #3, correctly flagged). builder_impl +
   contract_version handshakes at ctor (validate), `submit_batch_and_wait_graph_rust`
   (per graph), `next_graph_batch` (per graph, builder_impl), and Python resolver.
   `test_check_graph_request_seam_obligations` (5 cases) passes.
5. **assemble_ls_from_gnn_probs** — PASS. No renorm (segmented-softmax
   pre-normalized; `debug_assert` sum≈1). Off-window keyed by COORD
   (`overflow.insert(legal_coords[i], ..)`), matching consumer `get`'s
   `overflow.get(&(q,r))`. Consumer `LegalSetPolicy::get` reads priors by coord
   with NO renorm-on-read → an unnormalized map would corrupt sampling, but the
   assemble output is normalized by construction. `assemble_off_window_argmax_
   survives` confirms the off-window argmax stays the global argmax. (Deploy-side
   center-consistency = S2, step-6.)
6. **Test-retarget honesty** — PASS. No grid invariant weakened. Each edit either
   restricts a CNN/dense-specific parametrized test to grid (semantically correct —
   graph has n_planes=0, no state_stride, no CNN arch: buffer_roundtrip,
   arch_resolver, network_encoding_dispatch, pool_encoding_resolve) or adds the
   graph encoding to an expected-set/bucket/allowlist (registry, resolver_paths
   gap [matches existing v7/v7e30 pattern], round_trip skip-after-stable-check,
   n_chain positively asserts the graph n_source==0 invariant). 206/1 pass.
7. **Idiom/quality** — PASS. `.expect("… poisoned")` on locks/condvars matches the
   dense path; PyErr propagation via `PyValueError::new_err`/`PyResult` correct; no
   raw `.unwrap()` on the production seam (only in tests). Doc comments name each
   guarded error. Minor: N1 literals, N4 assert-vs-Result. Clippy not independently
   re-run (thermal budget) — code compiles clean and all tests pass.

## Environment note
This worktree's mise python 3.14 site-packages `engine` lacks the built `.so`
(`ModuleNotFoundError: engine.engine`); Python tests were run with the repo
`.venv/bin/python`, whose engine extension is built WITH the graph seam
(`has next_graph_batch == True`). Rust tests built fresh in-worktree (`cargo -j4`,
debug profile).
