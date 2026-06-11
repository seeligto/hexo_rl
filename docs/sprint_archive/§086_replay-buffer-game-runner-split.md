<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §86. Structural split of `replay_buffer/` and `game_runner.rs` (2026-04-13)

**What.** Pure structural refactor of the two Rust files that the A1 aux
target alignment (§85) had inflated past the project's one-concept-per-file
threshold. Zero behaviour change; `cargo test` is the oracle. PyO3 surface
stable — every exported method on `ReplayBuffer` and `SelfPlayRunner` keeps
the same name and signature.

**Why now.** Landing A1 pushed `engine/src/replay_buffer/mod.rs` to 1,102 lines
(struct + storage + push + sample + resize + weight schedule + HEXB v2 save/load +
6 tests in one file) and `engine/src/game_runner.rs` to 1,313 lines (struct +
PyO3 facade + 502-line worker loop + `GumbelSearchState` + aggregate helpers +
drain + 9 tests). Both files violated the "one concept per file" rule and
slowed future diff review. This sprint is the cleanup pass.

**Post-A1 layout.**

```
engine/src/replay_buffer/
  mod.rs        ~220 lines  ReplayBuffer struct + #[pymethods] facade
  storage.rs    ~120 lines  resize, dashboard stats, weight schedule, next_game_id
  push.rs       ~380 lines  push, push_game, push_raw (test-only) + 2 tests
  sample.rs     ~420 lines  sample_batch + weighted-sample kernels + apply_sym + 4 tests
  persist.rs    ~280 lines  HEXB v2 save/load + round-trip test
  sym_tables.rs ~120 lines  UNCHANGED — 12-fold tables + WeightSchedule

engine/src/game_runner/
  mod.rs           ~430 lines  SelfPlayRunner struct + #[pymethods] facade + Drop + 3 tests
  worker_loop.rs   ~500 lines  start_impl — worker thread spawn + per-move MCTS loop
  gumbel_search.rs ~295 lines  GumbelSearchState + 6 gumbel tests
  records.rs       ~175 lines  aggregate_policy, aggregate_policy_to_local,
                               sample_policy, reproject_game_end_row
```

The old `engine/src/replay_buffer/sampling.rs` is **merged into `sample.rs`**:
after `sample_batch` itself moves out of `mod.rs`, maintaining a separate
"internal kernel" file next to a near-synonym "public entry" file creates
a naming trap (which file owns `apply_sym`?). One file for all sampling
concerns, still under the 500-line cap per file.

The 502-line worker loop was trimmed below the cap by extracting the
game-end ownership / winning-line reprojection block (~20 lines of the
inner per-row loop) to `records::reproject_game_end_row`. All other start()
logic is byte-identical.

**Visibility hygiene.** Fields and helpers that never cross a module
boundary defaulted to `pub(crate)`. `GumbelSearchState` is `pub(super)` —
only `worker_loop.rs` constructs and drives it. `Position` struct in the
old `game_runner.rs` was a dead `pub` type constructed nowhere in the
codebase; reducing its visibility exposed it as dead via `dead_code`, and
it was deleted in the same pass.

**Rust test binary build fix.** `cargo test` at HEAD failed to link
against libpython (`rust-lld: error: undefined symbol: PyErr_SetObject`,
etc.) because `engine/Cargo.toml` hard-coded `pyo3/extension-module`,
which strips Python symbols from the binary. That flag is correct for
the maturin cdylib build but wrong for the standalone test binary.
Restructured the feature wiring:

```toml
[features]
default = ["extension-module"]
extension-module = ["pyo3/extension-module"]
test-with-python = ["pyo3/auto-initialize"]
```

`maturin develop --release -m engine/Cargo.toml` picks up `extension-module`
via the default feature. `make test` now runs
`cargo test --no-default-features --features test-with-python`, which
replaces `extension-module` with `auto-initialize` so the test binary
links libpython directly and resolves every PyO3 C-API symbol.

**Tests.** 113 passing, zero test-body modifications. Test functions were
physically moved to the file that exercises the code they test (e.g.
`test_aux_hexb_v2_roundtrip` → `persist.rs`, `test_gumbel_topk_selection` →
`gumbel_search.rs`); the assertions themselves were not touched. Every
test continues to call its public entry point through its Rust name
(e.g. `buf.save_to_path(path)` in the persist test resolves to the
PyO3 facade in `mod.rs`, which delegates to `save_to_path_impl` in
`persist.rs`). Build is `cargo build --release` clean (zero warnings).

**Out of scope — tracked in `/tmp/refactor_todos.md`.**

- A1 reviewer's "cumulative-of-cumulative" mean_depth / root_concentration
  bias at the old `game_runner.rs:622-633` (now
  `game_runner/worker_loop.rs` stats block). Fixing this changes what the
  dashboard reports and would invalidate any smoke comparison against
  pre-refactor baselines, so it ships in a separate commit with its own
  regression test.

**Files touched.**

- `engine/Cargo.toml` — feature restructure (see Rust test binary build fix)
- `engine/src/replay_buffer/{mod,storage,push,sample,persist}.rs` (mod.rs rewritten, 4 new)
- `engine/src/replay_buffer/sampling.rs` (**deleted**, merged into sample.rs)
- `engine/src/game_runner/{mod,worker_loop,gumbel_search,records}.rs` (4 new)
- `engine/src/game_runner.rs` (**deleted**, promoted to directory)
- `Makefile` — `test` target passes `--no-default-features --features test-with-python`
- `CLAUDE.md` — Repository layout file tree updated
- `docs/01_architecture.md`, `docs/09_VIEWER_SPEC.md`, `docs/q12_s_ordering_audit.md`
  — file path references updated

**Commit:** `refactor(engine): split replay_buffer and game_runner into modules`

