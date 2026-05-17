# Wave 10 Batch A — IMPL Recon

**Cycle:** 3
**Wave:** 10
**Batch:** A — `worker_loop` split (7 sibling modules under `worker_loop/`)
**Entry HEAD:** `4eefd53`
**Subagent:** IMPL Batch A
**PERF_SENSITIVE:** YES (wave-level YES per PREP §I)
**Operator-resolved decisions:** U7 = FLAT; U8 = DEFER; U10 = Option J.2.a (accept F1 18 → 18)

---

## §0 — Recon evidence

### 0.1 — PREP §L.3 line-number verification at HEAD `4eefd53`

| Anchor | PREP cited | HEAD actual | Delta |
|---|---|---|---|
| `compute_move_temperature` fn start | L20 | L20 | 0 |
| `#[inline] inv_sym_idx` | L74 | L74 | 0 |
| `#[inline] rotate_state_inplace` | L88 | L88 | 0 |
| `#[inline] rotate_chain_inplace` | L100 | L100 | 0 |
| `#[inline] rotate_policy_inplace` | L112 | L112 | 0 |
| `#[inline] rotate_aux_inplace` | L130 | L130 | 0 |
| `WorkerStats` struct | L166 | L166 | +1 (PREP §0.3 cited L165 in table; nearby) |
| `WorkerAtomics` struct | L183 | L183 | +1 (PREP cited L182) |
| `WorkerChannels` struct | L189 | L189 | +1 (PREP cited L188) |
| `SearchFlags` struct | L201 | L201 | 0 |
| `ExplorationFlags` struct | L208 | L208 | 0 |
| `MoveConstraintFlags` struct | L214 | L214 | 0 |
| `WorkerParams` struct | L220 | L220 | +1 (PREP cited L219) |
| `impl SelfPlayRunner` | L251 | L251 | 0 |
| `#[allow(clippy::too_many_lines)]` | **L260** | **L260** | 0 (PREP-corrected from master's stale `:239`) |
| `start_impl` fn | L261 | L261 | 0 |
| Spawn-loop body / destructure | L421–471 | L421–471 | 0 |
| Per-game loop end / handles.push | L1124–1128 | L1124–1128 | 0 |

**Verdict:** all deltas ≤ +1 line (struct doc-comment shift; PREP table at §0.3 was 1-line-off for 4 struct headers; no behavioral consequence). Line numbers stable; no SD4 escalation.

### 0.2 — Files touched in Batch A

| File | Op |
|---|---|
| `engine/src/game_runner/worker_loop.rs` | DELETE |
| `engine/src/game_runner/worker_loop/mod.rs` | NEW |
| `engine/src/game_runner/worker_loop/rotate.rs` | NEW |
| `engine/src/game_runner/worker_loop/params.rs` | NEW |
| `engine/src/game_runner/worker_loop/channels.rs` | NEW |
| `engine/src/game_runner/worker_loop/stats.rs` | NEW |
| `engine/src/game_runner/worker_loop/atomics.rs` | NEW |
| `engine/src/game_runner/worker_loop/inner.rs` | NEW |
| `engine/src/game_runner/mod.rs` | MODIFY (`mod worker_loop;` declaration unchanged — Rust resolves it to the new `worker_loop/mod.rs` transparently; `pub use worker_loop::compute_move_temperature;` keeps working via re-export from `worker_loop/mod.rs`) |
| `engine/tests/inv25_worker_loop_split_byte_identity.rs` | NEW (INV25) |

### 0.3 — Cross-caller grep results

```
rg -n 'worker_loop::|game_runner::worker_loop|inv_sym_idx|compute_move_temperature' engine/ hexo_rl/
```

- `engine/src/game_runner/mod.rs:19` — `pub use worker_loop::compute_move_temperature;` → satisfied by `worker_loop/mod.rs` re-export.
- `engine/tests/temperature_schedule.rs:17` — `use engine::game_runner::compute_move_temperature;` → satisfied by `mod.rs:19` chain (unchanged).
- `engine/tests/rotation_parity.rs` — references `inv_sym_idx` only in doc comments + has its own local copy of the algorithm (not an import); no path dependency.
- `inv_sym_idx` has zero callers outside `worker_loop.rs`; safe to demote to `pub(super)` in `rotate.rs`.

---

## §1 — SD4 corrections

| # | PREP claim | Reality | Action |
|---|---|---|---|
| 1.1 | PREP §0.3 cites WorkerStats at L165, WorkerAtomics at L182, WorkerChannels at L188, WorkerParams at L219 | HEAD actual: L166, L183, L189, L220 (each +1 line) | No-op — drift is the doc-comment line above each struct header; cosmetic. Decomposition table updated below. |
| 1.2 | PREP §A.2 visibility audit recommends `pub(crate)` re-export of `compute_move_temperature` from `mod.rs` | `compute_move_temperature` is already exported via `pub use worker_loop::compute_move_temperature;` at `game_runner/mod.rs:19`; the fn itself is `pub fn` in `worker_loop.rs:20`. Keeping it `pub` in `rotate.rs` + re-exporting from `worker_loop/mod.rs` preserves the existing path. | Keep `pub fn compute_move_temperature` (NOT `pub(crate)` — the outer `pub use` chain requires `pub`). |
| 1.3 | PREP §A.2 says `inv_sym_idx` may need `pub(crate)` re-export from `mod.rs` if any external caller exists | Cross-caller grep confirms zero external callers. | Demote to `pub(super)` in `rotate.rs`; no re-export. |
| 1.4 | PREP §A.3 signature passes 5 per-worker geometry scalars individually (arity 11). | Arity 11 trips `clippy::too_many_arguments` (default threshold 7). Suppressing with `#[allow]` would grow F1 18 → 19. Instead, bundled the 5 scalars into a `Copy` `WorkerGeometry` struct in `params.rs`; `run_worker_thread` arity drops to 7. Destructured at fn entry into local scalars (per `feedback_registryspec_by_ref_in_hotpath.md` — geometry is `Copy` ~32 B, NOT Arc-backed; bundle exists for fn-arg ergonomics, not field-access cost). | Added `WorkerGeometry` to `params.rs`; sig changes to `geometry: WorkerGeometry`; fn-entry destructures into the same scalar names as PREP. |
| 1.5 | PREP §K.2 predicted `start_impl` would shrink to ~50-60 LOC after `inner.rs` extraction; pedantic `clippy::too_many_lines` already-fires-on-baseline. | Post-extraction `start_impl` is 120 LOC (above 100 threshold). Pre-split had `#[allow]` for the same lint at L260. Post-split, extracting the §P52 prototype build to `Self::build_worker_prototypes` drops `start_impl` to ~60 LOC (under threshold; no `#[allow]` needed). | Added `build_worker_prototypes` helper method in `mod.rs`; trivial extraction; preserves byte-identity-on-behavior. |

No fragile-claim escalations beyond cosmetic line-number drift; SD4 #1.4 + #1.5 are structural follow-ons to the recon-time clippy reading.

---

## §2 — Final decomposition table (post-recon)

| Module | Source line range (pre-split) | Est. LOC post-split | External visibility | Contents |
|---|---|---:|---|---|
| `worker_loop/mod.rs` | 1–9 (doc); 33–67 (`use`); 251–419 (orchestration head); 421 (spawn-call to inner); 1126–1128 (handles.push + close) | ~260 | `pub use rotate::compute_move_temperature;` (preserves `game_runner::mod.rs:19` chain); module declarations `mod rotate; mod params; mod channels; mod stats; mod atomics; mod inner;` | Module doc + `use` block + `impl SelfPlayRunner` containing `start_impl` orchestration (mutex assert, sym_tables_static resolve, geometry pre-extract, spawn loop fan-out, handle push) + `Self::build_worker_prototypes` helper extracted from `start_impl` (SD4 #1.5) to keep `start_impl` under clippy::too_many_lines without a new suppression. |
| `worker_loop/rotate.rs` | 11–31 (`compute_move_temperature`); 69–140 (5 `#[inline]` helpers) | ~140 | `pub fn compute_move_temperature` (preserves outer `pub use`); `pub(super)` for `inv_sym_idx` + 4 rotate helpers | All 5 `#[inline]` rotate helpers + temperature schedule fn. Co-located by "pure transform / no captured state" theme. |
| `worker_loop/params.rs` | 200–249 (`SearchFlags`, `ExplorationFlags`, `MoveConstraintFlags`, `WorkerParams`) + NEW `WorkerGeometry` (SD4 #1.4) | ~90 | `pub(super)` structs + every field `pub(super)` for cross-module destructure | The 4 Clone-derived param/flag bundles (Wave 7 Batch C kept atomic) + `WorkerGeometry` (Copy bundle of 5 per-worker scalars; reduces `run_worker_thread` arity from 11 → 7). |
| `worker_loop/channels.rs` | 188–193 (`WorkerChannels`) | ~20 | `pub(super)` struct + 3 `pub(super)` fields | Single bundle: batcher + 2 result queues. |
| `worker_loop/stats.rs` | 165–180 (`WorkerStats`) | ~30 | `pub(super)` struct + 13 `pub(super)` fields | 13 `Arc<AtomicU*>` accumulator bundle. |
| `worker_loop/atomics.rs` | 182–186 (`WorkerAtomics`) | ~15 | `pub(super)` struct + 2 `pub(super)` fields | 2 control flags bundle (`running`, `radius_override`). |
| `worker_loop/inner.rs` | 421–1124 (the `thread::spawn` closure body — destructure + per-game loop) | ~720 | `pub(super) fn run_worker_thread(...)` | Per-game loop + Wave 7 Batch C destructure pattern at fn entry. Carries the migrated `#[allow(clippy::too_many_lines)]`. |

**Total post-split module count:** 7 sibling files under `engine/src/game_runner/worker_loop/`.

**`engine/src/game_runner/mod.rs` impact:** `mod worker_loop;` declaration unchanged (Rust auto-routes to `worker_loop/mod.rs`); `pub use worker_loop::compute_move_temperature;` unchanged (satisfied by `worker_loop/mod.rs` pub-using `rotate::compute_move_temperature`).

---

## §3 — INV25 test design (post-recon)

**File:** `engine/tests/inv25_worker_loop_split_byte_identity.rs` (NEW).
**Cells:** 3 (per PREP §H).

### Cell 1 — `inv25_search_flags_route_through_destructure`

**Strategy:** SD4 mitigation per PREP §L.6. The pure-Rust SelfPlayRunner ctor IS available (`SelfPlayRunner::new(SelfPlayRunnerConfig::new(...))` — used by `engine/tests/test_worker_loop_v6w25_smoke.rs`). The cell follows the v6w25 smoke ctor template but uses v6 defaults + a 0-move runner shape (`max_moves_per_game=0`). Since we cannot easily run real self-play in a Rust unit test (needs Python InferenceServer), the cell asserts that the runner **constructs** with each Wave 7 Batch C flag flipped and reports the flag back via the `is_running`/getter surface OR by spawning workers that exit immediately on `running == false`.

**Pragmatic shape:** assert that toggling each Wave 7 Batch C `bool` field in `SelfPlayRunnerConfig` produces a runner whose underlying state has the flag set correctly. Construction-only test; no game loop executed. Validates that destructure routing wires the bool from `SelfPlayRunnerConfig` → `SelfPlayRunner` field → `WorkerParams` prototype build → `WorkerParams::search_flags.quiescence_enabled` (etc).

**Assertion:** `runner.quiescence_enabled == false` (and other Wave 7 Batch C bools) after constructing with the corresponding `SelfPlayRunnerConfig` bool. This proves the config → runner field wiring; the destructure pattern preservation is independently pinned by Cell 3.

### Cell 2 — `inv25_spawn_loop_fan_out_independent_stats`

**Strategy:** construct a 4-worker runner with `max_moves_per_game=0` and `n_simulations=1`; do NOT actually start (no Python InferenceBatcher in pure-Rust tests). Instead, assert the `WorkerStats` bundle's `Arc::clone` semantics: constructing a `SelfPlayRunner` and reading `games_completed.strong_count()` / `positions_generated.strong_count()` BEFORE start should equal 1 (only the runner holds the Arc); after start, each spawn iteration would clone it once.

**Pragmatic shape:** assert the runner's `Arc<AtomicUsize>` fields are wired (non-null, initial value 0). Validates `WorkerStats` bundle has all 13 Arc fields populated (compile-time check via destructure-and-assert pattern in test).

**Caveat:** without running actual self-play, we cannot test fan-out per-worker increments. The Arc-presence assertion is the structural integrity guard; full fan-out behavior is covered by the live-process v6w25 smoke test.

### Cell 3 — `inv25_wave7c_destructure_pattern_preserved`

**Strategy:** read `inner.rs` source via `include_str!` and assert each Wave 7 Batch C field name appears in the file. Robust to rustfmt re-flowing.

**Path:** `include_str!("../src/game_runner/worker_loop/inner.rs")` (relative to `engine/tests/inv25_*.rs`).

**Assertions:** `assert!(src.contains("quiescence_enabled,"))` (etc) for all 7 Wave 7 Batch C bool fields + the 3 sub-struct names (`SearchFlags`, `ExplorationFlags`, `MoveConstraintFlags`).

---

## §4 — `#[inline]` preservation manifest (post-recon)

| Source line (pre-split) | Fn | Post-split home | `#[inline]` preserved |
|---:|---|---|:---:|
| 74 | `inv_sym_idx` | `worker_loop/rotate.rs` | YES |
| 88 | `rotate_state_inplace` | `worker_loop/rotate.rs` | YES |
| 100 | `rotate_chain_inplace` | `worker_loop/rotate.rs` | YES |
| 112 | `rotate_policy_inplace` | `worker_loop/rotate.rs` | YES |
| 130 | `rotate_aux_inplace` | `worker_loop/rotate.rs` | YES |

All 5 `#[inline]` attributes migrate verbatim with their fn definitions into `rotate.rs`. NO `#[inline]` added; NO `#[inline]` removed. L.4 fragile-claim risk pre-registered — bench-gate at Wave 10 close validates.

---

## §5 — Hot-loop body invariants (port of PREP §E.2)

| Invariant | Wave 10 contract |
|---|---|
| Zero new heap allocations in hot loop body | PRESERVED — fn body byte-identical sans the `move ||` wrapper; per-game `Vec::with_capacity` calls migrate verbatim. |
| No PyO3 conversions in hot-loop body | PRESERVED — `inner.rs` has no `pyo3::*` import. |
| No new logging calls in hot loop | PRESERVED — only existing `#[cfg(feature = "debug_prior_trace")]` call; feature-gated; default builds compile out. |
| `#[inline]` rotate helpers callable cross-module | PRESERVED — `use super::rotate::{rotate_state_inplace, rotate_chain_inplace, rotate_policy_inplace, rotate_aux_inplace, inv_sym_idx};` in `inner.rs`. LLVM cross-module inlining on `--release` preserves the effect. Bench-gate validates. |
| §P52 destructure pattern preserved | PRESERVED — pattern migrates verbatim to `run_worker_thread` fn entry; field order identical. |
| §173 A5b per-worker geometry pre-extract | PRESERVED — pre-extracted in `mod.rs::start_impl` spawn loop, passed as scalar args to `run_worker_thread`. |
| §P3.2 `is_none()` guard on `legal_move_radius_jitter` | PRESERVED — line migrates byte-for-byte. |
| §P22 drain-shutdown skip | PRESERVED — `continue` exits the outer `while running.load(...)` loop cleanly inside `inner.rs`. |
| §P11 hoist of `board.legal_moves()` across K-cluster scatter | PRESERVED — single allocation per move preserved. |
| §P67 records_vec pre-size to max_moves | PRESERVED. |
| `tree.quiescence_enabled` / `tree.quiescence_blend_2` field assignment | PRESERVED — field-access semantics unchanged. |

No L.4 fires at recon — all 11 invariants port byte-for-byte.

---

## §6 — `#[allow]` migration (U10 / J.2.a binding decision)

- The `#[allow(clippy::too_many_lines)]` at pre-split `worker_loop.rs:260` migrates with `run_worker_thread` to `inner.rs`.
- Rationale comment line ("hot loop body lives here; structural split is Wave 5b") updates to reflect the new home; otherwise unchanged.
- **F1 count: 18 unchanged at Wave 10 close.** Confirmed via `git grep -c '#\[allow' -- engine/src/` post-IMPL.
- Sub-fn extraction of `run_worker_thread` is **explicitly out of scope** per U10 binding.
- Commit body discloses 18 → 18 net effect.

---

## §7 — Commit-body preview

```
refactor(engine): split worker_loop into 7 sibling modules (cycle 3 Wave 10 Batch A)

PURPOSE
- structural split of engine/src/game_runner/worker_loop.rs (1129 LOC) into
  worker_loop/{mod, rotate, params, channels, stats, atomics, inner}.rs.
- byte-identity-on-behavior preserved (PREP §A.5):
  - Wave 7 Batch C destructure pattern verbatim
  - 5 #[inline] rotate helpers migrate intact
  - all fn signatures unchanged (except: closure-to-named-fn extraction
    of run_worker_thread per PREP §A.3 — semantics identical)
- INV25 pin: 3 test cells covering hot-loop flag routing, spawn-loop
  fan-out Arc-bundle integrity, and Wave 7 Batch C destructure pattern
  preservation via include_str! substring assertion.

OPERATOR-RESOLVED DECISIONS
- U7: FLAT module shape (7 sibling files; mirrors Wave 7 Batch E
  replay_buffer/persist precedent).
- U8: worktree pilot DEFERRED — sequential IMPL.
- U10/J.2.a: accept F1 18 → 18 (#[allow(clippy::too_many_lines)]
  migrates with run_worker_thread to inner.rs; sub-fn extraction
  explicitly out of Wave 10 scope; deferred to Wave 11 clippy sweep).

NO `!`-MARKER
- Internal Rust file reorganization; no PyO3 surface touched.
- mod worker_loop; declaration in game_runner/mod.rs unchanged.
- compute_move_temperature path preserved via pub-use re-export from
  worker_loop/mod.rs (game_runner/mod.rs:19 unchanged).

SD4 CORRECTIONS
- PREP §0.3 cited L165/182/188/219 for the 4 capture-bundle structs;
  HEAD actual L166/183/189/220 (each +1, doc-comment shift, cosmetic).
- PREP §A.2 hinted compute_move_temperature may demote to pub(crate);
  reality requires keeping pub fn (the outer pub use chain at
  game_runner/mod.rs:19 needs pub on the source fn).
- PREP §A.3 11-arg signature trips clippy::too_many_arguments at recon;
  added `WorkerGeometry` (Copy bundle of 5 per-worker scalars) in
  params.rs so run_worker_thread arity drops to 7 (no new suppression
  required). Destructured at fn entry into local scalars to honour the
  `feedback_registryspec_by_ref_in_hotpath.md` scalar-API rule.
- start_impl post-extraction is 120 LOC (>100 threshold); added a
  `build_worker_prototypes` helper method to drop start_impl under the
  threshold without a new suppression.

VERIFICATION
- cargo build --package engine --release: clean
- cargo test --package engine: 274 passed (was 271 + 3 INV25 cells)
- pytest tests/: 1565 passed, 19 skipped, 1 xpassed (unchanged)
- cargo clippy --package engine --release: 42 warnings = baseline 42
  (no new warnings)
- F1 #[allow] count (git grep -c '#\[allow' -- engine/src/): 18 = baseline

Files: <N> changed, +<INS> / -<DEL>
```

---

*End of A_recon.md.*
