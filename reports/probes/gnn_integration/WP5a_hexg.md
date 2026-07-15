# WP-5a — HEXG training-data path (RUST half) — AS-BUILT

**Scope:** the Rust half of WP-5 (C8 write-side): the HEXG graph-position replay
ring + rebuild-at-sample + D6 aug + HEXG v1 persist + the record-construction /
§178 outcome-stamping logic. Consumes `docs/designs/gnn_training_path_design.md`
(§HEXG / §recording), `gnn_ragged_contract_v1.md` §2.6 + Part 3. Single-sourced on
`GraphWire::from_axis_graphs` (shared with the inference seam) and
`sym_tables::rotate_axial` (shared with the CNN cell-scatter).

## Record layout as-built (bytes/record MEASURED)

Fixed-max-slot compact record (contract §2.6 realization (ii)) — a byte-for-byte
parallel of the HEXB stride-ring, SoA:

| block | fields | bytes |
|---|---|---|
| stones (MAX_STONES=256) | `stones_qr` i16×2 + `stone_players` i8 | 256·(4+1) = 1280 |
| visits (MAX_VISITS=128) | `visit_qr` i16×2 + `visit_probs` f32 | 128·(4+4) = 1024 |
| scalars | n_stones/n_visits u16, current_player i8, moves_remaining u8, ply_index u16, is_full_search u8, outcome f32, value_valid u8, game_length u16, game_id i64, weight u16 | 26 |
| **total** | | **2330 B** |

**Measured 2330 B/record = 2.3 KB** → 0.58 / 1.15 / 2.3 GB across the
250k/500k/1M capacity schedule (design §2.1 estimate matched exactly). NO dense
planes, NO aux (design §1.3 DROP: ownership/winning_line/chain/ply — GnnNet has
only policy + dist65 heads). Mean live content ~1.1 KB (CSR fallback → ~0.55 GB
if RAM binds; specified, not built).

## Ring + persist behavior

- **Ring mechanics** (`hexg/{mod,storage,push}.rs`): `head`-overwrite +
  `resize_impl` linearise (`rotate_left(head·stride)`) + `resize` on the graph
  SoA strides — the exact HEXB pattern. Weight-bucket histogram
  (decrement-on-overwrite / increment-on-write), game-length `WeightSchedule`,
  weighted rejection sampler + game_id dedup — all lifted verbatim (weights are
  game-length-driven, representation-agnostic). `size`/`capacity`/
  `get_buffer_stats`/`next_game_id` mirror `ReplayBuffer` so `pool.py` reads them
  representation-blind. Over-`MAX_STONES`/`MAX_VISITS` push is a LOUD error.
- **HEXG v1 persist** (`hexg/persist.rs`): own magic `0x48455847` ("HEXG"),
  version 1, header stamps `encoding_name` + `MAX_STONES`/`MAX_VISITS` (slot-geom
  sig). Save writes live entries (variable n_stones/n_visits) + all scalars +
  game_id + weight; load writes slots directly + rebuilds the bucket histogram.
  **Cross-format LOUD-FAIL both directions (tested):** a HEXB file → HEXG loader
  fails the magic check; a HEXG file → dense `ReplayBuffer::load` fails the HEXB
  magic check. Version-reject + slot-geometry-reject + encoding-mismatch all LOUD.

## Rebuild-at-sample cost (MEASURED)

`sample_graph_batch(256, augment=true)` on the REAL WP-A self-play distribution
(320 `wpa_positions.json` positions, mean 76 stones → 490 nodes/graph, 125,424
nodes/batch — matches WP-A mean 490 exactly):

- **313.6 ms/batch-256 serial (0.314 s/step)**, warm, always-on `verify_contract`,
  numpy-boundary copies + target alignment included.
- Design predicted 0.24 s/step serial; measured **0.31 s/step** — same ballroom,
  ~30% higher (v1 carries a `record_at` intermediate alloc per record + the
  numpy H2D copy + no BUILD-HOT perf sub-package). **8.7% of the 3.6 s/step floor
  budget → NOT floor-binding** (design §4.3; the floor is self-play-inference-
  bound, the train step overlaps generation async). Batch-parallel rayon is the
  documented BUILD-HOT follow-on (v1 = serial for correctness).

Sample path (`hexg/sample.rs`): weighted-sample indices → per record draw a
uniform D6 element → coord-rotate stones AND visit-keys via `rotate_axial` →
`hexo_graph::build_axis_graph` (stamps `builder_impl = 1` by construction — F7) →
align rotated visit-keys to built legal nodes by coord → the per-legal-node target
(no off-window drop) → block-diagonal fuse via `GraphWire::from_axis_graphs`. ONE
call emits graph + target together (F1 single-source). Emits `(GraphWire,
GraphTargets)` — the wire is the SAME payload `collate_graph_batch` reads;
`GraphTargets` carries `policy_target[Lg]` + `outcomes[B]` + `value_valid[B]` +
`is_full_search[B]` + `target_argmax_cells` (the AugRoundTrip runtime canary).

## D6 aug + ADV-7 TRUE reproduction evidence

- **Single-source `rotate_axial(q,r,sym)`** lifted into `sym_tables.rs`
  (reflect-then-rotate, `sym 0` = identity) — BOTH the CNN cell-scatter
  (`with_shape`, refactored to call it, byte-identical LUTs — all §133 D6
  sym-table tests + 316 lib tests pass) and the HEXG coord/visit-key rotation call
  it, so "D6 element s" is the identical geometry on both paths.
- **Round-trip coherence** (`rotate_axial_roundtrips_under_inverse`): `s` then
  `s⁻¹` recovers every coord for all 12 elements (reflections are involutions;
  rotations invert to `(6−n)%6`).
- **ADV-7 TRUE (2 legs):**
  1. **Unconstructability (positive):** the real `sample_graph_batch(augment=true)`
     output is coherent on EVERY draw — the target-argmax cell is always a legal
     node of the emitted graph (Rust `augmented_sample_target_is_coherent_every_element`
     over 48 draws; Python `test_augmented_sample_target_is_coherent` collates 20
     augmented draws WITH `target_argmax_cells` — a desync would raise). The
     single-call emission reads the target off the SAME rebuilt `legal_node_gather`
     the wire carries → graph/target desync is structurally unconstructable.
  2. **Canary discriminates (negative):** a manufactured desync (target-argmax
     poisoned onto a non-legal / STONE cell) FIRES the collate's
     `AugRoundTripMismatch` (Python `test_adv7_desync_target_argmax_raises`); the
     Rust `adv7_desync_is_caught_by_the_canary` proves the argmax-canonical canary
     passes real legal cells and fails occupied cells. NOTE: for a dense radius-6
     legal ball an UNROTATED in-ball cell can stay legal, so the coord-existence
     canary is a CANARY not a universal proof — the primary defense is the
     single-call unconstructability (leg 1), exactly as the design (WP-3 S1) frames it.

## Recording logic (§1.2/§1.3) — `record_position_graph` + `finalize_graph_outcome`

Built as pure, unit-tested functions in `game_runner/records.rs` (co-located with
the dense record fns):
- `record_position_graph(board, ls, …)` → ONE compact `GraphRecord` (whole-board,
  no K-loop): stones from `board.cells_iter()`; visit target read from the ragged
  `LegalSetPolicy` BY COORD over the FULL legal set — **the `records.rs:62`
  off-window skip is NOT inherited** (design §6.1); top-`MAX_VISITS` truncation by
  mass so the fixed slot never over-caps; outcome/value_valid = placeholders.
- `finalize_graph_outcome(rec_player, winner, terminal_reason, ply_cap_value,
  draw_reward)` → the §178 KEEP-verbatim per-row split (`terminal_reason==2` →
  `ply_cap_value` + masked; win/loss ±1; organic draw → `draw_reward`) — reads
  winner/reason/player only, NO cell geometry, so **INV26 / §178 transfers to
  graph rows UNCHANGED**. Unit test pins all four cases.

## Dormancy argument (dense byte-identical)

The bench measures dense CNN push/sample/inference metrics. All are byte-identical:
- **`replay_buffer/hexg/**` is a NEW parallel module** — the dense `ReplayBuffer`
  (`mod/storage/push/sample/persist`) code is untouched; only `pub mod hexg;` added.
- **`sym_tables.rs`**: the scatter-table CONSTRUCTION loop now calls `rotate_axial`
  (byte-identical reflect-then-rotate). Runs ONCE at init, NOT on the hot
  push/sample path; the §133 D6 sym-table tests + all 316 lib tests confirm
  byte-identical LUTs, so dense `apply_sym` (which reads them) is unaffected.
- **`inference_bridge.rs`**: extracted `GraphWire::from_axis_graphs`;
  `next_graph_batch` refactored to call it. `next_graph_batch` is the GRAPH
  inference path — dormant for all 11 dense encodings (`is_graph()` false). The
  dense inference hot path (`next_inference_batch`/`submit_inference_results`/
  `get_feature_buffer`) is byte-identical.
- **`records.rs`**: only ADDED two new pure fns (no dense caller); dense
  `aggregate_policy`/`finalize` untouched.
- **`game_runner/worker_loop`**: UNCHANGED (the live graph-record DISPATCH is not
  wired — see below), so the bench-gated worker hot path is literally identical.

## Test counts + suite status

- **Rust** (`cargo test -j4`): **319 lib pass** (303 baseline + 13 HEXG +
  3 records-graph), 3 ignored, 0 failed; all integration suites green.
  HEXG tests: ring push/wrap/persist-roundtrip/version-reject/HEXB-magic-reject/
  dense-loader-rejects-HEXG/slot-geometry-reject/grid-encoding-reject/
  rebuild-at-sample-parity-vs-direct-builder/rotate-axial-roundtrip/aug-coherence/
  ADV-7-canary. records: capture-stones-and-visit-mass / top-k-truncation /
  §178-outcome-split.
- **Python** (`pytest -m "not slow and not integration"`): collection 2691
  collected, **0 errors**; my `tests/training/test_gnn_hexg_buffer.py` = 6 pass
  (push→sample→collate→GnnNet.forward→ragged_policy_ce+binned_value_loss→backward,
  all finite; ADV-7 raise; cross-format reject). `tests/{training,selfplay,model}`
  = 239 + 37 (touched-area) green (losses.py `ragged_policy_ce` addition).
- **clippy**: 65 warnings total = pre-existing baseline; **0 in new code**.
- **`check.wasm`**: green (hexo-graph untouched, wasm-clean).
- **encoding audit**: 50 hits / 22 files = pre-existing baseline (HEXG files
  contribute 0 after the `size_of::<u64>()` cleanup); no registry diff.

## NOT wired (remaining integration step)

The live worker-loop DISPATCH of `record_position_graph` is NOT wired: the
`is_graph()` branch in `play_one_move` → per-game graph records vec →
`finalize_game_graph` → a graph results queue → `SelfPlayRunner.collect_graph_data`
drain → `pool.py` graph branch → `HexgBuffer.push_graph_positions`, plus forcing
`legal_set=true` for graph specs (so the recorded target is `MovePolicy::Ls`, not
the dense drop). The record-construction + §178-stamping LOGIC and the ENTIRE
buffer/sample/persist/aug path it feeds are built + tested end-to-end (the C8
write-side correctness); the drain is a separable bench-gated plumbing pass across
`game_runner/**` + `pool.py`.

## Files touched

New: `engine/src/replay_buffer/hexg/{mod,storage,push,sample,persist,tests}.rs`,
`tests/training/test_gnn_hexg_buffer.py`, this report.
Modified: `engine/src/replay_buffer/{mod.rs (pub mod hexg), sym_tables.rs
(rotate_axial lift)}`, `engine/src/inference_bridge.rs`
(`GraphWire::from_axis_graphs` + refactor + test accessors), `engine/src/lib.rs`
(register HexgBuffer + GraphTargets), `engine/src/game_runner/records.rs`
(`record_position_graph` + `finalize_graph_outcome` + tests),
`hexo_rl/training/losses.py` (`ragged_policy_ce`).

## Fix pass (B1/B2/game_id-rebase)

Closes the 2 real gaps + 1 theory item from `WP5a_redteam.md` (VERDICT
GAPS-FOUND). Scope: `engine/src/replay_buffer/hexg/{persist,sample,tests}.rs`
only — no dense/orchestrator/game_runner/inference_bridge/lib.rs touched.

### FIX 1 — B1 SILENT-CORRUPT: load failure-atomicity (`persist.rs`)

**Before:** `load_from_path_impl` zeroed the weight-bucket histogram and wrote
each slot directly into `self` *during* the per-record read loop, then set
`self.size` only at the end. A truncated/corrupt payload died mid-loop via `?`
with the histogram already reset and a prefix of slots already overwritten —
`self.size` stale against the old value → `size != Σhist`, a broken invariant
(confirmed by the red-team: pre-load `hist=[0,0,5]`, post-failed-load
`hist=[0,0,2]` while `size` stayed 5).

**Mechanism chosen:** two-pass parse-then-commit, buffering the whole parsed
payload transiently. **Pass 1** reads the entire file through the existing
bounds-checked `Cursor` into a local `Vec<ParsedRecord>` (owned, variable-length
`ns`/`nv` — NOT the fixed `MAX_STONES`/`MAX_VISITS` slot size); any `?` failure
returns immediately with `self` completely untouched. **Pass 2** (the commit)
only runs once every record parsed cleanly, resets the histogram once, and
writes all `size` slots + rebuilds the histogram + sets `size`/`head` — all
infallible writes, so pass 2 cannot itself leave a partial state. Net effect:
a failed load is a pure no-op on `self`; a successful load is byte-identical
to the old code's end state. Transient-memory tradeoff (documented in the
function's doc comment): pass 1 holds the payload twice briefly (raw bytes +
`parsed`), but `parsed`'s per-record allocations are sized to the record's
actual `ns`/`nv`, not the fixed 256/128 slot the committed buffer reserves —
so peak overhead scales with the file on disk, not the ring's worst case.

### FIX 2 — B2 SILENT target under-weighting: sample-align mass-drop guard (`sample.rs`)

**Before:** `push_graph_position` took visit coords with no legality check;
`sample_graph_batch_impl`'s align loop only visits the rebuilt legal-node set,
so mass on an illegal/off-window cell was dropped with the row's CE target
silently summing to `<1` — the argmax canary stayed quiet because the reported
argmax is always a legal cell by construction.

**Fix:** always-on check, comparing `Σ(rec.visits)` ("stored mass") against
`Σ(aligned target)` ("aligned mass", accumulated in the same align loop that
already touches every legal node — no extra pass). **Tolerance:** relative,
`REL_TOL = 1e-4`, with an absolute floor `ABS_FLOOR = 1e-6` for the near-zero
case (e.g. a quick-search row with `n_visits==0`, where both masses are ~0 and
the relative-tolerance branch would divide by ~0). On trip, returns a labeled
`PyValueError` naming `game_id` and `ply`, plus stored/aligned/dropped values.
Implementation note: the check itself (`mass_drop_check`) is a plain
`fn(..) -> Result<(), String>` outside the `impl` block, deliberately NOT
touching `PyErr` — `PyErr`'s `Display`/`Debug` impls call `Python::attach`,
which panics under plain `cargo test` (no `pyo3/auto-initialize`; see
`engine/Cargo.toml`'s `test-with-python` feature comment: "Our tests don't
call `Python::with_gil()`"). Keeping the message-construction pure-Rust lets
the labeled string be unit-tested directly without a GIL.
`sample_graph_batch_impl` wraps `mass_drop_check`'s `Err(String)` into
`PyValueError::new_err(msg)` only at the PyO3 boundary.

**Exactness check (why legit round-trips never trip):** `rotate_axial` (used
to rotate both stones and visit-map keys) is pure `i32` lattice arithmetic —
no float drift in *coordinates*; the *probabilities* are carried through
unchanged (only their key changes), so a legit record's aligned mass equals
its stored mass bit-for-bit whenever every visit coord lands on a legal node
(guaranteed by the `record_position_graph` producer contract, which only ever
emits mass on `board.legal_moves()`). Verified: the existing 1,696-position
parity fixture path (`tests/training/test_gnn_hexg_buffer.py`, `wpa_positions.json`)
and the D6-aug-coherence Rust test (`augmented_sample_target_is_coherent_every_element`,
48 draws) both stayed green post-fix, and a new negative-control test
(`legit_push_sample_roundtrip_does_not_trip_mass_drop_guard`) drives 24
unaugmented + 24 augmented draws on the fixture record with no raise.

### FIX 3 — game_id re-base (`persist.rs`)

**Before:** `next_game_id` was never touched by `load_from_path_impl`, so it
silently reset to 0 relative to the buffer's prior state; a fresh self-play
game after a resume-load could reuse a `game_id` already present among the
loaded records, mis-firing the Multi-Window correlation-guard dedup in
`sample_indices` (treats unrelated loaded/fresh positions as the same game).

**Fix:** after a successful commit (pass 2), `self.next_game_id =
self.next_game_id.max(max_gid + 1)` where `max_gid = parsed.iter().map(|r|
r.game_id).max()`. Monotonic (`.max()` against the existing counter, never
lowers it) and guarded for the empty-ring case by construction — `Option::max()`
over an empty `parsed` is `None`, so a zero-entry load is a no-op on
`next_game_id` (regression-tested by `load_of_empty_file_does_not_touch_next_game_id`).

### New tests (7, `hexg/tests.rs`; 13 → 20 HEXG tests)

- `failed_truncated_load_is_loud_and_leaves_buffer_untouched` — (a) from the
  spec: pre-populates a 5-record victim, attempts a load from a separate,
  truncated-mid-payload file, asserts `Err`, then asserts `size`/`head`/
  histogram/every record/every `game_id`/every `weight` are byte-identical to
  the pre-call snapshot, and that sampling afterward is still coherent.
- `sample_rejects_illegal_cell_visit_mass_drop` — (b) integration leg: 0.9
  mass poisoned onto an occupied `(0,0)` cell (the red-team's exact
  `probe_canary.py` repro) → `sample_graph_batch_impl` returns `Err`.
- `mass_drop_check_message_names_game_id_ply_and_dropped_mass` — (b) message
  leg, pure-Rust: asserts the labeled string contains `game_id=`, `ply=`,
  `dropped`.
- `mass_drop_check_tolerates_float_noise` — negative control: a `1e-6` gap and
  the `0.0`-vs-`0.0` case must not trip.
- `legit_push_sample_roundtrip_does_not_trip_mass_drop_guard` — (b) negative
  control: 24× unaugmented + 24× augmented legit round-trips, no raise.
- `load_rebases_next_game_id_past_loaded_max` — (c): loads game_ids `2,3,4`,
  asserts `next_game_id == 5`, then asserts a freshly minted id (`5`) does not
  collide with any loaded id.
- `load_of_empty_file_does_not_touch_next_game_id` — (c) empty-ring guard.

### Test evidence

**Rust** (`cargo test -j4 --manifest-path engine/Cargo.toml`, `-j4` cap
respected — laptop thermal): lib suite **326 passed, 0 failed, 3 ignored**
(319 baseline + 7 new); all 20 HEXG tests green including the 13 pre-existing
ones (ring wrap, persist round-trip, cross-format reject, rebuild-at-sample
parity, D6 aug coherence, ADV-7 canary — all unaffected). Full integration
suite (`cargo test -j4`, all targets) green, no regressions. `cargo clippy
--lib --tests` (project's `[lints.clippy] pedantic = warn`): the two-pass
`load_from_path_impl` restructuring initially tripped `too_many_lines`
(133/100) — split into `parse_records` (a free fn, PASS 1) and
`commit_records` (PASS 2 method) to bring the orchestrator under the
threshold; **0 clippy warnings in the 3 touched files** after the split
(re-verified: `cargo test` still 326/0/3 post-split).

**Python** (`maturin develop -m engine/Cargo.toml` rebuild, then
`pytest tests/training/test_gnn_hexg_buffer.py -q`): **6 passed** — unaffected
by the fix pass (push→sample→collate→GnnNet.forward→losses→backward all
finite; ADV-7 raise; cross-format reject).

**Full touched-area sweep**
(`pytest tests/training tests/selfplay -q -m "not slow and not integration"`):
**178 passed, 1 deselected** — dense paths stay green after the `.so` rebuild.

**Red-team reproductions re-run post-fix** (`probe_fuzz.py`, `probe_canary.py`,
`probe_ring.py`, `probe_partial.py`, all preserved under
`/tmp/claude-1000/-home-timmy-Work-Hexo-hexo-rl/2c5d9d06-3416-4182-85c9-8c7837722c89/scratchpad/wp5a_redteam/`):

- `probe_fuzz.py`: **all HELD**, including the B1 repro — histogram now
  reads `hist=[0, 0, 5]` before AND after the failed load (previously
  `[0, 0, 2]` post-failure); "buffer state consistent after failed load"
  now PASSes.
- B2 repro (occupied-cell peak + off-radius peak, driven standalone since the
  probe script's `try/except AugRoundTripMismatch` predates this fix and
  doesn't catch the new, earlier `ValueError`): both now raise
  `HEXG sample: visit mass dropped at sample-align ... for game_id=... ply=...
  stored=... aligned=... dropped=...`; the legit-all-legal control does not
  raise. The probe script itself now crashes with an uncaught `ValueError` at
  the first poisoned case — that crash **is** the LOUD-fail proof (it
  previously ran to completion silently under-weighting the target).
- `probe_ring.py`: **all HELD except one pre-existing, unrelated item** —
  "next_game_id restored across persist" now **PASSes** (`next_game_id` = 5
  after reloading ids `2..4`, previously 0/T1). The one flagged item,
  "resize-wrapped preserves logical order (save byte-identical pre/post)",
  fails because the probe compares the FULL file bytes including the header's
  `capacity` field, which legitimately changes on resize (4→16) —
  `storage.rs`/`resize_impl`/`save_to_path_impl` are untouched by this fix
  pass (out of scope), and a manual byte-diff confirms only offset 16 (the
  `capacity` u64) differs; the record payload after the 47-byte header is
  byte-identical. This exact discrepancy is already called out in the
  red-team's own prose ("the only byte diff is the header capacity field
  4→16") — a pre-existing artifact of the probe's raw-byte-equality
  assertion, not a regression from FIX 1/2/3.
- `probe_partial.py`: **all HELD**, including the `n_visits==0` zero-mass
  full-search row (stored=aligned=0, correctly does not trip the new guard).

### Verdict

All 3 red-team-flagged issues (B1, B2, T1) closed inside the declared scope.
No dense/orchestrator/game_runner/inference_bridge/lib.rs files touched. One
probe assertion in `probe_ring.py` (Attack 5, resize byte-equality) remains
red for a reason unrelated to and outside this fix pass's scope.

## Fix pass 2 (N1/N2/N3 — attacks on the FIX pass itself)

Closes the 3 hostile-input-only findings from `WP5a_redteam.md` § "Attacks on
the FIX code itself" (post-fix-pass-1 re-verification, FINAL VERDICT
GAPS-REMAIN-minor). Scope: `engine/src/replay_buffer/hexg/{persist,push,
tests}.rs` only — `sample.rs` untouched this round.

**N1 (`persist.rs:~178`)** — `max_gid + 1` panicked (`attempt to add with
overflow`, unlabeled `PanicException` through PyO3) in debug on a loaded
`game_id == i64::MAX`, and silently wrapped to `i64::MIN` (no-rebase) in
release. Fix: `max_gid.saturating_add(1)`. New test
`load_with_i64_max_game_id_does_not_panic_and_saturates` round-trips a
`game_id = i64::MAX` record through save→load and asserts `next_game_id ==
i64::MAX` (no panic, saturates rather than wrapping).

**N2+N3 (`push.rs`, `push_graph_position` / `push_record_impl`)** — the
`sample.rs` mass-drop guard (fix pass 1) is NaN-blind (`NaN > x` is false on
both branches, so a NaN visit prob passes the guard and reaches
`policy_target`/the loss — N2) and sign-blind (a `+/-` pair on legal cells
aligns to a stored-equals-aligned 0.0, so the guard passes and a NEGATIVE
entry reaches the CE target, flipping its gradient sign — N3). Fix: one guard
closes both, at push time (earlier than the sample-time guard it supersedes)
— a new pure helper `push::validate_visit_prob(q, r, prob) -> Result<(),
String>` (same GIL-free pattern as `sample::mass_drop_check`, tested directly
without `Python::attach`) checks `prob.is_finite() && prob >= 0.0` for every
visit in `push_record_impl`, before any mutation of `self`, naming the
offending `(q, r)` coord and the raw value on rejection. New tests:
`push_rejects_nan_visit_prob`, `push_rejects_negative_visit_prob` (both assert
the `PyResult` is `Err`, the pure-helper message names the coord + value, and
the rejected push leaves `buf.size == 0`), and
`legit_push_unaffected_by_prob_validation_guard` (negative control — the
existing fixture record's finite non-negative visit mass passes unchanged).

**Test evidence:** `cargo test -j4 --lib` (full crate): **330 passed, 0
failed, 3 ignored** (326 fix-pass-1 baseline + 4 new: N1 test + N2 test + N3
test + legit-push negative control). `cargo test -j4 --lib
replay_buffer::hexg`: **24 passed, 0 failed** (20 fix-pass-1 baseline + 4
new). `maturin develop --release` rebuild, then `pytest
tests/training/test_gnn_hexg_buffer.py -q`: **6 passed**, unaffected.

**Red-team re-verification re-run post-fix**
(`probe_reverify.py`, same scratch dir): F1 (B1 atomicity, harder
wrapped-ring + load-after-load attacks), F3 (game_id re-base, including the
N1 `i64::MAX` overflow case), and F2-A/F2-B (per-record firing,
false-positive hunt) all **PASS** unmodified. F2-C (NaN) now hits an
*uncaught* `ValueError` at the unwrapped `push_graph_position` call the
oracle script makes — that crash is itself the proof the fix works (push now
rejects the NaN before it ever reaches sample), but it also halts the script
before F2-D/F2-E execute (the oracle was written assuming the pre-fix
sample-time-only guard, same pattern already noted for the B2 probe in Fix
pass 1 above). Wrote a small adapted re-check,
`probe_n2n3_postfix.py` (same scratch dir), wrapping the push calls in
`try/except ValueError` to cover F2-C/F2-D/F2-E plus a legit-push negative
control end-to-end: **all HELD** — NaN and negative probs both rejected at
push with the coord/value named in the message, the buffer left untouched
(`size == 0`) on rejection, the pre-existing F2-E duplicate-coord case still
trips the (unmodified) sample-time mass-drop guard, and the legit-push
control samples a coherent `sum(policy_target) ~= 1.0` target.

### Verdict (fix pass 2)

N1/N2/N3 all closed inside the declared scope
(`persist.rs`/`push.rs`/`tests.rs` only). No `sample.rs`, dense,
orchestrator, game_runner, inference_bridge, or lib.rs files touched.
