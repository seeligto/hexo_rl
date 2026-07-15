# WP-5b COMMIT A — live worker-loop graph-recording dispatch — AS-BUILT

**Scope:** wiring only (per `docs/designs/gnn_wp5b_commitA_delta.md`) — the
WP-5a-built `record_position_graph`/`finalize_graph_outcome`/`HexgBuffer`
primitives are consumed, not reinvented. Dense/CNN path byte-identical
(verified: full dense test sweep green post-change, no dense hot-path fn
edited beyond one hoisted branch each).

## What wired where

### Rust (`engine/src/`)

| # | File | Change |
|---|---|---|
| R1 | `game_runner/worker_loop/mod.rs` (`legal_set` derivation, ~L159) | `legal_set = s.is_graph() \|\| matches!(s.policy_pool, LegalSetScatterMax)`. Gated on `is_graph()` — every grid spec's prior policy_pool-derived value is unchanged. |
| R2 | `game_runner/worker_loop/params.rs` (`WorkerGeometry`) + `mod.rs` (spawn loop) | Added `is_graph: bool` field, derived once per worker spawn from `self.registry_spec.map(is_graph)`. |
| R3 | `game_runner/worker_loop/inner.rs` (`PerGameInit`, `init_per_game_board`) | Added `graph_records: Vec<GraphRecord>` — always `Vec::new()` (zero-cap, no alloc for grid). |
| R4 | `inner.rs` (`play_one_move` record site) + new `record_position_graph_dispatch` fn | ONE hoisted `if is_graph {...} else { record_position(...) }` at the record call site. Graph arm extracts `MovePolicy::Ls(ls)` (guaranteed by R1) and calls `records::record_position_graph`; the `Dense` arm is `unreachable!()` (always-on, not a stripped `debug_assert!` — structurally impossible given R1, so a die-loud panic is correct if the invariant is ever broken). `#[cold]`/`#[inline(never)]`. |
| R5 | `inner.rs` new `finalize_game_graph` fn + `run_one_game` finalize dispatch | Sibling to `finalize_game`; duplicates the winner/`terminal_reason`/`version_seen` classification (`finalize_game` itself untouched), stamps each `GraphRecord` via `records::finalize_graph_outcome`, sets `game_length = plies.div_ceil(2)` (compound moves, matches `pool.py`'s dense `(plies+1)//2` convention), pushes the SAME `recent_game_results` metadata row `finalize_game` does, and caps `graph_results_queue` at `results_queue_cap` (reuses the existing `positions_dropped` counter — dense backpressure-drop parity, not a new primitive). |
| R6 | `game_runner/mod.rs` (`SelfPlayRunner` struct + `new`) + `worker_loop/{mod,channels}.rs` | Added `graph_results: Arc<Mutex<VecDeque<GraphRecord>>>` (constructed unconditionally, idle for grid) + threaded into `WorkerChannels`/the worker spawn closure alongside `results_queue`. |
| R7 | `game_runner/mod.rs` new `collect_graph_data` PyO3 method | Drains `graph_results`, returns `Vec<GraphRecordRow>` (9-tuple matching `push_graph_position`'s positional signature, `game_id` excluded per Adjudication 2). Sibling to `collect_data`; grid runners' `graph_results` queue stays permanently empty so this is never called on the dense path. |

R8 (bulk `push_graph_positions`) **not built**, per the delta doc's explicit
ruling — the singular-loop drain in `pool.py` is the commit-A choice (avoids
reopening the bench-gated `hexg` module for an unmeasured optimization).

### Python

| # | File | Change |
|---|---|---|
| P1 | `hexo_rl/training/orchestrator.py` (`init_replay_buffer`) | THE ONE RESOLVER: dispatches `HexgBuffer(capacity, encoding=spec.name)` when `_window_set(...).is_graph`, else `ReplayBuffer` (byte-identical for grid). |
| P2 | `orchestrator.py` (`restore_buffer_from_checkpoint`) | Graph branch (`isinstance(buffer, HexgBuffer)`) RE-RAISES on a restore failure instead of the dense swallow-and-warn (Adjudication 1 / §RUN3-STEP0 regression pin). Dense branch unchanged. |
| P3 | `orchestrator.py` (`init_recent_buffer`) | Graph spec (`registry_spec.is_graph`) skips the dense `RecentBuffer` construction even when `recency_weight > 0`, logs `recent_buffer_skipped_graph` (loud, names commit B as the sample-side fix). Dense branch unchanged. |
| P4 | `hexo_rl/selfplay/pool.py` (`WorkerPool.__init__`) | Added `self._is_graph`; `_feat_len`/`_chain_len` set to `0` for graph (skips the degenerate `n_kept_planes * trunk²` derivation) instead of computing a meaningless value. |
| P5 | `pool.py` (`_run_stats_loop`) | ONE hoisted branch: graph arm calls `collect_graph_data()` and loops `replay_buffer.push_graph_position(*rec, game_id=-1)`; the dense `collect_data`/`push_many`/reshape/recent_buffer block is wrapped in the `else` arm **verbatim** (same code, one indent level deeper — confirmed via the dense self-play test sweep below). |
| P6 | `hexo_rl/training/buffer_persist.py` | No change — the `recent_buffer is not None` guard already present; confirmed by reading, per the delta doc's "note only." |

## Deviations from the delta doc

1. **`configs/variants/run4_gnn.yaml` not created.** The delta doc's touch
   list names it as a "Config / tests" line item, but the actual commit-A
   test plan (§10) drives self-play via inline `SelfPlayRunnerConfig`
   construction (mirroring `test_gnn_seam_smoke.py`'s existing precedent),
   never through this yaml. Deferred as an operator/launch-config artifact
   (the actual `run4` launch decision — bot_batch_share, buffer_persist_path
   namespacing, recency_weight declaration — is a separate, later act than
   wiring the dispatch). No test in this commit needs it.
2. **`finalize_game_graph` caps `graph_results_queue` at `results_queue_cap`**
   (reusing the existing `positions_dropped` atomic) — the delta doc's R6 row
   doesn't explicitly mandate this, but it mirrors `finalize_game`'s existing
   dense backpressure-drop hygiene applied to the new queue. Not a new
   correctness primitive (same counter, same pattern); flagged since it's an
   addition beyond the letter of R5/R6.
3. **One `clippy::manually reimplementing div_ceil` self-fix**: the first
   draft of `game_length` computation used `(plies + 1) / 2`; replaced with
   `plies.div_ceil(2)` (matching the file's own `compound_move` convention
   two lines above in `play_one_move`) to keep 0 new clippy warnings in
   touched files.

## Concerns / findings (not fixed — out of commit-A scope)

**Pre-existing gap discovered while building the end-to-end integration
test: `hexo_graph::legal_moves_from_stones` (WP-1's native builder,
`engine/hexo-graph/src/lib.rs`) has no empty-board fallback.** A fresh
(0-stone) board yields ZERO legal graph nodes from the native builder, so an
MCTS search rooted at game start raises `EmptyLegalSet` inside
`collate_graph_batch`'s structural check on the very first leaf-batch;
`run_mcts_search` never completes a root expansion, and the game finalizes at
`ply==0` after exhausting its move budget on `RootExpansionFailed` retries.
`Board::legal_moves_set()` (the dense engine, `board/moves.rs:95-101`) DOES
special-case the empty board (a 5×5 opening region) — `hexo-graph`'s
candidate generator does not mirror it (`legal_moves_from_stones` only
expands radius-balls around EXISTING stones). This is **not** a WP-5b
commit-A regression — I never touched `hexo-graph/`, `infer_and_expand_graph`,
or `graph_collate.py` — it is a latent WP-1/WP-3 gap that nobody had
exercised before, because every prior graph-seam test/smoke (WP-3 step-7,
WP-5a's fixtures) drove pre-populated boards, never an actual game-0 position
through the live MCTS+inference loop. **Practical consequence: organic graph
self-play (`random_opening_plies=0`) cannot currently start a game at all
through the live worker loop.** The integration test built for this commit
routes around it with `random_opening_plies=2` (the same Board-level,
MCTS-bypassing opening mechanism dense self-play already uses for
diversity), which is a legitimate workaround for testing purposes but is
**not a fix** — a real `run4` launch config needs either
`random_opening_plies >= 1` or a WP-1 patch (empty-board fallback in
`legal_moves_from_stones`, mirroring `Board::legal_moves_set()`) before it
can run organically from move 0. Flagging for controller/reviewer triage;
recommend a follow-up ticket against WP-1, not a commit-A fix (touches
`hexo-graph/src/lib.rs`, outside this delta's touch list).

## Test evidence

**Rust** (`cargo test -j4 --manifest-path engine/Cargo.toml`, full crate, all
targets): **334 lib tests passed, 0 failed, 3 ignored** (330 baseline + 4 new
— `finalize_game_graph_win_loss_split_matches_178`,
`finalize_game_graph_ply_cap_masks_value_valid`,
`finalize_game_graph_organic_draw_stays_supervised`,
`finalize_game_graph_caps_queue_at_results_queue_cap`, all in a new
`inner.rs::graph_finalize_tests` module — R5 "live-loop wiring" test per the
delta doc §10 test-plan row 2, exercising the real `finalize_game_graph` fn,
not just the pure `records::finalize_graph_outcome` helper WP-5a already
unit-tests). All non-lib integration-test binaries green (18 binaries, 0
failed). `cargo clippy --lib --tests`: 64 lib warnings (matches the
pre-existing baseline magnitude WP5a reported; verified none of the 6
touched Rust files carry a NEW warning — `run_worker_thread`/`play_one_move`'s
pre-existing `too_many_lines` flags were already tripped before this diff by
~155-165 lines of body, my +2/+13 line additions didn't newly trigger them;
one self-inflicted `div_ceil` suggestion was fixed inline, see Deviations).

**Python** (`.venv/bin/python -m pytest tests/training tests/selfplay
tests/model -q -m "not slow and not integration"`): **239 passed, 2
deselected** (the new graph integration test correctly excluded by the
`-m integration` marker), 0 failures, 0 collection errors. Full-repo
collection (`pytest -m "not slow and not integration" --collect-only`):
2707/2724 collected, 0 errors.

**New integration test** (`tests/selfplay/test_gnn_record_dispatch.py -m
integration`): **1 passed** (~13s, stable across 4 repeated runs). Drives
REAL `SelfPlayRunner.start()`/`.stop()` self-play on `gnn_axis_v1` to 3
completed games, then: (a) `collect_graph_data()` row count == exactly
`sum(plies - random_opening_plies)` across the drained completed games
(R7/P5 drain correctness); (b) off-window visited-cell mass is reachable
(122-210 hits observed across calibration runs — R1 legal_set force + R4
record dispatch proof: the `records.rs:62` off-window skip is NOT inherited
on the live path); (c) every drained row round-trips
`HexgBuffer.push_graph_position` with no exception (Adjudication 3 live
control); (d) `save_to_path`/`load_from_path` round trip +
`get_buffer_stats()` histogram sums to buffer size; (e)
`sample_graph_batch(augment=True)` → `collate_graph_batch` →
`GnnNet.forward_batch` → `ragged_policy_ce` + `binned_value_loss` all finite.
Note: if R1 (`legal_set` force) were unwired, this test would not merely
fail an assertion — the worker thread would **panic** on the first recorded
move (`record_position_graph_dispatch`'s `unreachable!()` guard), so a
completed run with non-empty rows is itself structural proof R1+R4 are wired.

**18-assertion contract + 9 ADV payloads + orchestrator unit tests**
(`tests/selfplay/test_graph_collate.py tests/training/test_gnn_hexg_buffer.py
tests/selfplay/test_gnn_seam_smoke.py tests/test_orchestrator_gnn_buffer.py
tests/test_orchestrator_gnn_build.py -m "not slow"`): **36 passed** — all
pre-existing WP-3/WP-5a suites untouched-green, plus 6 new orchestrator P1/P2/P3
unit tests in `tests/test_orchestrator_gnn_buffer.py` (buffer resolver
dispatch both directions, graph restore loud-fail + dense swallow regression
pin, recency graph-skip + dense-unaffected regression pin).

**Encoding audit** (`.venv/bin/python -m hexo_rl.encoding audit`):
`info=65 warn=3 error=1` — the 3 warnings + 1 error are pre-existing
(`gnn_bc_040000.pt` no-metadata, a corpus filename heuristic mismatch, and
50 unjustified-literal hits across 22 files); confirmed neither
`pool.py` nor `orchestrator.py` appear in the hardcode-hit dump — my changes
introduced zero new scattered-literal findings.

**Dense byte-identical verification**: `tests/test_worker_pool.py` (real
`WorkerPool` self-play smoke, dense encoding, exercises the reindented
`_run_stats_loop` `else` arm end-to-end — `collect_data`/`push_many`
confirmed still populating the buffer) +
`tests/test_selfplay_runner_encoding_e2e.py` +
`tests/test_selfplay_encoding_aware.py` + `tests/selfplay/
test_v6w25_microsmoke.py` + `tests/test_rotation_buffer_compat.py` +
`tests/test_selfplay_registry_plumbing.py` +
`tests/selfplay/test_pool_inference_pool_size.py`: **33 passed, 3 skipped, 1
xpassed** (pre-existing skip/xpass, unrelated to this diff), 0 failures.

## Files touched

**Rust:** `engine/src/game_runner/worker_loop/{mod,inner,params,channels}.rs`,
`engine/src/game_runner/mod.rs`.
**Python:** `hexo_rl/training/orchestrator.py`, `hexo_rl/selfplay/pool.py`.
**Tests (new):** `tests/selfplay/test_gnn_record_dispatch.py`,
`tests/test_orchestrator_gnn_buffer.py`. **Tests (modified):** none (the R5
unit tests were added as a new `#[cfg(test)] mod graph_finalize_tests` inside
`engine/src/game_runner/worker_loop/inner.rs`, not a separate file).
**Report:** this file.

No files outside this list were touched. `buffer_persist.py` was read (P6)
but not modified. `configs/variants/run4_gnn.yaml` was not created (Deviation
1). `engine/hexo-graph/src/lib.rs` was read (investigating the EmptyLegalSet
finding) but not modified.

## Fix pass (commit-A red-team ADV-A/B/C/D)

Closes the two LOW-severity write-seam gaps from
`reports/probes/gnn_integration/WP5b_commitA_redteam.md` #1
(RAGGED-PAYLOAD end-to-end): `outcome` finiteness and per-stone `player`
range were unguarded on `push_graph_position`, both weaponizable via the
"poison between drain and collate" threat class, and both inconsistent with
the WP-5a die-loud posture already applied to the visit-prob field.

**Fix** (`engine/src/replay_buffer/hexg/push.rs`): extended the existing
`push_record_impl` validation block (same idiom/placement as
`validate_visit_prob` — pure, GIL-free helpers, checked before any mutation
of `self`) with two new guards:

- `validate_outcome(rec.outcome)` — rejects non-finite (NaN/±inf) outcome,
  naming the offending value.
- `validate_stone_player(q, r, player)` per stone in `rec.stones` — rejects
  any player ∉ {+1, −1}, naming the offending coord + value.

Both are boundary checks (drain-loop push path), not per-sim hot path — no
bench concern, matches the delta doc's own framing of the fix as ~6 lines.

**Tests added:**

- Rust (`engine/src/replay_buffer/hexg/tests.rs`, new section "WP5b
  commit-A red-team fix-pass regression tests (ADV-A / ADV-B)"):
  `push_rejects_nan_outcome`, `push_rejects_inf_outcome`,
  `push_rejects_bad_stone_player`,
  `legit_push_unaffected_by_outcome_and_stone_player_guards`.
- Python, graduated from the red-team's ADV cases, alongside the existing
  WP-5a cases in `tests/training/test_gnn_hexg_buffer.py` (new section
  "WP5b commit-A red-team fix-pass (ADV-A / ADV-B / ADV-D)"):
  - ADV-A `test_adv_a_push_rejects_non_finite_outcome` (parametrized
    NaN/+inf/−inf) — via the real PyO3 `push_graph_position` entry point.
  - ADV-B `test_adv_b_push_rejects_stone_player_out_of_range` (parametrized
    0/2/5/−3).
  - ADV-D `test_adv_d_illegal_visit_coord_raises_through_push_sample_roundtrip`
    — checked first: no existing Python test drove `push_graph_position` →
    `sample_graph_batch` with an illegal (occupied-cell) visit coord; the
    only prior coverage was the Rust-internal
    `hexg/tests.rs::sample_rejects_illegal_cell_visit_mass_drop` (same
    semantics, different language boundary — doesn't exercise PyO3 error
    propagation). Added as a genuine gap-close, not a duplicate.
  - ADV-C `tests/selfplay/test_gnn_record_dispatch.py` extended in place
    (no new file): asserts every drained `collect_graph_data()` row's
    `outcome` is finite (post-drain, pre-push), and every sampled
    `tg.outcomes` entry is finite (post-sample-rebuild) — near-free given
    the e2e already computes losses on the same tensor.

**Verification:**

- `cargo test -j4 --manifest-path engine/Cargo.toml --lib`: **338 passed, 0
  failed, 3 ignored** (334 baseline + 4 new: `push_rejects_nan_outcome`,
  `push_rejects_inf_outcome`, `push_rejects_bad_stone_player`,
  `legit_push_unaffected_by_outcome_and_stone_player_guards`).
- `maturin develop --release` (`.venv`): clean rebuild, `engine-0.1.0`
  installed editable.
- `.venv/bin/python -m pytest tests/training/test_gnn_hexg_buffer.py
  tests/test_orchestrator_gnn_buffer.py -q`: **20 passed** (14 in
  `test_gnn_hexg_buffer.py`, up from 9 pre-fix: +5 new ADV-A/B/D cases incl.
  parametrizations; 6 unchanged in `test_orchestrator_gnn_buffer.py`).
- `.venv/bin/python -m pytest tests/selfplay/test_gnn_record_dispatch.py -q
  -m integration`: **1 passed** (~12s), now also asserting ADV-C finiteness
  on both the drained-row and sampled-target outcome fields.
- Broader graph-suite sanity (`test_gnn_hexg_buffer.py
  test_orchestrator_gnn_buffer.py test_graph_collate.py test_gnn_seam_smoke.py
  test_orchestrator_gnn_build.py -m "not slow"`): **44 passed**, all
  pre-existing WP-3/WP-5a suites still green.
- **Red-team probe re-run** (byte-identical scripts from
  `/tmp/…/scratchpad/wp5bA_redteam/`, run against the freshly-built `.so`,
  `PYTHONPATH=.`, no source touched):
  - `probeA_guards.py`: `push.outcome_nan` / `push.outcome_inf` /
    `push.stone_player_0` / `push.stone_player_5` (and `stone_player_-3` in
    §3) now **REJECTED** (`ValueError: push_graph_position: outcome NaN is
    not finite…` / `…stone (0, 0) has invalid player 5 (must be +1 or -1)`)
    — the script's own `expect_reject=False` baked-in labels these "GAP" in
    its self-scored summary (stale — it encodes the PRE-fix expectation),
    but the raw REJECTED/ValueError lines are the ground truth the task
    asked to confirm.
  - `probeC2_deterministic.py` (deterministic homogeneous-buffer tamper,
    forces the sample lottery): `outcome_nan`, `outcome_inf`,
    `stone_player_5` all now `HELD-at-push` (previously `GAP`/silent
    NaN-loss). `outcome_1e30` (finite, extreme) still `GAP` by design — out
    of the stated fix scope (non-finite only, per spec); `visit_illegal_far
    cell`/`visit_dup_coord` still `HELD-at-sample/collate` (pre-existing
    B2 mass-drop guard, untouched) — no regression.

All 4 fix-pass files (Rust push.rs guard, Rust tests, 2 Python test files)
touched; nothing `git add`ed per instruction.
