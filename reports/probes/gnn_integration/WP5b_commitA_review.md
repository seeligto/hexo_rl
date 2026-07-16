# WP-5b COMMIT A — fresh-eyes review (live worker-loop graph-recording dispatch)

**Reviewer role:** independent verifier, did not author the change. Read-only on source.
**Inputs:** delta doc `docs/designs/gnn_wp5b_commitA_delta.md`, impl report
`reports/probes/gnn_integration/WP5b_commitA_impl.md`, diff patch (game_runner/**,
orchestrator.py, pool.py), new tests.

## VERDICT: **PASS** (5 non-blocking findings, all LOW/INFO)

All binding rulings honored, dense path byte-identical, row-count conservation pinned,
off-window reachability proven through the live path. No FIXES-REQUIRED. Findings F3/F4
are launch-config / WP-1 follow-up gates (pre-registered, not commit-A defects).

---

## Per-item status

| # | Check | Status |
|---|---|---|
| 1 | Dense byte-identical (Rust + pool.py + orchestrator) | **PASS** |
| 2 | Binding rulings (restore LOUD, game_id=-1, legal_set code-force, recency write-skip, hexg zero-diff) | **PASS** |
| 3 | Resume cross-format failure modes + drain row-count conservation | **PASS** |
| 4 | Declared deviations (positions_dropped reuse, no run4_gnn.yaml, div_ceil) | **PASS** (adjudicated below) |
| 5 | Off-window reachability via MovePolicy::Ls end-to-end | **PASS** |
| 6 | Test runs | **PASS** (counts below) |
| 7 | Test quality (finite losses, no over-fit) | **PASS** |

---

## 1. Dense byte-identical — PASS

**Rust.** Dense hot path structurally untouched:
- `play_one_move` record site: ONE `if is_graph {…} else { record_position(…) }`; else arm
  arg list verbatim. `is_graph` is a `WorkerGeometry` Copy bool set once per worker spawn.
- `run_one_game` finalize site: ONE `if is_graph { finalize_game_graph } else { finalize_game(…) }`;
  else arm arg list verbatim, `finalize_game` body untouched.
- `record_position_graph_dispatch` + `finalize_game_graph` are new `#[cold]`/no-dense-caller fns.
- `graph_records: Vec::new()` (zero-cap, no alloc) for grid; `graph_results` Mutex/VecDeque
  constructed unconditionally but only locked on the graph finalize branch — idle for grid.
- `WorkerGeometry` destructure gains `is_graph`; **INV25 byte-identity integration test still
  green** (3 passed) — the substring-anchored destructure pattern survives.
- No dense drop-counter / channel-capacity / stats-schema change. `collect_data`/`CollectDataOut`/
  `GameResultRow` unchanged.
- `finalize_game_graph` winner/`terminal_reason` classification (inner.rs:1788-1798) is a **byte-identical
  twin** of `finalize_game` (inner.rs:1648-1671) — same `u8::from(winning_cells.is_empty())` idiom,
  same `plies >= max_moves ? 2 : 3` draw split.

**pool.py.** Dense CNN block (`collect_data`/reshape/`push_many`/`recent_buffer.push`/`positions_pushed`)
moved verbatim into the `else` arm, one indent deeper. Common post-branch code
(`games_completed`/`x_wins`/…, `drain_game_results`, sims/sec, per-entry loop) unchanged and
representation-blind. See F1 for the single trivial deviation (`_in_ch` recompute).

**orchestrator.py.** P1 dense branch = `ReplayBuffer(...)` unconditional-equivalent; P2 dense branch
keeps swallow-and-warn verbatim; P3 dense `else` builds `RecentBuffer` verbatim.

## 2. Binding rulings — PASS

- **Restore LOUD (graph) / UNCHANGED (dense):** P2 `if isinstance(buffer, HexgBuffer): log.error(...); raise`
  then falls through to dense `log.warning("buffer_restore_failed")`. Graph re-raises; dense swallow
  intact. Unit-test-pinned both directions.
- **game_id=-1 untagged:** P5 `push_graph_position(*rec, game_id=-1)`; `collect_graph_data` returns a
  9-tuple excluding game_id. Matches Adjudication 2 — no `next_game_id` consumed on self-play write ⇒
  resume-collision-free by construction.
- **legal_set code-force, registry untouched:** R1 in `mod.rs`
  `s.is_graph() || matches!(s.policy_pool, LegalSetScatterMax)`. **`git diff --stat` on
  `engine/src/encoding/registry.toml` is EMPTY**; gnn_axis_v1 still `policy_pool="none"`,
  `representation="graph"`.
- **Recency write-side skip only:** P3 graph → `recent_buffer=None` + loud `recent_buffer_skipped_graph`;
  P5 graph branch never calls `recent_buffer.push`. Sample-side deferred to commit B.
- **Singular-loop drain, ZERO hexg diff:** **`git diff --stat -- engine/src/replay_buffer/hexg/` is
  EMPTY**. R8 bulk method not built (per ruling). pool.py loops singular `push_graph_position`.

## 3. Failure modes / conservation — PASS

- **Graph run vs dense file:** P2 re-raise + HexgBuffer magic LOUD-fail ⇒ aborts, cannot proceed
  silently. Pinned by `test_...graph_reraises_on_stale_dense_file`.
- **Dense run vs graph file:** dense swallow (pre-existing, acceptable per adjudication — dense keeps
  resilience). Not a new silent path.
- **Row-count conservation:** e2e asserts the full chain
  `len(rows) == sum(plies - random_opening_plies)` (drain==generated) → `buf.size == len(rows)`
  (pushed==drained) → `n_loaded == len(rows)` and `size == len(rows)` and `sum(hist) == size`
  (persist round-trip conserves). **pushed == drained == buffer-size is genuinely pinned.** Both drains
  happen post-`stop()` (workers joined) so no in-flight race.
- **No drain mis-route:** graph rows flow worker→`graph_results`→`collect_graph_data`→HexgBuffer; dense
  rows flow via `results`→`collect_data`→ReplayBuffer. Disjoint queues; `finalize_game_graph` pushes to
  `graph_results_queue`, `finalize_game` to `results_queue`. No crossing.

## 4. Declared deviations — adjudicated

- **positions_dropped reuse for graph backpressure — FINE.** A run is pure-graph or pure-dense; both
  meanings are "per-position rows dropped under backpressure", semantically consistent, never conflated
  within a run. Queue-cap parity unit-tested (`...caps_queue_at_results_queue_cap`, drop count exact).
  Minor: a graph drop is not separately labeled in stats (see F2).
- **No `run4_gnn.yaml` — FINE for commit A.** Wiring-only; no test consumes it (tests build inline
  `SelfPlayRunnerConfig`). Launch config is a later operator act (see F3 for the mandatory gate).
- **`plies.div_ceil(2).min(u16::MAX)` — FINE.** Equivalent to pool.py dense `(plies+1)//2` clamped to
  65535 for non-negative plies. Correct.

## 5. Off-window reachability — PASS

R1 forces `legal_set=true` for graph ⇒ `play_one_move` builds `MovePolicy::Ls` ⇒
`record_position_graph_dispatch` extracts `Ls` (the `Dense` arm is `unreachable!()`, panic-on-invariant-break)
⇒ `records::record_position_graph` records the full legal set by coord. e2e asserts
`total_off_window > 0` (visited cells outside the trunk window), proving the `records.rs:62` off-window
skip is NOT inherited on the live path. `MovePolicy::Ls` threaded end-to-end confirmed.

## 6. Tests run — all green

| Suite | Result |
|---|---|
| `tests/selfplay/test_gnn_record_dispatch.py -m integration` (e2e) | **1 passed** (~12s) |
| orchestrator P1/P2/P3 + contract + ADV + seam (`test_orchestrator_gnn_buffer` + `test_graph_collate` + `test_gnn_hexg_buffer` + `test_gnn_seam_smoke` + `test_orchestrator_gnn_build`) | **36 passed** |
| `cargo test -j4 --lib` | **334 passed**, 0 failed, 3 ignored |
| INV25 worker-loop byte-identity (`--test inv25_worker_loop_split_byte_identity`) | **3 passed** |
| Dense guard `tests/selfplay -m "not slow and not integration"` | **53 passed**, 2 deselected |

## 7. Test quality — PASS

- e2e drives finite losses end-to-end: `forward_batch` outputs + `ragged_policy_ce` + `binned_value_loss`
  all `torch.isfinite` asserted, plus `wire.builder_impl == 1` (F7 native-builder guard — catches the
  Python-builder trap on the sample leg).
- Not over-fit: off-window assertion is loose (`> 0`, flake-resistant), drain count tight (pins R7/P5),
  finalize unit tests assert exact outcome/value_valid per terminal_reason (pins the §178 split live-loop
  wiring). `unreachable!()`-as-proof for R1 is sound.
- **Known-gap workaround verified sound:** `random_opening_plies=2` routes around the pre-existing
  empty-board EmptyLegalSet gap. The test cannot silently pass with a broken dispatch — a broken R1 would
  panic (`unreachable!()`, `panic=abort`) before any game completes, and `len(rows) > 0` +
  `games_completed >= n_games` assertions fail hard otherwise.

---

## Findings (all non-blocking)

- **F1 (LOW / nit):** pool.py `_in_ch` is recomputed each stats-loop iteration inside the dense `else`
  arm (was hoisted once before the loop pre-commit). Value-identical (`self._feat_len`/`_trunk_size`
  are constant); `_run_stats_loop` is a periodic Python drain loop, NOT the self-play hot path. Cosmetic;
  no action needed.
- **F2 (LOW / note):** graph backpressure drops reuse `positions_dropped` with no distinct label, so a
  graph drop is indistinguishable from a dense drop in stats. Semantically consistent and runs are
  disjoint, so harmless now; add a distinct log only if graph-drop visibility later matters.
- **F3 (INFO / launch gate — pre-registered, not a defect):** no committed graph launch config. The
  eventual `run4` yaml MUST declare `recency_weight:0` (else recency silently no-ops until commit B lands
  the `recent_frac` sampler — delta §7/§12), namespace `buffer_persist_path` per §RUN3-STEP0 (so P2's
  loud-fail fires only on a genuine mismatch, not a sibling run's file), and set `bot_batch_share:0`.
- **F4 (INFO / pre-existing WP-1, ticketed separately):** `hexo_graph::legal_moves_from_stones` has no
  empty-board fallback ⇒ organic `random_opening_plies=0` graph self-play cannot start at move 0. Out of
  commit-A touch list; well-flagged. Launch config needs `random_opening_plies >= 1` OR a WP-1 patch.
- **F5 (LOW / optional hardening):** the three `getattr(spec, "is_graph", False)` sites default to False
  if the object ever isn't a PyO3 `RegistrySpec`. In practice the `window_set`/`_resolve_encoding_for_pool`
  seam always returns the PyO3 spec (whose `is_graph` is a real `#[getter]` bool — verified), so the
  default never fires; a wrong-type object would starve (empty buffer), not silently corrupt. Direct
  `spec.is_graph` (AttributeError-loud) would be marginally stronger but is not required.
