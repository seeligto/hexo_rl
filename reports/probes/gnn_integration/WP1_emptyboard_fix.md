# WP-1 empty-board launch-blocker fix — AS-BUILT

**Ticket:** `hexo_graph::legal_moves_from_stones` (`engine/hexo-graph/src/lib.rs`)
has no empty-board fallback — a 0-stone board yields ZERO legal graph nodes,
so live graph MCTS raises `EmptyLegalSet` at game 0 and organic graph
self-play (`random_opening_plies=0`) cannot start.
Discovery: `reports/probes/gnn_integration/WP5b_commitA_impl.md`
("Concerns / findings"). Workaround it replaced:
`tests/selfplay/test_gnn_record_dispatch.py`'s `random_opening_plies=2`.

## Dense reference (authority for the fix)

`Board::legal_moves_set()`, `engine/src/board/moves.rs:90-146`:

- **0-stone branch** (`moves.rs:95-101`): hardcoded literal region —
  `(dq, dr)` for `dq, dr ∈ -2..=2` (a 5×5 **rectangle**, 25 cells) around the
  absolute origin. **Independent of `radius`** — this branch never reads
  `self.legal_move_radius` at all (the comment: "Empty board: 5×5 region,
  same as `Board::new()` init").
- **General (≥1 stone) branch** (`moves.rs:102-146`): for every existing
  stone, all empty cells within hex-distance ≤ `self.legal_move_radius`
  (default 5, `DEFAULT_LEGAL_MOVE_RADIUS`, jittered per game in some
  configs). Implemented as an axial hex-ball loop: `dq ∈ -r..=r`,
  `dr ∈ [max(-r, -r-dq), min(r, r-dq)]` — equivalent to the standard
  `|dq|,|dr|,|dq+dr| ≤ r` hex-ball constraint.

**1-stone verification (per task instruction, done before assuming
correctness):** confirmed the 1-stone case falls into the *general* branch
(not the empty-board special case — `self.cells` is non-empty with 1 stone),
and that `hexo-graph`'s existing (unmodified) `legal_moves_from_stones`
general path already implements the *same* hex-ball formula (`dq.abs().max(
dr.abs()).max((dq+dr).abs()) <= radius`, offsets over `stones`) — same
shape, proven equal by a new Rust unit test that independently reimplements
dense's own loop and diffs the resulting cell sets
(`single_stone_legal_set_matches_dense_ball_formula`, both green). The only
gap was the 0-stone special case, which the general branch cannot produce
(there is no stone to iterate over).

## Implementation

**`engine/hexo-graph/src/lib.rs`** — `legal_moves_from_stones`: added an
`if stones.is_empty()` branch that emits the *exact* dense 5×5 rectangle
(`dq, dr ∈ -2..=2`, 25 cells, sorted), returning before the general
per-stone hex-ball loop. Radius-independent, matching dense. The general
branch (≥1 stone, i.e. every existing byte-exact-parity-tested position) is
byte-for-byte unchanged. Updated `verify_contract`'s stale `EmptyLegalSet`
escape-hatch comment (previously asserted the empty board was the *only*
legitimately-empty-legal-set input — no longer true; the `|| n_stones == 0`
disjunct is now vacuous-but-harmless defensive code, not load-bearing).

**Downstream fix required to actually clear the mandatory e2e gate**
(`engine/src/replay_buffer/hexg/sample.rs`): the dense-mirrored 5×5
**rectangle** is closed under only 4 of the 12 `rotate_axial` D6 elements
(identity + 180°-rotation family, `sym ∈ {0,3,6,9}`; verified by direct
enumeration — the other 8 map it to a *different* 25-cell rectangle,
whereas a true hex-ball would be closed under all 12). `HexgBuffer::
sample_graph_batch_impl`'s rebuild-at-sample path rotates the *stored visit
keys* by a random `sym` but rebuilds the graph from the (necessarily
un-rotatable, since it's empty) stone list, so for `sym ∉ {0,3,6,9}` a
rotated visit coordinate could land outside the un-rotated rebuilt legal
set — silently dropping mass, caught loudly by the existing `mass_drop_check`
contract guard (`ValueError: HEXG sample: visit mass dropped at
sample-align...`). This code path was **unreachable before this fix**
(0-stone records could never be pushed — the game never got past
`EmptyLegalSet`), so it's a newly-surfaced consequence of fixing the primary
bug, not a pre-existing regression. Fix: force `sym = 0` (identity) whenever
`rec.stones.is_empty()` — an empty board carries no orientation information
to exploit via augmentation anyway, so this loses no training signal.
Minimal, 1-line-of-logic change plus a documentation comment; does not touch
`hexo-graph`'s crate/semantics.

## Test evidence

**Rust (`cargo test -j4`, workspace root — `engine` + `hexo-graph`):**
339 lib tests passed (was 338 baseline pre-fix; +1 new:
`empty_board_record_survives_d6_augmented_sample_align` in
`replay_buffer::hexg::tests`), 0 failed, 3 ignored. `hexo-graph` crate: 8/8
lib tests passed (was 6; +2 new: `empty_board_matches_dense_5x5_fallback`
[replaces `empty_board_builds_dummy_only`, now assert-correct for the fix],
`single_stone_legal_set_matches_dense_ball_formula`). All integration-test
binaries green. `cargo clippy --lib --tests -p engine -p hexo-graph`: 0 new
warnings in any touched file (`engine/hexo-graph/src/lib.rs`,
`engine/src/replay_buffer/hexg/{sample,tests}.rs`) — pre-existing warning
count elsewhere unaffected.

**`make check.wasm`:** green, both before and after the fix (`cargo check -p
hexo-graph --no-default-features --features wasm --target
wasm32-unknown-unknown`) — the fix is pure `Vec`/integer arithmetic, no new
std-thread/rayon/PyO3 surface.

**Byte-exact parity suite** (`tests/test_hexo_graph_parity.py`, the
1,696-position suite from commit `d5a64f7`, `wpa_positions.json`-sourced):
**PASS, n=1696, ints byte-asserted, max_float_diff=0.00e+00.** Edge-count
delta vs the pre-fix baseline run: `17743286` vs `17742046` = **+1240 = 4 ×
310** — exactly the 4 pre-existing empty-board degenerate inputs (`cp ∈
{1,-1} × mr ∈ {1,2}`, already present in `_build_inputs()` per WP-1 review
MUST-FIX #2) going from 0 legal/0 edges to 25 legal/310 edges, and *nothing
else* — direct evidence the fix did not perturb any of the 1692 stone-bearing
positions (which remain byte-exact against the Python oracle, unchanged code
path). The test was edited to carve the `n_stones == 0` inputs out of the
Python-oracle strict-equality comparison (the oracle's `legal_moves_from_stones`
has no empty-board case — it's untested/out-of-domain for that fn's one real
caller, `strix_v1_bot.py`, which short-circuits its own opening move before
reaching the graph builder) and instead assert the dense-mirrored 25-cell
shape directly, so the regression class (a re-introduced vacuous empty-board
legal set) still fails the suite. The 1-stone degenerate inputs (8 of the
1696) remain fully oracle-byte-exact (untouched code path). No new positions
needed to be added — the existing degenerate fixtures already covered both
classes; they were mis-encoding the pre-fix (broken) behavior as "correct"
and now correctly enforce the fixed (dense-mirrored) behavior.

**e2e proof** (`tests/selfplay/test_gnn_record_dispatch.py -m integration`,
parametrized `random_opening_plies ∈ {0, 2}`): **2 passed** (organic-ply0,
workaround-ply2), stable across 4 repeated full runs (no flakes). `ply0`
proves organic graph self-play now starts a game from an actual empty board
end-to-end: real MCTS root expansion → record → drain → push →
persist round-trip → sample (D6-augmented) → collate → `GnnNet.forward_batch`
→ finite losses. Before the `sample.rs` follow-on fix, `ply0` failed inside
`sample_graph_batch` with the `mass_drop_check` `ValueError` described above
— confirms the follow-on fix was necessary, not incidental.

**Mandated Python suites, all green:**
- `tests/selfplay/test_gnn_record_dispatch.py -m integration`: 2 passed.
- `tests/training/test_gnn_hexg_buffer.py tests/test_orchestrator_gnn_buffer.py`:
  20 passed.
- `tests/selfplay -k "contract or adv"`: 10 passed, 46 deselected.
- `tests/selfplay/test_graph_collate.py tests/training/test_gnn_hexg_buffer.py
  tests/selfplay/test_gnn_seam_smoke.py tests/test_orchestrator_gnn_buffer.py
  tests/test_orchestrator_gnn_build.py -m "not slow"`: 44 passed.
- Dense byte-identical regression sweep (`test_worker_pool.py`,
  `test_selfplay_runner_encoding_e2e.py`, `test_selfplay_encoding_aware.py`,
  `test_v6w25_microsmoke.py`, `test_rotation_buffer_compat.py`,
  `test_selfplay_registry_plumbing.py`, `test_pool_inference_pool_size.py`):
  33 passed, 3 skipped, 1 xpassed (pre-existing skip/xpass, unrelated).
- `tests/training tests/selfplay tests/model -m "not slow and not integration"`:
  247 passed, 3 deselected, 0 failures.

## Files touched

**Rust:** `engine/hexo-graph/src/lib.rs` (empty-board fallback +
`verify_contract` comment fix + 2 new unit tests, replaced 1 stale test),
`engine/src/replay_buffer/hexg/sample.rs` (D6-augmentation identity guard
for empty-stone records), `engine/src/replay_buffer/hexg/tests.rs` (1 new
regression test).
**Python (tests only):** `tests/selfplay/test_gnn_record_dispatch.py`
(parametrized `random_opening_plies ∈ {0, 2}`, docstring updated to
FORMERLY/FIXED), `tests/test_hexo_graph_parity.py` (carved `n_stones == 0`
inputs out of oracle-equality assertions into dense-shape assertions,
docstring updated).
**Report:** this file. Nothing git-added per instructions.
