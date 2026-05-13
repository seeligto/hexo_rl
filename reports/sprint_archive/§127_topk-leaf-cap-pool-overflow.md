<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §127 — Top-K leaf cap eliminates MCTS pool overflow — 2026-04-28

**Files:** `engine/src/mcts/mod.rs`, `engine/src/mcts/backup.rs`,
`engine/tests/pool_overflow.rs`.

### Why

5090 96-thread sweep (v2/v3 prompts) saw `mcts_pool_overflow_count > 0`
on every cell of the bimodal-retry grid. Root cause: leaf expansion
created one child per legal move. Empty-board legality is 25 cells, but
once a game has 100+ stones spread out the radius-8 hex ball per stone
unions into 1k+ legal cells. Worst-case nodes consumed per search =
`n_simulations × leaf_batch × n_legal`, which blew past `MAX_NODES = 1M`
on long games.

The pre-existing mitigation (§prior — fabricate `is_terminal=true` with
quiescence-corrected NN value, AtomicU64 counter for visibility) was a
hot-path data-corruption sink: every overflow biased visit counts and
value targets without surfacing the issue, and the bench had to drop
contaminated runs after the fact rather than the engine refusing to
generate them.

### What

`MAX_CHILDREN_PER_NODE = 192` (public const in `mcts/mod.rs`).
`expand_and_backup_single` now sorts legal moves by NN policy prior
descending (tie-break `window_flat_idx` ascending — deterministic
regardless of `FxHashSet` iteration order) and takes the top K. Fast
path with no sort when `legal_moves.len() ≤ K` preserves pre-cap
behaviour at the empty-board / early-game regime where K is never
binding.

The fabricated-terminal overflow path is removed entirely. If overflow
still fires (it cannot under the bench config: `400 sims × 8 batch × 192
≈ 614k slots, fits 1M`), the counter increments for telemetry, then the
function panics. Silent corruption is no longer possible.

Q40 (subtree reuse) interaction: K is per-node, not per-tree. Children
identity is stable across re-roots since the chosen top-K set is
determined by local policy + flat_idx, both invariant under root
rotation. Documented in the const doc-comment.

### Bound calculation

```
nodes per search ≈ n_sims × leaf_batch × MAX_CHILDREN_PER_NODE
                = 400 × 8 × 192
                ≈ 614k
MAX_NODES = 1_000_000   →   ~38 % headroom
```

K can drop to 128 once threat-probe shows no regression at the lower
cap (would lift headroom to ~59 %).

### Tests

* `engine/src/mcts/mod.rs::tests::test_topk_truncates_at_max_children` —
  600-cell fixture (200 in-window + 400 out-of-window). Sort path
  selects exactly K, all in-window, monotonic priors.
* `engine/src/mcts/mod.rs::tests::test_topk_tie_break_by_flat_idx` —
  K+1 cells with identical priors. Highest flat_idx is the dropped
  one; deterministic regardless of HashSet iteration order.
* `engine/src/mcts/mod.rs::tests::test_topk_fast_path_keeps_all_when_under_cap` —
  50-cell fixture. `sort_used == false`; all 50 cells appear in
  output.
* `engine/tests/pool_overflow.rs::topk_eliminates_pool_overflow_across_full_game` —
  drives 200 plies of self-play with `n_sims=400, leaf_batch=8`,
  uniform priors. Asserts `pool_overflow_count() == 0` and no node
  exceeds K children.
* `engine/tests/pool_overflow.rs::normal_sized_pool_does_not_overflow_on_empty_root` —
  sanity: default-sized pool expands the root cleanly.

Old fabricated-terminal regression tests deleted — they intentionally
triggered overflow on a tiny pool to validate the `is_terminal=true`
shortcut, which no longer exists.

### Bench result

`make bench` (n=5, 120 s pool, --no-compile): `mcts_pool_overflows_total
median=0, per_run=[0,0,0,0,0]` — top-K cap holds across the full bench
window. (Pos/hr re-baseline is **out of scope** for this commit; lands
separately once the new const stabilises in the bench harness for a few
runs.)

### Out of scope (intentional)

* Pos/hr / NN throughput re-baseline — deferred to a separate commit so
  the K=192 baseline is anchored on its own rather than mixed with this
  semantic change.
* Cleanup of the AtomicBool warned-flag pattern elsewhere in the
  codebase — separate audit pass.
* `docs/rules/perf-targets.md` updates — gated on the rebaseline.
* Q40 subtree reuse implementation — only the top-K interaction is
  documented (commented in `MAX_CHILDREN_PER_NODE` doc).
* Channel-drop re-run — separate scope.

---

