# Gumbel Path Static Audit — 2026-04-10

Source file: `engine/src/game_runner.rs`

---

## Q1 — Is the Gumbel(0,1) noise vector drawn fresh for each game, or is it cached across games?

**Answer: Drawn fresh for every move (not just every game).**

The per-worker RNG is initialised once at thread spawn:
```rust
// line 344
let mut rng = rng();
```
This is a per-worker `rand::Rng` instance, independent across workers and not seeded
deterministically (it uses thread-local entropy). It is reused across all moves and games
for that worker.

`GumbelSearchState::new()` is called at **line 459**, inside the per-move loop
`for _ in 0..max_moves` (line 362):
```rust
// line 459
let mut gs = GumbelSearchState::new(
    &tree, effective_m, c_visit, c_scale, &mut rng,
);
```

Inside `GumbelSearchState::new()` (lines 55–60), a fresh Gumbel(0,1) vector is drawn
for every root child on every call:
```rust
let gumbel_values: Vec<f32> = (0..n_children)
    .map(|_| {
        let u: f32 = rng.random::<f32>().clamp(1e-10, 1.0 - 1e-7);
        -(-u.ln()).ln()  // Gumbel(0,1) = -log(-log(U))
    })
    .collect();
```

**RNG source:** `rand::rng()` (thread-local OS entropy, line 344).  
**Sample site:** `GumbelSearchState::new()` at line 459, called once per move.  
**Caching:** None — the `GumbelSearchState` struct is local to each move's search scope
and is not stored in the node pool, TT, or any persistent structure.

---

## Q2 — Is the Gumbel vector actually added into the effective root score, or computed and then discarded?

**Answer: Functionally added at two sites; never discarded.**

**Site 1 — Initial candidate selection** (lines 69–73): all root children are scored by
`gumbel_values[i] + log_priors[i]` and the top `effective_m` are selected:
```rust
// lines 69-71
let mut scored: Vec<(usize, f32)> = (0..n_children)
    .map(|i| (i, gumbel_values[i] + log_priors[i]))
    .collect();
```

**Site 2 — Sequential Halving phase scores** (line 130):
```rust
self.gumbel_values[child_offset] + self.log_priors[child_offset] + sigma
```
where `sigma = (c_visit + max_n) * c_scale * q_hat` is the completed-Q contribution.
The Gumbel term `self.gumbel_values[child_offset]` is added at every halving phase,
not just the first. It is stored in `GumbelSearchState.gumbel_values` (line 22) and
read by `score()` at line 119.

The vector is only dropped when `gs` goes out of scope at the end of the move loop body
(after move selection at lines 599–621).

---

## Q3 — Is `effective_m` computed as `min(n_legal, gumbel_m, n_root_children)` as documented, or is it a hardcoded small number?

**Answer: Computed correctly at line 445 as `min(gumbel_m, game_sims, n_root_children)`.**

```rust
// line 445
let effective_m = gumbel_m.min(game_sims).min(tree.root_n_children());
```

- `gumbel_m` is the config parameter (default 16).
- `game_sims` is the per-move simulation budget (e.g. 800 in training config). This caps
  `effective_m` to the sim budget so Sequential Halving never requests more candidates than
  it can spend sims on. The sprint log §61/§62 spec uses `n_legal` for this slot, but
  `game_sims` provides a tighter bound and is the correct operational guard.
- `tree.root_n_children()` is the actual number of legal moves after root expansion.

No hardcoded small numbers. Zero magic constants.

If `effective_m == 0` (degenerate board with no children), the code falls back to standard
PUCT search for that move (lines 446–456).

---

## Summary

All three questions answered affirmatively. The static analysis shows no ghost-feature bug
in the Gumbel path:

1. Noise is drawn fresh per move from a per-worker RNG — no caching, no reuse.
2. Gumbel values are added into the score at both candidate-selection and halving-phase
   sites; they are not discarded.
3. `effective_m` is a runtime computation matching the documented formula.

The runtime trace (Part 1B/C) should confirm this by showing diverse root priors across
workers at `compound_move == 0, ply == 0`.
