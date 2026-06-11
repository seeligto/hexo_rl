# Board representation — infinite board strategy

**Registry-aware update (§172):** encoding-derived constants
(`board_size`, `n_planes`, `cluster_window_size`, `plane_layout`,
`legal_move_radius`, `value_pool`, `policy_pool`, ...) now live in
`engine/src/encoding/registry.toml` and are accessed via
`hexo_rl.encoding.lookup(name)` /
`engine::encoding::lookup_or_panic(name)`. The conventions described
below remain accurate for v6 / v6w25 but are no longer hardcoded — see
`docs/designs/encoding_registry_design.md`.

**Multi-window selfplay operational (§173 α):** v6w25 K-cluster
selfplay is now operational on the Rust training path. All buffer
strides, sym tables, and aggregation helpers read geometry from
`RegistrySpec` at runtime. No scattered hardcodes remain in the
hot path. See `docs/07_PHASE4_SPRINT_LOG.md` §173 for bench gate
results and `docs/designs/encoding_alpha_multiwindow_selfplay_design.md`
for the K-cluster encoding implementation spec.

**Turn structure (2 stones per compound turn):** P1 opens with 1 stone
(ply 0), then both players alternate placing **2 stones per turn**.
`Board.moves_remaining` tracks the phase (2 = about to place stone 1, 1 =
stone 2; `state/core.rs:109`). The board is a stone *set* with a
commutative-XOR zobrist (`state/core.rs:523`), so a turn's two stones are
an **unordered pair** — `{A,B}` and `{B,A}` yield identical state/hash.
`apply_move` places one stone and performs no win check; win detection is
a separate per-stone call (a turn can win on its first stone). For how
MCTS / buffer / training treat the compound turn, see
`docs/rules/phase-4-architecture.md` "Compound-turn handling" and the
audit `docs/handoffs/compound_turn_pipeline_audit.md`.

The board is infinite. The NN requires fixed-size tensors. We resolve this as follows:

**Internal storage (Rust):** `HashMap<(q,r), Player>` — sparse, genuinely unbounded.
No allocation for empty cells. No fixed grid size in the data structure.
**Transposition Table (TT):** Uses `FxHashMap` with **128-bit Zobrist hashing** (splitmix128) for O(1) state lookups, critical for MCTS efficiency. 128-bit keys eliminate collision risk at sustained >150k sim/s throughput.

**NN view window (Hybrid Attention-Anchored Windowing):** The board state is dynamically grouped into K distinct clusters (colonies) of stones. The Rust core returns K distinct **2-plane (19×19) cluster snapshots** (current player + opponent stones). Python's `GameState.to_tensor()` stacks these snapshots with `move_history` to assemble the 8-plane wire-format tensor (or 18-plane history tensor for Python history-aware callers, sliced at consumer site via `spec.kept_plane_indices`). To prevent "Attention Hijacking" (where the model ignores distant but winning threats), we use **Attention-Anchored Windowing**: windows are centered on high-attention regions and critical formations, not just centroids.

**K-aggregation invariant** — applies at every live inference and replay-push site; aug-only fixtures are the sole exception:

- *Live MCTS forward (worker_loop.rs:319–411):* all K cluster views → NN → **min-pool on value**, **scatter-max on policy** (each legal move takes the maximum prior across all clusters that cover it). Per-worker, every ply.
- *Replay buffer push (worker_loop.rs:649–682):* one row per cluster per ply. K-fold expansion in the buffer; trainer sees per-cluster rows independently.
- *Aug-only sites* (RandomBot validation in pretrain.py, early-game monitoring probe, pass slot in records.rs): pick cluster index 0 deliberately as a fixture choice — shallow rollouts keep all stones in the origin window, so cluster 0 is the only relevant view. Not a boundary bug. Sprint §164 P1.
