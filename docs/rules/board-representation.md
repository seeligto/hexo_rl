# Board representation — infinite board strategy

The board is infinite. The NN requires fixed-size tensors. We resolve this as follows:

**Internal storage (Rust):** `HashMap<(q,r), Player>` — sparse, genuinely unbounded.
No allocation for empty cells. No fixed grid size in the data structure.
**Transposition Table (TT):** Uses `FxHashMap` with **128-bit Zobrist hashing** (splitmix128) for O(1) state lookups, critical for MCTS efficiency. 128-bit keys eliminate collision risk at sustained >150k sim/s throughput.

**NN view window (Hybrid Attention-Anchored Windowing):** The board state is dynamically grouped into K distinct clusters (colonies) of stones. The Rust core returns K distinct **2-plane (19×19) cluster snapshots** (current player + opponent stones). Python's `GameState.to_tensor()` stacks these snapshots with `move_history` to assemble the 8-plane wire-format tensor (or 18-plane history tensor for Python history-aware callers, sliced at consumer site via KEPT_PLANE_INDICES). To prevent "Attention Hijacking" (where the model ignores distant but winning threats), we use **Attention-Anchored Windowing**: windows are centered on high-attention regions and critical formations, not just centroids.

**K-aggregation invariant** — applies at every live inference and replay-push site; aug-only fixtures are the sole exception:

- *Live MCTS forward (worker_loop.rs:299–401):* all K cluster views → NN → **min-pool on value**, **scatter-max on policy** (each legal move takes the maximum prior across all clusters that cover it). Per-worker, every ply.
- *Replay buffer push (worker_loop.rs:649–682):* one row per cluster per ply. K-fold expansion in the buffer; trainer sees per-cluster rows independently.
- *Aug-only sites* (RandomBot validation in pretrain.py, early-game monitoring probe, pass slot in records.rs): pick cluster index 0 deliberately as a fixture choice — shallow rollouts keep all stones in the origin window, so cluster 0 is the only relevant view. Not a boundary bug. Sprint §164 P1.
