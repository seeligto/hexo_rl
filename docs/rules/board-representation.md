# Board representation — infinite board strategy

The board is infinite. The NN requires fixed-size tensors. We resolve this as follows:

**Internal storage (Rust):** `HashMap<(q,r), Player>` — sparse, genuinely unbounded.
No allocation for empty cells. No fixed grid size in the data structure.
**Transposition Table (TT):** Uses `FxHashMap` with **128-bit Zobrist hashing** (splitmix128) for O(1) state lookups, critical for MCTS efficiency. 128-bit keys eliminate collision risk at sustained >150k sim/s throughput.

**NN view window (Hybrid Attention-Anchored Windowing):** The board state is dynamically grouped into K distinct clusters (colonies) of stones. The Rust core returns K distinct **2-plane (19×19) cluster snapshots** (current player + opponent stones). Python's `GameState.to_tensor()` stacks these snapshots with `move_history` to assemble the full 18-plane temporal tensor. To prevent "Attention Hijacking" (where the model ignores distant but winning threats), we use **Attention-Anchored Windowing**: windows are centered on high-attention regions and critical formations, not just centroids.

**Value Aggregation (Min-Pooling):** When multiple windows are evaluated for a single board state, the scalar Value ($v$) is aggregated using **Min-Pooling** (from the perspective of the current player) to ensure that if any window contains a losing threat, the entire state is treated as high-risk.

**Legal moves:** All empty cells within a margin of existing stones, across all K clusters. The network outputs K policy distributions which are mapped back to global coordinates and unified via softmax.
