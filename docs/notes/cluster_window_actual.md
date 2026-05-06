# Cluster-Window Aggregation — ckpt_12190 Value Head

**Date:** 2026-04-24  
**Scope:** `hexo_rl/model/network.py`, `hexo_rl/selfplay/inference.py`,
`engine/src/board/state.rs`, `engine/src/board/moves.rs`

---

## (a) How K windows are selected per forward pass

Window selection happens in Rust via `Board::get_cluster_views()`, called from
`GameState.from_board()` on every board update.  The algorithm is two-phase:

**Phase 1 — cluster identification (`get_clusters()`).**
BFS over all occupied cells.  Two stones belong to the same cluster if their
hex-axial distance is ≤ 8.  Result: one or more disjoint stone groups.

**Phase 2 — center assignment.**  For each cluster:

- *Small cluster* (bounding-box span ≤ 15 in both q and r): one window,
  centered at the integer bbox centroid `((min_q+max_q)/2, (min_r+max_r)/2)`.
- *Large cluster* (span > 15 in either axis): one window per *anchor* inside
  the cluster.  Anchors come from two sources:
  - **Action anchors** — a ring buffer of the last 4 moves played (any player).
  - **Threat anchors** — centres of every open 3-in-a-row and 4-in-a-row
    formation, computed by scanning all stones on both axes.
  - Anchors within hex-distance 5 of an already-selected anchor are deduplicated.
  - Fallback to bbox centroid if no anchors fall inside the cluster.

Each selected center defines a 19 × 19 axial window.  The full batch of K
windows is passed to `LocalInferenceEngine.infer_batch()` as a single
`(K, 18, 19, 19)` float tensor.

---

## (b) Selection mechanism: centroid-based, learned-attention, or static-grid?

**Hybrid content-driven geometry — not learned.**

- Small clusters → geometric centroid of stone bounding box.
- Large clusters → discrete anchor set derived from recent move history and
  open-threat formation centres.

No parameters are learned for window placement.  There is no attention
mechanism, no neural routing, and no fixed grid of candidate centers.
Selection is a deterministic function of current board state.

---

## (c) Selection depends on board content vs fixed geometry

**Fully content-dependent.**

- Cluster membership changes with every stone placed (BFS over stone positions).
- Small-cluster centroid shifts whenever the bounding box changes.
- Large-cluster anchors change with each move (action anchor ring buffer) and
  whenever a 3-in-a-row or 4-in-a-row forms or is broken (threat anchors).

Fixed geometry plays no role: there is no pre-defined grid of candidate tiles.

---

## Value aggregation: min-pool in inference, not inside the NN

The NN value head itself is a pure global average-pool + global max-pool over
the 19 × 19 spatial grid of each individual window, followed by two FC layers
and tanh.  It knows nothing about K.

Min-pooling happens **outside** the model, in `LocalInferenceEngine.infer_batch()`
(`hexo_rl/selfplay/inference.py:77`):

```python
v = float(board_values.min())   # pessimistic: worst window is the board value
```

Policy aggregation also happens in the inference engine: for each legal cell,
max probability across all windows that contain that cell.

---

## D2 result: H2 **not implicated**

Because selection is content-dependent (not static-grid), H2 was testable.
D2 replayed all 50 `main_island_v1` fixture positions, called
`get_cluster_views()` on the live board, identified the largest hex-adjacent
threatening group of the current player, and computed stone-weighted coverage
(fraction of group stones inside any selected window).

Results (`reports/investigations/main_island_d2/`):

| Metric | Value |
|---|---|
| Positions analysed | 50 / 50 |
| K mean / max | 1.2 / 4 |
| Coverage mean / min | 1.000 / 1.000 |
| Positions with coverage < 50% | **0 / 50 (0%)** |
| H2 threshold | > 30% of positions failing |
| **H2 verdict** | **NOT IMPLICATED** |

At ply 21–39 (the fixture range), the main island spans well within 15 cells in
both q and r on every sampled position, so a single bbox-centroid window always
captures all stones.  The 4 positions that produced K > 1 (positions 15, 34,
41, 49) had detached secondary groups that triggered additional anchor windows,
but the main group remained fully covered.

**Structural conclusion:** the cluster-window mechanism does not systematically
exclude the largest threatening group at the game depths present in this fixture.
H2 (windowing-induced main-island blindness) is ruled out for ckpt_12190 on
`main_island_v1`.
