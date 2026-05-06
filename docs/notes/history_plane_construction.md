# History-Plane Construction — Ground Truth for §121

**Date:** 2026-04-24  
**Sources:** `engine/src/board/state.rs` (lines 186–520), `hexo_rl/env/game_state.py` (lines 139–242)

---

## Q1: When the window re-centers, does each history plane use the current centroid or the centroid at its own ply?

**Each history plane uses the centroid at its own ply — not the current centroid.**

The centroid is baked into the snapshot at move time. `GameState.from_board()` calls `rust_board.get_cluster_views()` immediately, which maps every stone through `window_flat_idx_at(q, r, cq, cr)` using the board's bounding-box centroid at *that moment*. The resulting `views` arrays (`List[np.ndarray]`, each shape `(2, 19, 19)`) are stored frozen on the `GameState` dataclass.

`to_tensor()` then assembles history planes by indexing `prior.views[k]` directly:

```python
# game_state.py:237-239
if k < len(prior.views):
    tensor[k, t]     = prior.views[k][0]   # prior my-stones, prior centroid
    tensor[k, 8 + t] = prior.views[k][1]   # prior opp-stones, prior centroid
```

No re-projection into the current coordinate frame occurs. The pixel at position `(wq, wr)` in history plane `t` means "stone at (current_center + wq - HALF, current_center + wr - HALF) at ply `now - t`" only if both plies share the same centroid. Otherwise the pixels are semantically misaligned across time.

**Consequence for §121:** As the game grows and the centroid drifts, successive history planes occupy different absolute coordinate frames. A stone at the same (wq, wr) cell represents a different absolute position in plane 0 vs plane 3 if the centroid shifted between those plies.

---

## Q2: Are older stones that drifted outside the current window clipped in history planes, or preserved?

**Clipped — and the clipping happens at snapshot time, not at tensor-assembly time.**

`to_planes()` / `get_cluster_views()` in Rust filters every stone through:

```rust
// state.rs:468-474
let flat = self.window_flat_idx(q, r);
if flat < TOTAL_CELLS {
    if cell == my_cell { planes_2[flat] = 1.0; }
    else if cell == opp_cell { planes_2[TOTAL_CELLS + flat] = 1.0; }
}
// flat == usize::MAX → silently dropped
```

`window_flat_idx_at` returns `usize::MAX` for any stone outside the 19×19 window centred on the current bounding-box at capture time. That stone is permanently absent from the stored view.

Because the snapshot is frozen at capture time, history plane `t` only contains stones that were inside *that ply's* window. Stones that have since moved into the window (from new play in the same region) cannot retroactively appear in old history planes, and stones that were in an earlier window but now lie outside the current window cannot be recovered.

**Consequence for §121:** Early-game stones that anchored in a corner, then got left behind as the bounding box expanded, may have been clipped from their own history plane if the game already drifted at the time of capture. The NN cannot recover them.

---

## Q3: Is plane 16 (moves_remaining) broadcast uniformly, or position-dependent?

**Strictly uniform broadcast — no positional information.**

Rust path (`encode_state_to_buffer`, state.rs:431-435):

```rust
let mr_val = if self.moves_remaining == 2 { 1.0 } else { 0.0 };
for i in 0..TOTAL_CELLS {
    out[16 * TOTAL_CELLS + i] = mr_val;
}
```

Python path (`to_tensor`, game_state.py:218-220):

```python
mr_val = np.float16(0.0 if self.moves_remaining == 1 else 1.0)
tensor[:, 16, :, :] = mr_val
```

Both paths assign the same scalar to every cell in the plane. The flag is a 1-bit global state encoded as {0.0, 1.0}: 1.0 means the current player is starting their 2-move turn (both moves remaining), 0.0 means they have already placed one stone and must place their second.

Note the scalar encoding is inverted between Rust and Python comments (`moves_remaining == 2` in Rust, `moves_remaining != 1` in Python) but they are arithmetically identical: the flag is 1 iff `moves_remaining == 2`.

Similarly, plane 17 (ply parity) is `ply % 2` broadcast uniformly across all TOTAL_CELLS — also purely scalar, position-independent.

---

## Q4: Are planes 0-7 and planes 8-15 synchronized to the same plies, or offset?

**Synchronized — both come from the same prior `GameState` object at each timestep.**

`to_tensor` inner loop (game_state.py:233-239):

```python
for t in range(1, HISTORY_LEN):
    if t > len(history):
        break
    prior = history[-t]           # single object for timestep t
    if k < len(prior.views):
        tensor[k, t]     = prior.views[k][0]  # my stones at t
        tensor[k, 8 + t] = prior.views[k][1]  # opp stones at t
```

`prior.views[k][0]` and `prior.views[k][1]` are the two planes of the same 2-plane snapshot captured at the same ply. They share the same window centroid, the same cluster assignment, and the same set of clipped/included stones. There is no offset between the current-player history and the opponent history.

**Plane-pair mapping:**

| Plane index | Content |
|-------------|---------|
| 0 | Current player's stones at ply `t` (now) |
| 1 | Current player's stones at ply `t-1` |
| … | … |
| 7 | Current player's stones at ply `t-7` |
| 8 | Opponent's stones at ply `t` (now) |
| 9 | Opponent's stones at ply `t-1` |
| … | … |
| 15 | Opponent's stones at ply `t-7` |
| 16 | `moves_remaining == 2` (broadcast scalar) |
| 17 | `ply % 2` (broadcast scalar) |

**Edge case — cluster count mismatch:** If the game had fewer active clusters at ply `t-k` than it does now, `prior.views` will have fewer entries. The `if k < len(prior.views)` guard leaves those history planes as zero rather than raising. Both the current-player and opponent planes for that timestep are zeroed together, so they remain synchronized (both zero).

---

## Summary for §121 intervention choice

| Property | Fact |
|----------|------|
| Re-centering reference | Each history plane uses its own ply's centroid — **not** the current centroid |
| Out-of-window stones | Clipped at snapshot time; irrecoverable in history planes |
| Plane 16 encoding | Uniform scalar broadcast; zero spatial structure |
| Plane 0-7 vs 8-15 alignment | Synchronized: same ply, same centroid, same snapshot per timestep |

The dominant invariance failure for §121 is **inter-ply centroid drift**: history planes are not spatially aligned to the current frame, so the NN must learn to tolerate sliding coordinate frames across the temporal depth dimension. Any intervention that re-projects prior snapshots into the current centroid would break backward-compatibility with the replay buffer and the Rust self-play path, both of which assume frozen views.
