# ZOI Scope Audit — Static Analysis
**Date:** 2026-04-11  
**Investigator:** static read of engine source  
**Question:** Does ZOI filtering reduce the MCTS branching factor, or only filter the final move selection?

---

## Verdict

**ZOI is a post-search move-selection filter. It never enters the MCTS tree.**

The tree always expands with the full radius-8 legal set (up to 200+ cells in mid-game).
ZOI only restricts which of those legal moves can be *sampled* as the final chosen move
after MCTS completes. This directly confirms the depth-stuck hypothesis.

---

## 1. ZOI configuration

**File:** `configs/selfplay.yaml:59–61`

```yaml
playout_cap:
  zoi_enabled: true
  zoi_lookback: 16    # anchors: last N moves in game history
  zoi_margin: 5       # hex-distance threshold for inclusion
```

`hex_distance(q1,r1,q2,r2) = (|dq| + |dr| + |dq+dr|) / 2` — standard axial coords.

---

## 2. ZOI call sites in `engine/src/game_runner.rs`

### Struct fields (lines 194–196)

```rust
zoi_enabled: bool,
zoi_lookback: usize,
zoi_margin: i32,
```

Stored in `SelfPlayRunner`. Captured into per-worker-thread locals at lines 334–336.

### THE ONLY ZOI FILTER — lines 626–643 (post-search move selection)

```rust
// Line 626: ── Sample and apply move (ZOI-filtered legal set) ──
let full_legal = board.legal_moves();             // line 627 — ALL legal moves
if full_legal.is_empty() { break; }

let legal = if zoi_enabled && move_history.len() >= 3 {
    let filtered: Vec<_> = full_legal.iter()
        .filter(|(q, r)| {
            move_history.iter().rev().take(zoi_lookback).any(|(q0, r0)| {
                hex_distance(*q, *r, *q0, *r0) <= zoi_margin   // line 635
            })
        })
        .cloned()
        .collect();
    if filtered.len() < 3 { full_legal } else { filtered }   // fallback: line 640
} else {
    full_legal
};
```

**Timing:** executes AFTER `tree.expand_and_backup()` (the MCTS search call at line 438)
returns. The variable `legal` is only used downstream for move sampling (lines 645–673).

**What it filters:** the pool of moves available for final move selection. It does NOT
affect which nodes the MCTS tree visits, which children are created, or which positions
receive NN evaluations.

**No other ZOI call sites exist in `game_runner.rs`.**

---

## 3. MCTS child expansion — `engine/src/mcts/backup.rs`

### `expand_and_backup_single()` — lines 81–153

```rust
let legal_moves = board.legal_moves_set();    // line 103 — NO ZOI
if legal_moves.is_empty() { /* terminal */ return; }

let n_ch = legal_moves.len();                 // line 111 — all legal moves
for (j, &(q, r)) in legal_moves.iter().enumerate() {  // line 125
    self.pool[ci] = Node { ... };              // one child per legal move
}
```

**Finding:** MCTS creates one child node per legal move from `board.legal_moves_set()`.
No ZOI, no restriction. In a mid-game position with 10+ stones, this is 100–300 children.

---

## 4. Board legal move generation — `engine/src/board/moves.rs:29–80`

`legal_moves_set()` returns all empty cells within `LEGAL_MOVE_RADIUS = 8` hex-distance
of any existing stone. No ZOI parameter. No configurable filtering. Pure game-rule set.

On an empty board: 5×5 init region (25 cells).  
After a few moves: grows rapidly — each stone adds up to 217 new candidate cells.

---

## 5. Gumbel MCTS path — `engine/src/game_runner.rs:40–73`

### Root expansion (line 447)
```rust
let root_sims = infer_and_expand(&mut tree, 1);
```
Root is expanded with ALL legal moves before Gumbel candidate selection begins.

### `GumbelSearchState::new()` — lines 44–73
```rust
let children = tree.get_root_children_info();   // line 51 — ALL root children
let n_children = children.len();                // line 52
let effective_m = m.min(n_children);            // line 68
// Top-m selected from ALL n_children by gumbel_noise + log_prior score
let candidates: Vec<usize> = scored.iter().take(effective_m)...;  // line 73
```

**Finding:** `gumbel_m = 16` draws from ALL root children (200+), selecting the top-16
by gumbel noise + log prior. The remaining budget is concentrated on these 16 via
Sequential Halving. BUT:
- Root still has 200+ children created.
- Non-root expansion is still unbounded (PUCT, full legal set).
- ZOI not applied to the candidate pool.

---

## 6. Move selection post-MCTS (lines 645–673)

Final move sampled from `legal` (ZOI-filtered at lines 631–643).  
For Gumbel winner: verified against `legal.contains(&(mq, mr))` (line 660).  
If winner not in ZOI set: falls back to `sample_policy(&policy, &legal, &board)`.

This is where ZOI can silently override Gumbel's winner if the best Gumbel move
falls outside the ZOI radius — a subtle interaction worth monitoring.

---

## Summary table

| Component | File | Lines | ZOI Applied? | Branching factor affected? |
|---|---|---|---|---|
| `Board::legal_moves_set()` | `board/moves.rs` | 29–80 | ❌ | Source of truth: radius-8 |
| MCTS child expansion | `mcts/backup.rs` | 103 | ❌ | **Not reduced** |
| Gumbel root expansion | `game_runner.rs` | 447 | ❌ | All legal moves |
| Gumbel candidate pool | `game_runner.rs` | 51–73 | ❌ | m=16 of 200+ |
| **Post-search move filter** | `game_runner.rs` | **626–643** | **✅ YES** | Move selection only |
| Final move sampling | `game_runner.rs` | 645–673 | Uses filtered `legal` | Selection only |

**Bottom line:** The MCTS tree branches on ALL legal moves. ZOI does not reduce the
branching factor. 200 simulations spread across 200+ children produce near-zero depth
beyond the first level, with FPU+PUCT providing the only narrowing.
