# `hexo` Python Package Evaluation (v0.2.0)

## Overview
`hexo` is a community-contributed Python engine for Hex Tac Toe on an infinite hex grid.

## Findings

### 1. Turn Structure (1-2-2-2)
- **Pass.** `Hexo.__init__` automatically places `P1` at `(0,0)` and sets `_to_move = Player.P2`.
- Subsequent `push()` calls expect two stones per player, then switch sides.
- Correctly handles the 1-2-2-2 turn structure from the start.

### 2. 3-Axis Win Detection
- **Pass.** Uses axial axes `((1, 0), (0, 1), (1, -1))`.
- Win length is configurable (default 6).
- Uses `_has_winning_line` after each placement to check for terminal state.
- **Limitation:** Does not use bitboards (uses `set` and directional scan), but for Python-side utilities, this is fine.

### 3. BKE Notation Parsing
- **Fail.** No native BKE parser found in the package (as of v0.2.0).
- Only supports custom JSON state: `{"turns": [[ [q1,r1], [q2,r2] ], ...], "pending": [[q3,r3], ...]}`.
- We will still need our own `BKEParser` in `python/bootstrap/`.

### 4. Infinite Board Handling
- **Pass.** Implements `placement_radius` (default 8).
- Only considers empty cells within distance 8 of any existing stone as legal.
- Matches the community rules for "infinite" board growth.

### 5. Coordinate System
- **Pass.** Uses axial `(q, r)` as `Coord = tuple[int, int]`.
- Matches our `native_core` representation.

## Decision Rule
Integrate `hexo` for:
- Validating scraped games from archives.
- Managing game state during corpus conversion.
- Serializing/Deserializing games to JSON.

We will keep our `native_core` Rust board for performance-critical MCTS and training, but use `hexo` for Python orchestration.
