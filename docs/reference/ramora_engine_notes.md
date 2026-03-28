# Ramora0 Engine Notes

Source: `vendor/ramora_engine/cpp/engine.h` (read 2026-03-28).

---

## Summary

The Ramora0 engine (`opt::MinimaxBot`) is a pure C++ iterative-deepening alpha-beta minimax
engine with quiescence search, a transposition table, and a learned pattern-based evaluation
function. It is the strongest known public bot for Hex Tac Toe and is used as the bootstrap
corpus source for this project.

---

## Board representation

- Axial (q, r) hex coordinates. The board is conceptually infinite but practically bounded
  to the range [-70, 69] per axis (140×140 flat arrays with offset `OFF=70`).
- Each cell stores `int8_t`: `P_NONE=0`, `P_A=1`, `P_B=2`.
- "Turns" are pairs of placements `(move1, move2)` because players place 2 stones per turn
  (except the very first turn where player A places 1 stone).

## Input interface

The engine is called via `get_move(const GameState& gs)`:

```cpp
struct GameState {
    struct Cell { int q, r; int8_t player; };
    std::vector<Cell> cells;   // all placed stones
    int8_t cur_player;         // P_A (1) or P_B (2)
    int8_t moves_left;         // 1 (first turn) or 2 (normal)
    int    move_count;         // total stones on board
};
```

The Python wrapper (`ai_cpp.cpp`) extracts this from the Python game object and passes it in.

## Output interface

```cpp
struct MoveResult {
    int q1, r1, q2, r2;   // two moves
    int num_moves;          // 1 (opening / edge-case) or 2 (normal turn)
};
```

The engine always returns a full turn (two cell coordinates). The caller uses `num_moves` to
know whether the second move is valid.

## Pattern-based evaluation

The evaluation function uses a learned pattern table loaded from `ai.py`:
- Windows of `eval_length` cells are scanned in all 3 hex directions.
- Each window is encoded as a base-3 number (0=empty, 1=P_A, 2=P_B).
- A pre-computed `_pv` array maps pattern index → score.
- The running eval score is maintained **incrementally** during make/undo — O(eval_length)
  per move, not O(board_size²).

## Search algorithm

1. **Iterative deepening**: starts at depth 1, increments until time expires or a win is found.
2. **Alpha-beta with TT**: flat_map keyed by Zobrist hash XOR'd with `cur_player` and
   `moves_left`. TT stores `(depth, score, flag, move, has_move)`.
3. **Move generation**:
   - Candidates are cells within distance 2 of any placed stone.
   - Scored by incremental `_move_delta` (pattern eval change).
   - Limited to `CANDIDATE_CAP=15` for inner nodes, `ROOT_CANDIDATE_CAP=20` for root.
   - Turns = all pairs of candidates → up to ~105/190 turns per node.
   - Optionally includes a "colony" candidate far from existing stones to explore new groups.
4. **Threat filtering**: if opponent has an immediate win threat, only moves that block it
   are considered (significantly prunes the tree).
5. **Instant win detection**: if current player has 5 stones in a 6-window with no opponent
   stones, that completing move is played immediately.
6. **Quiescence search**: extends to depth `MAX_QDEPTH=16` when threat cells exist, to avoid
   horizon effect near forced wins.
7. **History heuristic**: cells that caused beta cutoffs accumulate `depth²` history score,
   improving move ordering in subsequent searches.
8. **Time control**: checks every 1024 nodes; throws `TimeUp` exception on deadline, rolls
   back via saved state snapshot.

## Zobrist hashing

Uses splitmix64 with two separate tables per cell (`g_zobrist_a`, `g_zobrist_b`) for player
A and B respectively. The TT key also mixes in `cur_player` and `moves_left` to distinguish
same-board states reached with different move-remaining counts.

---

## Known bug — must fix before generating training data

**Reported by:** P_P (community)

**Location (as described in community docs):**
- Bug: `engine.h` line 1094 (the TT lookup block in `_minimax`)
- Fix: `engine.h` line 1265 (the TT write-back at end of `_minimax`)

**Bug description:**
In `_minimax`, when the search fails low (all moves fail to raise `alpha`), the variable
`best_move` may not have been set to a valid move, yet the TT write-back unconditionally
stores `has_move: true`:

```cpp
// Line 1265 — BUG: best_move may be uninitialized if value <= orig_alpha
_tt[ttk] = {depth, value, flag, best_move, true};
```

When `flag == TT_UPPER` (fail-low), the `best_move` field is the default-initialized
`Turn{}` (two `pack(0,0)` coordinates — the origin cell, which may not even be a valid
candidate). Subsequent iterations use this bogus move for TT move ordering, causing
incorrect evaluation chains that silently corrupt search results.

**The fix (one-liner):** Preserve the previous TT move when failing low, and only set
`has_move: true` when a real best move was found:

```cpp
// Line 1265 — FIXED
bool move_found = (value > orig_alpha);   // maximizing; minimizing uses value < orig_beta
_tt[ttk] = {depth, value, flag,
            move_found ? best_move : (has_tt_move ? tt_move : Turn{}),
            move_found || has_tt_move};
```

(The symmetric condition for the minimizing branch uses `value < orig_beta`.)

**Consequence for bootstrap:** Training on corpus generated from the unpatched engine
silently introduces incorrect position evaluations, degrading the value head. Apply the
fix before any corpus generation.

---

## Python wrapper interface

The pybind11 wrapper (`ai_cpp.cpp`) exposes `PyMinimaxBot`:

```python
from ai_cpp import PyMinimaxBot

bot = PyMinimaxBot(time_limit=0.1)        # seconds per move
moves = bot.get_move(game_object)          # returns list of (q, r) tuples
# moves[0] = first placement, moves[1] = second placement (if num_moves==2)
```

The wrapper imports Python's `ai` module to load pattern weights, and `game` module to
extract `GameState` from the Python game object. This tight coupling means the engine must
be built in-place with `setup.py build_ext --inplace` and called from within the
`vendor/ramora_engine/` directory.

## Our usage plan

We wrap the engine via a subprocess calling the community bot API (`HexTacToeBots`
tournament runner). See `docs/05_community_integration.md` section 2 for the
`RamoraEngine` Python wrapper. **Apply the line-1265 patch before building.**
