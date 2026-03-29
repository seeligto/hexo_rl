# Evaluation of `hexo` Python Package

## Overview
The `hexo` package (v0.2.0) provides a fast, reusable engine for Hexo on an infinite hex grid. It handles core game logic, including turn enforcement, win detection, and move validation.

## Findings

### Turn Structure
- **Requirement:** X places 1 on turn 0, then 2-2-2.
- **Implementation:** `Hexo.new()` automatically places the opening stone for P1 (X) at `(0, 0)` and sets the next player to P2 (O). Subsequent calls to `push()` or `play()` enforce 2-stone turns.
- **Match:** Perfect.

### 3-Axis Win Detection
- **Requirement:** 6-in-a-row on any of the three hex axes.
- **Implementation:** Uses `AXES = ((1, 0), (0, 1), (1, -1))` for straight-line detection of `win_length = 6`.
- **Match:** Perfect.

### BKE Notation Parsing
- **Requirement:** Parse BKE (hexagonal-tic-tac-toe-notation).
- **Implementation:** Not natively supported in the `hexo` package v0.2.0. However, `Hexo.from_state()` supports a JSON-like state, and the core `push()` method can be used to replay moves parsed from BKE.
- **Match:** Partial (requires external parser).

### Infinite Board Handling
- **Requirement:** Handle theoretically infinite board.
- **Implementation:** Uses a `placement_radius` (default 8) to restrict moves to a valid area around existing stones. This prevents the search space from exploding while allowing the board to grow naturally.
- **Match:** Perfect.

### Coordinate System
- **Requirement:** Axial coordinates `(q, r)`.
- **Implementation:** Uses `Coord = tuple[int, int]` representing axial `(q, r)`.
- **Match:** Perfect.

## Decision
Integrate `hexo` into `python/bootstrap/` for:
1. Validating scraped games.
2. Managing game state during corpus conversion.
3. Serving as a ground-truth engine for `BotProtocol` wrappers if needed.

We will need to implement our own BKE parser to convert the notation into a sequence of `push()` calls.

## Integration Plan
- Use `hexo.Hexo` for game state management in `python/bootstrap/`.
- Implement BKE parser in `python/bootstrap/notation.py`.
