# Community Integration — Ecosystem, Notation, Formations, Bot API

This document captures everything learned from the Hex Tac Toe community Discord and knowledge base that directly affects implementation. Do not treat this as background reading — several decisions here override or sharpen earlier architecture choices.

---

## 1. Community resources (confirmed active as of 2026-03-27)

| Resource | URL | Relevance |
|---|---|---|
| Strongest public engine | https://github.com/Ramora0/HexTicTacToe | **Bootstrap source** — use this, don't build minimax from scratch |
| Shared bot collection + tournament runner | https://github.com/Ramora0/HexTacToeBots | Integration target for evaluation |
| Bot API spec (v1) | https://github.com/hex-tic-tac-toe/htttx-bot-api/blob/main/definitions/bot-api-v1.yaml | Our bot must comply to participate in community tournaments |
| Notation standard | https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation | BKE format — needed for importing community games |
| Community GitHub org | https://github.com/hex-tic-tac-toe | Standards home |
| Game explorer | https://hex-tic-tac-toe.did.science/ | Human-playable reference |

---

## 2. The Ramora0 engine — bootstrap source

**Do not build our own minimax bot.** The Ramora0 C++ engine (`cpp/engine.h`) is already the strongest public bot, written by imaseal, and publicly available. Use it directly to generate the bootstrap corpus.

### Known bug — must fix before generating training data

**File:** `cpp/engine.h`, line 1094  
**Bug (reported by P_P):** The engine incorrectly treats a fail-low result as an exact score. When `old_alpha < v < alpha` (search fails low), the actual score is an upper bound — but the parent node, without proper alpha adjustments, treats it as exact. This can corrupt evaluations silently.

**Fix location:** line 1265 — preserve the old transposition table move when the best move is not found in the current search. This is a one-liner fix standard in chess engines.

**Consequence for bootstrap:** If we generate games from the unpatched engine, some positions will have incorrect evaluations baked into training data. Apply the fix before any corpus generation. The bug is documented and the fix is known — this is low-effort.

### How to use the engine

The engine exposes a standard interface. Wrap it behind the community Bot API spec (see section 4) and generate games via the tournament runner in `HexTacToeBots`. This is far faster than calling it through Python subprocesses naively.

```python
# python/bootstrap/ramora_wrapper.py

import subprocess
import json
from pathlib import Path

class RamoraEngine:
    """
    Wraps the compiled Ramora0 C++ engine via the community bot API.
    Assumes the engine binary is built at native_core/vendor/ramora/engine.
    Apply the line-1094 patch before building.
    """
    def __init__(self, binary_path: str, depth: int = 5):
        self.binary = Path(binary_path)
        self.depth = depth
        assert self.binary.exists(), f"Engine binary not found: {binary_path}"

    def get_move(self, board_state: dict) -> tuple[int, int]:
        """
        Send a board state dict (in bot-api-v1 format) and get back a move.
        """
        payload = json.dumps({
            "board": board_state,
            "depth": self.depth,
        })
        result = subprocess.run(
            [str(self.binary)],
            input=payload, capture_output=True, text=True, timeout=30,
        )
        response = json.loads(result.stdout)
        return response["row"], response["col"]
```

### Recommended depth mix for corpus generation

| Depth | Speed (approx) | Use |
|---|---|---|
| 3 | ~50,000 games/hour | Bulk corpus — tactical basics |
| 5 | ~5,000 games/hour | Quality games — stronger patterns |
| 7 | ~200 games/hour | Avoid — too slow, prior too strong |

Recommended split: **70% depth-3, 30% depth-5**. The depth-5 games provide stronger positional signal; the depth-3 games provide volume and diversity.

---

## 3. BKE notation

BKE (Boat-Kaitlyn-Epic) is the community standard for recording games and positions. It encodes positions relative to the board's center using **Ring** (letter) and **Offset** (number), which makes notation orientation-independent.

### Coordinate system

- Center tile = first X placement
- Ring A = first ring around center, Ring B = second, etc.
- Offset = distance from the 0-offset line, in the chirality direction that minimises offset values
- Optional sector form: `B2.1` = Ring B, sector 2, offset 1 within that sector

```
Examples:
  A0        = first ring, 0 offset (on a central line)
  A0 A2     = Closed Game opening (O's two moves)
  x A1 A4   = X's response (main line)
  B2.1      = same tile as B5 in full offset notation
```

### Parser notes

A TypeScript reference parser exists (`standardNotationParser.ts`, written by BSoD). Known issues:
- Does minimal validation — safe for client-side use only, vulnerable to malicious input server-side
- Has a comment typo (`s = -q - s` should be `s = -q - r`) but the code itself is correct
- Community parsers run to ~400 lines — don't underestimate this

We need a Python BKE parser for importing community game records. Write it in `python/bootstrap/bke_parser.py`. Test against the known opening table in section 5 of this doc.

```python
# python/bootstrap/bke_parser.py

from dataclasses import dataclass

@dataclass
class BKEMove:
    player: str       # 'x' or 'o'
    ring: str         # 'A', 'B', ..., 'AA', ...
    offset: int
    sector: int | None = None

def parse_bke_game(notation: str) -> list[BKEMove]:
    """
    Parse a BKE game string into a list of BKEMove objects.
    
    Input:  'A0 A2 x A1 A4 o B2 D16'
    Output: [BKEMove('o','A',0), BKEMove('o','A',2), BKEMove('x','A',1), ...]
    
    Note: 'o' is the first player (places first), 'x' is second.
    Current player is tracked implicitly — if no player prefix, 
    same player as previous move.
    """
    ...

def bke_to_axial(ring: str, offset: int, zero_line_direction: tuple) -> tuple[int, int]:
    """Convert BKE ring/offset to axial (q, r) coordinates."""
    ...
```

---

## 4. Bot API compliance

Our bot must implement the `htttx-bot-api-v1` spec to participate in community tournaments
via `HexTacToeBots`. The live spec is at `docs/reference/bot-api-v1.yaml`.

**Note (2026-03-28):** The live spec differs substantially from the earlier draft described
here. The notes below reflect the current `bot-api-v1.yaml`. The spec is still marked
v1-alpha / draft and may continue to evolve.

### What the live spec actually defines

The spec defines a single required endpoint and a capability schema:

```
GET /capabilities.json
  Returns a Capabilities object declaring which API modes the bot supports.
```

There is **no** `/move` endpoint, **no** `POST /move`, and **no** `tags` or
`supported_features` fields in the live spec. All of that was an earlier draft.

### Capabilities schema (from bot-api-v1.yaml)

```yaml
Capabilities:
  meta:                        # optional — human-readable bot info
    name: string
    description: string
    author: string             # single string, not array
    version: string            # semantic versioning preferred
  stateless:                   # optional — declare stateless HTTP support
    versions:
      v1-alpha:                # set this object to opt in
        api_root: string       # base path relative to capabilities.json
                               # defaults to "stateless/v1-alpha" if absent
        move_time_limit: bool  # if true, bot must respect time_limit in move requests
  basic_websocket:             # optional — declare real-time WebSocket support
    versions:
      v1-alpha:                # set this object to opt in
        api_root: string       # defaults to "bws/v1-alpha"
        move_time_limit: bool
        evaluation_time_limit: bool
        config:
          dynamic: bool        # supports config changes mid-game
        free_setup: bool       # supports arbitrary initial positions
        move_skips: bool       # client can make moves on bot's behalf
        dual_sided: bool       # bot can play both sides in one session
        free_move_order: bool  # client can request moves out of turn order
        evaluation: bool       # bot supports evaluation-only requests
        resettable_state: bool # client can resend setup packet at any point
        interruptable: bool    # bot supports interrupt packets
```

The move/turn endpoints themselves are defined by the versioned sub-APIs (`stateless/v1-alpha`
and `bws/v1-alpha`), not by this capabilities document. The stateless API's turn handle is
inferred to be at `{api_root}/turn`.

### Implementation plan

For Phase 6, implement the stateless capability first (simpler — pure HTTP, no WebSocket):

```python
# python/api/bot_server.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/capabilities.json")
def capabilities():
    return {
        "meta": {
            "name": "HexTacToe-AlphaZero",
            "description": "AlphaZero-style MCTS + ResNet.",
            "author": "<your name>",
            "version": "0.1.0",
        },
        "stateless": {
            "versions": {
                "v1-alpha": {
                    "move_time_limit": True,
                }
            }
        },
    }

# The stateless turn endpoint lives at /stateless/v1-alpha/turn
# (exact request/response schema defined by the v1-alpha stateless sub-spec,
#  which is not yet part of bot-api-v1.yaml — check for updates before implementing)
```

Run with: `uvicorn python.api.bot_server:app --port 8080`

### Differences from earlier draft

| Earlier draft | Live spec (2026-03-28) |
|---|---|
| `GET /capabilities` | `GET /capabilities.json` |
| `POST /move` with `board` + `time_limit_ms` | No `/move` endpoint in spec |
| `tags: string[]` | Not present |
| `supported_features: []` | Not present |
| `authors: string[]` | `author: string` (single) |
| No WebSocket | `basic_websocket` capability defined |
| Move API fully specified | Move API in versioned sub-specs not yet published |

**Action required before Phase 6:** Re-fetch `bot-api-v1.yaml` and check for a companion
stateless sub-spec. The turn request/response schema is not yet documented in the current
YAML — the community noted this spec is still evolving.

### CLI-to-API wrapper

The community discussed a localhost wrapper pattern: bots launched via shell + stdio are
wrapped behind the API. Implement this so our bot can run in both modes:

```bash
# Standalone API mode
python scripts/serve_bot.py --port 8080 --checkpoint checkpoints/best.pt

# CLI mode (for HexTacToeBots tournament runner)
python scripts/bot_cli.py --checkpoint checkpoints/best.pt
```

---

## 5. Formation library — for reward shaping and evaluation

This is the most architecturally important section. The community knowledge base defines a hierarchy of formations — from basic threats to forced wins. Our reward shaping should use this vocabulary, not generic n-in-a-row counting.

### Formation hierarchy (weakest → strongest)

```
Pre-emp (singlet)
  └── Threat (forces response)
        └── Double Threat (requires 2 blocks)
              └── Unstoppable formations (forced win, perfect play)
```

### Unstoppable formations (forced wins in isolation)

These are confirmed forced wins with perfect play. The network discovering how to create and recognise these is a key training milestone.

| Formation | BKE | Description |
|---|---|---|
| **Triangle** | `x A0 A1` | 3-cell. Core of "triangle theory". Impossible to defend against in isolation. |
| **Open Three** | `x A0 A3` | 3 cells with a gap. Thought to be undefendable. |
| **Rhombus** | `x A0 A1 A2` | 4-cell rhombus. Proceeds from open two island. |
| **Arch / Banana** | `x A0 A2 B1` | 4 cells in a curve. |
| **Trapezoid** | `x A0 A1 A2 A3` | 5-cell pentagon. Derived from triangle standard defense. |
| **Line / Quad** | `x A0 A3 B0` | Forces opponent to cap ends, then X makes Bone/Triangle/Rhombus. |
| **Ladder** | `x C0 C3 C6` | Triangles spaced 2 apart. Always wins with perfect play. |
| **Bone** | `x A0 A1 A3 A4` | Bowtie — equivalent to 5 open pre-emptives. |

### Revised reward shaping using formation vocabulary

Replace the generic n-in-a-row reward table in `01_ARCHITECTURE.md` with this:

```python
# python/training/formation_rewards.py
# All rewards decay to 0 over training — see decay schedule in 01_ARCHITECTURE.md

FORMATION_REWARDS = {
    # Pre-emps (weakest signal)
    "singlet_created":          +0.01,
    "singlet_blocked":          +0.005,

    # Threats
    "threat_created":           +0.03,
    "threat_blocked":           +0.02,
    "double_threat_created":    +0.08,

    # Forced-win shapes (strong signal — detecting these is key)
    "triangle_created":         +0.20,
    "open_three_created":       +0.15,
    "rhombus_created":          +0.18,
    "ladder_created":           +0.25,
    "bone_created":             +0.15,

    # Critical defensive failures
    "allowed_unstoppable":      -0.30,   # opponent reached a forced-win shape
}
```

**Warning (confirmed by community field report):** An RL bot in this community was observed farming threat rewards instead of finishing wins. This is exactly why the decay schedule is mandatory and terminal reward must always dominate. The shaped rewards above are initialization hints, not objectives.

### Formation detection in Rust

```rust
// native_core/src/formations.rs
// Called after each move — incremental scan from new piece only (community-confirmed pattern)

pub struct FormationDetector;

impl FormationDetector {
    /// Scan from the newly placed piece across all 3 hex axes.
    /// Returns a list of detected formations for that player.
    pub fn detect_after_move(
        board: &Board,
        last_move: (i32, i32),
        player: i8,
    ) -> Vec<Formation> {
        let mut found = Vec::new();
        for axis in HexAxis::all() {
            let segment = board.get_segment(last_move, axis, player);
            found.extend(classify_segment(&segment, board, player, axis));
        }
        found
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Formation {
    Singlet,
    Threat,
    DoubleThreat,
    Triangle,
    OpenThree,
    Rhombus,
    Arch,
    Trapezoid,
    Ladder,
    Bone,
    ForcedWin,   // catch-all for any confirmed unstoppable shape
}
```

**Implementation note from community:** Scan only from the newly placed piece across the 3 axes, store line endpoints for tracked segments, avoid full-board rescans. Extra bookkeeping (tracking all segments globally) can become slower than targeted recomputation — profile before optimising.

---

## 6. Opening book

The community knowledge base provides a table of named openings in BKE notation. These are directly usable as:

1. **Curriculum data**: oversample these positions in early training
2. **Opening book**: in evaluation/deployment, use policy head output within the book, then switch to full MCTS search
3. **Benchmark positions**: fixed positions to track policy improvement over training iterations

### Named openings (BKE)

```python
# python/opening_book/openings.py

NAMED_OPENINGS = {
    "closed_game":          "A0 A2",
    "closed_main_line":     "A0 A2 x A1 A4",
    "longsword":            "A0 A2 x A1 A4 o B2 D16",
    "shortsword":           "A0 A2 x A1 A4 o B2 B8",
    "sword":                "A0 A2 x A1 A4 o B2 C12",
    "wrongsword":           "A0 A2 x A1 A4 o B2 E20",
    "triangle_var":         "A0 A2 x A3 A4",
    "triangle_blanket_def": "A0 A2 x A3 A4 o A1 B5",
    "triangle_std_pair":    "A0 A2 x A3 A4 o B5 B8",
    "triangle_defensive":   "A0 A2 x A3 A4 o B6 B9",
    "pistol":               "A0 B2",
    "pistol_snail":         "A0 B2 x A1 A5",
    "pistol_triangle":      "A0 B2 x A4 A5",
    "pistol_river":         "A0 B2 x A1 B1",
    "pistol_koi":           "A0 B2 x C1 C2",
    "open_game":            "B0 B4",
    "island":               "E0 E1",
    "near_island":          "C0 C1",
    "101":                  "A0 A3",
    "shotgun":              "A0 C5",
    "pickaxe":              "A0 A2 x B0 B4 o A1 B2 x C4 D5",
    "pair":                 "A0 A1",
}

def load_opening_positions() -> list[dict]:
    """
    Parse all named openings into board states.
    Returns list of {name, state, depth} dicts for use in curriculum sampling.
    """
    parser = BKEParser()
    positions = []
    for name, bke in NAMED_OPENINGS.items():
        moves = parser.parse(bke)
        state = GameState.initial()
        for move in moves:
            state = state.apply(bke_to_axial(move))
        positions.append({"name": name, "state": state, "depth": len(moves)})
    return positions
```

---

## 7. What the community is watching for

From the Discord: no RL bot has yet surpassed the strongest public minimax bot. The community is aware this is an open problem and there's genuine interest in seeing it solved. A few things they will care about:

- **Can it explain its moves?** Outputting top-k policy moves + value estimates (via the bot API metadata field) gives the community something to analyse.
- **Does it play known theory?** Early evaluation games should be checked against the opening table — if the network is ignoring `A0 A2 x A1 A4` (closed game main line) in favour of random-looking moves, something is wrong.
- **Win rate vs Ramora0 at depth 5** is the natural community benchmark. Target: ≥ 55% win rate = clearly stronger.
- **SPRT-style testing**: P_P has a working SPRT runner. Once we reach a candidate, run SPRT against the Ramora0 baseline for a statistically valid strength claim.
