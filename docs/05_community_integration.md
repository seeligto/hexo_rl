# Community Integration — Ecosystem, Notation, Formations, Bot API

This document captures everything learned from the Hex Tac Toe community Discord and knowledge base that directly affects implementation. Do not treat this as background reading — several decisions here override or sharpen earlier architecture choices.

---

## 1. Community resources (updated 2026-03-29)

| Resource | URL | Relevance |
|---|---|---|
| Strongest public engine | <https://github.com/Ramora0/SealBot> | Bootstrap source — pybind11 minimax engine |
| Shared bot collection + tournament runner | <https://github.com/Ramora0/HexTacToeBots> | Evaluation target |
| `hexo` Python package | <https://github.com/PierreLapolla/hexo> (PyPI: `hexo`) | **Investigate first** — may replace our board/notation layer |
| Bot API spec (v1) | <https://github.com/hex-tic-tac-toe/htttx-bot-api> | Deploy target: <https://explore.htttx.io/> |
| Notation standard | <https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation> | BKE — read limitations in section 3 |
| Community GitHub org | <https://github.com/hex-tic-tac-toe> | Standards home |
| Game archive (primary) | <https://[site-redacted]/games> | 42k+ rated games — main corpus source |
| Game archive (secondary) | <https://[site-redacted]/games/> | Additional game hosting site |
| Bot Dev Discord | <https://discord.gg/7P8NeXnM4> | Dedicated bot dev server — monitor for new bots |

---

## 2. The SealBot engine — bootstrap source

**Do not build our own minimax bot.** SealBot (by imaseal, repo: `Ramora0/SealBot`) is the strongest public bot, available as a pybind11 module. Use it directly to generate the bootstrap corpus.

### How to use the engine

SealBot is imported directly as a Python module via pybind11 — no subprocess or binary compilation needed. The wrapper at `python/bootstrap/bots/sealbot_bot.py` handles path setup and the `BotProtocol` interface.

```python
# python/bootstrap/bots/sealbot_bot.py (simplified)
from minimax_cpp import MinimaxBot as _MinimaxBot
from game import Player as SealPlayer

class SealBotBot(BotProtocol):
    """Wraps the SealBot pybind11 minimax engine."""
    def __init__(self, time_limit: float = 0.05):
        self._bot = _MinimaxBot(time_limit)
        ...

    def get_move(self, state, board) -> tuple[int, int]:
        result = self._bot.get_move(game)
        return (result.q, result.r)
```

### Time limit configuration for corpus generation

SealBot uses a time limit (seconds per move) rather than a fixed depth:

| Time limit | Speed (approx) | Use |
|---|---|---|
| 0.03s | Fast | Bulk corpus — tactical basics |
| 0.05s | Medium | Default — good balance of speed and quality |
| 0.10s | Slow | Quality games — stronger patterns |

Configured via `sealbot_time_limit` in bootstrap YAML configs. The `sealbot_time_mix` option allows mixing multiple time limits for diversity.

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

Our bot must implement the `htttx-bot-api-v1` spec to participate in community tournaments via `HexTacToeBots`. This is an HTTP API.

### Required endpoints

```yaml
# From bot-api-v1.yaml (community standard)

GET /capabilities
  # Returns bot metadata and supported features
  Response:
    meta:
      name: string
      description: string
      authors: string[]
      version: string
    tags: string[]          # flexible extension: ["alphazero", "rl", "mcts"]
    supported_features: []

POST /move
  # Given a board state, return the bot's chosen move
  Request:
    board: BoardState
    time_limit_ms: int
  Response:
    row: int
    col: int
    metadata:               # optional
      confidence: float     # value head output for chosen move
      top_moves: []         # policy head top-k
```

### Implementation

```python
# python/api/bot_server.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

class MoveRequest(BaseModel):
    board: dict
    time_limit_ms: int = 5000

@app.get("/capabilities")
def capabilities():
    return {
        "meta": {
            "name": "HexTacToe-AlphaZero",
            "description": "AlphaZero-style MCTS + ResNet. First RL bot in community.",
            "authors": ["<your name>"],
            "version": "0.1.0",
        },
        "tags": ["alphazero", "mcts", "reinforcement-learning", "resnet"],
        "supported_features": [],
    }

@app.post("/move")
def make_move(req: MoveRequest):
    state = board_dict_to_game_state(req.board)
    move, metadata = inference_server.get_best_move(
        state, 
        time_budget_ms=req.time_limit_ms,
    )
    return {
        "row": move[0],
        "col": move[1],
        "metadata": metadata,
    }
```

Run with: `uvicorn python.api.bot_server:app --port 8080`

### CLI-to-API wrapper

The community discussed a localhost wrapper pattern: bots launched via shell + stdio are wrapped behind the API. Implement this so our bot can run in both modes:

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
// engine/src/formations/mod.rs
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
- **Win rate vs SealBot** is the natural community benchmark. Target: ≥ 55% win rate = clearly stronger.
- **SPRT-style testing**: P_P has a working SPRT runner. Once we reach a candidate, run SPRT against the SealBot baseline for a statistically valid strength claim.

---

## 8. `hexo` Python package — evaluate before reimplementing

Pedro published a Python package called `hexo` on PyPI (2026-03-29):

- PyPI: `pip install hexo`
- Repo: <https://github.com/PierreLapolla/hexo>
- Described as: "python-chess for chess but for HexO"

**Before implementing any board logic, BKE parsing, or game utilities in Python,
the agent must evaluate this package first.** It may already provide:

- Board state representation and move application
- BKE notation parsing and serialisation
- Win detection
- Legal move generation
- Game record I/O

If it covers these correctly, use it rather than our own implementations in the
Python layer. Our Rust core handles the hot paths (MCTS, win detection at speed) —
but for Python-side utilities (scraper, corpus conversion, opening book), using
a maintained community library is strictly better than maintaining our own.

**Evaluation task (Phase 3 sub-task):**

```bash
pip install hexo
python3 -c "import hexo; help(hexo)"
# Read the repo README and source
# Test: does it correctly handle the turn structure (1 then 2-2-2)?
# Test: does it parse all openings from our NAMED_OPENINGS table correctly?
# Test: does it detect all winning patterns we tested in Phase 0?
```

Document findings in `docs/reference/hexo_package_notes.md`.
If it passes all checks, update the bootstrap pipeline to use it.
If it has gaps, use it where it works and note what it misses.

**Note:** The package is brand new (landed March 2026) and the author said
"we'll see if it is actually useful" — so evaluate critically, don't assume completeness.

---

## 9. SealBot engine internals — candidate generation pattern

The community's strongest bot (SealBot) uses these constants:

```cpp
static constexpr int    CANDIDATE_CAP      = 15;  // max candidate moves per node
static constexpr int    ROOT_CANDIDATE_CAP = 20;  // more candidates at root
static constexpr int    PAIR_SUM_CAP       = 14;
static constexpr int    NEIGHBOR_DIST      = 2;   // cells within distance 2 of any stone
static constexpr double DELTA_WEIGHT       = 15;
static constexpr int    MAX_QDEPTH         = 16;  // quiescence search depth
static constexpr int    WIN_LENGTH         = 6;
```

**What this means for us:**

`NEIGHBOR_DIST=2` is the key insight — SealBot only considers moves within 2 cells
of any existing stone as candidates. This is a massive search space reduction and
explains why it can be strong with only 15 candidates per node.

Our legal move generation uses `bounding_box + 2 cell margin` which is similar but
not identical — we generate all cells in the window, not just cells within distance 2
of a stone. For MCTS this is fine (the policy head filters bad moves via low priors),
but it is useful context for understanding the benchmark we're training against.

`MAX_QDEPTH=16` indicates SealBot runs a quiescence search extension — it keeps
searching forced sequences (threats, blocks) up to 16 plies deeper than normal search.
Our MCTS handles this naturally through deep rollouts, but it explains why SealBot
is particularly strong at calculating forced wins.

---

## 10. O/X asymmetry — confirmed by community

Multiple community members independently confirmed O is slightly stronger than X.
The clearest framing: *"O controls the game, X just happens to have a gift at the start."*

The asymmetry arises from the turn structure: X places 1 stone on turn 0, then both
players place 2 stones per turn. X's single opening move is a tempo gift, but O
then gets persistent 2-stone turns from the start and effectively dictates flow.

**Implications for training:**

- The value head must learn asymmetric win probabilities from the opening position
- Self-play should track Elo and win rate broken down by side (O vs X) separately
- If O win rate in self-play exceeds ~58%, the asymmetry is being correctly learned
- The bootstrap corpus should be roughly balanced: ~50% games where human/bot
  plays O, ~50% where they play X — do not oversample one side

**Implication for opening book:** Our named openings are currently recorded from
O's perspective (O moves first). The value head will need sufficient games from
both perspectives to avoid a systematic bias.

---

## 11. BKE notation limitations (updated)

Beyond what was previously documented, the community has now explicitly confirmed:

- **Cannot express positions where X has more moves than O.** The format assumes
  O moves first and the move counts stay balanced or O-leading.
- **Cannot start from an arbitrary position** (e.g. a pre-set Triangle formation).
  The format always assumes the center X as the implicit first move.
- **Practical workaround used by community:** Parse top-left cell as (0,0) and
  treat all positions as relative — but this breaks orientation-independence.

**Impact on our opening book:**
The named openings in section 6 are all expressible in BKE (they start from the
natural game start). Arbitrary training positions (e.g. "evaluate this Triangle")
cannot be expressed in BKE and must use axial coordinates directly.

Do not attempt to encode mid-game tactical positions in BKE. Use `(q, r, player)`
triples for anything that isn't a game record from move 1.

---

## 12. Tactics puzzles — potential high-quality training data

Community member thejackofclubs is building a "mate-in-N" puzzle extractor from
the game archive — finding positions where one player can force a win in 2-3 moves.

These positions are extremely high-signal training data if they become available:

- The correct move is known (forced win)
- The position is real (from actual community games)
- They stress-test the policy head on the exact situations where errors are costly

Monitor the Bot Dev Discord (discord.gg/7P8NeXnM4) for when this dataset is
published. If available before Phase 5, add it to the corpus alongside human games.
