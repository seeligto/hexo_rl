# Notation Spec Notes

Source: `docs/reference/notation/README.md` (read 2026-03-28).

---

## What the live spec actually defines

The live notation spec uses **raw axial coordinates** `[q, r]`, not BKE ring/offset
notation. This is a significant difference from what `docs/05_community_integration.md`
section 3 describes.

### Grammar

```
<game>        ::= <metadata>? <turn>*
<metadata>    ::= <data>+ ";"
<data>        ::= <key> "[" <value> "]"

<turn>        ::= <turn_number> <coordinate> <coordinate> <amount_of_threats>? ";"
<turn_number> ::= <digits> "."
<coordinate>  ::= "[" <integer> "," <integer> "]"    ← axial [q, r]
<amount_of_threats> ::= "!"*
```

### Example game record

```
name[GameName 0]platform[WebsiteXY 0]playercross[BlueWhale 0]playercircle[GreenSnake 0]timecontrol[Fischer 60+5]endreason[win]winner[cross]datetime[2026-03-18 23:20:14];
1. [-1,0][0,1];
2. [-1,1][-2,2];
3. [1,-1][-5,5];
4. [-1,2][1,0];
5. [5,0][-3,2];
```

### Key rules

1. **First cross move is always at [0,0] and is never recorded.** The game starts with
   cross already placed at the origin. Turn 1 is circle's first turn.
2. **Each recorded turn is one player's pair of placements.** Both coordinates in a turn
   belong to the same player.
3. **Coordinates are axial `[q, r]`** — same coordinate system used by our Rust board.
   No conversion needed between file format and internal representation.
4. **Turn ordering:** circle goes first in the recorded turns (turn 1 = circle's turn),
   then cross (turn 2), alternating from there.
5. **`amount_of_threats`** (`!` marks): optional annotation, one `!` per threat. Can be
   ignored for parsing purposes.
6. **Whitespace** between tokens is ignored (except inside `<value>` strings in metadata).
7. **Metadata** is optional. All attributes are optional within it. `version` is recommended.

### Metadata attributes

| Key | Description | Value format |
|---|---|---|
| `version` | Notation version | integer |
| `name` | Display name of match | string |
| `platform` | Website/app | string |
| `utcdatetime` | UTC game start | `YYYY-MM-DD HH:MM:SS` |
| `playercross` | Cross player name | string |
| `playercircle` | Circle player name | string |
| `timecontrol` | Time control | `basetime+increment` or `basetime` |
| `endreason` | How game ended | `win`, `time`, `resign`, `draw` |
| `winner` | Who won | `cross`, `circle` |

### Time control format

```
<timecontrol> ::= <integer> ("+" <integer>)?
```
Absolute (single integer) or Fischer (base+increment). Unit not specified in spec —
context suggests seconds.

---

## Differences from docs/05_community_integration.md section 3

Our section 3 (BKE notation) describes a **completely different notation system** that
is not the live spec. The BKE (Boat-Kaitlyn-Epic) system uses Ring letters and Offset
numbers (`A0`, `A2`, `x A1 A4`, etc.) and was an earlier community notation.

| Our docs (section 3) | Live spec |
|---|---|
| BKE ring/offset: `A0 A2 x A1 A4` | Axial coords: `[-1,0][0,1]` |
| Ring letters (A, B, …, AA, …) | Integer `[q, r]` pairs |
| Offset = distance from zero-line | Standard axial hex coordinates |
| Sector form: `B2.1` | Not present |
| `x`/`o` player tokens mid-game | Turn numbers + two coords per turn |
| No metadata header | Optional metadata section |
| TypeScript reference parser by BSoD | Not mentioned in live spec |

**Impact on implementation:**

- The `BKEParser` described in `docs/05_community_integration.md` and planned for
  `python/opening_book/bke_parser.py` is for the **old** BKE format.
- The `NAMED_OPENINGS` dict in section 6 (e.g. `"closed_game": "A0 A2"`) uses the
  old BKE format. Those positions are still valid game positions — only the notation
  string format differs.
- Community game records downloaded from the platform will use the live axial format.
- For importing community games, implement a parser for the axial format — it is simpler
  than BKE (just integer coordinate pairs with a structured turn grammar).
- The BKE parser may still be needed to interpret the named openings table and any
  older community resources that use that format.

## Parser implementation plan

```python
# python/opening_book/game_record_parser.py
# Parses the live axial notation format

import re
from dataclasses import dataclass

@dataclass
class GameTurn:
    turn_number: int
    q1: int; r1: int   # first placement of this turn
    q2: int; r2: int   # second placement of this turn
    threats: int = 0   # count of '!' annotations

@dataclass
class GameRecord:
    metadata: dict[str, str]
    turns: list[GameTurn]
    # Implicit: cross plays [0,0] as move 0 (never in turns list)
    # Turn 1 = circle; turn 2 = cross; etc.

def parse_game_record(text: str) -> GameRecord:
    """Parse a game record in the live axial notation format."""
    ...
```

Key properties:
- Cross stone at [0,0] is implied — prepend it when reconstructing game state.
- Turn N corresponds to: circle if N is odd, cross if N is even.
- Two `[q,r]` coordinate pairs per turn, in order of placement.
- Threat count (`!` marks) is optional metadata per turn.
