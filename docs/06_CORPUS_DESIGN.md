# Corpus Pipeline Design — `python/corpus/`

> Design document only. No implementation. Last updated: 2026-04-01.

---

## Context

- **203 human games** on disk in `data/corpus/raw_human/` (UUID-named JSON files).
  Format: `{id, players[], playerTiles, gameOptions, moveCount, gameResult, moves[]}`.
  Each move: `{moveNumber, playerId, x, y, timestamp}` — `x`/`y` are already axial
  `(q, r)` coordinates, no conversion needed.
  Filter already applied at scrape time: `rated=true, moveCount≥20, reason=six-in-a-row`.
- **SealBotBot wrapper** complete and tested (`python/bootstrap/bots/sealbot_bot.py`).
- **Human opening + bot continuation** strategy confirmed: N=8 move threshold (see §4).
- **Parallelism target**: 10–12 worker processes.
- **Colony bug risk**: SealBot uses a flat 140×140 board — multiple distant clusters may
  alias and corrupt its pattern evaluation. Must track and log when
  `cluster_count > 4` at handoff.

---

## 1. Directory Layout

`python/corpus/` is the new orchestration layer. It sits **above** `python/bootstrap/`
and imports from it; it never replaces it.

```
python/corpus/
├── __init__.py
├── sources/
│   ├── __init__.py
│   ├── base.py               # CorpusSource ABC + GameRecord dataclass
│   ├── human_game_source.py  # reads data/corpus/raw_human/ — no scraping
│   └── hybrid_game_source.py # HumanGameSource + SealBotBot continuation
├── pipeline.py               # CorpusPipeline: dedup, convert, push to buffer
└── metrics.py                # per-source counters (colony bug rate, flip rate, pos/hr)
```

**Rationale for separation:**
`python/bootstrap/` is stable and phase-complete. Putting the new pipeline there would
conflate the scraping / minimax-pretraining concern with the ongoing corpus generation
concern. `python/corpus/` has a different job: it is a long-running, incremental feeder
that pushes positions into `RustReplayBuffer` during Phase 4+. These are different
lifecycles.

---

## 2. `CorpusSource` ABC

```
CorpusSource
  name() -> str                 # used in log fields + metrics labels
  __iter__() -> Iterator[GameRecord]   # yields one GameRecord per completed game
  __len__() -> int | None       # optional; enables progress bars; None if unknown
```

`GameRecord` — the pipeline's unit of consumption:

```
GameRecord
  game_id_str: str              # source-specific opaque identifier
                                #   human: UUID from JSON filename
                                #   hybrid: "{seed_uuid}:c{continuation_idx}"
  moves: list[tuple[int, int]]  # ordered (q, r) sequence, full game
  winner: int                   # +1 (player 1 won) or -1 (player 2 won)
  source: str                   # "human" | "hybrid" | "api" | …
  metadata: dict                # optional extra fields; pipeline ignores this
```

**What the ABC does NOT specify:**
- How the source generates games (scrape, process, play, HTTP)
- Whether games are pre-loaded or streamed lazily
- Anything about tensors, policies, or the replay buffer

This is intentional. The pipeline converts `GameRecord → (state_tensor, policy, value)`
triples. That conversion is the pipeline's job, not the source's.

**Planned future sources and why the ABC fits:**
| Source | `__iter__` behaviour |
|---|---|
| Second website scraper | Read from a second cache dir on disk |
| `CommunityAPIBot` source | Play games over HTTP via `BotProtocol`, yield `GameRecord` per game |
| Full human replay (no bot) | Read JSON, replay all N moves, yield one `GameRecord` |

All three fit the same ABC without widening it.

---

## 3. `HumanGameSource`

Wraps the existing scraper cache. **Does not re-scrape.** Reads whatever JSON files are
currently in `data/corpus/raw_human/`.

### JSON → `GameRecord` conversion

**Player assignment:**
Move `moveNumber == 2` is always Player 1's opening stone (the single-stone first move).
The `playerId` of that move is P1; the other player is P2.

```
p1_id = moves[0]["playerId"]   # first entry is moveNumber=2 (the opening stone)
```

(Move number 1 is a game-creation event; it is absent from the JSON. Move 2 is always
the first stone placement.)

**Winner:**
```
winner = +1 if gameResult["winningPlayerId"] == p1_id else -1
```

**Moves:**
Extract `(x, y)` from each entry in the `moves` array, in order. These are already
`(q, r)` axial coordinates.

**Filtering:**
`HumanGameSource` re-validates the already-applied scraper filter:
- `gameOptions.rated == true`
- `moveCount >= 20`
- `gameResult.reason == "six-in-a-row"`

Any file that fails re-validation is skipped with a `log.warning("human_game_skipped")`.
This guards against corrupt or partial downloads without crashing the pipeline.

**Output `GameRecord`:**
```
game_id_str = UUID (from filename, without .json extension)
moves       = [(x,y), …] ordered
winner      = +1 or -1
source      = "human"
metadata    = {players: [...], elo_p1: int, elo_p2: int}
```

---

## 4. `HybridGameSource`

Seeds: `HumanGameSource`. Continuation: `SealBotBot`.

### Parameters

| Parameter | Default | Notes |
|---|---|---|
| `n_opening_moves` | 8 | Target stone-placement threshold; see turn-boundary rule below |
| `time_limit` | 0.05 | Passed to `SealBotBot.__init__` |
| `n_workers` | 10 | `multiprocessing.Pool` workers |
| `games_per_seed` | 3 | Bot continuations generated per human opening |

**`games_per_seed = 3` — justification:**
With 203 human games, 3 continuations gives 609 hybrid games (~30,000–50,000 positions),
a reasonable addition to the pretrain corpus without over-representing any single opening.
SealBot's time-limited search is not fully deterministic across runs (wall-clock variance
changes the search tree), so 3 continuations genuinely explore different continuations.
If the corpus budget is later raised (e.g. WolverinDEV export arrives), raise this to 5.

### Turn-boundary handoff rule

The 1-2-2 turn structure means `n_opening_moves = 8` individual stone placements lands
**mid-turn**:

| Ply | Player | `moves_remaining` after ply |
|---|---|---|
| 0 | P1 opens | 2 (P2's turn now) |
| 1 | P2 first | 1 |
| 2 | P2 second | 2 (P1's turn now) |
| 3 | P1 first | 1 |
| 4 | P1 second | 2 (P2's turn now) |
| 5 | P2 first | 1 |
| 6 | P2 second | 2 (P1's turn now) |
| 7 | P1 first | **1** ← mid-turn |
| 8 | P1 second | 2 (P2's turn now) |

After 8 plies, `moves_remaining == 1` for P1. The bot would enter mid-turn.

**Rule:** after replaying ≥ `n_opening_moves` plies, advance one more ply from the
human game if `state.moves_remaining < 2`. The bot always receives a position where
`moves_remaining == 2` (i.e., the start of a fresh double-move turn).

With N=8: replay ply 7, detect `moves_remaining == 1`, replay ply 8, hand off.
Effective handoff at ply 9; bot enters as P2 with `moves_remaining == 2`.

**Games shorter than N+1 plies are discarded** (not enough human moves to reach the
handoff point). Log as `log.warning("hybrid_seed_too_short")`.

### Which player is the current player at handoff?

After snapping to a turn boundary, the current player alternates depending on the total
ply count. With N=8 (effective 9 plies), it is **always P2**. This is fine — both P1
and P2 positions are valuable training data. If the distribution of who-the-bot-is
matters for analysis, tag it in `metadata`.

### Value labels

The value labels for ALL positions in a hybrid game — both the human opening portion
(plies 0..handoff-1) and the bot continuation — are determined by the **hybrid game's
terminal outcome**, not the original human game's winner. This is the correct label:
it reflects what actually happened in the game the network will train on.

The original human winner is retained in `metadata` for the value-flip metric (§6).

### Continuation seed variation

Each of the `games_per_seed` continuations from the same human opening must use
a distinct RNG seed when initialising `SealBotBot` (or the worker's random state),
to maximise variation across continuations. Tag the seed index as
`metadata["continuation_idx"]`.

---

## 5. `CorpusPipeline`

```
CorpusPipeline(
    sources: list[CorpusSource],
    buffer:  RustReplayBuffer,
    metrics: CorpusMetrics,
)
```

### Responsibilities

1. **Consume** `GameRecord`s from each source in sequence (or interleaved — see below).
2. **Dedup** by `game_hash` before converting. Skip duplicates silently, count in metrics.
3. **Convert** each unique `GameRecord` to `(state_tensor, policy, value)` triples by
   replaying the move sequence using `Board` + `GameState` (reusing `bootstrap/dataset.py`
   logic — do not duplicate it).
4. **Assign game_id**: a monotonic `int64` counter maintained by the pipeline. This is
   the `game_id` passed to `buf.push_game()`. It must never be −1.
5. **Push** to `RustReplayBuffer` via `buf.push_game(states, policies, outcomes, game_id)`.
6. **Emit metrics** to `CorpusMetrics` after each game.

### Dedup hash

`game_hash = SHA256("|".join(f"{q},{r}" for q, r in game_record.moves))`

The hash is over the **ordered** move sequence, not a sorted bag. Sorted hashing would
false-positive on distinct games that happen to contain the same set of moves in
different orders (this can happen in short or symmetric games). Ordered hashing correctly
identifies exact-duplicate game records across pipeline runs.

The seen-hash set lives in memory during a pipeline run. It is not persisted across runs;
the buffer's ring-buffer semantics already handle position-level redundancy gracefully.

### `game_id` assignment

The pipeline holds a `next_game_id: int` counter initialised to 0 at startup.
Each successfully pushed game increments the counter by 1. The counter is not
persisted across runs — this is fine because the `RustReplayBuffer` itself is rebuilt
each training session. The monotonic property within a run is all that is needed to
prevent positions from the same game from appearing in a single training batch.

### Source ordering

By default, sources are consumed in order: `HumanGameSource` first, then
`HybridGameSource`. This means human positions enter the buffer first and are
proportionally sampled more heavily in early training steps. This is intentional
(human games are the highest-quality signal).

If interleaving is ever needed (e.g. to balance source distribution within a batch),
add an `interleave: bool` flag to the pipeline — do not design for this now.

---

## 6. Metrics to Track

All metrics are per-source (tagged by `CorpusSource.name()`). `CorpusMetrics` is a
lightweight dataclass that accumulates counts and flushes them via `structlog` at
configurable intervals (default: every 50 games).

### Colony bug exposure rate

```
colony_bug_games   / total_hybrid_games
```

A hybrid game is "exposed" if `cluster_count > 4` **at the handoff ply** (the moment
the bot is first called). This is a single check per game, not per move.

SealBotBot already logs `sealbot_colony_bug_risk` per-move. The pipeline-level metric
is coarser: did any handoff position trigger the risk? Log it as:

```json
{"event": "corpus_colony_bug_exposure", "source": "hybrid", "game_id_str": "...",
 "clusters_at_handoff": 6, "ply_at_handoff": 9}
```

Acceptable threshold: < 5% of hybrid games. If this rate is high, investigate whether
the human game corpus has a systematic tendency to produce multi-colony positions by
move 9 (it may, if players use the colony meta opening).

### Value label flip rate

```
flip_games / total_hybrid_games
```

A hybrid game is a "flip" if `hybrid_winner != human_winner`. Both are `+1` or `−1`.
High flip rate (> 50%) suggests the bot is systematically playing differently from
human strategy at the handoff point, which may mean the value labels on the opening
positions are noisy. Log at game level:

```json
{"event": "corpus_value_flip", "source": "hybrid", "game_id_str": "...",
 "human_winner": 1, "hybrid_winner": -1}
```

This metric does not block the pipeline — flipped games are still included. It is a
diagnostic to monitor. If flip rate exceeds 70%, reconsider the handoff depth or bot
strength.

### Positions per hour per source

Standard throughput metric aligned with the primary worker metric. `CorpusMetrics`
tracks `positions_pushed` and `wall_time_start` per source and emits:

```json
{"event": "corpus_throughput", "source": "hybrid", "positions_per_hour": 12400,
 "positions_total": 24800}
```

Log this every 100 games or at pipeline completion.

---

## 7. Open Questions

The following require a decision before implementation begins.

**Q1. Dedup hash scope**
Ordered vs. sorted was resolved above (ordered). But: should dedup be run across
*both* `HumanGameSource` and `HybridGameSource`? A hybrid game's move list will never
match a full human game's move list (different lengths), so cross-source dedup does
nothing. Recommend: dedup within each source only. Confirm this is acceptable.

**Q2. Human opening positions — include in training or not?**
For a hybrid game, should the training triples from the human opening portion (plies
0..handoff-1) be included in the buffer, or only the bot portion (plies handoff..end)?
Arguments for including: more positions per game, human moves are high quality.
Arguments against: the value label for those positions is the *hybrid outcome*, which
may not match the human game's outcome (flip). This is a training signal quality
question, not a correctness question. **Decision needed.**

**Q3. `games_per_seed` variation strategy**
SealBot is time-limited and not fully deterministic, but if the host system is
consistently fast, continuations may converge. Should we deliberately inject entropy
(e.g. random first-move perturbation from the handoff position) to ensure 3 genuinely
distinct continuations? Or rely on timing variance? **Decision needed before measuring
whether games_per_seed=3 actually produces diverse continuations.**

**Q4. Hybrid game: which player does the bot play as?**
With the snap-to-turn-boundary rule and N=8, the bot always enters as P2.
Is it acceptable that P1's moves in the bot continuation are played by a *second
instance* of SealBotBot, or should the bot play both sides? Both-sides is simpler
(one bot instance per game, alternating). One-side-each is more diverse. The current
`BotProtocol` does not distinguish — **confirm which configuration is intended.**

**Q5. Minimum game length filter for hybrid games**
If SealBot continuation is very short (e.g. 3 plies, because it finds a forced win),
should the hybrid game be included? Very short continuations produce few positions.
Recommend a minimum of 10 bot plies (configurable). Confirm this threshold.

**Q6. Persistence of the dedup set**
The current design does not persist the seen-hash set across runs. This means the
same human game will enter the buffer in every pipeline run. Is that acceptable, or
should we write `data/corpus/seen_hashes.pkl`? For a ring buffer that replaces old
positions, multi-run duplication is not harmful — but worth confirming explicitly.
