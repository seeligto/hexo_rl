# §176 Wave A2 — Eval Architecture + BotProtocol Extension Surface

**Scope:** Map opponent dispatch path. Identify minimal extension surface for KrakenBot + compound-move bots. Recommend `BotProtocol.get_turn()` vs caching.

**Status:** Complete. Verdict: **CACHING_CLEAN**.

---

## (a) Bot Wrapper Inventory (post-§176 P78)

All bots migrated from `hexo_rl/bootstrap/bots/` → `hexo_rl/bots/` per commits 3ea5873 (P78a), d554e7d (P78b), 7233d5d (P78c+d). Bootstrap bots dir deleted.

| File | Line | Class | Base |
|---|---|---|---|
| `hexo_rl/bots/random_bot.py` | 14 | `RandomBot` | `BotProtocol` |
| `hexo_rl/bots/sealbot_bot.py` | 29 | `SealBotBot` | `BotProtocol` |
| `hexo_rl/bots/krakenbot_bot.py` | 59 | `KrakenBotBot` | `BotProtocol` |
| `hexo_rl/bots/our_model_bot.py` | 24 | `OurModelBot` | `BotProtocol` |
| `hexo_rl/bots/community_api_bot.py` | 24 | `CommunityAPIBot` | `BotProtocol` |
| `hexo_rl/bootstrap/bot_protocol.py` | 20 | `BotProtocol` | `ABC` |

**Verification:** `ls hexo_rl/bots/ ✓` (6 files, no bootstrap/bots/ dir).

---

## (b) BotProtocol Contract (§176 P38 pin)

Location: `hexo_rl/bootstrap/bot_protocol.py:20–68`

**Abstract methods:**
```python
def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
    """Return a legal move (q, r) for the current position."""
    ...

def name(self) -> str:
    """Human-readable bot identifier."""
    ...
```

**Concrete methods:**
```python
def reset(self) -> None:
    """Clear per-game state. Default no-op; override for caching bots."""
    return
```

**Contract stability (§176 P38, line 23-28):** Argument set pinned. `rust_board` authoritative (read-only, not yet applied). `state` convenience snapshot, bots may ignore either. No underscore-prefix renaming required.

**Signature stable:** All 5 implementations (random, sealbot, krakenbot, our_model, community_api) use identical `(state, rust_board) → tuple[int, int]` contract.

---

## (c) Compound-Turn Handling Locus

Per CLAUDE.md line 14, rule: "player 1 opens with 1 move then both players alternate 2 moves per turn" → sequential stones, caller loops per stone.

**Locus:** `hexo_rl/eval/evaluator.py:201–210`, the main game loop:

```python
while not board.check_win() and board.legal_move_count() > 0:
    if ply < self._eval_random_opening_plies:
        q, r = random.choice(board.legal_moves())
    elif board.current_player == model_player_side:
        q, r = model_player.get_move(state, board)  # line 205
    else:
        q, r = opponent.get_move(state, board)      # line 207

    state = state.apply_move(board, q, r)
    ply += 1
```

**Pattern:** Caller (evaluator) loops `for ply in game`. Each `get_move()` returns **one stone**. State updated after each call. No per-ply caching loop visible — compound moves handled **inside bots** via `_pending_move` cache.

**Existing caching pattern** (both SealBotBot + KrakenBotBot):
- `get_move()` line 81–84 (KrakenBotBot): return cached `_pending_move` if present, null it, early exit.
- Line 112–118: if compound turn (`len(result) >= 2`) and `moves_remaining > 1`, cache `result[1]` into `_pending_move`.
- `reset()` line 123: null `_pending_move` at game start.

**Verification:** `evaluator.py:201` is the per-ply loop; `krakenbot_bot.py:79–121` shows caching implementation.

---

## (d) Eval Dispatch Trace

Entry: `configs/eval.yaml` opponent key (e.g., `"sealbot"`).

**Call chain for opponent="sealbot":**

1. `hexo_rl/eval/eval_pipeline.py:394–411` (run_evaluation)
   - Calls `ctx.evaluator.evaluate_vs_sealbot(...)` per opponent_runners.py.

2. `hexo_rl/eval/opponent_runners.py:126–150` (_run_sealbot)
   - Reads `pipeline.sealbot_cfg` (time_limit, n_games, model_sims).
   - Calls `ctx.evaluator.evaluate_vs_sealbot(n_games=50, time_limit=0.5, model_sims=128)`.

3. `hexo_rl/eval/evaluator.py:246–258` (evaluate_vs_sealbot)
   - Creates `sealbot = SealBotBot(time_limit=0.5)` (line 256, DI fallback).
   - Calls `self.evaluate(sealbot, n_games=50, sims=128, phase="sealbot")`.

4. `hexo_rl/eval/evaluator.py:165–236` (evaluate)
   - Per-game loop (line 192): `for i in range(n_games)`.
   - Per-ply loop (line 201): `while not board.check_win()`.
   - Line 207: `q, r = opponent.get_move(state, board)` → **dispatches to SealBotBot.get_move()**.

5. `hexo_rl/bots/sealbot_bot.py:39–106` (SealBotBot.get_move)
   - Line 41–42: clear cache at turn start if `moves_remaining > 1`.
   - Line 45–48: return cached second stone if present.
   - Line 86: call `self._bot.get_move(game)` → C++ minimax engine.
   - Line 98–103: cache result[1] if compound turn.
   - Line 105: return result[0].

**File:line pointers:** eval_pipeline.py:394 → opponent_runners.py:139 → evaluator.py:246 → evaluator.py:201 → bots/sealbot_bot.py:39.

---

## (e) Config Schema: eval.yaml Opponents

Location: `configs/eval.yaml:7–90`.

| Key | Type | Default | Semantics |
|---|---|---|---|
| `best_checkpoint.enabled` | bool | `true` | Run vs rotating best trained model |
| `best_checkpoint.stride` | int | `1` | Run every stride × eval_interval steps |
| `best_checkpoint.n_games` | int | `100` | Games per round |
| `best_checkpoint.model_sims` | int | `128` | MCTS sims for model player |
| `best_checkpoint.opponent_sims` | int | `128` | MCTS sims for opponent (if model) |
| `sealbot.enabled` | bool | `true` | Run vs SealBot |
| `sealbot.stride` | int | `4` | Run every 4 eval rounds |
| `sealbot.n_games` | int | `50` | Games per round |
| `sealbot.think_time_strong` | float | `0.5` | SealBot search budget (seconds) |
| `sealbot.think_time_fast` | float | `0.1` | Unused; legacy |
| `sealbot.model_sims` | int | `128` | MCTS sims for model player |
| `random.enabled` | bool | `true` | Run vs RandomBot |
| `random.stride` | int | `1` | Run every round |
| `random.n_games` | int | `20` | Games per round |
| `random.model_sims` | int | `96` | MCTS sims for model player |
| `bootstrap_anchor.enabled` | bool | `true` | Run vs frozen reference (v7full) |
| `bootstrap_anchor.stride` | int | `1` | Run every round |
| `bootstrap_anchor.n_games` | int | `100` | Games per round |
| `bootstrap_anchor.path` | str | `checkpoints/bootstrap_model_v7full.pt` | Frozen reference checkpoint |
| `bootstrap_anchor.model_sims` | int | `128` | MCTS sims for model player |
| `bootstrap_anchor.opponent_sims` | int | `128` | MCTS sims for opponent |

**eval_random_opening_plies** (line 88): Integer. Default 0. Per §174 line 1192: was 4 → 0 to remove free positional diversity mask. At 4, model got WR boost; at 0, SealBot prep lands cleanly (explains §168→§174 WR drop from 14.5% → 0% when knob flipped). No code fix — knob already in configs/eval.yaml.

**Defaults module:** `hexo_rl/eval/defaults.py:29–34` centralizes SSR16 set (random_model_sims=96, sealbot_model_sims=128, eval_temperature=0.5, eval_random_opening_plies=4 [old], eval_seed_base=42).

---

## (f) INV Pin Risk

Files under `tests/refactor_invariants/`:
- `test_public_api.py` — freezes `run_training_loop` signature (INV13/14, §176 §E).
- `test_augment_required.py` — verifies augmentation passes on buffer.
- `test_buffer_schedule.py` — pins buffer-growth schedule.
- `test_cosine_radius_pairing.py` — cosine + legal_move_radius must pair (L9, §156).
- `test_source_weighting.py` — corpus source weighting not silently disabled.
- `test_viewer_isolation.py` — viewer does not mutate game state (INV12).

**BotProtocol / opponent impact:** None of these INVs mention opponent names or BotProtocol. No schema-snapshot tests. No round-trip dispatch tests.

**Risk: Y/N table:**
| INV | Fires on new opponent name? | Fires on `get_turn` method add? | Risk |
|---|---|---|---|
| INV13 (public_api) | N | N | Low |
| INV14 (buffer_schedule) | N | N | Low |
| Augment_required | N | N | Low |
| Cosine_radius | N | N | Low |
| Source_weighting | N | N | Low |
| Viewer_isolation | N | N | Low |

**Verdict:** No INV pins fire on BotProtocol shape change or new opponent. Safe to add `get_turn()` or new opponent names without test regression.

---

## (g) Other Touch Points

**Encoding:** `hexo_rl/encoding/` does not whitelist or name-check bots. Uses `lookup(name)` via registry TOML. No encoding/bot binding.

**Network:** `hexo_rl/model/network.py:304` (HexTacToeNet class) has no whitelist. Comments at lines 601, 729, 851 mention `KClusterMCTSBot` in docstrings but do not import or depend on it. Pool API is abstract (shape-agnostic).

**Model loader paths:** Grep `KrakenBot|MinimaxBot|MCTSBot|SealBot` across `hexo_rl/`:
- `hexo_rl/bots/` (5 files) — bot wrappers themselves.
- `hexo_rl/model/` (3 docstring mentions, no code dependency).
- `hexo_rl/bootstrap/corpus_metrics.py:671–718` — compound move counting for corpus stats, not model-specific.
- No model-loader names bots specifically.

**Verification:** `grep -r "KrakenBot\|MinimaxBot\|MCTSBot\|SealBot" hexo_rl/encoding/ hexo_rl/model/ --include="*.py" | grep -v "docstring\|comment"` returns 0 imports.

---

## (h) Minimal-Diff Plan (Design Only)

Goal: Add KrakenBot (all 3 variants: RandomBot, MinimaxBot, MCTSBot) + optional `BotProtocol.get_turn() -> list[(q, r)]`.

**Table: Extension approach 1 (get_turn method, opt-in):**

| File | Change | LOC est. | INV risk | Bench-gate |
|---|---|---|---|---|
| `hexo_rl/bootstrap/bot_protocol.py` | Add default `get_turn(self) -> list[tuple[int, int]]` method. Delegates to `[self.get_move(...)[0]] + self._pending_move or []`. | +15 | N | N |
| `hexo_rl/bots/krakenbot_bot.py` | Remove `_pending_move` logic. Implement `get_turn()` returning both stones. Simplify `get_move()` to pop from that list. | −10 (net) | N | N |
| `hexo_rl/bots/sealbot_bot.py` | Same as krakenbot_bot. | −10 (net) | N | N |
| `hexo_rl/eval/evaluator.py` | Optional: add `_eval_use_get_turn: bool` knob. If true, call `opponent.get_turn()` once per compound turn (saves one search). Loop refactored to batch stones. | +25 (opt) | N | N |
| **Total** | — | **+15−20 (net) or +40 with evaluator opt-in** | N | N |

**Table: Extension approach 2 (caching, current state):**

| File | Change | LOC est. | INV risk | Bench-gate |
|---|---|---|---|---|
| `hexo_rl/bots/` | Add new bot wrapper (RandomBot variant, MinimaxBot MCTS variant, etc.). Each uses `_pending_move` cache. Pattern copy from SealBotBot/KrakenBotBot. | +80–100 per bot | N | N |
| `configs/eval.yaml` | Add opponent config block (e.g., `krakenbot_random:`, `krakenbot_minimax:`, `krakenbot_mcts:`). | +10 per opponent | N | N |
| `hexo_rl/eval/opponent_runners.py` | Add runner closure for each new opponent name. Pattern copy from `_run_sealbot`. | +20 per opponent | N | N |
| `hexo_rl/eval/eval_pipeline.py` | Wire config into runner dispatch (lines 196–206). | +5–10 (loop) | N | N |
| **Total (3 KrakenBot variants)** | — | **150–180 LOC** | N | N |

**Both approaches ≤200 LOC. No INV fire. No bench-gate needed (cold paths).**

---

## (i) Verdict

**Verdict bin: `CACHING_CLEAN`**

**Mechanism justification:**

Caller loop (evaluator.py:201–210) is **stone-by-stone, sequential**. No stone-batching, no compound-move awareness above evaluator. Compound-turn bots (KrakenBot, SealBot) already cache second stone in `_pending_move` and return it on next `get_move()` call.

This pattern is clean because:

1. **Caller contract is stable** (§176 P38). Evaluator never needs to know bot is compound-returning. Single-stone per-call is the API contract.

2. **Caching is deterministic.** Second stone is bound to a specific board state (cached immediately after first stone applied). Reset guard (`reset()` + `moves_remaining > 1` check) prevents cache corruption across turn boundaries.

3. **Eval dispatch is call-by-opponent-name** (opponent_runners.py pattern). Each new bot type (KrakenBot variant) is a new opponent config entry + runner closure. Adding `get_turn()` would require evaluator loop refactor to recognize compound-returning bots, or a bot-level feature-detection method. Caching side-steps this.

4. **INV pins stable.** No schema-snapshot tests break on new opponent names or new config keys (only stride/n_games/think_time, all already in other opponents).

5. **No architectural blocker.** Evaluator loop at line 201 can dispatch to any `BotProtocol` without code change. Caching bots self-synchronize via `reset()`.

**Alternative (get_turn) rejected:** Requires either (a) evaluator loop refactor to detect and batch compound stones, or (b) a bot feature-flag to declare `is_compound_returning: bool` and handle mixed single/compound bots. Both add complexity evaluator-side when the caching pattern already works and is test-proven on SealBot + KrakenBot.

**Cost:** Approach 2 (caching) adds **150–180 LOC for 3 KrakenBot variants** (new wrappers + config + runners). Approach 1 (get_turn) saves **20 LOC per bot** in wrapper code but adds **25 LOC evaluator-side** if opt-in per-stone batching is desired.

**Recommendation for Wave B:** Use **caching + bot wrappers** (Approach 2). Lowest friction. Add `krakenbot_random`, `krakenbot_minimax`, `krakenbot_mcts` as three opponent config entries, following SealBotBot pattern verbatim. Update opponents_runners.py with three copy-paste runner closures. No evaluator refactor.

---

## Summary Stats

- **Wrapper inventory:** 5 BotProtocol implementations post-§176 P78. BotProtocol at `hexo_rl/bootstrap/bot_protocol.py:20`.
- **Compound-turn locus:** Evaluator loop (evaluator.py:201–210, stone-by-stone). Bots cache via `_pending_move`.
- **Dispatch path:** Config key → opponent_runners.py runner → evaluator.evaluate() → per-ply get_move() call.
- **INV risk:** 0 pins fire on new opponent or get_turn method.
- **Minimal-diff plan:** +150–180 LOC for 3 KrakenBot variants (caching approach). +40 LOC for get_turn opt-in (not recommended).
- **Verdict:** CACHING_CLEAN. Existing caching pattern is deterministic, stable, and needs no evaluator refactor.

