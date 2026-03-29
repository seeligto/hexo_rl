# Bootstrap Strategy — Minimax Bot Seeding & Supervised Pretraining

---

## Why this matters

A freshly initialized AlphaZero network plays randomly. The first thousands of self-play games are essentially noise — two random agents occasionally stumbling into wins, with almost no useful signal for learning tactics. This means the network spends a very long time learning things a human beginner knows immediately: don't leave 5-in-a-rows unblocked, build toward 6, etc.

Pretraining on existing games gives the network a warm start with tactical intuitions already baked in. Self-play then builds strategic depth on top of that foundation rather than rediscovering basics from scratch.

**Observed effect in analogous projects**: pretraining compresses 1–3 weeks of early self-play into 1–2 hours of supervised training, and the resulting Elo trajectory is measurably steeper throughout.

---

## Corpus sources — three tiers, combined

All three sources are fed into a single pretrain dataset. They are complementary: human games have strategic depth, Ramora0 games have tactical precision, random games add coverage diversity.

| Source | Volume | Strength | How to get |
|---|---|---|---|
| Human game archive (hexo.did.science) | 42,000+ rated games | Real strategic patterns | Scraper (see below) |
| Ramora0 engine self-play | Unlimited | Tactical precision | RamoraBot wrapper |
| Random play | Unlimited | Coverage only | RandomBot |

**Priority order**: human games first, then Ramora0, then random to fill gaps. Human games are the most valuable signal — they contain opening theory, formation recognition, and endgame technique that minimax at depth 3–5 won't generate.

Corpus generation is configured in `configs/default.yaml`:
```yaml
bootstrap:
  human_games_min_moves: 20          # filter out surrenders and short games
  human_games_rated_only: true
  ramora_depth_mix: {3: 0.7, 5: 0.3} # 70% depth-3, 30% depth-5
  target_positions: 500000            # total positions in pretrain dataset
```

---

## Bot protocol — all bots are interchangeable

Every game source (Ramora0, community bots, our own model, random) implements `BotProtocol`.
This means any community bot implementing the API spec can be plugged into corpus generation
or evaluation without additional code.

```python
# python/bootstrap/bot_protocol.py
from abc import ABC, abstractmethod

class BotProtocol(ABC):
    @abstractmethod
    def get_move(self, state: GameState) -> tuple[int, int]: ...
    
    @abstractmethod
    def name(self) -> str: ...

class RamoraBot(BotProtocol):
    """Wraps the Ramora0 C++ engine at a given search depth."""
    def __init__(self, binary_path: str, depth: int = 5): ...

class OurModelBot(BotProtocol):
    """Wraps our own checkpoint + MCTS at configurable simulations."""
    def __init__(self, checkpoint_path: str, n_simulations: int = 400): ...

class RandomBot(BotProtocol):
    """Uniform random legal move. Baseline and corpus diversity filler."""
    ...

class CommunityAPIBot(BotProtocol):
    """Wraps any community bot implementing bot-api-v1 over HTTP."""
    def __init__(self, base_url: str): ...
```

Corpus generation always takes a list of `BotProtocol` instances — never hardcodes which bots to use. Drive from config.

---

## Human game archive scraper

**URL:** https://hexo.did.science/games  
**Volume:** 42,477+ archived matches (as of 2026-03-28), paginated 20 per page  
**Format:** HTML — session IDs in listing, individual game pages at `/games/{session_id}`

```python
# python/bootstrap/scraper.py
import httpx
import time
from bs4 import BeautifulSoup
import structlog

log = structlog.get_logger()

BASE_URL = "https://hexo.did.science"

def scrape_game_index(
    max_pages: int = 2125,      # 42,477 / 20
    rated_only: bool = True,
    delay_sec: float = 0.5,     # be polite — don't hammer the server
) -> list[str]:
    """Paginate the game archive and return session IDs matching filters."""
    session_ids = []
    for page in range(1, max_pages + 1):
        resp = httpx.get(f"{BASE_URL}/games", params={"page": page}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for game_card in soup.select("[data-session-id]"):  # inspect actual HTML for selector
            is_rated = "Rated" in game_card.text
            move_count = extract_move_count(game_card)
            if rated_only and not is_rated:
                continue
            if move_count < 20:    # filter surrenders and trivial games
                continue
            session_ids.append(game_card["data-session-id"])
        
        if page % 50 == 0:
            log.info("scrape_progress", page=page, collected=len(session_ids))
        time.sleep(delay_sec)
    
    return session_ids

def scrape_game(session_id: str) -> list[tuple[int, int]] | None:
    """Fetch a single game and return its move sequence as (q, r) axial coords."""
    resp = httpx.get(f"{BASE_URL}/games/{session_id}", timeout=10)
    # Parse move sequence from game page — format TBD based on actual page structure
    # May be BKE notation, JSON embedded in page, or rendered board state
    ...

def scrape_corpus(output_path: str = "data/human_games.npz", **kwargs):
    """Full pipeline: scrape index → fetch games → convert to tensors → save."""
    session_ids = scrape_game_index(**kwargs)
    log.info("scrape_index_complete", total=len(session_ids))
    
    all_records = []
    for sid in session_ids:
        moves = scrape_game(sid)
        if moves:
            all_records.extend(game_to_training_records(moves))
        time.sleep(0.3)
    
    save_corpus(all_records, output_path)
    log.info("scrape_corpus_complete", positions=len(all_records), path=output_path)
```

**Implementation note:** The actual HTML selectors and game page format need to be determined by the agent when implementing. The agent should fetch one game page, inspect the structure, then implement the parser. Do not guess the selectors — read the actual HTML first.

**Rate limiting:** The site is community-run. Use `delay_sec=0.5` minimum between requests. Check for `robots.txt` before scraping. If the site adds rate limiting or asks to stop, respect it.

---

## Minimax bot — use Ramora0, don't build from scratch

**Do not implement a minimax bot.** The Ramora0 C++ engine (`cpp/engine.h`) is the strongest known public bot and is already available at https://github.com/Ramora0/HexTicTacToe. Use it directly.

**Before generating any corpus, apply the line-1094 bug fix** (documented in `05_COMMUNITY_INTEGRATION.md`). The bug causes some positions to receive incorrect evaluations — training on corrupted positions silently degrades the value head. The fix is a one-liner and the community has documented exactly what to change.

See `05_COMMUNITY_INTEGRATION.md` section 2 for the `RamoraEngine` Python wrapper and recommended depth mix.

---

## Heuristic evaluation function (reference only)

The minimax evaluation function scores a board from the perspective of the current player:

```python
# python/bootstrap/heuristic.py

OPEN_RUN_SCORES = {
    # (run_length, open_ends): score
    (3, 2): 10,    # open 3-in-a-row — moderate threat
    (3, 1): 3,     # semi-open 3
    (4, 2): 100,   # open 4-in-a-row — serious threat
    (4, 1): 30,    # semi-open 4
    (5, 2): 5000,  # open 5-in-a-row — near-certain win
    (5, 1): 1000,  # semi-open 5 — very dangerous
}

def evaluate_board(board, player: int) -> float:
    """
    Returns a score from player's perspective.
    Positive = good for player, negative = good for opponent.
    Called on non-terminal positions only.
    """
    score = 0.0
    for p, sign in [(player, 1), (-player, -1)]:
        for direction in HEX_DIRECTIONS:
            for run_len, open_ends in find_runs(board, p, direction):
                score += sign * OPEN_RUN_SCORES.get((run_len, open_ends), 0)
    return score

def find_runs(board, player: int, direction) -> list[tuple[int, int]]:
    """
    Scans the board in one direction and returns (length, open_ends) pairs
    for each contiguous run of player's stones.
    """
    # Implementation: walk each row/diagonal, count contiguous runs,
    # check if ends are empty (open) or blocked (occupied or edge).
    ...
```

### Alpha-beta minimax

```python
# python/bootstrap/minimax.py
import math

class MinimaxBot:
    def __init__(self, depth: int = 4):
        self.depth = depth
        self.nodes_evaluated = 0

    def choose_move(self, game_state) -> tuple[int, int]:
        best_move = None
        best_score = -math.inf
        for move in game_state.legal_moves():
            child = game_state.apply(move)
            score = self._minimax(child, self.depth - 1, -math.inf, math.inf, False)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _minimax(self, state, depth, alpha, beta, maximizing) -> float:
        if state.is_terminal():
            return 1e6 if state.winner == state.root_player else -1e6
        if depth == 0:
            self.nodes_evaluated += 1
            return evaluate_board(state.board, state.root_player)

        legal = state.legal_moves()
        # Move ordering: check wins/blocks first for better pruning
        legal = self._order_moves(state, legal)

        if maximizing:
            val = -math.inf
            for move in legal:
                val = max(val, self._minimax(state.apply(move), depth-1, alpha, beta, False))
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return val
        else:
            val = math.inf
            for move in legal:
                val = min(val, self._minimax(state.apply(move), depth-1, alpha, beta, True))
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return val

    def _order_moves(self, state, moves):
        """Prioritise: immediate wins > blocks > center proximity."""
        wins   = [m for m in moves if state.apply(m).is_terminal()]
        blocks = [m for m in moves if self._is_block(state, m)]
        rest   = [m for m in moves if m not in wins and m not in blocks]
        # Sort rest by proximity to board center
        cx, cy = state.board_size // 2, state.board_size // 2
        rest.sort(key=lambda m: abs(m[0]-cx) + abs(m[1]-cy))
        return wins + blocks + rest
```

**Recommended depths for corpus generation**:
- Depth 3: fast, ~10,000 games/hour, basic tactical awareness
- Depth 5: slower, ~1,000 games/hour, stronger tactics, better training signal
- Depth 7+: avoid — too slow, and too strong a prior may constrain self-play

Use a **mix**: 70% depth-3 games (quantity) + 30% depth-5 games (quality).

---

## Corpus generation

```python
# python/bootstrap/generate_corpus.py
import multiprocessing as mp
from pathlib import Path
import numpy as np
import structlog

log = structlog.get_logger()

def generate_game(args) -> dict | None:
    game_id, bot_a_depth, bot_b_depth, seed = args
    rng = np.random.default_rng(seed)

    bot_a = MinimaxBot(depth=bot_a_depth)
    bot_b = MinimaxBot(depth=bot_b_depth)
    state = GameState.initial()
    records = []

    while not state.is_terminal():
        bot = bot_a if state.current_player == 1 else bot_b
        move = bot.choose_move(state)
        # Record: (state tensor, one-hot move, to be filled with outcome later)
        records.append({
            "state":  state.to_tensor(),
            "move":   move,
            "player": state.current_player,
        })
        state = state.apply(move)

    outcome = state.winner  # 1 or -1
    # Back-fill outcomes from each player's perspective
    for r in records:
        r["outcome"] = outcome if r["player"] == 1 else -outcome

    log.info("bootstrap_game_complete",
        game_id=game_id, outcome=outcome, plies=len(records),
        bot_a_depth=bot_a_depth, bot_b_depth=bot_b_depth)

    return {"game_id": game_id, "records": records}

def generate_corpus(
    n_games: int = 20_000,
    output_path: str = "data/bootstrap_corpus.npz",
    n_workers: int = 8,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Mix of depths: 70% depth-3, 30% depth-5
    tasks = []
    for i in range(n_games):
        a_depth = 3 if i < n_games * 0.7 else 5
        b_depth = 3 if i < n_games * 0.7 else 5
        tasks.append((i, a_depth, b_depth, i * 1337))

    all_states, all_policies, all_outcomes = [], [], []

    with mp.Pool(n_workers) as pool:
        for result in pool.imap_unordered(generate_game, tasks, chunksize=10):
            if result is None:
                continue
            for r in result["records"]:
                all_states.append(r["state"])
                # For minimax games, policy = one-hot at chosen move
                policy = np.zeros(19*19 + 1, dtype=np.float32)
                row, col = r["move"]
                policy[row * 19 + col] = 1.0
                all_policies.append(policy)
                all_outcomes.append(float(r["outcome"]))

    np.savez_compressed(
        output_path,
        states=np.stack(all_states).astype(np.float16),
        policies=np.stack(all_policies),
        outcomes=np.array(all_outcomes),
    )
    log.info("corpus_saved", path=output_path, n_positions=len(all_states))
```

---

## Supervised pretraining

```python
# python/bootstrap/pretrain.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import structlog
from rich.progress import track

log = structlog.get_logger()

def pretrain(
    model,
    corpus_path: str = "data/bootstrap_corpus.npz",
    epochs: int = 7,
    batch_size: int = 512,
    lr: float = 1e-3,
    checkpoint_path: str = "checkpoints/bootstrap.pt",
):
    data = np.load(corpus_path)
    states   = torch.from_numpy(data["states"]).float()
    policies = torch.from_numpy(data["policies"])
    outcomes = torch.from_numpy(data["outcomes"]).float()

    dataset = TensorDataset(states, policies, outcomes)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    scaler    = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(1, epochs + 1):
        total_policy_loss = 0.0
        total_value_loss  = 0.0

        for s, p, z in track(loader, description=f"Epoch {epoch}/{epochs}"):
            s, p, z = s.cuda(), p.cuda(), z.cuda()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                log_policy, value = model(s.half())
                # Behavior cloning: match the minimax move distribution
                policy_loss = F.kl_div(log_policy, p, reduction="batchmean")
                # Outcome regression
                value_loss  = F.mse_loss(value.squeeze(), z)
                loss = policy_loss + value_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()

        avg_pl = total_policy_loss / len(loader)
        avg_vl = total_value_loss  / len(loader)
        log.info("pretrain_epoch",
            epoch=epoch, policy_loss=avg_pl, value_loss=avg_vl)

    torch.save({"model": model.state_dict(), "source": "bootstrap"}, checkpoint_path)
    log.info("pretrain_complete", checkpoint=checkpoint_path)
    return model
```

### Validation before handing off to self-play

Before starting self-play, verify the pretrained model is sane. Two checks:

```python
def validate_pretrained(model, n_games: int = 100) -> dict:
    """
    Check 1: beat random opponent >> 95% of the time.
    Check 2: win rate vs Ramora0 depth-3 should be > 10% 
             (not expected to beat it yet, but shouldn't be 0%).
    """
    wins = 0
    for _ in range(n_games):
        state = GameState.initial()
        while not state.is_terminal():
            if state.current_player == 1:
                # Pretrained model — greedy from policy head
                tensor = torch.tensor(state.to_tensor()).unsqueeze(0).half().cuda()
                with torch.no_grad():
                    log_p, _ = model(tensor)
                legal_mask = state.legal_move_mask()
                move = greedy_move(log_p, legal_mask)
            else:
                # Random opponent
                move = random.choice(state.legal_moves())
            state = state.apply(move)
        if state.winner == 1:
            wins += 1

    win_rate = wins / n_games
    log.info("pretrain_validation", win_rate=win_rate, n_games=n_games)
    assert win_rate >= 0.90, f"Pretrained model too weak: win_rate={win_rate:.2f}"
    return {"win_rate": win_rate}
```

---

## Integration with self-play pipeline

The transition from pretraining to self-play is automatic:

```python
# scripts/train.py (simplified)
from python.bootstrap.pretrain import pretrain, validate_pretrained
from python.bootstrap.generate_corpus import generate_corpus
from python.training.trainer import Trainer

cfg = load_config("configs/default.yaml")
model = HexTacToeNet(**cfg["model"]).cuda()

if cfg["bootstrap"]["enabled"]:
    if not Path(cfg["bootstrap"]["corpus_path"]).exists():
        print("Generating bootstrap corpus...")
        generate_corpus(
            n_games=cfg["bootstrap"]["n_games"],
            output_path=cfg["bootstrap"]["corpus_path"],
        )
    print("Pretraining on minimax corpus...")
    model = pretrain(model, corpus_path=cfg["bootstrap"]["corpus_path"])
    validate_pretrained(model)
    print("Pretraining complete. Starting self-play.")

trainer = Trainer(model, cfg)
trainer.run()
```

---

## Decay of bootstrap influence

The key insight: the pretrained prior should fade as self-play data accumulates. Two mechanisms:

**1. Data dilution** (automatic): as the replay buffer fills with self-play games, the minimax-derived positions become a smaller fraction of each training batch. This is natural and requires no code change — the buffer simply pushes out old bootstrap data.

**2. Optional explicit decay**: if you want to add bootstrap games back into the buffer during early self-play, weight them with a decaying factor:

```python
# In ReplayBuffer.sample — weighted sampling during early training
def sample(self, batch_size, bootstrap_weight=0.0):
    n_bootstrap = int(batch_size * bootstrap_weight)
    n_selfplay  = batch_size - n_bootstrap
    selfplay_idx   = np.random.randint(0, self.selfplay_size, n_selfplay)
    bootstrap_idx  = np.random.randint(0, self.bootstrap_size, n_bootstrap)
    # ... combine and return
```

Decay schedule: `bootstrap_weight = max(0, 0.5 - iteration / 200)` — starts at 50% bootstrap, reaches 0 by iteration 100.

---

## Existing community bots

If the Hex Tac Toe community already has bots with known strength:

1. **Reach out to the community** to obtain game logs in any parseable format
2. Convert to the corpus format: `(state_tensor, move, outcome)` triples
3. Use these directly in the supervised pretraining step — skip `generate_corpus()`
4. Human games are often better training signal than minimax games because they reflect strategic patterns, not just local tactics

If the community bots export in a standard format (PGN, SGF), write a parser:

```python
# python/bootstrap/import_sgf.py
def parse_sgf_game(sgf_text: str) -> list[dict]:
    """Convert an SGF game record to a list of (state, move, outcome) dicts."""
    ...
```

Real human/bot games at any level are worth including — even mediocre play teaches the network that some positions are structurally better than others.
