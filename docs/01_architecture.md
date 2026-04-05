# Architecture — Hex Tac Toe AlphaZero

---

## 1. Game environment

### Board representation

The board uses **axial (cube) coordinates** internally. This makes the three hex directions (E/W, NE/SW, NW/SE) trivial to express and win detection O(direction × run_length). The 2D offset representation is used only when converting to a tensor for the network.

```
Hex directions (axial):
  E:  (+1,  0)    W:  (-1,  0)
  NE: ( 0, +1)    SW: ( 0, -1)
  NW: (-1, +1)    SE: (+1, -1)
```

A 19×19 board is the practical bound. Dynamic expansion is supported: if any stone is within 3 cells of an edge, the active window grows by 2 in that direction.

### State representation

```python
@dataclass(frozen=True)
class GameState:
    board: np.ndarray        # shape (19, 19), values: 0=empty, 1=p1, -1=p2
    current_player: int      # 1 or -1
    moves_remaining: int     # 1 or 2 (p1 opens with 1, then 2 each)
    move_history: tuple      # last N board snapshots for tensor stacking
    zobrist_hash: int        # 128-bit incremental hash (splitmix128) for transposition table
    ply: int                 # total half-moves played
```

### Tensor encoding

The state is converted to a `(C, H, W)` float16 tensor for the network:

```
Channel layout (18 channels, 19×19):
  0–7:   Current player's stones in last 8 board states (binary planes)
  8–15:  Opponent's stones in last 8 board states (binary planes)
  16:    moves_remaining broadcast as 0.0 (1 move left) or 1.0 (2 moves left)
  17:    turn parity — 0.0 if ply is even, 1.0 if odd
```

#### Rust / Python encoding split

`Board.get_cluster_views()` (Rust) returns **2-plane snapshots** per cluster
(plane 0 = current player's stones, plane 1 = opponent's stones — 722 floats).
It does **not** return 18 planes because:

- Planes 1–7 and 9–15 (history) require the full move sequence, which lives
  only in Python's `GameState.move_history`.
- Encoding 18 planes in Rust and crossing the PyO3 boundary with 6 498 zeros
  per cluster would be a 9× overhead for no gain.

`GameState.to_tensor()` (Python) assembles the final `(18, 19, 19)` tensor by
stacking the current 2-plane snapshot with up to 7 prior snapshots from
`move_history`. Missing history slots (early in the game) are left as zeros.

The **Rust self-play hot-path** (`game_runner.rs`) has no Python history. It
calls `Board.encode_18_planes_to_buffer()` to expand each 2-plane view to
18 planes in-place, leaving history planes as zeros. This is a valid
approximation for the RL warmup phase; full history encoding is available on
the Python path (worker.py, evaluator.py, pretrain.py).

### Turn structure

Turn 0: player 1 places 1 stone.
Turn 1+: each player places exactly 2 stones per turn (two sequential moves).

The `moves_remaining` counter tracks whether a player has used their first move. The MCTS tree and network both see this counter — the network learns the strategic difference between the first and second placement of a double-move turn.

### Win detection (Rust bitboard)

Each player's board is stored as a bitboard: six `u64` values covering 384 bits (361 used for 19×19). Win detection is a bitwise sliding AND in each of the three hex directions:

```rust
fn check_win(board: &Bitboard, direction: Direction) -> bool {
    let shift = direction.shift();
    let mut run = board.bits;
    for _ in 0..5 {
        run &= board.bits >> shift;
    }
    run != 0  // any set bit = 6-in-a-row found
}
```

This runs in nanoseconds per move — no loop over cells, no Python overhead.

---

## 2. Neural network

### Architecture (Multi-Window Cluster-Based Approach)

To solve the "Attention Hijacking" (Colony Meta) exploit where the network becomes blind to distant lethal threats, we employ a Multi-Window Cluster-Based Approach. The infinite board is dynamically partitioned into distinct spatial clusters (colonies).

Input:

- `K` dynamic `local_map` tensors: Shape `(K, 18, 19, 19)` float16. The Rust
  core groups active stones into K distinct clusters (distanced by max 8 cells)
  and returns 2-plane snapshots per cluster via `get_cluster_views()`.
  `GameState.to_tensor()` assembles the full 18-plane tensor (see "Tensor
  encoding" above).

Backbone (Single ResNet-12 Trunk):

- Processes the `K` tensors as a single batch (effective batch size = batch_size * K) through a 19×19 ResNet-12 with Squeeze-and-Excitation (SE) blocks on every residual block (reduction ratio 4).

Value Aggregation (Pooling):

- The network outputs `K` values. The true state value is aggregated using logical pooling (e.g., `min()` if it's the opponent's turn to act in a critical colony, or a weighted average) to ensure lethal threats in any cluster override localized advantages.

Policy Mapping:

- The network outputs `K` policy distributions (each 362 logits).
- The local 19×19 coordinates of each distribution are mapped back to the absolute global `(q,r)` coordinates using the respective cluster centers provided by the Rust core.
- The aggregated legal moves are unified via a final softmax to form a single global policy vector for MCTS.

Value head (dual-pooling):
  Global avg pool(128) → (128,) | Global max pool(128) → (128,)
  Concat → (256,) → FC(256 → 256) → ReLU → FC(256 → 1) → Tanh
  Output: scalar in [-1, 1] — win probability for current player.
  Loss uses pre-tanh logit: BCE(sigmoid(v_logit), (z+1)/2) where z ∈ {-1, +1}.

### Training details

- Optimizer: AdamW, lr=2e-3, weight_decay=1e-4
- LR schedule: cosine decay over training, restarts every 200 iterations
- Loss: `L = L_policy + L_value + w_aux · L_opp_reply`
  - `L_policy = -sum(π_mcts · log π_net)` (cross-entropy with MCTS visit distribution)
  - `L_value = BCE(sigmoid(v_logit), (z+1)/2)` where z ∈ {-1, +1} is the game outcome
  - `L_opp_reply = -sum(π_opp · log π_opp_net)` (auxiliary opponent reply prediction, weight 0.15)
- Mixed precision: `torch.cuda.amp.autocast()` + `GradScaler`
- `torch.compile()`: currently disabled (CUDA graph thread-local conflict with shared inference+training model). Re-enable when architecture allows separate models.
- Batch size: 256 (fits in RTX 3070 8GB with FP16)

### Checkpointing

Every N training steps:

1. Save full model state dict + optimizer state
2. Save a separate `inference_only.pt` (weights only, for deployment)
3. Record step number, Elo estimate, and loss values in `checkpoint_log.json`

---

## 3. MCTS

### PUCT formula

```
PUCT(s, a) = Q(s,a) + c_puct · P(s,a) · sqrt(N(s)) / (1 + N(s,a))
```

- `Q(s,a)`: mean value of action a from state s (backed up from simulations)
- `P(s,a)`: prior probability from the network's policy head
- `N(s)`: total visit count of parent node
- `N(s,a)`: visit count of this action
- `c_puct`: exploration constant, default 1.5 (tunable per phase)

### Virtual loss

To support parallel MCTS (multiple threads sharing one tree), virtual loss is applied: when a thread selects a node, it temporarily decrements Q by a fixed penalty. This discourages other threads from selecting the same path simultaneously. The penalty is reversed when the real value is backed up.

### Batched leaf evaluation

The critical GPU optimization:

```
N parallel self-play games → each game runs MCTS
When any game hits a leaf node → adds state to pending_batch
When pending_batch reaches batch_size (e.g. 64) →
  single model.forward(batch) call → 64 (policy, value) pairs returned
  → distribute results back to waiting games → they continue backup
```

This keeps the GPU busy rather than idle between single-position evaluations.

### Temperature scheduling

```python
def get_temperature(ply: int, phase: str) -> float:
    if phase == "training":
        return 1.0 if ply < 30 else 0.1   # explore early, exploit late
    elif phase == "evaluation":
        return 0.0                          # deterministic (argmax)
    elif phase == "bootstrap":
        return 0.5                          # moderate exploration vs minimax data
```

### Dirichlet noise

Applied to the root node's prior during self-play (not evaluation):

```python
noise = np.random.dirichlet([dirichlet_alpha] * n_legal_moves)
P_root = (1 - epsilon) * P_net + epsilon * noise
# dirichlet_alpha = 0.3 (typical for board games)
# epsilon = 0.25
```

---

## 4. Self-play pipeline

### Worker architecture

```
Main process
  ├── InferenceServer (GPU thread)            — thin loop over Rust batch queue
  │     while True:
  │       ids, fused = batcher.next_inference_batch(B, wait_ms)
  │       log_policy, value = model(fused)
  │       batcher.submit_inference_results(ids, log_policy.exp(), value)
  ├── SelfPlayRunner (N Rust threads)
  │     Each worker thread:
  │       loop:
  │         game = new_game()
  │         while not game.terminal:
  │           features = leaf_encoder(game_state)
  │           # blocks in Rust until Python inference returns
  │           policy, value = batcher.submit_request_and_wait(features)
  │           mcts.expand_and_backup(policy, value)
  │           move = sample(mcts_policy)
  │           game.apply(move)
  │         sample_queue.push(game.records)
  └── Trainer (runs in main process, sampling from replay_buffer)
```

The hot-path concurrency is Rust-owned (not Python multiprocessing). Python is responsible for the NN forward pass only, while Rust owns game-thread scheduling, request blocking, and wake-up semantics.

### Replay buffer (Rust — ReplayBuffer)

The replay buffer lives entirely in Rust and is exposed to Python via PyO3.
The Python `ReplayBuffer` class has been deleted; `ReplayBuffer` (from `engine`) is the only buffer.

**Storage layout:**

- `states: Vec<u16>` — f16 bits stored as u16, logical shape `[capacity, 18, 19, 19]`
- `policies: Vec<f32>` — logical shape `[capacity, 362]`
- `outcomes: Vec<f32>` — logical shape `[capacity]`
- `game_ids: Vec<i64>` — multi-window correlation guard (prevents clusters from the same game appearing in one training batch)

**Key properties:**

- **12-fold hex augmentation** — applied lazily at sample time. 6 rotations × 2 (with/without reflection). Scatter-copy via pre-computed symmetry tables. Cells that fall outside the 19×19 window after transformation are left as zero.
- **Zero-copy transfer** — Python receives numpy arrays directly via PyO3's `IntoPyArray`; no type conversion in the hot path.
- **f16-as-u16 storage** — states are stored as raw u16 (f16 bit-pattern) to halve VRAM footprint; reinterpreted as f16 on PyO3 return.

**Python API:**

```python
from engine import ReplayBuffer

buf = ReplayBuffer(capacity=500_000)
buf.push(state_f16, policy_f32, outcome_f32, game_id)        # single position
buf.push_game(states_f16, policies_f32, outcomes_f32, game_id)  # full game batch
states, policies, outcomes = buf.sample_batch(batch_size, augment=True)
```

**Performance (2026-03-31 baseline, 16 workers):** 219,444 pushes/sec; 936 µs/batch (3.66 µs/pos, augmented, batch=256); raw (no augmentation) 951 µs/batch.

---

## 5. Reward design

### Terminal rewards (default, always active)

| Outcome | Reward for winner | Reward for loser |
|---|---|---|
| Win by 6-in-a-row | +1.0 | -1.0 |
| Draw (if applicable) | -0.5 | -0.5 |

Negative draw reward (configurable via `draw_reward` in `configs/training.yaml`). Teaches the network to press for wins rather than accept draws. In a game with ~51.6% P1 win rate, draws are suboptimal for the stronger player. Source: KrakenBot practice (docs/10_COMMUNITY_BOT_ANALYSIS.md §5.1D). Changed from +0.01 on 2026-04-04; raised from -0.1 to -0.5 on 2026-04-05 after first overnight self-play run produced 56% draws — at that frequency -0.1 was dominated by win/loss signal and failed to discourage draw-seeking.

### Optional shaped rewards (decay to zero)

Applied during early training only. Weight: `w = max(0.0, 1.0 - iteration / decay_steps)`.

**Use formation vocabulary, not generic n-in-a-row counting.** The community knowledge base defines a precise hierarchy of tactical formations. Rewarding these is more semantically meaningful than rewarding arbitrary run lengths — and critically, it teaches the network to pursue *structurally winning* positions rather than just long lines.

| Formation event | Shaped reward |
|---|---|
| Singlet (pre-emp) created | +0.01 |
| Singlet blocked | +0.005 |
| Threat created | +0.03 |
| Threat blocked | +0.02 |
| Double threat created | +0.08 |
| Triangle created (`x A0 A1`) | +0.20 |
| Open Three created (`x A0 A3`) | +0.15 |
| Rhombus / Arch created | +0.18 |
| Ladder created | +0.25 |
| Bone created | +0.15 |
| Opponent reached unstoppable formation | -0.30 |

Formation detection runs incrementally in Rust — scan only from the newly placed piece across the 3 hex axes, store segment endpoints. Do not rescan the full board. See `05_COMMUNITY_INTEGRATION.md` for the `FormationDetector` Rust API.

Shaped rewards are returned alongside the terminal reward signal.

**Warning (confirmed by community field report):** An RL bot in this community was directly observed farming threat rewards instead of finishing games. This is precisely what happened when shaped rewards were weighted too strongly without decay. The decay schedule is mandatory, not optional. Terminal reward must always dominate.

---

## 6. Evaluation

### Elo ladder

Every `eval_interval` training steps, the new checkpoint plays a round-robin tournament against:

- Previous checkpoint
- Best checkpoint so far
- SealBot engine (time_limit=0.03)
- SealBot engine (time_limit=0.10)

The **win rate vs SealBot** is the primary community benchmark. Target: ≥ 55% win rate over 100 games = meaningfully stronger than the current best public bot.

Elo is updated using standard formula:

```
E_a = 1 / (1 + 10^((R_b - R_a) / 400))
R_a_new = R_a + K * (S_a - E_a)   # K=32, S_a ∈ {0, 0.5, 1}
```

If the new checkpoint beats the current best by a win rate ≥ 55%, it becomes the new best and self-play workers load its weights.

### Human-comparable benchmarks

As Elo grows, periodically generate games against community members (via exported SGF or custom format) for human review. This is qualitative — the community's strategic intuition is ground truth for whether the AI is "really playing well."

---

## 7. Rust / Python interface (PyO3)

### Exposed API

```python
from engine import Board, MCTSTree

# Board
b = Board(size=19)
b.apply_move(row, col, player)
b.check_win()           # -> bool
b.legal_moves()         # -> list[tuple[int,int]]
b.to_tensor()           # -> np.ndarray (channels, H, W)
b.zobrist_hash()        # -> int

# MCTS (tree lives in Rust, Python drives the search)
tree = MCTSTree(c_puct=1.5, dirichlet_alpha=0.3, epsilon=0.25)
tree.new_game(board)
leaves = tree.select_leaves(n=64)              # returns batch of states needing eval
tree.expand_and_backup(leaves, policies, values)  # feed network results back
policy = tree.get_policy(temperature=1.0)

# Benchmarking
stats = GameBenchmarks.run_mcts_throughput(n_simulations=10_000)
# -> {"simulations_per_sec": float, "tree_depth_avg": float, ...}
```

### Viewer / analysis exports

```python
# Threat detection (viewer-only — never called from MCTS or training)
threats = b.get_threats()  # -> list[(q, r, level, player)]
# Scans all 3 hex axes with a sliding window of width 6.
# A threat is a window where one player has N stones (N≥3) and the rest are empty.
# Returns the EMPTY cells within threatening windows, not the stone cells.
# Levels: 3=warning, 4=forced, 5=critical.

# MCTS tree introspection (used by viewer play-against-model)
top = tree.get_top_visits(n=5)  # -> list[(coord_str, visits, prior)]
v = tree.root_value()           # -> float (value at root for current player)
```

### Build

```bash
cd engine
cargo build --release
maturin develop --release -m engine/Cargo.toml  # installs into current Python env
```

---

## 8. Monitoring

Event-driven fan-out. `train.py` calls `emit_event(payload)` which dispatches
to all registered renderers. Two built-in renderers:

- **Terminal dashboard** (`hexo_rl/monitoring/terminal_dashboard.py`): Rich Live
  panel, updates at max 4Hz. Shows loss/throughput/buffer/system stats. Alert
  line for mode collapse (entropy < 1.0), grad spikes (norm > 10), loss increase
  runs, eval gate failures.

- **Web dashboard** (`hexo_rl/monitoring/web_dashboard.py`): Flask+SocketIO on
  localhost:5001. Serves `index.html` SPA. Event history replay on browser
  reconnect (last 500 events). Routes: `/` (dashboard), `/viewer` (game viewer),
  `/viewer/game/<id>`, `/viewer/recent`, `/viewer/play`.

**Invariant:** no renderer is imported by training, selfplay, or engine code.
Renderers are passive observers that receive events via `register_renderer()`.

See `docs/08_DASHBOARD_SPEC.md` for the full event schema and
`docs/09_VIEWER_SPEC.md` for the game viewer specification.
