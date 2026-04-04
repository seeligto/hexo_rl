# Community Bot Architecture Analysis

> Generated 2026-04-04. Sources: GitHub repos read via MCP.
> Some files in KrakenBot and Orca were not accessible due to API rate limits —
> gaps are noted inline.

---

## 1. KrakenBot Architecture Summary

**Repo:** `Ramora0/KrakenBot` — by imaseal (the SealBot developer)
**Language:** Python + Cython (MCTS hot path) + C++ (minimax engine, pybind11)
**Claimed strength:** Strongest NN bot for Hex Tac Toe

### 1.1 Network: HexResNet

| Parameter | Value |
|---|---|
| Input planes | **2** (current player, opponent) — no history, no threats |
| Board size | **25×25** (fixed, toroidal) |
| Stem | Conv2d 3×3 circular → GroupNorm(8) → ReLU |
| Residual blocks | **10** (default) |
| Channels | **128** |
| Normalization | **GroupNorm** (8 groups) — not BatchNorm |
| SE blocks | **None** |
| Padding mode | **`circular`** (torus self-play) / **`zeros`** (infinite grid eval) — runtime switchable via `set_padding_mode()` |
| Estimated params | ~1.5M (10 blocks × 128ch, no SE overhead) |

**Four output heads:**

1. **Value head:** Conv1×1(128→32) → GroupNorm → ReLU → masked global avg+max pool
   → concat(64) → FC(64→256) → ReLU → FC(256→1) → **tanh**. Output: scalar in [-1, 1].

2. **Pair policy head (PairPolicyHead):** Bilinear attention over cell embeddings.
   Two 1×1 convolution projections (Q and K, both 128→64). Pair logits:
   `A = (Q^T · K) / sqrt(d)`, symmetrized `A = (A + A^T) / 2`, clamped [-100, 100].
   Diagonal masked to -∞ (can't place both stones on same cell). Padding cells masked.
   **Output: `[B, 625, 625]`** — a full N×N attention matrix where entry (i,j) is the
   log-probability of placing stone 1 at cell i and stone 2 at cell j.
   **Marginalization:** `pair_logits.logsumexp(dim=-1)` → single-move logits `[B, 625]`
   for stone-1 selection in MCTS.

3. **Moves-left head:** Shares pooled features with value head.
   FC(64→256) → ReLU → FC(256→1) → ReLU. Predicts remaining game moves.

4. **Chain head:** Conv1×1(128→32) → ReLU → Conv1×1(32→6).
   Output: `[B, 6, 25, 25]` — per-cell unblocked chain length for current player
   (3 hex directions) + opponent (3 directions). Trained with masked MSE.

### 1.2 Torus Encoding

- **`TORUS_SIZE = 25`**: all coordinates taken mod 25. Win detection wraps around edges.
- Circular CNN padding makes the network's receptive field seamlessly wrap around
  the torus boundaries — no edge artifacts, no dead padding zones.
- For infinite-grid evaluation, padding switches to `zeros` and a dynamic bounding box
  (margin=6, min_size=25) is used to create a proxy `ToroidalHexGame` with coordinate offsets.
- **Self-play is exclusively on the torus.** The torus is a proxy for the infinite grid:
  at 25×25 = 625 cells the board is large enough that wrap-around collisions are rare
  in practice (games typically occupy <100 cells).

### 1.3 Compound Action Space (Pair Logits)

This is KrakenBot's most distinctive architectural choice:

- The N×N pair attention matrix **jointly models both stones** of a 2-stone turn.
  The network sees the correlation between stone placements — e.g., it can learn
  that placing stone 1 as a threat and stone 2 as a block is a good compound move.
- **Symmetrization** `(A + A^T)/2` encodes the insight that the order of placement
  within a turn doesn't matter (both stones land before the opponent moves).
- **Marginalization** via logsumexp gives single-stone logits for the two-level
  MCTS tree at the root.
- **Cost:** The 625×625 = 390,625 logit matrix is large but sparse in practice
  (most cells are masked). The bilinear projection (two 1×1 convs) is lightweight.

### 1.4 MCTS

| Parameter | Value |
|---|---|
| PUCT C | 1.0 |
| FPU | KataGo-style dynamic (reduction=0.25, grows with √visited_mass) |
| Dirichlet | α = 10/n_candidates, frac = 0.25 |
| Self-play sims | 200 (quick) / 600 (full, 25% probability) |
| Max tree depth | 50 |
| Non-root top-K pairs | 50 |
| Candidate distance | 8 (hex distance from nearest stone) |

**Two-level compound move tree at root:**
- Level 1: stone_1 selection via PUCT on marginalized priors
- Level 2: stone_2 selection via PUCT on conditional priors `pair_probs[s1_idx]`
- Level-2 nodes expanded lazily on first visit to each stone_1 action
- Children indexed by `(s1_idx, s2_idx)` tuple

**Non-root flat pair selection:**
- Actions encoded as `s1 * 625 + s2` (flat pair index)
- Top-50 pairs used as candidates — no Dirichlet noise

**Backpropagation:**
- Sign alternates at **pair boundaries** (not per-stone). Within a pair both entries
  get the same sign; across pair boundaries the sign flips.

**Temperature schedule:**
- Turns 0–5: T=2.5 (very exploratory)
- Turns 6–19: T=1.0
- Turn 20+: T=0.3

**Cython acceleration:** `_mcts_cy.pyx` (18KB) and `_puct_cy.pyx` (2.3KB) accelerate
leaf selection and backpropagation.

### 1.5 Training Pipeline

**Two-stage pipeline:**
1. **Distillation:** Generate positions from early human games (≤11 stones) played
   out by C++ MinimaxBot (0.04s ±0.015s per move). Store as Parquet with board,
   current_player, pair moves, eval_score, win_score, winning_singles, winning_pairs.
   MAX_BOARD_SPAN = 19 (discard if bounding box > 19×19).

2. **Self-play RL:** Resume from distillation checkpoint. Batch 256 games in lockstep.
   Double-buffered GPU overlap (process group A on GPU while doing CPU work for group B).
   150-move game cap. Draw penalty = -0.1. Cold start = 1024 games first round.

**Loss functions (distillation):**
- **Value:** MSE(predicted, win_target)
- **Policy (pair CE):** Cross-entropy on flattened N² pair logits against target pair
  `(m1*N + m2)`. For winning positions, replaced by `-log P(winning_pair_set)`.
- **Entropy regularization:** weight 0.01
- **Moves-left:** MSE on normalized predictions (÷150)
- **Chain:** Masked MSE on per-cell chain predictions
- Default weights: value=1.0, policy=1.0, entropy=0.01, moves_left=0.1, chain=0.1

**Optimizer:** Adam, lr=1e-3, weight_decay=1e-4
**Scheduler:** Cosine annealing `0.5 * (1 + cos(π * progress))`
**AMP:** GradScaler when `--amp` flag used
**Batch size:** 256
**Grad clipping:** `clip_grad_norm_(params, 50.0)`
**Validation split:** 20% by game_id (no position leakage)
**Symmetry augmentation:** D6 (12 transforms) — 6 rotations × 2 reflections, precomputed
permutation tables `[12, 625]` with direction-aware chain permutations.

### 1.6 C++ Engine

- `engine.h` (60KB) — full minimax engine with iterative deepening, alpha-beta,
  pattern-based evaluation
- `ai_cpp.cpp` — pybind11 wrapper exposing `MinimaxBot`
- `types.h` — `GameState` = vector of `Cell{q,r,player}` + cur_player + moves_left
- `ankerl_unordered_dense.h` — high-performance hash map for board storage
- `pattern_data.h` — pattern evaluation data
- Methods: `has_instant_win()`, `has_near_threats()`, PV extraction, pickle support

### 1.7 Developer Notes (scratch.txt, todo.md)

Pipeline command sequence:
```
distill train (--amp --epochs 3) → self-play train (--resume from distill) → eval vs minimax
```

TODO items (incomplete): quiescence search, weaker minimax opponent, checkpoint gating
(not yet implemented), TensorRT inference.

---

## 2. Orca Architecture Summary

**Repo:** `Saiki77/hexbot-building-framework` — by Saiki77
**Language:** Python + C (ctypes FFI, auto-compiles on first import)
**Distribution:** PyPI package `hexbot`, ships with pretrained weights (65 iterations)
**Claimed strength:** Strong framework bot; multiple architecture variants available

### 2.1 Network: HexNet / HybridHexNet

**Standard HexNet:**

| Parameter | Value |
|---|---|
| Input planes | **7** (own stones, opp stones, legal moves, player indicator, stones remaining, own threats, opp threats) |
| Board size | **19×19** (centroid-windowed) |
| Stem | Conv2d 3×3 pad=1 no-bias → BN → ReLU |
| Residual blocks | **12** |
| Channels | **128** (standard) / **256** (large/hybrid) |
| Normalization | **BatchNorm** |
| SE blocks | **No** (standard) / **Yes, ratio=4** (HybridHexNet) |
| Padding mode | **zeros** (standard) / **circular** (hex-native-circular variant) |
| Estimated params | ~3.9M (standard) / ~5.2M+ (hybrid-256ch) |

**Three output heads:**

1. **Policy head:** Conv1×1(128→2) → BN → ReLU → flatten → Linear(722, 361).
   Output: `[B, 361]` single-cell logits.

2. **Value head:** Conv1×1(128→1) → BN → ReLU → flatten → Linear(361, 256) → ReLU
   → Linear(256, 1) → **tanh**. Output: scalar in [-1, 1].

3. **Threat head:** Conv1×1(128→1) → BN → ReLU → spatial map **blended into policy
   logits** with weight 0.3 → flatten → Linear(361, 4). Output: 4 threat floats
   (own_level, own_multi_axis, opp_level, opp_multi_axis).

**Key trick — threat-policy blend:** The threat head's spatial activation map is added
directly to policy logits before softmax (weight `THREAT_POLICY_BLEND=0.3`), biasing
MCTS toward threat-relevant cells. This creates a feedback loop where the network
learns to attend to threats AND the search naturally explores them.

**HybridHexNet (CNN + Global Attention):**
- 12× SEResBlock (squeeze ratio 4) instead of plain ResBlocks
- After ResBlocks: reshape to (B, 361, C) → learnable positional embedding →
  2× MultiheadAttention (8 heads, pre-norm, residual) → reshape back to spatial
- Default 256 channels; `hybrid-small` = 128ch, 6 blocks, 4 heads, 1 attn layer

**Alternative architectures available:**
- **TransformerHexNet:** 12 ResBlocks + 2 transformer layers (8 heads, FF dim=512, GELU).
  ~5.2M params. ~30% slower per step, untested competitively.
- **HexGNN:** Graph neural network on hex adjacency. 8 message-passing layers,
  attention-weighted global pooling for value head. ~2.5M params. All D6 symmetries
  are trivial on graphs. Variable board size without retraining.
- **HexMaskedNet:** Standard architecture but 3×3 kernels masked to only hex-neighbor
  positions (6 of 9 weights active). Same param count, same speed, respects hex topology.
- **HexNativeNet:** Only 7 learnable weights per kernel (center + 6 neighbors).
  ~3.0M params (fewer due to 7 vs 9 weights). Optional circular padding for torus.
- **MultiscaleNet:** Local CNN + global attention two-tower. ~1.1M params (compact).

**Checkpoint migration:** Can upgrade channel count (5→7) and filter width (64→128)
without retraining from scratch — zero-pads new channels, small-random-pads new filters.

### 2.2 Board Encoding (7 Channels)

| Plane | Content |
|---|---|
| 0 | Current player's stones |
| 1 | Opponent's stones |
| 2 | Legal moves (candidate cells) |
| 3 | Current player indicator (constant 0 or 1) |
| 4 | Stones remaining this turn (normalized ÷2) |
| 5 | Current player's **threat map** (cells where placing creates 4+ in a row, scaled min(count/6, 1)) |
| 6 | Opponent's **threat map** (same scale) |

**Windowing:** Centroid of all occupied cells → center 19×19 window. Moves outside the
window get zero probability (invisible to the network).

**Threat label computation:** Continuous scale — 2-in-a-row=0.15 (preemptive), 3=0.40,
4=0.80, 5=0.85, 6+=1.0. Multi-axis score detects proto-forks (2+ axes with 2+=0.50,
2+ axes with 3+=1.0). Only counts "live" lines that can still reach 6.

### 2.3 MCTS and Search

| Parameter | Value |
|---|---|
| PUCT C | **1.5** |
| Dirichlet | α=0.3, ε=0.25 |
| Self-play sims | **400** (default, curriculum overrides: 30→200) |
| Batch size | 64 (batched MCTS) |
| Temperature threshold | 35 moves (then greedy) |
| Virtual loss | Yes (batched MCTS) |
| Transposition cache | Dict-based, 100K entries, class-level shared |

**Progressive widening (distinctive feature):**
- Initial expansion: only 6 children
- At 20 visits → 10 children, 50 → 16, 100 → 25
- Forces tree **deeper instead of wider** — critical for connect-6 where seeing
  several full 2-stone turns ahead matters more than exploring many moves at one position.

**Quiescence in MCTS:** If C engine reports 3+ winning moves for either side, overrides
NN value to ±1.0 (forced win — opponent can block at most 2 per turn). For 2 winning
moves, blends 0.3 boost.

**AB Hybrid pre-check:** Before MCTS, runs C engine alpha-beta at depth 4. If proven
win/loss (`|value| ≥ 1.0`), returns immediately without MCTS.

**Three search engines available:**
1. **MCTS** (single-threaded, standard)
2. **BatchedMCTS** (virtual loss, batch NN eval, AB pre-check, transposition cache)
3. **BatchedNNAlphaBeta** (three-phase: C alpha-beta collects leaves → batch NN eval →
   re-search with cached values. 15× faster than per-leaf callbacks. Reaches depth 8–12
   vs MCTS depth 3–5.)

**Rollout blend:** Mixes NN priors with C engine heuristic scores. Distant play mode
uses 0.15 weight for adjacent moves, 0.05 for far moves.

### 2.4 Training

**Optimizer:** Adam, lr=1e-3, weight_decay=1e-4
**Scheduler:** CosineAnnealingWarmRestarts(T₀=50, T_mult=2, η_min=1e-4)
**Batch size:** 1024
**Grad clipping:** norm 1.0
**AMP:** Auto FP16 on CUDA
**Replay buffer:** 400K positions
**Self-play workers:** 5 (ProcessPoolExecutor)
**Games per iteration:** 100
**Training steps per iteration:** 200

**Loss functions:**
- Policy: cross-entropy vs MCTS visit distribution
- Value: MSE (tanh output vs game result)
- Threat: MSE (4 floats vs computed threat labels)

**Data augmentation (7 copies per sample):**
- 3 grid-safe transforms (priority ×0.8): 180° rotation, transpose, 180°+transpose
- 4 axial rotations (priority ×0.7): 60°, 120°, 240°, 300°

**Defensive training boosts:** Blocking moves get 3.0× priority, survival moves get 2.0×.

### 2.5 Curriculum (6 Levels)

| Level | Name | Sims | Opponent | Advance Win Rate | Min Iters |
|---|---|---|---|---|---|
| 1 | Basics | 30 | random | 80% | 5 |
| 2 | Blocking | 50 | heuristic | 60% | 10 |
| 3 | Tactics | 100 | self | 55% | 15 |
| 4 | Forks | 150 | self | 55% | 20 |
| 5 | Colony | 200 | self | 55% | 30 |
| 6 | Endgame | 200 | self | N/A | N/A |

**Plateau detection:** If ELO stalls for 10 iterations (delta < 15), auto-boosts
MCTS sims by 50 (capped at 400).

### 2.6 Endgame Solver

- C engine alpha-beta, max depth 12, time limit 5s
- `solver_or_mcts()` hybrid: tries solver first, falls back to MCTS
- Persistent transposition cache (pickle, 100K entries, evicts oldest 25%)

### 2.7 C Engine

- `engine.c` (~2300 lines, 100KB) — not read due to rate limit
- Board representation: likely sparse hash map or bitboard (uses Zobrist hashing)
- Exposed via ctypes: `board_place()`, `board_undo()`, `board_encode_state_full()`
  (computes all 7 channels including threats in C), `board_count_winning_moves()`,
  alpha-beta search with collect/inject mode for batched NN evaluation
- Auto-compiles on first import — no build system needed

---

## 3. Ambrosia Summary

**Repo:** `hex-tic-tac-toe/ambrosia` — by Mina (@trueharuu)
**Language:** Pure Rust
**Stage:** Early / experimental ("currently very limited" per README)

### 3.1 Approach

- **No NN, no search depth.** Pure 1-ply heuristic evaluator.
- Board: sparse `FxHashMap<Hex, Player>` — identical approach to ours.
- Axial coordinates, 64-bit Zobrist hashing, full undo/redo support.
- Correct 1-then-2-per-turn implementation.
- Win detection: sliding along 3 hex axes from each placed stone.
- Move generation: expand all occupied cells by configurable radius.

### 3.2 Evaluation Function

Linear feature model: `score = Σ (feature_i × weight_i)`. `Genome` struct holds weights
(scaffolding for evolutionary optimization, not yet implemented).

**9 hand-crafted features:**

| Feature | Description |
|---|---|
| LongestRun | Longest consecutive line for current player |
| ThreatScore | `len² × (1 + open_ends)`, skipping dead lines |
| OpenThreats(n) | Runs within n of win length with ≥1 open end |
| DoubleThreats | Cells creating near-win on 2+ axes (fork detection) |
| GapThreats | X_X patterns that would be near-win if gap filled |
| OpponentThreat | ThreatScore for opponent (defensive awareness) |
| LargestCluster | Largest BFS-connected component |
| IsolatedPieces | Own stones with no friendly neighbor within distance 2 |
| CentreProximity | Σ 1/(1 + distance_from_origin) for own stones |

### 3.3 Dependencies

`clap`, `rand`, `rustc-hash` (FxHashMap), `tiny-skia` (2D rendering).
**No ML crates.** No `tch`, `burn`, `candle`, `ort`.

### 3.4 Strength Assessment

Without any search depth, significantly weaker than SealBot or any NN bot.
Good Rust foundation — adding alpha-beta would dramatically improve it.
**Not a candidate for corpus generation or evaluation benchmarking.**
Worth watching if Mina adds search or evolutionary weight optimization.

---

## 4. Comparison Table

| Feature | KrakenBot | Orca | HeXO (ours) |
|---|---|---|---|
| **Language** | Python + Cython + C++ | Python + C (ctypes) | Rust + Python |
| **Board representation** | Python dict (sparse) | C hash map (sparse) | Rust HashMap (sparse) |
| **Board encoding** | 25×25 torus (circular padding) | 19×19 centroid window | K × 19×19 cluster windows |
| **Input channels** | 2 (own, opp) | 7 (own, opp, legal, player, remaining, own threats, opp threats) | 18 (8 own history, 8 opp history, moves_remaining, turn parity) |
| **Threat planes** | None (learned implicitly) | 2 explicit threat planes + threat-policy blend | None (learned implicitly) |
| **Network depth** | 10 res blocks | 12 res blocks | 12 res blocks |
| **Network width** | 128 channels | 128 ch (standard) / 256 ch (hybrid) | 128 channels |
| **Normalization** | GroupNorm (8 groups) | BatchNorm | BatchNorm |
| **SE blocks** | None | Yes (hybrid, ratio=4) | Yes (every block) |
| **Padding mode** | Circular (torus) / zeros (infinite) | Zeros (standard) / circular (hex-native) | N/A (cluster-based) |
| **Hex-aware convolutions** | No (standard 3×3) | HexMaskedConv / HexNativeConv variants | No (standard 3×3) |
| **Policy head** | Bilinear pair attention N×N | Conv→Linear (361 cells) | Per-cluster softmax, unified |
| **Action space** | **Compound** (joint pair logits) | **Sequential** (single-cell policy) | **Sequential** (2 MCTS plies per turn) |
| **Value head** | avg+max pool → FC → **tanh** | flatten → FC → **tanh** | avg+max pool → FC → **BCE sigmoid** |
| **Auxiliary heads** | Moves-left (MSE) + Chain (6ch, masked MSE) | Threat (4 floats, MSE) + threat-policy blend | Opponent reply prediction |
| **Value aggregation** | Single eval (torus = 1 window) | Single eval (1 centroid window) | **Min-pooling** across K clusters |
| **Symmetry augmentation** | D6 × 12 transforms | 7 transforms (3 grid + 4 axial, weighted priority) | D6 × 12 transforms |
| **MCTS sims (self-play)** | 200 / 600 (25% full) | 30–400 (curriculum) | 800 |
| **PUCT C** | 1.0 | 1.5 | Config-driven |
| **FPU** | KataGo dynamic (reduction=0.25) | Standard | Config-driven |
| **Dirichlet** | α=10/n, frac=0.25 | α=0.3, ε=0.25 | Config-driven |
| **Progressive widening** | No (all legal pairs at root) | **Yes** (6→10→16→25) | No |
| **AB hybrid pre-check** | No | **Yes** (depth 4 before MCTS) | No |
| **Quiescence** | No | **Yes** (3+ winning moves = forced win) | No |
| **Replay buffer** | Parquet files on disk | 400K positions (Python) | Rust f16 ring buffer (250K→1M) |
| **Training batch** | 256 | 1024 | Config-driven |
| **Optimizer** | Adam (lr=1e-3, wd=1e-4) | Adam (lr=1e-3, wd=1e-4) | Config-driven |
| **LR schedule** | Cosine annealing | CosineWarmRestarts (T₀=50, T_mult=2) | Config-driven |
| **Grad clipping** | 50.0 | 1.0 | Config-driven |
| **Loss: policy** | Pair CE (N² logits) | CE vs visit distribution | CE vs visit distribution |
| **Loss: value** | MSE | MSE | BCE (sigmoid) |
| **Loss: auxiliary** | Entropy reg (0.01) + moves-left (0.1) + chain (0.1) | Threat MSE | Opponent reply (0.15) |
| **Distillation stage** | Yes (minimax→NN via Parquet) | Optional (SFT from game collections) | Yes (SealBot corpus → pretrain) |
| **Curriculum** | No | **6 levels** (random→heuristic→self) | No |
| **Endgame solver** | No | **Yes** (C AB, depth 12, persistent TT cache) | No |
| **Checkpoint gating** | TODO (not implemented) | ELO eval every 2 iters | SealBot eval gate |
| **C/C++ engine** | C++ minimax (pybind11, pattern eval) | C engine (ctypes, auto-compile, AB + encoding) | Rust MCTS + board + buffer |
| **Estimated total params** | ~1.5M | ~3.9M (standard) / ~5.2M (hybrid) | ~3.5M (12 blocks, SE, 128ch) |

---

## 5. Actionable Findings for HeXO

### 5.1 Clearly Better, Low Risk to Adopt

**A. KataGo-style Dynamic FPU (from KrakenBot)**

KrakenBot uses a dynamic first-play urgency that blends observed mean Q with the NN
parent value, weighted by visited policy mass, with reduction growing with √mass.
This is strictly better than a fixed FPU value — it adapts as the subtree fills in.

- **Effort:** Small config + MCTS code change in Rust.
- **Risk:** None. Well-established in KataGo literature. Can A/B test trivially.
- **Priority:** HIGH. Implement before next sustained training run.

**B. Quiescence Check for Forced Wins (from Orca)**

Orca checks: if 3+ winning moves exist for either side, override NN value to ±1.0
(forced win — opponent can block at most 2 per turn in a 2-stone-per-turn game).
For 2 winning moves, blend a boost.

- **Effort:** Moderate. Requires `Board::count_winning_moves()` in Rust (we already
  have `Board::get_threats()` which is close). Then add a quiescence hook at MCTS
  leaf expansion.
- **Risk:** Low. Pure improvement — prevents the NN from "hallucinating" through
  proven forced wins/losses.
- **Priority:** HIGH. Directly improves tactical accuracy, which is our current weakness.

**C. Entropy Regularization in Policy Loss (from KrakenBot)**

KrakenBot adds `-entropy * 0.01` to the loss, encouraging the policy to maintain
some exploration mass. This prevents premature policy collapse during self-play
training.

- **Effort:** Trivial — one line in the loss function.
- **Risk:** None. Standard regularization technique.
- **Priority:** MEDIUM. Add before sustained self-play run.

**D. Draw Penalty (from KrakenBot)**

KrakenBot uses `-0.1` as the value target for drawn games instead of 0.0. This
teaches the network to actively avoid draws, which is strategically correct in
a game where first-player advantage should be leveraged.

- **Effort:** Trivial config change.
- **Risk:** None.
- **Priority:** MEDIUM. Add to training config.

### 5.2 Requires Architectural Changes, Potentially Worth It

**E. Threat Planes as Input (from Orca)**

Orca feeds 2 explicit threat planes (own + opponent threat maps) as NN input channels.
Plus their threat head's spatial map is blended directly into policy logits (weight 0.3).

- **Tradeoff:** We currently rely on the NN to learn threat patterns from raw stones.
  Explicit threat planes give the network a shortcut — it doesn't have to learn
  threat detection from scratch. However, the threat computation must be consistent
  with the NN's internal representation, or it may confuse training.
- **Effort:** Moderate. We have `Board::get_threats()` in Rust. Need to add 2 planes
  to the tensor encoding (18→20 channels), update the network stem, and retrain.
- **Risk:** Medium. Changes input encoding which requires retraining from scratch or
  checkpoint migration. Orca's checkpoint migration approach (zero-pad new channels)
  could help.
- **Priority:** MEDIUM. Worth ablating after Phase 4.5. The threat-policy blend is
  particularly interesting as it creates a learned-but-guided attention mechanism.

**F. Progressive Widening in MCTS (from Orca)**

Orca's 6→10→16→25 progressive widening forces deeper trees. This is especially
valuable for connect-6 where seeing several full 2-stone turns ahead matters more
than exploring many first moves.

- **Tradeoff:** Deeper trees find tactical sequences better, but may miss surprising
  moves that a wider search would discover. The sweet spot depends on how good the
  policy network is at pruning.
- **Effort:** Moderate. Changes the MCTS node expansion logic in Rust.
- **Risk:** Medium. Interacts with our sequential action space — our tree is already
  somewhat deeper than KrakenBot's because we expand 2 MCTS plies per turn.
- **Priority:** MEDIUM. Worth ablating. Our 800 sims/move gives us more budget
  than Orca's 200–400, so the optimal widening schedule would differ.

**G. Alpha-Beta Hybrid Pre-Check (from Orca)**

Before MCTS, run a quick depth-4 alpha-beta. If a proven win/loss is found, skip
MCTS entirely.

- **Tradeoff:** Saves MCTS computation in tactically decided positions. Requires
  a competent alpha-beta implementation, which we don't have (we use SealBot as an
  external minimax, not an internal one).
- **Effort:** HIGH. Would need to implement alpha-beta in Rust or wrap SealBot's
  C++ engine at the MCTS level. The latency of calling out to SealBot per position
  may negate the savings.
- **Risk:** High complexity for uncertain gain. SealBot's minimax is strong but
  was designed as a standalone bot, not as an MCTS subroutine.
- **Priority:** LOW for now. Revisit if tactical blindness becomes a bottleneck.

### 5.3 Interesting but Needs Ablation Data

**H. Compound vs Sequential Action Space**

KrakenBot's pair attention matrix (N×N joint logits) is the most novel architectural
choice we've seen. It directly models the correlation between the two stones of a turn.

- **Our approach (sequential):** 2 MCTS plies per turn, each selecting one stone.
  Q-flip at turn boundaries, Dirichlet skipped at intermediate plies. This was
  resolved as Q6 in our open questions.
- **KrakenBot's approach (compound):** Single N×N logit matrix, bilinear attention
  between cell embeddings, symmetrized. Two-level MCTS decomposition at root,
  flat pair selection at non-root nodes.
- **Tradeoffs:**
  - Compound sees correlations (threat + block in one turn) but has O(N²) output
    which is expensive and sparse.
  - Sequential is simpler, cheaper, and lets MCTS naturally explore different
    first-stone choices, but may miss synergistic pairs.
  - KrakenBot's marginalization trick (logsumexp for single-stone logits) is elegant
    but adds computation at every MCTS node.
- **Verdict:** Both approaches are viable. We chose sequential and it's working.
  Switching would be a major architectural change. **Only reconsider if we see
  evidence of systematic pair-synergy blindness in eval games.**
- **Priority:** LOW. Monitor KrakenBot's competitive results as a data point.

**I. Moves-Left Head (from KrakenBot)**

Predicts how many moves remain in the game. Used as an auxiliary training signal
(MSE loss, weight 0.1).

- **Benefit:** May help the value head calibrate — a position with 50 moves left
  is more uncertain than one with 5 moves left. KataGo found this useful.
- **Effort:** Small. Add one more head to the network.
- **Risk:** Low, but unclear benefit for our architecture (we already have opponent
  reply prediction as auxiliary).
- **Priority:** LOW. Ablation candidate for Phase 5.

**J. Chain Head (from KrakenBot)**

Per-cell per-direction unblocked chain length (6 channels). Trained with masked MSE.

- **Benefit:** Forces the backbone to explicitly represent threat structures,
  which may improve both policy and value head accuracy.
- **Effort:** Moderate. Need to compute chain targets (we have the threat detection
  infrastructure). Add 6-channel head.
- **Risk:** Low.
- **Priority:** LOW. Ablation candidate — compare against our opponent reply prediction.

### 5.4 Things That Conflict with Our Approach

**K. Torus vs Multi-Window Encoding**

This is the fundamental architectural divergence between KrakenBot and HeXO.

| | Torus (KrakenBot) | Multi-Window (HeXO) |
|---|---|---|
| **Pros** | Single evaluation per position; circular padding eliminates edge artifacts; full D6 symmetry trivially preserved; simpler implementation; no value aggregation needed | Handles truly distant colonies; no phantom connectivity across seam; scales to any board size; window-centered attention | |
| **Cons** | Phantom 6-in-a-row across torus seam (unconfirmed risk); fixed maximum board span (25×25); all distant stones share same evaluation context | Multiple NN evals per position (K windows); value aggregation strategy (Q2 still open); policy unification complexity; higher inference cost |
| **When better** | Games stay compact (<25×25 span); self-play training (torus is fine as proxy); games with 1–2 clusters close together | Games with genuinely distant colonies (>25 cells apart); competitive play against colony-style opponents; late-game positions with scattered stones |

**Our assessment:** The torus is a pragmatic shortcut that works well for self-play
training (where the bot controls both sides and naturally keeps play compact). For
competitive play on the truly infinite grid, multi-window is more principled. However,
KrakenBot's `set_padding_mode()` trick shows you can train on the torus and eval on
the infinite grid by switching to zeros padding — this is worth investigating.

**Action:** Q10 in our open questions already tracks this. imaseal's KrakenBot results
will be the first real data point on whether torus training transfers to infinite-grid
play. **Do not switch until we see competitive results.**

**L. GroupNorm vs BatchNorm (from KrakenBot)**

KrakenBot uses GroupNorm (8 groups) instead of BatchNorm. GroupNorm is batch-size-
independent, which is valuable when batch sizes vary (e.g., single-sample inference
during MCTS vs batch-256 during training). BatchNorm can behave differently in
train vs eval mode.

- **Tradeoff:** GroupNorm adds ~1-2% computational overhead but eliminates train/eval
  mode discrepancies. With our Rust-based inference server and fixed batch sizes, this
  is less critical for us.
- **Priority:** LOW. Not worth switching mid-training. Consider for Phase 5 architecture
  revision.

**M. 2-Channel vs 18-Channel Input**

KrakenBot uses only 2 input channels (current + opponent stones). We use 18 (8 history
planes per player + 2 metadata planes). KrakenBot compensates by having the chain head
force the backbone to learn temporal patterns.

- **Tradeoff:** Fewer input channels → faster inference, smaller stem. But temporal
  history helps the network understand move sequences and infer opponent intent.
  KrakenBot's approach relies entirely on the backbone to infer history from the
  current position, which is a harder learning task.
- **Verdict:** Our 18-channel approach is more principled and gives the network
  more information. Keep it. The real question is whether the additional information
  actually helps training convergence — this would be an ablation.
- **Priority:** No action needed. Our approach is defensible.

### 5.5 Summary: What KrakenBot Does for the 8-Cell Placement Rule

The game rules state that player 1 places 1 stone on the first turn, then both
players place 2 stones per turn. KrakenBot handles this as follows:

- `moves_left_in_turn` starts at 1 for player A's first turn, then 2 thereafter.
- The pair policy head naturally handles this: for the first turn, only the diagonal
  of the N×N matrix matters (single stone), and the MCTS only runs level 1 of the
  two-level tree.
- Temperature is very high (T=2.5) for the first 5 turns to ensure diverse openings.

This is essentially the same approach as ours (tracking `moves_remaining` and
adjusting MCTS behavior), but implemented through the compound action space rather
than sequential plies.

---

## Appendix: Files Not Read (Rate-Limited)

### KrakenBot
- `cpp/engine.h` (60KB) — core minimax engine
- `minimax_bot.py` (35KB) — Python minimax wrapper
- `diagnostic_*.py` — debugging tools
- `mcts/_mcts_cy.pyx`, `mcts/_puct_cy.pyx` — Cython MCTS acceleration
- `mcts/pattern_table.py` — pattern evaluation data
- `play.py` — interactive play interface
- `parallel_selfplay.py`, `train_loop.py` — partially read via imports

### Orca
- `engine.c` (100KB) — C game engine (board, win detection, AB, threats, encoding)
- `hexbot.py` (42KB) — 30+ API functions
- `hexgame.py` (18KB) — Python game wrapper
- `bot.py` (9KB) — ReplayBuffer, self-play functions
- `orca/multiscale_net.py` — multi-scale two-tower architecture
- `orca/data.py` (45KB) — data loading pipeline
- `orca/threats.py`, `orca/openings.py`, `orca/ensemble.py`,
  `orca/distributed.py`, `orca/zoo.py`, `orca/leaderboard.py`

Re-running with a higher-rate GitHub token would fill these gaps.
