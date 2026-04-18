# KrakenBot Deep-Dive Analysis

**Date:** 2026-04-16  
**Source:** `github.com/Ramora0/KrakenBot` (cloned `/tmp/krakenbot`)  
**Purpose:** Reverse-engineer every design decision that made KrakenBot dominant; identify what HexaZero should adopt, adapt, or avoid.

> Code is ground truth throughout. Where docs and code disagree, code wins.

---

## Table of Contents

1. [Input Representation](#1-input-representation)
2. [Network Architecture](#2-network-architecture)
3. [MCTS / Search](#3-mcts--search)
4. [Training Pipeline](#4-training-pipeline)
5. [Bootstrap Strategy](#5-bootstrap-strategy)
6. [Game Logic & Rules Encoding](#6-game-logic--rules-encoding)
7. [Inference & Deployment](#7-inference--deployment)
8. [Code Quality & Engineering](#8-code-quality--engineering)
9. [Final Synthesis](#9-final-synthesis)

---

## 1. Input Representation

### 1.1 KrakenBot — Exact Tensor Spec

**File:** `model/resnet.py`, lines 167–212 (`board_to_planes_torus`)  
**Shape:** `[B, 2, 25, 25]`

| Channel | Semantic | Encoding |
|---------|----------|----------|
| 0 | Current player's stones | 1.0 where occupied, else 0.0 |
| 1 | Opponent's stones | 1.0 where occupied, else 0.0 |

**No history planes. No turn-encoding planes. No feature planes.** The network sees only the bare two-stone-occupancy snapshot at the current ply.

Board size: `BOARD_SIZE = 25` (line 20). Fixed toroidal grid for self-play. Dynamic bounding-box crop with margin=6 for evaluation on the infinite grid (`board_to_planes`, lines 189–200).

**Coordinate mapping:** `(q, r)` → flat index `q * 25 + r` (`mcts/tree.py`, lines 234–239). Simple row-major layout, no special hex encoding.

**Circular padding:** All convolutions use `padding_mode='circular'` during self-play (lines 82–85, 27–33), meaning the 25×25 grid wraps toroidally. Mode switched to `'zeros'` for evaluation on infinite grid via `set_padding_mode()` (lines 160–164).

### 1.2 KrakenBot — Auxiliary Chain Output (Not Input)

The network predicts chain lengths as an *output* (training target), not as an input feature:

**File:** `model/resnet.py`, lines 102–106 (`chain` head output)  
**Shape:** `[B, 6, H, W]`

| Channels | Semantic |
|----------|----------|
| 0–2 | Current player's unblocked chain length along hex directions 0, 1, 2 |
| 3–5 | Opponent's unblocked chain lengths along directions 0, 1, 2 |

This is a supervised auxiliary signal extracted from the board state and fed back as a loss term. It is **never concatenated into the input tensor**.

### 1.3 KrakenBot — 2-Moves-Per-Turn Encoding

Turn structure is **not encoded in input planes at all**. The tree handles it structurally: root uses a two-level decomposition (stone-1 selection, then stone-2 selection), and `game.moves_left_in_turn` tracks whether we are on the first or second stone. The NN always sees a 2-channel snapshot regardless of turn phase.

### 1.4 KrakenBot — Symmetry Augmentation

**File:** `model/symmetry.py`, lines 18–54

Full D6 group: 12 transforms (6 rotations × 2 reflections). Precomputed as:
- `PERMS[12, 625]` — forward permutation (old flat index → new flat index)
- `INV_PERMS[12, 625]` — inverse permutation
- `DIR_PERMS[12][3]` — direction index permutation (for chain targets)

Applied at training sample time (`training/selfplay/train_loop.py`, lines 200–234): a random D6 symmetry is applied to both input planes and visit-count targets. Chain targets use `apply_symmetry_chain` (lines 106–115 of `symmetry.py`) which permutes the 6 channels according to `DIR_PERMS`.

### 1.5 HexaZero — Exact Tensor Spec

**File:** `hexo_rl/env/game_state.py`, lines 186–244 (`to_tensor`)  
**Shape:** `[K, 18, 19, 19]` where K = number of active clusters (typically 1–5)

| Plane(s) | Semantic | Encoding |
|----------|----------|----------|
| 0 | Current player stones (now) | Binary float16 |
| 1–7 | Current player stones (t-1 … t-7) | Binary float16, zeros for missing history |
| 8 | Opponent stones (now) | Binary float16 |
| 9–15 | Opponent stones (t-1 … t-7) | Binary float16, zeros for missing history |
| 16 | `moves_remaining == 2` broadcast | 1.0 if 2 moves left, 0.0 if 1 |
| 17 | Ply parity broadcast | `ply % 2` cast to float16 |
| 18–23 | Q13 chain lengths (6 channels) | moved to replay-buffer aux sub-buffer post-§97; not in NN input |

History: `HISTORY_LEN = 8` (line 10), stored as deque of prior `GameState` objects. Missing slots filled with zeros. Chain planes computed in Python (`_compute_chain_planes`, lines 98–137), capped at 6, normalized to [0, 1].

**Board size:** 19×19 per cluster window. Flat action space: 362 (19×19 + 1 pass).

**Post-§97:** chain-length planes moved to replay-buffer aux sub-buffer. NN input is now 18 channels (history/scalar only). Chain head trained from sub-buffer target, not input slice.

### 1.6 Input Representation — Side-by-Side

| Aspect | KrakenBot | HexaZero |
|--------|-----------|----------|
| Shape | `(B, 2, 25, 25)` | `(B*K, 18, 19, 19)` |
| Stone channels | 2 (current + opponent) | 2 (current + opponent) |
| History planes | **0** | 14 (7 ply × 2 players) |
| Turn/phase planes | **0** | 2 (moves_remaining, ply parity) |
| Chain/threat planes (input) | **0** | **0** (moved to aux sub-buffer post-§97) |
| Total input channels | **2** | **18** |
| Board geometry | Toroidal 25×25 (self-play) | 19×19 windowed clusters |
| Coord mapping | `q*25 + r` flat | Per-cluster offset via Rust |
| Symmetry augmentation | D6 (12-fold), at training time | D6 (12-fold), at buffer sample time |
| Chain as input | No | **Yes** (planes 18–23) |
| Chain as output target | Yes | Yes |

**Key gap:** KrakenBot achieves strong play with 2 input channels, no history, and no explicit feature planes. HexaZero uses 24 channels including history and chain features as inputs. This raises the question of whether the added input complexity is beneficial or just noise. The empirical answer is unknown — KrakenBot's 2-channel approach may work because the network instead *learns* all implicit features through its chain auxiliary head training signal.

---

## 2. Network Architecture

### 2.1 KrakenBot — Full Model Definition

**File:** `model/resnet.py`, lines 23–149 (`ResBlock` + `PairPolicyHead` + `HexResNet`)

**Stem:**
```
Conv2d(2 → 128, 3×3, circular padding, bias=False) → GroupNorm(8, 128) → ReLU
```

**Trunk: 10 × ResBlock**
```
ResBlock(128):
  Conv2d(128 → 128, 3×3, circular padding, bias=False) → GN(8,128) → ReLU
  Conv2d(128 → 128, 3×3, circular padding, bias=False) → GN(8,128)
  + skip → ReLU
```

No SE blocks. GroupNorm throughout (not BatchNorm). Circular padding in all convolutions preserves toroidal structure.

**Value head** (lines 92–96, 137–138):
```
Conv2d(128 → 32, 1×1) → GN(8,32) → ReLU
Masked global avg-pool + max-pool → concat [avg, max] → dim=64
Linear(64 → 256) → ReLU → Linear(256 → 1) → Tanh
Output: scalar in [-1, 1]
```

**Moves-left head** (lines 98–100, 140–141) — shares pooled features with value:
```
Linear(64 → 256) → ReLU → Linear(256 → 1) → ReLU
Output: non-negative scalar (remaining moves in game)
```

**Chain auxiliary head** (lines 102–106, 146–147):
```
Conv2d(128 → 32, 1×1) → ReLU → Conv2d(32 → 6, 1×1)
Output: [B, 6, H, W] — per-cell, per-direction chain length predictions
```

**Policy head: `PairPolicyHead`** (lines 39–73) — *the biggest architectural difference*:
```
q_proj = Conv2d(128 → 64, 1×1)   # [B, 64, H, W]
k_proj = Conv2d(128 → 64, 1×1)   # [B, 64, H, W]

# Flatten spatial dims:
Q = q_proj(trunk).flatten(2)     # [B, 64, N]   N = H*W
K = k_proj(trunk).flatten(2)     # [B, 64, N]

# Bilinear dot-product:
A = bmm(Qᵀ, K) * scale           # [B, N, N]   scale = 64^(-0.5)

# Symmetrize (order of stones must not matter):
A = (A + Aᵀ) / 2

# Mask self-pairs (can't place both stones on same cell):
A.masked_fill(eye(N), -inf)

# Mask padding cells if board < 25×25:
A.masked_fill(invalid_pairs, -inf)

Output: raw logits [B, N, N]
```

The policy head outputs a **joint distribution over (stone1, stone2) pairs** directly. The symmetrization `(A + Aᵀ)/2` enforces that swapping the two stones gives the same probability — correct since both are placed this turn and order within a turn doesn't matter.

**Parameter count estimate:** ~3.0M total.

### 2.2 HexaZero — Full Model Definition

**File:** `hexo_rl/model/network.py`, lines 93–227 (`HexTacToeNet`)

**Stem:**
```
Conv2d(24 → 128, 3×3, padding=1, bias=False) → BN(128) → ReLU
```

**Trunk: 12 × ResidualBlock (with SE)**
```
Conv2d(128 → 128, 3×3, padding=1, bias=False) → BN → ReLU
Conv2d(128 → 128, 3×3, padding=1, bias=False) → BN
SEBlock(128, reduction=4):
  GlobalAvgPool → Linear(128→32) → ReLU → Linear(32→128) → Sigmoid → scale
+ skip → ReLU
```

**Policy head:**
```
Conv2d(128 → 2, 1×1) → BN → ReLU → Flatten(722) → Linear(722 → 362) → log_softmax
Output: [B, 362] log-probabilities (flat single-stone policy)
```

**Value head** (dual pooling):
```
GlobalAvgPool(128) concat GlobalMaxPool(128) → [B, 256]
→ Linear(256 → 256) → ReLU → Linear(256 → 1) → Tanh
Also stores pre-tanh logit for BCE loss
```

**Auxiliary heads** (training only, active when flags set):
- Opponent reply: same structure as policy head → `[B, 362]`, weight 0.15
- Ownership: `Conv2d(128→1, 1×1) → Tanh`, weight 0.1
- Threat: `Conv2d(128→1, 1×1)` raw logits, weight 0.1, pos_weight=59
- Q13 chain: `Conv2d(128→6, 1×1)`, weight 1.0
- Uncertainty: `GlobalAvgPool → Linear(128→1) → Softplus`, disabled (weight 0.0)

**Parameter count estimate:** ~4.2M total.

### 2.3 Network Architecture — Side-by-Side

| Aspect | KrakenBot | HexaZero |
|--------|-----------|----------|
| Res blocks | 10 | 12 |
| Filters | 128 | 128 |
| Normalization | **GroupNorm** (8 groups) | BatchNorm |
| SE blocks | **None** | Every block (reduction=4) |
| Padding | **Circular** (torus) | Zero (standard) |
| Policy head type | **Bilinear N×N pair logits** | Flat 362-dim single-stone |
| Policy output shape | `[B, N, N]` joint pairs | `[B, 362]` per stone |
| Value head | avg+max pool → FC256 → tanh | avg+max pool → FC256 → tanh |
| Moves-left head | **Yes** (shared pooled features) | **No** |
| Chain aux head | Yes (`[B, 6, H, W]`) | Yes (`[B, 6, H, W]`) |
| Threat head | No | Yes |
| Opp-reply head | No | Yes |
| Ownership head | No | Yes |
| torch.compile | No | Disabled (CUDA graph incompatibility) |
| Parameters | ~3.0M | ~4.2M |

**Critical difference — policy head:** KrakenBot models the *joint* distribution over stone pairs. HexaZero models a *sequential* per-stone distribution. The bilinear attention approach means the network explicitly learns which pairs of cells are good to occupy together — the interaction between the two stones within a turn is represented in the model weights, not delegated entirely to the search tree.

**GroupNorm vs BatchNorm:** GroupNorm is batch-size-independent, which matters when self-play batch sizes are small or variable. BatchNorm's running statistics can degrade in the inference model (which runs with batch=1 at leaf eval time). This may be a subtle source of training/inference distribution mismatch in HexaZero.

---

## 3. MCTS / Search

### 3.1 KrakenBot — PUCT and Selection

**File:** `mcts/tree.py`, lines 70–72 and `mcts/_puct_cy.pyx`, lines 6–7

```
PUCT_C = 1.0
FPU_REDUCTION = 0.25
```

Selection formula (Python fallback, lines 316–374):
```
Q(s,a) = values[a] / visits[a]            (or FPU for unvisited)
U(s,a) = PUCT_C * sqrt(N(s)) * P(a) / (1 + N(s,a))
score(a) = Q + U
```

**Dynamic FPU (KataGo-style):** When unvisited children exist, compute a policy-mass-weighted blend of sibling Q-values, then subtract `FPU_REDUCTION * sqrt(policy_mass)`. This gives a more principled lower bound than a constant Q=0.

**No Gumbel search.** Exhaustive grep for "gumbel", "sigma", "sequential halving", "completed_q" across the entire repo returns zero matches. KrakenBot uses standard AlphaZero PUCT with KataGo-style FPU only.

**Dirichlet noise** (`mcts/tree.py`, lines 388–409):
- Alpha: adaptive — `10.0 / n_candidates` (not a fixed constant)
- Epsilon: 0.25 (standard)
- Applied at root level-1 AND level-2 after expansion
- Not applied at non-root nodes

**Virtual loss:** Not implemented (no `virtual_loss` field in tree nodes).

**Tree reuse** (`mcts/tree.py`, lines 711–800, `graft_reused_subtree`): After each turn, the child `PosNode` that was actually played is saved. On the next turn, its statistics are grafted into the new root's two-level structure. This means accumulated simulations from the previous turn carry forward, effectively giving the search a head start.

**Transposition table:** Not implemented. Each tree is purely path-local.

### 3.2 KrakenBot — 2-Move-Per-Turn in the Tree

Root uses a **two-level decomposition** (lines 157–177, 806–915):

```
Level 1 (stone_1): move_node   — PUCT over candidate first stones
Level 2 (stone_2): move_node.level2[s1_idx]  — PUCT over candidate second stones
Child PosNode: created after pair (s1, s2) is committed
```

At level-2, a fresh `MCTSNode` is lazily created for each `s1` once it is selected. Non-root positions use flat pair encoding: `action = s1_idx * N_CELLS + s2_idx`, with only the top `NON_ROOT_TOP_K = 50` pairs expanded (line 74).

**Backprop sign:** Pair-depth tracked on each path entry; sign flips at pair (turn) boundaries (lines 998–1033). Within a turn (level-1 → level-2 → PosNode), the sign logic counts which "pair depth" each node belongs to.

**Single-move turns:** First move of game (Player A places 1 stone): `select_single_move` called instead of `select_move_pair` (lines 1181–1185). Tree handles this correctly without special-casing the network input.

### 3.3 HexaZero — MCTS

**Files:** `engine/src/mcts/` (Rust), `engine/src/game_runner/gumbel_search.rs`

**PUCT constants** (`configs/selfplay.yaml`):
```yaml
c_puct: 1.5
fpu_reduction: 0.25
```

Same KataGo-style dynamic FPU as KrakenBot. cpuct is 50% higher (1.5 vs 1.0).

**Virtual loss:** Yes, `VIRTUAL_LOSS_PENALTY = 1.0`, applied during traversal and reversed during backup.

**2-move handling:** `moves_remaining` field on each node. During backup, if `parent.moves_remaining == 1`, the value is negated (negamax sign flip at turn boundaries). Less explicit than KrakenBot's two-level root decomposition, but functionally equivalent for non-root nodes.

**Gumbel MCTS** (`engine/src/game_runner/gumbel_search.rs`):
```
GumbelSearchState:
  gumbel_values[i]   = Gumbel(0,1) noise per root child
  log_priors[i]      = log P(a) from network
  candidates         = active candidate set (starts at m, halved each phase)
  c_visit, c_scale   = sigma scaling constants (50.0, 1.0)

Gumbel score: g(a) + log P(a) + σ(Q(a))
  where σ(Q) = (c_visit + max_N) × c_scale × Q_hat

Sequential Halving:
  1. Expand root (1 inference call)
  2. Apply Dirichlet noise
  3. Select top-m candidates by Gumbel score
  4. For each of num_phases = ceil(log2(m)) phases:
     - sims_per = remaining_budget / (remaining_phases × |candidates|)
     - Force each candidate to receive sims_per simulations
     - Halve candidates (keep top half by updated Gumbel score)
  5. Select winner
```

**`completed_q_values` / `gumbel_targets`** — what they contain:

From `engine/src/mcts/mod.rs`, `get_improved_policy()` (lines 174–304):
```
π_improved = softmax(log P(a) + σ(Q(a)))
  Q(a) = MCTS mean value estimate for action a
  σ(Q) = (c_visit + max_N) × c_scale × Q(a)
  For unvisited actions: Q = v_mix (network value blended with visited siblings)
```

**Does `gumbel_targets` include fast games?** Yes — if `completed_q_values: true`, fast games (64 sims at 25% prob) also receive improved policy targets. If false, fast games get zero policy (value-only targets). Current default: `completed_q_values: false`.

**Tree reuse:** Not implemented in HexaZero.

### 3.4 MCTS — Side-by-Side

| Aspect | KrakenBot | HexaZero |
|--------|-----------|----------|
| Base algorithm | Standard PUCT | PUCT + Gumbel Sequential Halving (optional) |
| cpuct | 1.0 | 1.5 |
| FPU reduction | 0.25 (KataGo dynamic) | 0.25 (KataGo dynamic) |
| Dirichlet alpha | Adaptive: 10/n | Fixed 0.3 |
| Dirichlet epsilon | 0.25 | 0.25 |
| Virtual loss | **No** | Yes (penalty=1.0) |
| Tree reuse | **Yes** (graft subtree) | **No** |
| Sims / move | 100 quick, 600 full (25% full) | 400 standard |
| Full sim prob | 25% (`full_search_prob`) | — |
| Policy targets | Visit-count softmax | Completed Q (when enabled) |
| 2-move tree | Two-level root decomposition | moves_remaining + negamax flip |
| Transposition table | No | No |
| Quiescence override | No | Yes (≥3 winning moves → forced value) |
| Temperature schedule | 2.5 (turns 0–5), 1.0 (6–19), 0.3 (≥20) | Cosine 1.0 → 0.05 |

**Playout cap randomization:** KrakenBot uses 75%/25% quick/full search split. This is functionally similar to HexaZero's approach of always running 400 sims, but KrakenBot's full-search positions (600 sims) get much higher-quality policy targets and are flagged — policy loss is applied *only* on full-search turns (`train_loop.py` line 602). This selective policy training is a key detail.

---

## 4. Training Pipeline

### 4.1 KrakenBot — Optimizer and Schedule

**File:** `training/selfplay/train_loop.py`, line 1462

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=1e-4)
```

LR schedule: single 5× drop at 2/3 of total training rounds (lines 1701–1706):
```python
lr_drop_round = (2 * total_rounds) // 3
cur_lr = args.lr / 5.0 if round_num >= lr_drop_round else args.lr
```

Gradient clipping: `clip_grad_norm_(model, 5.0)` (line 1631).

### 4.2 KrakenBot — Loss Function

**File:** `training/selfplay/train_loop.py`, lines 581–627

| Term | Formula | Weight | Notes |
|------|---------|--------|-------|
| Value | `MSE(v_pred[~draw], v_target[~draw])` | 1.0 | Drawn games fully masked |
| Policy | `-sum(π_visit · log_softmax(pair_logits))` | 1.0 | **Full-search turns only** |
| Moves-left | `MSE(ml_pred/150, ml_target/150)` | 0.1 | Drawn games masked |
| Chain | `MSE(chain_pred, chain_target)` masked | 0.1 | Per-cell, per-direction |

**Total:** `vloss + ploss + 0.1 * ml_loss + 0.1 * chain_loss`

Critical detail: policy loss **only applied on full-search turns** (line 602). Quick-search (100 sim) turns contribute to value and chain losses but not policy. This is correct — with only 100 sims, the visit distribution doesn't reliably reflect the true policy improvement.

Value loss uses **MSE**, not BCE. Draws are excluded from value loss entirely.

### 4.3 KrakenBot — Replay Buffer

**File:** `training/selfplay/self_play.py`; `training/selfplay/train_loop.py`

Storage: pandas DataFrame saved as parquet, with per-example dicts containing sparse visit counts (JSON-encoded). Pre-compiled `.pt` cache to avoid repeated JSON parsing.

Window: last 12 rounds, exponential decay with oldest weight = 0.5. Decay exponent ≈ `0.5^(1/11) ≈ 0.94` per round.

Outcome balancing: A-wins and B-wins reweighted to equal total mass (lines 171–195 of `train_loop.py`). This corrects for any first-player advantage bias in the training distribution.

Cold start: 1024 games to populate buffer before steady state of 256/round.

### 4.4 KrakenBot — Graduation Gate

**File:** `training/selfplay/train_loop.py`, lines 1746–1790

Every 10 rounds, evaluate current model vs. anchor (256 games, 50/50 sides):
- If score ≥ 0.76 → **graduate**: anchor ← current model, save `anchor.pt`, bank Elo
- Otherwise: continue training against same anchor

Banked Elo: `banked_elo = compute_elo(banked_elo, score)` accumulates strength across generations.

This gating mechanism prevents two failure modes: (1) the model drifting into a bad local minimum and being used for self-play data generation, (2) chaotic oscillation where a weaker model replaces a stronger one.

### 4.5 HexaZero — Training Pipeline

**Optimizer:** AdamW, lr=0.002, cosine annealing to 2e-4 over 200K steps (`configs/training.yaml`).

**Loss terms** (`hexo_rl/training/losses.py`):

| Term | Formula | Weight |
|------|---------|--------|
| Policy | CE or KL depending on `completed_q_values` | 1.0 |
| Value | `BCE(v_logit, (z+1)/2)` | 1.0 |
| Opp reply | CE | 0.15 |
| Ownership | MSE | 0.1 |
| Threat | BCE (pos_weight=59) | 0.1 |
| Q13 chain | Huber (smooth_L1) | 1.0 |

**Replay buffer:** Rust ring buffer, f16-as-u16 storage, game-length weighted sampling, 12-fold augmentation at sample time.

**Bootstrap mixing:** starts 80% corpus / 20% self-play, decays to 10% corpus floor at step ~20K.

**No graduation gate.** Continuous training and checkpointing; any self-play games from any recent checkpoint contribute to training. Risk: a temporarily weaker checkpoint could poison the buffer.

### 4.6 Training Pipeline — Side-by-Side

| Aspect | KrakenBot | HexaZero |
|--------|-----------|----------|
| Optimizer | **SGD (momentum=0.9)** | AdamW |
| Initial LR | 0.005 | 0.002 |
| LR schedule | Single 5× drop at 2/3 training | Cosine anneal (200K steps) |
| Grad clip | 5.0 | Not confirmed in code |
| Value loss | **MSE (draws masked)** | BCE on logit |
| Policy loss | CE, full-search turns only | CE or KL |
| Policy selective | **Yes** (full-search flag) | **No** |
| Moves-left loss | 0.1 × MSE | **Not present** |
| Chain loss | 0.1 × MSE | 1.0 × Huber |
| Outcome balancing | **Yes** (A/B win reweight) | No |
| Graduation gate | **Yes** (76% threshold) | **No** |
| Anchor model | **Maintained explicitly** | Not present |

---

## 5. Bootstrap Strategy

### 5.1 KrakenBot — Distillation Pipeline

**Files:** `training/distill/generate_distill.py`, `training/distill/train_resnet.py`

**Step 1 — Generate bot games** (`generate_distill.py`):
- Plays 100K games using the C++ minimax bot on both sides
- Starting positions: randomly sampled from human game archives, filtered to positions with ≤11 stones and odd stone count (i.e., start-of-turn for Player A), deduplicated
- Board representation: infinite grid with 19×19 bounding box cap
- Bot time: 0.04s ± random jitter per move pair
- Stores: board state, move pairs played, eval scores, game outcome, winning cells for game-ending positions (lines 270–306)
- Training targets: when game ends in win, all pairs containing any winning cell get equal probability mass as policy target (lines 39–42 of `train_resnet.py`)

**Step 2 — Supervised pretraining** (`train_resnet.py`):
```
Losses: policy (CE on pair visits/winning-move labels)
      + value (tanh output, MSE vs outcome)
      + moves-left (MSE)
      + chain (MSE)
Epochs: 5 (--epochs 5)
AMP: enabled (--amp)
Output: training/resnet_results/checkpoint.pt
```

**Step 3 — Self-play with SFT annealing** (`train_loop.py`, lines 1409–1416, 1693–1699):
```python
# At training round r:
sft_weight = max(0.0, 1.0 - r / sft_anneal_rounds)  # linear decay to 0
# sft_anneal_rounds default = 10
```

Self-play begins immediately from round 0. The distill corpus is sampled at weight `sft_weight * 0.3` (sft_weight=0.3 is the `--sft-weight` flag default) alongside self-play data. After 10 rounds, SFT weight hits zero and training is pure self-play.

This is a one-way, short transition. There is no iterative alternation between bot-play and self-play.

### 5.2 HexaZero — Bootstrap Strategy

**Files:** `hexo_rl/bootstrap/pretrain.py`, `hexo_rl/bootstrap/generate_corpus.py`

Bootstrap source: SealBot (pybind11 C++ minimax engine), accessed via `BotProtocol`.

Starting positions: empty board (no human-seed seeding currently implemented for bot games).

Bootstrap count: not fixed in code; depends on how many SealBot games are generated before training starts.

SL→SP transition: exponential decay from 80% corpus to 10% floor over 20K steps (`configs/training.yaml` `mixing` section). More gradual than KrakenBot's 10-round hard cutoff.

**No graduation gate.** Any checkpoint that happens to be generating self-play games contributes to the training distribution.

### 5.3 Bootstrap — Side-by-Side

| Aspect | KrakenBot | HexaZero |
|--------|-----------|----------|
| Bot source | C++ minimax with learned patterns | SealBot (pybind11 minimax) |
| Starting positions | **Human game seeds (≤11 stones)** | Empty board |
| Bootstrap count | **100K games** | Variable |
| SL pretraining | 5 epochs, full distill corpus | pretrain.py implemented |
| Transition type | **Linear 10-round anneal to zero** | Exponential 20K-step decay to 10% |
| Iterative SP↔bot | No | No |
| Human seeding | **Yes (positions_human_labelled.pkl)** | Not yet implemented for bot games |

**Most likely training progression for KrakenBot:**
1. ~1–2 days: generate 100K minimax games from human-seeded positions
2. ~4 hours: supervised pretrain for 5 epochs (GPU)
3. Weeks: self-play RL with 10-round SFT annealing then pure SP
4. Graduation gating every 10 rounds ensures monotonic strength increase
5. Banked Elo tracks absolute strength across all generations

The key insight: **human starting positions** give the bot diverse, realistic mid-game configurations from the first move. An empty-board bot generates a biased distribution of openings that may not reflect human play patterns. This is the single most impactful bootstrap improvement available to HexaZero.

---

## 6. Game Logic & Rules Encoding

### 6.1 KrakenBot

**Internal representation:**
- Infinite grid (`HexGame`, `game.py` lines 28–122): sparse `dict[(q,r) → Player]`
- Toroidal self-play (`ToroidalHexGame`, lines 130–277): flat array, coordinates wrap mod 25

**Win detection:** 3-axis 6-cell window scan. For each occupied cell, scan in both directions; count consecutive friendly pieces; if ≥6 with no opponent blocker, position is won. Implemented identically in both `HexGame.make_move` and the C++ minimax.

**Legal move generation:** Cells within hex distance ≤ 2 of any existing stone (`_NEIGHBOR_OFFSETS_2`, `minimax_bot.py` lines 118–124). Same as SealBot's `NEIGHBOR_DIST=2`. Does NOT consider all empty cells.

**MCTS candidate gating:** `MAX_CAND_DIST = 8` (cells within distance 8 of any stone, `mcts/tree.py` line 82). Wider than the legal-move generator's distance-2 because the tree must reason about future threats at longer range.

**Coordinate system:** Axial (q, r). Three hex directions: `(1,0), (0,1), (1,-1)`.

**Board size:** Infinite for official games. 25×25 torus for self-play. Evaluation uses dynamic bounding box (min 25×25, margin 6).

### 6.2 HexaZero

**Internal representation:** Rust `HashMap<(q,r), Player>` — sparse, unbounded (`engine/src/board/`).

**Win detection:** Rust implementation using the same 3-axis 6-cell window logic.

**Legal move generation (ZOI):** Within hex distance ≤ 5 of last 16 moves, falling back to full legal set if < 3 candidates (`configs/selfplay.yaml`). Wider than KrakenBot's distance-2, which could increase branching factor unnecessarily.

**Coordinate system:** Axial (q, r), same as KrakenBot.

### 6.3 Game Logic — Side-by-Side

| Aspect | KrakenBot | HexaZero |
|--------|-----------|----------|
| Board repr | Sparse dict (inf) / flat array (torus) | Rust HashMap (sparse) |
| Win detection | 3-axis 6-cell window | Same |
| Legal move radius | Distance ≤ 2 | Distance ≤ 5 of last 16 moves |
| MCTS candidate dist | Distance ≤ 8 | ZOI radius (same config) |
| Board for self-play | Toroidal 25×25 | Infinite with windowing |
| Coordinate system | Axial (q, r) | Axial (q, r) |

**ZOI gap:** KrakenBot's distance-2 legal moves is tighter than HexaZero's distance-5 ZOI. Tighter candidate generation reduces branching factor and focuses search on relevant cells. The tradeoff: distant cells may occasionally be strategically important; the distance-2 constraint may miss them. KrakenBot compensates with the MCTS-level distance-8 gating (which covers speculative moves at greater range).

---

## 7. Inference & Deployment

### 7.1 KrakenBot

**MCTSBot** (`mcts_bot.py`, lines 24–151): pure PyTorch, loaded via `torch.load`. No ONNX export. CUDA/MPS/CPU selection at init. Respects `time_limit` parameter.

**Infinite grid evaluation:** switches Conv2d padding mode from 'circular' to 'zeros' (`set_padding_mode`), uses dynamic bounding box for tensor assembly.

**ONNX/TensorRT:** Noted in `todo.md` line 10 as "TensorRT or smth" — not implemented.

**API:** Not explicitly bot-api-v1 compliant. Bot provides `get_move(game) → tuple[int,int] | list[tuple[int,int]]` with `pair_moves: bool = True`. Compatible with both infinite and toroidal game objects.

**Adaptive crossover evaluation** (`train_loop.py`, lines 937–1046): finds the minimax time at which MCTS scores 50% win rate — tracks ELO-equivalent vs minimax depth. Uses 3 log-spaced time brackets, interpolates crossover, momentum smoothing (0.3).

### 7.2 HexaZero

Inference through Rust `InferenceBatcher` → Python GPU forward. Batched leaf evaluations from Rust worker threads.

No ONNX export. API compliance with bot-api-v1 targeted but not yet verified.

---

## 8. Code Quality & Engineering

### 8.1 KrakenBot

**Languages:** Python 3.10+ (primary), C++ (minimax engine + PUCT Cython), Python type hints throughout.

**Dependencies:** torch≥2.0, numpy≥1.24, pandas≥2.0, pybind11≥2.11, optional wandb, optional pygame.

**Config:** Pure argparse. No YAML, no Hydra. Defaults hardcoded in argument definitions. Command examples in `scratch.txt` (informal notes).

**Logging:** wandb integration (optional, graceful fallback). tqdm progress bars. Custom `CpuProfile` class for MCTS timing.

**Notable engineering patterns:**
- **Pattern canonicalization** (`mcts/pattern_table.py`): reduces 3^6 = 729 window patterns to ~400 canonical forms via reversal + piece-swap symmetry. O(1) lookup.
- **Dual board modes** (`set_padding_mode`): same model weights work on toroidal (training) and infinite (evaluation) boards by switching padding mode at inference time.
- **Parquet + .pt cache**: self-play data stored as parquet, then pre-compiled to `.pt` for fast tensor loading. Avoids JSON re-parsing on every training iteration.
- **Outcome balancing**: A-win / B-win reweighting so neither player's games dominate training signal.
- **Anchor + graduation**: monotonic quality gating prevents regression.
- **Banked Elo**: persistent strength estimate across all generations.

### 8.2 HexaZero

**Languages:** Python + PyTorch (training), Rust + PyO3/maturin (MCTS, board, replay buffer).

**Config:** YAML (model.yaml, training.yaml, selfplay.yaml, etc.) deep-merged in `scripts/train.py`. Explicit variant override files. Rigorous config-key overlap detection.

**Logging:** structlog (JSON to file) + rich (console). Event-driven fan-out to terminal + web dashboard. Game recorder, replay poller, metrics writer.

**Notable engineering patterns:**
- **Gumbel Sequential Halving** in Rust hot path — theoretically superior to standard PUCT for fixed budgets, not present in KrakenBot.
- **128-bit Zobrist hashing** (splitmix128) for collision-free transposition at high sim rates.
- **f16-as-u16 ring buffer** with zero-copy PyO3 transfer.
- **12-fold augmentation** in Rust sample path with precomputed scatter-copy tables.
- **Quiescence value override**: forces ±1.0 when ≥3 winning moves detected — KrakenBot doesn't do this.

---

## 9. Final Synthesis

### 9.1 Top 5 Things KrakenBot Does That We Should Adopt

Ranked by expected impact on playing strength:

---

**#1 — Bilinear Pair Policy Head** *(expected impact: very high)*

KrakenBot's `PairPolicyHead` models the joint distribution over (stone1, stone2) pairs directly via bilinear attention: `A[i,j] = (Q_i·K_j + Q_j·K_i)/2`. HexaZero outputs a flat 362-dim single-stone distribution; the tree implicitly chains two stone placements, but the network has no direct representation of stone-pair interactions.

The bilinear head is the single most important architectural difference. In a 2-moves-per-turn game, the synergy between the two stones in a turn is a first-class strategic concept (double threats, complementary connections). Forcing the network to model this as a joint distribution means it can directly learn "stone A is good together with stone B" rather than relying on the tree to discover this implicitly.

**Files to modify:**
- `hexo_rl/model/network.py` — replace policy head with `PairPolicyHead`
- `engine/src/mcts/mod.rs` — update policy projection from `[B, N, N]` → 2-stone action space
- `engine/src/game_runner/records.rs` — update policy target storage
- `hexo_rl/training/losses.py` — update policy loss to work with N×N logits
- `engine/src/replay_buffer/` — store pair policy targets instead of flat 362-dim

Reference: `KrakenBot/model/resnet.py` lines 39–73.

---

**#2 — Graduation Gate + Anchor Model** *(expected impact: high)*

KrakenBot maintains an explicit `anchor.pt` — the strongest model that passed the 76% threshold. The current model must beat the anchor in 256 games before it replaces it. HexaZero has no such gating: any checkpoint in the ring buffer contributes to self-play, including potentially weaker intermediate checkpoints.

Without gating, the training distribution can be poisoned by temporary regressions. Self-play from a weaker model teaches the network bad habits. Graduation gating provides monotonic quality guarantees.

**Files to modify:**
- `hexo_rl/training/loop.py` — add eval-vs-anchor logic and promotion check
- `scripts/train.py` — add `--graduation-threshold` and `--eval-every` CLI flags
- `hexo_rl/bootstrap/bots/our_model_bot.py` — used for anchor vs. current head-to-head

Reference: `KrakenBot/training/selfplay/train_loop.py` lines 1746–1790.

---

**#3 — Tree Reuse Between Turns** *(expected impact: moderate–high)*

`graft_reused_subtree` saves the subtree that was actually played and grafts it into the new root's level-1/level-2 structure. This carries forward accumulated simulation statistics from the previous turn, effectively giving the search a free head start equivalent to hundreds of simulations — all for zero inference cost.

Given HexaZero's 400 sims/move, grafting even 200 sims from the previous turn would represent a 50% efficiency gain.

**Files to modify:**
- `engine/src/mcts/mod.rs` — add `graft_subtree` logic
- `engine/src/game_runner/worker_loop.rs` — save and pass the played child node between moves

Reference: `KrakenBot/mcts/tree.py` lines 711–800.

---

**#4 — Selective Policy Loss (Full-Search Turns Only)** *(expected impact: moderate)*

KrakenBot applies policy loss only on full-search (600-sim) turns. Quick-search (100-sim) turns contribute to value and chain losses but not policy. This is sound: with 100 sims on a 625-cell board, the visit distribution is too noisy to be a reliable policy improvement signal.

HexaZero currently applies policy loss uniformly across all positions. Adding a full-search flag and masking the policy loss accordingly would improve sample quality at no additional compute cost.

**Files to modify:**
- `engine/src/replay_buffer/push.rs` — store `is_full_search` flag per position
- `hexo_rl/training/losses.py` — mask policy loss by `full_search` flag
- `engine/src/game_runner/worker_loop.rs` — tag positions with `full_search` based on sim count

Reference: `KrakenBot/training/selfplay/train_loop.py` line 602.

---

**#5 — Bootstrap from Human Starting Positions** *(expected impact: moderate)*

KrakenBot seeds bot games from real human game positions (≤11 stones), giving immediate coverage of realistic mid-game configurations. HexaZero generates corpus games from the empty board, producing a biased distribution of openings that may not reflect how actual competitive games develop.

Human-seeded bot games generate more diverse, more realistic training examples. The resulting pretrained model enters self-play with better generalization over typical game states.

**Files to modify:**
- `hexo_rl/bootstrap/generate_corpus.py` — add `--seed-from-human` mode that samples positions from `data/corpus/` human games
- `hexo_rl/bootstrap/human_seeding.py` — filter to positions ≤11 stones, odd stone count, no wins

Reference: `KrakenBot/training/distill/generate_distill.py` lines 133–178.

---

### 9.2 Top 3 Things HexaZero Does Better (Keep These)

**#1 — Gumbel MCTS (Sequential Halving)**

KrakenBot uses standard PUCT. HexaZero has full Gumbel Sequential Halving with completed Q-value policy targets. Per the Danihelka et al. (ICLR 2022) paper, Gumbel provides:
- Unbiased policy improvement even with very few simulations
- Better sample efficiency in the self-play data generation
- Completed Q-values that are strictly better policy targets than visit-count distributions

This is a genuine research advance that KrakenBot lacks. Keep it. The key gap is that `completed_q_values: true` is disabled by default — it should be the default for sustained runs (the known bug in `trainer.py` nested-dict lookup was flagged in architecture review C1; fix that and enable it).

**#2 — SE Blocks on Every Residual Block**

KrakenBot uses plain residual blocks with GroupNorm. HexaZero adds Squeeze-and-Excitation after every block. SE blocks add channel-wise attention at ~2% parameter overhead, enabling the network to adaptively emphasize relevant feature maps. This is particularly valuable for strategic games where different feature channels (threat patterns, connection patterns) have variable importance depending on board position.

Keep SE blocks. The extra ~0.2M parameters are well worth the expressivity gain.

**#3 — Q13 Chain Planes as Input Features**

KrakenBot only uses chain lengths as an *output* target (auxiliary loss). HexaZero feeds them directly into the network as input planes 18–23. This gives the network explicit, pre-computed strategic context — it doesn't have to re-derive chain lengths from raw stone positions on every forward pass.

Whether this helps or hurts is an open question (it might create redundancy or conflate strategic and positional processing), but it's a reasonable hypothesis that explicit features accelerate early learning. Keep unless ablation shows it hurts.

---

### 9.3 Bootstrap Training Progression (Estimated)

Based on KrakenBot's code, the most likely training progression to reach competitive strength:

| Phase | Duration | Activity |
|-------|---------|----------|
| Data generation | 4–8 hours (8-core CPU) | 100K minimax games, human-seeded starting positions |
| SL pretrain | 2–4 hours (GPU) | 5 epochs on distill corpus |
| Early SP + SFT | Rounds 0–10 | Self-play with SFT annealing; SFT weight drops to 0 |
| Pure SP RL | Rounds 11–N | Self-play only; graduation every 10 rounds at 76% threshold |
| Convergence | ~100–500 SP rounds | Estimated from graduation frequency and Elo banking |

Each SP round generates 256 games (~5–15 minutes depending on sims and hardware). A graduation every 10 rounds = ~2 hours per generation. At 50 generations, that's ~100 hours of pure SP. Total wall time from scratch to competitive: likely 1–2 weeks of continuous GPU training.

No evidence of curriculum (easy→hard bots) or iterative bot-play cycles. Pure: SL bootstrap → SP RL.

---

### 9.4 Input Channel Side-by-Side Table

| # | KrakenBot | HexaZero |
|---|-----------|----------|
| 0 | Current player stones (now) | Current player stones (now) |
| 1 | Opponent stones (now) | Current player stones (t-1) |
| 2 | — | Current player stones (t-2) |
| 3 | — | Current player stones (t-3) |
| 4 | — | Current player stones (t-4) |
| 5 | — | Current player stones (t-5) |
| 6 | — | Current player stones (t-6) |
| 7 | — | Current player stones (t-7) |
| 8 | — | Opponent stones (now) |
| 9 | — | Opponent stones (t-1) |
| 10 | — | Opponent stones (t-2) |
| 11 | — | Opponent stones (t-3) |
| 12 | — | Opponent stones (t-4) |
| 13 | — | Opponent stones (t-5) |
| 14 | — | Opponent stones (t-6) |
| 15 | — | Opponent stones (t-7) |
| 16 | — | moves_remaining broadcast |
| 17 | — | ply parity broadcast |
| 18 | — | Q13 chain (axis 0, cur player) |
| 19 | — | Q13 chain (axis 0, opponent) |
| 20 | — | Q13 chain (axis 1, cur player) |
| 21 | — | Q13 chain (axis 1, opponent) |
| 22 | — | Q13 chain (axis 2, cur player) |
| 23 | — | Q13 chain (axis 2, opponent) |
| **Total** | **2** | **24** |

KrakenBot achieves competitive strength with only the current board state. This is a strong signal that temporal history (planes 1–7, 9–15 in HexaZero) may not be necessary for this game. An ablation removing history planes would clarify this — the Q13 chain planes already capture much of the same spatial context implicitly.

---

### 9.5 Concrete Next Steps

Listed in implementation priority order:

**Priority 1 — Fix `completed_q_values` and enable by default**  
File: `hexo_rl/training/trainer.py` (nested-dict bug from architecture review C1)  
Currently `completed_q_values: false` by default, silently using CE instead of KL. This is the lowest-effort, highest-impact fix: proper Gumbel targets provide better policy supervision than visit-count softmax.

**Priority 2 — Add graduation gate**  
File: `hexo_rl/training/loop.py` — add `eval_vs_anchor()` and promotion logic  
File: `configs/training.yaml` — add `graduation_threshold: 0.76`, `eval_every_steps: N`  
This is architectural: changes how checkpoints propagate. Do this before the next sustained run.

**Priority 3 — Selective policy loss (full-search flag)**  
Files: `engine/src/game_runner/worker_loop.rs`, `engine/src/replay_buffer/push.rs`, `hexo_rl/training/losses.py`  
Low implementation cost, direct quality improvement on policy supervision.

**Priority 4 — Bootstrap from human starting positions**  
Files: `hexo_rl/bootstrap/generate_corpus.py`, `hexo_rl/bootstrap/human_seeding.py`  
Modify `human_seeding.py` to export early-game positions for bot game seeding. Filter: ≤11 stones, odd stone count, no win detected, deduplicated.

**Priority 5 — Bilinear pair policy head**  
Files: `hexo_rl/model/network.py`, `engine/src/mcts/mod.rs`, `engine/src/game_runner/records.rs`, `hexo_rl/training/losses.py`, `engine/src/replay_buffer/`  
Highest expected impact but largest engineering scope. Requires changing the fundamental output representation from `(B, 362)` to `(B, N, N)`. Track as a separate sprint.

**Priority 6 — Tree reuse**  
Files: `engine/src/mcts/mod.rs`, `engine/src/game_runner/worker_loop.rs`  
Significant MCTS efficiency gain. Requires careful handling of the two-level root structure in Gumbel mode (Gumbel candidates must be rebuilt on reuse since Gumbel noise is per-tree).

**Priority 7 — Investigate GroupNorm vs BatchNorm**  
File: `hexo_rl/model/network.py` — swap BN → GN in trunk  
BatchNorm with batch_size=1 at leaf eval degrades. GroupNorm is batch-size-independent. Run a quick ablation (1K steps) before committing to the change.

**Priority 8 — Moves-left auxiliary head**  
Files: `hexo_rl/model/network.py`, `hexo_rl/training/losses.py`  
Small addition, moderate benefit. Shares pooled features with value head (no extra trunk cost).

---

*Analysis performed 2026-04-16. KrakenBot commit history reviewed; full codebase read.  
All file:line references are to `/tmp/krakenbot/` (KrakenBot) or `/home/timmy/Work/hexo_rl/` (HexaZero).*
