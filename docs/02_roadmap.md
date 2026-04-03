# Roadmap — Hex Tac Toe AlphaZero

Each phase has clear entry criteria, exit criteria, and a set of deliverables. Do not advance to the next phase until exit criteria are met — skipping validation steps is the most common cause of wasted training runs.

---

## Phase 0 — Foundation (1–2 weeks)

**Goal**: Correct, tested game logic. Nothing else matters until the environment is bug-free.

### Tasks

- [x] Implement `Board` in Rust: axial coordinates, legal move generation, move application
- [x] Implement bitboard win detection in all 3 hex directions
- [x] Implement Zobrist hashing (incremental update per move)
- [x] Expose via PyO3 — verify `Board` usable from Python
- [x] Implement turn structure: 1 move on ply 0, 2 moves per player thereafter
- [x] Implement `GameState` Python dataclass wrapping the Rust board
- [x] Write comprehensive tests:
  - All winning patterns (E, NE, NW directions, at edges, at corners)
  - Legal move generation at various board states
  - Zobrist hash collision test (hash distinct positions, check uniqueness)
  - Turn counter correctness across a full game

### Exit criteria

- All tests pass
- Win detection verified manually on 20+ hand-crafted positions
- A full random game can be played to completion without errors
- `Board.to_tensor()` produces correctly shaped output verified by inspection

---

## Phase 1 — Minimal training loop (1–2 weeks)

**Goal**: End-to-end pipeline that trains, even if slowly and badly.

### Tasks

- [x] Implement `HexTacToeNet` (ResNet-12, 128 filters, dual heads)
- [x] Implement `MCTSTree` in Rust (PUCT, single-threaded, no batching yet)
- [x] Implement single-process self-play (no worker pool yet)
- [x] Implement `ReplayBuffer` (NumPy ring arrays) (later replaced by Rust ReplayBuffer in Phase 3.5)
- [x] Implement training step (FP16, GradScaler, AdamW)
- [x] Implement basic checkpoint save/load
- [x] Implement basic console logging (loss, games played, buffer size)
- [x] Wire everything together: self-play → buffer → train → repeat

### Configuration (fast_debug.yaml)

```yaml
board_size: 9          # smaller board for debugging
res_blocks: 4
filters: 64
n_simulations: 50      # tiny — just for pipeline validation
batch_size: 64
n_workers: 1
```

### Exit criteria

- Training runs for 1 hour without crashing
- Policy loss decreases from its initial value
- Value loss converges toward a reasonable range (< 0.5)
- Checkpoint save/load round-trips correctly

---

## Phase 2 / 3.5 — Performance & scale (COMPLETE)

**Goal**: Hit throughput targets for real training on 19×19.

### Tasks

- [x] Add virtual loss to MCTS for parallel tree access
- [x] Implement batched leaf evaluation (inference server + pending batch queue)
- [x] Implement Rust-native Game-Level Parallelism (Phase 3.5)
- [x] Apply `torch.compile()` to the model
- [x] Implement Dirichlet noise at root
- [x] Implement temperature scheduling
- [x] Add `structlog` structured logging (JSON to file, pretty to console)
- [x] Add `rich` progress bars and live training dashboard
- [x] Implement benchmark harness: MCTS throughput, GPU utilization, positions/hour
- [x] Profile and fix bottlenecks: Enabled TF32, dynamic hardware scaling

### Throughput targets (Phase 3.5 / 4)

| Metric | Target | Status |
|---|---|---|
| MCTS throughput | ≥ 150,000 sim/s | **PASS** (160,882 sim/s) |
| NN inference | ≥ 8,000 pos/s | **PASS** (11,479 pos/s) |
| Worker throughput | ≥ 500,000 pos/hr | **PASS** (1,734,003 pos/hr) |
| GPU utilization | ≥ 80% | **PASS** (95.4%) |
| VRAM usage | ≤ 80% Total VRAM | **PASS** (9%) |
| Batch fill % | ≥ 50% | **PASS** (99.8%) |

> These are Phase 3.5 exit baselines. See CLAUDE.md § Benchmarks for current Phase 4.0 baselines (2026-04-03).

### Exit criteria

- All Phase 3.5 throughput targets met (verified by benchmark harness)
- `python scripts/benchmark.py` runs and produces a full report
- No worker process crashes over a 4-hour run
- Structured logs written correctly to file

---

## Phase 3 — Bootstrap from minimax (1–2 weeks)

**Goal**: Pretrain the network on existing bot games so early self-play starts from a competent prior rather than random.

### Tasks

- [x] Integrate or implement minimax bot (depth 3–5, α-β pruning) in Python or Rust (Implemented as SealBotBot wrapper)
- [x] Heuristic evaluation function for minimax: counts of open 3/4/5-in-a-rows per player (Integrated via SealBot engine)
- [x] Generate supervised corpus: 10,000–50,000 minimax vs minimax games (Implemented via generate_corpus.py with persistent cache)
- [x] Implement `BootstrapTrainer`:
  - Policy head: behavior cloning loss (cross-entropy with minimax move distribution)
  - Value head: game outcome regression
  - 5–10 epochs over the corpus
- [x] Validate: pretrained network beats random policy significantly
- [x] Transition: warm-start self-play from pretrained weights
- [x] Track Elo from bootstrap baseline so improvement is measurable

### Exit criteria

- Pretrained model win rate vs random opponent ≥ 95%
- Pretrained model shows recognizable tactical play (verified by manual game review)
- Self-play Elo at iteration 100 is measurably higher than without pretraining (run both, compare)

---

## Phase 3.5 — Multi-Window Cluster Refactor [COMPLETED]

**Goal**: Solve the "Attention Hijacking" (Colony Meta) exploit where the network becomes blind to distant lethal threats.

### Tasks

- [x] **Rust Core Update**: Implement dynamic stone clustering (distance ≤ 8) and multi-window 2-plane snapshot generation (K × 19×19 per cluster). Rust returns 2-plane views; Python assembles 18-plane tensors.
- [x] **Network Refactor**: Simplify to a single-trunk ResNet-12 that processes K clusters as a batch.
- [x] **Pipeline Integration**: Implement Value Pooling (min-pooling for pessimistic threat detection) and Policy Mapping (global coordinate translation).
- [x] **Un-constrain Bots**: Remove 19x19 bounds from SealBotBot to enable full colony meta play.
- [x] **ReplayBuffer (Rust)**: Port replay buffer and 12-fold hex augmentation to Rust (f16-as-u16 storage, zero-copy PyO3 transfer). Python `ReplayBuffer` deleted.
- [x] **128-bit Zobrist**: Upgrade hashing from 64-bit to 128-bit (splitmix128) to eliminate collision risk at sustained >150k sim/s.
- [x] **Benchmarking**: Verify throughput stays >5,000 pos/sec (Actual: ~52,000 pos/sec).

### Exit criteria

- Multi-window model maintains >50,000 pos/sec throughput on GPU. ✓
- Replay buffer sampling (batch=256) latency < 1,000 µs (raw: 951 µs; augmented: 936 µs). ✓
- `smoke_test_clusters.py` confirms detection of distant lethal threats. ✓
- `benchmark.py` passes all checks. ✓

---

## Phase 4.0 — Self-Play RL (CURRENT)

**Goal**: Run the distributed self-play loop continuously to produce training data from the bootstrapped model checkpoint.

The split-responsibility architecture is fully in place:

- **Rust workers** (`SelfPlayRunner`) drive MCTS and produce raw 2-plane snapshots + game records.
- **Python** (`GameState.to_tensor()`, `TensorBuffer`) assembles full 18-plane temporal tensors from `move_history`.
- **ReplayBuffer** (Rust) stores, augments, and serves training batches with zero-copy transfer.

### Tasks

- [x] Validate end-to-end self-play loop (worker → buffer → trainer) runs stably
- [x] Confirm positions/hour metric reported in structured logs and dashboard
- [x] Verify loss decreases over first training steps vs bootstrap checkpoint (4,940 steps run, 5 issues found and fixed)
- [x] Confirm auto-promotion triggers when win rate ≥ 55% vs bootstrap model
- [x] Run `make bench.full` — all 10 targets pass with Phase 4.0 config (2026-04-02)
- [x] Pretrain validated: policy loss 5.0 → 2.07, ≥95/100 wins vs RandomBot
- [x] Codebase cleanup: directories renamed (engine, hexo_rl, monitoring), Rust prefix removed from all PyO3 exports
- [x] Dashboard rebuilt — event fan-out pattern (events.py → terminal + web renderers)
- [x] Game viewer implemented — `/viewer` endpoint, threat overlay, replay scrubber, play-against-model
- [x] Threat detection in Rust — `Board.get_threats()`, `engine/src/board/threats.rs`
- [x] Benchmark rebaselined — 2026-04-03, correct 12-block × 128-channel model, all 10 metrics PASS
- [ ] **Sustained training run** — 24–48 hour run, monitor for policy entropy collapse, value loss plateau
- [ ] **Q2 ablation** — value aggregation strategy: min vs mean vs attention (highest-priority open question)

### Exit criteria

- Self-play runs ≥ 24 hours without worker crash or buffer corruption
- Policy loss and value loss both decrease from bootstrap baseline over first 500 RL steps
- Worker throughput ≥ 1,000,000 pos/hr sustained (current baseline: 1,177,745)

---

## Phase 4.1 — Evaluation & Elo

**Goal**: Automated, reliable measurement of model strength over time.

### Tasks

- [ ] Implement Elo ladder: each checkpoint gets an Elo rating
- [ ] Implement tournament runner (round-robin, configurable games per pair)
- [ ] Integrate fixed reference opponents (minimax depth 3, depth 5)
- [ ] Auto-promote best checkpoint when win rate ≥ 55% vs previous best
- [ ] Generate and save PGN/SGF game records from eval games for review
- [ ] Plot Elo over training iterations (logged to TensorBoard / wandb)

### Exit criteria

- Elo updates correctly and monotonically over first 500 training iterations
- A game record from an evaluation game can be loaded and replayed
- Auto-promotion works: new best checkpoint is loaded by workers automatically

---

## Phase 5 — Long-run training (ongoing)

**Goal**: Train to human-competitive strength.

### Milestones

| Milestone | Indicator |
|---|---|
| Beats minimax depth-5 consistently | Win rate ≥ 70% over 100 games |
| Reaches basic tactical competence | Manual review: no obvious blunders |
| Beats minimax depth-7+ | Win rate ≥ 60% |
| Human-competitive | Community players report it plays strong moves |
| Surpasses top humans | Win rate ≥ 55% vs community's strongest known players |

### Ongoing tasks

- Monitor training stability (loss should not diverge)
- Periodically export opening analysis for community review
- Tune hyperparameters (c_puct, Dirichlet alpha, LR) based on Elo trajectory
- Archive best checkpoints at each milestone

---

## Phase 6 — Tooling & community interface (parallel with Phase 5)

**Goal**: Make the model useful to the community beyond just playing.

### Tasks

- [ ] Opening book query mode: given a position, show top moves + estimated win probability
- [ ] Game analysis mode: replay a game with move-by-move network evaluation
- [ ] SGF import/export for game records
- [ ] Optional: simple web interface for playing against the model

---

## Known risks and mitigations

| Risk | Mitigation |
|---|---|
| Win detection bug causes incorrect training data | Phase 0 exit criteria are strict — do not skip |
| GPU underutilization kills training speed | Phase 2 benchmark gate — must hit targets before proceeding |
| Self-play collapses into repetitive patterns | Monitor policy entropy; apply Dirichlet noise; check temperature |
| Reward hacking from shaped rewards | Decay schedule is mandatory; monitor for non-terminal rewards dominating |
| Bootstrap model too strong — self-play can't improve on it | Use minimax depth ≤ 5; deeper creates too strong a prior |
| Rust/PyO3 ABI issues across environments | Pin `maturin` and `pyo3` versions in `Cargo.toml` and `requirements.txt` |
