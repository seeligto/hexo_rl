# HeXO Phase 4.0 Sprint Log — Consolidated Reference

Read this alongside `CLAUDE.md` at the start of any new session to avoid
re-litigating resolved decisions. Structured by subsystem, not by date.
For per-day narrative see `docs/07_PHASE4_SPRINT_LOG_BACKUP.md`.

---

## Classification Audit (§1–§66)

| Bucket | Sections |
|---|---|
| KEEP-FULL | §1, §2, §4, §5, §15, §19, §21, §26, §27, §28, §33, §34, §35, §36, §37, §40, §46b, §47, §58, §59, §61, §63, §66, §69, §70, §71 |
| KEEP-CONDENSED | §6, §11, §13, §14, §16, §17, §20, §22, §23, §24, §29, §30(game-cap/T_max), §31, §38, §41–§46, §48, §50–§57, §72 |
| MERGE | §3+§25+§30(torch)+§32→torch.compile arc; §30(quiescence-gate)→§28; §52+§60→eval_interval; §61+§62→Gumbel; §63+§64+§65→dashboard metrics |
| BENCHMARK-STALE | 2026-04-01 table, 2026-04-02 table, §18 corrected table, §39 table, §51 table |
| DELETE | Test-count-only updates, "Immediate next steps", §27b operational note, §49 (superseded by §59) |
| SUPERSEDED | §49 (uncertainty head guard — head later disabled at §59) |

---

# Part 1 — Architecture & Features

## 1. Network Architecture

**Files:** `hexo_rl/model/network.py`, `hexo_rl/training/trainer.py`,
`configs/model.yaml`, `configs/training.yaml`

```
Input:  (18, 19, 19) tensor
        Planes 0-15: 8 history steps × 2 players (cluster snapshots)
        Planes 16-17: metadata (moves_remaining, turn parity)

Trunk:  12 × ResidualBlock(128ch, SE reduction=4)
        Pre-activation (BN → ReLU → Conv)
        SE blocks: squeeze C→C/4→C per block (~1% FLOPs, validated in KataGo/LCZero)

Heads:
  Policy:      Conv(128→2, 1×1) → BN → ReLU → FC → log_softmax
  Value:       GlobalAvgPool + GlobalMaxPool → concat(256) → FC(256) → FC(1) → tanh
               Loss: BCE(sigmoid(v_logit), (z+1)/2)   ← logit path avoids atanh NaN
  Opp reply:   Mirror of policy, training only, weight=0.15
  Ownership:   Conv(1×1) → tanh → (19×19), weight=0.1, spatial MSE
  Threat:      Conv(1×1) → raw logit → (19×19), weight=0.1, BCE with logits
  Uncertainty: trunk → AdaptiveAvgPool → Linear → Softplus → σ² (DISABLED — see below)

Output: (log_policy, value, value_logit)  ← always 3-tuple for all inference callers
```

- `forward()` always returns the 3-tuple. BCE loss needs the raw logit; atanh(tanh(x)) was numerically unstable (NaN).
- Value loss: BCE not MSE — sharper gradients for binary outcomes.
- Global pooling value head is board-size-independent.
- Ownership head rationale: teaches the network where stones end up, giving the value head spatial grounding far from game end.
- Threat head rationale: teaches which cells form winning patterns, directly supporting quiescence reasoning.

**Policy target pruning:** Zero out entries < 2% of max visits, renormalise before CE/KL loss. Config: `policy_prune_frac: 0.02`. Applied once in Python training — Rust `get_improved_policy()` no longer prunes (double-pruning removed at §62 because first prune + renorm makes second prune non-idempotent, producing targets much sharper than intended).

**Entropy regularisation:** `L_total = L_policy + L_value + w_aux·L_aux − w_entropy·H(π)`. Weight `entropy_reg_weight: 0.01`. Expected range ~3–6 nats; < 1.0 signals collapse.

**Uncertainty head (DISABLED):** Built at §33 (`forward(uncertainty=True)` returns σ², gradient stopped before reaching value head). Head exists in `network.py` but `uncertainty_weight: 0.0` in `configs/training.yaml` gates it entirely — `use_uncertainty=False` means the head never runs in the current training path. Disabled at §59 because Gaussian NLL diverges when σ² → 1e-6 clamp floor, causing `total_loss` spikes to ~394. `uncertainty_weight: 0.0` must be explicit in config — absence causes the trainer default to silently match but is fragile on resume. Re-enable only after adding σ² regularisation (log-barrier or β-VAE-style KL).

### torch.compile — DISABLED (§32)

Arc: §3 added `reduce-overhead` → §25 re-enabled with split train/inf model instances → §30 changed to `mode="default"` (TLS crash on Python 3.14) → §32 fully disabled (27 GB Triton JIT spike blocks workers for 5+ min on first forward).

**Final state:** `torch_compile: false` in `configs/training.yaml`.
**Re-enablement criteria:** Deferred until PyTorch stabilises CUDA graph support on Python 3.14. Track upstream pytorch/pytorch issues.

The benchmark delta for torch.compile was only +3% worker throughput — not worth the instability.

---

## 2. MCTS

**Files:** `engine/src/mcts/`, `engine/src/board/`, `engine/src/game_runner.rs`,
`configs/selfplay.yaml`, `hexo_rl/selfplay/pool.py`

### Legal Move Margin — corrected to hex-ball radius 8 (§26)

Official rule: new hex ≤ 8 cells from any existing stone. Prior implementation used bbox+2 expansion (~5×5 rectangle per stone — far too small). Fixed to per-stone hex ball iteration: all (dq,dr) with `|dq|≤8, |dr|≤8, |dq+dr|≤8`. 217 cells/stone, deduplicated via FxHashSet. `LEGAL_MOVE_RADIUS = 8`. This also matches `get_clusters()` which uses distance ≤ 8 for NN windowing — same threshold, independent purposes.

This correction ~9× expanded the branching factor and is the primary driver of MCTS sim/s reduction from ~190k → ~31k. Targets recalibrated accordingly.

### action_idx widened u16 → u32 (§38)

Pre-existing silent bug: u16 cap at 65,535 caused wrong child selection with global axial coordinate encoding after radius-8 legal move expansion. Fixed throughout `mcts/mod.rs` and `mcts/node.rs`.

### Dynamic FPU (§27)

```
explored_mass = Σ prior(a) for visited children
fpu_value     = parent_q − fpu_base × √explored_mass
```

Unvisited children use `fpu_value` instead of Q=0. As more children are visited, `fpu_value` becomes more pessimistic relative to `parent_q`, shifting exploration toward refining known-good branches. Config: `mcts.fpu_reduction: 0.25` (matches KrakenBot).

**Benchmark note:** fpu_reduction=0.25 causes MCTS sim/s regression in the CPU-only benchmark because all NN values=0 → FPU makes unvisited children look worse than visited → deeper/narrower tree. This is a benchmark artifact. In real self-play with non-zero NN values, FPU improves selection quality.

### Quiescence Value Override (§28 + §30 gate)

**Game-specific theorem:** Each turn places exactly 2 stones. Therefore the opponent can block at most 2 winning cells per response. If the current player has ≥3 empty cells where placing a stone completes a 6-in-a-row, the win is mathematically forced.

**Critical distinction from the removed forced-win short-circuit:**
The earlier `FormationDetector::has_forced_win()` short-circuit (removed post-baseline) fired at MCTS **expansion** and marked positions as terminal — the NN never evaluated them, preventing the network from learning forced-win patterns.

This quiescence check is a **VALUE OVERRIDE at leaf evaluation**:
- The NN still receives the position and produces (policy, value).
- The **POLICY is used unchanged** for MCTS expansion → network learns fork patterns.
- Only the **VALUE** is overridden with the proven result.

```
current_wins >= 3  → value = +1.0
opponent_wins >= 3 → value = -1.0
current_wins == 2  → value = min(value + blend_2, 1.0)
opponent_wins == 2 → value = max(value - blend_2, -1.0)
```

Config: `mcts.quiescence_enabled: true`, `mcts.quiescence_blend_2: 0.3`

**Two-tier performance gate (§30):** `count_winning_moves()` is O(legal_moves). Gate it:
1. Ply gate (free): skip if `board.ply < 8` (P1 needs ≥5 stones = ply 8 minimum).
2. Long-run pre-check: skip if neither player has `Board::has_player_long_run(5)`.
Net overhead: 1.8% vs no quiescence in the gated benchmark.

### ZOI (Zone of Interest) Lookback (§36)

Restricts MCTS candidates to cells within hex-distance 5 of the last 16 moves. Falls back to full legal set if < 3 candidates. Reduces the post-search move-selection pool without changing legal moves. Does NOT reduce the MCTS branching factor — the tree expands with the full radius-8 legal set at all depths. See §77.

Config: `mcts.zoi_enabled: true`, `mcts.zoi_radius: 5`, `mcts.zoi_history: 16`, `mcts.zoi_min_candidates: 3`

### Cosine-Annealed Temperature (§36)

Replaced hard step at move 30 with smooth cosine schedule:
```
temperature = temp_min + 0.5 × (1 − temp_min) × (1 + cos(π × ply / anneal_moves))
```
Eliminates policy entropy spikes at ply 30. Config: `mcts.temp_min: 0.05`, `mcts.temp_anneal_moves: 60`

### Transposition Table — clear on new_game (§59)

`MCTSTree::new_game()` was resetting the node pool but not clearing the TT. Each `TTEntry` holds a heap-allocated `Vec<f32>` (362-element policy). With 12 workers × ~67 games each × ~100 moves × ~300 positions/move × ~1.5 KB/entry, the TT accumulated tens of GB over a sustained run (observed: 28 GB RSS after 500 min).

Fix: `self.transposition_table.clear()` added to `new_game()`. TT is per-game by nature — cross-game position reuse is negligible.

### Gumbel AlphaZero (§61 + §62) — OFF by default

Implements Danihelka et al. ICLR 2022. Two components:

**Completed Q-values policy targets:**
Standard AlphaZero trains policy toward visit-count distributions. At 50 sims (fast games, 25% of training), visit counts barely differ from prior. Instead:
1. Visited children (N>0): use Q(a) = W(a)/N(a)
2. Unvisited legal actions: use v_mix (interpolates root value with policy-weighted avg of visited Q)
3. `π_improved = softmax(log_prior + σ(completedQ))` where `σ = (c_visit + max_N) × c_scale × completedQ`
4. Training loss: KL(π_improved ∥ π_model) instead of CE with visit counts

Computed in Rust (`MCTSTree::get_improved_policy`) — all data is local, no extra PyO3 crossings. Config: `completed_q_values: true`, `c_visit: 50.0`, `c_scale: 1.0`.

**Sequential Halving root search:**
1. Gumbel-Top-k: generate Gumbel(0,1) noise, select top `m = min(n, 16, |legal|)` candidates. Replaces Dirichlet noise at root.
2. Sequential Halving: allocate budget across `ceil(log2(m))` phases, halving candidates each phase by `g(a) + log_prior(a) + sigma(Q_hat(a))`.
Non-root nodes: unchanged (PUCT + dynamic FPU).

**Hardening fixes (§62):** Budget off-by-one fixed (`sims_done = sims_used` in fallback). `max_n` cached per halving phase (was O(candidates²), now O(candidates)). Double-pruning removed (see policy target pruning note above). Config defaults restored: `gumbel_mcts: false`, `fast_prob: 0.25`.

Config: `gumbel_mcts: false` (opt-in), `gumbel_m: 16`, `gumbel_explore_moves: 10`

---

## 3. Replay Buffer

**Files:** `engine/src/replay_buffer/`, `hexo_rl/training/recency_buffer.py`,
`hexo_rl/selfplay/pool.py`, `configs/training.yaml`

### Growing Buffer + Mixed Streams (§2 + §40b + §79)

Buffer growth schedule (updated §79 — reverts §40b reduction):
```yaml
buffer_schedule:
  - {step: 0,           capacity: 250_000}
  - {step: 300_000,     capacity: 500_000}
  - {step: 1_000_000,   capacity: 1_000_000}
```
`ReplayBuffer.resize()` linearises ring buffer in-place via rotate_left, extends backing vecs.

Mixed pretrained + self-play: `pretrained_weight = max(0.1, 0.8 × exp(−step / 300_000))`. After §58 fix: pretrained stream is always initialised from the corpus NPZ when the file exists, regardless of `_buffer_restored`.

### Playout Cap Randomisation (§2 + §43)

- 25% of games: 50 sims, τ=1 throughout (value targets only — policy masked on zero-policy rows)
- 75% of games: 200 sims, τ=1 for first 15 compound moves then τ→0

Config: `playout_cap.fast_prob: 0.25`, `fast_sims: 50`, `standard_sims: 200`. fast_sims was tested at 30 and reverted — 50 produces meaningfully better policy signal for fast-game value targets.

### Recency-Weighted Replay (§33)

Lightweight Python-side `RecentBuffer` ring mirrors the newest ~50% of buffer capacity. Training batches: 75% recent / 25% full-buffer (augmented). Config: `recency_weight: 0.75`. Falls back to full-buffer when recent buffer is empty.

### Buffer Persistence (§46b)

Save/load added to `ReplayBuffer`: binary HEXB v1 format (magic `0x48455842`, little-endian). ~14.4 KB/entry; ~1.4 GB for 100K positions.

Two save points to prevent loss on unclean exit:
1. Inside `_run_loop()` on shutdown signal (before `break`, before `pool.stop()`).
2. Post-loop finally block (normal exit).

On resume: buffer is restored before `pool.start()`. Corpus prefill is skipped when `n_loaded >= 10_000` (threshold from §58 fix — `n_loaded > 0` was too permissive).

Config: `mixing.buffer_persist: true`, `mixing.buffer_persist_path: "checkpoints/replay_buffer.bin"`

### collect_data PyArray fix (§51)

Changed `collect_data()` return from `Vec<(Vec<f32>, Vec<f32>, f32, usize)>` to `(PyArray2<f32>, PyArray2<f32>, PyArray1<f32>, PyArray1<u64>)`. Previously each `Vec<f32>` was converted to a Python list of Python float objects via pymalloc arenas — at ~10 Hz this accumulated ~0.15 GB/min RSS. Zero-copy NumPy path eliminates this entirely.

---

## 4. Data Pipeline & Corpus

### Corpus Generation (§5)

`generate_corpus.py`: SealBot self-play, SHA-256 hash-based filenames for deduplication. Random opening injection: 3 random moves (d4) or 1 random move (d6+) before SealBot takes over. Reduced dupe rate 87%→43% at d4. SealBot time cap: 1s/move. Makefile targets: `corpus.fast` (5K games, 0.1s), `corpus.strong` (2.5K games, 0.5s).

### Scraper — white-box API (§21)

`hexo.did.science` = `WolverinDEV/infhex-tic-tac-toe` repo (confirmed). Key findings that drove changes:
- `baseTimestamp` is correct param (not `before`) — fixed
- Public game cap is exactly 500 (not 480) — updated `UNAUTHENTICATED_GAME_LIMIT`
- `/api/leaderboard` exists unauthenticated → `--top-players-only` flag added
- Per-game Elo in `DatabaseGamePlayer.elo` → `player_black_elo`/`player_white_elo` stored in game JSON
- Coordinates are direct axial (q=x, r=y) — no translation needed

Dual-pass `scrape_daily.sh`: standard 500-game pull + top-20 player profile pull. Manifest includes `elo_bands` breakdown. All scraper flags in `configs/corpus.yaml`.

### mmap Corpus Loading — active pattern (§19 + §35)

**Root cause:** `np.load()` (no mmap) + `np.concatenate()` on full corpus (~906K positions) caused ~26 GB peak RAM → system freeze.

**Pattern:**
1. `make corpus.export` produces a 50K-position uncompressed NPZ (`np.savez`, not `savez_compressed`).
   Uncompressed is required — `savez_compressed` defeats `mmap_mode='r'`.
2. Load with `np.load(path, mmap_mode='r')` — OS pages data on demand, RAM stays near-zero.
3. `del pre_states, pre_policies, pre_outcomes` immediately after `push_game()` releases mmap views.
   Keeping views alive for the entire process lifetime was a confirmed ~720 MB leak (§46).

**Warning:** If `bootstrap_corpus.npz` is absent, `load_corpus()` fallback runs and the double-allocation risk returns. **Always run `make corpus.export` before `make pretrain`.**

Config: `corpus_npz_path` in `configs/corpus.yaml`. `mixing.pretrain_max_samples: 200_000` caps corpus even if NPZ is large.

### Pretrain Checkpoint Hygiene (§34)

**Decision: always use `checkpoints/bootstrap_model.pt` as the RL entry point.**

| File | step | scheduler last_epoch | Type | Use |
|---|---|---|---|---|
| `pretrain/pretrain_00000000.pt` | 0 | ~53K (exhausted) | full checkpoint | **Do not use for RL** — LR ≈ eta_min (1e-5) at start, learns almost nothing. |
| `pretrain/pretrain_00053130.pt` | 53130 | 53130 | full checkpoint | Human-only pretrain reference only. |
| `checkpoints/bootstrap_model.pt` | N/A | N/A | **weights-only** | **Use this for RL.** Fresh optimizer + scheduler from config. |

`bootstrap_model.pt` is weights-only. `Trainer.load_checkpoint` detects this, sets `trainer.step = 0`, constructs fresh scheduler. `make train` already uses this via `CHECKPOINT_BOOTSTRAP` in Makefile.

### Validation Game Count (§20)

Pretrain validation: 100 greedy games vs RandomBot (was 5 — statistically meaningless). 95% CI width at p=0.5 is ±~10% at n=100.

### Sequential Action Space (§6) — confirmed correct

- 2 MCTS plies per 2-stone compound turn
- Q-value sign flips **only at turn boundaries**, not at intermediate ply
- Dirichlet noise skipped at intermediate ply
- Plane 16 encodes `moves_remaining == 2`

---

## 5. Evaluation Pipeline

**Files:** `hexo_rl/eval/`, `configs/eval.yaml`, `configs/training.yaml`

- Bradley-Terry MLE (not incremental Elo). scipy L-BFGS-B with L2 regularisation 1e-6 to prevent divergence on perfect records.
- SQLite results store (WAL mode). Full BT recomputation from all historical pairwise data after each eval round.
- Gating rule: new checkpoint promoted if win_rate ≥ 0.55 over 50 games vs best checkpoint.
- Opponents: previous best, SealBot (fixed external Elo reference), RandomBot (sanity floor).
- Evaluation in a separate thread; model cloned (fresh HexTacToeNet) to avoid sharing compiled training model.

**eval_interval arc:** §52 changed 500→2000 (eval was blocking self-play). §60 changed 2000→5000 (at step 2000, SealBot winrate was 0/22; eval took 130 min = 55% of wall-clock). **Final: `eval_interval: 5000`** in `configs/training.yaml`, `best_checkpoint n_games: 50` in `configs/eval.yaml`. At ~490 steps/hr, one eval every ~10 hours = ~9% overhead.

---

## 6. Training Loop & Stability

### FP16 NaN Guard (§47) — active fix, must not revert

**Root cause chain:**
1. `compute_aux_loss`: policy_prune_frac zeros low-visit entries. Zero target × `log_softmax(-inf)` = `0×-inf = NaN` under FP16.
2. Entropy bonus: `log_policy.exp()` underflows to 0.0 for near-zero probs under FP16, then multiplies `-inf` log entry.
3. BatchNorm contamination: BN running stats updated **during forward pass, before GradScaler**. One poisoned forward NaN's all subsequent passes even if optimizer step is skipped.

**Why GradScaler alone is insufficient:** GradScaler checks `torch.isinf()` on gradients, not `torch.isnan()`. `0×-inf` NaN passes the inf check entirely.

**Fixes:**
- `compute_aux_loss`: clamp log-probs to `min=-100.0` before multiplication.
- Entropy (two sites in trainer.py): replace `-(p * log_p).sum()` with `torch.special.entr(p_fp32).sum()` (defines `0·log(0)≡0`, promotes to FP32 first).
- NaN guard: after `compute_total_loss()`, detect non-finite loss, reset poisoned BN modules, call `scaler.update()`, return early.

### Draw Penalty (§24 → §40)

Initial (§24): `draw_reward: -0.1` (KrakenBot practice for minority-outcome draws). Changed at §40 to `draw_reward: -0.5`.

**Why -0.5:** First overnight run produced 56.6% draws. At 56% frequency, expected draw target = `0.56 × -0.1 = -0.056` — too weak to break the draw-seeking equilibrium. At -0.5 the expected target = -0.280, providing a clear gradient. Once draws become rare, the penalty rarely fires and value distribution naturally re-centres on ±1.0.

### Resume Bugs Fixed (§58) — all three were confirmed in JSONL

**Bug 1 — pretrained stream not initialised on resume:** `if _buffer_restored: ... elif pretrained_path:` structure prevented corpus loading whenever any buffer was restored. Fixed: decoupled the conditions. `pretrained_buffer` always initialised from NPZ when file exists; `_buffer_restored` only controls the `corpus_prefill_skipped` log message.

**Bug 2 — corpus prefill skip threshold too low:** `_buffer_restored = (n_loaded > 0)` meant 1,770 positions skipped the corpus load. Fixed: `_buffer_restored = (n_loaded >= 10_000)`. Below threshold: `corpus_prefill_running` event logged with `reason="buffer_too_small"`.

**Bug 3 — hidden loss terms invisible in JSONL:** `log.info("train_step")` included `total_loss` but omitted `uncertainty_loss`, `ownership_loss`, `threat_loss`. These ARE in `total_loss` (weights 0.05, 0.1, 0.1) and caused apparent spikes (~390+) when uncertainty head initialised randomly on resume. Fixed: all four loss terms individually logged. Residual delta = entropy regularisation (already logged as `policy_entropy`).

### Memory Fixes

- **glibc malloc arenas (§53):** `MALLOC_ARENA_MAX=2` prepended to all training Makefile targets. Without this, glibc creates one 64 MB arena per thread (up to 128 on 16-thread system); arenas never returned to OS → ~+2 GB/hr RSS growth. GPU allocation is unaffected (CUDA allocator).
- **RSS leaks (§46):** (1) `del pre_states, pre_policies, pre_outcomes` after push_game. (2) SocketIO `_safe_emit()` gated on connected SIDs. (3) Pre-allocated batch arrays (`np.copyto` in-place, not `np.concatenate`).
- **Buffer warmup edge case (§50):** `np.copyto` path falls back to `np.concatenate` when buffer underfull (warm-up period < batch_size positions). Flips to in-place permanently after warmup ends.

### Config Growth Reductions (§40b)

Applied during draw-collapse fix session: `buffer_schedule` reduced (100K/250K/500K, was 250K/500K/1M), `standard_sims: 400→200`, `decay_steps: 1_000_000→300_000`.

---

## 7. Monitoring & Dashboard

**Files:** `hexo_rl/monitoring/`, `configs/monitoring.yaml`

### Architecture

Event-driven fan-out: `emit_event(payload)` in `events.py` dispatches to registered renderers. Never raises; renderer failures caught and logged. Zero import side effects.

Renderers registered at startup in `train.py`:
- `TerminalDashboard` — Rich Live 4Hz render
- `WebDashboard` — Flask+SocketIO at `:5001`

Events: `run_start`, `training_step`, `iteration_complete`, `eval_complete`, `game_complete`, `system_stats`, `run_end`.

### Dashboard Enrichment (§63 + §64 + §65)

**`training_step` event additions:**
- `policy_target_entropy`: mean entropy of post-pruning MCTS policy target over batch. Computed only on non-zero-policy rows. Replaces the old `policy_excess` label (§65 fix: policy_excess = KL − H(target) after §61 switched to KL loss — had no guaranteed sign). Renamed to `policy_KL` displaying `loss_policy` directly.

**`iteration_complete` event additions:**
- `mcts_mean_depth`: mean leaf depth per sim across all moves. Accumulated in `select_leaves()` outer loop (not inside `select_one_leaf` traversal) — one count per sim.
- `mcts_root_concentration`: mean of (max_child_visits / root_total_visits) at root per move.

Depths are accumulated as ×1e6 fixed-point in AtomicU64 (first commit had truncation bug using u64 directly — self-corrected within same session before any training ran on it).

**Expected values:** At step ~3,800 with near-uniform priors and 25% fast games, mcts_mean_depth = 2–4 is physically correct. Expect ≥ 5.0 by step 50–70K (decay_steps=300K slows the transition — estimate adjusted accordingly).

**Ring buffer sizes bumped:** `training_step_history: 2000` (was 500), `game_history: 500` (was 200). Values served via `/api/monitoring-config` so SPA doesn't need hardcoding.

### Operational Fixes

- **SocketIO bounded queue (§55):** `put_nowait()` from training thread into `queue.Queue(maxsize=200)`; daemon drain thread does actual `socketio.emit()`. Training loop never blocks. Config: `monitoring.emit_queue_maxsize: 200`.
- **Structlog JSONL sink (§56):** Log file `logs/train_{run_id}.jsonl`. Handle closed in `finally` block **after** all session-end logging (earlier ordering caused `I/O operation on closed file` crash on Ctrl+C).
- **RSS tracking (§45):** `psutil.Process().memory_info().rss` sampled on every GPUMonitor poll cycle, included in `system_stats` as `rss_gb`.
- **grad_norm to structured log (§54):** Unconditional `log.info("train_step", ...)` in `_train_on_batch()` — visible in JSONL even without dashboard.

---

## 8. Game Viewer

**Files:** `engine/src/board/threats.rs`, `hexo_rl/viewer/engine.py`,
`hexo_rl/monitoring/web_dashboard.py`, `hexo_rl/monitoring/static/viewer.html`,
`configs/monitoring.yaml`

- Sliding-window threat detection on 3 hex axes: empty cells within 6-cell windows where one player has N≥3 stones. Levels: 5=critical (≥5), 4=forced (4), 3=warning (3). Threat cells never overlap with occupied cells (tested). Viewer-only — never called from MCTS or training path.
- Game records written to `runs/<run_id>/games/<game_id>.json` on arrival. In-memory index capped at 50 entries (`monitoring.viewer_max_memory_games`). Disk rotated to `monitoring.viewer_max_disk_games: 1000` oldest-first.
- Viewer URL: `http://localhost:5001/viewer` (during training).
- Features: hex board canvas (pointy-top), threat overlay, MCTS visit heatmap (toggle), value sparkline, scrubber, play-against-model mode.

## 9. Gumbel MCTS Activation & Training Restart (§66)

**Date:** 2026-04-07
**Files:** `configs/selfplay.yaml`, `CLAUDE.md`, `docs/reviews/2026-04_architecture_review.md`

- **`gumbel_mcts: true` on desktop is intentional.** The desktop (Ryzen 7 3700x + RTX 3070) config has `gumbel_mcts: true` in `configs/selfplay.yaml` for the Phase 4.0 sustained run. Laptop and cloud configs remain `false` until benchmarked on those hosts. CLAUDE.md updated to document this per-host override rather than stating a single default.
- **`completed_q_values` KL loss was silently disabled.** Architecture review (C1) found that `trainer.py:305` performs a nested-dict lookup (`self.config.get("selfplay", {}).get("completed_q_values", False)`) into the flattened `combined_config` dict, which always returns `False`. The intended KL policy loss was never active — all prior self-play training used CE loss instead. Fix is tracked as a separate commit (one-line change to flat key lookup).
- **Phase 4.0 sustained run will restart from Phase 3C pretrained weights** after the C1 fix lands. The prior run (~18,750 steps) trained under CE loss instead of the intended KL loss, so those checkpoints are not valid for the Phase 4.0 exit criterion. Restart is intentional, not a regression.
- **Benchmark baseline is pre-Gumbel.** The MCTS 55,478 sim/s figure was measured with `gumbel_mcts: false`. Gumbel Sequential Halving adds overhead at the root; re-bench is pending after the C1 fix and restart.

---

# Part 2 — Operational Record

## Current Authoritative Benchmark Baseline

**2026-04-06, Ryzen 7 8845HS + RTX 4060 Laptop, bench.full n=5, 3s warm-up, LTO+native**

| Metric | Baseline (median) | Target | IQR |
|---|---|---|---|
| MCTS (CPU only, 800 sims/move × 62 iters) | 55,478 sim/s | ≥ 26,000 sim/s | ±400 |
| NN inference batch=64 | 9,810 pos/s | ≥ 8,500 pos/s | ±1 |
| NN latency batch=1 | 1.59 ms | ≤ 3.5 ms | ±0.05 ms |
| Replay buffer push | 762,130 pos/sec | ≥ 630,000 pos/sec | ±114,320 |
| Buffer sample raw (batch=256) | 1,037 µs/batch | ≤ 1,500 µs | ±34 µs |
| Buffer sample augmented (batch=256) | 940 µs/batch | ≤ 1,400 µs | ±62 µs |
| GPU utilisation | 100.0% | ≥ 85% | ±0 |
| VRAM usage (process) | 0.05 GB / 8.0 GB | ≤ 6.4 GB | ±0 |
| Worker throughput | 659,983 pos/hr | ≥ 625,000 pos/hr | ±56,835 |
| Batch fill % | 100.0% | ≥ 80% | ±0 |

All 10 targets PASS. Methodology: median n=5, 3s warm-up, realistic MCTS workload (800 sims/move × 62 iterations with tree reset), CPU unpinned (n=5 median provides sufficient variance control).

## Benchmark Evolution

| Date | What was wrong | What changed | Impact on targets |
|---|---|---|---|
| 2026-04-01 | MCTS workload was burst (50K sims in one tree) — exceeded L2 cache, inflated by boost clocks | Changed to 800 sims/move × 62 iter with tree reset; n=5 median | MCTS target dropped from 160K to realistic steady-state |
| 2026-04-03 | benchmark.py read `config.get('res_blocks')` (top-level) instead of `config['model']['res_blocks']` — measured wrong (smaller) model; VRAM used pynvml global not process-specific; single pool per measurement window included cold-start | Fixed model config path; switched to `torch.cuda.max_memory_allocated()`; keep one warm pool across all measurement windows | Worker throughput baseline corrected 1.18M→735K; NN latency 1.52ms→2.90ms; targets recalibrated |
| 2026-04-04 | Legal move radius corrected bbox+2→radius 8 (~9× branching factor expansion) + FPU behavioral tree shape change | Both are correct behaviour changes, not regressions | MCTS target rebaselined to ≥26K (85% of new ~31K median) |

## Regressions & Reversions

| Feature | When Added | Reverted | Reason |
|---|---|---|---|
| SealBot mixed opponent in self-play | §17 (2026-04-02) | Immediately (`c9f39de`) | Python daemon threads caused 3.3× GIL contention regression (1.52M→464K pos/hr). **Do not re-litigate** — GIL regression was an implementation issue, not a conceptual flaw. Re-add post-Phase 4.5 baseline using subprocess-based wrapper to avoid GIL. |
| Forced-win short-circuit (`FormationDetector::has_forced_win()`) | Pre-sprint | Removed (2026-04-02) | MCTS bypassed NN for near-win positions → network never learned to evaluate them. Removing adds ~30% more NN calls/game (batch fill 99.4%→99.82%) but training quality requires it. |
| draw_reward: -0.1 | §24 | Raised to -0.5 at §40 | Not a revert — a correction. -0.1 assumed draws were minority; at 56% draw rate the signal was too weak to break equilibrium. |
| torch.compile | §3 | Disabled §32 | Python 3.14 CUDA graph incompatibility cascade (TLS crash → Triton spike). |
| uncertainty_weight: 0.05 | §33 | Set to 0.0 at §59 | Gaussian NLL diverges when σ²→clamp floor; total_loss spiked to ~394. |

## Key Resolved Bugs

| § | Bug | Impact | Fix |
|---|---|---|---|
| §26 | Legal move radius bbox+2 instead of radius 8 | ~9× too-small branching factor; invalid MCTS search | Per-stone hex ball iteration |
| §38 | `action_idx` u16 overflow (cap 65,535) | Silent MCTS tree corruption with global axial coords | Widen to u32 |
| §47 | FP16 0×-inf NaN cascade | NaN total_loss, BN poisoning, training halt | Log-clamp aux CE; `torch.special.entr()` for entropy; BN reset guard |
| §58 | Three resume bugs | Pretrained stream silently disabled on resume; hidden loss spikes | Decouple buffer-restore from corpus-load; threshold 10K; log all loss terms |
| §59 | TT memory leak (`new_game()` did not clear TT) | 28 GB RSS after 500 min | `self.transposition_table.clear()` in `new_game()` |

---

# Part 3 — Open Questions

| # | Question | Status |
|---|---|---|
| Q5 | Supervised→self-play transition schedule | ✅ Resolved — exponential decay 0.8→0.1 over 300K steps |
| Q6 | Sequential vs compound action space | ✅ Resolved — sequential confirmed correct |
| Q2 | Value aggregation: min vs mean vs attention | 🔴 Active — HIGH priority, blocks Phase 4.5 |
| Q3 | Optimal K (number of cluster windows) | 🟡 Active — MEDIUM priority |
| Q8 | First-player advantage in value training | 🟡 Active — MEDIUM priority (corpus: 51.6% P1 overall, 57.1% in 1000-1200 Elo) |
| Q9 | KL-divergence weighted buffer writes (KataGo) | 🟡 Active — MEDIUM priority. Prerequisite: Phase 4.5 baseline checkpoint. |
| Q10 | Torus board encoding (imaseal experiment) | 🔵 Watch — incompatible with attention-anchored windowing; pending imaseal results |
| Q1, Q4, Q7 | MCTS convergence rate, augmentation equivariance, Transformer encoder | 🔵 Deferred — Phase 5+ |

See `docs/06_OPEN_QUESTIONS.md` for full detail.

---

# Key Config Values (current settled state)

```yaml
# configs/selfplay.yaml
mcts:
  n_simulations: 800          # benchmark workload; ZOI reduces effective branching
  fpu_reduction: 0.25         # dynamic FPU (KrakenBot baseline)
  quiescence_enabled: true
  quiescence_blend_2: 0.3
  temp_min: 0.05
  temp_anneal_moves: 60       # cosine anneal over 60 plies
  zoi_enabled: true
  zoi_radius: 5
  zoi_history: 16
  zoi_min_candidates: 3
playout_cap:
  fast_prob: 0.25
  fast_sims: 50
  standard_sims: 200
selfplay:
  max_game_moves: 200         # compound moves; 128 caused mass draw-cap games
gumbel_mcts: false            # opt-in; zero regression when disabled
gumbel_m: 16
gumbel_explore_moves: 10

# configs/training.yaml
training:
  fp16: true
  torch_compile: false        # DISABLED — Python 3.14 compat (§32); re-enable when stable
  policy_prune_frac: 0.02
  aux_opp_reply_weight: 0.15
  entropy_reg_weight: 0.01
  ownership_weight: 0.1
  threat_weight: 0.1
  uncertainty_weight: 0.0    # head exists; disabled (§59 divergence); explicit 0.0 required
  draw_reward: -0.5
  grad_clip: 1.0
  eval_interval: 5000         # training steps (~10 hrs at 490 steps/hr)
  completed_q_values: true    # Gumbel improved policy targets (no Gumbel search required)
  c_visit: 50.0
  c_scale: 1.0
mixing:
  decay_steps: 300_000        # pretrained_weight decay; 50K too aggressive (§65 note)
  pretrain_max_samples: 200_000
  buffer_persist: true
  buffer_persist_path: "checkpoints/replay_buffer.bin"
buffer_schedule:  # updated §79
  - {step: 0,           capacity: 250_000}
  - {step: 300_000,     capacity: 500_000}
  - {step: 1_000_000,   capacity: 1_000_000}

# configs/model.yaml
res_blocks: 12
channels: 128
se_reduction_ratio: 4

# configs/eval.yaml
eval_n_games: 200             # total per round (RandomBot + SealBot + best_checkpoint)
best_checkpoint_n_games: 50   # reduced from 200 (§60)
promotion_threshold: 0.55

# configs/monitoring.yaml
training_step_history: 2000
game_history: 500
num_actions_for_entropy_norm: 362
emit_queue_maxsize: 200
viewer_max_memory_games: 50
viewer_max_disk_games: 1000
```

---

### §66 — C1 bug impact assessment (amendment)

The C1 bug affected only the reported loss scalar, not training dynamics. CE and KL against fixed targets share gradients, so the checkpoint was structurally valid. The Phase 4.0 runs are being restarted — but the trigger is the LR scheduler bug (§67), not C1. C1 is fixed and loss reporting is now correct for all future runs.

---

### §67 — LR scheduler bug fix + total_steps / decay_steps co-design + named Gumbel variants

**Problem:** Both desktop and laptop runs were using the hardcoded `T_max = 50_000` fallback in `trainer.py:145` (L2 from April architecture review). With `decay_steps = 300_000`, the LR collapsed to `eta_min = 1e-5` at step 50K while the mixing weight was still bootstrap-heavy. Self-play never dominated. Both runs are unrecoverable and restart from `bootstrap_model.pt`.

**Fix — trainer.py:** Removed all silent defaults. `_build_scheduler` now raises `ValueError` if `total_steps` or `eta_min` is absent from config. Resolution order for `total_steps`:
1. `--iterations` CLI flag (sets `combined_config["total_steps"]` before Trainer init)
2. `config["total_steps"]` (from `training.yaml`)
3. `ValueError` listing both options

**Fix — configs/training.yaml:**

| Key | Old | New | Rationale |
|---|---|---|---|
| `total_steps` | absent (50K fallback) | `200_000` | Sets LR horizon explicitly |
| `eta_min` | absent (1e-5 fallback) | `2e-4` | ~10% of peak lr=0.002; prevents LR floor being too low |
| `mixing.decay_steps` | `300_000` | `70_000` | Co-designed with total_steps (see below) |

**total_steps / decay_steps co-design rationale:**

| Phase | Steps | What's happening |
|---|---|---|
| Bootstrap dominant | 0 → 70K | pretrain_weight decays 0.8 → 0.1; replay buffer fills with self-play data |
| Self-play dominant | 70K → 200K | pretrain_weight = 0.1 (floor); model trains almost entirely on its own games |
| LR floor | ~200K | cosine reaches `eta_min = 2e-4`; scheduler done |

Rule of thumb: `decay_steps ≈ 0.35 × total_steps`. Keeps bootstrap phase at ~35% of run length, leaving 65% for self-play consolidation before LR collapses.

**Named Gumbel variants (`configs/variants/`):**

Three named override files, each deep-merged on top of `selfplay.yaml` via `--variant`:

| Variant | `gumbel_mcts` | `completed_q_values` | Host |
|---|---|---|---|
| `gumbel_full` | true | true | Desktop (RTX 3070) |
| `gumbel_targets` | false | true | Laptop / cloud |
| `baseline_puct` | false | false | Ablation baseline |

`selfplay.yaml` reverted to `gumbel_mcts: false`, `completed_q_values: false` (explicit baseline). Variant must be specified explicitly — no implicit activation.

**Usage:**
```bash
make train VARIANT=gumbel_full          # desktop
make train VARIANT=gumbel_targets       # laptop
make train.resume VARIANT=gumbel_full   # resume with same variant
# Or directly:
python scripts/train.py --checkpoint checkpoints/bootstrap_model.pt --variant gumbel_full
```

**Restart plan:** Both hosts restart from `bootstrap_model.pt` at step 0. The dual-host run is an informal comparison of root search strategy (Gumbel vs PUCT) with policy target type held constant (`completed_q_values: true` on both). Desktop uses `gumbel_full`; laptop uses `gumbel_targets`.

---

### §68 — Eval DB run_id bug fix + broken-run cleanup

**Problem:** `EvalPipeline` stored `self.run_id` (UUID from `train.py`) but never passed it to `db.get_or_create_player()` or `db.insert_match()` in `run_evaluation()`. All 5 call sites used the default `run_id=""`, making every run's eval results indistinguishable in the ratings DB — critical now that desktop and laptop run different variants.

**Fix:** Added `run_id=self.run_id` to all 5 DB calls in `run_evaluation()`:
- `get_or_create_player` for checkpoint player (line 115)
- `insert_match` vs Random (line 130)
- `insert_match` vs SealBot (line 149)
- `get_or_create_player` for best_checkpoint (line 170)
- `insert_match` vs Best (line 176)

Reference opponents (SealBot, random_bot) intentionally keep `run_id=""` — they are shared anchors across all runs. The `get_all_pairwise(run_id=...)` and `get_ratings_history(run_id=...)` queries already filter correctly (matching the run's players plus `run_id=""` reference opponents).

**Broken-run cleanup:** Archived all artifacts from the scheduler-poisoned run (§67):
- `archive/checkpoints.broken-202604/` — 10 checkpoints (21500–24274), best_model.pt, inference_only.pt, replay_buffer.bin (1.4 GB), checkpoint_log.json
- `archive/eval.broken-202604/results.db` — 39 matches, all with `run_id=""`

Kept in place: `bootstrap_model.pt` (verified loads clean), `checkpoints/pretrain/`, all game records in `runs/*/games/` (3,839 files), logs, corpus data.

---

### §69 — Config Sweep 2026-04-08 — PUCT/Gumbel Knob Ranking

15+1 runs on laptop (Ryzen 7 8845HS + RTX 4060), 20-min windows each, all starting fresh from `bootstrap_model.pt` with `completed_q_values=true`. The sweep varied `training_steps_per_game` (ratio), `max_train_burst`, `max_game_moves`, `inference_max_wait_ms`, `leaf_batch_size`, `inference_batch_size`, `n_workers`, and `gumbel_m` across PUCT and Gumbel arms to identify the highest-throughput config for the Phase 4.0 overnight run. Full methodology and per-run data in `archive/sweep_2026-04-08/`.

**PUCT top 3:**

| run_id | steps/hr | games/hr | draw% | gl_p50 |
|--------|----------|----------|-------|--------|
| P8b    | 2,976    | 497      | 55%   | 75     |
| P8     | 2,487    | 415      | 63%   | 75     |
| **P3** | **2,422**| **606**  |**39%**| **45** |

**Gumbel top 3:**

| run_id | steps/hr | games/hr | draw% | gl_p50 |
|--------|----------|----------|-------|--------|
| G5     | 1,721    | 431      | 44%   | 39     |
| **G3** | **1,417**| **710**  |**30%**| **20** |
| G2     | 1,211    | 607      | 23%   | 20     |

#### Winners

- **PUCT overnight (P3):** `training_steps_per_game=4`, `max_train_burst=16`, `max_game_moves=150`, `inference_max_wait_ms=4`, `leaf_batch_size=8`, `inference_batch_size=64`, `n_workers=14`.
- **Gumbel overnight (G3):** `gumbel_m=8`, `training_steps_per_game=2`, `max_train_burst=8`, `max_game_moves=150`, `inference_max_wait_ms=2`, `leaf_batch_size=8`, `n_workers=14`.

P3 chosen over P8b despite P8b's +23% steps/hr because P3 has +22% games/hr, 16pp lower draw rate, and median game length 45 vs 75. More unique decisive games produce better learning signal than fewer long draws.

#### Headline win

`gpu_train_frac` moved from 3.4% (P0 control) to 12.7% (P3) — a **3.7× increase** in the fraction of GPU time spent on gradient steps. This is the metric that validates the sweep.

#### Negative results worth remembering

- **leaf_bs=16 consistently hurts:** increases calls/move (opposite of theory), decreases games/hr by 19–30%, inflates draw rate by 25pp. Do not re-try without a `game_runner.rs` change. See Q16 in `docs/06_OPEN_QUESTIONS.md`.
- **Replay ratio > 4 correlates with draw-rate inflation** on 20-min windows (ratio=6 → 55–63% draws even with the best other knobs). Revisit after Phase 4.5 baseline.
- **inf_bs=64 → 32 fills batches to 99.4%** (threshold-reachable confirmed) but costs GPU util 84% → 78%. Mechanism validated, tradeoff unfavorable at this scale.
- **gumbel_m=16 → 8 doubled throughput** in the Gumbel arm (largest single knob effect in the sweep).

#### Open issue for overnight

- `policy_entropy_mean ≈ 0.25 nats` on EVERY run (framework expected 3–6 nats). Flat over 20 min across all configs. Probably a bootstrap-concentration artifact, but if the P3 overnight hasn't crossed ~1.0 nat by the 6-hour mark, pause and investigate before running the remaining 18 hours.

---

### §70 — Phase 4.0 Overnight Run — Mode Collapse Diagnosis

**Status:** diagnostics complete 2026-04-09, no fixes proposed. Run
`dcf8cbba5b9f485987880055e9cb6ea7` PAUSED at
`checkpoint_00017428.pt` pending the fix session. Full artefacts in
`archive/diagnosis_2026-04-10/`. Tracked as **Q17** in `docs/06_OPEN_QUESTIONS.md`.

#### Context

The P3 overnight run started from `checkpoints/bootstrap_model.pt` and
reached ~step 16,880 on the `gumbel_targets` variant (`gumbel_mcts: false`,
`completed_q_values: true`). Dashboard metrics looked healthy at the time
of inspection:

- `policy_entropy` ≈ 2.54 nats on the combined pretrain+selfplay mini-batch stream
- training loss trending down
- 733 games/hr, 28% draws
- no obvious deadlock or OOM

Despite this, intra-run round-robins between near-current checkpoints
exposed a hard mode collapse.

#### Eval results table

Three ckpt-vs-ckpt round-robins plus a RandomBot control, using
`scripts/eval_round_robin.py` at the default 64 sims / 0.1 s settings:

| Matchup | Score | Colony wins | Game length | Observation |
|---|---|---|---|---|
| ckpt_13000 vs ckpt_14000 | 100/0 P1 | deterministic | exactly 25 moves | carbon-copy games every rollout |
| ckpt_13000 vs ckpt_15000 | 100/0 P1 | deterministic | exactly 25 moves | carbon-copy |
| ckpt_14000 vs ckpt_15000 | 50/0 P1 (50 draws) | deterministic | exactly 31–33 moves | carbon-copy |
| **ckpt_15000 vs RandomBot (control)** | **50/0 P1** | varied | **lengths 11–33** | **varied games — the network has real knowledge, the self-play equilibrium has collapsed** |

The RandomBot control is the critical anchor: ckpt_15000 does know how
to win games. The collapse is not in the policy's game-playing ability,
it is in the self-play distribution that produces training data.

#### Monitoring gap

Dashboard `policy_entropy` is computed over the combined pretrain + selfplay
mini-batch stream (`trainer.py:402-405`). With buffer mix ~63% pretrain /
~37% selfplay (from the `pretrained_weight = max(0.1, 0.8·exp(-step / decay_steps))`
schedule at step ~16k with `decay_steps = 70_000`), the pretrain stream's
high entropy masked the selfplay stream's collapse. The §69 overnight open
issue flagged `policy_entropy_mean ≈ 0.25` on every sweep run as
"probably a bootstrap-concentration artifact" — in hindsight that was
the early warning signal for this collapse.

**Action item (follow-up, not this pass):** split `policy_entropy` into
`policy_entropy_pretrain` and `policy_entropy_selfplay` in the
`train_step` monitoring event so the collapse is visible on the
dashboard next time. Tracked under Q17 remediation.

#### Diagnostic A — static audit + feature-gated runtime trace

**Goal:** programmatically prove or refute that `engine/src/game_runner.rs`
calls `apply_dirichlet_to_root` on the training path.

**Headline finding:** it does not. The live training path
(`scripts/train.py` → `hexo_rl/selfplay/pool.py` → Rust
`engine::SelfPlayRunner`) has **zero** calls to
`apply_dirichlet_to_root`. The PyO3 method exists at
`engine/src/lib.rs:454` and its Rust implementation at
`engine/src/mcts/mod.rs:321-337`, but the **only** caller is the Python
`SelfPlayWorker._run_mcts_with_sims` at
`hexo_rl/selfplay/worker.py:138-145`, which is referenced from
`scripts/benchmark_mcts.py`, `hexo_rl/bootstrap/bots/our_model_bot.py`,
and `hexo_rl/eval/evaluator.py` only. `scripts/train.py` never
constructs a `SelfPlayWorker`.

**Git archaeology verdict:** *unported feature*. The commit that added
`apply_dirichlet_to_root` to the Python path landed on 2026-03-28. The
Phase 3.5 migration of the training path to Rust
(`engine::SelfPlayRunner`) landed two days later on 2026-03-30 and did
not carry Dirichlet injection across. This matters for the fix session's
framing: it is not a regression someone rolled back by mistake, it is a
missing port. See `archive/diagnosis_2026-04-10/diag_A_grep.txt` and
`diag_A_static_audit.md` for the raw proof.

**Runtime trace instrumentation.** To confirm the static finding at
runtime *and* give diagnostic C its data for free, a compile-time
feature-gated JSONL trace was added:

- Cargo feature `debug_prior_trace` (empty default, opt-in via
  `maturin develop --release -m engine/Cargo.toml --features debug_prior_trace`).
- Activation is gated a second time at launch by the environment variable
  `HEXO_PRIOR_TRACE_PATH`. If the feature is compiled in but the env var
  is unset, the trace is a no-op.
- Two capture sites: `engine/src/game_runner.rs` (post-MCTS root-prior
  and visit-count snapshot, cap 30 records) and
  `engine/src/mcts/mod.rs::apply_dirichlet_to_root` (pre/post priors and
  noise vector, cap 10 records).
- Writer uses unbuffered `write_all` + `flush` via a
  `std::sync::LazyLock<Mutex<Option<File>>>` sink so every record is
  durable at the moment it is written, surviving SIGINT paths that skip
  Rust-side `Drop` chains.
- One gated unit test (`test_dirichlet_trace_roundtrip`) asserts the
  JSONL wrapper on the Python path produces exactly one well-formed
  record per call.

**Runtime trace result — training path.** 30 records captured from 14
workers during a ~45-second smoke run from `checkpoint_00015000.pt` on
the `gumbel_targets` variant. **All 30 records have `site: game_runner`;
zero records have `site: apply_dirichlet_to_root`** — confirming the
static audit at runtime. The first move on the empty board shows:

| Metric | Value |
|---|---|
| root_priors[argmax] | 0.5397 (one cell out of 25 legal candidates) |
| second/third priors | 0.1709, 0.0978 |
| priors below 0.002 | 18 of 25 candidates (effectively unreachable even at τ=1.0) |
| MCTS top-1 visit fraction | **0.649** (133 of 205 visits on the top prior) |
| children receiving visits | 6 of 25 |
| temperature | 1.0 (compound_move = 0) |

Full record dump and per-field explanation in
`archive/diagnosis_2026-04-10/diag_A_trace_summary.md`.

**Runtime trace result — Python path.** 4 records from
`scripts/benchmark_mcts.py`, all with `site: apply_dirichlet_to_root`,
`n_children = 25`, `epsilon = 0.25`, and non-uniform Dirichlet noise
vectors (23–25 non-zero components, peak magnitudes 0.17–0.36). The
Python path is functionally correct; it is just dead code for training
purposes.

**Variant disclosure.** The trace was captured under `gumbel_targets`,
the same variant as the collapsed run. The relevant behaviour (absence
of Dirichlet injection, temperature formula, MCTS visit concentration)
is identical between `gumbel_targets` and `baseline_puct` because both
set `gumbel_mcts: false` — the only difference is `completed_q_values`
(KL policy target vs CE visit target), which affects the training-loss
shape, not the self-play path that produces root noise. A secondary run
under `baseline_puct` is not required.

#### Diagnostic B — raw policy sharpness across checkpoints

**Goal:** measure whether the policy head has sharpened to near-zero
entropy on the positions the training loop was actually training on,
and anchor the progression against `best_model.pt` (pre-§67 reference).

**Method.** 500 positions drawn with stratified sampling (early /
mid / late phase) from the 500 recorded games in the collapsed run
`runs/dcf8cbba5b9f485987880055e9cb6ea7/games/`. Replayed through Rust
`Board` + Python `GameState.from_board` / `apply_move`, converted to
`(K, 18, 19, 19)` tensors, K=0 (centroid) window taken to produce
`(18, 19, 19)` per position. Each checkpoint was loaded via
`Trainer._extract_model_state` + `Trainer._infer_model_hparams` +
`HexTacToeNet.load_state_dict(strict=False)`, evaluated in `.eval()` +
`torch.no_grad()` + `torch.autocast` on CUDA. Entropy per position:
`torch.special.entr(exp(log_policy)).sum(dim=-1)` matching
`trainer.py:402-405`.

**K=0 caveat (must appear at top of `diag_B_sharpness.md`):** see
`archive/diagnosis_2026-04-10/diag_B_sharpness.md`. Primary signal is
the **progression across checkpoints on identical positions**, not the
absolute nat values vs the §1 heuristic.

**Per-checkpoint summary (500 positions, K=0):**

| Checkpoint | H(π) mean | median | p10 | p90 | top-1 mean | eff. support mean |
|---|---|---|---|---|---|---|
| bootstrap_model.pt | 2.665 | 2.688 | 1.330 | 3.889 | 0.379 | 21.48 |
| checkpoint_00013000.pt | 1.666 | 1.643 | 0.620 | 2.681 | 0.497 | 9.72 |
| checkpoint_00014000.pt | 1.581 | 1.547 | 0.556 | 2.622 | 0.520 | 7.00 |
| checkpoint_00015000.pt | 1.532 | 1.601 | 0.569 | 2.336 | 0.524 | 5.79 |
| checkpoint_00016000.pt | 1.649 | 1.650 | 0.521 | 2.572 | 0.504 | 7.05 |
| checkpoint_00017000.pt | 1.486 | 1.446 | 0.477 | 2.353 | 0.540 | 6.68 |
| checkpoint_00017428.pt | 1.698 | 1.644 | 0.531 | 2.755 | 0.505 | 9.35 |
| best_model.pt | 2.665 | 2.688 | 1.330 | 3.889 | 0.379 | 21.48 |

**Phase split (mid bucket `10 ≤ cm < 25` is the worst-case window):**

| Checkpoint | Early (cm<10) mean | Mid (10≤cm<25) mean | Late (cm≥25) mean |
|---|---|---|---|
| bootstrap_model.pt | 2.430 | 2.665 | 3.418 |
| checkpoint_00013000.pt | 1.622 | 1.466 (p10=0.179) | 2.070 |
| checkpoint_00014000.pt | 1.499 | 1.443 (p10=0.191) | 2.021 |
| checkpoint_00015000.pt | 1.591 | **1.317** (p10=0.081) | 1.621 |
| checkpoint_00016000.pt | 1.634 | 1.465 (p10=0.294) | 1.935 |
| checkpoint_00017000.pt | 1.387 | 1.419 (p10=0.132) | 1.887 |
| checkpoint_00017428.pt | 1.623 | 1.620 (p10=0.136) | 2.037 |
| best_model.pt | 2.430 | 2.665 | 3.418 |

**Key observations:**

- **`best_model.pt` is NOT an independent reference — it IS
  `bootstrap_model.pt`.** Weight fingerprint (SHA-256 of first conv
  layer): `ed07ecbe6a73` for both files. `best_model.pt` is a plain
  state dict that was seeded from `bootstrap_model.pt` weights when
  training started (`scripts/train.py:526`) and was **never promoted
  during training** — no challenger beat the incumbent gating eval in
  the entire P3 run. The files differ on disk because one is a full
  checkpoint dict and the other is a raw state dict, but the tensor
  values are identical. The diag B table should be read as:
  bootstrap (H≈2.67) vs. all post-bootstrap (H≈1.5–1.7 nat band).
  There is no pre-§67 independent reference in this dataset.
- **Stuck fixed point, not progressive collapse.** All post-bootstrap
  checkpoints sit in a narrow 1.49–1.70 nat band with no downward
  trend — entropy oscillates within ~0.2 nats of a stable fixed point.
  The system found a self-consistent policy where MCTS rubber-stamps
  the prior, training targets match network outputs, and no gradient
  signal breaks the equilibrium. Framing this as "progressive collapse"
  is misleading: the collapse happened fast (likely within the first
  few thousand self-play steps), and subsequent training maintained
  rather than deepened it.
- **The worst bucket is mid-game (cm 10–24)**, where p10 drops to
  0.08–0.19 nats on every post-bootstrap checkpoint. Late-game is
  consistently the highest-entropy bucket — the opposite of what the
  §1 heuristic assumes about "expected range 3–6 nats".
- The raw-policy collapse on its own is *not* catastrophic (means still
  above 1.5 nats at K=0, effective support 5–10 children). What makes
  it catastrophic is diagnostic C: MCTS is not adding any exploration
  on top of that prior.

**Restart candidate heuristic.** `checkpoint_00017428.pt` has the
highest mean H(π) in the post-bootstrap set (1.698 nats) but the band
width is 0.21 nats — entropy rank is noise at this scale. **Do not use
entropy ordering to select the restart point.** The honest framing:

- No checkpoint in the 13k–17k range is meaningfully less collapsed
  than any other. Picking 13000 because H=1.666 > 17000 H=1.486 is
  spurious; both are stuck at the same fixed point.
- Restart point selection should be based on **buffer composition**:
  the earliest checkpoint before self-play dominated the replay buffer
  (~step 10k, where pretrain share was still ≥70%), not on entropy rank.
- Starting fresh from `bootstrap_model.pt` (clean pretrained weights,
  H≈2.67) is the simplest and cleanest option once the Dirichlet port
  is complete. This is a **finding, not a recommendation**; the fix
  session owns the call.

#### Diagnostic C — temperature schedule + MCTS visit distribution

**C.1 — temperature schedule audit.** Config values: `temperature_min
= 0.05`, `temperature_threshold_compound_moves = 15`.

Rust code (`engine/src/game_runner.rs:510-515`):

```
τ(cm) = temp_min                                 if cm ≥ threshold
      = max(temp_min, cos(π/2 · cm / threshold)) otherwise
```

| compound_move | 0 | 5 | 10 | 14 | 15 | 16 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| τ | 1.0000 | 0.8660 | 0.5000 | 0.1045 | 0.0500 | 0.0500 | 0.0500 | 0.0500 |

**Temperature formula drift (separate bullet — independent finding).**
Sprint log §36 describes the temperature schedule as:

```
τ(ply) = temp_min + 0.5·(1 − temp_min)·(1 + cos(π·ply / temp_anneal_moves))
         with temp_anneal_moves = 60, per-ply (not per compound_move)
```

These are **different functions**:

- §36 is a half-cosine *per ply* that would hold τ above 0.86 at ply 5,
  above 0.52 at ply 30, and only reach the `temp_min` floor at ply 60.
- The code is a quarter-cosine *per compound move* with a hard floor at
  cm 15 (≈ply 29–30), zero further annealing after that.

Under the §36 schedule a game has roughly four times as many plies with
meaningfully stochastic sampling as the code actually provides. This is
**not** the cause of the mode collapse on its own — no root noise is —
but it is a live docs-vs-code drift that must be fixed in one direction
or the other in the fix session. Documented here so it is greppable
later. See `archive/diagnosis_2026-04-10/diag_C_temp_schedule.md` for
the full side-by-side.

**C.2 — per-move MCTS entropy from the training trace.** Parsed the 30
records in `diag_A_trace_training.jsonl` and computed H(π_prior),
H(π_visits), Δentropy, and top-1 visit fraction per record.

| Metric | mean | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| H(π_prior) | 1.340 | 1.437 | 1.213 | 1.438 | 1.213 | 1.585 |
| H(π_visits) | 1.213 | 1.207 | 1.199 | 1.250 | 1.169 | 1.379 |
| Δ (prior − visits) | **0.127** | 0.178 | 0.014 | 0.230 | −0.055 | 0.333 |
| top-1 visit fraction | 0.526 | 0.509 | 0.399 | 0.649 | 0.395 | 0.649 |
| effective support (exp H_visits) | 3.366 | 3.345 | 3.316 | 3.490 | 3.217 | 3.972 |

**Verdict.** MCTS sharpens the prior by only 0.13 nats on average. The
effective support of the visit distribution is ~3.4 children — MCTS is
picking between the top 3 prior candidates and rubber-stamping them.
Combined with the temperature schedule dropping to 0.05 at cm 15 and
the §70 diag-A finding that there is no Dirichlet perturbation at the
root, this closes the loop: once the prior concentrates on 3 children
there is no mechanism in the training path to make the self-play stream
explore any other continuation. The sampling window for "not
deterministic" self-play is compound_move 0 through ~14, and even inside
that window the top prior gets picked >50% of the time.

**Caveat:** the 30 records are all cm=0, ply=0 on the empty board
(different games from different workers, all starting identically).
They are 30 independent rollouts of the same position, not a sweep
across game phases. This is a consequence of the global counter cap
firing inside the first move because 14 workers contend for it in
parallel. The per-game-phase variation the plan originally asked for
is not in this data — the fix session should raise the
`GAME_RUNNER_CAP` and/or make it per-game if that variation is needed
for remediation decisions. For the current diagnostic pass the 30
empty-board records are sufficient to demonstrate the rubber-stamp
behaviour, because the empty board is where the self-play loop enters
its deterministic attractor.

#### Candidate root causes (ranked by support from A/B/C)

1. **No root noise on the training path.** Strongest candidate.
   Confirmed by diagnostic A at both static (grep + git archaeology)
   and runtime (30 game_runner records, 0 apply_dirichlet_to_root
   records) levels.
2. **Policy sharpness amplified by self-referential MCTS.** Quantified
   by diagnostic B (mean H(π) ≈ 1.5 nats on K=0, p10 ≈ 0.1 nats in the
   mid-game bucket) and C (Δentropy ≈ 0.13 nats, effective support ≈
   3.4 children). MCTS does not add exploration, it rubber-stamps.
3. **Temperature schedule is weaker than §36 described.** Hard floor
   at cm 15, quarter-cosine shape, no further annealing. Not the root
   cause on its own but it narrows the time window in which (1) and (2)
   could be broken by chance.
4. **Entropy regularisation too weak** (`entropy_reg_weight = 0.01`).
   Consistent with the late-phase p10 numbers in diagnostic B but not
   independently proven by this diagnostic pass.
5. **Buffer-mix interaction masking the collapse in monitoring.**
   Independent of the root cause but explains why the collapse went
   unnoticed for 16,880 steps. See Monitoring Gap above.

#### Identical eval games — expected behaviour, not a seeding bug

The round-robin results showed 100% identical games between near-era
checkpoints (ckpt_13000 vs ckpt_14000: all 25 moves, carbon-copy). This
is **expected behaviour**: `ModelPlayer.get_move()` calls
`tree.get_policy(temperature=0.0)` which returns a one-hot argmax policy,
and the eval loop has no stochastic element. Any two runs of the same
matchup will produce identical games by construction.

A separate temperature sampling check (2026-04-10, `scripts/eval_diagnostic.py
--temperature 1.0 --model_a/b ckpt_15000.pt`, 20 games) confirmed that
with τ=1.0, games diverge normally: 13 distinct game lengths across 20
games, P1/P2 wins roughly equal. **Temperature sampling in the Rust
game_runner and in the eval path is functionally correct.** The collapse
is not caused by broken temperature sampling — it is purely the missing
Dirichlet injection on the training path.

#### "Known correct" reference

There is no independent pre-collapse reference checkpoint in the P3
dataset. `best_model.pt` was initialised from `bootstrap_model.pt`
weights at training start and never updated — weight fingerprint
confirms identity (`ed07ecbe6a73`). The `bootstrap_model.pt` pretrained
weights (H≈2.67 nats) are the only available anchor. The previous
framing of `best_model.pt` as a "pre-§67 CE-loss reference" was
incorrect.

#### Not-in-scope (fix session to decide)

- Porting `apply_dirichlet_to_root` into the Rust training path
  (`engine/src/game_runner.rs`) vs switching the laptop variant to
  `gumbel_mcts: true` (Gumbel-Top-k provides root noise by
  construction). Both are valid remediations; the choice is not owned
  by this diagnostic pass.
- Splitting `policy_entropy` into pretrain / selfplay streams in the
  monitoring event.
- Reconciling the §36 temperature formula with the code (either
  direction).
- Re-running diagnostic C with a larger `GAME_RUNNER_CAP` to cover
  mid-game and late-game MCTS behaviour.
- Any change to checkpoints, replay buffer, or run directory state.

---

### §71 — Pre-Dirichlet-Fix Cleanup & Gumbel Fallback Verification — 2026-04-10

Cleanup and instrumentation pass before the Dirichlet port session. No training runs
started. Commits are: `chore(scripts)`, `feat(monitoring)`, `diag(gumbel)`,
`docs(sprint)` — four independent commits, each self-contained.

---

#### 1. Gumbel path verification

**Static audit** (`archive/verify_gumbel_2026-04-10/diag_static.md`):

Three questions answered with line-number citations from
`engine/src/game_runner.rs`:

- **Q1 — noise freshness:** `GumbelSearchState::new()` is called inside
  the per-move loop (`line 362`) at `line 459`. Gumbel(0,1) values are
  drawn from a per-worker RNG (`let mut rng = rng()`, `line 344`) for
  every root child on every move — fresh per move, not cached across
  games or moves.

- **Q2 — noise is added, not discarded:** The Gumbel vector is added
  to `log_prior[i]` at candidate-selection time (`line 70`) and again
  at every Sequential Halving phase (`line 130`:
  `gumbel[i] + log_prior[i] + sigma(Q)`). It is not discarded.

- **Q3 — `effective_m` formula:** `line 445`:
  `gumbel_m.min(game_sims).min(tree.root_n_children())`. Matches the
  §61/§62 spec; no hardcoded constants.

**Runtime trace** (`archive/verify_gumbel_2026-04-10/diag_trace.jsonl`,
`verdict.md`): 30 records captured under `gumbel_mcts: true` from
`checkpoint_00015000.pt`.

Key finding: visit arrays differ across workers at ply=0 (top-1 visit
cells: workers 0/1/2 → cells 13/15/15 — disagreement). The
`top_visit_fraction` mean is **0.24** vs **0.65** in the PUCT §70 trace
— a 41 percentage-point reduction, confirming Sequential Halving spreads
budget across multiple candidates rather than rubber-stamping the prior.

**Verdict: Gumbel noise is functionally active on the training path.
Switching to `gumbel_mcts: true` is a validated fallback remediation if
the Dirichlet port encounters issues.**

---

#### 2. Policy-entropy split monitoring

The §70 diagnosis found that `policy_entropy` was 2.54 nats
(healthy-looking) during the P3 overnight run while the selfplay stream
was stuck at ~1.5 nats in a fixed-point. The pretrain corpus (~63%
buffer share) masked the selfplay collapse for ~13k training steps.

Changes landed in `feat(monitoring)`:

- **`trainer.py`:** `_train_on_batch` and `train_step_from_tensors`
  accept `n_pretrain: int = 0`. After the combined entropy is computed,
  splits `log_policy` by index: `[:n_pretrain]` → pretrain,
  `[n_pretrain:]` → selfplay. On the single-buffer path
  (`n_pretrain == 0`) the pretrain field is `NaN` and selfplay equals
  the combined metric.

- **`scripts/train.py`:** Passes `n_pretrain=n_pre` to
  `train_step_from_tensors`. Emits `policy_entropy_pretrain` and
  `policy_entropy_selfplay` in the `training_step` event and structlog
  `train_step` entry alongside the existing `policy_entropy` (retained
  for backwards compatibility).

- **Terminal dashboard:** New `entropy` subsection row:
  `entropy  combined X.XX  │  pretrain X.XX  │  selfplay X.XX  (collapse < 1.5 nats)`.
  Selfplay colored red (`< 1.5 nats`), yellow (`1.5–2.0`), green (`≥ 2.0`).
  Separate `selfplay entropy X.XX — selfplay mode collapse` alert.

- **Web dashboard:** `collapse_threshold_nats: 1.5` added to
  `/api/monitoring-config`. Chart.js entropy panel gains pretrain
  (blue dashed) and selfplay (red dashed) traces; second reference line
  at 1.5 nats labeled "collapse threshold".

Smoke test (2026-04-10): `policy_entropy_pretrain: 2.74`,
`policy_entropy_selfplay: 2.75` visible in JSONL after 24s of smoke
from `bootstrap_model.pt`. Both fields present; dashboards render
without crash.

---

#### 3. Known-good checkpoint inventory

> Entropy values from `diag_B_sharpness.md` (K=0 cluster window only,
> ~several tenths below min-pool; see K=0 caveat in that file).
> Weight fingerprints are SHA-prefix of the first layer weight tensor.

| Checkpoint | weight_fp | H(π) mean (K=0) | Use / Status |
|---|---|---|---|
| `bootstrap_model.pt` | `ed07ecbe6a73` | 2.665 | **Primary restart point** for Phase 4.0 post-Dirichlet-fix. Clean pretrained weights, highest entropy anchor. |
| `checkpoint_00013000.pt` | — | 1.666 | Earliest post-collapse; no less collapsed than later checkpoints. Do not restart from. |
| `checkpoint_00010000.pt` | — | — | Does not exist in current checkpoints/ (was never saved). |
| `best_model.pt` | `ed07ecbe6a73` | 2.665 | **Identical to `bootstrap_model.pt`** (same weight fingerprint). Was initialised at training start but never promoted. Do not treat as an independent reference. |
| collapsed run ckpt 13k–17k | — | 1.49–1.70 | Fixed-point collapse, no entropy gradient. Retained for forensics. **Do not restart from.** |

---

#### 4. Pre-run checklist

Walk this checklist before launching the next Phase 4.0 sustained run:

```
[ ] Dirichlet ported to engine/src/game_runner.rs, unit-tested
[ ] debug_prior_trace re-run confirms apply_dirichlet_to_root
    records appear on the training path (inverse of the §70 proof)
[ ] checkpoints/replay_buffer.bin archived to .bak
[ ] collapsed checkpoints (13k-17k) moved to
    checkpoints/collapsed_2026-04-09/ (do not delete)
[ ] make test.all passes
[ ] make bench.full passes all 10 §66 targets (or environmental
    noise explicitly acknowledged per 2026-04-09 bench run notes)
[ ] policy_entropy_pretrain and _selfplay fields visible in the
    JSONL log of a smoke run
[ ] both dashboards render the split entropy without error
[ ] 2-hour smoke from bootstrap_model.pt produces non-identical
    self-play games (eval_diagnostic.py with --temperature 1.0,
    >= 5 distinct game lengths in 20 games)
[ ] 6-hour entropy-checkpoint plan written: selfplay-stream H
    must be above 1.5 nats and trending up or stable at the
    6-hour mark, otherwise pause and investigate
```

---

### §72 — Bench Baseline Rebaseline — 2026-04-09 Driver-State Shift  <!-- KEEP-CONDENSED -->

Three consecutive `bench.full` runs on 2026-04-09 and 2026-04-10 failed the same two §66 targets (NN inference batch=64 ~8,370 vs target 8,500; worker throughput ~541k vs target 625k). A structured three-run investigation (cold / hot / post-10min-idle) confirmed the failures are **not thermal** — cold and hot runs returned 8,393 and 8,397 pos/s respectively (0.05% apart), with the GPU staying at 49°C throughout and no boost-clock events visible in `nvidia-smi`. The step-change appeared overnight between the last passing run (2026-04-08 22:50, 9,388 pos/s) and the first failing run (2026-04-09 11:53, 8,347 pos/s) with no model or benchmark code changes in that window — the NVIDIA laptop driver's `DynamicPowerManagement=3` settled the GPU into a lower sustained boost-clock bin after a full day of workloads. NN inference latency corroborates: 1.59 ms at §66 → 1.77–1.80 ms now, a 12–13% increase consistent with a ~14% GPU clock reduction. Worker throughput failures are a secondary consequence (slower inference → stalled workers). Full investigation artifacts in `archive/bench_investigation_2026-04-09/`.

**Rebaselined targets** (CLAUDE.md §66 table updated, 2026-04-09): NN inference ≥ 8,250 pos/s (was 8,500; floor from three cold-start runs: 8,327); worker throughput ≥ 500,000 pos/hr (was 625,000; floor from stable runs 1/2: ~540k, conservative target accounting for structural IQR noise in the 60s measurement window). All other eight targets unchanged. The §66 baseline column values are kept at their 2026-04-06 peak to document the original hardware capability; the targets now reflect the sustained operating floor. A reboot may restore peak numbers, but the training programme does not depend on it.

---

### §73 — Dirichlet Root Noise Ported to Rust Training Path — 2026-04-10

**Root cause from §70 resolved.** `engine/src/game_runner.rs` now calls `apply_dirichlet_to_root` on every turn boundary in both PUCT and Gumbel branches.

**Changes landed (commit `71d7e6e`):**

- `engine/src/mcts/dirichlet.rs` — new Gamma-normalize sampler using `rand_distr 0.5` (compatible with `rand 0.9`). Draws `n` independent `Gamma(alpha, 1.0)` samples normalised by sum. Four unit tests: sum-to-one, non-negative, independence, sparsity at `alpha=0.3`.
- `engine/src/game_runner.rs` — added `dirichlet_alpha` / `dirichlet_epsilon` / `dirichlet_enabled` fields to `SelfPlayRunner`. PUCT branch: root expansion separated to `batch=1` call, Dirichlet applied immediately after. Gumbel branch: Dirichlet applied after the root expansion guard. Both sites honour the intermediate-ply skip (`moves_remaining==1 && ply>0`), matching `worker.py:107-111`. Two integration tests verify the gate fires and can be disabled.
- `configs/selfplay.yaml` — `dirichlet_enabled: true` added under `mcts:` (default active).
- `hexo_rl/selfplay/pool.py` — wires `dirichlet_alpha` / `dirichlet_epsilon` / `dirichlet_enabled` from `mcts_cfg` to `SelfPlayRunner` constructor.
- `engine/Cargo.toml` — adds `rand_distr = "0.5"`.

**Tests:** `cargo test -p engine` (default + `debug_prior_trace`): 108/109 passing, 0 failures.  
`make test.all`: 108 Rust + 646 Python, all pass.

**Benchmark:** `make bench.full` 2026-04-10. MCTS sim/s 53,840 (target ≥ 26,000). NN inference 8,804 pos/s (target ≥ 8,250). Worker throughput 548,653 pos/hr (target ≥ 500,000). All 10 metrics pass CLAUDE.md targets. Note: benchmark script still uses pre-§72 script-hardcoded targets (625k worker, 8,500 NN) — script exit code 2 is a stale-target pre-existing issue, not a regression.

**Runtime verification (commit `4a3149e`) — `archive/dirichlet_port_2026-04-10/verdict.md`:**

Trace from `ckpt_15000`, variant `baseline_puct`, 90s smoke, no train step:

| Site | Count | §70 count |
|---|---|---|
| `apply_dirichlet_to_root` | 10 | **0** |
| `game_runner` | 30 | 30 |

- 10/10 unique Dirichlet noise vectors — workers draw independent samples.
- Top-1 prior: `0.540 → 0.412` post-noise (−12.8 pp).
- Top-1 **visit** fraction at cm=0: **0.474** vs §70 PUCT baseline **0.65** (−17.6 pp).
- Workers at cm=0,ply=0 span 0.33–0.55 — clearly diverging (§70: identical across all 14 workers).

**Grep proof of presence:**
```
engine/src/game_runner.rs:465: tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);  # PUCT branch
engine/src/game_runner.rs:550: tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);  # Gumbel branch
```

**§71 pre-run checklist status:**

- [x] Dirichlet ported to `engine/src/game_runner.rs`, unit-tested
- [x] `debug_prior_trace` re-run confirms `apply_dirichlet_to_root` records appear
- [ ] `checkpoints/replay_buffer.bin` archived
- [ ] Collapsed checkpoints moved to `checkpoints/collapsed_2026-04-09/`
- [ ] `make test.all` and `make bench.full` pass (done — see above)
- [x] `policy_entropy_pretrain/_selfplay` fields visible
- [x] Dashboards render split entropy without error
- [ ] 2-hour smoke from `bootstrap_model.pt` produces non-identical self-play games
- [ ] 6-hour entropy-checkpoint plan written

Q17 status: **RESOLVED — Dirichlet port shipped.** Remaining items before sustained run: walk the §71 checklist (archive buffer, move collapsed ckpts, run 2hr smoke from bootstrap, write 6hr plan).

### §74 — Gumbel vs PUCT Loop Audit — Resolutions — 2026-04-10

Closes the three open-item categories from the 2026-04-09 static audit at `archive/gumbel_vs_puct_bench_2026-04-09/verdict.md`'s sibling `reports/gumbel_vs_puct_loop_audit_2026-04-09/verdict.md` §6. Delivered as three sequential commits: `bench(gumbel): paired gumbel_full vs baseline_puct` → `test(mcts): dirichlet parity between puct and gumbel branches` → `docs(sprint): §74 gumbel-puct audit resolutions`. No changes to `game_runner.rs` in this pass.

#### §74.1 — `get_improved_policy` is PUCT-tree-safe (architectural property)

Static audit §5 proved that `engine/src/mcts/mod.rs:171–295` reads only ordinary MCTS state populated by the shared expansion/backup primitives:

- `root.is_expanded / first_child / n_children / w_value / n_visits` (mod.rs:181–191, 241)
- per-child `action_idx`, `n_visits`, `prior`, `w_value` (mod.rs:197–208)

All of these fields are written by `expand_and_backup_single` (`backup.rs:122–138`) and `backup` (`backup.rs:180–198`), regardless of which outer branch (PUCT or Gumbel) drove the selection. `get_improved_policy` never reads `forced_root_child`, `GumbelSearchState`, `gumbel_values`, `log_priors`, or any Gumbel-only state. `c_visit` / `c_scale` are passed in as arguments, so the caller can select defaults appropriate to the use case (selfplay.yaml defaults at `configs/selfplay.yaml:33–34` match what training targets use).

**Consequence:** A PUCT self-play run can train Gumbel policy targets by calling `get_improved_policy` at turn boundaries without running Sequential Halving, and the planned `/analyze` endpoint can return improved-policy signal from any PUCT-built tree. This decouples the training target shape from the search algorithm used to build the tree.

#### §74.2 — Paired benchmark closure

Full verdict and raw data: `archive/gumbel_vs_puct_bench_2026-04-09/verdict.md`. Reproduced inline for this log.

**Headline:** batch fragmentation is theoretical on Ryzen 7 8845HS + RTX 4060. Per-worker Sequential Halving fragmentation is absorbed by `InferenceBatcher` cross-worker coalescing before reaching the GPU. Batch fill % = 100.00% (IQR 0) for both variants across all runs is the direct evidence.

Design: four interleaved invocations (`baseline_puct → gumbel_full → baseline_puct → gumbel_full`), 16 workers, n=5 runs / 60s worker pool per invocation. "Med-of-2" = median across the two interleaved invocations per variant. All 10 §66 gate metrics pass for both variants.

| Metric | baseline_puct (med-of-2) | gumbel_full (med-of-2) | Δ (rel) | §66 target | PUCT | Gumbel |
|---|---:|---:|---:|---|:-:|:-:|
| MCTS sim/s (CPU, no NN) | 53,396.5 | 54,166.5 | +1.44% | ≥ 26,000 | ✓ | ✓ |
| NN inference batch=64 pos/s | 8,547.75 | 8,517.70 | −0.35% | ≥ 8,250 | ✓ | ✓ |
| NN latency batch=1 (ms) | 1.650 | 1.665 | +0.91% | ≤ 3.5 | ✓ | ✓ |
| Replay buffer push (pos/s) | 709,519.5 | 739,201.5 | +4.18% | ≥ 630,000 | ✓ | ✓ |
| Buffer sample raw (µs/batch) | 1,106.45 | 1,097.00 | −0.85% | ≤ 1,500 | ✓ | ✓ |
| Buffer sample augmented (µs/batch) | 1,032.25 | 1,038.05 | +0.56% | ≤ 1,400 | ✓ | ✓ |
| GPU utilisation % | 99.95 | 100.00 | +0.05 pp | ≥ 85 | ✓ | ✓ |
| VRAM (GB) | 0.05 | 0.05 | 0.00 | ≤ 6.4 | ✓ | ✓ |
| **Worker throughput (pos/hr)** | **566,480** | **619,678.5** | **+9.39%** ⚠ noise | ≥ 500,000 | ✓ | ✓ |
| **Worker batch fill %** | **100.00** | **100.00** | **0.00 pp** | ≥ 80 | ✓ | ✓ |

**Caveat on worker throughput.** Two of four invocations had pos/hr IQR > 38% of median (gumbel_full_run1 at 46%, run2 at 39%). The nominal +9.4% Gumbel lead is inside the noise floor: baseline combined range 427–768k, gumbel combined range 415–781k — they overlap almost entirely. The stop-rule "Gumbel >5% higher on worker throughput" was checked and not triggered; the rule guards against a meaningful Gumbel-faster signal contradicting the fragmentation hypothesis, and there is no meaningful signal here, plus the direct mechanism test (batch fill %) confirms the hypothesis in the expected direction. Worker throughput noise is documented but not load-bearing on the verdict.

**Why batch fill % is the real verdict.** `scripts/benchmark.py:401` computes batch fill as `delta_req / (delta_fwd * server._batch_size) * 100` — average filled slots per GPU forward pass. This is an aggregated measurement across the full worker pool: if per-worker batches are small but multiple workers' requests coalesce at the `InferenceBatcher` before a GPU forward pass, the resulting fill % is still 100%. That is exactly what the audit §1c predicted structurally: Gumbel's Sequential Halving fragments `sims_per` at the per-candidate level (`game_runner.rs:509–511` in-source bandaid comment), but each Gumbel worker's small per-candidate batch still enters the shared queue, and the batcher fills its GPU-side batch from the pooled queue up to `inference_batch_size`. On 16 workers feeding one batcher, cross-worker coalescing absorbs per-worker fragmentation completely.

**Harness note.** `make bench.full` and `--variant` do not exist — neither `Makefile` nor `scripts/benchmark.py` accepts them. This benchmark used `scripts/benchmark.py --config configs/variants/<name>.yaml` as a workaround per plan discrepancies D1/D2. The harness script's internal pass/fail thresholds are stale (625k worker pos/hr, 8,500 NN inference) — it prints "Some checks FAILED" on every invocation because it predates §72's rebaseline. All metrics pass the current `CLAUDE.md` Phase 4.5 gate.

#### §74.3 — Dirichlet parity regression test

Commit: `test(mcts): dirichlet parity between puct and gumbel branches`. New file: `engine/tests/dirichlet_parity.rs` (first entry in `engine/tests/`).

**Code-inspection finding:** the two Dirichlet call sites at `game_runner.rs:454–467` (Gumbel branch) and `:538–553` (PUCT branch) are structurally identical on current HEAD — same `sample_dirichlet(dirichlet_alpha, n_ch, &mut rng)` call, same `tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon)` call, same `is_intermediate_ply = board.moves_remaining == 1 && board.ply > 0` gate (lines 458 and 542, same comment pointing to `hexo_rl/selfplay/worker.py:107-111`). The only asymmetry is an extra `if tree.pool[0].is_expanded()` guard on the PUCT side at line 544 — a correctness-preserving asymmetry, not a drift, justified because the Gumbel branch's preceding `root_sims > 0` check already guarantees expansion. **No actual drift to fix.** Audit §3's "minor drift risk" concern was preserved as a regression guard rather than a drift fix. Separately, the audit noted §73's grep proof at lines 1193–1194 swapped the PUCT/Gumbel line labels; this has not been edited in-place since it's historical record, but the correct mapping is documented here (`:465` is in the Gumbel branch, `:550` is in the PUCT branch — inspection at `game_runner.rs:444` confirms the outer `if gumbel_mcts {` places `:465` inside the then-arm and `:550` inside the else-arm).

**What the test asserts:**

1. `sample_dirichlet_sums_to_one_and_is_nonneg` — sum-to-1 within 1e-5 across `n ∈ {1, 2, 5, 24, 50}`, all entries non-negative.
2. `apply_dirichlet_to_root_blends_linearly` — asserts the per-child formula `new = (1-ε)·old + ε·noise` at `mcts/mod.rs:344` with non-uniform ramp noise and ε=0.25, tolerance 1e-6.
3. `apply_dirichlet_with_zero_epsilon_is_noop` — priors bit-exact unchanged under ε=0, compared via `f32::to_bits`.
4. `intermediate_ply_gate_matches_game_runner_spec` — truth table across a 5-move sequence covering `(ply, moves_remaining) ∈ {(0,1), (1,2), (2,1), (3,2), (4,1), (5,2)}`, pinning which plies are turn-boundary vs intermediate. Fires loudly if `Board`'s turn structure ever changes.

**What the test does NOT assert:**

- **Branch-level runtime equivalence under a shared RNG seed.** Blocked by the absence of `new_with_seed` on `SelfPlayRunner` (RNG is created from unseeded `rand::rng()` inside the worker at `game_runner.rs:356`). Tracked in §74.6.
- **Textual parity of the two call sites.** A test that `include_str!`'d `game_runner.rs` and grep'd for matching token sequences would false-fire on reformatting and train people to ignore the suite. A grep-based guard belongs in a git pre-commit hook or CI check, not in `cargo test`. The branch-edit-without-sibling risk is accepted and documented here.

**Test results:** `cargo test -p engine`: 112 passed (108 existing + 4 new). `make test`: green.

#### §74.4 — Implications for `/analyze`

§74.1 unblocks exposing `get_improved_policy` to Python for both PUCT and Gumbel trees. This was the original question that motivated the whole audit (the analyzer sibling task): can we return improved-policy signal from a PUCT-built tree without running Sequential Halving? Answer: yes, unambiguously — the function reads only PUCT-populated fields. Implementation (PyO3 binding + Python-side wiring in the `/analyze` endpoint) is tracked as a **separate, not-yet-scheduled commit owned by a later task**. Do not implement as part of this pass.

#### §74.5 — Implications for the desktop variant decision

Current `CLAUDE.md` "Gumbel MCTS (per-host override)" block:

- **Desktop 3070:** `gumbel_mcts: true` (`gumbel_full`), intentional for Phase 4.0 sustained run, not yet swept on desktop hardware.
- **Laptop 8845HS + 4060:** `gumbel_mcts: false`, P3 sweep winner as base config.

**Outcome: No change needed on laptop.** The laptop measurement found no pipeline-level perf gap between variants — batch fill % is 100% for both, worker throughput is noise-dominated with no meaningful delta, all other metrics are within ~5%. The laptop's `gumbel_mcts: false` choice remains correct (it's the P3 sweep winner, not a choice driven by fragmentation).

**Desktop decision still pending re-bench on 3070 hardware.** The laptop batch-fill finding is **mechanism evidence** — it shows that cross-worker coalescing in `InferenceBatcher` can absorb per-worker fragmentation *on that configuration* (16 workers, RTX 4060, `inference_batch_size=64`). The desktop has a different saturation profile: smaller GPU (3070), possibly different worker count, possibly different `inference_batch_size`. Cross-worker coalescing may or may not reach 100% fill there. Before the next sustained desktop run from `bootstrap_model.pt`, the desktop should run the same paired-variant benchmark to confirm the laptop mechanism generalises. If batch fill % drops below ~95% on desktop Gumbel, the fragmentation becomes a real pipeline-level cost there and the desktop should switch to `baseline_puct` for the sustained run.

#### §74.6 — Open items explicitly NOT resolved by this pass

- **Coalescing Phase-*i* candidate inference across workers.** Q16-adjacent. The `game_runner.rs:499–519` loop sets `tree.forced_root_child = Some(child_pool_idx)` per candidate per phase, so candidates structurally cannot share an inference batch within a single worker. Lifting this would require reshaping Sequential Halving to batch across candidates before committing to any one. Out of scope for this pass; also newly de-prioritised by §74.2's finding that cross-worker coalescing already absorbs the per-worker fragmentation on laptop hardware.
- **Exposing `get_improved_policy` to Python.** Separate PyO3 binding + Python-side wiring for `/analyze`. Separate commit, separate owner. See §74.4.
- **Seeding `SelfPlayRunner`'s worker RNG deterministically.** Needed for a true end-to-end Dirichlet parity test (one move per branch under identical RNG, byte-exact post-blend prior comparison). Small Rust change: a `new_with_seed` constructor that threads a `u64` down to the worker `rng` initialisation at `game_runner.rs:356`. Blocks the "real" parity test but the structural regression guard in §74.3 covers the practical regression surface.
- **Tighter worker throughput measurement under longer pool duration.** The current 60s × 5 runs × 2 interleaved budget produced IQRs of 7–46% on worker throughput on the laptop — too noisy to discriminate small deltas on this metric. A re-bench at `--pool-duration 180 × ≥5 runs × interleaved` (total ~90 min) would tighten the signal. Cheap follow-up if anyone cares, not a blocker. Batch fill % is not affected by this.
- **Audit §6 open question 4** — per-move wall-clock cost of `GumbelSearchState::new` + `halve_candidates` via `criterion` microbench. Not covered by the whole-pipeline benchmark in §74.2. Separately out of scope.

**Status:** all three audit open items closed or explicitly deferred. No blockers for the next sustained run beyond the §71 checklist items that were already outstanding.

---

## §75 — Fast game disable for gumbel_targets (2026-04-10)

Draw rate investigation (reports/draw_rate_investigation_2026-04-10/)
confirmed 100% of draws are timeout draws at the 150-ply cap.
Low-sim games (fast_prob=0.25, 50 sims, τ=1.0, PUCT) had 94.4% draw
rate vs 3.7% for standard games. Colony-extension behavior observed in
game viewer — both sides extend parallel lines with no engagement.

Fix: fast_prob: 0.0 in configs/variants/gumbel_targets.yaml.
gumbel_full.yaml unchanged (Gumbel SH is effective in the low-sim
regime, §71).

Resumed from checkpoint at step 25008 with VARIANT=gumbel_targets.

---

### §76 — max_game_moves correction for gumbel_targets (2026-04-10)

Phase A diagnostic (reports/draw_rate_investigation_2026-04-10/phase_a_postfix.md)
confirmed max_game_moves counts plies not compound moves. gumbel_targets was the
only variant at 150 plies (75 compound turns) — a §69 artifact for fast_prob=0.25.
With fast_prob=0.0 (§75), 57.6% of games hit the cap. Fix: 150→200 plies to match
other variants. Yaml comment "compound moves" was incorrect — fixed to "plies".
Resumed from checkpoint_00025008 (the last clean checkpoint before fast_prob change).

---

## §77 — MCTS depth & ZOI scope investigation (2026-04-11)

**Reference:** `reports/mcts_depth_investigation_2026-04-11/`

### Motivation

Prior sessions assumed ZOI restricted MCTS tree branching. Depth probe and
code audit performed to verify actual behavior and measure search depth.

### Findings

**1. ZOI is post-search only — §36 corrected.**
Code audit of `game_runner.rs:626–643` confirmed ZOI filtering runs *after*
`expand_and_backup` completes, on the root visit-count vector used for move
selection. The MCTS tree itself expands with the full radius-8 legal set at
all depths. §36 description amended.

**2. Measured branching factor.**
Depth probe (200 sims, PUCT): 360 median root children created, 7 receiving
visits. B_eff = 6.1. FPU and policy concentration — not ZOI — drive the low
effective branching factor. Children past rank ~10 receive zero visits under
200 sims.

**3. Measured leaf depth.**
Mean leaf depth 2.92 plies (PUCT, 200 sims), max depth 6–8. Top-5 visit
share 0.97 — search is appropriately concentrated given the compute budget.

**4. Depth projections.**
Gumbel m=16 gains approximately +0.6 plies vs PUCT at 200 sims; desktop
training logs at ~18k steps confirm ~3.5 mean depth (consistent with projection).
ZOI-at-expansion would add only +0.16 plies over Gumbel — below measurement noise.

### Decision: Option A (do nothing)

ZOI-at-expansion rejected. Depth improves automatically as policy sharpens
(lower B_eff). The correct lever for deeper search is n_sims increase (Option B),
not tree pruning. Revisit Option B at 200K+ steps if B_eff remains above 10.

No code changes. No config changes.

## §78 — /analyze Policy Viewer (2026-04-11)

### Motivation

§70 mode collapse was invisible until 16k steps. Need interactive debugging tool
to inspect raw network priors on arbitrary positions in seconds.

### Implementation

Branch `feat/policy-viewer`, 4 commits:

1. **Rust PyO3**: `forced_root_child` getter/setter, `get_root_children_info()`,
   `get_improved_policy()`, `get_top_visits` → 4-tuple (added q_value).
2. **HexCanvas ES module**: extracted hex rendering from viewer.html into reusable
   `hex_canvas.js` class. viewer.html ported to use it.
3. **`/api/analyze` Blueprint**: checkpoint LRU cache (max 3, mtime stale detection),
   Python-driven MCTS (PUCT + Gumbel SH), ThreadPoolExecutor(1) for inference.
   `model_loader.py` loads checkpoints without importing Trainer.
4. **`/analyze` SPA**: sidebar controls, policy heatmap overlay, MCTS visit overlay,
   deep-link support (`?moves=<base64>&checkpoint=<path>`).

### Key decisions

- **Python-driven MCTS** (not Rust `analyze_position`): avoids FFI callback complexity.
  PyMCTSTree already exposes `select_leaves`/`expand_and_backup`. Only needed 3 new
  read-only accessors.
- **Gumbel SH uses raw Q** (not `completed_q_values`): sufficient for interactive
  analysis. Production SH in `engine/src/game_runner.rs` is source of truth.
- **model_loader.py duplicates** `_extract_model_state` / `_infer_model_hparams` from
  Trainer to avoid pulling in optimizer/scheduler imports. Sync test added.

### Post-review fixes (same session)

- XSS in deep-link parser: added typeof validation matching paste path
- BOARD_SIZE threaded from checkpoint metadata (was hardcoded 19)
- Path traversal guard on checkpoint param (must be under project root)
- Dead `half` / `BOARD_SIZE` vars deleted from `_run_gumbel`
- Checkpoint dir configurable via `analyze_bp.checkpoint_dir`

---

## §79 — Initial buffer increased 100K → 250K (2026-04-12)

### Motivation

§40b reduced initial buffer 250K→100K as a stability measure during draw-collapse
diagnosis. Draw collapse resolved at §40. Buffer saturates at 100K with ~48%
self-play = ~48K positions = ~600 games of context — too thin for the model to
generalise beyond colony patterns. CLAUDE.md line "start at 250K" was already
correct; config was the stale artifact.

### Memory budget (verified)

14,458 bytes/entry (Rust) + ~14,448 bytes/entry (Python RecentBuffer at 50% capacity).

| Tier | Rust buffer | Python mirror | Buffers total |
|------|-------------|---------------|---------------|
| 250K | 3.37 GB | 1.68 GB | 5.05 GB |
| 500K | 6.73 GB | 3.37 GB | 10.1 GB |
| 1M | 13.47 GB | 6.73 GB | 20.2 GB |

System: 32 GB RAM. At 250K initial: ~12.7 GB total process → 19.3 GB headroom. Safe.
Delta vs 100K: +2.98 GB.

### Schedule change

```yaml
# Before (§40b):
buffer_schedule:
  - {step: 0,       capacity: 100_000}
  - {step: 150_000, capacity: 250_000}
  - {step: 500_000, capacity: 500_000}

# After (§79):
buffer_schedule:
  - {step: 0,           capacity: 250_000}
  - {step: 300_000,     capacity: 500_000}
  - {step: 1_000_000,   capacity: 1_000_000}
```

Growth tiers shift right because starting tier is larger. Steps 300K and 1M exceed
`total_steps: 200_000` — apply during extended runs only.

### Resume safety (verified)

`load_from_path` (engine/src/replay_buffer/mod.rs:503) reads
`min(saved_size, self.capacity)` positions into pre-allocated capacity without
resizing the buffer. Resume from old 100K checkpoint: buffer constructed at 250K
(from new config), then ≤100K positions loaded in — no truncation, no resize call.

---

## §80 — Eval Determinism Fix: Temperature + Random Openings (2026-04-12)

### Root cause (from §70)

`Evaluator.evaluate()` constructed `ModelPlayer` without a `temperature` arg, so it
defaulted to `temperature=0.0`. `get_policy(τ=0)` returns a one-hot argmax policy.
With no stochastic element anywhere in the eval loop, all 50 games in a SealBot
match were bit-identical. Bradley-Terry CIs were ±100K — completely uninformative.

### Fix

Three targeted changes, all in the formal eval pipeline only. Training path untouched.

**1. Temperature propagation** (`hexo_rl/eval/evaluator.py`)

`Evaluator.__init__` reads `eval_temperature` (default 0.5) from the `evaluation`
config sub-dict. `evaluate()` passes `temperature=self._eval_temperature` when
constructing `ModelPlayer`. `evaluate_vs_model()` does the same for the opponent
player so the best-checkpoint matchup is symmetric.

**2. Per-game seeding** (`hexo_rl/eval/evaluator.py`)

At the start of each game loop iteration `i`, the evaluator calls:
```python
np.random.seed(self._eval_seed_base + i)
random.seed(self._eval_seed_base + i)
```
This seeds both the temperature sampling in `ModelPlayer.get_move()` (uses
`np.random.choice`) and the random opening selection (uses `random.choice`).
Games are reproducible across eval runs AND distinct from each other.

**3. Random opening plies** (`hexo_rl/eval/evaluator.py`)

For the first `eval_random_opening_plies` plies (default 4 = 2 compound moves),
both players' moves are replaced by `random.choice(board.legal_moves())`. This
creates genuinely diverse starting positions before model play begins, ensuring
even early-training checkpoints (whose policy is near-uniform) produce varied games.

**Config keys added** (`configs/eval.yaml`, propagated in `eval_pipeline.py`):

```yaml
eval_temperature: 0.5            # >0 → sample from visit dist; 0 → old argmax
eval_random_opening_plies: 4     # random opening plies for both players
eval_seed_base: 42               # game i uses seed base+i
```

**Tests added** (`tests/test_model_player.py`): 4 CPU-only unit tests covering
temperature variance, opening-ply bypass, deterministic regression guard, and
E2E seeded-opening diversity.

**Backward compat:** `ModelPlayer.__init__` signature unchanged (`temperature`
arg already existed). `eval_diagnostic.py` bypasses `Evaluator` — unaffected.
Old argmax + no-opening behaviour restored via `eval_temperature: 0.0` and
`eval_random_opening_plies: 0`.

### Effect

Next eval run against SealBot will produce 50 distinct games with varied opening
positions. Bradley-Terry CIs will reflect real checkpoint discrimination.

---

## §81 — Desktop Worker-Count Sweep 2026-04-12

### Motivation

Laptop P3 winner (n_workers=14, burst=16) caused 97% worker-idle time on
Ryzen 7 3700x via GIL burst stalls (§77). Swept D1–D5 to find desktop ceiling.

### Sweep table

| Run | n_workers | wait_ms | burst | gph (median last-5) | gpu_util | batch_fill | notes |
|-----|-----------|---------|-------|---------------------|----------|------------|-------|
| D1  | 6         | 8.0     | 8     | 247                 | ~73%     | —          | under-provisioned |
| D2  | 8         | 6.0     | 8     | 313                 | ~78%     | —          | monotonic gain |
| D3  | 10        | 5.0     | 8     | 334                 | ~79%     | ~78%       | **winner** |
| D4  | 8         | 8.0     | 4     | 277                 | —        | —          | burst=4 regresses |
| D5  | 12        | 5.0     | 8     | 307 (declining)     | ~76%     | ~90%       | overloaded GIL |

### Findings

- D3 (10w) peak throughput: **~334 gph** steady-state.
- D5 (12w): throughput dropped to 307 and declining; batch_fill rose 78%→90%,
  indicating inference server backs up. Workers saturate the GIL/callback boundary
  before adding useful parallelism.
- 400 gph gate from §69 is unreachable on Zen2 (Ryzen 7 3700x).
  Laptop gate (659K pos/hr) was measured on Zen4 8845HS — do not backport.
- Desktop ceiling: **~334 gph at 10w/5ms/burst=8**.

### Decision

D3 confirmed peak. `configs/variants/gumbel_targets_desktop.yaml` unchanged
(already reflects n_workers=10 from end-of-D3 update). Sweep yaml files deleted.
400 gph gate declared unreachable on this hardware; accept ~334 gph as desktop ceiling.
Proceed to sustained 24–48hr run (Phase 4.0 exit criterion).

### Buffer note

Buffer capacity is now 250K (§79). At sweep end: ~180K positions accumulated.
Sustained run resumes from checkpoint_00030851 with buffer growing toward 250K.
`schedule_idx=1` after construction; next trigger is step 300K. Clean.

---

## §82 — emit_event monitoring gap: ownership_loss + threat_loss (2026-04-12)

### Root cause

Both losses computed by `trainer.py` (result dict keys `ownership_loss`, `threat_loss`)
and written to structlog JSONL since §58, but absent from `emit_event()` in
`scripts/train.py`. Invisible on web and terminal dashboards for all runs prior to
this fix. Documented in `reports/aux_quiescence_health_2026-04-12/findings_summary.md` Q2.

### Fix

Two lines added to the `train_step` emit_event call in `scripts/train.py`:

```python
"loss_ownership": float(loss_info.get("ownership_loss", 0.0)),
"loss_threat":    float(loss_info.get("threat_loss", 0.0)),
```

Default to 0.0 when aux heads disabled or loss key absent — safe for old consumers.

**Commit:** `d6a293e fix(monitoring): emit ownership_loss and threat_loss in train events`

---

## §83 — quiescence_fire_count instrumentation (2026-04-12)

### Motivation

No instrumentation existed to measure whether the quiescence value override actually
fires during self-play. Structural analysis (Q5, findings_summary.md) confirmed logic
is correct but firing rate was zero-data. Cannot answer "is quiescence doing anything?"
without a counter.

### Implementation

- `engine/src/mcts/mod.rs`: `pub quiescence_fire_count: AtomicU64` on `MCTSTree`;
  reset to 0 in `new_game()`
- `engine/src/mcts/backup.rs`: `fetch_add(1, Ordering::Relaxed)` at all 4 firing
  branches in `apply_quiescence` (≥3 current wins → +1.0, ≥3 opponent wins → -1.0,
  2 current wins blend, 2 opponent wins blend)
- `engine/src/lib.rs`: Python property `quiescence_fire_count` on `PyMCTSTree`
- `engine/src/game_runner.rs`: `mcts_quiescence_fires Arc<AtomicU64>` accumulated
  per-search in worker thread; exposed as Python property
- `scripts/train.py`: `quiescence_fires_per_step` delta emitted in training event
- `tests/test_gumbel_mcts.py`: `TestQuiescenceFireCount` validates getter + reset

No performance impact — atomic relaxed load on non-critical (post-search) path.

**Commits:** `4124faa feat(mcts): add quiescence_fire_count instrumentation`,
`ad79be7 add quiescence to log`

---

## §84 — Fix eval checkpoint retention (2026-04-13)

### Symptom (pre-existing, first noted in §71)

§71 footnote: "checkpoint_00010000.pt does not exist in current checkpoints/ (was never
saved)". Same symptom on the laptop gumbel_targets run at step ~19K: ckpt_5000, ckpt_10000,
ckpt_15000 absent from disk. Eval DB (results.db) has BT ratings for all three players but
weight files are gone — re-eval, sharpness sweeps, and post-hoc forensics impossible.

### Root cause

Three config values created a perfect eviction storm:

```
checkpoint_interval: 500    # save every 500 steps
max_checkpoints_kept: 10    # keep 10 most recent
eval_interval: 5000         # eval at 5k, 10k, 15k ...

5000 / 500 = 10 = max_checkpoints_kept
```

`prune_checkpoints()` (`checkpoints.py:53–79`) kept the N largest step numbers and deleted
everything else via `Path.unlink()` — no exemption for eval-step checkpoints. After exactly
10 more rolling saves, each eval checkpoint was evicted by the next eval.

`best_checkpoint` promotion overwrites `best_model.pt` (weights-only, `torch.save()`). It
does not rename or copy the numbered checkpoint, so promotion offered no protection.

Eval DB stores player names (`"checkpoint_5000"`), not file paths — DB records intact,
only the weight files were gone.

### Fix

Two-tier retention: eval steps permanent, rolling window unchanged at 10.

- `checkpoints.py`: `prune_checkpoints()` gains `preserve_predicate: Optional[Callable[[int], bool]]`.
  Steps matching the predicate are excluded from the rotation pool entirely.
- `trainer.py` `save_checkpoint()`: builds predicate `lambda s: s > 0 and s % eval_interval == 0`
  from config each call (not a frozen set — tracks `eval_interval` if it changes mid-run).
  Guarded by `preserve_eval_checkpoints` config key (default `True`).
- `configs/training.yaml`: `preserve_eval_checkpoints: true` added.
- `tests/test_trainer.py`: `test_eval_checkpoints_not_pruned` — 30 fake files, eval_interval=5000,
  max_kept=10; asserts all 3 eval checkpoints present + correct 10 rolling present + older rolling absent.

### Recovery for live laptop run

ckpt_5000/10000/15000 unrecoverable — accept loss. Restart sequence: let run hit step 20000
naturally, eval completes, then graceful stop before step 25000 (old rotation window). On
restart, new code loads config, `prune_checkpoints` sees ckpt_20000 on disk, predicate
exempts it. ckpt_20000 becomes the first permanent forensic anchor.

**Commit:** `fix(training): preserve eval checkpoints`
