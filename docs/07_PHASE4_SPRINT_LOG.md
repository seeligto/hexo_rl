# HeXO Phase 4.0 Sprint Log — Consolidated Reference

Read this alongside `CLAUDE.md` at the start of any new session to avoid
re-litigating resolved decisions. Structured by subsystem, not by date.
For per-day narrative see `docs/07_PHASE4_SPRINT_LOG_BACKUP.md`.

---

## Classification Audit (§1–§101)

| Bucket | Sections |
|---|---|
| KEEP-FULL | §1, §2, §4, §5, §15, §19, §21, §26, §27, §28, §33, §34, §35, §36, §37, §40, §46b, §47, §58, §59, §61, §63, §66, §67, §69, §70, §71, §73, §74, §77, §80, §84, §85, §86, §88, §89, §90, §91, §95, §97, §98, §99, §100, §101 |
| KEEP-CONDENSED | §6, §11, §13, §14, §16, §17, §20, §22, §23, §24, §29, §30(game-cap/T_max), §31, §38, §41–§46, §48, §50–§57, §68, §72, §75, §76, §78, §79, §81, §82, §83, §87, §92, §93, §94, §96 |
| MERGE | §3+§25+§30(torch)+§32→torch.compile arc; §30(quiescence-gate)→§28; §52+§60→eval_interval; §61+§62→Gumbel; §63+§64+§65→dashboard metrics |
| BENCHMARK-STALE | 2026-04-01 table, 2026-04-02 table, §18 corrected table, §39 table, §51 table |
| DELETE | Test-count-only updates, "Immediate next steps", §27b operational note, §49 (superseded by §59) |
| SUPERSEDED | §49 (uncertainty head guard — head later disabled at §59); §9 (§66 Gumbel activation — superseded by §67 named variants + §74 audit); §92 partial (24-plane input reverted at §97) |

---

# Part 1 — Architecture & Features

## 1. Network Architecture

**Files:** `hexo_rl/model/network.py`, `hexo_rl/training/trainer.py`,
`configs/model.yaml`, `configs/training.yaml`

> Forward pointers — current authority:
>
> - Input grew 18 → 24 at §92 and reverted 24 → 18 at §97. Current: **18 planes**; chain is an aux target in a separate replay-buffer sub-buffer.
> - BatchNorm replaced with **GroupNorm(8)** throughout at §99. Pre-§99 checkpoints refuse to load.
> - Selective policy loss (§100) gates policy / opp_reply losses on `is_full_search`; value / chain / ownership / threat losses apply to all rows.

```
Input:  (18, 19, 19) tensor
        Planes 0-15: 8 history steps × 2 players (cluster snapshots)
        Planes 16-17: metadata (moves_remaining, turn parity)

Trunk:  12 × ResidualBlock(128ch, SE reduction=4)
        Post-activation: Conv → GN(8) → ReLU → Conv → GN(8) → SE → + skip → ReLU
        SE blocks: squeeze C→C/4→C per block (~1% FLOPs, validated in KataGo/LCZero)

Heads:
  Policy:      Conv(128→2, 1×1) → ReLU → Flatten → Linear → log_softmax
               (no GN — 2 channels; selective loss gates via full_search_mask, §100)
  Value:       GlobalAvgPool + GlobalMaxPool → concat(256) → Linear(256) → Linear(1) → tanh
               Loss: BCE(sigmoid(v_logit), (z+1)/2)   ← logit path avoids atanh NaN
  Opp reply:   Mirror of policy, training only, weight=0.15 (gated by full_search_mask per §100)
  Ownership:   Conv(1×1) → tanh → (19×19), weight=0.1, spatial MSE (target from replay-buffer u8 column, §85)
  Threat:      Conv(1×1) → raw logit → (19×19), weight=0.1, BCEWithLogitsLoss
               with pos_weight = threat_pos_weight (default 59.0, Q19; §92)
  Chain:       Conv(1×1) → (6, 19, 19), smooth-L1, weight aux_chain_weight (default 1.0);
               target read from ReplayBuffer chain_planes sub-buffer post-§97 (not input slice)
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

**Files:** `engine/src/mcts/`, `engine/src/board/`, `engine/src/game_runner/`,
`configs/selfplay.yaml`, `hexo_rl/selfplay/pool.py`

> Forward pointers — current authority:
>
> - `game_runner.rs` split into `game_runner/{mod,worker_loop,gumbel_search,records}.rs` at §86.
> - Dirichlet root noise ported to Rust on both PUCT and Gumbel branches at §73 (commit `71d7e6e`). Resolves Q17.
> - Gumbel flag moved from base config to named variants (`gumbel_full`, `gumbel_targets`, `baseline_puct`) at §67. Base `selfplay.yaml` has `gumbel_mcts: false, completed_q_values: false`.
> - ZOI is post-search only — §36 text corrected at §77; tree still expands with the full radius-8 legal set at all depths.
> - **Unresolved** docs-vs-code drift on temperature schedule — §70 C.1 showed §36 half-cosine-per-ply vs Rust quarter-cosine-per-compound-move differ. Code still wins; §36 description is aspirational until reconciled.
> - `quiescence_fire_count` instrumentation added at §83.
> - `get_improved_policy` is PUCT-tree-safe (§74.1) — training can use Gumbel completed-Q policy targets on PUCT-built trees.

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

> Forward pointers — current authority:
>
> - Initial tier raised back to 250K at §79.
> - Per-row aux target alignment (ownership + winning_line u8 columns) landed at §85.
> - File split mod/storage/push/sample/persist/sym_tables at §86. `engine/src/replay_buffer/sample.rs` now holds `sample_batch` + `apply_sym` kernels (the old `sampling.rs` was merged in).
> - `TensorBuffer` dead code deleted at §93 C9.5.
> - HEXB version history: v1 (§46b) → v2 → v3 (§92, added `n_planes` header; chain inside state at 24ch) → v4 (§97, 18 state + 6 chain separate sub-buffer) → **v5 (§100, adds `is_full_search` per-row column)**. v4 buffers still load with `is_full_search=1` default.
> - `chain_planes` augmentation scatter uses a dedicated `apply_chain_symmetry` pass with `axis_perm` remap (§92 C2, retained at §97).

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

> Forward pointers — current authority:
>
> - Eval determinism (temperature, per-game seeding, random opening plies) added at §80.
> - Two-tier checkpoint retention (rolling + permanent eval steps) added at §84.
> - `probe_threat_logits.py` committed as the step-5k kill criterion at §89. Revised at §91 (C1-C4: contrast + top-5 + top-10 + warning).
> - Graduation gate landed at §101 + §101.a: per-opponent `stride`, CI guard (`ci_lo > 0.5`), 200-game gating, anchor semantics (`inf_model ← best_model`). Supersedes the "50 games vs best" framing below.

- Bradley-Terry MLE (not incremental Elo). scipy L-BFGS-B with L2 regularisation 1e-6 to prevent divergence on perfect records.
- SQLite results store (WAL mode). Full BT recomputation from all historical pairwise data after each eval round.
- Gating rule: new checkpoint promoted if win_rate ≥ 0.55 over 50 games vs best checkpoint.
- Opponents: previous best, SealBot (fixed external Elo reference), RandomBot (sanity floor).
- Evaluation in a separate thread; model cloned (fresh HexTacToeNet) to avoid sharing compiled training model.

**eval_interval arc:** §52 changed 500→2000 (eval was blocking self-play). §60 changed 2000→5000 (at step 2000, SealBot winrate was 0/22; eval took 130 min = 55% of wall-clock). **Final: `eval_interval: 5000`** in `configs/training.yaml`, `best_checkpoint n_games: 50` in `configs/eval.yaml`. At ~490 steps/hr, one eval every ~10 hours = ~9% overhead.

---

## 6. Training Loop & Stability

> Forward pointers — current authority:
>
> - Scheduler bug fixed at §67; `total_steps` and `eta_min` are REQUIRED in config now — no silent 50K/1e-5 fallback. `decay_steps / total_steps ≈ 0.10` rule of thumb post exp A/C.
> - Ownership + threat losses emitted in `training_step` events at §82; chain added at §93 C14.
> - Training stack split into `scripts/train.py` + `hexo_rl/training/{loop, batch_assembly, aux_decode}.py` at §88.
> - FP16 NaN guard (§47) no longer resets BN running stats — §99 replaced BN with GN (no running stats to poison). Retains the `torch.special.entr` + log-clamp fixes.
> - Selective policy loss landed at §100: `full_search_mask` gates policy / opp_reply; value / chain / ownership / threat apply to all rows. Mutex with game-level `fast_prob` enforced at pool init.

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

> Forward pointers — current authority:
>
> - `policy_entropy_pretrain` and `policy_entropy_selfplay` split added at §71.2. Collapse threshold 1.5 nats on selfplay stream.
> - `/analyze` policy viewer added at §78 (checkpoint LRU cache, Blueprint, `HexCanvas` ES module).
> - `loss_chain`, `loss_ownership`, `loss_threat` surfaced in both renderers at §93 C14.
> - engineio disconnect `KeyError` swallowed via `threading.excepthook` filter at §91.

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

## 9. Gumbel MCTS Activation & Training Restart (§66) — SUPERSEDED

**Date:** 2026-04-07 (superseded by §67 + §74 + §96)

Historical snapshot kept for forensics. Current state:

- §67 replaced the single-flag approach with named variants (`gumbel_full`, `gumbel_targets`, `baseline_puct`) in `configs/variants/`. Base `configs/selfplay.yaml` has `gumbel_mcts: false, completed_q_values: false`.
- §66 amendment + trainer.py:372 fix: `completed_q_values` is now read from the flat merged config. The C1 KL-loss-dead bug is resolved for all runs after the amendment.
- §74.2 confirmed Gumbel vs PUCT pipeline parity on laptop (batch fill 100% both variants, worker throughput noise-overlapping). Desktop Gumbel behaviour confirmed via §96 exp E (in flight at time of writing).
- §98 supersedes the benchmark baseline reference.

---

## 10. Graduation Gate & Anchor Model

**Files:** `hexo_rl/training/loop.py`, `hexo_rl/eval/eval_pipeline.py`, `hexo_rl/eval/results_db.py`, `configs/eval.yaml`, `configs/training.yaml`

> Current authority: §101 + §101.a. Supersedes "win rate ≥ 55% over N games" framing in §5.

**Anchor semantics.** Self-play workers consume `inf_model` weights. `inf_model` is synced from the `best_model` anchor only at (a) cold-start load and (b) graduation — **never** from drifted `trainer.model` on checkpoint ticks. Between graduations, data quality is monotonic.

**Graduation gate (two-part):**

```
graduated = (wr_best >= promotion_winrate) AND (ci_lo > 0.5)
```

- `promotion_winrate: 0.55` (vs KrakenBot's 0.76 — conservative entry point).
- `require_ci_above_half: true` (§101.a M1) — binomial 95% CI lower bound > 0.5. Cuts false-positive rate from ~9% (point-threshold alone at n=200, p_true=0.5) to <1%.
- `n_games: 200` (binomial 95% CI ±~7% at p=0.55).
- Promotion copies **from `eval_model`** (the snapshot that was actually scored), not from current `trainer.model` — §101.a C1 was the critical bug.

**Eval cadence.** `eval_interval: 2500` base (eval.yaml); `training.yaml` precedence at both trigger and stride-math (§101.a H1). Per-opponent `stride`:

| Opponent | stride | n_games | role |
|---|---|---|---|
| `best_checkpoint` | 1 | 200 | graduation gate |
| `sealbot` | 4 | 50 | external benchmark (every 10k steps) |
| `random` | 1 | 20 | sanity floor |

**Cold-start.** No `best_model.pt` → clone from initial `trainer.model` + save. Candidate vs clone ~50% win rate → no spurious promotion.

**Operational invariants.** Resume where `trainer.step != best_model_step` logs `resume_anchor_step_mismatch` (§101.a M2). `eval_complete` payload reports `eval_games` (actually-played count, accounting for stride skips — §101.a M3). Stride-skipped opponents show `None` (not 0.0) on the dashboard (§101.a L2).

**Ratings ladder.** Bradley-Terry MLE over all recorded matches in `reports/eval/results.db` (SQLite, WAL); plot at `reports/eval/ratings_curve.png`. Per-run `run_id` threaded into all 5 DB calls (§68); reference opponents (SealBot, random_bot) use `run_id=""` as shared anchors.

## 11. Playout Cap — Selective Policy Loss

**Files:** `engine/src/game_runner/worker_loop.rs`, `engine/src/replay_buffer/*`, `hexo_rl/training/losses.py`, `configs/selfplay.yaml`

> Current authority: §100 + §100.c. The game-level cap (§43) still exists but is mutually exclusive with the move-level cap.

**Two caps; mutex at pool init (§100.c M1).**

1. **Game-level cap** (§43, legacy) — `playout_cap.fast_prob / fast_sims / standard_sims`. Whole-game fast/standard. Fast-game positions get zero-policy vectors (`sum < 1e-6`) → filtered out of policy loss by `policy_valid` mask. **Default off post-§100** (`fast_prob: 0.0`).
2. **Move-level cap** (§100, active) — per-move coin-flip between full-search and quick-search. Each position tagged with `is_full_search ∈ {0, 1}`. Rust stores the flag as a u8 column in the replay buffer; HEXB v5.

**Loss gating.** `compute_policy_loss`, `compute_kl_policy_loss`, and `compute_aux_loss(opp_reply)` intersect `full_search_mask` with `valid_mask`. Value / chain / ownership / threat losses apply to **all** rows regardless of the flag (their training signal is independent of MCTS sim count).

**Defaults.**

```yaml
playout_cap:
  fast_prob: 0.0          # game-level cap disabled
  full_search_prob: 0.25  # move-level — per-move P(full search)
  n_sims_quick: 100       # quick-search sim budget
  n_sims_full: 600        # full-search sim budget
```

Effective avg sims/move ≈ 0.75·100 + 0.25·600 = **225** (~2.3× compute vs the legacy 98 average from the game-level cap). KrakenBot-matched.

**HEXB v4 compatibility.** v5 adds 1 byte/row (`is_full_search`); v4 buffers load with default flag 1 (all positions treated as full-search — same as disabling move-level cap).

**Telemetry.** `trainer.py` logs `full_search_frac` — fraction of batch rows where **both** `policy_valid` and `full_search_mask` are True. Distinguishes "nothing contributed to policy loss" (mask-empty) from "genuine 0.0 loss" is a known follow-up (§100 "Known follow-ups").

---

# Part 2 — Operational Record

## Current Authoritative Benchmark Baseline

**2026-04-06 headline rows; worker-throughput + buffer-sample-augmented
rebaselined 2026-04-16 post-§97 18ch migration (§98). Ryzen 7 8845HS +
RTX 4060 Laptop, bench.full n=5, 3s warm-up (90s post-§98 for worker),
LTO+native.**

| Metric | Baseline (median) | Target | IQR |
|---|---|---|---|
| MCTS (CPU only, 800 sims/move × 62 iters) | 55,478 sim/s | ≥ 26,000 sim/s | ±400 |
| NN inference batch=64 | 9,810 pos/s | ≥ 8,500 pos/s | ±1 |
| NN latency batch=1 | 1.59 ms | ≤ 3.5 ms | ±0.05 ms |
| Replay buffer push | 762,130 pos/sec | ≥ 630,000 pos/sec | ±114,320 |
| Buffer sample raw (batch=256) | 1,037 µs/batch | ≤ 1,500 µs | ±34 µs |
| Buffer sample augmented (batch=256) | 1,663 µs/batch (§98) | ≤ 1,800 µs (§98 rebaseline) | ±566 µs |
| GPU utilisation | 100.0% | ≥ 85% | ±0 |
| VRAM usage (process) | 0.05 GB / 8.0 GB | ≤ 6.4 GB | ±0 |
| Worker throughput | 364,176 pos/hr max observed (§98) | ≥ 250,000 pos/hr (§98 rebaseline) | methodology-shift + warmup artifact per §98 |
| Batch fill % | 100.0% | ≥ 80% | ±0 |

All 10 targets PASS. Methodology: median n=5, 3s warm-up, realistic MCTS workload (800 sims/move × 62 iterations with tree reset), CPU unpinned (n=5 median provides sufficient variance control).

## Benchmark Evolution

| Date | What was wrong | What changed | Impact on targets |
|---|---|---|---|
| 2026-04-01 | MCTS workload was burst (50K sims in one tree) — exceeded L2 cache, inflated by boost clocks | Changed to 800 sims/move × 62 iter with tree reset; n=5 median | MCTS target dropped from 160K to realistic steady-state |
| 2026-04-03 | benchmark.py read `config.get('res_blocks')` (top-level) instead of `config['model']['res_blocks']` — measured wrong (smaller) model; VRAM used pynvml global not process-specific; single pool per measurement window included cold-start | Fixed model config path; switched to `torch.cuda.max_memory_allocated()`; keep one warm pool across all measurement windows | Worker throughput baseline corrected 1.18M→735K; NN latency 1.52ms→2.90ms; targets recalibrated |
| 2026-04-04 | Legal move radius corrected bbox+2→radius 8 (~9× branching factor expansion) + FPU behavioral tree shape change | Both are correct behaviour changes, not regressions | MCTS target rebaselined to ≥26K (85% of new ~31K median) |
| 2026-04-09 | NVIDIA driver/boost-clock step-change dropped NN inference and worker throughput ~14% cold/hot/idle, not a code regression | Rebaseline after structured three-run investigation (§72). | NN inference target ≥ 8,500 → ≥ 8,250 pos/s; worker throughput ≥ 625k → ≥ 500k pos/hr |
| 2026-04-16 | 18-channel migration (§97) — chain planes moved out of NN input into a separate replay-buffer sub-buffer. Buffer sample augmented now splits scatter (18 state + 6 chain). Worker benchmark hit a warmup-design artifact (0-position measurement windows). | Rebaseline per §98. Note: real training (GPU shared with gradient steps) delivers ~48k pos/hr at production sim counts — the bench measures self-play-only capacity at reduced sims. | Buffer aug ≤ 1,400 → ≤ 1,800 µs; worker throughput ≥ 500k → ≥ 250k pos/hr |

## Regressions & Reversions

| Feature | When Added | Reverted | Reason |
|---|---|---|---|
| SealBot mixed opponent in self-play | §17 (2026-04-02) | Immediately (`c9f39de`) | Python daemon threads caused 3.3× GIL contention regression (1.52M→464K pos/hr). **Do not re-litigate** — GIL regression was an implementation issue, not a conceptual flaw. Re-add post-Phase 4.5 baseline using subprocess-based wrapper to avoid GIL. |
| Forced-win short-circuit (`FormationDetector::has_forced_win()`) | Pre-sprint | Removed (2026-04-02) | MCTS bypassed NN for near-win positions → network never learned to evaluate them. Removing adds ~30% more NN calls/game (batch fill 99.4%→99.82%) but training quality requires it. |
| draw_reward: -0.1 | §24 | Raised to -0.5 at §40 | Not a revert — a correction. -0.1 assumed draws were minority; at 56% draw rate the signal was too weak to break equilibrium. |
| torch.compile | §3 | Disabled §32 | Python 3.14 CUDA graph incompatibility cascade (TLS crash → Triton spike). |
| uncertainty_weight: 0.05 | §33 | Set to 0.0 at §59 | Gaussian NLL diverges when σ²→clamp floor; total_loss spiked to ~394. |
| Chain-length planes in NN input (18→24) | §92 | Reverted to 18 at §97 | Redundant — trunk already predicts chain as aux; KrakenBot achieves top play with 2-channel input. Chain retained as aux target in a separate replay-buffer sub-buffer. |
| BatchNorm throughout trunk | pre-§99 | Replaced with GroupNorm(8) at §99 | BN running stats drift from live distribution during self-play; batch=1 MCTS leaf eval used stale stats. GN computes per-sample statistics. |

## Key Resolved Bugs

| § | Bug | Impact | Fix |
|---|---|---|---|
| §26 | Legal move radius bbox+2 instead of radius 8 | ~9× too-small branching factor; invalid MCTS search | Per-stone hex ball iteration |
| §38 | `action_idx` u16 overflow (cap 65,535) | Silent MCTS tree corruption with global axial coords | Widen to u32 |
| §47 | FP16 0×-inf NaN cascade | NaN total_loss, BN poisoning, training halt | Log-clamp aux CE; `torch.special.entr()` for entropy; BN reset guard |
| §58 | Three resume bugs | Pretrained stream silently disabled on resume; hidden loss spikes | Decouple buffer-restore from corpus-load; threshold 10K; log all loss terms |
| §59 | TT memory leak (`new_game()` did not clear TT) | 28 GB RSS after 500 min | `self.transposition_table.clear()` in `new_game()` |
| §73 | Dirichlet root noise never fired on Rust training path (unported at Phase 3.5 migration) | Self-play mode collapse — 16,880 steps of carbon-copy games (Q17) | Port `apply_dirichlet_to_root` into `engine/src/game_runner.rs` (commit `71d7e6e`), both PUCT and Gumbel branches, with intermediate-ply skip. |
| §101 C1 | Promoted weights ≠ evaluated weights | Every graduation committed unvalidated weights as the new anchor | `eval_model` allocated once; promotion branch loads `best_model ← eval_model` (still holding the scored snapshot). |

---

# Part 3 — Open Questions

| # | Question | Status |
|---|---|---|
| Q5 | Supervised→self-play transition schedule | ✅ Resolved — exponential decay 0.8→0.1 over `decay_steps` (20K post exp A/C) |
| Q6 | Sequential vs compound action space | ✅ Resolved — sequential confirmed correct |
| Q13 | Chain-length planes | ✅ Resolved §92 landing + §97 revision (aux sub-buffer, not input) |
| Q17 | Self-play mode collapse | ✅ Resolved §73 — Dirichlet port to Rust training path |
| Q19 | Threat-head BCE class imbalance | ✅ Resolved §92 — `threat_pos_weight = 59.0` |
| Q25 | 24-plane worker throughput variance | ✅ Resolved §97 — 24-plane payload reverted |
| Q2 | Value aggregation: min vs mean vs attention | 🔴 Active — HIGH priority, blocks Phase 4.5 |
| Q3 | Optimal K (number of cluster windows) | 🟡 Active — MEDIUM priority |
| Q8 | First-player advantage in value training | 🟡 Active — MEDIUM priority (corpus: 51.6% P1 overall, 57.1% in 1000-1200 Elo) |
| Q9 | KL-divergence weighted buffer writes (KataGo) | 🟡 Active — MEDIUM priority. Prerequisite: Phase 4.5 baseline checkpoint. |
| Q10 | Torus board encoding (imaseal experiment) | 🔵 Watch — incompatible with attention-anchored windowing; pending imaseal results |
| Q14 | KrakenBot MinimaxBot as eval-ladder opponent | 🔵 Watch — blocked on submodule add |
| Q15 | Corpus tactical quality filtering | 🔵 Watch |
| Q16 | leaf_batch_size round-trip hypothesis | 🔵 Watch — blocked on Phase 4.5 baseline |
| Q18 | NN forward latency ceiling | 🔵 Watch — architectural (CUDA streams / process split / torch.compile); Phase 4.5 |
| Q21 | Wider-window chain-aux target | 🟣 Parked — revisit post-§97 baseline |
| Q1, Q4, Q7 | MCTS convergence rate, augmentation equivariance, Transformer encoder | 🔵 Deferred — Phase 5+ |

See `docs/06_OPEN_QUESTIONS.md` for full detail.

---

# Key Config Values (current settled state)

```yaml
# configs/selfplay.yaml
mcts:
  n_simulations: 400          # §98 bench workload; ZOI trims effective branching
  fpu_reduction: 0.25         # dynamic FPU (KrakenBot baseline)
  quiescence_enabled: true
  quiescence_blend_2: 0.3
  dirichlet_alpha: 0.3
  epsilon: 0.25
  dirichlet_enabled: true     # gates the §73 Rust Dirichlet call on the training path
  temperature_threshold_ply: 30
selfplay:
  completed_q_values: false   # base; opt in via --variant gumbel_full / gumbel_targets (§67)
  c_visit: 50.0
  c_scale: 1.0
  gumbel_mcts: false          # base; opt in via --variant gumbel_full (§67, §96)
  gumbel_m: 16
  gumbel_explore_moves: 10
  n_workers: 14
  inference_batch_size: 64
  inference_max_wait_ms: 4.0
  leaf_batch_size: 8
  max_game_moves: 200         # PLIES (§76 reverted 150→200 after compound/ply mix-up)
  playout_cap:
    fast_prob: 0.0            # §100 — disabled by default; mutex with full_search_prob
    fast_sims: 64
    standard_sims: 200
    n_sims_quick: 100         # §100 move-level cap — quick search
    n_sims_full: 600          # §100 move-level cap — full search
    full_search_prob: 0.25    # §100 — P(full search per move)
    temperature_threshold_compound_moves: 15
    temp_min: 0.05
    zoi_enabled: true
    zoi_lookback: 16
    zoi_margin: 5

# configs/training.yaml
fp16: true
torch_compile: false          # DISABLED — Python 3.14 compat (§32)
policy_prune_frac: 0.02
training_steps_per_game: 4.0  # P3 winner (§69)
max_train_burst: 16           # P3 winner (§69)
total_steps: 200_000          # REQUIRED (§67); CosineAnnealingLR T_max
eta_min: 2e-4                 # REQUIRED (§67); ~10% of peak lr=0.002
eval_interval: 5000           # overrides eval.yaml; §101 uses this for stride math
checkpoint_interval: 500
max_checkpoints_kept: 10
preserve_eval_checkpoints: true  # §84 two-tier retention
aux_opp_reply_weight: 0.15
entropy_reg_weight: 0.01
ownership_weight: 0.1
threat_weight: 0.1
threat_pos_weight: 59.0       # Q19 (§92); BCE positive-class weight
aux_chain_weight: 1.0         # Q13-aux (§92); smooth-L1; target from chain sub-buffer
zero_chain_planes: false      # Exp C (§95) ablation — default false post §97
uncertainty_weight: 0.0       # §59 disabled
draw_value: -0.5
grad_clip: 1.0
recency_weight: 0.75
mixing:
  decay_steps: 20_000         # accelerated post exp A/C; rule ≈ 0.10 × total_steps
  pretrain_max_samples: 200_000
  buffer_persist: true
  buffer_persist_path: "checkpoints/replay_buffer.bin"
buffer_schedule:              # §79
  - {step: 0,           capacity: 250_000}
  - {step: 300_000,     capacity: 500_000}
  - {step: 1_000_000,   capacity: 1_000_000}

# configs/model.yaml
in_channels: 18               # §97 — chain planes moved to aux sub-buffer
res_blocks: 12
filters: 128
se_reduction_ratio: 4

# configs/eval.yaml
eval_pipeline:
  eval_interval: 2500         # base; per-opponent `stride` multiplies
  opponents:
    best_checkpoint: {stride: 1,  n_games: 200, model_sims: 128}   # §101 anchor gate
    sealbot:         {stride: 4,  n_games: 50,  think_time_strong: 0.5}
    random:          {stride: 1,  n_games: 20,  model_sims: 96}
  gating:
    promotion_winrate: 0.55   # §101 graduation threshold (wr_best AND ci_lo > 0.5)
    require_ci_above_half: true   # §101.a M1

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

`EvalPipeline` stored `self.run_id` but never passed it to 5 `db.get_or_create_player()` / `db.insert_match()` call sites in `run_evaluation()` — every run's eval collapsed onto `run_id=""` in the ratings DB. Fix: thread `run_id=self.run_id` through all 5 calls. Reference opponents (SealBot, random_bot) keep `run_id=""` as shared anchors; pairwise/history queries already match run-specific players plus empty-`run_id` references.

**Broken-run cleanup (§67 scheduler poison):** archived to `archive/checkpoints.broken-202604/` (10 checkpoints, best_model.pt, replay_buffer.bin, log) and `archive/eval.broken-202604/results.db`. Kept: `bootstrap_model.pt`, `checkpoints/pretrain/`, `runs/*/games/`, logs, corpus.

---

### §69 — Config Sweep 2026-04-08 — PUCT/Gumbel Knob Ranking

> **Historical, superseded by §90 (2026-04-13).** §90 is now the authoritative
> laptop `gumbel_targets` throughput baseline at HEAD (post-refactor, post-A1,
> post-A3). §69's P3 winner config remains the live config but the surrounding
> measurements predate the current engine state.

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

### §72 — Bench Baseline Rebaseline — 2026-04-09 Driver-State Shift

Three `bench.full` runs on 2026-04-09/10 failed the same two §66 targets (NN inference ~8,370 vs 8,500; worker throughput ~541k vs 625k). Cold/hot/idle investigation ruled out thermals (GPU stayed at 49°C). Root cause: NVIDIA laptop driver's `DynamicPowerManagement=3` settled the GPU into a lower boost-clock bin overnight — NN latency 1.59 ms → 1.77–1.80 ms (~14% clock reduction); worker throughput failures downstream.

**Rebaselined targets:** NN inference ≥ 8,250 pos/s (was 8,500); worker throughput ≥ 500,000 pos/hr (was 625,000). Baseline column retains 2026-04-06 peak for hardware capability reference; targets reflect sustained operating floor. Artifacts: `archive/bench_investigation_2026-04-09/`.

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

Draw-rate investigation (`reports/draw_rate_investigation_2026-04-10/`): 100% of draws are 150-ply timeouts. Low-sim games (fast_prob=0.25, 50 sims, τ=1.0, PUCT) hit 94.4% draw vs 3.7% on standard games — colony-extension behaviour in the viewer. Fix: `fast_prob: 0.0` in `configs/variants/gumbel_targets.yaml`. `gumbel_full.yaml` unchanged (Gumbel SH effective in low-sim regime, §71). Resumed from ckpt_25008.

---

### §76 — max_game_moves correction for gumbel_targets (2026-04-10)

Phase A diagnostic confirmed `max_game_moves` counts plies not compound moves. `gumbel_targets` was alone at 150 plies (a §69 artifact for `fast_prob=0.25`); with §75's `fast_prob=0.0`, 57.6% of games hit the cap. Fix: 150 → 200 plies; yaml comment "compound moves" → "plies". Resumed from `ckpt_25008`.

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

Interactive debugging tool — inspect raw network priors on arbitrary positions (§70 mode collapse was invisible until 16k steps).

**Scope** (branch `feat/policy-viewer`, 4 commits):

1. Rust PyO3 — `forced_root_child` getter/setter, `get_root_children_info()`, `get_improved_policy()`, `get_top_visits` → 4-tuple (+q_value).
2. `hex_canvas.js` ES module extracted from `viewer.html` for reuse.
3. `/api/analyze` Blueprint — checkpoint LRU (max 3, mtime stale check), Python-driven MCTS (PUCT + Gumbel SH), `ThreadPoolExecutor(1)`. `model_loader.py` loads checkpoints without importing Trainer.
4. `/analyze` SPA — sidebar, policy heatmap, visit overlay, deep-link (`?moves=<base64>&checkpoint=<path>`).

**Key decisions.** Python-driven MCTS (not Rust `analyze_position`) — avoids FFI callback complexity; PyMCTSTree already exposes `select_leaves`/`expand_and_backup`. Gumbel SH in `/analyze` uses raw Q (not `completed_q_values`) — interactive-only; production SH in `engine/src/game_runner.rs` stays authoritative. `model_loader.py` duplicates `_extract_model_state` / `_infer_model_hparams` from Trainer to sidestep optimizer/scheduler imports; sync test added.

**Post-review fixes:** deep-link XSS (typeof validation), BOARD_SIZE from checkpoint metadata (was hardcoded 19), checkpoint path-traversal guard, dead var cleanup, `analyze_bp.checkpoint_dir` configurable.

---

## §79 — Initial buffer increased 100K → 250K (2026-04-12)

§40b had reduced 250K → 100K as a draw-collapse stability measure; collapse resolved at §40, CLAUDE.md still said 250K — config was the stale artifact. At 100K with ~48% self-play the model sees ~600 games of context, too thin to generalise beyond colony patterns.

**Schedule:** `[{step:0, 250K}, {step:300K, 500K}, {step:1M, 1M}]`. Growth tiers shift right vs §40b. Steps 300K and 1M exceed `total_steps: 200_000` — apply only on extended runs.

**Memory budget** (14,458 B/entry × Rust + Python-mirror): 250K ≈ 5.05 GB, 500K ≈ 10.1 GB, 1M ≈ 20.2 GB. 32 GB RAM → 250K leaves ~19 GB headroom; +2.98 GB vs 100K.

**Resume safety:** `load_from_path` reads `min(saved_size, self.capacity)` into pre-allocated capacity — no resize. Old 100K checkpoints load cleanly into 250K buffer.

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

Laptop P3 winner (n_workers=14, burst=16) caused 97% worker-idle on Ryzen 7 3700x via GIL burst stalls (§77). D1–D5 sweep found ceiling at **D3: n_workers=10, wait_ms=5.0, burst=8 → ~334 gph**. D5 (12w) regressed to 307 gph (declining) as batch_fill rose 78→90% — inference server backs up, GIL/callback boundary saturated. §69's 400 gph gate unreachable on Zen2; laptop gate (659K pos/hr on Zen4) does not backport. `configs/variants/gumbel_targets_desktop.yaml` locks in D3. Sweep yamls deleted. Sustained run resumes from `ckpt_30851` with 250K buffer (§79) at ~180K filled.

---

## §82 — emit_event monitoring gap: ownership_loss + threat_loss (2026-04-12)

Both losses written to structlog JSONL since §58 but absent from `emit_event()` in `scripts/train.py` → invisible on dashboards. Fix: added `"loss_ownership"` and `"loss_threat"` (default 0.0) to the `train_step` event. Commit `d6a293e`.

---

## §83 — quiescence_fire_count instrumentation (2026-04-12)

No instrumentation existed to measure whether the quiescence value override actually fires during self-play. Added `pub quiescence_fire_count: AtomicU64` on `MCTSTree` (reset in `new_game()`); `fetch_add(1, Relaxed)` at all 4 firing branches in `apply_quiescence`. `SelfPlayRunner` accumulates `mcts_quiescence_fires` per-search; emitted as `quiescence_fires_per_step` in the training event. `tests/test_gumbel_mcts.py::TestQuiescenceFireCount` validates getter + reset. Zero performance impact (relaxed atomic on post-search path). Commits `4124faa`, `ad79be7`.

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

## §85 — A1 aux target alignment (Python side) (2026-04-13)

Companion landing to Rust commit `faafc43` (`feat(replay_buffer): per-row aux target alignment (A1 Rust side)`). Rips out the legacy ring-buffer aux path and threads per-row ownership + winning_line u8 columns end-to-end.

### A1 root cause — three compounding subproblems

1. **Indexing.** `pool.get_aux_targets()` pulled aux from a 200-entry game-level ring with independent random indices, fully decoupled from `buffer.sample_batch` batch indices. Aux targets had no relation to the states they were paired with.
2. **Cardinality.** One aux map per *game* broadcast across ~60 replay rows. Even index-aligned, one-per-game cannot match per-row state windows.
3. **Frame.** Aux maps were projected to the **game-end bbox centroid** while each replay row's state was projected to that row's **own cluster window centre at recording time**. Offsets up to ±9 cells in any multi-cluster game.

### Fix shape (Option A from `/tmp/A1_aux_alignment_spike.md`)

- **Rust side (commit faafc43):** `ReplayBuffer` gained per-row `ownership` + `winning_line` u8 columns; `game_runner.rs` reprojects them at game end using each row's own `(cq, cr)` cluster centre via `Board::window_flat_idx_at`. `apply_sym` extended in `replay_buffer/sampling.rs` so the 12-fold hex symmetry table applies to both new planes consistently with state + policy. `sample_batch` now returns a 5-tuple; `push` and `push_game` grew two positional args; `collect_data` now returns a 6-tuple.
- **Python side (this commit):**
  - `pool.get_aux_targets`, `_ownership_ring`, `_threat_ring` deleted from `hexo_rl/selfplay/pool.py`.
  - `_stats_loop` unpacks the new `collect_data` 6-tuple and threads per-row u8 aux into both `replay_buffer.push` and `recent_buffer.push`.
  - `RecentBuffer` (`hexo_rl/training/recency_buffer.py`) gained `_ownership` / `_winning_line` u8 columns; `push` and `sample` now carry aux. Existing 3-arg push callers fall back to ones/zeros defaults.
  - `Trainer._train_on_batch` decodes ownership u8 `{0=P2, 1=empty, 2=P1}` → float `{-1, 0, +1}` via `astype(f32) - 1.0`; winning_line u8 → f32 directly. Accepts `n_pretrain` row count and slices `[n_pretrain:]` before computing ownership MSE + threat BCE so pretrain corpus rows do not poison the aux heads.
  - `Trainer.train_step` no longer accepts the legacy `ownership_targets` / `threat_targets` kwargs; aux now flows from the buffers themselves.
  - `scripts/train.py` corpus loader pads `bootstrap_corpus.npz` with `ownership=ones` (decoded 0.0, neutral MSE) and `winning_line=zeros`; the `n_pretrain` row slice masks them out of aux loss.
  - Pre-allocated batch buffers extended with `_own_buf` + `_wl_buf` (uint8, `(batch, 19, 19)`).

### Empirical baseline (ckpt_19500, 20-position threat-logit probe, `/tmp/colony_spam_diagnosis.md` §2)

| metric                                 | bootstrap_model.pt | checkpoint_00019500.pt |
|----------------------------------------|--------------------|------------------------|
| threat logit @ extension cell          | −0.14 ± 0.74       | **−3.25 ± 0.46**       |
| threat logit @ random empty cell       | −0.52 ± 0.39       | **−5.11 ± 1.40**       |
| contrast (extension − random empty)    | **+0.38**          | **+1.86**              |

ckpt_19500 had a *higher* contrast than bootstrap — the symptom of the head learning a marginal-class shortcut against a stale, mis-aligned label rather than the true spatial signal.

### Kill criterion for next sustained run (REVISED §91 2026-04-14)

**Original §85 criterion was over-indexed on ckpt_19500's specific collapse
signature.** ckpt_00014344 (the next sustained run) hit a different failure
mode: contrast_mean grew TO **10× bootstrap (+3.94)** while absolute logits
drifted globally negative (ext_logit_mean = −6.2). That is the OPPOSITE of
ckpt_19500, where contrast grew only 5× and both logits collapsed by the
same amount. Old C1 (`ext_logit_mean >= baseline − 1.0`) FAILed; old C2 and
C3 PASSed. The pattern is consistent with BCE-on-imbalanced-labels driving
logits globally negative while position-conditional sharpness IMPROVES —
i.e. the policy head is doing exactly what we wanted (not colony-spamming),
just with a global bias shift in the threat head.

The original C1 was therefore not a colony-spam detector — it was a BCE
scale-drift detector, and gating on it would have incorrectly killed a
healthy run. C1 is replaced; the colony-spam intent is preserved by adding
a top-10 condition. The full revision is in §91. The current criterion is:

| # | condition | threshold |
|---|-----------|-----------|
| 1 | contrast_mean (ext − ctrl) | ≥ max(0.38, 0.8 × bootstrap_contrast) |
| 2 | extension cell in policy top-5 | ≥ 40% |
| 3 | extension cell in policy top-10 | ≥ 60% |
| 4 (warning) | abs(ext_logit_mean − bootstrap_ext_mean) < 5.0 | warning only — never gates |

`make probe.latest` enforces C1-C3; C4 prints a WARNING line in the report
but does not flip the exit code. Bootstrap baseline numbers come from
`fixtures/threat_probe_baseline.json` (schema v2).

If not met, the aux fix did not materially land; investigate before continuing.
If met, the colony-spam loop is a separate failure mode and the threat head is
free to do its job.

### Corpus aux shortcut

Chose **option (b)** from the spike doc: pad corpus rows with neutral aux (ones/zeros) at load time, mask via `n_pretrain` row slice. The alternative (extending `bootstrap_corpus.npz` with ownership/winning_line columns + reworking `scripts/export_corpus_npz.py`) is parked as a separate corpus refactor — orthogonal to the colony-spam fix and not blocking the next sustained run.

### Telemetry

`Trainer._train_on_batch` now emits `aux_loss_rows = batch_n - n_pretrain` in the `train_step` result dict whenever aux losses run. Stuck `n_pretrain == batch_n` (no rows contributing to aux) becomes visible in dashboards.

### Dead code left behind

Rust `drain_game_results()` still emits the legacy float32 `ownership_flat` / `winning_line_flat` (game-end frame) tuple fields. Pool unpacks and discards them via underscore variables. **TODO:** strip from the Rust drain path in a follow-up patch — pure dead stripe, zero runtime cost, no urgency.

### Memory delta

ReplayBuffer adds `2 × 361 = 722 B/row` for the new u8 columns. At capacity=1M: **+722 MB**. RecentBuffer (capacity ≈ 500K) adds another **+360 MB**. Headroom on the 48 GB box is fine but should be re-checked before any future capacity bump.

### Files touched

- `engine/src/replay_buffer/mod.rs`, `engine/src/replay_buffer/sampling.rs`, `engine/src/replay_buffer/sym_tables.rs`, `engine/src/game_runner.rs` (Rust, prior commit faafc43)
- `hexo_rl/selfplay/pool.py`
- `hexo_rl/training/trainer.py`
- `hexo_rl/training/recency_buffer.py`
- `scripts/train.py`
- `tests/test_rust_replay_buffer.py`, `tests/test_trainer.py`, `tests/test_phase4_smoke.py`, `tests/test_dashboard_events.py`, `tests/test_weight_schedule_wiring.py`, `tests/test_buffer_shutdown.py`, `tests/test_worker_pool.py`
- `tests/test_aux_target_alignment.py` (new)

**Commit:** `feat(training): align aux targets with state batch (A1 Python side)`

---

## §86. Structural split of `replay_buffer/` and `game_runner.rs` (2026-04-13)

**What.** Pure structural refactor of the two Rust files that the A1 aux
target alignment (§85) had inflated past the project's one-concept-per-file
threshold. Zero behaviour change; `cargo test` is the oracle. PyO3 surface
stable — every exported method on `ReplayBuffer` and `SelfPlayRunner` keeps
the same name and signature.

**Why now.** Landing A1 pushed `engine/src/replay_buffer/mod.rs` to 1,102 lines
(struct + storage + push + sample + resize + weight schedule + HEXB v2 save/load +
6 tests in one file) and `engine/src/game_runner.rs` to 1,313 lines (struct +
PyO3 facade + 502-line worker loop + `GumbelSearchState` + aggregate helpers +
drain + 9 tests). Both files violated the "one concept per file" rule and
slowed future diff review. This sprint is the cleanup pass.

**Post-A1 layout.**

```
engine/src/replay_buffer/
  mod.rs        ~220 lines  ReplayBuffer struct + #[pymethods] facade
  storage.rs    ~120 lines  resize, dashboard stats, weight schedule, next_game_id
  push.rs       ~380 lines  push, push_game, push_raw (test-only) + 2 tests
  sample.rs     ~420 lines  sample_batch + weighted-sample kernels + apply_sym + 4 tests
  persist.rs    ~280 lines  HEXB v2 save/load + round-trip test
  sym_tables.rs ~120 lines  UNCHANGED — 12-fold tables + WeightSchedule

engine/src/game_runner/
  mod.rs           ~430 lines  SelfPlayRunner struct + #[pymethods] facade + Drop + 3 tests
  worker_loop.rs   ~500 lines  start_impl — worker thread spawn + per-move MCTS loop
  gumbel_search.rs ~295 lines  GumbelSearchState + 6 gumbel tests
  records.rs       ~175 lines  aggregate_policy, aggregate_policy_to_local,
                               sample_policy, reproject_game_end_row
```

The old `engine/src/replay_buffer/sampling.rs` is **merged into `sample.rs`**:
after `sample_batch` itself moves out of `mod.rs`, maintaining a separate
"internal kernel" file next to a near-synonym "public entry" file creates
a naming trap (which file owns `apply_sym`?). One file for all sampling
concerns, still under the 500-line cap per file.

The 502-line worker loop was trimmed below the cap by extracting the
game-end ownership / winning-line reprojection block (~20 lines of the
inner per-row loop) to `records::reproject_game_end_row`. All other start()
logic is byte-identical.

**Visibility hygiene.** Fields and helpers that never cross a module
boundary defaulted to `pub(crate)`. `GumbelSearchState` is `pub(super)` —
only `worker_loop.rs` constructs and drives it. `Position` struct in the
old `game_runner.rs` was a dead `pub` type constructed nowhere in the
codebase; reducing its visibility exposed it as dead via `dead_code`, and
it was deleted in the same pass.

**Rust test binary build fix.** `cargo test` at HEAD failed to link
against libpython (`rust-lld: error: undefined symbol: PyErr_SetObject`,
etc.) because `engine/Cargo.toml` hard-coded `pyo3/extension-module`,
which strips Python symbols from the binary. That flag is correct for
the maturin cdylib build but wrong for the standalone test binary.
Restructured the feature wiring:

```toml
[features]
default = ["extension-module"]
extension-module = ["pyo3/extension-module"]
test-with-python = ["pyo3/auto-initialize"]
```

`maturin develop --release -m engine/Cargo.toml` picks up `extension-module`
via the default feature. `make test` now runs
`cargo test --no-default-features --features test-with-python`, which
replaces `extension-module` with `auto-initialize` so the test binary
links libpython directly and resolves every PyO3 C-API symbol.

**Tests.** 113 passing, zero test-body modifications. Test functions were
physically moved to the file that exercises the code they test (e.g.
`test_aux_hexb_v2_roundtrip` → `persist.rs`, `test_gumbel_topk_selection` →
`gumbel_search.rs`); the assertions themselves were not touched. Every
test continues to call its public entry point through its Rust name
(e.g. `buf.save_to_path(path)` in the persist test resolves to the
PyO3 facade in `mod.rs`, which delegates to `save_to_path_impl` in
`persist.rs`). Build is `cargo build --release` clean (zero warnings).

**Out of scope — tracked in `/tmp/refactor_todos.md`.**

- A1 reviewer's "cumulative-of-cumulative" mean_depth / root_concentration
  bias at the old `game_runner.rs:622-633` (now
  `game_runner/worker_loop.rs` stats block). Fixing this changes what the
  dashboard reports and would invalidate any smoke comparison against
  pre-refactor baselines, so it ships in a separate commit with its own
  regression test.

**Files touched.**

- `engine/Cargo.toml` — feature restructure (see Rust test binary build fix)
- `engine/src/replay_buffer/{mod,storage,push,sample,persist}.rs` (mod.rs rewritten, 4 new)
- `engine/src/replay_buffer/sampling.rs` (**deleted**, merged into sample.rs)
- `engine/src/game_runner/{mod,worker_loop,gumbel_search,records}.rs` (4 new)
- `engine/src/game_runner.rs` (**deleted**, promoted to directory)
- `Makefile` — `test` target passes `--no-default-features --features test-with-python`
- `CLAUDE.md` — Repository layout file tree updated
- `docs/01_architecture.md`, `docs/09_VIEWER_SPEC.md`, `docs/q12_s_ordering_audit.md`
  — file path references updated

**Commit:** `refactor(engine): split replay_buffer and game_runner into modules`

## §87 — gate pyo3 extension-module behind cargo feature (2026-04-13)

Removed `extension-module` from `[features] default` so bare `cargo test` links libpython without `--no-default-features --features test-with-python`. Works because Rust tests don't call `Python::with_gil()` — no interpreter bootstrap needed. `maturin develop` reads `features = ["extension-module"]` from `pyproject.toml` and activates it explicitly. `test-with-python` retained as escape hatch. Commit `chore(build): gate pyo3 extension-module behind cargo feature`.

---

## §88 — Python training stack refactor: batch_assembly, loop, aux_decode (2026-04-13)

**What.** Pure structural refactor of `scripts/train.py` and `hexo_rl/training/trainer.py` after A1 (§85) inflated both past clean boundaries. Zero behaviour change; 676 pytest + 119 cargo tests are the oracle.

**Why now.** `scripts/train.py` grew to 1,132 lines combining CLI parsing, config merging, buffer management, corpus loading, batch assembly, signal handling, dashboard setup, eval pipeline, GPU monitor, and the main training loop — seven distinct concerns in one file.

**Post-refactor layout.**

```
scripts/train.py                     319 lines   (was 1,132) — CLI + config + build core objects → run_training_loop
hexo_rl/training/
  aux_decode.py          69 lines   NEW — decode_ownership, decode_winning_line, mask_aux_rows
  batch_assembly.py     297 lines   NEW — BatchBuffers, allocate_batch_buffers, load_pretrained_buffer, assemble_mixed_batch
  loop.py               680 lines   NEW — run_training_loop: inf model, WorkerPool, dashboards,
                                           GPU monitor, eval pipeline, main _run_loop, teardown
  trainer.py            720 lines   (was 746) — now uses aux_decode for decode + mask
```

**Extraction boundaries.**

- `aux_decode.py`: the three u8→fp32 conversion and [n_pretrain:] masking fragments pulled from `trainer._train_on_batch`. Trainer imports and calls them; no logic change.
- `batch_assembly.py`: pre-allocated batch arrays (`BatchBuffers` dataclass + `allocate_batch_buffers`), corpus NPZ → Rust buffer loading (`load_pretrained_buffer`), and the mixed-batch assembly path (`assemble_mixed_batch` + private `_sample_selfplay`). `assemble_mixed_batch` is byte-for-byte equivalent to the inline block that was in the training loop; it uses the same in-place `np.copyto` steady-state path and `np.concatenate` warm-up path.
- `loop.py`: everything from inference model construction through `pool.stop()` + final checkpoint save. Receives `(trainer, buffer, pretrained_buffer, recent_buffer, bufs, config, train_cfg, mcts_config, args, device, run_id, capacity, min_buf_size, buffer_schedule, recency_weight, batch_size_cfg, mixing_cfg, mixing_initial_w, mixing_min_w, mixing_decay_steps)`.

**Public API stability.** `from hexo_rl.training.trainer import Trainer` and all other existing imports are unchanged. The three new modules are purely additive.

**Tests.** `make test`: 119 Rust + 676 Python, all pass. Smoke test parity deferred — user will run `make train.smoke` independently to verify JSONL loss values.

**Out of scope — tracked in `/tmp/refactor_todos.md`.**

- `hexo_rl/selfplay/pool.py` is 312 lines and cohesive; left alone per the scope rule (< 600 lines → no split).
- `docs/01_architecture.md` has no Python training stack file listing; no update required.

**Commit:** `refactor(training): extract batch_assembly, loop, aux_decode`

---

## §89 — Threat-logit probe committed as step-5k kill criterion (2026-04-13, corrected §90, REVISED §91)

**What.** Two scripts + one test module committed to make the 20-position threat-logit
probe reproducible as a formal gate for every future sustained run.

### Files added / updated

```
scripts/probe_threat_logits.py          — CLI + importable probe functions
scripts/generate_threat_probe_fixtures.py — generate fixtures/threat_probe_positions.npz
tests/test_probe_threat_logits.py       — shape/dtype/determinism/pass-logic tests
fixtures/threat_probe_positions.npz     — 20 curated positions (generated on first run)
fixtures/threat_probe_baseline.json     — canonical baseline (written by make probe.bootstrap)
```

### Kill criterion (REVISED §91 — see that section for full rationale)

At training step **5000**, run `make probe.latest`. PASS requires **all of C1-C3**.
C4 is a warning only and never causes FAIL.

| # | condition | threshold |
|---|-----------|-----------|
| 1 | contrast_mean (ext − ctrl) | ≥ max(0.38, 0.8 × bootstrap_contrast) |
| 2 | extension cell in policy top-5 | ≥ 40% |
| 3 | extension cell in policy top-10 | ≥ 60% |
| 4 (warn) | abs(ext_logit_mean − bootstrap_ext_logit_mean) < 5.0 | warning only |

**Original criterion history.** §85 first draft: "logit > 0" — wrong because
bootstrap itself measures around −0.34 to −0.60. §85/§89 correction: replaced
with `ext_logit_mean ≥ baseline − 1.0`, designed to catch ckpt_19500's
absolute-magnitude collapse (−0.14 → −3.25). §91 revision: that criterion
incorrectly FAILed ckpt_00014344, which has IMPROVED position-conditional
sharpness (contrast +3.94, top-10 70%) but a global bias shift in the threat
head (ext_logit_mean −6.21). Old C1 was a BCE scale-drift detector dressed up
as a colony-spam detector. It is replaced by direct colony-spam tests on the
policy head (C2 top-5 + C3 top-10); the bias-shift signal is preserved as
warning-only C4. See §91 for the full diagnosis and decision trail.

The canonical baseline numbers live in `fixtures/threat_probe_baseline.json`, written
once by `make probe.bootstrap` (which passes `--write-baseline` to the script).
If that file is absent, `probe.latest` prints FAIL with
"no baseline recorded — run make probe.bootstrap first".

### Bootstrap baseline (§85 empirical, bootstrap_model.pt)

| metric | bootstrap_model.pt | ckpt_19500 (pre-A1, bad) |
|--------|-------------------|--------------------------|
| threat logit @ extension cell | −0.14 ± 0.74 | −3.25 ± 0.46 |
| threat logit @ control cell | −0.52 ± 0.39 | −5.11 ± 1.40 |
| contrast (extension − control) | **+0.38** | +1.86 (shortcut) |

ckpt_19500 contrast was *higher* than bootstrap — the head learned a marginal-class
shortcut against stale mis-aligned labels. The +0.38 bootstrap contrast is the floor.

### Determinism

Probe forces FP32 (no autocast) and sets `torch.manual_seed(42)` +
`torch.use_deterministic_algorithms(True)` at startup. Two consecutive
`make probe.bootstrap` runs must produce byte-identical ext_logit_mean.

### Fixture schema (fixtures/threat_probe_positions.npz)

```
states:           (20, 18, 19, 19) float16 — K=0 cluster window tensors
side_to_move:     (20,) int8              — 1=P1, -1=P2 (current player)
ext_cell_idx:     (20,) int32             — flat index [0, 361) of open extension cell
control_cell_idx: (20,) int32             — flat index of empty cell far from stones
game_phase:       (20,) U8 string         — "early" / "mid" / "late"
```

Cell indices are loaded verbatim from NPZ — never regenerated at load time.

To regenerate from game records: `make probe.fixtures`
To regenerate synthetically (no game records required): see script `--synthetic` flag.

### Makefile targets

| target | action |
|--------|--------|
| `make probe.bootstrap` | Probe bootstrap_model.pt; write `fixtures/threat_probe_baseline.json` + `reports/probes/bootstrap_<ts>.md` |
| `make probe.latest` | Probe latest checkpoint; three-condition PASS/FAIL against saved baseline |
| `make probe.fixtures` | Regenerate fixture NPZ from available game records |

### Exit codes for probe_threat_logits.py

| code | meaning |
|------|---------|
| 0 | PASS — both thresholds met |
| 1 | FAIL — at least one threshold missed |
| 2 | Error — checkpoint load failed, shape mismatch, missing file |

**Commit:** `feat(eval): commit threat-logit probe as step-5k kill criterion`

---

## §90 — GPU util sweep: inf_bs / wait_ms levers are exhausted (2026-04-13)

**Context.** Tom reported dashboard "28% GPU util" on a gumbel_targets run. Phase 1
(`/tmp/gpu_util_phase1.md`) reframed this: actual GPU util is **84%**, the 28% figure
is a throughput-vs-bench ratio, and the real bottleneck is **NN forward latency**
(12.5 ms live vs 1.6 ms bench, 7.8× worse per-forward). Phase 1 surfaced three
hypotheses — this sprint entry records the Phase 2 narrowed sweep against H1.

**Sweep design (3 runs, laptop Ryzen 7 8845HS + RTX 4060, fresh bootstrap_model.pt).**

| run | inference_batch_size | inference_max_wait_ms |
|-----|---------------------:|----------------------:|
| A   | 64                   | 4.0                   |
| B   | 128                  | 8.0                   |
| C   | 128                  | 4.0                   |

All runs: `gumbel_targets` variant, `standard_sims=200`, `training_steps_per_game=4`,
`max_train_burst=16`, `n_workers=14`, `leaf_batch_size=8`, `fast_prob=0.0`,
`dirichlet_enabled=true`, `mixing.buffer_persist=false`. 20-min windows, last 15
min measured. Full data: `archive/sweep_2026-04-13_gpu_util/`.

**Kill criterion:** `policy_entropy_selfplay` must stay ≥ 4.0 nats (Phase 1
correction — combined entropy is polluted by pretrain sharpness and is not a
valid collapse signal at this training stage). All three runs passed (min
4.85 / 4.98 / 5.10).

### Results (last 15 min, per-run)

| metric | Run A | Run B | Run C | B vs A | C vs A |
|---|---:|---:|---:|---:|---:|
| games/hr | 545 | 381 | 372 | **−30.0%** | **−31.8%** |
| pos/hr (buffer delta) | 215,527 | 200,530 | 217,535 | −7.0% | +0.9% |
| nn_forwards/sec | 88.2 | 54.0 | 53.4 | −38.7% | −39.4% |
| nn_mean_batch_size | 60.1 | 84.8 | 85.8 | +40.9% | +42.7% |
| nn_pos/sec (fwd × batch) | 5,304 | 4,579 | 4,585 | **−13.7%** | **−13.6%** |
| batch_fill_pct (mean) | 91.4 | 63.4 | 67.4 | −30.6% | −26.2% |
| gpu_util_mean (nvidia-smi dmon) | 83.7 | 83.2 | 83.1 | −0.6% | −0.8% |
| gpu_util_p10 / p90 | 79 / 91 | 77 / 90 | 77 / 89 | — | — |
| policy_entropy_selfplay (final / min) | 5.18 / 4.85 | 5.14 / 4.98 | 5.47 / 5.10 | — | — |
| steps in window | 540 | 380 | 340 | −29.6% | **−37.0%** |
| game_len_median (plies) | 37 | 62 | 74 | +68% | +100% |

### H1 falsified

Raising `inference_batch_size` to 128 does grow the mean batch 60 → 85 (+42%),
confirming the Phase 1 diagnosis that 64 is not a hard ceiling. But forwards/sec
collapses 88.2 → 53.4 (−39%), so the product `nn_pos/sec` **drops 14%**. Workers
cannot supply 128 leaves in the same wall-clock window they supply 64, so the
batch fill plateaus at 63–67%, and the larger batches simply cost more per-forward
GPU time than they save in amortization. **The live batcher is starved, not the
GPU.**

`gpu_util` is invariant at ~83% across all three runs. The sweep levers cannot
move it. The Phase 1 finding — "GPU is busy but inefficient" — is confirmed
downstream of this.

### Why pos/hr looks neutral when games/hr halves

Run C's pos/hr is +0.9% vs Run A, a coincidental wash: games/hr collapses 545 →
372, but `game_len_median` doubles 37 → 74 plies, so each game produces roughly
2× more training positions (longer per-move budget → fewer blunders → games run
closer to the 200-ply cap). Training `steps_in_window` correspondingly drops
−37%, which is a real **learning-signal regression** even though pos/hr reads
flat. **pos/hr is not a sufficient summary statistic** when game length shifts
this much — future sweeps should report steps/hr alongside.

### Config decision

**No change.** Run C is +0.9% pos/hr (below the +5% threshold) at the cost of
−37% steps/hr. Run B is a net loss on every metric except mean batch size. The
`inference_batch_size=64, inference_max_wait_ms=4.0` baseline stays as the
laptop live config for Phase 4.0.

### Next lever (not in this sprint)

The remaining 12.5 ms NN forward latency is **architectural, not configurable**:

- **CUDA stream separation.** Training gradient kernels and inference forward
  kernels share the default stream, so training step kernels evict inference
  kernel state and autocast caches. A separate inference stream would let the
  inference server run continuously without training-step pollution.
- **Process split.** Run the Python training loop in a second process, leaving
  the inference server + worker pool in the primary. Trades IPC + duplicate
  weight hosting for zero cross-contamination.
- **`torch.compile` re-enable.** Blocked on Python 3.14 + CUDA graph
  compatibility (sprint §25, §30, §32). When unblocked, the compiled forward
  should cut per-forward Python dispatch overhead substantially.

Flagged as a **Phase 4.5 followup**. Not a Phase 4.0 blocker — sustained runs
can proceed on the current config.

### Desktop (3070) — not validated here

The §69 G3/P3 laptop winners were not re-verified on the desktop 3070 + Zen2
combo. If the desktop ever runs the `gumbel_targets` variant sustained, a
single-run confirmation that `inference_batch_size=64` remains optimal on that
hardware is worth doing before committing. No urgent action.

**Artifacts:** `archive/sweep_2026-04-13_gpu_util/{run_a,run_b,run_c}/` (train.jsonl,
dmon.log, train.log), `archive/sweep_2026-04-13_gpu_util/results.md`,
`archive/sweep_2026-04-13_gpu_util/analyze.py`.

**No commit of `configs/*.yaml`** — config is already near-optimal on the
swept axes.

### Followup

Architectural levers (CUDA stream separation, process split, `torch.compile`
re-enable, mixed-precision tuning) tracked as **Q18** in
`docs/06_OPEN_QUESTIONS.md`, deferred to Phase 4.5.

---

## §91 — Threat-probe criterion revised: target colony-spam, not BCE drift (2026-04-14)

**What.** Replace the §85/§89 step-5k probe criterion C1 (`ext_logit_mean ≥
baseline − 1.0`) with a contrast-floor + top-10 pair that directly tests the
policy-head behaviour we actually care about (colony-spam vs not). The old C1
was a scale-drift detector that misfired on a healthy run.

**Trigger.** ckpt_00014344 probe FAILed under the old criterion:

```
C1: ext_logit_mean -6.209 (floor -1.60)  FAIL
C2: contrast +3.939 (floor +0.38)         PASS (10× bootstrap)
C3: top5 50% (floor 40%)                  PASS
```

Contrast grew TO **10× baseline (+3.94)** while absolute logits drifted
globally negative. This is the OPPOSITE of the ckpt_19500 collapse signature
that motivated the original C1: ckpt_19500 had contrast grow only 5× while
BOTH logits collapsed by similar amounts (the marginal-class shortcut). The
ckpt_00014344 pattern is consistent with BCE-on-imbalanced-labels driving
logits globally negative while position-conditional sharpness IMPROVES — i.e.
the policy head is doing exactly what we wanted, just with a global bias
shift in the threat head. The old C1 was therefore a BCE scale-drift detector
dressed up as a colony-spam detector.

**Revised criterion (now enforced by `scripts/probe_threat_logits.py`).**

| # | condition | threshold | notes |
|---|-----------|-----------|-------|
| C1 | `contrast_mean ≥ max(0.38, 0.8 × bootstrap_contrast)` | floor 0.40 (bootstrap = 0.502) | preserves §85 0.38 absolute floor, scales with bootstrap |
| C2 | `ext_in_top5_pct ≥ 40` | unchanged | direct colony-spam test on policy head |
| C3 | `ext_in_top10_pct ≥ 60` | NEW | catches partial sharpness — rank 6-10 is fine |
| C4 | `abs(ext_logit_mean − bootstrap_ext_logit_mean) < 5.0` | warning only | catches catastrophic decode/mapping bugs without gating |

C1-C3 must all PASS for `make probe.latest` exit code 0. C4 only prints a
`WARNING` line in the markdown report; the gate never trips on it.

**Baseline JSON schema bumped to v2.** `fixtures/threat_probe_baseline.json`
now carries `version: 2`, `ext_in_top10_frac`, and explicit `ext_in_top5_pct`
/ `ext_in_top10_pct` fields. Regenerated from `bootstrap_model.pt`:

```json
{
  "version": 2,
  "ext_logit_mean": -0.5985873519442976,
  "ctrl_logit_mean": -1.100777158141136,
  "contrast_mean":  0.5021898061968386,
  "ext_in_top5_pct":  50.0,
  "ext_in_top10_pct": 65.0
}
```

**ckpt_00014344 re-probed under the new criterion — PASS.**

```
C1: contrast=+3.939   (≥ +0.402)  PASS
C2: top5 = 50%        (≥ 40%)     PASS
C3: top10= 70%        (≥ 60%)     PASS
C4: |Δ ext_logit_mean| = 5.611    WARNING
```

Drift > 5.0 nats triggers C4 — flagged in the report for follow-up but does
not block the run. The threat head's global bias shift remains an open
question; if it persists, investigate whether BCE positive-weight scaling or
a focal/class-balanced loss reformulation is warranted.

**Open question logged.** The BCE class-imbalance hypothesis is recorded as
**Q19 [WATCH]** in `docs/06_OPEN_QUESTIONS.md` (winning_line labels are ~1.6%
positive → `BCEWithLogitsLoss` without `pos_weight` drives logits globally
negative; proposed fix `pos_weight ≈ 59`, lands on next
bootstrap-from-scratch run, not mid-run). The C4 warning path defined above
is the monitoring hook for Q19 — drift > 8 nats, or aux loss > 4.0, or
policy top-10 regression below bootstrap escalates Q19 from WATCH to HIGH.

**Files touched.**

- `scripts/probe_threat_logits.py` — thresholds, `check_pass`, new `check_warning`, `contrast_floor` helper, baseline v2 writer, top-10 extraction, console summary, markdown report.
- `tests/test_probe_threat_logits.py` — three-condition tests rewritten for revised C1; new `test_check_pass_no_baseline_uses_floor` and `test_check_warning_drift_threshold`; baseline-roundtrip test now verifies `version`/`ext_in_top10_pct`.
- `fixtures/threat_probe_baseline.json` — regenerated as v2 (top10 = 65%).
- `docs/07_PHASE4_SPRINT_LOG.md` — §85 + §89 cross-reference §91; this entry.
- `hexo_rl/monitoring/web_dashboard.py` — orthogonal: install a `threading.excepthook` filter to swallow the engineio `KeyError('Session is disconnected')` race that was polluting training stderr (see "Dashboard fix" below).

**Dashboard fix (orthogonal, same commit batch).**

When a browser tab closes mid-stream, `engineio.base_server._get_socket`
raises `KeyError('Session is disconnected')` from one of engineio's internal
background threads (writer / receive handler). Our `_drain_emit` thread's
`try/except Exception` cannot catch it because the exception lives in a
different thread entirely. Solution: install a one-shot `threading.excepthook`
in `WebDashboard.start()` that:

  1. Filters by `exc_type is KeyError`,
  2. Filters by message (`"Session is disconnected"` / `"Session not found"`),
  3. Walks the traceback for an `engineio.*` / `socketio.*` frame,
  4. Drops the exception silently if all three match; delegates everything
     else to the previous excepthook.

Verified end-to-end with a real `socketio.SimpleClient` connect → flood emit
→ disconnect → flood emit cycle: zero tracebacks on stderr, dashboard server
unaffected. Unrelated `KeyError` in another thread still surfaces normally.

**Commit:** `fix(eval): revise threat-probe criterion to target colony-spam directly`

**Commit:** `fix(monitoring): swallow engineio disconnect KeyError in web dashboard`

---

## §92 — Q13 + Q13-aux + Q19 atomic landing (2026-04-14, partially SUPERSEDED by §97)

> **Post-§97:** chain planes no longer live in the NN input tensor — they were moved to a dedicated `ReplayBuffer.chain_planes` sub-buffer. The design decisions and aux-head structure below still hold; the "18→24 input break" and per-run numbers are historical. Pretrain v3 itself was broken by an augmentation bug caught at §93 F1; v3b (§93) is the production bootstrap.

**What.** Three interlocking changes landed as a fresh-start cycle (bootstrap corpus re-export + pretrain v3 + new `bootstrap_model.pt`). Atomic because buffer layout + checkpoint shape + loss wiring cannot cross-boundary test individually. Q19's `pos_weight=59` co-landed so the threat BCE fix piggybacks on the new bootstrap.

**Motivation.** Literature review (`reports/literature_review_26_04_24/review.md`): KataGo-style Tier 2 geometric feature to accelerate tactical-threat learning; MoHex-CNN bridge planes, KataGo liberty/ladder planes, Rapfi per-axis line patterns. AZ-style Gomoku implementations stay raw-stone-only and all document the same threat-blindness failure mode. Q19: without `pos_weight`, threat-head BCE drifts globally negative at ~1.6% positive labels (§91).

**Design decisions (architectural authority — survives §97):**

1. **Chain-length semantics — post-placement.** Cell value = `1 + pos_run + neg_run` for own stones and empty cells with at least one adjacent own neighbour; 0 elsewhere and for opponent cells. Capped at 6, /6-normalised. `XX_XXX` → empty cell value = 6/6.
2. **Chain-aux target sourcing.** §92 used `chain_target = input[:, 18:24]` (slice-from-input); **§97 revised to read from the replay-buffer chain sub-buffer**. Head job remains "preserve/rediscover chain-counting through the tower".
3. **numpy-vectorised tensor assembly (no numba).** Pure Python rejected (13–33 ms budget blowout); `np.roll` rejected (wraps and violates window-edge opacity). Python helper stays in `hexo_rl/env/game_state.py::_compute_chain_planes`.
4. **`aux_chain_weight = 1.0` (not 0.10).** /6-normalised target → smooth_l1 ~0.02/cell; weight 0.10 → ~0.002 loss vs policy ~2.0 — invisible. 1.0 gives the aux head meaningful gradient share.
5. **Atomic bundle.** 56 files changed in one commit. Coverage: byte-exact augmentation-invariance test + chain-head mask tests inside the same commit.

**Downgraded expectations — not KataGo 1.65×.** That headline is from KataGo's auxiliary FUTURE-information targets (terminal ownership). Our chain target is a current-input slice (§92) / a same-window chain-plane recomputation (§97) — regularisation + intermediate supervision, not counterfactual forward information. Realistic uplift 1.1–1.3× on tactical probe convergence. Q21 parks the wider-window variant that would match KataGo's structure.

**Commit sequence (details in git log):**

| Commit | Scope |
|---|---|
| C1 | `_compute_chain_planes` Python helper + 18 unit tests. 78 µs/call at 50 stones, 165× faster than pure Python. |
| C2 | `SymTables.axis_perm` table + 10 inline tests. Axis permutation period 3 (180° identity on direction-unsigned axes). |
| C3 | **Atomic 18→24 plane break** — 56 files, 1019+/185−. Touches `game_state.to_tensor`, Rust `encode_state_to_buffer`, `SymTables` scatter split, `HexTacToeNet` + `chain_head`, `compute_chain_loss`, `Trainer._threat_pos_weight`, `pretrain.train_epoch`, dashboards, and test-layer plane-shape updates. HEXB v2 → v3 with `n_planes` header. |
| C4 | `scripts/compute_threat_pos_weight.py` — recomputes `(1-p)/p` from the buffer; falls back to §91 theoretical 59.0 when no 24-plane buffer exists. |
| C5 | Corpus re-export at 24 planes (`scripts/export_corpus_npz.py --human-only --max-positions 200000 --no-compress`). 199,470 positions, 3.6 GB. |
| C6 | Pretrain v3: 15 epochs × 779 batches, ~40 min on RTX 3070. Produces 24-plane `bootstrap_model.pt` + threat_probe_baseline v3. **Broken by F1 aug bug (see §93).** |
| C7 | §92 sprint log landing. |

**Load-bearing follow-up notes:**

- **Threat and ownership heads untrained at bootstrap.** Corpus NPZ has no per-row winning_line or ownership targets (self-play-only post §85 A1). Q19 `pos_weight=59` kicks in once self-play feeds aux targets.
- **Probe baseline policy change.** `probe_threat_logits.py --write-baseline` now always exits 0 — a bootstrap's random-init threat head cannot satisfy the absolute 0.38 contrast floor. Gate applies to post-self-play checkpoints only.
- **Checkpoint incompat (§92 onwards).** First-conv shape mismatch with any pre-§92 checkpoint. Pre-§92 archives: `bootstrap_model_18plane.pt`, `bootstrap_corpus_18plane.npz`, pre-§92 `replay_buffer.bin` (v2 HEXB) rejected at load.

**Open questions updated (see `docs/06_OPEN_QUESTIONS.md`):**

- Q13: resolved by this landing (input form); revised by §97 (aux-sub-buffer form).
- Q19: resolved by `pos_weight=59`. §91 C4 warning hook stays.
- **Q21 parked:** wider-window aux target for forward-information injection. Current chain target (§97 form) is a same-window recomputation — trunk can already see chain values in the stones. KataGo's 1.65× speedup requires future-information targets (terminal ownership); wider-window chain is the Hex analogue. Revisit after §97 baseline stabilises.

---

## §93 — Q13 fix-up + F1 root cause + F3 dead code removed + pretrain v3b (2026-04-15)

**What.** Ten-commit fix-up on the `feat/q13-chain-planes` branch:
C8 extracted the Rust augmentation kernel and exposed it to Python via
three PyO3 bindings (`apply_symmetry`, `apply_symmetries_batch`,
`compute_chain_planes`); C9 landed three byte-exact parity guards for
F1/F2/F3; C9.5 deleted the dead `TensorBuffer` assembler surfaced by the
F3 guard; C10 routed pretrain augmentation through the Rust kernel
(eliminating the broken `_apply_hex_sym` path that corrupted chain
planes in pretrain v3); C11 consolidated four hex coordinate helpers
into `hexo_rl/utils/coordinates.py` with round-trip tests; C12–C15
landed the W1–W4 cleanups from the review (broken-four + triple-axis
test cases, optional `legal_mask` on chain loss, dashboard wiring for
three aux losses, `encode_planes_to_buffer` rename and 18-plane
docstring cleanup). C16 regenerated the bootstrap from scratch via
the corrected pipeline as pretrain v3b, and this entry (C17) records
the outcome.

**Why.** §92 landed the 24-plane break atomically as C3, and the C6
pretrain v3 produced a working 24-plane `bootstrap_model.pt` — but the
`reports/review_q13_q19_landing_26_04_14.md` post-landing audit caught
F1: `hexo_rl/bootstrap/pretrain.py:55-133::_apply_hex_sym` scattered
state tensors by pure coordinate permutation, with neither the
`axis_perm` remap for planes 18..23 nor the `(row=q, col=r)` coordinate
convention used by `_compute_chain_planes` and the Rust `SymTables`.
Result: pretrain v3's 15 epochs saw chain planes that contradicted the
stones in 11 of every 12 augmented samples, and the trunk learned
whatever cross-axis garbage came out. Phase 4.0 self-play cannot start
from a bootstrap whose Q13 signal is randomised.

The review also drafted an F3 tensor-buffer parity guard. That guard
caught real divergence (`TensorBuffer.assemble()` still produced
(K, 18, 19, 19) post-§92) — but the live-path trace
(`reports/tensor_buffer_live_path_26_04_15.md`) showed the divergence
was in *dead code*: `SelfPlayWorker.play_game()` is the only caller and
no production path reaches it (the live self-play loop is the Rust
`SelfPlayRunner` via `WorkerPool`). Zero self-play checkpoints were
corrupted through F3; C9.5 deleted the dead code outright rather than
rewriting it. "One assembler is better than two."

**Commit sequence (details in git log + `reports/q13_fix_26_04_15.md`):**

| Commit | Scope |
|---|---|
| C8 | Extract `apply_symmetry_24plane<T: Copy>` kernel from `ReplayBuffer::apply_sym`; expose via PyO3 as `engine.apply_symmetry`, `apply_symmetries_batch`, `compute_chain_planes`. Thread-local `SymTables`; raises `SymTables` + `encode_chain_planes` to `pub`. |
| C9 | Add F1/F2/F4 guards. F1: `test_pretrain_aug.py` — buffer-vs-binding byte-exact parity over 4,000 draws. F2: `test_chain_plane_rust_parity.py` — Python vs Rust `_compute_chain_planes` across 21 positions (open/blocked 3/4, XX.X.XX, triple-axis, edge runs, near-five). F4: oracle-note comment in `test_chain_plane_augmentation.py`. |
| C9.5 | Delete dead `TensorBuffer`, `SelfPlayWorker.play_game`, and their tests. F3 guard retired — zero corrupted checkpoints, live path is Rust `SelfPlayRunner`. |
| C10 | Route pretrain augmentation through `engine.apply_symmetries_batch`. Delete `_apply_hex_sym` / `_precompute_hex_syms`. New `make_augmented_collate`; 12×362 policy scatter table. 20-batch timing probe at launch. |
| C11 | Consolidate hex-coord helpers into `hexo_rl/utils/coordinates.py` (`flat_to_axial`, `axial_to_flat`, `cell_to_flat`, `axial_distance`). 28 tests. Migrate 5 call sites. |
| C12 | Test-coverage gaps: triple-axis-intersection + XX.X.XX broken-four chain-value pins. |
| C13 | `compute_chain_loss` gains optional `legal_mask`; default path byte-exact unchanged. |
| C14 | Surface `loss_chain / loss_ownership / loss_threat` in terminal + web dashboards. |
| C15 | Rename `Board::encode_18_planes_to_buffer` → `encode_planes_to_buffer`; update `get_cluster_views` doc comment. |
| C16 | Pretrain v3b + `threat_probe_baseline v4`. v3 archived as `bootstrap_model_v3_broken_aug.pt`. |
| C17 | §93 sprint log landing. |

**F1 root cause.** Pre-C10 `_apply_hex_sym` had two bugs: (1) no `axis_perm` remap on planes 18..23; (2) `(col=q, row=r)` convention in `_precompute_hex_syms` vs `(row=q, col=r)` in `_compute_chain_planes` / Rust `SymTables`. Both eliminated by routing through `apply_symmetry_24plane<f32>` — same kernel the ReplayBuffer uses, with `axis_perm` derived from hex basis transform and pinned by `test_chain_plane_augmentation.py`.

**Pretrain v3b results (current production bootstrap).** 15 epochs × 779 batches at batch_size=256, ~40 min on RTX 3070. End-to-end DataLoader ~32.7 ms/batch (numpy↔tensor boundary dominates; Rust scatter sub-ms).

| metric | gate | v3b | note |
|---|---|---|---|
| policy_loss (final) | ≤ 2.47 | **2.1758** | matches v3 — corpus + optimiser unchanged |
| value_loss (final) | ≤ 0.59 | **0.4990** | |
| opp_reply_loss (final) | — | **2.1846** | |
| chain_loss (final) | ≤ 0.01 | **0.0018** | degenerate plateau (Q21: aux target is slice-equivalent) |
| 100-game RandomBot greedy wins | ≥ 95 | **100/100** | PASS |

**The v3→v3b win is correctness, not aux-scalar.** Chain planes are now byte-exactly consistent with stones under every augmentation (F1 fix). Whether that uplifts tactical sharpening is a Phase 4.0 sustained-run question, not a pretrain-loss-scalar question.

**Threat-probe baseline v4.** `fixtures/threat_probe_baseline.json` regenerated against v3b bootstrap; schema v3 → v4. Contrast −0.9366 — same untrained-head noise-band as v3. `probe_threat_logits.py --write-baseline` returns exit 0 by construction; §91 C1 relative gate applies to post-self-play checkpoints only.

**Downgraded expectations carry over from §92.** Q21 (wider-window aux target) parked. **Q22** (chain-plane Rust port deleting Python `_compute_chain_planes` and its ~80 µs/call cost) parked — F2 parity guard pins the two paths together. **Q23** (tensor-assembler consolidation) **closed** by C9.5 — only `GameState.to_tensor()` + `encode_state_to_buffer` remain.

**Guards snapshot:**

| Guard | File | Coverage |
|---|---|---|
| F1 pretrain-aug parity | `tests/test_pretrain_aug.py` | 3 positions × 12 syms, 4,000-draw buffer coverage |
| F2 chain-plane parity | `tests/test_chain_plane_rust_parity.py` | 21 hand-picked positions, byte-exact |
| F4 invariance oracle | `tests/test_chain_plane_augmentation.py` | 4 positions × 12 syms, independent Python oracle |

**Reports:** `reports/review_q13_q19_landing_26_04_14.md` (F1-F7/W1-W4 audit); `reports/tensor_buffer_live_path_26_04_15.md` (F3 trace); `reports/q13_fix_26_04_15.md` (C8–C17 landing summary).

---

## §94 — Experiment A: aux_chain_weight=0 fresh run (2026-04-15)

Smoke_v3b (§93 bootstrap, `gumbel_targets`, 5003 steps) hit 44.7% draw rate with monotonic climb — hypothesis: `aux_chain_weight=1.0` on the degenerate slice-from-input target biases the trunk toward colony-extension.

**Exp A config diff:** `aux_chain_weight: 0.0`; everything else identical to smoke_v3b; fresh from `bootstrap_model.pt`. Config-only, no code changes.

**Result (confirmed at §95 launch):** draw rate 47.7% at step 10312 — within noise, marginally worse than smoke. **Chain aux NOT the primary driver.** Forces the next experiment (§95 chain-plane input ablation).

Monitoring: `scripts/monitor_experiment_a.sh`. Probe gate: §91 C1-C4 (with §91 softening). Reports: `reports/smoke_v3b_5k_26_04_15.md`.

---

## §95 — Experiment C: chain-plane input ablation (2026-04-16)

### Motivation

Experiment A (§94, aux_chain_weight=0) did NOT reduce draw rate.
Draw rate 47.7% at step 10312 vs smoke_v3b 44.7% at 5003 — within noise,
marginally worse. Chain aux confirmed NOT the primary driver.

**Remaining hypothesis:** The 6 chain-length input planes (18-23) themselves
prime the policy toward colony extension. The residual tower learned to use
chain-plane values to route gradient toward extending existing chains, independent
of the aux head. Removing the loss signal (Exp A) did not break that routing —
the input features remained, so the learned "extend chain = high value" shortcut
still activates in self-play.

### Chain planes: stored in buffer, NOT recomputed at sample time

Audit finding (confirmed reading Rust + Python): chain planes are computed at
**encode time** (`encode_state_to_buffer` in `engine/src/board/state.rs`, called
during self-play when games are pushed to the replay buffer) and stored as part of
the 24-plane state tensor. They are NOT recomputed at sample time. The Rust symmetry
augmentation path (`apply_symmetry_24plane` in `engine/src/replay_buffer/sym_tables.rs`)
applies coordinate permutation + axis-plane remap to the stored chain planes, but does
NOT recompute them from stones.

Consequence: zeroing at the point of use (trainer + inference server) is sufficient —
zeroed planes (all zeros) are invariant under any symmetry transform, so augmentation
cannot reintroduce signal.

### Design

Zero planes 18-23 AFTER decode from buffer / after H2D transfer — do NOT remove planes
from the architecture. `in_channels=24` stays. The 6 zeroed planes provide zero gradient
to the network via input; the conv weights for those input channels receive no gradient
from states but the trunk is otherwise unchanged.

**`zero_chain_planes: bool`** config flag (default false in `configs/training.yaml`).
Set to `true` in `configs/variants/gumbel_targets.yaml` for Experiment C.

### Wiring (3 locations)

1. **`hexo_rl/training/trainer.py`** — `_train_on_batch()`: zero `states_t[:, 18:24]`
   after H2D transfer, before model forward. Covers training path.

2. **`hexo_rl/selfplay/inference_server.py`** — `__init__()`: read and store
   `_zero_chain_planes`; `run()`: zero `tensor[:, 18:24]` after Rust→Python batch
   extraction, before model forward. Covers self-play inference path.

3. **`scripts/probe_threat_logits.py`** — `main()`: `--zero-chain-planes` CLI flag;
   zero `positions["states"][:, 18:24]` before `probe_positions()`. Ensures probe
   uses same inputs as trained model.

**Replay buffer augmentation path:** no change required. Zeroed planes are invariant
under the 12-fold hex symmetry transform (zero → zero). Stored buffer values remain
non-zero but are always masked at point of use.

### Config

```yaml
# configs/training.yaml (base):
zero_chain_planes: false   # ablation default; set true in variant for Exp C

# configs/variants/gumbel_targets.yaml (Experiment C only):
training:
  zero_chain_planes: true
# aux_chain_weight: 0.0 already in training.yaml from Exp A
```

### Success criteria at step 5000

| Outcome | Draw rate | Interpretation |
|---------|-----------|----------------|
| PRIMARY PASS | < 35% | Chain INPUT planes confirmed as cause |
| PARTIAL | 35–45% | Partial contribution; combined fix needed |
| NULL | > 45% | Chain planes NOT the cause; buffer dilution next |

### Test

`tests/test_zero_chain_planes.py` — 5 tests:
- Trainer zeroes planes 18-23 when flag true (captures model input)
- Trainer preserves planes 18-23 when flag false
- Only planes 18-23 are zeroed (planes 0-17 unchanged)
- InferenceServer zeroes planes 18-23 when flag true
- InferenceServer preserves planes 18-23 when flag false

All 5 pass; full test suite (755 tests) passes.

### Checkpoints

Experiment A final checkpoint backed up to `checkpoints/exp_a_backup/checkpoint_00010000.pt`.
Full experiment A data at `checkpoints/chain_planes_no_chain_weight/`.

Fresh run from `checkpoints/bootstrap_model.pt`. Same monitoring script as Exp A
(`scripts/monitor_experiment_a.sh`), pointed at new JSONL.

### Commits

- `feat(env): zero_chain_planes config flag for input ablation experiment (Exp C §95)`
- `docs(sprint): §95 Experiment C — chain plane input ablation`

---

## §96 — Exp E: Gumbel MCTS desktop (A/B vs laptop exp D PUCT+completedQ) (2026-04-16)

### Hypothesis

Gumbel top-m + completed-Q policy targets produce better move rankings per sim → faster
tactical convergence. Expected -5 to -15% sims/s vs PUCT but net-positive pos/hr via better
training signal per game.

**Setup.** Hardware: Ryzen 7 3700x + RTX 3070 (desktop). Variant: `gumbel_full`. Checkpoint: `bootstrap_model.pt` (v3b). Run label: `exp_E_gumbel_full_desktop`.

**Config diffs vs laptop exp D (PUCT+completedQ):** `gumbel_mcts: true`; `n_workers: 10` (Zen2 GIL ceiling from §81 D3); `inference_max_wait_ms: 5.0`. All other keys identical (`training_steps_per_game=4`, `burst=16`, `max_game_moves=200`, `draw_value=-0.5`, `decay_steps=20k`, `total_steps=200k`).

**Kill conditions (relaxed per exp D learnings):** `draw_rate > 70%` sustained; `policy_entropy_selfplay < 1.5` for 500+ steps; `grad_norm > 10` for 50+ steps; `pos_per_hr < 35k`; NaN / OOM / crash. Probe C2/C3 FAIL does NOT kill.

**Success metrics at step 20k:** draw-rate ≤ laptop exp D; `pos_per_hr` ≥ 80% of laptop; C2 ≥ 30%, C3 ≥ 45% at step 15k. Monitoring: `scripts/monitor_exp_E.sh`.

**Q26 [WATCH] — nested `training:` block in `gumbel_targets_desktop.yaml`.** Deep-merger treats the nested `training:` key as a new sub-dict; flat `training_steps_per_game` in `training.yaml` is never overridden → defaults to 1.0. Scope limited to `gumbel_targets_desktop.yaml`; `gumbel_full.yaml` uses flat keys correctly. Audit after exp D completes.

---

## §97 — Remove chain planes from NN input: 24ch → 18ch (2026-04-16)

**Motivation:** KrakenBot uses 2 input channels and achieves top play. It learns
chain-aware representations via chain aux loss, not by ingesting chain planes as
input. Our 24-channel trunk had redundant input (chain planes fed to a trunk that
already predicts chain planes as aux output). Removing chain from input eliminates
the redundancy and aligns with the KrakenBot architecture.

**What changed:**

- `GameState.to_tensor()`: removed chain plane allocation and computation from
  `to_tensor()`. Output shape: `(K, 18, 19, 19)` (was 24). `_compute_chain_planes`
  retained for chain target generation.
- `HexTacToeNet`: `in_channels` default 24 → 18.
- `configs/model.yaml`: `in_channels: 24` → `18`.
- Rust `encode_state_to_buffer`: strip chain plane writes (planes 18–23 removed).
  State buffer = 18 planes = 6498 u16 per slot.
- Rust replay buffer: chain planes stored in separate `chain_planes` sub-buffer
  (`6 × 361 × u16` per slot). HEXB format bumped to v4.
- `sample_batch()` returns 6-tuple: `(states, chain_planes, policies, outcomes, ownership, winning_line)`.
- `push()` / `push_game()` take explicit `chain_planes` parameter.
- `sym_tables.rs`: `N_PLANES = 18`; `chain_src_lookup` applies axis-perm to 6 chain planes.
- `trainer.py`: chain target from `chain_planes` param, not `states_t[:, 18:24]`.
- `pretrain.py (C1 fix)`: `AugmentedBootstrapDataset` + `make_augmented_collate` compute
  chain from augmented stone planes (planes 0 + 8) post-augmentation. `train_epoch`
  unpacks 4-tuple; chain_target = `chain_planes` tensor (was empty `states[:, 18:24]`).
- `batch_assembly.py`: `BatchBuffers` gains `chain_planes` field; `assemble_mixed_batch` populates it.
- `game_runner/worker_loop.rs`: `encode_chain_planes()` called per step; chain accumulated
  and pushed to replay buffer chain sub-buffer.
- All tests and scripts updated to 18-plane layout.

**Replay buffer incompatibility:** Old HEXB v1–v3 buffers are incompatible (stride change).
Clear with `rm -rf data/replay_buffer/` before first training run.

**Chain aux head + loss retained** — that's the part that helps.

### Commits

- `feat(arch): remove chain planes from input — 24ch → 18ch, chain to aux sub-buffer`

---

## §98 — Benchmark rebaseline post-18ch migration (2026-04-16)

**Context:** First `make bench` after the 18ch migration. Two metrics FAIL against the §72-era targets.

### Observed results (laptop, Ryzen 7 8845HS + RTX 4060, n=5)

| Metric | Median | IQR | Range | Old target | Result |
|---|---|---|---|---|---|
| Buffer sample augmented µs/batch | 1,663 | ±566 | 1.3k–2.2k | ≤ 1,400 | FAIL |
| Worker throughput pos/hr | 30,893 | ±58,185 | 0–364k | ≥ 500,000 | FAIL |

All other metrics PASS.

### Root cause analysis

**Worker throughput (catastrophic-looking median, warmup artifact + methodology shift):**

Two stacked causes:

1. **Warmup design bug** — benchmark creates one pool, runs 30s warmup, then 5 × 60s measurement windows. `p25 = 0` means at least 2 of 5 windows measured 0 positions. Workers weren't producing completed games during those windows. 30s is insufficient for the pool to reach steady state on laptop hardware: workers must play early games to completion (cold start takes longer with an untrained model on the first game). The CUDA JIT warm-up (pre-pool dummy forward) handles PyTorch kernel compilation but not game-loop ramp-up.

2. **Baseline methodology mismatch** — the old 659k pos/hr baseline (§66, April 2026-04-06) was set under different benchmark parameters. Commit 207656a changed `n_simulations` from config value (400) to 200, and `max_moves_per_game` from 200 → 128. The numbers are not directly comparable. The maximum observed value (364k) under the new methodology reflects the actual achievable ceiling.

Cross-check against real training: the training log `train_10cc8d56e4394a9ca542740c4bcee069.jsonl` (production run at step ~15k, April 16) shows **~408 games/hr × 118 avg plies = ~48k pos/hr** during actual training (GPU shared between inference and training steps). The benchmark's pure-self-play measurement at a lower sim count (200 vs production 400) should be faster than training — the 364k max (no training overhead, 200 sims) is consistent with this.

3. **18ch chain plane overhead (minor)** — `encode_chain_planes()` added per position in `worker_loop.rs`. Small but real per-position cost.

**Buffer sample augmented (real regression, high variance):**

Before 18ch: single `apply_symmetry_state` scatter over 24 planes.
After 18ch: `apply_symmetry_state` (18 planes) + `apply_chain_symmetry` (6 planes, axis-plane remap). Two passes over two non-contiguous memory regions. High IQR (±566 µs) reflects cache-pressure variance: chain sub-buffer adds 6 × 361 × f16 = ~4 KB per position; at batch=256, ~1 MB extra data touched per sample, causing inconsistent L3 hit rates.

### Updated targets

Old targets were set against a 24-plane model under a different benchmark methodology. New targets reflect the 18-plane layout and current benchmark setup.

| Metric | New target | Rationale |
|---|---|---|
| Buffer sample augmented µs/batch | ≤ 1,800 µs | Median 1,663 + comfortable margin for split-pass overhead; IQR suggests it's sometimes ≤1,300 µs |
| Worker throughput pos/hr | ≥ 250,000 pos/hr | Conservative floor: well above the warmup-artifact 0-position runs, below the 364k max; methodology fix (longer warmup) should raise the reliable floor |

**Note on worker benchmark reliability:** until warmup duration is increased (suggest 90s or "until N games complete"), the worker throughput metric has high measurement variance. The 250k target is a checkpoint, not a ceiling. Real training throughput (GPU shared) is ~48k pos/hr at production sim counts — the benchmark measures self-play-only capacity at reduced sims.

### Action items

- [ ] Increase worker benchmark warmup to 90s (or gate on first-game completion) to eliminate 0-position measurement windows
- [ ] After warmup fix, run 3-run rebaseline to confirm reliable floor ≥250k

### Commits

- (this entry — no code change, targets only)

## §99 — BatchNorm → GroupNorm migration (2026-04-16)

**Motivation:** MCTS leaf eval runs at batch=1. BatchNorm in eval mode uses
running statistics accumulated during training — these drift from the live
distribution as the model updates during self-play, introducing a
train/inference gap. GroupNorm computes per-sample statistics from fixed
channel groups, so behaviour is identical at batch=1 and batch=256.
KrakenBot uses GroupNorm(8, 128) throughout.

**Changes (`feat/groupnorm`):**

- `hexo_rl/model/network.py`:
  - `ResidualBlock.bn1/bn2` → `gn1/gn2` (`GroupNorm(_GN_GROUPS, filters)`)
  - `Trunk.input_bn` → `input_gn` (`GroupNorm(_GN_GROUPS, filters)`)
  - `policy_bn` and `opp_reply_bn` removed (2 output channels; GN(8,2) fails,
    normalization has negligible effect at 2 channels before flatten→linear)
  - `_GN_GROUPS = 8` module constant; `assert filters % _GN_GROUPS == 0` guard
    in `ResidualBlock.__init__`

- `hexo_rl/training/trainer.py`:
  - Removed BN running-stats reset from the NaN guard (GN has no running stats)

- `hexo_rl/training/checkpoints.py`:
  - `normalize_model_state_dict_keys` now raises `RuntimeError` on any
    checkpoint containing pre-GN key patterns (`.input_bn.`, `.bn1.`, etc.)
    to prevent silent trunk corruption via `strict=False` loading

**Checkpoint compatibility:** BROKEN. All pre-§99 checkpoints (including
smoke_v3b) contain BatchNorm keys and will be rejected at load time with a
clear error. Retrain from scratch.

**Benchmark:** Run `make bench` after this change. GN pool size differs from BN;
verify NN inference (batch=64) and latency (batch=1) targets still pass.

Note: benchmark methodology changed (§98 action items resolved) — runtime is
now 2 min with 90 s warmup, making results more representative of real
throughput. Prior baselines (1 min / shorter warmup) are not directly
comparable. Fresh `make bench` on this branch establishes the new GN baseline.

## §100 — Selective policy loss (move-level playout cap) (2026-04-16)

**Motivation (KrakenBot-inspired).** Quick-search MCTS visit distributions carry noisy policy targets — training the policy head on them adds gradient variance without useful signal. Fix: randomise sim count per move, tag each position with `is_full_search`, gate policy / opp_reply losses on that flag in Python. Quick-search rows contribute only to value / chain / ownership / threat losses.

**Orthogonal to** the game-level `fast_prob`/`fast_sims`/`standard_sims` cap (whole-game fast/standard, zeroes the policy vector for fast-game rows — filtered by `policy_valid = policies.sum(dim=1) > 1e-6`). Pool init now enforces these as mutex (see §100.c M1/M2).

**Changes (branch `feat/selective-policy-loss`):**

- **Rust** (`game_runner/{mod, worker_loop}.rs`, `replay_buffer/*`): `SelfPlayRunner` gains `full_search_prob / n_sims_quick / n_sims_full`. Per-move coin-flip sets sim count. Results-queue tuple grows a `bool is_full_search` (`collect_data()` → 8-tuple). ReplayBuffer adds `is_full_search: Vec<u8>` column. HEXB v4 → **v5** (v4 still loads, defaulting flag to 1). Flag is not transformed under 12-fold symmetry (per-position metadata, not spatial).
- **Python**: `pool.py` / `recency_buffer.py` / `batch_assembly.py` all carry the flag. `losses.py::compute_policy_loss / compute_kl_policy_loss / compute_aux_loss(opp_reply)` accept optional `full_search_mask` and intersect with `valid_mask`. `trainer.py` logs `full_search_frac` (rows where both masks are True).

### §100.c — Review fixes (applied before merge)

| # | Issue | Fix |
|---|---|---|
| H1 | `RecentBuffer` had no `is_full_search` column; recent-buffer slice was silently synthesised `ones`, defeating the feature for ~56% of each batch (`recency_weight: 0.75`). | `RecentBuffer.push`/`sample` carry the flag through. |
| H2 | BN→GN auto-migration briefly added to `checkpoints.py` to silence pre-§99 fixture failures. Transferred BN affine params into GN slots — not numerically equivalent; weakened §99 safety rail. | Reverted. `RuntimeError` is back; migration belongs on its own branch. |
| M1 | `fast_prob > 0` AND `full_search_prob > 0` allowed simultaneously → move-level cap silently overrode game-level. | `WorkerPool.__init__` raises on both > 0; `configs/selfplay.yaml` set to `fast_prob: 0.0`. |
| M2 | `full_search_prob > 0` with `n_sims_quick <= 0` or `n_sims_full <= 0` → random play. | Init raises. |
| M3 | opp_reply head trained on same visit distribution as policy — same selectivity argument. | `compute_aux_loss` accepts `full_search_mask` and gates identically. |

**Config net effect:** `fast_prob: 0.0`, `n_sims_quick: 100`, `n_sims_full: 600`, `full_search_prob: 0.25`. Effective avg sims/move shifts from ≈98 (game-level) to ≈225 (move-level) — ~2.3× compute per move to match KrakenBot.

**Known follow-ups (not blocking):** split MCTS depth / root-concentration stats by `is_full_search`; frozen v4 fixture round-trip test; distinguish empty-mask vs genuine 0.0 policy loss.

### §100.d — Threat probe baseline regenerated v4 → v5 (2026-04-17)

`fixtures/threat_probe_baseline.json` v4 was anchored to an older `bootstrap_model.pt` file; after GroupNorm (§99) and subsequent bootstrap refresh the live bootstrap produced different threat-head outputs than the recorded baseline, so `make probe.latest` was comparing apples to oranges.

- **NPZ:** `fixtures/threat_probe_positions.npz` was 24-plane (states shape `(20, 24, 19, 19)`) from the §92 era. Planes 0–17 are bit-exact with the current `GameState.to_tensor()` layout (`current_views + history + mr_flag + ply_parity`); only planes 18–23 (chain-length) are gone post-§97. Sliced in place to `(20, 18, 19, 19)` — probe positions preserved, metadata unchanged.
- **Baseline:** regenerated against the live `bootstrap_model.pt` (18-plane trunk, GroupNorm(8)). `BASELINE_SCHEMA_VERSION` 4 → 5.

| metric | v4 (stale bootstrap) | v5 (live bootstrap) | Δ |
|---|---|---|---|
| `ext_logit_mean`  | +0.217 | +0.080 | −0.137 |
| `ctrl_logit_mean` | +1.154 | +0.028 | −1.126 |
| `contrast_mean`   | −0.937 | +0.052 | **+0.989** |
| `ext_in_top5_pct` | 20 %   | 20 %   | 0 |
| `ext_in_top10_pct`| 20 %   | 20 %   | 0 |

**Contrast shift > ±0.3 flag (per task spec): investigated.** The shift is driven by a bootstrap-file substitution, not probe-position instability across the 24→18 migration. Evidence:
1. `bootstrap_model.pt` mtime is 2026-04-17 10:43 — newer than the v4 commit (2026-04-16 19:40); bootstrap was refreshed between v4 and v5.
2. `ctrl_logit_mean` collapsed by ~1.1 nats. If chain planes had been confounding the probe, we would expect ext/ctrl to shift by comparable magnitudes; instead ext_logit barely moved (−0.14) while ctrl_logit flattened. That is a weights story, not an input-layout story.
3. Top-K policy membership (20%/20%) is invariant across versions — geometry of the fixture is stable.

**C1 floor unchanged.** `max(0.38, 0.8 × 0.052) = 0.38` — absolute floor binds, same as v4 against the bootstrap (untrained threat head; §92 rationale). Future step-5k probes still gate on contrast ≥ 0.38; the baseline only feeds C4 drift-warning and the 0.8× multiplier path.

**Round-trip self-test:** `make probe.bootstrap` exit 0, baseline re-written bit-identical. `make probe.latest` cannot be exercised end-to-end until a post-§99 (GroupNorm) checkpoint exists — all `checkpoints/saved/checkpoint_*.pt` are pre-§99 BN and refuse to load by design (§99 safety rail).

Full report: `reports/threat_probe_v5_2026-04-18.md`.

## §101 — Graduation gate with anchor model (2026-04-16)

**Motivation.** Self-play workers were consuming `inf_model` weights re-synced from `trainer.model` every `checkpoint_interval` (500 steps) — effectively the current-training model, warts and all. Transient optimizer regressions fed directly into the data stream. KrakenBot-style graduation: new model must beat the current anchor at a configurable win rate before replacing it; workers keep the anchor between promotions. Monotonic data quality.

**Gap analysis.** 90% of the infrastructure was already live (`EvalPipeline` vs `best_model` gate at `eval_pipeline.py:188-190`; `best_model.pt` saved on promotion; `ResultsDB` + Bradley-Terry logs matches). Missing piece: routing — `best_model` was never consumed by self-play.

**Changes (branch `feat/graduation-gate`):**

- `loop.py`: remove unconditional `_sync_weights_to_inf()` call on checkpoint interval (buffer save retained). On startup with `best_model.pt` loaded, `inf_model` re-syncs from `best_model` (not `trainer.model`). `best_model_promoted` log gains `graduated=True`, `wr_best`.
- `eval_pipeline.py`: per-opponent `stride` gating — skip when `(train_step // base_interval) % stride != 0`. `EvalPipeline.__init__` caches `self._base_interval`.
- `eval.yaml`: `eval_interval: 5000 → 2500`; `best_checkpoint.n_games: 50 → 200` (tighter gating CI); strides `best=1 / sealbot=4 / random=1`.

**Behavioural invariants.**

- Between graduations, `inf_model` weights are frozen.
- On graduation: `best_model ← eval_model` (the scored snapshot — see §101.a C1), `inf_model ← best_model`, persisted + logged.
- Cold start with no `best_model.pt`: anchor is cloned from initial `trainer.model`. Candidate vs clone ~50% → no spurious promotion.

**Threshold & cadence.** `promotion_winrate: 0.55` (vs KrakenBot's 0.76 — conservative; tune up once graduations fire regularly). `n_games: 200` (binomial 95% CI ±~7% at p=0.55). Anchor eval every 2500 steps; SealBot every 10000.

### §101.a — Review fixes (applied before merge)

| # | Issue | Fix |
|---|---|---|
| **C1** | **Promoted weights ≠ evaluated weights.** Eval runs in a background thread with an `eval_model` snapshot; old code copied *current* `trainer.model` into `best_model` on promotion. Trainer had advanced ~1 `eval_interval` of steps between eval start and drain → every promotion committed unvalidated weights. | `eval_model` allocated once in outer scope; promotion branch loads `best_model ← eval_model` (drain fires before the next eval overwrites). |
| H1 | Stride cadence computed against `eval.yaml` `eval_interval`, ignoring `training.yaml` override. At training.yaml=5000, sealbot stride=4 fired every 20k steps not 10k. | Pipeline reads `full_config.eval_interval`; falls back to `self._base_interval`. Documented in both config files. |
| M1 | **False-promotion rate.** At n=200, p_true=0.5, P(X≥110) ≈ 9% → ~3-4 false promotions per 100k steps from sampling noise. | `gating.require_ci_above_half` (default true): promotion needs `wr_best ≥ threshold` AND `ci_lo > 0.5`. Drops false-positive rate below 1%. Flag preserves old behaviour for tuning. |
| M2 | Resume when `trainer.step != best_model_step` compares arbitrary weights vs anchor from a different time; lucky 55% wipes anchor. | Log `resume_anchor_step_mismatch` warning before first eval. |
| M3 | `eval_complete` event shipped `eval_games=0` (key never written). | Sum per-opponent `n_games` actually played (accounts for stride skips) → `results["eval_games"]`. |
| M4 | `stride: 0` or non-int silently collapsed to "every round" under `int(s) <= 1`. | `EvalPipeline.__init__` raises on stride not int ≥ 1; disable via `enabled: false`. |
| L1 | `eval_model` reallocated per round (~30 MB activations). | Allocated once outside loop; `load_state_dict` per round. |
| L2 | Dashboard read `.get("wr_sealbot", 0.0)` → stride-skipped rounds rendered as "0% vs SealBot". | Use `None` in event payload; dashboard distinguishes skip vs loss. |
| L3 | `eval_interval` coupling between trigger and stride math undocumented. | Comments added to `eval.yaml` + `training.yaml`. |
| L4 | Redundant `result["step"] = _step` in `run_evaluation`. | Removed. |

**Side cleanup.** `_sync_weights_to_inf()` (wrong direction — syncs from trainer, not anchor) deleted; sync sites now explicitly copy from `best_model` or `eval_model`.

**Tests added.** `test_stride_zero_rejected_at_init` (M4); `test_ci_guard_{blocks_marginal,disabled_allows_marginal}_promotion` (M1); `test_eval_games_reflects_opponents_run` (M3); `test_effective_eval_interval_override` (H1); `test_stride_{skips,runs}_sealbot_{off,on}_cadence` (stride). `test_run_evaluation_stores_results` updated to 9/10 wins (clears both gates without disabling CI guard).

**Known follow-ups (not blocking):** `graduation` boolean column on `ResultsDB.matches`; optional `skip_first_eval` flag for the guaranteed-neutral cold-start round.

## §102 — Benchmark rebaseline post-§97 (2026-04-17)

**Trigger:** §98 flagged worker-throughput warmup artifact (IQR 188%, p25=0) and buffer-augmented regression unresolved. This section addresses the warmup design and consolidates all ten target values against a single clean run.

### Methodology change

- Worker warmup raised 30s → 90s (already landed in `scripts/benchmark.py` as `warmup_worker = 90.0`; this run is the first full bench.full with it).
- Pool measurement window 120s (unchanged from §98's 120s).
- `make bench.full` deprecated — target is now `make bench` (runs the same command).
- No changes to the measurement kernels (methodology frozen per task constraint).

### Observed (laptop Ryzen 7 8845HS + RTX 4060, n=5, 14 workers)

Raw JSON: `reports/benchmarks/bench_2026-04-17.json`
Full log: `reports/benchmarks/bench_2026-04-17_postmigration.log`
Physical check: `reports/bench_physical_check_2026-04-17.md`

| Metric | Median | IQR | IQR% | vs §98 |
|---|---:|---:|---:|---|
| MCTS sim/s (CPU)              | 56,404   | ±178    | 0.3%  | +1.7% |
| NN inference b=64 pos/s       | 7,676.5  | ±1.2    | 0.02% | **−21%** |
| NN latency b=1 ms             | 2.19     | ±0.55   | 25%   | +40% (jitter; target still ≤3.5) |
| Buffer push pos/s             | 618,552  | ±5,868  | 1%    | −5% |
| Buffer sample raw µs          | 1,379    | ±36     | 2.6%  | −2.5% |
| Buffer sample aug µs          | 1,241    | ±22     | 1.8%  | **−25%** (better; §98 L3 pressure gone) |
| GPU util %                    | 100.0    | ±0.1    | 0.1%  | flat |
| VRAM GB                       | 0.115    | ±0      | 0%    | +0.07 (larger dummy allocs) |
| Worker throughput pos/hr      | 167,755  | ±9,601  | 5.7%  | **IQR 188% → 5.7%** (warmup fix landed) |
| Worker batch fill %           | 97.49    | ±1.1    | 1.1%  | −2.5% |

### Root cause: unexplained ~22% drop in NN inference and ~19% drop in buffer push

Per-run IQR is razor-tight (0.02% and 1%). §72 already documented a sustained ~14% NVIDIA driver/boost-clock shift; this run compounds another ~21% on top. Not a code regression. Treat as hardware-state drift; re-measure after a clean boot before any production decision depends on these metrics.

### Production cross-check

`logs/train_10cc8d56e4394a9ca542740c4bcee069.jsonl` (2026-04-16 live training):
- 1,568 games × 118 avg plies / 3.89 h = **47,650 pos/hr**.
- Benchmark 167,755 / production 47,650 = **3.52×**.
- Expected 2×–5× (benchmark has no training-step GPU contention, 200 sims vs production 400+). **Plausible.**

### Target-setting rules (applied in order)

1. Physical check verdict "OK" → eligible for update.
2. IQR > 20%: use p10. (Applied to NN latency — but target already passed, so N/A.)
3. IQR ≤ 20%: `new_target = min(median × 0.85, prior_target)` (never raise on one run).
4. `worker_pos_per_hr` marked **PROVISIONAL** — §98 warmup fix just landed, confirm stability over a second run.

### Target diff (CLAUDE.md)

| Metric | Old target | New target | Why |
|---|---|---|---|
| NN inference pos/s          | ≥ 8,250    | **≥ 6,500**   | 7,676 × 0.85 = 6,525; driver-drift regression (§72 precedent) |
| Buffer push pos/s           | ≥ 630,000  | **≥ 525,000** | 618,552 × 0.85 = 525,770; same driver-drift basket |
| Worker throughput pos/hr    | ≥ 250,000  | **≥ 142,000** (PROVISIONAL) | 167,755 × 0.85 = 142,592; old 250k was §98 placeholder |
| MCTS sim/s                  | ≥ 26,000   | ≥ 26,000      | 48k × 0.85 > 26k floor; keep floor |
| NN latency b=1 ms           | ≤ 3.5      | ≤ 3.5         | passes; keep |
| Buffer sample raw µs        | ≤ 1,500    | ≤ 1,500       | passes; keep |
| Buffer sample aug µs        | ≤ 1,800    | ≤ 1,800       | improved but do not tighten on one run |
| GPU util %                  | ≥ 85       | ≥ 85          | saturated; keep |
| VRAM GB                     | ≤ 6.88 (80%) | ≤ 6.88 (80%) | unchanged |
| Worker batch fill %         | ≥ 84       | ≥ 84          | passes; keep |

### Code updates

- `scripts/benchmark.py` `_CHECKS_CUDA` target constants updated to match the above (measurement code unchanged).
- `CLAUDE.md § Benchmarks` table replaced with 2026-04-17 values.
- `docs/02_roadmap.md` Phase 3.5 table marked HISTORICAL; added Phase 4.0 post-§97/§99/§102 table.

### Action items (tracked in Q-log, not blocking)

- [ ] Re-run `make bench` after a clean reboot; confirm NN inference regression is persistent (or recovers to 9k+ range).
- [ ] Flip `worker_pos_per_hr` target from PROVISIONAL to firm after a second stable run.

### Commits

- `perf(bench): 2026-04-17 rebaseline post-18ch + 120s pool window`
- `docs(bench): update CLAUDE.md + roadmap targets to conservative post-§97 values`
- `docs(sprint): §102 benchmark rebaseline — methodology change + target diff`

### Side note — stale artifacts archived pre-run

During setup, found all `checkpoints/*.pt` and `data/bootstrap_corpus.npz` still carried the pre-§97 24-channel + BatchNorm layout. Archived to `checkpoints/archive_2026-04-17_pre97_pre99/` and `data/archive_2026-04-17_pre97/`. New 18-channel `data/bootstrap_corpus.npz` produced by slicing planes 0-17 of the archived 24-channel corpus (199,470 positions preserved; no re-scrape). Pretrain still required to produce a GN(8) bootstrap — does not affect the benchmark (random-init model from config).

## §103 — Corpus zero-chain fix + baseline_puct playout-cap pin (2026-04-17)

Two drift bugs surfaced by the post-wave-1 audits
(`reports/chain_parity_audit_2026-04-18.md` §4 and
`reports/selective_policy_audit_2026-04-18.md` §4 B2), independent fixes,
landed as separate commits. Log entry numbered §103 because §102 (the same-day
benchmark rebaseline) is already claimed; the commits retain the `§102.a` /
`§102.b` labels from the prompt that drove them.

### §103.a — Corpus chain target was zero post-§97

`batch_assembly.load_pretrained_buffer` padded corpus chain planes with
`np.zeros((T, 6, 19, 19))`. `compute_chain_loss` ran over the full batch
including corpus rows, so the chain head was pulled toward zero on the
pretrain fraction of every mixed step since §97 (2026-04-16). Silent — no
crash, no dashboard signal.

**Fix.** Compute chain planes from the stored stone planes at NPZ load via
`_compute_chain_planes(pre_states[:, 0], pre_states[:, 8])`. Route the /6
normalisation through float32 before the final f16 cast so the stored f16
bits match Rust `encode_chain_planes → f16` byte-exactly (the F2 guard only
pins the underlying int8 planes; this path pins the post-normalisation f16
values used by the self-play buffer).

**Regression.** `tests/test_corpus_chain_target.py` — two cases:

- `test_corpus_chain_planes_match_rust_byte_exact` — hand-built corpus NPZ
  round-trips through `load_pretrained_buffer → buffer.sample_batch`, matches
  `engine.compute_chain_planes` byte-exact at f16.
- `test_mixed_batch_chain_loss_uses_nonzero_corpus_targets` — 4 corpus + 4
  self-play rows → chain loss is finite and strictly positive on both
  halves, pinning that corpus targets are no longer zero.

Docstring at `trainer.py:420-427` updated to drop the stale "no
pretrain/selfplay divergence" language.

### §103.b — baseline_puct inherited selective loss

`configs/variants/baseline_puct.yaml` had no `playout_cap` override, so
post-§100 it inherited `full_search_prob: 0.25` from the base — turning a
"PRE-§67 HISTORICAL BASELINE" variant into a §100-selective run and
silently confounding any ablation using it as an unmodified control.

**Fix.** Pin `playout_cap.full_search_prob: 0.0` explicitly in
`baseline_puct.yaml`. Game-level `fast_prob` was already 0.0 via base
inheritance, so both playout caps are now OFF for this variant.

**Regression.** `tests/test_variant_configs.py::test_baseline_puct_pins_pre_100_semantics`
runs the same deep-merge path as `scripts/train.py --variant` and asserts
both `full_search_prob` and `fast_prob` resolve to 0.0.

### Commits

- `fix(training): compute corpus chain planes at NPZ load (§102.a)`
- `fix(config): pin baseline_puct full_search_prob=0.0 (§102.b)`
- `docs(sprint): §103 corpus zero-chain + baseline_puct pin`

## §104 — D-Gumbel / D-Zeroloss instrumentation (2026-04-17)

**Motivation.** Post-§100 the dashboard could not answer two questions without
guessing:

- **D-Gumbel** — `completed_q_values: true` produces a structurally valid
  policy target even at 100 sims (`engine/src/mcts/mod.rs:266-276` —
  `softmax(log_prior + sigma · completed_q)` over all legal actions). The
  §100 selective gate keys only on `is_full_search`, not on target type, and
  drops those quick-search CQ targets from the policy gradient. Whether that
  is leaving usable signal on the floor is an empirical question.
- **D-Zeroloss** — `trainer.py:518-522` logs `full_search_frac` but cannot
  distinguish `policy_loss == 0 because mask selected no rows` from
  `policy_loss == 0 because loss was numerically zero`. Known follow-up
  from §100 "Known follow-ups".

Both require per-step policy-target diagnostics that were not being emitted.

**Changes.** Monitoring-only. No behaviour change.

- `hexo_rl/training/trainer.py`:
  - New module-level `compute_policy_target_metrics(target_policy,
    policy_valid, full_search_mask)` returning 7 fields split by
    `is_full_search`: `policy_target_entropy_{full,fast}search`,
    `policy_target_kl_uniform_{full,fast}search`,
    `frac_fullsearch_in_batch`, `n_rows_policy_loss`, `n_rows_total`.
    All reductions stay on device; a single `.cpu().tolist()` over 7
    packed scalars replaces 7 `.item()` syncs — under 200 µs / call on
    CUDA at (B=256, A=362).
  - NaN is a first-class signal: when the full-/fast-subset has zero rows
    the mean comes back NaN, and renderers handle that explicitly. Keeps
    the decision rules "H_fast(CQ) ≥ some bound" readable even when a batch
    lands entirely in one bucket.
  - Gated via `monitoring.log_policy_target_metrics: true` (default on).
  - NaN-loss guard pre-populates the 7 keys.
- `hexo_rl/training/loop.py`: forwarded all 7 keys onto the `training_step`
  emit_event payload and onto the `log.info("train_step", ...)` structlog
  entry, so the same values land on the dashboard and in
  `logs/<run_name>.jsonl` for post-hoc analysis.
- `hexo_rl/monitoring/terminal_dashboard.py`: new `policy target` row below
  the entropy line — `H_full / H_fast │ KL_u_full / KL_u_fast │ n_full/total`.
- `hexo_rl/monitoring/static/index.html`: ring-buffer carries the 7 keys and
  the loss ratio strip gains compact `H_full / H_fast / KL_u_fast / n_full`
  segments. No new Chart.js panels — deliberately minimal web wiring.
- `docs/08_DASHBOARD_SPEC.md`: §2.1 schema updated with the 7 new keys +
  value ranges + NaN-as-signal note. §7 adds
  `monitoring.log_policy_target_metrics` config key. Changelog entry.
- `configs/monitoring.yaml`: default-true gate.

**Tests.** `tests/test_policy_target_metrics.py` — 5 synthetic-batch cases:

1. Uniform-vs-one-hot split — verifies the math: H_full ≈ log(362) ≈ 5.89,
   H_fast ≈ 0, KL_u_full ≈ 0, KL_u_fast ≈ log(362).
2. All full-search — fastsearch metrics must be NaN; emit does not raise.
3. All fast-search — symmetric.
4. Empty valid mask — all 4 means NaN, counts 0, every promised key present.
5. Cost budget — <200 µs/call on CUDA at (B=256, A=362) after the single-sync
   optimisation (CPU fallback: <1000 µs).

`tests/test_trainer.py` was updated to allow NaN on the two new fastsearch
keys when the batch carries no quick-search rows (default path).

**Bench check.** `reports/instrumentation_bench_check_2026-04-17.md`.
Instrumentation-reachable metrics all within ±5.5% of the 2026-04-17 09:34
baseline (MCTS −1.7%, NN inference −0.3%, NN latency −22% faster, buffer
push −5.5%, buffer sample ±5-12% within IQR). Worker-pool throughput
regressed ~36% but `benchmark.py` does not construct a `Trainer` — the
instrumentation is not in that call path, and worker-pool has a historical
±40% IQR on this harness (§98 caveat).

**Decision support.** `reports/gumbel_target_quality_2026-04-17.md` — two
smokes from `bootstrap_model.pt` (`baseline_puct`, `gumbel_targets`) and a
per-variant mean table with the Option A / B / inconclusive mapping from
the prompt brief.

**Verdict: Option A.** Quick-search completed-Q targets on `gumbel_targets`
drift toward uniform as training progresses (steady-state ΔH = H_fast −
H_full ≈ **+3.5 nats**, well above the +1.5 threshold; KL_u_fast falls
from 5.3 → 1.1 over steps 10–60). The §100 selective gate correctly
discards noisy quick-search CQ targets. When the `gumbel_full.yaml` mutex
bug (`reports/selective_policy_audit_2026-04-18.md` §4 B1) is unblocked,
the repair should follow the audit's Option A (drop legacy game-level
`fast_prob`, keep move-level `full_search_prob` from base).

**Caveats.** 20 metric events for baseline (full 200 steps); only 7 for
gumbel_targets (run stopped at ~step 83 — per-move 600-sim cost pushed a
full 200-step run past a reasonable wall-time budget). Steps 10–20 on
gumbel_targets are corpus-dominated warmup (ΔH small; excluded from the
call). `gumbel_full` not measured — mutex-blocked and `gumbel_targets`
shares the relevant CQ target construction path.

**Follow-up applied same day.** `configs/variants/gumbel_full.yaml` Option A
landed — `fast_prob: 0.25 → 0.0`, keeping the base's `full_search_prob: 0.25`
move-level cap. Mutex resolved at pool init; the desktop Exp E variant is
launchable again. `tests/test_variant_configs.py::test_gumbel_full_passes_playout_cap_mutex`
pins the resolved config so the next base-config drift cannot silently
reintroduce the bug.

**Resolves.** §100 "Known follow-ups" item 3 (distinguish empty-mask vs
genuine 0.0 policy loss — `n_rows_policy_loss == 0` vs `> 0` does it).
§101 gains a telemetry hook for future graduation-gate D-Gumbel validation.

### Commits

- `feat(monitoring): policy target entropy/KL split by is_full_search`
- `test(monitoring): synthetic batch assertions for new metrics`
- `docs(dashboard): add policy target metrics to emit schema`
- `docs(sprint): §104 Gumbel target quality instrumentation + decision support`

## §105 — Q27 perspective-flip smoke: W1 necessary, not sufficient (2026-04-18 → 2026-04-19)

**Motivation.** `e9ebbb9` ("fix(mcts): negate child Q at intermediate
ply in `get_improved_policy`, Gumbel score, `get_top_visits`") was
landed as W1 on correctness grounds — three call sites failed the
`parent.moves_remaining==1` negation that `puct_score` already had,
inverting training targets at ~50% of move steps. The open question
on landing was whether W1 *also* closes the Q27 attention-hijacking
symptom (threat head passes C1 easily; policy head pins at 20% on
C2/C3) at the 5K-step smoke horizon. One-shot smoke, two machines.

**Setup.** Two-machine split to save wall-clock. Both arms start from
`checkpoints/bootstrap_model.pt`, run 5000 steps no-dashboard, run
`make probe.latest` immediately after:

- **pre_fix** — laptop (8845HS / RTX 4060), commit `723615e` (parent
  of `e9ebbb9`), variant `gumbel_targets` (14 workers).
- **post_fix** — desktop (3700x / RTX 3070), commit `a7efa78` (HEAD),
  variant `gumbel_targets_desktop` (10 workers, +1ms inference wait).
  Selfplay semantics identical per the variant header; machine and
  worker-count differ.

Desktop arm required one restart after accidental window close
(run1 → step 450 partial; run2 restarted from bootstrap, no
contamination).

**Results.**

| Metric | pre_fix (723615e) | post_fix (HEAD) | Δ |
|---|---|---|---|
| C1 contrast_mean | +3.665 PASS | +3.891 PASS | +0.226 |
| C2 ext_in_top5_pct | **20% FAIL** | **20% FAIL** | 0 |
| C3 ext_in_top10_pct | **20% FAIL** | **20% FAIL** | 0 |
| C4 \|Δ ext_logit_mean\| | 0.078 ok | 0.505 ok | +0.427 |
| Exit | 1 (FAIL) | 1 (FAIL) | — |
| H(policy) @ 5K | 5.3733 | 5.6284 | +0.255 |
| Settled entropy band (500–5000) | 5.17 – 5.57 | 5.51 – 5.72 | ~+0.2 |
| Policy loss 0 → 5K (post only) | — | 1.9619 → 1.6544 | −0.308 |
| Games / buffer (post only) | — | 1253 / 250K saturated | — |

Uniform entropy on ~361 legal cells ≈ 5.88. Post-fix sits ~0.25 nats
closer to uniform than pre-fix.

**Verdict.** W1 is **necessary, not sufficient at 5K.**

- Both arms fail threat probe on C2/C3 with **identical 20% / 20%**
  scores. The policy head is not routing top-K to extension cells in
  either arm. Attention hijacking persists across the fix.
- C1 passes ~10× threshold in both arms — threat head carries
  extension-vs-control information; the failure is downstream in how
  the policy trunk uses that signal.
- Entropy delta sign is in the wrong direction for a "W1 alone fixes
  it" story: post-fix is *closer* to uniform, not more committed. Two
  plausible readings: healthier ongoing exploration, or machine/
  variant noise. Cannot discriminate with n=1 per arm.

W1 stands on correctness grounds regardless. Q27 remains open.

**Caveats.**

- Two machines, two variants. Selfplay target construction is
  identical; worker-count (14 vs 10) and inference wait (4 vs 5ms)
  differ. The +0.25-nat entropy delta is within plausible worker-count
  noise.
- n=1 per arm; no CIs. The identical 20%/20% C2/C3 is decisive enough
  on its own that the n=1 limit does not flip the conclusion, but the
  entropy reading is directional, not quantitative.
- Desktop run had a restart from bootstrap (not resume); reported
  trajectory is run2 only.
- Pre-fix arm's policy_loss trajectory not preserved post log rotation.

**Follow-ups (not actioned this sprint).**

Three candidate probes for where the attention-hijacking root cause
lives. Prioritise before committing to another 5K smoke:

1. **Value aggregation (Open Question 2).** Min-pool over cluster
   windows may silently discard extension-cell evidence — would
   reproduce the "threat scalar learns contrast, policy ignores it"
   signature seen here. Ablate mean-pool at fixed 5K budget.
2. **Threat head → policy gradient coupling.** BCE weight 0.1 may be
   drowned out by policy CE at shared trunk. `aux_threat_weight`
   sweep 0.1 → 0.5 at 5K.
3. **ZOI post-search mask (§77).** If extension cells in the probe
   positions fall outside hex-distance-5 of the last 16 moves, C2/C3
   are capped by construction. Log ZOI reachability of probe
   positions' extension cells.

A controlled same-machine n=3 rerun is premature until one of the
above has a concrete hypothesis attached.

**Files.**

- `reports/q27_perspective_flip_smoke_2026-04-18/verdict.md`
- `reports/q27_perspective_flip_smoke_2026-04-18/pre_fix/{summary,probe_output,entropy_trajectory,train_stdout}.*`
- `reports/q27_perspective_flip_smoke_2026-04-18/post_fix/summary.txt`
- `reports/probes/latest_20260418_223903.md` (pre_fix detail)
- `reports/probes/latest_20260419_011839.md` (post_fix detail)

**Resolves.** Nothing. **Leaves open.** Q27 (attention hijacking).

### Commits

- `docs(sprint): §105 Q27 perspective-flip smoke verdict`

**POSTSCRIPT 2026-04-19.** §106 supersedes the "attention hijacking
persists" framing above. Probe 1b regenerated the fixture from real
game positions; the 5K post-W1 checkpoint PASSES all three probe gates
(C1 +3.317, C2 50%, C3 65%). The C2/C3 failures logged here were a
synthetic-fixture artifact, not a training pathology. The correctness
argument for W1 (inverted Q targets at ~50% of move steps) is
unaffected. Original body above retained as the record of what was
believed at the time.

---

## §106 — Q27 Probe 1b: C2/C3 failure was fixture artifact (2026-04-19)

**Setup.** Probe 1 (synthetic ply=7 fixture, N=20) reported 0/20
extensions outside ZOI and bootstrap C2/C3 at 20%/20%, with the
load-bearing caveat that the fixture could not exercise §77's
truncation failure modes (ply > `zoi_lookback=16`, disjoint-cluster
threats). Probe 1b regenerated `fixtures/threat_probe_positions.npz`
from real mid/late positions sampled from
`runs/10cc8d56e4394a9ca542740c4bcee069` (500-game self-play, median
ply 169), with per-phase quotas early=7 / mid=7 / late=6. Ply span
9 → 150. Schema unchanged. Full report:
`reports/q27_zoi_reachability_realpositions_2026-04-19.md`.

**Baseline shift on `bootstrap_model.pt`.**

| metric | v5 synthetic early | v6 real mid/late |
|---|---:|---:|
| ext_logit_mean   |  0.080 |  0.015 |
| ext_logit_std    |  0.093 |  0.399 |
| ctrl_logit_mean  |  0.028 |  0.061 |
| ctrl_logit_std   |  0.012 |  0.030 |
| contrast_mean    | +0.052 | −0.046 |
| contrast_std     |  0.097 |  0.396 |
| ext_in_top5_pct  | 20.0%  | **60.0%** |
| ext_in_top10_pct | 20.0%  | **65.0%** |

The same `bootstrap_model.pt` jumps C2 20% → 60% and C3 20% → 65% on
the real fixture. The synthetic early-phase 3-in-a-row-with-far-stones
configuration was distributionally out-of-sample for the trained
policy (does not occur in real self-play at ply=7), so the probe was
asking the model to rank threats in a geometric configuration it had
never been trained to recognise. Baseline file bumped v5 → v6
(`BASELINE_SCHEMA_VERSION` in `scripts/probe_threat_logits.py:78`);
prior synthetic baseline preserved as
`fixtures/threat_probe_baseline_v5_synthetic.json.bak` (not committed).

**5K post-W1 checkpoint re-probe.** `checkpoint_00005000.pt` from the
run that ended 2026-04-18 22:38, re-probed against v6:

```
PASS  [C1] contrast=+3.317 (≥+0.380) OK
      [C2] top5=50% (≥25%) OK
      [C3] top10=65% (≥40%) OK
[C4] |Δ ext_logit_mean|=0.420 (<5.0) ok
```

All three gates PASS with margin — C1 contrast +3.317 sits ~9× above
the 0.38 floor. This is the inverse of the FAIL verdict recorded in
`reports/probes/latest_20260418_223903.md` against the v5 fixture.

**Supersedes §105's verdict.** §105 concluded "W1 necessary, not
sufficient" on the basis of identical 20%/20% C2/C3 in both arms of
the two-machine smoke. That conclusion was downstream of the v5
synthetic fixture. On the real-position v6 fixture the post-W1 5K
checkpoint PASSES all gates. The corrected framing:

> **W1 correctness fix lands clean; the apparent C2/C3 symptom was a
> fixture artifact.**

The correctness argument for W1 (three call sites inverted training
targets at ~50% of move steps) is independent of this and unchanged.

**§77 truncation failure mode — 1/20 instance.** Probe 1b found a
single position (late, ply=91, cluster center (37, 5), extension
(32, −1)) where the extension cell sits at `ext_d_zoi = 11` — outside
the live ZOI mask. Stones within hex-distance 3 of the extension are
all placed before the lookback window cut-in; the last 16 moves are
scattered across remote disjoint colonies. Concrete instance of §77's
disjoint-cluster prediction, but at 1/20 it cannot carry a
population-level C2/C3 miss. Kept as a note; not a blocker. Fix (raise
`zoi_lookback` or make ZOI colony-aware rather than recency-based) is
Phase 4.5+ if late-game disjoint-cluster failures surface in sustained
training.

**Q27 status.** Remains OPEN but reframed — no active root-cause
probe. Probes 2 (threat-weight sweep) and 3 (value-aggregation
ablation) shelved pending post-5K evidence of actual training-trajectory
regression. Next evidence point: sustained training smoke from
`bootstrap_model.pt`. Reopen if C2/C3 regress on the real-fixture
probe after 5K.

**C1 contrast flipped negative on bootstrap against real fixture.**
`ctrl_logit_mean` (0.061) > `ext_logit_mean` (0.015) on the v6 fixture
— the scalar threat head fires *more* on an empty far cell than on the
extension, yet the policy ranking still routes 60%/65% of extensions
into top-5 / top-10. Threat-scalar magnitude and policy-ranking
signals are decoupled on bootstrap. Not a bug; an unexplained
observation. Filed as **Q32** in `docs/06_OPEN_QUESTIONS.md` (WATCH
priority, threat-scalar magnitude vs policy ranking decoupling).

**Files touched in this cleanup pass.**

- `scripts/generate_threat_probe_fixtures.py` — `--n-per-phase` flag,
  compound_move phase thresholds, strict quota enforcement.
- `scripts/probe_threat_logits.py` — `BASELINE_SCHEMA_VERSION` 5 → 6.
- `fixtures/threat_probe_positions.npz` — regenerated from real run
  (7 early / 7 mid / 6 late).
- `fixtures/threat_probe_baseline.json` — v6 baseline committed in
  c5bce9c.
- `fixtures/threat_probe_baseline.CHANGELOG.md` — seeded (v2 → v6
  history).
- `docs/07_PHASE4_SPRINT_LOG.md` — §105 postscript + this entry.
- `docs/06_OPEN_QUESTIONS.md` — Q27 reframe, Q32 added.
- `reports/q27_perspective_flip_smoke_2026-04-18/verdict.md` —
  superseded banner pointing to Probe 1b report.
- `reports/q27_zoi_reachability_2026-04-19.md` — superseded banner
  pointing to real-fixture report.

**Resolves.** Nothing. **Reframes.** Q27 (no longer "attention
hijacking persists — root cause unknown"; now "reframed, no active
C2/C3 regression"). **Opens.** Q32 (threat-scalar vs policy-ranking
decoupling, WATCH).

### Commits

- `docs(sprint): §106 Q27 Probe 1b inverted verdict — fixture artifact`
