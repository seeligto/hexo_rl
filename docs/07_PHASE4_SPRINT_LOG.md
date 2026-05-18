# HeXO Phase 4.0 Sprint Log — Consolidated Reference

Read this alongside `CLAUDE.md` at the start of any new session to avoid
re-litigating resolved decisions. Structured by subsystem, not by date.
For per-day narrative see `docs/07_PHASE4_SPRINT_LOG_BACKUP.md`.

<!-- Compressed 2026-05-13 (full pass §66–§174): every closed § distilled to
     verdict + load-bearing mechanism + settled pins + commit/report pointer.
     Forensic detail extracted verbatim to reports/sprint_archive/ before
     compression. Net: 11,111 → ~2,690 lines (-76%). Backup retained at
     docs/07_PHASE4_SPRINT_LOG_BACKUP.md. Spec: docs/compression/sprint_log_compression_spec.md. -->

---

## Classification Audit (§1–§101)

| Bucket | Sections |
|---|---|
| KEEP-FULL | §1, §2, §4, §5, §15, §19, §21, §26, §27, §28, §33, §34, §35, §36, §37, §40, §46b, §47, §58, §59, §61, §63, §66, §67, §69, §70, §71, §73, §74, §77, §80, §84, §85, §86, §88, §89, §90, §91, §95, §97, §98, §99, §100, §101, §158 |
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
> - Input grew 18 → 24 at §92 and reverted 24 → 18 at §97; dropped 18 → 8 at §131.
>   Current: **8 planes** (KEPT_PLANE_INDICES = [0,1,2,3,8,9,10,11] from the
>   18-plane index space). Chain is an aux target in a separate replay-buffer
>   sub-buffer.
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

### torch.compile — DISABLED (§32) → RE-ENABLED (§116)

Arc: §3 added `reduce-overhead` → §25 re-enabled with split train/inf model instances → §30 changed to `mode="default"` (TLS crash on Python 3.14) → §32 fully disabled (27 GB Triton JIT spike blocks workers for 5+ min on first forward) → **§116 re-enabled `reduce-overhead` on PT2.11+Py3.14 (both blockers resolved)**.

**Current state:** `torch_compile: false` pending §116 AC-power bench gate.
**§116 probe verdict:** GO — all 3 modes work, 0 graph breaks, 1.50× throughput / 1.87× latency vs eager, VRAM spike 59.5 MB (was 27 GB). See `reports/investigations/torch_compile_retry_20260423/report.md`.

Note: `triton.cudagraphs=False` in PT2.11 — reduce-overhead gains come from kernel fusion, not CUDA graph replay.

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
> - §36 temperature description reconciled to match Rust code at §70 C.1 resolution (quarter-cosine per compound_move, threshold=15, temp_min=0.05 floor). The legacy ply-based step schedule lives on only in `hexo_rl/selfplay/utils.py::get_temperature`, called from the Python `SelfPlayWorker` used by eval-adjacent paths (`our_model_bot`, `benchmark_mcts`), not on the training path.
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

### Cosine-Annealed Temperature (§36; reconciled at §70 C.1)

Replaced the hard step at move 30 with a quarter-cosine schedule. Live
Rust implementation in `engine/src/game_runner/worker_loop.rs:20-31`
(`compute_move_temperature`), driven off **compound move** (not ply):

```
compound_move = (ply + 1) / 2  for ply > 0, else 0
τ(cm) = max(temp_min, cos(π/2 · cm / temp_threshold))   if cm <  temp_threshold
τ(cm) = temp_min                                        if cm >= temp_threshold
```

| compound_move | 0 | 5 | 10 | 14 | 15 | 16 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| τ | 1.0000 | 0.8660 | 0.5000 | 0.1045 | 0.0500 | 0.0500 | 0.0500 | 0.0500 |

Config: `selfplay.playout_cap.temperature_threshold_compound_moves: 15`,
`selfplay.playout_cap.temp_min: 0.05`. `mcts.temp_anneal_moves` /
`mcts.temp_min` are not read by the Rust training path — the live keys
are under `selfplay.playout_cap`. The legacy ply-based half-cosine
formulation described earlier in this section (and the `1.0 if ply<30
else 0.1` step schedule in `docs/01_architecture.md`) is obsolete on
the training path; the Python `get_temperature(ply, ...)` step schedule
in `hexo_rl/selfplay/utils.py` survives only for `SelfPlayWorker`, used
by eval-adjacent bots.

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

## Falsified Hypotheses Register

Do not re-litigate. Each row points at the § that closed it.

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §154 | v9 hex-trunk closes self-play gap | §154 MCTS-matched eval | Probe gates pass but selfplay drops to 0–1% SealBot WR. Probes cannot validate dynamic equivariance. |
| §155 R10 | Super-additive interaction of 5 smoke MCTS+exploration knobs drives 91% draws | §156 R12 | Cosine temperature alone is load-bearing (~5% draws → ~91%). Dirichlet / opening_plies / playout_cap are synergy partners, not drivers. |
| §169 A4 | Padding semantics (canvas_realness + PartialConv2d) recovers bbox direction | §169 P4 eval | A4 trains to 3.47 loss (below A1 anchor 3.57) but 0% SealBot WR. bbox direction is structural — K-aggregation as cross-cluster contrast, bbox-centroid frame instability, R=8 perception. |
| §170 A2/A3 | Learned PMA pool replaces K-cluster min/max | §169 ablation matrix | A2 4.5% / A3 7.5% WR vs A1 14.5%. A1 K-cluster min/max canonical. |
| §170 P3 gpool-bias-on-all | Gpool-bias as global lever for both policy + value heads | §170 P4 | gpool-bias-policy-only is the load-bearing mechanism; full gpool-bias is NULL on value. |
| pre-§148 | v6 corpus is human-quality (bot mix at uniform weight does not contaminate) | §147 audit | ~41% bot games at source_weight=1.0; Elo weighting degenerated to uniform via rng.choice on uniform weights. v7 human-only Elo-weighted is canonical. |
| §174 e50 | More pretrain epochs improve self-play | §174 closeout | e50 selfplay regressed vs e30 (median 12 vs 17 plies). e50 G4 marginal fail (0.489 vs band 0.462). Value head over-fits to corpus-mode signal that selfplay cannot reproduce. |
| §174 radius compression | LEGAL_MOVE_RADIUS 8→5 at bootstrap fixes v6w25 selfplay collapse | §174 R=1..R=8 smoke | Median plies identical across all radii. Radius does not move bootstrap quality. Smokes were already R=8. |
| §174 bootstrap recipe | v6w25 selfplay collapse is a bootstrap recipe issue | §174 closeout | Loss surface normal (3.96 nats vs uniform, matches v7full trajectory). Opening-fraction starvation refuted (16.09% vs 17.15% v6). Collapse is at argmax-degeneracy / selfplay-interaction layer, not corpus/loss. |
| pre-§73 | Dirichlet root noise active on Phase 3.5+ training path | §73 Q17 | Unported at Rust migration. 16,880 steps of carbon-copy self-play (Q17 mode collapse). Fixed in commit `71d7e6e`. |
| pre-§47 | FP16 AMP is numerically robust on aux losses | §47 | 0×−inf cascade in aux CE caused NaN total_loss, BN poisoning. Log-clamp + `torch.special.entr()` fix. |
| pre-§101 C1 | Promoted weights = evaluated weights | §101 C1 | Allocator reuse → every graduation committed unvalidated weights as anchor. Fixed at §101 C1. |
| §169 P0 | A4 collapse is broadcast-scalar-plane dependency | §169a probe | Spatial pathway not dead; collapse is structural at K=1 inference. |
| §131 (pre) | 18-plane input dimensionality is load-bearing | §131 ablation | KEPT_PLANE_INDICES=[0,1,2,3,8,9,10,11] (8 of 18) suffice. Chain moved to aux sub-buffer (§97 line). |
| forced-win short-circuit (pre-baseline) | MCTS expansion-time forced-win detection accelerates training | removed pre-§0 | Network never evaluated near-win positions → no fork learning. Removed; quiescence value-override at leaf-eval is the correct alternative. |
| §171 A4 P2-reopen | Distribution-shift fine-tune over 5% adversarial corpus (frozen-spine) recovers MCTS signal on A4 | §171 A4 P2-reopen C closeout | MCTS-64 0/200 Wilson95 [0%, 1.88%] — DEAD bin cleanly met. Falsifies §169 P0 SPATIAL_RICH for frozen-spine class. |

---

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
| 2026-04-22 | `cda9dde` dedup always-on has residual 33 µs cost; `push_many_impl` element-wise `to_bits()` loops caused LLVM codegen spillover regressing push (460k→576k) and sample (1,715→1,533 µs). | `push.rs` transmute fix (`6c0bfa9`) recovered push and aug; sample_raw residual +33 µs is correctness cost of always-on dedup. Recalibrate sample_raw target only. | `buffer_sample_raw` ≤ 1,500 → ≤ 1,550 µs |

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

# Mechanism Lessons

Distilled meta-lessons. Cite a row before re-deriving its rule.

| # | Lesson | Origin |
|---|---|---|
| L1 | Corpus quality = model quality floor. Verify corpus completeness BEFORE diagnosing trainer pathology. Silent filter bugs do NOT show in loss curves. | §114, §147 |
| L2 | Probe gates cannot validate dynamic equivariance. Require MCTS-matched eval, not just argmax probes. | §154 v9 falsified |
| L3 | One sole-load-bearing knob is the default; "super-additive interaction of N knobs" is usually wrong. Bisect within the conjunction. | §155 → §156 |
| L4 | The encoding decides; the pool variant tweaks. Training loss alone is NOT a sufficient signal for downstream WR. | §169 ablation matrix |
| L5 | Pool overflow is silent corruption. `expand_and_backup_single` inserting all legal moves caused warned-flag fabricated `is_terminal=true` → poisoned buffer throughout. Top-K expansion (192) sorted by policy prior is the fix. | §127, §128 |
| L6 | Bench metric `positions_pushed` is bimodal (burst artifact). `positions_generated` continuous counter replaces it. Pre-§128 throughput numbers are obsolete. | §128 |
| L7 | Always run bench twice on new hardware; discard first run (CUDA JIT warmup). | §90, §125 |
| L8 | More pretrain epochs is not strictly better. Value-head over-fits to corpus-mode signal that selfplay cannot reproduce. | §174 e50 |
| L9 | Cosine temperature schedule is the load-bearing knob in draw-collapse. Pair with LEGAL_MOVE_RADIUS jitter when active. | §156, §157 |
| L10 | Cross-encoding checkpoint loading is brittle. Encoding header in `persist.rs` (§173 HEXB v7) rejects mismatched loads. | §172, §173 |
| L11 | K-cluster encoding has no board-AI precedent but is structural twin of MVCNN view-pooling, SwAV multi-crop, PointNet++ set-abstraction, deep MIL pooling. 12pp gain at matched MCTS perception is structural inductive bias, not TTA. | §170 P4, §167 T2 |
| L12 | Never recalibrate gate thresholds to match failing runs. Never extend smoke runs past stated step limits without explicit go-ahead. | §155, §144 |
| L13 | Subagent prompts include pre-registered pass criteria; implicit done-when causes scope creep. Independent review subagent at sprint close in fresh context, not implementer's. | §170, §171, §172 A9 |
| L14 | Pre-flight cold smoke must use canonical sprint bootstrap, not dev defaults. | §171 P2 |
| L15 | Pre-§148 v6 corpus retired wholesale. All v6-era anchors (Q41 51%, Q52 24%) carry contaminated baseline; do not cite as comparison. | §147, §148 |
| L16 | RegistrySpec by value (~174 B) on MCTS hot path kills `worker_pos_per_hr` ~10%. Use `&'static`, scalar extraction, or `#[inline]` accessors. | §173 A5b |
| L17 | Always grep receiving code before scheduling a sprint item as "one-line config change". §122 rotation "one-liner" was a ~50-80 line port. | §122, §131 |

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

# Part 4 — Sprint Body (compressed §66–§174)

Each closed § retains: date, verdict, load-bearing mechanism, settled pins, commit/report pointer.
Forensic detail: `reports/sprint_archive/§<NNN>_<slug>.md` (extracted verbatim 2026-05-13).

## § Index — INDEX-ONLY entries

| § | Topic | Date | Pointer |
|---|---|---|---|
| §75 | Fast game disable for gumbel_targets (`fast_prob: 0.0`); 100% timeout draws under K=50 sims | 2026-04-10 | `reports/sprint_archive/§075_*.md` |
| §76 | `max_game_moves` 150 → 200; plies vs compound moves clarification | 2026-04-10 | `reports/sprint_archive/§076_*.md` |
| §78 | `/analyze` policy viewer (4 commits, branch `feat/policy-viewer`) | 2026-04-11 | `reports/sprint_archive/§078_*.md` |
| §79 | Initial buffer 100K → 250K; growth schedule `[250K@0, 500K@300K, 1M@1M]` | 2026-04-12 | `reports/sprint_archive/§079_*.md` |
| §81 | Desktop n_workers sweep D3 winner (10 workers, wait 5ms, burst 8); 3700x GIL ceiling | 2026-04-12 | `reports/sprint_archive/§081_*.md` |
| §82 | `loss_ownership` + `loss_threat` added to `emit_event` (`d6a293e`) | 2026-04-12 | `reports/sprint_archive/§082_*.md` |
| §83 | `quiescence_fire_count` atomic instrumentation (`4124faa`, `ad79be7`) | 2026-04-12 | `reports/sprint_archive/§083_*.md` |
| §87 | Cargo feature gate for pyo3 `extension-module` | 2026-04-13 | `reports/sprint_archive/§087_*.md` |
| §102 | Bench rebaseline post-§97; methodology shift; NN inference & buffer-push targets lowered (driver drift) | 2026-04-17 | `reports/sprint_archive/§102_*.md` |
| §106 | Q27 Probe 1b: synthetic fixture artifact; v6 baseline regenerated from real positions; supersedes §105 FAIL verdict | 2026-04-19 | `reports/sprint_archive/§106_*.md` |
| §107 | Post-W1 sustained launch + I1 colony-extension + I2 cluster-variance instrumentation | 2026-04-19 | `reports/sprint_archive/§107_*.md` |
| §108 | Desktop post-W1 `gumbel_full` launch + I1/I2 JSONL mirror | 2026-04-19 | `reports/sprint_archive/§108_*.md` |
| §109 | Q33 selfplay-entropy: pe_self ≈ 5.35 is `H(p_model)` not target entropy; CQ targets sharpen | 2026-04-21 | `reports/sprint_archive/§109_*.md` |
| §110 | Q33-B trainer-fit sanity: pe_self fixed point ~5.36 (Q1=5.36, Q4=5.36, Δ=0.004); split E1/E2 candidates | 2026-04-21 | `reports/sprint_archive/§110_*.md` |
| §111 | Q33-C augmentation discriminator HALT (Python-API-only toggle; no config knob) | 2026-04-21 | `reports/sprint_archive/§111_*.md` |
| §113 | `buffer_sample_raw` target recalibration 1500 → 1550 µs (always-on dedup correctness cost) | 2026-04-22 | `reports/sprint_archive/§113_*.md` |
| §115 | CLAUDE.md 734 → 87 lines; topic-scoped rules under `docs/rules/`; 3 skills scaffolded | 2026-04-22 | `reports/sprint_archive/§115_*.md` |
| §120 | RecentBuffer augmentation deployed (`19b1392`); soft-abort at step 14000 — augmentation alone insufficient | 2026-04-25 | `reports/sprint_archive/§120_*.md` |
| §122 | Phase 5 architectural redesign scoping (B1 D17 channel ablation, B2 backbone-form memo, B3 retrain cost, B4 buffer compat) | 2026-04-25 | `reports/sprint_archive/§122_*.md` |
| §123 | Bench methodology fix: `pool_model` uses `mode="default"` (not `reduce-overhead`) to avoid `cudagraph_trees` background-thread TLS crash | 2026-04-25 | `reports/sprint_archive/§123_*.md` |
| §126 | Sweep harness migration: knob registry replaces `sweep_epyc4080.sh`; `make sweep` / `make sweep.long` | 2026-04-26 | `reports/sprint_archive/§126_*.md` |
| §129 | Disk-budget guard (`DiskGuard`); checkpoint/game-record pruning; rotating JSONL | 2026-04-28 | `reports/sprint_archive/§129_*.md` |
| §130 | Per-game self-play rotation port; closes §121 C1 directional bias | 2026-04-29 | `reports/sprint_archive/§130_*.md` |
| §133 | D6 sym-table verification for HEXB v6 8-plane buffer (6 tests; plane indices invariant under D6) | 2026-04-29 | `reports/sprint_archive/§133_*.md` |
| §134 | bootstrap-v6: 8-plane pretrain on 6,259 human games; 100/100 vs RandomBot; threat C2=50, C3=60 | 2026-04-30 | `reports/sprint_archive/§134_*.md` |
| §135 | Bench gate W4 8-plane: 9/10 metrics improved or flat; +14% worker pos/hr; no regressions | 2026-04-30 | `reports/sprint_archive/§135_*.md` |
| §136 | Post-§131 W1+W2 audit fix wave (19 commits): 8→18 plane refs, Q49 RNG audit COUPLED-NEGLIGIBLE | 2026-04-30 | `reports/sprint_archive/§136_*.md` |
| §138 | W4 Option C smoke (5080, 8-plane + rotation): axis_density 0.55, downward trend; CONTINUE to 40k | 2026-04-30 | `reports/sprint_archive/§138_*.md` |
| §143 | γ-knob audit + W4C smoke v3 recommendation (`temperature_threshold_compound_moves` 15→10, ε 0.25→0.10) | 2026-05-01 | `reports/sprint_archive/§143_*.md` |
| §144 | W4C smoke v3 Stage 1 ABORT; gates recalibrated for 100-ply games; `max_game_moves` 100→150 | 2026-05-01 | `reports/sprint_archive/§144_*.md` |
| §145 | Smoke v4 ABORT — γ + truncation slack insufficient; fallback to Option α' (radius cap) | 2026-05-02 | `reports/sprint_archive/§145_*.md` |
| §151 | Numba @njit audit — NO-GO; no qualifying hot-path Python loop; Rust-port alternatives flagged | 2026-05-04 | `reports/sprint_archive/§151_*.md` |

---

## §66–§101 — Phase 4.0 foundation (KEEP-DISTILLED per §1–§101 Classification Audit)

### §66 — Gumbel MCTS activation & training restart (SUPERSEDED) — 2026-04-07
SUPERSEDED by §67 (named variants) + §74 (audit) + §96 (Gumbel desktop A/B). C1 KL-loss-dead bug resolved at amendment; affected only loss reporting, not training dynamics. Forensics: `reports/sprint_archive/§066_c1-bug-impact-amendment.md`.

### §67 — LR scheduler bug + total_steps / decay_steps co-design + named Gumbel variants — 2026-04-07
`trainer.py:145` hardcoded `T_max=50_000` fallback collapsed LR to `eta_min=1e-5` at step 50K while bootstrap was still dominant. Fix: `_build_scheduler` raises if `total_steps`/`eta_min` absent. Locked: `total_steps: 200_000`, `eta_min: 2e-4`, `decay_steps: 70_000` (rule of thumb `decay_steps ≈ 0.35 × total_steps`). Three named variants in `configs/variants/`: `gumbel_full` (Gumbel root + CQ), `gumbel_targets` (CQ only), `baseline_puct` (neither). Restart from `bootstrap_model.pt`. Forensics: `reports/sprint_archive/§067_lr-scheduler-fix-gumbel-variants.md`.

### §68 — Eval DB run_id bug + broken-run cleanup — 2026-04-07
`EvalPipeline.run_id` was stored but never passed to 5 DB call sites → all runs collapsed onto `run_id=""`. Fix: thread `run_id=self.run_id`. Reference opponents keep `run_id=""` as shared anchors. §67 broken runs archived to `archive/checkpoints.broken-202604/`. Forensics: `reports/sprint_archive/§068_*.md`.

### §69 — Config Sweep 2026-04-08 — PUCT/Gumbel knob ranking — HISTORICAL
Superseded by §90 throughput baseline. 15+1 runs on laptop varied `training_steps_per_game`, `max_train_burst`, `max_game_moves`, `inference_max_wait_ms`, `leaf_batch_size`, `inference_batch_size`, `n_workers`, `gumbel_m`. **P3 winner config remains live** (tsp=1.5, burst=16). Full sweep data: `archive/sweep_2026-04-08/`. Forensics: `reports/sprint_archive/§069_*.md`.

### §70 — Phase 4.0 overnight run: mode collapse diagnosis — 2026-04-09 → 2026-04-10
**RESOLVED at §73**. Root cause: Dirichlet root noise was wired through Python `SelfPlayWorker` (eval/benchmark) but unported to `engine::SelfPlayRunner` at the 2026-03-30 Rust migration. 30/30 trace records `site: game_runner`, 0/30 `site: apply_dirichlet_to_root`. MCTS rubber-stamped a sharp prior → deterministic fixed point → no gradient signal. Dashboard `policy_entropy` averaged pretrain stream with collapsed selfplay stream, masking collapse for 16,880 steps. Restart from `bootstrap_model.pt`. Forensics: `archive/diagnosis_2026-04-10/`, `reports/sprint_archive/§070_*.md`.

### §71 — Pre-Dirichlet-fix cleanup + Gumbel fallback verification — 2026-04-10
Gumbel SH functionally active on training path (30 trace records, `top_visit_fraction` 0.24 vs PUCT 0.65 — Sequential Halving spreads budget). `policy_entropy_pretrain` / `policy_entropy_selfplay` split landed; collapse threshold 1.5 nats on selfplay stream. Pre-run checklist (10 items) walked before next sustained. Forensics: `reports/sprint_archive/§071_*.md`.

### §72 — Bench rebaseline: NVIDIA driver/boost-clock step-change — 2026-04-09
Three `bench.full` runs failed same two §66 targets (NN inference ~8,370 vs 8,500; worker throughput ~541k vs 625k). Cold/hot/idle ruled out thermals (49°C). Root cause: `DynamicPowerManagement=3` driver settled GPU into lower boost-clock bin; NN latency 1.59ms → 1.77-1.80ms (~14%). Rebaselined: NN inference ≥ 8,250 pos/s, worker throughput ≥ 500,000 pos/hr. Treat as hardware-state drift. Forensics: `archive/bench_investigation_2026-04-09/`, `reports/sprint_archive/§072_*.md`.

### §73 — Dirichlet root noise ported to Rust training path — 2026-04-10 (Q17 RESOLVED)
Commit `71d7e6e`. New `engine/src/mcts/dirichlet.rs` (Gamma-normalize sampler, `rand_distr 0.5`). `engine/src/game_runner.rs` calls `apply_dirichlet_to_root` on every turn boundary, both PUCT (line 550) and Gumbel (line 465) branches, with intermediate-ply skip (`moves_remaining==1 && ply>0`). Verification: 10 unique noise vectors at cm=0, top-1 prior `0.540 → 0.412`, top-1 visit fraction `0.474` vs §70 baseline `0.65` (−17.6pp). Tests: 108 Rust + 646 Python pass. Forensics: `archive/dirichlet_port_2026-04-10/`, `reports/sprint_archive/§073_*.md`.

### §74 — Gumbel vs PUCT loop audit — 2026-04-10
Three sub-resolutions: §74.1 `get_improved_policy` is PUCT-tree-safe (reads only fields populated by shared expand/backup primitives; unblocks `/analyze` on PUCT trees + Gumbel training targets on PUCT-built trees). §74.2 paired benchmark — batch_fill 100% for both variants on laptop 4060 (16 workers); cross-worker coalescing absorbs per-worker fragmentation; +9.4% Gumbel lead inside noise floor (IQR 39-46%). §74.3 Dirichlet parity regression test (`engine/tests/dirichlet_parity.rs`, 4 tests covering sum-to-one, linear blend, ε=0 noop, intermediate-ply gate). No code change to `game_runner.rs`. Forensics: `reports/gumbel_vs_puct_loop_audit_2026-04-09/`, `reports/sprint_archive/§074_*.md`.

### §77 — MCTS depth & ZOI scope — 2026-04-11
ZOI is post-search only (operates on root visit-count vector for move selection). MCTS tree expands with full radius-8 legal set at all depths. Measured at 200 sims: 360 root children created, 7 receiving visits (B_eff=6.1); mean leaf depth 2.92 plies; top-5 visit share 0.97. Decision: Option A (do nothing) — depth improves automatically as policy sharpens; correct lever is `n_sims` not tree pruning. `docs/01_architecture.md` §36 amended. Forensics: `reports/mcts_depth_investigation_2026-04-11/`, `reports/sprint_archive/§077_*.md`.

### §80 — Eval determinism fix: temperature + random openings — 2026-04-12
Root cause: `Evaluator` constructed `ModelPlayer` without `temperature` arg → defaulted to 0.0 → one-hot argmax → all 50 games bit-identical, BT CIs ±100K. Fix: `eval_temperature: 0.5`, per-game `np.random.seed(seed_base + i)`, `eval_random_opening_plies: 4` (random plies for both sides). Training path untouched. Old behaviour restored via `eval_temperature: 0.0`. Forensics: `reports/sprint_archive/§080_*.md`.

### §84 — Eval checkpoint retention (two-tier) — 2026-04-13
Storm: `checkpoint_interval=500 × max_checkpoints_kept=10 = eval_interval=5000` → every eval ckpt evicted by next eval (DB has BT ratings but weight files gone). Fix: `preserve_eval_checkpoints: true` default; `prune_checkpoints()` accepts `preserve_predicate` lambda exempting steps matching `s % eval_interval == 0`. Test `test_eval_checkpoints_not_pruned` pins the contract. Forensics: `reports/sprint_archive/§084_*.md`.

### §85 — A1 aux target alignment (Python + Rust) — 2026-04-13
Three compounding A1 subproblems: (1) `get_aux_targets()` pulled aux from a 200-entry game-level ring with independent random indices (no relation to batch indices); (2) one aux map per game broadcast across ~60 rows; (3) aux maps projected to game-end bbox centroid, replay rows to per-row cluster window centre (offsets ±9 cells). Fix Option A: per-row `ownership` + `winning_line` u8 columns reprojected at game end using each row's `(cq, cr)`; `apply_sym` extended for 12-fold scatter consistency; `sample_batch` 5-tuple → 6-tuple. ReplayBuffer +722 B/row. Rust commit `faafc43`; Python this entry. Kill criterion **revised at §91** (C1-C4 contrast + top-K). Forensics: `reports/sprint_archive/§085_*.md`.

### §86 — Structural split of `replay_buffer/` + `game_runner.rs` — 2026-04-13
Pure refactor. `replay_buffer/mod.rs` 1,102 → split into `{mod,storage,push,sample,persist,sym_tables}.rs`. `game_runner.rs` 1,313 → split into `game_runner/{mod,worker_loop,gumbel_search,records}.rs`. PyO3 surface stable. `Cargo.toml` feature wiring: `default = ["extension-module"]`, `test-with-python` escape hatch. 113 Rust tests pass with zero body modifications. Forensics: `reports/sprint_archive/§086_*.md`.

### §88 — Python training stack refactor — 2026-04-13
`scripts/train.py` 1,132 → 319 LOC (CLI + config + build core objects → `run_training_loop`). New modules: `hexo_rl/training/aux_decode.py` (69), `batch_assembly.py` (297), `loop.py` (680). `trainer.py` 746 → 720. Public API stable. 119 Rust + 676 Python tests pass. Forensics: `reports/sprint_archive/§088_*.md`.

### §89 — Threat-logit probe committed as step-5k kill criterion — 2026-04-13 (REVISED §91)
`scripts/probe_threat_logits.py` + `tests/test_probe_threat_logits.py` + `fixtures/threat_probe_positions.npz` (20 curated positions) + `fixtures/threat_probe_baseline.json`. Make targets: `probe.bootstrap`, `probe.latest`, `probe.fixtures`. FP32 forced, `torch.manual_seed(42)`, deterministic. **Criterion locked at §91** (C1 contrast + C2/C3 top-K + C4 warning-only). Forensics: `reports/sprint_archive/§089_*.md`.

### §90 — GPU util sweep: inf_bs / wait_ms levers exhausted — 2026-04-13
`(inf_bs, wait_ms)` sweep A=(64,4), B=(128,8), C=(128,4). Bottleneck is NN forward latency (12.5ms live vs 1.6ms bench), not batcher config. Raising inf_bs grows mean batch but forwards/sec collapses (workers can't supply 128 leaves in same wall-clock). **pos/hr is NOT a sufficient summary** when game length shifts — future sweeps must report steps/hr. Architectural levers (CUDA streams, process split, `torch.compile` re-enable) deferred to Q18. Forensics: `archive/sweep_2026-04-13_gpu_util/`, `reports/sprint_archive/§090_*.md`.

### §91 — Threat-probe criterion revised: colony-spam, not BCE drift — 2026-04-14
§85/§89 C1 (`ext_logit_mean ≥ baseline − 1.0`) was a BCE scale-drift detector misfiring on healthy runs (ckpt_00014344: contrast +3.94 IMPROVED while abs logits drifted globally negative — opposite of ckpt_19500). Replaced with **C1 contrast floor + C2 top-5 + C3 top-10**; C4 warning-only. `BASELINE_SCHEMA_VERSION` 2 → v6 later (real fixture, §106). Q19 `pos_weight ≈ 59` flagged for §92. Forensics: `reports/sprint_archive/§091_*.md`.

### §92 — Q13 + Q13-aux + Q19 atomic landing (PARTIALLY SUPERSEDED §97) — 2026-04-14
56-file atomic commit: 18→24 plane break, chain-length post-placement semantics, `aux_chain_weight=1.0`, threat `pos_weight=59`, HEXB v2 → v3. Pretrain v3 broken by F1 augmentation bug — fixed at §93 v3b. **§97 reverted chain planes from NN input** (chain stays as aux target in dedicated sub-buffer); design decisions survive. Q21 (wider-window aux) parked. Forensics: `reports/sprint_archive/§092_*.md`.

### §93 — Q13 fix-up + F1 root cause + pretrain v3b — 2026-04-15
10-commit fix-up on `feat/q13-chain-planes`. F1: `_apply_hex_sym` had two bugs — no `axis_perm` remap on planes 18..23 + `(col=q,row=r)` vs `(row=q,col=r)` convention mismatch. Fix: route pretrain augmentation through Rust `apply_symmetries_batch`. F3 tensor-buffer parity guard caught real divergence but in dead code (`TensorBuffer` only called by Python `SelfPlayWorker.play_game` — production uses Rust `SelfPlayRunner`); deleted at C9.5. v3b: 15 epochs, policy_loss 2.18, value_loss 0.50, 100/100 vs RandomBot. **Threat probe baseline v4**. Forensics: `reports/q13_fix_26_04_15.md`, `reports/sprint_archive/§093_*.md`.

### §94 — Experiment A: `aux_chain_weight=0` fresh run — 2026-04-15
Result: 47.7% draw rate at step 10312 vs smoke_v3b 44.7% at 5003. Chain aux confirmed **NOT primary draw-collapse driver**. Forces §95 (input ablation). Forensics: `reports/smoke_v3b_5k_26_04_15.md`, `reports/sprint_archive/§094_*.md`.

### §95 — Experiment C: chain-plane INPUT ablation — 2026-04-16
Audit confirmed chain planes computed at encode-time + stored, NOT recomputed at sample-time; zero planes are symmetry-invariant. Design: zero planes 18-23 after H2D decode (don't remove from architecture). `zero_chain_planes: bool` config flag. Wired in trainer, inference_server, probe_threat_logits. 5 tests. Forensics: `reports/sprint_archive/§095_*.md`.

### §96 — Experiment E: Gumbel MCTS desktop (vs laptop Exp D PUCT+CQ) — 2026-04-16
Hardware: Ryzen 3700x + RTX 3070 (desktop). Variant `gumbel_full`. Bootstrap v3b. Kill conditions relaxed per Exp D learnings. Q26 [WATCH]: nested `training:` block in `gumbel_targets_desktop.yaml` not picked up by deep-merger. Forensics: `reports/sprint_archive/§096_*.md`.

### §97 — Remove chain planes from NN input: 24ch → 18ch — 2026-04-16
KrakenBot architectural alignment. Chain learned via aux loss, not input ingestion. Removed redundancy. `GameState.to_tensor()` returns (K, 18, 19, 19); `HexTacToeNet.in_channels: 24 → 18`. Rust replay buffer chain planes moved to separate `chain_planes` sub-buffer (`6 × 361 × u16` per slot); HEXB v3 → v4. `sample_batch` returns 6-tuple. `apply_chain_symmetry` separate pass with `axis_perm` remap. Old HEXB v1-v3 buffers incompatible. Forensics: `reports/sprint_archive/§097_*.md`.

### §98 — Bench rebaseline post-18ch migration — 2026-04-16
Two failures: buffer_sample_augmented 1,663 µs (≤ 1,400 target) — real regression from split-pass scatter (state 18 + chain 6, two non-contiguous regions); worker_throughput 30,893 pos/hr (≥ 500K target) — warmup design bug (30s insufficient on laptop; p25=0 means at least 2/5 windows measured 0 positions). Rebaselined: `buffer_sample_aug ≤ 1,800 µs`; `worker_throughput ≥ 250,000 pos/hr` (PROVISIONAL — needs longer warmup). Forensics: `reports/sprint_archive/§098_*.md`.

### §99 — BatchNorm → GroupNorm(8) migration — 2026-04-16
Motivation: BN running stats drift from live distribution during self-play; batch=1 MCTS leaf eval uses stale stats. KrakenBot uses GN(8, 128) throughout. `_GN_GROUPS=8` module constant; `policy_bn` + `opp_reply_bn` removed (2 output channels too few for GN). `normalize_model_state_dict_keys` raises `RuntimeError` on pre-GN key patterns (prevents silent trunk corruption via `strict=False`). **BREAKS pre-§99 checkpoints**; retrain from scratch. Forensics: `reports/sprint_archive/§099_*.md`.

### §100 — Selective policy loss (move-level playout cap) — 2026-04-16
Per-move coin-flip between full-search (600 sims) and quick-search (100 sims); each position tagged `is_full_search ∈ {0,1}`. Policy + opp_reply losses gated on flag in Python; value/chain/ownership/threat apply to all rows. HEXB v4 → v5 (+1 u8/row, flag not under symmetry). §100.c review fixes: H1 RecentBuffer carries flag; H2 BN→GN auto-migration reverted (numerically wrong); M1 mutex (`fast_prob > 0` AND `full_search_prob > 0` raises at pool init); M3 opp_reply gated. **Defaults: `fast_prob: 0.0`, `n_sims_quick: 100`, `n_sims_full: 600`, `full_search_prob: 0.25`** (avg ~225 sims/move ≈ KrakenBot). §100.d threat probe baseline regenerated v4 → v5 against live bootstrap. Forensics: `reports/sprint_archive/§100_*.md`.

### §101 — Graduation gate with anchor model — 2026-04-16
Self-play workers consume `inf_model`. `inf_model` syncs from `best_model` anchor only at (a) cold-start and (b) graduation — **never** from drifted `trainer.model` on checkpoint ticks. Monotonic data quality between graduations. Gate: `wr_best ≥ promotion_winrate (0.55)` AND `ci_lo > 0.5`. §101.a review fixes: **C1 promoted weights ≠ evaluated weights** — `eval_model` allocated once in outer scope; promotion loads `best_model ← eval_model`. H1 `eval_interval` reads from `training.yaml` override. M1 `require_ci_above_half` default true (drops false-positive rate <1% vs ~9% naïve). M2 `resume_anchor_step_mismatch` warning. `_sync_weights_to_inf()` (wrong-direction) deleted. Per-opponent stride: `best=1 n=200 / sealbot=4 n=50 / random=1 n=20`. Forensics: `reports/sprint_archive/§101_*.md`.

---

## §103–§157 — Phase 4.0/Phase B' arc (KEEP-DISTILLED)

### §103 — Corpus zero-chain fix + `baseline_puct` playout-cap pin — 2026-04-17
Two drift bugs from §97/§100. §103.a: `load_pretrained_buffer` padded corpus chain planes with `np.zeros((T,6,19,19))` → chain head pulled toward zero on pretrain fraction of every mixed step. Fix: compute chain planes from stored stone planes 0,8 at NPZ load; route /6 normalisation through float32. §103.b: `baseline_puct.yaml` had no `playout_cap` override → inherited post-§100 `full_search_prob: 0.25`, silently §100-selective. Pin `playout_cap.full_search_prob: 0.0`. Tests `test_corpus_chain_target.py`, `test_baseline_puct_pins_pre_100_semantics`. Forensics: `reports/sprint_archive/§103_*.md`.

### §104 — D-Gumbel / D-Zeroloss instrumentation — 2026-04-17
Monitoring-only; no behavior change. `compute_policy_target_metrics` returns 7 fields split by `is_full_search`. Single `.cpu().tolist()` over 7 packed scalars replaces 7 `.item()` syncs (<200 µs/call on CUDA at B=256, A=362). NaN as first-class signal. Gate `monitoring.log_policy_target_metrics: true`. **D-Gumbel verdict: Option A** — quick-search CQ targets drift toward uniform (ΔH +3.5 nats, above +1.5 threshold); §100 selective gate correctly discards. `gumbel_full.yaml` Option A landed: `fast_prob: 0.25 → 0.0`. Resolves §100 known follow-up 3. Forensics: `reports/gumbel_target_quality_2026-04-17.md`, `reports/sprint_archive/§104_*.md`.

### §105 — Q27 perspective-flip smoke: W1 necessary, not sufficient — 2026-04-18 → 2026-04-19 (SUPERSEDED §106)
**Postscript:** §106 supersedes. C2/C3 20%/20% identical-FAIL was a synthetic-fixture artifact, not training pathology. The correctness argument for W1 (commit `e9ebbb9`, three call sites that failed `parent.moves_remaining==1` Q-negation — `get_improved_policy`, Gumbel score, `get_top_visits`) stands independently. Two-machine smoke (laptop pre_fix `723615e` vs desktop post_fix `a7efa78`), 5K steps each from `bootstrap_model.pt`. Forensics: `reports/q27_perspective_flip_smoke_2026-04-18/`, `reports/sprint_archive/§105_*.md`.

### §112 — Q33-C2 augmentation discriminator (E1 confirmed) — 2026-04-21
Unblocks §111 HALT. `feat(training): expose augment as training.augment config knob` (commit `eb17389`). Two 25-min smokes from `checkpoint_00017000.pt`, `w_pre=0`. **Result: |Δpe_Q4| = 0.049 ≪ 0.5 threshold** — augmentation-off does NOT reduce `pe_self`. **Verdict E1: pe_self ≈ 5.4 fixed point is self-play-distribution behaviour, not a 12-fold augmentation rotation bug.** Q33 / Q37 **RESOLVED (non-pathology)**. Phase 4.5 unblocked on pe_self premise. Config knob `training.augment: true` (default) with hard `ValueError` on missing key at loop entry. Forensics: `reports/q33c2_augmentation_discriminator_2026-04-21.md`, `reports/sprint_archive/§112_*.md`.

### §114 — bootstrap-v4: full-corpus retrain + eval (SUPERSEDED §148) — 2026-04-22
**L1, L15 origin.** Two silent corpus bugs found post-§70/§85: (1) `POSITION_END=50` truncation in `export_corpus_npz.py` discarded all ply ≥ 50 (~40% of positions); bootstrap was endgame-blind. (2) `update_manifest.py` read `player_black_elo`/`player_white_elo` (old format), missed `players[].elo` — 5694/5706 games unrated; Elo-weighted sampling effectively off. Fix sequence: `aa16624` Elo fallback; `ddd408f` drop POSITION_END cap (305,410 positions); `8b446c5` POSITION_END=150 (P95.5, trims time-scramble). Eval v4: C1 contrast +0.36, H2H WR 67%, SealBot WR 18.7% (n=150). Retcons Q17: Dirichlet (§73) was necessary but corpus completeness was the structural fix. **Rule: verify corpus before tuner**. Superseded by v5 (§118), v6 (§134), v7 (§148), v7full (§150). Forensics: `reports/sprint_archive/§114_*.md`.

### §116 — D-ladder investigation + torch.compile retry + landing/revert — 2026-04-23 → 2026-04-24
**D-ladder (curr_10k catastrophic forgetting).** Verdict: P-regressed (distributional), V intact on corpus. D1 curr WR ex-draws 6%; D2 deep matched 4% (deep regression); D3 KL on corpus 0.181 (close); D4 V MSE ratio 1.027 (matched). Mismatch = distributional. Smoking gun: D3-extra early-game synthetic probe — empty board curr argmax agreement with boot = 0%; ply 2-7 curr entropy 5.47-5.70 (≈ uniform). Curr forgot how to open. Hypothesis: replay buffer under-covered early-game positions during sustained run. Revert live ckpt to `bootstrap_model.pt`. Forensics: `reports/investigations/diag_D_20260423/`.

**torch.compile retry GO.** Both §32 blockers resolved on PT 2.11 + Py 3.14: TLS crash gone, Triton 27 GB spike gone (59.5 MB peak). reduce-overhead: 1.50× throughput / 1.87× latency vs eager, 6.4 s compile. **§116.a landing on master (`1e2d82b` + `41ffad5` resume/OptimizedModule fixes), then REVERTED (`e102a0a`)** after second resume deadlock at step 6002 (futex_do_wait on 78 threads — trainer+inference dual-JIT contention). Mode-plumbing + OptimizedModule unwrap fixes stay. **Re-enable preconditions:** deadlock repro harness OR compile-sequencing OR training-loop smoke. Forensics: `reports/investigations/torch_compile_retry_20260423/`, `reports/sprint_archive/§116_*.md`.

### §117 — TF32 + channels_last probe + per-host autodetect — 2026-04-23
**L7 corollary.** Four-arm matrix × two hosts. TF32 cross-host divergent: sm_89 (4060) −5.8% latency; sm_86 (3070) +5.9% (FP32-tail Linears route to small-K TF32 kernel that serializes poorly). channels_last −7 to −17% on both (SE block `s.view(b,c,1,1)` breaks CL propagation 12× per forward). **Decision: TF32 per-host autodetect (`_TF32_MEASURED: sm_86→False, sm_89→True`); channels_last rejected.** Replaces unconditional `torch.set_float32_matmul_precision("high")`. Forensics: `reports/investigations/tf32_channels_last_20260423/`, `reports/sprint_archive/§117_*.md`.

### §118 — Early-game forgetting fix wave — 2026-04-23 → 2026-04-24
**Verdict — root cause:** `pe_self ≈ 5.4` is self-play-starvation rate on **off-canonical** early-game positions, not policy collapse. Under prod (`decay_steps=20000`, `full_search_prob=0.25`), only ~13% of batch policy-gradient came from fresh SP rows; ply 2-7 off-canonical drifted toward `log(N_legal)`. **Axis is canonical vs off-canonical, not ply depth.** Smoke discriminator: A fsp=0.5 → 26% SP-grad share, H 3.97; B decay=2500 → 17.6%, H 3.32. Landed: `early_game_entropy` 10-pos probe; `selfplay.random_opening_plies: 4`; `dirichlet_alpha: 0.3 → 0.05`; `full_search_prob: 0.25 → 0.5`; `pretrain_max_samples: 200_000 → 0`. Phase 5 validation (bootstrap-v5, step 6000): early_game_entropy 3.55 (gate <4.0), threat C1 +3.44, throughput +10%. Q8/Q33/Q37 framing flipped to "sampling-rate starvation". Forensics: `reports/investigations/phase5_validation_20260424/`, `reports/sprint_archive/§118_*.md`.

### §119 — Main-Island Neglect: mechanism located, RecentBuffer augmentation gap — 2026-04-25
User-flagged visual pattern: parallel horizontal formations at equidistant spacing, main island neglected. **5-hypothesis discriminator cascade.** D1 history ordering RULED OUT (10% counterfactual disagreement). D2 windowing coverage RULED OUT (100% largest-group coverage). D3/D4 rotation equivariance PARTIAL (board-coord agreement 51.5% — model is axis-asymmetric but doesn't explain E-W preference). **D5 trajectory: self-play E-W axis share 65% vs corpus 38% (+27pp).** D6 augmentation audit MECHANISM CLOSED: RecentBuffer rows are sampled at Python call site without `apply_sym` LUTs; at step >10k RecentBuffer contributes ~67% of batch → 67% of policy-gradient rows see identity-only symmetry coverage → absolute-position FC policy head learns axis-asymmetric features freely. **Causal chain**: un-augmented RecentBuffer → asymmetric features → axis-biased MCTS → axis-biased trajectories → buffer reinforces 67%. Decision: Option A (augment RecentBuffer in §120). Methodology lesson: when visual pattern is stable + geometrically specific, treat geometry as falsifiable hypothesis first. Forensics: `reports/investigations/main_island_d{1..6}/`, `reports/sprint_archive/§119_*.md`.

### §121 — Directional bias resolves, clustering magnitude is architectural — 2026-04-25
Seven diagnostics (D10-D16) closed mechanism account of axis-clustering. **Two independent components**: (1) Directional heuristic (within-turn, rotation-equivariant) — model places 2nd stone far W of 1st (W=38.2% selfplay vs corpus 12-14%); washes out under D16 rotation probe (W=12.3%, W/E=0.96). Fixed by **permanent self-play rotation from ply 0** (landed §130). (2) Clustering magnitude (cross-turn, rotation-invariant) — aggregate axis density 0.60 per axis doesn't shift under rotation; learned strategic prior "identify axis early and cluster"; preserved by rigid transformation. **Architectural**: trunk lacks inductive bias for hex-axis strategies at right abstraction level. §120 RecentBuffer augmentation closed symmetry gap (~4.7/12 → 12/12 elements per batch row) but cannot correct relational biases. Methodology lesson: split signal (different components resolve differently) is more informative than uniform PASS/FAIL. §122 opens for architectural redesign. Forensics: `reports/investigations/phase121_d{10..16}*/`, `reports/sprint_archive/§121_*.md`.

### §124 — InferenceServer dispatch fix: TorchScript trace + bench methodology shift — 2026-04-25
**L17 corollary.** TorchScript-trace `InferenceServer.forward` at `__init__` via `torch.jit.trace`; gated by `selfplay.trace_inference` (default true); falls back to untraced module if trace raises (e.g. compile-wrapped). ScriptModule shares parameter storage so `load_state_dict_safe` propagates without re-tracing. Merges D2H (`cat(probs, value)` → single `.cpu().numpy()`). Eliminates ~32.6% CPython `nn.Module._call_impl` dispatch cost per forward (py-spy 200Hz, 180s, n_workers=10, 3070). **+34% pos/hr on 3070.** Trace neutral on EPYC 4080S (still GPU-bound). Why trace not compile: simpler (no Dynamo guard, no cudagraph TLS, no Triton spike); ~matches compile throughput on GPU-bound HW while lifting dispatch-bound HW. `make bench` switches to compile OFF + trace ON as production gate; `make bench.compile` retained for engineering datum. NN inference target lowered 6,500 → 4,000 pos/s. Forensics: `reports/sprint_archive/§124_*.md`.

### §125 — EPYC 4080S sweep verdict + py-spy → perf_timing — 2026-04-25
**L7 origin.** Validate sweep on EPYC 7702 + 4080S: `n_workers=24, batch=192, wait=4ms, leaf=8, burst=16` → 377k pos/hr. n_workers=16 regressed 388k → 234k with trace (workers can't refill 192-batch before GPU drains). py-spy 0.4.2 does not support Py 3.14 — replaced with `diagnostics.perf_timing: true` (per-batch fetch_wait_us / h2d_us / forward_us / d2h_scatter_us). **Profile result (n_workers=24, batch=192, trace ON):** forward = 80.4% of cycle (p50 11.96 ms). EPYC 4080S is **GPU-compute-bound, not dispatch-bound** — 3070 py-spy extrapolation was wrong. Next lever: torch.compile on top of trace (compile fuses CUDA kernels trace cannot); stack path `torch.compile(model).then(jit.trace(compiled._orig_mod, ex))` verified possible, not yet implemented. Forensics: `reports/sprint_archive/§125_*.md`.

### §127 — Top-K leaf cap eliminates MCTS pool overflow — 2026-04-28
**L5 origin.** 5090 96-thread sweep saw `mcts_pool_overflow_count > 0` every cell. Root cause: leaf expansion created one child per legal move; once 100+ stones spread out, radius-8 hex ball per stone unions into 1k+ cells; worst-case nodes per search = `n_sims × leaf_batch × n_legal` blew past `MAX_NODES = 1M`. Pre-existing mitigation (fabricate `is_terminal=true` with quiescence-corrected NN value via AtomicU64 counter) was a **hot-path silent-corruption sink** — biased visit counts and value targets, contaminated runs dropped after the fact. Fix: `MAX_CHILDREN_PER_NODE: usize = 192` (sorted by NN policy prior desc, tie-break flat_idx asc — deterministic regardless of FxHashSet order). Fast path with no sort when `legal_moves.len() ≤ K`. Fabricated-terminal removed entirely; counter retained for telemetry, then panic. **Bound**: `400 × 8 × 192 ≈ 614k slots fits 1M` (~38% headroom). Tests in `mcts/mod.rs` + `engine/tests/pool_overflow.rs`. Forensics: `reports/sprint_archive/§127_*.md`.

### §128 — Bench metric `positions_generated` replaces `positions_pushed` — 2026-04-28
**L6 origin.** `worker_pos_per_hr` measured via `pool.positions_pushed` which increments by K cluster views × 1 per ply at **game completion** (batch write). 120s window vs ~160s game → most windows capture zero completions → bimodal IQR 80.9%. `positions_generated` is per-ply continuous AtomicUsize — continuous, no burst, stable. Relationship: `positions_pushed = K_avg × positions_generated`, K_avg ≈ 7 empirically. Targets updated: CUDA 142,000 → 20,000; MPS 200,000 → 25,000; CPU 80,000 → 11,000. Desktop n=5 baseline: 27,835 pos/hr IQR ±8.6%, all 5 runs unimodal. **Pre-§128 throughput numbers obsolete**. Forensics: `reports/sprint_archive/§128_*.md`.

### §131 — 18 → 8 plane migration (P1+P2+P3) — 2026-04-29
**KEPT_PLANE_INDICES = [0,1,2,3,8,9,10,11]** (cur ply-0..3 + opp ply-0..3 in 18-plane space; positions 0-7 in 8-plane). Ply-4..7 history + scalar metadata dropped. Dense > sparse for `ckpt_14000` conditioning surface preservation. STATE_STRIDE 6498 → 2888. Chain_planes unchanged at 6 planes. **P1 Rust**: `N_PLANES: 18→8`, HEXB v5 → v6 with `n_planes` header validation, slice-on-push integration; inference path untouched. **P2 Python**: `BUFFER_CHANNELS=8`; `pool.py` `_feat_len = 2888`; `batch_assembly.py` 8→18 scatter bridge in `_train_on_batch` (temporary). Corpus regenerated 1,090,296 positions `(N,8,19,19)`. **P3 Model**: `HexTacToeNet.in_channels: 18→8`; hard-reject guard `trunk.input_conv.weight.shape[1] == 18 → RuntimeError`; bridge block removed; `InferenceBatcher`/`SelfPlayRunner` defaults 6498 → 2888; `worker_loop.rs` `slice_kept_planes_18_to_8` deleted (encode sites call `encode_state_to_buffer_channels` directly). 958 py + 138 rs lib + 35 rs integration pass. Forensics: `reports/sprint_archive/§131_*.md`.

### §137 — W3 validation gates: Q41 WARN + Q52 PASS + Q44 done → Phase 4.0 UNBLOCKED — 2026-04-30
Q41 v6 vs v5 H2H (n=200): 102/200 (51.0%), Wilson [44.1%, 57.8%] — WARN (near-parity). Old gate `lower-CI ≥ 50%` fired even at exact parity; **revised**: PASS ≥ 48%, WARN [43%, 48%), BLOCK < 43%. Q52 v6 vs SealBot (n=150): 36/150 (24.0%) Wilson [17.9%, 31.4%] — **PASS** (gate ≥ 14%). Beats v4 anchor 18.7% by +5.3pp. **Colony-win fraction: 5.6% vs v4's 82%** — low colony fraction is **POSITIVE** (colony wins caused self-play training explosion; §131 channel cut dropped colony-related planes; v6 wins via 6-in-a-row). Q44 laptop bench refloor: 33,174 pos/hr (+19% vs desktop §128). Phase 4.0 UNBLOCKED. Forensics: `reports/sprint_archive/§137_*.md`.

### §141 — W4C policy-head diagnosis: policy intact, locus is search/encoding — 2026-05-01
ckpt_5500 (W4C smoke §138) recorded 1.3% SealBot WR vs bootstrap-v6's 24%. §139-§140 cleared value head + rotation LUT. **Policy probe (n=200 × 5 metrics × 4 categories × 2 models)**: H(p) decreased on real positions (sharpened, not flattened — falsifies Hypothesis A); Spearman ρ ≥ 0.66 on corpus/sealbot/threat; top-1 agreement ≥ 64%; threat extension prob retained at 94% of bootstrap. Colony positions diverged (top-1 agreement 18.5%, rank 201/362) — desired §137 behaviour, not defect. **Verdict: policy head NOT regression locus.** Protocol fixes (pretrain floor 0.1→0.5, max_game_moves 200→100) unlikely to help. Self-play board-extent 329 cells, NN window 19×19 (±9) — most board invisible. Locus: **search/encoding boundary**. Forensics: `reports/w4c_diag/policy_diagnosis.md`, `reports/sprint_archive/§141_*.md`.

### §142 — Encoding-window coverage audit: ply-31 fragmentation pivot — 2026-05-01
ckpt_5500 selfplay crosses 19×19 single-window boundary at **ply 31** (median pct_outside 0% → 21.9%, sharp). Any-cluster windowing delays onset but doesn't prevent: 8/16 draws end with ≥80% of stones invisible. End-of-game single-window blindness median 97.7% on draws. **Pathology is distribution-endogenous** — against SealBot (5 games) max ply 29, 0% outside throughout; fragmentation only emerges when two mutually permissive policies play each other. Axis: q (NE-SW) consistent with §138 axis_density. **Recommendation: Option γ (tighten exploration)** — Option α (cap LEGAL_MOVE_RADIUS) is fallback. Forensics: `reports/w4c_diag/encoding_audit.md`, `reports/sprint_archive/§142_*.md`.

### §146 — Option α' implementation: cap LEGAL_MOVE_RADIUS 8 → 5 — 2026-05-02
`engine/src/board/moves.rs:9` `LEGAL_MOVE_RADIUS: 8 → 5`. CLUSTER_THRESHOLD untouched (still 8 — governs colony adjacency). Tests: 216 → 90 (single-stone hex ball: 91-1), new `legal_move_radius_capped_at_5`. Laptop smoke (21 games, bootstrap-v6, gumbel_full): draw_rate 0/21, mean game length 16 plies (R=8 was ~110). With p=0.2 the n=0 draws prob is 0.9% — real shift. **Bandaid; SUPERSEDED at §156 (cosine-temp is the real load-bearing knob).** Forensics: `reports/sprint_archive/§146_*.md`.

### §147 — Bootstrap corpus contamination audit — 2026-05-03
**L1, L15 origin.** Pre-§148 audit of `data/bootstrap_corpus.npz` (v6) found it was generated by `make corpus.export` (all sources), NOT `make corpus.export.pretrain` (human-only). `pretrain_human_only: true` flag was not honored. Bot games made up **41% of raw game count** at `source_weight=1.0`. Per-position Elo-band weights also not applied. **v6 anchor numbers tainted** — Q41 51% parity and Q52 24% vs SealBot trained against bootstrap that learned partly from bot-style play. Phase 4.0/Phase B' diagnostics (§141-§144) inherit caveat. **Decision: rebuild v7 from scratch with `make corpus.export.pretrain`**. Preserve v6 as `bootstrap_corpus_v6.npz`/`bootstrap_model_v6.pt`. Forensics: `reports/sprint_archive/§147_*.md`.

### §148 — Corpus rebuild: v7 human-only Elo-weighted — 2026-05-03
**Elo-weight bug fix in `scripts/export_corpus_npz.py`:** when `--max-positions` omitted (uncapped), `rng.choice(n, n, replace=False, p=w)` degenerates to permutation — per-position Elo weight had no effect on which positions kept; `weights_out = np.ones(...)` made `WeightedRandomSampler` sample uniformly. Patched: when uncapped, save per-position `source_weight × elo_band_weight / game_length` as `weights_out`. v7 corpus: 6,259 human games (≥15 plies decisive), 355,271 qualifying positions, 353,091 sampled, 2,435 MB, fp16 `(N,8,19,19)`. Elo bands: sub_1000=81,985 (w=0.5), 1000_1200=202,111 (w=1.0), 1200_1400=69,739 (w=1.5), 1400+=1,436 (w=2.0). v7 retrain (15 ep, batch 256, ~97 min on 3070): final loss 3.31, val_acc 0.75, 100/100 vs RandomBot. **SealBot n=200**: v7 16% vs v6 11% (z=1.46, p=0.14 n.s.); H2H 49%; threat C1 +0.00 (corpus-shift artifact) but C2 45% / C3 75% PASS. **Promoted**. HF push `timmyburn/hexo-bootstrap-{corpus,models}` versioned + canonical. Forensics: `reports/corpus_v7/promotion.md`, `reports/sprint_archive/§148_*.md`.

### §149 — v7 verification + v7e30 fine-tune promotes — 2026-05-03
Pretrain saturation audit on v7 found last-3-epoch cumulative Δ = 1.6% of total descent — fails strict <1% gate (cosine LR reached eta_min, idled). Patched `pretrain.py`: `--resume`, `--lr-peak`, `--inference-out` flags. **v7e30 fine-tune**: resumed `pretrain_00000000.pt`, fresh cosine peak `5e-4 → 1e-5` for 15 more epochs; final loss 3.246. **SealBot n=500**: v6 11.4%, v7 13.2%, v7e30 **16.4%** (z=2.29 vs v6, **p=0.022 significant**). Threat C2/C3 preserved through fine-tune; C1 still flat (corpus-shift + flatter v7-family distribution). Human-fixture C1 probe: v7 +0.06 vs v6 +0.51 (lower), but v7 C2 42% vs v6 25% (+17pp) — flatter policy with preserved/improved top-K ranking. **v7e30 promoted to canonical**. Hygiene: `make train.fresh` wipes `replay_buffer.bin`; `buffer_state_at_corpus_load` event; Q41 BLOCK threshold 43% → 38%. Forensics: `reports/sprint_archive/§149_*.md`.

### §150 — v7full: 30-epoch full retrain promotes — 2026-05-03
**Canonical bootstrap anchor.** User ran full retrain on vast.ai 5080: single-cycle cosine 30 epochs, peak `2e-3`, **`eta_min=5e-5`** (raised from `1e-5`). Wall ~83 min. Final loss **3.1573** (vs v7e30 3.2462, Δ -0.089). `--eta-min` flag added (`1f822ae`). **SealBot n=500: 87/500 = 17.4% [14.3%, 21.0%]** (z=2.70 vs v6, **p=0.007 significant**; vs v7e30 z=0.42 p=0.67 n.s. but every metric directional + v6-anchor edge becomes significant). Threat: C1 +0.20 (partial recovery from v7e30 0.0 toward v6 0.6); C2 50% / C3 70% **PASS**. Colony wins 12/87 = 13.8% (line baseline). Promoted; canonical `bootstrap_model.pt` = v7full sha `29306533…`. **Recipe locked**: 30 ep cosine, peak `2e-3`, `eta_min=5e-5`. Forensics: `reports/corpus_v7/threat_probe_v7full.md`, `reports/sprint_archive/§150_*.md`.

### §152 — Phase B' instrumented smoke: Class-4 dominant — 2026-05-04
Run `w4c_smoke_v6_instrumented_5080` aborted at step 2560/5000 after four-class signal saturated. **Verdict matrix**: Class 4 (q-axis stride-5 distance-5 spam) DOMINANT, ρ(stride5_run, is_ply_cap) = +0.50, p = 5e-42; Class 3 (buffer composition) STRONGLY ACTIVE downstream; Class 2 (value-head drift) ACTIVE downstream (dec=−0.69±0.03); Class 1 (stale dispatch) NOT TESTED (eval_interval=5000 zeroed model_version). **Causal story**: v7full mildly prefers stride-5 east-west extensions; smoke γ knobs + Dirichlet + playout_cap full=0.5 sims=600 + cosine temp amplify into dominant policy mode (cap_max median 30 vs T2 baseline 3 = 10×). 87% cap-rate → 98% draw-coded buffer → value head trains to overshoot draw_value to −0.69 → reinforces cap-prone policy. **Pattern**: mixed-color stones along single hex row at distance-5 spacing (`x____o____o____x_`). Persists despite §130 rotation (per-game uniform across dihedral group). Existing macros miss it: `colony_extension_fraction` gates at hex_dist > 6; `axis_distribution` measures distance-1 adjacency. v8 priority order written. Forensics: `reports/phase_b_prime/instrumented/diagnosis.md`, `reports/sprint_archive/§152_*.md`.

### §154 — v9 hex-trunk turn FALSIFIED — 2026-05-05 (FALSIFIED ROW: Falsified Hypotheses Register §154)
HexConv2d + corner_mask + per-move rotation cadence on `phase_b_prime_v9_hex_native`. Probe gates PASS but selfplay drops to 0-1% SealBot WR. **Mechanism**: probes cannot validate dynamic equivariance — only MCTS-matched eval can. Class-5 eval-gate guard (colony-attractor on wr_best) now v10 priority 1. Branch retained as architecture-research substrate (knobs default off; production paths unaffected). Memory: `project_phase_b_prime_v9_falsified.md`.

### §155 / §156 / §157 — Cosine-temperature draw-collapse arc — 2026-05-05 → 2026-05-06 [MERGE]
**L3, L9 origin.** §155 T1 ran R0-R5 knob-isolation harness (`scripts/v7full_training_knob_isolation.py`) ruling out Dirichlet / cosine-temp / opening_plies / parallelism alone (all ≤ 5.5% draws). T1.1 ran R6-R10 MCTS-side: only **R10 (full conjunction)** hit 91.0% [86.2%, 94.2%] draws under frozen v7full **without training updates** — proximate cause is super-additive interaction of smoke MCTS regime + exploration knobs. FALSIFIED by §156. T2 bootstrap-floor multi-anchor gate landed default-off (`bootstrap_anchor` opponent + `bootstrap_floor.min_winrate=0.45` AND-combined with `wr_best ≥ 0.55`).

**§156 R11-R14 bisection (each removes one knob from R10):**
| Variant | Knob removed | draws | draw_rate |
|---|---|---:|---|
| R10 | (full smoke regime) | 182/200 | 91.0% |
| R11 | Dirichlet ε=0.10 → 0 | 176/200 | 88.0% NULL |
| **R12** | **cosine temp → fixed τ=0.5** | **10/200** | **5.0% LOAD-BEARING** |
| R13 | opening_plies 1 → 4 | 170/200 | 85.0% NULL |
| R14 | playout cap → uniform 600 | 198/200 | 99.0% INVERSELY load-bearing |

**Load-bearing knob = cosine temperature schedule.** Fix: `temperature_threshold_compound_moves: 0` + `temp_min: 0.5`. Mandatory pairings: `legal_move_radius_jitter: true` (R12 colony rate 67% mitigation), `bootstrap_floor.enabled: true min_winrate: 0.45`. R12 colony rate = 67% is the §147 v5 / §154 v9 colony attractor — jitter holds it at trace levels.

**§157 5k validation on 5080 (commits 9412a38, 83be4d7, f2e4555):** wall 3h 18m, draw_rate 7.5% last-200, stride5 P90=4, row_max P90=13, colony_ext_frac max 0.086, 0 NaN. **SealBot offline eval n=200 final-ckpt: 19.0% (38/200)** — beats 17% gate, matches §150 v7full 17.4% within sample noise (Δ +1.6pp). wr_anchor 0.28 → 0.42 → 0.37 (bootstrap-floor refused all sub-floor promotions correctly). **Operator decisions (Gate 6):** Path B selected (skip 40k sustained, pivot to encoding migration); `legal_move_radius_jitter: true` propagated to top-level; bootstrap-floor default-on + frozen v7full path; cosine-temp NOT propagated (variant-pinned).

**§157 follow-ups:** #1 `eval_interval` must be ≥ 2500 (smoke eval cadence at 500 produced 6 skipped events; only 3/10 fired); #2 stride5/row_max dashboard-only on master; #3 final eval skipped on `--iterations` exit; #4 `sealbot_colony_bug_risk` legacy guard; #5 draw_rate is NOT abort signal (user verdict: draws are model missing open-4s, not pathology).

Forensics: `reports/phase_b_prime/training_knob_isolation/`, `reports/phase_b_prime/5k_smoke/results.md`, `reports/sprint_archive/§155-§157_cosine-temp-draw-collapse-arc.md`.

---

## §158–§163 — Hygiene + Refactor waves — 2026-05-06 [MERGE]

**Hygiene wave (§158, §158a, §158b):** L3 variant-config cleanup. §158 retired 6 superseded variants (`smoke_A/B`, `w4c_smoke_v6_*` family) + 3 stale docs. §158a coordinated wave: `phase118_recovery.yaml` + dead `dry_run_batch.py`; `calib_R1-R4.yaml` + `run_calibration_run.sh` + Makefile target; `sweep_*ch.yaml` + Phase-1 harness (`run_sweep.py`, `tournament_sweep.py`); `baseline_puct.yaml` + coordinated references. 924 py pass post-wave. §158b L8 Stage 3 disk reclaim: ~9 GB freed (workspace 48 → 39 GB); cumulative L8 total ~58 GB. Commits `98722cb`, `33a324f`, `c1fceaf`, `96f0b27`, `f777922`, `f8c5ccc`. Forensics: `reports/sprint_archive/§158-§158b_hygiene-wave-2026-05-06.md`.

**Refactor wave (§159, §159a, §160, §160a, §161, §162, §163):** Audit-driven structural splits, all FF-merged to master.
- **§159**: `hexo_rl/training/loop.py` 1464 → 686 LOC (-53%). Five new modules: `anchor.py` (anchor I/O), `signals.py` (signal handlers), `orchestrator.py` (training hooks), `lifecycle.py` (subsystem builders), `eval/pipeline_setup.py`. 8 commits + sprint log.
- **§159a**: StepCoordinator extraction. `loop.py` 686 → 357 (-48%); new `step_coordinator.py` 893 LOC (closure → class; 18 per-step decisions O2-O6/D0-D10 preserve identical ordering DAG). +20 behavioural tests. Constraints R1-R25 honored. 4 commits.
- **§160 / §160a**: `eval_pipeline.py` 529 → 472. New `gate_logic.py` (GateConfig + `evaluate_gate` + ci_confidence wired via `norm.ppf`), `reporting.py`. **2 of 4 audit claims stale** (BT + SQLite already extracted pre-§160). §160a: `_load_anchor_model` prefix-strip dedup'd to `normalize_model_state_dict_keys` (inherits BN-key + 18-plane guards).
- **§161**: Q-§159b §B item 15 lifecycle.py coverage (+5 tests for build_subsystems, resolve_anchor branches, teardown).
- **§162**: `selfplay/pool.py` 705 → 539. New `instrumentation.py` 176 (PoolInstrumentation: per-worker draws, terminal-reason counts, mv_range history, recent move histories). Stride-5 abort path retired (Q-§162a WATCH); P90 stays passive event field. Bench gate PASS.
- **§163**: `mcts/mod.rs` 1493 → 974 (-519). New `mcts/policy.rs` 533 (5 methods + 11 tests: get_policy, get_improved_policy, get_root_children_info, apply_dirichlet_to_root, get_top_visits). Pure-move (subagent audit verified). **Audit row PARTIALLY STALE**: 6 of 8 listed concerns already in `selection.rs`/`backup.rs` pre-§163. Bench gate PASS.

Pattern: audit's L9 rows treated as low-fidelity index (claims stale at §160, §160a, §162, §163). Forensics: `reports/sprint_archive/§159-§163_refactor-wave-2026-05-06.md`.

---

## §164–§174 — Encoding migration arc (KEEP-DISTILLED)

### §164 — Phase 5+ entry probe wave: P1 anchor / P2 perception (CATASTROPHIC) / P3 corner-mask — 2026-05-07
Three probes dispatched in parallel; nothing landed on master.
- **P1 (Principled — memory misread).** Live self-play forwards all K cluster views to NN, min-pools value, scatter-max policy. Replay buffer push emits one row per cluster per ply. Index-0 picks exist only in `pretrain.py:564,568` (Aug-only RandomBot validation) + `early_game_probe.py:103` (Aug-only monitoring fixture). Bootstrap corpus picks **first cluster covering played move**, not index 0 — Principled by design.
- **P2 (CATASTROPHIC).** Rule: r=8 placement (hexo v0.2.0); our perception: r=5. **Bot WR 78% vs `far_line` (scripted r=6-8 6-axis adversary)** with 22% opp colony reach-6; brain-dead script outperforms strongest engine's 17.4% empirical edge vs SealBot. Mechanism: stones at hex_dist 6-8 from any bot stone form their **own cluster** (`CLUSTER_THRESHOLD=5`) with their own window; policy treats them as low-priority. **Deployment hotfix REQUIRED before hexo.did.science deploy.** Three options: (a) bump constants to 8 (re-opens fragmentation), **(b) hybrid r=5 selfplay + r=8 inference (PREFERRED short-term)**, (c) 25×25 window (large scope; encoding migration). PyO3 bindings for `set_legal_move_radius` + `set_cluster_threshold` missing.
- **P3 (Neutral within noise).** Corner-mask (hex_dist ≤ 9, 271 cells survive of 361) bench A/B on 5080: aug −8.12%, worker +20% (borderline noise). Smoke OFF vs ON last-100: draws 11% → 7%, X/O/draw 52/37/11 → 47/46/7 (better balance), colony_extension_fraction max 0.030 → 0.0. **SealBot Δ = −2.5pp NOT statistically significant**. Mask safe to ship. **D3 verdict: adopt inscribed hex** on §152 dihedral-symmetry argument.

Forensics: `audit/probes/{p1,p2,p3}*.md`, `reports/probes/p3_smoke/`, `reports/sprint_archive/§164_*.md`.

### §165 — v8 encoding migration design + spike wave — 2026-05-07
Three parallel spike subagents. **S1 bbox crop**: fixed-max single-tensor bbox-of-all-stones at HALF=16, BBOX_SIDE=33, m=8, 9-11 planes, K-aggregation REMOVED, N_ACTIONS=1090. **S2 KataGo GPool**: 2 sites at `{6,10}` (50%/83% of 12-block trunk), `KataConvAndGPool` operator verbatim, `c_main=128, c_mid=128, c_gpool=32`. Trunk FLOPs DECREASE 2.1% at 19×19. SE × GPool compose cleanly. Policy + opp_reply FC → KataGo 1×1 conv + linear_pass = **−482k params (FREE WIN)**. **S3 v8 corpus regen + cutover plan**: ~10-15 min corpus + ~3 hr retrain = 3.25 hr wall on 5080. Raw human JSONs persisted; no re-scrape needed. **Final v8 spec Path β**: 25×25 + 11 planes + R=8 + 96-channel trunk + GPool {6,10} + KataGo policy head + N_ACTIONS=625 (no pass). Phase A-E sequence: short-lived branches off master, FF-merge on bench-gate PASS. Forensics: `audit/encoding_spikes/{s1,s2,s3,SPIKE_SUMMARY}.md`, `reports/sprint_archive/§165_*.md`.

### §166 — Phase A: encoding pipeline core (gated coexistence) — 2026-05-07
Operator-revised strategy: NOT hard cutover. v8 lands as **gated coexistence**: v6 path canonical default + byte-exact; v8 path opt-in via `configs/model.yaml encoding.version: v8`. Preserves rollback envelope, unblocks Phase B without putting v6 at risk. **4 Phase A commits + 1 contract** on `encoding/phase_a_pipeline`:
- Bucket C (`ad8dd10`): `EncodingSpec` NamedTuple + `resolve_encoding(config)` in `hexo_rl/utils/encoding.py`; v8 constants in `constants.py` (BOARD_SIZE_V8=25, NUM_CELLS_V8=625, BUFFER_CHANNELS_V8=11, N_ACTIONS_V8=625, LEGAL_MOVE_RADIUS_V8=8, MARGIN_M_V8=8); `configs/model.yaml` encoding section default v6.
- Bucket A (`ee2de0b`): Rust `sym_tables.rs` shape-parameterized; v8 const symbols (STATE_STRIDE_V8=6875, CHAIN_STRIDE_V8=3750, POLICY_STRIDE_V8=625, AUX_STRIDE_V8=625, HALF_V8=12); `SymTables::with_shape(board_size, n_planes)`; `src_plane_lookup` `[[usize;N_PLANES];N_SYMS]` → `Vec<Vec<usize>>`; `apply_symmetry_state` / `apply_chain_symmetry` read `n_cells` from `sym_tables.n_cells`.
- Bucket B (`66b9f9c`): `hexo_rl/bootstrap/dataset_v8.py` (replay_game_to_triples_v8, encode_position_v8, bbox-of-all-stones centroid integer-truncation + n_clipped telemetry); `_compute_chain_planes` shape-derived; `get_policy_scatters(board_size=25, has_pass=False)`; `--encoding v6/v8` flag on `export_corpus_npz.py`.
- Bucket D (`b47136c`): **P2 hotfix-(c) bundled** — `Board.set_legal_move_radius(r)` + `legal_move_radius()` PyO3 bindings; dataset_v8 replay Board uses R=8.

**Tests**: 1028 py + 151 Rust lib + 6 Rust integration GREEN (41 net new: 9 resolver + 8 v8 sym + 28 dataset_v8 inc. 4 P2). v6 999 py + 143 Rust unchanged → byte-exact regression guard satisfied. **Bench gate** (n=5 post-Phase-A laptop): 8/10 PASS, 2 WATCH (`buffer_push_per_s` -7.7%, `worker_pos_per_hr` -6.2%); n=5 close-out confirmed both as boost-clock variance noise (§102). 5080 baseline for Phase B captured. Forensics: `reports/encoding_phase_a/`, `reports/sprint_archive/§166_*.md`.

### §167 — Phase B v8 variant exploration sprint — 2026-05-07 → 2026-05-08
5-arm matrix on `encoding/phase_b_variants`:

| Arm | Channels | Depth | GPool sites | Head GPool | Notes |
|---|---|---|---|---|---|
| B0 | 128 | 12 | none | no | Control (encoding-shape change only) |
| **B1** | **128** | **12** | **{6,10}** | **yes** | **Primary candidate** |
| B2 | 96 | 12 | {6,10} | yes | Capacity probe (Path β shrink) |
| B3 | 128 | 10 | {5,8} | yes | Depth probe |
| B4 | 160 | 12 | {6,10} | yes | Width probe |

**B1 NaN incident**: 2026-05-07 14:19 retrain hit single-batch fp16 overflow at step -22000 (epoch 14 of 30) in `KataConvAndGPool.linear_g` (3·c_gpool=96 → c_out=128). Standard PyTorch GradScaler chain didn't skip the step — `clip_grad_norm_(NaN, max=1.0)` computed `norm=NaN`, NaN clip_coef, optimizer wrote NaN. **Patch (`4c7dbb5`)**: `if not torch.isfinite(loss): continue` before backward (defense-in-depth, single CPU `isfinite` check, no perf hit). B1 retry: ~24% steps NaN-skipped, final loss 3.227 (best clean). B4 OOM'd at batch=256; fallback batch=128 ran ~80% NaN-skipped (only ~6 epochs effective).

**SealBot WR (argmax-only, n=200, t=0.5): ALL 5 v8 arms = 0/200 = 0%** [0%, 1.9%]. v7full v6-argmax baseline: r=5 6.5% / r=8 12.5% / r=10 15%. B1 across radii r=8/10/12: 0%/0%/0%. **The v8 0% is NOT v8-architecture falsification** — structural limitation of argmax-only cross-encoding eval. K-cluster's inference-time multi-window pooling acts like a tiny "ensemble" that bbox lacks. Both effects vanish under MCTS (Phase D §168). **Canonical pick: B1** (best clean loss 3.227). Threat probe deferred (v6-only fixture). bbox clip rate ~6%/stone (above S1's 1% Path α trigger) — Path α escalation deferred. **v7full radius curve 6.5% → 12.5% → 15% confirms cross-encoding gap real**; memory `feedback_v6_v8_same_training_data.md` (corpora share 6,259 raw games — encoding changes density only) + `project_v8_argmax_handicap.md`. Forensics: `reports/encoding_phase_b/VARIANT_SUMMARY.md`, `reports/sprint_archive/§167_*.md`.

### §168 — Eval harness generalization + v6w25 plumbing (Phase D restructured) — 2026-05-08
Branch `encoding/eval_generalization`. **Eval harness generalized along (encoding × inference method) axes** — single invocation handles any `(checkpoint, method)` tuple.

**Architectural decisions locked:**
- `checkpoint_loader.load_model_with_encoding(path, device)` auto-detects encoding from state-dict (in_channels=11 → v8; =8 + filename "v6w25" → v6w25; =8 default → v6); returns `(model, EncodingSpec, label)`.
- `build_inference_method(name, model, device, label)` dispatches V6/V8 Argmax/MCTS/Fast bots per `(encoding, method)` tuple.
- `V8MCTSBot` — sequential PUCT in Python; Rust `MCTSTree` is v6-locked.
- **v6w25 = runtime parameterization** (NOT cargo features per §166 contract §4.3): `Board.set_cluster_threshold(8)` + `set_cluster_window_size(25)`. v6 and v6w25 cluster encoders coexist in one binary.
- `scripts/run_sealbot_eval.py` (renamed from `eval_v8_vs_sealbot.py`).

**v6w25 artifacts (5080):**
- Corpus: 319,207 positions, 3.8 GB, sha256 `85c045934c90…`.
- Bootstrap: 21 MB, sha256 `571a82f844fc…`, pretrain 1h 33m.

**v6w25 sanity SealBot argmax @ r=8 n=200: 14.5% [10.3%, 20.0%]** (29/200, 2 draws, mean ply 51.5). v7full @ r=8 baseline = 12.5% [8.6%, 17.8%] (§167). CIs overlap — cluster-threshold widening (5→8) does NOT materially help argmax-only WR.

**Commits**: `3f2bf10` (eval harness), `ed440a3` (Rust v6w25 constants), `0c62138` (pretrain + HexTacToeNet wiring). 1085 py + 151 Rust unit + 6 Rust integration PASS. T3 matched-MCTS comparison deferred to §169. Forensics: `reports/sprint_archive/§168_eval-harness-v6w25-plumbing.md`.

### §169 — Four-way encoder ablation (A1/A2/A3/A4 + §169a probe) [MERGE] — 2026-05-08
Branch `encoding/four_way_ablation`. Closes the bbox-vs-K-cluster question.

**Verdict matrix:**

| arm | loss | argmax @ r=8 (n=200) | MCTS-128 (n=200) | params (M) | verdict |
|---|---|---|---|---|---|
| **A1 K-cluster + min/max (v6w25 anchor)** | 3.57 | **14.5%** | 25% (§169 P1 sanity MCTS-32 n=20) | 5.29 | **CANONICAL** |
| A2 K-cluster + PMA pool | 4.25 | 4.5% | 3.5% | 6.30 | NEGATIVE — K=1-pretrain regime; SAB sees single token per batch |
| A3 K-cluster + PMA + global token | 3.62 | 7.5% | 2.5% | 6.37 | PARTIAL POSITIVE — global gate climbed 0.10 → 0.66 (6.6×); closes 95% of loss gap, halves argmax gap, no MCTS lift |
| A4 bbox + canvas_realness + PartialConv2d | **3.47** | **0.0%** | **0.0%** | 3.85 | NEGATIVE — bbox direction STRUCTURAL, not padding semantics |

**Key mechanism (L4 origin):** **The encoding decides; the pool variant tweaks.** Training loss alone NOT sufficient signal — A4 has lowest loss but zero WR; A2 has highest loss but still beats A4 at argmax. Three candidate mechanisms for bbox failure: K-aggregation as cross-cluster contrast (K=1-pretrain forfeits multi-window inference); bbox-centroid frame instability (centroid moves up to m=8 cells per move); R=8 perception expansion (8× more action geometry per state). Per-cluster bbox at CLUSTER_THRESHOLD=8 falling back to unified bbox specced in `audit/encoding_spikes/s1_bbox_algorithm.md` §5.2; operator's call for §170 follow-up.

**§169a A4 spatial-pathway-deadness probe.** Pre-registered: E1 dead `mean(KL_S) < 0.10 AND KL_S/KL_R < 0.05`; E2 alive `mean(KL_S) > 1.00 OR KL_S/KL_R > 0.30`. **Results**: KL_S = 1.533 > 1.0 → **E2 PASS spatial path alive**. argmax visits 133/200 distinct cells. Ratio KL_S/KL_R = 0.273 → scalar-dominated but not scalar-only. PartialConv2d + canvas_realness propagate spatial signal correctly through trunk + KataGo policy head. User's "spatial-dead" hypothesis FALSIFIED. Surfaces alternatives: distribution-shift fine-tune (§171 A4 P2-reopen), cross-encoding eval gap audit, scalar-ablation follow-up (§170 P0).

**PMA-collapse smoke** flagged on A3 fixture as probe artifact (synthetic 2-cluster fixture trained model has strong absolute-position preference; gate=0.66 + non-zero argmax delta on real games prove healthy use). Threat probes all SKIPPED (v6w25 fixture gap; §170 follow-up).

Commits: pool registry + wire `pool_type='pma'` + retrain config + A2 eval; A3 global encoder + crop helper + pretrain + eval; A4 canvas_realness + PartialConv2d + retrain + eval; §169a probe. Forensics: `reports/ablation_169/{A1-A4}_*.json`, `reports/sprint_archive/§169-§170_four-way-ablation-wave.md`.

### §170 — gpool-bias retrain wave + Gate 6 operator surface [MERGE] — 2026-05-08 → 2026-05-09
Six sub-passes (P0–P4 P2). Closes the v6w25 inference-side question.

**Verdict matrix:**

| Pass | Arm | Result | Verdict |
|---|---|---|---|
| P0 | A4 scalar-ablation probe | SPATIAL_RICH (KL_zeroed=4.19 ≫ 1.50) | NULL — falsifies SCALAR_DOMINATED; argmax stable 0/200 |
| P1 | A3 MCTS-N curve (PMA-value-semantics) | flat MCTS-32/64/128 = 2.5% | NULL — Cochran-Armitage p=0.0277 entirely from argmax vs any-MCTS split |
| P3 | A1 + gpool-bias on both heads | NULL on value head | NULL |
| **P4 P1** | **A1 + gpool-bias-policy-only retrain** | **22% argmax SealBot @ r=8 n=200** | **CANONICAL** |
| P4 P2 | Adversarial corpus prep for §171 A4 fine-tune | corpus assembled, fine-tune handoff | landed |

**Key mechanism:** **gpool-bias-policy-only (P4 P1)** is the load-bearing intervention. Full gpool-bias on both heads (P3) is NULL on value. Earlier attribution of 22% to A4 fine-tune was wrong — A4 was already 0%; the lift is from policy-side gpool-bias applied on A1 K-cluster.

**Canonical pins:**
- **v6w25 = canonical for pretrain + eval + matched-MCTS** (P4 P1 mechanism).
- **v7full = canonical for self-play** pending α (§173) implementation.

**Gate 6 operator surface:** `scripts/run_gate6_eval.py` — A1+gpool-bias matched-MCTS at n=200, 5080. CLI: `--bootstrap`, `--encoding`, `--inference`, `--n-games`.

Forensics: `reports/gpool_bias/`, `reports/sprint_archive/§170_gpool-bias-retrain-wave.md`. Branch `encoding/four_way_ablation` retired after canonical pin. Memory: `project_170_p3_falsified.md`.

### §171 — Sprint scoping + blocked handoff + §171 A4 P2-reopen (DEAD) — 2026-05-09 → 2026-05-11
Branch `phase4/p171_selfplay_smoke`. P0 corpus-sha manifest transposition corrected (`bootstrap_corpus_v8.npz` is vanilla v8 sha `adb88412…`; canvas_realness variant is `bootstrap_corpus_v8_canvas_realness.npz` sha `110ea6b2…`). **P3 BLOCKED** by two-layer issue: Layer 1 canvas vs trunk `board_size` semantics inconsistency (A2.1 trainer trunk=25 vs A2.2 pool guard canvas=19); **Layer 2 load-bearing**: `Board::to_planes()` hardcoded 18×19×19 regardless of `cluster_window_size` — selfplay's `state.to_tensor()` calls single-window path; v6w25 pretrain worked because `dataset_v6w25.py` uses `get_cluster_views()`. v6w25 sustained selfplay requires **α multi-window** (§173). Recommendation: Option α structural fix in §172.

**Pin scope (§171 P3.6):** v6w25 canonical for pretrain+eval+matched-MCTS; v7full canonical for selfplay pending α; α = Phase 4.5+ scope.

**§171 A4 P2-reopen C (DEAD).** 2026-05-11. Distribution-shift fine-tune side-arm on `phase4/encoding_registry`. Pre-registered: E1 ALIVE `MCTS-64 > 8% AND CI_LB > 5%`; E2 DEAD `MCTS-64 ≤ 2% AND CI_UB < 4%`. Recipe: resume `A4_canvas_realness.pt`, mixed corpus 95% bootstrap + 5% adversarial via `WeightedRandomSampler`, 3000 steps, peak LR 5e-5/eta_min 5e-6, freeze input_conv + tower[0..7], trainable tower[8..11] + heads (35.2%). **Result: MCTS-64 0/200 Wilson95 [0.000%, 1.88%] — DEAD cleanly met.** §169 P0 SPATIAL_RICH framing FALSIFIED for frozen-spine class. Closes bbox+canvas_realness+frozen-spine line. **Correction**: original close-out claimed "22% → 0% argmax collapse"; the 22% was misattributed (§170 P3 A1+gpool-bias, not A4 baseline). A4 was already 0% pre-fine-tune (`reports/ablation_169/A4_eval.json`); fine-tune did not damage anything new. DEAD verdict rests strictly on MCTS-64 axis. Forensics: `reports/sprint_171_a4/`, `reports/sprint_archive/§171_sprint-scoping-blocked-handoff.md`. Memory: `project_171_a4_p2_reopen_c_dead.md`, `project_bootstrap_argmax_drift_check_20260511.md`.

### §172 — Encoding Registry Single-Source-of-Truth + Phase B v7full sustained — 2026-05-09 → 2026-05-11
Branch `phase4/encoding_registry`. **Trigger**: §171 P3 plane-export blocker — `Board::to_planes` hardcoded `BOARD_SIZE=19` even when `Board::with_encoding(v6w25_spec)` set `cluster_window_size=25` → silent shape corruption. Root cause: scattered encoding state across 23 surfaces.

**Phase A (A1–A10):** TOML at `engine/src/encoding/registry.toml` as canonical single source. Lazy `&'static` lookup. `RegistrySpec` Rust + `hexo_rl.encoding` Python module. New encoding adds 1 TOML entry; `python -m hexo_rl.encoding audit` verifies parity. Variant configs use `encoding: <name>` only; scattered overrides rejected at load. Threaded through Rust hot path / Python selfplay / training+scripts / model+eval / configs.

**A10 close-out (13 commits f687602..576f69d):** §10 amended consistency-not-equality; §11.6 cross-table consistency (6 INVs joined on corpus_sha256); 3 **HIGH-RISK silent-corruption hazards** retired:
1. `engine/src/game_runner/mod.rs:159` `SelfPlayRunner::new` pyo3 default kwarg silent v6 fallback (feature_len=8*19*19, policy_len=19*19+1) → now derives from `spec.state_stride()`/`spec.policy_stride()`.
2. `engine/src/inference_bridge.rs:295` `InferenceBatcher::new` same pattern; `encoding_spec` kwarg added.
3. `engine/src/replay_buffer/sym_tables.rs:26` `N_ACTIONS=362` audit confirmed all consumers v6-only; v8 pinned to `spec.policy_stride()=625`.

A6 round-trip test (5 encodings PASS); A7 α design doc (`docs/designs/encoding_alpha_multiwindow_selfplay_design.md`); A8 docs cleanup (README + CLAUDE.md + docs tree); A9 three parallel review subagents PASS. 1306 py + 223 rs pass.

**Phase B (v7full sustained, B1-redo + B2):** B1 first launch surfaced G1 (`bootstrap_anchor` strict-load failure on `tower.*` ↔ `trunk.tower.*` aliases) + G2 (92.3% draw rate inherited cosine). B1-redo `sprint_172_p3_v7full_r12.yaml` + `cf73390`: R12 cosine disable + `_load_anchor_model` migrated to `load_model_with_encoding` delegate. 1200-step smoke PASS — draw_rate 0.923 → 0.040 (23×), colony_extension_fraction = 0.0, bootstrap_anchor LOADED.

**B2 30K sustained on 5080** (commit `e90e49d` mid-run wired `argmax_n` DRIFT detector + `eval_interval` 1000 → 5000; `--iterations` is LR-schedule denominator NOT step-stop — ran 3024 over). Only step-20K promoted (best_arena 0.610 CI_LB 0.512). **sealbot STALLED at 0.05 four consecutive rounds (15K/20K/25K/30K)** vs §150 anchor 17.4% n=500 — 12.4pp short on point estimate. bootstrap_anchor oscillating 0.50-0.65; best_arena post-promotion parity 0.55-0.56; argmax_n DRIFT detector inert (0/20 across all 6 rounds); colony_fraction 0.0 throughout.

**B verdict — no v7full graduation.** Self-play distribution overfit: single-window v7full selfplay generates positions model learns to play well vs self but doesn't exercise threat structures SealBot exploits at r=8. **Encoder-specific gap, not value-drift pathology** (argmax_n DRIFT 0/20). Transfer gap is structural at the encoder level, not head-recalibration. **Forward: §173 α multi-window K-cluster selfplay.**

**Gate 6 decision packet (operator):** (a) Do NOT promote v7full P3 ckpt (save step-20K as B3 anchor candidate); (b) YES open §173 α implementation; (c) δ A4 DEAD verdict closes bbox+canvas_realness+frozen-spine; (d/e) merge `phase4/encoding_registry` → master directly with `sprint-172-close` tag; (f) no new architectural reopens.

Forensics: `reports/sprint_172_summary.md`, `reports/sprint_archive/§172_encoding-registry-sprint.md`. Memory: `project_172_a{1..10}_complete.md`, `project_172_b{1,1_redo,2}_complete.md`. Encoding registry contract: `docs/designs/encoding_registry_design.md`.

### §173 — α multi-window K-cluster selfplay implementation — 2026-05-11
Branch `phase4.5/m173_alpha_multiwindow`. **Scope corrected** from §172 A7 design (buffer+trainer parameterization) to **constants-parameterization only** — no changes to worker_loop architecture (L319-411 dispatch, L662-694 K-fanned push). Retire every v6 hardcoded literal in replay-buffer + game-runner hot paths, replacing with `RegistrySpec`-derived reads.

**Commits (10, A3 → A8'' + A7 merge):**
- `68934a5` A3: `kept_plane_indices` + `n_source_planes` TOML; Python `n_cells` parity bug fix (was `board_size²`, now `trunk_size²`).
- `38b0544` A4: spec-wire strides + `sym_tables_for()` + H5 pass-slot guard.
- `5928f9d` A5a: spec-wire strides in worker_loop + mod.rs.
- `3a11d71` A5b: `aggregate_policy*` spec-threading (initial -10.47% worker_pos_per_hr regression from RegistrySpec by value — see L16; recovered to +6.01% via scalar-API + `#[inline]`).
- `7f43fdc` A6: Python triple-setter migration (`with_encoding_name`); setter guards raise when encoding is set.
- `00a25f2` A7: audit CLI regex tighten + allowlist gap closure.
- `8fd28e5` A8: cold-smoke variant + v6w25 baseline JSON + microsmoke tests.
- `2af7d99` A8': lift α multi-window guard in `pool.py` + trunk_size geometry refactor.
- `aedbb2a` A8'': spec-thread aux + `window_flat_idx` via `spec.n_cells()`.

**HAZARD closure** (H1-α through H6-α all CLOSED): sym_tables shape mismatch, rotate_aux ownership corruption, view truncation, aggregate_policy 626 vs 362 vector, sample.rs:220 pass-slot OOB, mod.rs:342 STATE_STRIDE v6. **H7-α (HEXB v7 encoding header field)** CARRIED to §174.

**Bench gate (n=5, post-A5b v2 vs pre-α):** all 10 targets PASS. MCTS −0.4%, NN inference −0.9%, NN latency −0.9%, buffer_push +3.1%, buffer_aug +1.2%, GPU util flat, worker_pos_per_hr +28.9% (within variance — methodology shift: pre-α 3s warmup vs post-A5b 90s).

**Phase B verdict:** A8 cold smoke (5K-step v6w25 sustained) DEFERRED — operator did not authorize 5080 launch before A10 close-out. Equivalent empirical signal: A8' microsmoke PASS (shapes 8×25×25/6×25×25/626, zero NaN/Inf); A8'' window_flat_idx green; post-A5b bench all PASS.

**G3/G4/G5 gating status:** G3 monotonic depth scaling UNVERIFIED; G4 value-head |max| ±50% band UNVERIFIED (baseline JSON `v6w25_baseline_value_max.json`: 0.308, band [0.154, 0.462]); G5 per-cluster variance drift ≤30% UNVERIFIED. §174 sustained must verify before any longer run.

**A9 review:** 3 parallel subagents PASS sprint-level. 4 SOFT-FAIL F1-F4 tagged for §174 (audit CLI strict-mode legacy literals; v8 `to_planes()` semantic mismatch pre-existing; G4/G5 empirically unverified pending smoke; MCTS architecture preserved as designed). No HARD-FAIL.

§174 prerequisites (BLOCKERS): (1) HEXB v7 format bump (encoding-name header field; H7-α carried); (2) A8 cold smoke G3/G4/G5 verification; (3) `n_source_planes` producer-side migration (deferred cleanup, non-blocking).

Locked §174 parameters (from P0): eval n=100 (sealbot/bootstrap_anchor/best_arena), eval_interval=5000, train:selfplay 2:1, buffer growth 500K @ step 250K, sustained bootstrap `bootstrap_model_v6w25.pt`, arena anchor static.

Forensics: `reports/sprint_173/`, `docs/designs/encoding_alpha_multiwindow_selfplay_design.md`, `reports/sprint_archive/§173_alpha-multiwindow-kcluster-selfplay.md`. Memory: `project_173_a3_a6_bundle_complete.md`, `feedback_registryspec_by_ref_in_hotpath.md`.

### §174 — v6w25 sustained: bootstrap investigation + escalation — CLOSED 2026-05-13
Branch `phase4.5/m176a_v7mw`. Three bootstrap recipes tested; none clears selfplay viability gate at R=8 MCTS-128.

| Bootstrap | SealBot MCTS-128 | Selfplay median plies | G4 |
|---|---|---|---|
| 30-epoch (canonical anchor) | **0% (0/200)** | 6 | PASS within band |
| e50 | 10% (10/100, artifact-suspect) | 6 | MARGINAL FAIL 0.489 |
| v6→v6w25 transfer FT | **0% (0/200)** | 8 | n/a |

**Timeline:**
1. First 30K run — FP16 GradScaler overflow on 25×25 geometry; LR 2e-3 → 1e-3 fix per §173 A8 v3 ablation.
2. Second 30K run — eval crash mid-pipeline (orthogonal; deferred).
3. e50 retrain (50 ep vs 30) hypothesis: more capacity-utilisation tightens heads → REGRESSED. Value head grew (0.489 vs band [0.154, 0.462]) — over-fits to corpus-mode signal selfplay cannot reproduce.
4. **Radius ablation FALSIFIED.** R∈{1..8} smokes off e30 bootstrap, otherwise identical. Median game length identical across all eight. Radius does NOT move bootstrap quality (smokes already R=8). `legal_move_radius_schedule` retained as downstream-training lever, not bootstrap-time hyperparameter.

**Root cause (`reports/s174_v6w25_investigation.md`):** Loss surface normal — v6w25 30-ep achieves 3.96 nats vs uniform (best of v6/v7full/v6w25); matches v7full value-loss trajectory. **Opening-fraction starvation H1 refuted:** ply ≤ 10 fraction 16.09% (v6w25) vs 17.15% (v6) — gap 1.06pp, not the multi-× predicted. Collapse is at **argmax-degeneracy / selfplay-interaction layer**, not corpus or loss. Transfer recipe inherited v6 trunk but lost opening knowledge in re-initialised policy FC.

**eval_random_opening_plies audit (Track 2):** §168 → §174 sealbot WR drop (14.5% → 0%) fully explained by `eval_random_opening_plies` 4 → 0 in `configs/eval.yaml:88`. With 4 random plies model got free positional diversity masking weaknesses; with 0 SealBot's preparation lands cleanly. No code fix — flip already in place.

**G-gates wired:** G3 `avg_game_length` in `iteration_complete` + per-game `game_length`. **G4 NEW THIS SPRINT** — `_g4_value_head_band_check` in every `run_evaluation` round; structlog WARNING on out-of-band; constants gate-internal. G5 `cluster_value_std_mean` + `cluster_policy_disagreement_mean` emitted in `iteration_complete` + `train_step_summary`.

**Infrastructure landed (Track 2):**
- Encoding auto-detect across `make pretrain` / `make eval` / `make selfplay.smoke` / `make transfer` — checkpoint metadata authoritative; CLI flag overrides on ambiguity.
- v6 → v6w25 transfer script (`scripts/transfer_v6_to_v6w25.py`).
- Makefile encoding-override knobs: `PRETRAIN_ENCODING`, `EVAL_ENCODING`, `SMOKE_ENCODING`, `TRANSFER_SOURCE`, `TRANSFER_OUTPUT`.
- vast.yaml audited clean: LR 1e-3, eval_interval 10000 (halved wall-time vs §173 P0; preserves stride math), random_opening_plies 0 on both paths.

**Forward: §175 v6 sustained.** Recipe: 100K steps, n=100 SealBot eval, matched cosine LR schedule inherited from §174 vast.yaml. Selfplay encoding v6 (single-window 19×19, existing path). v6w25 retained as future re-entry target once selfplay-friendly bootstrap recipe is found — current evidence says fix is at policy/value head + selfplay-interaction layer, NOT corpus layer.

**Retained baselines:**

| Name | Encoding | Use |
|---|---|---|
| `bootstrap_model.pt` (v6) | v6 | **§175 anchor** |
| `bootstrap_model_v7full.pt` | v7full | §150 anchor (17.4% n=500) |
| `bootstrap_model_v6w25.pt` (e30) | v6w25 | retained for analysis |
| `bootstrap_model_v6w25_e50.pt` | v6w25 | G4 marginal fail; dominated |
| `bootstrap_model_v6w25_transfer_ft.pt` | v6w25 | 0% MCTS-128; retained |

Forensics: `reports/s174_v6w25_investigation.md`, `reports/s174_bootstrap_investigation.md`, `reports/s174_bootstrap_fix.md`, `reports/s174_r8_falsification.md`, `reports/sprint_archive/§174_v6w25-sustained-bootstrap-investigation.md`.

---

## §176 — Python codebase refactor cycle (2026-05-13 → 2026-05-14)

**Scope:** 80-proposal audit + 6-phase execute on `refactor/python-audit` branch.

**Phases:**
- Phase 0 — master plan fixup (`c4eaa53` CLAUDE.md scattered-keys clarification) + drift annotations (`838b5ed` open-questions log)
- Phase 1a/b — HEAD-blocker fixes (B2 `sweep_harness` restore, B1 `HexTacToeNet` encoding whitelist→registry)
- Phase 2 — invariant pre-flight tests (12 INV pins under `tests/refactor_invariants/`) + 4 low-risk additions
- Phase 3a — W1 deletions (-274 LOC net)
- Phase 3b — W1 extracts + SSR fixes + small renames (24 commits)
- Phase 4 — W2 splits + extracts (25 commits; 3 bench-gated items)
- Phase 5 — W3 cross-bucket consolidation (15 commits; 4 bench-gated items)

**Outcomes:**
- 86 commits landed `c4eaa53..HEAD` (post-rebase HEAD `7233d5d`); 171 files changed, +12102/-7446 (net +4656; dominated by P39 6-module pretrain split, P70 train.py orchestrator decomposition, and INV/fixture test scaffolding)
- 75 of 80 proposals landed (5 NEEDS-WORK resolved at Phase 0; 3 deferred to W3 sub-items)
- Cross-bucket SSR debt cleared: `utils/encoding.py` + `utils/constants.py` v6/v8 entries retired; ~37 callers migrated to §172 registry; `hexo_rl/bootstrap/bots/` retired (P78a–d) — all three SSR grep targets return 0 post-merge
- 12 behavior invariants pinned as regression tests under `tests/refactor_invariants/` (all green post-merge)
- HEAD-blocker B1 (v7-family pretrain crash via whitelist) + B2 (sweep_harness broken imports after §163 deletion) fixed
- Test count: 1518 → 1574 (+56 from new fixtures + INV pins). Single pre-existing failure `test_no_stale_plane_refs` baselined and unchanged
- Bench: all hot-path edits verified within ±5% on 10-metric gate (P3, P4, P8, P22, P24); cold-path skips documented per `docs/refactor-template.md`
- `make test.py` post-merge: 1574 passed / 1 failed (pre-existing) / 17 skipped / 4 deselected / 1 xpassed
- §175 selfplay state unaffected throughout — refactor branch strictly isolated; rebase onto `phase4.5/m176a_v7mw` HEAD `838b5ed` was conflict-free (no file overlap with in-flight commits)

**Deferred to future micro-refactor cycle** (tracked as `Q-§176-residual` in `06_OPEN_QUESTIONS.md`):
- P24b/c: `HexTacToeNet.__init__` (262 LOC), `forward` (162 LOC), `aggregated_forward_K` (113 LOC) further decomposition — partial landed in Phase 5
- P70: `scripts.train::seed_everything` circular-import shim lifted inside orchestrator helper — clean candidate

Forensics: `reports/refactor_audit/00_MASTER_PLAN.md`, `reports/refactor_audit/p6_phase4.5_inflight_commits.txt`, Phase 5 reviewer verdict (in conversation history of `phase4.5/m176a_v7mw` § auditor session).

---

## §176 — Phase A — KrakenBot eval ladder validation + colony POC (2026-05-14)

Branch `phase4.5/s176_phase_a_validation`. Five-wave empirical investigation (A1–A4 parallel + B + C + D + E fresh-context review) closing **Q14 partial**, opening §176 Phase B implementation scope. §175 v6 sustained continues on vast; aborts at next eval boundary post-merge.

**Waves:**
- **A1** (`reports/s176_a1_kraken_smoke.md`) — submodule pinned `d9c5bfb`; verdict `INTEGRABLE_NOW` for MinimaxBot+RandomBot, `NEEDS_WEIGHTS_DOWNLOAD` for both MCTSBot variants (`vendor/bots/krakenbot/.gitignore:8`, no public mirror). MinimaxBot latency 222–232 ms @ time_limit∈{0.1,…,2.0}.
- **A2** (`reports/s176_a2_eval_arch.md`) — verdict `CACHING_CLEAN`. Evaluator loop stone-by-stone (evaluator.py:201-210); `_pending_move` cache already proven on SealBot+KrakenBot. Minimal-diff plan ~150-180 LOC, 0 INV pins fire.
- **A3** (`reports/s176_a3_selfplay_forensics.md`) — operator's "one large diffuse cluster" claim **REFUTED** by 21,371 §175 game records (vast run `c7e74d2842404a82bdd9f62edf740ea2`). Single-cluster fraction monotone-down 18.1% (20K) → 6.3% (50K); attractor is multi-island fragmentation. Step-change at 40K (n_components 9.61→13.77, +43%). **POC metric = `n_components` raw BFS, Cohen's d −0.822** (largest among 8 candidates). In-trainer `colony_extension_fraction` flat zero — does NOT capture §175 attractor; justifies new POC.
- **A4** (`reports/s176_a4_falsified_scan.md`) — 9 falsified rows + 11 mechanism lessons (L1–L17 subset) + 4 regressions + 4 open Qs. 15-item do-not list each empirically sourced. Surfaced 7 master-prompt gaps (pool freeze, e30/e50 pretrain, radius+cosine pairing, v6 corpus blacklist, extended smoke boundary, frozen-spine rejection, realistic plumbing budget).
- **B** (`reports/s176_b_smoke.md`) — `KrakenBotRandomBot` + `KrakenBotMCTSBot` skeleton + `scripts/tournament_validate.py` + 3 wrapper tests (PASS). 15-game smoke verdict `PROCEED-TO-C`. Flagged: `bootstrap_model.pt` is v6w25; `our_v6_*` Wave C must pin `bootstrap_model_v6.pt`.
- **C** (`reports/s176_c_tourney/summary.md`) — 1050-game round-robin / 7 bots / 50 games/pair / laptop wall 85 min. Mid-tourney critical fix: KrakenBot MinimaxBot returns `[(0,0)]` sentinel when `_generate_turns` rejects all compounds (vendor lines 184/219/325/330); naive uniform-random fallback caused 0.42 sentinel/game in mid-game (ply ≥20). Smart neighbour-2 fallback (`_smart_legal_fallback` in `hexo_rl/bots/krakenbot_bot.py`) using KrakenBot's own `_D2_OFFSETS` pool prevented 438 uniform-random degradations across full run (fb_n2=438 / fb_rand=0).
- **D** (`reports/s176_d_plan.md`) — §176 Phase B implementation plan, 6 commits ≤10 cap, ~990 LOC delta, zero bench gates (all cold paths).
- **E** (`reports/s176_e_review.md`) — fresh-context audit verdict **CLEAR** across all 5 dimensions. 7-row risk register; 3 non-blocking strengthening notes for S6.

**BT ladder (anchor=sealbot, n=50/pair):**

| Bot | Elo | CI lo | CI hi | Wins/300 | Colony>0.3 rate |
|---|---:|---:|---:|---:|---:|
| sealbot | 0 | 0 | 0 | 274 (91.3%) | 35.0% [29.6, 40.9] |
| our_v6_mcts128 | −62 | −150 | +26 | 263 (87.7%) | 33.5% [28.0, 39.4] |
| kraken_minimax_strong | −494 | −612 | −376 | 182 (60.7%) | 7.1% [4.2, 11.8] |
| kraken_minimax_fast | −499 | −618 | −381 | 181 (60.3%) | 15.5% [10.9, 21.4] |
| kraken_random | −3072 | sat. | sat. | 7 (2.3%) | 85.7% [48.7, 97.4] (n=7) |
| randombot | −3091 | sat. | sat. | 0 | n/a |
| our_v6_argmax | −3102 | sat. | sat. | 0 | n/a |

**V1–V6 verdicts** (full text + numbers in `reports/s176_c_tourney/summary.md`):

| ID | Hypothesis | Verdict | Mechanism |
|---|---|---|---|
| V1 | strongest Kraken MCTS > SealBot | `N/A_MCTSBOT_BLOCKED`; modified-V1 FAIL | strongest tested Kraken (MinimaxBot @ 1.0s) is −494 Elo vs sealbot |
| V2 | Kraken MinimaxBot @ 1.0s > MCTSBot | `N/A_MCTSBOT_BLOCKED`; side-finding: MinimaxBot @ 0.1s ≈ @ 1.0s (Δ −5 Elo within CI) | iterative-deepening saturates at depth 4 + sentinel-fallback rate inflates draws |
| V3 | MinimaxBot colony ≤ MCTSBot colony − 10pp | `N/A_MCTSBOT_BLOCKED`; modified-V3 PASS | sealbot 35.0% vs kraken_minimax_strong 7.1%, gap 27.9pp, CIs non-overlapping by 18pp |
| V4 | SealBot colony > all Kraken | FAIL (kraken_random 85.7% > sealbot 35.0%) | caveat: kraken_random only 7 wins, CI wide |
| V5 | our_v6 strictly between Random and weakest Kraken | FAIL — our_v6_mcts128 is the 2nd-strongest bot in the tourney, not a weak baseline | our_v6 −62 BT vs kraken_random −3072 / randombot −3091 |
| V6 | cross-pair colony ≠ self-pair (opponent-coupled) | PASS — sealbot colony ranges 0.000 (vs argmax) → 0.412 (vs kraken_minimax_strong), 41pp spread | colony emergence is opponent-driven, not bot-intrinsic |

**D1–D5 master-prompt decision verdicts:**

| ID | Verdict | Source |
|---|---|---|
| D1 BotProtocol caching not `get_turn` | BACKED | A2 CACHING_CLEAN + Wave C 1050-game stability |
| D2 tourney includes all Kraken variants | PARTIAL — MCTSBot blocked, defer to §177+ until weights | A1 NEEDS_WEIGHTS_DOWNLOAD |
| D3 MinimaxBot colony < MCTSBot colony | PARTIAL — modified BACKED (vs SealBot 27.9pp); MCTSBot blocked | V3 PASS |
| D4 mix 75/15/10 | NEEDS_REVISION → adjusted bot pool: sealbot 50% / our_v6 30% / kraken_strong 15% / kraken_random 5% (Elo-derived per-source weights) | Wave C BT ladder |
| D5 Source A first, Source B target | BACKED | V6 opponent-coupled colony + A4 do-not #1 mandates subprocess for B |

**New mechanism lessons (L18+ candidates):**

- **L18** — *Pretrained-bootstrap-at-MCTS-128 can match an external minimax engine.* Wave C: our_v6 bootstrap (`bootstrap_model_v6.pt`, untrained on selfplay) is 25/50 H2H vs SealBot @ 0.5s, BT delta −62 (95% CI [−150, +26], WR Wilson95 [36.3%, 63.7%], n=50 — point-estimate-parity-consistent, not strongly-asserted parity). Implication: a "weak external opponent" framing for §175-style transfer-gap diagnoses is wrong at matched MCTS perception — the degradation is internal self-play head drift, not opponent-strength regression. **Refined by L22 below:** "head drift" is sampled-policy distribution flattening into colony attractor under T=0.5, not argmax-mode regression past bootstrap.
- **L19** — *KrakenBot MinimaxBot `_generate_turns` sentinel `[(0,0)]` is the upstream strength floor, not the time_limit.* time_limit ∈ {0.1, 1.0} produces BT delta −5 Elo. The 0.42 sentinel/game rate in mid-board positions (ply ≥20) caps strength independently of search budget. Any bot wrapper consuming a third-party bot MUST validate the returned cell against the live engine board, not trust the bot's output blindly.
- **L20** — *Argmax-only proxies (MCTS-1) are structurally below RandomBot.* `our_v6_argmax` (n_sims=1, temperature=0) went 0/300 — below randombot's 0/300-with-99-draws. The argmax-handicap memory (`feedback_v6_v8_same_training_data.md`) generalises beyond cross-encoding diagnostics: argmax-only mode is a degenerate strength sensor, only useful as a relative-direction signal, never as an absolute baseline.

**New Falsified Hypotheses Register row candidates:**

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §176 Phase A | A3 — §175 selfplay terminal states are "one large diffuse cluster" | A3 §c-§d | Single-cluster fraction monotone-down 18.1%→6.3% across 20K–50K cohorts; modal pattern is multi-island fragmentation |
| §176 Phase A | V2 — KrakenBot MinimaxBot @ 1.0s > @ 0.1s | Wave C V2 | BT delta −5 Elo, head-to-head 20-30 favouring 0.1s; iterative deepening saturates at depth 4 in our off-distribution game |
| §176 Phase A | V5 — our v6 bootstrap @ MCTS-128 strictly between RandomBot and weakest Kraken | Wave C V5 | bootstrap MCTS-128 is the 2nd-strongest bot in the tourney (BT −62 vs sealbot); ~3030 Elo above RandomBot |

**Forward pointer:** §176 Phase B implementation (S1–S6 per `reports/s176_d_plan.md` §2) opens on a fresh branch (TBD). Recommended 6 commits ≤10 cap. Mix-ratio bot-pool weights (sealbot 50 / our_v6 30 / kraken_strong 15 / kraken_random 5) per S4 design doc. Source B (live cross-bot) is design-only this sprint; subprocess isolation mandatory per A4 do-not #1.

Forensics: `reports/s176_{a1,a2,a3,a4}_*.md`, `reports/s176_b_smoke{.md,/}`, `reports/s176_c_tourney/{summary.md,verdicts.txt,ratings.csv,h2h_matrix.csv,colony_table.csv,per_game.jsonl}`, `reports/s176_d_plan.md`, `reports/s176_e_review.md`. Memory: `project_176_phase_a_close.md` (to be written).

---

## §176 — Phase A Gate 1 + Gate 2 + Gate 3 close-out (2026-05-15)

Branch `phase4.5/s176_phase_a_validation`. Three-gate operator-mandated cycle: (1) fresh-context independent review of Phase A artifacts (Wave E was the implementer-adjacent fresh-context audit; Gate 1 is a second pass per L13), (2) §175 interrupt + 70K vast tourney for tide-vs-recover empirical answer, (3) step-20K checkpoint promotion to weights-only bootstrap artifact.

### Gate 1 — operator review, verdict **STRENGTHEN_ONLY**

`reports/s176_gate1_operator_review.md`. 7 dimensions audited: D1–D5 cites (12 claims PASS), Wave C BT-ladder reproducibility (3 H2H pairs reproduced from `per_game.jsonl` — sealbot vs our_v6_mcts128 25/25, sealbot vs kraken_strong 49/1, sealbot vs randombot 50/0), `_smart_legal_fallback` correctness (`hexo_rl/bots/krakenbot_bot.py:34` imports `_D2_OFFSETS` from vendor — no divergent reimplementation), Falsified Register hygiene (16 rows walked; §17 GIL daemon row explicitly cited in plan S5 mandate), L18 sufficiency (PASS with strengthening note — n=50 H2H, BT CI [-150, +26] crosses zero; lesson body should disclose CI), risk register (7 rows ≥ 5 floor; 5 non-blocking strengthening notes captured for Phase B prep).

One process-note dimension: V1–V6 pre-registration is not committed as a separate artifact (`reports/s176_c_tourney/verdicts.txt` is post-hoc results dump). Honest FAIL declarations on V1, V4, V5 are weak evidence of integrity; operator vouches pre-registration. Gate 2 verdicts (V70K-1..5) committed BEFORE tourney in `reports/s176_gate2_verdicts.txt` — establishes audit-trail pattern for Phase B.

### Gate 2 — §175 interrupt + 70K tourney on vast 5080, verdict **MIXED**

`reports/s176_gate2_tourney/{summary.md,verdicts_v70k.md,ratings.csv,h2h_matrix.csv,per_game.jsonl}` + `reports/s175_forensics/`.

§175 interrupt was a no-op: session ended cleanly at step 70176 at 2026-05-14T20:56Z (SIGINT during sealbot eval game 37/100; `shutdown_save=True` triggered, buffer persisted, final checkpoint flushed). tmux session `s175` detached, kept for state preservation. Forensics archived locally: 21 eval-DB rows, checkpoint_log.json, training-step tail (5000 lines), shutdown events.

Tourney: 5 bots × 50 games/pair × 10 pairs = 500 games, 6551.1 s wall (well under 4-hr cap). Participants: `our_v6_latest` (step 70000, MCTS-128), `our_v6_step20k` (step 20000, MCTS-128), `our_v6_bootstrap` (bootstrap_model_v6.pt, MCTS-128), `sealbot` (think_time=0.5s), `kraken_minimax_strong` (time_limit=1.0s).

**Pre-registered verdicts (V70K-1..5):**

| ID | Hypothesis | Observed | Verdict |
|---|---|---|---|
| V70K-1 | 70K vs SealBot WR ≥ 17.4% | 25/50 = 50.0%, Wilson95 [36.6, 63.4] | **PASS (greedy mode)** |
| V70K-2 | 20K stronger than 70K H2H | step20k vs latest 50/0 = 100% | **PASS strong** |
| V70K-3 | 70K improved over own bootstrap | latest vs bootstrap 50/0 = 100% | **PASS strong** |
| V70K-4 | 70K col-frac (winner-side) ≤ 65% | 100% col>0.3 rate, mean col-frac 63.3%, n_components 14.90 | **FAIL strong** |
| V70K-5 | 70K vs Kraken strong WR ≥ 60% | 49/50 = 98.0%, Wilson95 [89.5, 99.6] | **PASS strong** |

**Critical methodology divergence.** `tournament_validate.py` runs `OurModelBot` at `temperature=0.0` (greedy argmax). §175 `eval_pipeline.py` defaults to `eval_temperature=0.5` (stochastic). Two distinct play modes from the same weights produce radically different rankings. §175 trajectory was 18.0% → 4.0% (T=0.5, n=100) across 20K → 70K. This tourney shows 0.0% → 50.0% (T=0.0, n=50) across the same checkpoints. The H2H 50-0 results between checkpoints reflect 2-effective-unique-games × 25 P1/P2 repetitions inflated to Wilson95 [92.9%, 100%] — over-confident at greedy-mode determinism.

**Interpretation:**
1. V70K-2 PASS strong is robust across both modes (step20k > latest in greedy 50-0; step20k > latest in sampled 18% > 4%). **20K-as-bootstrap decision validated.** Both methodologies agree.
2. V70K-3 PASS strong contradicts L18's "regressed past own bootstrap" framing for greedy mode but is silent on sampled mode (no eval-pipeline measurement of latest_70K vs bootstrap). **L18 needs refinement** (see L21 below).
3. V70K-4 FAIL strong (100% col>0.3 rate for latest_70K wins, n_components 14.90) is the most important finding — **attractor fully captured the latest_70K argmax distribution**, sitting right at the §176 Phase B warning threshold (`n_components ≥ 15`). Argues for aggressive bot-game mixing in Source A; Section 3 weights in `s176_d_plan.md` remain sound.
4. V70K-1 PASS is mode-qualified. Greedy parity with sealbot at step 70K is real (25/25) but DOES NOT contradict the §175 sampled-eval 4% slide.

### Gate 3 — step-20K promotion, verdict **PROMOTED with n=20 caveat**

`reports/s176_gate3/smoke_eval.md` + `checkpoints/bootstrap_model_v6_step20k.{pt,json}`.

- Source: §175 `checkpoint_00020000.pt` (run `c7e74d2842404a82bdd9f62edf740ea2`), source SHA256 `540ac1cf91be38c21b8c10267d36828f34aec242d89c51bc4fd0ea6f2a8680ca`.
- Artifact: `checkpoints/bootstrap_model_v6_step20k.pt`, SHA256 `297e0ce0e48c8c9c417d923610de6ed0166c7295789ebc7bd029c90bb42bce6a`, 17.0 MB (weights-only per §34: only `model_state` + `metadata` retained; optimizer/scaler/scheduler stripped).
- Sidecar: `checkpoints/bootstrap_model_v6_step20k.json`.
- Extraction verification: tensor-equality 143/143 keys MATCH source; round-trip `load_state_dict(strict=True)` returns `<All keys matched successfully>`; `python -m hexo_rl.encoding audit` reports `v6 v6 OK` (declared==inferred).
- Smoke eval: matched §175 methodology (`eval_temperature=0.5`, per-game seed) → 1/20 = 5.0% Wilson95 [0.9%, 23.6%]; binomial P(X≤1 | n=20, p=0.18) = 0.10 — consistent with §175 18/100 anchor at α=0.10; n=20 noise dominates point estimate. Master-prompt STOP boundary (< 5%) not strictly triggered. **Promotion approved.**
- Vast parity: artifact + sidecar pushed; SHA matches both hosts.
- `bootstrap_model.pt` symlink UNTOUCHED (§175-era reproducibility preserved).

### Retained baselines — new row

| Anchor | Path | SHA256 | Source step / run | Eval WR (n=100) | Note |
|---|---|---|---|---|---|
| v6_step20k (§176 Gate 3) | `checkpoints/bootstrap_model_v6_step20k.pt` | `297e0ce0…2bce6a` | §175 step 20000 / `c7e74d…40ea2` | 18.0% [11.7, 26.7] vs SealBot | Empirical §175 sampled-eval peak; weights-only per §34 |

### New mechanism lessons (L21+)

- **L21** — *Eval temperature mode change can invert checkpoint rankings.* §175 step-20K vs SealBot at T=0.5 is 18.0% (peak across §175); same checkpoint at T=0.0 (greedy argmax) is 0.0%. §175 step-70K at T=0.5 is 4.0%; at T=0.0 it ties sealbot 25/25. The argmax-mode and sampled-mode are effectively **two different bots** from the same weights. Any cross-tooling comparison must declare temperature; defaulting to `eval_temperature=0.5` is the convention for §175-era continuity.
- **L22** — *L18 head-drift refinement.* §175 internal drift between 20K and 70K is **policy-distribution flattening into colony attractor** under T=0.5 sampling, NOT loss of argmax dominance over bootstrap. V70K-3 PASS strong (latest dominates bootstrap 50-0 in greedy) + V70K-4 FAIL strong (100% colony-spam wins for latest, n_components 14.90) + §175 eval trajectory (18% → 4% sampled) jointly pin the mechanism. L18 should read "*sampled-policy regression into colony attractor*", not "*regressed past bootstrap*".
- **L23** — *H2H 50-0 in greedy-argmax tourneys is 2 effective unique games × 25 P1/P2 repetitions.* `tournament_validate.py` with `temperature=0.0`, `dirichlet_enabled=False`, `random_opening_plies=0`, and sealbot's roughly-deterministic 0.5s response produces 2 distinct game trajectories per pair (one per opening side). Wilson95 intervals on the inflated n=50 are over-confident. For 20K-as-bootstrap-style discriminator runs, this is acceptable as a sign-test (direction); for absolute strength estimates, prefer eval_pipeline at T=0.5 to inject per-game variance.

### New Falsified Hypotheses Register rows

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §176 Phase A Gate 2 | V70K-4 — §175 step-70K winner-side col-frac ≤ 65% (attractor weakened by training past 50K) | V70K-4 strong FAIL (100% col>0.3 rate, n_components 14.90) | Attractor captured the policy. Greedy-mode wins are uniformly colony-spam patterns. |
| §176 Phase A Gate 2 | L18 strict reading — "§175 latest_70K regressed past its own bootstrap on the selfplay axis" | V70K-3 PASS strong (50-0 H2H latest dominates bootstrap in greedy argmax) | Drift is sampled-mode policy-distribution flattening, not argmax-mode regression. See L22. |
| §176 Phase A Gate 1 | Operator-prompted L18 framing requires no statistical disclaimer at n=50 | Gate 1 dim (vi) — BT 95% CI [-150, +26] crosses zero; H2H 25/25 is parity-consistent only at n=50, NOT strongly-asserted parity | L18 lesson body should disclose CI; framing already correct ("can match") |

### Phase B anchor decision

**Anchor for §176 Phase B sustained: `checkpoints/bootstrap_model_v6_step20k.pt`** (Gate 3 artifact). Validated by:
- V70K-2 PASS strong (20K dominates latest in greedy 50-0)
- §175 sampled-eval (18% vs 4% across 20K vs 70K)
- Both methodologies agree direction.

Phase B implementation opens on a fresh branch (S1–S6 per `reports/s176_d_plan.md`). Mix-ratio bot-pool weights per Section 3 unchanged. **Add eval-temperature pin to all sustained smoke prompts** (recommend T=0.5 for §175 continuity).

### Phase A close

PR #8 mergeable. §175 tmux detached but session preserved. Replay buffer (2.9 GB + 77 MB .recent.npz), 24 checkpoints from step 5000 to 70176, structlog jsonl (49 MB train + 51 MB events) all intact on vast under `c7e74d2842404a82bdd9f62edf740ea2`.

Forensics added: `reports/s176_gate1_operator_review.md`, `reports/s176_gate2_verdicts.txt`, `reports/s176_gate2_tourney/{summary.md,verdicts_v70k.md,…}`, `reports/s176_gate3/smoke_eval.md`, `reports/s175_forensics/{eval_db_rows.json,train_tail_5000.jsonl,checkpoint_log.json,…}`.

---

## §176 — Phase B pre-launch baseline (2026-05-15)

Pre-Phase-B fix wave landed on `phase4.5/s176_phase_a_validation`: six commits absorbing F01–F05 SHOULD-FIX from `reports/s176_review_findings.md` + F06 + F09 N1+N3+N4 cheap STRENGTHEN items. Phase B launch prompt committed as pre-registered done-when artifact per Gate 1 §145 Note 5.

**Forward pointer:** §176 Phase B prompt artifact at `reports/s176_phase_b_prompt.md` (pinned at commit `3994459`). A future Opus session reads that file top-to-bottom and executes Phase B S1–S6 end-to-end. Anchor for Phase B: `checkpoints/bootstrap_model_v6_step20k.pt` (SHA `297e0ce0…2bce6a`, 18.0% n=100 vs SealBot [11.7%, 26.7%]). Forensics specimen retained: `checkpoints/checkpoint_00070000.pt` (L22 attractor-capture witness; F07 deferred to Phase B S6 close-out for retention metadata sidecar OR `docs/rules/checkpoint-archive-policy.md` retention note).

Phase B scope: S1 wrapper audit + anchor n=100 re-baseline (F08), S2 dual-temperature eval ladder + Q14 close, S3 `n_components` colony POC at `pool.py game_complete` selfplay-mode emit, S4 Source A static corpus mixing design doc (lowered-expected-benefit framing per L22), S5 Source B live cross-bot games design doc (elevated to primary-fix-mechanism candidate per L22 + V6 PASS opponent-coupling), S6 close-out. Pre-registered verdicts V-PhaseB-1..9 freeze at the artifact's commit SHA per L13 + A4 do-not #9.

Phase B does NOT run a sustained training smoke; design docs queue for §177+ implementation. Pre-Phase-B fix wave commits `1d5b6b5..2014669`:

| Commit | Item |
|---|---|
| `1d5b6b5` | F01 Gate 1+2+3 close-out + L21–L23 + 3 Falsified rows (+ F06 + F09 N4 L18 inline) |
| `834f761` | F04 CLAUDE.md current phase refresh |
| `9994d5d` | F05 roadmap §175 row + §176 Phase A/B rows |
| `6c30f03` | F02 S2 eval_temperature=0.5 pin per L21 |
| `80a0205` | F03 S3 n_components threshold mode-scope qualifier |
| `2014669` | F09 N1+N3 risk register rows 8+9 |
| `3994459` | Phase B prompt artifact pre-launch baseline |

**Master merge + anchor SHA parity verification (2026-05-15).** `phase4.5/s176_phase_a_validation` merged into master via `--no-ff` (14 commits, merge SHA pinned at master push). Branch preserved local + remote per L13 traceability + bisect anchor. Anchor `bootstrap_model_v6_step20k.pt` PARITY verified across hosts: laptop SHA `297e0ce0e48c8c9c417d923610de6ed0166c7295789ebc7bd029c90bb42bce6a` (17,035,312 bytes) == vast `/workspace/hexo_rl/checkpoints/bootstrap_model_v6_step20k.pt` SHA `297e0ce0e48c8c9c417d923610de6ed0166c7295789ebc7bd029c90bb42bce6a` (17,035,312 bytes). Forensics specimen retained on vast: `/workspace/hexo_rl/checkpoints/checkpoint_00070000.pt`, SHA `1f6aa40852e57db6e3cdeac64adb879590370b8596975f2b86f9023f459224dc`, 51,131,811 bytes — L22 witness for Phase B S6 sidecar consumption.

**Phase B prompt pinned ready-to-launch (2026-05-15).** Pre-launch audit returned LAUNCHABLE_WITH_STRENGTHEN; two minor fixes folded in commit `cf14d72` (+10/-8 LOC, within 30-LOC cap): (a) Q11 naming clarification — S6 now creates a NEW `Q-§176-mechanism` Q-row carrying §174/L22 mechanism question forward, Q11 RESOLVED 2026-04-28 status + body preserved with only a one-line `See also` cross-reference appended; V-PhaseB-9 updated to match. (b) Wave A1 SHA parity re-check folded in as 30-second pre-launch insurance. Final Phase B launch artifact = `reports/s176_phase_b_prompt.md` at commit SHA `cf14d729f81f3a3f59071ad07dda3448e97c15ae`, blob SHA `6b23df987260b4affb6baa6f48efb34d24a28e2d`. A future Opus 4.7 x-high session reads that file top-to-bottom with no prior context required and executes Phase B S1–S6 end-to-end.

---

## §177-pre — Wave A1 baseline (n=100 dual-temperature SealBot eval) — 2026-05-15

Operator-driven Wave A1 ahead of S1–S6 implementation: 30-s SHA parity re-check + n=100 dual-temperature SealBot eval of the §176 Phase A anchor + variant config authoring + §177 training launch from the step-20K anchor (NOT Phase B's design-only scope; Phase B S1–S6 sequence still queued for a future session against this baseline).

**Anchor SHA parity re-check (vast 5080, post-master-pull 7d4b4fb).** PASS both:
- `bootstrap_model_v6_step20k.pt` → `297e0ce0e48c8c9c417d923610de6ed0166c7295789ebc7bd029c90bb42bce6a` (17,035,312 B) ✓ matches expected
- `checkpoint_00070000.pt` → `1f6aa40852e57db6e3cdeac64adb879590370b8596975f2b86f9023f459224dc` (51,131,811 B) ✓ matches expected

**n=100 dual-temperature SealBot baseline** (MCTS-128, `random_opening_plies=0`, `time_limit=0.5`, seed_base=42):

| Mode | n | wins | WR | Wilson 95% | mean_ply | elapsed_sec |
|---|---|---|---|---|---|---|
| greedy T=0.0   | 100 | 0  | 0.0%  | [0.0%, 3.7%]  | 77.4 | 1722 |
| sampled T=0.5  | 100 | 12 | 12.0% | [7.0%, 19.8%] | 52.7 | 1129 |

Reports: `reports/phase_b_wave_a1/{baseline_greedy.json, baseline_sampled.json, baseline_*.log}`.

**Pre-registered verdicts BL-1..BL-4 (operator prompt) — disposition.** Three of the four were NULL'd: the operator's pre-registration conflated greedy vs sampled modes for the anchor-reproduction test, but §175 step-20K's documented 18% is the **sampled** number per `eval.yaml::eval_temperature=0.5`, and L21 explicitly establishes **greedy=0%** for the same checkpoint. NULL'd per L13 + skill `investigation-probe-smoke-verdict` anti-pattern #1 (don't rewrite report to fit verdict).

| ID | User-pre-registered hypothesis | Literal verdict | Disposition |
|---|---|---|---|
| BL-1 | Greedy n=100 reproduces §175 step-20K eval (PASS: WR ∈ [11.7, 26.7]) | FAIL (0% vs 18% expected) | **NULL — basis invalidated by L21.** §175 18% is sampled mode (`eval_temperature: 0.5` pin); L21 explicitly states step-20K **greedy = 0.0%**. The 0/100 greedy result EXACTLY reproduces L21's documented greedy-mode value. Reframed: GREEDY_L21_REPRO = PASS. |
| BL-2 | Sampled diverges from greedy ≥ 6pp absolute per L21 | PASS (12pp) | **PASS** — L21 dual-mode divergence confirmed at the Phase B baseline. Locks dual-temperature gate scope for Phase B S2. |
| BL-3 | Sampled WR < greedy WR (matches §175 trajectory direction) | FAIL (sampled > greedy by 12pp) | **NULL — basis invalidated by L22.** Sampled > greedy at step-20K is the normal **pre-flattening** ordering (exploration-spread baseline). The §175 trajectory direction (sampled < greedy) emerges **post-flattening** at step-70K per V70K-3 PASS strong + §175 eval slide 18%→4%. BL-3's framing should be applied to a post-flattening checkpoint, not the pre-flattening anchor. Sampled-canonical meta-claim PASSes independently via eval.yaml `eval_temperature: 0.5` pin + L21 convention. |
| BL-4 | Colony fraction in sampled wins ≥ greedy wins by ≥ 10pp | undefined | **NULL — greedy wins=0 ⇒ divisor zero.** No greedy-side colony rate computable. Sampled-side colony measurement deferred to Phase B S3 implementation. |

**Phase B prompt S1 re-baseline verdict (canonical).** Per Section 6 V-PhaseB-5 + Section 3 S1 PASS criterion (point-estimate-in-original-CI):

- Sampled n=100 = 12.0%, point estimate **12.0% ∈ [11.7%, 26.7%]** (original §175 anchor CI).
- **V-PhaseB-5 verdict: PASS** (anchor preserved by extraction; tensor-equality + behavioral-equality both green).
- Wilson95 of 12/100 = [7.0%, 19.8%] overlaps but does not contain §175 [11.7%, 26.7%]; point-estimate criterion is what the Phase B prompt pre-registered, and it is met.
- F08 SHOULD-RE-BASELINE deferred-strengthen item now CLOSED.

**Mode comparison signature (Wave A1 forensics).**

- Greedy mean_ply 77.4 (long grinding losses, no termination via win).
- Sampled mean_ply 52.7 (12 wins terminate games earlier; sampled exploration shortens losing trajectories too).
- Sampled-greedy elapsed asymmetry: 1722 s vs 1129 s (greedy 1.5× sampled wall-time despite identical MCTS budget) — sampled's shorter games dominate.

**Variant config + training launch.** Authored `configs/variants/v6_sustained_s177.yaml` (commit `166ac7c` on master): clone of §175 `v6_sustained.yaml` with bootstrap delta only (CLI `--checkpoint` flag); L9 cosine-temp + jitter pairing preserved, `random_opening_plies=0`, `eval_interval=10000`, n_games=100, total_steps=100000. Dashboard port falls back to `monitoring.yaml` default 5001 (vast port 8080 occupied by jupyter-notebook pid 690). §177 training launched on vast 5080 in tmux `s177` from `bootstrap_model_v6_step20k.pt` along the §175 recipe — empirical zero-point for Phase B Wave C source-mixing experiments. No bot-game mixing (Source A/B arrive after Phase B S1–S6 land).

---

## Supplementary tables — preserved from per-§ bodies

### §70 mode-collapse evidence (round-robin signature)

| Matchup | Score | Game length |
|---|---|---|
| ckpt_13000 vs ckpt_14000 | 100/0 P1 | exactly 25 moves, carbon-copy |
| ckpt_14000 vs ckpt_15000 | 50/0 P1 + 50 draws | 31-33 moves, carbon-copy |
| ckpt_15000 vs RandomBot | 50/0 P1 | 11-33, varied |

H(π) band 1.49–1.70 (post-collapse) vs bootstrap 2.665 — entire post-bootstrap band sits within 0.21 nats. Fixed point, not progressive collapse. Restart should select on buffer composition, not entropy rank.

### §73 Dirichlet port verification (commit `71d7e6e`)

| Site | Count post-port | §70 count |
|---|---|---|
| `apply_dirichlet_to_root` | 10 | **0** |
| `game_runner` | 30 | 30 |

10 unique noise vectors across workers. Top-1 prior: `0.540 → 0.412` post-noise (−12.8pp). Top-1 visit fraction at cm=0: 0.474 vs §70 baseline 0.65 (−17.6pp). Workers at cm=0 ply=0 span 0.33–0.55 (diverging vs §70 identical across 14 workers).

### §91 / §100.d threat-probe criterion (locked, REVISED from §85/§89)

| # | Condition | Threshold |
|---|---|---|
| C1 | contrast_mean ≥ max(0.38, 0.8 × bootstrap_contrast) | floor 0.40 (bootstrap=0.502) |
| C2 | ext_in_top5_pct ≥ 40 | direct colony-spam test |
| C3 | ext_in_top10_pct ≥ 60 | catches partial sharpness |
| C4 | abs(ext_logit_mean − bootstrap_ext_logit_mean) < 5.0 | **warning only, never gates** |

C1–C3 must all PASS for `make probe.latest` exit 0. C4 is BCE-drift / Q19 monitoring hook. Baseline `fixtures/threat_probe_baseline.json` v6 (§106 real-position regen): contrast 0.502, top5 50%, top10 65%.

### §116 torch.compile retry — three-mode comparison (PT 2.11 + Py 3.14)

| Metric | Eager | default | reduce-overhead | max-autotune-no-cudagraphs |
|---|---|---|---|---|
| Throughput batch=64 (pos/s) | 2,529 | 3,665 | **3,788** | 3,744 |
| Throughput speedup vs eager | 1.00× | 1.45× | **1.50×** | 1.48× |
| Latency batch=1 (mean ms) | 3.553 | 2.844 | **1.897** | 3.007 |
| Latency speedup vs eager | 1.00× | 1.25× | **1.87×** | 1.18× |
| Compile time | — | 11.8 s | **6.4 s** | 29.9 s |
| Graph breaks | 0 | 0 | 0 | 0 |

**§116.a landed then reverted (`1e2d82b` + `41ffad5` mode-plumbing/OptimizedModule unwrap stay; `e102a0a` flag flipped back to `false`)** — second resume at step 6002 hit futex_do_wait on 78 threads (trainer+inference dual-JIT). Re-enable preconditions documented inline.

### §118 → §121 axis-clustering causal chain

```
RecentBuffer un-augmented (67% of late-training batch)
  → absolute-position FC policy head learns axis-asymmetric features freely
  → MCTS visits concentrate on preferred E-W axis (no symmetry pressure)
  → self-play generates axis-biased trajectories
  → RecentBuffer samples reinforce bias at 67% of gradient
  → loop closes; bias grows monotonically until truncation or intervention
```

§120 closed the symmetry coverage gap (4.7/12 → 12/12 group elements per batch row) but **augmentation alone is insufficient** for relational biases (D13 heuristic preserved under rigid transformation). Two independent components in §121 — directional heuristic (rotation-equivariant, fixed by §130 permanent rotation) + clustering magnitude (rotation-invariant, architectural). §122 architectural redesign blocked on B1 D17 ablation + B2 backbone-form memo + B3 retrain cost + B4 buffer compat.

### §156 R10 within-bisection (each variant removes ONE knob from R10)

| Variant | Knob removed | n | draws | draw_rate (95% CI) | mean_ply | stride5 P50/P90 | colony_wins | wall |
|---|---|---:|---:|---|---:|---:|---:|---:|
| R10 | (full smoke regime) | 200 | 182 | 91.0% [86.2%, 94.2%] | 140 | 84/97 | 9 | 2702s |
| R11 | Dirichlet ε=0.10 → 0 | 200 | 176 | 88.0% [82.8%, 91.8%] | 139 | 76/86 | 15 | 2649s |
| **R12** | **cosine temp → fixed τ=0.5** | 200 | 10 | **5.0% [2.7%, 9.0%]** | 63 | **3/4** | 134 | 738s |
| R13 | opening_plies 1 → 4 | 200 | 170 | 85.0% [79.4%, 89.3%] | 135 | 82/100 | 15 | 2620s |
| R14 | playout cap → uniform 600 | 200 | 198 | 99.0% [96.4%, 99.7%] | 149 | 132/133 | 0 | 3576s |

R12 colony rate 67% (134/200) is the §147 v5 / §154 v9 colony attractor — mitigated by `legal_move_radius_jitter: true` + `bootstrap_floor.min_winrate: 0.45`.

### §157 5k smoke abort-signatures (5080, 1256 games, wall 3h 18m)

| Signature | End-of-run | Threshold | Status |
|---|---|---|---|
| stride-5 P90 (rolling 50 games) | 4 | 60 | ✅ |
| row max P90 (rolling 50 games) | 13 | 50 | ✅ |
| colony_ext_frac max (per-game) | 0.086 | 0.40 | ✅ |
| colony_terminal_fraction | 0.000 | — | ✅ |
| draw_rate (last 200 games) | 7.5% | 70% (WARN-only) | ✅ |
| grad_norm | 0.98–1.62 | 10.0 hard-abort | ✅ |
| NaN losses | 0 | any | ✅ |

Final-ckpt SealBot offline eval n=200: **19.0% (38/200)** — beats 17% gate, matches §150 v7full 17.4% n=500 within sample noise.

### §167 Phase B v8 variant matrix retrains (5080 + laptop, 30 epochs each)

| Arm | Final loss | NaN-skip rate | SealBot argmax n=200 |
|---|---|---|---|
| B0 (128×12, no GPool) | 3.2737 | 0% | 0/200 |
| **B1 retry (128×12 + GPool {6,10})** | **3.227** | 24% NaN-skipped (`4c7dbb5` `isfinite` guard) | 0/200 |
| B2 (96×12 + GPool, laptop) | 3.276 | 0% | 0/200 |
| B3 (128×10 + GPool {5,8}) | 3.2536 | 0% | 0/200 |
| B4 (160×12 + GPool, batch 128 fallback) | 3.2249 (~6 ep effective) | 80% NaN-skipped | 0/200 |

v7full v6-argmax baseline radius curve: r=5 6.5% / r=8 12.5% / r=10 15%. B1 across radii r=8/10/12: 0%/0%/0%. **Cross-encoding argmax-only handicap is structural** — K-cluster's inference-time multi-window pooling acts as tiny ensemble that bbox lacks. Effect vanishes under MCTS (Phase D §168).

### §169 four-way ablation matrix (post-§169a probe)

| Arm | Encoding | Pool | Loss (30 ep) | argmax @ r=8 n=200 | MCTS-128 n=200 | params (M) |
|---|---|---|---|---|---|---|
| **A1 (canonical)** | v6w25 (25×25 K-cluster) | min/max | **3.57** | **14.5%** [10.3%, 20.0%] | 25% (§169 P1 MCTS-32 n=20) | 5.29 |
| A2 | v6w25 K-cluster | PMA | 4.25 | 4.5% [2.4%, 8.3%] | 3.5% [1.7%, 7.0%] | 6.30 |
| A3 | v6w25 K-cluster | PMA + global token | 3.62 | 7.5% [4.6%, 12.0%] | 2.5% [1.1%, 5.7%] | 6.37 |
| A4 | v8 bbox + canvas_realness + PartialConv2d | KataGo head | **3.47** | **0.0%** [0.0%, 1.9%] | **0.0%** [0.0%, 1.9%] | 3.85 |

A3 learned `pool_global_gate` climbed 0.10 → 0.66 (6.6× over init) — global branch earns weight, lifts policy argmax, but doesn't fix PMA's K=1-pretrain-regime cross-cluster blindness at search time.

### §170 P4 P1 gpool-bias-policy-only (CANONICAL)

A1+gpool-bias-policy-only retrain → **22% argmax SealBot @ r=8 n=200**. Full gpool-bias on both heads (P3) is NULL on value. Earlier attribution of 22% lift to A4 fine-tune was wrong (memory `project_bootstrap_argmax_drift_check_20260511.md` documents correction); A4 was already 0% pre-fine-tune. Mechanism: gpool-bias on policy head only, applied on A1 K-cluster.

### §172 A10 close-out — high-risk hazard retirement

| # | Site | Hazard | Closure |
|---|---|---|---|
| H1 | `engine/src/game_runner/mod.rs:159` `SelfPlayRunner::new` | pyo3 default kwargs silent v6 fallback (`feature_len=8*19*19`, `policy_len=19*19+1`) | derive from `spec.state_stride()`/`spec.policy_stride()`; legacy-caller backward-compat retained |
| H2 | `engine/src/inference_bridge.rs:295` `InferenceBatcher::new` | same pattern as H1 | `encoding_spec` kwarg added |
| H3 | `engine/src/replay_buffer/sym_tables.rs:26` `N_ACTIONS=362` | v6-only consumers; v8 silently uses wrong value | audit confirmed all v6-only; v8 pinned to `spec.policy_stride()=625`; Rust unit test pins |

A10 commits: ab760ae (T1 stamp model_variant), ae97525 (T2 migrations consolidate), a133d52 (T3 DeprecationWarning), 2dc086f (T4 RegistrySpec accessors), 1262e0c (T5 retire `*_V8` const presets), 823e241 (T6 retire `config["board_size"]`), e2a73f5 (T7 cross-table consistency INV-1..6), e83e78a (T8a allowlist 881→201 hits), f7c2bc8 (T8b HIGH-RISK pyo3 fix), 47b7f17 (T9 `<auto>` config form), 1595008 (T10 model-variant backfill), 576f69d (T11 pyo3 `from_py_object` TODO).

### §172 Phase B B2 milestone curve (30K v7full sustained, n=20)

| Step | sealbot | bootstrap_anchor | best_arena (n=100) | argmax_n | elo | promoted |
|---|---|---|---|---|---|---|
| 5K  | 0.100 | 0.350 | 0.410 | 0.000 | -94.2 | F |
| 10K | 0.200 | 0.600 | 0.570 | 0.000 | +50.5 | F (CI block) |
| 15K | 0.050 | 0.650 | 0.500 | 0.000 |  -9.4 | F |
| **20K** | **0.050** | **0.650** | **0.610** | 0.000 | +34.0 | **T** (only promotion) |
| 25K | 0.050 | 0.500 | 0.560 | 0.000 | -63.2 | F (CI block) |
| 30K | 0.050 | 0.600 | 0.550 | 0.000 | -36.3 | F (CI block) |

§150 v7full anchor SealBot 17.4% n=500. B2 finished sealbot 0.050 n=20 Wilson95 [0.009, 0.236]. **REGRESSION gate did not fire** (UB 0.236 covers anchor LB 0.143). DRIFT gate cold (argmax_n 0/20 all rounds). Self-play improving vs self (best_arena 0.41 → 0.61) while sealbot stalled — **encoder-specific transfer gap, not value-drift pathology**.

### §173 bench gate (pre-α vs post-A5b v2, n=5, 90s warmup, compile OFF)

| Metric | Pre-α median | Post-A5b v2 | Δ | Target | Status |
|---|---|---|---|---|---|
| MCTS sim/s | 80,601 | 80,287 | −0.4% | ≥ 26,000 | PASS |
| NN inference pos/s | 14,278 | 14,148 | −0.9% | ≥ 8,250 | PASS |
| NN latency ms | 1.551 | 1.537 | −0.9% | ≤ 3.5 | PASS |
| Buffer push pos/s | 992,777 | 1,023,047 | +3.1% | ≥ 630,000 | PASS |
| Buffer sample raw µs | 757 | 764 | +0.9% | ≤ 1,550 | PASS |
| Buffer sample aug µs | 759 | 768 | +1.2% | ≤ 1,800 | PASS |
| GPU util % | 94.0 | 94.0 | — | ≥ 85% | PASS |
| VRAM GB | 0.105 | 0.105 | — | ≤ 6.4 | PASS |
| Worker pos/hr | 80,715 | **104,141** | **+28.9%** | ≥ 250,000 | PASS |
| Batch fill % | 99.999 | 99.976 | −0.02pp | ≥ 80% | PASS |

A5b initial −10.47% worker_pos_per_hr regression recovered to +6.01% via scalar-API + `#[inline]` (`feedback_registryspec_by_ref_in_hotpath.md` — RegistrySpec ~174-byte copy per MCTS sim).

### §173 HAZARD ledger (closed)

| HAZARD | Description | Closed by |
|---|---|---|
| H1-α | SymTables v6 unconditional → K-window rotation silent shape mismatch | A5a (`sym_tables_for()`) |
| H2-α | `rotate_aux_inplace` TOTAL_CELLS=361 silent ownership corruption for v6w25 | A5a |
| H3-α | `views[k][..TOTAL_CELLS]` truncates chain encoding for v6w25 | A5a |
| H4-α | `aggregate_policy*` BOARD_SIZE=19 — 362-vector where 626 required | A5b |
| H5-α | `sample.rs:220` pass-slot copy: latent OOB for v8 (`has_pass_slot=false`) | A4 |
| H6-α | `mod.rs:342 STATE_STRIDE` v6 constant in `collect_data` | A5a |
| H7-α | HEXB on-disk format has no encoding-name header — blocks first v6w25 persist | **CARRIED to §174** |

Python `EncodingSpec.n_cells` parity bug (used `board_size²` instead of `trunk_size²`) closed by A3.

### §174 bootstrap matrix (post-mortem)

| Bootstrap | Recipe | Final loss | SealBot MCTS-128 (random_plies=0) | Selfplay median plies @ R=8 | G4 status |
|---|---|---|---|---|---|
| **v6 (`bootstrap_model.pt`)** | 30 ep cosine 2e-3/5e-5 | — | reference | reference | PASS — §175 anchor |
| v7full | 30 ep cosine 2e-3/5e-5 | 3.1573 | 17.4% n=500 (§150) | — | PASS |
| v6w25 e30 | 30 ep cosine 1e-3/5e-5 | 3.96 nats vs uniform | **0% (0/200)** | 6 | PASS within band |
| v6w25 e50 | 50 ep cosine 1e-3/5e-5 | (lower) | 10% (10/100, artifact-suspect) | 6 | **MARGINAL FAIL** 0.489 vs band [0.154, 0.462] |
| v6w25 transfer FT | v6 trunk + Xavier policy FC + drop-restart FT | — | **0% (0/200)** | 8 | — |

Eval random_opening_plies 4 → 0 in `configs/eval.yaml:88` fully explains §168 → §174 sealbot WR drop (14.5% → 0%) — with 4 random plies model got free positional diversity masking weaknesses; with 0 SealBot's preparation lands cleanly.

### G-gate wiring status (Track 2 audit, 2026-05-13)

| Gate | Description | Wiring |
|---|---|---|
| G3 | Monotonic depth scaling | `avg_game_length` in `iteration_complete` (orchestrator.py:336); per-game `game_length` in structlog `game_complete` (pool.py:593) |
| G4 | Value-head |max| ±50% band [0.154, 0.462] around v7full 0.308 | **NEW §174 Track 2** — `_g4_value_head_band_check` runs at start of every `run_evaluation`; result persisted in `results["value_fc2_weight_abs_max"]` + `results["g4_value_head_band_pass"]`; structlog WARNING on violation; constants gate-internal (variants do not override) |
| G5 | Per-cluster variance drift ≤30% | `cluster_value_std_mean` + `cluster_policy_disagreement_mean` + `cluster_variance_sample_count` emitted in `iteration_complete` (orchestrator.py:349-351) + `train_step_summary` (orchestrator.py:404-406); drift detection is post-hoc operator computation |

`random_opening_plies` two distinct fields (selfplay vs eval paths): `selfplay.random_opening_plies` (`configs/selfplay.yaml:66` default 1, vast.yaml override 0); `eval_pipeline.eval_random_opening_plies` (`configs/eval.yaml:88` default 0, was 4 pre-§174). Pipeline build path `pipeline_setup.py:52` loads eval.yaml directly — separate from training base-config list.

### §174 escalation decision matrix

| Track 1 finding | §175 action |
|---|---|
| e30 v6w25 ≥ §150 v7full anchor on MCTS-128 sealbot n=100 | Launch sustained with e30 v6w25 |
| e30 v6w25 < §150 anchor by > 5pp absolute | Re-evaluate: retrain with different recipe OR fall back to v7full for §174 |
| e30 v6w25 within ±5pp of §150 anchor (within noise) | Launch sustained — gap is in measurement noise; α + radius curriculum are net new levers |

Track 1 returned 0% MCTS-128 across all three v6w25 bootstrap recipes → escalation to §175 v6 sustained (100K steps, n=100 SealBot eval, matched cosine LR from §174 vast.yaml, selfplay encoding v6 single-window 19×19 existing path).

---

## §178 — Rust engine refactor cycle 1 close (2026-05-15 → 2026-05-16)

Branch `refactor/rust-engine`. Three-wave cycle bracketed by a 90-proposal Phase 3 audit pass at `audit/rust-engine/00_aggregated_proposals.md` + per-file split addendum at `audit/rust-engine/01_file_split_addendum.md`. Wave 1 = foundation (docs + clippy floor + LazyLock migration). Wave 2 = dead-code purge (six commits, net −789 LOC, bitboard module + mcts dead setters + dead lib.rs PyO3 surfaces). Wave 3 = hot-path correctness (P1 silent v6w25 corruption, P2 v8 `has_pass_slot` dispatch, P3 cross-language `EncodingSpec` retirement) + pre-3d hygiene (Python test consumer cleanup, NN-latency triangulation). Cycle bench gate PASS; INV15+INV16+INV17 pinned; 6 settled decisions migrated below.

**Forward pointer:** archive consolidation at `reports/sprint_archive/§178_rust_engine_audit.md` (user post-cycle action — see `audit/rust-engine/wave_3/3d/archive_prep_note.md`). Source-of-truth audit tree retained at `audit/rust-engine/` for forensic reference until archive lands; SD1–SD6 are cite-from-future-prompts surface.

### Wave-by-wave commit lineage

| Wave | Commits | Range | LOC ± | Reviewer | Notes |
|---|---:|---|---|---|---|
| 1 — foundation | 5 | `4bff8c7..5391e79` | +75 / −25 | 5× APPROVE | docs (P87, P84, P88), clippy lint config (P60 — 3 erasing_op errors resolved, exit 0 maintained throughout), once_cell::Lazy → std::sync::LazyLock (P70) |
| 2 — dead-code purge | 6 | `a311347..fd22bc2` | +760 / −1549 (net **−789**) | 7× APPROVE (Reviewer F = P86 investigation) | bitboard module −347 LOC (P16+P41), 11 V6W25 consts + src_plane_lookup −121 (P17+P44), Zobrist demotion (P85), 12 PyMCTSTree setters + `vl_adaptive` (P15+P27), 5 dead Board PyO3 surfaces + view_window twin + 3 inference_bridge PyO3 (P24+P25), mcts/mod.rs → mcts/tests.rs pure-cut-paste (file-split addendum). P26 SKIP (integration test caller) + P86 RETAIN (zero runtime cost, historical reproducibility) — see SD1+SD2. |
| 3a — P1 CRITICAL + state.rs split + INV15 | 3 | `5d411c4..54baab8` | +1148 / −786 | APPROVE | P1 silent `TOTAL_CELLS=361` v6w25 corruption: `encode_state_to_buffer_channels` + `encode_chain_planes` parameterised on `n_cells`; `state.rs` split atomic with kernel mod into `state/{core,encode,cluster}.rs`. INV15 (3 v6w25 + v6-byte-identity tests). Bench WATCH → resolved at 3b. |
| 3b — P2 v8 has_pass_slot + INV16 | 1 | `867164e` | +428 / −51 | APPROVE | `aggregate_policy[_to_local]` + `get_policy`/`get_improved_policy` thread `has_pass_slot` from `spec`; v8 (n=625) corner cell no longer zeroed by unconditional pass-slot write at `records.rs:68`. INV16 (3 v8 + v6 dispatch tests). Bench PASS. Parent-prompt CONSTRAINTS inverted v6/v8 semantics → implementer followed registry per SD4. |
| 3c — P3 EncodingSpec retirement + INV17 | 2 | `a2b0be1..8ea6436` | +356 / −558 (net −204) | APPROVE | Python `engine.EncodingSpec` wrapper retired → `engine.RegistrySpec.from_registry(name)` classmethod (P3.1); Rust `PyEncodingSpec` pyclass + `PyBoard::with_encoding` + getter retired (P3.2, breaking change `!`-marked). 16 files, INV17 Rust (3 tests) + Python (2 tests). Bench PASS-with-WATCH; NN latency / push / sample aug carried to pre-3d triangulation. |
| pre-3d — H1 test cleanup + H2 NN triangulation | 1 | `d74972a` | +138 / −441 (net −303) | n/a (hygiene) | 39 → 1 Python test failures resolved: `test_chain_plane_rust_parity` retired whole-file (Python `_compute_chain_planes` canonical), `test_corpus_chain_target` migrated to Python kernel, `InferenceServer.submit_and_wait/.infer` rewritten as direct sync forward (closes 00b7d2b coordinated-Python-PR promise), `apply_symmetry(s,i)` → `apply_symmetries_batch(s[None],[i])[0]` migration. H2 fresh bench confirms NN latency reverts to Wave 2 baseline; SD6 second-confirm point. |

### Cycle metrics

- Test counts at cycle close: **Rust 249** tests (across 17 binaries, 0 fail) + **Python 1549 / 1 failed / 18 skipped / 4 deselected / 1 xpassed** (the 1 failure is `tests/test_policy_target_metrics.py::test_cost_budget_under_200us_at_b256` — a timing-budget flake under concurrent `make bench` GPU contention; the test's own docstring discloses 10× idle-baseline tolerance specifically to handle this scenario, and the budget tripped today only because `make test.py` ran while `make bench` was holding 100% GPU. Not a Wave 3 code regression — pre-cycle suite was 1550/0 fail post-H1 measurement.)
- Clippy floor: `cargo clippy --release` exit 0, **190 lib warnings** (Wave 1 opened gate at 199; Wave 2 closed at 192; Wave 3 closed at 190 — strict downward trend across cycle).
- Cycle bench gate: **PASS** per `audit/rust-engine/wave_3/3d/cycle_bench_verdict.md`. 9 metrics measured at HEAD = `d74972a`; `all_targets_met` PASS. Cumulative vs Wave 2 close (SD5 anchor): MCTS sim/s +3.23%, NN inf +0.13%, NN latency +8.91% (PASS-WITH-WATCH per SD6), buffer push +14.86% IMPROVED, sample raw −16.08% IMPROVED, sample aug −8.25% IMPROVED, GPU flat, worker pos/hr −4.36% (PASS-WITH-WATCH per SD6), batch fill +1.01%. Cumulative vs Phase 0: MCTS +72.97%, NN latency −6.27%, push +0.04%, raw −7.53%, aug +5.09%, worker −10.60% (inside ±12.5% IQR envelope), batch fill −0.72% — net cycle improvement on load-bearing MCTS sim/s, neutral elsewhere.

### Settled decisions

The six entries below were maintained in `audit/rust-engine/cycle_settled_decisions.md` during the cycle and are cited from future Phase 4/5 prompts that touch the affected surfaces.

- **SD1 — P26 SKIP: `PyRegistrySpec::from_static` retained as `pub`.** Audit P26 proposed demoting `pub fn from_static` to `pub(crate)`; reality found at implementation: `engine/tests/test_worker_loop_v6w25_smoke.rs:69` calls it for §173 A5a v6w25 SelfPlayRunner construction guard. Integration tests are external to the crate; `pub(crate)` would break compilation. SKIP confirmed. Unblock conditions: migrate that test to `PyEncodingSpec::from_registry` (now retired — would need to be `PyRegistrySpec::from_registry`, available post-P3.1), OR expose `pub(crate)` in-crate constructor. Cite in any future prompt that touches `engine/src/lib.rs` `PyRegistrySpec` or registry constructor surface.

- **SD2 — P86 RETAIN: `v7` and `v7e30` registry entries kept.** P86 INVESTIGATE recommended INVESTIGATE; investigation found wire format `v7` byte-identical to `v6`, `v7e30` byte-identical to `v7full` — both legitimate historical names with zero production dispatch but loadable §148/§149/§150 ckpts depending on them. Cost RETAIN = 0 runtime (LazyLock lazy-init); RETIRE = 14 file edits + 2 ckpt renames + breakage risk. RETAIN AS-IS. SSR-collapse spirit applies to *duplicate* SoTs, not historical-tag entries in a single SoT TOML. Cite in any prompt touching `engine/src/encoding/registry.toml`.

- **SD3 — Per-commit scope-expansion-by-deletion is permitted.** Wave 2 + Wave 3 produced 8 forced scope expansions total (Wave 2: A→benches, B→d6_sym_tables, D→selection.rs; Wave 3: 3a→state.rs tests, 3b→3 test sites, 3c→4 Rust mod-tests + 5 positional-arg sites + `compat.py`/`pool.py` cascade, pre-3d H1→`inference_server.py`). All disclosed in commit bodies, all reviewer-approved, all minimal. Rule: expansion file MUST reference an in-scope deleted symbol; edit MUST be minimal (delete dead ref or migrate to surviving API); commit body MUST disclose; reviewer MUST confirm forced+minimal. Half-deletion that leaves the codebase non-compiling is worse. Cite under CONSTRAINTS in all Phase 4 implementer prompts. **Mechanism Lesson candidate L24** if pattern recurs in future cycles.

- **SD4 — Implementer/reviewer corrections to audit MD take precedence.** Wave 2 + Wave 3 found audit MD inaccuracies via implementation-time `rg` inventory: P15 audit said 10 dead PyMCTSTree setters → real 12; P26 audit said zero callers → integration test caller existed; P1 audit estimated 12 literal sites → impl observed 19 substitutions (some lines carried two `TOTAL_CELLS` each); 3b parent prompt CONSTRAINTS INVERTED v8 vs v6 pass-slot semantics (implementer followed registry); 3c PREP §D enumerated 3 deleted Rust mod-tests → reality was 4. Rule: implementer must run verification `rg` checks per their prompt's CONSTRAINTS section; if reality contradicts the audit MD findings count/caller list, report in commit body + adjust scope; if reality CONTRADICTS the proposal premise (e.g. "zero callers" falsified) → STOP and report. Cite under DONE-WHEN in all Phase 4 implementer prompts. **Mechanism Lesson candidate L25** if pattern recurs.

- **SD5 — Bench baseline re-anchored at Wave 2 close.** Phase 0 baseline measured at `072d0db` (pre-Wave-1) became stale after 6 dead-code-purge commits dropped 788 LOC. Re-snapshot at `fd22bc2` (`audit/rust-engine/wave_3/00_bench_baseline_post_wave_2_run2.txt`) is the formal Wave 3 cumulative bench-gate baseline. Phase 0 baseline preserved unchanged as canonical "before refactor cycle" measurement for Phase 5 bench audit + this § entry's transparency table. Cite as the "baseline" file path in any Wave 4+ bench-gate prompt.

- **SD6 — Mechanism-absent bench WATCH on small-absolute deltas is measurement variance until 2-3 commits confirm.** Promoted at 3d cycle close. NN latency b=1 trajectory across 6 measurements: 2.47 → 2.58 → 2.49 → 2.62 → 2.47 → 2.69, oscillating-revert without code mechanism (P1/P2/P3 don't touch NN paths; 3d and pre-3d H2 are the same HEAD `d74972a`). Independently confirmed on buffer push (3b WATCH → 3c → H2 +25.55%) and sample aug (3c → H2 −21.08%). Rule: WATCH metric is treated as measurement variance — NOT actionable — when code commit does not mechanistically explain it AND subsequent commits show non-monotonic/reverting behavior (or fresh-bench triangulation reverts toward baseline); escalated to investigation only when monotonic over 3+ commits AND code-level mechanism connects. Operator implication: future cycles do NOT freeze on a single-commit bench WATCH; require 2-commit confirmation OR fresh-bench triangulation. Cite this entry in Phase 4/5 bench-gate subagent prompts as the verdict heuristic.

### INV pin additions

- **INV15** — v6w25 encode round-trip regression pin (3a, `engine/tests/inv15_v6w25_encode_roundtrip.rs`, 3 `#[test]` fns: corner-cell byte-identity, v6 byte-identity-unchanged regression guard, v6w25 chain-plane axis runs)
- **INV16** — v8 has_pass_slot dispatch pin (3b, `engine/tests/inv16_v8_pass_slot_dispatch.rs`, 3 `#[test]` fns: v8 aggregate_policy preserves corner cell, v8 aggregate_policy_to_local preserves corner cell, v6 pass-slot zeroing regression guard)
- **INV17** — PyRegistrySpec.from_registry classmethod supersedes PyEncodingSpec (3c, Rust `engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs` 3 fns + Python `tests/test_inv17_pyregistryspec_retired.py` 2 fns)

### Open items deferred to Wave 4 / next cycle

- `engine/src/encoding/spec.rs` legacy Rust `EncodingSpec` struct + `Board::with_encoding` `cfg(test)`-only survivors (PREP 3c §A deferral). Kept as test-only fixtures pending operator review of test-only surface; not blocking but inelegant.
- `engine/src/lib.rs` → `pyo3/{board,mcts,encoding,utils}.rs` split (file-split addendum Wave-5 sequencing, dependencies now all settled — P3 done in 3c, Wave 2 deletes done in P15+P24+P25+P26-SKIP). Eligible Wave 4 candidate.
- `engine/src/game_runner/worker_loop.rs` split (881 LOC, INVESTIGATE) — blocked on P69 inline-test coverage per file-split addendum.
- `engine/src/game_runner/mod.rs` split (936 LOC) — DEFER confirmed; re-eval gates `[P22, P58]`.
- Remaining Phase 3 audit-pass proposals not addressed by Waves 1–3: tracked in `audit/rust-engine/00_aggregated_proposals.md`; next-cycle audit-pass should re-classify against post-cycle HEAD.

### Falsified Hypotheses Register additions

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §178 (Wave 3 3b) | v8 `has_pass_slot=true` (parent prompt CONSTRAINTS claim) | 3b SD4 catch + INV16 Test A | Registry `engine/src/encoding/registry.toml` declares v8 has_pass_slot=false; implementer ran rg on registry source per SD4 and disregarded inverted parent claim; INV16 Test A pins the corner-cell-preserved-when-has_pass_slot-false contract |
| §178 (Wave 3 3a) | P1 audit claim "12 `TOTAL_CELLS` literal sites in production code" | 3a impl rg sweep | Real count was 19 substitutions (some lines carried two literals each); disclosed in P1.1 commit body; SD4 application |
| §178 (Wave 2 P15) | P15 audit claim "10 dead PyMCTSTree setters" | Implementer D rg sweep | Real count 12; commit `1d68d5b` lists correct enumeration; SD4 application |

### Archive reference

`reports/sprint_archive/§178_rust_engine_audit.md` (user post-cycle action — preserve full audit tree forensics for cross-cycle reference per `audit/rust-engine/wave_3/3d/archive_prep_note.md`). Memory: pending — recommend `project_178_rust_engine_audit.md` covering cycle outcome + SD6 entry + INV15/16/17 pin paths for cross-session lookup.

---

## §179 — Rust engine refactor cycle 2 close (2026-05-16)

Branch `refactor/rust-engine-cycle-2`. Five-wave cycle bracketed by the §178 cycle 1 close at master `e0e7c47`. Wave 4 = MCTS hot-loop allocation cleanup + InferenceBatcher pool sizing (3 commits, +184 LOC, PASS-WITH-WATCH after SD6 bisection triangulation). Wave 5 pre-flight = three pre-existing test-suite flakes triaged (1 commit, test-only fixes, no production touch). Wave 5a = PyO3 boundary hardening + held fold-in (P74/P75/P76/P77) + Python `inference_pool_size` wiring (5 commits, net +784 LOC, zero-copy strengthens A+E, SD6 single WATCH). Wave 5b = `engine/src/lib.rs` structural split into `engine/src/pyo3/{board, encoding, mcts, utils}.rs` (4 commits, net +79 LOC, lib.rs 793 → 34 LOC at split, 45 LOC post-Wave-6 L26 rustdoc). Wave 6 = clippy ride-through + idiom polish + tail (P19, P42, P63, P64, P66) (5 commits, +110 LOC, clippy 186 → 42 warnings, −77.4%). Cycle bench gate PASS at Wave 6 close per `audit/rust-engine/cycle_2/wave_6/wave_close_bench_verdict.md`; all 10 `all_targets_met` checkpoints GREEN. INV15+INV16+INV17 pinned through entire cycle (Rust 9/9 + Python 2/2 at every wave close). SD1–SD6 preserved and cited per commit body across all 18 cycle-2 commits.

**Forward pointer:** archive consolidation at `reports/sprint_archive/§179_rust_engine_audit.md` (decision recorded at `audit/rust-engine/cycle_2/close/archive_prep_decision.md`). Source-of-truth audit tree retained at `audit/rust-engine/cycle_2/` for forensic reference until archive lands. Phase 5 bench audit verdict at `audit/rust-engine/cycle_2/close/03_bench_audit.md`; cycle 3 baseline anchored at `audit/rust-engine/cycle_2/close/04_baseline_next_cycle.txt`.

### Wave-by-wave commit lineage

| Wave | Commits | Range | LOC ± | Reviewer | Notes |
|---|---:|---|---|---|---|
| 4 — MCTS hot-loop + InferenceBatcher pool | 3 | `68a6e97..408a5c5` | +281 / −97 (net **+184**) | 3× APPROVE | Batch A select+backup atomic (P5+P6+P7+P8+P9+P36 `567ac4f`); Batch B cluster scratch + legal_moves + Gumbel policy (P10+P11+P33 `0ffe116`); Batch C InferenceBatcher pool sizing (P55 `408a5c5`). Wave-close bench initially FAIL on MCTS sim/s + worker pos/hr (>2% mechanism); SD6 bisection at A-only HEAD + A+B HEAD reverted both to neutral-or-positive → PASS-WITH-WATCH. Six WATCH metrics carried into Wave 5 monitoring (5 mechanism-absent variance, 1 with code mechanism within IQR). |
| 5 pre-flight — test floor triage | 1 | `4cec7c0` | +60 / −23 | n/a (hygiene) | Three pre-existing flakes resolved test-only: `test_hexb_v6_legacy_load_python` sample count 20→200 (birthday-bound miss 5.76% → 2e-19); `test_bootstrap_entropy_range` sister-guard for `policy_fc != 362`; `replay_buffer/persist::tests` + `replay_buffer_v6_roundtrip` `unique_test_path()` helper (pid + nanos + atomic counter, tempdir-reuse fix). Test floor preserved 1550→1553 Python; 249→249 Rust. |
| 5a — PyO3 boundary hardening + Python wiring | 5 | `4cec7c0..7aab309` | +953 / −169 (net **+784**) | 5× APPROVE (1 doc nit) | Batch A `#[pymethods]` zero-copy + PyO3 0.28 (P34+P71+P76+P77 `30cf281`); Batch B worker_loop hardening + `inference_pool_size` Python wiring (P22+P51+P52+P67 `e264e04`); Batch C wire-signature crossload + HashSet hoist (P13+P45 `325fe8b`); Batch D quiescence fetch_add + bench-fidelity hoist (P32+P38 `0db73bb`, MCTS sim/s +8.71% from bench-fidelity); Batch E `Arc<Vec<f32>>` inference path + Python defensive copy drop (P74+P75 `7aab309`). Zero-copy STRENGTHENS on A + E per architecture-doc claim. One SD6 WATCH (buffer sample augmented, mechanism-absent, fresh-bench reverts toward baseline). 1 NEEDS-WORK doc nit on Batch C v6w25 wire_signature rustdoc — held to Wave 6 Batch D. |
| 5b — `lib.rs` structural split | 4 | `7aab309..25e796b` | +854 / −775 (net **+79**) | APPROVE | Sequential per-file extraction: `pyo3/encoding.rs` (`e4c9e27`); `pyo3/utils.rs` (`634d65a`); `pyo3/board.rs` (`f4c47d2`); `pyo3/mcts.rs` (`25e796b`). `lib.rs` 793 LOC → 34 LOC (target ≤80 met, 95.7% reduction). Wave 5a method-body edits (P34/P71/P76/P77) byte-identically relocated to per-file homes. G.6 shadow-extern mod collision mitigated via `use ::pyo3::prelude::*;` in lib.rs (leading `::` forces extern-crate resolution); preflight at `audit/rust-engine/cycle_2/wave_5b/g6_preflight.txt` confirmed 44+ compile errors without the mitigation. Three WATCH metrics (buffer raw + aug + worker batch fill) SD6 variance per fresh-bench triangulation; Wave 6 close reverted all three. Test counts grew 249→255 Rust, 1550→1553 Python. Mechanism Lesson candidate **L26** opened (see Settled decisions below). |
| 6 — clippy ride-through + tail | 5 | `25e796b..ba19b1c` | +378 / −268 (net **+110**) | 5× APPROVE (1 NEEDS-WATCH `i32::midpoint`) | Batch A `cargo clippy --fix` LOW (`2b0dd08`, 186→81 warnings); Batch B manual MED (`8b269bd`, 81→58); Batch C site-local HIGH `#[allow]` w/ cycle 3 P-anchors (`2e17672`, 58→42 — 15 attribute lines, 16 lint IDs, all rationale-commented); Batch D docs cascade — Wave 5a v6w25 policy_logit_count nit + L26 rustdoc + P23/P40/P48/P59 (`546bae3`, doc-only); Batch E residual proposals P19 (`n_chain_planes()` accessor) + P42 + P63 + P64 + P66 (`ba19b1c`). Clippy floor 186 → 42 (−77.4%, far beat PREP target ≤110 and stretch ≤90). Wave 6 close bench PASS no WATCH-with-mechanism; Wave 5b 3-WATCH set all reverted toward baseline confirming SD6 variance. 1 NEEDS-WATCH carried to cycle 3: `i32::midpoint` signed-semantics at 4 sites (`cluster.rs:75,96` + `state/core.rs:365,366`); tests + bench neutral but worth forensic re-check if v6w25 K-cluster eval anomaly surfaces. |

### Cycle metrics

- Test counts at cycle close: **Rust 255 tests** across 21 binaries (+6 vs cycle 1 close 249); **Python 1553 passed / 0 failed / 18 skipped / 4 deselected / 1 xpassed** (+3 vs cycle 1 close 1550 and net 0 failures vs cycle 1 close's 1 transient timing flake — the `test_cost_budget_under_200us_at_b256` flake did not recur under cycle 2 bench cadence).
- Clippy floor trajectory: **191 (cycle 2 baseline = cycle 1 close) → 189 (Wave 4) → 186 (Wave 5a) → 186 (Wave 5b, structural split no lint touch) → 42 (Wave 6 close) — net −149 / −78.0% across cycle 2.** Strict downward trend; the Wave 6 single-wave −144 is the largest clippy-floor reduction in cycle 1 or cycle 2 history. 15 site-local `#[allow]` attributes (16 lint IDs) anchor the residual to cycle 3 P79 (builder) + P68 (module split) refactors (see `audit/rust-engine/cycle_2/close/06_allow_to_cycle3_anchor_map.md`).
- Cycle bench gate: **PASS** at Wave 6 close per `audit/rust-engine/cycle_2/wave_6/wave_close_bench_verdict.md` AND **PASS** at cycle close median of 3 fresh runs per `audit/rust-engine/cycle_2/close/03_bench_audit.md`. 10 metrics measured at HEAD `ba19b1c`; `all_targets_met` PASS on all 3 runs. **Cycle close median anchor** (cycle 3 baseline): MCTS 66,289 sim/s (+75.16% vs Phase 0 / +4.53% vs cycle 1 close); NN inf b=64 4,859.7 pos/s (stable); NN latency b=1 2.84 ms (+14.98% vs cycle 1 close, SD6 mechanism-absent variance candidate — cycle 2 NN forward path untouched); buffer push 826,625 pos/s (+11.88% vs Phase 0; bench measures slow path P12 deferred); sample raw 970.5 µs (−18.19% vs cycle 1 close, P45 HashSet hoist mechanism); sample aug 990.8 µs (−25.69% vs cycle 1 close, same P45 fn); worker pos/hr 33,354 (−9.30% vs Phase 0 within ±11% IQR; +2.30% vs cycle 2 baseline); worker batch fill 99.28% (+0.95% vs cycle 2 baseline); GPU util 100.0% (saturated). Net cycle-2 impact at the median: bench-neutral on MCTS sim/s + worker pos/hr (variance dominates on laptop 4060 Max-Q with ±11% IQR on worker pos/hr); STRENGTHENS on zero-copy compliance (3 PyO3 surfaces converted to `IntoPyArray`; 2 String per-child drops; Arc-based inference payload); IMPROVED on sample raw + aug via P45.
- Total LOC delta cycle 2: **+2,526 / −1,332 = net +1,194 LOC** across 18 commits over the 5 implementation waves (Wave 4 +184 + Wave 5 pre-flight +37 + Wave 5a +784 + Wave 5b +79 + Wave 6 +110). Cycle 2 net growth is dominated by Wave 5a Batch B's `WorkerCtx` introduction (P52, +~500 LOC for the worker-loop boundary hardening) + Wave 5b's per-file `register(m)` function additions; cycle 2 is the inverse of cycle 1's −789 LOC dead-code purge wave.

### Settled decisions (SD1-SD6 preserved; L26 promoted from candidate to Mechanism Lesson)

The six entries below were maintained in `audit/rust-engine/cycle_settled_decisions.md` during the cycle 1 close and remained the operating record through cycle 2. Cycle 2 introduced no new SDs but adopted **L26** as a Mechanism Lesson promoted from Wave 5b PREP candidate.

- **SD1 — P26 SKIP: `PyRegistrySpec::from_static` retained as `pub`.** Cited verbatim in Wave 4 Batch A, Wave 5a Batch A (`#[pyclass(from_py_object)]` annotation preserved `pub` visibility), Wave 5b Batch 1 (extraction to `pyo3/encoding.rs` — `from_static` stays `pub` per integration-test caller `engine/tests/test_worker_loop_v6w25_smoke.rs:69`), Wave 6 Batches A-E (no touch). Zero violations across cycle 2's 18 commits.
- **SD2 — P86 RETAIN: `v7` and `v7e30` registry entries kept.** Cycle 2 had zero `engine/src/encoding/registry.toml` diff across all 18 commits; all 8 encodings (`v6, v6w25, v7full, v7, v7e30, v7mw, v8, v8_canvas_realness`) live and registry-driven. Wave 5a Batch C (P13 wire-signature crossload) added `wire_signature()` accessor that consumes `v7`/`v7e30` registry entries to validate cross-encoding HEXB v7 load rejection — the historical entries became actively load-bearing for the crossload guard.
- **SD3 — Per-commit scope-expansion-by-deletion is permitted.** Cycle 2 produced 8 disclosed SD3 expansions: Wave 4 Batch A (P9 `pending` tuple shape + `MoveDiff` import trim); Wave 4 Batch B (`aggregate_policy_to_local` 8th parameter into 2 integration tests, 4 call sites); Wave 4 Batch C (channel-size co-parameterisation `max(n*2, 1024)`); Wave 5a Batch A (`engine/tests/perspective_parity.rs` axial-tuple destructure migration); Wave 5a Batch B (`engine/src/replay_buffer/sym_tables.rs` `LazyLock<SymTables>` v6-default accessor); Wave 5a Batch C (`wire_signature()` derived accessor); Wave 5b commit 1 (`replay_buffer/mod.rs:302` struct-literal → `PyRegistrySpec::from_static`); Wave 5b commit 3 (`PyBoard::from_inner` demoted to `pub(crate)` + new `inner_ref()` accessor); Wave 6 Batch A (wildcard-import revert at test sites); Wave 6 Batch D (P48 test rename held back — 3+ load-bearing callers, SD3-minimal); Wave 6 Batch E (P19 deferred test-site migrations on `apply_chain_symmetry` signature change). All 11 expansions disclosed in commit bodies, all reviewer-approved minimal. **Mechanism Lesson candidate L24** strengthening evidence: now observed across 2 consecutive cycles with consistent pattern, recommend promotion to Mechanism Lesson at the cycle 3 close if pattern recurs again.
- **SD4 — Implementer/reviewer corrections to audit MD take precedence.** Cycle 2 produced 11+ SD4 applications: Wave 4 PREP §A line drifts (P5/P9 ±1); Wave 4 Batch B path drift (P10 caller in `state/cluster.rs:38-55` vs function at `board/moves.rs:277-318`); Wave 4 Batch C audit `:326-329` → live `:283-286` (pre-cycle-2 restructure); Wave 5a Batch A (P71 `skip_from_py_object` audit claim vs reality `Option<PyRegistrySpec>` requires `FromPyObject` → implementer used `from_py_object` opt-in); Wave 5a Batch C (P13 tuple-type drift `(u8,u8,u16,...)` → `(usize,usize,usize,...)`); Wave 5a Batch D (P32 readers check — 5 reader sites verified at external-synchronization boundaries); Wave 5a Batch E (P74/P75 line drifts ±48 from cycle-1 file restructure); Wave 5b commit 1 (PREP §B `pub(super) fn register` → `pub(crate) fn register` — lib.rs is grandparent not direct parent); Wave 5b commit 4 (PREP §F.2 assertion `engine.Board.__module__ == 'engine'` falsified — PyO3 default is `'builtins'`); Wave 6 Batch B (8 PREP §B lints with zero live sites at HEAD already swept by Batch A); Wave 6 Batch D (PREP §E claim 4 v6w25 doc-nit sites → 3 in source tree); Wave 6 Batch E (line-number drifts on P42, P64, P66 from Wave 3a state.rs split). **Mechanism Lesson candidate L25** also strengthening evidence over 2 consecutive cycles; recommend promotion at cycle 3 close if pattern recurs again.
- **SD5 — Bench baseline re-anchored at Wave close.** Cycle 2 re-anchored the bench baseline three times: cycle 2 baseline at HEAD `68a6e97` (cycle 1 close) per cycle 2 PREP, then Wave 5a baseline at HEAD `4cec7c0` (pre-flight close), then Wave 5b baseline at HEAD `7aab309` (Wave 5a close), then Wave 6 baseline at HEAD `25e796b` (Wave 5b close). Each wave's bench gate compared its close against its own baseline; cycle-2 cumulative reporting at cycle close uses the chained delta (see `audit/rust-engine/cycle_2/close/03_bench_audit.md`). Phase 0 baseline at `audit/rust-engine/00_bench_baseline.txt` (HEAD `072d0db`) preserved unchanged as the canonical cross-cycle reference point. New cycle 3 baseline at `audit/rust-engine/cycle_2/close/04_baseline_next_cycle.txt` (HEAD `ba19b1c`, median of 3 fresh measurements).
- **SD6 — Mechanism-absent bench WATCH on small-absolute deltas is measurement variance until 2-3 commits confirm.** SD6 was exercised heavily across cycle 2. Wave 4 close: 6 WATCH metrics, 4 mechanism-absent (NN latency, buffer push, sample raw, sample aug); SD6 bisection bench at intermediate HEADs A-only + A+B reverted MCTS sim/s and worker pos/hr toward baseline-or-positive → confirmed environmental noise dominated the wave-close single-run anomaly; final verdict PASS-WITH-WATCH. Wave 5a close: 1 WATCH (buffer sample aug +6.13%) confirmed non-monotonic over run 1/run 2 → SD6 variance. Wave 5b close: 3 WATCH (buffer raw +5.02%, buffer aug +17.45%, worker batch fill −2.28pp) all mechanism-absent and non-monotonic. Wave 6 close: Wave 5b 3-WATCH set REVERTED toward baseline (buffer raw −19.04%, buffer aug −22.39%, worker batch fill within 0.7pp); SD6 discipline confirmed all three were cycle-2 measurement variance, not refactor cost. Cycle 2 strengthens SD6: **bidirectional variance confirmed** — metrics swing both higher and lower than baseline across consecutive measurements without code mechanism, exactly matching the original SD6 wording "non-monotonic / variance". No SD6 escalation triggered across the 5 wave-close bench-gate runs.

**Mechanism Lesson L26 promotion (NEW at cycle 2 close):** Rust local `pub mod pyo3` shadows the `pyo3` extern crate inside its resolution scope. When `engine/src/lib.rs` declares `pub mod pyo3;` to host the cycle-2 PyO3 split, naive `use pyo3::prelude::*;` resolves to the local module (not the extern crate), causing 44+ compile errors. **Mitigation:** lib.rs uses `use ::pyo3::prelude::*;` (leading `::` forces extern-crate resolution); submodules in `engine/src/pyo3/` use the unqualified form (local `mod pyo3` is not in their resolution scope). Wave 5b preflight (`audit/rust-engine/cycle_2/wave_5b/g6_preflight.txt`) confirmed the collision empirically before commit 1 landed. Wave 6 Batch D L26 rustdoc at `engine/src/lib.rs` near `pub mod pyo3;` documents the mechanism for future cross-cycle reference. **Operator action:** cite L26 in any cycle 3 prompt that opens a new local `mod <name>` whose name matches an extern crate (esp. `pyo3`, `tokio`, `serde`, `std`). Promoted from candidate to Mechanism Lesson at §179 close.

### INV pin status

| Pin | File | Tests | Status at HEAD `ba19b1c` |
|---|---|---|---|
| INV15 (v6w25 encode round-trip) | `engine/tests/inv15_v6w25_encode_roundtrip.rs` | 3 Rust | GREEN |
| INV16 (v8 has_pass_slot dispatch) | `engine/tests/inv16_v8_pass_slot_dispatch.rs` | 3 Rust | GREEN |
| INV17 Rust (PyRegistrySpec supersedes PyEncodingSpec) | `engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs` | 3 Rust | GREEN |
| INV17 Python (PyEncodingSpec retired) | `tests/test_inv17_pyregistryspec_retired.py` | 2 Python | GREEN |

**Zero new INV pins in cycle 2.** All five wave-close artifacts confirm INV15+INV16+INV17 Rust 9/9 + Python 2/2 PASS. PyO3 boundary hardening + lib.rs structural split + clippy ride-through did not change correctness contracts; cycle 2 was structural and idiomatic, not behavioral.

### Open items deferred to cycle 3

Consolidated from naming-inventory + `#[allow]` anchor map + Wave 6 "Open items for cycle 3" + DEFER-bin proposals (per `audit/rust-engine/cycle_2/wave_6/PREP_plan.md` §A.2):

- **Cycle 3 anchor refactor candidates:**
  - **P79 builder pattern** for `SelfPlayRunner::new` (40-param ctor) + `ReplayBuffer::push`/`push_game`/`push_many` PyO3 surfaces + `apply_sym` / `scan_line` / `scan_line_general` / `record_game_runner` interior helpers. Closes 15 `#[allow]` attributes per `audit/rust-engine/cycle_2/close/06_allow_to_cycle3_anchor_map.md`.
  - **P68 module splits** for `engine/src/encoding/registry.rs::parse_one`, `engine/src/encoding/spec.rs::validate`, `engine/src/replay_buffer/persist.rs::load_from_path_impl`, `engine/src/game_runner/worker_loop.rs::start_impl` (the last is gated on P69 inline-test coverage). Closes 4 `#[allow(clippy::too_many_lines)]` attributes.
  - **FF.2 / FF.3 / FF.10** — Python ↔ Rust full-schema `EncodingSpec` duplicate retirement. Carries cycle 1 P3-pattern (legacy `EncodingSpec` cfg(test) survivor at `engine/src/encoding/spec.rs` + `Board::with_encoding`) to final retirement. Bundles in 6 GENERICISE + 5 CONSOLIDATE naming candidates per `audit/rust-engine/cycle_2/close/05_naming_inventory.md`.
  - **K_max registry field (Option A TOML schema)** — Wave 5a PREP §C.3 deferred; Wave 6 ratified for cycle 2 deferral. Decision point: add to TOML registry or keep operator-driven Python-side default. Separate config-system PR; not gated on P79+P68.
  - **Naming-cleanup wave** — folds into the FF.2/FF.3/FF.10 PR per cycle 2 naming-inventory recommendation; no standalone wave needed. Adds 2 new proposals (P91 `min_max_v6_head` → generic; P92 `compute_v8_mask` → generic + registry `mask_polarity` field).

- **i32::midpoint signed-semantics forensic** at `engine/src/board/cluster.rs:75,96` and `engine/src/board/state/core.rs:365,366` (Wave 6 Batch A NEEDS-WATCH; non-blocking; cycle-3 if v6w25 K-cluster anomaly surfaces). Documented in §"Falsified Hypotheses Register additions" below.
- **P19 deferred test-site migrations** (`apply_chain_symmetry` signature change + `#[cfg(test)]` body `N_CHAIN_PLANES` callers — Wave 6 Batch E SD3 hold-back).
- **Legacy Rust `EncodingSpec` cfg(test) survivor** at `engine/src/encoding/spec.rs` + `Board::with_encoding`. Cycle 1 PREP 3c §A + cycle 2 PREP §A.2 + cycle 2 naming-inventory U1 all flag for FF.2/FF.3/FF.10 retirement.
- **`worker_loop.rs` split** (881 → ~917 LOC after Wave 5a Batch B; INVESTIGATE per cycle 1 file-split addendum). Gated on P69 inline-test coverage. Cycle 3 candidate alongside P68 module splits.
- **`game_runner/mod.rs` split** (~789 LOC after Wave 5a Batch B). DEFER confirmed; re-eval gates `[P22, P58]`. P22 landed in Wave 5a Batch B so 1 of 2 gates closed. P58 (SelfPlayRunner Drop ordering race) still gating.
- **Per Wave 6 PREP §A.2 DEFER bin:** P4, P12, P14, P18, P20, P21, P28, P29, P30, P31, P37, P39, P43, P46, P47, P49, P50, P53, P54, P56, P57, P58, P62, P65, P69, P72, P73, P78, P79, P80, P81, P82, P83, P89, P90.

Remaining ~75 proposals (post cycle 1's 18 landed) tracked in `audit/rust-engine/00_aggregated_proposals.md`. Cycle 3 audit-pass should re-classify against post-cycle-2 HEAD `ba19b1c`.

### Falsified Hypotheses Register additions

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §179 (Wave 4 close, PREP §G) | "Net IMPROVEMENT predicted from Batch A MCTS allocation cleanup" | Single wave-close bench showed −3.03% MCTS sim/s + −3.87% worker pos/hr with code mechanism | Bisection bench at intermediate HEADs revealed environmental noise dominated (A-only HEAD: worker pos/hr +10.95% vs baseline; A+B HEAD: MCTS sim/s +2.16%). Median Wave 4 effect bench-neutral or bench-positive; the wave-close run was hit by laptop thermal / background load. SD6 fresh-bench triangulation discipline applied. |
| §179 (Wave 5b commit 4, PREP §F.2) | "`engine.Board.__module__ == 'engine'` after PyO3 split" | F.2 smoke assertion at IMPL time | PyO3 default `__module__` is `'builtins'` regardless of `#[pymodule]` placement; smoke updated to assert `'builtins'` invariance. SD4 application. |
| §179 (Wave 6 Batch A NEEDS-WATCH) | "`(min_q + max_q) / 2` and `i32::midpoint(min_q, max_q)` are byte-identical for signed integers" | Reviewer flag during Batch A `cargo clippy --fix` review | `i32::midpoint(a, b)` rounds toward `-∞` (floor); `(a+b) / 2` truncates toward `0`. They differ by 1 when `(a+b)` is negative-odd: `i32::midpoint(-5, -2) == -4`; `(-5 + -2) / 2 == -3`. Tests + bench GREEN at Wave 6 close (downstream window-flat-idx absorbs ±1 shifts); cycle 3 forensic carries the watch in case v6w25 K-cluster eval surfaces anomaly attributable to the shift. |
| §179 (Wave 5a Batch A) | "P71 audit MD claim: deprecation closes via `skip_from_py_object`" | SD4 implementer `rg` sweep | Reality at HEAD: `Option<PyRegistrySpec>` is used as PyO3 input in `inference_bridge.rs:285` + `game_runner/mod.rs:201`, requiring `FromPyObject`. Implementer used `from_py_object` (opt-in) instead — both clear the deprecation, substantive correction caught at impl time. SD4 application. |
| §179 (Wave 5b commit 4, IMPL_complete) | "lib.rs `register(m)` fns can be `pub(super)`" | SD4 implementer correction at IMPL | lib.rs is the GRANDPARENT of the `register` fns (lib.rs → pyo3 mod → pyo3::encoding mod → register fn), not the direct parent. `pub(super)` would scope to `pyo3` only; lib.rs needs `pub(crate)` for resolution. SD4 application. |
| §179 (Wave 4 close, SD6 reaffirmation) | "Wave 4 single-bench-close FAIL = real regression on MCTS sim/s + worker pos/hr (mechanism + >2% threshold)" | SD6 bisection triangulation discipline | SD6 mandates fresh-bench triangulation when multiple mechanism-absent metrics regress in lockstep on a single bench run (4 metrics regressed without Wave 4 code mechanism on those code paths). Bisection at A-only HEAD + A+B HEAD reverted MCTS sim/s and worker pos/hr toward baseline-or-positive. Verdict downgraded FAIL → PASS-WITH-WATCH per SD6. **Strengthens SD6:** confirms SD6 escalation gate (mechanism + monotonic over 2-3 commits) is the correct discriminator, not the single-wave-close median rule. |

### Archive reference

`reports/sprint_archive/§179_rust_engine_audit.md` (post-cycle action — preserves full audit tree forensics for cross-cycle reference). Source-of-truth audit tree at `audit/rust-engine/cycle_2/` retained on the operator's workstation. Memory: pending — recommend `project_179_rust_engine_audit.md` covering cycle 2 outcome + L26 promotion + SD compliance preservation + P79/P68/FF.2/FF.3/FF.10 cycle 3 anchor handoff for cross-session lookup.

---

## Sprint memory map (cross-conversation pointers)

| Topic | Memory file |
|---|---|
| §171 A4 P2-reopen C DEAD verdict (commits ee8032a..be8b14a) | `project_171_a4_p2_reopen_c_dead.md` |
| Bootstrap argmax drift check 2026-05-11 (vast 5080) | `project_bootstrap_argmax_drift_check_20260511.md` |
| §172 B2 v7full sustained closed 2026-05-11 | `project_172_b2_complete.md` |
| §172 B1-redo (G1 anchor loader fix + R12 cosine variant) | `project_172_b1_redo_complete.md` |
| §172 A10 close-out (13 commits + HIGH-RISK fixes) | `project_172_a10_complete.md` |
| §172 A9 review verdict PASS | `project_172_a9_complete.md` |
| §172 A8 docs cascade (README + CLAUDE.md + tree) | `project_172_a8_complete.md` |
| §172 A6 round-trip test parameterized over registry | `project_172_a6_complete.md` |
| §172 A5 ckpt + corpus sidecar metadata schemas + audit CLI | `project_172_a5_complete.md` |
| §172 A4 plumbing pass (6 commits) | `project_172_a4_complete.md` |
| §172 A3 registry.toml + Rust/Python modules | `project_172_a3_complete.md` |
| §172 A2 design doc | `project_172_a2_complete.md` |
| §172 A1 Q1-Q5 resolved | `project_172_a1_resolved.md` |
| §171 P3 resolved (A1+A2 reopen cycle) | `project_171_p3_blocked.md` |
| §171 P2 complete (sprint_171_p3_5080.yaml) | `project_171_p2_complete.md` |
| Audit Board mutators after `Board::with_encoding` | `feedback_encoding_post_mutators_audit.md` |
| §170 closed 2026-05-09 P3 FALSIFIED + P4 P1 CANONICAL | `project_170_p3_falsified.md` |
| Current vast.ai host `ssh6.vast.ai:13053` | `project_current_vast_host.md` |
| v7/v8 corpora share 6,259 raw human games (encoding changes density) | `feedback_v6_v8_same_training_data.md` |
| v8 0% SealBot WR is K-cluster argmax handicap (structural) | `project_v8_argmax_handicap.md` |
| §167 Phase B closed 2026-05-08 (5-arm matrix) | `project_phase_b_verdict.md` |
| §173 A3-A6 bundle closed (α multi-window operational) | `project_173_a3_a6_bundle_complete.md` |
| RegistrySpec by-ref in hotpath rule (L16) | `feedback_registryspec_by_ref_in_hotpath.md` |
| §169 four-way closed 2026-05-08 (A1 canonical, bbox falsified) | `project_169_four_way_complete.md` |
| §169 A3 partial positive verdict | `project_a3_p3_verdict.md` |
| §155 T1.1 v6 92% draws reproducible under frozen v7full | `project_phase_b_prime_v10_t1_1_verdict.md` |
| §154 v9 hex-trunk turn FALSIFIED | `project_phase_b_prime_v9_falsified.md` |
| Use throughput-optimal configs (n_workers/batch/wait/burst) for new smokes | `feedback_smoke_use_optimal_throughput_config.md` |
| Phase B' v9 §153 engineering complete | `project_phase_b_prime_v9_hex_native.md` |
| Phase B' Class-4 east→west stride-5 stone spam | `project_phase_b_prime_class4_q_axis_stride5.md` |
| §131 P3 complete (model 18→8, bridge removed, ckpt guard live) | `project_hexbv6_p3_complete.md` |
| §131 P2 complete (buffer 8 planes, corpus regenerated) | `project_hexbv6_p2_complete.md` |
| §121 closed 2026-04-25 (split verdict) | `project_d16_selfplay_rotation.md` |
| D-ladder 2026-04-23 curr_10k forgetting | `project_diag_d_20260423.md` |
| Supply-side perf wave 2026-04-22 (+8.5% pos/hr, +12.6% bench) | `project_supply_wave_2026-04-22.md` |
| Q33-B pe_self ≈ 5.36 fixed point (Q37 RESOLVED at §112) | `project_q33b_verdict.md` |
| Calibration in-flight (4×4hr sweep R1-R4 launched 2026-04-17) | `project_calibration_inflight.md` |
| Q17 RESOLVED 2026-04-10 (Dirichlet ported) | `project_phase40_status.md` |
| §128 bench metric switch to positions_generated | `project_bench_metric_switch_128.md` |
| Doc conventions (positions/hour, augment=False in convergence tests) | `feedback_doc_conventions.md` |
| Corpus strategy (500 game wall, SealBot wrapper) | `project_corpus_strategy.md` |
| Bench audit 2026-04-01 (cells_iter fix, P0 resolved) | `project_bench_audit_20260401.md` |
| Throughput regression analysis (forced-win removal → +30% NN calls) | `project_throughput_regression_analysis.md` |
| Clone fix 2026-04-01 (Board::Clone skip legal_cache copy) | `project_clone_fix_20260401.md` |
| Bench variance (warm-up + n=5 median tames to <7% IQR) | `feedback_bench_variance.md` |
| Architecture upgrade 2026-04-01 (SE blocks, BCE value, aux head) | `project_arch_upgrade_20260401.md` |
| Test speed (cap game-loop smokes at ~100 moves) | `feedback_test_speed.md` |
| Omarchy Linux (no cpupower, cannot pin CPU frequency) | `user_system_omarchy.md` |
| Current dev host is laptop (4060 Max-Q) | `feedback_current_host_is_laptop.md` |
| torch_compile=False required in test configs | `feedback_torch_compile_tests.md` |
| §116 torch.compile landed 2026-04-24 (reduce-overhead) | `project_torch_compile_116.md` |
| venv only; never install to system site-packages | `feedback_venv_only.md` |
| Early draw rate (steps 0-500) is noise | `feedback_draw_rate_early_noise.md` |
| Rolling window resets cause draw_rate spikes (track trend) | `feedback_draw_rate_windowing.md` |
| Do NOT abort on draw_rate alone (§157 user verdict) | `feedback_draw_rate_not_abort_signal.md` |
| Smoke variants must set eval_interval ≥ 2500 | `feedback_smoke_eval_interval_min_2500.md` |
| Use X/(X+O) ex-draws as fairness signal, not raw rates | `feedback_winrate_balance.md` |
| Post-sustained: characterize 80.7% stall before Q35 vs other Phase 4.5 target | `project_stall_diagnostic_deferred.md` |
| py-spy verdict 2026-04-25 (dispatcher module-dispatch bound) | `project_dispatch_pyspy_2026-04-25.md` |
| §118 recovery run launched 2026-04-24 | `project_phase118_recovery.md` |
| Reduce-overhead CUDA graph TLS is per-thread | `feedback_torch_compile_threading.md` |
| At current NN size, torch.compile regresses selfplay pos/hr | `feedback_compile_selfplay_dispatch_bound.md` |
| W3 complete 2026-04-30 (Q41 WARN, Q52 PASS, Phase 4.0 UNBLOCKED) | `project_w3_complete_20260430.md` |
| Low colony fraction in eval is POSITIVE | `feedback_colony_fraction.md` |
| Bootstrap corpus error (bot games at uniform source_weight=1.0) | `project_bootstrap_corpus_bot_contamination.md` |
| §160 eval split landed 2026-05-06 | `project_s160_eval_split.md` |

---

## §180 — Rust engine refactor cycle 3 close (Waves 6.5, 7, 8, 9, 10, 11)

Branch `refactor/rust-engine-cycle-3`. Six-wave cycle bracketed by the §179 cycle 2 close at master `ba19b1c`. Wave 6.5 = `i32::midpoint` revert + INV18/INV18b truncate-semantics pin (1 commit, +50 LOC, NEEDS-WATCH from cycle 2 Wave 6 Batch A closed at cycle 3 entry per `audit/rust-engine/cycle_3/00_i32_midpoint_forensic.md`). Wave 7 = P79 builder pattern across 12 large-arity ctors + P68 small module splits for 3 monolithic >100-LOC fn bodies (5 commits `3ef3100..a37a50c`, net +1,230 LOC, 5× PASS-or-PASS-WITH-NOTES, bench 9 PASS / 4 WATCH, INV19+INV20+INV21 pinned). Wave 8 = FF.2/FF.3/FF.10 EncodingSpec retirement cohort + Wave-8 Batch D naming-fold (4 commits `a37a50c..43d5d8a`, net +438 LOC, 3× PASS-WITH-NOTES + 1× PASS, bench 4 PASS / 5 WATCH all mechanism-absent + SD6-bidirectionally-reverted, INV22 Python parametrized + INV23 e2e pinned, 2 `!`-markers across A+C). Wave 9 = P55 K_max Option A registry field + InferenceBatcher pool auto-derive (1 commit `4eefd53`, net +210 LOC, PASS-WITH-NOTES, bench SKIPPED per zero hot-path touch, INV24 K_max + INV22 extension pinned). Wave 10 = `worker_loop` structural split into 7 sibling modules + P69 inline-test scaffold for InferenceBatcher early-return paths (2 commits `f53975e` + `8ba72be`, net +829 LOC, PASS-WITH-NOTES + PASS, bench GREEN 6 PASS / 3 WATCH, INV25 byte-identity-on-behavior pinned with 3 cells incl. `include_str!` destructure substring assertion). Wave 11 = cycle 3 close engineering (tail clippy + J.2.b `run_worker_thread` 712-LOC body sub-fn extraction into 8 sub-fns + REVIEW-driven §173 A5b rationale annotation, 3 commits `e678757` + `5a63a23` + `5e0c09d`, net +346 LOC, per-wave bench SKIPPED per Phase 6 CYCLE-BENCH n=10 supersedes). Cycle bench gate **PASS** at cycle 3 close (n=4×10 laptop primary 40 samples + n=2×10 vast.ai mirror 20 samples; 10/10 metrics within Phase 4.5 floors; worker pos/hr −9.95% vs Wave 10 ref WATCH-edge per SD7 bidirectional variance — bimodal CUDA across runs 1+2 low / 3+4 high, mechanism-absent for Batch B §173 A5b discipline). Cycle 4 entry baseline anchored at `audit/rust-engine/cycle_3/close/04_baseline_next_cycle.txt` (HEAD `5e0c09d`, 2026-05-18).

**Forward pointer:** archive consolidation at `reports/sprint_archive/§180_rust_engine_audit.md` (post-cycle operator action — mirrors cycle 1 §178 / cycle 2 §179 archive precedent). Source-of-truth audit tree retained at `audit/rust-engine/cycle_3/` for forensic reference until archive lands. Phase 5 bench audit verdict at `audit/rust-engine/cycle_3/close/03_bench_audit.md`; cycle 4 baseline at `audit/rust-engine/cycle_3/close/04_baseline_next_cycle.txt`.

### Wave-by-wave commit lineage

| Wave | Commits | Range | LOC ± | Reviewer | Notes |
|---|---:|---|---|---|---|
| **6.5 — `i32::midpoint` revert + truncate semantics pin** | 1 | `(cycle 3 entry parent)..3ef3100` | +50 / −5 | n/a (cycle entry) | Cycle 2 Wave 6 Batch A NEEDS-WATCH (clippy `--fix` converted 4 sites in `cluster.rs:75,96` + `state/core.rs:365,366` from `(a+b)/2` truncate-toward-0 to `i32::midpoint` floor-toward-−∞) closed at cycle 3 entry. Revert + 2 new INV pins (INV18 `window_center` + INV18b `cluster_center` negative-bbox truncate semantics). v6/v6w25 checkpoint calibration preserved pre-Wave-6.5 byte-identity per forensic at `audit/rust-engine/cycle_3/00_i32_midpoint_forensic.md`. |
| **7 — P79 builder + P68 module splits** | 5 | `3ef3100..a37a50c` | +2,147 / −917 (net **+1,230**) | 5× APPROVE-or-PASS-WITH-NOTES (3 PASS + 2 PASS-WITH-NOTES; 0 CRITICAL / 0 MAJOR / 4 MINOR / 17 NIT) | Batch A `f715780`! SelfPlayRunner takes SelfPlayRunnerConfig builder (38-param ctor + 14 Python sites + 9 Rust SD3 expansion); Batch B `8315d15` config struct for ReplayBuffer push API impl methods (3 sibling structs — `PushSingleConfig` / `PushGameConfig` / `PushManyConfig` — zero shared subset per SD4 #1); Batch C `54a60f4` WorkerChannels + WorkerParams bundle (anonymous closure destructure pattern; PREP `fn signature` framing reframed via L27 candidate); Batch D `39ccc7d` config-struct sweep `apply_sym` / `scan_line` / `scan_line_general` / `record_game_runner`; Batch E `a37a50c` P68 module splits — `parse_one` → `parse.rs`, `RegistrySpec::validate` → `validate.rs`, `load_from_path_impl` → `persist/load.rs` (pure-file-move; L29 helper-extraction-blocks-allow-retire surface). 9 PASS / 4 WATCH bench; SD7-candidate evidence base started (Buffer push +5.04% Batch B + Buffer aug −5.57% Batch D both directionally inconsistent with PREP predictions). |
| **8 — FF.2/FF.3/FF.10 EncodingSpec retirement cohort + naming-fold** | 4 | `a37a50c..43d5d8a` | +1,339 / −901 (net **+438**) | 3× PASS-WITH-NOTES + 1× PASS (0 CRITICAL / 0 MAJOR / 3 MINOR / 12 NIT) | Batch A `a6ca01b`! Python `EncodingSpec` @dataclass retire — consumers migrate to `engine.RegistrySpec` (type alias at `hexo_rl/encoding/__init__.py`; 10 new PyO3 `#[getter]` accessors + 5 `pub fn → #[getter]` conversions for attribute-surface parity); Batch B `ba82d67` legacy Rust 4-field `EncodingSpec` + `Board::with_encoding` ctor + 5 tests retired (cycle 1 P3 final closure); Batch C `9f0f2dc`! SelfPlayRunner encoding round-trip collapse to `encoding_name: Option<String>` + `WireFormatSpec` + `WIRE_FORMAT_SPECS` + `legacy_spec_for_registry_name` retire + 6 of 7 `audit: legacy-v6-fallback` arms collapse to `PyValueError` (1 operator-locked comment-and-keep at `mcts/mod.rs:226` bench-harness-only — L31 candidate); Batch D `43d5d8a` `min_max_v6_head` → `min_max_window_head` rename + naming-fold sweep (replay aliases collapse to `_REPLAYERS` dispatcher; `_build_v6/v8_model` → `_build_min_max/kata_model` + unified `_build_model_from_spec`; `_v6/v8_net` → `_net_from_spec`). 4 PASS / 5 WATCH bench; ALL 5 WATCHes mechanism-absent (zero MCTS / NN / replay-buffer hot-path touch); 3 Wave-7→Wave-8 bidirectional reversions (Buffer push, Buffer aug, ride-through on raw) STRENGTHEN SD7. 2 `!`-markers (cross-language surface retirements). |
| **9 — P55 K_max Option A registry field** | 1 | `43d5d8a..4eefd53` | +233 / −23 (net **+210**) | PASS-WITH-NOTES (0 CRITICAL / 0 MAJOR / 1 MINOR / 4 NIT) | Batch A `4eefd53` — `k_max: u32` schema field added to `engine/src/encoding/registry.toml` (8 entries) + Rust `RegistrySpec` struct field + 2-line parser delta + 4-line validator `k_max >= 1` rule + PyO3 `#[getter]` accessor + INV24 (3 Rust cells: presence-and-positive + golden snapshot of 8 (name, k_max) tuples + single-window-implies-k_max=1 discipline) + INV22 extension (18 → 19 `_REQUIRED_FIELDS`) + consumer auto-derive in `SelfPlayRunner::new` ctor `inference_pool_size` default-path with PREP §L.6 `.max(512)` floor mitigation (preserves v6 default 512 byte-for-byte; v6w25 grows to 1792). Bench SKIPPED per zero hot-path touch (registry-load-time + ctor-time + PyO3 accessor only). NO `!`-marker (non-breaking schema field add; zero Python `engine.RegistrySpec(...)` kwarg-construct sites). L33 candidate opened (1st instance: schema field add as cleanest mechanism for cross-language consumer-cap defaults). |
| **10 — `worker_loop` 7-module split + P69 inline-test scaffold** | 2 | `4eefd53..8ba72be` | +1,958 / −1,129 (net **+829**) | PASS-WITH-NOTES + PASS (0 CRITICAL / 0 MAJOR / 0 MINOR / 5 NIT across both batches) | Batch A `f53975e` — file split of `engine/src/game_runner/worker_loop.rs` (1,129 LOC) into 7 sibling files under `worker_loop/`: `mod.rs` (orchestration + `start_impl` + extracted `build_worker_prototypes`) + `inner.rs` (`run_worker_thread` named-fn extraction carrying migrated `#[allow(clippy::too_many_lines)]`) + `rotate.rs` (5 `#[inline]` rotate helpers + `compute_move_temperature`) + `params.rs` (4 Wave 7 Batch C bundle structs + new `WorkerGeometry` Copy bundle) + `channels.rs` + `stats.rs` + `atomics.rs`. INV25 pinned (3 cells; Cell 3 `include_str!` substring assertion of destructure pattern is anti-tautological). Operator-bound to U7 FLAT shape, U10/J.2.a defer `run_worker_thread` sub-fn extraction to Wave 11. Bench GREEN 6 PASS / 3 WATCH all mechanism-absent + SD6 envelope. Batch B `8ba72be` — P69 inline-test scaffold at `engine/src/inference_bridge.rs::tests` (2 `#[cfg(test)]` cells: `submit_batch_and_wait_rust_returns_err_when_closed` + `submit_batch_and_wait_rust_returns_err_on_length_mismatch`). U9 = Option B.2.c scaffold-only binding; `infer_and_expand` test target inside `worker_loop/inner.rs` deferred (closure-vs-fn extraction decision separate). Test count 274 → 276. 2 Wave-8→Wave-10 bidirectional reversions (Buffer push + Buffer sample augmented) STRENGTHEN SD7. |
| **11 — Cycle 3 close engineering (tail clippy + J.2.b sub-fn extraction)** | 3 | `8ba72be..5e0c09d` | +991 / −645 (net **+346**) | Batch A clippy-clean; Batch B PASS-WITH-NOTES (annotation pass in `5e0c09d` follows REVIEW); 0 REWORK | Batch A `e678757` retired `#[allow(dead_code)]` on `InferenceBatcher::submit_batch_and_wait_rust` (P69 inline tests from Wave 10 Batch B closed the dead-code gap; clippy silent post-edit). Batch B `5a63a23` extracted `run_worker_thread` 712-LOC body into 8 sub-fns (`run_one_game`, `init_per_game_board`, `infer_and_expand`, `run_mcts_search`, `play_one_move`, `select_move`, `record_position`, `finalize_game`) + 5 helper structs (`InferContext`, `ClusterVarianceAtomics`, `MoveAccumulators`, `MovePlayContext`, `PerGameInitCtx`, `PerGameInit`, `MoveOutcome`, `McTSSearchResult`); parent body fell to ~108 LOC clippy-counted, retiring `#[allow(clippy::too_many_lines)]` at pre-Wave-11 `inner.rs:52`. PREP-vs-IMPL drifts D1 (`build_per_game_state` SKIPPED — INV25 substring assertion protection) + D2 (`infer_and_expand` promoted to `#[inline] fn`; PREP recommended closure but `#[allow]` retire forced top-level form) + D3 (F1 17→29 vs PREP target 16; mechanism = §173 A5b hot-path-by-value discipline forbids arg-bundling on 8 sub-fns). Batch B follow-up `5e0c09d` annotated the 8 new `clippy::too_many_arguments` + 1 `fn_params_excessive_bools` allows with §173 A5b rationale (cycle 4 retire-or-keep predicate anchor). Per-wave bench gate SKIPPED per Phase 6 CYCLE-BENCH n=10 supersedes. |

### Cycle metrics

- Test counts at cycle close: **Rust 276 tests** across 22 binaries (+21 vs cycle 2 close 255: Wave 6.5 +6 INV18+INV18b; Wave 7 +6 INV19+INV20+INV21; Wave 8 +4 INV23; Wave 9 +3 INV24; Wave 10 +3 INV25 + 2 P69 inline; Wave 11 0 new); **Python 1565 passed / 0 failed / 19 skipped / 1 xpassed / 33 warnings** (+12 vs cycle 2 close 1553: Wave 8 +12 INV22 17 parametrized cells + 2 P79 INV20 facade + reorg).
- Clippy floor trajectory: **42 (cycle 3 baseline = cycle 2 close) → 42 (Wave 6.5; revert is byte-neutral) → 42 (Wave 7; 5 batches lint-neutral with new P79+P68 allows offset by Wave 6 Batch C retirements that landed in cycle 2) → 42 (Wave 8; FF.2/FF.3/FF.10 retirement lint-neutral) → 42 (Wave 9; schema field add lint-silent) → 42 (Wave 10; split byte-neutral via U10/J.2.a binding) → 42 (Wave 11; net 0 — Batch A −1 + Batch B +12 from §173 A5b hot-path discipline − 1 for L52 retire = 0 net clippy delta, but F1 substantially grew).** Clippy WARNING count stable across entire cycle 3 (cycle 2 Wave 6 Batch C established the canonical 42-warning floor).
- `#[allow]` floor trajectory: **F1 absolute 22 (cycle 3 entry parent `3ef3100`) → 18 (Wave 7 close; Batch A +2 + Batch B −3 + Batch C −2 + Batch D −4 = net −4 absolute via NON-BREAKING facade constraint at Batches B+C+D; +0 from Batch E pure-file-move) → 18 (Wave 8; net 0 across 4 batches; registry-flow restructuring + cross-language @dataclass retirement is not a clippy sweep) → 18 (Wave 9; net 0; schema field add lint-silent) → 18 (Wave 10; net 0 per U10/J.2.a binding — migrating `#[allow(too_many_lines)]` from pre-split worker_loop.rs:260 to inner.rs:52 is structural-not-numeric) → 29 (Wave 11 close; Batch A −1 dead_code + Batch B −1 too_many_lines + Batch B +12 §173 A5b hot-path-by-value discipline + 1 ride-through = +11 net).** PREP §J ≤6 aspirational target NOT MET; cycle 4 absorbs residual under new "P79 hot-path discipline" anchor per `06_allow_to_cycle4_anchor_map.md`.
- Cycle bench gate: **PASS** at cycle 3 close. Hosts: laptop primary (Ryzen 7 8845HS + RTX 4060 Laptop GPU, n=4 × n=10 internal = 40 samples per metric) + vast.ai mirror (Ryzen 9 9900X + RTX 5080, n=2 × n=10 internal = 20 samples per metric). Headline laptop median worker pos/hr **29,118** (range 26,425–30,565) vs Wave 10 reference 32,334 = **−9.95%** — SD7 verdict WATCH-edge: bidirectional bimodal CUDA across n=4 runs (runs 1+2 cluster low 26-28k / runs 3+4 cluster high 30-31k), mechanism-absent for Batch B sub-fn extraction (§173 A5b hot-path-by-value discipline preserved; scalar-API at fn entry). Worker batch fill **98.82%** (+0.5pp vs Wave 10). Vast.ai mirror median worker pos/hr **83,692** (range 80,101–87,283), 2.87× laptop. Cross-host ratios: MCTS sim/s vast 1.23× laptop, NN inf b=64 3.18× (5080 SM advantage), NN lat b=1 0.58× (3.18× faster in absolute ms), buffer push 1.22×, GPU util laptop 100% saturating, vast 94%. 10/10 canonical metrics within Phase 4.5 floors at both hosts. Per-host floor formula `tightest_of(median − 2σ, 5th_percentile)` applied at cycle 4 entry baseline `04_baseline_next_cycle.txt`. Full Phase 6 raw data + verdict at `audit/rust-engine/cycle_3/close/cycle_bench.md`.
- `!`-marker count cumulative cycle 3: **3** (`f715780` Wave 7 Batch A SelfPlayRunner config builder; `a6ca01b` Wave 8 Batch A FF.2 Python EncodingSpec dataclass retire; `9f0f2dc` Wave 8 Batch C FF.10 SelfPlayRunner encoding round-trip collapse).
- Total LOC delta cycle 3: **+6,174 / −3,199 = net +2,975 LOC** across 16 commits over 6 waves (Wave 6.5 +50 + Wave 7 +1,230 + Wave 8 +438 + Wave 9 +210 + Wave 10 +829 + Wave 11 +346 + REVIEW annotation +0 + miscellaneous +-126 reconciliation). Cycle 3 net growth is dominated by Wave 7's P79 builder pattern (config-struct boilerplate; +1,230 LOC) + Wave 10's `worker_loop` split (per-file module docstrings + `WorkerGeometry` bundle + sub-module helpers; +829 LOC) + Wave 11's 8-sub-fn extraction (signature + doc-comment overhead; +346 LOC). Inverse of cycle 1's −788 LOC dead-code purge wave; consistent with cycle 2's +1,194 LOC structural-growth pattern.

### Settled decisions (SD1-SD6 preserved; SD7 PROMOTED at cycle 3 close)

The six SD entries below carry forward from cycle 1 (`§178`) + cycle 2 (`§179`); SD7 promotes at cycle 3 close per `cycle_settled_decisions.md` update + U12 default.

- **SD1 — P26 SKIP: `PyRegistrySpec::from_static` retained as `pub`.** Cited verbatim across cycle 3 Waves 7-11; integration-test caller `engine/tests/test_worker_loop_v6w25_smoke.rs:69` still resolves via `engine::PyRegistrySpec`; preserved in cycle 3 Wave 8 Batch A FF.2 retire (Python `EncodingSpec` dataclass → `EncodingSpec = engine.RegistrySpec` type alias; `PyRegistrySpec` PyO3 surface unchanged). Zero violations across cycle 3's 16 commits.

- **SD2 — P86 RETAIN: `v7` and `v7e30` registry entries kept.** Cycle 3 had ZERO `engine/src/encoding/registry.toml` entry removals — Wave 9 ADDED `k_max: u32` schema field + bumped `schema_version` 2 → 3 across all 8 entries; all 8 encodings (`v6, v6w25, v7full, v7, v7e30, v7mw, v8, v8_canvas_realness`) live and registry-driven. SD2 retention strengthened by Wave 8 Batch C FF.10 RegistrySpec consumer-collapse (every `encoding_name` consumer now reads from registry — historical entries actively-load-bearing rather than zero-runtime LazyLock).

- **SD3 — Per-commit scope-expansion-by-deletion is permitted.** Cycle 3 produced 14 disclosed SD3 expansions: Wave 7 Batch A (9 Rust + 14 Python compile-forced ctor migration sites across 13 files); Wave 7 Batch D (2 cfg-test + cfg-debug compile-forced); Wave 8 Batch A (16 Python explicit-import sites + 11 method-form Python call sites + 2 file deletions); Wave 8 Batch C (30 file touches across SelfPlayRunner config collapse + WireFormatSpec retirement + worker_loop callers); Wave 9 Batch A (4 test-fixture `schema_version` literal updates + 1 struct fixture field-add); Wave 10 Batch A (zero — `pub use` chain preservation + Rust auto-routing transparent to integration-test compile); Wave 11 (zero — pure-internal refactor + clippy attribute deletions). All disclosed in commit bodies, all reviewer-approved minimal. **Mechanism Lesson candidate L24 strengthening evidence: now observed across 3 consecutive cycles with consistent pattern.** Recommend formal Mechanism Lesson promotion at cycle 4 close.

- **SD4 — Implementer/reviewer corrections to audit MD take precedence.** Cycle 3 produced 25+ SD4 applications across 6 waves: Wave 7 Batches A-E (4 + 3 + 7 + 5 + 4 = 23 SD4 corrections — fundamental PREP framing errors caught at recon (anonymous closure destructure pattern per L27 candidate), shared-struct subset analysis errors (per L28 candidate), pure-file-move scope corrections (per L29 candidate)); Wave 8 Batches A-D (3 + 2 + 4 + 1 = 10 SD4 corrections); Wave 9 Batch A (4 SD4 corrections); Wave 10 Batches A+B (5 + 2 = 7 SD4 corrections); Wave 11 Batches A + B (1 + 3 SD4 corrections — Batch B 3 drifts: D1 INV25 substring assertion protection; D2 closure-vs-fn promotion forced by `#[allow]` retire; D3 F1 substantially over PREP target via §173 A5b discipline). **Mechanism Lesson candidate L25 strengthening evidence: now observed across 3 consecutive cycles.** Recommend formal Mechanism Lesson promotion at cycle 4 close.

- **SD5 — Bench baseline re-anchored at Wave close.** Cycle 3 re-anchored at Wave 7 close (`wave_7_close_baseline.txt`), Wave 8 close (`wave_8_close_baseline.txt`), Wave 10 close (`wave_10_close_baseline.txt`); Wave 9 SKIPPED bench (PREP §I) so Wave 8 close baseline carried forward; Wave 11 SKIPPED per-wave bench (PREP §E.1) so Wave 10 close baseline carried forward to cycle 3 close. **Phase 0 baseline at `audit/rust-engine/00_bench_baseline.txt` (HEAD `072d0db`) preserved unchanged** — canonical cross-cycle reference point. **New cycle 4 baseline at `audit/rust-engine/cycle_3/close/04_baseline_next_cycle.txt`** (HEAD `5e0c09d`, n=10 median + n=10 vast.ai mirror — laptop floor = `tightest_of(median − 2σ, 5th_percentile)`; vast.ai floor same formula). Cycle 4 baseline anchored 2026-05-18 at HEAD `5e0c09d` (laptop n=4 × n=10 internal median; vast.ai n=2 × n=10 internal mirror).

- **SD6 — Mechanism-absent bench WATCH on small-absolute deltas is measurement variance until 2-3 commits confirm.** SD6 exercised heavily across cycle 3. Wave 7 close: 4 WATCH metrics (MCTS sim/s +8.88%, NN latency b=1 −12.50%, Buffer push +5.04%, Buffer sample augmented −5.57%) — 2 mechanism-absent + 2 SD7-candidate-directionally-inconsistent. Wave 8 close: 5 WATCH metrics, ALL mechanism-absent + ALL bidirectionally-reverted (3 from Wave 7 + 2 new mechanism-absent reversions). Wave 9 SKIPPED bench. Wave 10 close: 3 WATCH metrics, 2 bidirectional reversions from Wave 8 (Buffer push + Buffer sample augmented) STRENGTHEN SD7 + 1 mechanism-absent variance candidate. Wave 11 SKIPPED per-wave bench. Cycle 3 strengthens SD6 + provides the 7-instance evidence base for SD7 promotion.

- **SD7 PROMOTED (NEW at cycle 3 close):** **Mechanism-absent WATCH bidirectional reversion = measurement variance.** Full SD7 statement + evidence base + discriminator rule in `audit/rust-engine/cycle_settled_decisions.md` (cycle 3 close adjudication; see ESCALATION ITEM below regarding file tracking). Discriminator: bidirectional reversion → variance (discount per SD7); monotonic with no code mechanism → flag for SD6 investigation; monotonic with code mechanism → real regression (revert/redesign). Datapoint base: 7 confirming bidirectional-reversion instances (5 cycle 2 + 2 cycle 3 Wave 7→8) + 5+ reversion-supporting datapoints across cycles 2+3. Anchors to SD6.

### Process patterns / Mechanism Lessons

Per Mechanism Lessons promotion convention (2+ confirming instances): L27 promotes at cycle 3 close per U13 default. L24/L25 strengthen across 3 cycles but await 4th-cycle confirmation. L28-L33 each have 1 instance at cycle 3 close; documented for future-cycle 2nd-instance candidacy.

- **L26 (cycle 2 close — `pub mod pyo3` shadow rule).** Preserved through cycle 3 — `engine/src/lib.rs` `use ::pyo3::prelude::*;` leading `::` discipline intact at every Wave 7-11 commit. Wave 11 does NOT touch `engine/src/lib.rs` or any `engine/src/pyo3/**` submodule (per `audit/rust-engine/cycle_3/wave_11/PREP_plan.md` §I.2 hard constraint).

- **L27 PROMOTED (NEW at cycle 3 close):** **Anonymous closure destructure pattern for thread-spawned worker bodies — PREP framing in fn-form should be verified against actual `thread::spawn(move \|\| {...})` closure context at recon-time.** Discriminator: when PREP wording says "rewrite worker fn signature" / "extract sub-fn", IMPL recon MUST verify whether the target is (a) a named `fn name(...)`, (b) an anonymous `thread::spawn(move \|\| {...})` closure with no fn name, or (c) a hybrid. The framing affects refactor mechanics: (a) supports straightforward fn-signature rewrite; (b) requires closure-destructure + struct-def rewrite (no fn signature exists); (c) needs case-by-case decomposition. Recon MUST run `rg 'fn <name>\('` to verify fn existence before adopting fn-framing language from PREP.

  **Datapoint base (3 instances):**
  1. **Wave 7 Batch C** (`54a60f4`, 1st instance) — closure destructure pattern for thread-spawned worker bodies; PREP §C wording said "rewrite worker fn signatures" but reality was `thread::spawn(move || {...})` anonymous closure inside `start_impl`; IMPL recast as closure destructure + struct-def rewrite (5 SD4 corrections including this framing).
  2. **Wave 8 Batch C** (`9f0f2dc`, 2nd instance) — `sym_tables_v6_default` migration; PREP wording about `worker_loop` migration prompted L27-pattern verification (migration happened BEFORE the `thread::spawn(move || {...})` in `start_impl` fn scope, but the pattern was reaffirmed via recon — confirmed L27-candidate via L32 candidate at Wave 8 close).
  3. **Wave 10 Batch A** (`f53975e`, 3rd instance) — `worker_loop` split preserves the destructure pattern verbatim at `inner.rs::run_worker_thread` entry, even after closure-to-named-fn extraction. The spawn-site at `mod.rs:174-184` wraps the named-fn call in `thread::spawn(move || { inner::run_worker_thread(...) })` — preserving the L27 closure-spawn pattern + the bool-flag destructure that INV25 Cell 3 pins via `include_str!` substring assertion.

  **Promoted per U13 default.** Operator action: cite L27 in any future-cycle prompt opening a P79-class refactor touching threaded-worker code OR proposing fn-extraction of body content inside a `thread::spawn(move || {...})` closure.

- **L28 (1 instance — Wave 7 Batch B sibling-struct fallback pattern).** When PREP claims "share a `<X>Params` struct", recon MUST field-set-intersect across all N consumer pairs before adopting shared-struct shape; sibling-struct shape is the SD4-correct fallback when intersection is empty. Wave 7 Batch B (`8315d15`) PREP §B literally said "share a `PushParams` struct" — actual analysis showed zero shared subset across the 3 facade↔impl pairs (3D vs 4D ndarray ranks; scalar-vs-array shapes; push_many omits game_id); sibling-struct shape (`PushSingleConfig` / `PushGameConfig` / `PushManyConfig`) was the SD4-correct fallback. **NOT PROMOTED — 1 instance.** Cycle 4+ candidate for 2nd-instance confirmation if a future P79-class refactor with multi-pair config-struct surfacing arises.

- **L29 (1 instance — Wave 7 Batch E pure-file-move locks `#[allow]` retirement).** `#[allow(clippy::too_many_lines)]` cannot retire via file move alone — the body's LOC count is preserved. Helper extraction is required to actually retire the attribute, and is out-of-scope for pure-move refactors. Wave 7 Batch E preserved 3 `#[allow(too_many_lines)]` lines across the 3 split fn bodies (`parse_one` 199, `validate` 178, `load_from_path_impl` 256). Operator pre-decision (PREP §M item 3): pure-file-move scope rejects helper extraction; SD4 documented "by design"; cycle 4+ revisit if structural changes to the registry parser or persist loader surface. **NOT PROMOTED — 1 instance.** Cycle 4+ candidate.

- **L30 (1 instance — Wave 8 Batch A type alias for cross-language @dataclass retire).** `EncodingSpec = engine.RegistrySpec` byte-identical to every consumer call site post-PyO3 `#[getter]` parity expansion, zero deprecation debt, sole source of truth at Rust side. Beats deprecation-stub class (future cleanup debt). Operator pre-decision (Wave 8 PREP §M item 3): alias direction = type alias, NOT re-import. **NOT PROMOTED — 1 instance.** Cycle 4+ candidate.

- **L31 (1 instance — Wave 8 Batch C comment-and-keep for bench-harness arms during cross-version compile).** When retiring a fallback arm referenced ONLY by bench harness (not production callers), comment-and-keep with explicit `// audit: bench-harness-only` block beats both PyValueError conversion (migration risk: bench harness lacks explicit `encoding_name` kwarg threading) and bench harness migration (broader scope outside FF.10 anchor). Wave 8 Batch C anchor: `engine/src/mcts/mod.rs:226` (`PyMCTSTree::run_simulations_cpu_only` constructs trees from `Board::new()` without registry_spec; None arm is bench-only). 10-line explanatory comment block above the arm cites why the path cannot reach production code. **NOT PROMOTED — 1 instance.** Cycle 4+ candidate.

- **L32 / L33 (1 instance each at cycle 3 close).**
  - **L32** (Wave 8 Batch C reaffirms L27 anonymous closure destructure) — folded into L27 promotion at cycle 3 close.
  - **L33** (Wave 9 Batch A schema field add as cleanest mechanism for cross-language consumer-cap defaults) — when operator needs a per-encoding default value flowing to Rust ctor + Python YAML defaults + PyO3 surface, a single `registry.toml` schema field add beats scattered Rust constants + per-consumer `or_else(...)` patches. K_max Option A (vs Option B runtime param vs Option C scattered constants) landed in 1 commit with 1 consumer auto-derive making the field load-bearing day-one + INV pin ensuring future operator tunes surface explicitly. **NOT PROMOTED — 1 instance.** Cycle 4+ candidate for 2nd-instance confirmation.

### Bench gate verdict

**PHASE 5 BENCH VERDICT: PASS.** 10 of 10 `all_targets_met` checkpoints GREEN at HEAD `5e0c09d` across n=4×10 laptop (40 samples) + n=2×10 vast.ai mirror (20 samples); SD7 bidirectional variance confirmed across cycles 2+3 (7 confirming + 5+ reversion-supporting datapoints); cycle 4 entry baseline anchored at `04_baseline_next_cycle.txt` per host floor `tightest_of(median − 2σ, 5th_percentile)`; 1 WATCH metric with bidirectional-and-mechanism-absent flagged as SD7 variance (worker pos/hr −9.95% vs Wave 10 ref, bimodal CUDA across n=4 runs — runs 1+2 cluster low at 26-28k / runs 3+4 cluster high at 30-31k). 2 STRENGTHEN deltas on buffer push/sample (mechanism-absent for Wave 11 — no buffer-push code path touched in cycle 3; gain attributed to CUDA/dispatch state).

Vast.ai v3 anomaly (forensic): first vast pass on non-canonical `/root/hexo_rl` workspace saw run 2 batch-fill collapse (99.40% → 56.26%) and worker pos/hr drop (84,556 → 67,069). DID NOT REPRODUCE on v4 canonical `/workspace/hexo_rl` (both runs 99%+ batch fill). Attribution: shared-host noisy-neighbor during v3 run 2. v4 is canonical; v3 archived at `bench_vast_*_v3_root_hexo_rl.txt` for forensic reference.

### INV pin status

| Pin | File | Tests | Status at HEAD `5e0c09d` |
|---|---|---|---|
| INV15 (v6w25 encode round-trip) | `engine/tests/inv15_v6w25_encode_roundtrip.rs` | 3 Rust | GREEN |
| INV16 (v8 has_pass_slot dispatch) | `engine/tests/inv16_v8_pass_slot_dispatch.rs` | 3 Rust | GREEN |
| INV17 Rust (PyRegistrySpec supersedes PyEncodingSpec) | `engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs` | 3 Rust | GREEN |
| INV17 Python (PyEncodingSpec retired) | `tests/test_inv17_pyregistryspec_retired.py` | 2 Python | GREEN |
| INV18 (window_center i32::midpoint negative-bbox truncate semantics) | `engine/tests/inv18_window_center_negative_bbox.rs` | 3 Rust | GREEN (cycle 3 Wave 6.5) |
| INV18b (cluster_center i32::midpoint negative-bbox truncate semantics) | `engine/tests/inv18b_cluster_center_negative_bbox.rs` | 2 Rust | GREEN (cycle 3 Wave 6.5) |
| INV19 (SelfPlayRunner config builder byte-equivalence) | `engine/tests/inv19_selfplayrunner_config_builder_byte_equivalence.rs` | 3 Rust | GREEN (cycle 3 Wave 7 Batch A) |
| INV20 Rust (ReplayBuffer push config field shape) | `engine/tests/inv20_replay_buffer_push_config_field_shape.rs` | 3 Rust | GREEN (cycle 3 Wave 7 Batch B) |
| INV20 Python (ReplayBuffer facade kwargs) | `tests/test_inv20_replay_buffer_facade_kwargs.py` | 2 Python | GREEN (cycle 3 Wave 7 Batch B) |
| INV21 (P68 module splits byte-identity) | `engine/tests/inv21_p68_module_splits_byte_identity.rs` | 3 Rust | GREEN (cycle 3 Wave 7 Batch E) |
| INV22 Python (PyO3 EncodingSpec parity post-FF.2) | `tests/test_inv22_python_encoding_spec_parity.py` | 17 Python parametrized | GREEN (cycle 3 Wave 8 Batch A) |
| INV23 (SelfPlayRunner encoding_name e2e post-FF.10) | `engine/tests/inv23_selfplayrunner_encoding_name_e2e.rs` | 4 Rust | GREEN (cycle 3 Wave 8 Batch C) |
| INV24 (K_max registry field discipline) | `engine/tests/inv24_k_max_registry_field.rs` | 3 Rust | GREEN (cycle 3 Wave 9 Batch A) |
| INV25 (worker_loop split byte-identity-on-behavior) | `engine/tests/inv25_worker_loop_split_byte_identity.rs` | 3 Rust | GREEN (cycle 3 Wave 10 Batch A; preserved Wave 11 Batch B per destructure pattern anchor) |

**33 Rust INV cells across 11 files + 17 Python INV22 parametrized cells + 2 Python INV17 cells + 2 Python INV20 cells.** Cycle 3 contribution: 10 new INV cells across 5 new pin files + INV22 Python parametrized addition + INV17 Python preserved. All GREEN at HEAD `5e0c09d`.

### Falsified Hypotheses Register additions

Cycle 3 PREP-vs-IMPL deltas where PREP framing was falsified at IMPL time:

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §180 (Wave 7 Batch A) | "SelfPlayRunner ctor has 40 params" | A SD4 #1 + REVIEW_A §3 verification | Actual 38 params (verified at `game_runner/mod.rs:196-235` pre-refactor); 2-param undercount from PREP. |
| §180 (Wave 7 Batch B) | "Share single `PushParams` struct across 3 facade↔impl pairs" | B SD4 #1 + B_recon §3 field-set-intersection | Zero shared subset across the 3 pairs (3D vs 4D ndarray rank; scalar-vs-array shapes; push_many omits game_id); sibling-struct shape (`PushSingleConfig` / `PushGameConfig` / `PushManyConfig`) was the SD4-correct fallback. L28 candidate. |
| §180 (Wave 7 Batch C) | "WorkerParams has 9 bools" | C SD4 #1 + REVIEW_C §3 phase 2 | Actual 7 bools (re-derived partition 3+2+2 across SearchFlags/ExplorationFlags/MoveConstraintFlags); 2-bool overcount from PREP. |
| §180 (Wave 7 Batch C) | "Rewrite worker fn signatures" | C SD4 #6 + REVIEW_C §3 phase 2 | No fn exists — worker body is anonymous `thread::spawn(move \|\| {...})` closure; IMPL recast as closure destructure + struct-def rewrite. L27 PROMOTED (3 instances total). |
| §180 (Wave 7 Batch E) | "Inner-helper extraction (9 `check_*` helpers + 4 header readers + payload reader) clears `#[allow(too_many_lines)]` at all 3 split sites" | E SD4 #1 + REVIEW_E §3 phase 3 | Pure-file-move-only per user constraint; `#[allow(too_many_lines)]` retained on all 3 monolithic fn bodies. L29 candidate. |
| §180 (Wave 8 Batch A) | "`hexo_rl.encoding.spec.EncodingSpec` @dataclass shape preserved by alias `from engine import RegistrySpec as EncodingSpec`" | Wave 8 §M item 3 decision + L30 candidate | Type alias (`EncodingSpec = engine.RegistrySpec`) is byte-identical and zero-deprecation-debt; circular-import vector blocked the re-import path. SD4 application. |
| §180 (Wave 8 Batch C) | "4 `audit: legacy-v6-fallback` arms" | PREP §L.4.4 pre-registered + Batch C SD4 disclosure | Actual 7 arms across 3 files (`inference_bridge.rs`, `game_runner/mod.rs`, `mcts/mod.rs`); 6 collapsed to `PyValueError`; 1 comment-and-keep at `mcts/mod.rs:226` bench-harness-only (L31 candidate). |
| §180 (Wave 9 Batch A) | "K_max field count: 8 schema_version line edits" | Wave 9 PREP §A.5 vs IMPL: 8 per-entry edits + 1 header doc-comment v3 block addition (5 new lines) | Header doc-comment v3 block addition was implicit but unstated; clean SD4 disclosure in commit body. |
| §180 (Wave 10 Batch A) | "PREP §A.3 11-arg `run_worker_thread` signature" | A SD4 #1.4 + REVIEW_A §8 CB7 | Arity 11 trips `clippy::too_many_arguments` default 7; bundling 5 per-worker scalars into `Copy` `WorkerGeometry` struct in `params.rs` reduces arity to 7 + preserves scalar-API discipline at fn entry per `feedback_registryspec_by_ref_in_hotpath.md`. SD4 application. |
| §180 (Wave 10 Batch A) | "PREP §K.2 `start_impl` ~50-60 LOC post-extraction" | A SD4 #1.5 + REVIEW_A §9 CB8 #2 | Actual 120 LOC > 100-line clippy threshold; extracted `build_worker_prototypes` private helper to drop `start_impl` under threshold without new suppression. SD4 application. |
| §180 (Wave 11 Batch A) | "F2 stays 15 (Batch A `:176` not F2-anchored)" | Batch A IMPL recon §6 anomaly | `:176` IS F2-anchored (`^\s*#\[allow` matches column-0 attribute via empty `\s*`); F2 drops 15 → 14 at Batch A close (PREP attributed −1 to Batch B; IMPL attributed to Batch A). Cosmetic accounting drift; net cycle-3-close F1/F2 deltas unaffected. |
| §180 (Wave 11 Batch B) | "PREP §B.4 keep `infer_and_expand` as closure" | B_recon §B.6 RECON OVERRIDE | To retire `#[allow(too_many_lines)]` at L52, parent body MUST fall under 100 LOC; cold/warm extractions alone yield ~440-505 residual; only by extracting `infer_and_expand` + `run_mcts_search` can parent fall under 100 LOC. `infer_and_expand` promoted to `#[inline] fn` (L31-hazard mitigation). |
| §180 (Wave 11 Batch B) | "PREP §D.2 F1 17 → 16 (delete `too_many_lines` at L52; no new allows)" | B_recon §B.7 + §B.12 Drift D3 | F1 17 → 29 (+12). Mechanism: 8 extracted sub-fns + 1 mirror-struct §173 A5b hot-path-by-value attributes; bundling into `Copy` structs would defeat scalar-API discipline. PREP target MISSED by +13; cycle 4 absorbs under "P79 hot-path discipline" anchor. |
| §180 (Wave 11 Batch B) | "PREP §B.4 5-sub-fn plan" | B_recon §B.12 4th drift | IMPL extracted 8 sub-fns (added `run_one_game` + `select_move` beyond PREP's 5); required to bring parent body under 100 LOC + split `play_one_move` (126 LOC clippy) under threshold. SD4 application. |

### Open items deferred to cycle 4

Consolidated from naming-inventory + `#[allow]` anchor map + per-wave open items + DEFER-bin proposals:

1. **8 hot-path-by-value `too_many_arguments` allows from Batch B** (§173 A5b cohort; PERMANENT KEEP under "P79 hot-path discipline" anchor per `06_allow_to_cycle4_anchor_map.md`). Cycle 4 retire predicate: (a) bundle-struct re-introduction validated by §173 A5b-style counter-bench showing bench-neutral, OR (b) PyO3 0.30+ kwarg-builder API surface change.

2. **PyO3 0.30+ kwarg-builder API** — close 7 P79 PyO3-surface allows (`SelfPlayRunnerConfig` struct + ::new; `ReplayBuffer::push/push_game/push_many`; `SelfPlayRunner` struct + ::new).

3. **SD4 too_many_lines sub-fn extraction follow-ups** — close 3 P68 allows at `engine/src/encoding/spec/validate.rs:31` + `engine/src/encoding/registry/parse.rs:17` + `engine/src/replay_buffer/persist/load.rs:27`. Cycle 2 Wave 7 Batch E PURE-FILE-MOVE rationale: SD4 documented "by design"; cycle 4+ revisit if structural changes surface.

4. **Q-§176 residual** — DEFERRED per U16 default (cycle 4+ candidate; orthogonal to refactor cycle).

5. **P24b/P24c HexTacToeNet decomposition** — operator-call defer to §177+.

6. **P70 train::seed_everything circular-import shim** — operator-call defer to §177+.

7. **Wave 7 C/E pytest count `1555 → 1558 → 1565` amendments** — NOT amended (history rewrite forbidden per CLAUDE.md discipline); informational only. Canonical convention going forward: `pytest tests/ -q` form (no marker filters).

8. **Wave 9 commit `4eefd53` Co-Authored-By trailer missing** — NOT amended (history rewrite forbidden); informational only. Wave 10 commits resumed the trailer.

9. **`#[allow]` framing standardization in commit bodies** — Wave 7+8 used cycle-2-anchor-cohort framing; Wave 11 adopts canonical F1 absolute (`git grep -c '#\[allow' -- engine/src/`) going forward. NOT amended at cycle 3 close; future-cycle commit-body convention.

10. **P69 `infer_and_expand` test target** — DEFERRED per Wave 10 U9 (Option B.2.c scaffold-only). Cycle 4 candidate now that `infer_and_expand` is top-level fn post-Wave-11 Batch B (no longer a closure; testable in isolation).

11. **Naming inventory final pass results** — see `audit/rust-engine/cycle_3/close/05_naming_inventory.md`. Cycle 3 close left **~30 architectural identifier sites + ~20 historical doc-comment refs** (vs cycle 2's 47 — substantial reduction from Wave 8 FF.2/FF.3/FF.10 cohort). 3 TBD candidates for cycle 4: (a) U3 `compute_v8_mask` polarity-flip rename (still gated on registry schema decision per cycle 2 U3); (b) `dataset_v6w25.py` + `dataset_v8.py` + `pretrain_legacy.py` filename consolidation under cycle 4 dataset-builder unification; (c) Python `_REPLAYERS` dispatcher dict + `_build_model_from_spec` collapse already done in Wave 8 Batch D; remaining cycle-4 cleanup is cosmetic.

12. **CRITICAL ESCALATION — `cycle_settled_decisions.md` tracking status:** see §"Critical escalation" below.

### Critical escalation — `cycle_settled_decisions.md` tracking status

`audit/rust-engine/cycle_settled_decisions.md` is **NOT in `.gitignore`** (verified via `git check-ignore` returning empty) BUT is **also NOT yet tracked by git** at cycle 3 close (verified via `git ls-files audit/rust-engine/cycle_settled_decisions.md` returning empty). The file is local-only on the operator's workstation.

**Implication:** the SD7 promotion entry added at cycle 3 close (per `cycle_settled_decisions.md` UPDATE deliverable D7) lands in a local-only file. Without operator action to `git add audit/rust-engine/cycle_settled_decisions.md` + commit + push, the SD record is NOT preserved cross-session — future cycle 4+ Phase 4 implementer/reviewer prompts will reference an SD7 entry that exists only on the cycle-3-close operator's workstation.

**Operator action required at cycle 3 close (post-AGGREGATION):**

1. Decide whether `audit/rust-engine/cycle_settled_decisions.md` SHOULD be tracked. Per cycle 2 §"Sprint-log handoff" wording ("At cycle close, the contents of this file migrate to the new sprint § entry in `docs/07_PHASE4_SPRINT_LOG.md`"), the file IS the cross-cycle authority for the SD record — its loss would be a hard regression.
2. If YES tracking: `git add audit/rust-engine/cycle_settled_decisions.md` + include in cycle-3-close merge commit.
3. If NO tracking: confirm that the §180 sprint-log entry (this file, eventually landed in `docs/07_PHASE4_SPRINT_LOG.md`) is sufficient SD record + accept loss of cycle-local intermediate-state forensic file.

**Recommendation:** **TRACK IT.** Cycle 1 + cycle 2 close patterns implicitly assume this file is the SoT for cross-cycle SD record; cycle 3 close discovers the file has never been git-tracked. Adding `audit/rust-engine/cycle_settled_decisions.md` to the merge commit at cycle 3 close future-proofs the SD record + provides cycle 4+ subagents an unambiguous canonical reference.

The 5 partially-tracked recon files under `audit/rust-engine/cycle_3/wave_{8,9,10}/` (`A_recon.md` / `B_recon.md` / `C_recon.md` / `D_recon.md`) suggest the `audit/` tree is eligible for tracking on a case-by-case basis; the broader `audit/rust-engine/cycle_2/` + `cycle_3/` tree is currently untracked. A cycle 4 PREP-time decision on `audit/` tracking policy is warranted.

### Archive reference

`reports/sprint_archive/§180_rust_engine_audit.md` (post-cycle action — preserves full audit-tree forensics for cross-cycle reference). Source-of-truth audit tree at `audit/rust-engine/cycle_3/` retained on operator workstation. Memory: pending — recommend `project_180_rust_engine_audit.md` covering cycle 3 outcome + SD7 promotion + L27 promotion + cycle 4 anchor handoff (PyO3 0.30+ surface; SD4 sub-fn extraction; §173 A5b discipline retire predicate; CRITICAL ESCALATION on cycle_settled_decisions.md tracking).

---

## §66–§101 Classification Audit — quick-look table

| Bucket | Sections | Compressed body location |
|---|---|---|
| KEEP-FULL → KEEP-DISTILLED | §66, §67, §69, §70, §71, §73, §74, §77, §80, §84, §85, §86, §88, §89, §90, §91, §97, §98, §99, §100, §101 | Body above |
| KEEP-FULL retained for L-rule origin | §47 (NaN guard), §58 (resume bugs), §59 (TT clear), §83 (quiescence_fire), §95 (chain ablation), §61/§62/§67 (Gumbel), §63/§64/§65 (dashboard metrics) | Part 1 narrative §1–§11 |
| KEEP-CONDENSED → INDEX-ONLY | §6, §11, §13, §14, §16, §17, §20, §22, §23, §24, §29, §30, §31, §38, §41–§46, §48, §50–§57, §68, §72, §75, §76, §78, §79, §81, §82, §83, §87, §92, §93, §94, §96 | § Index table above + Part 1 forward pointers |
| MERGED (torch.compile arc) | §3, §25, §30(torch), §32, §116, §123, §124, §125 | §116 / §124 / §125 entries above |
| MERGED (Gumbel arc) | §61, §62, §67, §74, §96, §104 | §67 + §74 + §96 + §104 entries above |
| MERGED (dashboard metrics) | §63, §64, §65, §82, §83, §104 | §82/§83/§104 entries above |
| MERGED (eval_interval / graduation) | §52, §60, §101, §101.a, §137 | §101 + §137 entries above |
| SUPERSEDED | §9 (→§66/§67/§74), §49 (→§59), §66 (→§67+§74+§96), §92 (→§97), §66 C1 (→§67) | Annotated inline |
| DELETE / archived | Test-count-only updates, §27b operational, §49 (superseded by §59), 2026-04-01 + 2026-04-02 stale bench tables | Git history retains |

---

## §102–§174 Classification — applied per spec

| § | Topic | Verdict applied |
|---|---|---|
| §102 | Bench rebaseline post-§97 | INDEX-ONLY → § Index row |
| §103 | Corpus zero-chain fix + baseline_puct playout-cap pin | KEEP-DISTILLED (above) |
| §104 | D-Gumbel / D-Zeroloss instrumentation | KEEP-DISTILLED |
| §105 | Q27 perspective-flip smoke W1 | KEEP-DISTILLED (superseded by §106) |
| §106–§111 | Q27 fixture artifact / post-W1 launches / Q33 diagnostic / Q33-B / Q33-C HALT | INDEX-ONLY → § Index rows |
| §112 | Q33-C2 E1 confirmed; Q33/Q37 RESOLVED | KEEP-DISTILLED |
| §113 | buffer_sample_raw recalibration 1500→1550 µs | INDEX-ONLY |
| §114 | bootstrap-v4 (L1, L15 origin) | KEEP-DISTILLED |
| §115 | CLAUDE.md split + skill scaffolding | INDEX-ONLY |
| §116 | D-ladder + torch.compile retry GO + §116.a revert | KEEP-DISTILLED |
| §117 | TF32 + channels_last per-host autodetect | KEEP-DISTILLED |
| §118 | Early-game forgetting fix wave (off-canonical axis) | KEEP-DISTILLED |
| §119 | Main-Island Neglect mechanism + RecentBuffer gap | KEEP-DISTILLED |
| §120 | RecentBuffer augmentation deployed; soft-abort at step 14000 | INDEX-ONLY |
| §121 | Directional bias resolves, clustering magnitude architectural | KEEP-DISTILLED |
| §122 | Phase 5 architectural redesign scoping | INDEX-ONLY (superseded by §165/§166 actual pivot) |
| §123 | Bench methodology fix: compile + InferenceServer threading | INDEX-ONLY |
| §124 | InferenceServer dispatch fix: TorchScript trace | KEEP-DISTILLED |
| §125 | EPYC 4080S sweep verdict + py-spy → perf_timing | KEEP-DISTILLED |
| §126 | Sweep harness migration (knob registry) | INDEX-ONLY |
| §127 | Top-K leaf cap eliminates MCTS pool overflow (L5 origin) | KEEP-DISTILLED |
| §128 | positions_generated metric (L6 origin) | KEEP-DISTILLED |
| §129–§130, §133–§136, §138 | Disk-budget / rotation port / D6 sym verify / v6 pretrain / W4 bench / W1+W2 audit / W4 Option C smoke | INDEX-ONLY |
| §131 | 18→8 plane migration P1+P2+P3 | KEEP-DISTILLED |
| §137 | W3 validation gates → Phase 4.0 UNBLOCKED | KEEP-DISTILLED |
| §141 | W4C policy-head diagnosis: locus is search/encoding | KEEP-DISTILLED |
| §142 | Encoding-window coverage audit: ply-31 pivot | KEEP-DISTILLED |
| §143–§145 | γ-knob audit / smoke v3 ABORT / v4 ABORT → α' | INDEX-ONLY |
| §146 | α' implementation: cap LEGAL_MOVE_RADIUS 8→5 | KEEP-DISTILLED |
| §147 | Bootstrap corpus contamination audit (L1, L15 origin) | KEEP-DISTILLED |
| §148 | Corpus rebuild v7 human-only Elo-weighted | KEEP-DISTILLED |
| §149 | v7 verification + v7e30 fine-tune promotes | KEEP-DISTILLED |
| §150 | v7full 30-epoch full retrain promotes (canonical anchor) | KEEP-DISTILLED |
| §151 | Numba @njit audit (NO-GO) | INDEX-ONLY |
| §152 | Phase B' instrumented smoke: Class-4 dominant | KEEP-DISTILLED |
| §155–§157 | Cosine-temp draw-collapse arc (L3, L9 origin) | MERGE entry (above) |
| §158–§163 | Hygiene wave + Refactor wave | MERGE entry (above) |
| §164 | Phase 5+ entry probe wave (P1/P2/P3) | KEEP-DISTILLED |
| §165 | v8 encoding migration design + spike wave | KEEP-DISTILLED |
| §166 | Phase A: encoding pipeline core (gated coexistence) | KEEP-DISTILLED |
| §167 | Phase B v8 variant exploration (B0–B4) | KEEP-DISTILLED |
| §168 | Eval harness generalization + v6w25 plumbing | KEEP-DISTILLED |
| §169 | Four-way encoder ablation (A1/A2/A3/A4 + §169a) | MERGE entry (above) |
| §170 | Six sub-passes (P0/P1/P3/P4 P1/P4 P2/P4 close) | MERGE entry (above) |
| §171 | P0/P1/P3 BLOCKED + A4 P2-reopen C DEAD | MERGE entry (above) |
| §172 | Encoding Registry SSoT (Phase A + Phase B v7full sustained) | KEEP-DISTILLED |
| §173 | α multi-window K-cluster selfplay (constants-parameterization) | KEEP-DISTILLED |
| §174 | v6w25 sustained: bootstrap investigation + escalation | KEEP-DISTILLED |
