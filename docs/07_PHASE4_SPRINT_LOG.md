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
| §S184 | Strategy δ: a sorted-`Vec` representation for `legal_moves_set` beats the `FxHashSet` rebuild | §S184 vast bench | −32.5% sims/s. The ring loop `push`es ~7× duplicate cells (overlapping radius balls); `sort_unstable`+`dedup` on the bloated array costs more than `FxHashSet`'s hash-with-inline-dedup insert. |
| §S185 | The residual ~44% `legal_moves_set` self-time is `cells.contains_key`-dominated (the §S184 post-mortem's interim inference) | §S185 laptop flamegraph | `FxHashSet::insert` 56.8% vs `contains_key` 27.7% — insert is dominant. δ's failure was a fix-design error, not a contains_key mechanism. |
| §S186 | Strategy β: incremental `legal_cov` delta maintenance amortizes below the once-per-leaf rebuild | §S186 vast + laptop bench | −49.5% sims/s. The delta runs once per descent *step* (`apply_move` ×depth + `undo_move` ×2·depth), not once per leaf — de-amortized to ~3× the rebuild's work, on the hot path. The residual `legal_moves_set` cost is a structural floor; see the "Perf-investigation arc" appendix. |

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
| L18 | A flamegraph shows where time is *spent*, not where it is *recoverable*. A tall profiler line is a question ("is this cost necessary?"), not proof of headroom. Before speccing a fix, confirm a genuinely cheaper algorithm *exists*. | §S180–§S186 perf arc |

---

## Perf-investigation arc §S180→§S186 — CLOSED (do not re-litigate)

**Closed 2026-05-22.** The `legal_moves_set` hot line (~44% whole-program
self-time — §S185 laptop flamegraph; 41.8% in the §S182-era `perf report`)
drew the four-strategy menu of `investigation/rust-perf-2026-05-20/09_*`.
**Merged wins:** §S182 capacity-`reserve` fix **+66.4% sims/s** vast (killed
the hashbrown rehash cascade), §S183 micro-opt bundle **+1.12%** (sqrt hoist
+ `mul_add` FMA + relaxed atomics). **Both rewrites of the residual FAILED:**
§S184 δ (sorted-`Vec` representation) **−32.5%**, §S186 β (incremental `u16`
coverage map) **−49.5%** — δ traded the representation for a worse one, β
de-amortized the once-per-leaf rebuild onto every descent step. α and γ were
rejected by design. **Verdict: the residual ~44% `legal_moves_set` self-time
is a structural floor, not headroom** — the once-per-leaf `FxHashSet` rebuild
is already the cheap way to produce a leaf's legal-move set. **Do NOT
re-spec a `legal_moves_set` optimization without NEW mechanism evidence** (a
concretely cheaper algorithm, not a tall profiler line — see L18). Full
record: `investigation/rust-perf-2026-05-20/` plans `09`/`13` + post-mortems
`11`/`16`; sprint entries §S182/§S183/§S184/§S186.

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
| Q-§S181-structural | Config-invisible capture channel | ✅ Resolved (diagnosis axis) §S181 — training-loop value-head discrimination collapse; resolution path FU-1/FU-2 open |
| Q-§S181-value-head-arch | Does removing coverage-blind `v_max` pool prevent value-head collapse? | 🔴 Active — HIGH; §S181-T2; folded into FU-2 |
| Q-§S181-probe-redesign | Do MCTS-in-loop probes catch colony capture C1–C4 miss? | 🔴 Active — HIGH; §S181-T4; PR-A first |

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

## §177 — v6 sustained from step-20K anchor: closed by recipe-attractor reproduction (2026-05-18)

**Branch:** `phase4.5/m176a_v7mw` (continuation), tmux session `s177` on vast 5080.
**Anchor:** `bootstrap_model_v6_step20k.pt` (SHA `297e0ce0…2bce6a`, §175 step-20K promotion).
**Launch commits:** `166ac7c` (variant `v6_sustained_s177.yaml`), `d70507a` (Wave A1 baseline doc), `072d0db` (vast checkup script).
**Reference §177-pre:** lines 1428–1470.

**Closing trajectory (vast, per §178 pre-design investigation):** SealBot WR 2→0% across step 10K→40K, mirroring §175 18→4% trajectory in different anchor. Combined with §175, falsifies the residual "anchor weight is the lever" hypothesis: same recipe + different anchors → same colony-attractor capture.

**Closed by interrupt** at the point §178 pre-design investigation gathered sufficient evidence to motivate the bot-mix + ply-cap-value mechanism intervention. No further sustained running was useful.

### L24/L25/L26 mechanism-lesson candidates (pending operator confirm — defer to L24+ register if accepted)

| ID | Candidate lesson | Evidence | Implication |
|---|---|---|---|
| L24 | **Recipe-dependent colony attractor, not anchor-dependent.** Same §175 recipe captures the colony attractor regardless of bootstrap anchor (v6.pt clean AND v6_step20k.pt continuation both ride the attractor). | §175 18→4% + §177 2→0% across two anchors; combined PASS in `reports/s178_pre_design_investigation.md` SA-A | Refines L18 (anchor-mistake signal coupling) — anchor swap is NOT a lever; recipe/objective intervention required. §178 trials this. |
| L25 | **G4 value-head band FAIL concurrent with colony capture.** §175 + §177 both showed `value_fc2_weight_abs_max` drift outside `[0.154, 0.462]` coincident with sealbot WR collapse. | §175 stages probe + §177-pre L21/L22 dual-mode | Value head flattening tracks colony entrenchment; G4 is the upstream WR-collapse predictor. |
| L26 | **Ply-cap truncation outcome = organic draw outcome silently dilutes finish-pressure on long colony-prone games.** Pre-§178 Rust path writes `outcome = draw_reward` for both winner=None && ply≥max AND winner=None && legal_count=0. Only `terminal_reason` metadata distinguishes — and value-head never sees it. | §178 investigation SA-C VC-2; §178 T2 resolves via `ply_cap_value` split | Operator pre-commit `draw_value -0.5→-0.1` alone REMOVES finish-pressure; `ply_cap_value` split (-0.5 vs -0.1 in §178 — operator dialed back from design -0.8) restores it. |

### Falsified Hypotheses Register row added

| § | Hypothesis | Evidence FAILing | Closer |
|---|---|---|---|
| §177 | Step-20K anchor escapes the colony attractor under §175 recipe | §177 2→0% across 10K→40K reproduces §175 18→4% on different anchor | §178 launch — same recipe, different mechanism (bot-mix + ply-cap split) |

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

*NOTE: This §178 = Rust engine refactor cycle 1 close. See also the §S178+ scheme (Sustained Training Sprint, post-§180) for v6 sustained run + bot-corpus mixing — the prefix `S` disambiguates the two numbering schemes that coexist permanently.*

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

*NOTE: This §179 = Rust engine refactor cycle 2 close. See also the §S179+ scheme (Sustained Training Sprint, post-§180) when those entries land — prefix `S` disambiguates.*

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

*NOTE: This §180 = Rust engine refactor cycle 3 close. Sustained Training Sprint numbering (§S178+) starts AFTER this entry to avoid confusion. Always cite the prefix (§S…) or the branch (e.g. `phase4.5/s178_botmix`) when referring to a sustained-training sprint.*

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

## §S178 — v6 sustained bot-mix recipe launch (2026-05-18)

*DISCRIMINATOR: This is §S178 = Sustained Training Sprint 178. NOT to be confused with §178 (line 1697) which is the Rust engine refactor cycle 1 close. The two numbering schemes coexist permanently post-cycle-3. Always cite the branch (`phase4.5/s178_botmix`) or the prefix (§S178) for disambiguation.*

**Branch:** `phase4.5/s178_botmix` (off master `ddfa42e`); **anchor:** `checkpoints/bootstrap_model_v6.pt` (clean v6 base bootstrap; §175 anchor preserved for direct A/B vs §175 + §177).
**Design:** `docs/designs/S178_design.md` (commit `b26999b`).
**Investigation:** `reports/s178_pre_design_investigation.md` (gitignored, vast-only).
**Implementation:** 9 commits `b26999b..22597fc` on `phase4.5/s178_botmix` (design + T2 Rust split + INV26 + T3 Python wire + T1 bot corpus generator + T4/T5/T7 training-path + T6 yaml).

### Two mechanism levers vs §175/§177 colony-attractor capture

1. **SealBot-vs-anchor bot corpus pool** at `bot_batch_share=0.15`. Top-level batch slot (NOT inside selfplay decay, NOT subject to LRU eviction). Pre-generated via `make corpus.bot ANCHOR=checkpoints/bootstrap_model_v6.pt N_GAMES=700 OUT=data/bot_corpus_s178_sealbot_vs_v6.npz`. KrakenBot DROPPED (operator call, supported by Wave C BT data); bot-vs-bot games SKIPPED (no anchor-mistake signal). Bootstrap corpus UNCHANGED (human-only per §148).
2. **`ply_cap_value` split from `draw_reward`.** Rust `finalize_game` outcome branch now distinguishes terminal_reason==2 (ply-cap, → `ply_cap_value`) from winner=None,ply<max (→ `draw_reward`). Operator pre-commit `draw_value: -0.5 → -0.1`. Operator override on `ply_cap_value`: design called for `-0.8` (BCE 0.10, near-loss); operator dialed back to `-0.5` (BCE 0.25, soft-penalty) pending §178 outcome — may revisit.

### Pre-registered V178-1..8 verdicts (design §7)

| ID | Hypothesis | PASS criterion | FAIL criterion | NULL criterion |
|---|---|---|---|---|
| V178-1 | SealBot WR @ step 10K beats §177 step-10K (2%) | ≥5%, n=100 | ≤2% | 2–5% inconclusive |
| V178-2 | SealBot WR @ step 30K > 0% | ≥1 win / 100 | 0/100 | — |
| V178-3 | colony_fraction in bootstrap_anchor wins @ step 30K below §177 (~64%) | ≤50% | ≥70% | 50–70% |
| V178-4 | Non-monotonically-declining SealBot WR (no strict-decline trajectory) | observed | strict monotone decline | — |
| V178-5 | G4 value_fc2_weight_abs_max in [0.154, 0.462] by step 20K | in-band | below-band | — |
| V178-6 | Bot-pool policy-loss > 0.5 nat through step 30K (corpus dominance does NOT compete it down) | observed | bot ≈ corpus loss | — |
| V178-7 | draw_rate + ply_cap_rate both stable/down | both | either ≥5pp rise | — |
| V178-8 | v_pred(ply≥140) ∈ [-0.9, -0.7] by step 15K (ply-cap-value penalty learned) | in-band | drifts toward 0 | — |

### Risk register pointer

`docs/designs/S178_design.md` §8 — 12 rows, operator-confirmed pre-launch. Hard-abort thresholds preserved (§157/L9): grad_norm 10.0, stride5_p90 60, row_max_p90 50, colony_ext_frac_max 0.40. NEW soft-abort: `pct_at_cap_warn 0.30` (≥5pp rise without WR recovery).

### Operator overrides this sprint

- `ply_cap_value: -0.8 → -0.5` (variant yaml + training.yaml default). Design value retained in `docs/designs/S178_design.md` body for traceability; revisit if §178 outcome insufficient.
- §S178 discriminator on sprint log heading: sprint log already uses §178/§179/§180 for Rust engine refactor cycles 1/2/3 (lines 1697/1764/1921). Branch + design + verdict IDs all keep "s178"/"S178" branding.

### Pre-launch hygiene (operator, vast 5080)

1. `make bench` on vast 5080 to confirm no regression vs cycle-3-close baseline at `audit/rust-engine/cycle_3/close/04_baseline_next_cycle.txt`. Local laptop bench-gate SKIPPED (no AC; design §11). Touch points: T2 `finalize_game` + T4 `assemble_mixed_batch` in-place copyto.
2. `tmux kill-session -t s177` (if still alive on vast).
3. `data/bootstrap_corpus_v6.npz` present on vast (regen or scp; SA-E vast-only finding).
4. `rm checkpoints/replay_buffer.bin` (clean start; §177 buffer discarded).
5. `make corpus.bot ANCHOR=checkpoints/bootstrap_model_v6.pt N_GAMES=700 OUT=data/bot_corpus_s178_sealbot_vs_v6.npz` to generate bot corpus.
6. Launch: `python scripts/train.py --checkpoint checkpoints/bootstrap_model_v6.pt --variant v6_botmix_s178 --iterations 100000`.

### Forward pointers

- §179 candidate: flip `bot_corpus_refresh.enabled: true` if §178 colony_frac trajectory shows aging (currently hook present + disabled, warning-only log when triggered).
- INV26 pins ply_cap_value-distinct outcome path; INV19 byte-equivalence extends 38→39 atomically with T2.

### Pending post-launch follow-ups (for future sessions)

Pre-launch branch extension (T11/T12/T13, 2026-05-18; on `phase4.5/s178_botmix`):
- **F-fix-1** threat-target colony fix landed at **T11** (commit `1aa0c8f`): `find_winning_line` now scans all stones via fallback so the threat-head target is non-empty when `winner` is set via the `player_wins` all-stones fallback (HTT 2-moves-per-turn off-line second-move case). INV27 Rust + Python parametrized GREEN.
- **ply_cap_value revised** to `0.0` at **T12** (commit `1b47eb1`): literature-canonical per AlphaZero/KataGo (was `-0.5` = no-op vs draw_value). Variant only; training.yaml default kept at `-0.5` for back-compat.
- **T13** (this commit): sprint-log cross-pointers + this follow-up block.

Open §S179 candidates if §S178 colony reduction is modest:
- (a) `completed_q_values: false` A/B at step 30K-50K (visit-count CE policy target).
- (b) Policy Surprise Weighting / KL-weighted buffer writes (Q9; KataGo confirmed major efficiency win).
- (c) Soft policy target T=4 with 8x weight (KataGo b18c384 innovation).
- (d) `bot_batch_share` → 0.20.
- (e) `game_length_weights` neutralization (H4).

Open hygiene items (NOT in §S178 scope; defer to follow-up commits on master):
- **H1** — `bootstrap_model.pt` cleanup (rename/delete + repoint `Makefile` / `anchor.py` / `opponent_runners`; provenance per `reports/bootstrap_model_pt_provenance.md`).
- **H2** — `data/bootstrap_corpus_v6.npz` on vast (regen on vast if absent; can run before bench gate).
- **H3** — §177 vast tmux liveness (SSH check; kill if alive before bot-corpus generation).
- **H4** — `game_length_weights` colony-bias (§S179 candidate, not pre-launch).
- **H5** — `Makefile` BOOTSTRAP default → explicit error (separate hygiene commit).

Source B live cross-bot subprocess infra design deferred to §179+.
Length-discounted value target DROPPED from candidate pool per operator (terminal-reward purity principle).

---

## §S178a — Tier-1 hygiene wave (mid-§S178) — 2026-05-18

Five-item parallel hygiene wave on branch `phase4.5/tier1_hygiene` (off master
`96337c4`); §S178 sustained run on vast.ai untouched throughout (tmux `sS178`
LIVE; active anchor `bootstrap_model_v6.pt` + active bot corpus
`bot_corpus_s178_sealbot_vs_v6.npz` + active variant `v6_botmix_s178.yaml` all
untouched in every diff). Wave-driver:
`reports/tier1_hygiene_wave.md` (this aggregation report).

**A — quarantine fresh-init `bootstrap_model.pt`** (commit `c1173a8`).
Closes §S178 open hygiene **H1**. The unknown-provenance v6w25-architecture
file at `checkpoints/bootstrap_model.pt` was the silent `Makefile:11` default
when `BOOTSTRAP=` was unset; per `reports/bootstrap_model_pt_provenance.md`
§3+§5 it carries Kaiming-uniform fresh-init weights with 0/143 tensor match
vs every sibling — not a trained anchor. Repointed canonical defaults
(`Makefile:11`+`:47`, `hexo_rl/training/anchor.py`
`_BOOTSTRAP_ANCHOR_CANDIDATES`, `hexo_rl/eval/opponent_runners.py:217`
`bootstrap_anchor`) to `bootstrap_model_v6.pt`; dropped the row from
`scripts/migrations/2026_05_12_checkpoint_manifest.json`. Forensic copy
preserved at `checkpoints/archive_quarantine/bootstrap_model_random_init_v6w25.pt`
(gitignored; SHA `d00b8604…1586253`). 4 residual references outside the
contracted touch list deferred to follow-up (Q-§S178a F-A1).

**B — `tests/test_scraper.py` restore** (NO-OP, no commit).
The tracked-path test does not exist anywhere in reachable git history
(neither `7aea774` nor `913c5a0` touched a `tests/test_scraper.py`). A
gitignored local copy DOES exist on this worktree, imports
`hexo_rl.bootstrap.scraper` cleanly, and runs 19/19 PASS in venv — there
is no broken `elo_band_key` import to fix locally. The wave-driver
premise was sourced from a vast-side bench-gate report describing a
different stale gitignored file (Q-§S178a F-B1 follow-up).

**C — `scripts/verify_anchor.py` anchor verification CLI** (commit
`2740766`; 254 LOC new file). Reports `sha256` (16 hex), `format`,
`key_count`, `param_count`, `head_shapes`, `fresh_init_signature` (bool),
`verdict` (TRAINED | FRESH_INIT_SUSPECT | UNKNOWN). Heuristic per
provenance report §3: `value_fc2.weight.abs().max() / sqrt(1/fan_in)`
within `[0.8, 1.25]` → suspect. Sidecar JSON at `<ckpt>.verify.json`.
Exit codes 0 / 1 / 2. Verified end-to-end against the active anchor
(exit 0 TRAINED, ratio 2.013) and the quarantined fresh-init file
(exit 1 SUSPECT, ratio 0.999).

**D — CLAUDE.md split** (NO-OP, no commit). `wc -l CLAUDE.md` = 105 ≤
200 target; all 8 `docs/rules/*.md` topic files exist; CLAUDE.md is
index-shaped (Prime Directive + threat-probe gate + encoding registry +
rule files index + deep-dive index + MCP tools) with no fat content to
extract.

**E — variant config hygiene audit + cleanup** (commit `ba4f1e7`).
Audited 7 non-active variants (`v6_botmix_s178.yaml` SKIPPED, active
§S178). Cleaned `v6_sustained.yaml` (§175 closed-by-interrupt) and
`v7mw_sustained.yaml` (§176a experimental v7mw) — dropped 11/12
base-equal scalars each. Audit report at
`reports/variant_hygiene_audit.md` (306 LOC) covers per-variant
noise/override/extension tables. Deferred audit-only:
`m173_alpha_cold_smoke.yaml`, `smoke_radius_curriculum.yaml`,
`_sweep_template.yaml`, `v6_sustained_s177.yaml` (§S178 contrast),
`vast.yaml` (operator-designated exemplar). All 22+ removed keys
spot-checked base-equal; `tests/test_variant_configs.py` 5/5 PASS;
`#[allow` count = 29 (cycle-3 close baseline preserved). Follow-ups
Q-§S178a F-E1, F-E2.

**Verification.** Cherry-pick order A → C → E on
`phase4.5/tier1_hygiene` (B + D no-op). 5 independent REVIEW subagents
(no IMPL context, fresh per-agent) each PASS. `make test` exit 0 on
branch HEAD: 1588 passed, 21 skipped, 1 xpassed, in 155.74s. No
hot-path touch; no Rust `#[allow]` regression; active-run paths
confirmed untouched in every diff. Operator FF-merges to master
post-inspection (no tag — hygiene wave, not cycle close).

**Follow-up entries:** `docs/06_OPEN_QUESTIONS.md` — Q-§S178a row
(LOW/MED priority items absorbed during normal §S178+/§S179 work).

---

## §S179 — §S178 mechanism CLOSE: bot-mix + ply_cap split insufficient

*DISCRIMINATOR: §S179 = Sustained Training Sprint 179 (this entry). NOT §179
(line 1921) = Rust engine refactor cycle 3 close. Cite the run-id or `§S179`
prefix to disambiguate.*

**Status:** FAILED. Colony attractor reproduced. Same dead-end as §175.

**Run identity:**
- Branch at launch: master `98296f9` (post-§S178a + §S179a/§S179b/§S179c +
  F-A1 merges; `98296f9` = cadence-fix `eval_interval 5K→10K`).
- Anchor: `checkpoints/bootstrap_model_v6.pt` (SHA
  `7ab77d2cb091e3a67a0900e8c312f11fd7f9e87c8ea31cdd27102b9298372103`).
- Bot corpus: `data/bot_corpus_s178_sealbot_vs_v6.npz` (700g, static).
- Variant: `configs/variants/v6_botmix_s178.yaml` (post-cadence-fix:
  `eval_interval` 10K, SealBot stride 1 n=100; `ply_cap_value` 0.0 per §S178
  T12; `draw_value` -0.1).
- Run-id: `243e321f76504c6d908ab2f64eef8100`; tmux `sS179` on vast (5080).
- Launched 2026-05-18 17:36 UTC; SIGINT-stopped 2026-05-20 07:04 UTC at
  step **62,740** (31,370 games, ~37.4 h elapsed). Clean exit — final
  checkpoint + buffer flushed, `session_end` written. Last eval @ step 60K
  (`eval_interval` 10K); steps 60K→62.7K trained without further eval.

**Eval trajectory** (SealBot/anchor n=100, greedy-bot n=20; colony =
colony-formation wins ÷ player wins):

| Step | wr_sealbot | wr_anchor | wr_best | colony@anchor | colony@best | colony@sealbot | Elo | Promoted |
|---|---|---|---|---|---|---|---|---|
| 10K | 8% | 59% | 55% | 49% | 62% | 13% | 410 | ❌ |
| 20K | 11% | 68% | 66% | 79% | 85% | 91% | 414 | ✅ peak |
| 30K | 12% | 64% | 49% | 70% | 80% | 83% | 302 | ❌ |
| 40K | 2% | 66% | 60% | 85% | 88% | 100% | 266 | ✅ |
| 50K | 2% | 70% | 63% | 83% | 89% | 100% | 224 | ✅ |
| 60K | 4% | 75% | 45% | 77% | 93% | 100% | 224 | ❌ |

**Diagnostic.**

SealBot WR peaked 12% @ step 30K, crashed to 2–4% by 40K–60K. Peak Elo
promotion @ step 20K (Elo 414). Anchor WR climbed monotonically 59→75 over
the same span. colony@sealbot pinned 100% the last 3 rounds (every SealBot
win is a colony-formation win). Canonical anchor↑/sealbot↓ divergence — the
§155 T2 / §175 colony-capture signature.

Threat probes did not trigger the C1–C3 kill criterion through step 60K
(run ran uninterrupted to 62.7K) — the threat circuit is intact; the failure
is policy-distribution-level, not threat-representation-level. L22 confirmed
again: the sampled policy diverts into colony patterns despite correct
threat representation.

§S178 mechanism (`bot_batch_share=0.15` SealBot-vs-v6 corpus +
`ply_cap_value=0.0` + `draw_value=-0.1` + cosine-OFF + F-fix-1 threat-target
colony fix) bought ~one extra promotion vs §175 trajectory (§175 peak 17% @
15.5K; §S179 peak 12% @ 30K) but did NOT escape the attractor. The
anti-corrective force decomposition in `docs/designs/S178_design.md §3.1`
predicted 0.82:1 (DIRECT-corrective vs recent-selfplay-colony) = BORDERLINE.
Borderline lost.

**Falsified.** §S178 design hypothesis H-S178-1: "`bot_batch_share=0.15`
with SealBot-vs-v6 corpus + `ply_cap_value=0.0` + cosine-OFF is a sufficient
anti-colony lever for stable training to step ≥50K."

### Falsified Hypotheses Register addition

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S179 (H-S178-1) | `bot_batch_share=0.15` SealBot-vs-v6 corpus + `ply_cap_value=0.0` + cosine-OFF is a sufficient anti-colony lever for stable training to step ≥50K | §S179 eval trajectory (close 2026-05-20) | SealBot WR 8→11→12→2→2→4; anchor 59→75; colony@sealbot pinned 100% from step 40K. Mechanism buys ~1 extra promotion vs §175 but does not escape the colony attractor. Borderline 0.82:1 corrective-force decomposition (S178_design §3.1) lost. |

**Archive.** `archive/s179_recipe_fail/` on vast — 8 eval-aligned/peak/final
checkpoints (`ckpt_step{10,20-peak,30,40,50,60}k.pt`, `ckpt_final_step62740.pt`,
`best_model_final.pt`) + `eval_db.sqlite` + `metadata.json` + `training_tail.log`.
Replay buffer (2.9 GB, colony-saturated) + dense intermediate checkpoints
deleted post-archive-verify (low forensic value). `best_model.pt` reflects
the last promotion (step 50K) — the step-20K peak is preserved separately
as `ckpt_step20k_peak.pt`.

**Successor.** §S180a launching: §S179 recipe + single config flip
`completed_q_values: false` (pre-registered §S179-candidate (a) — visit-count
CE policy target). Tests CQV as a colony-attractor amplifier. Single
isolated variable.

### Process patterns / Mechanism Lessons

L-numbering: L1–L17 promoted in the register table above; L18–L33 are
§-local candidates (latest L33 = §180 Wave 9). §S179 adds L34/L35.

- **L34 (anchor↑/sealbot↓ divergence = canonical colony-capture signature
  — 3rd confirming instance; promotable).** The calibration rule
  "`wr_bootstrap_anchor` drop below 50–55% = regression" is
  necessary-but-not-sufficient: it assumes the anchor is a colony-resistant
  reference. The v6 anchor SHARES the colony weakness. Rising `wr_anchor`
  with falling `wr_sealbot` is the textbook colony-capture pattern — the
  model improves at exploiting v6's colony weakness while losing the ability
  to play threat hex. Always cross-check the anchor trajectory against a
  colony-resistant opponent (SealBot). Confirming instances: §155 T2, §175,
  §S179. Calibration memo
  (`feedback_alphazero_sustained_eval_calibration`) amended accordingly.

- **L35 (§S178 mechanism — bot-mix 0.15 + ply_cap split + cosine-OFF — is an
  insufficient anti-colony lever; 1 instance).** The borderline
  anti-corrective force decomposition (0.82:1 DIRECT vs colony reinforcement)
  predicted at design time was confirmed insufficient empirically. The
  mechanism buys ~one extra promotion vs the §175 baseline but does not
  escape the attractor. Future mechanism interventions must target a ≥2:1
  corrective-force ratio OR isolate the amplification mechanism
  (CQV / `game_length_weights` / value-head pretrain bias) for direct
  nullification. Cycle 4+ candidate for 2nd-instance confirmation.

---

## §S180a — CQV-flip A/B CLOSE: CQV not the colony lever

*DISCRIMINATOR: §S180a = Sustained Training Sprint 180a (this entry). Cite
the run-id or `§S180a` prefix to disambiguate from §180 (Rust engine refactor
cycle 3).*

**Status:** FAILED. Different signature than §S179 — not colony capture,
weaker learning signal. CQV-flip RULED OUT as colony lever via single-knob
A/B vs §S179.

**Run identity:**
- Branch at launch: master `6f08042`
- Anchor: `bootstrap_model_v6.pt` (SHA `7ab77d2c…372103`)
- Variant: `configs/variants/v6_botmix_s180a_cqv_off.yaml`
- Single-knob delta vs §S179: `completed_q_values: true → false`
- Run-id: `e68e79a53793421a886611c625f9c802`
- tmux: `sS180a` on vast (5080)
- Launched 2026-05-20 07:20 UTC; SIGINT-stopped 2026-05-20 20:52 UTC at
  step **22,624** (11,312 games, ~13.5 h elapsed). Clean exit — final
  checkpoint flushed, `session_end` written. Killed after V180a-2 FAIL
  @ step 20K.

**Eval comparison (single-knob A/B):**

| @step | metric | §S179 (CQV true) | §S180a (CQV false) | delta |
|---|---|---|---|---|
| 10K | wr_sealbot | 8% | 8% | 0 |
| 10K | wr_anchor | 59% | 58% | -1 |
| 20K | wr_sealbot | 11% | 7% | -4 |
| 20K | wr_anchor | 68% | 53% | -15 |
| 20K | wr_best | 66% | 48% | -18 |

**Diagnostic.**

§S179 = colony capture (anchor↑ + sealbot↑ then crash). §S180a = not
learning (anchor↓ + sealbot↓ + wr_best <50% at step 20K, weaker than own
step-10K ckpt). Visit-count CE policy target produces weaker gradient than
CQV without escaping the colony attractor. CQV is NOT the colony amplifier
— it gave better learning, but learning the same wrong thing.

Threat probes PASS throughout both runs — circuit health independent of
colony failure mode (L22 reconfirmed).

**Falsified.** Hypothesis H-S180a-1: "`completed_q_values: false` produces
more diverse policy target, escaping colony attractor." Add to Falsified
Hypotheses Register.

### Falsified Hypotheses Register addition

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S180a (H-S180a-1) | `completed_q_values: false` produces a more diverse policy target, escaping the colony attractor | §S180a eval trajectory (close 2026-05-20) | Visit-count CE = uniformly weaker metrics at step 20K (wr_sealbot -4pp, wr_anchor -15pp, wr_best -18pp vs §S179). Not colony capture — slower learning of the same trapped state. CQV ruled out as colony lever. |

**Archive.** `archive/s180a_cqv_off_fail/` on vast — 3 eval-aligned/final
checkpoints (`ckpt_step{10,20}k.pt`, `ckpt_final_step22624.pt`) +
`best_model_final.pt` + `eval_rounds_s180a.json` (12 `evaluation_*` events
extracted from train jsonl; `results.db` has no `eval_rounds` table) +
`metadata.json` + `training_tail.jsonl`. Replay buffer (2.9 GB) + dense
intermediate checkpoints deleted post-archive-verify.

**Successor.** §S180b — 3-knob escalation targeting direct anti-colony force:
1. `completed_q_values: true` (restore — stronger gradient confirmed)
2. `bot_batch_share: 0.15 → 0.30` (double direct anti-colony signal;
   per `docs/designs/S178_design.md §3.1`, raises DIRECT:colony ratio from
   0.82:1 to ~1.64:1)
3. `game_length_weights` neutralize (uniform 1.0/1.0/1.0 — kill colony
   upweighting in selfplay slice; Q-§S179-residual confirmed lever)

Multi-knob delta justified: §S179 + §S180a establish 2 baseline arms;
§S180b tests combined direct-force escalation. If §S180b PASS, follow-up
ablation isolates. If §S180b FAIL, surface is dead — escalate to §S181
with code-level levers (PSW or refresh hook).

### Process patterns / Mechanism Lessons

§S180a adds L36/L37.

- **L36 (single-knob A/B discipline retired when suspect-set unranked).**
  §S180a single-knob A/B isolated CQV cleanly = ruled out. But 4 remaining
  candidate levers (`bot_batch_share`, `game_length_weights`, PSW, refresh
  hook) have no quantitative ranking from §S179/§S180a data. Single-knob
  discipline on 4 unranked suspects = 4 × ~30h = 120h. Pragmatic shift:
  combine cheapest 2-3 unfired levers when all target the same mechanism.
  §S180b applies this.

- **L37 (visit-count CE = weaker gradient than CQV in colony-rich regime).**
  Empirical finding worth documenting: with bot-mix corpus + ply_cap split +
  cosine-OFF, switching from CQV to visit-count CE produced uniformly weaker
  metrics at step 20K (wr_sealbot -4pp, wr_anchor -15pp, wr_best -18pp).
  Possible mechanism: in the colony regime, the MCTS visit distribution is
  diffuse (low value-head signal), so the visit-count CE target = high-
  entropy near-uniform. CQV reweighting concentrates the target on
  high-value children = stronger learning signal despite the colony bias.
  Future variants with visit-count CE should pair with a value-head signal
  restoration mechanism.

### perf/legal-moves-cache-cap — CANDIDATE branch (merge held)

Rust-perf wave (`investigation/rust-perf-2026-05-20/`): `legal_moves_set`
pre-reserve fix. Branch `perf/legal-moves-cache-cap` HEAD `f8ff7b8` (off
master `3146144`), tag `perf-legal-moves-cache-cap-candidate`. Single-file
+26 LOC in `engine/src/board/moves.rs` — O(1) bbox+ball-area capacity
reserve before the legal-set rebuild loop, kills the hashbrown
power-of-2 rehash cascade. Laptop bench gate (8845HS, `--profile
profiling`, criterion n=800, median runs 2+3): **2.4936 ms → 1.4595 ms =
+70.9% sims/s**, uniform across n=100/400/800. perf report confirms
mechanism — `reserve_rehash` 31.4% → 1.2%, out of top-5. Independent
review `07_legal_moves_review.md` = MERGE-READY, all 8 checks pass.

**Merge HELD** — §S178/§S180b live on vast; no master push, no FF-merge
until §S178 close + vast cross-host (`9900X + 5080`) re-bench (same
3-run discard-first protocol). If vast delta within ±5pp of laptop →
FF-merge + push; if diverges >5pp → investigate L3-cache sensitivity
(8845HS 16MB vs 9900X 64MB) before merge. Residual: `legal_moves_set`
still #1 self-time (41.8%) post-fix → next perf wave targets the rebuild
insert cost itself (TLS scratch / incremental legal-set; plan Option (c)).

**CLOSED → §S182** — vast cross-host re-bench PASS (+66.4%, within ±5pp of
laptop +70.9%); FF-merged to master 2026-05-22.

---

## §S180b — 3-knob escalation CLOSE: config-level surface exhausted

*DISCRIMINATOR: §S180b = Sustained Training Sprint 180b (this entry).*

**Status:** FAILED. V180b-4 HARD FAIL @ step 50K — wr_sealbot collapsed to
**0%** (CI [0.0, 3.7]). 4th colony reproduction (§175, §S179, §S180a, §S180b).
Config-level anti-colony surface area exhausted.

**Run identity:**
- Branch at launch: master `3146144`
- Anchor: `bootstrap_model_v6.pt` (SHA `7ab77d2c…372103`)
- Variant: `configs/variants/v6_botmix_s180b_3knob_escalation.yaml`
- 3-knob delta vs §S179: `completed_q_values: true` (restored) +
  `bot_batch_share: 0.15 → 0.30` + `game_length_weights` neutralized
  (`[1.0, 0.50, 0.15] → uniform [1.0, 1.0, 1.0]`)
- Run-id: `fd9ea56e320646e5aeae11aefbe296bb`
- tmux: `sS180b` on vast (5080)
- Launched 2026-05-20 21:04 UTC; SIGINT-stopped 2026-05-21 23:00 UTC at
  step **53,890** (~26 h elapsed). Clean exit. Killed after V180b-4 FAIL
  @ step 50K.

**Eval trajectory:**

| @step | wr_sealbot | CI95 | colony@sb | wr_anchor | colony_a | wr_best | elo |
|---|---|---|---|---|---|---|---|
| 10K | 11% | [6.3, 18.6] | 7/100 | 61% | 36/100 | 52% | 422 |
| 20K | 7% | [3.4, 13.7] | 3/100 | 56% | 35/100 | 50% | 342 |
| 30K | 12% | [7.0, 19.8] | 11/100 | 61% | 40/100 | 61% | 354 |
| 40K | 19% | [12.5, 27.8] | 12/100 | 68% | 43/100 | 57% | 330 |
| 50K | **0%** | [0.0, 3.7] | 0/100 | 65% | **59/100** | 62% | 237 |

Pre-registered verdicts: V180b-1 @10K PASS; V180b-2 @20K FAIL; V180b-3 @30K
FAIL; **V180b-4 @50K FAIL**.

**Diagnostic — colony capture, masked.**

The 3-knob escalation crushed every *visible* colony metric: self-play
`colony_extension_fraction` ~0.04% of games (near-extinct), `colony@sealbot`
0–12% throughout (never the §S179 91%). Yet the policy still collapsed:
wr_sealbot 11→7→12→19→0. The 40K 19% was a transient pre-crash peak, not
recovery. At 50K the L34 capture signature fired — anchor 65% (high) /
sealbot 0% (collapsed) / `colony_a` jumped 43→59 per 100. The colony lives
in *anchor games*, a channel none of the 3 knobs touch. The model overfit
to beating the colony-prone anchor and lost all generalization to SealBot.

Threat probes PASS throughout (C1–C3, contrast 4.2–5.4, top5 40–70%,
top10 65–75%) — circuit health independent of colony collapse, L22
reconfirmed a 4th time.

**Falsified.** Hypothesis H-S180b-1: "combined config-level escalation
(CQV + 2× bot_batch_share + neutral game_length_weights) supplies enough
direct anti-colony force to escape the attractor."

### Falsified Hypotheses Register addition

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S180b (H-S180b-1) | 3-knob config escalation supplies enough direct anti-colony force to escape the attractor | §S180b eval trajectory (close 2026-05-21) | Every visible colony metric crushed (self-play colony ~0.04%, colony@sealbot 0–12%) yet wr_sealbot still collapsed 19%→0% @50K with L34 anchor↑/sealbot↓ divergence. Capture channel is config-invisible. |

**Archive.** `archive/s180b_3knob_fail/` on vast — 6 checkpoints
(`ckpt_step{10,20,30,40,50}k.pt` + `ckpt_step53500.pt`) +
`best_model_final.pt` + `eval_rounds_s180b.jsonl` (5 `evaluation_round_complete`
events) + `metadata.json` + `training_tail.jsonl`. Replay buffer (3.1 GB) +
14 dense intermediate checkpoints deleted post-archive-verify (`checkpoints/`
4.2 GB → 569 MB).

**Successor.** §S181 — code-level levers. Config-level surface (CQV,
bot_batch_share, game_length_weights, cosine, ply_cap split) is exhausted
across §S178/§S179/§S180a/§S180b with zero escape. Next intervention must
be code-level: prioritized-sample-weighting (PSW) on bot-corpus rows, OR a
bot-corpus refresh hook that regenerates SealBot-vs-current games mid-run.

### Process patterns / Mechanism Lessons

§S180b adds L38.

- **L38 (config-level anti-colony surface is exhausted; capture channel is
  config-invisible).** §S178/§S179/§S180a/§S180b swept the full config-level
  lever set — bot-mix share (0.15, 0.30), CQV on/off, ply_cap split, cosine
  on/off, game_length_weights (biased, neutral). Every arm reproduces the
  colony attractor. §S180b is the decisive instance: it drove every *visible*
  colony metric to near-zero (self-play colony 0.04%, colony@sealbot 0–12%)
  and still collapsed via L34 anchor↑/sealbot↓ divergence. Conclusion: the
  capture operates through a channel no YAML knob reaches — diagnosis-by-
  metric is exhausted. §S181+ must use code-level levers (PSW / corpus
  refresh hook) and instrument the anchor-game colony channel directly.

---

## §S181 — structural diagnosis (research wave)

*DISCRIMINATOR: §S181 = structural-diagnosis research wave (this entry).
INSPECTION-ONLY — no training, no hot-path edit, no config edit. Successor
to §S180b after the config-level anti-colony surface was declared exhausted
(L38).*

**Status:** COMPLETE. 4 parallel inspection tracks, all reviewed and PASSED.
Verdict: the colony attractor is a **training-loop-generated value-head
discrimination collapse** that the architecture PERMITS and the current
metrics are BLIND to. It is NOT bootstrapped and NOT MCTS-driven —
handoff hypotheses #1 and #5 both FALSIFIED.

**Wave identity:**
- Branch: `phase4.5/s181_structural_research`
- Mode: inspection-only (4 standalone probes under
  `scripts/structural_diagnosis/`; no selfplay/training/MCTS core touched)
- Inputs: §S180b archive (`archive/s180b_3knob_fail/`), anchor
  `bootstrap_model_v6.pt` (SHA `7ab77d2c…`), v6 human corpus, eval archives
- Aggregation: `audit/structural/00_aggregation.md`

### Per-track verdicts

| Track | Question | Verdict |
|---|---|---|
| T1 — bootstrap+corpus bias | Does the bootstrap/corpus encode colony bias pre-self-play? | **BIAS-RULED-OUT.** Value head extension-favouring (open-5 +0.978 > 5-blob +0.583; open-4 +0.704 > 4-blob +0.378); Δ(colony−ext) value −0.150, Welch p=0.355 (wrong sign, n.s.). Corpus winning lines 91.3% extension-shaped, rising to 100% at 1400+ Elo. Policy plays the 6th-move win 90%. Falsifies handoff hypothesis #1. Correction: §S178 line uses encoding `v6` (k_max=1), NOT v6w25. |
| T2 — value head + encoding architecture | Does the architecture structurally encourage colony interpretation? | **ARCHITECTURAL-BIAS-CONFIRMED (PERMISSIVE, not FORCED).** Dual-pool value head `v_max` half is a coverage-blind monotone peak detector — `GMP(colony) ≡ GMP(extension)` exactly (max\|diff\|=0.0), no architectural counterweight guaranteed. Permits colony-value-saturation; does not compel it. Density-centred cluster windowing favors compact structure. K-cluster min-pool is v6w25-only — does NOT fire for the §S178 line. Ranked fixes A1–A4; recommends A2 (multi-scale avg-pool ~40 LOC) + A3 (colony-penalty aux ~60 LOC) paired. |
| T3 — MCTS colony dynamics | Does MCTS+PUCT amplify or correct the bias? | **MCTS-NEUTRAL.** c_puct ×0.5/×2.0 and Dirichlet ×4 sweeps move colony-visit fraction <6pp / <3pp — no search-config escape hatch (extends L38). DECISIVE side result: §S180b step-50k value head collapsed FLAT — colony−extension value spread −0.016 vs anchor +0.617. L37 INVERTED: colony positions yield LOWER-entropy, MORE-concentrated visit targets (sharp-and-wrong, strong self-reinforcing CE gradient). |
| T4 — probe + dashboard redesign | Why did the probes miss 4 collapses? | **PROBE-REDESIGN-NEEDED.** C1–C4 static threat-logit probes cleared the gate ~11× at the §S180b 0/100 crash — categorically blind. `colony_a` (anchor-game colony fraction) already in the `evaluation_round_complete` payload at 36/100 by step 10K — 40K before crash — never surfaced first-class. 4 MCTS-in-loop probes designed; retrospective fire-step 20–40K before the §S180b crash. Land order PR-A (`colony_a` ~40 LOC) first. |

### Convergent diagnosis

The colony attractor is **H6 — a training-loop value-head discrimination
collapse**: the value head flattens (loses colony/extension separation)
during self-play, removing the signal MCTS needs to prefer extension;
search then collapses onto the colony-biased policy prior. Measured: value
spread +0.617 (anchor, healthy) → −0.016 (§S180b step-50k, captured). The
architecture **H7 — offers no resistance**: the `v_max` coverage-blind
monotone peak detector has no counterweight. The metrics **H3/H4 — are
blind**: C1–C4 static, `colony_a` never first-class. This IS the
config-invisible capture channel L38 named — and it is directly observable
with a 40-position static value-spread probe, one forward pass per
checkpoint.

### Ranked hypothesis list

| Rank | Hypothesis | Status |
|---|---|---|
| 1 | H6 — training-loop value-head discrimination collapse (value spread +0.617→−0.016) | TOP — MEASURED; primary driver |
| 2 | H7 — architecture permits H6, no resistance (`v_max` coverage-blind, weight-independent) | HIGH — MEASURED; fix surface = T2 A2 |
| 3 | H3/H4 — probes blind, `colony_a` config-invisible | HIGH — CONFIRMED 4×; instrumentation, not a fix |
| 4 | H8 — colony target sharp-and-wrong (L37 inverted; H_frac 0.484 vs 0.805) | MED — explains *why* CE reinforces |
| 5 | H9 — aux opp-reply head lower loss floor in colony regime | MED — mechanism inferred, not measured |
| 6 | H10 — density-centred windowing favors compact structure | LOW for v6 K=1; v6w25 re-entry only |
| — | H1 — bootstrap/corpus encode colony bias | **FALSIFIED** (T1) |
| — | H5 — MCTS dynamics favor colony | **FALSIFIED** (T3); MCTS-search config sub-surface exhausted |
| — | H2 (v6 form) — min-pool architectural asymmetry | **N/A** for v6 anchor (`value_pool="none"`) |

### Next-step decision tree

```
FU-1 — value-spread checkpoint-ladder probe  (~30–60 min, 0 GPU, ~2 dev-hr)
  Probe §S180b ladder ckpt_step{10,20,30,40,50}k for value spread.
  │
  ├─ spread degrades monotonically from step 10K
  │    → loop installs bias early + gradually
  │    → FU-2 architecture A/B must hold the canary from step 0;
  │      PSW/refresh-hook (if used) must act before step 20K
  │
  └─ spread holds healthy then late cliff
       → phase transition exists; target the cliff step directly

FU-2 — value-head re-architecture A/B  (T2 A2+A3; ~3–5 days, ~3–4 GPU-days)
  Requires FU-1 first (pins the canary target). Fresh re-pretrain
  mandatory (A2 breaks value_fc1 shape). Wire value-spread canary +
  hard-abort gate (abort if spread < +0.20).
  │
  ├─ A2+A3 holds spread > +0.20 AND wr_sealbot does not collapse
  │    → architecture was the load-bearing permissive element (H7 = fix)
  │    → adopt A2+A3; close the structural line
  │
  └─ spread still collapses under A2+A3
       → loop installs the bias regardless of architecture
       → escalate to buffer-level levers (PSW / bot-corpus refresh hook),
         success metric = value-spread canary, NOT loss/value-acc (Goodhart)

PR-A — colony_a first-class metric + ALERT-2  (T4; ~40 LOC, ~2 hr)
  Independent of FU-1/FU-2 — land in parallel. Lowest-LOC highest-leverage
  fix in the wave: would have fired 20–40K steps before the §S180b crash.
```

### Process patterns / Mechanism Lessons

§S181 adds L41/L42/L43.

- **L41 (MCTS-search config sub-surface is also exhausted — extends L38).**
  T3 swept the last untested config sub-surface: `c_puct` ×0.5/×2.0 and
  `dirichlet_alpha` ×4. Colony-visit fraction moved <6pp / <3pp — inside
  n=20 noise. Higher c_puct mildly *worsens* colony preference (it
  up-weights an already colony-biased prior). The MCTS-search knobs join
  the YAML config-level levers as exhausted. Do NOT propose c_puct /
  Dirichlet retune as an anti-colony lever — FALSIFIED in
  `audit/structural/03_mcts_colony_dynamics.md` §4/§5.

- **L42 (value-head discrimination collapse is the config-invisible
  capture channel — directly observable).** The L38 "config-invisible
  capture channel" is the value head losing colony/extension separation
  during self-play. MEASURED: colony−extension value spread +0.617
  (anchor) → −0.016 (§S180b step-50k). A flat value head gives MCTS no
  signal to prefer extension over colony, so search collapses onto the
  colony-biased policy prior. It is "config-invisible" only because no
  dashboard metric tracked it — it is fully observable with a 40-position
  static probe, one forward pass per checkpoint. Wire `value_spread`
  (mean V over a colony bank − mean V over an extension bank) as a
  first-class canary + hard-abort gate (abort if spread < +0.20). This
  supersedes diagnosis-by-config-metric.

- **L43 (colony training target is sharp-and-wrong, not diffuse — L37
  inverted).** §S180a L37 hypothesized "colony regime → diffuse MCTS
  visit target → weak CE signal". T3 MEASURED the opposite: colony
  positions yield LOWER-entropy, MORE-concentrated visit targets
  (H_frac 0.484 vs extension 0.805) — the policy prior is sharply peaked
  on blob-adjacent cells. CE against a sharp wrong target is a STRONG
  self-reinforcing gradient pushing the policy further into the colony
  mode — a worse failure mode than a diffuse target. This is the textbook
  attractor mechanic and matches the operator's "structured wrong choice,
  not random collapse" game-read. L37's *empirical finding* (visit-count
  CE weaker than CQV) stands; its *proposed mechanism* (diffuse target)
  is corrected.

### Falsified Hypotheses Register additions

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S181-T1 | `bootstrap_model_v6.pt` + v6 human corpus jointly encode a colony bias in the value head and/or policy head before self-play (handoff hypothesis #1) | §S181-T1 bootstrap+corpus bias audit (`audit/structural/01_bootstrap_corpus_bias.md`) | Value-head Δ(colony−ext)=−0.150, Welch p=0.355 (wrong sign, n.s.); near-win sub-probe rates open-5 extension +0.978 > 5-blob +0.583 (open-4 +0.704 > 4-blob +0.378); policy plays the 6th-move win 90%; corpus winning lines 91.3% extension rising to 100% at 1400+ Elo. No colony bias exists pre-self-play — the attractor is generated by the training loop. |
| §S181-T3 | MCTS-search parameters (`c_puct`, `dirichlet_alpha`/`epsilon`) are a viable anti-colony escape lever (handoff hypothesis #5) | §S181-T3 MCTS colony-dynamics audit (`audit/structural/03_mcts_colony_dynamics.md`) | c_puct ×0.5/×2.0 moves colony-visit fraction <6pp; Dirichlet ×4 moves it <3pp — both inside n=20 noise. Higher c_puct mildly worsens colony preference. MCTS+PUCT neither amplifies nor corrects the bias; it faithfully passes through a colony-biased value/policy head. MCTS-NEUTRAL — extends L38 to the search sub-surface. |

### Files produced

- `audit/structural/00_aggregation.md` — this wave's synthesis
- `audit/structural/0{1,2,3,4}_*.md` — 4 track audits + sidecar JSONs
- `scripts/structural_diagnosis/{probe_value_bias,probe_architecture,mcts_colony_probe,new_probes}.py`
- `reports/s181_next_wave_skeleton.md` — next-wave operations skeleton

### Successor

§S181 does NOT launch a training run. Next wave = `reports/s181_next_wave_skeleton.md`:
FU-1 value-spread ladder probe FIRST (pins the flatten step, ~30 min),
THEN FU-2 value-head re-architecture A/B (T2 A2+A3) gated on FU-1. PR-A
(`colony_a` first-class metric) lands in parallel — independent, ~40 LOC.
PSW / refresh-hook levers are demoted: they reshape the buffer but only
help if the reshaped buffer retrains value-head discrimination — pair
either with the L42 value-spread canary as the success criterion, not
loss/value-acc (Goodhart).

### §S181 FU-1 — value-spread checkpoint-ladder probe

Ran the FU-1 ladder probe (`scripts/structural_diagnosis/fu1_value_spread_ladder.py`,
inspection-only, CPU, ~0 compute). Pins WHEN the value-head discriminator
flattens across the §S180b trajectory. Full report:
`audit/structural/05_fu1_value_spread_ladder.md`.

**Bank.** Reused the T3 40-position canonical bank verbatim (20 colony +
20 extension; `mcts_colony_probe.py` builders, deterministic). Fixture
SHA-256 `934204713620d171…dcc23991`. **Brief correction:** the +0.617
anchor spread is a T3 measurement, NOT T1 — T1's `probe_value_bias.py`
bank is 50+50 and gives a different figure (−0.150). FU-1 used the T3
bank, the only one consistent with +0.617. Anchor reproducibility gate
PASS — FU-1 reproduces T3 `value_head` to 4 dp (+0.6173 vs +0.617).

**Ladder.** V_spread (mean V colony − mean V extension):

| step | 0 | 10k | 20k | 30k | 40k | 50k | 53.5k |
|---|---|---|---|---|---|---|---|
| V_spread | +0.617 | +0.260 | −0.110 | +0.140 | −0.051 | −0.016 | +0.099 |

**Verdict.** Drift classifier returns `OSCILLATION` (non-monotone;
+0.250 up-swing 20k→30k). But the mechanical label is a **noise
artifact** — post-20k swings (±0.25 peak-to-peak) sit within ~1.3×
SE(spread)=0.185 of a flat spread ≈ 0. The substantive signature is
**EARLY / FRONT-LOADED COLLAPSE**: 69 % of spread lost by step 10k,
sub-+0.20 (FU-2 abort gate) at or before 10k, negative by 20k, then
flat-dead. The discriminator dies inside the first 10–20k steps and
never recovers. Ladder limitation: no sub-10k checkpoint — cannot
resolve a step-0 onset from an ~8k cliff.

**colony_a denominator = WINS-ONLY** (`evaluator.py:213-216`,
`display.py:40`: `colony / wins`). The §S180b colony_a series is the
fraction of the model's *wins vs the anchor* that are colony-shaped —
computed over a shrinking small win pool as wr collapses. A colony_a
trigger fires late + noisy; the value-spread static probe (stable
denominator, crosses +0.20 before step 10k) is the better canary.

**Recommendation (operator decides).** Early-collapse read routes to
FU-2 (value-head re-arch A/B, T2 A2+A3) with the value-spread canary
wired as a hard-abort gate **from step 0** — a buffer-level refresh
hook cannot act fast enough for a collapse >half complete by step 10k.
Cheaper de-risk before committing FU-2: a finer 0–20k ladder (new
checkpoints every ~2k from a short §S180b-config re-run) to pin
step-0-onset vs mid-cliff. Not auto-launched.

### §S181 FU-1.5 + PR-A — finer ladder + value_spread canary

Closed 2026-05-23 — Stage 4 merged. 4 commits on master
(`b8ebb1d..9cfadd7`) covering FU-1.5 (probe SHA gate + pre-registered
V-FL classifier + 10-rung 2k-cadence ladder + audit doc), PR-A
(value_spread canary as a first-class colony-capture trigger), the
§S180b config + launch script for the re-run, and the FU-1 audit
artifacts brought along. Independent landed fix `de149e6
fix(mcts): canonical child order — kill FxHashSet-order leak`
addresses a latent `FxHashSet` iteration-order leak found during the
investigation (behaviourally inert per 5 independent tests; not a
regression cause; landed as a hygiene improvement). REVIEW
(Opus 4.7, high-effort, fresh context) returned **PASS 19/19** across
scopes A/B/C.

**Verdict (mechanical, code — L13 guard).** `V-FL-A — STEP-0-ONSET`.

**Ladder (clean-host vast re-run, anchor + bank SHA gates PASS):**

| step | 0 | 2k | 4k | 6k | 8k | 10k | 12k | 14k | 16k | 18k | 20k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V_spread | +0.617 | **+0.175** | **−0.118** | +0.055 | +0.217 | +0.390 | +0.167 | +0.523 | +0.221 | +0.208 | +0.108 |

Single-interval loss 0→2k = **−0.4421 = 86.7 % of the total
trajectory loss**. V_spread crosses the +0.20 FU-2 abort gate
*between step 0 and step 2k*. Step 4k goes negative.

**Key finding — resolves FU-1's open question:** the value-head
collapse is front-loaded to the first 2k steps. The FU-1 10k-cadence
ladder under-sampled this region and could not localize the onset;
FU-1.5's 2k cadence places >86 % of the collapse before step 2k.

**FU-2 routing (pre-registered V-FL-A → A2 arm):** **A2 multi-scale
avg-pool value head load-bearing.** An aux-loss A3 cannot catch a
value-head crash that is >86 % complete by step 2k — there is no
time for an aux gradient to push back before the discriminator is
already through the abort gate.

**Substantive nuance (not the verdict).** §S180b crashed
monotonically to a permanent flat-dead band by step 20k; FU-1.5
crashes fast then **oscillates** post-onset (peak +0.523 at step
14k, range −0.118 to +0.523). Both 20k endpoints sit near zero
(FU-1.5 +0.108, §S180b −0.110, both within ~1 × SE of zero). The
endpoints agree; the paths diverge — consistent with chaotic
post-onset divergence in the colony-attractor neighbourhood.

**§S180b cross-check @ step 10k (brief ±15pp variance gate):**
wr_sealbot 9 % vs §S180b 11 % (−2 pp); elo 421.3 vs 422 (−0.7);
wr_anchor 66 % vs 61 %; wr_best 52 % vs 52 %; colony_wins_anchor
41 vs 36. All within noise — FU-1.5 IS a faithful §S180b
reproduction.

**Investigation context (the host-state regression).** The first
FU-1.5 launch attempt hit an anomalous vast-host regime (`depth ~2.5
/ games ~77-96 plies` vs every pre-05-22 run at depth 3.4-3.8 /
plies 27-57). Five-test code exoneration of §S182/§S183
(deterministic depth probe; deterministic full-game probe; laptop
self-play 100-game smoke A/B; laptop full-training 100-game A/B;
vast 100-game A/B post-reinstall — every test bit-identical old vs
new code) ruled out the perf wave. Root cause: stale vast-host state.
Clean reinstall (`rm -rf hexo_rl` → fresh `git clone` → `make install`:
fresh `.venv` torch 2.11.0+cu128 + fresh engine build + 1555 tests
PASS) restored the §S180b regime. The relaunch on the clean host is
the data this audit reports on.

**PR-A — value_spread canary.** First-class colony-capture trigger
wired as a checkpoint-save callback emitting a `value_spread` event
+ alert rule (WARNING < +0.30, SOFT-ABORT < +0.20 — the FU-1 / FU-2
abort gate). 40-position T3 bank frozen at
`tests/fixtures/value_spread_bank.json` (SHA `9342…23991`); INV pin
`tests/test_inv_value_spread_bank.py` asserts the SHA. Renderer
failure cannot propagate (canary fire-and-forget). `colony_a` stays
in the eval payload — additive, not destructive. No hot-path touch
(canary fires post-save, not in the inner loop). Micro-bench: 15.1
ms → 30.4 ms ckpt-save on RTX 4060 (+15.3 ms, under the 50 ms gate).
Replaces `colony_a` as the *trigger* signal (FU-1 §5: `colony_a` is
wins-only / late / small-denominator; value_spread is a stable
static probe that crosses +0.20 before step 2k).

**Independent landed fix (`de149e6`).** `pick_topk_children` Path A
emitted children in raw `FxHashSet` iteration order (a knowingly-left
"identical to pre-cap behaviour" hash leak); §S182's capacity-reserve
changed that order. The fix unifies both paths to canonical
`(prior desc, flat asc)` — the canonical key Path B already used.
MCTS is now order-stable regardless of hashbrown capacity. Five
independent tests prove behavioural identity (it was never a
regression cause); landed as hygiene. Bench-gated: MCTS sim/s
−3.0 % (under 5 % gate). Regression test
`test_topk_child_order_independent_of_hashset_capacity` pins the
invariant.

**L44 — value-head crashes can be front-loaded.** A 2k-cadence
ladder on §S180b's recipe shows >86 % of the value-spread loss
occurs in the first 2k steps from the bootstrap. The 10k-cadence
ladder of FU-1 under-sampled this region. Future structural-
diagnosis waves that probe value-head onset should default to ≤2k
cadence in the first 10k steps.

**L45 — host-state hygiene is research-load-bearing.** The
05-21→05-22 vast regime change (depth 3.4→2.5 stable; uniform
across all 05-22 vast launches before the reinstall) was *not*
code. It was accumulated host cruft (stale `.venv`, stale compiled
engine, leftover state from prior runs). Five independent code
A/B tests exonerated the perf wave. A clean `git clone` +
`make install` restored the §S180b regime. **Implication:** when a
"first variance since the N-knob baseline" appears, host-state
reinstall belongs in the discriminator set alongside the code
bisect — not as a follow-up.

**L46 — post-onset oscillation ≠ flat-dead.** §S180b reached a
permanent flat-dead V_spread band after step 20k; FU-1.5 (same
recipe, clean host) reaches a similar mean-near-zero by step 20k
but the path oscillates (+0.523 peak at step 14k). Two faithful
reproductions of the same recipe can take different routes through
the chaotic colony-attractor neighbourhood. The 20k endpoint
agrees within ~1 × SE; the trajectory does not.

**Next.** FU-2 NOT auto-launched. **Operator decides FU-2 A2 arm**
(multi-scale avg-pool value-head re-architecture; ~3-4 GPU-days
pretrain cycle) per L34/L42/V-FL-A routing.

---

## §S181-AUDIT — Track A source decomposition + PR-B hygiene

**Wave.** Cheap inspection-only audit + no-regret hygiene patch in
parallel. Track A on `phase4.5/s181_audit_track_a` (6 commits
`c2c9e5e..e717c61`, FF-merged 2026-05-23). PR-B on
`phase4.5/s181_pr_b_hygiene` (3 commits `c2a0f31..03425de` post-rebase,
FF-merged 2026-05-23). REVIEW pass (Opus 4.7 fresh-context, scopes
A/B/C all PASS).

**Provenance.** Designed against L47 (training-loop dominates
architecture by ≥1.0 V_spread per ~1000 steps; banked on the parked
`phase4.5/s181_fu2_a2_arch` branch, audit doc
`audit/structural/08_fu2_a2_sustained_ladder.md`). The FU-2 wave proved
A2's architectural anchor inversion (−0.508 PT-3 baseline) was dragged
to +0.13 within 1000 steps — same end-state as pre-A2 from the
sign-opposite direction. Next surface = the LOOP. Track A localizes
which loop-level source carries the bias; PR-B lands stage-1 hygiene
mechanism-aligned with L47.

### Per-subtask Track A verdicts (LITERAL, L13 guard)

| ID | hypothesis | verdict | quantitative finding |
|---|---|---|---|
| A1 | H-BOT — bot corpus position-level bias | INCONCLUSIVE | colony frac 26.0%, asymmetry +0.078 (8× < anchor +0.617) |
| A2 | H-AUG — augmentation oversamples colony | INCONCLUSIVE + FALSIFIED | unique-variants ratio 1.0 (mechanism dead); feat-var ratio 0.54 is downstream signature |
| A3 | H-BANK — T3 bank confound | **CONFOUND** | Pearson r=0.27, T3 V_spread amplified ~3× vs alt bank |
| A4 | H-CE-STRENGTH — per-class gradient | INCONCLUSIVE | grad L2 ratio 1.21 (L43 entropy confirmed col 0.50 / ext 0.80) |
| A5 | H-PRETRAIN — pretrain position z | INCONCLUSIVE | colony frac 31.2%, asymmetry +0.157 (~25% of anchor) |

No subtask reached STRONG-CONFIRM / ASYMMETRIC-CONFIRMED /
PRETRAIN-COLONY-BIASED. The largest source contributor is H-PRETRAIN
(asymmetry +0.157 × ~30% sample share = +0.057/step upper bound),
followed by H-BOT (+0.014/step), with H-CE-STRENGTH as a 1.21×
multiplier on every colony sample and H-AUG falsified to zero.

### A6 dominant-source identification + routing

**Dominant source: NONE pre-registered-confirmed.** Per routing table:
"None confirmed → escalate to Track B (live training-loop gradient
instrumentation)". The largest unmeasured surface is the self-play
buffer's per-sample gradient signal — Track B is a short instrumented
run that closes the per-step pull accounting gap.

**Composite next-wave lever (operator decides — NOT auto-launched):**

1. Track B (PRIMARY) — short instrumented run logging per-sample
   gradient magnitude bucketed by colony / extension / neither on the
   live self-play buffer.
2. Combined lever IF Track B confirms loop-side imbalance:
   PSW (Prioritized Stratified Window on the buffer) + refresh hook
   (periodic bot-corpus regeneration) + per-class target temperature
   on colony positions.
3. **Dual-bank V_spread canary update.** Per A3, augment PR-A's
   `value_spread_canary.py` to compute V_spread on BOTH T3 and alt
   banks. Alt bank fixture pinned at
   `tests/fixtures/value_spread_bank_alt.json` with SHA in
   `meta.sha256`. Compute on both ≈ 1 sec per ckpt.

**DEFERRED.** Track C (encoding swap, ~1 GPU-day), EMA (changes
self-play inference model), WDL migration (same cost profile as A2).
**DO NOT** re-propose architecture-only fixes (L47 stands).

### PR-B — hygiene patch (Stage-1, landed)

Three trainer-side hygiene changes, ~80 LOC net:

1. **Param-group split for AdamW** (`c2a0f31`). New helper
   `build_param_groups(model, weight_decay)` — 1D params (BN/LN
   scales) OR `.bias`-suffixed → no-decay; 2D+ weights → decay.
   Standard nanoGPT/KataGo pattern; pure hygiene, orthogonal to
   colony attractor.
2. **Cosine `eta_min` floor raised** to 5e-4 (from 2e-4) in
   `configs/training.yaml` (`a43d5eb`). Mechanism alignment with L47:
   prevent loss-of-plasticity at late-run when the colony attractor
   manifests; KataGo precedent (never drop below 0.5× peak).
3. **Policy entropy regularization** set to 0.005 (from 0.01) in
   `configs/training.yaml` (`03425de`). Mechanism alignment with L47:
   counter-pressure on policy collapse via entropy bonus. Wiring
   pre-existing (trainer.py:512 + losses.py:215 — `total = total -
   entropy_weight * entropy_bonus`, correct sign).

**Tests.** `tests/test_optimizer_param_groups.py` (4 unit tests) +
`tests/test_inv_optimizer_param_groups.py` (5 INV pin tests). 9/9 PASS;
broader trainer/optimizer/losses regression 84/84 PASS (excluding
pre-existing minimax_bot ImportError, unrelated).

**Deliberately excluded (cite L47).**
- **EMA.** Smooths gradient flow but changes the self-play inference
  model; contaminates audit baseline + future ladder probes. Defer
  until Track A localizes the dominant lever and we want a clean
  baseline.
- **`beta2 = 0.95`.** L47 says recent gradients are colony-biased;
  shortening AdamW's gradient memory window AMPLIFIES the bias.
  Counter-indicated.
- **WDL migration.** Same cost profile as A2 (re-pretrain, value-head
  re-arch). V-FU2-C falsification of architecture-only fixes makes
  this a low-priority commitment until Track A/B/C surface a
  source-targeted lever.

### L47 (adopted to master from the parked FU-2 branch)

**L47 — value-head architectural inductive-bias fixes are insufficient
without loop-side intervention.** A2's starting-point inversion
(extension-favouring anchor V_spread −0.508) was dragged through zero
to +0.13 colony-favouring within ~1000 training steps under the
§S178/§S180b 3-knob recipe — same end-state as pre-A2 reached
monotonically from +0.617, from the SIGN-OPPOSITE direction. The
training loop's value-target dynamics (selfplay outcome × MCTS
visit-count CE × corpus mix) override the architecture's structural
starting point by a margin of ≥0.5 in V_spread magnitude per 1000
steps (T3-bank metric; see L48 for revised magnitude). The next surface
IS the loop — PSW / refresh hook / value-target perturbation / class-
weighted gradient. **Source:** parked branch
`phase4.5/s181_fu2_a2_arch` audit doc
`audit/structural/08_fu2_a2_sustained_ladder.md`.

### L48 — V_spread metric is partially T3-bank-specific (CONFOUND)

A3 measured Pearson r=0.27 between V_spread(T3) and V_spread(alt
corpus-derived bank) across the FU-1.5 ladder (anchor + 10 ckpts at 2k
cadence). Anchor V_spread on T3 is +0.617; on alt bank +0.212 (~34%).
Alt-bank trajectory range [+0.016, +0.258] vs T3 range [−0.118,
+0.617] — alt magnitudes ~3× smaller throughout the ladder. **L47's
≥1.0/1000-step magnitude is revised downward** by the same factor when
read on real corpus positions: actual per-step pull ≈ +0.33/1000 steps,
not ≥1.0. The mechanism stands; the absolute amplitude was amplified
by T3 bank's specific synthetic structures. **Operational corollary:**
V_spread canary should be augmented to compute on BOTH banks; T3 stays
as historical reference, alt grounds the metric in corpus reality.

### L49 — no single corpus-level source confirmed dominant; multi-source attractor

Across A1 + A4 + A5, no single hypothesis reached its pre-registered
confirmation threshold. H-PRETRAIN is the largest contributor (+0.157
asymmetry, ~25% of anchor V_spread direction), H-BOT contributes
+0.078 (~13% of anchor), H-CE-STRENGTH provides a 1.21× per-colony-
sample gradient multiplier. Sum of source-attributable per-step pull
upper bounds (~+0.07/step) exceeds the observed alt-bank-corrected
movement rate (~+0.0003/step) by ~200× — meaning sources have
**head-room**; the actual per-step rate is gated by optimizer
damping + neither-class counter-currents (61% bot, 63% pretrain) +
unmeasured self-play buffer dynamics (Track B). **Implication:** no
single corpus-side knob fixes the attractor. The next wave should be
multi-lever (PSW + refresh hook + per-class target temperature) +
Track B instrumentation, NOT a single targeted intervention.

---

## §S181-AUDIT Wave 1 — Track B infra + C-LITE-1 + Track D + REAL_RUN_RECIPE

**Wave.** Capstone diagnostic before the next sustained run. Closes the
Track A "no single source" cliff (L49) by landing Track B's
instrumentation infra (operator-mediated run pending), capturing v7full
as a real-run anchor candidate, and synthesizing a parameterized recipe
gated on Track B's V-B verdict. FF-merged to master 2026-05-23 as 4
commits `7cd0dc0..71268ab` across 4 short-lived branches (PR-C / Track
B / Track C-LITE / Track D — all pruned post-merge). REVIEW pass (Opus
4.7 fresh-context, scopes A-F all PASS).

### Landed commits

- `7cd0dc0` PR-C: dual-bank V_spread canary (T3 + alt per L48). 412 LOC
  net, INV pin extended to alt fixture SHA `a68b810f…20a20ff`. 31/31
  PASS (8 INV + 23 unit). Back-compat preserved for legacy single-bank
  payloads. No hot-path touch.
- `3201c39` Track B: B1 per-source gradient-norm attribution
  (`hexo_rl/training/track_b_attribution.py`, ~3× cost via 3
  retain_graph autograd.grad slices) + B2 buffer position-class
  snapshot (`hexo_rl/training/track_b_buffer_snapshot.py`, ckpt-cadence
  hook in StepCoordinator) + B3 post-run trunk feature drift script +
  `v6_botmix_s181_track_b.yaml` variant + launch-and-analysis spec.
  43/43 PASS (12 Track B + 31 PR-C/INV). NOT LAUNCHED — operator-
  mediated vast launch (~6 h, ~$1.50) blocks V-B verdict aggregation.
- `b93d994` Track C-LITE-1: v7full anchor dual-bank V_spread (T3
  +0.2171 / alt +0.4078). Verdict C-LITE-1-A — encoding regression
  candidate CONFIRMED; v7full's alt-bank V_spread is ~2× v6's. T3
  borderline at +0.017 over SOFT-ABORT is L48-explained (T3 calibrated
  on v6's value head; alt is the corpus-grounded reference). C-LITE-2
  v6w25 stock probe DEFERRED — C-LITE-1's answer + REAL_RUN_RECIPE
  conditional path covers the encoding question for the v6 family.
- `47d658e` REAL_RUN_RECIPE: 7-section synthesis with parameterized
  Wave 2 lever stack gated on V-B-{A,B,C,D,E}. Anchor primary
  `bootstrap_model_v7full.pt`; encoding `v7full`; success criteria
  RR-G1..RR-G6 (LITERAL L13); compute budget ~$5 + 2-3 days dev.
- `71268ab` Track D: pipeline regression audit §150→§S178+ (596
  lines). Smoking-gun rank: (1) bot-corpus value-target imprint +
  staleness, (2) pretrain corpus colony pull × recency_weight=0.75,
  (3) ply_cap_value=0.0 × full_search_prob=0.5 cross, (4) bot-corpus
  staleness × outcome feedback, (5) recency_weight ×
  selfplay-buffer compounding (INCONCLUSIVE).

### Track B — B4 run pending (operator-mediated)

Run spec at `audit/structural/track_b/B_launch_and_analysis_spec.md`:
3000-step instrumented run on vast 5080 with `v6_botmix_s181_track_b`
variant, `bootstrap_model_v6.pt` anchor, dual-bank canary firing every
500 steps. Pre-registered V-B-{A..E} decision tree:

| ID | rule | downstream |
|---|---|---|
| V-B-A | one source ≥60% of total grad pull | source-targeted lever |
| V-B-B | all three sources 25-45% | multi-source damping (EMA + 2-stone aux + per-class target temp) |
| V-B-C | buffer colony-heavy ≥50% by step 2k | refresh hook + EMA priority |
| V-B-D | trunk centroids collapse ≥50% by step 1k | aux heads forcing trunk discrimination |
| V-B-E | none match | escalate, no real-run launch |

V-B-{verdict} feeds REAL_RUN_RECIPE §3 conditional lever pick. No L50+
banked this wave — Wave 2 will bank lessons from the actual B4 run +
verdict application.

### Wave 2 — operator decision point

`audit/structural/REAL_RUN_RECIPE.md` §7 captures the Wave 2 sequence:
B4 run → V-B aggregation → conditional lever pick → EMA always; 2-stone
aux iff V-B-D; per-source lever per V-B → pre-launch smoke (~$1.50,
~6 h) → main 100k-step run on v7full anchor (~$3, ~14 h). Total Wave 2
estimated cost ~$5 + 2-3 days dev. Operator decides launch timing; do
NOT auto-launch.

### Branches pruned post-merge

`phase4.5/s181_pr_c_dual_bank`, `phase4.5/s181_track_b_instrumented`,
`phase4.5/s181_track_c_lite`, `phase4.5/s181_track_d` — all FF-merged
and deleted. No push of these branch names; master HEAD `71268ab`
carries every commit.

---

## §S181-AUDIT Wave 2 — refresh-hook-less lever stack peak-and-collapse

**Status.** Lever stack `uniform_self` (v7full anchor + EMA decay 0.999 +
per-class target temperature on selfplay slice `sample_rate=0.20`) hit
project-record SealBot WR **33% peak @ step 20k** (1.9× §150 baseline
17.4%), then monotonic decline 11% @ 30k → 5% @ 40k. HARD-ABORT @ step
47642 on RR-G3 / §S180b 8% threshold breach. Wave 2 PROVES anti-colony
lever stack works short-term but **fails sustained without bot corpus
refresh**. Phase 4.5 remains BLOCKED; Wave 3 with refresh hook +
sliding-window WR gate + per-class temp scope revision is the successor.

### Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (ssh6.vast.ai:13053) |
| branch | `phase4.5/s181_wave2_lever_vba_selfplay` `54bd9da` (origin, NOT master) |
| variant | `v7_real_run_main` (`configs/variants/v7_real_run_main.yaml`) |
| anchor | `bootstrap_model_v7full.pt` SHA `568d8a33…d61e8e98` |
| encoding | v7full (single-window 19×19, 8 planes) |
| iterations target | 100 000; **actual 47 642** (operator-aborted) |
| run_id | `aad5e948c4b94bf395daeddbb57415b9` |
| launch / abort | 2026-05-24 00:09 UTC / 2026-05-25 00:05 UTC (~24h wall) |
| cost | ~$5 (1A B4 $1.30 + 4B smoke $0.70 + 5 main ~$5 = **~$7 total Wave 2**) |
| checkpoints saved | 23 @ 2k cadence + best_model.pt (last promotion step 30k) |
| events captured | 23 dual-bank canary fires; 4 SealBot evals; 3 best_model promotions |
| canonical deliverable | `reports/track_b_main/checkpoints/checkpoint_00020000.pt` (33% peak) |
| key audit doc | `audit/structural/wave2_real_run_analysis.md` |

Lever stack vs §S180b 3-knob recipe:
- encoding v6 → **v7full** (anchor match; C-LITE-1 verdict)
- EMA of weights (decay 0.999, every 10 optimizer steps), dispatched
  through `Trainer.inference_state_dict`
- per-class target temperature on selfplay slice (T_colony=1.5, others
  1.0, pretrain slice untouched, `selfplay_sample_rate=0.20` perf opt
  per smoke L48 throughput recovery)

### Path through V-B verdict (Option D)

B4 LITERAL verdict was **V-B-E** (no clean match): per
`audit/structural/track_b/B_aggregation.md`:
- V-B-A NO — uniform_self mean share 0.563 < 0.60 single-source gate
- V-B-B NO — pretrain mean 0.092 outside [0.25, 0.45] band
- V-B-C NO — buffer colony_frac stable 9-10% (no feedback loop)
- V-B-D NO — trunk inter-centroid 84% of anchor at step 1000

`B_verdict_synthesis.md` surfaced 4 routing options + recommended
**Option D** (partial smoke before full real-run commit) on the
near-miss V-B-A reading: selfplay-family share (uniform_self + recent)
= 90.8% — single-source partition missed by 3.7pp. Operator took
Option D. Smoke S-A PASS authorized main launch. Documented in
B_verdict_synthesis.md as "respects both the literal V-B-E *and* the
near-miss mechanism interpretation".

### SealBot WR trajectory

| step | wr_sealbot | CI95 | wr_anchor | colony_wins_sealbot | promoted |
|---:|---:|---|---:|---:|:---:|
| 10 000 | 24.0 % | [16.7, 33.2] | 66.0 % | 3 | ✓ |
| **20 000** | **33.0 %** | [24.6, 42.7] | 61.0 % | 6 | ✓ |
| 30 000 | 11.0 % | [6.3, 18.6] | 65.0 % | 4 | ✓ |
| **40 000** | **5.0 %** | **[2.2, 11.2]** | **70.0 %** | 5 | ✗ |

Peak step 20k; monotonic decline thereafter. Step-40k 5% below RR-G3
13% gate AND below §S180b 8% HARD-ABORT threshold. promoted=False at
step-40k = bootstrap_floor 0.45 gate failed = training producing
weaker models than current best_model.

### L34 anchor↑/sealbot↓ divergence

| transition | anchor Δ | sealbot Δ | L34 fires? |
|---|---:|---:|:---:|
| 10k → 20k | -5 pp | +9 pp | INVERSE (healthy) |
| 20k → 30k | +4 pp | -22 pp | **YES (1st)** |
| 30k → 40k | +5 pp | -6 pp | **YES (2nd)** |

Classic colony-attractor capture signature: model becomes *relatively
stronger* vs frozen anchor while *absolutely weaker* vs SealBot.
SOFT-ABORT trigger requires 5 consecutive — 2 observed when run was
killed.

### Dual-bank V_spread canary

alt-bank held +0.18–0.30 throughout 46k steps (well above +0.07
sustained gate). T3 started +0.27, oscillated +0.03–+0.09 through
step 28k, then collapsed to -0.26 by step 46k.

**Critical**: alt V_spread stayed above gate the entire run yet
eval-measured wr_sealbot collapsed 33→5%. The held-out V_spread canary
failed to track the actual performance collapse. (L50)

### Pre-registered success criteria — verdict

| ID | criterion | result |
|---|---|:---:|
| RR-G1 | T3 ≥ +0.20 sustained 0→50k | **FAIL** (crossed below at step 1000 in smoke; collapsed deep negative in main) |
| RR-G2 | alt ≥ +0.07 sustained 0→50k | PASS (held +0.18+ throughout) |
| RR-G3 | SealBot WR ≥ 13% @ step 30k (Wilson95 LB ≥ 9%) | **FAIL** (11% / LB 6.3%) |
| RR-G4 | SealBot WR ≥ 18% @ step 50k | **FAIL** (extrapolated from 5% @ 40k) |
| RR-G5 | colony_a < 50/100 in eval rounds | PASS (colony_wins_sealbot 3-6 throughout) |
| RR-G6 | L34 anchor↑/sealbot↓ clean | **FAIL** (2 consecutive instances) |

4 of 6 RR-G* FAIL. RR-G2 + RR-G5 PASS isolate the L50 mechanism:
held-out V_spread + colony-policy gates can both PASS while
eval-measured tactical WR collapses.

### Mechanism diagnosis

Three candidate mechanisms (ranked):

**M1 — Bot corpus opportunistic fit + degeneration (HIGH).**
21,899-position static bot corpus at `bot_batch_share=0.30` exposes
~77 SealBot positions/batch. By step 20k the corpus has been re-
encountered ~70 times (batch 256 × 30 % × 20k steps = 1.54 M bot
positions / 21,899 corpus size); distributional decay vs the
evolving model grows with policy-distance after corpus saturation.
Past step 20k: selfplay drifts model policy off the corpus
distribution → 30% bot-batch becomes off-distribution noise → fit
decays. Track D C4 (bot staleness × outcome feedback) candidate now
CONFIRMED as the dominant Wave 2 failure mechanism.

**M2 — Per-class temp dilution over selfplay slice (MEDIUM).**
T_colony=1.5 softens visit-count CE targets on 20% of selfplay
colony-classified rows. As selfplay buffer accumulates sharper
late-game policies, softening the model's own best moves degrades
tactical learning. Combined with M1 drift, compounds the post-peak
collapse. NOTE: alt V_spread stayed high → M2 doesn't hurt value-head
discrimination; the proxy for M2 damage is policy-sharpness data
not captured in this Wave 2 instrumentation.

**M3 — EMA averaging artifact (LOW, ruled out).** Smoke (Stage 4B,
also EMA-enabled) was clean S-A PASS at step 3000. Collapse appears
at step 20-40k, well past EMA's effective warmup. EMA decay 0.999
is monotonically a smoother (can lag, doesn't actively degrade).

**Most likely: M1 + M2 compound.** Bot-corpus drift is dominant
driver; per-class temp amplifies tactical degradation as selfplay
slice grows past corpus-dominated early window. Wave 3 must address
BOTH: refresh hook (M1) + per-class temp scope revision (M2).

### Wave 1 vs Wave 2 reframing of Track D C4

REVIEW (Stage 1A) surfaced an apparent tension between Wave 1 B4 and
Wave 2 main on the bot-corpus mechanism:

- **Wave 1 B4** (3000 steps, B_track_d_xref.md): C4 ranked "small
  absolute magnitude" — bot corpus gradient-pull share 0.092 mean,
  smallest of three sources.
- **Wave 2 main** (47642 steps, M1 above): C4 confirmed as DOMINANT
  failure mechanism via staleness amplification.

Both findings are correct AT THEIR TIME WINDOW. B4 measured
gradient-pull share at step 0-3000 when the corpus is in-
distribution; the static-staleness mechanism doesn't yet bite. Wave 2
main captured the 20k+ horizon where the model has drifted off the
static distribution and the 30% bot-batch share becomes off-
distribution noise. Same channel, different time windows, two
metrics measuring different aspects (instantaneous pull vs cumulative
distributional decay). The Track D ranking is NOT inverted — it is
extended with a time dimension B4 did not measure.

### Falsified Hypotheses Register additions

- **Static bot corpus alone is a sufficient anti-colony anchor for
  sustained training past peak fit point.** FALSIFIED by Wave 2:
  21,899-position static SealBot-vs-v6 corpus held the colony
  attractor at bay through step 20k (33% SealBot peak); past step 20k
  the model's policy drifted off the corpus distribution and the
  30% bot batch share became off-distribution noise. Future runs need
  dynamic regeneration of the bot corpus against the current model.

- **alt V_spread + dual-bank canary alone is a sufficient gate for
  real-run quality.** FALSIFIED by Wave 2 (L50): alt V_spread stayed
  +0.18–0.30 throughout 46k steps (well above +0.07 sustained gate)
  while wr_sealbot collapsed 33% → 5%. Value-head discrimination on
  fixed held-out banks is not a sufficient proxy for actual
  selfplay/eval performance. Future runs must ALSO gate on sealbot WR
  sliding-window trajectory (Wave 3 hard-abort triggers).

### L50 — alt-bank V_spread is necessary but not sufficient for sustained eval quality

**Rule.** Held-out value-head discrimination metrics (T3 + alt bank
V_spread) can both PASS while training-loop policy quality
deteriorates. The metric measures value-head separation on a fixed
position bank; it does not capture the model's *policy* on actual
self-play / eval games.

**Why.** Wave 2 evidence: alt V_spread sustained +0.18–0.30 across
46k steps (well above the +0.07 sustained gate) yet wr_sealbot
collapsed 33% (step 20k peak) → 5% (step 40k). The value head
remained discriminative on the static bank while the policy
deteriorated tactically on live SealBot games.

**How to apply.** Sustained run gates must include sliding-window
SealBot WR trajectory tracking as a hard-abort lever, not advisory.
alt-bank V_spread remains useful as an early-warning signal but
cannot stand alone. L48 framing (alt is the corpus-grounded
reference vs T3 synthetic) refined: alt-bank is corpus-grounded but
only for the value-head sub-task, not the policy-head sub-task.

### L51 — Bot-corpus staleness predicted as Track D C4 — Wave 2 confirms

**Rule.** A static bot corpus regularization signal has an effective
training-lifetime bounded by the ratio (model drift rate) ÷
(corpus-replay rate). Past that lifetime, the signal becomes
off-distribution noise and contributes negatively rather than as
anti-colony anchor.

**Why.** Wave 2 lever stack used a 21,899-position SealBot-vs-v6
corpus (`bot_corpus_s178_sealbot_vs_v6.npz`) at `bot_batch_share=0.30`.
By step 20k the corpus had been re-encountered ~70 times (batch 256
× 30% × 20k steps = 1.54 M bot positions / 21,899 corpus size); the
model had imprinted the SealBot tactical distribution → peak 33% WR.
Past step 20k the model's selfplay policy drifted off the corpus's
position distribution → the 30% bot slice became increasingly
off-distribution → fit decayed monotonically to 5% by step 40k.

**How to apply.** ANY future sustained run with a static bot corpus
must specify (a) maximum effective training-lifetime, (b) refresh
trigger, (c) refresh cooldown. Wave 3 `bot_corpus_refresh.enabled=true`
is the design-named remedy (refresh against current EMA model on
each best_model_promotion + 5pp WR delta, cooldown 5k steps,
max_regens 19 per 100k run).

### L52 — Per-class target temperature on selfplay slice over-softens late tactical learning when bot corpus drifts off-distribution

**Rule.** Per-class CE-target softening on the SELFPLAY slice
reduces gradient magnitude on the model's strongest moves. This is
benign while selfplay positions are dominated by early-game /
corpus-class shapes, but degrades tactical sharpness once selfplay
buffer accumulates late-game shapes the model is sharpening on
itself. Combined with a stale anti-colony signal (L51), the model
loses tactical strength faster than the static bot corpus can
correct.

**Why.** Wave 2 used T_colony=1.5 with `selfplay_sample_rate=0.20`
on selfplay slice. M2 mechanism: softening visit-count CE on
colony-classified selfplay rows attenuates the model's own best-move
signal. In the bot-corpus-fresh window (step 0-20k) this is balanced
by the 30% bot corpus pulling toward SealBot's tactical distribution.
Past corpus exhaustion (step 20k+ per L51), the per-class temp
softening is the dominant remaining anti-colony pressure on selfplay
rows — and it's tactical-softening, not target-replacement, so it
actively de-sharpens rather than re-pointing.

**How to apply.** Per-class target temperature should apply only to
slices where the model is NOT learning its own play (pretrain + bot
slices). Drop the selfplay slice from per-class temp. Wave 3
implementation adds `apply_to_selfplay: false` flag to the existing
`apply_to_pretrain: true` companion. Combined with Wave 3 refresh
hook (L51), the levers separate: bot corpus stays current via
refresh (anti-colony pressure), per-class temp targets only static-
distribution rows (pretrain + bot), selfplay slice learns its own
play unmodified.

### Wave 2 canonical deliverable preservation

`reports/track_b_main/checkpoints/checkpoint_00020000.pt` is the
project-record SealBot WR 33% snapshot. Preserved as historical
reference even if Wave 3 produces a stronger sustained model. Not
promoted, not anchor candidate — Wave 2 mechanism-evidence ckpt.
Operator decides post-Stage-1 whether to archive to
`reports/canonical_models/wave2_step20k_peak33pct.pt` for long-term
retention.

`best_model.pt` (step-30k promotion, 11% WR degradation point) +
`checkpoint_00010000/30000/40000.pt` retained at
`reports/track_b_main/checkpoints/` as the full trajectory ladder.

### Open handles (Wave 3 carries forward)

- Refresh hook activation per L51 (design at
  `docs/designs/s179c_bot_refresh_hook.md`)
- Sliding-window SealBot WR hard-abort gate per L50
- Per-class temp scope revision per L52 (`apply_to_selfplay: false`)
- REAL_RUN_RECIPE v2 update with new PRIMARY success criterion
  (rolling-mean SealBot WR ≥20% sustained 30k-50k vs current
  RR-G3+RR-G4 single-step gates)
- Compute budget ~$5 + 2-3 days dev (same as Wave 2 allocation)

### Branches

| branch | commit | status |
|---|---|---|
| `phase4.5/s181_wave2_ema` | `6ef4aad` | cherry-picked to master Stage 1C (commit `95624af`) |
| `phase4.5/s181_wave2_b4_analysis` | `814c4ef` | cherry-picked to master Stage 1C (commits `e973d1f`, `6fab8c9`) |
| `phase4.5/s181_wave2_lever_vba_selfplay` | `54bd9da` | KEEP as historical reference (origin push pending operator); audit docs cherry-picked to master Stage 1C (commit `96562fa`) |

Wave 2 lever code (per_class_target_temperature.py + variant configs)
stays on the lever branch — Wave 3 revises scope (L52) so the code
is reference-only.

---

## §S181-AUDIT Wave 3 — refresh hook + per-class temp scope flip plateau-then-collapse

**Status.** Wave 3 lever stack (v7full anchor + EMA + bot corpus refresh
hook per L51 + per-class target temp scope flip per L52 + L50 sliding-
window WR hard-abort gate) ran 50 000 steps across 3 sessions, then
hit L50 Trigger C auto-abort @ step 50k after wr_sealbot collapsed to
**2% @ step 45k**. Wave 3 plateaued the model at 16-25% wr_sb across
steps 10k-30k (5 evals) — a softer trajectory shape than Wave 2's
peak-then-monotonic-decline — but hit the SAME end-state colony
attractor (anchor 70% / sealbot 2% / 68pp divergence). Refresh hook +
per-class scope flip are necessary-not-sufficient against the
colony-attractor capture. Wave 4 design with structural levers (2-stone
aux / WDL / PSW) required. **Phase 4.5 remains BLOCKED.**

### Run identity

| field | value |
|---|---|
| host | vast.ai 5080 (ssh6.vast.ai:13053) |
| branch | `phase4.5/s181_wave3_design` 4435f4d (pushed to origin) |
| variant | `v7_wave3_main` (with mid-run L50 widening @ 4435f4d) |
| anchor | `bootstrap_model_v7full.pt` SHA `568d8a33…d61e8e98` |
| encoding | v7full (single-window 19×19, 8 planes) |
| iterations target | 100 000; **actual 50 000** (L50 Trigger C auto-fire) |
| sessions | 3 (s1: cadence-fix-stop @ 11 746 / s2: L50-B-fire @ 40 000 / s3: L50-C-fire @ 50 000) |
| total wall | ~32 h |
| total cost | ~$8.30 (slight overrun of $8 hard cap on operator "continue to 50k regardless" override) |
| total games | ~25 000 |
| canonical archive | `reports/track_b_main_wave3/` (353 MB) |
| key audit doc | `audit/structural/wave3_real_run_analysis.md` (452 lines) |

Wave 3 lever stack vs Wave 2:
- EMA + v7full anchor (Wave 2 baseline, kept)
- **Bot corpus refresh hook ACTIVATED** per L51 (Wave 2 was disabled)
- **Per-class target temp SCOPE FLIP** per L52 (Wave 2 was selfplay-only;
  Wave 3 is pretrain+bot only, selfplay slice sharp tactical CE preserved)
- **L50 sliding-window WR hard-abort gate** active (Wave 2 didn't have this)

### SealBot WR trajectory (peak-and-collapse)

| step | wr_sb | wr_anc | wr_best | c_sb | promoted | elo |
|---:|---:|---:|---:|---:|:---:|---:|
| 5 000 | 16.0% | 60.0% | 52.0% | 2 | ✗ | 435 |
| **10 000** | **25.0%** | 62.0% | 58.0% | 3 | ✗ | 417 |
| 15 000 | 18.0% | 46.0% | 59.0% | 7 | ✗ | 428 |
| 20 000 | 16.0% | 68.0% | 55.0% | 3 | ✗ | 407 |
| 25 000 | 23.0% | 59.0% | 66.0% | 8 | **✓** | 377 |
| 30 000 | 20.0% | 65.0% | 65.0% | 11 | **✓** | 313 |
| 35 000 | 10.0% | 61.0% | 57.0% | 6 | ✗ | 212 |
| **45 000** | **2.0%** | **70.0%** | **69.0%** | 2 | **✓** | 397 |

Peak step 10k 25% (vs Wave 2 peak step 20k 33%). Plateau 16-25%
steps 10k-30k. Catastrophic collapse 23%→2% over steps 25k→45k. SAME
end-state attractor as Wave 2 (anchor 70% / sb ≤5%).

### L34 anchor↑/sealbot↓ divergence (3 fires)

| transition | anchor Δ | sealbot Δ | L34? |
|---|---:|---:|:---:|
| 15k → 20k | +22 pp | -2 pp | **YES (1st)** |
| 25k → 30k | +6 pp | -3 pp | **YES (2nd)** |
| 35k → 45k | +9 pp | -8 pp | **YES (3rd)** |

3 L34 fires across trajectory. Step 45k = final form: 70%/2% = 68pp
divergence (Wave 2 peak was 65pp at step 40k). Same colony-attractor
end-state.

### L50 hard-abort fires (mechanism validated end-to-end)

- **Fire #1 — Trigger B @ step 40 000** (session 2): "SealBot WR 10.0% <
  peak 23.0% × 50% past step 25,000 — Wave-2-style collapse". Operator
  widened `wr_collapse_from_peak_ratio: 0.5 → 0.25` and resumed from
  ckpt_00040000.pt.
- **Fire #2 — Trigger C @ step 50 000** (session 3): "SealBot WR 2.0% <
  5% past step 15,000 — §S180b-style early death". Clean session_end.

Both fired AS DESIGNED. Stage 2B's L50 gate is validated end-to-end.

### Refresh hook validation (4 cycles end-to-end)

| cycle | requested step | swap step | n_pos_after |
|---:|---:|---:|---:|
| 1 | 5 000 | 10 000 | 6 724 |
| 2 | 15 000 | 20 000 | 7 453 |
| 3 | 30 000 | 35 000 | 8 006 |
| 4 | 45 000 | 50 000 | 9 932 |

Mechanism works perfectly. Atomic NPZ swap + hot-reload + corpus growth
across cycles (6.7k → 9.9k positions as EMA model strengthens + sustains
longer games). Reload_sec ≤ 6s each. The mechanism is NOT defective —
the verdict is that it's INSUFFICIENT against the attractor.

### Mechanism diagnosis (M1 + M2 + M3 compound)

**M1 — Refresh hook insufficient (high).** 30% bot-batch-share + dynamic
EMA-anchored fresh corpus IS overpowered by the 70% selfplay-buffer
share that drives the gradient toward colony shapes. Mechanism works,
share is wrong.

**M2 — Per-class temp scope flip insufficient (high).** Wave 2's M2
mechanism (per-class temp on selfplay slice over-softens tactical CE)
WAS addressed by L52 scope flip (apply_to_selfplay: false). But the
attractor operates ABOVE the per-class CE level. Wave 3 STILL collapsed.
Preserving selfplay CE sharpness ≠ preserving the right policy direction.

**M3 — best_model rotation amplifies the attractor (moderate).** Promotion
at step 25k, 30k, 45k. Step 45k promotion happened WHILE wr_sb crashed
to 2% — the promotion criteria (wr_best ≥ 0.55 + bootstrap_floor 0.45)
reinforce colony exploitation via positive feedback on internal metrics
that don't penalize anchor-exploit.

### L53 — Refresh hook + per-class temp scope flip insufficient against the colony attractor

**Rule.** A 30%-batch-share dynamic-refreshed bot corpus + per-class
CE softening scoped to static (pretrain+bot) rows is INSUFFICIENT to
prevent the colony-attractor capture in v7full sustained training.

**Why.** Wave 3 ran 50k steps with both levers active. wr_sb peaked at
25% (step 10k), oscillated 16-25% across steps 10k-30k, then collapsed
to 2% by step 45k. Same end-state attractor as Wave 2 reached via
different trajectory shape. Refresh hook cycles complete cleanly but
the fresh corpus signal is overpowered by selfplay-dominated gradient.

**How to apply.** Wave 4+ must employ STRUCTURAL interventions, not
RATIO interventions. Candidates: 2-stone opponent-reply aux head
(addresses M3), WDL value-head migration (changes value-target
structure), KL-weighted buffer writes, PSW (s179b parked design),
class-weighted gradient scaling (vs CE softening). The refresh hook +
per-class temp scope flip should REMAIN as defensive substrate but a
fundamental mechanism change is needed atop them.

### L54 — Trigger B peak×0.5 threshold catches collapse late

**Rule.** L50 Trigger B fires AFTER trajectory drops to half its peak —
by then the collapse is already deep.

**Why.** Wave 3 fired Trigger B at step 40k drain (wr_sb 10%, peak 23%,
threshold 11.5%) — but the trajectory crashed 23% (step 25k) to 10%
(step 35k) BEFORE the fire. ~15k steps of collapse training before
auto-abort. Trigger C fired even later (step 50k after wr_sb hit 2%).

**How to apply.** Future runs should add a derivative trigger: "rolling-
mean WR drops ≥ Xpp across 2 consecutive evals past step 20k". X=5pp
would have caught Wave 3 at step 35k (rolling-mean 21.5% → 15% = 6.5pp
drop). Or tighten Trigger B's peak ratio earlier (ratio 0.7 + min_step
20k catches earlier).

### L55 — Wave 3 plateau-then-collapse is SAME L34-attractor with delay, not new mechanism

**Rule.** A long sustained mid-WR phase (plateau) is NOT evidence of
attractor break. It's consistent with the SAME attractor + lever-stack-
induced inertia before the wr_sb-measurable collapse.

**Why.** Wave 2 and Wave 3 end at the SAME L34 signature (anchor ~70%
/ sealbot ≤5% / 65pp divergence). The end-state is identical. Wave 3
lever stack delayed wr_sb-measurable manifestation by ~10-15k steps but
didn't break the attractor.

**How to apply.** Wave 4+ design should NOT chase a different trajectory
shape (longer plateau ≠ better if collapsing at 2%). Design for a
different ATTRACTOR — different value-head / policy-head / value-target
structure with no colony-exploit-of-anchor-bias attractor. WDL, 2-stone
aux, or PSW change WHAT THE MODEL OPTIMIZES, not just the
gradient-share of the existing optimization.

### Falsified Hypotheses Register additions

- **Refresh hook + per-class temp scope flip is sufficient to prevent
  colony-attractor capture in v7full sustained training.** FALSIFIED by
  Wave 3 — wr_sb collapsed to 2% by step 45k despite both levers active.
- **Plateau (long sustained mid-WR phase) is a positive sign of attractor
  break.** FALSIFIED by Wave 3 — model plateaued 16-25% wr_sb across
  10k-30k then catastrophically collapsed to 2%.
- **best_model promotion is a reliable signal of model improvement
  toward Phase 4.5 readiness.** FALSIFIED by Wave 3 — best_model promoted
  AT step 45k (wr_best 69%) WHILE wr_sb crashed to 2%. Promotion gate
  rewards anchor-exploit, not anti-colony improvement.

### Wave 3 canonical deliverable preservation

No single project-record snapshot. Candidates:
- `checkpoint_00010000.pt` — Wave 3 peak 25% wr_sb (CI wide; not promoted)
- `checkpoint_00025000.pt` — first promotion + 23% wr_sb
- `checkpoint_00045000.pt` — **colony-attractor reference** (70%/2% — useful
  for Wave 4 mechanism ablation work)

Wave 2's `wave2_step20k_peak33pct.pt` remains the project-record peak
SealBot WR snapshot. Wave 3 archive at `reports/track_b_main_wave3/`
holds the trajectory for future analysis.

### Wave 4 escalation path (operator decides priority)

Per dispatcher §5C routing (PRIMARY all FAIL → "Mechanism wrong"):

1. **2-stone opponent-reply aux head** (V-B-D conditional, never tested
   in sustained context). Highest priority — forces trunk to discriminate
   on-policy reply patterns, addresses M3.
2. **WDL value-head migration** (parked since §S178; A2 falsified
   arch-only fix, BUT Wave 3 shows loop-side levers ALSO insufficient
   → arch + loop combined hypothesis worth testing).
3. **PSW (Policy Surprise Weighting)** — design `s179b` parked. Penalizes
   high-KL transitions in selfplay buffer writes.
4. **Class-weighted gradient scaling** (different mechanism class than
   per-class CE softening Wave 3 already tested).

L53/L54/L55 + `audit/structural/wave3_real_run_analysis.md` are the
Wave 4 design starting point.

### Cross-references

- `audit/structural/wave3_real_run_analysis.md` — full analysis (452 lines)
- `audit/structural/wave3_smoke.md` — Stage 3 smoke WS-A PASS-WITH-NOTES
- `audit/structural/wave3_launch_readiness.md` — Stage 2 close-out
- `audit/structural/wave2_real_run_analysis.md` — Wave 2 baseline (L50/L51/L52)
- `audit/structural/REAL_RUN_RECIPE.md` — Wave 3 success criteria
- `docs/designs/s179c_bot_refresh_hook.md` — refresh hook design (validated)
- `docs/designs/s179b_policy_surprise_weighting.md` — PSW (Wave 4 candidate)
- `reports/track_b_main_wave3/` — 353 MB local archive (ckpts + logs + events JSONL)

### Branches (Stage 4 → Stage 6 disposition)

| branch | commit | status |
|---|---|---|
| `phase4.5/s181_wave3_design` | `4435f4d` | active; pending Stage 6 REVIEW → master merge |

Wave 3 design branch has 10 commits ahead of master: Stage 2A refresh
hook + Stage 2B WR hard-abort gate + Stage 2C per-class temp scope +
Stage 2D REAL_RUN_RECIPE v2 + Stage 2E launch readiness + Stage 3A
smoke variant + Stage 3C smoke audit + Stage 4A main variant + mid-run
yaml widening + this sprint-log entry.

---

## §S181-AUDIT Wave 4 — subtract-the-variable + multi-aux suite

**Status.** Both Track 4A (subtract bot mix) and Track 4B (multi-aux
density on Wave 3 lever stack) ran sustained sessions on vast 5080
and SIGINTed early when verdict patterns landed. Track 4A peaked
at step 5k 19% and collapsed to 11% by step 10k (W4A-B verdict
LITERAL). Track 4B peaked at step 10k 23% and collapsed to 11% by
step 15k (W4B-B verdict LITERAL). Both reached ~11% wr_sealbot by
step 12-15k via different paths — bot mix removal accelerated the
collapse (Track 4A), multi-aux density delayed it ~5k steps but
didn't prevent it (Track 4B). **Colony-attractor mechanism lives
downstream of all config + density levers tested in Waves 1-4.**
Wave 5 strategic reckoning required: value-target propagation
(TD-λ / n-step), WDL 3-class head, or game-theoretic regularization.
**Phase 4.5 remains BLOCKED.**

### Run identity

| field | Track 4A | Track 4B |
|---|---|---|
| branch | `phase4.5/s181_wave4_subtract` 5b7f85e | `phase4.5/s181_wave4_multiaux` 0523147 |
| variant | `v7full_baseline_minus_bot` | `v7full_wave4_multiaux_w4ac` |
| anchor | `bootstrap_model_v7full.pt` SHA `568d8a33…` | (same) |
| iter target / actual | 60 000 / ~12 000 (SIGINT) | 60 000 / ~15 500 (SIGINT) |
| run_id | `1b8c649a…` | `8e4568c6…` |
| wall | ~7 h | ~9.5 h |
| spend | ~$3 | ~$3 |

Total Wave 4 spend ~$6 (under $7 cap).

### Track 4A subtract-the-variable verdict (W4A-B LITERAL)

Three DELTAs vs Wave 3 main: `bot_batch_share 0.30→0.0`,
`bot_corpus_refresh.enabled true→false`, `per_class_target_temperature.enabled
true→false`. Preserved Wave 2-3 hygiene (EMA + entropy 0.005 +
eta_min 5e-4 + PR-B param-group + L50 hard-abort STRICT thresholds).

| step | wr_sb | wr_anchor | wr_best | col_anchor | promoted | elo |
|---:|---:|---:|---:|---:|:---:|---:|
| 5 000 | **19.0%** | 57.0% | 59.0% | 40 (70%) | ✗ | 448 |
| 10 000 | **11.0%** | 53.0% | 68.0% | 40 (75%) | ✓ | 382 |

Δ 5k→10k: wr_sb -8pp, wr_anchor -4pp, wr_best +9pp, elo -66. BOTH
anchor and sealbot DECLINING (broader value-head degradation, NOT
classic L34 anchor↑/sealbot↓). Colony share 70→75% creeping. Audit
doc `audit/structural/wave4_track_a_subtract.md`.

### Track 4B multi-aux density verdict (W4B-B LITERAL)

Parent v7_wave3_main (KEEPS bot mix + refresh hook + per-class temp).
Multi-aux density bumps in training.yaml: sigma2 Huber 0.1 (NLL
formulation diverged at σ²→0; switched to Huber-on-squared-error per
4B-impl-5 d80de72), ownership 0.1→0.2, threat 0.1→0.2, ply_index
0.0→0.1 (NEW 4B-impl-3 head).

| step | wr_sb | wr_anchor | wr_best | col_sb | col_anchor | promoted | elo |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 5 000 | 20.0% | 64.0% | 62.0% | 4 (20%) | 50 (78%) | ✓ | 465 |
| 10 000 | **23.0%** | 62.0% | 61.0% | 3 (13%) | 41 (66%) | ✓ | 362 |
| 15 000 | **11.0%** | (~62.5 partial) | — | 5 (45%) | — | — | — |

Δ 5k→10k: wr_sb +3pp, col composition decreasing (78→66 anchor,
20→13 sb) — initially looked W4B-A-trending. Δ 10k→15k: wr_sb -12pp,
col_sb composition jumped 13→45% (homogenization confirmed). Audit
doc `audit/structural/wave4_track_b_sustained.md`.

### Cross-run shape comparison

| step | Wave 3 wr_sb | Track 4A wr_sb | Track 4B wr_sb |
|---:|---:|---:|---:|
| 5 000 | 16% | 19% | **20%** |
| 10 000 | **25%** (peak) | 11% (collapsed) | 23% (peak) |
| 15 000 | 18% | (SIGINT step 12k) | 11% (collapsed) |

Track 4B's peak (23%) ≈ Wave 3's peak (25%). Track 4A's peak (19%)
is the only data point where "no bot mix" started ahead before
accelerating its own decline. Multi-aux density configurations
shift WHEN the collapse happens, not WHETHER it happens.

### Lessons (L56-L60 banked)

**L56 (Wave 4 Track 4A).** Bot mix is NOT the load-bearing failure
variable in the colony-attractor mechanism. Removing it produces a
FASTER decline than keeping it (Track 4A step-10k 11% vs Wave 3
step-10k 25%). The §S178+ hypothesis ("bot mix introduces the
colony") is inverted or non-monotonic: bot mix may DELAY or attenuate
the colony pattern; removing it accelerates the mechanism.

**L57 (Wave 4 Track 4A).** Track 4A peaked at step 5k (19%) not step
10k like Wave 3 (25%). Without bot mix, the trainer's
`training_steps_per_game=2.0` rate forces heavy reuse of early
selfplay buffer, causing premature exposure to bootstrap distribution
drift. Recipe sensitivity to selfplay supply rate.

**L58 (Wave 4 Track 4B).** Multi-aux density (sigma2 Huber +
ownership 0.2 + threat 0.2 + ply_index 0.1) DELAYS the colony
attractor by ~5k steps but does NOT prevent it. Peak shifts from
Wave 3 step 10k 25% to Track 4B step 10k 23% (essentially same
magnitude). The KataGo aux-density hypothesis is FALSIFIED for
HeXO's colony attractor.

**L59 (Wave 4 close).** The colony attractor mechanism is INSENSITIVE
to: bot mix presence/absence, refresh hook, per-class target
temperature, multi-aux density, EMA/entropy/PR-B hygiene. The
mechanism lives DOWNSTREAM of all currently-tested levers — in the
training objective itself (value head + target propagation), trainer
step structure, or MCTS+selfplay interaction. Wave 5 must operate
at this boundary.

**L60 (Wave 4 Track 4B).** Colony composition (col_sb%, col_anchor%)
trajectory SHAPE predicts the W4B verdict before raw wr_sb does.
Track 4B step 5→10k showed composition DECREASING (positive signal),
but step 15k composition shot up (sb 13→45%) AS the wr_sb decline was
confirmed. Watch composition deltas as leading indicator of pattern
homogenization. (Per operator's colony-framing memo
`feedback_colony_is_meta_not_kill_signal`: colony presence is not a
kill signal; colony GROWTH + L34 + stride5 spike is.)

### Multi-aux infrastructure SHIPPED (regardless of verdict)

Even though the science hypothesis was falsified, the Wave 4 implementation
work is permanently useful and landed on master:

- `4B-impl-5` d80de72: sigma2 Huber-on-squared-error formulation
  (replaces Gaussian NLL that diverged at σ²→0); density bumps for
  ownership/threat (0.1→0.2); uncertainty re-enabled (0.0→0.1)
- `4B-impl-1` 4fee9d1: `position_index: u16` field in
  `ReplayBuffer`; HEXB v8 wire format with v7 backward-compat load;
  `sample_batch_with_pos` 8-tuple facade (legacy `sample_batch` 7-tuple
  byte-identical preserved); INV20 contract extended; bench gate PASS
  (n=5 on laptop, +25.79% buffer_push_per_s after hot-path cleanup)
- `4B-impl-3` f3509ce: `ply_index_head` 2-layer MLP + `compute_ply_index_loss`
  Huber on `clamp(position_index / 100, 0, 1)` + end-to-end plumbing
  through `BatchAssemblyResult` → `step_coordinator` →
  `train_step_from_tensors`
- `7c80a2d`: `load_state_dict_strict` benign-missing-keys allowlist
  for `ply_index_head.*` (extensible pattern for future aux heads)
- Tests: 7 new uncertainty loss tests + 7 new ply_index loss tests +
  buffer round-trip + INV20 facade pin extension; 1714+ pytest +
  191 cargo tests green

### Wave 5 strategic reckoning (Task 5 pre-write)

Per dispatcher: "if BOTH Track 4A and Track 4B colony, strategic
reckoning is forced." Both colonyed within 15k steps. Surfaces NOT
yet tested:

1. **Value-target propagation**: terminal z → all positions may be
   fundamentally wrong for HeXO. Try n-step bootstrap or TD-λ.
2. **WDL 3-class softmax** value head (replaces tanh scalar).
3. **Game-theoretic regularization**: explicit anti-colony loss term
   penalizing homogeneous-pattern composition.

Wave 5 scope estimate: ~3 weeks dev + ~$20 vast. Major commitment.
Operator decision pending; parked for now per "Stage 6 close + park
for Wave 5 design" route.

### Sprint-log header pre-written

`## §S181-AUDIT Wave 5 — strategic reckoning + structural target rework`

### Falsified Hypotheses Register additions

- "Bot mix is the load-bearing failure variable in the colony
  attractor mechanism" → FALSIFIED 2026-05-27 by Track 4A (W4A-B).
  Removing bot mix produced FASTER decline.
- "Multi-aux density (KataGo-style diverse aux signal) prevents
  single-attractor lock" → FALSIFIED 2026-05-27 by Track 4B (W4B-B).
  Delayed colony attractor ~5k steps but did not prevent it.

### Forensics

- Branches: `phase4.5/s181_wave4_subtract` (Track 4A), `phase4.5/s181_wave4_multiaux` (Track 4B impl + audit docs)
- Audit docs: `audit/structural/wave4_track_a_subtract.md`,
  `audit/structural/wave4_track_b_sustained.md`,
  `audit/structural/wave4_track_b_multiaux_design.md` (pre-impl design + scope reduction history)
- Stage 6 REVIEW: opus 4.7 fresh-context subagent dispatched 2026-05-27, verdict GREEN
- Commits: 5b7f85e (Track 4A variant) + d80de72/4fee9d1/f3509ce/6e75bfd/5fea1ce/3f0d9cd/7c80a2d/37b216d/0523147 (Track 4B impl + audit chain)

---

## §S181-AUDIT Wave 5 entry — compound-turn investigation

**Status:** OPEN — research/audit session landed; sustained-run verdict
operator-pending. (Placeholder: operator fills run details after the
Wave 5 design session.)

**Premise (per L59 boundary).** Wave 4 close established the colony
mechanism is insensitive to every config / hygiene / aux-density lever;
it lives downstream in the training objective + MCTS/selfplay interaction.
Wave 5 entry tests a structural hypothesis: HeXO's 2-stones-per-compound-
turn is modelled as two *sequential single-stone* decisions, which may
favour colony (order-invariant) over extension (needs coordination).

**Phase 5 read-only pipeline audit (this session, 2026-05-28).** Full
static audit of compound-turn handling across all 7 stages:
`audit/structural/compound_turn_pipeline_audit.md`.

Key results (see audit for citations + the five critical questions):
- Hypothesis's STRONG form FALSIFIED — the engine is **not** order-blind.
  Board/zobrist order-invariant; TT merges `{A,B}`≡`{B,A}`; MCTS Q-flips
  per *turn boundary* not per stone, with a correct 2-ply within-turn
  look-ahead. Genuinely sequential facets: greedy per-stone *commitment*
  (2 fresh searches/turn, no subtree reuse) and *per-ply storage* of the
  intermediate position.
- **Bug found — CF-1:** a first-stone win is scored `terminal_value=-1.0`
  (backup.rs:223-228) because `apply_move` does not flip the player after
  stone 1; the value convention is current_player-perspective, so the
  sign is inverted only for stone-1 wins. Distorts the stone-1 policy
  target (visit mass pushed off winning-cell-first). Mechanism-plausible
  minor colony contributor; causal link NOT established. Fix + A/B
  pre-registered in the audit, NOT yet implemented (read-only scope).
- **CF-2 (inferred-strong):** v6/v7full NN input drops the
  `moves_remaining`/`ply_parity` planes (registry.toml:78) — the value
  head gets no explicit turn-phase signal, mechanism-aligned with the
  L47 value-head discrimination collapse.
- **CF-6 (undetermined):** FPU sign at `mr==1` parents may not honour the
  per-turn flip; needs a `puct_score` unit test to resolve.

**Open question opened:** Q-COMPOUND-TURN in `docs/06_OPEN_QUESTIONS.md`.

**Operator next steps (Wave 5 design, pending):** CF-1 sign-fix unit test
+ A/B; CF-2 value-spread probe with planes 16/17 added; the structural
reckoning options pre-written in the Wave 4 entry (value-target
propagation / WDL value head / anti-colony regularization).

---

## §S182 — perf wave: legal_moves_set capacity fix MERGED

*DISCRIMINATOR: §S182 = Rust-perf wave merge (this entry). The §S178 bot-mix
training line and the Rust-perf waves advance on independent sprint numbers;
§S181 stays reserved for the §S180b code-level-lever training successor.*

**Merged.** `perf/legal-moves-cache-cap` FF-merged to master 2026-05-22.
Tag `perf-legal-moves-cache-cap` at `46fa489`. Perf-fix commit `f8ff7b8`
(+26 LOC `engine/src/board/moves.rs`) — O(1) bbox+ball-area capacity reserve
before the `legal_moves_set` rebuild loop; kills the hashbrown power-of-2
`reserve_rehash` cascade. Closes the §S180b-era CANDIDATE-branch merge-hold.

**Bench gate — cross-host PASS.** Criterion `mcts_sims_cpu_only`,
`--profile profiling`, n=800, median of runs 2+3 after discard-first warm-up:

| host | baseline (master `3146144`) | post (`46fa489`) | sims/s Δ |
|---|---|---|---|
| laptop 8845HS | 2.4936 ms | 1.4595 ms | **+70.9%** |
| vast 9900X+5080 | 1.9974 ms | 1.2001 ms | **+66.4%** |

Cross-host gap 4.5pp — inside the ±5pp merge gate. Uniform across sizes
(vast +62.4% / +68.5% / +66.4% at n=100/400/800; laptop ~+70% all sizes).
The L3-cache sensitivity concern (8845HS 16 MB vs 9900X 64 MB) did not
materialize — the mechanism is an allocation-pattern fix, not cache-
residency, so it is hardware-portable. Raw: `investigation/rust-perf-
2026-05-20/raw/vast/` (gitignored).

**Mechanism confirmed.** Laptop perf report: `hashbrown::reserve_rehash`
31.4% → 1.2%, dropped out of top-5. Residual post-fix top-3 self-time:
`legal_moves_set` 41.8%, `expand_and_backup_single` 28.0%, `select_leaves`
12.0%.

**Successors.** §S183 — quick-win micro-opt wave (P1 `sqrt` hoist + F1/F2
`mul_add` + A1/A4 atomic-ordering relax), branch `perf/quick-wins-mcts`.
§S184 — `legal_moves_set` rebuild-cost reduction targeting the residual
41.8% self-time (planning doc `09_rebuild_fix_plan.md`).

---

## §S183 — perf wave: MCTS quick-win micro-opt bundle MERGED

*DISCRIMINATOR: §S183 = Rust-perf quick-win wave (this entry), the §S182
successor. Independent of the §S178 bot-mix training line; §S181 stays
reserved for the §S180b code-level-lever training successor.*

**Merged.** `perf/quick-wins-mcts` FF-merged to master 2026-05-22, commit
`4781fae` (off master `f9ae886`). Five mechanically-distinct hot-path edits,
one commit, from `investigation/rust-perf-2026-05-20` ranks 1/2/3/6/14:

- **P1** `selection.rs` — hoist `parent_n.sqrt()` out of `puct_score` into
  the `pick_best_puct` caller as `sqrt_parent_n`; loop-invariant across the
  K≤192 children, was recomputed per child. Signature + all callers updated.
- **F1** `selection.rs` — FPU `parent_q − fpu_reduction·sqrt(mass)` →
  `(−fpu_reduction).mul_add(sqrt, parent_q)`. The clippy-flagged
  `suboptimal_flops` site is the FPU expr, not the PUCT `q+u` term (`q +
  num/den` is add-of-division, not FMA-able).
- **F2** `policy.rs` ×5 — `a·b+c` → `a.mul_add(b,c)` (v_mix, 3× logit,
  Dirichlet mix).
- **A1** `worker_loop/inner.rs` ×8 — `running.load` `SeqCst` → `Relaxed`;
  `running` is a payload-free stop-signal flag, `handles.join()` after
  `stop()` supplies the real happens-before.
- **A4** `inner.rs` — `positions_generated.fetch_add` `SeqCst` → `Relaxed`;
  monotonic counter, reader was already `Relaxed`.

**Bench gate — cross-host.** Laptop 8845HS inconclusive (protocol +1.33%
n=800 but ~3.8% run-spread swamps the signal). Vast 9900X+5080 (lower noise
floor) resolved it — criterion `mcts_sims_cpu_only`, `--profile profiling`,
discard run 1, median runs 2+3:

| n | baseline (`f9ae886`) | post (`4781fae`) | sims/s Δ |
|---|---|---|---|
| 100 | 111.88 µs | 110.29 µs | +1.4% |
| 400 | 567.0 µs | 549.4 µs | +3.2% |
| 800 | 1.1817 ms | 1.1686 ms | **+1.12%** |

All three sizes positive; n=800 +1.12% clears the ≥1% acceptance gate. A
thin margin (baseline n=800 spread ~3.8%) — cross-host all-sizes-positive
consistency is what carries it past the inconclusive laptop result.

**Verification.** `cargo test` 282 pass / 0 fail. clippy `suboptimal_flops`
13 → 7 (−6: F1 + 5×F2). Fresh-context REVIEW subagent verdict MERGE-READY,
all 7 checks pass (scope, P1/F1/F2/A1/A4 correctness, clippy delta, bench,
hygiene).

**Successor.** §S184 — `legal_moves_set` rebuild-cost reduction (residual
41.8% self-time). Plan `09_rebuild_fix_plan.md`: recommended strategy δ
(FxHashSet-free sorted-Vec representation), branch
`perf/legal-moves-rebuild-reduce`, ≥20% n=800 acceptance gate. After §S184
merges, write the perf-wave Mechanism Lesson (flamegraph-first for
throughput, static-first for correctness).

---

## §S184 — perf wave: legal_moves_set sorted-Vec δ ABORTED

*DISCRIMINATOR: §S184 = Rust-perf wave (this entry), the §S183 successor.
Independent of the §S178 bot-mix training line; §S181 stays reserved for the
§S180b code-level-lever training successor.*

**Aborted — not merged.** Branch `perf/legal-moves-rebuild-reduce` (commit
`194b5a0`, off master `e5c2b0a`) implemented `09_rebuild_fix_plan.md`
strategy δ — swap the per-Board `legal_moves_set` cache from
`FxHashSet<(i32,i32)>` to a sorted-deduped `Vec<(i32,i32)>`, targeting the
residual 41.8% `legal_moves_set` self-time left after §S182.

**Bench gate — vast 5080/9900X.** Criterion `mcts_sims_cpu_only`,
`--profile profiling`, discard run 1, median runs 2+3:

| n | baseline `e5c2b0a` | post δ `194b5a0` | sims/s Δ |
|---|---|---|---|
| 100 | 111.36 µs | 168.43 µs | **−33.9%** |
| 400 | 554.85 µs | 822.22 µs | **−32.5%** |
| 800 | 1.16825 ms | 1.73035 ms | **−32.5%** |

All sizes regressed, every run, p<0.05. Decision gate (Negative → Abort):
branch reverted, master unchanged at `e5c2b0a`.

**Mechanism (post-mortem `11_s184_postmortem.md`).** IMPL was correct — the
regression is inherent to δ. The rebuild loop `push`es every non-occupied
ring cell *including duplicates* from overlapping radius-5 balls; in a dense
leaf board every interior cell is covered by ~7 stones, so the pre-dedup Vec
is a several-× blow-up. δ then `sort_unstable()`s that *blown-up* array —
`O(N log N)` on the pre-dedup `N`, not the ≤~200 deduped count the plan
assumed. The `FxHashSet` it replaced deduplicated *inline at insert* and
never exceeded the unique count. δ traded cheap-hash-with-free-dedup for
cheap-push + expensive-sort-of-bloated-array. +48% rebuild → −32.5% sims/s.

**Falsified.** Plan §3 claim *"the O(n log n) sort is cheaper than the
removed hashbrown insert work"* — false by ~48%. δ swapped `FxHashSet::insert`
(hash + bucket write, with **free inline dedup**) for `Vec::push` +
`sort_unstable` + `dedup` on a pre-dedup array several× larger than the
deduped count (overlapping radius-5 balls); the sort on the blown-up `N`
costs more than the insert it replaced. The §S185 flamegraph confirms
`FxHashSet::insert` IS the dominant rebuild cost (56.8% of `legal_moves_set`)
— δ replaced the #1 hot op with something worse.

**Lesson (L39).** δ's failure was a **fix-design error, not a mechanism
miss**. The §07 static review correctly identified `FxHashSet::insert` as
the dominant rebuild cost (§S185 flamegraph: 56.8%). The plan's error:
assuming a representation swap to `Vec::push`+`sort`+`dedup` would beat it —
but overlapping-ball duplicates blow the pre-dedup array up several× and the
`O(N log N)` sort on that costs more. Lesson: when a fix *replaces* a hot
operation rather than *eliminating* it, the replacement's cost must be
modeled, not assumed cheaper — and only a bench (not a flamegraph) catches a
bad replacement. β is preferred precisely because it *eliminates* the
rebuild. (The §S184 post-mortem's interim guess that the residual was
`cells.contains_key` was itself refuted by the §S185 flamegraph — see
`13_s185_plan.md` §4.)

**Successor.** §S185 — laptop flamegraph (`perf` DWARF call-graph, 72k
samples) localized the 44% `legal_moves_set` self-time: `FxHashSet::insert`
56.8%, `cells.contains_key` 27.7%, `reserve` 5.6%, ring-iteration 9.9%.
84.5% (insert+probe) is pure per-leaf redundant rebuild. Representation
swaps are dead (δ falsified); the only surviving strategy is **β —
incremental delta maintenance** of `legal_cache` through
`apply_move`/`undo_move`, which deletes the rebuild and *both* costs (~37%
whole-program). Plan `13_s185_plan.md` = SPEC-BETA: per-cell `u16` coverage
map for the union-coverage hazard, ~180–270 LOC / 4 engine files,
debug-mode recompute-and-assert canary, ≥20% n=800 gate, branch
`perf/legal-moves-incremental`, IMPL = sprint §S186. The perf-wave Mechanism
Lesson stays unwritten until §S186 resolves.

---

## §S186 — perf wave: incremental legal-moves β ABORTED, arc closed

*DISCRIMINATOR: §S186 = Rust-perf wave (this entry), the §S184/§S185
successor and the final §09 perf-investigation strategy. §S181 stays
reserved for the §S180b code-level-lever training successor.*

**Aborted — not merged. Perf-investigation arc CLOSED.** Branch
`perf/legal-moves-incremental` (commit `1cae62f`, off master `70abacb`)
implemented `13_s185_plan.md` strategy β — replace the per-leaf full
rebuild of `legal_moves_set` with incremental delta maintenance through
`apply_move`/`undo_move`, backed by a per-cell `u16` coverage map
(`legal_cov`). Post-review cleanup folded into the commit: `UnsafeCell`→
plain field (β made the wrapper vestigial; `Board` now auto-`Sync`) +
stale `legal_cache`/`cache_dirty` doc-comment fixes.

**Bench gate — cross-host.** Criterion `mcts_sims_cpu_only`,
`--profile profiling`, discard run 1, median runs 2+3:

| n | vast base | vast β | vast Δ | laptop base | laptop β | laptop Δ |
|---|---|---|---|---|---|---|
| 100 | 111.6 µs | 203.2 µs | −45.1% | 134.2 µs | 248.6 µs | −46.0% |
| 400 | 548.6 µs | 1088.9 µs | −49.6% | 721.6 µs | 1305.4 µs | −44.7% |
| 800 | 1.145 ms | 2.268 ms | −49.5% | 1.483 ms | 2.776 ms | −46.6% |

β roughly halves MCTS throughput — every size, every run, both hosts,
tight CIs. Decision gate Negative → abort, branch deleted, master
unchanged at `70abacb`.

**Mechanism.** IMPL was correct — the §S186 debug canary
(recompute-and-assert in debug builds) stayed green across 282 tests;
REVIEW verdict MERGE-READY-pending-bench. β is a *cadence* error: it
de-amortizes the once-per-leaf rebuild onto every descent step.
`apply_move` (×depth per sim) now walks a 91-cell radius ball;
`undo_move` (×depth) walks two. ~3× the ball-walk work of the rebuild it
replaced, moved onto the hot descent path that previously only flipped a
`cache_dirty` flag. legal-moves work was 44% self-time → ~3× → ~1.9×
total slowdown; the measured +82–98% time matches.

**Falsified.** `13_s185_plan.md` §5 acknowledged β adds apply/undo work
yet estimated a "large but not full 37%" *gain* — wrong by ~85 points.
The plan's model ("rebuild once-per-leaf, an incremental delta must be
cheaper") inverted reality: the delta runs once per descent *step*, and a
descent has `depth` steps.

**Arc closed.** §09's four strategies are exhausted: α rejected (loses the
intra-leaf O(1) fast path), γ rejected (workload always mutates clones),
δ §S184 FAILED −32%, β §S186 FAILED −50%. The residual 44%
`legal_moves_set` self-time is structural — the once-per-leaf `FxHashSet`
rebuild is already the cheap way to produce a leaf's legal-move set.
§S182 +66.4% was the genuine win; net merged perf gain §S182+§S183 ≈
+68%.

**Lesson (L40 + perf-wave ML).** L40: an incremental fix that relocates a
per-leaf cost onto a per-step path *de-amortizes* — model call-frequency,
not just per-call cost. ML: a flamegraph shows where time is *spent*, not
where it is *recoverable*. The 44% `legal_moves_set` line drew δ and β;
both made MCTS slower (−32%, −50%). A tall profiler line is a question
("is this cost necessary?"), not an answer ("there is 44% to win here").
The honest test before building a fix — does a genuinely cheaper
algorithm *exist*. §S182 took the one real inefficiency (hashbrown rehash
cascade); the rest was structural.

**Successor.** None — perf-investigation arc (§S180→§S186) CLOSED.
Post-mortems `investigation/rust-perf-2026-05-20/11_s184_postmortem.md` +
`16_s186_postmortem.md`; plans `09`/`13` retained as the strategy record.
§S181 remains reserved for the §S180b code-level-lever training successor.

---

## §P5-CT — compound-turn defect probe (CF-1 banked, CF-2 gated)

**Date 2026-05-28.** Discriminate compound-turn-adjacent defects from the
§S150 loop-regression alternative as proximate driver of the colony /
value-head discrimination collapse. Backing audit
`audit/structural/compound_turn_pipeline_audit.md`; investigation report
`reports/investigations/compound_turn_cf1_cf2_20260528.md`.

**ARM 1 — CF-1 BANKED (free, code-complete).** `expand_and_backup_single`
(backup.rs:223) hardcoded the terminal value `-1.0`, which is a sign
**inversion** for a **first-stone win**: `apply_move` (core.rs:528-532) keeps
the player on stone-1 (mr 2→1, no flip), so the `check_win` leaf's
side-to-move is still the winner → should be `+1.0`. Fix derives the sign
from `board.moves_remaining` (`mr==1`⇒+1.0, `mr==2`⇒-1.0). TDD discriminator
in `engine/src/mcts/tests.rs` (`test_cf1_stone1_win_scored_as_win` FAILS on
old code / passes on new; `test_cf1_stone2_win_still_scored_as_loss_to_mover`
proves no Case-B regression). Placed crate-internal because
`expand_and_backup_single` + `Board::cells/last_move` are `pub(crate)`
(brief's `engine/tests/inv*.rs` placement impossible). Independent review:
all 4 checks PASS. Bench (focused `mcts_sims_cpu_only` pre/post, laptop):
−1.4% / +0.8% / +1.2% — within noise, MCTS ≥73k floor PASS.

**ARM 2 — CF-2 GATED (pretrain operator/GPU-pending).** v6 drops turn-phase
planes 16/17 (registry.toml:78); v8 keeps them but a v8 smoke is a
**confounded + argmax-degenerate** CF-2 test → REJECTED. Clean recipe = new
`v6tp` registry entry keeping 16/17 (channels encoder already emits them — no
encode.rs change) + ~30-epoch §150 pretrain. **Sequencing:** run CF-1-fixed
v6 smoke FIRST; escalate to v6tp pretrain only on FAIL/SPLIT.

**CF-4 DEFERRED:** not "free" — needs a per-row ply field across the hot-path
record tuple + PyO3 boundary; ply-index aux already inert (L58).

**Pre-registered verdict (V_spread@2k primary):** PASS = V_spread>+0.10
through 5k AND SealBot WR within 10pp of peak @20k; FAIL = V_spread≤0 by 2k
(L44 sig) → pivot to §S150 hunt; SPLIT = filler-first policy improves but
V_spread collapses → CF-1 banked as correctness-only. **E1 vs E2 UNDECIDED**
pending smoke. Cosine temp stays OFF (L9).

**L61 — a hardcoded terminal sign is a compound-turn trap.** Negamax
terminal values must be derived from the leaf's side-to-move, never
hardcoded, in any engine where a move can fail to flip the player (multi-ply
turns). The `-1.0` constant was correct for every turn-final win and silently
inverted only first-stone wins — a class invisible to single-stone-game
intuition. **Why:** caught by re-deriving the sign from `apply_move`'s
turn-structure, not from the "side-to-move just lost" heuristic. **How to
apply:** any future MCTS terminal-handling edit must case on
`moves_remaining`; pin with a stone-1-vs-stone-2 unit test.

### Phase 5a — productionized CF-1 + CF-2 + CF-4 (2026-05-28)

Operator decision after the cheap arm passed: productionize all three as a
5-stage surgical wave on `master` (zero file-overlap Rust↔Python). Six
commits `fd25a37..fc0308c`: CF-1 sign fix (B1.1), v6tp registry (B1.2), CF-4
ply-index emission through `collect_data` (B1.3), v6tp Python mirror + smoke
config (B2.1), v6tp 10-plane corpus export + pretrain recipe (B2.2), CF-4
pool.py wiring (B2.3).

- **CF-4 was NOT pool.py-only** as the brief implied: `collect_data` carried
  no per-row ply index and no game_id, so the Rust record tuple had to emit
  it (the "not free" cost the probe deferred). Landed via
  `RecordTuple→WorkerResultRow→collect_data` (8→9 arrays). Bench-cold
  (per-move, not per-sim). Inert on the baseline (`ply_index_weight: 0.0`).
- **BENCH-GATE PASS:** controlled A/B on `mcts_sims_cpu_only` — no change at
  n=100/400/800; ≈257k sim/s vs 73k floor.
- **REVIEW PASS** (2 parallel fresh-context): Rust sign/plumbing correct;
  Python is a **clean single-variable delta** vs v6 (corpus seed-stable,
  policies/outcomes encoding-independent, augmentation plane-generic, fresh
  10-plane pretrain, lever stack OFF).
- **Caveat (reviewer):** CF-1 is a global engine change ⇒ a v6tp smoke is not
  comparable to the *historical* §175 v6 datapoint; attributing V_spread
  recovery to CF-2 alone needs a co-built **CF-1-only v6 control**.
- **AGGREGATION (operator GPU-pending):** `scripts/p5a_v6tp_pretrain.sh` →
  `--variant v6tp_p5a_smoke --iterations 30000`. SUCCESS = SealBot WR ≥25%
  past 30k AND V_spread >+0.10 @20k. FAILURE (V_spread collapse despite the
  plane) → Phase 5b **TD-λ** (NOT aux heads). Report:
  `reports/investigations/compound_turn_cf1_cf2_20260528.md` §8.

#### Phase 5a smoke VERDICT — QUALIFIED SUCCESS (2026-05-29, ran on vast)

30k smoke completed (lever stack OFF). Full data + reasoning: report §9.
- **SealBot WR (sampled T=0.5, in-run):** 22/18/21/21/22 % (5k–25k) — flat
  ~21% plateau, above §150 anchor 17.4%, far above §S178/§S181 collapse (0–5%).
- **SealBot WR (greedy T=0, standalone 30k ckpt): 33%** [24.6, 42.7] — clears
  the ≥25% bar, ≈2× anchor.
- **Temperature isolated as the dominant factor** of the 33-vs-21 gap: same
  KClusterMCTSBot path 0.33@T0 → 0.23@T0.5 (−10pp); path effect (KCluster vs
  ModelPlayer @T0.5) ~2pp within noise. Confirms L21/L22 — sampled flatness,
  not weak argmax.
- **V_spread (T3):** net-positive throughout (+0.22..+0.59 vs §S181's −0.016),
  degraded in the tail (+0.6→+0.22, −0.31 excursion @25.5k) — attractor
  pressing late but not winning.
- **Colony = decision, not spam:** stride5_p90=4 (vs 60), colony_ext_frac=0.0
  (vs 0.40); colony share of wins 36%→82% is legit meta per
  [[feedback_colony_is_meta_not_kill_signal]].
- **Net:** colony attractor SUPPRESSED (the §S181 target). Favors E1
  (compound-turn defects were contributing); E2 (§S150 loop-regression)
  weakened. Open: CF-1 (global) vs CF-2 (turn-phase plane) attribution →
  **CF-1-only v6 control** (`v6_p5a_control`, running) resolves it; eval at
  BOTH T=0 and T=0.5.
- **L62 — always eval at both T=0 (greedy strength) and T=0.5 (sampled-policy
  health); a single temperature misreads the model** (here: 33% vs 22% on the
  same checkpoint). **L63 — the encoding registry is NOT the single source of
  truth it claims; ~14 modules hardcode "v6≡8 planes" (WIRE_CHANNELS/
  BUFFER_CHANNELS/KEPT_PLANE_INDICES/{8,11} literals). Needs a dedicated
  encoding-width audit wave** (report §9.7).

### Next-session PROMPT 2 — H-PLANE-MISMATCH CONFIRMED + hardcode ledger (2026-05-29)

Two no-GPU deliverables, run in parallel with the CF-1 control. Report
`reports/investigations/hplane_mismatch_20260529.md`; probe
`scripts/structural_diagnosis/hplane_activation_dump.py`.

**H-PLANE-MISMATCH — CONFIRMED.** v6-family wire planes 1-3/9-11 (my/opp history
t-1..t-3) are LIVE in pretrain (Python `to_tensor` fills them from the
`move_history` deque; corpus mean-abs ≈ 0.04 ≈ the live t0 planes, nonzero in
94-99.8% of rows) but EXACTLY ZERO in self-play (Rust `encode_state_to_buffer` +
`_channels` zero them; pinning test). Matched-sample dump: corpus history-sum
0.0434 vs self-play 0.0000. So 6 of 8 (v6) / 6 of 10 (v6tp) wire planes carry full
mass in pretrain and zero in RL — a transfer cliff. Fresh-context review PASS
(re-dumped Rust path → history 0.000000). **Correction:** the zeroing is only the
history planes; turn-phase 16/17 are live on both paths.

**CRITICAL caveat — NOT a colony cause.** The mismatch is INVARIANT across all
v6-family runs; §150 (17.4%, no collapse) and §175/§S178/§S181 (collapse) share
it. It cannot be the colony differentiator — it is a constant **regression-class
baseline handicap**, scoped against/before Phase 6, NOT a colony remedy.

**Recommendation:** register `v6_live2` (`kept_plane_indices=[0,8,16,17]`, 4
planes = my/opp t0 + turn-phase) — drops exactly the mismatched history planes,
makes pretrain==selfplay. Gate a fresh-pretrain **MCTS-matched** smoke vs v6tp
(probe gates can't validate dynamic equivariance, L2). Populated-history-on-Rust
is infeasible (split-responsibility).

**Hardcode ledger (L63 follow-through):**
`audit/structural/encoding_width_hardcode_ledger.md` — **P0=2**
(`orchestrator.py:286` in_channels fallback=18; `generate_bot_corpus.py` hardcoded
v6), P1=6 (diagnostic/probe surfaces). Every reactively-fixed LIVE path verified
clean. Ledger only (clean bisect); fix wave is separate, P0s first; add v6tp as
the non-8-plane regression canary.

**L64 — a documented encoding asymmetry that is constant across runs is a
baseline handicap, never a per-run failure differentiator.** Don't attribute a
divergent failure (colony in some runs, not others) to an invariant. **Why:** the
history-plane shift read as colony-relevant in the brief but is identical in
collapsing and non-collapsing runs. **How to apply:** before proposing an
invariant as a cause, confirm it varies across the outcomes you're explaining; if
it doesn't, it's a floor, not a trigger.

### PROMPT 1 — CF-1 control launched on vast; second opponent DROPPED (2026-05-29)

- **CF-1-only v6 control LAUNCHED on vast** (5080, `phase4.5/p5a_v6tp`). Pretrain
  (`p5a_v6_control_pretrain.sh`, 8-plane v6, seed 42) running; a guarded chain
  (`scripts/_chain_v6ctl_smoke.sh`, tmux `v6ctlsmoke`) auto-launches the 30k
  `v6_p5a_control` smoke on `PRETRAIN_DONE`. v6tp 30k artifacts archived (vast
  `checkpoints/v6tp_archive/` + pulled to laptop `checkpoints/v6tp_archive/`)
  BEFORE the control clobbers the shared `checkpoints/` dir — this preserves the
  QUALIFIED-SUCCESS v6tp 30k model regardless.
- **Second-opponent disambiguation DROPPED (operator, 2026-05-29).** KrakenBot was
  the only non-SealBot bundled bot and is too weak to serve as a strong second
  opponent: its MinimaxBot emits illegal-move sentinels mid-game so the wrapper
  falls back to a neighbor-2 heuristic (early v6tp@T0 vs Kraken ≈90% reflected the
  weakened opponent, not model strength). No other strong bundled bot exists; the
  only alternative is `CommunityAPIBot` against a live `explore.htttx.io/bots/<name>`
  endpoint (needs a URL + the bot online). A trial `run_sealbot_eval --opponent`
  generalization + its eval were **reverted** (uncommitted) — the kraken arm is not
  pursued. The v6tp ~21% sampled / 33% greedy therefore stays read as
  distance-from-SealBot, NOT validated as a general plateau.
- **Attribution verdict (CF-1 vs CF-2) BLOCKED** on the control 30k results
  (~13hr). Process per report §9.6 + PROMPT 1 — spam-signal primary, V_spread
  DEMOTED (operator note 2). CF-1 + CF-4 banked as correctness regardless.

### v6_live2 encoding LANDED (H-PLANE-MISMATCH fix scaffolding, 2026-05-29)

Registered `v6_live2` (`kept_plane_indices=[0,8,16,17]`, 4 planes = my/opp t0 +
turn-phase) = v6tp minus the dead history planes. Scope/gate:
`reports/investigations/v6_live2_scoping_20260529.md`. Landed the no-GPU
scaffolding + made it actually runnable (NOT yet pretrained/smoked — GPU queued
behind the control). Code uncommitted.

- **Wiring** (all green): registry.toml entry + Python `_REGISTERED_NAMES` +
  resolvers detector (`in_ch==4 → v6_live2`) + corpus/anchor path maps + export
  `--encoding v6_live2` + `configs/variants/v6_live2_smoke.yaml` (`in_channels:4`)
  + `scripts/p5a_v6_live2_pretrain.sh`. Engine rebuilt (`make build`). Verified:
  audit parity, lookup, 4-plane model construct+forward, detector (neutral label
  → v6_live2), export slice → (T,4,19,19), round-trip + 16 Rust registry tests.
- **Fresh-context review caught 6 run-blockers** the static wiring missed — all
  one root cause the hardcode-ledger UNDER-COUNTED: chain-plane recompute hardcodes
  the **opponent t0 stone at corpus/buffer slot 4** (the v6 position), but v6_live2
  has opp at slot 1. Fixed via a registry-derived `opp_stone_slot(spec)` helper
  (`hexo_rl/encoding/resolvers.py`) at all 5 recompute sites
  (`batch_assembly.py` ×3: load_pretrained / load_bot / `_augment_recent_rows`;
  `pretrain_dataset.py` collate) + the checkpoint_loader allow-list & spec branch
  (`checkpoint_loader.py`). **Backward-compat pinned:** `opp_stone_slot`==4 for
  v6/v6tp/v6w25/v8/v7full, ==1 only for v6_live2. Regression test
  `tests/test_v6_live2_wiring.py`. Full Python suite green except 2 PRE-EXISTING
  `test_analyze_api` failures (confirmed via stash — unrelated to v6_live2).
- **L65 — the encoding-width ledger's grep patterns missed the `states[:, 4]`
  opponent-plane-index class.** A hardcode audit keyed on identifier names
  (WIRE_CHANNELS/KEPT_PLANE_INDICES/{8,11}) will not catch a bare positional
  slice like `states[:, 4]` that encodes "opp lives at plane 4". **Why:** the
  ledger reported all live paths clean, yet a 4-plane run hard-crashed at 5 such
  sites. **How to apply:** when adding an encoding with a NEW plane COUNT, grep
  positional slices (`[:, N]`, `[i, N]`) over state tensors too, and prefer a
  dry end-to-end run / fresh-context trace over a name-grep ledger alone.

### FINAL VERDICTS — PROMPT 1 + PROMPT 2 (2026-05-31)

All data in: CF-1-only v6 control (killed @20k, verdict decisive) + v6_live2 30k
smoke (ADOPT). Handoff: `reports/investigations/v6_live2_session_handoff_20260530.md`.

**PROMPT 1 — CF-2 (turn-phase planes 16/17) is LOAD-BEARING.** The CF-1-only v6
control COLLAPSED vs SealBot (9%→1%→3% @ 5/10/15k) with the self-anchor *rising*
(0.47→0.63) — the anchor↑/sealbot↓ colony-capture divergence (L34) — while v6tp
held ~21% and v6_live2 matched/beat it. CF-1's terminal-sign fix alone does NOT
suppress the attractor; the turn-phase signal carries v6tp. **Keep the plane.**
Collapse manifested as the WR/anchor divergence, NOT `colony_extension_fraction`
(stayed 0.0) — a strength collapse vs SealBot, not extension-spam. CF-1 + CF-4
banked as correctness regardless. **Second opponent DROPPED** (KrakenBot too weak —
illegal-move fallbacks); SealBot-style-specificity NOT disambiguated (only
`CommunityAPIBot` vs a live htttx.io endpoint remains). Residual = **policy
flatness** (10pp T0/T0.5 gap: v6tp 33/21, v6_live2 40/20) — policy-target hygiene,
deferred. CF-5/CF-6 UNFIXED/UNDETERMINED (gate any future KataGo FPU).

**PROMPT 2 — H-PLANE-MISMATCH CONFIRMED → v6_live2 ADOPTED.** History planes
1-3/9-11 live in pretrain (mean-abs ≈ live t0 ≈ 0.04), exactly zero in self-play —
a 6-plane pretrain↔selfplay cliff. INVARIANT across collapse/non-collapse runs ⇒
baseline handicap, not the colony cause (L64). Fix realized: `v6_live2=[0,8,16,17]`
(drop dead history, keep stones + turn-phase), 30k smoke = **ADOPT** —
**greedy 40% [31,50] (n=100) > v6tp 33%**; sampled ~0.20 ≈ v6tp ~0.21 (within CI,
trajectory 0.15/0.20/0.20/0.29/0.16); anchor 0.45→0.52 + best 0.37→0.54 climbing
with SealBot (genuine self-improvement, no colony-capture); spam clean throughout;
threat head healthy (C1 PASS 3.870 @ 8.5k vs bootstrap 0.063). Ledger: P0=2, P1=6
+ the missed positional-slice class (L65, fixed). Ledger P0s + a proper
de-hardcoding sweep still owed.

**Synthesis — production encoding = v6_live2.** `[0,8,16,17]` literally IS
PROMPT 1's answer (keep the load-bearing turn-phase = CF-2) + PROMPT 2's answer
(drop the history cliff). Simplest of the three (4 planes vs v6tp's 10),
matches-or-beats v6tp vs SealBot, no spam. **Adopt v6_live2.** 30k model archived
`checkpoints/v6_live2_rl/checkpoint_00030000.pt`; vast run hung in its final
in-run eval (instance reset) — training completed, model safe. Open: commit the
arc, ledger-P0 sweep, policy-flatness + real-second-opponent (deferred).

### De-hardcoding sweep — `resolve_arch` resolver + INV pin (2026-05-31) — PASS

Owed ledger-P0 sweep closed as a clean, test-pinned, one-commit-per-site arc
(NOT reactive). Design (question-first, operator-confirmed): REJECT
checkpoint shape-sniffing; ADOPT ONE registry-derived resolver
`hexo_rl/encoding/resolvers.py::resolve_arch(name) -> ArchSpec` {in_channels,
kept_indices, cur_stone_slot, opp_stone_slot, k_max, policy_logit_count,
history_planes, turn_phase_planes} — explicit, by name, every field from
`lookup(name)`. Folded the pre-existing `opp_stone_slot` + new `cur_stone_slot`
onto a shared `_kept_slot_of`. Rust mirror: `impl RegistrySpec` accessors
(cur/opp_stone_slot, history/turn_phase_planes) — init-time only, NO MCTS
hot-path call site ⇒ **no bench gate** (confirmed; 196 Rust tests green).

14 commits `0e89ccd..05e3365` on `phase4.5/v6_live2`. Sites routed:
- **P0-1** `orchestrator.py` fresh-run `in_channels` (was literal 18) →
  `_resolve_fresh_in_channels` → `resolve_arch(enc).in_channels`.
- **P0-2** `generate_bot_corpus.py` (was hardcoded-v6 end-to-end, no flag) →
  `--encoding` + `_resolve_generator_encoding` (19×19-single-window guard) +
  `spec.kept_plane_indices` slice threaded through factory/play/save.
- **P1-1..6** early_game_probe / build_value_probe_fixture / value_probe
  (plane-count skip→NaN guard) / windowing_diagnostic / analyze_api /
  v6_argmax_bot — each slices the RESOLVED encoding's kept set (or
  registry-scanned by `in_channels`), never v6's 8.
- **cur-slot** `batch_assembly.py` `pre/bot_states[:, 0]` → `cur_stone_slot`.
- **2 NEW L65-class finds beyond the ledger** (the grep caught what name-grep
  missed): `structural_diagnosis/track_a/{position_classifier,a3_h_bank}.py`
  hardcoded opp at v6 slot 4 (`state[4]` / `states[:, 4]`) — routed via
  plane-count→registry slot derivation.

**INV pin (both facets, GREEN):** (1) `resolve_arch == registry` parametrized
over ALL registered encodings (count-agnostic via `_load()`); (2)
`test_inv_no_positional_plane_slice` greps the live tree for bare `[:, <int>]`
plane slices, fails on any not in the documented SOURCE-layout allowlist
(game_state encoder writes, axis_distribution source read, dataset_v8 native
builder, bench synthetic input, policy index). **Teeth verified** by injecting
`arr[:, 4]` → RED, restore → GREEN.

**Verdict = PASS.** All P0+P1 + new finds resolver-routed; INV green; full suite
**1733 py + 196 rs green, 0 fail** (the anticipated `test_analyze_api` 400==200
failures did not occur). Fresh-context review: NONE surviving on live paths, 3
spot-checks PASS. **L66 — a name-grep hardcode ledger is structurally blind to
positional slices (`states[:, N]`) and leading-axis plane reads (`state[N]`);
ship a registry-derived resolver AND a grep-INV with TEETH (inject-and-revert)
so the next new-plane-count encoding fails a test, not a run** (refines L65 from
"grep positional slices too" to "pin them in CI"). Note: repo has no configured
formatter (no `[tool.*]`/pre-commit); `ruff format` would reflow ~1188 lines vs
the hand-aligned house style — deliberately NOT applied; new code matches
surrounding style; all ruff F401/F841/E402 in touched files are pre-existing.
Acceptable residuals (P2, unchanged): module-level `BUFFER_CHANNELS =
lookup("v6").n_planes` v6-family default consts; Rust `sym_tables.rs` v6
fallbacks (guarded). PARTIAL/none — no site needed a deeper refactor.

---

## §P-INF — inference attribution: GPU-bound vs FFI/dispatch (Rust-rewrite question)

**Date 2026-05-31.** Settle empirically whether self-play inference wall-clock is
GPU-bound or dispatch/FFI/GIL-bound **before** anyone specs moving inference to Rust
(per L18/L39/§S186: a tall line is a question, not headroom). Report
`reports/investigations/inference_attribution_2026-05-31.md`.

**Premise (code-confirmed).** PyO3 is crossed **per fused GPU batch, not per-leaf**:
worker submit (`submit_batch_and_wait_rust`, inference_bridge.rs:177) is `pub(crate)`
Rust→Rust (called worker_loop/inner.rs:534); the only per-batch Python-facing crossings
are `next_inference_batch` (fetch) + `submit_inference_results` (return). 2 FFI crossings
amortise over 64–192 leaves.

**Method.** Real WorkerPool selfplay, `diagnostics.perf_timing=true` +
**`perf_sync_cuda=true`** (mandatory — without the post-H2D/post-forward
`cuda.synchronize()`, `forward_us` collapses to async launch and GPU time mis-attributes
to the `.cpu()` D2H sync). 5-bucket attribution; `submit_us` added (perf-gated) to time
the 2nd crossing so Σ5 = full cycle. Driver `scripts/perf/inference_attribution_probe.py`,
laptop 4060, `v6_live2_smoke_laptop`, bootstrap_model_v6_live2.pt.

**Result (2 independent runs, per-batch p50).** forward/RT = **83.2% / 79.98%**;
**FFI=(fetch+submit)/RT = 5.7% / 7.8%**; h2d ~1.8%, d2h ~2%. Sum-check closes (untimed
tail 0.90% via independent inter-emit timestamps). Non-forward residual is dominated by
**batch-fill stall** (22% of batches hit the 16 ms `max_wait` timeout; high-fetch_wait ⇒
*lower* batch_n = worker starvation), not dispatch.

**Verdict — Rust inference REJECTED on evidence.** The `forward ≥ 80%` clause is knife-edge
(reviewer's run 79.98% < gate; **not** post-hoc moved) so the literal E1 gate is
INCONCLUSIVE-leaning-GPU-bound — but the **decision** rests on the FFI clause only (Rust can
touch nothing else): FFI <8% on both runs, and an **upper bound** (both ran half-full batches
under sync; production batch-fill ~99% ⇒ fuller batches + no stall ⇒ FFI fraction strictly
smaller). The 9 ms GPU forward is untouchable by a rewrite; §124 TorchScript trace already
captured the dispatch win in the Python server. If selfplay throughput is ever the target, the
only recoverable lever is the batch-fill stall (feeders: n_workers↑ / max_wait↓ / batch↓),
config-side. Concurs with §090, §124, §125 (80.4% forward on 4080S), L18.

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

### §CANARY-VAL — Spam-canary threshold validation (stride5 / colony_ext) — 2026-05-31

**Verdict (split): stride5_run = RECALIBRATE; colony_extension_fraction = METRIC-INSUFFICIENT.**
Report: `reports/investigations/canary_validation_20260531/REPORT.md`
(+ `compute_metrics.py`, `query.py`, `per_game_metrics.csv`). Independent
re-derivation review: 574-game blind 20% sample, 0/0 metric mismatches.

Validated the two abort canaries against real games (probe gates can't test
dynamic play, L2): RECENT v6_live2-window 866 games (vast-staged
`/tmp/hexo_vast_stage/logs_replays/`) + HIST §175 collapse 2000 games
(`reports/s176_a3_games/replays_05_14.jsonl`). §152 spam-positive raw games are
**unarchived** — only `phase_b_prime/instrumented/diagnosis.md` survives.

- **stride5_run separates spam cleanly but threshold 60 is dead.** Benign
  (2866 games) per-game max=3, rolling p90≤4. §152 spam (diagnosis.md, re-verified
  ×2): P50=8, P90=21, max=34. Deployed `stride5_p90=60` > spam p90 21 → 0% TPR on
  the spam it was built for. §152 author proposed P90>15. **Recommend
  `stride5_p90` 60→15** (0% FPR: benign p90≤4≪15). 60 was an eyeballed "preserved
  §157/L9" guardrail, decoupled from the §152 diagnosis.
- **colony_ext_frac is blind to the §175 attractor.** Flat ~0 during real
  collapse (A3 confirmed; here 2/2000 ≥0.40, p90=0.0). AUC≈0.5 for that mode.
  Detects only rare isolated-far-stone extension. Strategic-vs-spam *colony* axis
  (cluster coherence) is unmonitored — needs within-run-normalized fragmentation
  (A3 n_components d=−0.822 within-run, but confounded across runs/training-stage:
  RECENT clean n_comp 14.93 > HIST 8.40). Specced as follow-up.
- **Past verdicts:** colony-arc kills (§175/§S179/§S180/§S181) fired on SealBot-WR,
  NOT these canaries → not invalidated. But "spam clean: stride5_p90≪60" citations
  (§157, v6_live2 adoption) leaned on an inert gate; v6_live2 still spam-clean by
  direct measurement (stride5 max 2, colony_ext 0).
- Gates are LIVE aborts (`step_coordinator.py:147-164`, `raise HardAbort`).

**WIRED 2026-05-31 (operator directive).** stride5 spam gate now LIVE via
`monitors.hard_abort_stride5_p90` (default 30) + `hard_abort_stride5_p90_consec`
(default 3), mirroring the grad-norm gate (NOT the dead `hard_abort:` block, which
stays inert). Fires when pool rolling-50 stride5 P90 ≥ threshold for 3 consecutive
eval points. Code: `instrumentation.current_stride5_p90` → `pool.current_stride5_p90`
→ `step_coordinator` D5c → `loop.py`. TDD: 3 tests (`test_hard_abort_stride5_*`);
coordinator suite 23/0/0, instrumentation 14/0/0, tests/training 105/0/0.
FPR safety: scanned 25,601 on-disk games — §175 attractor (22,320) rolling-p90
max=4, 0 games ≥60; ONLY pre-radius-5 (obsolete §146) + cosine-temp full-conjunction
(R10-R14 p90 86-133, cosine OFF in prod L9) exceed it → **0 false positives on any
radius-5 run.** colony_ext left as telemetry (annotated non-enforced in 10 configs).
Default-on, ≤0 disables. Not in any hot path (called once per eval-interval ≥2500).

## §P6 — v6_live2 flatness re-measure + Hammerhead NNUE 2nd opponent — 2026-06-01

Dispatcher for the post-fix flatness verdict. The 3 correctness fixes (DRAW-MASK
value-target mask on ply-capped rows, CF-5 `aux_opp_reply_weight: 0.0`, CF-6
no-bug) are already landed (`3291ebe`); this is the wire→run→eval→verdict layer.
Report + pre-registered verdict: `reports/investigations/v6_live2_flatness_30k_20260601.md`.

**STEP 0 (recorded, NOT changed):** `v6_live2` **k_max=1 (single-cluster)**. The
§169 K-cluster min/max 12pp-argmax lead lives in the separate v6w25/v7mw k_max=8
line, never ported. This is **line-divergence / parsimony, NOT a silent
regression** — the v6→v6tp→v6_live2 production lineage was always k_max=1; the
H-PLANE fix touched only the plane set. The prior 40%-greedy baseline shares
k_max=1, so the flatness delta is uncontaminated. K-cluster restore = a separate
single-variable arm w/ its own matched baseline, flagged for the rethink session.

**STEP 1 DONE — Hammerhead NNUE wired as eval-only 2nd opponent.** Vendored
`vendor/bots/hammerhead` (submodule of github.com/seeligto/hammerhead) + root
`Cargo.toml` `workspace.exclude`; built into `.venv` (maturin release). The
engine `Bot` is stateful/incremental-from-origin (hard `MustStartAtOrigin`, no
set-position API) and hexo_rl keeps no ordered history → `NnueBot(BotProtocol)`
**diff-syncs** the board each move + applies one **translation** to the origin.
Correctness rests on translation-invariance (Hammerhead's static search depends
only on occupied-set + side-to-move, not replay order) so the exact opening is
irrelevant; ranges compatible (hh `max_piece_distance=8` ≥ r5; suggests within
r2 ≤ r5). Reused the SealBot path (no bespoke): `Evaluator.evaluate_vs_nnue`,
`opponent_runners._run_nnue` (appended LAST → byte-for-byte insert order),
`EvalRoundResult.wr_nnue`, `eval_pipeline` cfg/pid, `configs/eval.yaml`
`opponents.nnue` (**default OFF** — keeps the in-run 30k single-variable),
`run_sealbot_eval.py --opponent {sealbot,nnue}`. **Zero hot-path touch** (no
bench), pinned by `test_nnue_eval_path_only.py`. TDD `test_nnue_bot.py` 11/11
green (incremental + cold-start sync == board, non-origin opening, full game,
auto-reset; caught a real premature-origin-lock bug). End-to-end smoke on the
real baseline 30k checkpoint PASS. **Second-opponent interpretation:** general
strength → HeXO_vs_NNUE ≫ HeXO_vs_SealBot; SealBot-overfit → ≈/< despite NNUE
being weaker.

**STEP 2 launch-ready (operator/vast ~13hr):** `train.py --checkpoint
checkpoints/bootstrap_model_v6_live2.pt --variant v6_live2_smoke --iterations
30000`. Single variable vs the prior 30k = the 3 fixes only (verified config
delta; cosine OFF L9; stride5_p90 + grad-norm + SealBot-WR hard-aborts live).

**STEP 3/4 pending the run:** eval the post-fix 30k dual-opponent × dual-temp
(SealBot/NNUE, T=0/T=0.5, n=100, mcts-128) → pre-registered flatness bins
(BUGS-WERE-IT / TUNING-NEEDED / REGRESSION) vs baseline greedy 40% / sampled
~0.20 / ~2× gap. Canary = spam (stride5); V_spread DEMOTED. Not committed
(operator-gated).

**Local NNUE baseline matches (2026-06-01, operator request, mcts-128 T=0 n=20,
NNUE 500ms):** bootstrap v6_live2 **70%** [48,86] (14/6); pre-fix 30k **80%**
[58,92] (16/4) — both ~2× their SealBot WR (34% / 40%) ⇒ **general-strength
direction** (NNUE a genuine lower rung); wiring validated end-to-end.

**Post-fix 30k smoke RUNNING on vast (RTX 5080, 2026-06-01), NNUE ON in-run
(n=50).** Launched once on the laptop by mistake (operator: smoke belongs on
vast) → killed. vast was on pre-Phase-6 `a604804` (no DRAW-MASK engine, no NNUE)
→ rsync'd source, `make build` rebuilt engine WITH DRAW-MASK (cargo-PATH gotcha
[[feedback_vast_bench_scripting]] hit + fixed), built hammerhead for in-run NNUE,
verified engine+nnue import. §149 buffer trap pre-cleaned (archived stale
buffer+ckpts). Confirmed clean: encoding v6_live2, buffer_size_before_corpus_load
=0, step 20 healthy (loss 3.07, grad_norm 1.26), no early_game_probe_failed,
no abort. tmux `v6l2`, log `/tmp/v6l2_smoke.log`. ~13hr est; 1hr health-watch
running.

**Probe fix — early_game_probe for the 4-plane model (resolver, no hardcode).**
The in-run probe failed `expected 4 channels, got 8`: a stale version-matched
8-plane `early_game_probe_v6_live2_v1.npz` loaded blindly (the BUILDER was
encoding-aware, the LOADER had no channel check). Fixed TDD: `load_fixture`
validates the loaded plane count against `resolve_arch(encoding_name).in_channels`
and regenerates a stale fixture; `compute()` gained a model-channel guard.
On-disk fixture auto-healed 8→4. Probe-only (never affected training/verdict);
`tests/test_early_game_probe_encoding.py` +2.

**VERDICT (2026-06-02) — run completed 30k on vast (exit 134 = benign teardown
abort post-save; model safe). Bin = TUNING-NEEDED + GENERAL strength.**
- T=0 greedy n=100 standalone: **SealBot 0.32** [0.24,0.42] vs baseline greedy
  **0.40** (within noise, NOT improved); **NNUE 0.77** [0.68,0.84]. In-run sampled
  (5–25k): SealBot 0.10/0.25/0.18/0.15/0.18, NNUE 0.30/0.56/0.54/0.64/0.50.
- **Flatness PERSISTS** (greedy/sampled ≈ 1.8×, the 2× gap survives) → NOT
  BUGS-WERE-IT. Threat/value-head probe healthy + climbing across ckpts
  (C1 3.81→4.88→5.35→5.47, all C1–C3 PASS) ⇒ DRAW-MASK/CF-5 fixed the VALUE side;
  residual is **policy-side flatness, genuine**. Next lever per memo = **O1
  forced-win one-hot POLICY target** (rethink session). Not a regression (within
  baseline noise, spam clean, no promotion any round).
- **Second-opponent RESOLVED = GENERAL strength** (not SealBot-specific overfit):
  HeXO vs NNUE 0.77 ≫ vs SealBot 0.32, and NNUE is the weaker bot ⇒ the SealBot
  ceiling is a real general strength gap. KrakenBot blocker closed via NNUE.
- **NNUE wrapper cold-start bug** (the `(2,-10)` OutOfRange that crashed the first
  NNUE T0 eval) fixed: within-range replay filter + full-reset retry + legal
  fallback; resilience test added; re-run clean (4/5000 fallback moves).

**Parallelization (operator asked; DECIDED — not implemented):** in-run eval is
already async daemon-thread + skip-if-busy, so slow eval never blocks training;
**leave it sequential** — parallelizing it contends for the GPU (eval MCTS shares
the GPU-forward-bound selfplay path) = breaks training. Standalone
`run_sealbot_eval.py` is safely parallelizable (`--jobs N` subprocess-sharding,
each shard the proven sequential path, deterministic merge) — biggest payoff is
STEP-3 4×n=100 on vast; deferred (GPU busy with the 30k smoke, can't verify
locally). Future item; design captured in the report.

## §O1 — flatness diagnostics (D1-D4) + forced-win one-hot POLICY target — 2026-06-02

Report: `reports/investigations/diagnostics_o1_20260602.md`. Resolved the open
§P6 flatness diagnostics on the post-fix v6_live2 30k (vast run `f8aaf414`,
artifacts pulled to `/tmp/hexo_postfix/`), then implemented O1 gated on D1.

**Diagnostics (CPU, read-only on the existing run):**
- **D1 = HEALTHY** (gate for O1). Value loss = binary CE on scalar win-prob
  (`losses.py:75-102`, floor log(2)=0.693). Stratified by plies-to-terminal on
  371 real-terminal games: 0-4 bucket **0.271** ≪ floor (mean logit +1.0),
  5-12 0.519, 13-30 0.690, 31+ 0.762; overall 0.684 dragged to floor by genuinely
  uncertain deep-early positions. Value head reads near-won positions → bottleneck
  is policy. Independent re-bucket reviewer AGREED (exact match).
- **D2 = NATURAL-LENGTH dominant (70.3%).** 38/128 caps weak-conversion (depth-1
  missed win / open-5), 90/128 natural-length — but 67/90 carry unconverted open-4
  (2-move-win) pressure (only 23 truly balanced). O1 depth-2 reaches the open-4
  subset; residual length = cap-height (arch session). Spot-trace: model walked
  past a completed-6 win ~40 plies.
- **D3 = ALREADY-RECENTERED → SKIP.** Window = bbox centroid recomputed every
  `apply_move` (`core.rs:345-351,480-494`); first-move-(0,0) pin is a no-op; NOT a
  flatness lever.
- **D4 = soft-but-HEALTHY.** disagreement flat ~0.68 (lagging, not collapsed),
  early-game entropy ~3.15 stable, losses falling (value best 0.42), aux null
  (CF-5 confirmed), grad ~1.08. One self-corrected V_spread transient (s15-22k
  negative → +0.349, §S181 signature, didn't propagate).

Working theory CONFIRMED (soft policy → indecisive finishing → value parks at
log(2) as genuine uncertainty, one mechanism) → lever = policy sharpening (O1),
NOT value-head reroute.

**O1 LANDED (uncommitted working tree, NOT pushed):** `Board::forced_win_move`
(depth-1 + depth-2, turn-phase via `moves_remaining`, NOT ply parity) +
`first_winning_move` (`moves.rs`); `apply_forced_win_one_hot` (`records.rs`)
overrides the training policy target to a (near-)one-hot at the proven winning
move; the hardened row is forced full-search so PCR's `full_search_mask` can't drop
it. Config `forced_win_policy_{enabled,depth,weight}` as `#[pyo3(get,set)]`
(default OFF/2/1.0 — INV19/25/26 byte-equivalence untouched) → `pool.py` ←
`configs/selfplay.yaml`. INV pin `inv_o1_forced_win_one_hot_wiring.rs`
(source-presence) + behavioral unit tests + Python prune-survival test. **305 Rust
+ 4 Python tests green**; soundness reviewed **5/5, no silent drop** (one
latent-unreachable fast-game branch documented). Downstream survival proven at
every stage (aggregate/rotate/augment/prune/loss). Default-OFF = byte-identical.

**Stage C:** `configs/eval.yaml` `eval_profiles` (cheap n=50/10k SealBot-only +
milestone n=200/20k dual-opponent dual-temp) — config-only, declarative, base
defaults unchanged, wiring flagged for arch session. k_max NOT changed.

**Bench gate (vast 5080, n=5, make bench):** baseline MCTS 112,135 sim/s (≫73k) /
worker 96,043 pos/hr / all PASS; post-O1 (default OFF) — see report (no hot-path
touch: O1 fires once per move at target extraction, not per-sim).

**NOT done this batch:** O1 validation smoke (next GPU run, pre-registered: O1 ON
must narrow greedy/sampled gap 1.8×→~1.3× on a matched 30k). Handoff →
claude.ai arch-finalization session (O1 ready; cap-height; K-cluster restore;
eval_profiles wiring).

## §PRELONG — pre-long-run triage (T1-T3 → window-WIDEN arm) — 2026-06-04

Report: `reports/investigations/prelong_triage_2026-06-04.md`. Separated the causes
the "flatness" frame conflated after the O1 smoke dissolved the 1.8× premise.
Local artifacts (vast 30k gone); the perception geometry is encoding-invariant so the
local **pre-fix** v6_live2 checkpoints answer the routing question. Scripts in
`scripts/structural_diagnosis/prelong_triage_*.py`.

- **Mechanism (proven):** v6_live2 = `k_max=1` single 19×19 window. A win cell at
  chebyshev > 9 from the bbox-centroid centre → `to_flat=usize::MAX` → **no policy
  logit** (362=19×19+pass, prior 0) AND truncated out of the MCTS child array on
  spread boards (`backup.rs` MAX_CHILDREN 192) → **genuinely unreachable**.
- **T1 = DECISIVE GAME** (qualitative). SealBot self-play (global eval) misses **0/69**
  forced wins, caps low at the 150 cap where HeXO caps ~12% (noise-free) / 25.7% (run).
  *Red-team caught* the SealBot leg was deterministic/seed-blind (n=120→~9 lines, gap
  not significant); re-run with random openings (27 shapes, cap ~5%) restores it.
  Human corpus is six-in-a-row-filtered ⇒ length only (median 49), not draw rate.
- **T2 (keystone) = OFF-WINDOW.** Trained 30k (157 misses): **94.9% off-window**
  (independent re-derive 94.6%), lift **2.02×**, 97% on ≥2-cluster boards, far-cheb
  median 11; model takes **every in-window immediate win** (0/18 in-window depth-1);
  off-window fraction **training-invariant** (bootstrap 97%→30k 95%). All depth-2 =
  within-turn open/closed-4→6 (operator note); detector verified genuine (turn-phase
  guard, real check_win, dedup).
- **T3 = FLAT/structural** (cap 14.7/13.3/12.2% n=90, statistically flat); off-window
  fraction flat-high → the residual is the wall, not training horizon. Off-window is
  the dominant **miss** mode but a **minority cap** cause (24% of caps had no
  forced-miss; 64% of forced-miss games won).
- **ROUTING (post-red-team):** the binding lever is window **RADIUS, not count** —
  every off-window cell is cheb ≤ 11, so a **23×23 single-window WIDEN covers 100%**
  (width oracle: 19×19=5%, 23×23=100%). Stand up a **window-widen arm (primary) vs
  K-cluster (comparator)** BEFORE the 300k — NOT K-cluster-first (it failed 3× per
  §174 + doesn't fix single-cluster off-window misses + v8-25×25 argmax-dilution).
  Do NOT go-long-blind on `k_max=1`. O1 stays banked (P4; ~5% in-window depth-2 only).
- **Adversarial review:** 10-agent workflow (4 lenses CONFIRM mechanism, 3 red-team
  attacks LANDED → routing corrected not buried, 2 stat-gap agents → ranked
  next-diagnostics incl. multi-window value oracle, off-window-DEFENSE on losses,
  value-calibration-at-cap). Verbatim: reports/investigations/prelong_triage_data/verify_results.json.
- **Process fixes (working tree, NOT committed):** P1 eval-profile opening-plies pin
  (kills the greedy@4-vs-sampled@0 artifact class) + P2 cadence (`configs/eval.yaml`);
  P3 V_spread canary CORE met (T3 bank fires on v6_live2, `t3_spread=0.528`; alt-bank
  4-plane rebuild deferred); P4 O1 shelved+armed comment (`configs/selfplay.yaml`);
  P5 `--no-web-dashboard` flag (`train.py`/`lifecycle.py`, kills exit-134 teardown).
  Owed (operator): commit the hammerhead/NNUE stack; CONFIG-3.

## §PRELONG-CLOSE — v6_live2 + cleanup milestone committed, FF-merged, tagged — 2026-06-04

Closed the v6_live2-adopt + cleanup milestone: the §PRELONG process fixes + the
Hammerhead/NNUE eval stack are committed on `phase4.5/v6_live2`, FF-merged to
`master`, and tagged. This is the clean base for the perception arm — which is
**NOT** started here (no 23×23 / window-widen / K-cluster code lands in this
milestone; routing call deferred to the arch-finalization session per §PRELONG).

**Owed-work commit wave (5 commits, each its own concern):**

| commit | what |
|---|---|
| `65d6b30` feat(eval) | Hammerhead minimax+NNUE 2nd ladder opponent — submodule `vendor/bots/hammerhead`, Cargo `workspace.exclude`, `NnueBot` + eval wiring (`evaluator`/`opponent_runners`/`eval_pipeline`/`result_types`), `run_sealbot_eval.py --opponent {sealbot,nnue}`, `eval.yaml`/`v6_live2_smoke` cfg. **Eval-path only** (grep-pinned `test_nnue_eval_path_only.py`); zero hot-path touch → no bench gate. |
| `fa4850c` feat(eval) | P1/P2 canonical prolonged-run `eval_profiles` (cheap/milestone) — opening-plies pinned IDENTICALLY across temps (kills the greedy@4-vs-sampled@0 O1-smoke artifact class) + 12.5k cadence. Config-only (SoT); selection hook deferred. |
| `0c7474a` docs(selfplay) | P4 O1 forced-win policy = SHELVED + ARMED comment (value unchanged, stays `false`); reactive lever IFF the long-run V_spread excursion fails to self-correct at 200-300k. |
| `10922b6` feat(train) | P5 `--no-web-dashboard` — suppress only the Flask-SocketIO dashboard (kills the benign exit-134 SIGABRT teardown), keep the terminal dashboard. + gate test. |
| `c7ca6ef` fix(monitoring) | early_game_probe 8→4-plane auto-heal (resolver-derived `in_channels`, regenerate stale fixture) + compute() channel-mismatch skip + canonical v6_live2/v7full fixtures. The v6_live2-adoption monitoring fix. |

**P3 V_spread canary = NO-OP here.** The core T3-bank fix already landed on
`master` (`321b136`, unmodified in the working tree). Alt-bank 4-plane rebuild
stays DEFERRED (operator follow-up; the T3 bank alone arms the long-run canary).

**Verification (pre-merge gate):**
- `make test` GREEN — **1768 passed, 54 skipped, 1 xpassed, 0 failed** (Rust +
  Python). The v6w25-roundtrip PYTHONHASHSEED flake did not fire (clean run).
- `make bench` n=5 laptop (`n_workers=14`) — **all 10 targets PASS**. MCTS (CPU,
  no NN) median **89,891 sim/s** (range 86.9k-90.5k, IQR ±1,109) ≫ ≥73,000 floor;
  flat vs the §S183 baseline 88,006 → O1 (default-OFF) + B1 cold-path = no
  regression, as expected (neither touches the hot path).
- Fresh-context review: O1 `forced_win_policy_enabled: false` in the merged
  config; hammerhead referenced only on the eval path; no perception-arm/23×23
  leak in the owed commits.

**Merge + tag:** FF-merge `phase4.5/v6_live2` → `master`; tag
**`v6_live2-adopt-close`** at the last CODE commit `c7ca6ef` (not this
docs/sprint-log commit, per archive-tag convention). Pushed `master` + tag.

**Left uncommitted (out of scope — investigation/perception-arm tooling,
NOT this milestone):** `scripts/structural_diagnosis/*` (§PRELONG triage
probes), `investigation/`, `docs/compression/`, `scripts/{export_selfplay_games,
generate_demo_replays,transfer_v6_to_v6w25,update_manifest}.py`,
`scripts/s174_bootstrap_fix_run.sh`. The §PRELONG triage report itself is
gitignored under `reports/`.

**Next:** the perception arm (23×23 single-window WIDEN primary vs K-cluster
comparator) on this clean `master`, per the §PRELONG routing — separate chapter.

## §D-WALLCAUSATION — does the off-window wall CAUSE colony? + recorder/tripwire fixes — 2026-06-05

Branch `phase4.5/wallcausation_fixes` (Phase B, not committed pending operator).
Report `reports/investigations/wallcausation_2026-06-04.md`; go-long validation
`reports/investigations/golong_validation_v2.md`.

**Phase A — causation: INCONCLUSIVE, leaning AGAINST.** Regenerated single-window
`ModelPlayer` self-play from archived colony checkpoints (s180b 10k–53.5k, s179
10k–60k, + healthy v6live2) → forced-win detector both sides → correlated off-window
incidence vs recorded colony signal (`metadata.json eval_trajectory`). s180b corr
0.96 but off-window is **coincident-not-leading** (flat 0–2% through 10k–40k while
colony_anchor climbs 36→43 / elo 422→330; spikes only at the 50k hard-fail, CI[13,32]
disjoint from earlier [0,7.7]); healthy v6live2 carries 11% off-window > colony
checkpoints; s179-60k (colony 77%) below its own 50k peak. **Decisive caveat
(structural):** `ModelPlayer` drops off-window at selection (`evaluator.py:113`) →
`max_spread≤18` (window diameter) → it CANNOT reach the training-self-play spread-306
regime where the wall fires (§OFFWINDOW 25.6%). Instrument is asymmetrically biased
against detecting causation → can't claim clean FALSIFIED; lag+non-monotonicity kill
FIRMED. → **wall→colony NOT firmed → D-SCATTER NOT triggered.** Decisive follow-up =
Rust worker_loop self-play regen (off-window-searchable), operator-gated (§A.5).

**Phase B — recorder + tripwire fixes (3, tested, suite green; commit pending).**
(1) Live forced-win tripwire was INERT: `_emit_forced_win_trend` passed
`mover_side=0`, never matches engine {1,−1} → n=0. Fixed: fold both engine sides via
`forced_win_detector.engine_player_sides(enc)` (zero-literal, derived from a fresh
board); `update_trend_from_file_incremental` generalized to `int|Sequence[int]`.
(2) `checkpoint_step` stuck at 0 (every historical record): `pool.update_checkpoint_step`
had zero callers. Fixed: seed at StepCoordinator init + refresh at the promotion sync
(`eval_drain.py`, the ONLY `sync_inference_weights` site) — NOT per train step
(red-team caught that over-attribution: self-play runs the inference model, swapped only
on promotion). (3) Moveset: VERIFIED already-correct on master (`move_history.push`
unconditional, inner.rs:968) — empty historical records are legacy. Bench-exempt
(Python-only, off hot path). +6 tests in test_forced_win_detector/test_step_coordinator,
new test_eval_drain.py.

**Phase C — mechanism + precedent (confirmed on master).** Off-window drop is 3-layer:
target drop `records.rs:62` (via `usize::MAX` from `core.rs:409`) + uniform 1/n_ch prior
`backup.rs:112` + 192-child cap `backup.rs:105` (binding limiter). records is the SOLE
TRAINING-TARGET drop site (eval/ModelPlayer drop at inference only). **§S181 did NOT pin
the single-window action space** — it pinned value-head discrimination collapse (H6) +
permissive value-head arch (H7); off-window is a DISTINCT (policy/action-target)
mechanism → the "no-reopen colony" fence basis is WEAKER than assumed; the open question
(unsettled) is whether off-window is the upstream cause of H6.

**Phase D — go-long validation: GO-WITH-NOTE.** Single-window go-long READY iff the
fixed tripwire is a HARD gate (n=game-sides semantics). All 8 standing checks PASS; #9
wall-honesty / #11 tripwire-live (n>0) / #12 causation-not-firmed satisfied; no blockers.

**Lessons.** L: `ModelPlayer` (deploy/eval path) is spread-capped at the action window
(`evaluator.py:113`) and reproduces neither the spread-306 wall regime nor the
colony/draw regime — both are training-self-play (Rust worker) phenomena; use the Rust
path, not ModelPlayer, to study training-time pathologies. L: the recorder's
`checkpoint_step` must be tagged at the inference weight-sync (promotion), not per train
step — self-play uses the promoted model, not the live trainer model.

## §D-MULTICLUSTER S-PRE — §174 kill-gate precheck (CONDITIONALLY CLEAN) — 2026-06-06

**Goal.** Predict the §174 argmax-degeneracy for the legal-set ACTION space CHEAPLY,
BEFORE paying S0 (the Rust refactor). The handoff brief specced a static argmax-distribution
probe; CORRECTED — §174 is a bootstrap→selfplay HANDOFF collapse (policy entropy ~2.4 nats,
explicitly "not the lever"), so a static probe would false-CLEAR. Faithful instrument =
MCTS viability, Python-runnable via `KClusterMCTSBot` (no Rust; §173 multi-window machinery).

**Method.** A/B isolating ONLY the off-window drop (`records.rs:62`): CONTROL = single-window
362 (`to_flat ≥ n_actions-1` dropped) vs TREATMENT = legal-set no-drop, same v6_live2 head,
same Python K-cluster MCTS. The self-play A/B was UNDER-POWERED — sims-64 self-play stays
K=1 (0% K>1 over 33,715 expansions, all models) → pivoted to MOVE-AGREEMENT on REAL spread
positions (K≥2 AND off-window present), the regime where the drop can bite.

**Result.** Move-agreement 0.85 (30k) / 0.79 (54.5k); TREATMENT picks an off-window move
the single-window CONTROL cannot reach ~9% (checkpoint-independent). No collapse / no
degeneracy → §174's "single-window-tuned policy breaks under multi-cluster scatter-max" is
REFUTED for the 362-multiwindow legal-set. Real self-play geometry IS multi-window (K up to
7; 79% of positions have ≥1 off-window legal move). Code-faithfulness PASS (5/5); red-team +
completeness reviewers applied (3 agents).

**Conditions (all must hold for GO):** (1) **S1 MANDATORY** — eval-only clears the *argmax*
mechanism, NOT the training-loop handoff (value-head over-fit, §174 e50 mode); §174 failed
that 3× → S1 still >50%-likely to fail. (2) **S0 MUST lock 19×19 / 362-head + multi-window**
— NOT a single larger head (that reintroduces the §174-626 risk; registry has no multi-window
362 encoding yet). (3) efficacy + off-window-pick QUALITY untested → S3 adversarial gate
(`exploit_probe.py` ≤ 0.06), NEVER vs-bot WR.

**Verdict: CONDITIONALLY CLEAN** — §174 *argmax* kill-gate CLEARED → the multi-cluster line
is NOT killed; GO/no-go on S0 is now an EFFICACY + COST + S1-risk call, not a §174-argmax call.

**Lessons.** L: a cheap pre-S0 §174 prediction IS possible via eval-path multi-window MCTS,
but it clears only the argmax mechanism — the bootstrap→selfplay→training handoff is
irreducibly an S1 (post-S0) gate (consistent with the existing "ModelPlayer eval-path ≠
training self-play" lesson; the off-window drop is 3-layer in training — `records.rs:62` +
`backup.rs:112` uniform prior + `backup.rs:105` 192-child cap — the eval probe models only
the first). L: sims-64 Python self-play never spreads (K=1) regardless of model; use
move-agreement on REAL replay positions, not self-play, to exercise the multi-window regime
cheaply.

Full: `reports/investigations/multicluster_s174_precheck_2026-06-06.md`.
Probe: `scripts/multicluster_s174_precheck.py` + `_moveagreement.py` + `_measure_k.py`.

## §D-FRAGILITY — why do the long runs break + spread? (A/B/C diagnosis) — 2026-06-07

Verdict: **A (LR/training-stability transient); B (off-window value-corruption) DECISIVELY
FALSIFIED; MODERATE confidence (~0.72–0.75).** Eval-only on the LIVE v6_live2 §D-GOLONG run
(30k→90k arc pulled from vast); diagnosis only, no re-pretrain/Rust/engine change.

PHASE 0 — the run did NOT "break twice." ONE causal chain: a FALSE single-point SealBot-WR
gate abort at 87.5k (`v6l2golong.log:146636`, wr_history `[[62500,0.29],[75000,0.05]]` — only
2 points; the 87.5k drain re-eval had already RECOVERED to ~0.19) + a BENIGN exit-134 SIGABRT
(`terminate called without an active exception`, `:147047`) during graceful teardown AFTER
checkpoint+buffers saved. Gate fix committed `b340e99` (B/C triggers now require
`wr_collapse_consecutive_evals=3`; the 2-point history cannot fire it). The "second abort" is
the benign teardown the mandate anticipated, not an independent divergence.

PHASE 1 (decisive) — value-head won/lost discrimination (AUC) across 10 checkpoints (40k→90k)
on a matched 8000-position arc pool (`scripts/structural_diagnosis/fragility_value_discrim.py`,
reusing `forced_win_detector` + `golong_game_analysis` + `load_inference_model`):
  • B(iii) FALSE (lead refutation, confound-free): AUC_spread PEAKS at 75k under ALL spread
    metrics while `value_fc2` weight-norm COLLAPSES (0.224→0.143) + g4 band fails → the 75k
    event is a weight/CALIBRATION wobble, NOT a ranking-discrimination collapse (AUC is
    scale-invariant). fc2 self-corrected non-monotonically to ~0.20 by 85–90k.
  • B(i) FALSE: AUC_compact never reaches 0.70 (max 0.688); spread-vs-compact direction FLIPS
    by metric (bbox/ncomp spread>compact; density spread<compact) = a ply confound (spread ply
    72.9 vs compact 21.6), NOT a clean spread deficit.
  • B(ii) = off-window artifact: clear-won OFF-window-only spread reads ~0/neg (CORRECT —
    unconvertible, no logit); clear-won IN-WINDOW (convertible) spread reads positive
    (+0.13..+0.47, peaks 75k).
  • B(iv) precondition IS MET (independently replicated): per-game off-window forced-win rate
    RISES with spread (Spearman(bbox, off-window rate) +0.356, p≈9e-11). → **B is dead on its
    CONSEQUENCE (i+iii), not its precondition** (the stronger refutation): the throttle is
    real, but the value head handles it correctly and spread-discrimination does not collapse.

PHASE 2 — 75k event at FLAT lr ~0.00197 (53k re-warm 2e-3 plateau; did NOT decay through
recovery → A's literal lr-decay mechanism doesn't hold, but sustained-high-lr wobble does);
grad_norm stable 0.9–1.4; t3_spread dipped 0.30@77.5-83k → recovered 0.43.

PHASE 3 — SealBot-WR softening (b340e99) is necessary+sufficient; stride5 (passive p90=4) +
grad_norm stay FAST (never false-fired, don't self-correct); g4 band is already warn-only
(fired a TRUE warning at 75k without aborting).

Verification: 7-agent fresh-review + red-team workflow (`wf_72614f66-0b1`) — 6/7 upheld A
(methodology SOUND); 1 red-team (recovery-illusory, conf 0.25) capped confidence, did not
overturn. Caught + corrected a reviewer error (3 agents cited the in-run EMA to wrongly claim
B(iv) fails).

Decision: fix LR/stability; bank `checkpoint_00087500.pt` as deployable single-window
baseline; **multi-cluster PARKED on the fragility basis** (B falsified → not a fragility fix;
off-window remains a real ACTION blind spot for the SEPARATE adversarial-human-exploit question
only). **REVISIT if the 100k full SealBot eval ≤ ~0.15** or loss/draw/grad destabilize. No
evidence prior long runs all broke at 75–90k (first run to traverse it; recovered) → fragility
NOT systematic.

Full: `reports/investigations/fragility_diagnosis_2026-06-07.md`.
Instruments (local, uncommitted): `scripts/structural_diagnosis/fragility_value_discrim.py`,
`investigation/fragility_2026-06-07/`.

## §D-COHERENCE — in-window vs off-window forced-win conversion decomposition — 2026-06-08

Verdict: **V-INWINDOW** (significant sub-material) — the golong `forced_win_conversion`
decline is driven by IN-WINDOW finishing, NOT the off-window structural defect. Eval-only,
read-only on banked golong self-play records + checkpoints; ZERO engine/config/Rust/pretrain
change (`git diff --stat` empty). Source data = vast
`checkpoints/_archive_golong_kill_20260608T065342Z/` (the killed run, pulled). Usable arc
buckets (promoted-inference checkpoint_step tags on Rust GameRecorder self-play records):
30k (n=256 gs) / 53k (377) / 87.5k (293); 75k=1, step-0 legacy unusable.

**CORRECTION (2026-06-08 — RETRACTED by §D-GLOBALCONC Phase 2b; see that entry below).** The
V-INWINDOW verdict and the "**NOT multi-cluster (off-window not the driver, 19 %)**" routing in this
entry are **RETRACTED**. They rested on the legacy depth-1/ply flatten win-cell unit, which mislabeled
depth-2 wins that COMPLETE off-window but place their FIRST stone in-window as in-window. Under the
turn-correct completing-cell unit (`forced_win_detector.winning_turn_cells`, the cell that LANDS the
win — re-run `coherence_decomposition.py --unit turn`): the in-window decline LOSES CI significance
(per-game-side d−0.036 CI[−0.098,+0.028]; turn-level d−0.116 CI[−0.258,+0.034] — both straddle 0),
the off-window share RISES significantly (+0.074 CI[+0.009,+0.141]), and the decline shift-share flips
**81/19 → 46/54** → off-window is the larger, significant, rising leg. The GLOBAL `forced_win_conversion`
decline (−0.075) and the over-spread WHAT (own ncomp 14→22) are **unit-INVARIANT and STAND**; only the
in/off-window SPLIT and the multi-cluster dismissal change. This RE-OPENS the off-window / multi-cluster
line (→ §D-GLOBALCONC Phase-3 Branch C). History preserved below as originally written.

**Phase 1 (decomposition).** Off-window share is FLAT (turn-level 0.516→0.480→0.554, Δ+0.038;
per-game-side Δ−0.006). Shift-share of the global turn-level decline (off-window converts ≈0
⇒ global = conv_in·(1−share)): **in-window-drop = 81 %, off-window-rise = 19 %** → the
red-team "decline over-determined by off-window rise" premise is REFUTED. In-window
conversion drops: recurrence-robust per-game-side Δ−0.089 (CI[−0.156,−0.016], seed-stable,
**just SUB-material** vs the 0.10 bar); turn-level Δ−0.151 is ~40 % recurrence-inflated (don't
trust as "material"); non-converting game-side fraction rose 17.0→25.7 %. Survivorship
inverted: in-window forced-win COUNT per game-side ROSE 1.49→1.90.

**Phase 2 (mechanism, self-contained — re-derived, NOT leaning on §D-FRAGILITY).** On a fixed
common pool of in-window forced-win positions, NEITHER NN head degrades: POLICY finishing-mass
flat (p_win 0.237→0.246, top1 ~0.30, entropy flat), VALUE healthy (AUC won>not-won flat
0.79→0.81; 75k global-sharpness peak independently reproduced). The drop is DISTRIBUTIONAL:
under the same models, p_win on 87.5k-sourced positions (0.187) ≪ 30k-sourced (0.256), Δ−0.069
(~27 %, ≈ the conversion drop), uniform at every checkpoint, `#win-cells`-invariant.

**Phase 2b (WHY — operator-driven).** Operator read the games: "defending + scattered attacks,
way too spread out." Confirmed: the mover's OWN force OVER-FRAGMENTS along the arc — own
components 14.0→16.6→**22.1** (+58 %) while stones rise only +9 % (components/stone 0.34→0.49,
elevated within matched ply- AND stone-bands → training-checkpoint property, not game-length);
largest-blob fraction 0.35→0.24; local support around the win 1.59→1.45 falls; opponent
interference 0.90→1.12 rises. So forced wins are thin, unsupported, single-threat opportunities
in scattered structure. This is the **opposite pole** from the §175/§S181 colony attractor
(over-homogenization) — spread ran too far into force-fragmentation.

**Lever (to strategy layer).** = reward/target shaping for in-window line-COHERENCE, sharpened
to **force-CONCENTRATION of the attacking mass**; NOT O1 (policy-target — Phase 2 refutes the
policy-head mechanism, O1 stays falsified), NOT multi-cluster (off-window not the driver, 19 %).
Guard rail: a concentration lever must land BETWEEN over-spread and colony-homogenization — gate
on conversion/fragmentation here AND colony_fraction/stride5 monitors AND SealBot-WR (the spread
also drove the 24→38 % WR gains; don't trade strength for finishing efficiency blindly).

Verification: independent REVIEW (UPHELD — all numbers reproduced, leak-check clean) +
RED-TEAM (6/7 pillars clean; over-spread reinforced by matched-band controls; one owed
qualification = sub-material magnitude, folded in). NOTE: the off-window target drop is the
`window_flat_idx_at_geom` reprojection in `engine/src/game_runner/records.rs`; the widely-quoted
"`records.rs:62`" is the pass-slot skip, not the off-window drop (mechanism right, line wrong;
the `forced_win_detector` docstring perpetuates it — left untouched per read-only mandate).

Full: `reports/investigations/coherence_decomposition_2026-06-08.md`. Instruments (local):
`scripts/structural_diagnosis/coherence_decomposition.py` (Phase 1),
`coherence_inwindow_policy.py` (Phase 2), `coherence_overspread.py` (Phase 2b);
`investigation/coherence_2026-06-08/` (replays + checkpoints + JSON + REVIEW.md/REDTEAM.md).

## §D-OVERSPREAD — WHY does the model over-spread? (5-driver discriminator) — 2026-06-08

Verdict: **NO clean driver; the hypothesized D1→D3→D5 value-first stack is FALSIFIED.** Eval-only,
read-only on banked golong replays + the 11-rung checkpoint ladder; ZERO engine/config/Rust/pretrain
change (`git diff --stat` = only this sprint-log). 12-agent parallel workflow (5 driver probes →
per-driver red-team + independent review + ordering attack). Pre-registration (lighting + thresholds
locked before any probe): `investigation/overspread_2026-06-08/PREREGISTRATION.md`.

**Drivers.** D1 value-discrimination ceiling **OUT** — value RANKS concentration; the clean
(turn-fork, stone-matched) strand AUC 0.69–0.79 RISING (the mover_ncomp/largest_frac headline
strands are stone-confounded — red-team correction; OUT rests on the fork-redundancy strand only).
D2 off-window-structure-biases-play **OUT** — abandonment flat/down, model commits to boundary lines
MORE (P(pick interior) 0.64→0.56); the boundary-share rise is a SYMPTOM of over-spread, not a driver.
D3 target-doesn't-credit-forks **OUT** — fork-affinity HIGH (110–180× the no-credit null after
removing the 52% finishing-move confound) and does NOT fall (75k sharpness peak). D4 exploration
**OUT by constancy** — 33 exploration knobs byte-identical across 3 relaunches; cosine on LR not temp.
D5 self-play-co-adaptation **INCONCLUSIVE** — Part-2 (losses are spread-force) instrument-blocked (no
eval move-sequences banked + window-masked ModelPlayer); Part-1 decline-leg "spearman ±1.0" is a
training-step monotonicity artifact (87.5k WR injected; colony_wins co-falls → partly the opposite
pole). All §2 re-validation guards PASS (each OUT ruled out by in-context evidence, not a borrowed
prior). Review UPHELD (git-clean, numbers re-derived, no leaks).

**Ordering red-team — DECISIVE co-movement.** value_mean, policy p_win, policy-fork-mass AND
MCTS-fork-mass ALL peak at 75k (the §D-FRAGILITY/§D-COHERENCE sharpness transient) while over-spread
(ncomp 14→22) rises MONOTONICALLY → a monotone phenomenon cannot be caused by a non-monotone one ⇒
value+search are a COUPLED WAVE riding ON TOP of the spread substrate, not its generator. Value-first
falsified in the wrong direction; value/search are clean signals to lean ON, not holes to patch.

**D5 follow-on (2 purpose instruments; reframe: over-spread = own-force FRAGMENTATION, a single-window
weights property, NOT off-window reach — the block was un-RECORDED moves, not the window mask).** Leg A
INTERNAL (banked self-play, n≈210–320/bucket): the more-fragmented side is NEVER the loser
(P(more-frag lost)=0.40–0.50 ≤0.5 everywhere; winner has MORE components; mildly favored early, neutral
by the finish) → **over-spread NOT punished internally** → the co-adaptation PRECONDITION is CONFIRMED.
Leg B EXTERNAL (generated SealBot games, KClusterMCTSBot @ temp 1.0, REGIME=REPRO verified): the naive
cut-frac Δ(loss−win) (+0.16/+0.04/+0.21, growing) is a **GAME-LENGTH confound** (losses longer: 30k
69v46, 87.5k 63v47 plies); STONE-MATCHED (fixed-ply) Δ mostly NULL/reversed (30k ply40 −0.13, 50k
−0.08/−0.01 straddle; 87.5k ply60 +0.16 CI>0 but n_win=2, uninterpretable) — win class underpowered (WR
0.09–0.14). → **external loss-spread NOT established** (length-mediated, not clean spread-force-loses).
**D5 = precondition CONFIRMED, external clause INCONCLUSIVE** — leading framing, not a clean LIT driver.
Branch 2 (compact-reference regularizer) rests on the confirmed internal neutrality. Instruments:
`overspread_d5_internal_punish.py` + `overspread_d5_sealbot_lossspread.py`.

**TURN-vs-PLY standing hole (operator insight, folded in).** A turn = 2 stones; `count_winning_moves`
/ quiescence are depth-1 (single-stone) — wrong unit. New turn-correct primitive
`scripts/structural_diagnosis/turn_wins.py::count_winning_turns` = `|depth1 ∪ {depth2 second-stones}|`.
Empirics: depth-1 undercounts the turn win-set in **86.5%** of threat snapshots; at in-window
forced-win turn-starts the engine quiescence `credit_gap=1.000` / `ply_blind≈0.95` (NEVER fires +1).
Strength order: this-turn depth-2 completion > next-turn depth-1 ≥3 fork > single ply threat — the
engine credits only the weaker, blind to the stronger. But the gap is FLAT across the arc → a STANDING
structural hole that *permits* uncorrected spread, NOT the trend driver (a constant cannot drive +58%
— same logic that ruled out D4). Eval-MEASURED, engine NOT changed (Phase-B; depth-2 is O(threats²)/
leaf, deliberately omitted §28/§30). Recommend promoting `count_winning_turns` into
`forced_win_detector.py` (unify the f-vs-s inconsistency); audit `probe_threat_logits.py` for the
same depth-1 blind spot.

**Routed fix (NOT value-first; operator-gated; 30k SIGNATURE smoke before any sustained run).**
(1) Close the credit-gap hole — turn-correct HARD concentration/fork credit (promote
`count_winning_turns` into the value/target/quiescence path; aux value target predicting turn-fork
redundancy) so a thin win ≠ a concentrated turn-fork (heads aren't broken — *strengthen the soft
signal into a hard one*). GUARDS: not A2 avg-pool, not a config knob; quiescence variant bench-gated.
(2) D5 compact-reference self-play regularizer — the co-adaptation backstop (GUARD: not bot-mix
anti-colony, not a PFSP league). **Decisive missing instrument FIRST:** D1/D3 OUT ⇒ the soft signal
already failed to prevent spread, so the self-play DYNAMIC (D5) may be load-bearing — but D5 is
instrument-blocked. Before paying for a fix, build a **spread-uncapped, move-recording** SealBot-eval
(lift the n_actions window mask) and test directly whether losses are spread-force. Gate any lever on
the fragmentation/conversion metrics AND colony_fraction/stride5 (opposite pole) AND SealBot-WR.

**Lessons.** L: when every menu-driver is OUT/INCONCLUSIVE, the over-spread trend is generated by the
self-play DYNAMIC upstream of value/search — diagnosed via CO-MOVEMENT (a monotone effect can't come
from a peaked cause), not by a single lit driver. L: a STATIC structural hole (depth-1 credit gap,
constant) is a standing CONDITION, not a trend DRIVER — apply the D4 constancy logic to any constant.
L: `count_winning_moves`/quiescence are depth-1; the turn-correct unit (`count_winning_turns`)
matters wherever forks/winning-counts are credited (value override, target shaping, threat probe).
L (citation): §D-WALLCAUSATION's `evaluator.py:113`/`max_spread≤18` is imprecise — the spread bound
is the n_actions window mask at `hexo_rl/eval/evaluator.py:108-118`, no `max_spread` variable
(mechanism holds). Falsified-register candidate: **§D-OVERSPREAD D1 (value-discrimination ceiling
drives over-spread) FALSIFIED** — value ranks turn-fork concentration 0.69–0.79 RISING.

Full: `reports/investigations/overspread_driver_2026-06-08.md`. Instruments (local, uncommitted):
`scripts/structural_diagnosis/turn_wins.py` + `overspread_forkredundancy.py` + `overspread_d{1..5}_*.py`
+ red-team scripts; `investigation/overspread_2026-06-08/` (PREREGISTRATION + notes + JSON + workflow).

## §D-GLOBALCONC — mid-game GLOBAL-concentration discriminator + turn-unit fix — 2026-06-08

Verdict: **GLOBAL-SIGNAL-ABSENT — neither head carries a clean concentration signal at the build-up
scale.** Eval-only GATE for the §D-OVERSPREAD follow-on (strategy-layer red-team: every §D-OVERSPREAD
concentration signal was measured at LOCAL TACTICAL positions; the fragmentation it explains is a
GLOBAL build-up property — different scales). Read-only on banked golong replays + the 11-rung
ladder; tracked-source change = the sanctioned Phase-2 detector edit + tests (+ turn_wins shim +
coherence_decomposition `--unit`). Pre-reg locked before any probe:
`investigation/globalconc_2026-06-08/PREREGISTRATION.md`. Verified by a 6-agent fresh-context
REVIEW + 4-lens RED-TEAM (UPHELD_WITH_CAVEATS; it refuted the initial policy claim + caught a
determinism bug — folded in below).

**Phase 1 (the gate).** Pool = BUILD-UP turn-starts (ply-band swept, immediate-forced-win turns
EXCLUDED — NOT the §D-OVERSPREAD tactical regime), n=9000, 30k/53k/87.5k buckets, corr(ncomp,stones)
=0.604. (a) **VALUE** `AUC_globalconc` = P(value ranks CONCENTRATED build-up > SCATTERED) at matched
stones AND matched eventual outcome (stratified Mann-Whitney): **0.40–0.42 mean every band, max
anywhere 0.579, never reaches 0.60**; stone-confound REFUTED as the cause (AUC invariant across
stone-band widths 1–12 incl. EXACT width-1 match; value is stone-agnostic); CI 2–12 SD below 0.50;
won-only ~0.50 (NEUTRAL) / lost-only ~0.33 (INVERTED). → value does NOT favor global concentration.
(b) **POLICY PRIOR** main-vs-PERIPHERAL `AUC` (adjacency controlled) = **0.565 mean, 0.547–0.582,
upper CI never reaches 0.60, flat-declining** — the prior carries only generic ADJACENCY
(`AUC_adj`≈0.78), NOT main-mass concentration. → **GLOBAL-SIGNAL-ABSENT on BOTH heads** (NOT the
clean MIXED first reported; the red-team showed the 0.78 was the trivial adjacency floor). Consistent
with §D-OVERSPREAD "no clean driver." Re-scopes **§D-OVERSPREAD D1**: value ranks LOCAL turn-fork
concentration (0.69–0.79) but is GLOBALLY absent (0.41) — D1 was mis-scoped onto local tactics
(frame meta-lesson #1, empirically confirmed). Value-sign audited (won +0.017 ≫ lost −0.166).

**LIVE COMPETING HYPOTHESIS (un-refuted; the fix must discriminate).** value AUC<0.50 = value rates
more-fragmented build-up HIGHER. H1: value is a HOLE (over-credits threat-spread w/o convertibility).
H2: value is CORRECT — §D-OVERSPREAD D5 internal-neutrality (more-fragmented side NEVER the self-play
loser) means a value head that doesn't penalise build-up fragmentation is a FAITHFUL estimator of the
spread-tolerant equilibrium, and the liability is at the FINISH not build-up. H1→Branch A; H2→Branch C
+ Branch A pushes away from self-play-optimal (risks the 24→38% WR the spread bought).

**Phase 2a (turn-unit fix; one clean commit; full `make test` green: 1831 pass; 31 detector tests).**
Promoted `winning_turn_cells`/`count_winning_turns`/`is_fork_turn`/`FORK_THRESHOLD` from turn_wins.py
into `forced_win_detector.py` (turn_wins → re-export shim). Unified the f-vs-s inconsistency onto the
COMPLETING cell `pair[1]`. PROVEN bounded: `forced`/`converted` invariant to the f-vs-s choice
(non-empty iff any win) → `forced_win_conversion` unchanged; only off-window classification of depth-2
wins shifts. **DETERMINISM FIX (red-team-caught live-metric regression):** `get_threats()` order is
unstable → `winning_turn_cells`/off-window binding were non-deterministic (65/4068 mismatches); fixed
by sorting `depth2_wins` candidates (0/3069 after; regression test). `probe_threat_logits.py` audited
— NOT depth-1-count based (threat HEAD + level≥3); unaffected.

**Phase 2b (re-validation — FLAG HARD; RETRACTS the 19% routing).** Deterministic (two turn runs
bit-identical). GLOBAL conversion IDENTICAL across units (0.750→0.676 — invariance proven). The
§D-COHERENCE in-window ATTRIBUTION does NOT survive: in-window decline loses CI significance at BOTH
levels (per-game-side −0.036 CI[−0.098,+0.028]; turn-level −0.116 CI[−0.258,+0.034] — both straddle
0), off-window share RISES significantly (+0.074 CI[+0.009,+0.141]), shift-share flips **81/19 →
46/54**, V-INWINDOW → AMBIGUOUS. Cause: depth-2 wins completing off-window but first-stone in-window
were mislabeled in-window by the legacy flatten unit. → **RETRACTS §D-COHERENCE's "NOT multi-cluster
(19%)"** — off-window is the larger, significant, rising leg. Does NOT touch the Phase-1 finding nor
the over-spread WHAT (ncomp 14→22).

**Routed fix DESIGN (no impl, no engine change, no run; operator-gated).** Lead with the H1/H2
DISCRIMINATOR (fine-tune value/aux head on a banked ckpt → re-run globalconc_probe [value AUC>0.50?]
AND overspread_d5_internal_punish [does internal-neutrality reverse?]). Branch A: value-side (+ opt.
policy-side, since policy is ALSO absent ~0.565) GLOBAL-concentration AUXILIARY PREDICTION target
(largest-region-share / support-weighted attacking-mass concentration). GUARDS: not A2 avg-pool, not
a config knob, not LOCAL turn-fork credit (wrong scale); reward supported attacking mass not raw
clustering, gate between over-spread and the §175/§S181 colony pole; apply the WR guard-rail to
itself (H2 risk); won't fix the turn-pair scale. Branch C (RE-OPENED, possibly SAFER primary under
H2): off-window/multi-cluster for the CONVERSION/finishing leg (where §D-OVERSPREAD Leg B showed the
loss actually happens — length/finishing). Smoke reads TRAJECTORY (components/stone reverses, value
AUC>0.50, D5 internal-neutrality reverses, colony_fraction/stride5 clean, SealBot-WR holds).

**FOURTH SCALE still unprobed (red-team-named): the TURN-PAIR / second stone.** Both arms are
per-single-stone; neither measures the JOINT concentration of the two stones a turn places. Branch A
(per-position) wouldn't fix turn-pair sequencing. Next probe if the build-up fix doesn't move
components/stone.

**Lessons.** L: a concentration signal can exist at the LOCAL tactical scale (D1 0.69–0.79) yet be
ABSENT at the GLOBAL strategic scale where the decision is made — match the instrument's regime to the
decision's (frame meta-lesson #1, empirically confirmed). L: a policy "concentration" AUC that lumps
isolated moves into "spread" collapses to the trivial adjacency floor — control for adjacency
(main-vs-peripheral) or you measure "plays near its stones," not concentration (red-team catch). L:
value-absent at a scale is direction-ambiguous (hole vs faithful-estimator) — discriminate with a
self-play-internal-punishment counterfactual before forcing the feature. L: the depth-1→turn-correct
unit change is `forced`/`converted`-invariant but moves the in/off-window SPLIT — re-validating
§D-COHERENCE RETRACTED its 81/19 → 46/54 (off-window re-opened). L: `get_threats()` order is unstable
→ any primitive selecting a single threat cell (`winning_turn_cells` `pair[1]`) must sort or it makes
a live metric non-deterministic. Falsified-register: **§D-COHERENCE "in-window finishing, NOT
multi-cluster (off-window 19%)" RETRACTED** under the turn-correct unit (off-window ≈54%, the
significant rising leg). Initial §D-GLOBALCONC "MIXED / policy-sees-concentration" SELF-CORRECTED to
GLOBAL-SIGNAL-ABSENT by the red-team (0.78 = adjacency floor).

Full: `reports/investigations/globalconc_probe_2026-06-08.md`. Instruments (local):
`scripts/structural_diagnosis/globalconc_probe.py`; `coherence_decomposition.py --unit {ply,turn}`;
`investigation/globalconc_2026-06-08/` (PREREGISTRATION + JSON + run{,2}.log + verify_workflow.js).

## §D-RECONVERGE — off-window PLACEMENT conversion-lift discriminator (THE GATE) + corrections — 2026-06-08

Converges the §D-OVERSPREAD/§D-GLOBALCONC arc back to the ORIGINAL §PRELONG off-window frontier. A
unit bug in the founding §D-COHERENCE conversion metric (depth-1/ply flatten) mis-routed the
investigation away from off-window ("NOT multi-cluster, 19%"); §D-GLOBALCONC Phase-2b corrected the
unit (completing cell) and re-opened off-window as the larger, rising leg (46/54). The retraction was
BORDERLINE (off-window share +0.074 CI[+0.009,+0.141], point-pinned by the determinism fix) — enough
to KILL the 19% dismissal, NOT enough to commit multi-cluster Rust-weeks. This is the cheap
discriminator the borderline demanded.

**Phase 0 (COMMITTED + PUSHED `origin/phase4.5/overspread_driver`).** `7e786b9` (detector turn-unit
promotion + f-vs-s unification onto the completing cell + `get_threats()` determinism SORT, bundled so
no intermediate commit carries the non-determinism + 7 tests + `turn_wins` shim) + `1361ca0`
(`coherence_decomposition --unit` + `globalconc_probe.py` + sprint-log §D-GLOBALCONC + §D-COHERENCE
CORRECTION note). Commit-safety PROVEN: GATED `forced_win_conversion` invariant (forced iff win-set
non-empty; converted reads outcome — both f-vs-s-independent), only the WATCH `off_window_forced_win_rate`
shifts toward correctness+determinism. `make test` green (1829 py + rust). **Flake (`test_shape_fallback`)
verified PRE-EXISTING** (registry `(8,626)→v6w25` shape-probe fragility): parent suite 1822 + change
1829 BOTH green; the change adds ZERO registry registration → cannot trigger it (STOP condition
impossible).

**Phase 1a — THE GATE — VERDICT: LIFT (off-window PLACEMENT is the binding conversion constraint).**
EVAL-ONLY, read-only, NO Rust/training. Reuse `_ControlDropMCTSBot` (single-window, off-window priors
dropped — production path) vs `KClusterMCTSBot` (multi-window legal-set), SAME model, ONE switch. Pool
= golong self-play off-window forced-win turn-starts (corrected `winning_turn_cells`+`is_off_window`);
160 OFF-leg (**all-off 151 = 94%**, control conversion ≈0), 160 IN-leg reference. `recovery =
MULTI_off / R_in` (R_in = in-window finishing skill). Pre-registered LIFT (recovery≥0.50 + every-ckpt
lift-CI>0 + placement≥0.5) — **all met**:

| ckpt | R_in | OFF control | OFF multi | lift CI | recovery | off-placement-frac | all-off multi |
|---|---|---|---|---|---|---|---|
| 30k | 0.581 | 0.025 | 0.412 | [+0.310,+0.464] | 0.71 | 0.73 | 0.424 |
| 50k-PEAK | 0.631 | 0.056 | 0.512 | [+0.376,+0.544] | 0.81 | 0.65 | 0.523 |
| 87.5k | 0.519 | 0.050 | 0.312 | [+0.193,+0.340] | 0.60 | 0.66 | 0.311 |
| **pooled** | — | ~0.04 | — | **+0.369** | **0.71** | **0.68** | **0.419** |

Multi recovers 60–81% of in-window finishing on off-window forced wins, MOSTLY by placing the
off-window cell (placement 0.65–0.73); single-window production is structurally walled (~0.02–0.06).
Ablation-clean (drop fired 28–30k expansions, K>1 ~96%, max_k 7–8, `dropped_all_turns=0` — NOT
vacuous). The all-off subleg (control ≈0 → multi ~0.42; ANY conversion there REQUIRES off-window
placement) is the clincher. → the §D-COHERENCE +54% off-window leg is a REAL finishing liability the
action space addresses → **Branch C (off-window/multi-cluster) is the validated lever.** Phase 2
(H1/H2) is CONDITIONAL on NO-LIFT → NOT run.

**Phase 1b — determinism CLEAN.** Repeat-call sweep (951 games × 8): **0 mismatches** across the live
chain — `analyze_recorded_game` 0/13314, `winning_turn_cells`+binding 0/6657 over **3069** turn-starts
(the exact 65/4068 class, now 0), depth1/2 SETs 0/6657, `coherence` both units 0/6657. The
`depth2_wins` sort closed the only live-chain leak. **Two LATENT risks flagged OUTSIDE the chain** (NOT
fixed — design-only): `offwindow_adversary_bot.py:261` `blocks[0]` from unsorted `get_threats()`
(affects the ARMED exploitability monitor's reproducibility — one-line `sorted()` fix, operator-gated);
`generate_threat_probe_fixtures.py:129` sorts by level only → ties retain unstable order (fixture
REGENERATION non-deterministic; live gate reads the baked fixture → unaffected at runtime).

**Phase 3 — routing DESIGN-only (operator-gated).** Branch C re-validated on the new conversion basis
(NOT the retracted 19%); also covers the unprobed TURN-PAIR fourth scale (legal-set is over both
stones). §D-MULTICLUSTER gates re-evaluated: **S0** (Rust 362-multiwindow head) still needed — the
inference lift raises the EFFICACY prior, does not pay for it; **S1** is the DOMINANT residual risk and
UNCHANGED — the probe models only the FIRST of the THREE training-layer off-window drops
(`records.rs:62`); the bootstrap→selfplay→training-loop handoff (`backup.rs:112`/`:105` + value
over-fit, §174 e50) is irreducibly post-S0 and **§174 failed S1 3× (>50%-likely to fail)**; **S3**
(`exploit_probe`≤0.06, NEVER vs-bot WR) unchanged. **GATE EXT-LINK (OPEN, BLOCKING S0):** the LIFT is
SELF-PLAY conversion; the self-play→external link is **SIGN-AMBIGUOUS** — §PRELONG-BRIDGE's 0.0pp
(n=400, Wilson95 upper 1.52pp) + D5 Leg B null/underpowered say a vs-SealBot-WR A/B reads ≈0 BY
CONSTRUCTION (false-clears), while §D-EXPLOIT's 18%-vs-6% adversary (p=0.00017) says off-window matters
MORE vs adversarial/human play. → the open gate must be the ADVERSARIAL/spread-uncapped instrument
(`exploit_probe`≤0.06 OR a spread-uncapped move-recording eval, off-window-targeting WR Δ CI-lower>0),
NOT a SealBot-WR tourney. Discharge BEFORE any Rust-weeks; do not assert the link a 4th time.

**Verification — 4-agent fresh-context REVIEW + 3-lens RED-TEAM (NOT the implementer): UPHELD WITH
CAVEATS.** REVIEW re-derived 30k BIT-IDENTICALLY + confirmed corrected unit / one-switch ablation /
non-vacuous drop / clean Phase-0 tree. RED-TEAM: lift GENUINE off-window placement (all-off alone
clears all gates; binding-only DILUTES) — correction: control ≈0 not ==0 (residual nets out,
conservative); retraction is a CORRECT monotone reclassification (387/3069 flip IN→OFF-only, 0 reverse;
`pair[1]`=landing stone 12/12; forced/converted invariant 0/37279); frame-gate HIGH → the EXT-LINK gate
sharpened to the adversarial instrument (above). Zero tracked-source contamination.

**Lessons.** **L (CLAUDE.md candidate, promoted): a unit error in a founding measurement mis-routes
every downstream investigation — verify the measurement UNIT before building a frame on it.** The
§D-COHERENCE depth-1/ply flatten counted a depth-2 win's in-window FIRST stone as in-window
convertibility, hiding that the win LANDS on the off-window completing stone; that one-cell mislabel
sent a multi-week detour ("NOT multi-cluster, 19%") that §D-GLOBALCONC Phase-2b + this gate reversed.
L: a BORDERLINE retraction earns a CHEAP eval-only discriminator before an expensive lever — the
LIFT (recovery 0.71) cost one GPU run and converted "re-opened, magnitude-borderline" into "validated
binding constraint." L: an inference conversion-lift is a NECESSARY-condition / capability probe — name
the self-play→external KILL link as an explicit OPEN gate, and pick the RIGHT external instrument
(adversarial, not a fixed-bot WR that false-clears). Falsified-register: no new falsification (confirms
§D-GLOBALCONC's off-window re-opening + Branch C).

Full: `reports/investigations/offwindow_reconverge_2026-06-08.md`. Instruments (local):
`scripts/structural_diagnosis/offwindow_placement_lift.py` + `determinism_audit.py`;
`investigation/reconverge_2026-06-08/` (PREREGISTRATION + JSON + review_scratch). Phase-0 commits
`7e786b9`,`1361ca0` pushed.

## §D-EXTLINK — discharge the off-window external gate before any S0 Rust-weeks — 2026-06-08

Verdict: **EXT-LINK-REAL.** The off-window blind spot is a real EXTERNAL / adversarial defect, not just
the §D-RECONVERGE self-play conversion constraint → Branch C justified for **least-exploitability
(Objective A)**; S0 spec MAY proceed (still S1-dominant-risk gated). Discharged on the ADVERSARIAL /
spread-uncapped instrument, NEVER SealBot-WR (false-clears by construction). Eval-only, frozen
checkpoints; pre-registration LOCKED before the run (`investigation/extlink_2026-06-08/PREREGISTRATION.md`).

**Phase 0 (COMMITTED `a7ba110`, sole sanctioned commit).** Fixed the gate instrument's determinism:
`offwindow_adversary_bot.py` blocked level-5 threats via `blocks[0]` from UNSORTED `get_threats()`.
Characterized (pure-engine, 24612 replay positions): `get_threats()` order unstable **74.6%** (intra-
process, every run); OLD raw `blocks[0]` varied **165/166** block-relevant (≥2 level-5) positions → the
§D-EXPLOIT numbers were computed on a non-deterministic instrument (the §114 lesson). One-line
`sorted(...)` (block SET unchanged; representative pinned). Post-fix `get_move` **0 mismatches / 24612 ×
8 repeats**. `make test` green (1829 py + rust).

**Phase 1 — THE GATE (deterministic instrument, {30k / 50k-PEAK / 87.5k}, n=200/arm, sims=128, random×6).**
- **1a (adversarial forced-win) reproduces 18v6 STRONGER.** All 3 ckpts FORCEABLE: exploit
  0.255 / 0.235 / 0.215, control 0.075 / 0.06 / 0.05; pooled margin **+0.173 [+0.134, +0.213]**, every
  per-ckpt margin CI-lower > 0. Checkpoint-INDEPENDENT (drifts slightly DOWN with training). CLEAN
  one-switch ablation: `any_offwindow_forcing_position_rate` equal across arms (0.325/0.325, 0.29/0.30,
  0.295/0.30) — builder skill constant; arms diverge ONLY at conversion. The point estimate SHIFTED UP
  vs the pre-fix §D-EXPLOIT 0.18 — exactly the consequence of the block non-determinism (Phase 0 real).
- **1b (off-window-targeting WR Δ at power).** WR Δ +0.162 [+0.122, +0.202]; off-window win class
  **n=178** (~89× the §D-OVERSPREAD Leg-B n_win=2 trap). NOTE 1a ≈ 1b (exploit wins ~100% off-window →
  not independent corroboration); per-ckpt FORCEABLE margins carry the verdict, pooled z=8.45 decorative
  (same run).
- **1b-causal (the genuine independent leg; uncapped KClusterMCTSBot vs capped ModelPlayer defender,
  one switch = the off-window cap, 50k-PEAK n=100).** Uncapping CLOSES the margin **+0.16 → +0.03
  (−81%, drop z=2.56)**: the uncapped defender faces the forcing setup MORE (0.38 vs 0.27) yet loses 7×
  LESS (0.03 vs 0.22) — it BLOCKS, not avoids. The off-window advantage is CAUSALLY the action cap.
  Residual 0.03 ≤ the S3 gate (0.06) = a fix-DIRECTION efficacy prior — NOT an S3/S1 clearance (an
  inference-time multi-window overlay on single-window-TRAINED weights, not a trained multi-cluster model).

**Phase 2 (routing DESIGN-only, operator-gated).** **Objective A** (least-exploitability — gated REAL,
the deployment-vs-humans goal §D-FRAGILITY kept off-window alive for) vs **Objective B** (recover
self-play strength — NOT gated). The golong collapse (−0.32 external, peak→trough 0.38@50k→0.05@75k,
recovered ~0.19@87.5k) is dominated by the over-spread fragmentation self-play dynamic (§D-OVERSPREAD
no-clean-driver, value-first falsified); the off-window conversion leg ≈ **−0.040** (54% of the −0.075
GLOBAL conversion decline — NOT −0.075; that is the total) = ~13% of the −0.32 → **Branch C alone will
NOT recover Objective B** (bounded-small). Branch C's effect on over-spread is TWO-SIDED (entrench vs
channel), resolvable ONLY by the S1 TRAJECTORY smoke. §D-MULTICLUSTER gates unchanged: S0 (Rust 362 +
multi-window, NOT a single larger head) expensive, not paid by the lift; **S1 the dominant residual
(>50% fail, §174 ×3) UNCHANGED**; S3 post-fix off-window-pick quality ≤0.06. Optional cheap Python
multi-window S1 pre-check is KILL-ONLY (can falsify, cannot clear). Latent determinism carry-forwards:
`generate_threat_probe_fixtures.py:129` (fixture-regen, baked gate unaffected) + legacy §PRELONG copies.

**Verification — 4-agent fresh-context REVIEW + 3-lens RED-TEAM (`wf_8e547df9`, NOT the implementer):
UPHELD, no REFUTE.** REVIEW re-derived 30k 51/200=0.255 BIT-EXACTLY + confirmed pre-reg locked before
runs (mtimes) + Phase-0 sole-commit + read-only. RED-TEAM: instrument (low — regime genuinely off-window
cheb 10-15, win class powered, 1b-causal sound) / frame (low) / **magnitude (medium — caught the
off-window-leg mislabel: −0.075 is the GLOBAL total, the leg is its 54% share ≈ −0.040; corrected
throughout, conclusion STRENGTHENED)**. Zero tracked-source contamination.

**Lessons.** L: a gate instrument must be DETERMINISTIC before a load-bearing decision rides on it
(§114) — `get_threats()` order was unstable 74.6% intra-process; the §D-EXPLOIT 18v6 was computed on a
non-deterministic adversary and the deterministic rate is HIGHER (0.215-0.255). L: discharge the
off-window external gate on the ADVERSARIAL / spread-uncapped instrument, NEVER vs-bot WR (false-clears
— off-window wins need a dominant-but-exploitable state a fixed bot never creates). L: a one-switch
causal defender swap (capped→uncapped) is the cleanest external-defect proof AND a fix-direction prior;
it is NOT an S1/S3 clearance. L (CLAUDE.md re-validate-unit, again): the off-window LEG (−0.040) ≠ the
GLOBAL conversion decline (−0.075) — the dispatcher's "−0.075 off-window conversion" mislabeled the
total; the red-team caught it. EXT-LINK gates Objective A only; the −0.32 driver is elsewhere
(over-spread). No new falsification (confirms §D-EXPLOIT + §D-RECONVERGE Branch C on the external axis).

Full: `reports/investigations/extlink_gate_2026-06-08.md`. Instruments (local):
`scripts/exploit_probe.py` (deterministic), `investigation/extlink_2026-06-08/` (PREREGISTRATION +
`determinism_verify.py` + `analyze_p1.py` + `uncapped_defender_causal.py` + JSON + `review_workflow.js`).
Phase-0 commit `a7ba110`.

## §D-FOUNDING — re-establish the golong failure signal on the right instrument (checkpoint-relative round-robin) — 2026-06-08

Tested the unexamined premise every prior §D assumed: did the golong suffer a self-play STRENGTH
regression? Six investigations chased the CAUSE of the "−0.32 collapse" — a vs-SealBot WR (the
project's own flagged-WRONG strength instrument), never re-established on the instrument-matched
(checkpoint-relative MCTS-vs-MCTS) measure. EVAL-ONLY, git-diff clean (all code untracked under
`investigation/founding_2026-06-08/`).

**DATA RECOVERY (the enabler).** The post-peak ladder (75k…112.5k) believed lost was still on the live
vast box in `checkpoints/_archive_golong_kill_20260608T065342Z/` — full ladder 5k→112.5k + BANK (50k
PEAK / 85k PRE / 90k POST) + `best_model_75k_deceptive.pt` + 80M final log + AS-RUN yaml + replays.
Pulled + `torch.load`-verified (v6_live2 4-plane auto-detect). First §D able to measure the post-peak
segment checkpoint-relatively.

**VERDICT — TWO-FACED, and that is the result.**
- **On-distribution (standard openings): FLAT.** 12-rung all-pairs round-robin, MCTS-vs-MCTS, 64 sims
  (compute-bound at 128; relative Elo robust to sims), temp 0.5, color-balanced, n=40/pair (2640
  games), Bradley-Terry Elo. Slope **+0.13/1k, bootstrap CI [−0.25,+0.55], P(>0)=0.73, r²=0.014**; late
  rungs 90–112.5k statistically = 50k; heavily non-transitive (25/66 pairs invert; s45k beats s112.5k
  15–4). **No CI-resolved self-play strength regression → Objective B (recover self-play strength) is
  ILL-POSED.**
- **Off-distribution (6 random opening plies): RESOLVED late FELL.** Instrument 2×2 (temp × opening,
  rungs 50/75/90/100/112.5k, 400g/cell): holding temp=0.5, open0→open6 drops 90–112.5k from ≈50k
  (n.s.) to −55/−83/−88 Elo (CIs exclude 0); argmax (temp0) deepens only modestly. **Opening
  randomization, NOT temperature, is the lever.** Slope −1.4 to −1.7/1k, P(neg) 0.95–0.98, growing
  post-75k. **→ Objective A (off-distribution exploitability / off-window brittleness) is REAL**,
  triangulating SealBot-WR collapse + §D-EXTLINK off-window. Mechanism: scatter enlarges live-stone
  bbox → more completing cells off-window → spread-specialized late model (more off-window-dependent)
  punished hardest = the single-window × over-spread interaction, surfaced by scatter not a bot.

**Over-spread = style/symptom correlate, NOT the strength driver** (powered, n=759 self-play): the
naive "loser more-spread" (colony z=+11.5) is a SHORT-GAME blowout artifact — length-controlled it is
neutral/inverted (glen≥95 colony z=−1.7; winner more-spread by pw-dist z=−5.9); the loser−winner gap
is stable/shrinking with step; monotone-rising spread co-moves with FLAT on-distribution strength.
Its causal role is on the EXPLOITABILITY axis (the off-window mechanism), not Objective B.

**Premise corrections** (CLAUDE.md re-validate; stated plainly): **C1** no "canonical 200k floor"
exists anywhere in this log — unsourced/unverified. **C2** the run was NOT auto-killed by the SealBot
gate — the Wave3-B WR gate fired a `level:"warning"` at 87.5k (5.0% < 14.5%), the operator ran 25k
MORE steps to 112722, then an EXTERNAL process kill mid-eval; failure mode = instrument MISDIAGNOSIS
(matchup-WR read as a strength meter) misdirecting six investigations, not a premature auto-kill.
**C3** STRENGTH-ROSE/FELL/FLAT + §D-FOUNDING were dispatcher-defined, not pre-existing.

**INTELLECTUAL-HONESTY / falsified-register:** the first-pass reported **STRENGTH-ROSE +1.46/1k
"CI-cleared"** — a BUG (inverse-CI-variance WLS gave the zero-width BT anchor ~10¹⁵× weight, pinning
the fit through (35k,0)). The fresh-context red-team caught it; corrected → FLAT. L: never weight a LS
fit by a CI that can be exactly zero (the BT anchor); use a game-level bootstrap. L (re-validate the
unit, again): the SealBot −0.32 "collapse" was REAL but it measures OFF-DISTRIBUTION EXPLOITABILITY
(Objective A), not self-play STRENGTH (Objective B) — the founding measurement's MEANING was never
validated against the decision it gated (matchup-WR ≠ strength), the exact failure CLAUDE.md's
"verify the measurement unit" rule warns of. L: a borderline retraction earns a CHEAP eval-only
discriminator before any lever — the temp×opening 2×2 (minutes on the 5080) flipped "FLAT" into the
correct two-faced read; the red-team did not just verify, it changed the conclusion.

**ROUTING (design only, operator-gated):** Objective B ill-posed → do NOT open a 7th self-play-
strength cause-hunt. Pour effort into Objective A — the off-distribution/off-window exploitability the
spread specialization buys: (1) the single-window→multi-cluster/K-window ENCODING decision (§D-GOLONG
4d) now has measured evidence (the opening-scatter FELL is a measured off-window blind-spot instance);
Branch C (compact-reference regularizer) addresses the spread STYLE, the encoding swap the MECHANISM;
gate on an adversarial/spread-uncapped eval, never SealBot-WR. (2) Any fresh canonical run: steer+abort
on a checkpoint-relative mini-round-robin (Objective-B floor) PLUS an off-distribution/adversarial gate
(Objective-A), SealBot-WR demoted to logged style-diagnostic; run length governed by these, not a
guessed floor (C1).

Full: `reports/investigations/founding_signal_2026-06-08.md`. Banked data:
`reports/eval/golong_vast_pull_20260608/` (arena DB + ratings curve), pulled ladder in
`checkpoints/_archive_golong_kill_20260608T065342Z/`. Instruments (local, untracked):
`investigation/founding_2026-06-08/` — `rr_driver.py` (round-robin + BT + bootstrap slope),
`argmax_discriminator.py` (temp×opening 2×2), `overspread_causal.py`, `spread_trajectory.py`, +
`rr_agg`/`argmax_agg`/`ctrl_agg`/`rr_5rung_agg` outputs. Housekeeping: `a7ba110` (off-window
determinism fix) still UNPUSHED on `phase4.5/overspread_driver` (2 ahead) — operator-gated.
