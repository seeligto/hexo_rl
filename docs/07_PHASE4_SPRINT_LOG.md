# HeXO Phase 4.0 Sprint Log — Consolidated Reference

Read this alongside `CLAUDE.md` at the start of any new session to avoid
re-litigating resolved decisions. Structured by subsystem, not by date.
For per-day narrative see `docs/07_PHASE4_SPRINT_LOG_BACKUP.md`.

<!-- Compressed 2026-05-13 (full pass §66–§174): every closed § distilled to
     verdict + load-bearing mechanism + settled pins + commit/report pointer.
     Forensic detail extracted verbatim to docs/sprint_archive/ before
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

**Entropy regularisation:** `L_total = L_policy + L_value + w_aux·L_aux − w_entropy·H(π)`. Weight `entropy_reg_weight: 0.01`. Healthy measured band 2.1–2.9 nats (D-RUN2 2026-07-04; the earlier "~3–6 nats" citation is falsified — live selfplay entropy trails 2.42–2.90, see F3-C1); < 1.0 signals collapse.

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

> **SUPERSEDED (§S178 2026-05-18 + D-TEMPDECAY C1 2026-06-12):** the live base
> default is now `temperature_threshold_compound_moves: 0` / `temp_min: 0.5` =
> schedule OFF (constant τ=0.5, the anti-colony posture). The `15` / `0.05`
> cosine-ON parameterization below is the obsolete §143 setting — historical only.

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
| §D-LADDER (2026-06-24) | The d1m 150k+ stall is a colony-attractor / off-window divergence (self-play-strong-but-SealBot-weak mid cluster) | §D-LADDER fixed-depth-SealBot reproduction | The "SealBot exploits the 150k mid cluster" trajectory FLIPPED between a wall-clock time-limited bar (s150k WR 0.35) and a reproducible fixed-depth-5 bar (s150k 0.55) → a SealBot-instance artifact, NOT a model off-window defect. The single-axis model⊥minimax intransitivity is itself real + reproducible (transitive-null P≈0.003→0.004, 8/9 cycles SealBot-routed) but its specific shape is bar-dependent. Endpoint verdict = TRUE-STALL (deploy-matched Gumbel@150 self-ladder 120k→226k flat; brief s200k peak not held). |
| §D-STRIX S3 (2026-07-02) | Custom CUDA kernel (hexo-strix/Vladdy pattern) would speed HeXO forward path | §D-STRIX S3 verdict, RED-TEAM confirmed | Their kernel solves GNN variable-size ragged batching — a problem HeXO's dense CNN+attention does not have. K-cluster multi-window batching varies batch COUNT, not tensor SHAPE (fixed-geometry windows `torch.cat`'d on batch dim; `hexo_rl/selfplay/inference.py`) — the standard variable-N case cuDNN/flash already handles. If forward-throughput ever binds: torch.compile → smaller net → quantized eval, in that order. **[SCOPE (D-L WP2, 2026-07-13): this row is CUDA-kernel / ragged-batch-perf ONLY — it does NOT adjudicate the axis-graph REPRESENTATION, which was banked NOTE-ONLY and never falsified. Representation status advanced to RE-OPENED pending D-L WP3 probe; see `docs/designs/gnn_readjudication.md`. Kernel verdict here unchanged.]** |


### Consolidated Falsified-Register additions (relocated from split-out phase bodies)

_These rows were emitted inline in phase bodies §176–§S181 that now live under `docs/sprint_archive/`. Copied here VERBATIM so the live register is complete; full context in the cited archive file. Central table above + these = the consolidated falsified register. CLAUDE.md holds the re-validation protocol (not a table); re-validate context before dropping any row._

*(from `docs/sprint_archive/§S176-S177_sustained_recipe.md`)*
**New Falsified Hypotheses Register row candidates:**

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §176 Phase A | A3 — §175 selfplay terminal states are "one large diffuse cluster" | A3 §c-§d | Single-cluster fraction monotone-down 18.1%→6.3% across 20K–50K cohorts; modal pattern is multi-island fragmentation |
| §176 Phase A | V2 — KrakenBot MinimaxBot @ 1.0s > @ 0.1s | Wave C V2 | BT delta −5 Elo, head-to-head 20-30 favouring 0.1s; iterative deepening saturates at depth 4 in our off-distribution game |
| §176 Phase A | V5 — our v6 bootstrap @ MCTS-128 strictly between RandomBot and weakest Kraken | Wave C V5 | bootstrap MCTS-128 is the 2nd-strongest bot in the tourney (BT −62 vs sealbot); ~3030 Elo above RandomBot |

**Forward pointer:** §176 Phase B implementation (S1–S6 per `reports/s176_d_plan.md` §2) opens on a fresh branch (TBD). Recommended 6 commits ≤10 cap. Mix-ratio bot-pool weights (sealbot 50 / our_v6 30 / kraken_strong 15 / kraken_random 5) per S4 design doc. Source B (live cross-bot) is design-only this sprint; subprocess isolation mandatory per A4 do-not #1.

Forensics: `reports/s176_{a1,a2,a3,a4}_*.md`, `reports/s176_b_smoke{.md,/}`, `reports/s176_c_tourney/{summary.md,verdicts.txt,ratings.csv,h2h_matrix.csv,colony_table.csv,per_game.jsonl}`, `reports/s176_d_plan.md`, `reports/s176_e_review.md`. Memory: `project_176_phase_a_close.md` (to be written).

---

*(from `docs/sprint_archive/§S176-S177_sustained_recipe.md`)*
### New Falsified Hypotheses Register rows

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §176 Phase A Gate 2 | V70K-4 — §175 step-70K winner-side col-frac ≤ 65% (attractor weakened by training past 50K) | V70K-4 strong FAIL (100% col>0.3 rate, n_components 14.90) | Attractor captured the policy. Greedy-mode wins are uniformly colony-spam patterns. |
| §176 Phase A Gate 2 | L18 strict reading — "§175 latest_70K regressed past its own bootstrap on the selfplay axis" | V70K-3 PASS strong (50-0 H2H latest dominates bootstrap in greedy argmax) | Drift is sampled-mode policy-distribution flattening, not argmax-mode regression. See L22. |
| §176 Phase A Gate 1 | Operator-prompted L18 framing requires no statistical disclaimer at n=50 | Gate 1 dim (vi) — BT 95% CI [-150, +26] crosses zero; H2H 25/25 is parity-consistent only at n=50, NOT strongly-asserted parity | L18 lesson body should disclose CI; framing already correct ("can match") |

*(from `docs/sprint_archive/§S176-S177_sustained_recipe.md`)*
### Falsified Hypotheses Register row added

| § | Hypothesis | Evidence FAILing | Closer |
|---|---|---|---|
| §177 | Step-20K anchor escapes the colony attractor under §175 recipe | §177 2→0% across 10K→40K reproduces §175 18→4% on different anchor | §178 launch — same recipe, different mechanism (bot-mix + ply-cap split) |

---

*(from `docs/sprint_archive/§S178-S180_botmix_colony.md`)*
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

*(from `docs/sprint_archive/§S178-S180_botmix_colony.md`)*
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

*(from `docs/sprint_archive/§S178-S180_botmix_colony.md`)*
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

*(from `docs/sprint_archive/§S181_structural_audit.md`)*
### Falsified Hypotheses Register additions

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §S181-T1 | `bootstrap_model_v6.pt` + v6 human corpus jointly encode a colony bias in the value head and/or policy head before self-play (handoff hypothesis #1) | §S181-T1 bootstrap+corpus bias audit (`audit/structural/01_bootstrap_corpus_bias.md`) | Value-head Δ(colony−ext)=−0.150, Welch p=0.355 (wrong sign, n.s.); near-win sub-probe rates open-5 extension +0.978 > 5-blob +0.583 (open-4 +0.704 > 4-blob +0.378); policy plays the 6th-move win 90%; corpus winning lines 91.3% extension rising to 100% at 1400+ Elo. No colony bias exists pre-self-play — the attractor is generated by the training loop. |
| §S181-T3 | MCTS-search parameters (`c_puct`, `dirichlet_alpha`/`epsilon`) are a viable anti-colony escape lever (handoff hypothesis #5) | §S181-T3 MCTS colony-dynamics audit (`audit/structural/03_mcts_colony_dynamics.md`) | c_puct ×0.5/×2.0 moves colony-visit fraction <6pp; Dirichlet ×4 moves it <3pp — both inside n=20 noise. Higher c_puct mildly worsens colony preference. MCTS+PUCT neither amplifies nor corrects the bias; it faithfully passes through a colony-biased value/policy head. MCTS-NEUTRAL — extends L38 to the search sub-surface. |

*(from `docs/sprint_archive/§S181_structural_audit.md`)*
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

*(from `docs/sprint_archive/§S181_structural_audit.md`)*
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

*(from `docs/sprint_archive/§S181_structural_audit.md`)*
### Falsified Hypotheses Register additions

- "Bot mix is the load-bearing failure variable in the colony
  attractor mechanism" → FALSIFIED 2026-05-27 by Track 4A (W4A-B).
  Removing bot mix produced FASTER decline.
- "Multi-aux density (KataGo-style diverse aux signal) prevents
  single-attractor lock" → FALSIFIED 2026-05-27 by Track 4B (W4B-B).
  Delayed colony attractor ~5k steps but did not prevent it.


### INV pin pointer (post-split)

INV15–INV25 enumerations remain inline in the §178/§179/§180 refactor-cycle sections above. INV26/INV27 were defined in §S178 (now `docs/sprint_archive/§S178-S180_botmix_colony.md`); verbatim pin lines copied here. Canonical enforcement: `engine/tests/inv*.rs`.

**Implementation:** 9 commits `b26999b..22597fc` on `phase4.5/s178_botmix` (design + T2 Rust split + INV26 + T3 Python wire + T1 bot corpus generator + T4/T5/T7 training-path + T6 yaml).
- INV26 pins ply_cap_value-distinct outcome path; INV19 byte-equivalence extends 38→39 atomically with T2.
- **F-fix-1** threat-target colony fix landed at **T11** (commit `1aa0c8f`): `find_winning_line` now scans all stones via fallback so the threat-head target is non-empty when `winner` is set via the `player_wins` all-stones fallback (HTT 2-moves-per-turn off-line second-move case). INV27 Rust + Python parametrized GREEN.

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
| L9 | Cosine temperature schedule is the load-bearing knob in draw-collapse. ~~Pair with LEGAL_MOVE_RADIUS jitter when active~~ — CORRECTED 2026-07-04 (D-RUN2): `legal_move_radius_jitter` is dead code for all registry-spec encodings since §172/§173 (triple-verified); the live radius lever is `legal_move_radius_schedule` (curriculum). Cosine-temp ban/variant-pinning UNCHANGED. | §156, §157; corrected D-RUN2 |
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
Forensic detail: `docs/sprint_archive/§<NNN>_<slug>.md` (extracted verbatim 2026-05-13).

## § Index — INDEX-ONLY entries

| § | Topic | Date | Pointer |
|---|---|---|---|
| §75 | Fast game disable for gumbel_targets (`fast_prob: 0.0`); 100% timeout draws under K=50 sims | 2026-04-10 | `docs/sprint_archive/§075_*.md` |
| §76 | `max_game_moves` 150 → 200; plies vs compound moves clarification | 2026-04-10 | `docs/sprint_archive/§076_*.md` |
| §78 | `/analyze` policy viewer (4 commits, branch `feat/policy-viewer`) | 2026-04-11 | `docs/sprint_archive/§078_*.md` |
| §79 | Initial buffer 100K → 250K; growth schedule `[250K@0, 500K@300K, 1M@1M]` | 2026-04-12 | `docs/sprint_archive/§079_*.md` |
| §81 | Desktop n_workers sweep D3 winner (10 workers, wait 5ms, burst 8); 3700x GIL ceiling | 2026-04-12 | `docs/sprint_archive/§081_*.md` |
| §82 | `loss_ownership` + `loss_threat` added to `emit_event` (`d6a293e`) | 2026-04-12 | `docs/sprint_archive/§082_*.md` |
| §83 | `quiescence_fire_count` atomic instrumentation (`4124faa`, `ad79be7`) | 2026-04-12 | `docs/sprint_archive/§083_*.md` |
| §87 | Cargo feature gate for pyo3 `extension-module` | 2026-04-13 | `docs/sprint_archive/§087_*.md` |
| §102 | Bench rebaseline post-§97; methodology shift; NN inference & buffer-push targets lowered (driver drift) | 2026-04-17 | `docs/sprint_archive/§102_*.md` |
| §106 | Q27 Probe 1b: synthetic fixture artifact; v6 baseline regenerated from real positions; supersedes §105 FAIL verdict | 2026-04-19 | `docs/sprint_archive/§106_*.md` |
| §107 | Post-W1 sustained launch + I1 colony-extension + I2 cluster-variance instrumentation | 2026-04-19 | `docs/sprint_archive/§107_*.md` |
| §108 | Desktop post-W1 `gumbel_full` launch + I1/I2 JSONL mirror | 2026-04-19 | `docs/sprint_archive/§108_*.md` |
| §109 | Q33 selfplay-entropy: pe_self ≈ 5.35 is `H(p_model)` not target entropy; CQ targets sharpen | 2026-04-21 | `docs/sprint_archive/§109_*.md` |
| §110 | Q33-B trainer-fit sanity: pe_self fixed point ~5.36 (Q1=5.36, Q4=5.36, Δ=0.004); split E1/E2 candidates | 2026-04-21 | `docs/sprint_archive/§110_*.md` |
| §111 | Q33-C augmentation discriminator HALT (Python-API-only toggle; no config knob) | 2026-04-21 | `docs/sprint_archive/§111_*.md` |
| §113 | `buffer_sample_raw` target recalibration 1500 → 1550 µs (always-on dedup correctness cost) | 2026-04-22 | `docs/sprint_archive/§113_*.md` |
| §115 | CLAUDE.md 734 → 87 lines; topic-scoped rules under `docs/rules/`; 3 skills scaffolded | 2026-04-22 | `docs/sprint_archive/§115_*.md` |
| §120 | RecentBuffer augmentation deployed (`19b1392`); soft-abort at step 14000 — augmentation alone insufficient | 2026-04-25 | `docs/sprint_archive/§120_*.md` |
| §122 | Phase 5 architectural redesign scoping (B1 D17 channel ablation, B2 backbone-form memo, B3 retrain cost, B4 buffer compat) | 2026-04-25 | `docs/sprint_archive/§122_*.md` |
| §123 | Bench methodology fix: `pool_model` uses `mode="default"` (not `reduce-overhead`) to avoid `cudagraph_trees` background-thread TLS crash | 2026-04-25 | `docs/sprint_archive/§123_*.md` |
| §126 | Sweep harness migration: knob registry replaces `sweep_epyc4080.sh`; `make sweep` / `make sweep.long` | 2026-04-26 | `docs/sprint_archive/§126_*.md` |
| §129 | Disk-budget guard (`DiskGuard`); checkpoint/game-record pruning; rotating JSONL | 2026-04-28 | `docs/sprint_archive/§129_*.md` |
| §130 | Per-game self-play rotation port; closes §121 C1 directional bias | 2026-04-29 | `docs/sprint_archive/§130_*.md` |
| §133 | D6 sym-table verification for HEXB v6 8-plane buffer (6 tests; plane indices invariant under D6) | 2026-04-29 | `docs/sprint_archive/§133_*.md` |
| §134 | bootstrap-v6: 8-plane pretrain on 6,259 human games; 100/100 vs RandomBot; threat C2=50, C3=60 | 2026-04-30 | `docs/sprint_archive/§134_*.md` |
| §135 | Bench gate W4 8-plane: 9/10 metrics improved or flat; +14% worker pos/hr; no regressions | 2026-04-30 | `docs/sprint_archive/§135_*.md` |
| §136 | Post-§131 W1+W2 audit fix wave (19 commits): 8→18 plane refs, Q49 RNG audit COUPLED-NEGLIGIBLE | 2026-04-30 | `docs/sprint_archive/§136_*.md` |
| §138 | W4 Option C smoke (5080, 8-plane + rotation): axis_density 0.55, downward trend; CONTINUE to 40k | 2026-04-30 | `docs/sprint_archive/§138_*.md` |
| §143 | γ-knob audit + W4C smoke v3 recommendation (`temperature_threshold_compound_moves` 15→10, ε 0.25→0.10) | 2026-05-01 | `docs/sprint_archive/§143_*.md` |
| §144 | W4C smoke v3 Stage 1 ABORT; gates recalibrated for 100-ply games; `max_game_moves` 100→150 | 2026-05-01 | `docs/sprint_archive/§144_*.md` |
| §145 | Smoke v4 ABORT — γ + truncation slack insufficient; fallback to Option α' (radius cap) | 2026-05-02 | `docs/sprint_archive/§145_*.md` |
| §151 | Numba @njit audit — NO-GO; no qualifying hot-path Python loop; Rust-port alternatives flagged | 2026-05-04 | `docs/sprint_archive/§151_*.md` |

---

## §66–§101 — Phase 4.0 foundation (KEEP-DISTILLED per §1–§101 Classification Audit)

### §66 — Gumbel MCTS activation & training restart (SUPERSEDED) — 2026-04-07
SUPERSEDED by §67 (named variants) + §74 (audit) + §96 (Gumbel desktop A/B). C1 KL-loss-dead bug resolved at amendment; affected only loss reporting, not training dynamics. Forensics: `docs/sprint_archive/§066_c1-bug-impact-amendment.md`.

### §67 — LR scheduler bug + total_steps / decay_steps co-design + named Gumbel variants — 2026-04-07
`trainer.py:145` hardcoded `T_max=50_000` fallback collapsed LR to `eta_min=1e-5` at step 50K while bootstrap was still dominant. Fix: `_build_scheduler` raises if `total_steps`/`eta_min` absent. Locked: `total_steps: 200_000`, `eta_min: 2e-4`, `decay_steps: 70_000` (rule of thumb `decay_steps ≈ 0.35 × total_steps`). Three named variants in `configs/variants/`: `gumbel_full` (Gumbel root + CQ), `gumbel_targets` (CQ only), `baseline_puct` (neither). Restart from `bootstrap_model.pt`. Forensics: `docs/sprint_archive/§067_lr-scheduler-fix-gumbel-variants.md`.

### §68 — Eval DB run_id bug + broken-run cleanup — 2026-04-07
`EvalPipeline.run_id` was stored but never passed to 5 DB call sites → all runs collapsed onto `run_id=""`. Fix: thread `run_id=self.run_id`. Reference opponents keep `run_id=""` as shared anchors. §67 broken runs archived to `archive/checkpoints.broken-202604/`. Forensics: `docs/sprint_archive/§068_*.md`.

### §69 — Config Sweep 2026-04-08 — PUCT/Gumbel knob ranking — HISTORICAL
Superseded by §90 throughput baseline. 15+1 runs on laptop varied `training_steps_per_game`, `max_train_burst`, `max_game_moves`, `inference_max_wait_ms`, `leaf_batch_size`, `inference_batch_size`, `n_workers`, `gumbel_m`. **P3 winner config remains live** (tsp=1.5, burst=16). Full sweep data: `archive/sweep_2026-04-08/`. Forensics: `docs/sprint_archive/§069_*.md`.

### §70 — Phase 4.0 overnight run: mode collapse diagnosis — 2026-04-09 → 2026-04-10
**RESOLVED at §73**. Root cause: Dirichlet root noise was wired through Python `SelfPlayWorker` (eval/benchmark) but unported to `engine::SelfPlayRunner` at the 2026-03-30 Rust migration. 30/30 trace records `site: game_runner`, 0/30 `site: apply_dirichlet_to_root`. MCTS rubber-stamped a sharp prior → deterministic fixed point → no gradient signal. Dashboard `policy_entropy` averaged pretrain stream with collapsed selfplay stream, masking collapse for 16,880 steps. Restart from `bootstrap_model.pt`. Forensics: `archive/diagnosis_2026-04-10/`, `docs/sprint_archive/§070_*.md`.

### §71 — Pre-Dirichlet-fix cleanup + Gumbel fallback verification — 2026-04-10
Gumbel SH functionally active on training path (30 trace records, `top_visit_fraction` 0.24 vs PUCT 0.65 — Sequential Halving spreads budget). `policy_entropy_pretrain` / `policy_entropy_selfplay` split landed; collapse threshold 1.5 nats on selfplay stream. Pre-run checklist (10 items) walked before next sustained. Forensics: `docs/sprint_archive/§071_*.md`.

### §72 — Bench rebaseline: NVIDIA driver/boost-clock step-change — 2026-04-09
Three `bench.full` runs failed same two §66 targets (NN inference ~8,370 vs 8,500; worker throughput ~541k vs 625k). Cold/hot/idle ruled out thermals (49°C). Root cause: `DynamicPowerManagement=3` driver settled GPU into lower boost-clock bin; NN latency 1.59ms → 1.77-1.80ms (~14%). Rebaselined: NN inference ≥ 8,250 pos/s, worker throughput ≥ 500,000 pos/hr. Treat as hardware-state drift. Forensics: `archive/bench_investigation_2026-04-09/`, `docs/sprint_archive/§072_*.md`.

### §73 — Dirichlet root noise ported to Rust training path — 2026-04-10 (Q17 RESOLVED)
Commit `71d7e6e`. New `engine/src/mcts/dirichlet.rs` (Gamma-normalize sampler, `rand_distr 0.5`). `engine/src/game_runner.rs` calls `apply_dirichlet_to_root` on every turn boundary, both PUCT (line 550) and Gumbel (line 465) branches, with intermediate-ply skip (`moves_remaining==1 && ply>0`). Verification: 10 unique noise vectors at cm=0, top-1 prior `0.540 → 0.412`, top-1 visit fraction `0.474` vs §70 baseline `0.65` (−17.6pp). Tests: 108 Rust + 646 Python pass. Forensics: `archive/dirichlet_port_2026-04-10/`, `docs/sprint_archive/§073_*.md`.

### §74 — Gumbel vs PUCT loop audit — 2026-04-10
Three sub-resolutions: §74.1 `get_improved_policy` is PUCT-tree-safe (reads only fields populated by shared expand/backup primitives; unblocks `/analyze` on PUCT trees + Gumbel training targets on PUCT-built trees). §74.2 paired benchmark — batch_fill 100% for both variants on laptop 4060 (16 workers); cross-worker coalescing absorbs per-worker fragmentation; +9.4% Gumbel lead inside noise floor (IQR 39-46%). §74.3 Dirichlet parity regression test (`engine/tests/dirichlet_parity.rs`, 4 tests covering sum-to-one, linear blend, ε=0 noop, intermediate-ply gate). No code change to `game_runner.rs`. Forensics: `reports/gumbel_vs_puct_loop_audit_2026-04-09/`, `docs/sprint_archive/§074_*.md`.

### §77 — MCTS depth & ZOI scope — 2026-04-11
ZOI is post-search only (operates on root visit-count vector for move selection). MCTS tree expands with full radius-8 legal set at all depths. Measured at 200 sims: 360 root children created, 7 receiving visits (B_eff=6.1); mean leaf depth 2.92 plies; top-5 visit share 0.97. Decision: Option A (do nothing) — depth improves automatically as policy sharpens; correct lever is `n_sims` not tree pruning. `docs/01_architecture.md` §36 amended. Forensics: `reports/mcts_depth_investigation_2026-04-11/`, `docs/sprint_archive/§077_*.md`.

### §80 — Eval determinism fix: temperature + random openings — 2026-04-12
Root cause: `Evaluator` constructed `ModelPlayer` without `temperature` arg → defaulted to 0.0 → one-hot argmax → all 50 games bit-identical, BT CIs ±100K. Fix: `eval_temperature: 0.5`, per-game `np.random.seed(seed_base + i)`, `eval_random_opening_plies: 4` (random plies for both sides). Training path untouched. Old behaviour restored via `eval_temperature: 0.0`. Forensics: `docs/sprint_archive/§080_*.md`.

### §84 — Eval checkpoint retention (two-tier) — 2026-04-13
Storm: `checkpoint_interval=500 × max_checkpoints_kept=10 = eval_interval=5000` → every eval ckpt evicted by next eval (DB has BT ratings but weight files gone). Fix: `preserve_eval_checkpoints: true` default; `prune_checkpoints()` accepts `preserve_predicate` lambda exempting steps matching `s % eval_interval == 0`. Test `test_eval_checkpoints_not_pruned` pins the contract. Forensics: `docs/sprint_archive/§084_*.md`.

### §85 — A1 aux target alignment (Python + Rust) — 2026-04-13
Three compounding A1 subproblems: (1) `get_aux_targets()` pulled aux from a 200-entry game-level ring with independent random indices (no relation to batch indices); (2) one aux map per game broadcast across ~60 rows; (3) aux maps projected to game-end bbox centroid, replay rows to per-row cluster window centre (offsets ±9 cells). Fix Option A: per-row `ownership` + `winning_line` u8 columns reprojected at game end using each row's `(cq, cr)`; `apply_sym` extended for 12-fold scatter consistency; `sample_batch` 5-tuple → 6-tuple. ReplayBuffer +722 B/row. Rust commit `faafc43`; Python this entry. Kill criterion **revised at §91** (C1-C4 contrast + top-K). Forensics: `docs/sprint_archive/§085_*.md`.

### §86 — Structural split of `replay_buffer/` + `game_runner.rs` — 2026-04-13
Pure refactor. `replay_buffer/mod.rs` 1,102 → split into `{mod,storage,push,sample,persist,sym_tables}.rs`. `game_runner.rs` 1,313 → split into `game_runner/{mod,worker_loop,gumbel_search,records}.rs`. PyO3 surface stable. `Cargo.toml` feature wiring: `default = ["extension-module"]`, `test-with-python` escape hatch. 113 Rust tests pass with zero body modifications. Forensics: `docs/sprint_archive/§086_*.md`.

### §88 — Python training stack refactor — 2026-04-13
`scripts/train.py` 1,132 → 319 LOC (CLI + config + build core objects → `run_training_loop`). New modules: `hexo_rl/training/aux_decode.py` (69), `batch_assembly.py` (297), `loop.py` (680). `trainer.py` 746 → 720. Public API stable. 119 Rust + 676 Python tests pass. Forensics: `docs/sprint_archive/§088_*.md`.

### §89 — Threat-logit probe committed as step-5k kill criterion — 2026-04-13 (REVISED §91)
`scripts/probe_threat_logits.py` + `tests/test_probe_threat_logits.py` + `fixtures/threat_probe_positions.npz` (20 curated positions) + `fixtures/threat_probe_baseline.json`. Make targets: `probe.bootstrap`, `probe.latest`, `probe.fixtures`. FP32 forced, `torch.manual_seed(42)`, deterministic. **Criterion locked at §91** (C1 contrast + C2/C3 top-K + C4 warning-only). Forensics: `docs/sprint_archive/§089_*.md`.

### §90 — GPU util sweep: inf_bs / wait_ms levers exhausted — 2026-04-13
`(inf_bs, wait_ms)` sweep A=(64,4), B=(128,8), C=(128,4). Bottleneck is NN forward latency (12.5ms live vs 1.6ms bench), not batcher config. Raising inf_bs grows mean batch but forwards/sec collapses (workers can't supply 128 leaves in same wall-clock). **pos/hr is NOT a sufficient summary** when game length shifts — future sweeps must report steps/hr. Architectural levers (CUDA streams, process split, `torch.compile` re-enable) deferred to Q18. Forensics: `archive/sweep_2026-04-13_gpu_util/`, `docs/sprint_archive/§090_*.md`.

### §91 — Threat-probe criterion revised: colony-spam, not BCE drift — 2026-04-14
§85/§89 C1 (`ext_logit_mean ≥ baseline − 1.0`) was a BCE scale-drift detector misfiring on healthy runs (ckpt_00014344: contrast +3.94 IMPROVED while abs logits drifted globally negative — opposite of ckpt_19500). Replaced with **C1 contrast floor + C2 top-5 + C3 top-10**; C4 warning-only. `BASELINE_SCHEMA_VERSION` 2 → v6 later (real fixture, §106). Q19 `pos_weight ≈ 59` flagged for §92. Forensics: `docs/sprint_archive/§091_*.md`.

### §92 — Q13 + Q13-aux + Q19 atomic landing (PARTIALLY SUPERSEDED §97) — 2026-04-14
56-file atomic commit: 18→24 plane break, chain-length post-placement semantics, `aux_chain_weight=1.0`, threat `pos_weight=59`, HEXB v2 → v3. Pretrain v3 broken by F1 augmentation bug — fixed at §93 v3b. **§97 reverted chain planes from NN input** (chain stays as aux target in dedicated sub-buffer); design decisions survive. Q21 (wider-window aux) parked. Forensics: `docs/sprint_archive/§092_*.md`.

### §93 — Q13 fix-up + F1 root cause + pretrain v3b — 2026-04-15
10-commit fix-up on `feat/q13-chain-planes`. F1: `_apply_hex_sym` had two bugs — no `axis_perm` remap on planes 18..23 + `(col=q,row=r)` vs `(row=q,col=r)` convention mismatch. Fix: route pretrain augmentation through Rust `apply_symmetries_batch`. F3 tensor-buffer parity guard caught real divergence but in dead code (`TensorBuffer` only called by Python `SelfPlayWorker.play_game` — production uses Rust `SelfPlayRunner`); deleted at C9.5. v3b: 15 epochs, policy_loss 2.18, value_loss 0.50, 100/100 vs RandomBot. **Threat probe baseline v4**. Forensics: `reports/q13_fix_26_04_15.md`, `docs/sprint_archive/§093_*.md`.

### §94 — Experiment A: `aux_chain_weight=0` fresh run — 2026-04-15
Result: 47.7% draw rate at step 10312 vs smoke_v3b 44.7% at 5003. Chain aux confirmed **NOT primary draw-collapse driver**. Forces §95 (input ablation). Forensics: `reports/smoke_v3b_5k_26_04_15.md`, `docs/sprint_archive/§094_*.md`.

### §95 — Experiment C: chain-plane INPUT ablation — 2026-04-16
Audit confirmed chain planes computed at encode-time + stored, NOT recomputed at sample-time; zero planes are symmetry-invariant. Design: zero planes 18-23 after H2D decode (don't remove from architecture). `zero_chain_planes: bool` config flag. Wired in trainer, inference_server, probe_threat_logits. 5 tests. Forensics: `docs/sprint_archive/§095_*.md`.

### §96 — Experiment E: Gumbel MCTS desktop (vs laptop Exp D PUCT+CQ) — 2026-04-16
Hardware: Ryzen 3700x + RTX 3070 (desktop). Variant `gumbel_full`. Bootstrap v3b. Kill conditions relaxed per Exp D learnings. Q26 [WATCH]: nested `training:` block in `gumbel_targets_desktop.yaml` not picked up by deep-merger. Forensics: `docs/sprint_archive/§096_*.md`.

### §97 — Remove chain planes from NN input: 24ch → 18ch — 2026-04-16
KrakenBot architectural alignment. Chain learned via aux loss, not input ingestion. Removed redundancy. `GameState.to_tensor()` returns (K, 18, 19, 19); `HexTacToeNet.in_channels: 24 → 18`. Rust replay buffer chain planes moved to separate `chain_planes` sub-buffer (`6 × 361 × u16` per slot); HEXB v3 → v4. `sample_batch` returns 6-tuple. `apply_chain_symmetry` separate pass with `axis_perm` remap. Old HEXB v1-v3 buffers incompatible. Forensics: `docs/sprint_archive/§097_*.md`.

### §98 — Bench rebaseline post-18ch migration — 2026-04-16
Two failures: buffer_sample_augmented 1,663 µs (≤ 1,400 target) — real regression from split-pass scatter (state 18 + chain 6, two non-contiguous regions); worker_throughput 30,893 pos/hr (≥ 500K target) — warmup design bug (30s insufficient on laptop; p25=0 means at least 2/5 windows measured 0 positions). Rebaselined: `buffer_sample_aug ≤ 1,800 µs`; `worker_throughput ≥ 250,000 pos/hr` (PROVISIONAL — needs longer warmup). Forensics: `docs/sprint_archive/§098_*.md`.

### §99 — BatchNorm → GroupNorm(8) migration — 2026-04-16
Motivation: BN running stats drift from live distribution during self-play; batch=1 MCTS leaf eval uses stale stats. KrakenBot uses GN(8, 128) throughout. `_GN_GROUPS=8` module constant; `policy_bn` + `opp_reply_bn` removed (2 output channels too few for GN). `normalize_model_state_dict_keys` raises `RuntimeError` on pre-GN key patterns (prevents silent trunk corruption via `strict=False`). **BREAKS pre-§99 checkpoints**; retrain from scratch. Forensics: `docs/sprint_archive/§099_*.md`.

### §100 — Selective policy loss (move-level playout cap) — 2026-04-16
Per-move coin-flip between full-search (600 sims) and quick-search (100 sims); each position tagged `is_full_search ∈ {0,1}`. Policy + opp_reply losses gated on flag in Python; value/chain/ownership/threat apply to all rows. HEXB v4 → v5 (+1 u8/row, flag not under symmetry). §100.c review fixes: H1 RecentBuffer carries flag; H2 BN→GN auto-migration reverted (numerically wrong); M1 mutex (`fast_prob > 0` AND `full_search_prob > 0` raises at pool init); M3 opp_reply gated. **Defaults: `fast_prob: 0.0`, `n_sims_quick: 100`, `n_sims_full: 600`, `full_search_prob: 0.25`** (avg ~225 sims/move ≈ KrakenBot). §100.d threat probe baseline regenerated v4 → v5 against live bootstrap. Forensics: `docs/sprint_archive/§100_*.md`.

### §101 — Graduation gate with anchor model — 2026-04-16
Self-play workers consume `inf_model`. `inf_model` syncs from `best_model` anchor only at (a) cold-start and (b) graduation — **never** from drifted `trainer.model` on checkpoint ticks. Monotonic data quality between graduations. Gate: `wr_best ≥ promotion_winrate (0.55)` AND `ci_lo > 0.5`. §101.a review fixes: **C1 promoted weights ≠ evaluated weights** — `eval_model` allocated once in outer scope; promotion loads `best_model ← eval_model`. H1 `eval_interval` reads from `training.yaml` override. M1 `require_ci_above_half` default true (drops false-positive rate <1% vs ~9% naïve). M2 `resume_anchor_step_mismatch` warning. `_sync_weights_to_inf()` (wrong-direction) deleted. Per-opponent stride: `best=1 n=200 / sealbot=4 n=50 / random=1 n=20`. Forensics: `docs/sprint_archive/§101_*.md`.

---

## §103–§157 — Phase 4.0/Phase B' arc (KEEP-DISTILLED)

### §103 — Corpus zero-chain fix + `baseline_puct` playout-cap pin — 2026-04-17
Two drift bugs from §97/§100. §103.a: `load_pretrained_buffer` padded corpus chain planes with `np.zeros((T,6,19,19))` → chain head pulled toward zero on pretrain fraction of every mixed step. Fix: compute chain planes from stored stone planes 0,8 at NPZ load; route /6 normalisation through float32. §103.b: `baseline_puct.yaml` had no `playout_cap` override → inherited post-§100 `full_search_prob: 0.25`, silently §100-selective. Pin `playout_cap.full_search_prob: 0.0`. Tests `test_corpus_chain_target.py`, `test_baseline_puct_pins_pre_100_semantics`. Forensics: `docs/sprint_archive/§103_*.md`.

### §104 — D-Gumbel / D-Zeroloss instrumentation — 2026-04-17
Monitoring-only; no behavior change. `compute_policy_target_metrics` returns 7 fields split by `is_full_search`. Single `.cpu().tolist()` over 7 packed scalars replaces 7 `.item()` syncs (<200 µs/call on CUDA at B=256, A=362). NaN as first-class signal. Gate `monitoring.log_policy_target_metrics: true`. **D-Gumbel verdict: Option A** — quick-search CQ targets drift toward uniform (ΔH +3.5 nats, above +1.5 threshold); §100 selective gate correctly discards. `gumbel_full.yaml` Option A landed: `fast_prob: 0.25 → 0.0`. Resolves §100 known follow-up 3. Forensics: `reports/gumbel_target_quality_2026-04-17.md`, `docs/sprint_archive/§104_*.md`.

### §105 — Q27 perspective-flip smoke: W1 necessary, not sufficient — 2026-04-18 → 2026-04-19 (SUPERSEDED §106)
**Postscript:** §106 supersedes. C2/C3 20%/20% identical-FAIL was a synthetic-fixture artifact, not training pathology. The correctness argument for W1 (commit `e9ebbb9`, three call sites that failed `parent.moves_remaining==1` Q-negation — `get_improved_policy`, Gumbel score, `get_top_visits`) stands independently. Two-machine smoke (laptop pre_fix `723615e` vs desktop post_fix `a7efa78`), 5K steps each from `bootstrap_model.pt`. Forensics: `reports/q27_perspective_flip_smoke_2026-04-18/`, `docs/sprint_archive/§105_*.md`.

### §112 — Q33-C2 augmentation discriminator (E1 confirmed) — 2026-04-21
Unblocks §111 HALT. `feat(training): expose augment as training.augment config knob` (commit `eb17389`). Two 25-min smokes from `checkpoint_00017000.pt`, `w_pre=0`. **Result: |Δpe_Q4| = 0.049 ≪ 0.5 threshold** — augmentation-off does NOT reduce `pe_self`. **Verdict E1: pe_self ≈ 5.4 fixed point is self-play-distribution behaviour, not a 12-fold augmentation rotation bug.** Q33 / Q37 **RESOLVED (non-pathology)**. Phase 4.5 unblocked on pe_self premise. Config knob `training.augment: true` (default) with hard `ValueError` on missing key at loop entry. Forensics: `reports/q33c2_augmentation_discriminator_2026-04-21.md`, `docs/sprint_archive/§112_*.md`.

### §114 — bootstrap-v4: full-corpus retrain + eval (SUPERSEDED §148) — 2026-04-22
**L1, L15 origin.** Two silent corpus bugs found post-§70/§85: (1) `POSITION_END=50` truncation in `export_corpus_npz.py` discarded all ply ≥ 50 (~40% of positions); bootstrap was endgame-blind. (2) `update_manifest.py` read `player_black_elo`/`player_white_elo` (old format), missed `players[].elo` — 5694/5706 games unrated; Elo-weighted sampling effectively off. Fix sequence: `aa16624` Elo fallback; `ddd408f` drop POSITION_END cap (305,410 positions); `8b446c5` POSITION_END=150 (P95.5, trims time-scramble). Eval v4: C1 contrast +0.36, H2H WR 67%, SealBot WR 18.7% (n=150). Retcons Q17: Dirichlet (§73) was necessary but corpus completeness was the structural fix. **Rule: verify corpus before tuner**. Superseded by v5 (§118), v6 (§134), v7 (§148), v7full (§150). Forensics: `docs/sprint_archive/§114_*.md`.

### §116 — D-ladder investigation + torch.compile retry + landing/revert — 2026-04-23 → 2026-04-24
**D-ladder (curr_10k catastrophic forgetting).** Verdict: P-regressed (distributional), V intact on corpus. D1 curr WR ex-draws 6%; D2 deep matched 4% (deep regression); D3 KL on corpus 0.181 (close); D4 V MSE ratio 1.027 (matched). Mismatch = distributional. Smoking gun: D3-extra early-game synthetic probe — empty board curr argmax agreement with boot = 0%; ply 2-7 curr entropy 5.47-5.70 (≈ uniform). Curr forgot how to open. Hypothesis: replay buffer under-covered early-game positions during sustained run. Revert live ckpt to `bootstrap_model.pt`. Forensics: `reports/investigations/diag_D_20260423/`.

**torch.compile retry GO.** Both §32 blockers resolved on PT 2.11 + Py 3.14: TLS crash gone, Triton 27 GB spike gone (59.5 MB peak). reduce-overhead: 1.50× throughput / 1.87× latency vs eager, 6.4 s compile. **§116.a landing on master (`1e2d82b` + `41ffad5` resume/OptimizedModule fixes), then REVERTED (`e102a0a`)** after second resume deadlock at step 6002 (futex_do_wait on 78 threads — trainer+inference dual-JIT contention). Mode-plumbing + OptimizedModule unwrap fixes stay. **Re-enable preconditions:** deadlock repro harness OR compile-sequencing OR training-loop smoke. Forensics: `reports/investigations/torch_compile_retry_20260423/`, `docs/sprint_archive/§116_*.md`.

### §117 — TF32 + channels_last probe + per-host autodetect — 2026-04-23
**L7 corollary.** Four-arm matrix × two hosts. TF32 cross-host divergent: sm_89 (4060) −5.8% latency; sm_86 (3070) +5.9% (FP32-tail Linears route to small-K TF32 kernel that serializes poorly). channels_last −7 to −17% on both (SE block `s.view(b,c,1,1)` breaks CL propagation 12× per forward). **Decision: TF32 per-host autodetect (`_TF32_MEASURED: sm_86→False, sm_89→True`); channels_last rejected.** Replaces unconditional `torch.set_float32_matmul_precision("high")`. Forensics: `reports/investigations/tf32_channels_last_20260423/`, `docs/sprint_archive/§117_*.md`.

### §118 — Early-game forgetting fix wave — 2026-04-23 → 2026-04-24
**Verdict — root cause:** `pe_self ≈ 5.4` is self-play-starvation rate on **off-canonical** early-game positions, not policy collapse. Under prod (`decay_steps=20000`, `full_search_prob=0.25`), only ~13% of batch policy-gradient came from fresh SP rows; ply 2-7 off-canonical drifted toward `log(N_legal)`. **Axis is canonical vs off-canonical, not ply depth.** Smoke discriminator: A fsp=0.5 → 26% SP-grad share, H 3.97; B decay=2500 → 17.6%, H 3.32. Landed: `early_game_entropy` 10-pos probe; `selfplay.random_opening_plies: 4`; `dirichlet_alpha: 0.3 → 0.05`; `full_search_prob: 0.25 → 0.5`; `pretrain_max_samples: 200_000 → 0`. Phase 5 validation (bootstrap-v5, step 6000): early_game_entropy 3.55 (gate <4.0), threat C1 +3.44, throughput +10%. Q8/Q33/Q37 framing flipped to "sampling-rate starvation". Forensics: `reports/investigations/phase5_validation_20260424/`, `docs/sprint_archive/§118_*.md`.

### §119 — Main-Island Neglect: mechanism located, RecentBuffer augmentation gap — 2026-04-25
User-flagged visual pattern: parallel horizontal formations at equidistant spacing, main island neglected. **5-hypothesis discriminator cascade.** D1 history ordering RULED OUT (10% counterfactual disagreement). D2 windowing coverage RULED OUT (100% largest-group coverage). D3/D4 rotation equivariance PARTIAL (board-coord agreement 51.5% — model is axis-asymmetric but doesn't explain E-W preference). **D5 trajectory: self-play E-W axis share 65% vs corpus 38% (+27pp).** D6 augmentation audit MECHANISM CLOSED: RecentBuffer rows are sampled at Python call site without `apply_sym` LUTs; at step >10k RecentBuffer contributes ~67% of batch → 67% of policy-gradient rows see identity-only symmetry coverage → absolute-position FC policy head learns axis-asymmetric features freely. **Causal chain**: un-augmented RecentBuffer → asymmetric features → axis-biased MCTS → axis-biased trajectories → buffer reinforces 67%. Decision: Option A (augment RecentBuffer in §120). Methodology lesson: when visual pattern is stable + geometrically specific, treat geometry as falsifiable hypothesis first. Forensics: `reports/investigations/main_island_d{1..6}/`, `docs/sprint_archive/§119_*.md`.

### §121 — Directional bias resolves, clustering magnitude is architectural — 2026-04-25
Seven diagnostics (D10-D16) closed mechanism account of axis-clustering. **Two independent components**: (1) Directional heuristic (within-turn, rotation-equivariant) — model places 2nd stone far W of 1st (W=38.2% selfplay vs corpus 12-14%); washes out under D16 rotation probe (W=12.3%, W/E=0.96). Fixed by **permanent self-play rotation from ply 0** (landed §130). (2) Clustering magnitude (cross-turn, rotation-invariant) — aggregate axis density 0.60 per axis doesn't shift under rotation; learned strategic prior "identify axis early and cluster"; preserved by rigid transformation. **Architectural**: trunk lacks inductive bias for hex-axis strategies at right abstraction level. §120 RecentBuffer augmentation closed symmetry gap (~4.7/12 → 12/12 elements per batch row) but cannot correct relational biases. Methodology lesson: split signal (different components resolve differently) is more informative than uniform PASS/FAIL. §122 opens for architectural redesign. Forensics: `reports/investigations/phase121_d{10..16}*/`, `docs/sprint_archive/§121_*.md`.

### §124 — InferenceServer dispatch fix: TorchScript trace + bench methodology shift — 2026-04-25
**L17 corollary.** TorchScript-trace `InferenceServer.forward` at `__init__` via `torch.jit.trace`; gated by `selfplay.trace_inference` (default true); falls back to untraced module if trace raises (e.g. compile-wrapped). ScriptModule shares parameter storage so `load_state_dict_safe` propagates without re-tracing. Merges D2H (`cat(probs, value)` → single `.cpu().numpy()`). Eliminates ~32.6% CPython `nn.Module._call_impl` dispatch cost per forward (py-spy 200Hz, 180s, n_workers=10, 3070). **+34% pos/hr on 3070.** Trace neutral on EPYC 4080S (still GPU-bound). Why trace not compile: simpler (no Dynamo guard, no cudagraph TLS, no Triton spike); ~matches compile throughput on GPU-bound HW while lifting dispatch-bound HW. `make bench` switches to compile OFF + trace ON as production gate; `make bench.compile` retained for engineering datum. NN inference target lowered 6,500 → 4,000 pos/s. Forensics: `docs/sprint_archive/§124_*.md`.

### §125 — EPYC 4080S sweep verdict + py-spy → perf_timing — 2026-04-25
**L7 origin.** Validate sweep on EPYC 7702 + 4080S: `n_workers=24, batch=192, wait=4ms, leaf=8, burst=16` → 377k pos/hr. n_workers=16 regressed 388k → 234k with trace (workers can't refill 192-batch before GPU drains). py-spy 0.4.2 does not support Py 3.14 — replaced with `diagnostics.perf_timing: true` (per-batch fetch_wait_us / h2d_us / forward_us / d2h_scatter_us). **Profile result (n_workers=24, batch=192, trace ON):** forward = 80.4% of cycle (p50 11.96 ms). EPYC 4080S is **GPU-compute-bound, not dispatch-bound** — 3070 py-spy extrapolation was wrong. Next lever: torch.compile on top of trace (compile fuses CUDA kernels trace cannot); stack path `torch.compile(model).then(jit.trace(compiled._orig_mod, ex))` verified possible, not yet implemented. Forensics: `docs/sprint_archive/§125_*.md`.

### §127 — Top-K leaf cap eliminates MCTS pool overflow — 2026-04-28
**L5 origin.** 5090 96-thread sweep saw `mcts_pool_overflow_count > 0` every cell. Root cause: leaf expansion created one child per legal move; once 100+ stones spread out, radius-8 hex ball per stone unions into 1k+ cells; worst-case nodes per search = `n_sims × leaf_batch × n_legal` blew past `MAX_NODES = 1M`. Pre-existing mitigation (fabricate `is_terminal=true` with quiescence-corrected NN value via AtomicU64 counter) was a **hot-path silent-corruption sink** — biased visit counts and value targets, contaminated runs dropped after the fact. Fix: `MAX_CHILDREN_PER_NODE: usize = 192` (sorted by NN policy prior desc, tie-break flat_idx asc — deterministic regardless of FxHashSet order). Fast path with no sort when `legal_moves.len() ≤ K`. Fabricated-terminal removed entirely; counter retained for telemetry, then panic. **Bound**: `400 × 8 × 192 ≈ 614k slots fits 1M` (~38% headroom). Tests in `mcts/mod.rs` + `engine/tests/pool_overflow.rs`. Forensics: `docs/sprint_archive/§127_*.md`.

### §128 — Bench metric `positions_generated` replaces `positions_pushed` — 2026-04-28
**L6 origin.** `worker_pos_per_hr` measured via `pool.positions_pushed` which increments by K cluster views × 1 per ply at **game completion** (batch write). 120s window vs ~160s game → most windows capture zero completions → bimodal IQR 80.9%. `positions_generated` is per-ply continuous AtomicUsize — continuous, no burst, stable. Relationship: `positions_pushed = K_avg × positions_generated`, K_avg ≈ 7 empirically. Targets updated: CUDA 142,000 → 20,000; MPS 200,000 → 25,000; CPU 80,000 → 11,000. Desktop n=5 baseline: 27,835 pos/hr IQR ±8.6%, all 5 runs unimodal. **Pre-§128 throughput numbers obsolete**. Forensics: `docs/sprint_archive/§128_*.md`.

### §131 — 18 → 8 plane migration (P1+P2+P3) — 2026-04-29
**KEPT_PLANE_INDICES = [0,1,2,3,8,9,10,11]** (cur ply-0..3 + opp ply-0..3 in 18-plane space; positions 0-7 in 8-plane). Ply-4..7 history + scalar metadata dropped. Dense > sparse for `ckpt_14000` conditioning surface preservation. STATE_STRIDE 6498 → 2888. Chain_planes unchanged at 6 planes. **P1 Rust**: `N_PLANES: 18→8`, HEXB v5 → v6 with `n_planes` header validation, slice-on-push integration; inference path untouched. **P2 Python**: `BUFFER_CHANNELS=8`; `pool.py` `_feat_len = 2888`; `batch_assembly.py` 8→18 scatter bridge in `_train_on_batch` (temporary). Corpus regenerated 1,090,296 positions `(N,8,19,19)`. **P3 Model**: `HexTacToeNet.in_channels: 18→8`; hard-reject guard `trunk.input_conv.weight.shape[1] == 18 → RuntimeError`; bridge block removed; `InferenceBatcher`/`SelfPlayRunner` defaults 6498 → 2888; `worker_loop.rs` `slice_kept_planes_18_to_8` deleted (encode sites call `encode_state_to_buffer_channels` directly). 958 py + 138 rs lib + 35 rs integration pass. Forensics: `docs/sprint_archive/§131_*.md`.

### §137 — W3 validation gates: Q41 WARN + Q52 PASS + Q44 done → Phase 4.0 UNBLOCKED — 2026-04-30
Q41 v6 vs v5 H2H (n=200): 102/200 (51.0%), Wilson [44.1%, 57.8%] — WARN (near-parity). Old gate `lower-CI ≥ 50%` fired even at exact parity; **revised**: PASS ≥ 48%, WARN [43%, 48%), BLOCK < 43%. Q52 v6 vs SealBot (n=150): 36/150 (24.0%) Wilson [17.9%, 31.4%] — **PASS** (gate ≥ 14%). Beats v4 anchor 18.7% by +5.3pp. **Colony-win fraction: 5.6% vs v4's 82%** — low colony fraction is **POSITIVE** (colony wins caused self-play training explosion; §131 channel cut dropped colony-related planes; v6 wins via 6-in-a-row). Q44 laptop bench refloor: 33,174 pos/hr (+19% vs desktop §128). Phase 4.0 UNBLOCKED. Forensics: `docs/sprint_archive/§137_*.md`.

### §141 — W4C policy-head diagnosis: policy intact, locus is search/encoding — 2026-05-01
ckpt_5500 (W4C smoke §138) recorded 1.3% SealBot WR vs bootstrap-v6's 24%. §139-§140 cleared value head + rotation LUT. **Policy probe (n=200 × 5 metrics × 4 categories × 2 models)**: H(p) decreased on real positions (sharpened, not flattened — falsifies Hypothesis A); Spearman ρ ≥ 0.66 on corpus/sealbot/threat; top-1 agreement ≥ 64%; threat extension prob retained at 94% of bootstrap. Colony positions diverged (top-1 agreement 18.5%, rank 201/362) — desired §137 behaviour, not defect. **Verdict: policy head NOT regression locus.** Protocol fixes (pretrain floor 0.1→0.5, max_game_moves 200→100) unlikely to help. Self-play board-extent 329 cells, NN window 19×19 (±9) — most board invisible. Locus: **search/encoding boundary**. Forensics: `reports/w4c_diag/policy_diagnosis.md`, `docs/sprint_archive/§141_*.md`.

### §142 — Encoding-window coverage audit: ply-31 fragmentation pivot — 2026-05-01
ckpt_5500 selfplay crosses 19×19 single-window boundary at **ply 31** (median pct_outside 0% → 21.9%, sharp). Any-cluster windowing delays onset but doesn't prevent: 8/16 draws end with ≥80% of stones invisible. End-of-game single-window blindness median 97.7% on draws. **Pathology is distribution-endogenous** — against SealBot (5 games) max ply 29, 0% outside throughout; fragmentation only emerges when two mutually permissive policies play each other. Axis: q (NE-SW) consistent with §138 axis_density. **Recommendation: Option γ (tighten exploration)** — Option α (cap LEGAL_MOVE_RADIUS) is fallback. Forensics: `reports/w4c_diag/encoding_audit.md`, `docs/sprint_archive/§142_*.md`.

### §146 — Option α' implementation: cap LEGAL_MOVE_RADIUS 8 → 5 — 2026-05-02
`engine/src/board/moves.rs:9` `LEGAL_MOVE_RADIUS: 8 → 5`. CLUSTER_THRESHOLD untouched (still 8 — governs colony adjacency). Tests: 216 → 90 (single-stone hex ball: 91-1), new `legal_move_radius_capped_at_5`. Laptop smoke (21 games, bootstrap-v6, gumbel_full): draw_rate 0/21, mean game length 16 plies (R=8 was ~110). With p=0.2 the n=0 draws prob is 0.9% — real shift. **Bandaid; SUPERSEDED at §156 (cosine-temp is the real load-bearing knob).** Forensics: `docs/sprint_archive/§146_*.md`.

### §147 — Bootstrap corpus contamination audit — 2026-05-03
**L1, L15 origin.** Pre-§148 audit of `data/bootstrap_corpus.npz` (v6) found it was generated by `make corpus.export` (all sources), NOT `make corpus.export.pretrain` (human-only). `pretrain_human_only: true` flag was not honored. Bot games made up **41% of raw game count** at `source_weight=1.0`. Per-position Elo-band weights also not applied. **v6 anchor numbers tainted** — Q41 51% parity and Q52 24% vs SealBot trained against bootstrap that learned partly from bot-style play. Phase 4.0/Phase B' diagnostics (§141-§144) inherit caveat. **Decision: rebuild v7 from scratch with `make corpus.export.pretrain`**. Preserve v6 as `bootstrap_corpus_v6.npz`/`bootstrap_model_v6.pt`. Forensics: `docs/sprint_archive/§147_*.md`.

### §148 — Corpus rebuild: v7 human-only Elo-weighted — 2026-05-03
**Elo-weight bug fix in `scripts/export_corpus_npz.py`:** when `--max-positions` omitted (uncapped), `rng.choice(n, n, replace=False, p=w)` degenerates to permutation — per-position Elo weight had no effect on which positions kept; `weights_out = np.ones(...)` made `WeightedRandomSampler` sample uniformly. Patched: when uncapped, save per-position `source_weight × elo_band_weight / game_length` as `weights_out`. v7 corpus: 6,259 human games (≥15 plies decisive), 355,271 qualifying positions, 353,091 sampled, 2,435 MB, fp16 `(N,8,19,19)`. Elo bands: sub_1000=81,985 (w=0.5), 1000_1200=202,111 (w=1.0), 1200_1400=69,739 (w=1.5), 1400+=1,436 (w=2.0). v7 retrain (15 ep, batch 256, ~97 min on 3070): final loss 3.31, val_acc 0.75, 100/100 vs RandomBot. **SealBot n=200**: v7 16% vs v6 11% (z=1.46, p=0.14 n.s.); H2H 49%; threat C1 +0.00 (corpus-shift artifact) but C2 45% / C3 75% PASS. **Promoted**. HF push `timmyburn/hexo-bootstrap-{corpus,models}` versioned + canonical. Forensics: `reports/corpus_v7/promotion.md`, `docs/sprint_archive/§148_*.md`.

### §149 — v7 verification + v7e30 fine-tune promotes — 2026-05-03
Pretrain saturation audit on v7 found last-3-epoch cumulative Δ = 1.6% of total descent — fails strict <1% gate (cosine LR reached eta_min, idled). Patched `pretrain.py`: `--resume`, `--lr-peak`, `--inference-out` flags. **v7e30 fine-tune**: resumed `pretrain_00000000.pt`, fresh cosine peak `5e-4 → 1e-5` for 15 more epochs; final loss 3.246. **SealBot n=500**: v6 11.4%, v7 13.2%, v7e30 **16.4%** (z=2.29 vs v6, **p=0.022 significant**). Threat C2/C3 preserved through fine-tune; C1 still flat (corpus-shift + flatter v7-family distribution). Human-fixture C1 probe: v7 +0.06 vs v6 +0.51 (lower), but v7 C2 42% vs v6 25% (+17pp) — flatter policy with preserved/improved top-K ranking. **v7e30 promoted to canonical**. Hygiene: `make train.fresh` wipes `replay_buffer.bin`; `buffer_state_at_corpus_load` event; Q41 BLOCK threshold 43% → 38%. Forensics: `docs/sprint_archive/§149_*.md`.

### §150 — v7full: 30-epoch full retrain promotes — 2026-05-03
**Canonical bootstrap anchor.** User ran full retrain on vast.ai 5080: single-cycle cosine 30 epochs, peak `2e-3`, **`eta_min=5e-5`** (raised from `1e-5`). Wall ~83 min. Final loss **3.1573** (vs v7e30 3.2462, Δ -0.089). `--eta-min` flag added (`1f822ae`). **SealBot n=500: 87/500 = 17.4% [14.3%, 21.0%]** (z=2.70 vs v6, **p=0.007 significant**; vs v7e30 z=0.42 p=0.67 n.s. but every metric directional + v6-anchor edge becomes significant). Threat: C1 +0.20 (partial recovery from v7e30 0.0 toward v6 0.6); C2 50% / C3 70% **PASS**. Colony wins 12/87 = 13.8% (line baseline). Promoted; canonical `bootstrap_model.pt` = v7full sha `29306533…`. **Recipe locked**: 30 ep cosine, peak `2e-3`, `eta_min=5e-5`. Forensics: `reports/corpus_v7/threat_probe_v7full.md`, `docs/sprint_archive/§150_*.md`.

### §152 — Phase B' instrumented smoke: Class-4 dominant — 2026-05-04
Run `w4c_smoke_v6_instrumented_5080` aborted at step 2560/5000 after four-class signal saturated. **Verdict matrix**: Class 4 (q-axis stride-5 distance-5 spam) DOMINANT, ρ(stride5_run, is_ply_cap) = +0.50, p = 5e-42; Class 3 (buffer composition) STRONGLY ACTIVE downstream; Class 2 (value-head drift) ACTIVE downstream (dec=−0.69±0.03); Class 1 (stale dispatch) NOT TESTED (eval_interval=5000 zeroed model_version). **Causal story**: v7full mildly prefers stride-5 east-west extensions; smoke γ knobs + Dirichlet + playout_cap full=0.5 sims=600 + cosine temp amplify into dominant policy mode (cap_max median 30 vs T2 baseline 3 = 10×). 87% cap-rate → 98% draw-coded buffer → value head trains to overshoot draw_value to −0.69 → reinforces cap-prone policy. **Pattern**: mixed-color stones along single hex row at distance-5 spacing (`x____o____o____x_`). Persists despite §130 rotation (per-game uniform across dihedral group). Existing macros miss it: `colony_extension_fraction` gates at hex_dist > 6; `axis_distribution` measures distance-1 adjacency. v8 priority order written. Forensics: `reports/phase_b_prime/instrumented/diagnosis.md`, `docs/sprint_archive/§152_*.md`.

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

Forensics: `reports/phase_b_prime/training_knob_isolation/`, `reports/phase_b_prime/5k_smoke/results.md`, `docs/sprint_archive/§155-§157_cosine-temp-draw-collapse-arc.md`.

---

## §158–§163 — Hygiene + Refactor waves — 2026-05-06 [MERGE]

**Hygiene wave (§158, §158a, §158b):** L3 variant-config cleanup. §158 retired 6 superseded variants (`smoke_A/B`, `w4c_smoke_v6_*` family) + 3 stale docs. §158a coordinated wave: `phase118_recovery.yaml` + dead `dry_run_batch.py`; `calib_R1-R4.yaml` + `run_calibration_run.sh` + Makefile target; `sweep_*ch.yaml` + Phase-1 harness (`run_sweep.py`, `tournament_sweep.py`); `baseline_puct.yaml` + coordinated references. 924 py pass post-wave. §158b L8 Stage 3 disk reclaim: ~9 GB freed (workspace 48 → 39 GB); cumulative L8 total ~58 GB. Commits `98722cb`, `33a324f`, `c1fceaf`, `96f0b27`, `f777922`, `f8c5ccc`. Forensics: `docs/sprint_archive/§158-§158b_hygiene-wave-2026-05-06.md`.

**Refactor wave (§159, §159a, §160, §160a, §161, §162, §163):** Audit-driven structural splits, all FF-merged to master.
- **§159**: `hexo_rl/training/loop.py` 1464 → 686 LOC (-53%). Five new modules: `anchor.py` (anchor I/O), `signals.py` (signal handlers), `orchestrator.py` (training hooks), `lifecycle.py` (subsystem builders), `eval/pipeline_setup.py`. 8 commits + sprint log.
- **§159a**: StepCoordinator extraction. `loop.py` 686 → 357 (-48%); new `step_coordinator.py` 893 LOC (closure → class; 18 per-step decisions O2-O6/D0-D10 preserve identical ordering DAG). +20 behavioural tests. Constraints R1-R25 honored. 4 commits.
- **§160 / §160a**: `eval_pipeline.py` 529 → 472. New `gate_logic.py` (GateConfig + `evaluate_gate` + ci_confidence wired via `norm.ppf`), `reporting.py`. **2 of 4 audit claims stale** (BT + SQLite already extracted pre-§160). §160a: `_load_anchor_model` prefix-strip dedup'd to `normalize_model_state_dict_keys` (inherits BN-key + 18-plane guards).
- **§161**: Q-§159b §B item 15 lifecycle.py coverage (+5 tests for build_subsystems, resolve_anchor branches, teardown).
- **§162**: `selfplay/pool.py` 705 → 539. New `instrumentation.py` 176 (PoolInstrumentation: per-worker draws, terminal-reason counts, mv_range history, recent move histories). Stride-5 abort path retired (Q-§162a WATCH); P90 stays passive event field. Bench gate PASS.
- **§163**: `mcts/mod.rs` 1493 → 974 (-519). New `mcts/policy.rs` 533 (5 methods + 11 tests: get_policy, get_improved_policy, get_root_children_info, apply_dirichlet_to_root, get_top_visits). Pure-move (subagent audit verified). **Audit row PARTIALLY STALE**: 6 of 8 listed concerns already in `selection.rs`/`backup.rs` pre-§163. Bench gate PASS.

Pattern: audit's L9 rows treated as low-fidelity index (claims stale at §160, §160a, §162, §163). Forensics: `docs/sprint_archive/§159-§163_refactor-wave-2026-05-06.md`.

---

## §164–§174 — Encoding migration arc (KEEP-DISTILLED)

### §164 — Phase 5+ entry probe wave: P1 anchor / P2 perception (CATASTROPHIC) / P3 corner-mask — 2026-05-07
Three probes dispatched in parallel; nothing landed on master.
- **P1 (Principled — memory misread).** Live self-play forwards all K cluster views to NN, min-pools value, scatter-max policy. Replay buffer push emits one row per cluster per ply. Index-0 picks exist only in `pretrain.py:564,568` (Aug-only RandomBot validation) + `early_game_probe.py:103` (Aug-only monitoring fixture). Bootstrap corpus picks **first cluster covering played move**, not index 0 — Principled by design.
- **P2 (CATASTROPHIC).** Rule: r=8 placement (hexo v0.2.0); our perception: r=5. **Bot WR 78% vs `far_line` (scripted r=6-8 6-axis adversary)** with 22% opp colony reach-6; brain-dead script outperforms strongest engine's 17.4% empirical edge vs SealBot. Mechanism: stones at hex_dist 6-8 from any bot stone form their **own cluster** (`CLUSTER_THRESHOLD=5`) with their own window; policy treats them as low-priority. **Deployment hotfix REQUIRED before hexo.did.science deploy.** Three options: (a) bump constants to 8 (re-opens fragmentation), **(b) hybrid r=5 selfplay + r=8 inference (PREFERRED short-term)**, (c) 25×25 window (large scope; encoding migration). PyO3 bindings for `set_legal_move_radius` + `set_cluster_threshold` missing.
- **P3 (Neutral within noise).** Corner-mask (hex_dist ≤ 9, 271 cells survive of 361) bench A/B on 5080: aug −8.12%, worker +20% (borderline noise). Smoke OFF vs ON last-100: draws 11% → 7%, X/O/draw 52/37/11 → 47/46/7 (better balance), colony_extension_fraction max 0.030 → 0.0. **SealBot Δ = −2.5pp NOT statistically significant**. Mask safe to ship. **D3 verdict: adopt inscribed hex** on §152 dihedral-symmetry argument.

Forensics: `audit/probes/{p1,p2,p3}*.md`, `reports/probes/p3_smoke/`, `docs/sprint_archive/§164_*.md`.

### §165 — v8 encoding migration design + spike wave — 2026-05-07
Three parallel spike subagents. **S1 bbox crop**: fixed-max single-tensor bbox-of-all-stones at HALF=16, BBOX_SIDE=33, m=8, 9-11 planes, K-aggregation REMOVED, N_ACTIONS=1090. **S2 KataGo GPool**: 2 sites at `{6,10}` (50%/83% of 12-block trunk), `KataConvAndGPool` operator verbatim, `c_main=128, c_mid=128, c_gpool=32`. Trunk FLOPs DECREASE 2.1% at 19×19. SE × GPool compose cleanly. Policy + opp_reply FC → KataGo 1×1 conv + linear_pass = **−482k params (FREE WIN)**. **S3 v8 corpus regen + cutover plan**: ~10-15 min corpus + ~3 hr retrain = 3.25 hr wall on 5080. Raw human JSONs persisted; no re-scrape needed. **Final v8 spec Path β**: 25×25 + 11 planes + R=8 + 96-channel trunk + GPool {6,10} + KataGo policy head + N_ACTIONS=625 (no pass). Phase A-E sequence: short-lived branches off master, FF-merge on bench-gate PASS. Forensics: `audit/encoding_spikes/{s1,s2,s3,SPIKE_SUMMARY}.md`, `docs/sprint_archive/§165_*.md`.

### §166 — Phase A: encoding pipeline core (gated coexistence) — 2026-05-07
Operator-revised strategy: NOT hard cutover. v8 lands as **gated coexistence**: v6 path canonical default + byte-exact; v8 path opt-in via `configs/model.yaml encoding.version: v8`. Preserves rollback envelope, unblocks Phase B without putting v6 at risk. **4 Phase A commits + 1 contract** on `encoding/phase_a_pipeline`:
- Bucket C (`ad8dd10`): `EncodingSpec` NamedTuple + `resolve_encoding(config)` in `hexo_rl/utils/encoding.py`; v8 constants in `constants.py` (BOARD_SIZE_V8=25, NUM_CELLS_V8=625, BUFFER_CHANNELS_V8=11, N_ACTIONS_V8=625, LEGAL_MOVE_RADIUS_V8=8, MARGIN_M_V8=8); `configs/model.yaml` encoding section default v6.
- Bucket A (`ee2de0b`): Rust `sym_tables.rs` shape-parameterized; v8 const symbols (STATE_STRIDE_V8=6875, CHAIN_STRIDE_V8=3750, POLICY_STRIDE_V8=625, AUX_STRIDE_V8=625, HALF_V8=12); `SymTables::with_shape(board_size, n_planes)`; `src_plane_lookup` `[[usize;N_PLANES];N_SYMS]` → `Vec<Vec<usize>>`; `apply_symmetry_state` / `apply_chain_symmetry` read `n_cells` from `sym_tables.n_cells`.
- Bucket B (`66b9f9c`): `hexo_rl/bootstrap/dataset_v8.py` (replay_game_to_triples_v8, encode_position_v8, bbox-of-all-stones centroid integer-truncation + n_clipped telemetry); `_compute_chain_planes` shape-derived; `get_policy_scatters(board_size=25, has_pass=False)`; `--encoding v6/v8` flag on `export_corpus_npz.py`.
- Bucket D (`b47136c`): **P2 hotfix-(c) bundled** — `Board.set_legal_move_radius(r)` + `legal_move_radius()` PyO3 bindings; dataset_v8 replay Board uses R=8.

**Tests**: 1028 py + 151 Rust lib + 6 Rust integration GREEN (41 net new: 9 resolver + 8 v8 sym + 28 dataset_v8 inc. 4 P2). v6 999 py + 143 Rust unchanged → byte-exact regression guard satisfied. **Bench gate** (n=5 post-Phase-A laptop): 8/10 PASS, 2 WATCH (`buffer_push_per_s` -7.7%, `worker_pos_per_hr` -6.2%); n=5 close-out confirmed both as boost-clock variance noise (§102). 5080 baseline for Phase B captured. Forensics: `reports/encoding_phase_a/`, `docs/sprint_archive/§166_*.md`.

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

**SealBot WR (argmax-only, n=200, t=0.5): ALL 5 v8 arms = 0/200 = 0%** [0%, 1.9%]. v7full v6-argmax baseline: r=5 6.5% / r=8 12.5% / r=10 15%. B1 across radii r=8/10/12: 0%/0%/0%. **The v8 0% is NOT v8-architecture falsification** — structural limitation of argmax-only cross-encoding eval. K-cluster's inference-time multi-window pooling acts like a tiny "ensemble" that bbox lacks. Both effects vanish under MCTS (Phase D §168). **Canonical pick: B1** (best clean loss 3.227). Threat probe deferred (v6-only fixture). bbox clip rate ~6%/stone (above S1's 1% Path α trigger) — Path α escalation deferred. **v7full radius curve 6.5% → 12.5% → 15% confirms cross-encoding gap real**; memory `feedback_v6_v8_same_training_data.md` (corpora share 6,259 raw games — encoding changes density only) + `project_v8_argmax_handicap.md`. Forensics: `reports/encoding_phase_b/VARIANT_SUMMARY.md`, `docs/sprint_archive/§167_*.md`.

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

**Commits**: `3f2bf10` (eval harness), `ed440a3` (Rust v6w25 constants), `0c62138` (pretrain + HexTacToeNet wiring). 1085 py + 151 Rust unit + 6 Rust integration PASS. T3 matched-MCTS comparison deferred to §169. Forensics: `docs/sprint_archive/§168_eval-harness-v6w25-plumbing.md`.

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

Commits: pool registry + wire `pool_type='pma'` + retrain config + A2 eval; A3 global encoder + crop helper + pretrain + eval; A4 canvas_realness + PartialConv2d + retrain + eval; §169a probe. Forensics: `reports/ablation_169/{A1-A4}_*.json`, `docs/sprint_archive/§169-§170_four-way-ablation-wave.md`.

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

Forensics: `reports/gpool_bias/`, `docs/sprint_archive/§170_gpool-bias-retrain-wave.md`. Branch `encoding/four_way_ablation` retired after canonical pin. Memory: `project_170_p3_falsified.md`.

### §171 — Sprint scoping + blocked handoff + §171 A4 P2-reopen (DEAD) — 2026-05-09 → 2026-05-11
Branch `phase4/p171_selfplay_smoke`. P0 corpus-sha manifest transposition corrected (`bootstrap_corpus_v8.npz` is vanilla v8 sha `adb88412…`; canvas_realness variant is `bootstrap_corpus_v8_canvas_realness.npz` sha `110ea6b2…`). **P3 BLOCKED** by two-layer issue: Layer 1 canvas vs trunk `board_size` semantics inconsistency (A2.1 trainer trunk=25 vs A2.2 pool guard canvas=19); **Layer 2 load-bearing**: `Board::to_planes()` hardcoded 18×19×19 regardless of `cluster_window_size` — selfplay's `state.to_tensor()` calls single-window path; v6w25 pretrain worked because `dataset_v6w25.py` uses `get_cluster_views()`. v6w25 sustained selfplay requires **α multi-window** (§173). Recommendation: Option α structural fix in §172.

**Pin scope (§171 P3.6):** v6w25 canonical for pretrain+eval+matched-MCTS; v7full canonical for selfplay pending α; α = Phase 4.5+ scope.

**§171 A4 P2-reopen C (DEAD).** 2026-05-11. Distribution-shift fine-tune side-arm on `phase4/encoding_registry`. Pre-registered: E1 ALIVE `MCTS-64 > 8% AND CI_LB > 5%`; E2 DEAD `MCTS-64 ≤ 2% AND CI_UB < 4%`. Recipe: resume `A4_canvas_realness.pt`, mixed corpus 95% bootstrap + 5% adversarial via `WeightedRandomSampler`, 3000 steps, peak LR 5e-5/eta_min 5e-6, freeze input_conv + tower[0..7], trainable tower[8..11] + heads (35.2%). **Result: MCTS-64 0/200 Wilson95 [0.000%, 1.88%] — DEAD cleanly met.** §169 P0 SPATIAL_RICH framing FALSIFIED for frozen-spine class. Closes bbox+canvas_realness+frozen-spine line. **Correction**: original close-out claimed "22% → 0% argmax collapse"; the 22% was misattributed (§170 P3 A1+gpool-bias, not A4 baseline). A4 was already 0% pre-fine-tune (`reports/ablation_169/A4_eval.json`); fine-tune did not damage anything new. DEAD verdict rests strictly on MCTS-64 axis. Forensics: `reports/sprint_171_a4/`, `docs/sprint_archive/§171_sprint-scoping-blocked-handoff.md`. Memory: `project_171_a4_p2_reopen_c_dead.md`, `project_bootstrap_argmax_drift_check_20260511.md`.

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

Forensics: `reports/sprint_172_summary.md`, `docs/sprint_archive/§172_encoding-registry-sprint.md`. Memory: `project_172_a{1..10}_complete.md`, `project_172_b{1,1_redo,2}_complete.md`. Encoding registry contract: `docs/designs/encoding_registry_design.md`.

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

Forensics: `reports/sprint_173/`, `docs/designs/encoding_alpha_multiwindow_selfplay_design.md`, `docs/sprint_archive/§173_alpha-multiwindow-kcluster-selfplay.md`. Memory: `project_173_a3_a6_bundle_complete.md`, `feedback_registryspec_by_ref_in_hotpath.md`.

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

Forensics: `reports/s174_v6w25_investigation.md`, `reports/s174_bootstrap_investigation.md`, `reports/s174_bootstrap_fix.md`, `reports/s174_r8_falsification.md`, `docs/sprint_archive/§174_v6w25-sustained-bootstrap-investigation.md`.

---


> **Split-out:** the following closed phases moved to `docs/sprint_archive/` (see links). Their falsified-register rows are consolidated in the register section above; INV pins are inline/noted there.

- **[§S176-S177_sustained_recipe.md](sprint_archive/§S176-S177_sustained_recipe.md)** — §176 Python/PhaseA + §177 sustained recipe-attractor + Supplementary tables

## §178 — Rust engine refactor cycle 1 close (2026-05-15 → 2026-05-16)

*NOTE: This §178 = Rust engine refactor cycle 1 close. See also the §S178+ scheme (Sustained Training Sprint, post-§180) for v6 sustained run + bot-corpus mixing — the prefix `S` disambiguates the two numbering schemes that coexist permanently.*

Branch `refactor/rust-engine`. Three-wave cycle bracketed by a 90-proposal Phase 3 audit pass at `audit/rust-engine/00_aggregated_proposals.md` + per-file split addendum at `audit/rust-engine/01_file_split_addendum.md`. Wave 1 = foundation (docs + clippy floor + LazyLock migration). Wave 2 = dead-code purge (six commits, net −789 LOC, bitboard module + mcts dead setters + dead lib.rs PyO3 surfaces). Wave 3 = hot-path correctness (P1 silent v6w25 corruption, P2 v8 `has_pass_slot` dispatch, P3 cross-language `EncodingSpec` retirement) + pre-3d hygiene (Python test consumer cleanup, NN-latency triangulation). Cycle bench gate PASS; INV15+INV16+INV17 pinned; 6 settled decisions migrated below.

**Forward pointer:** archive consolidation at `docs/sprint_archive/§178_rust_engine_audit.md` (user post-cycle action — see `audit/rust-engine/wave_3/3d/archive_prep_note.md`). Source-of-truth audit tree retained at `audit/rust-engine/` for forensic reference until archive lands; SD1–SD6 are cite-from-future-prompts surface.

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

`docs/sprint_archive/§178_rust_engine_audit.md` (user post-cycle action — preserve full audit tree forensics for cross-cycle reference per `audit/rust-engine/wave_3/3d/archive_prep_note.md`). Memory: pending — recommend `project_178_rust_engine_audit.md` covering cycle outcome + SD6 entry + INV15/16/17 pin paths for cross-session lookup.

---

## §179 — Rust engine refactor cycle 2 close (2026-05-16)

*NOTE: This §179 = Rust engine refactor cycle 2 close. See also the §S179+ scheme (Sustained Training Sprint, post-§180) when those entries land — prefix `S` disambiguates.*

Branch `refactor/rust-engine-cycle-2`. Five-wave cycle bracketed by the §178 cycle 1 close at master `e0e7c47`. Wave 4 = MCTS hot-loop allocation cleanup + InferenceBatcher pool sizing (3 commits, +184 LOC, PASS-WITH-WATCH after SD6 bisection triangulation). Wave 5 pre-flight = three pre-existing test-suite flakes triaged (1 commit, test-only fixes, no production touch). Wave 5a = PyO3 boundary hardening + held fold-in (P74/P75/P76/P77) + Python `inference_pool_size` wiring (5 commits, net +784 LOC, zero-copy strengthens A+E, SD6 single WATCH). Wave 5b = `engine/src/lib.rs` structural split into `engine/src/pyo3/{board, encoding, mcts, utils}.rs` (4 commits, net +79 LOC, lib.rs 793 → 34 LOC at split, 45 LOC post-Wave-6 L26 rustdoc). Wave 6 = clippy ride-through + idiom polish + tail (P19, P42, P63, P64, P66) (5 commits, +110 LOC, clippy 186 → 42 warnings, −77.4%). Cycle bench gate PASS at Wave 6 close per `audit/rust-engine/cycle_2/wave_6/wave_close_bench_verdict.md`; all 10 `all_targets_met` checkpoints GREEN. INV15+INV16+INV17 pinned through entire cycle (Rust 9/9 + Python 2/2 at every wave close). SD1–SD6 preserved and cited per commit body across all 18 cycle-2 commits.

**Forward pointer:** archive consolidation at `docs/sprint_archive/§179_rust_engine_audit.md` (decision recorded at `audit/rust-engine/cycle_2/close/archive_prep_decision.md`). Source-of-truth audit tree retained at `audit/rust-engine/cycle_2/` for forensic reference until archive lands. Phase 5 bench audit verdict at `audit/rust-engine/cycle_2/close/03_bench_audit.md`; cycle 3 baseline anchored at `audit/rust-engine/cycle_2/close/04_baseline_next_cycle.txt`.

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

`docs/sprint_archive/§179_rust_engine_audit.md` (post-cycle action — preserves full audit tree forensics for cross-cycle reference). Source-of-truth audit tree at `audit/rust-engine/cycle_2/` retained on the operator's workstation. Memory: pending — recommend `project_179_rust_engine_audit.md` covering cycle 2 outcome + L26 promotion + SD compliance preservation + P79/P68/FF.2/FF.3/FF.10 cycle 3 anchor handoff for cross-session lookup.

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

**Forward pointer:** archive consolidation at `docs/sprint_archive/§180_rust_engine_audit.md` (post-cycle operator action — mirrors cycle 1 §178 / cycle 2 §179 archive precedent). Source-of-truth audit tree retained at `audit/rust-engine/cycle_3/` for forensic reference until archive lands. Phase 5 bench audit verdict at `audit/rust-engine/cycle_3/close/03_bench_audit.md`; cycle 4 baseline at `audit/rust-engine/cycle_3/close/04_baseline_next_cycle.txt`.

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

`docs/sprint_archive/§180_rust_engine_audit.md` (post-cycle action — preserves full audit-tree forensics for cross-cycle reference). Source-of-truth audit tree at `audit/rust-engine/cycle_3/` retained on operator workstation. Memory: pending — recommend `project_180_rust_engine_audit.md` covering cycle 3 outcome + SD7 promotion + L27 promotion + cycle 4 anchor handoff (PyO3 0.30+ surface; SD4 sub-fn extraction; §173 A5b discipline retire predicate; CRITICAL ESCALATION on cycle_settled_decisions.md tracking).

---


> **Split-out (continued):**

- **[§S178-S180_botmix_colony.md](sprint_archive/§S178-S180_botmix_colony.md)** — §S178/§S178a bot-mix launch + §S179 mechanism close + §S180a/§S180b colony-lever closes
- **[§S181_structural_audit.md](sprint_archive/§S181_structural_audit.md)** — §S181 structural diagnosis + §S181-AUDIT Waves 1-5
- **[§S182-S186_perf_arc.md](sprint_archive/§S182-S186_perf_arc.md)** — §S182-§S186 perf wave arc (legal_moves_set, MCTS micro-opt, aborts)
- **[§P5-CT_PINF_probes.md](sprint_archive/§P5-CT_PINF_probes.md)** — §P5-CT compound-turn defect probe + §P-INF inference attribution

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


> **Split-out (continued):**

- **[§D_diagnosis_eval_arc.md](sprint_archive/§D_diagnosis_eval_arc.md)** — §P6/§O1/§PRELONG + §D-WALLCAUSATION…§D-STRENGTHAXIS diagnosis & eval-foundation arc
- **[§D_multicluster.md](sprint_archive/§D_multicluster.md)** — §D-MULTICLUSTER 362-multiwindow design + §D-MULTICLUSTER-S0 ragged build
- **[§D_loopfix_tempdecay.md](sprint_archive/§D_loopfix_tempdecay.md)** — §D-VALPROBE/§D-VALCEIL/§D-PROMOGATE/§D-LOOPFIX/§D-RERUNPREP + §D-TEMPDECAY

## §D-TEMPSTRENGTH — does within-game temp decay make a STRONGER model? (head-to-head) — 2026-06-14

**Verdict: TEMP-STRENGTH-NULL.** D-TEMPDECAY judged temp decay on value CALIBRATION
(`value_accuracy_corpus`, flat → NEGATIVE); the operator's real hypothesis is *sharper late play
→ STRONGER model* — a STRENGTH question the Phase-3 smoke skipped (eval off). This dispatcher
re-ran control (flat τ=0.5) vs a20 (quarter-cosine thr=12/floor 0.20), golong@50k PEAK → 65k
(15k steps), eval-read ON, on vast 5080. Both arms generate from the SAME frozen 50k incumbent
(golong regime, no promotions → temp the only delta, red-team #4 cleared).

**Head-to-head (post-980bc4d round_robin, temp 0.0 + on-distribution opening-jitter 8 for
effective-n, n=60/pair, distinct-game bootstrap CI):** a20 vs control WR = 0.508 / 0.467 / **0.483**
at 55k/60k/65k — **every CI straddles 0.50** (65k CI [0.367, 0.608]). Effective-n HONEST
(copy_multiplier 1.002, 59–60 distinct/pair, no low-power warning) — the NULL is real, not
underpowered. BT-Elo: control marginally ahead at 65k (+130 vs +99) / 60k (+76 vs +65), tied at
55k — direction leans control, CIs overlap (not separated). Both arms improve over the 50k anchor
by 60k (only control@55k dips to 0.441 vs anchor, CI straddles 0.5 — not a real regression).

**Supporting:** a20 draw-tax plateaus ~0.106 (~1.8× control's flat 0.059), flat-not-rising by 15k,
0.30 gate never threatened (the smoke's "rising 0.096→0.106" was the asymptotic approach).
Coherence: a20 forced_win_conversion 0.182 < control 0.285 — a20 manufactures MORE forcing turns
(643 vs 438) but converts FEWER. off-window forced rate high both arms (0.62/0.71, the §D-EXTLINK
blind spot, not lever-specific).

**Pre-registered prediction (indirect signals → non-positive) CONFIRMED.** a15 escalation runs
ONLY on POSITIVE → does NOT run. Do NOT extend (O1). Falsified-register: within-game temp decay
floor 0.20 buys NO head-to-head strength + a persistent draw-tax — non-positive on BOTH strength
and calibration. The D-TEMPDECAY value-NEGATIVE verdict stands independently. Report
`reports/investigations/tempstrength_2026-06-14.md`; tooling `scripts/{run_tempstrength_rr,
analyze_tempstrength,coherence_tempstrength}.{sh,py}` + `--opening-jitter-plies` on
`eval_round_robin.py`; configs `configs/variants/tempstrength_{control,a20}.yaml`. Branch
`phase4.5/tempdecay`. Total vast run ~15.7h. Commits pending operator ask.

## §D-GUMBELSIMS — minimum-sim Gumbel operating point (Phase 0: design + method-validated) — 2026-06-14

**Status: Phase 0 CLOSED (design + harness + dev smoke); Phase 1–3 operator-run (5080/vast), GATED on
Arm-C 50k encoding verdict.** Branch `phase4.5/gumbelprep`. The prize D-GUMBELPREP buried: Gumbel's
marquee result is policy improvement at DRASTICALLY fewer sims (Danihelka ICLR'22, n≈16–50). PUCT live
≈350/move; if Gumbel matches PUCT-600 at n≈50 → ~7–12× games/GPU-hr at equal quality (attacks the GPU/
self-play bottleneck). CHARACTERIZATION→DIRECTIONAL: proxy finds a candidate BRACKET, Phase-3 strength
defines the committed n.

**No new Rust (the load-bearing finding).** Matched-position reads looked to need a Rust Gumbel-on-
position API (eval/ModelPlayer is PUCT-only; only self-play exercises SH). But PyO3 already exposes every
SH primitive (`set_forced_root_child`, `get_root_children_info`, `get_improved_policy`,
`apply_dirichlet_to_root`) and `analyze_api._run_gumbel` already drives faithful SH from Python — the
TARGET is the IDENTICAL Rust `get_improved_policy`. Harness is Python-only, no bench gate.

**6-agent fresh-review + red-team of the design (all verified vs source; ~all adopted, 2 refined):**
M1 per-seed-PAIR JSD not JSD-of-means; M4 **Dirichlet gap** (production applies in the Gumbel branch
`inner.rs:747`, driver didn't); M14 read-B improved-vs-improved (golong `completed_q_values:true` →
PUCT also records `get_improved_policy`); M8 static coherence DEMOTED to provisional witness (D-TEMPDECAY
static-vs-dynamic inversion class); M11 Phase-3 LOWER-anchor arm (the throughput-optimal n may be BELOW
the proxy knee); M12 played-move witness (trajectories steered by the SH winner, not the target); M9
refined: value target is z (`trainer.py:182`), so the read is v_mix-reliance not "value-target
degradation". Full disposition: `reports/gumbelsims/DESIGN.md` §11.

**Dev method-validation smoke (4060, golong@50k, dirichlet-OFF) — METHOD VALIDATED + metric REVISED.**
The pre-registered improved-policy JSD is UNUSABLE: at production c_visit=50/c_scale=1 the completed-Q
target is near-ONE-HOT (top≈1.0) → JSD ≈ argmax, and its n=400 SELF-floor ≈0.5–1.0 (the one-hot crown
flips between near-equal moves on value-indifferent positions — the reference disagrees with itself).
**Revised primary reads:** (A1) **visit-policy per-seed-pair JSD** vs n=400 (stable shape; self-floor
≈0.19–0.29) + (A2) **played-move value-regret** under the n=400 reference Q (decision-relevant,
indifference-robust); improved-JSD + argmax-agreement = witnesses; **dirichlet-OFF primary** (Dirichlet
inflates the floor ~2×), dirichlet-ON + component-2 production self-play = transfer check. Standalone
smoke: visit-JSD → floor by n≈50, regret → ~0.05 by n≈50 for m=16 (the Danihelka regime). Production CLI
smoke (jittered fixture) conservatively reported knee_n=400 under wide small-fixture CIs — correctly
refuses a false early knee; the knee is fixture-sensitive → full Phase-1 fixture (≥80–120 distinct games,
R=12, 300–500 positions, regime-stratified, cluster-bootstrap CI) on the 5080 resolves it.

**Artifacts (commits pending operator ask):** `hexo_rl/eval/gumbel_sims.py` (pure math: SH-budget
parity, per-seed-pair JSD, game-id cluster bootstrap; `tests/eval/test_gumbel_sims.py` 14 green) +
`hexo_rl/eval/gumbel_search_py.py` (faithful SH+PUCT driver, parity fixes) + `scripts/gumbel_sims_sweep.py`
(Phase-1 fixture+curve harness, `--smoke`) + `scripts/gumbel_sims_smoke.py` + `reports/gumbelsims/
{DESIGN,SMOKE_RESULT,PHASE1_RUNBOOK}.md`; `gumbel_ab_runbook.md` Step 0 added. Arm-C 50k untouched.

### Phase 1 RESULT (2026-06-17) — GUMBEL-SIMS-NULL on the multiplier; Phase 3 LEAN launched

**Ran Phase 1 on the WINNING encoding (v6_live2_ls) per the gate-cleared operator decision** —
generator `armc_50000_final.pt` (Arm-C 50k final, sha `1f1b3c23…`), variant `v6_live2_ls_ab`
(NOT the script default v6_live2_golong single-window — the multi-window legal-set must match the
trained checkpoint). Fixture AUDIT PASS: 480 positions, 120/120 distinct (copy_mult 1.0), legal
100→289 spanning mid+late (no opening-b≈25 regime — multi-window union is ~100 cells by ply 2, so
the runbook's generic b≈25 note doesn't apply to this encoding). Full report (gitignored):
`reports/gumbelsims/PHASE1_RESULT.md` + `curve_off_m{8,16,32}.json` + `curve_on_m16.json`.

**Result — NO CLEAN KNEE, m-robust across 8/16/32 (dirichlet OFF), confirmed dirichlet ON.**
`knee_n=None` every cell. visit-JSD is floor-limited (floor 0.30–0.52; the n=400 reference
disagrees with itself on diffuse multi-window positions) and only enters the band at n=400 — the
mid-range plateaus are FALSE (2× Δ<δ at 100→200 then Δ>δ at 200→400, curve resumes dropping).
value-regret monotonic, NO bottoming (0.11–0.15 at n=400 vs smoke's 0.05); the sharp early drop is
the mechanical SH-warmup at n≈1.5–2×m, not quality saturation. Higher m monotonically better at
fixed budget. Sensitivity gates PASS (measurement trustworthy, genuinely plateau-above-floor).

**The pre-registered candidate knee (m=16≈n=50) did NOT reproduce** — it was a low-diversity
artifact of the 5-deterministic-position standalone smoke; this matches the production-CLI smoke
(jittered fixture → knee_n=400, "correctly refused a false early knee"). The cheap 7–12× multiplier
is NOT supported by the proxy. RED-TEAM confounds: (1) flat-armc model (PARTIALLY controlled — the
strong golong also got knee_n=400 on a jittered fixture); (2) fixture diversity raises the floor
[method]; (3) multi-window large action space (100–289 legal) genuinely needs more sims [REAL
encoding property]. (2)+(3) sufficient.

**Operator routing: Phase-3 LEAN head-to-head** (the proxy compared Gumbel-n vs Gumbel-400, NOT
vs PUCT — it cannot rule out Gumbel making BETTER targets per-sim). Arms `configs/variants/
p3_armc_{gumbel,puct}.yaml`: ONE variable = search regime. Arm-Gumbel (gumbel_mcts, m=32,
n_sims_full=100, c_visit/c_scale 50/1.0) vs Arm-PUCT (standard, n_sims_full=600); fsp=0.5/quick=100/
encoding/anchor `4198d5cb…`/corpus/18w held constant. Architecture: vast TRAINS only (15k steps
each, ~16h/$11); laptop EVALS (round_robin via KClusterMCTSBot + `--encoding v6_live2_ls`,
distinct-game bootstrap CI) on pulled final checkpoints → zero vast eval time. Eval-dispatch fix
(`defender_dispatch.build_model_bot` routing legal-set models through no-drop KClusterMCTSBot)
committed this branch + synced to vast. Pre-flight GREEN (opponent imports, anchor sha, corpus).
POSITIVE = Gumbel-100 non-inferior to PUCT-600 + real pos/hr multiplier + coherence not degraded.

### Throughput optimization (2026-06-17) — host tuning is a PREREQUISITE for Gumbel affordability

During the Phase-3 A/B, the vast Gumbel arm ran with the host **95% idle** (GPU 11%, load 0.93/24,
batch fill 22%, 2310+ `waiting_for_games`) — single Python process / 43 threads (thread pool +
InferenceServer), **inference-latency-bound**, not resource-saturated. `scripts/thr_optimize.py`
(coordinate descent + interaction grid + n=5 validation via `benchmark.py` pool throughput) found
the knob directions, validated thermal-clean by an alternating A/B (laptop mobile-4060 throttles →
absolutes unreliable, ratios via interleaving). Full report (gitignored): `reports/thr_opt/REPORT.md`.

**Knob directions (consistent both Gumbel + PUCT):** `n_workers`↑ (latency overlap), **`inference_batch_size`
SMALLER (64≫128≫256)** — small batch fills fast / low dispatch latency, **`leaf_batch_size` 8→16**
(fewer round-trips; NB changes search co-batching/targets → future-only, both-arms-matched),
`inference_max_wait_ms`→2. Recommended vast Gumbel host config: **`n_workers=32, inference_batch_size=64,
leaf_batch_size=16, inference_max_wait_ms=2`**.

**Vast confirmation (5080) — the laptop MASSIVELY understated it (laptop 1.68× vs vast 15×):** the
faster GPU makes the latency-bound base WORSE (workers starve it, 22% fill). Gumbel base-w18 **9.7k →
opt-w32 150k pos/hr = ~15×**; the batch/leaf/wait knobs alone at w18 give 7.1×. PUCT (less throttled,
58% fill base; compute-bound, optimum ≥w48): base-w18 13.0k → opt-w48 32.9k = 2.5×.

**Affordability multiplier (each arm at its vast optimum): Gumbel-opt 149.6k ÷ PUCT-opt 32.9k = 4.5×**
(vs naive 600/100=6×). **CRITICAL: contingent on host tuning — at the as-run 18w-base config Gumbel
(9.7k) is 0.75× = SLOWER than PUCT (13.0k); optimization flips 0.75→4.5×.** Gumbel's affordability
advantage exists ONLY with proper host tuning. End-to-end training speedup ≤4.5× (trainer becomes the
bottleneck once `waiting_for_games` is gone — needs a real optimized run) and is only bankable if
Gumbel-100 strength is non-inferior (the Phase-3 A/B, in progress). `p3_armc_*.yaml` left at 18w-base
(strength-A/B record, lf8); optimized config is future-canonical-run-only.

### Phase 3 GAME-LEVEL DATA DIVE (2026-06-18) — why the arms play differently (mechanism + red-team)

Data-dive over the actual games (no new training): `reports/investigations/gumbel_vs_puct_games_2026-06-18.md`
(+ harness/scripts in `reports/investigations/gumbel_vs_puct_data/`). Sources: 360 round-robin eval games
(`p3_rr`, distinct-game CI valid: copy_mult 1.0, distinct/pair 120), 200 exploit-probe games/arm
(`p3_coherence`), training self-play replays pulled from vast (`/root/hexo_rl/logs/replays`, segmented to
the 3 Phase-3 arms by timestamp+character fingerprint). Findings:

- **Game-character is a SEARCH artifact, NOT a learned net trait (the key reframe).** Self-play diverges
  hard (draw 0.00 vs 0.25; plies 33 vs 95) but at **matched eval search (sims=128, both nets)** PUCT's net
  draws **0%** and plays **~38-ply** games — the 25% draws + 95-ply games are n=600-search-induced (cautious
  deep search → 150 self-play cap → scored draws). The trained nets are far more alike in tempo than the
  self-play logs imply. Eval confirmed symmetric (60 games per color ordering).
- **Off-window "inversion" is EXPOSURE, not defense (red-team-reversed, verified from seeds).** Naive read
  P(loss | reached) Gumbel 0/17 vs PUCT 17/34 LOOKS like defense — but `adv_side` is set by seed parity
  (even→+1 axis (1,-1); odd→−1 axis (1,0)). Gumbel reaches forcing positions ONLY on even seeds (17/50),
  NEVER on odd (0/50); **all 17 PUCT losses are odd-seed positions Gumbel structurally never enters.** On the
  17 shared even-seed positions BOTH arms hold **17/17 — identical defense.** So 0/17-vs-17/34 is a
  population confound (classic §D-COHERENCE re-validation trap); the gap is which geometry self-play enters,
  NOT defense skill. "Gumbel off-window-robust" WITHDRAWN. All 17 PUCT losses = 1 tactic (-4,3), non-strict
  (`strict_off_window_forced=0` both arms; model had in-window block). May be a 15k transient (armc-50k 0/200).
- **Strength edge concentrated in SHORT games (refutes attrition story).** PUCT H2H 60% (72-48) is strongest
  in the 0–30-ply bucket (**68%**); 60–90 bucket is **50/50**. Traces to PUCT's deeper self-play search →
  tactically sharper net. Significance MARGINAL even at 1 seed: H2H Wilson [0.511,0.683]; Gumbel/PUCT **Elo
  CIs overlap 16 Elo**; Gumbel-beats-anchor 57% is NOT significant (Wilson [0.477,0.652]).
- **Short Gumbel wins are largely UNCONTESTED 6-lines** (loser 0–2 stones adjacent to winning line) = immature
  defense at 15k steps — but PUCT wins are *also* uncontested → shared 15k weakness, only length differs.
- **No colony-vs-coherent divergence per position** (Q4): legal-set size + connected-component colonies are
  near-equal at matched ply; big terminal gaps are pure game-length artifacts.
- **Affordability:** 4.5× (Gumbel-opt w32 149.6k ÷ PUCT-opt w48 32.9k) — contingent on the **worker optimum**
  (free byte-identical lever). Gumbel-opt (tspg=1.0) arm STILL RUNNING — preserves drawless/short character;
  final strength / matched-wall-clock gap-closure = OPEN.
- **Depth lever (route to GUMBEL-SIMS-POSITIVE):** depth=f(n,m)≈n/m (Gumbel 2.83 < PUCT 3.52). Map the **2D
  (m,n) Pareto front** — lower m is FREE (n fixed → 4.5× holds, breadth↓), raise n is COSTLY (throughput↓).
  Proposed: proxy sweep m∈{4,8,16,32}×n∈{100,200,400} measuring `mcts_mean_depth` (`MCTSTree.last_search_stats()[0]`)
  + value-regret vs n=800, then a **multi-seed** Gumbel-(low-m) vs PUCT-600 A/B.

- **Strength sign solid, magnitude/CI oversold (red-team).** PUCT>Gumbel sign robust (paired bootstrap of
  Gumbel−PUCT Elo = [−143,−38.5], 99.98%) but marginal CIs OVERLAP (the "separated CIs" read is wrong), H2H
  p≈0.029, +89 Elo is n=1. Per-phase "concentrated in short games" is NOT significant (Fisher p=0.23; 90+
  bucket 66.7% PUCT contradicts it). Short Gumbel wins partly low-resistance (30% of ≤20-ply wins uncontested
  vs PUCT 1%). The "one variable" is an algorithm+sim-budget BUNDLE (Gumbel∧n=100 vs PUCT∧n=600).

**OVERARCHING LIMIT: n=1 training run per arm — and operator has elected to STAY at n=1 (no multi-seed
training).** Eval CI captures eval noise only, zero training-seed variance. So the bankable claims are the
mechanically-robust ones (game-character=search artifact; depth=f(n,m); affordability throughput 4.5×); the
strength *magnitude* and any off-window inversion remain single-seed and are reported as such, not banked.
Two NEAR-FREE eval-only de-risks remain available without any new training: (1) §2 matched-position replay —
force the Gumbel ckpt into the 17 odd-seed (-4,3) lines to confirm exposure-not-defense; (2) §7 (m,n) proxy
sweep — locate lowest-m hitting PUCT-600 depth at n=100 + decouple the algo/sim bundle. Findings
adversarially cross-checked by an independent red-team subagent (report §12; the off-window reversal was
re-verified from raw seeds). **Verdict: GUMBEL-SIMS-NEGATIVE at matched step (sign solid, +89 Elo n=1); the
"coherence inversion" is DOWNGRADED to an exposure artifact, not a defense win; affordability/depth route
untested.** Report: `reports/investigations/gumbel_vs_puct_games_2026-06-18.md`.

### §D-GUMBELSIMS session verdict (2026-06-18) — Phase-1+3 + throughput + game dive + red-team

Full verdict: `reports/gumbelsims/SESSION_VERDICT.md` (gitignored). n=1/arm, 15k steps — directional.

**Bankable (mechanical, training-level-independent):** (1) Phase-1 NULL — no cheap Gumbel knee
(smoke n≈50 was a low-diversity-fixture artifact; m-robust OFF+ON). (2) Depth=f(n,m): Gumbel-100/m32
**2.83** < PUCT-600 **3.52** — Gumbel buys breadth not depth; `m` is a free depth↔breadth lever
(m=8/n=100 cuts value-regret 0.165→0.109 ~34% at unchanged throughput; PUCT-depth needs n≥400 = costs
throughput). (3) Game-character (drawless/short Gumbel vs drawy/long PUCT) is a SEARCH artifact — both
nets draw 0%/~38ply at matched eval. (4) Throughput **4.5×** (Gumbel-opt÷PUCT-opt at vast worker
optima), CONTINGENT on host tuning — at as-run 18w-base Gumbel is 0.75× SLOWER; tuning flips it
(host was 95%-idle, inference-latency-bound). Recommended Gumbel host: n_workers=32, ib=64, leaf=16,
wait=2.

**Strength (single-seed, qualified):** PUCT stronger at 15k (Elo +154 vs +65, H2H 60%, sign 99.98%)
**BUT the gap NARROWS with training — PUCT-wr 0.75(5k)→0.71(10k)→0.60(15k)** → very likely an
undertraining transient (15k ≪ 500k). Pre-registered POSITIVE (Gumbel non-inferior strength) FAILS
at 15k, but the affordability case is alive (findings 4+6).

**Red-team REVERSALS (both confirmed by direct test this session):** the off-window "Gumbel defends
0/17, PUCT forceable 17/17" was a **seed-parity EXPOSURE confound** — the forced-position test
(`scripts/forced_offwindow_test.py`) replayed all 17 PUCT losses, swapped the defender to Gumbel at
the forcing onset → **Gumbel lost 17/17, held 0/17** (PUCT-control 17/17 validates). Both nets share
the identical off-window blind spot; zero defense difference. My original read fell into the
§D-COHERENCE founding-measurement trap; the red-team + this direct test corrected it. Also: "loss
concentrated in short games" NOT significant (Fisher p=0.23).

**Running:** Gumbel-opt affordability arm (vast) — w32/ib64/wait2/lf8 + tspg=1.0, ~7.2k steps/hr
(2.8× the throttled 18w), kill at ~9.6h (≈ PUCT-15k GPU-hours, ~69k steps on fresh data) → eval vs
PUCT-15k. Tests findings 4+6 convergence: does cheap-Gumbel-run-longer match PUCT at equal cost.
Commits: `63060eb` (eval-dispatch+arms), `2930a68` (exploit_probe dispatch), `e614327` (throughput opt).

### §D-GUMBELSIMS 50k affordability test (2026-06-18) — undertraining dominated the 15k read

Optimized Gumbel arm (p3_armc_gumbel_opt: w32/ib64/wait2/lf8 + tspg=1.0) run to **50k steps**
(~6.9h GPU-hrs, *cheaper* than PUCT-15k's 9.6h; ~7.2k steps/hr = 2.8× the throttled 18w). 4-model
round_robin (anchor / PUCT-15k / Gumbel-opt-50k / PUCT-50k=armc), n=100, on vast.

**Elo (anchor=0):** PUCT-15k **135**, Gumbel-opt-50k **136**, PUCT-50k **208**. Head-to-head:
Gopt-50k vs PUCT-15k **51-49 (dead even)**; Gopt-50k vs PUCT-50k **46-54**; PUCT-50k vs PUCT-15k 63.5-36.5.

- **Affordability ≈ parity-to-edge:** Gumbel-opt-50k TIES PUCT-15k (Elo 136≈135) at **~72% the cost**.
  Not a blowout — Gumbel needed 3.3× the steps to offset weaker-per-step targets — but a real slight
  cost edge at a fixed GPU budget.
- **Matched-step: PUCT-600 still wins, gap NARROWING:** Gopt-50k vs PUCT-50k 46% (Elo +72) vs the 15k
  matched gap 40% (+89). Catch-up real but incomplete. PUCT improves with training too (PUCT-50k ≫ 15k).
- **Off-window blind spot = UNDERTRAINING, resolved:** PUCT FORCEABLE(15k,17/100) → DEFENDED(50k,0/200);
  Gumbel-opt-50k DEFENDED (0/100). The 15k "Gumbel defends / PUCT forceable" was exposure (forced test:
  Gumbel loses 17/17 when forced) + undertraining (both defend by 50k). Fully a non-issue at 50k.
- **Trajectory (Gopt vs PUCT-15k):** 0.31(15k)→0.59(25k)→0.42(35k)→0.49(50k) — noisy, reaches parity by 50k.
- **SealBot:** Gopt-50k 44% [31,58], up from 15k arms' 24-32%; not beating SealBot (50k ceiling holds).

**Bottom line:** the 15k Phase-3 verdict ("PUCT wins + Gumbel defends off-window") was dominated by
undertraining artifacts. At 50k: off-window is a wash (both defend), the per-step strength gap is
shrinking (PUCT-600 still ahead), and Gumbel-100 reaches PUCT-15k strength at lower cost. GUMBEL-SIMS
= affordability-parity (modest edge), NOT the clean 4.5× win — the throughput multiplier is real but
mostly consumed offsetting Gumbel's weaker-per-step (shallower 2.83-depth) search. Files:
`reports/p3_rr50_agg/`, `reports/p3_coherence/gopt50.*`, `reports/p3_sealbot/gopt50.log`,
`reports/p3_trajopt_*/`. Verdict mirrored to `reports/gumbelsims/SESSION_VERDICT.md`.

---

## §D-LONGRUN-READY — Long-run pre-flight: consolidate → re-pretrain → m-gate → launch spec — 2026-06-18

**Operator intent:** stay with Gumbel, commit a good long run on v6_live2_ls. Stop PUCT churn
(reuse the banked PUCT arm). New 8300-game corpus available.

**Discipline the verdict forces:** Gumbel = NULL/affordability-PARITY — weaker per step
(depth 2.83 < PUCT 3.52), ~28% cheaper (host-tuning + m + n=1 contingent). Phase-1 sim-efficiency
thesis dead. The m-lever (untested in training) is the one thread that flips parity → edge.
Long run GATED on m-sweep: picks Gumbel config AND decides whether Gumbel earns the run over
simpler PUCT-600. Do NOT launch at provably-shallow m=32.

**Deliverables (produced this session):**

| Artifact | Path |
|---|---|
| Phase B m=8 config | `configs/variants/phase_b_mgate_m8.yaml` |
| Phase B m=16 config | `configs/variants/phase_b_mgate_m16.yaml` |
| Phase B m=32 config | `configs/variants/phase_b_mgate_m32.yaml` |
| A1 vast consolidation runbook | `docs/handoffs/longrun_a1_vast_consolidation_runbook.md` |
| A2 bootstrap re-pretrain spec | `docs/handoffs/longrun_a2_bootstrap_pretrain_spec.md` |
| Phase C long-run launch spec | `docs/handoffs/longrun_phase_c_launch_spec.md` |

**Phase A (parallel, operator-run):**
- A1: /root → /workspace consolidation. Reports/ preserved first (data-loss-risk step).
  git bundle transport (vast fetch dead). Engine rebuild + smoke. /root removed after gate.
- A2: fresh bootstrap pretrain on 8300-game corpus (473k pos, sha 8f7115ab), 30 epochs,
  v6_live2 encoding. Produces `checkpoints/bootstrap_model_v6_live2_8300.pt`. SHA256 pinned
  into phase_b configs after production.

**Phase B — m-gate (pre-registered discriminator):**
Arms Gumbel m=8/m=16/m=32, all n=100 (throughput fixed, m = pure depth↔breadth lever),
optimised throughput w32/ib64/wait2/lf8, tspg=1.0. ~12k steps each. From new A2 bootstrap.
Reads: (1) strength head-to-head among m-arms (distinct-game CI, round_robin); (2) each
arm vs BANKED PUCT-15k (reuse `reports/p3_rr_agg/` — NO PUCT re-run); (3) depth +
value-regret; (4) pos/hr (should be ~flat). Eval-opponent pre-flight before any launch.

Pre-registered routing: GUMBEL-EARNS-IT (best m-arm non-inferior to PUCT, cheaper) →
Gumbel@best-m long run. GUMBEL-PARITY-ONLY or GUMBEL-WORSE → PUCT-600 long run (stronger
per-step, simpler, no host-tuning dependency).

**Phase C (design-only until B resolves + operator GO):**
Resolved config (Gumbel@best-m OR PUCT-600), v6_live2_ls, A2 bootstrap.
Corpus schedule: canonical 0.8→0.1 floor, ONE variable (search config).
Loop-turning gate: promotion MUST fire by step 15k or ABORT. Hard-abort monitors live.
Banking: every promotion + 25k/50k/... schedule; rsync transport; bank from step 1.
SealBot: periodic, trajectory read, not a point-in-time bar.

---

## §D-LONGRUN-C — Phase C Gumbel long-run: 200k verdict — 2026-06-21

**Run:** `longrun_v6_live2_ls_gumbel_m16`, vast RTX 5080, tmux `longrun_c3`. Killed at step **234,067** (33k past 200k target; no configured max_steps). Config: `configs/variants/longrun_v6_live2_ls_gumbel_m16.yaml`. Encoding: v6_live2_ls (4 planes). Gumbel m=16, n_sims=100, w32/ib64/wait2.

### Two bugs contaminated the run

**Bug 1 — bootstrap_floor gate (steps 0–120k):** Base `eval.yaml` has `bootstrap_floor.enabled: true` by default. `bootstrap_anchor` opponent is disabled in this variant → `wr_bootstrap_anchor=null` → floor always fails → every promotion blocked. Fixed mid-run by adding `bootstrap_floor.enabled: false` to variant gating config. Effect: two qualifying evals at 60k (WR=0.64, CI_lo=0.501) and 90k (WR=0.66, CI_lo=0.522) were silently blocked. Self-play ran on a stale `best_model.pt` for the entire first 120k steps, degrading sample quality.

**Bug 2 — playout_cap policy waste (entire run):** `full_search_prob: 0.5` with `n_sims_quick=n_sims_full=100`. In Gumbel, the improvement operator produces valid policy targets at ALL positions regardless of sim count — the `full_search_mask` filter is a PUCT design. Effect: ~50% of policy gradient signal discarded every game at zero search-quality cost. Fixed in config: `full_search_prob: 1.0` (2026-06-21). Does not affect correctness of the targets kept, just halves training efficiency throughout.

### Eval trajectory (14 rounds)

| Step | Promoted | WR-best | CI | WR-random | WR-sealbot | Elo-est |
|------|----------|---------|-----|-----------|------------|---------|
| 15k  | no  | 0.48 | [0.348, 0.615] | 1.0 | — | 789 |
| 30k  | no  | 0.48 | [0.348, 0.615] | 1.0 | — | 621 |
| 45k  | no  | 0.54 | [0.404, 0.670] | 1.0 | — | 540 |
| 60k  | **blocked** | 0.64 | [0.501, 0.759] | 1.0 | 0.32† | 421 |
| 75k  | no  | 0.40 | [0.276, 0.538] | 1.0 | — | 226 |
| 90k  | **blocked** | 0.66 | [0.522, 0.776] | 1.0 | — | 367 |
| 105k | no  | 0.44 | [0.312, 0.577] | 1.0 | — | 192 |
| 135k | **YES** | 0.68 | [0.542, 0.792] | 1.0 | — | 834 |
| 150k | no  | 0.46 | [0.330, 0.596] | 1.0 | — | 549 |
| 165k | no  | 0.58 | [0.442, 0.706] | 1.0 | — | 543 |
| 180k | no  | 0.54 | [0.404, 0.670] | 1.0 | 0.40† | 359 |
| 195k | **YES** | 0.66 | [0.522, 0.776] | 1.0 | — | 395 |
| 210k | no  | 0.48 | [0.348, 0.615] | 1.0 | — | 216 |
| 225k | no  | 0.46 | [0.330, 0.596] | 1.0 | — | 234 |

Notes: "blocked" = gate would have passed (CI_lo > 0.5, WR ≥ 0.55) but bootstrap_floor bug prevented it. Two legitimate promotions: 135k and 195k (both post-fix). WR-random=1.0 throughout (fully beats random). Elo-est is relative to current best_model, not absolute — not comparable across promotion boundaries. †WR-sealbot values marked † are §D-ARGMAX-collapsed (argmax + no opening plies → n_eff ≈ 2); see honest post-hoc trajectory below.

### SealBot trajectory (stride=4 production evals — COLLAPSED; post-hoc honest re-run)

**Production eval numbers (60k=32%, 180k=40%) are §D-ARGMAX-collapsed and not load-bearing.** The production eval uses `temperature=0` (argmax) + no random opening plies → both sides deterministic from a fixed start → n_eff ≈ 2 distinct games inflated to n=50. The Wilson CI is overconfident by √25 = 5×.

**Honest post-hoc trajectory** (`scripts/generate_trajectory_games.py`, 50 games/ckpt, n_sims=100, 4 random opening plies, seeded per-game): 400 games total, game files written to `runs/step_*/games/` for viewer.

| Step | WR vs SealBot | CI (Wilson) | Regime |
|------|--------------|-------------|--------|
| 30k  | 42% | [0.294, 0.558] | bootstrap_floor bug active |
| 60k  | 46% | [0.330, 0.596] | bootstrap_floor bug active |
| 90k  | 44% | [0.312, 0.577] | bootstrap_floor bug active |
| 120k | 42% | [0.294, 0.558] | bootstrap_floor bug active |
| 150k | 54% | [0.404, 0.670] | **post-fix** (first promo 135k) |
| 180k | 50% | [0.366, 0.634] | post-fix |
| 210k | 56% | [0.423, 0.688] | post-fix (second promo 195k) |
| 234k | 54% | [0.404, 0.670] | post-fix |

**Reading:** CIs all overlap at n=50 (±13pp) — no individual checkpoint is significantly better than another. The *regime shift* is the load-bearing signal: pre-fix plateau at 42–46% (below 50%, stale best_model), post-fix step to 50–56% (model beating SealBot). The step coincides exactly with the first promotion at 135k. Playout_cap bug suppressed the ceiling throughout — 50% policy signal wasted per game.

Cross-check vs §D-GUMBELSIMS 50k arm (Gopt-50k, honest eval): 44% SealBot. This run reaches 50–56% by 150k+, consistent with continued improvement beyond the 50k regime despite two bugs.

### WR pattern: alternating high/low

Post-bootstrap-floor-fix, WR alternates: 0.68 (promote) → 0.46 → 0.58 → 0.54 → 0.66 (promote) → 0.48 → 0.46. Consistent with best_model hard-reset on each promotion: evaluation target immediately gets harder → WR drops → model climbs back over ~30–45k steps. This is healthy self-play loop-turning behavior, not instability.

Pre-fix pattern (60k→75k dip from 0.64→0.40 with NO promotion/no opponent change) is real oscillation — attributable to playout_cap waste + stale best_model degrading sample quality.

### Session verdict

**CONTAMINATED-BUT-POSITIVE.** Two bugs throughout, but directional signal is clear.

1. **Loop turns.** With gate fixed, model promotes and climbs. 2 promotions in 100k post-fix steps. Self-play loop is healthy.
2. **SealBot: positive regime shift post-fix.** Honest trajectory shows 42–46% (pre-fix) → 50–56% (post-fix). Model crosses 50% SealBot WR by step 150k and stays there. Production eval numbers (32%, 40%) were §D-ARGMAX-collapsed — not a reliable read.
3. **No failure modes.** Zero colony collapse, draw_rate 0.001 throughout, grad_norm healthy (1.8–3.9), zero hard-abort triggers.
4. **Playout_cap fix is load-bearing for next run.** ~50% policy signal discarded every game. `full_search_prob: 1.0` doubles effective gradient per game at zero cost. Most impactful single fix available.
5. **No PUCT comparison.** No matched PUCT 200k run. Affordability case (4.5× throughput) requires both bugs fixed to assess cleanly.

**Configs fixed** (2026-06-21):
- `full_search_prob: 1.0` in `configs/variants/longrun_v6_live2_ls_gumbel_m16.yaml`
- `bootstrap_floor.enabled: false` (already applied mid-run, now permanent in variant)

**Open question:** does a clean Gumbel long-run (both bugs fixed, full_search_prob=1.0) reach 60%+ SealBot by 200k? The post-fix trajectory trend (50→56%) is encouraging but the playout_cap waste caps the ceiling. A clean re-run is the test.

---

## §D-MONITORFIX → PARTIAL + §D-MONITORFIX-CONFIRM (2026-06-23)

Red-team + rework of `scripts/d1m_monitor.py` (D-1M Gumbel-1M run monitor, read-only ssh consumer; PID 1512427 on vast never touched). No-restart, consumer-script-only.

**Fix ledger (D-MONITORFIX):**
- **F1, F4 — NO-OP.** Two handoff "confirmed errors" were already-correct on inspection. Investigation caught stale handoff claims. *Bank: handoff verdicts are claims under red-team, not given.*
- **F2 — wr_sealbot green-gate.** Dropped the absolute `≥0.55` bar (a soft §101 anchor, never a gate). Green now requires a robust SLOPE read: Theil-Sen slope>0 ∧ measurement-error CI lower-bound>0 (small-sample t df=n−2, propagating each point's logged `ci_sealbot` half-width as σ; bootstrap + analytic, conservative bound) ∧ effect-size rise (slope×span) ≥ floor. Reworked + reverified vs the original false-green Monte-Carlo (the old z=1.96 normal-CI false-greened noisy plateaus at n=5–8).
- **F3 — entropy LOW-FLOOR only.** Gated series = `policy_entropy_selfplay` (net selfplay policy-head entropy, `trainer.py:1088` `entr(p_fp32[n_pretrain:])`). Retained the same-regime selfplay-collapse floors (collapse 1.0 / warn 1.5, the §71.2 calibration on the selfplay stream); dropped the cross-regime 2.0 UPPER bar — high entropy is healthy, bounded by the DERIVED `ln(policy_logit_count)` ceiling (registry lookup, never a literal).
- **F5 — depth descriptive vs run self-baseline.** No absolute depth floor (shallow-by-design under Gumbel-SH). `depth_health()` judges depth vs the run's own rolling median (±5% stable, >12% below = regression); root-concentration co-move is DESCRIPTIVE under Gumbel-SH.
- **F6 — Tier-1 signals surfaced.** Gumbel-target entropy + KL-to-uniform (full-search head), opening diversity (early-game entropy + top1 mass = §D-ARGMAX effective-n), fp16 AMP-scale sharp-drop canary — all self-baseline reads.
- **F7 — eval overhead** from timestamp gaps (approx); owed-notes for distinct-game / value-calibration.
- **F8 — interim replay-analyzer** (`scripts/d1m_replay_analyzer.py`, new file): per-game `longest_line` + `n_components` reconstructed from self-play replay JSONL for the golong kill-gate. All outputs labelled INTERIM; definitions pinned to engine line/adjacency conventions to avoid an S7 measurement discontinuity. Hand-validated.

**PHASE-A confirm outcomes (D-MONITORFIX-CONFIRM, read-only, pre-registered):**
Live reads (ssh grep of `logs/d1m/d1m_gumbel_m16_n150.jsonl` @ step 149370): 3 sealbot rounds — 30k=0.23, 60k=0.24, 120k=0.27 (Theil-Sen +0.044/100k); `policy_entropy_selfplay`=2.668 (trailing 2.42–2.90); Gumbel-target fullsearch entropy=0.208.

- **F2-C2 → PASS.** 3/3 wr_sealbot records carry `ci_sealbot` (100%) — the measurement-error CI has its per-point σ input on all points.
- **F3-C1 → PASS.** Band gates `policy_entropy_selfplay` (selfplay net entropy, live 2.668); floor 1.5 is the §71.2 selfplay-stream collapse calibration — SAME regime. Gumbel-target entropy (0.208) is surfaced in the separate F6 descriptive panel, NOT run through the 1.0/1.5 floor (which would falsely RED it). No cross-regime borrow.
- **F3-C2 → PASS.** Floor = §71.2 selfplay-collapse calibration + retained 1.0 low-anchor; sits ~1.2 nats BELOW the live 2.668 reading (and ~0.9 below the trailing min 2.42) — independent of the current value, not circular. Upper ref is the derived `ln(362)=5.89`, distinct from the §105 high-entropy 5.x settled band (different regime).
- **F2-C1 → FAIL (literal "reads GREEN" bar) → HOLD-PARTIAL.** Synthetic at the live slope + the run's real per-point noise (logged `ci_sealbot`→Wilson n=100 σ≈0.042, run through the SHIPPED gauge) greens only at **n≥9**; reads honest "inconclusive (+slope/100k, CI straddles 0)" at n=5–8; flat plateaus never false-green. **Phase B proved the FAIL untightenable no-restart:** (1) dropping the effect-size floor to 0 leaves n=5–8 inconclusive → the binding constraint is the CI-lower>0, NOT the effect-size floor — the pre-registered FAIL cause is **REFUTED**; (2) greening the live climb at n=6 needs ci_level≤0.70, at which flat-plateau false-green = **4.10% ≫ 0.5%** bound. The non-green is correct statistics (a +4.4pp/100k slope is sub-noise vs n=100 Wilson σ≈0.042 until n≥9), not a false-negative. Gauge left as-is (correct). *Bank: a confirm can falsify its own pre-registered remedy.*

**Verdict bin: HOLD-PARTIAL.** F2-C1 fails + untightenable no-restart → reverts to descriptive/owed (the F2 slope-gate fix is correct and ships unchanged; only the green-on-live-climb-at-small-n is owed). F2-C2/F3-C1/F3-C2 PASS. `make test` = baseline (2022 pass, 1 pre-existing unrelated fail in `scripts/forced_offwindow_test.py`, separate hygiene) — no new regression.

**Owed (engine-add / S7, not buildable no-restart):**
- **F2-C1** — green on a live-rate sealbot climb is unattainable at the run's realistic n (3 now; ~5–8 through the 200k gate at current ~30–60k cadence; n≥9 ≈ step 500k+). Only resolver = more sealbot eval games/point (tighter per-point CI) = restart-gated. Not a gauge defect.
- **I7a** — `game_complete` has no game-id/hash → monitor distinct-game fraction owed (F8 interim-salvages via move-seq hashing: 1.000, no argmax-collapse).
- **I7b** — `train_step` has no per-sample (predicted_value, outcome) → value-calibration curve owed.
- **S7** — Rust `game_complete` emit of `longest_line`/`n_components`; F8 is the interim Python stand-in. One documented discontinuity to reconcile: per-player split vs color-blind `get_clusters` (analyzer dual-emits both).

---

## §D-CONSOLIDATE — land scattered Wave-1 work to master (2026-06-23)

Brought four parallel-session outputs to `master` via the `phase4.5/gumbelprep` integration branch. **Local merge only** — the live D-1M run (PID 1512427 on vast, ~step 149k) keeps its own `.so` and was never touched. Outcome bin: **CLEAN-MERGE** (reached after a threading fix the dispatcher had pre-scoped to PARTIAL).

**What merged** — master `81cb468` = origin/master `a71ae28` ⊕ gumbelprep `4b54c58`:
- **docs (S4):** sprint-log split/archive + §D-MONITORFIX verdicts (already on the branch base).
- **scripts (S1):** `d1m_monitor.py` + `d1m_replay_analyzer.py` (read-only consumers, no hot-path).
- **mcts Rust (D-QFIX-LAND A2a+A1):** `completed_q.rs` dedup + explicit `InteriorSelector` planner split.
- **audit:** authoritative 148-stat D-STATAUDIT re-run → `docs/audits/stat_audit_2026-06-23/` (supersedes the ephemeral /tmp 97-stat first audit; promotion-gate finding INVERTED — CI half-draw bug is conservative/false-negative, promotions trustworthy). `_audit/` mirror de-duped.

**Rust red-team (deferred pass, completed): BYTE-PURE.** Two independent builds — new code (worktree) AND old `e132e67` inline math (oracle regen in a clean tree) — both pass the same golden `f32::to_bits()` fixtures (18/18 byte-identical), so committed goldens are genuine HEAD provenance, not captured-from-refactored. 3 adversarial static lenses (arith / default-path / config-INV19) high-confidence; S3 `gumbel_search.rs` empty diff (A2b correctly cancelled — no math divergence to unify); INV19 38-arg positional ctor surface intact. Default `interior_selector=puct` ⇒ leaf selection identical to HEAD.

**Finding — A1 hard-read incompletely threaded (regression found + fixed).** `pool.py` hard-reads `mcts_cfg["interior_selector"]` (KeyError on missing, by design — no silent default). The A1 landing fixed `benchmark.py` but missed 9 test-fixture files → 23 `make test` regressions; the handback's "all WorkerPool call sites threaded" claim is FALSIFIED. Fix (`4b54c58`): added `"interior_selector": "puct"` to each pool-config fixture (matches `configs/selfplay.yaml`; preserves the hard-read). Dev scripts + eval already safe (they load `selfplay.yaml`). `make test` green afterward — only the untracked-WIP `forced_offwindow_test.py` probe fails locally (absent on master). *Bank: a "verified all call sites" handback claim is itself a claim under red-team.*

**Topology note:** local `master` was stale (behind `origin/master` by 2 — `a71ae28` exploit_probe legal-set fix + `3e878fb` hammerhead install). Caught at merge time; reset to the fetched `origin/master` and re-merged. gumbelprep's gumbelsims exploit_probe work is byte-identical to `a71ae28` → clean auto-merge, no conflict.

**Stayed WIP (excluded from master):** untracked `configs/variants/*`, `docs/handoffs/*`, `scripts/*` (incl. `forced_offwindow_test.py`), `exports/`. Cleanup: removed `worktree-d-qfix-land` + `statAudit_wt` worktrees; deleted `worktree-d-qfix-land` + `audit/stataudit-rerun` branches (content merged).

**Owed (operator / next-run, out of scope here):** push `master` + rebuild the vast `.so` from merged master for the **NEXT** run only (rsync + git bundle, never `git pull`; never under the live PID). Next-run queue unchanged: Dirichlet-off A/B, CI half-draw fix, S7 bundle. Hygiene: `forced_offwindow_test.py` off-window-leak (allowlist/move). `/workspace`-vs-`/root` vast-repo migration still flagged, not actioned.

---

## §D-LADDER — deploy-matched BT-MLE Elo ladder: TRUE-STALL + falsified colony mechanism — 2026-06-24

**Question:** is the d1m 150k+ "parity stall" (live promotion eval 150/180/210k @ WR 0.47/0.53/0.495) a real strength plateau, a measurement-ceiling artifact of the down-powered eval, or a regression? Decided with a deploy-matched self-anchored BT-MLE Elo ladder, NOT the live promotion gate. Full record: `reports/d_ladder_2026-06-24/{PREREG,FINDINGS}.md`; instrument `scripts/eval/gumbel_ladder.py` + `tests/test_gumbel_ladder_aggregate.py` (untracked WIP).

**HARD INVARIANT honored:** the live run (PID 1512430, `/workspace/hexo_rl` on vast) was NEVER touched — only read-only rsync of banked checkpoints + read-only `/proc`/`ls`. All GPU eval on the laptop 4060.

**Stage 0.5 (read-only greps):** S0.5-A = **SILENT** — bot corpus never active this run (`bot_corpus_path:null`, `bot_batch_share:0.0`, n_bot=0 every step; the 173 "sealbot" log events are the eval opponent, not training rows). S0.5-B = **COLONY_ONLY** — the register's only bot-mix entry was anti-colony (Track 4A, removal → faster decline); no isolated 0.30-vs-0.0 strength A/B exists → bot-floor lever ranks below clean-strength levers.

**Verdict = TRUE-STALL** (combined n=40/pair, 1120 games; copy_mult 1.0, distinct/pair 40, head_fired 100%, 0 draws). D_self (Elo(t)−Elo(s120k), distinct-game bootstrap): s150k −11 [−63,+44] · s175k +35 [−17,+88] · s200k +54 [≈0,+111] · **s226k +3 [−51,+58]**. No D_self robustly excludes 0 (the s200k "climb" sits at lo≈0, flips sign across bootstrap seeds; 226k≈120k). Not CEILING, not REGRESSION. n=20→n=40 moved D_self(226k) −23→+3 (toward flat). The live stall is REAL in the deployed-strength dimension — not a pure measurement artifact, but it also isn't "fix eval and let run finish."

**Robust structural finding (de-risked):** the model relates NON-SCALAR-ly to a minimax opponent — single-axis intransitivity, 8-9/9 three-cycles SealBot-routed, transitive-null P≈0.003. Reproduces with BOTH a time-limited AND a fixed-`max_depth=5` SealBot (P≈0.004) → bar-independent. ⇒ self-play Elo is an INCOMPLETE strength order.

**FALSIFIED — colony / off-window mechanism** (register row above): the SPECIFIC "SealBot exploits the 150k mid cluster" trajectory FLIPPED between bars (s150k WR 0.35 time-limited → 0.55 fixed-depth-5). A real off-window defect would be exploited by both bars; the inversion ⇒ a wall-clock-SealBot-instance artifact. DROP the colony/off-window narrative + the planned off-window GPU probe.

**Instrument findings (verified):** (1) the live in-loop eval is NOT deploy-matched — deploy = Gumbel SH @ n_sims_full=150, no temp; but `evaluator.py` ModelPlayer/KClusterMCTSBot = PUCT visit-policy + `eval_temperature=0.5` (`defaults.py:32`/`eval.yaml:120`; 0.0 vs SealBot) + model_sims=64. So the stall reads ran a PUCT+temp+64-sim head the model never deploys. (2) 150 is a non-canonical Gumbel SH budget for m=16 (allocator `inner.rs:781` profile 2/4/10/23 vs 128's clean 2/4/8/16). (3) wall-clock time-limited bots are machine-speed-dependent (SealBot@0.5s reached median depth 4 / mean 4.6 here) — use fixed `max_depth` for a reproducible bar. Literature (AGZ evaluator τ→0; mctx `gumbel_scale=0`; LightZero `deterministic=True`) confirms eval = greedy deploy head, no temperature; opening BOOK > random plies.

**Stage-2 routing (TRUE-STALL + bot SILENT):** #1 = **pretrain-floor 0.10→lower** training-fork; #2 opening-diversity. Relaunch eval recipe: Gumbel@150 (or swept power-of-2) no temp, opening book, fixed-depth SealBot + a 2nd external bar, distinct-game bootstrap BT-Elo, restored power, adaptive screen→confirm. Runbook: `docs/handoffs/d_ladder_relaunch_runbook.md`.

**Process note (banked):** an early read had a doubled-denominator WR bug (WR(A,B)+WR(B,A)=0.5) that inverted the apparent self-ladder; caught by the independent re-derivation. And the n=20 screen's "colony-attractor" point-estimate story was over-claimed → corrected to TRUE-STALL by the pre-registered CI gate + adversarial red-team, then the fixed-depth de-risk falsified the mechanism. Pre-registration before the run + fresh re-derivation + adversarial bar-swap each caught a distinct error.

**CORRECTION — 2026-07-02 (§D-FORENSIC F1):** the entire d1m lineage self-played **single-window
`v6_live2`** (declared `v6_live2_ls` silently overridden at load; see §D-FORENSIC). TRUE-STALL
stands as measured, but it characterizes the **single-window self-play regime only** — 120k→226k
flat is a statement about a net that never generated an off-window move in training. It does NOT
bound what multi-window self-play (first ever run in the §D-WS3 v3 arms) can reach.

---

## §D-LOCALIZE — gap-localization (VALUE-TARGET) + deploy-matched in-loop eval + search-scaling launch — 2026-06-25

**Question (follow-on to §D-LADDER TRUE-STALL + model⊥minimax):** WHERE does the model⊥SealBot gap live — LINES (→ imitation/bot-mix + corpus regen), VALUE (→ value-target lever), or TACTICS (→ search lever)? Decided by read-only re-eval of the 68 banked lost mid-cluster games (no fresh games). Full record: `docs/handoffs/d_localize_findings.md`; data `reports/d_localize_2026-06-25/` (gitignored).

**Verdict = VALUE-TARGET (value head blind to SealBot-reachable losses).** 68 lost net-vs-SealBot@d5 games (s150k=18, s175k=26, s200k=24 — the intransitive triangle); decisive blunder per game via FULL per-ply SealBot-d6 `last_score` scan + WIN→LOSS-persists-to-terminal filter. Raw gates: VALUE 61 / LINES 13 / TACTICS 4 — **LINES NEVER standalone** (all 13 are VALUE co-fires). The raw VALUE gate fires at the d6-WIN decisive ply (only confirms the position was winning), so a CORRECTED discriminator (`p2_value_discriminator.py`) re-uses the logged **post-blunder** net_value: **VALUE-BLIND 56/61 = 92%** — the value head reports net_value 0.6–1.0 (winning) at positions d6 calls forced-mate-against (e.g. net **0.99** at a forced loss). Severe value-head blindness, not an instrument artifact.

**Adversarial:** REVIEW (held-out s175k, independent re-derivation) CONFIRMS 11/11, 0 discrepancies, d6 determinism proven. RED-TEAM: the 13 LINES are REAL (d7 5/5 survive — the MODEL missed the line, ref_mass 0.000–0.010 rank 16–216 — not a reference miss); persistence filter changed 0/68. Off-window completing-cell decision-time rate 43% (terminal 18% was a window-recentring artifact) — but §D-LADDER already FALSIFIED off-window as a deployable defect (fixed-depth inversion 0.35→0.55) → descriptive of the SealBot matchup, NO lever built on it.

**Stage-2 routing:** #1 = **VALUE-TARGET / horizon-aware value calibration** on the deploy Gumbel@150 head. **bot-mix / SealBot-vs-anchor imitation corpus DROPPED** (LINES not plurality); §S178 bot-mix recipe stays UNLAUNCHED. Corpus regen OFF / spec-only (gated on P2=LINES, did not fire) — no GPU-week launched. n stays 150. Predicts **P3 PLATEAU** (search can't fix a mis-calibrated value head).

**P4 deliverable — deploy-matched in-loop strength eval SHIPPED + committed:** `hexo_rl/eval/deploy_strength_eval.py` — Gumbel SH greedy, **g=0** (`gumbel_scale=0.0`), no temp, deploy sims; fixed-depth-5 SealBot; adaptive screen(80)→confirm(200); distinct-game bootstrap BT-Elo gate; fail-safe (no PUCT/temp/64 fallback). Default-OFF opponent `deploy_strength` (existing runs bitwise-unchanged). RED-TEAM: g0 verified, gate traced end-to-end, false-negative bounded (~2.4% only at the exact 0.55 bar, stride re-screens), **14 tests pass**. Residual: pin the screen-band config defaults before deploy. Fixes the §D-LADDER "triple-miss" instrument mismatch.

**P0/P1/P3:** P0 — depth-5 = reproducible SealBot bar; **unit = HTTT TURNS** (depth-5 ≈ 10 stones; 5 = median+1). P1 — banked jsonl PARTIAL (moves, no value) → re-eval, no fresh games. **P3 (search-scaling 150-vs-256) = PLATEAU-by-150** (CONFIRMED on vast, 1200 games, ~7h co-tenant): no checkpoint has CI_lo(WR@256) > CI_hi(WR@150) — 150→256 drops/flatlines everywhere (s150k .575→.400, s175k .350→.250), n512 no climb, depth-4 red-team matches depth-5 → keep n=150, 256 not worth 1.7×; confirms value-blindness is not a search-budget artifact. Runbook `docs/handoffs/d_localize_p3_search_scaling_vast_runbook.md`; verdict `reports/d_localize_p3/P3_VERDICT.md`.

**HARD INVARIANT — live run NOT touched:** the d1m run (PID 1512430, `/workspace/hexo_rl`) is healthy + actively training (~249k; the apparent "stale log" was vast clock-skew, not a hang). Declined the unattended kill (not explicitly authorized; a healthy 2-day run). P3 co-tenants instead — strictly more conservative than the dispatcher's "stop live PID" precondition. To free the full 5080: `ssh vast 'kill -INT 1512430'`. Live status: `docs/handoffs/d_localize_run_status.md`.

## §D-FULLSPEC — full-spectrum value-target discriminator: TARGET vs FEATURE → ENTANGLED — 2026-06-26

**Question (follow-on to §D-INJECT NO-GO):** loss-only hard −1 value-distill craters wins in lockstep with correcting losses (KILL-A ⊥ KILL-C). Is that an irreducible FEATURE limit, or did loss-only just lack the contrast gradient? One frozen-trunk probe decides: distill the value head toward BOTH classes (d7-confirmed losses −1 ∪ genuine wins +1), class-balanced, and measure the collateral DS3 never did (KILL-C). Full record `reports/d_fullspec_2026-06-26/D_FULLSPEC_findings.md`; data on vast, pulled local (gitignored). Nothing committed.

**Verdict = ENTANGLED (FEATURE problem).** Full-spectrum contrast did NOT rescue separability — the D-INJECT anti-correlation PERSISTS. **Founding-measurement trap caught:** win/loss sets match on stones (median 51 vs 52) but the ENTIRE class gap is the turn-phase broadcast planes (v6_live2 plane 2 `moves_remaining`, plane 3 `ply_parity`) — a frozen trunk separates off turn-phase alone (turn-phase-only logistic holdout **AUC 0.807**). Judged on the **turn-phase-matched** set: KILL-A **0.762 PASS** (>0.35) but KILL-C **0.441 FAIL** (≥0.85 req) — a 0.41 crater, not near-threshold. Six independent matched KILL-C estimates all crater [0.324–0.462]: matched 0.441, caliper-1:1 0.455, tp0-only 0.324, 400-epoch plateau 0.382 (max-after-warmup 0.412, never nears 0.85), neutralized (planes 2,3=0) 0.330, in-sample train-fit 0.462. Naive KILL-C 0.520 is the shortcut (tp=1 easy wins); matched strips it.

**Adversarial:** REVIEW (independent, game-disjoint via recovered `game_id` byte-join, 3 fresh seeds, shared_games==0) → matched KILL-A 0.740 / KILL-C 0.482, **ENTANGLED confirmed**; turn-phase control verified-not-asserted (matched stratum has plane2==plane3==0). RED-TEAM (4 axes, `verdict_robust=true`): (1) within-matched turn-phase probe AUC 0.500 = chance → false-SEPARABLE impossible; (2) leakage none (game/byte/Hamming ≤4 all 0); (3) under-powered **REFUTED** — smoking gun: head fits only **46.2%** of its OWN tp0 training wins (vs 88.6% losses) → in-sample conflation, NOT sample size → labeling more wins won't help; (4) effective-n honest (34 matched wins = 28 distinct games; Wilson95 upper bound 0.61 ≪ 0.85).

**Cheaper precursor RAN → ENTANGLED_LT (light-trunk unfreeze, 2026-06-26).** Single-variable: unfreeze trunk @ light LR (swept {1e-5,3e-5,1e-4} + block11-only), same discriminator, strict game-disjoint split (shared_games=0, 308 games), same turn-phase-matched holdout. Best matched KILL-A 0.653 ✅ / KILL-C 0.532 ❌ — same conflated corner as E1 frozen (0.762/0.441); `joint_pass=0` over 8 estimates [0.455–0.614]. Overfit ruled out (best config train−holdout gap −0.019; high-LR overfit still craters holdout; in-sample KC ~0.6). RED-TEAM mechanism: the finetune raises KILL-A only by GLOBALLY depressing value (mean_v +0.055→−0.407 on 300 unrelated positions) — that IS the anti-correlation, and it damages the broader value head. Independent REVIEW agrees (0.648/0.568); RED-TEAM `verdict_robust=true`, `restart_warranted=true`. ⇒ trunk adaptation does NOT separate; a light-trunk finetune target-fix is NOT viable (and is destructive). `LIGHTTRUNK_findings.md`, `scripts/dvderisk_lighttrunk_probe.py`.

**E2 threat-plane re-probe RAN → ENTANGLED_R (2026-06-26).** Apples-to-apples input-feature ablation (`scripts/dvderisk_e2_featablation.py`): one small from-scratch conv, 4-plane control vs 8-plane (+4 opponent-agnostic threat planes cur/opp open-line-count + best-window-fill, verified correct vs `engine/src/board/threats.rs:check_window`, max_abs_diff 0.0), same game-disjoint + turn-phase-matched holdout. Control reproduces ENTANGLED (matched KILL-A 0.982 / KILL-C 0.070); treatment 0.930 / 0.155; ΔKILL-C **+0.084** (indep REVIEW +0.126) — below the +0.15 PARTIAL threshold, ~6× below 0.85. DECISIVE false-negative guard: a clean game-disjoint **linear probe** on the 8 threat summaries tops out at held-out matched **AUC 0.646** (train-fit 0.71 ≈ holdout, not overfit) → the threat features' held-out CEILING itself is far below separation; the treatment's only edge is in-sample (train-fit KILL-C +0.27, zero transfer). Both verify agents: ENTANGLED_R, `verdict_robust`, `restart_warranted=false`, control valid, planes correct, no leakage. `E2_REPROBE_findings.md`.

**Closeout RAN → CLOSED_ENTANGLED (2026-06-26).** Last gap (E1/LT/E2 all distilled via the value head's global pool): is win/loss recoverable from 272357's FROZEN representation by ANY readout? `scripts/dvderisk_closeout_probe.py` — flexible probes (logistic/MLP/conv) on POOLED vs PRE-POOL spatial [128,19,19], game-disjoint turn-phase-matched, + the literal richer continuous SealBot-value TARGET distill. Best pooled holdout AUC 0.70 / KILL-C 0.43; best pre-pool spatial AUC 0.62 / KILL-C 0.37 (pre-pool ≤ pooled → REOPENED_READOUT refuted, pooling discards nothing recoverable, no attention-pool/head lever). Richer continuous target (re-score validated, sign-AUC 0.731) → KILL-C 0.045 (craters harder than E1's ±1 → REOPENED_TARGET refuted, anti-correlation is representational not target-richness). NOT a false-negative (all probes fit TRAIN AUC 0.94-1.0 → ample capacity, generalization failure). Independent REVIEW (exact value-head reconstruction max-err 0.0) + RED-TEAM (Part B reproduced bit-for-bit, Wilson CIs all <0.85) agree, `verdict_robust`. Caveat: weak sub-threshold residual (pooled AUC ~0.70 > chance) → "absent" = below-usable-threshold.

**FINAL — DEEP/HORIZON architectural limit; cheap-lever space EXHAUSTED.** value-target distill (E1, 0.441) ∧ light-trunk finetune (LT, 0.532) ∧ threat input features (E2, 0.155/ceiling 0.646) ∧ any readout incl. pre-pool spatial (Closeout, 0.43/0.37) ∧ richer search-TARGET (Closeout, 0.045) ALL fail. The deep-forced-loss win/loss signal is decision-relevantly ABSENT from the frozen v6_live2 representation (consistent with D-PERCEPT 67% deep value-blind / ~15% short-lookahead-catchable, §D-LADDER P3 search PLATEAU-by-150). The `v6_live2_ft` bootstrap restart is NOT evidence-justified. Remaining directions are ALL expensive/architectural — substantially deeper/wider net or search-in-the-loop (MuZero-style learned lookahead) — none gated by a cheap discriminator, all gambles. RULE: no GPU-week without a cheap game-disjoint discriminator first clearing KILL-C≥0.85 vs a valid entangled control. Cheap discriminator space exhausted ⇒ treat the value blind-spot as a CHARACTERIZED KNOWN LIMIT of the current static-value architecture on adversarial forced-loss positions. Do NOT relabel more wins (in-sample 46% recall ruled win-count out).

### Falsified Hypotheses Register addition (§D-FULLSPEC 2026-06-26)
- **"The value blind-spot is a TARGET problem — loss-only distillation just lacked the win-contrast gradient; full-spectrum class-balanced distillation on existing v6_live2 features will SEPARATE wins from losses (KILL-A>0.35 AND KILL-C≥0.85 together)."** → **FALSIFIED.** Frozen-trunk full-spectrum distill on turn-phase-matched held-out: KILL-A 0.76 ✅ but KILL-C 0.44 ❌ (six estimates crater [0.32–0.46]); in-sample train-fit 46% rules out under-power; D-INJECT anti-correlation persists. The existing 4-plane features CONFLATE win/loss in the net_value>0 blind-spot neighborhood once the turn-phase shortcut (AUC 0.807) is controlled. ⇒ FEATURE problem, not TARGET. **Re-validation closed:** the registered light-trunk-unfreeze precursor RAN (2026-06-26) = ENTANGLED_LT — trunk adaptation also fails to separate (best 0.653/0.532, joint_pass=0, raises KILL-A only by globally depressing value). So a light-trunk finetune target fix is ALSO falsified; the FEATURE diagnosis holds from both frozen and unfrozen ends.
- **"Richer INPUT features (KataGo-style threat planes: open-line-count + best-window-fill) will carry the win/loss signal the 4 base planes lack and justify the v6_live2_ft restart."** → **FALSIFIED (E2 2026-06-26).** 4-plane control vs 8-plane treatment, same from-scratch conv, game-disjoint turn-phase-matched holdout: ΔKILL-C +0.084 (REVIEW +0.126), both crater (treatment 0.155 ≪ 0.85). A linear probe on the threat summaries ceilings at held-out AUC 0.646 (≈ train-fit → the features' ceiling, not capacity). Threat planes verified correct. ⇒ the bottleneck is NOT cheap input features; restart NOT justified.
- **"Win/loss is recoverable from 272357's frozen representation by some readout (the value head's global pool discards it / a richer search-derived value TARGET separates) — a cheap head/target fix avoids the restart."** → **FALSIFIED (Closeout 2026-06-26).** Flexible probes on pre-pool spatial [128,19,19] do NOT beat the pooled probe on game-disjoint holdout (AUC 0.62 ≤ 0.70; pooling discards nothing recoverable). The richer continuous SealBot-value target (re-score validated, sign-AUC 0.731) distilled into the frozen head craters wins (KILL-C 0.045, worse than ±1's 0.441 → anti-correlation is representational, not target-richness). Not under-fit (all probes train-AUC 0.94-1.0). ⇒ no cheap readout/target lever; the signal is decision-relevantly ABSENT from the frozen representation. **CONVERGENT CLOSE — cheap-lever space EXHAUSTED:** target-distill ∧ trunk-finetune ∧ input-features ∧ any-readout ∧ richer-target ALL entangle → DEEP/HORIZON architectural limit; remaining = expensive/architectural only (deeper net / search-in-the-loop). Treat as a characterized known limit.
- **Bonus (E0):** resume-whitelist bug confirmed + fixed (precedence inverted, `orchestrator.py`); **no catalogued run contaminated** (bootstraps weights-only → fallback path → bug inert); **§S180a cqv-off confound REFUTED**.

## §D-TACTICAL — are the deep value-blind losses search-fixable (level-k traps)? cheap SEARCH-side discriminator — 2026-06-26

**Question (follow-on to §D-FULLSPEC CLOSED_ENTANGLED).** §D-FULLSPEC exhausted the STATIC-value cheap-lever space and named two expensive survivors: deeper/wider net OR search-in-the-loop. §D-TACTICAL runs the SEARCH-side cheap discriminator §D-FULLSPEC never ran — reframing the deep value blind-spot as the textbook MCTS **level-k search-trap** weakness (HeXO selective-MCTS losing to SealBot=minimax on a sharp 6-in-a-row): does a BOUNDED tactical search in HeXO's OWN engine PROVE the proven-loss traps the deploy search (Gumbel-SH-150 g=0) misses, and how cheaply? Offline, no retrain, no GPU-week. Full record `reports/investigations/d_tactical_2026-06-26.md` + `reports/d_tactical_2026-06-26/` (gitignored). Engine: read-only forcing-move PyO3 bindings added to `engine/src/board/moves.rs` (`threat_moves`/`winning_moves`/`count_winning_moves`/`first_winning_move`/`has_player_long_run`/`forced_win_move`) + `engine/src/pyo3/board.rs`; probe harness `scripts/dtactical/`. Nothing else committed.

**Foundation (all gates PASS).** Pivoted off the 493 d7 plane bank (V1: window-clipped, 39% reconstruct, 94% of failures off-window-border) to a **reachable-replay corpus** — 61 decisive deploy blunders replayed from `per_game_seald5` true moves (complete boards, zero clipping). **V2** net-parity 61/61 Δ=0 (per-bucket s150k/s175k/s200k ckpts — NOT 272357, which made the DS1 net). **V3** fresh fixed-depth SealBot reproduces `proven_core=33` vs disc_merged 33. **V4** new bindings fuzz-match a Python ref 1364/1364. 33 value-blind proven-core (gate denominator); bands in SealBot TURNS: 16 short / 16 mid / 1 deep.

**Verdict = SEARCH-VIABLE-BUT-NOT-CHEAP** (a 4th cell beyond the pre-registered HYBRID-VIABLE / PARTIAL / TRAPS-TOO-DEEP trichotomy). **T0** (deploy baseline): Gumbel-SH-150 g=0 **mis-evaluates 85% (52/61)** of proven losses (the SEARCH, not just the static head, is blind); **67% (37/55) of saving moves have ~0 deploy-policy prior** → MCTS structurally cannot explore them → **explains the §D-LADDER P3 search-scaling PLATEAU** (more sims can't find a ~0-prior move) → fix must OVERRIDE the prior, not re-weight. **T1/T2** (cheap HeXO-native, NET-FREE + SEALBOT-FREE threat-space proof search): flips only **3/38 = 8%** of proven-core (all mate-in-2-turns; mid 0/16, deep 0/1), median 47 nodes, **0 soundness violations**. **Fails the pre-registered ≥40% gate decisively.** Mechanism (PV-confirmed): these mates mix forcing + QUIET developmental moves — a one-primitive threat search bails when the line goes quiet; full-width minimax hits the 16⁶ wall for even a 6-ply mate. **Existence proof:** SealBot resolves the 38 it can prove at d6-8 (fast for short mates, up to ~122s for deep) → the losses ARE bounded-tactical ⇒ **TRAPS-TOO-DEEP / expensive deeper-net frontier REJECTED.**

**Adversarial (both PASS).** REVIEW (fresh, independent): 3/38 flip set reproduced exactly; 3 flips genuine (fresh SealBot confirms mate); 8% real (MISSes reconstruct faithfully, genuine UNKNOWN on confirmed losses); deploy mis-eval 85% reproduced; 0 not-loss / 0 unproven-loss false-flips. RED-TEAM (distinct, 5 vectors, survives all): **V1 oracle-leak clean** (`solver.py` imports only `engine`, no `minimax_cpp`/net); **V2 (could-overturn) STRENGTHENED — broadening candidates REGRESSES** (open-3/dist-1/full-brute all 0/38 at ~100× cost) ⇒ the cheap-add ceiling is FUNDAMENTAL, SealBot-grade pattern-ordering+α-β+eval genuinely required, not a solver-weakness artifact; **V3 cost qualified** (measured native optimized **2.6–2.8M NPS**, original 949k: root-d6 ≈ 0.37–0.65 s/move = deploy-affordable for the SHORT band; per-leaf ×150 ≈ 55–98 s = NOT; d8 = training-time); V4 reachable by construction; V5 depth cliff sharper (all 3 flips mate-in-2; mate-in-3 already 0%).

**Lever = custom NATIVE-RUST `engine::tactics` solver (NO SealBot round-trip)** — undo-based ID-α-β (`core.rs:563` O(1) undo → no clone-per-node, the Python probe's 290 nodes/s tax) + TT(u128 zobrist) + killers/history + threat-ordered candidates + MCTS-Solver CF-1 backup + net-policy ordering. Hooks: `expand_and_backup_single`/root (deploy solver-backup, short band), `finalize_game` z-correction (training-time, all bands amortized — the load-bearing route). Kills the SealBot dependency + colony OOB + time-flakiness. Spec `reports/d_tactical_2026-06-26/NATIVE_RUST_SOLVER_design.md`.

**Ceiling (can it EXCEED SealBot?) → can-exceed, CONDITIONAL.** As SealBot-LABEL distillation = capped + already falsified (§D-INJECT/§D-FULLSPEC). As the self-play-embedded-search BOOTSTRAP (the actual lever, = the survivor §D-FULLSPEC named) = NOT capped: a solver-in-the-loop is a policy-improvement operator (corrected z + policy ratchet, no fixed teacher point). SealBot is a FLOOR not a ceiling — in-corpus it FAILS on 23/61 = 38% (18 unproven-loss + 5 not-loss sign-flip d6→d7), `CANDIDATE_CAP=15` makes quiet refutations structurally invisible (same blind spot as the cheap add), colony OOB, time-cap-corrupted d8. **Decisive caveat (Probe C, empirical):** augmenting the search with the CURRENT net's policy → **0/14 flips** (the blind net can't guide to ~0-prior moves) ⇒ exceedance is chicken-and-egg, gated on closing the training loop first; collapses to distillation if dropped. Does NOT repeal the static-head horizon (~15% value-learnable) — escapes the ceiling by carrying the deep tail as deploy computation + making the catchable subset learnable.

### Falsified Hypotheses Register addition (§D-TACTICAL 2026-06-26)
- **"The deep value-blind losses need the expensive frontier (deeper/wider net or MuZero) — they are not bounded-tactical."** → **REJECTED.** SealBot (a bounded minimax) proves all 33 value-blind proven-core at depth 6–8 turns (fast for short mates) ⇒ they ARE bounded-tactical game-tree properties; the deeper-net frontier is NOT earned. The lever is SEARCH-in-the-loop, discriminated from "deeper net" — advances §D-FULLSPEC's two-survivor close to one.
- **"A CHEAP tactical add (threat-space search / shallow minimax) flips ≥40% of the traps at <10× the deploy node budget (cheap-HYBRID-VIABLE)."** → **FALSIFIED.** A HeXO-native, net-free, SealBot-free threat-space proof search flips only 3/38 = 8% (all mate-in-2-turns; mid/deep 0%); BROADENING the candidate set (open-3 / dist-1 / full-brute) REGRESSES to 0/38 at ~100× cost (wider branching collapses reachable depth). 0 soundness violations; oracle-leak-clean; REVIEW+RED-TEAM confirm. ⇒ the cheap-add ceiling is fundamental: a competent SealBot-GRADE pattern-guided minimax is required, not a one-primitive probe. Deploy solver-backup is affordable (measured 2.6–2.8M NPS, root-d6 ~0.5 s/move) ONLY for the short ≤2-turn band; mid/deep are a bounded residual handled training-time.
- **"Policy-guided candidate ordering with the current deploy net would lift the cheap-search flip-rate (the AlphaZero ingredient already helps)."** → **FALSIFIED for the CURRENT net (Probe C).** Augmenting candidates with the net's top-K policy → 0/14 flips (the net is ~0-prior-blind on the very refuting moves, per T0). ⇒ policy-guidance only helps AFTER the net is trained with search-in-the-loop — confirms the exceed-SealBot route runs THROUGH the bootstrap, not around it.

## §D-WS3 — L1 solver-in-loop smoke: v2 CONFOUNDED (regime, not lever) → v3 three-arm re-run pre-registered — 2026-07-01/02

**Question (follow-on to §D-TACTICAL's native-solver lever).** Does the L1 solver-
in-loop SOFT visit-injection fine-tune (per-move `engine::tactics` proves the side-
to-move's forced win, blends visit mass onto the proving move into the POLICY
training target) teach the POLICY to play the saving move **standalone**
(GENERALIZES, held-out move-level trap-flip ≥25% → fund the GPU-week) or cap at the
~16% memorization floor (MEMORIZES → deploy-backup permanent for the class)? Spec
`docs/designs/coupled_valuez_decode_design.md` §0/§5 (Stage-1A); runbooks
`docs/handoffs/d_ws3_l1_smoke_runbook.md` (v2, superseded) and
`docs/handoffs/d_ws3v3_smoke_runbook.md` (v3, live).

**v2 RAN on vast (2026-07-01/02, single candidate arm `solver_visit_weight=0.3`,
`--resume checkpoint_00200000.pt`) → CONFOUNDED, not a clean read.** `checkpoint_00200000.pt`
is a FULL checkpoint (`model_state`+`config`+`optimizer_state`+`scaler_state`+`step`),
so `trainer_ckpt_load.load_checkpoint` classified the resume `is_full_ckpt=True` and
the run silently inherited the ANCHOR's LR scheduler STATE instead of the variant's
intended fresh `lr: 2e-3`→`eta_min: 5e-4` decay: run-logs confirm **~1.85e-3**
(`ws3_z2_l1.jsonl` last train_step lr=0.0018557751539875656 at step 200802;
`ws3_ctrl.jsonl` lr=0.0018456274188078784 at step 208000), the anchor's 1M-step
cosine evaluated near step 200k. The replay buffer had ALSO been rm'd before the
run (`ws3_ctrl.jsonl`'s first `warmup` event: `{"buffer": 0, "target": 256, "games": 0}`)
— training began on a cold buffer, `min_buffer_size` at the training.yaml default 256
(not a real prefill gate).

A solver-OFF CONTROL arm (`z2_solver_in_loop_control.yaml`) was launched under the
SAME broken regime and reproduced the candidate's damage almost exactly:

| metric | control (solver OFF) | candidate (solver ON, w=0.3) |
|---|---|---|
| trap-flip, registered-31 | 12.9% → **6.5%** | 12.9% → **6.5%** |
| trap-flip, combined-125 | 19.2% → 15.2% | 19.2% → 15.2% |
| deploy-disagree (KILL co-gate, n=55) | 49.1% | 47.3% |
| threat-probe C2 (top5, vs BOOTSTRAP baseline) | 30% PASS | 20% FAIL |
| off-window forced rate (deploy + modelplayer) | 0.0 | 0.0 |

Both arms crater identically on the same held-out corpus → the regime drift (LR +
cold buffer), not the solver injection, produced the damage. **VERDICT: the
injection is EXONERATED; the v2 read is INDETERMINATE** (a broken-regime null
result cannot support MEMORIZES, KILL, or GENERALIZES — nothing was actually
tested). C2 is the one metric that differed between arms but was gated against the
WRONG baseline (bootstrap `contrast_mean=+0.599`; the anchor's own is +7.3095, 12×
sharper — gating a resumed fine-tune against bootstrap is nearly a no-op floor).
Also measured (v2, offline; no in-run fire-rate logging existed then): solver
fire-rate 5.6% of moves overall / 16.4% late-game — a real starvation risk for
fresh, unseeded self-play.

**Banked lessons (mechanism, not just this run):**
1. A **weights-only warm-start is required to pin a fine-tune's LR** — a full-
   checkpoint `--resume`/`--checkpoint` CANNOT do it: `RESUME_CHECKPOINT_OWNED_KEYS`
   (`hexo_rl/training/orchestrator.py:233-252`) includes `lr`/`lr_schedule`/
   `eta_min`/`min_lr`/`total_steps`/`scheduler_t_max` — the variant's LR intent is
   silently discarded whenever `config`/`optimizer_state`/`scaler_state` are present
   in the source checkpoint (`trainer_ckpt_load.py`'s `is_full_ckpt` gate,
   L328-336). Strip to `{model_state, metadata, step}` BEFORE the resume.
2. **The replay buffer must be explicitly restored (or a real prefill gate set),
   not implicitly reused.** A cold/rm'd buffer under the training.yaml default
   `min_buffer_size: 256` trains on ~0 real self-play positions for the whole run.
3. **In-run solver fire-rate logging is mandatory**, not a post-hoc offline probe —
   otherwise a THIN-STILL (density-starved) read is indistinguishable from a true
   MEMORIZES read.
4. **Probe gates must compare against the run's own warm-start ANCHOR, not the
   generic bootstrap baseline** — the anchor is far sharper on the threat-probe
   (contrast +7.31 vs bootstrap's +0.60), so a bootstrap-gated C1 floor is nearly
   vacuous for a resumed fine-tune; only C2/C3's flat percentage thresholds bit at
   all, and even those should be read against the anchor's own values (C3 sits
   EXACTLY at its 40% threshold at n=20 — a borderline anchor score needs the
   comparison to be honest about that).

**v3 pre-registration (`docs/handoffs/d_ws3v3_smoke_runbook.md`).** Fixes: (V0)
weights-only warm-start pins `lr: 1.0e-4` flat (`lr_schedule: none`), per-arm
`mixing.buffer_persist_path` + `min_buffer_size: 25000` (POSITIONS) real prefill
gate, threat-probe gated vs `tests/fixtures/threat_probe_baseline_anchor200k.json`
(minted 2026-07-02, CPU: contrast_mean=+7.3095, top5=30%, top10=40%, n=20 —
matches the vast-measured anchor numbers from the v2 logs exactly), in-run solver
fire-rate Rust counters (`solver_moves_eligible/injected/...`) surfaced as
`training_step.solver_fire_rate`/`solver_fire_rate_seeded`. (V1) trap-corpus
START-POSITION SEEDING (`selfplay.seed_fraction`, KataGo `startPoses` mechanism;
`scripts/build_ws3v3_seed_corpus.py` mines NEW proven-loss traps from the untapped
`per_game_seald5.jsonl` games EXCLUDING every eval corpus at mine-time, expands each
into seeds at cuts k∈{0,2,4} plies before the parent; `scripts/check_ws3v3_disjointness.py`
MEASURES — not assumes — game-level, position-level, and leakage-distance
disjointness from the eval sets before the seeded arm's flip numbers are trusted).
(V2) `solver_visit_weight: 0.5` (0.3's "too aggressive, soften to 0.15" rationale is
itself FALSIFIED by the control-arm exoneration above — the weight was never shown
guilty — so v3 steps UP to 0.5 under the fixed regime instead of continuing to
soften; 1.0 held as an escalation). (V3) three arms — **ARM-CONTROL** (solver OFF,
no seeding — the pre-registered VALIDITY GATE, read FIRST: must hold flat on
deploy-disagree/trap-flip/threat-probe-vs-anchor before either ON arm is read at
all), **ARM-INJECT** (solver ON w=0.5, no seeding), **ARM-SEEDED** (solver ON w=0.5,
seeding ON, `seed_fraction: 0.15`).

**Verdict definitions (pre-registered, `docs/handoffs/d_ws3v3_smoke_runbook.md` §6):**
GENERALIZES (an ON-arm flip ≥25% AND control flat AND KILL≤16% AND co-gates clear →
fund the GPU-week + carry seeding forward); MEMORIZES (ON-arm flip ≤16% held-out
AND, for ARM-SEEDED, `solver_fire_rate_seeded` confirmed high — the density lever
fired, it just didn't teach → D-BOOTSTRAP fires, solver-in-loop from step 0);
THIN-STILL (fire-rate high but flip unmoved vs ARM-INJECT → one escalation to
`solver_visit_weight: 1.0` before declaring MEMORIZES); KILL (deploy-disagree >0.16
on an ON-arm with control clean → soften the weight, re-smoke); INDETERMINATE
(control itself drifts → the regime is still broken, fix before reading anything —
the exact trap v2 fell into). GPU run is operator-run on vast; this session shipped
the build (weights-only strip script, probe `--baseline-json`/`--write-baseline-to`,
seed-corpus miner + disjointness checker, three arm variants) — no training executed
locally (laptop is CPU/thermal-capped for this workload).

**v3 review + red-team fixes (2026-07-02, same session, before any GPU spend).**
Review + red-team of the v3 build surfaced two additional issues, fixed before
launch:

**FIX1 (BLOCKER — encoding).** `reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt`
carries STALE `metadata['encoding_name']='v6_live2'` (single-window).
`hexo_rl/training/trainer_ckpt_load.py::load_checkpoint` prefers checkpoint
metadata unconditionally, and `hexo_rl/training/orchestrator.py::init_trainer`
(the `:343-355` back-propagation loop) writes the ckpt-resolved encoding into
`combined_config`, which `hexo_rl/selfplay/pool.py` resolves for the Rust
self-play runner. Left uncorrected, **all three v3 arms would have self-played
single-window v6_live2 despite every variant declaring v6_live2_ls** — the
entire point of the multi-window off-window-injection routing this workstream
exists to test. Fix, at the artifact boundary (not the propagation mechanism —
`orchestrator.py`'s ckpt-wins precedence is intentional, §171 P3 depends on
it): `scripts/make_ws3v3_warmstart.py` gained `--encoding-name` (default
`v6_live2_ls`), which re-stamps `metadata['encoding_name']` in the stripped
payload AFTER validating — via the encoding registry — that the override
shares the original's WIRE SIGNATURE (`n_planes`, `board_size`,
`policy_logit_count`, `has_pass_slot`, `sym_table_id`; mirrors Rust
`RegistrySpec::wire_signature()`). v6_live2 and v6_live2_ls share this tuple
exactly (`(4, 19, 362, True, "size_19")` — they differ only in self-play/decode
MECHANICS: cluster_window_size, is_multi_window, value_pool, policy_pool), so
the re-stamp is safe; a genuinely different encoding (verified with `v6`,
`n_planes=8`) is REJECTED loudly. `orchestrator.py`'s propagation loop also
gained a LOUD `checkpoint_encoding_overrides_variant` structlog warning
(log-only, precedence unchanged) for any future case like this one.
`checkpoints/ws3v3_warmstart_200k.pt` was regenerated with the fix. END-TO-END
VERIFIED locally 2026-07-02 via a real 3-step launch
(`--variant ws3v3_arm_control --iterations 3 --min-buffer-size 64`,
`logs/train_60cc566090434086a8218697040a54ed.jsonl`): `checkpoint_encoding_resolved`
and `train_encoding_resolved` both read `v6_live2_ls`/`is_multi_window=true`,
no `checkpoint_encoding_overrides_variant` warning fired (variant and
ckpt-resolved encoding now agree), `train_step.lr` read `0.0001` flat across
all 3 steps, step continuation `200000→200003` (absolute).

**Retroactive implication for v2 (and possibly further back).** The same
propagation mechanism means **v2 (`docs/handoffs/d_ws3_l1_smoke_runbook.md`)
ALSO self-played single-window v6_live2**, not the `z2_solver_in_loop.yaml`
variant's declared v6_live2_ls — confirmed from `reports/vast_ws3_v2/logs/ws3_clean.jsonl`
and `ws3_ctrl.jsonl`: `checkpoint_encoding_resolved=v6_live2` with the identical
propagation path. Consequently v2's off-window injection routing
(`window_half=None` → legal_set ragged target) was **DEAD** — the single-window
Dense self-play path drops off-window wins outright — and the
`z2_solver_in_loop.yaml` header's "HARD INVARIANT graft A ... Confirmed, no
mismatch" claim was **FALSE AT RUNTIME** (retracted in the file's own comment
block). This adds a THIRD item to v2's confound list (alongside the LR-schedule
and cold-buffer holes already documented above): encoding — v2's self-play ran
v6_live2 single-window; the checkpoint metadata cascade overrode the variant's
declared encoding; the off-window injection path was never actually exercised.

**OPEN FORENSIC QUESTION (flagged, NOT investigated this session):** the same
200k anchor's own BAKED config (inside `checkpoint_00200000.pt`) reads
`encoding=v6_live2` / `cluster_window_size=None` — whether the entire
`d_decide`/`d1m` training lineage that PRODUCED this anchor self-played
single-window all along (the metadata cascading through a chain of resumes,
not just this workstream's own resume) is an open question that needs a
dedicated audit, not a fix folded into this one. Evidence pointer for that
audit: `checkpoint_encoding_resolved` events across the lineage's run logs +
each checkpoint's baked `config['encoding']`/`config['cluster_window_size']`.
**→ RESOLVED 2026-07-02 (§D-FORENSIC F1): CASCADE-CONFIRMED, full lineage,
from step 0** — worse than flagged: the override fired at the initial fresh
launch (string-form `encoding:` bug), not through resume cascades. One
consequence for the v3 verdict read: the control arm is itself absorbing a
FIRST-EVER multi-window self-play distribution shift (the 200k warm-start
never generated an off-window move) — control-vs-anchor drift ≠ candidate
damage. See §D-FORENSIC below.

**FIX2 (MAJOR — seed fire-density expectation).** Red-team measured that seed
landing positions at cuts k∈{0,2,4} (before the trap PARENT) prove NO forced
win by construction (the parent is a "saving move exists" position), and even
the tested trap's POST-blunder position was NOT native-solver-provable at
budget 50k (matches [[d-tactical-search-viable-not-cheap]] — native solver has
weak recall on quiet traps) — so ARM-SEEDED's "`fire_rate_seeded ~1.0`"
expectation was wrong. Fix: `scripts/measure_native_provable_fraction.py`
MEASURES the honest ceiling (combined held-out corpus, POST position,
`TacticalSolver(window_half=None, cand_cap=40, neighbor_dist=2)`, depth 16,
budget 20000). Laptop debug build paces ~60-80s/trap, so the local
measurement is a SAMPLED estimate (uniform n=40/125,
`--sample 40 --seed 20260702`, deterministic order-preserving subset) —
**sampled fraction: 0/40 proven = 0.000, Wilson 95% CI [0.000, 0.088]**
(n_skipped=0; ZERO proven in EVERY sampled mate_distance bucket, 2 through 9
turns — not a deep-mate-only failure; 3498s ≈ 87s/trap debug; see
`reports/d_ws3v3/native_provable_fraction_sample40.json` + `_records.jsonl`);
the FULL-125 exact number is a v3 runbook §1(i) vast precondition (release
build ~2.5× faster, ~1h, optional but recommended — replaces the sampled CI
with the population value). `scripts/build_ws3v3_seed_corpus.py`
now additionally emits POST-blunder seeds (`cut: -1`) ONLY for traps the native
solver can prove AT BUILD TIME (computed fresh, not the miner's stored
`native_loss_verified` flag — the DS1 stale-label lesson); given the ~0
measured fraction, this gate is expected to emit few-to-zero `_kpost` seeds —
correct behavior, it refuses to seed positions the solver can't convert.
ARM-SEEDED's mechanism is now documented as DUAL: (i) trap-decision z-labels
densify ALWAYS (the KataGo `startPoses` value, no solver required), (ii)
solver policy-injection densifies only on native-provable conversions — at
the measured ~0 ceiling this half of the lever is expected DORMANT on this
corpus, making ARM-SEEDED's live mechanism the z-label half. A pre-registered
**SEED-STARVED** branch was added to the runbook §6 verdict table:
`solver_fire_rate_seeded < 2× the organic rate` (organic v2 = 5.6%, so
**< 11.2%**) → the seeding lever did not densify INJECTION for this class —
per the 0/40 measurement this branch is the DEFAULT expectation, not an edge
case → do not read MEMORIZES off ARM-SEEDED's flip alone (the z-label
mechanism may still show in the flip); interpret ARM-SEEDED against
ARM-INJECT instead of standalone, and the follow-up lever is native-solver
recall (D-TACTICAL), not more GPU.

**FIX3 (MAJOR — launch commands + clobber).** The runbook's launch commands
passed `--variant` a YAML PATH (`configs/variants/ws3v3_arm_control.yaml`);
`scripts/train.py` takes the BARE NAME and constructs
`configs/variants/{name}.yaml` internally — the path form 404s
(`FileNotFoundError`, verified). Also, none of the three launches specified
`--checkpoint-dir`/`--run-name`, so all three arms would have written the SAME
`checkpoints/checkpoint_00208000.pt` (ABSOLUTE step — the trainer's
`self.step` starts at the warm-start's own step field 200000 and increments
per training step, `f"checkpoint_{self.step:08d}.pt"`, NOT relative
`00008000`) and the SAME default log file. Fixed everywhere (runbook, the
three `ws3v3_arm_*.yaml` header comments, `d_ws3_l1_smoke_runbook.md` §6.5's
still-nonexistent `-m hexo_rl.training.run --resume` form): bare `--variant`
name, per-arm `--checkpoint-dir checkpoints/ws3v3_<arm>` +
`--run-name ws3v3_<arm>`, `--out` added to both §5 `exploit_probe.py` calls
(`required=True` — omitting it fails argparse before anything runs), an
explicit LAUNCH-ORDERING STOP-GATE (launch ARM-CONTROL alone, gate on §3,
only then launch the other two — ~23 GPU-hours sit downstream of that gate),
and §3's "control flat" PASS condition operationalized numerically
(registered-31 flip count in [2,6], combined-125 in [19,29], C1≥5.848/
C2≥25%/C3≥40% vs the anchor baseline) instead of a qualitative "flat" read.

**Bench gate (2026-07-02, vast 5080, remote-host targets):** commits `87b9c96..dddb07b`
benched at 2×`make bench` (n=5 each) per side, patch-applied vs `0c494b7` baseline —
**PASS, 10/10 targets met in all 4 snapshots.** One WATCH per the >5% rule:
`buffer_push_per_s` −5.4% median-of-medians (pre 990k/1028k → post 931k/978k).
ACCEPTED as host noise, not code: the diff touches ZERO replay-buffer code
(`git diff 0c494b7..dddb07b -- engine/src/replay_buffer/` is empty; the push
microbench never exercises game_runner), best-case pairing is −1.2%, the driver is
the pre2=1028k HIGH outlier vs the documented §180 vast median 1007k (range
994k–1.01M), and post medians hold 77–86% headroom over the 525k target. JSONs:
`reports/benchmarks/20260702_ws3v3_{pre,post,post2}.json` + `2026-07-02_13-38.json`
(pre2), archived locally. Vast tree restored to pre-gate state after the runs.

---

## §D-FORENSIC — lineage encoding audit (F1) + solver budget-escalation diagnostic (F2) — 2026-07-02

**Dispatcher (CPU-cheap, parallel to the v3 smoke).** F1 resolved the G0-vs-anchor-config
contradiction flagged as §D-WS3's open forensic question; F2 tested the ONE knob the 0/40
native-provable finding hadn't (solver budget) — a BUDGET-BOUND result would have revived
ARM-SEEDED's injection half pre-launch. Both ran read-only/CPU on the laptop while the smoke
stayed operator-gated. Reports: `reports/investigations/f1_lineage_encoding_forensic_2026-07-02.md`,
`reports/investigations/f2_solver_budget_escalation_2026-07-02.md` (both with review addenda).

### F1 VERDICT = CASCADE-CONFIRMED — full lineage, from step 0 (worse than pre-registered)

d1m (`longrun_v6_live2_ls_gumbel_m16`) self-played **single-window `v6_live2` for its entire
history (step 0 → ≥272,357)** despite every variant declaring multi-window `v6_live2_ls`.
NOT a resume-cascade: the override fired at the initial fresh launch (2026-06-21).
**Root cause (E0-independent, two stacked holes):** (1) `_resolve_checkpoint_encoding`
(`trainer_ckpt_load.py:125`) treated the CANONICAL string form `encoding: v6_live2_ls` as
"unspecified" (dict-only isinstance) → filename/shape inference resolved the bootstrap
(`bootstrap_model_v6_live2_8300.pt`, no `_ls` substring; v6_live2/v6_live2_ls are
state-dict-shape-IDENTICAL) to single-window; (2) `save_checkpoint` restamped the wrong
resolution into `metadata['encoding_name']`, and the §172 A4.3 metadata-preference branch never
compared against the declared encoding → self-perpetuation through every resume. `encoding` was
checkpoint-owned both before AND after the E0 fix (`fe34e3f`, 2026-06-26) — E0 provably
irrelevant. **Evidence artifact-grade** (independently re-derived by fresh review, zero
discrepancies): production run-logs (`checkpoint_encoding_resolved=v6_live2` at 12:57/13:01
launch + 19:54 94k-resume), torch.load on 6 lineage ckpts (all stamped v6_live2), game replay
via `to_flat>=362` sentinel (d1m 2/48,512 ≈ 0.004% off-window vs positive-control `longrun_c3`
61/3,094 = 1.97%).

**Fixes LANDED (TDD, review SHIP-AFTER-FIXES → fixes applied same day):**
- `trainer_ckpt_load.py`: string-form `encoding:` now counts as explicit; NEW hard gate —
  `metadata['encoding_name']` disagreeing with an explicitly declared config encoding RAISES
  (mirrors `_resolve_checkpoint_encoding`; restores the documented §171 P3 "refuse to silently
  override" intent that the metadata stamp had bypassed). No-declaration configs keep
  metadata-wins backward-compat. Sanctioned encoding change = weights-only strip + re-stamp
  (the v3 warm-start path, unaffected).
- `anchor.py`: encoding mismatch on an anchor candidate emits dedicated ERROR
  `anchor_encoding_mismatch` + RAISES. Review BLOCKER caught the first version (return-None
  routed a VALID best_model.pt into quarantine-rename + silent fresh-init overwrite —
  reproduced end-to-end); fixed + pinned by integration test
  `test_resilient_load_encoding_mismatch_never_quarantines`.
- `orchestrator.py`: `train_encoding_resolved` now stamped `source="variant_declared_pre_checkpoint"`
  (it fires PRE-load; reading it as ground truth mis-routed the v2 forensic for a week —
  post-load truth is `checkpoint_encoding_resolved`).
- Regression tests pin both real failure shapes (weights-only generic-filename bootstrap; full
  ckpt with baked config + metadata stamp, string+dict declared forms). Targeted sweep
  537 passed / 0 failed; tests/training 117 passed.

**Lineage corrections applied:** D-DECODE G0 ("net was trained multi-window; deploy aligns to
training") FALSIFIED — correction block in `docs/handoffs/d_decode_results_and_build_plans.md`
(G0's trace hardcoded `ENC=v6_live2_ls`, never read the ckpt stamp). The position-level
0/50-vs-12/50 action-space result SURVIVES (structural). §D-LADDER TRUE-STALL re-scoped:
characterizes the single-window regime only. §D-WS3 v3: the control arm absorbs a FIRST-EVER
multi-window self-play distribution shift — control-vs-anchor drift ≠ candidate damage (noted
at the § itself).

**OPEN (F1 follow-ups):** (1) `hexo_rl/eval/checkpoint_loader.py` + `opponent_runners.py` are a
SEPARATE unfixed encoding-inference path (same ambiguity class, feeds SealBot eval /
exploit_probe / promotion-eval) — needs the same declared-vs-resolved gate; (2) the D-DECODE
full-game defense 0.0 + WR 0.50 results are MORE surprising under CASCADE-CONFIRMED (multi-window
deploy of a single-window-trained net) — re-verify before load-bearing use; (3) the 2/48,512
anomalous off-window hits are code-plausibly the legal-set escape valve (radius-5 ball independent
of window recentring) but the two plies were not position-verified.

### F2 VERDICT = ALGORITHM-BOUND — budget escalation buys ZERO conversions; injection half stays dead

Same 40-position fixture as the 0/40 (durably stored, hard-asserted, baseline reproduced
node-exact at 1×): **0/40 proven at 20k, 200k, 1M, AND 3M nodes** (all budget-exhausted; Wilson
[0, 0.088] at every tier). SealBot node cost MEASURED on the same positions via its own `_nodes`
counter: **median 11,054 at d8** (mean 2.04M, p95 16.3M — a band, not a point; 3M = 271× median,
≥ full d8 cost for 34/40). Native proves 0/40 at 271× the median SealBot spend → budget is not
the constraint. R3 LOSS-guard confirmed active at all tiers (`search.rs:253-270` + recall-verify).
Component inventory vs premise: 729-eval/scored-α-β/PVS+LMR/aged-TT all LANDED; **quiescence tail
DOES NOT EXIST**; eval-ordering tie-break unwired; net-policy ordering INERT by default.

**Localization (5 positions SealBot proves in 3–2,054 nodes):** the line is dropped at
**candidate-generation** — late-game roots have 347–408 legal moves, the widened candidate set
(178–201 cells) is truncated to cand_cap=40, and on `s150k_g241_p96` SealBot's principal defender
reply is NOT in the post-truncate set. Walking SealBot's own mating PV, native is budget-exhausted
UNKNOWN at every quiet interior ply and only proves the in-check mate-in-1-turn tail. The WIN-side
(OR-node, injection-facing, needs ONE line, no recall) is STILL unprovable at 1M → not just the
LOSS-frame recall cost. Cost arithmetic: sound LOSS certification compounds ~legal_move_count per
defender turn (mate-3 ≈ 5.8e6 lower bound, mate-5 ≈ 8e11) — 37/40 fixture positions are mate≥3,
structurally beyond any affordable tier. SealBot wins by an unsound-in-principle reduced movegen +
quiescence + eval ordering that happens to be right here.

**Consequences:** ARM-SEEDED's injection half does NOT revive (pre-registered fold into v3
declined; SEED-STARVED default stands, mechanism now localized). Post-smoke lever = native
RECALL work (quiet-move candidate-gen under real ordering + quiescence tail + eval-tie-break
wiring, possibly an unsound-but-tagged fast path for seeding/labeling kept OUT of sound z-label
channels) — consistent with §D-TACTICAL and D-SOLVER A2; not budget, not GPU.

**Process note (banked, matches [[workflow-agent-may-background-heavy-compute]]):** the F2 agent
died on a session limit while its detached 3M workers ran; its report pre-filled the 3M row with
PROJECTED wall stats 30 min before the data existed. Verdict-bearing numbers verified correct
against the real JSON post-completion; projected wall stats corrected (report provenance note).
Always diff a background-worker report's numbers against the artifacts it cites.

**Both probes' purpose served:** the GPU-week's verdict-reading frame is now fixed BEFORE it's
bought — v3's control arm carries the multi-window-shift caveat, and a SEEDED-arm null on
injection can't be mistaken for "budget was too low."

**RED-TEAM (distinct, adversarial — both verdicts STAND, strengthened):**
- F1 attack backfired into STRONGER evidence: the Dense/single-window path sorts off-window
  cells at prior 0.0 and truncates at `MAX_CHILDREN_PER_NODE=192` (`mcts/mod.rs:46`) — all 61
  positive-control (c3) off-window hits sat at in-window-avail 199–259 (> cap, escape-valve
  MATHEMATICALLY impossible → genuine multi-window decode), while d1m's 2 hits sat at 190/182
  (just under cap — textbook escape-valve). The two anomalous plies are now position-verified
  (F1 OPEN item 3 CLOSED). Honest residual: game-replay ground truth only exists for the
  ~120k/~200k eras; steps 0→94k rest on the config/log chain (correctly labeled in the report).
- F2: unit-mismatch attack bounded out (both node counters tick once per recursive search call);
  WIN-side gap closed LIVE at 3M (still unprovable, 237s); **cand_cap=200 (5× width) × 200k/1M
  probed on the localized positions: 0/6** — ALGORITHM-BOUND is not an artifact of holding
  candidate width fixed. Affordability arithmetic reproduces exactly (the "~10-20% throughput
  hit" headline is UNVERIFIED-not-refuted, non-load-bearing).
- Fix regression CAUGHT + FIXED (R3b): the anchor raise initially also fired on the hardcoded
  v6-family `_BOOTSTRAP_ANCHOR_CANDIDATES` fallback tier — a fresh non-v6 launch on a host with
  `bootstrap_model_v6.pt` present (and no best_model.pt yet) hard-crashed where it should skip →
  fresh-init. Fixed: raise scoped to the same-lineage best/.bak tiers; foreign bootstrap
  candidates skip with `anchor_encoding_mismatch_skipped` (warning); pinned by
  `test_resilient_load_foreign_bootstrap_mismatch_skips_not_raises`. 118 tests green.
- STILL-LIVE gaps (restated): ~12/61 variants declare no `encoding:` → metadata-wins compat
  branch leaves them as cascade-exposed as before (the recommended pre-flight hard gate is
  unbuilt); `hexo_rl/eval/checkpoint_loader.py` has NO gate at all (confirmed, not partial) on
  the eval/promotion path.

## §D-STRIX — comparative audit vs hexo-strix + landscape read (read-only, parallel to v3) — 2026-07-02

Dispatcher: S1 economics diff ∥ S4 play-UI spec → S2 portability (gated on S1) → S3 kernel verdict
(no work) → REVIEW (fresh) → RED-TEAM (distinct). Deliverables (each carries appended `## REVIEW`
+ `## RED-TEAM` sections): `docs/handoffs/d_strix_s1_economics_diff.md`,
`d_strix_s2_portability.md`, `d_strix_s4_play_ui_spec.md`, `d_strix_dispatcher_report.md`.
hexo-strix clone was sandbox-only (scratchpad), read-only, nothing vendored.

**Headline: the "days-from-0" premise EVAPORATED as a verified claim.** Their repo
(github.com/SootyOwl/hexo-strix @ 1b8ae4d, 17 commits, ~2h-old history) is a code+config-only
export — zero committed run logs, eval results, checkpoints, or throughput numbers; their own
config comments show a mixed record (`jk-long: collapsed at S3`, layerscale "regressed, loss
+20%"); the one positive ("beats sealbot 5s") is an unquantified code comment. Days-to-parity
formally NOT comparable anyway (their SealBot gate is time-limited/machine-speed-dependent vs our
fixed-depth-5 probe). Iteration-efficiency gap = config-implied, unconfirmed either direction.
Cheapest re-test: ask author to run their own `hexo-a0 eval-sealbot` and share output.

**Recipe diff (all MEASURED):** 222,146 vs 4,254,283 params (19.1×; production count confirmed
by instantiation — the pre-briefed "2.9M" matches an older smaller-trunk config res10/f112, NOT
aux heads); self-play sims 128 vs 400 (3×); 5-stage win-length+radius curriculum vs none; 2 vs 8
loss heads; buffer 250K vs 500K–1M; train==deploy Gumbel vs our PUCT-train/Gumbel-deploy.

**Portability verdicts (post REVIEW + RED-TEAM):**
- Radius curriculum: PORTABLE→**TESTABLE-CHEAP** (red-team downgrade). Radius scheduling already
  wired + live (`legal_move_radius_schedule`, vast.yaml lineage ran past its 50k transition) but
  no diagnostic ever isolated the transition's effect; §167 v7full argmax eval shows WR RISING
  with radius (6.5→15%) — open tension with small-early; L9 jitter protection unverified for
  radius-driven narrowing; vast.yaml's 5→8 endpoints are 25×25-calibrated, must re-derive for
  19×19 (v6_live2_golong precedent). win_length axis OUT (hardcoded Rust const, bench-gated).
  Spec in S2 doc; single-variable arm only, composes post-v3, never bundled with solver/seeding.
- Tiny net: TESTABLE-CHEAP, SURVIVES red-team. 2×2 width×aux-heads economics probe spec in S2
  doc; precondition added: verify launch anchor single-window-clean (F1) before comparing arms.
- Axis-graph line-topology input: NOTE-ONLY banked representation card (restart-gated; adjacent
  to D-FULLSPEC ENTANGLED_R but frozen-trunk context does not formally falsify from-scratch).
- No-window simplicity: confirmation of already-actioned F1/D-DECODE work.
- Train==deploy search: NOTE-ONLY; §D-GUMBELSIMS stands at GUMBEL-SIMS-NULL/affordability-PARITY
  (REVIEW corrected S2's stale 15k cite); open lever = matched-total-sim (m,n) sweep.
- Custom kernel: REJECT, banked to falsified register (row above) — GNN ragged-batching problem
  we don't have; K-cluster varies batch COUNT not tensor SHAPE.

**Play-UI (S4):** hero.did.science = DNS-dead; real site hexo.did.science is multiplayer-only, no
checkpoint plug-in hook → NOT zero-build. Full `hexo-play/` separate-repo spec delivered (FastAPI
+ JS hex board, compound-turn click-click-confirm, DeployHeadBot + SolverBackupBot via existing
eval path, SQLite log). Red-team: `/viewer` (09_VIEWER_SPEC, complete 2026-04-03) already covers
play-vs-model locally; hexo-play's real deltas are persistence + public deploy; realistic effort
1.5–2 days with admin/upload scope, ~1 day without. Operator picks.

**Pre-registration scored:** curriculum-as-big-driver WEAKENED (no verified outcome on either
side); tiny-net probe CONFIRMED worth specing; axis-graph banked as predicted; kernel REJECT as
predicted; landscape-panic-unwarranted HELD and STRENGTHENED (their verified record is thinner
than briefed).

**CORRECTION 2026-07-03 (§D-EVALGATE G2):** the line above previously read "...small-sample
caveat applies to operator's own wins over their bot too, n undocumented" — VOID. No head-to-head
vs hexo-strix (or its bot) exists at all; strength ordering vs hexo-strix = UNKNOWN in both
directions, not an undocumented-n result.

### §D-STRIX axis-graph re-adjudication (2026-07-13, D-L WP2) — representation card RE-OPENED (probe-gated)

Full re-proposal record: `docs/designs/gnn_readjudication.md` (the sanctioned path; register
discipline). Two dispositions, kept distinct:

- **CUDA kernel (falsified-register row above, §D-STRIX S3): UNCHANGED.** Still correct — HeXO's
  dense CNN+attention has no ragged-batching problem; the D-K tournament does not touch that
  premise. A scope footnote was added to the register row so it cannot be misread as an
  architecture kill.
- **Axis-graph line-topology REPRESENTATION card (banked NOTE-ONLY / restart-gated above): status
  advanced → `RE-OPENED pending D-L WP3 probe`.** The 2026-07-02 bank explicitly did NOT falsify
  it ("frozen-trunk context does not formally falsify from-scratch"); it deferred on absence of a
  measured strength read. That blocker is now removed by the D-K bridge tournament:
  strix-g128 **#1 deploy at +313 Elo** (`reports/tourney/TOURNAMENT.md`); strix-**raw** **#1 at
  +121 Elo**, a **+229 raw gap** over mantis-261k-raw (`argmax/ARGMAX_FINAL.md`, after the
  turn-assembly fidelity fix re-verified 18/18); **222,146 params**, 33.8 ms/turn. A large share
  of strix's strength lives in the net, not only search.

The re-open asserts only that the question now warrants a discriminating probe — NOT that the
architecture works. Discriminator = D-L WP3 axis-graph BC-prefit (GNN-BC vs CNN-BC, matched;
`docs/designs/gnn_bc_probe_design.md`), frozen verdicts ARCH-DOMINANT / ARCH-NULL / MIXED. run3's
primary-variable decision is deferred to the D-L convene (after WP1 + WP3); this addendum does not
change it. Nothing deleted here; history preserved.

## §D-EVALGATE — eval-side encoding-inference gate (G1) + record corrections (G2) + head-to-head protocol (G3) — 2026-07-03

**Why:** F1 proved the zero-gate encoding-inference class burned the d1m lineage; trainer-side landed, but the SAME class was open on the eval side (`checkpoint_loader.py` + call sites) — the exact instruments (SealBot eval, exploit_probe, promotion-eval) about to read the v3 arms. Gate required BEFORE any v3 verdict read.

**G1 build (2 waves: impl → fresh review BLOCK + red-team → fix wave).**
Core design — TWO semantics, split after review proved the naive trainer-port broke the sanctioned D-DECODE workflow:
- `declared_encoding` = assertion "ckpt IS X" → mismatch vs any stamp RAISES (`DeclaredEncodingMismatchError`). d1m shape pinned: stamped `v6_live2` + declared `v6_live2_ls` → raise.
- `decode_override` (new) = deliberate decode-time cross-decode (D-DECODE lever: same net, multi-window decode) → ALWAYS wins, NEVER silent: structured `encoding_decode_override` warning naming ckpt_stamp + decode_as. Both kwargs together → ValueError.
- Stamp resolution: 3 sources reconciled (metadata['encoding_name'] → config['encoding'] → top-level raw['encoding'] — the last was a priority-1 gate BYPASS pre-fix); internal stamp disagreement → raise (corrupt ckpt). Stamp authoritative over filename/shape sniff (sniff = last resort only). Malformed values raise symmetrically.
- Wired: exploit_probe + run_sealbot_eval (`--encoding`→decode_override, NEW `--expect-encoding`→assertion for v3 reads), round_robin `_CachedModelBot` (+`expect_encoding`), **promotion-floor anchor loader** `eval_pipeline._load_anchor_model` (red-team CRITICAL: fed `wr_bootstrap_anchor` which AND-combines into the live promotion boolean, default-enabled — was fully ungated; now config-declared→assert, else `require_encoding_source=True`), run_a1_solver_backup + run_z2_standalone_ladder (decode_override; Z2 dead labels now emitted as `decode_label`), gumbel_greedy_bot (hand-rolled silent resolver replaced with shared gated helpers).
- Deep-merge trap AVOIDED (helper-agent find): `configs/eval.yaml` deep-merges into variants → a base anchor `encoding:` key would inherit into ~15 path-only-override variants = crash regression. Base deliberately carries NO encoding key (warning comment added); regression test pins it.
- Tests: `tests/test_checkpoint_loader_encoding_gate.py` 32/32; eval sweep 155 passed/6 skipped + 119 passed/9 skipped consumer supplement; 1 pre-existing unrelated failure (offwindow OPPONENTS order) reproduces on stash.

**Acceptance (byte-identical baselines through gated loaders): HOLDS where it must.** Empirical old-vs-new resolution on local ckpts: both v2 flip-baseline ckpts (`reports/vast_ws3_v2/runs/{ws3_ctrl,ws3_z2_l1}/checkpoint_00208000.pt`) SAME (`v6_live2`); 16/17 red-team-sampled ckpts SAME. ONE intended drift: `checkpoints/ws3v3_warmstart_200k.pt` old=`v6_live2` new=`v6_live2_ls` (multi_window true) — this is the deliberately re-stamped LS warm-start the OLD eval loader was mislabeling (F1's exact bug class); drift = the correction working, documented in the module docstring (naive "byte-for-byte" claim deleted).

**GATE STATUS: v3 ON-arm reads may proceed** through `--expect-encoding v6_live2_ls` on exploit_probe/run_sealbot_eval (assertion raises on any mis-stamped arm ckpt). Recommended: v3 runbook reads use `--expect-encoding`, not bare `--encoding`.

**Operational flags (operator):**
1. Local `bootstrap_model_*.pt` anchors are unstamped bare state_dicts + `bootstrap_anchor.enabled: true` is the eval.yaml default → next floor-enabled run hard-fails at anchor load (intended loud-fail). Fix BEFORE next vast promotion run: per-variant `opponents.bootstrap_anchor.encoding:` key, or stamped COPIES (do NOT mutate SHA-pinned originals — `bootstrap_model_v6.pt` SHA is pinned in CLAUDE.md §S178).
2. Re-run ws3v3 arm instruments against vast-side `ws3v3_arm_*` ckpts with `--expect-encoding v6_live2_ls` to confirm clean pass (ckpts not local).

**Residuals (open, non-gating):** 12/61 no-encoding-variant preflight not enforced at variant level (mechanism `require_encoding_source` now live at anchor loader only); non-anchor candidate-side pipeline loads + sniff-default paths (round_robin/sealbot with no flags) still sniff when unstamped; `viewer/model_loader.py` + `bots/our_model_bot.py` = different class (pure shape inference, no name resolution — viewer will render `v6_live2_ls` ckpts as v6 until fixed).

**G2 (record corrections, propagated):** beats-claim VOID — no head-to-head vs hexo-strix ("tyto-bot" = hexo-strix, SootyOwl's bot; no literal "tyto" in repo) exists either direction; CORRECTION notes added in d_strix_dispatcher_report, d_strix_s1_economics_diff, §D-STRIX above, memory. Param count (4.25M) + URL facts (hero.did.science DNS-dead / hexo.did.science multiplayer-only) verified ALREADY correct everywhere — no edits needed.

**G3:** `docs/handoffs/d_strix_head_to_head_protocol.md` — pre-registered match protocol (n=40, 20/side strict alternation, distinct openings mandatory per §D-ARGMAX effective-n, deduped-bootstrap CI, disconnects excluded-and-logged) + artifact-ask template + claim-scope pinned BEFORE play: site cannot enforce compute → default reading = "deployment beats deployment", net-strength claim only if disclosed budgets match; 25-75% WR = INDETERMINATE (not parity). Operator executes socially.

### §D-EVALGATE A1 amendment executed (2026-07-03) — instrument re-baseline VERDICT: MATCH; probe fixture trap found+closed

A0 landed (evalgate wave + 20 per-variant `opponents.bootstrap_anchor.encoding:` declarations, operational flag 1 CLOSED; backlog tree committed in 15 grouped commits, suite green after un-staling 2 pre-existing pins).

**RED-TEAM provenance (now on record, was missing):** original flip baselines 4/31=12.9% / 24/125=19.2% were measured on the RAW `checkpoint_00200000.pt` (stale stamp `v6_live2`) via `gumbel_greedy_bot._build_engine` — blind `torch.load`, stamp NEVER consulted — decode forced by bare `--encoding v6_live2_ls`, host vast 5080. Decode geometry was already `_ls` → the old-loader mislabel never touched the flip DECODE, only the label. Strip `ws3v3_warmstart_200k.pt` weights verified byte-identical (per-tensor `torch.equal`) to the raw 200k.

**A1 re-measure (laptop, gated `--expect-encoding v6_live2_ls`, strip):** 3/31=9.7% / 22/125=17.6% — both inside pre-reg bands [2,6]/[19,29] → **MATCH, old numbers stand**. Attribution discriminator: OLD instrument on the SAME host = byte-identical 3/31, 22/125 → instrument correction is a **no-op on flip**; residual −1/−2 vs vast = host fp (scattered per-trap class flips, both directions). NEW **HOST-MATCH rule** in runbook §0.6: arm reads compare only against a same-host strip self-baseline (vast: re-run once, expect ≈4/31, 24/125).

**Probe C1–C3:** anchor numbers reproduce **byte-exact** through the gated loader (C1 +7.310/C2 30%/C3 40%, C4 Δ=0.000) on the `v6_live2` fixture. TRAP found: on `_ls`-stamped ckpts (strip + every v3 arm ckpt) the probe auto-detect now resolves `_ls` (stamp-aware fix), and the implicit fixture fallback silently swapped to the DEFAULT v6 fixture → manufactured C1 +10.97/C2 45%/C3 75% (different position set, would have PASSed). Fallback now hard-fails; `_ls`-stamped reads pass `--positions tests/fixtures/threat_probe_positions_v6_live2.npz` explicitly (valid: shape-identical family). Runbook §1(f)/§3/§4/§5 commands amended; `--expect-encoding` added to `run_l1_trapflip_smoke.py` (summary JSON records loader semantics).

**A2 STATUS: control arm may launch** per runbook §3 as amended (gated reads everywhere; first gated vast arm read doubles as the loader's vast confirmation).

### §D-PRELAUNCH (2026-07-04/05) — D-WS3V3 v3 smoke VERDICT: THIN-STILL; run2_mw_fresh LAUNCHED seeding-OFF

**P0/P1 (all landed, commits `20146be..9f8753c`):** run2 spec+yaml committed; B1 `--synthesize-metadata` mint (`run2_bootstrap_v6_live2_ls.pt`, weights-sha = pinned `ebf2ed39…`); `_ls` corpus regenerated (`export_corpus_npz.py --encoding v6_live2_ls`, per-cluster-row scatter, 610,954 rows, +22.7% previously-zeroed containing-window mass, sha `3813edc2…`); anchor corrective AMENDED — raw-bootstrap cp passes the sha pin but hard-fails `resolve_anchor` on encoding (observed live twice); OQ10 offwindow_adversary ENABLED stride-1; OQ8 widen-early signature gate (operator-procedural, EMA conv ≥0.85 slope ≤+0.02 2×25k reads n≥30); acceptance dry-run executed 2 real train steps + confirmed the D-1M C2 cosine/--iterations interaction live.

**v3 smoke (3 arms × 8000 steps @ 200k warmstart, vast 5080, ~$8 total):**
- **CONTROL r1 = canonical** (v3.1-CORRECTED): C1–C3 PASS improved (+9.671/30%/45%), comb-125 flat 23/24, reg-31 4→7 (+3, FIX1 first-true-multi-window-signal explanation — the strip is the F1 crippled-lineage net). deploy-disagree ABSOLUTE = base-rate drift (0.38 @2k steps; unusable in any unpaired form — ON-vs-CONTROL differential 0.31-0.36 sits BELOW either arm's drift-from-strip 0.42-0.48, i.e. common-regime-drift, not corruption).
- **SIGN-ERROR incident (register-grade):** trapflip `flip` = plays-the-SAVE (higher=better); a sign-inverted read manufactured an "off-window erosion" diagnosis → wasted r2 control re-run on `_ls` mixing corpus. r2 banked as accidental ablation: **`_ls`-corpus mixing in WARM-START regime regressed comb-125 conversion 23→17** (~1.5σ; yellow flag noted in run2 spec Decision 7; memory `trapflip-flip-semantics-sign-error`).
- **ARM-INJECT:** C1–C3 PASS, exploit floors 0.0/0.0 (deploy + modelplayer), fire 5-11% organic, conversion ≈ control (6/31, 21/125) → **injection lever no-effect** (matches FIX2a full-125 exact ceiling: 2/125 = 1.6% provable, Wilson [0.4%, 5.7%]).
- **ARM-SEEDED:** **NOT SEED-STARVED** — seeded-slice fire 19.95% = 3.2× organic 6.3% (floor 2×), nonzero 86% of intervals. Conversion STILL ≈ control/inject (7/31, 24/125; vs INJECT +3/125 noise). C1–C3 PASS, floors 0.0.
- **VERDICT: THIN-STILL** (density lever fired, taught nothing at `solver_visit_weight 0.5`). Pre-registered next step = ONE escalation to `visit_weight 1.0` before any MEMORIZES declaration — BANKED, does not gate run2. Machinery qualification (the real deliverable) BANKED: solver+counters live under deployed Gumbel (fire-rate reads via MONITORING SocketIO `training_step` stream, NOT the structlog JSONL — runbook §2 amended), encoding stamps clean, LR pin 1.0e-4 flat, per-arm buffer isolation, HOST-MATCH baselines byte-exact.
- **Instrument corrections landed:** v3.1 gate amendment (C1–C3 authoritative; disagree descriptive), anchor-incumbent re-stamp precondition (g2), fire-rate read location. Artifacts rsynced local (`reports/d_ws3v3/` + 4 ckpt dirs).

**run2_mw_fresh LAUNCHED 2026-07-05 (~16:50 UTC, vast 5080, $0.23/hr ≈ $84/15.3d):** seeding OFF per handshake (THIN-STILL ≠ GENERALIZES; verdict on crippled-lineage net = non-conclusive against mechanism; seeding banked as mid-run intervention if held-out trap-flip plateaus). Anchor = `_ls`-stamped mint; `_ls` mixing corpus (fresh-bootstrap regime — r2 warm-start yellow flag noted, first suspect if 5k/25k conversion gates read low); 5k probe gate = first mandatory stop.

### §D-PRELAUNCH SealBot-cost tuning (2026-07-06, run2 live-ops)

**Measured (vast + laptop):** the SealBot eval phase is the DOMINANT eval cost — 3.45h/100 games on vast (SealBot depth-5 ≈ 4s/move). `time_limit` is NOT the lever: depth-5 finishes in 0.6–6.9s (never hits 10s or 60s — 10s and 60s measured byte-identical); depth-6 uncapped is ~40s/move and depth-6@10s CAPS at 10.00s/move every move (2.65× depth-5 for a partial extra ply) → **depth-6 is a cost trap, kept at 5**.

**Changes (commit a95a70e):**
- Deploy SealBot arm DROPPED (`deploy_strength.sealbot_games: 0`; new knob, 0=off). It was reported-only — `sealbot_wr` is NOT in `decide_promotion` (gate = distinct-game bootstrap CI-lower > 0 AND wr_confirm ≥ 0.55 over the pooled screen+confirm vs-best games) — so dropping is promotion-neutral. Saves ~confirm_n SealBot games/deploy-round.
- Deploy `confirm_n` 200→100 (pooled 180 vs-best games; ample for a CI-lower gate).
- Normal-eval `sealbot.n_games` 100→50 (halves the 3.45h/round phase; Wilson95 hw ~14pp still a fine WR trajectory).
- Deploy stride kept at 4 (every 100k) — promotion granularity is a deliberate training lever (operator call).
- Projected runtime ~17d → **~13–14d**.

**MANUAL follow-up owed:** deploy no longer runs SealBot-vs-deploy-head. If the promotion trajectory or SealBot WR looks off at a milestone, run an OFFLINE SealBot-vs-deploy-head check (depth 5, or a one-off depth-6 rough read) — re-enable in-loop via `sealbot_games>0` only if it becomes load-bearing. SealBot WR (normal eval, every 25k, now n=50) remains the promotion-independent true-north.
