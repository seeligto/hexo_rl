# 20k `gumbel_targets` Reference-Run Characterization

Branch: `perf/investigation-2026-04-21`. Read-only Phase 2 pass — no code touched.

**Log**: `logs/train_c51d245de55c4a4bb39ac418397669bd.jsonl` (31,445 lines).
**Checkpoint**: `checkpoints/checkpoint_00020454.pt`.
**Parser**: stdlib `json` + `statistics` via `python3` (no pandas).

---

## 1. Run metadata

| Field | Value | Event source |
|---|---|---|
| `run_id` | `c51d245de55c4a4bb39ac418397669bd` | `run_id` |
| Start ts | 2026-04-20 19:10:40.xxxZ (logging_configured) | `logging_configured` |
| First `train_step` ts | 2026-04-20 19:11:23.430Z | `train_step` |
| Last `train_step` ts | 2026-04-21 07:09:05.607Z | `train_step` |
| `session_end` ts | 2026-04-21 07:09:37.654Z | `session_end` |
| `shutdown_requested` ts | 2026-04-21 07:09:30.087Z (Ctrl+C) | `shutdown_requested` |
| Wall-clock (start→session_end) | 43,137.7 s (11.98 h) | derived |
| `elapsed_sec` (from session_end) | 43,113.1 s | `session_end` |
| Train steps (logged `train_step` rows) | 8,734 (steps 11,721 → 20,454 inclusive) | `train_step` |
| `session_end.final_step` | 20,454 | `session_end` |
| `session_end.games_played` | 4,367 | `session_end` |
| Resume source | `checkpoints/checkpoint_00011720.pt` @ step 11,720 | `resumed` |
| Variant | `gumbel_targets` (inferred: `gumbel_mcts=false`, `completed_q_values=true`) | `startup.config.selfplay` |
| Network | 18-plane input, 12 × 128ch, SE, GroupNorm(8), FP16 | `startup.config` |
| MCTS | `n_simulations=400`, `c_puct=1.5`, Dirichlet on (α=0.3, ε=0.25) | `startup.config.mcts` |
| Playout cap | `full_search_prob=0.25`, `n_sims_quick=100`, `n_sims_full=600` | `startup.config.selfplay.playout_cap` |
| Workers | `n_workers=14`, `inference_batch_size=64`, `wait_ms=4.0` | `startup.config.selfplay` |
| Buffer | capacity 250K restored from `replay_buffer.bin` (`buffer_restored`) | `buffer_restored` |
| Corpus | 199,470 pos from `bootstrap_corpus.npz` (prefill_skipped — buffer already warm) | `corpus_prefill_skipped`, `corpus_loaded` |
| Host | laptop Ryzen 7 8845HS + RTX 4060 8GB (inferred: `vram_total_gb=8.586`, 14 workers per recent bench) | `gpu_stats`, `worker_pool_started` |
| `resume_anchor_step_mismatch` | `trainer_step=11720`, `best_model_step=0` — best_model.pt came from a different run | `resume_anchor_step_mismatch` |

**Note on resume anchor**: `best_model.pt` was at step 0 while the trainer resumed at 11,720. First eval at step 15,000 compared trained weights against an unrelated best-anchor — this affects `wr_best` interpretation but not throughput/perf metrics below.

---

## 2. Throughput

### Games/hour

| Source | Value |
|---|---|
| Counted `game_complete` events / total wall | **364.4 games/hr** |
| Counted `game_complete` / first→last `game_complete` span | 365.1 games/hr |
| `train_step_summary.games_per_hour` rolling | mean 466.9, p50 398.6, p95 786.8, stdev 161.2, min 295.5, max 1465.9 |

The summary field is a **rolling-window** estimate (window reset behaviour noted in `feedback_draw_rate_windowing`), so its `max=1465` is a short-window artefact. **The authoritative full-run figure is 364 games/hr** computed from raw event counts.

### Positions/hour (primary worker metric per `feedback_doc_conventions`)

Computed from `game_complete.plies` summed (total = 436,930 plies):

| Quantity | Value |
|---|---|
| Total plies (= positions pushed to buffer) | 436,930 |
| Over full wall (43,138 s) | **36,464 pos/hr** |
| Over game span (43,064 s) | 36,526 pos/hr |

Bench target (2026-04-18): ≥142,000 pos/hr **PROVISIONAL**. This run is **~26% of the bench target**, or **3.9× slower** than the `make bench` worker-pool figure of 164K pos/hr. The bench harness runs 200 sims/move with no training contention; this run runs 100 (quick) / 600 (full) with `full_search_prob=0.25` → expected sims/move ≈ 100·0.75 + 600·0.25 = 225 sims, ~1.1× the bench workload. The ~4× gap is therefore **not** the per-move sim delta — it's training-pool contention (see §4).

### MCTS throughput (`game_complete.sims_per_sec`)

| Stat | Value |
|---|---|
| n | 4,367 |
| mean | 3,324 sim/s |
| p50 | 3,655 |
| p95 | 3,933 |
| min | 19 (game with stall) |
| max | 10,599 |

Bench baseline (CPU-only, no NN, no contention): 69,680 sim/s. Self-play with NN-in-loop is **5.3% of CPU-only sim/s** — expected; every sim waits on NN inference. The p95 is the relevant ceiling: ~3,900 sim/s × 14 workers = ~55K aggregate sim/s during saturated periods.

---

## 3. Game length

From `game_complete` fields (`plies`, `game_length`):

| Stat | `plies` (raw half-moves) | `game_length` (compound moves) |
|---|---|---|
| n | 4,367 | 4,367 |
| mean | 100.05 | 50.35 |
| p50 | 63.0 | 32.0 |
| p95 | 200.0 | 100.0 |
| stdev | 72.74 | 36.17 |
| min | 11 | — |
| max | 200 | — |

`max_game_moves=200` hard cap hit by p95 — **≥5% of games truncate**. Mean plies 100 × 4,367 = 436,700 ≈ 436,930 total (matches §2).

`compound_move` ≈ `plies/2` (post-opening), confirming dual-move compound structure. Every position → 1 replay row (+ symmetry augmentation at sample time).

---

## 4. Train-step timing

8,733 Δt values between consecutive `train_step` events:

| Stat | Value |
|---|---|
| mean | 4,930.97 ms |
| p50 | 463.25 ms |
| p95 | 21,556.74 ms |
| stdev | 7,910.94 ms |
| min | 308.60 ms |
| max | 92,986.92 ms (likely eval-round wait) |

Sum of all Δt = 43,062 s ≈ 99.8% of wall. **This is the inversion signal**: mean ≫ median by 10× because the distribution is bimodal.

### Bimodal split (threshold: 1s)

| Bucket | n | % | Σ sec | % of wall |
|---|---|---|---|---|
| Fast (Δt <1s) — actual step cost | 4,724 | 54.1% | 1,949.5 s | **4.5%** |
| Slow (Δt ≥1s) — idle / waiting / eval | 4,009 | 45.9% | 41,112.7 s | **95.4%** |

Fast-bucket distribution (the **real** per-step GPU cost):

| Stat | Value |
|---|---|
| mean | 412.7 ms |
| p50 | 393.1 ms |
| p95 | 500.6 ms |
| p99 | 897.2 ms |

So **intrinsic train-step cost ≈ 400 ms** (fp16, bs=256, model 12×128, 18-plane input). At that cost, 8,734 steps should take ~58 min of GPU time. Observed: **11.98 h**. **The trainer spent ~95% of wall time idle**, waiting for the self-play pool to supply replay data or for eval rounds.

### Gaps (no-op periods)

| Threshold | Count | Σ sec | % of wall |
|---|---|---|---|
| Δt >5 s | 2,597 | 37,049 | 85.9 |
| Δt >10 s | 1,528 | 29,218 | 67.7 |
| Δt >60 s | 7 | 482 | 1.1 |

The 7 gaps >60 s line up with the two eval rounds (1,215 s and 1,454 s total — some split over multiple consecutive gaps) and long `waiting_for_games` stalls.

---

## 5. InferenceBatcher metrics

From `train_step_summary` (n=873). **No dedicated `inference_batch_summary` event exists in this run** — all batcher signal comes from the summary aggregation.

| Field | Value |
|---|---|
| `batch_fill_pct` mean | 93.93 |
| `batch_fill_pct` p50 | 94.20 |
| `batch_fill_pct` p95 | 94.30 |
| `batch_fill_pct` min | 73.30 |
| `batch_fill_pct` max | 94.40 |
| `inf_forward_count` total delta | 2,657,159 forwards |
| `inf_total_requests` total delta | 160,223,643 positions |
| **observed batch size (req/fwd)** mean | **59.78 pos/batch** |
| observed batch size median | 60.49 |

Configured `inference_batch_size=64`. Observed avg 59.78 / 64 = **93.4% fill** — matches `batch_fill_pct` field. **Batcher is well-fed** — this is not the bottleneck.

Forwards per second over run: 2,657,159 / 43,138 = **61.6 fwd/s** aggregate. Bench NN inference (batch=64): 7,646 pos/s → ~119 fwd/s of batch-64 capacity. So GPU inference utilization ≈ 52% of bench ceiling.

Workers (14) × sims per game × games: total sim count ≈ 160M inference requests + leaf batching → aggregate MCTS sim/s ≈ 3,710 (summary field). Bench MCTS (CPU-only) 69,680 sim/s ÷ per-move NN wait → expected in-training degradation.

---

## 6. Worker supply / demand

`waiting_for_games` events: **8,491** total, fired on a ~5s cadence (p50 Δt=5.01s, p95=5.44s). **4 of 8,491 ticks report `games=0`** (the rest show cumulative game count) — so this event **is not an idle flag**, it's a heartbeat with the running total.

**Cannot infer worker-idle from `waiting_for_games` directly** — the event fires unconditionally at the ~5s tick. The trainer-side idle signal is the **train_step Δt ≥1s** bucket (41,113 s, §4).

Gaps <10 s between consecutive `waiting_for_games` ticks sum to 43,048 s ≈ 99.8% of wall — confirming the heartbeat ran continuously (no dashboard crash).

**Implied worker supply gap**: trainer idle 41,113 s while 14 workers should produce positions continuously. At 36,464 pos/hr actual and `training_steps_per_game=2.0`, the trainer consumes ~728 positions per step × (8734 steps / 43,138 s) = ~177 pos/s. Workers produce 436,930 / 43,138 = **10.1 pos/s**. Deficit: **~167 pos/s of would-be consumption is instead idle wait time**.

Caveat: `training_steps_per_game=2.0` means the trainer targets 2 steps per new game (not per position). With 4,367 games and 8,734 steps = exactly 2.0 ratio — the trainer is **throttled by the game-production rate, not batch readiness**. The worker pool is the bottleneck.

---

## 7. GPU utilization

From `gpu_stats` (n=8,577, ~5s cadence, pynvml-sourced):

| Stat | Value |
|---|---|
| mean | 85.4% |
| p10 | 82.0% |
| p50 | 88.0% |
| p90 | 90.0% |
| min | 0.0% |
| max | 100.0% |

Cross-check from `train_step_summary.gpu_util` (n=873): mean 85.5, p10 81, p50 88, p90 95 — agrees.

**GPU held at ~85–88% during training bursts**. Given that the trainer spends 95% of wall idle (§4), this high number is pynvml reading inference-server activity from worker NN calls, **not** training forward/backward. The pynvml utility is a sampled integer and covers both consumers.

Target: ≥85% (met on paper). But given §4 shows trainer idle, the **effective GPU utilization for training is ~4.5% of wall** — the other 95% GPU time goes to self-play inference. This is the key finding.

---

## 8. VRAM

From `gpu_stats.vram_used_gb`:

| Stat | Value |
|---|---|
| mean | 5.03 GB |
| peak | 5.17 GB |
| total | 8.59 GB (RTX 4060) |
| utilization | 60.2% peak |

Cross-check `train_step_summary.vram_gb`: mean 5.04, peak 5.14 — agrees.

Bench target: ≤6.88 GB (80% of 8.6). **1.71 GB headroom** at peak. `mem_util_pct` (memory bandwidth) mean 54%, max 92% — bandwidth-bound bursts exist but not sustained.

---

## 9. SealBot evaluation

Two `evaluation_round_complete` events:

| Step | Games | wr_random | wr_best | wr_sealbot | CI_sealbot | Gate passed? | Elo est |
|---|---|---|---|---|---|---|---|
| 15,000 | 120 | 1.000 | 0.110 (CI 0.063–0.186) | — (not run) | — | false | +679.4 |
| 20,000 | 170 | 1.000 | 0.130 (CI 0.078–0.210) | **0.000** (CI 0.000–0.071) | false | false | −65.8 |

- SealBot gate ran only at step 20,000 (per `stride: 4`, 20K cadence — matches §101 H1 config).
- `wr_best=0.13` at step 20K is **far below** the `wr_best ≥ 0.55 AND ci_lo > 0.5` promotion bar. No graduation.
- Elo estimate swings from +679 (step 15K, random-only) to −65 (step 20K, including SealBot loss) — random-baseline Elo is uninformative when the model crushes random 100%.
- Eval wall-cost: 1,215.5 s (step 15K) + 1,454.2 s (step 20K) = 44.5 min of the 11.98 h run = **6.2% of wall time spent in eval**.

18 `checkpoint_saved` events at `checkpoint_interval=500` (configured) → 500-step cadence confirmed (2,457 s / 500 steps = ~4.9 s per step = matches training throughput).

---

## 10. Derived bottleneck ranking

1. **Worker pool supply** (primary): trainer idle 95% of wall, ~10.1 pos/s produced vs ~177 pos/s consumable.
2. **Game truncation**: ≥5% of games hit `max_game_moves=200` — each truncation costs ~200 plies of MCTS work with no terminal-value signal. Investigate whether relaxing the cap or detecting draws earlier helps.
3. **Eval cost**: 44.5 min / 12h = 6.2% of wall. Acceptable but non-trivial.
4. **Inference batcher**: well-fed at 94% fill — **not** a bottleneck.
5. **GPU compute**: 85–88% util reading, but misleading — the useful-work fraction for training is ~4.5% of wall.

---

## 11. Metrics that were NOT logged (Phase B probe targets)

These fields were **absent** or only indirectly inferrable — Phase B should add explicit events for each:

1. **Per-worker idle time / queue depth** — no `worker_idle`, `worker_busy`, `worker_waiting` events. Had to infer from train-step Δt gaps.
2. **InferenceBatcher wait-time distribution** — no `inference_batch_wait_ms` or per-batch latency percentiles. Only `batch_fill_pct` aggregate in `train_step_summary`.
3. **Queue depths** — no `results_queue_depth`, `inference_queue_depth`, `leaf_queue_depth`. Cannot tell which queue is the constraint.
4. **Per-game MCTS sim count breakdown** — `game_complete` has `sims_per_sec` but no `total_sims`, no `fast_sims_count` vs `full_sims_count` split. Cannot directly verify playout-cap behaviour per game.
5. **Per-step GPU fwd/bwd timing** — `train_step` has no `fwd_ms`, `bwd_ms`, `opt_ms`. Only coarse Δt between events, which is dominated by inter-step idle.
6. **Replay-buffer sample timing** — no `buffer_sample_ms` per batch. Given bench 1,654 µs/batch augmented, this should be cheap, but not confirmed in-flight.
7. **Per-worker game-production rate** — no per-worker `games_completed` or `worker_positions_per_sec`. Impossible to tell if one worker is stalling.
8. **Eval in-progress flag** — no `training_paused_for_eval` event. Had to diff `evaluation_start` against `evaluation_round_complete` timestamps (1215s + 1454s cost).
9. **CPU utilization / per-core** — no `cpu_util`, `cpu_freq_mhz`. Laptop-boost-clock drift (§98, §102 note) cannot be diagnosed in-flight.
10. **Board/tree memory** — no `tree_node_count`, `tt_entries`, `tt_hit_rate`. Transposition-table efficiency invisible.
11. **`waiting_for_games` is a heartbeat, not an idle signal** — needs a companion `worker_pool_stalled{n_idle_workers}` event to be actionable.
12. **Python GC pauses / allocation stats** — `tracemalloc_top10` fires 17 times but no per-allocation-site deltas; no `gc_pause_ms`.
13. **inference_batch_size_effective histogram** — only mean/fill-pct logged; no p10/p90 on actual batch size to catch small-batch stalls.
14. **MCTS leaf-batching fill** — `leaf_batch_size=8` configured; no event reports observed leaf-batch fill.
15. **Game-end reason** — `game_complete` has `winner` but no `termination_reason` (win / draw / truncation_cap). Crucial given ≥5% of games hit the 200-ply cap.

---

## 12. Event-source reference (for reproducibility)

| Metric | Event(s) used |
|---|---|
| Run metadata | `run_id`, `startup`, `resumed`, `worker_pool_started`, `selfplay_pool_started`, `buffer_restored`, `session_end`, `resume_anchor_step_mismatch` |
| Games/hr, pos/hr | `game_complete` (count + `plies` sum) |
| Game length | `game_complete.plies`, `game_complete.game_length` |
| Train-step timing | consecutive `train_step.timestamp` Δt |
| Loss trajectory | `train_step_summary.policy_loss`, `.value_loss` (first 1.8364 → last 1.4344 policy; 0.5654 → 0.5734 value) |
| Inference batcher | `train_step_summary.batch_fill_pct`, `.inf_forward_count`, `.inf_total_requests` |
| MCTS throughput | `game_complete.sims_per_sec` |
| GPU util / VRAM | `gpu_stats` (primary), `train_step_summary.gpu_util` / `.vram_gb` (cross-check) |
| Worker idle | `waiting_for_games` (heartbeat only), inferred from train-step Δt |
| Eval | `evaluation_start`, `evaluation_games_start`, `evaluation_games_complete`, `evaluation_round_complete` |
| Checkpoints | `checkpoint_saved` (n=18, 500-step cadence) |

Fast-bucket per-step cost (400ms @ bs=256, fp16, 12×128) is the reference number for §Phase D VRAM-headroom analysis.
