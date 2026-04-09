# Architecture Review — April 2026

**Date:** 2026-04-07
**Scope:** Full codebase audit (Rust engine + Python training/selfplay/monitoring/eval/bootstrap + configs + docs)
**Method:** 8 parallel subagent audits, findings deduplicated and verified by main agent
**Constraints:** READ-ONLY. No edits, no test runs, no benchmarks executed. Sprint log decisions not re-litigated.

---

## Executive summary

1. **CRITICAL: `completed_q_values` KL loss path is silently dead.** `trainer.py:305` looks up `self.config.get("selfplay", {}).get("completed_q_values", False)` but `self.config` is the flattened `combined_config` dict (no nested `"selfplay"` key). The lookup always returns `False`, so training uses cross-entropy loss instead of the intended KL loss with Gumbel completed-Q targets. Every training step since `completed_q_values: true` was set has been silently ignoring it.

2. **HIGH: `gumbel_mcts: true` in live config contradicts documented default.** `configs/selfplay.yaml:38` enables experimental Gumbel Sequential Halving root search. Sprint log §62 and CLAUDE.md both state the default is `false`. Three independent subagents flagged this. The sustained Phase 4.0 exit-criterion run would use an undocumented algorithm variant.

3. **HIGH: ~12 dead files/classes accumulating.** `MetricsWriter`, `GameReplayPoller`, `opening_classifier.py`, `game_record_parser.py`, `HexoComScraper`, `KrakenBotBot`, `BootstrapDataset`, `Formation` enum, `submit_request_and_wait_rust()`, `hexo_rl/api/` — all have zero inbound callers.

4. **MEDIUM: Feature buffer pool drain in game_runner.** `get_feature_buffer()` buffers used for game records are never returned to the pool, causing steady-state 26 KB/move heap allocations after pool exhaustion (512 slots).

5. **MEDIUM: 7 dead config keys in `monitoring.yaml`** (`win_rate_window`, `game_length_window`, `training_step_history`, `game_history`, `viewer_max_memory_games`, `viewer_max_disk_games`, `capture_game_detail`) — documented in sprint log but never wired into code.

6. **MEDIUM: `eval_interval` dual definition** with misleading precedence comment. `eval.yaml` takes precedence when eval pipeline is enabled, not `training.yaml` as the comment claims.

7. **MEDIUM: Stale temperature schedule in `docs/01_architecture.md`** — still shows hard-step `1.0 if ply < 30 else 0.1`, replaced by cosine annealing at sprint §36.

8. **LOW: Cosine LR scheduler T_max defaults to hardcoded 50,000** when `total_steps` is absent from config. `min_lr` defaults to `1e-5`. Neither is in `training.yaml`.

**Top 3 recommended actions:**

| # | Action | Effort | Impact |
|---|---|---|---|
| 1 | Fix `completed_q_values` lookup (`trainer.py:305`) — change to flat key lookup | 5 min | Enables KL policy loss as intended; potentially significant training quality improvement |
| 2 | Set `gumbel_mcts: false` in `selfplay.yaml` (or document the decision) | 1 min | Restores documented baseline for Phase 4.0 exit run |
| 3 | Delete ~12 dead files/classes in one cleanup commit | 30 min | Removes ~600 lines of unreachable code, eliminates phantom deps (wandb, tqdm) |

---

## Findings by severity

### CRITICAL

**C1. `completed_q_values` KL loss path silently dead**
- **File:line:** `hexo_rl/training/trainer.py:305`
- **Category:** config_drift
- **Description:** `use_kl = bool(self.config.get("selfplay", {}).get("completed_q_values", False))` performs a nested lookup into a key that doesn't exist in the flattened `combined_config` dict passed by `scripts/train.py:233`. The `"selfplay"` section is spread into the top level via `**self_config` at `train.py:182`, so `completed_q_values` is a top-level key. The nested lookup always returns `False`.
- **Proposed action:** Change to `use_kl = bool(self.config.get("completed_q_values", False))`. Add a regression test.
- **Risk:** Low (one-line fix, no architectural change)
- **Requires bench:** No (loss function change — validate via training log inspection, not perf bench)
- **Source:** Subagent 4 (model + training)

---

### HIGH

**H1. `gumbel_mcts: true` contradicts documented default**
- **File:line:** `configs/selfplay.yaml:38`
- **Category:** config_drift
- **Description:** Sprint log §62: "Config defaults restored: `gumbel_mcts: false`". CLAUDE.md: "`gumbel_mcts: false` by default". Current config has `true`. Enables Sequential Halving + Gumbel-Top-k root search for all workers.
- **Proposed action:** Set `gumbel_mcts: false` or add sprint log entry documenting the deliberate change.
- **Risk:** Low (config-only)
- **Requires bench:** Yes if kept enabled (`make bench.full` — worker throughput target)
- **Source:** Subagents 1, 2, 7

**H2. `MetricsWriter` — dead class, phantom wandb dep**
- **File:line:** `hexo_rl/monitoring/metrics_writer.py:18`
- **Category:** dead_code
- **Description:** Zero inbound imports anywhere. References `wandb` which is not in `requirements.txt`.
- **Proposed action:** Delete file.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagents 6, 8

**H3. `GameReplayPoller` — dead class with full daemon infrastructure**
- **File:line:** `hexo_rl/monitoring/replay_poller.py:41`
- **Category:** dead_code
- **Description:** Zero inbound imports. Full daemon thread, LRU cache, byte-offset tracking — never started.
- **Proposed action:** Delete file.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagents 6, 8

**H4. `game_record_parser.py` — entire module dead**
- **File:line:** `hexo_rl/opening_book/game_record_parser.py:1`
- **Category:** dead_code
- **Description:** Zero inbound imports from any file. `opening_book/` package is completely disconnected.
- **Proposed action:** Delete `hexo_rl/opening_book/` directory.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 8

**H5. `KrakenBotBot` — dead + BotProtocol signature mismatch**
- **File:line:** `hexo_rl/bootstrap/bots/krakenbot_bot.py:58`
- **Category:** dead_code
- **Description:** Zero callers. Takes `get_move(state, rust_board)` — two args instead of `BotProtocol`'s one-arg contract. Would TypeError if used.
- **Proposed action:** Fix signature to match BotProtocol or delete until properly integrated.
- **Risk:** None for deletion
- **Requires bench:** No
- **Source:** Subagent 8

**H6. `submit_request_and_wait_rust()` — dead Rust method, impl-block suppress**
- **File:line:** `engine/src/inference_bridge.rs:156`
- **Category:** dead_code
- **Description:** Only `submit_batch_and_wait_rust()` is called from `game_runner.rs`. The single-request variant is unreachable. `#[allow(dead_code)]` on the impl block masks this.
- **Proposed action:** Remove dead method; narrow `#[allow(dead_code)]` to specific methods if needed.
- **Risk:** Low (verify with `cargo test`)
- **Requires bench:** No
- **Source:** Subagent 8

**H7. `InferenceServer.submit_and_wait()` uses `.tolist()` at PyO3 boundary**
- **File:line:** `hexo_rl/selfplay/inference_server.py:65`
- **Category:** boundary
- **Description:** Converts numpy array to Python list of floats before passing to Rust. Full Python-object allocation (6498 elements). Used by `our_model_bot.py` and tests, not the live SelfPlayRunner path.
- **Proposed action:** Accept contiguous float32 ndarray via PyO3 `PyReadonlyArray`.
- **Risk:** Low
- **Requires bench:** Yes (`make bench` after fix)
- **Source:** Subagent 5

---

### MEDIUM

**M1. Feature buffer pool drain in game_runner**
- **File:line:** `engine/src/game_runner.rs:587`
- **Category:** perf_hot
- **Description:** `get_feature_buffer()` buffers used for game records are consumed but never returned to the 512-slot pool. After exhaustion, every subsequent call allocates a fresh `vec![0.0f32; 6498]` (26 KB) per move per cluster.
- **Proposed action:** Copy record data to a scratch buffer, then return the feature buffer to the pool immediately.
- **Risk:** Medium (touches hot path allocation pattern)
- **Requires bench:** Yes (`make bench.full` — worker throughput)
- **Source:** Subagent 2

**M2. TT policy clone allocates 1.4 KB per hit**
- **File:line:** `engine/src/mcts/selection.rs:140`
- **Category:** perf_hot
- **Description:** `entry.policy.clone()` allocates a new `Vec<f32>` (362 elements) on every transposition table hit in the MCTS selection loop. Unavoidable with current ownership model.
- **Proposed action:** Consider `Arc<[f32]>` for reference-counted sharing. Track TT hit rate in real self-play first.
- **Risk:** Medium (changes TT memory layout)
- **Requires bench:** Yes (`make bench.full`)
- **Source:** Subagents 1, 3

**M3. 7 dead monitoring config keys**
- **File:line:** `configs/monitoring.yaml:9-24`
- **Category:** dead_code
- **Description:** `win_rate_window`, `game_length_window`, `training_step_history`, `game_history`, `viewer_max_memory_games`, `viewer_max_disk_games`, `capture_game_detail` — all unread by any Python code. Sprint log §63 says they're served via `/api/monitoring-config` but the handler doesn't include them.
- **Proposed action:** Wire into web dashboard response or delete per dead-config policy.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 7

**M4. `eval_interval` dual definition with wrong precedence comment**
- **File:line:** `configs/eval.yaml:5` vs `configs/training.yaml:16`
- **Category:** config_drift
- **Description:** Both files define `eval_interval: 5000`. `eval.yaml` comment says "training.yaml takes precedence" but `train.py:471` reads from `eval.yaml` first when eval pipeline is enabled. Additionally, `train.py:555` re-assigns `eval_interval` from `train_cfg` unconditionally, overwriting the eval pipeline's value.
- **Proposed action:** Remove from `eval.yaml`; read only from `training.yaml`.
- **Risk:** Low
- **Requires bench:** No
- **Source:** Subagent 7

**M5. Stale temperature schedule in architecture doc**
- **File:line:** `docs/01_architecture.md:189-196`
- **Category:** clarity
- **Description:** Shows old hard-step schedule (`1.0 if ply < 30 else 0.1`), replaced by cosine annealing at sprint §36.
- **Proposed action:** Update to cosine formula with current config keys.
- **Risk:** None (doc-only)
- **Requires bench:** No
- **Source:** Subagent 7

**M6. `sims_per_sec` metric overcounts by ~23%**
- **File:line:** `hexo_rl/selfplay/pool.py:214`
- **Category:** perf_cold
- **Description:** Uses `n_simulations` (= `standard_sims`, 200) for all games. With `fast_prob=0.25` and `fast_sims=50`, actual average is 162.5. Dashboard figure is inflated.
- **Proposed action:** Compute weighted average: `avg_sims = fast_prob * fast_sims + (1-fast_prob) * standard_sims`.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 5

**M7. Legacy `SelfPlayWorker` hardcodes different playout cap**
- **File:line:** `hexo_rl/selfplay/worker.py:203,210`
- **Category:** config_drift
- **Description:** 90%/10% fast/standard split with `fast_sims ~ U[15,26)` vs production config's 25%/75% split with `fast_sims=50`. Used by `benchmark_mcts.py` and `our_model_bot.py`.
- **Proposed action:** Read from config or add deprecation warning. At minimum update docstring.
- **Risk:** Low
- **Requires bench:** No
- **Source:** Subagent 5

**M8. `Formation` enum and `has_forced_win()` — dead Rust code**
- **File:line:** `engine/src/formations/mod.rs:5`
- **Category:** dead_code
- **Description:** ~55 lines compiled into every build. Zero call sites. Forced-win short-circuit was deliberately removed (sprint log).
- **Proposed action:** Delete module or add targeted `#[allow(dead_code)]` if keeping as future hook.
- **Risk:** Low
- **Requires bench:** No
- **Source:** Subagent 8

**M9. `BootstrapDataset` class — superseded by `AugmentedBootstrapDataset`**
- **File:line:** `hexo_rl/bootstrap/dataset.py:121`
- **Category:** dead_code
- **Description:** Zero inbound imports. Functional replacement lives in `pretrain.py:139`.
- **Proposed action:** Delete class. Keep `replay_game_to_triples()` which is still used.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 8

**M10. Policy projection semantic divergence**
- **File:line:** `hexo_rl/selfplay/policy_projection.py:8-36` vs `hexo_rl/selfplay/inference.py:78-103`
- **Category:** redundancy
- **Description:** Two coordinate-transform implementations with different semantics (forward vs reverse, renorm vs max-pool). Legacy `SelfPlayWorker` uses both for the same position — MCTS search policy and buffer target are computed by different functions.
- **Proposed action:** Document divergence explicitly. Consider consolidating if legacy path is retained.
- **Risk:** Low (only affects legacy path)
- **Requires bench:** No
- **Source:** Subagent 5

**M11. Test uses wrong `decay_steps` constant**
- **File:line:** `tests/test_phase4_smoke.py:70,170`
- **Category:** test_gap
- **Description:** `decay_steps = 1_000_000.0` hardcoded in two tests. Production config is `300_000`. Tests validate the formula at the wrong constant, giving false confidence.
- **Proposed action:** Read from canonical config or use production constant.
- **Risk:** None (test-only)
- **Requires bench:** No
- **Source:** Subagent 5

**M12. No test coverage for ownership/threat aux heads**
- **File:line:** `hexo_rl/model/network.py:201-205`, `hexo_rl/training/trainer.py:329-338`
- **Category:** test_gap
- **Description:** Both heads are active in production (`ownership_weight: 0.1`, `threat_weight: 0.1`) but have zero test coverage in `test_network.py` or `test_trainer.py`.
- **Proposed action:** Add forward-pass shape tests and loss-computation tests.
- **Risk:** None (test-only)
- **Requires bench:** No
- **Source:** Subagent 4

**M13. Multi-axis threat overlap untested**
- **File:line:** `engine/src/board/threats.rs` (test suite)
- **Category:** test_gap
- **Description:** No test verifies that when two axes produce threats at the same (q,r) cell, the `best` accumulator takes the maximum level.
- **Proposed action:** Add `test_multi_axis_threat_takes_max_level`.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 3

**M14. `threats.rs` uses `std::collections::HashMap` instead of `FxHashMap`**
- **File:line:** `engine/src/board/threats.rs:58`, `engine/src/lib.rs:192`
- **Category:** perf_cold
- **Description:** Viewer-only path but inconsistent with the rest of the engine which uses FxHashMap throughout. Redundant HashMap allocation in `lib.rs:192` converting cells.
- **Proposed action:** Switch to FxHashMap. Pass `&self.inner.cells` directly.
- **Risk:** Low (viewer path only)
- **Requires bench:** No
- **Source:** Subagent 3

---

### LOW

**L1. `torch_compile` default `True` in pretrain.py**
- **File:line:** `hexo_rl/bootstrap/pretrain.py:656`
- **Category:** config_drift
- **Description:** `config.get("torch_compile", True)` — opposite of the canonical `False` everywhere else (§32).
- **Proposed action:** Change default to `False`.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 4

**L2. Cosine scheduler T_max hardcoded fallback**
- **File:line:** `hexo_rl/training/trainer.py:145`
- **Category:** config_drift
- **Description:** Falls back to `50_000` when `total_steps` absent from config. `min_lr` falls back to `1e-5`. Neither key is in `training.yaml`.
- **Proposed action:** Add `total_steps` and `min_lr` to `configs/training.yaml`. Consider raising error if `lr_schedule: cosine` but no T_max specified.
- **Risk:** Low
- **Requires bench:** No
- **Source:** Subagent 4

**L3. `HexoComScraper` — dead stub**
- **File:line:** `hexo_rl/bootstrap/scraper.py:141`
- **Category:** dead_code
- **Description:** `fetch_games_list()` returns `[]`. Never instantiated.
- **Proposed action:** Delete.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagents 6, 8

**L4. `hexo_rl/api/` — empty placeholder package**
- **File:line:** `hexo_rl/api/__init__.py:1`
- **Category:** dead_code
- **Description:** Single 1-line `__init__.py`. Zero imports anywhere.
- **Proposed action:** Delete.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 8

**L5. `opening_classifier.py` — test-only, no production caller**
- **File:line:** `hexo_rl/bootstrap/opening_classifier.py:1`
- **Category:** dead_code
- **Description:** Only imported by its own test. No corpus or eval pipeline references it.
- **Proposed action:** Delete with its test, or wire into corpus pipeline if classification is valuable.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagents 6, 8

**L6. `emit_queue_maxsize` not in config file**
- **File:line:** `configs/monitoring.yaml` (absent)
- **Category:** config_drift
- **Description:** Read by `web_dashboard.py:54` with default 200. Sprint log §55 documents it as a config key. Never added to YAML.
- **Proposed action:** Add to `monitoring.yaml`.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 7

**L7. `long_run_balanced.yaml` dead keys and undocumented model reduction**
- **File:line:** `configs/long_run_balanced.yaml:5,23-24`
- **Category:** config_drift
- **Description:** `res_blocks: 10` (vs production 12) undocumented. `min_lr: 0.0001` and `total_steps: 50000` are never read.
- **Proposed action:** Wire `min_lr`/`total_steps` into trainer or delete. Document the 10-block deviation.
- **Risk:** Low
- **Requires bench:** Yes if keeping 10-block model
- **Source:** Subagent 7

**L8. Stale replay buffer throughput in architecture doc**
- **File:line:** `docs/01_architecture.md:268`
- **Category:** clarity
- **Description:** Shows "219,444 pushes/sec" from 2026-03-31. Current baseline is 762,130 pos/sec (3.5x stale).
- **Proposed action:** Update to current baseline or cross-reference CLAUDE.md.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 7

**L9. ZOI config key names in sprint log don't match implementation**
- **File:line:** `docs/07_PHASE4_SPRINT_LOG.md:129`
- **Category:** clarity
- **Description:** Sprint log says `zoi_radius`, `zoi_history`, `zoi_min_candidates`. Actual keys are `zoi_margin`, `zoi_lookback`. No `zoi_min_candidates` key (hardcoded in Rust at 3).
- **Proposed action:** Update sprint log to match implemented key names.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 7

**L10. Phase 4.0 exit criterion stale**
- **File:line:** `docs/02_roadmap.md:186`
- **Category:** clarity
- **Description:** Exit criterion "Worker throughput >= 1,000,000 pos/hr" exceeds current baseline of 659,983. Will never be met with current config.
- **Proposed action:** Update to >= 625,000 (CLAUDE.md target) or document hardware requirement.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 7

**L11. `model.eval()` called per-batch in InferenceServer**
- **File:line:** `hexo_rl/selfplay/inference_server.py:103`
- **Category:** perf_hot
- **Description:** Idempotent but traverses full module tree on every batch. Model is already set to eval mode by `WorkerPool.start()`.
- **Proposed action:** Remove per-batch call.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 5

**L12. Redundant `p_fp32` computation in trainer**
- **File:line:** `hexo_rl/training/trainer.py:319,394`
- **Category:** perf_cold
- **Description:** `torch.exp(log_policy.float())` computed twice per step when entropy regularization is active.
- **Proposed action:** Compute once, reuse.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 4

**L13. `SipHash` in replay buffer sampling**
- **File:line:** `engine/src/replay_buffer/sampling.rs:61`
- **Category:** perf_cold
- **Description:** Uses `std::collections::HashSet<i64>` (SipHash) instead of `FxHashSet`. Rest of engine uses FxHash.
- **Proposed action:** Switch to `FxHashSet`.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 2

**L14. Undiscoverable scripts**
- **File:line:** `scripts/sealbot_quality_diagnostic.py:1`, `scripts/inject_corpus.py:1`, `scripts/push_corpus_preview.py:1`
- **Category:** clarity
- **Description:** Three scripts with no Makefile targets and no CLAUDE.md mention.
- **Proposed action:** Add Makefile targets or document.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 8

**L15. `tqdm` dep used by one file; rest of project uses `rich.progress`**
- **File:line:** `requirements.txt` / `hexo_rl/bootstrap/dataset.py:10`
- **Category:** deps
- **Description:** Inconsistent progress bar library.
- **Proposed action:** Switch to `rich.progress.track()` or defer `tqdm` import.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 8

**L16. `game_browser.py` duplicated move-reader functions**
- **File:line:** `hexo_rl/monitoring/game_browser.py:376-385`
- **Category:** redundancy
- **Description:** `_read_human_moves()` and `_read_bot_moves()` have identical bodies.
- **Proposed action:** Merge into one function.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 6

**L17. N+1 query in eval pipeline player name lookup**
- **File:line:** `hexo_rl/eval/eval_pipeline.py:211`
- **Category:** perf_cold
- **Description:** One `SELECT` per player ID. Grows with checkpoint count.
- **Proposed action:** Add batch `get_player_names_batch()` method.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 6

**L18. Missing SQLite indices on matches table**
- **File:line:** `hexo_rl/eval/results_db.py:15`
- **Category:** perf_cold
- **Description:** No `CREATE INDEX` for `player_a_id`, `player_b_id`, `run_id`, `ratings.player_id`. Full-table scans on aggregation queries.
- **Proposed action:** Add indices.
- **Risk:** None
- **Requires bench:** No
- **Source:** Subagent 6

---

### NIT

**N1.** `engine/src/lib.rs:172` — stray doc comment fragment ("Returns a list of all stones...") attached to `count_winning_moves`. (Subagent 3)

**N2.** `engine/src/board/threats.rs:118-130` — `_end_q`/`_end_r` parameters have leading underscore but ARE used. Suppresses future dead-param Clippy warnings. Rename to `end_q`/`end_r`. (Subagent 3)

**N3.** `engine/benches/mcts_bench.rs:10` — `bench_win_detection` duplicates `board_bench.rs` coverage. Remove from MCTS bench. (Subagent 3)

**N4.** `hexo_rl/training/trainer.py:179,223,273,277` — numpy imported three times inside `train_step()` under different aliases. Hoist to module level. (Subagent 4)

**N5.** `hexo_rl/model/network.py:34` — unused `from typing import Tuple, Union`. (Subagent 4)

**N6.** `hexo_rl/selfplay/worker.py:32` — duplicate `from engine import ReplayBuffer` (already imported on line 26 pattern). Merge. (Subagent 5)

**N7.** `hexo_rl/monitoring/terminal_dashboard.py:389` — deferred `from rich.console import Group` inside 4 Hz render loop. Move to top-level. (Subagent 6)

**N8.** `hexo_rl/selfplay/completed_q.py` — test-only module, should be documented as such or moved to `tests/`. (Subagent 5)

**N9.** `hexo_rl/bootstrap/generate_corpus.py:279-281` — `n_random = 3` hardcoded; should be in `configs/corpus.yaml`. (Subagent 6)

**N10.** `engine/src/game_runner.rs:369` — `Vec::new()` without capacity hint for `all_batch_features`. Use `Vec::with_capacity(leaves.len())`. (Subagent 2)

**N11.** `engine/src/game_runner.rs:348` — `records = Vec::new()` without capacity hint. Use `Vec::with_capacity(max_moves)`. (Subagent 2)

**N12.** MCTS `run_simulations_cpu_only` (`engine/src/mcts/mod.rs:415-428`) allocates `Vec<Vec<f32>>` per simulation in benchmark path. Pre-allocate outside loop. (Subagent 1)

**N13.** `docs/bench_batcher_sweep.csv` untracked in `docs/` instead of `reports/benchmarks/`. (Subagent 8)

---

## Boundary compliance check

| Rule | Status | Notes |
|---|---|---|
| Tensor history assembly stays in Python | PASS | No proposals to move it to Rust |
| No Python in MCTS hot loops | PASS | All MCTS in Rust; Python only for NN forward pass |
| Zero-copy PyO3 NumPy transfer | PASS | `sample_batch` and `collect_data` use `into_pyarray`. Legacy `submit_and_wait` uses `.tolist()` (H7) but is not on live training path |
| No Python-side replay buffer | PASS | Only Rust `ReplayBuffer` exists. `RecencyBuffer` is a thin Python deque for recent-window sampling — it stores references, not data |
| No forced-win short-circuit in expansion | PASS | Quiescence is value-override only (§28). Expansion always runs NN eval. Regression test at `mcts/mod.rs:878` |
| torch.compile disabled | PASS | `torch_compile: false` in `training.yaml`. Pretrain default is wrong (L1) but overridden by config |

---

## Config drift report

| Key | File 1 (value) | File 2 (value) | Effective | Issue |
|---|---|---|---|---|
| `eval_interval` | `training.yaml` (5000) | `eval.yaml` (5000) | `eval.yaml` when pipeline enabled; `training.yaml` at line 555 overwrites | Misleading comment; dual source of truth (M4) |
| `gumbel_mcts` | `selfplay.yaml` (true) | Sprint log §62 (false) | true | Undocumented deviation (H1) |
| `draw_reward` | `training.yaml` (-0.5) | `pool.py:74` default (-0.1) | -0.5 (config wins) | Stale fallback default in source |
| `completed_q_values` | `selfplay.yaml` (true) | Trainer lookup (nested) | false (bug) | Broken lookup path (C1) |
| `torch_compile` | `training.yaml` (false) | `pretrain.py:656` default (True) | false (config wins) | Wrong default in pretrain (L1) |

---

## Dead code inventory

### Files (zero inbound imports)

| File | Lines | Notes |
|---|---|---|
| `hexo_rl/monitoring/metrics_writer.py` | ~60 | Phantom wandb dep |
| `hexo_rl/monitoring/replay_poller.py` | ~130 | Full daemon infrastructure |
| `hexo_rl/opening_book/game_record_parser.py` | ~80 | + `__init__.py` |
| `hexo_rl/api/__init__.py` | 1 | Empty placeholder |
| `hexo_rl/bootstrap/opening_classifier.py` | ~100 | Test-only import |

### Classes/functions (zero call sites)

| Symbol | File | Notes |
|---|---|---|
| `MetricsWriter` | `monitoring/metrics_writer.py` | Entire class |
| `GameReplayPoller` | `monitoring/replay_poller.py` | Entire class |
| `BootstrapDataset` | `bootstrap/dataset.py` | Superseded by `AugmentedBootstrapDataset` |
| `HexoComScraper` | `bootstrap/scraper.py` | Stub, never instantiated |
| `KrakenBotBot` | `bootstrap/bots/krakenbot_bot.py` | + signature mismatch |
| `LossResult` | `training/losses.py` | Dead dataclass |
| `submit_request_and_wait_rust` | `inference_bridge.rs` | Masked by impl-block `#[allow(dead_code)]` |
| `Formation` enum | `formations/mod.rs` | + `has_forced_win()` |

### Dependencies

| Package | Status | Notes |
|---|---|---|
| `wandb` | Not in requirements.txt | Referenced only by dead `MetricsWriter` |
| `tqdm` | In requirements.txt | Used by 1 file (`dataset.py`); rest uses `rich.progress` |

---

## Deferred / out-of-scope

| Item | Sprint log ref | Why deferred |
|---|---|---|
| SealBot mixed opponent in self-play | §17, reverted at `c9f39de` | GIL contention. Re-add with subprocess wrapper post-Phase 4.5. |
| Forced-win short-circuit | Removed pre-sprint | Prevents NN from learning forced-win patterns. Quiescence value override (§28) is the correct replacement. |
| torch.compile | §3→§25→§30→§32 disabled | Python 3.14 CUDA graph TLS conflict. Only +3% throughput. |
| Uncertainty head | §33, disabled §59 | Gaussian NLL diverges at clamp floor. `uncertainty_weight: 0.0` is deliberate. |
| Virtual loss always-on | Design choice | Unconditional in batched selection. No off switch. Documented as intended. |
| Board::Clone leaves legal_cache empty | Sprint log clone fix | Verified correct at `state.rs:674`. |

---

## Recommended next prompts

Each prompt is scoped to land as one commit:

**1. Fix `completed_q_values` lookup + add regression test**
```
Fix trainer.py:305 — change self.config.get("selfplay", {}).get("completed_q_values", False)
to self.config.get("completed_q_values", False). Add test in test_trainer.py that passes
{"completed_q_values": True} in config and asserts compute_kl_policy_loss is invoked.
Revert gumbel_mcts to false in configs/selfplay.yaml. Run make test.py to confirm.
```

**2. Dead code cleanup (batch delete)**
```
Delete the following files:
  hexo_rl/monitoring/metrics_writer.py
  hexo_rl/monitoring/replay_poller.py
  hexo_rl/opening_book/ (entire directory)
  hexo_rl/api/ (entire directory)
  hexo_rl/bootstrap/opening_classifier.py + tests/test_opening_classifier.py
  hexo_rl/bootstrap/scraper.py:HexoComScraper class (lines 141-end)
  hexo_rl/training/losses.py:LossResult class
  engine/src/formations/mod.rs (+ remove pub mod formations from lib.rs)
  engine/src/inference_bridge.rs:submit_request_and_wait_rust() method
Remove tqdm from requirements.txt; switch dataset.py progress to rich.
Run make test.all to confirm nothing breaks.
```

**3. Config hygiene pass**
```
Add total_steps and min_lr to configs/training.yaml.
Remove eval_interval from configs/eval.yaml (keep only in training.yaml).
Delete 7 dead keys from monitoring.yaml. Add emit_queue_maxsize: 200.
Fix pretrain.py:656 torch_compile default to False.
Update docs/01_architecture.md temperature section and replay buffer throughput.
Run make test.py to confirm.
```

**4. Feature buffer pool return in game_runner**
```
In engine/src/game_runner.rs, after encode_18_planes_to_buffer writes into feat,
copy data to a stack-allocated or per-game scratch buffer, then call
batcher.return_feature_buffer(feat) immediately. Run make bench.full before and after
to verify worker throughput target (>= 625,000 pos/hr) is maintained.
```

**5. Test gap closure (ownership/threat heads + multi-axis threats)**
```
Add to test_network.py: forward pass with ownership=True and threat=True, assert shapes.
Add to test_trainer.py: train_step with ownership/threat targets, assert loss keys present.
Add to engine threats.rs tests: multi-axis overlap takes max level.
Fix test_phase4_smoke.py decay_steps to match production config (300_000).
Run make test.all.
```
