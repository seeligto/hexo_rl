# Stat Inventory — Phase 0 Enumeration

**Method:** grep/ripgrep over source for `log.info`, `log.warning`, `emit_event`, `"event":` dict keys,
and `@getter`/PyO3 property declarations. Rust atomics tracked via `WorkerStats` struct + `SelfPlayRunner`
PyO3 getters. Per-stat metadata derived from source only — NOT from banked logs (those are used only
for "present in banked sample" verification in coverage_gaps.md). Emit-site line numbers are at the
*statement* that fires the kwarg/key; function entry lines not used.

Prior audit count: 97 tracked stats. This enumeration found **113** distinct named stats/fields
that cross the emit boundary (structlog kwargs or event-dict keys that carry signal, excluding
pure bookkeeping strings like `event`, `ts`, `step`, `level`, `timestamp`, `path`, `error`,
`msg`, `trigger`, `checkpoint_path`). Delta: +16 vs prior count (see coverage_gaps.md).

---

## B1 — Training loop (`hexo_rl/training/**`)

| name | bucket | emit_site | aggregates | has_band |
|---|---|---|---|---|
| loss (total_loss) | B1 | hexo_rl/training/trainer.py:1019 | yes (batch) | no |
| policy_loss | B1 | hexo_rl/training/trainer.py:1020 | yes (batch) | no |
| value_loss | B1 | hexo_rl/training/trainer.py:1021 | yes (batch) | no |
| value_loss_main | B1 | hexo_rl/training/trainer.py:1022 | yes (batch) | no |
| value_loss_uncertainty | B1 | hexo_rl/training/trainer.py:1023 | yes (batch) | no |
| value_loss_aux | B1 | hexo_rl/training/trainer.py:1024 | yes (batch) | no |
| value_loss_composite | B1 | hexo_rl/training/trainer.py:1025 | yes (batch) | no |
| value_accuracy | B1 | hexo_rl/training/trainer.py:1027 | yes (batch) | no |
| value_accuracy_masked | B1 | hexo_rl/training/trainer.py:1028 | yes (batch) | no |
| value_accuracy_corpus | B1 | hexo_rl/training/trainer.py:1029 | yes (batch) | no |
| value_accuracy_selfplay | B1 | hexo_rl/training/trainer.py:1030 | yes (batch) | no |
| value_bce_corpus | B1 | hexo_rl/training/trainer.py:1031 | yes (batch) | no |
| value_bce_selfplay | B1 | hexo_rl/training/trainer.py:1032 | yes (batch) | no |
| value_rows_corpus | B1 | hexo_rl/training/trainer.py:1033 | yes (batch) | no |
| value_rows_selfplay | B1 | hexo_rl/training/trainer.py:1034 | yes (batch) | no |
| value_rows_masked | B1 | hexo_rl/training/trainer.py:1035 | yes (batch) | no |
| value_rows_corpus_supervised | B1 | hexo_rl/training/trainer.py:1036 | yes (batch) | no |
| value_rows_selfplay_supervised | B1 | hexo_rl/training/trainer.py:1037 | yes (batch) | no |
| policy_entropy | B1 | hexo_rl/training/trainer.py:957 | yes (batch) | yes (alert_entropy_min=1.0, warn=2.0) |
| policy_entropy_pretrain | B1 | hexo_rl/training/trainer.py:958 | yes (batch) | no |
| policy_entropy_selfplay | B1 | hexo_rl/training/trainer.py:959 | yes (batch) | yes (collapse_threshold_nats=1.5) |
| selfplay_model_entropy_batch | B1 | hexo_rl/training/trainer.py:960 | yes (batch) | yes (same as policy_entropy_selfplay) |
| policy_entropy_recent | B1 | hexo_rl/training/trainer.py:961 | yes (batch) | no |
| policy_entropy_uniform_selfplay | B1 | hexo_rl/training/trainer.py:962 | yes (batch) | no |
| policy_target_entropy | B1 | hexo_rl/training/trainer.py:963 | yes (batch) | no |
| policy_target_entropy_fullsearch | B1 | hexo_rl/training/trainer.py:948 (via compute_policy_target_metrics) | yes (batch) | no |
| policy_target_entropy_fastsearch | B1 | hexo_rl/training/trainer.py:948 | yes (batch) | no |
| policy_target_kl_uniform_fullsearch | B1 | hexo_rl/training/trainer.py:948 | yes (batch) | no |
| policy_target_kl_uniform_fastsearch | B1 | hexo_rl/training/trainer.py:948 | yes (batch) | no |
| frac_fullsearch_in_batch | B1 | hexo_rl/training/trainer.py:967 | yes (batch) | no |
| n_rows_policy_loss | B1 | hexo_rl/training/trainer.py:948 | yes (batch) | no |
| n_rows_total | B1 | hexo_rl/training/trainer.py:948 | yes (batch) | no |
| grad_norm | B1 | hexo_rl/training/trainer.py:964 | no | yes (alert_grad_norm_max=10.0) |
| lr | B1 | hexo_rl/training/trainer.py:966 | no | no |
| full_search_frac | B1 | hexo_rl/training/trainer.py:967 | yes (batch) | no |
| fp16_scale | B1 | hexo_rl/training/trainer.py:1045 | no | no |
| opp_reply_loss (aux_loss) | B1 | hexo_rl/training/trainer.py:1038 | yes (batch) | no |
| uncertainty_loss | B1 | hexo_rl/training/trainer.py:1039 | yes (batch) | no |
| avg_sigma | B1 | hexo_rl/training/trainer.py:1144 | yes (batch) | no |
| ownership_loss | B1 | hexo_rl/training/trainer.py:1040 | yes (batch) | no |
| threat_loss | B1 | hexo_rl/training/trainer.py:1041 | yes (batch) | no |
| chain_loss | B1 | hexo_rl/training/trainer.py:1042 | yes (batch) | no |
| ply_index_loss | B1 | hexo_rl/training/trainer.py:1151 | yes (batch) | no |
| aux_loss_rows | B1 | hexo_rl/training/trainer.py:1154 | yes (batch) | no |
| per_source_grad_norm (corpus/recent/uniform) | B1 | hexo_rl/training/trainer.py:865 | yes (batch) | no |
| axis_q | B1 | hexo_rl/training/events.py:122 | yes (recent games) | yes (axis_warn=0.45, axis_alert=0.50) |
| axis_r | B1 | hexo_rl/training/events.py:122 | yes (recent games) | yes |
| axis_s | B1 | hexo_rl/training/events.py:122 | yes (recent games) | yes |
| axis_max | B1 | hexo_rl/training/events.py:122 | yes (recent games) | yes |
| early_game_entropy_mean | B1 | hexo_rl/training/events.py:188 | yes (10-position fixture) | yes (EARLY_GAME_ENTROPY_WARN_THRESHOLD) |
| early_game_top1_mass_mean | B1 | hexo_rl/training/events.py:323 | yes (fixture) | no |
| buffer_position_class_snapshot — colony_frac | B1 | hexo_rl/training/track_b_buffer_snapshot.py:92 | yes (sample) | no |
| buffer_position_class_snapshot — extension_frac | B1 | hexo_rl/training/track_b_buffer_snapshot.py:93 | yes (sample) | no |
| buffer_position_class_snapshot — neither_frac | B1 | hexo_rl/training/track_b_buffer_snapshot.py:94 | yes (sample) | no |
| colony_mean_value_target | B1 | hexo_rl/training/track_b_buffer_snapshot.py:96 | yes (sample) | no |
| extension_mean_value_target | B1 | hexo_rl/training/track_b_buffer_snapshot.py:97 | yes (sample) | no |
| value_probe_drift — decisive_mean | B1 | hexo_rl/training/step_coordinator.py:1321 | yes (50-pos fixture) | no |
| value_probe_drift — decisive_std | B1 | hexo_rl/training/step_coordinator.py:1322 | yes (fixture) | no |
| value_probe_drift — draw_mean | B1 | hexo_rl/training/step_coordinator.py:1323 | yes (fixture) | no |
| value_probe_drift — draw_std | B1 | hexo_rl/training/step_coordinator.py:1324 | yes (fixture) | no |
| instrumentation_periodic — draw_target_fraction | B1 | hexo_rl/training/step_coordinator.py:1367 | yes (buffer) | no |
| instrumentation_periodic — colony_terminal_fraction | B1 | hexo_rl/training/step_coordinator.py:1368 | yes (buffer) | no |
| instrumentation_periodic — six_terminal_fraction | B1 | hexo_rl/training/step_coordinator.py:1369 | yes (buffer) | no |
| instrumentation_periodic — cap_terminal_fraction | B1 | hexo_rl/training/step_coordinator.py:1370 | yes (buffer) | no |

---

## B2 — Self-play / game-generation (`hexo_rl/selfplay/**`, pool)

| name | bucket | emit_site | aggregates | has_band |
|---|---|---|---|---|
| win_rate_p0 (x_winrate) | B2 | hexo_rl/training/events.py:251 | yes (games since start) | no |
| win_rate_p1 (o_winrate) | B2 | hexo_rl/training/events.py:252 | yes (games since start) | no |
| draw_rate | B2 | hexo_rl/training/events.py:253 | yes (games since start) | no |
| games_total | B2 | hexo_rl/training/events.py:246 | yes (cumulative) | no |
| games_this_iter | B2 | hexo_rl/training/events.py:247 | yes (iter delta) | no |
| games_per_hour | B2 | hexo_rl/training/events.py:248 | yes (rolling) | no |
| positions_per_hour | B2 | hexo_rl/training/events.py:249 | yes (rolling) | no |
| avg_game_length | B2 | hexo_rl/training/events.py:250 | yes (rolling) | no |
| sims_per_sec | B2 | hexo_rl/training/events.py:254 | yes (lifetime) | no |
| buffer_size | B2 | hexo_rl/training/events.py:255 | no | no |
| buffer_capacity | B2 | hexo_rl/training/events.py:256 | no | no |
| corpus_selfplay_frac | B2 | hexo_rl/training/events.py:257 | no | no |
| batch_fill_pct | B2 | hexo_rl/training/events.py:258 | yes (recent batches) | no |
| quiescence_fires_per_step | B2 | hexo_rl/training/events.py:228 | yes (iter delta) | no |
| colony_extension_stone_count | B2 | hexo_rl/selfplay/pool.py:770 | no | no |
| colony_extension_stone_total | B2 | hexo_rl/selfplay/pool.py:771 | no | no |
| colony_extension_fraction | B2 | hexo_rl/selfplay/pool.py:772 (game_complete event) | yes (rolling game window) | no |
| terminal_reason | B2 | hexo_rl/selfplay/pool.py:776 | no | no |
| model_version_min | B2 | hexo_rl/selfplay/pool.py:777 | no | no |
| model_version_max | B2 | hexo_rl/selfplay/pool.py:778 | no | no |
| model_version_distinct | B2 | hexo_rl/selfplay/pool.py:779 | no | no |
| model_version_range_size | B2 | hexo_rl/selfplay/pool.py:780 | no | no |
| stride5_run_p90 | B2 | hexo_rl/selfplay/pool.py:782 | yes (rolling games) | no |
| row_max_density | B2 | hexo_rl/selfplay/pool.py:784 | no | no |
| buffer_composition — draw_target_fraction | B2 | hexo_rl/training/step_coordinator.py:1346 | yes (buffer) | no |
| buffer_composition — six_terminal_fraction | B2 | hexo_rl/training/step_coordinator.py:1346 | yes (buffer) | no |
| buffer_composition — colony_terminal_fraction | B2 | hexo_rl/training/step_coordinator.py:1346 | yes (buffer) | no |
| buffer_composition — cap_terminal_fraction | B2 | hexo_rl/training/step_coordinator.py:1346 | yes (buffer) | no |
| buffer_composition — corpus_fraction | B2 | hexo_rl/training/step_coordinator.py:1346 | yes (buffer) | no |
| worker_draw_rate — per_worker | B2 | hexo_rl/training/step_coordinator.py:1353 | yes (50-game rolling) | no |
| model_version_summary — median_range | B2 | hexo_rl/training/step_coordinator.py:1358 | yes (rolling games) | no |
| model_version_summary — p90_range | B2 | hexo_rl/training/step_coordinator.py:1358 | yes | no |
| model_version_summary — max_range | B2 | hexo_rl/training/step_coordinator.py:1358 | yes | no |
| model_version_summary — median_distinct | B2 | hexo_rl/training/step_coordinator.py:1358 | yes | no |
| model_version_summary — spearman_rho_range_vs_draw | B2 | hexo_rl/training/step_coordinator.py:1358 | yes | no |
| gpu_util | B2 | hexo_rl/monitoring/gpu_monitor.py:119 (gpu_stats emit) | no | no |
| vram_used_gb | B2 | hexo_rl/monitoring/gpu_monitor.py:119 | no | no |
| rss_gb | B2 | hexo_rl/monitoring/gpu_monitor.py:128 (system_stats event) | no | no |
| cpu_util_pct | B2 | hexo_rl/monitoring/gpu_monitor.py:128 | no | no |
| disk_free_gb | B2 | hexo_rl/monitoring/disk_guard.py:60 | no | yes (warn_gb=10, fail_gb=5) |

---

## B3 — MCTS / search (`engine/src/mcts/**`, `hexo_rl/eval/gumbel_search_py.py`)

| name | bucket | emit_site | aggregates | has_band |
|---|---|---|---|---|
| mcts_mean_depth | B3 | engine/src/game_runner/mod.rs:584 (getter; accumulated from worker_loop via stats.rs) | yes (lifetime across all moves) | no |
| mcts_mean_root_concentration | B3 | engine/src/game_runner/mod.rs:595 (getter) | yes (lifetime across all moves) | no |
| mcts_quiescence_fires | B3 | engine/src/game_runner/mod.rs:603 (getter; cumulative) | yes (cumulative) | no |
| cluster_value_std_mean | B3 | engine/src/game_runner/mod.rs:612 (getter; K≥2 only) | yes (K≥2 moves) | no |
| cluster_policy_disagreement_mean | B3 | engine/src/game_runner/mod.rs:620 (getter; K≥2 only) | yes (K≥2 moves) | no |
| cluster_variance_sample_count | B3 | engine/src/game_runner/mod.rs:629 (getter) | yes (cumulative) | no |

*(Note: mcts_mean_depth and mcts_mean_root_concentration surface in B1/B2 events via pool.runner_stats() → events.py:260-261; the emit sites in events.py:260-261 and iteration_complete are the log boundary; the Rust getters are the ground-truth accumulate sites.)*

---

## B4 — Eval / promotion / Bradley-Terry / opponent-arm

| name | bucket | emit_site | aggregates | has_band |
|---|---|---|---|---|
| wr_best | B4 | hexo_rl/eval/eval_pipeline.py:452 (evaluation_round_complete) | yes (n_games vs best_checkpoint) | yes (promotion_winrate threshold, default 0.55) |
| ci_best (lo, hi) | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | yes (require_ci_above_half) |
| colony_wins_best | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| wr_sealbot | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes (n_games vs SealBot) | yes (legacy 0.55; demoted to diagnostic — alert_rules.check_sealbot_wr_hard_abort) |
| ci_sealbot | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | yes |
| colony_wins_sealbot | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| sealbot_gate_passed | B4 | hexo_rl/eval/eval_pipeline.py:452 | no | no |
| wr_random | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes (n_games vs random) | no |
| ci_random | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| colony_wins_random | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| wr_bootstrap_anchor | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes (n_games vs frozen anchor) | yes (bootstrap_floor.min_winrate default 0.45) |
| ci_bootstrap_anchor | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| colony_wins_bootstrap_anchor | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| wr_argmax_n | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes (n_games vs SealBot-argmax) | no |
| ci_argmax_n | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| colony_wins_argmax_n | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| wr_nnue | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes (n_games vs Hammerhead) | no |
| ci_nnue | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| colony_wins_nnue | B4 | hexo_rl/eval/eval_pipeline.py:452 | yes | no |
| offwindow_forced_win_rate | B4 | hexo_rl/eval/result_types.py:94 (populated by offwindow_adversary runner) | yes | yes (robustness_warn_threshold=0.06) |
| offwindow_strict_forced_rate | B4 | hexo_rl/eval/result_types.py:95 | yes | no |
| strength_aggregate | B4 | hexo_rl/eval/result_types.py:100 | yes (fixed ref-set) | yes (strength_abort_floor, currently 0.0 uncalibrated) |
| strength_cycle_density | B4 | hexo_rl/eval/result_types.py:101 | yes | yes (strength_cycle_density_max=0.15) |
| elo_estimate | B4 | hexo_rl/eval/eval_pipeline.py:440 (Bradley-Terry rating for current ckpt) | yes (all pairwise in DB) | no |
| value_fc2_weight_abs_max | B4 | hexo_rl/eval/eval_pipeline.py:293 | no | yes (G4 band [0.154, 0.462]) |
| g4_value_head_band_pass | B4 | hexo_rl/eval/eval_pipeline.py:294 | no | yes |
| eval_games | B4 | hexo_rl/eval/eval_pipeline.py:288 | yes (round total) | no |
| promoted | B4 | hexo_rl/eval/eval_pipeline.py:398 | no | no |
| wr_best — win_rate_vs_sealbot (eval_complete event) | B4 | hexo_rl/training/eval_drain.py:49 | yes | no |
| elo_estimate (eval_complete event) | B4 | hexo_rl/training/eval_drain.py:48 | yes | no |
| anchor_promoted (eval_complete event) | B4 | hexo_rl/training/eval_drain.py:51 | no | no |
| eval_round_wall_sec | B4 | hexo_rl/eval/result_types.py:70 | no | no |
| rr_5rung — colony_fraction_winner | B4 | hexo_rl/investigation/founding_2026-06-08/rr_5rung.jsonl (offline script) | yes (per game) | no |
| rr_5rung — n_components_winner | B4 | rr_5rung.jsonl | yes | no |
| rr_5rung — winner_completed_via_threat | B4 | rr_5rung.jsonl | no | no |
| rr_5rung — terminal_threat_count | B4 | rr_5rung.jsonl | yes | no |

---

## B5 — Monitoring / alert-rules / canary / dashboard-schema

| name | bucket | emit_site | aggregates | has_band |
|---|---|---|---|---|
| t3_spread (value_spread event) | B5 | hexo_rl/monitoring/value_spread_canary.py:469 | yes (40-pos T3 bank) | yes (WARN_THRESHOLD=0.30, SOFT_ABORT_THRESHOLD=0.20) |
| alt_spread (value_spread event) | B5 | hexo_rl/monitoring/value_spread_canary.py:469 | yes (40-pos alt bank) | yes (ALT_WARN_THRESHOLD=0.10, ALT_SOFT_ABORT_THRESHOLD=0.07) |
| both_pass (value_spread event) | B5 | hexo_rl/monitoring/value_spread_canary.py:469 | yes | no |
| mean_colony (value_spread event) | B5 | hexo_rl/monitoring/value_spread_canary.py:448 (DualCanaryResult.to_payload back-compat) | yes | no |
| mean_ext (value_spread event) | B5 | hexo_rl/monitoring/value_spread_canary.py:448 | yes | no |
| spread (value_spread event, back-compat alias for t3_spread) | B5 | hexo_rl/monitoring/value_spread_canary.py:307 | yes | yes (same as t3_spread) |
| t3_components (value_spread event) | B5 | hexo_rl/monitoring/value_spread_canary.py:301 | yes | no |
| alt_components (value_spread event) | B5 | hexo_rl/monitoring/value_spread_canary.py:302 | yes | no |
| wr_sealbot sliding window — wr_hard_abort_enabled alert | B5 | hexo_rl/monitoring/alert_rules.py:101 (check_sealbot_wr_hard_abort) | yes (N-eval window) | yes (wr_rolling_threshold=0.10, wr_early_death_threshold=0.05, collapse_from_peak_ratio=0.5) |
| strength_abort alert | B5 | hexo_rl/monitoring/alert_rules.py:199 (check_strength_regression_abort) | yes (N-eval window) | yes (strength_abort_floor=0.0, strength_cycle_density_max=0.15) |
| robustness_abort alert | B5 | hexo_rl/monitoring/alert_rules.py:295 (check_robustness_abort) | yes (N-eval window) | yes (robustness_warn_threshold=0.06) |
| policy_entropy — collapse alert | B5 | hexo_rl/monitoring/alert_rules.py:26 (check_entropy_collapse) | yes | yes (alert_entropy_min=1.0) |
| policy_entropy_selfplay — collapse alert | B5 | hexo_rl/monitoring/alert_rules.py:36 (check_selfplay_entropy_collapse) | yes | yes (collapse_threshold_nats=1.5) |
| grad_norm — spike alert | B5 | hexo_rl/monitoring/alert_rules.py:57 (check_grad_norm_spike) | no | yes (alert_grad_norm_max=10.0) |
| loss — consecutive increase alert | B5 | hexo_rl/monitoring/alert_rules.py:68 (check_loss_increase_window) | yes (window) | yes (alert_loss_increase_window=3) |
