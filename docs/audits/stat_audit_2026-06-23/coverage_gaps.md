# Coverage Gaps — Phase 0

## 1. Stats absent from §4 banked logs

### mcts_mean_depth
- Emit site: `engine/src/game_runner/mod.rs:584` (Rust getter), surfaced via `hexo_rl/training/events.py:309` in `train_step_summary` structlog event and `events.py:259` in `iteration_complete` dashboard event.
- **Not present in banked sample.** The banked `train_cdf2.jsonl` `train_step` rows do NOT contain `mcts_mean_depth`; it appears only in `train_step_summary`. The `events_cdf2.jsonl` `iteration_complete` event DOES contain `mcts_mean_depth` (confirmed present). Structlog `train_step` (not `train_step_summary`) is the canonical per-step log — `mcts_mean_depth` is absent from that row. Finding: depth is emitted per-iteration (log_interval cadence) not per training-step.

### mcts_mean_root_concentration
- Same emit path as `mcts_mean_depth`. Present in `iteration_complete` events JSONL; absent from `train_step` structlog rows in banked sample.

### depth / root_concentration (PREREG seed S1, S2)
- PREREG seed S1 claims "collapse=search-drop reading." Source code re-derived: `mcts_mean_depth` is the LIFETIME arithmetic mean of per-search averages (not reset per step; `mcts_stat_count` accumulates since `start()`). The "collapse=search-drop" interpretation is Axis-C concern (PUCT semantics applied to Gumbel-SH regime where visit-distribution is mechanically pinned by Sequential Halving). Re-derivation confirms the emit site exists; the framing concern is real and handed to Phase 1 B3 bucket auditor.

### value_accuracy_corpus, value_accuracy_selfplay, value_bce_corpus, value_bce_selfplay, value_rows_*
- Emit site: `hexo_rl/training/trainer.py:1028-1037` (§D-VALCEIL Q3 block).
- **Not present in banked sample.** Banked `train_step` rows in cdf2 and 76cf logs do not contain these keys. These are newer additions (§D-VALCEIL Q3). Emit exists in source; absent from banked logs = logs predate the feature.

### value_loss_main, value_loss_uncertainty, value_loss_aux, value_loss_composite
- Emit site: `hexo_rl/training/trainer.py:1022-1025`.
- **Not present in banked train_step rows** (cdf2 and 76cf logs). Absent from banked sample.

### full_search_frac
- Emit site: `hexo_rl/training/trainer.py:1043`.
- **Not present in banked train_step rows** in either banked log.

### fp16_scale
- Emit site: `hexo_rl/training/trainer.py:1045`.
- **Not present in banked train_step rows.**

### policy_entropy_pretrain, policy_entropy_selfplay, selfplay_model_entropy_batch, policy_entropy_recent, policy_entropy_uniform_selfplay
- **Present in `train_step_summary` structlog (cdf2) and `training_step` events (cdf2 events JSONL). NOT in `train_step` structlog rows.** The `train_step` structlog event is the per-step log; the entropy split lives only in `train_step_summary` (log_interval cadence). This is correct by design but means per-step granularity is not available for entropy splits in banked structlog sample.

### policy_target_entropy_fullsearch, policy_target_entropy_fastsearch, policy_target_kl_uniform_*, n_rows_policy_loss, n_rows_total
- Present in `training_step` events JSONL (cdf2); absent from `train_step` structlog in banked sample.

### value_spread (T3 + alt dual fields)
- Present in `events_cdf2.jsonl` as `value_spread` event with fields: `t3_spread`, `alt_spread`, `both_pass`, `spread` (back-compat), `mean_colony`, `mean_ext`, `t3_components`, `alt_components`, `warn_threshold`, `soft_abort_threshold`, `alt_warn_threshold`, `alt_soft_abort_threshold`.
- `alt_spread` IS present in banked events log. PREREG seed S7 ("alt_spread goes NaN claim vs handoff says it does not") — source code re-derived: `compute_value_spread_dual` has a branch that skips the alt bank when `in_channels` doesn't match and instead tries re-encoding via registry helpers; if re-encode fails, `alt_spread = float("nan")` is set at line 400. For the standard v6 8-plane model the re-encode branch is NOT triggered (`alt_applicable=True`) so `alt_spread` does NOT NaN. Handoff claim verified by source: alt_spread is NaN ONLY on cross-encoding runs (e.g. v6tp with 10 planes + re-encode failure). The banked log shows numeric alt_spread values, consistent with the handoff claim.

### buffer_composition, worker_draw_rate, model_version_summary
- **Not present in banked sample** (instrumentation_enabled=False in the logged run variant, or emitted on a different cadence not captured). Emit sites confirmed at `step_coordinator.py:1346/1353/1358`.

### value_probe_drift
- **Not present in banked sample.** Same instrumentation gate.

### track_b_buffer_snapshot (colony_frac, extension_frac, etc.)
- **Not present in banked sample.** Emitted as `buffer_position_class_snapshot` event; not in banked logs.

### per_source_grad_norm
- **Not present in banked sample.** Emitted as `per_source_grad_norm` structlog event; gated by `_track_b_grad_attribution` flag (off by default).

### rr_5rung fields (colony_fraction_winner, n_components_winner, winner_completed_via_threat, terminal_threat_count)
- **Present in banked rr_5rung.jsonl only.** These are offline round-robin eval metrics, not emitted during live training. No dashboard consumer wires them into the monitoring surface. Axis-E concern (construct validity): `colony_fraction_winner` measures a symptom not directly tied to the §D-COLONY metric tracked in self-play.

### offwindow_forced_win_rate, offwindow_strict_forced_rate
- **Not present in banked sample.** Opponent arm disabled in the logged variant.

### strength_aggregate, strength_cycle_density
- **Not present in banked sample.** Operator-gated feature; off in logged run.

### wr_argmax_n, wr_nnue
- **Not present in banked sample.** Opponent arms disabled.

### ply_index_loss
- **Not present in banked sample.** Enabled only when `ply_index_weight > 0`.

### stride5_run_p90, row_max_density
- Present in `game_complete` events JSONL (cdf2). Confirmed.

### eval_complete event
- **Not present in banked events JSONL.** No eval round fired in the banked logs (short training run without eval). The `eval_complete` dashboard event (`eval_drain.py:45-52`) and `evaluation_round_complete` structlog (`eval_pipeline.py:452`) are emitted only when an eval fires.

---

## 2. Emits with no monitor consumer

### `train_step` structlog (per-step)
- Contains: `grad_norm`, `total_loss`, `policy_loss`, `value_loss`, `value_loss_main`, `value_loss_uncertainty`, `value_loss_aux`, `value_loss_composite`, `value_accuracy`, `value_accuracy_masked`, `value_accuracy_corpus`, `value_accuracy_selfplay`, `value_bce_corpus`, `value_bce_selfplay`, `value_rows_*`, `ownership_loss`, `threat_loss`, `chain_loss`, `full_search_frac`, `lr`, `fp16_scale`.
- The dashboard renderers consume the `training_step` EVENTS (from `events.py` emit), NOT the `train_step` structlog. The `train_step` structlog is the JSONL-only record. The §D-VALCEIL Q3 keys (`value_accuracy_corpus`, etc.) appear ONLY in `train_step` structlog; they are NOT forwarded to the `training_step` dashboard event. No monitor consumer for these per-source accuracy keys.

### `per_source_grad_norm` structlog
- Track B gradient attribution; no dashboard event consumer. Structlog-only.

### `buffer_position_class_snapshot` structlog + event
- Both structlog and `emit_event` called. Dashboard spec does not define a renderer for this event. Terminal dashboard ignores it; web dashboard's generic event log would catch it but there's no dedicated panel.

### `gpu_stats` structlog
- Fields: `gpu_util_pct`, `mem_util_pct`, `vram_used_gb`, `vram_total_gb`, `temp_c`. The `system_stats` EVENTS JSONL has only `gpu_util_pct`, `vram_used_gb`, `vram_total_gb` (no `mem_util_pct`, no `temp_c`). `temp_c` is logged structlog-only; no dashboard consumer.

### `train_step_timing` structlog (perf mode)
- `h2d_us`, `fwd_loss_us`, `bwd_opt_us`, `total_us`, `batch_n`. Diagnostic mode only. No dashboard consumer.

### `vram_probe` structlog (perf mode)
- `vram_peak_gb`, `vram_allocated_gb`, `vram_reserved_gb`, `vram_frag_gb`, `num_ooms`. Diagnostic mode only. No dashboard consumer.

---

## 3. Count delta vs prior audit (97)

This enumeration counts **113** distinct tracked signal fields crossing the emit boundary. Delta = +16. Sources of the increase:
- §D-VALCEIL Q3 per-source value decomposition: +9 new keys (`value_accuracy_masked`, `_corpus`, `_selfplay`, `value_bce_corpus`, `value_bce_selfplay`, `value_rows_corpus`, `value_rows_selfplay`, `value_rows_masked`, `value_rows_*_supervised`) — new since prior audit.
- `value_loss_main`, `_uncertainty`, `_aux`, `_composite`: +4 decomp keys.
- `per_source_grad_norm` attribution: +3 sub-fields (corpus/recent/uniform).
- Gumbel-specific split keys (`policy_target_entropy_fullsearch/fastsearch`, `policy_target_kl_uniform_*`): these were probably partially counted before.
- `stride5_run_p90`, `row_max_density` in `game_complete`: +2 (new since prior audit).

Some prior-counted stats may have been structlog-only diagnostic entries not found here because they lack a unique `event` key; the prior count may have included diagnostic/lifecycle events that this enumeration excludes per the "signal stats" scope.
