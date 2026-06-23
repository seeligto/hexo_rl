# STAT_AUDIT.md — D-STATAUDIT master synthesis

> **148-stat re-run, authoritative; supersedes the ephemeral 97-stat first audit** (which was never committed — ran in a /tmp worktree, wiped by session cleanup). The promotion-gate finding is **INVERTED** relative to that first pass: the CI bug is conservative/false-negative, so promotions are trustworthy. Relocated from `_audit/` to this canonical home by D-CONSOLIDATE (2026-06-23).

## Verdict header

- **Source pin:** worktree `52067631f07b2ac2a83d25840d1a0b35cc29c90b` (PREREG §0; branch `phase4.5/gumbelprep` tip `e132e67` + tracked WIP). Audit-commit tip `5da9b24…`.
- **Inventory coverage:** 148 stats. Every stat below = exactly one row. 100% of the structured inventory.
- **Bin counts (final, post Phase-2 review + Phase-3 red-team):**

| Bin | Count |
|---|---|
| WRONG | 4 |
| BIASED | 16 |
| REDUNDANT | 7 |
| CORRECT | 121 |
| **Total** | **148** |

- **S7-delete=Y (mechanical stamp: WRONG ∨ REDUNDANT):** 11 rows.
- **Integrity result:** Phase-2 review **0 overrides** (105 subset rows all CONFIRMED). Phase-3 red-team **0 overturned**, 3 MODIFIED (regime-scope on `mcts_mean_root_concentration`; down-weight Axis-D on `strength_cycle_density`; correct seed5 distinct-count attribution). Dispatcher self-test: all 5 PASS. PREREG frozen pre-Phase-1, diff clean each boundary. **Audit PASS — adversarially robust.**

---

## ⚠️ CARVE-OUT — the mechanical S7=Y stamp OVER-COUNTS. Read this before deleting anything.

The S7-delete column below is the **mechanical** rule (WRONG ∨ REDUNDANT ⇒ Y). It marks **11** rows. **Most of those are FIX-IN-PLACE, not deletes** — the quantity is recoverable.

### REAL deletes (the SMALL set — only these may actually be removed):

1. **Exact-alias emits** (byte-identical duplicate of a canonical stat under a second name):
   - `selfplay_model_entropy_batch` == `policy_entropy_selfplay` (trainer.py:960, comment "alias; drop 2026-05-28" — deadline passed). Migrate dashboard band to `policy_entropy_selfplay` FIRST, then drop.
   - `value_loss_main` == `value_loss` (trainer.py:990, byte-identical float).
   - `spread` == `t3_spread` (value_spread_canary.py:305 byte-copy, pre-PR-C back-compat only).
2. **Trivially re-derivable, no consumer** (recoverable from a surfaced sibling in the SAME event, zero marginal info):
   - `model_version_range_size` == `model_version_max − model_version_min` (both in same game_complete event; verified all 381 banked games).
   - `aux_loss_rows` == `value_rows_selfplay` (identical `batch_n − n_pretrain`).
   - `frac_fullsearch_in_batch` == `full_search_frac` (identical numerator AND denominator; canonical = full_search_frac).
3. **Gumbel-unfixable HEALTH signal:**
   - `mcts_mean_root_concentration` — drop **as a health/confidence signal** (SH-pinned under Gumbel, Axis-C, unfixable). **KEEP the raw emit, guarded by planner** for any future PUCT-deployment A/B (red-team C.4 / §D.2 regime-scope: do NOT hard-delete; guard by `gumbel_mcts` so a PUCT ablation keeps the diagnostic).

### Everything ELSE stamped S7=Y is FIX-IN-PLACE (NOT a delete):

- `strength_aggregate` (WRONG) — phantom: NO producer exists; the `decide_promotion` replace-branch (gate_logic.py:57-62) is **dead code**. FIX = build the producer (or remove the dead gate branch). Do NOT silently drop the gate — it is the **sharpest promotion-integrity risk** (replaces the CI-guarded `wr_best` with an unguarded threshold compare, red-team §D.3). The quantity (fixed-reference strength) is intended and recoverable.
- `strength_cycle_density` (WRONG) — phantom: NO producer; non-transitivity abort always sees `.get(...,0.0)` ⇒ can never fire. FIX = emit real ladder-cycle density. Quantity recoverable.
- `alt_spread` (WRONG) — emits `float('nan')` SKIP sentinel on plane mismatch (encoding=None ⇒ re-encode branch never entered). FIX = emit an `alt_applicable` flag + wire encoding through `fire_canary` so the alt bank actually computes. Salvageable; do NOT delete the channel.
- `policy_entropy_selfplay` (REDUNDANT-stamp) — this is the **canonical** stat; the alias `selfplay_model_entropy_batch` is the one to drop. KEEP this.

---

## Master table (100% of inventory)

Legend — verdict: final after Phase-2/3 amendment. S7: mechanical Y/N (WRONG ∨ REDUNDANT ⇒ Y). action: keep/fix/drop. ★ = REAL delete (carve-out); all other Y = FIX-IN-PLACE.

| stat | bucket | emit_site | verdict | axes_failed | action | S7 | note |
|---|---|---|---|---|---|---|---|
| loss | B1 | trainer.py:1019 | CORRECT | — | keep | N | total loss scalar, no band |
| policy_loss | B1 | trainer.py:1020 | CORRECT | — | keep | N | CE vs Gumbel completed-Q improved policy, valid under Gumbel |
| value_loss | B1 | trainer.py:1021 | CORRECT | — | keep | N | canonical BCE-with-logits value loss |
| value_loss_main | B1 | trainer.py:1022 | REDUNDANT | F | drop ★ | Y | exact alias of value_loss (trainer.py:990) |
| value_loss_uncertainty | B1 | trainer.py:1023 | CORRECT | — | keep | N | weighted uncertainty-head contribution |
| value_loss_aux | B1 | trainer.py:1024 | BIASED | E | fix | N | opp_reply is POLICY-shaped, name implies value-aux; relabel/caveat |
| value_loss_composite | B1 | trainer.py:1025 | BIASED | E | fix | N | includes policy-shaped aux; rename mixed_head_loss_total or caveat |
| value_accuracy | B1 | trainer.py:1027 | BIASED | E | fix | N | includes draw/ply-capped rows, deflated; primary-display masked variant |
| value_accuracy_masked | B1 | trainer.py:1028 | CORRECT | — | keep | N | decided+supervised only (the clean stat) |
| value_accuracy_corpus | B1 | trainer.py:1029 | CORRECT | — | keep | N | unmasked corpus-slice accuracy |
| value_accuracy_selfplay | B1 | trainer.py:1030 | CORRECT | — | keep | N | unmasked selfplay-slice accuracy |
| value_bce_corpus | B1 | trainer.py:1031 | CORRECT | — | keep | N | corpus+supervised BCE; recombines to value_loss |
| value_bce_selfplay | B1 | trainer.py:1032 | CORRECT | — | keep | N | selfplay+supervised BCE |
| value_rows_corpus | B1 | trainer.py:1033 | CORRECT | — | keep | N | corpus row count, correct |
| value_rows_selfplay | B1 | trainer.py:1034 | CORRECT | — | keep | N | batch_n−n_pre complement; canonical per-source count |
| value_rows_masked | B1 | trainer.py:1035 | CORRECT | — | keep | N | supervised∩decided denom for value_accuracy_masked |
| value_rows_corpus_supervised | B1 | trainer.py:1036 | CORRECT | — | keep | N | corpus∩supervised denom for value_bce_corpus |
| value_rows_selfplay_supervised | B1 | trainer.py:1037 | CORRECT | — | keep | N | selfplay∩supervised denom for value_bce_selfplay |
| policy_entropy | B1 | trainer.py:957 | CORRECT | — | keep | N | Shannon nats, band 1.0/2.0 low=collapse correct; banked 2.14-2.89 |
| policy_entropy_pretrain | B1 | trainer.py:958 | CORRECT | — | keep | N | corpus-row entropy, NaN-guarded |
| policy_entropy_selfplay | B1 | trainer.py:959 | REDUNDANT | F | keep | Y | CANONICAL; the alias selfplay_model_entropy_batch is the drop. KEEP (fix = migrate band) |
| selfplay_model_entropy_batch | B1 | trainer.py:960 | REDUNDANT | F | drop ★ | Y | exact alias of policy_entropy_selfplay (comment "drop 2026-05-28"); migrate dashboard band first |
| policy_entropy_recent | B1 | trainer.py:961 | CORRECT | — | keep | N | recent-buffer slice entropy |
| policy_entropy_uniform_selfplay | B1 | trainer.py:962 | CORRECT | — | keep | N | uniform-slice entropy, falls back to selfplay |
| policy_target_entropy | B1 | trainer.py:963 | CORRECT | — | keep | N | improved-policy target entropy, Gumbel-valid |
| policy_target_entropy_fullsearch | B1 | trainer.py:948 | CORRECT | — | keep | N | full-search target entropy, NaN-guarded |
| policy_target_entropy_fastsearch | B1 | trainer.py:948 | CORRECT | — | keep | N | fast-search target entropy |
| policy_target_kl_uniform_fullsearch | B1 | trainer.py:948 | CORRECT | — | keep | N | KL=logN−H, correct |
| policy_target_kl_uniform_fastsearch | B1 | trainer.py:948 | CORRECT | — | keep | N | symmetric fast-search KL |
| frac_fullsearch_in_batch | B1 | trainer.py:967 | REDUNDANT | F | drop ★ | Y | identical num+denom to full_search_frac (canonical) |
| n_rows_policy_loss | B1 | trainer.py:948 | CORRECT | — | keep | N | full-search valid-policy count |
| n_rows_total | B1 | trainer.py:948 | CORRECT | — | keep | N | all policy-valid count |
| grad_norm | B1 | trainer.py:964 | CORRECT | — | keep | N | L2 norm, alert>10 generous; banked max 2.88 |
| lr | B1 | trainer.py:966 | CORRECT | — | keep | N | optimizer param_groups[0]['lr'] |
| full_search_frac | B1 | trainer.py:967 | CORRECT | — | keep | N | canonical full-search fraction |
| fp16_scale | B1 | trainer.py:1045 | CORRECT | — | keep | N | GradScaler loss scale |
| opp_reply_loss | B1 | trainer.py:1038 | CORRECT | — | keep | N | raw unweighted aux head loss, conditional |
| uncertainty_loss | B1 | trainer.py:1039 | CORRECT | — | keep | N | raw heteroscedastic loss; banked ~0.49 |
| avg_sigma | B1 | trainer.py:1144 | CORRECT | — | keep | N | sqrt(pred var) mean, descriptive |
| ownership_loss | B1 | trainer.py:1040 | CORRECT | — | keep | N | conditional ownership head loss |
| threat_loss | B1 | trainer.py:1041 | CORRECT | — | keep | N | conditional threat head loss |
| chain_loss | B1 | trainer.py:1042 | CORRECT | — | keep | N | conditional chain head loss |
| ply_index_loss | B1 | trainer.py:1151 | CORRECT | — | keep | N | conditional ply-index head loss |
| aux_loss_rows | B1 | trainer.py:1154 | REDUNDANT | F | drop ★ | Y | == value_rows_selfplay (identical batch_n−n_pretrain) |
| per_source_grad_norm | B1 | trainer.py:865 | CORRECT | — | keep | N | L2 per slice×group, diagnostic-mode-only |
| axis_q | B1 | events.py:122 | BIASED | D | fix | N | warn=0.45 below corpus baseline 0.4526 + live 0.50-0.54; raise threshold |
| axis_r | B1 | events.py:122 | BIASED | D | fix | N | same band miscalibration as axis_q; false alarm on healthy regime |
| axis_s | B1 | events.py:122 | CORRECT | — | keep | N | s=0.4479<0.45; composite warn charged to axis_q/r |
| axis_max | B1 | events.py:122 | CORRECT | — | keep | N | qualitative dominant-axis label |
| early_game_entropy_mean | B1 | events.py:188 | CORRECT | — | keep | N | legal-renorm Shannon, warn>4.5 (high=warn correct early) |
| early_game_top1_mass_mean | B1 | events.py:323 | CORRECT | — | keep | N | max legal-renorm prob, complement |
| colony_frac | B1 | track_b_buffer_snapshot.py:92 | CORRECT | — | keep | N | n_col/n_total; coverage gap (instrumentation off) |
| extension_frac | B1 | track_b_buffer_snapshot.py:93 | CORRECT | — | keep | N | n_ext/n_total; coverage gap |
| neither_frac | B1 | track_b_buffer_snapshot.py:94 | CORRECT | — | keep | N | partition complement; coverage gap |
| colony_mean_value_target | B1 | track_b_buffer_snapshot.py:96 | CORRECT | — | keep | N | mean V over colony class; coverage gap |
| extension_mean_value_target | B1 | track_b_buffer_snapshot.py:97 | CORRECT | — | keep | N | mean V over ext class; coverage gap |
| value_probe_decisive_mean | B1 | step_coordinator.py:1321 | CORRECT | — | keep | N | fixture mean V decisive; key=decisive_mean; coverage gap |
| value_probe_decisive_std | B1 | step_coordinator.py:1322 | CORRECT | — | keep | N | fixture std; coverage gap |
| value_probe_draw_mean | B1 | step_coordinator.py:1323 | CORRECT | — | keep | N | fixture mean V draw; coverage gap |
| value_probe_draw_std | B1 | step_coordinator.py:1324 | CORRECT | — | keep | N | fixture std draw; coverage gap |
| draw_target_fraction | B1 | step_coordinator.py:1367 | CORRECT | — | keep | N | live-config-derived window (PIPE-4 fix); coverage gap |
| colony_terminal_fraction | B1 | step_coordinator.py:1368 | CORRECT | — | keep | N | colony/total_games factual count |
| six_terminal_fraction | B1 | step_coordinator.py:1369 | CORRECT | — | keep | N | six_in_a_row/total_games |
| cap_terminal_fraction | B1 | step_coordinator.py:1370 | CORRECT | — | keep | N | ply_cap/total_games |
| win_rate_p0 | B2 | events.py:251 | CORRECT | — | keep | N | x_wins/games lifetime; p0+p1+dr=1.0 verified |
| win_rate_p1 | B2 | events.py:252 | CORRECT | — | keep | N | o_wins/games lifetime |
| draw_rate | B2 | events.py:253 | CORRECT | — | keep | N | draws/games lifetime; banked 0-0.153 |
| games_total | B2 | events.py:246 | CORRECT | — | keep | N | monotone cumulative count |
| games_this_iter | B2 | events.py:247 | CORRECT | — | keep | N | delta since last iteration_complete |
| games_per_hour | B2 | events.py:248 | CORRECT | — | keep | N | 60s sliding rate |
| positions_per_hour | B2 | events.py:249 | CORRECT | — | keep | N | gph*avg_gl derived throughput |
| avg_game_length | B2 | events.py:250 | CORRECT | — | keep | N | rolling 200-game mean, compound turns; banked 21-41.8 |
| sims_per_sec | B2 | events.py:254 | CORRECT | — | keep | N | n_sims*games/elapsed per drain; banked 3426-7428 |
| buffer_size | B2 | events.py:255 | CORRECT | — | keep | N | live size; schema-collision is hygiene only |
| buffer_capacity | B2 | events.py:256 | CORRECT | — | keep | N | live capacity readout |
| corpus_selfplay_frac | B2 | events.py:257 | CORRECT | — | keep | N | 1−w_pre batch-mix weight; distinct from corpus_fraction |
| batch_fill_pct | B2 | events.py:258 | CORRECT | — | keep | N | lifetime cumulative occupancy % |
| quiescence_fires_per_step | B2 | events.py:228 | BIASED | A | fix | N | name says per_step but value is per-10-step interval; divide by log_interval or rename |
| colony_extension_stone_count | B2 | pool.py:770 | CORRECT | — | keep | N | per-game count >6 hex-dist stones |
| colony_extension_stone_total | B2 | pool.py:771 | CORRECT | — | keep | N | per-game denom; banked 25-41 |
| colony_extension_fraction | B2 | pool.py:772 | CORRECT | — | keep | N | per-game ext/total; 0.15/0.25 are B5 cosmetic, no emit band |
| terminal_reason | B2 | pool.py:776 | CORRECT | — | keep | N | u8→string categorical map |
| model_version_min | B2 | pool.py:777 | CORRECT | — | keep | N | per-game min version |
| model_version_max | B2 | pool.py:778 | CORRECT | — | keep | N | per-game max version |
| model_version_distinct | B2 | pool.py:779 | CORRECT | — | keep | N | distinct-set count (uses distinct, not move count) |
| model_version_range_size | B2 | pool.py:780 | REDUNDANT | F | drop ★ | Y | == max−min, both in same game_complete event (verified 381 games) |
| stride5_run_p90 | B2 | pool.py:782 | CORRECT | — | keep | N | rolling P90 over 50 games; aggregates-label metadata error only |
| row_max_density | B2 | pool.py:784 | CORRECT | — | keep | N | max stones on any axis-row per game; banked 7-18 |
| corpus_fraction | B2 | step_coordinator.py:1346 | CORRECT | — | keep | N | 1−(sp_pushed/size) buffer frac; coverage gap |
| worker_draw_rate_per_worker | B2 | step_coordinator.py:1353 | CORRECT | — | keep | N | rolling 50-game draw rate/worker; coverage gap |
| mv_median_range | B2 | step_coordinator.py:1358 | BIASED | A | fix | N | ranges_sorted[n//2] upper-half median bias ≤+0.5 even n; use (n−1)//2 |
| mv_p90_range | B2 | step_coordinator.py:1358 | CORRECT | — | keep | N | int(n*0.9)−1 numpy-matching P90; coverage gap |
| mv_max_range | B2 | step_coordinator.py:1358 | CORRECT | — | keep | N | ranges_sorted[-1] over 200-game window; coverage gap |
| mv_median_distinct | B2 | step_coordinator.py:1358 | BIASED | A | fix | N | same upper-half median bias as mv_median_range; use (n−1)//2 |
| mv_spearman_rho_range_vs_draw | B2 | step_coordinator.py:1358 | CORRECT | — | keep | N | spearmanr(ranges,is_draw), n>=10 guard |
| gpu_util_pct | B2 | gpu_monitor.py:119 | CORRECT | — | keep | N | pynvml util.gpu; schema collision hygiene only |
| vram_used_gb | B2 | gpu_monitor.py:119 | CORRECT | — | keep | N | mem.used/1e9; banked ~1.01 |
| rss_gb | B2 | gpu_monitor.py:128 | CORRECT | — | keep | N | process RSS/1e9; banked ~6.32 |
| cpu_util_pct | B2 | gpu_monitor.py:128 | CORRECT | — | keep | N | psutil non-blocking, 0.0 first-call expected |
| disk_free_gb | B2 | disk_guard.py:60 | CORRECT | — | keep | N | shutil free/1e9; bands engineering-sound config-driven |
| mcts_mean_depth | B3 | game_runner/mod.rs:584 | CORRECT | — | keep | N | interior-PUCT descent depth (only root SH-forced); Gumbel-valid; banked ~3.7-3.9; no invented 3.0/3.4 literal |
| mcts_mean_root_concentration | B3 | game_runner/mod.rs:595 | WRONG | C | drop ★ (health) / keep emit guarded | Y | SH-pinned under Gumbel (root PUCT bypass selection.rs:124-126); banked flat 0.545-0.576. DROP as health signal; KEEP raw emit guarded by planner for PUCT A/B (regime-scoped, red-team C.4) |
| mcts_quiescence_fires | B3 | game_runner/mod.rs:603 | CORRECT | — | keep | N | 4-branch leaf override count, reset-per-move, no double-count |
| cluster_value_std_mean | B3 | game_runner/mod.rs:612 | BIASED | A,E | fix | N | per-leaf std correct but divided by LIFETIME cumulative count ⇒ frozen lifetime mean (banked flat ~0.42); emit windowed delta-mean |
| cluster_policy_disagreement_mean | B3 | game_runner/mod.rs:620 | BIASED | A,E | fix | N | same lifetime-cumulative divisor defect (banked flat ~0.69); emit windowed delta-mean |
| cluster_variance_sample_count | B3 | game_runner/mod.rs:629 | CORRECT | — | keep | N | exact distinct K>=2 count; eff-n witness for the two cluster means |
| wr_best | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | draw-half WR vs champion n=200; the LIVE gating instrument (ci_lo>0.5). PUCT-proxy of Gumbel net (S6) |
| ci_best | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | Wilson n=200; only CI that gates promotion |
| colony_wins_best | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | colony subset of win_count; anti-colony signal |
| wr_sealbot | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | draw-half WR vs SealBot n=50 monitor; band defect quarantined in flag. PUCT-proxy of Gumbel net (S6) |
| ci_sealbot | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | Wilson n=50, no gate dependency |
| colony_wins_sealbot | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | raw colony subset count vs SealBot |
| sealbot_gate_passed | B4 | eval_pipeline.py:452 | BIASED | D | fix | N | wr>=0.5 borrowed bar; SealBot fair WR ~18% ⇒ ~always-False; gates nothing. Make slope-based / re-band, drop "gate" name |
| wr_random | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | draw-half WR vs random n=20 floor |
| ci_random | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | Wilson n=20, no band |
| colony_wins_random | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | raw colony subset count |
| wr_bootstrap_anchor | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | draw-half WR vs frozen anchor n=100; floor gate 0.45 lacks CI guard (advisory) |
| ci_bootstrap_anchor | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | Wilson n=100; computed not consumed by floor gate |
| colony_wins_bootstrap_anchor | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | raw colony subset count |
| wr_argmax_n | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | model n_sims=1 vs SealBot DRIFT detector; advisory: §D-ARGMAX risk if variant zeroes opening_plies |
| ci_argmax_n | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | Wilson n=20 wide; no false CI-resolved claim |
| colony_wins_argmax_n | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | raw colony subset count |
| wr_nnue | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | draw-half WR vs Hammerhead NNUE n=100, default-off arm |
| ci_nnue | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | Wilson n=100 honest |
| colony_wins_nnue | B4 | eval_pipeline.py:452 | CORRECT | — | keep | N | raw colony subset count |
| offwindow_forced_win_rate | B4 | result_types.py:94 | CORRECT | — | keep | N | completing-cell (turn-correct) off-window rate; robustness BLOCK at 0.06; right adversarial instrument |
| offwindow_strict_forced_rate | B4 | result_types.py:95 | CORRECT | — | keep | N | stricter conditioned subset, distinct denom-condition |
| strength_aggregate | B4 | result_types.py:100 | WRONG | A,E | fix (build producer / remove dead branch) | Y | PHANTOM: no write site; replace-branch gate_logic.py:57-62 DEAD CODE. NOT a delete — top promotion-integrity risk (replaces CI-guarded wr_best unguarded) |
| strength_cycle_density | B4 | result_types.py:101 | WRONG | A,D | fix (build producer) | Y | PHANTOM: no write site; abort sees .get(...,0.0) always ⇒ never fires. Axis-D down-weighted (band IS calibrated, red-team C.6). NOT a delete |
| elo_estimate | B4 | eval_pipeline.py:440 | CORRECT | — | keep | N | BT point estimate anchored ckpt0; advisory surface BT CI alongside |
| value_fc2_weight_abs_max | B4 | eval_pipeline.py:293 | CORRECT | — | keep | N | value-head weight-mag canary, band 0.308±50% warning-only, Gumbel-independent |
| g4_value_head_band_pass | B4 | eval_pipeline.py:294 | CORRECT | — | keep | N | band flag for fc2 weight; canonical actionable form |
| eval_games | B4 | eval_pipeline.py:288 | CORRECT | — | keep | N | sum n across arms, coverage counter |
| promoted | B4 | eval_pipeline.py:398 | CORRECT | — | keep | N | strength_ok AND robustness_ok, fail-safe on missing arms; LIVE path = wr_best (strength-replace dead, charged to strength_aggregate) |
| eval_round_wall_sec | B4 | result_types.py:70 | CORRECT | — | keep | N | diagnostic timing, no band |
| t3_spread | B5 | value_spread_canary.py:469 | BIASED | D,E | fix | N | band anchored to synthetic 8-plane +0.617 but banked 4-plane T3=−0.5249 inverted; canary self-DEMOTED to INFORMATIONAL (failed to track eval WR). Keep number, re-anchor band |
| alt_spread | B5 | value_spread_canary.py:469 | WRONG | A,E | fix (emit alt_applicable flag + wire encoding) | Y | float('nan') SKIP sentinel on plane mismatch (encoding=None, re-encode never entered); seed7 INVERTED. Salvageable. NOT a delete |
| both_pass | B5 | value_spread_canary.py:469 | BIASED | E | fix | N | degrades to t3-only when alt skipped, mislabels "both"; L50 INFORMATIONAL demotion. Rename/flag single-bank |
| spread | B5 | value_spread_canary.py:307 | REDUNDANT | F | drop ★ | Y | byte-copy of t3_spread (value_spread_canary.py:305) back-compat only |
| mean_colony | B5 | value_spread_canary.py:307 | CORRECT | — | keep | N | honest mean V over 20 colony positions; banked −0.081 |
| mean_ext | B5 | value_spread_canary.py:307 | CORRECT | — | keep | N | honest mean V over 20 ext positions; banked +0.4439 |
| wr_sealbot_sliding_window_alert | B5 | alert_rules.py:101 | BIASED | B,E | fix | N | fires on raw wr no CI/n_eff; re-arms §D-FOUNDING-flagged instrument as primary. Attach deduped-bootstrap CI, reconcile primaries |
| strength_regression_abort_alert | B5 | alert_rules.py:199 | BIASED | B,D | fix | N | no CI exposed + floor=0.0 placeholder (TBD §2.5). Default-off keeps from WRONG; fed by phantom strength_aggregate. Lock floor + attach CI |
| robustness_abort_alert | B5 | alert_rules.py:295 | BIASED | B | fix | N | off-window trigger no CI/n_eff; band 0.06 derived; default-off PROMOTE+WARN. Closest to CORRECT of abort family; attach CI |
| entropy_collapse_alert | B5 | alert_rules.py:26 | CORRECT | — | keep | N | fires <1.0 nats (low=collapse correct); banked 2.14-2.89 above floor; NOT seed3 backwards band |
| selfplay_entropy_collapse_alert | B5 | alert_rules.py:36 | CORRECT | — | keep | N | fires <1.5 nats selfplay; banked 2.09-3.59 above floor |
| grad_norm_spike_alert | B5 | alert_rules.py:57 | CORRECT | — | keep | N | >10 NaN-precursor, NaN-guarded; banked max 2.88 |
| loss_increase_window_alert | B5 | alert_rules.py:68 | CORRECT | — | keep | N | 3 consecutive increases, off-by-one guarded |

---

## Coverage gaps (separated per PREREG §4 — absence-of-emit is a finding, not invented)

Most B1 per-source value-decomp, B3 cluster/depth, and ALL B4 eval stats are **absent from the §4 banked logs** — their verdicts are **STRUCTURAL** (source emit-site + threshold derived), NOT magnitude-measured. Do not read the master table as implying these magnitudes were observed.

- **No eval events in §4 banked logs.** Every B4 stat (`wr_*`, `ci_*`, `strength_aggregate`, `strength_cycle_density`, `sealbot_gate_passed`, `elo_estimate`, off-window arms) verdicted from emit site only — no live distribution. Both `events_*.jsonl` hold only game_complete/iteration_complete (short pre-eval runs).
- **`mcts_mean_depth` / `mcts_mean_root_concentration`** present in `iteration_complete` events, absent from `train_step` structlog rows. Root-concentration banked flat 0.545-0.576 (confirms SH-pinning).
- **§D-VALCEIL Q3 keys** (`value_accuracy_masked/_corpus/_selfplay`, `value_bce_*`, `value_rows_*`) + `value_loss_main/_uncertainty/_aux/_composite`, `full_search_frac`, `fp16_scale` — absent from banked `train_step` rows (logs predate feature / structlog-only).
- **Instrumentation-gated** (instrumentation_enabled=False): `colony_frac`/`extension_frac`/`neither_frac` + buffer-snapshot family, `value_probe_*`, `corpus_fraction`, `worker_draw_rate_per_worker`, `mv_*_range`/`mv_median_distinct`/`mv_spearman_*`, `draw_target_fraction`, terminal-fraction family.
- **Conditional-head** (off by flag): `ply_index_loss`, `per_source_grad_norm`, `opp_reply_loss` (aux off in newer log), early_game probe (failed in logged run).
- **Emits with no monitor consumer:** §D-VALCEIL per-source accuracy keys (train_step structlog only, not forwarded to dashboard event); `per_source_grad_norm`; `buffer_position_class_snapshot`; `temp_c` (gpu_stats structlog only).

---

## §1 note — promotion gate live path

The promotion gate's **live path is `wr_best` / `ci_best`** (`decide_promotion` → `wr_best≥0.55 AND ci_lo>0.5 AND bootstrap_floor`, gate_logic.py:120-122). The `strength_aggregate` ref-set producer is an **operator-gated follow-up, UNIMPLEMENTED at this pin** — no write site anywhere in `hexo_rl/`. Therefore the `gate_logic.py:57-62` branch where `strength_aggregate` REPLACES `wr_best` is **DEAD CODE**. Cross-checks the B4 bucket finding (bucket_B4.md §strength_aggregate; review.md i=108; red-team §C.2/§D.3): the replace-without-CI structure is the sharpest promotion-integrity exposure — a flawed aggregate would silently supplant the only CI-guarded gate. Until a producer exists, `promoted` is the legacy `wr_best` decision.

## S6 caveat — every wr_sealbot / wr_best is a PUCT-proxy of a Gumbel-trained net

Every `wr_*` arm runs the model's MCTS at eval time. The audited regime trains under **Gumbel** (`gumbel_mcts=true`), but the eval/deployment search regime decision has **not landed**. So every `wr_sealbot` / `wr_best` (and the gate built on them) is a **"PUCT-proxy of a Gumbel-trained net"** until the deployment-regime decision is made — the strength number measures the net under a planner that may differ from training. Outcome-based WRs are Gumbel-neutral (Axis-C pass) for the verdicts, but the deployment-regime mismatch is a standing construct caveat on the whole B4 strength surface.
