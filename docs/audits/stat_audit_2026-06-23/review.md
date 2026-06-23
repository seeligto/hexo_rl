# Phase 2 — REVIEW (fresh auditor, no Phase-1 bucket run)

Sandbox pin `52067631`. Source read-only from `/home/timmy/Work/Hexo/statAudit_wt`.
Empirical claims use ONLY §4 banked logs (read-only, in main repo).
Re-audited the frozen review subset (PREREG §5: 30% deterministic i%10∈{0,3,7} + every CORRECT + every WRONG)
against the 6-axis rubric. Each row CONFIRM or OVERRIDE.

## Independent source re-derivations performed (load-bearing)

- **strength_aggregate / strength_cycle_density (WRONG×2):** `EvalRoundResult` is a `TypedDict(total=False)`
  (result_types.py:51 — purely a static-type catalog, every key optional). grep across `hexo_rl/`+`engine/`+`scripts/`
  found NO producer (`results["strength_aggregate"]=` / `=...` assignment). Only `.get()` reads at
  eval_pipeline.py:390 and step_coordinator.py:1090. The two eval_pipeline lines 401/406 pass the value INTO a
  log/decision AFTER the .get() — not a producer. Docstring (result_types.py:97-99) names producer "operator-gated
  follow-up" = unbuilt. ⇒ keys NEVER populated; `decide_promotion` strength-replace branch (gate_logic.py:57-62)
  is DEAD (always falls to wr_best); cycle-density abort always sees 0.0. Phantom, Axis A. **Kills airtight.**
- **mcts_mean_root_concentration (WRONG):** selection.rs:123-126 — at root (cur==0) Gumbel sets `forced_root_child`
  and BYPASSES PUCT; the forced child gets the visit. So `max_child_visits/total` (mcts/mod.rs:199-212) is
  SH-schedule-pinned, not PUCT-confidence. Banked `mcts_root_concentration` flat 0.547 across 76 pts (events_cdf…),
  σ≈0.006 — exactly what an SH-pinned ratio predicts. Axis C, Gumbel-meaningless+unfixable. **Kill airtight.**
- **alt_spread (WRONG):** value_spread_canary.py:362-390 — when `encoding_spec is None` re-encode skipped,
  emits `alt_spread = float("nan")` sentinel as a metric value with no `alt_applicable` flag. Banked train_cdf…
  step 500: `alt_spread:NaN`, `value_spread_alt_skipped_plane_mismatch {alt_planes:8, model_in_channels:4}`,
  `alt_components.n=0`. Sentinel-as-value, Axis A. Seed7 INVERTED (handoff "does NOT NaN" FALSE). **Kill airtight.**
- **Redundancy identity checks:** frac_fullsearch_in_batch (`n_full/batch_n`, trainer.py:154) == full_search_frac
  (`(pvb&fs).sum()/pvb.numel()`, trainer.py:942-945) — same numerator AND denominator. model_version_range_size ==
  max-min (both in same game_complete event). spread == t3_spread (value_spread_canary.py:305 byte-copy). All confirmed.
- **mv_median_range (BIASED):** instrumentation.py:228 `ranges_sorted[n//2]` returns upper-of-two-middle for even n
  (≤+0.5 upward bias). Axis A weak. Confirmed.
- **Eval defaults (seed4/seed5 lens):** defaults.py:32-33 `EVAL_TEMPERATURE=0.5`, `RANDOM_OPENING_PLIES=4` break the
  §D-ARGMAX argmax+temp0+fixed-opening collapse for the temp-0.5 arms ⇒ wr_* CORRECT verdicts sound; seed5 INVERTED
  for live path is correct. win_rate = (wins+0.5·draws)/n (evaluator.py:231) — draw-half, correct for every wr_*.
  sealbot_gate_passed = wr>=0.5 (opponent_runners.py:155) — band defect quarantined in the flag, NOT on wr_sealbot.
- **Entropy bands (seed3):** check_entropy_collapse fires on `policy_entropy < 1.0` (alert_rules.py:31; config.py:35),
  selfplay < 1.5 (config.py:39) — LOW=collapse, correct direction. NOT the "≤2.6 backwards" band (lives at
  terminal_dashboard normalized-entropy, outside B5 alert sites). i=44 CORRECT sound.

## Per-subset adjudications

All 105 subset rows CONFIRMED. No overrides.

| i | name | phase1 | review | agree | note |
|---|---|---|---|---|---|
| 0 | alt_spread | WRONG | WRONG | yes | NaN skip-sentinel-as-value confirmed in source + banked step500. Kill airtight. |
| 2 | avg_game_length | CORRECT | CORRECT | yes | rolling 200-game mean; unit=compound turns, consistent w/ codebase. banked 21-41.8. |
| 3 | avg_sigma | CORRECT | CORRECT | yes | sqrt(pred var) mean, no band, descriptive. |
| 4 | axis_max | CORRECT | CORRECT | yes | qualitative label of dominant axis; the warn-band defect lives on axis_q/r (BIASED), not the label. |
| 7 | axis_s | CORRECT | CORRECT | yes | s=0.4479<0.45 baseline; per-axis formula fine; composite max-warn charged to axis_q/r. |
| 8 | batch_fill_pct | CORRECT | CORRECT | yes | lifetime cumulative occupancy %, correct formula. |
| 10 | buffer_capacity | CORRECT | CORRECT | yes | live capacity readout. |
| 11 | buffer_size | CORRECT | CORRECT | yes | live size; schema-collision is dashboard hygiene, not stat error. |
| 12 | cap_terminal_fraction | CORRECT | CORRECT | yes | ply_cap/total_games; coverage gap (instrumentation off). |
| 13 | chain_loss | CORRECT | CORRECT | yes | conditional head loss, correct. |
| 14 | ci_argmax_n | CORRECT | CORRECT | yes | Wilson n=20, honest width, no false CI-resolved claim. |
| 15 | ci_best | CORRECT | CORRECT | yes | Wilson n=200, the gating ci_lo>0.5 guard. |
| 16 | ci_bootstrap_anchor | CORRECT | CORRECT | yes | Wilson n=100; not consumed by floor gate (point-est). |
| 17 | ci_nnue | CORRECT | CORRECT | yes | Wilson n=100, honest. |
| 18 | ci_random | CORRECT | CORRECT | yes | Wilson n=20, sanity floor. |
| 19 | ci_sealbot | CORRECT | CORRECT | yes | Wilson n=50, no gate dep. |
| 20 | cluster_policy_disagreement_mean | BIASED | BIASED | yes | lifetime-cumulative mean logged raw per-iter (Axis A surfacing + E live-signal); honest n. fix=windowed delta. |
| 22 | cluster_variance_sample_count | CORRECT | CORRECT | yes | the eff-n witness/denominator for #4/#5; exact count, keep. |
| 23 | colony_extension_fraction | CORRECT | CORRECT | yes | per-game ext/total; 0.15 thresholds are B5 cosmetic, no band on emit. |
| 24 | colony_extension_stone_count | CORRECT | CORRECT | yes | per-game count, correct. |
| 25 | colony_extension_stone_total | CORRECT | CORRECT | yes | per-game denom, correct. |
| 26 | colony_frac | CORRECT | CORRECT | yes | n_col/n_total buffer snapshot; coverage gap. |
| 27 | colony_mean_value_target | CORRECT | CORRECT | yes | mean value_target over colony class; coverage gap. |
| 28 | colony_terminal_fraction | CORRECT | CORRECT | yes | colony/total_games; factual count. |
| 29 | colony_wins_argmax_n | CORRECT | CORRECT | yes | raw colony-win subset count. |
| 30 | colony_wins_best | CORRECT | CORRECT | yes | colony subset of win_count; anti-colony signal. |
| 31 | colony_wins_bootstrap_anchor | CORRECT | CORRECT | yes | raw colony subset count. |
| 32 | colony_wins_nnue | CORRECT | CORRECT | yes | raw colony subset count. |
| 33 | colony_wins_random | CORRECT | CORRECT | yes | raw colony subset count. |
| 34 | colony_wins_sealbot | CORRECT | CORRECT | yes | raw colony subset count. |
| 35 | corpus_fraction | CORRECT | CORRECT | yes | 1-(sp_pushed/size) buffer frac; coverage gap. |
| 36 | corpus_selfplay_frac | CORRECT | CORRECT | yes | 1-w_pre batch-mix weight; distinct from corpus_fraction. |
| 37 | cpu_util_pct | CORRECT | CORRECT | yes | psutil non-blocking; correct. |
| 38 | disk_free_gb | CORRECT | CORRECT | yes | shutil free/1e9; bands engineering-sound, config-driven. |
| 39 | draw_rate | CORRECT | CORRECT | yes | draws/games lifetime; banked 0-0.153. p0+p1+dr=1.0. |
| 40 | draw_target_fraction | CORRECT | CORRECT | yes | live-config-derived window (PIPE-4 fix); coverage gap. |
| 41 | early_game_entropy_mean | CORRECT | CORRECT | yes | legal-renorm Shannon; warn>4.5 (high=warn, correct for early probe); probe failed in banked. |
| 42 | early_game_top1_mass_mean | CORRECT | CORRECT | yes | max legal-renorm prob; coverage gap. |
| 43 | elo_estimate | CORRECT | CORRECT | yes | BT point estimate, anchored ckpt0; CI surfaced elsewhere (advisory). |
| 44 | entropy_collapse_alert | CORRECT | CORRECT | yes | fires <1.0 nats (low=collapse, correct dir); NOT the seed3 backwards band. |
| 45 | eval_games | CORRECT | CORRECT | yes | sum n across arms; coverage counter. |
| 46 | eval_round_wall_sec | CORRECT | CORRECT | yes | diagnostic timing, no band. |
| 47 | extension_frac | CORRECT | CORRECT | yes | n_ext/n_total; coverage gap. |
| 48 | extension_mean_value_target | CORRECT | CORRECT | yes | mean value_target over ext class; coverage gap. |
| 49 | fp16_scale | CORRECT | CORRECT | yes | GradScaler loss scale. |
| 50 | frac_fullsearch_in_batch | REDUNDANT | REDUNDANT | yes | identical num+denom to full_search_frac (verified). canonical=full_search_frac. |
| 51 | full_search_frac | CORRECT | CORRECT | yes | canonical form in primary train_step structlog. |
| 52 | g4_value_head_band_pass | CORRECT | CORRECT | yes | band flag for fc2 weight; warning-only, regime-derived. |
| 53 | games_per_hour | CORRECT | CORRECT | yes | 60s sliding rate; banked 240-420. |
| 54 | games_this_iter | CORRECT | CORRECT | yes | delta since last iteration_complete. |
| 55 | games_total | CORRECT | CORRECT | yes | monotone cumulative count. |
| 56 | gpu_util_pct | CORRECT | CORRECT | yes | pynvml util.gpu; schema collision is hygiene. |
| 57 | grad_norm | CORRECT | CORRECT | yes | L2 grad norm; alert>10 generous ceiling (banked max 2.88). |
| 58 | grad_norm_spike_alert | CORRECT | CORRECT | yes | >10 NaN-precursor heuristic, NaN-guarded. |
| 59 | loss | CORRECT | CORRECT | yes | total loss scalar, no band. |
| 60 | loss_increase_window_alert | CORRECT | CORRECT | yes | 3 consecutive increases; off-by-one guarded. |
| 61 | lr | CORRECT | CORRECT | yes | optimizer param_groups[0]['lr']. |
| 62 | mcts_mean_depth | CORRECT | CORRECT | yes | interior-PUCT descent depth (only root SH-forced); meaningful under Gumbel. banked 3.75. |
| 63 | mcts_mean_root_concentration | WRONG | WRONG | yes | SH-pinned max/total (selection.rs:124-126 root PUCT bypass); banked flat 0.547. Axis C. Kill airtight. |
| 64 | mcts_quiescence_fires | CORRECT | CORRECT | yes | 4-branch leaf override count, reset-per-move verified, no double count. |
| 65 | mean_colony | CORRECT | CORRECT | yes | colony-bank mean V; honest component, no band. |
| 66 | mean_ext | CORRECT | CORRECT | yes | ext-bank mean V; honest component, no band. |
| 67 | model_version_distinct | CORRECT | CORRECT | yes | Rust distinct-set count (uses distinct, not move count). |
| 68 | model_version_max | CORRECT | CORRECT | yes | per-game max version. |
| 69 | model_version_min | CORRECT | CORRECT | yes | per-game min version. |
| 70 | model_version_range_size | REDUNDANT | REDUNDANT | yes | == max-min, both in same event (verified all 381 games). |
| 71 | mv_max_range | CORRECT | CORRECT | yes | ranges_sorted[-1]; coverage gap. |
| 73 | mv_median_range | BIASED | BIASED | yes | ranges_sorted[n//2] upper-half bias ≤+0.5 even n. Axis A weak. fix=(n-1)//2. |
| 74 | mv_p90_range | CORRECT | CORRECT | yes | int(n*0.9)-1 index, numpy-matching; coverage gap. |
| 75 | mv_spearman_rho_range_vs_draw | CORRECT | CORRECT | yes | spearmanr(ranges, is_draw), n>=10 guard. |
| 76 | n_rows_policy_loss | CORRECT | CORRECT | yes | full-search valid-policy count denom. |
| 77 | n_rows_total | CORRECT | CORRECT | yes | all policy-valid count denom. |
| 78 | neither_frac | CORRECT | CORRECT | yes | partition complement; coverage gap. |
| 79 | offwindow_forced_win_rate | CORRECT | CORRECT | yes | completing-cell (turn-correct) off-window rate; right adversarial instrument. |
| 80 | offwindow_strict_forced_rate | CORRECT | CORRECT | yes | stricter conditioned subset, distinct denom-condition. |
| 81 | opp_reply_loss | CORRECT | CORRECT | yes | raw unweighted aux head loss; conditional. |
| 82 | ownership_loss | CORRECT | CORRECT | yes | conditional ownership head loss. |
| 83 | per_source_grad_norm | CORRECT | CORRECT | yes | L2 per slice×group; Track-B diagnostic-mode-only, coverage gap. |
| 84 | ply_index_loss | CORRECT | CORRECT | yes | conditional head loss. |
| 85 | policy_entropy | CORRECT | CORRECT | yes | Shannon over model output; band 1.0/2.0 nats (low=collapse, planner-indep). banked 2.14-2.89. |
| 86 | policy_entropy_pretrain | CORRECT | CORRECT | yes | entropy over corpus rows; NaN-guarded. |
| 87 | policy_entropy_recent | CORRECT | CORRECT | yes | recent-buffer slice entropy. |
| 89 | policy_entropy_uniform_selfplay | CORRECT | CORRECT | yes | uniform-slice entropy, fallback to selfplay. |
| 90 | policy_loss | CORRECT | CORRECT | yes | CE vs Gumbel completed-Q improved policy; well-defined. |
| 91 | policy_target_entropy | CORRECT | CORRECT | yes | entropy of improved-policy targets; Gumbel-valid. |
| 92 | policy_target_entropy_fastsearch | CORRECT | CORRECT | yes | fastsearch subset entropy; NaN-guarded. |
| 93 | policy_target_entropy_fullsearch | CORRECT | CORRECT | yes | fullsearch subset entropy; NaN-guarded. |
| 94 | policy_target_kl_uniform_fastsearch | CORRECT | CORRECT | yes | KL=logN-H over fastsearch rows. |
| 95 | policy_target_kl_uniform_fullsearch | CORRECT | CORRECT | yes | KL=logN-H over fullsearch rows. |
| 96 | positions_per_hour | CORRECT | CORRECT | yes | gph*avg_gl derived throughput. |
| 97 | promoted | CORRECT | CORRECT | yes | strength_ok AND robustness_ok; fail-safe on missing arms; legacy wr_best path (strength-replace is dead — charged to i=108). |
| 100 | row_max_density | CORRECT | CORRECT | yes | max stones on any axis-row per game; banked 7-18. |
| 101 | rss_gb | CORRECT | CORRECT | yes | process RSS/1e9. |
| 103 | selfplay_entropy_collapse_alert | CORRECT | CORRECT | yes | fires <1.5 nats selfplay (low=collapse); banked 2.09-3.59 safely above. |
| 105 | sims_per_sec | CORRECT | CORRECT | yes | n_sims*games/elapsed per drain; banked 3426-7428. |
| 106 | six_terminal_fraction | CORRECT | CORRECT | yes | six_in_a_row/total_games. |
| 107 | spread | REDUNDANT | REDUNDANT | yes | byte-copy of t3_spread (value_spread_canary.py:305). canonical=t3_spread. |
| 108 | strength_aggregate | WRONG | WRONG | yes | phantom: TypedDict field, NO producer, only .get(); dead gate branch. Axis A. Kill airtight. |
| 109 | strength_cycle_density | WRONG | WRONG | yes | phantom: NO producer, abort always sees 0.0. Axis A. Kill airtight. |
| 110 | strength_regression_abort_alert | BIASED | BIASED | yes | no-CI trigger + uncalibrated floor placeholder; default-off keeps from WRONG. NOTE: input fed by phantom i=108 — but logic math correct & sound w/ producer, so BIASED-fix is the right bin. |
| 111 | stride5_run_p90 | CORRECT | CORRECT | yes | rolling P90 over 50 games; inventory aggregates-label wrong is metadata-only. |
| 113 | terminal_reason | CORRECT | CORRECT | yes | u8→string map, correct. |
| 114 | threat_loss | CORRECT | CORRECT | yes | conditional head loss. |
| 115 | uncertainty_loss | CORRECT | CORRECT | yes | raw heteroscedastic loss; banked ~0.49. |
| 117 | value_accuracy_corpus | CORRECT | CORRECT | yes | _mean_over(correct, is_corpus); per-source split. |
| 118 | value_accuracy_masked | CORRECT | CORRECT | yes | decided+supervised-only winner-call accuracy (the clean stat). |
| 119 | value_accuracy_selfplay | CORRECT | CORRECT | yes | _mean_over(correct, is_selfplay). |
| 120 | value_bce_corpus | CORRECT | CORRECT | yes | per-source BCE, supervised-masked. |
| 121 | value_bce_selfplay | CORRECT | CORRECT | yes | per-source BCE, supervised-masked. |
| 122 | value_fc2_weight_abs_max | CORRECT | CORRECT | yes | weight-mag probe; band 0.308±50%, warning-only, no lever Goodharts it. |
| 123 | value_loss | CORRECT | CORRECT | yes | BCE-with-logits; canonical (value_loss_main is its alias). |
| 127 | value_loss_uncertainty | CORRECT | CORRECT | yes | unc_weight*uncertainty_loss weighted component. |
| 128 | value_probe_decisive_mean | CORRECT | CORRECT | yes | fixture mean V on decisive positions; coverage gap; key=decisive_mean. |
| 129 | value_probe_decisive_std | CORRECT | CORRECT | yes | fixture std; coverage gap. |
| 130 | value_probe_draw_mean | CORRECT | CORRECT | yes | fixture mean V on draw positions; coverage gap. |
| 131 | value_probe_draw_std | CORRECT | CORRECT | yes | fixture std; coverage gap. |
| 132 | value_rows_corpus | CORRECT | CORRECT | yes | corpus row count in batch. |
| 133 | value_rows_corpus_supervised | CORRECT | CORRECT | yes | corpus∩supervised denom for value_bce_corpus. |
| 134 | value_rows_masked | CORRECT | CORRECT | yes | supervised∩decided count, denom for value_accuracy_masked. |
| 135 | value_rows_selfplay | CORRECT | CORRECT | yes | batch_n - n_pre complement count. |
| 136 | value_rows_selfplay_supervised | CORRECT | CORRECT | yes | selfplay∩supervised denom for value_bce_selfplay. |
| 137 | vram_used_gb | CORRECT | CORRECT | yes | mem.used/1e9; banked ~1.01. |
| 138 | win_rate_p0 | CORRECT | CORRECT | yes | x_wins/games lifetime; distinct games, no argmax collapse. |
| 139 | win_rate_p1 | CORRECT | CORRECT | yes | o_wins/games lifetime. |
| 140 | worker_draw_rate_per_worker | CORRECT | CORRECT | yes | rolling 50-game draw rate/worker; coverage gap. |
| 141 | wr_argmax_n | CORRECT | CORRECT | yes | model n_sims=1 (argmax) vs SealBot; survives collapse via opening_plies=4+color-alt; DRIFT detector. |
| 142 | wr_best | CORRECT | CORRECT | yes | draw-half WR vs champion, n=200; the gating instrument w/ ci_lo>0.5. |
| 143 | wr_bootstrap_anchor | CORRECT | CORRECT | yes | draw-half WR vs frozen anchor, n=100; floor-gate lacks CI (advisory). |
| 144 | wr_nnue | CORRECT | CORRECT | yes | draw-half WR vs Hammerhead, n=100, default-off arm. |
| 145 | wr_random | CORRECT | CORRECT | yes | draw-half WR vs random, n=20 floor check. |
| 146 | wr_sealbot | CORRECT | CORRECT | yes | draw-half raw WR vs SealBot, n=50 monitor; band defect quarantined in sealbot_gate_passed flag. |
| 147 | wr_sealbot_sliding_window_alert | BIASED | BIASED | yes | no-CI/no-n_eff wr trigger (B) + re-arms §D-FOUNDING-flagged instrument as primary (E); consec-N mitigates. |

## Summary

- Subset size re-audited: 105 rows.
- Overrides: 0.
- All 4 WRONG kills (i=0, 63, 108, 109) independently re-derived from source + corroborated by §4 banked
  logs (alt_spread NaN at step500; root_concentration flat 0.547; strength_* no-producer phantom). Airtight —
  safe to feed S7 deletion.
- All CORRECT verdicts survive the rubber-stamp guard: bands checked for direction (entropy floors correct,
  not the seed3 backwards band); eval arms checked against the seed4/seed5 instrument-distribution lens
  (temp-0.5 + 4 opening plies break argmax collapse → n_eff not collapsed); colony_wins/value_rows confirmed
  as honest counts not duplicate emits.
- BIASED/REDUNDANT verdicts confirmed; redundancy identities (frac_fullsearch≡full_search_frac,
  range_size≡max-min, spread≡t3_spread) verified at source.
- One cross-link flagged (not an override): i=110 strength_regression_abort_alert is fed by the phantom
  i=108 strength_aggregate, but its own logic is correct and it is default-OFF — BIASED-fix remains the
  correct, most-severe-appropriate bin (the construct is sound once a producer exists; not WRONG).
