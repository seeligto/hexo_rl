# Bucket B2 — Self-Play Stats Audit

**Auditor:** sonnet (Phase 1, B2 assignment)
**Source root:** `/home/timmy/Work/Hexo/statAudit_wt/` (frozen SHA 52067631)
**Banked logs read:** `events_cdf24392b8414486a28424673f221575.jsonl` (2335 lines), `events_76cf0f3925ab45e889c7ff12a5b6b2ea.jsonl` (623 lines), `train_cdf24392b8414486a28424673f221575.jsonl` (3054 lines)

---

## Per-stat verdicts

---

### win_rate_p0
- **Emit site:** `hexo_rl/training/events.py:251`
- **Event:** `iteration_complete`
- **A — Formula:** `round(float(pool.x_winrate), 4)` where `x_winrate = x_wins / games_completed` (pool.py:455-458). Lifetime cumulative. Correct.
- **B — Eff-n:** Each self-play game is a distinct playthrough (unique UUID/moves). Lifetime cumulative n = games_total. No argmax collapse. No CI needed (diagnostic monitoring, not comparison stat). PASS.
- **C — Planner-semantics:** Game outcome; no planner-specific meaning. PASS.
- **D — Band calibration:** No band (has_band=false). PASS.
- **E — Construct validity:** Measures what it claims: P0 (X) win fraction. PASS.
- **F — Redundancy:** Not duplicate of any other single stat (draw_rate and win_rate_p1 are different). PASS.
- **Empirical:** 76 banked `iteration_complete` events; range 0.36–0.51. Sum p0+p1+dr = 1.0000±0.0005 (rounding). Correct.
- **Verdict: CORRECT — keep**

---

### win_rate_p1
- **Emit site:** `hexo_rl/training/events.py:252`
- **Event:** `iteration_complete`
- **A — Formula:** `round(float(pool.o_winrate), 4)` where `o_winrate = o_wins / games_completed` (pool.py:461-464). Correct.
- **B–F:** Same analysis as `win_rate_p0`. PASS all axes.
- **Empirical:** Range 0.38–0.60 in banked sample. Correct.
- **Verdict: CORRECT — keep**

---

### draw_rate
- **Emit site:** `hexo_rl/training/events.py:253`
- **Event:** `iteration_complete`
- **A — Formula:** `round(float(pool.draws / games_played), 4) if games_played > 0 else 0.0`. Lifetime cumulative draws / games. Correct.
- **B — Eff-n:** Lifetime cumulative. Distinct games. No CI needed. PASS.
- **C — Planner-semantics:** Outcome metric; no planner-specific meaning. PASS.
- **D — Band calibration:** No band. PASS.
- **E — Construct validity:** Draws (ply-cap + colony + organic) / total. Measures cumulative draw rate. PASS.
- **F — Redundancy:** Not redundant (different from p0/p1). PASS.
- **Empirical:** Range 0.0–0.153 in banked sample. Correct.
- **Verdict: CORRECT — keep**

---

### games_total
- **Emit site:** `hexo_rl/training/events.py:246`
- **Event:** `iteration_complete`
- **A — Formula:** `games_played` = `pool.games_completed` (cumulative count from Rust runner). Correct counter.
- **B–F:** Monotone counter; no aggregation concern; no planner semantics; no band; measures what it claims; not duplicate of `games_this_iter` (different quantity). PASS all.
- **Verdict: CORRECT — keep**

---

### games_this_iter
- **Emit site:** `hexo_rl/training/events.py:247`
- **Event:** `iteration_complete`
- **A — Formula:** `games_played - last_iter_games`. Delta since last `iteration_complete` event. Correct.
- **B–F:** PASS all axes. Not redundant with `games_total`.
- **Verdict: CORRECT — keep**

---

### games_per_hour
- **Emit site:** `hexo_rl/training/events.py:248`
- **Event:** `iteration_complete`
- **A — Formula:** `RollingGamesPerHour.update()` (loop.py:49-59). 60-second sliding window; rate = (dg/dt)*3600 where dg=delta games, dt=window elapsed. Correctly normalized per-hour. PASS.
- **B — Eff-n:** Throughput metric, not a comparison stat. No n_eff concern. PASS.
- **C–F:** No planner semantics; no band; measures throughput correctly; not redundant with `positions_per_hour` (different unit). PASS.
- **Empirical:** 420.7, 241.9, 240.4 in banked sample. Consistent with observed self-play rates.
- **Verdict: CORRECT — keep**

---

### positions_per_hour
- **Emit site:** `hexo_rl/training/events.py:249`
- **Event:** `iteration_complete`
- **A — Formula:** `gph * avg_gl if avg_gl > 0 else 0.0` (events.py:236). Derived: games/hr * avg_game_length (avg stones placed per game). Correct.
- **B–F:** PASS all. Not redundant: combines two distinct signals into a unified throughput metric.
- **Verdict: CORRECT — keep**

---

### avg_game_length
- **Emit site:** `hexo_rl/training/events.py:250`
- **Event:** `iteration_complete`
- **A — Formula:** Rolling mean over last 200 games (deque maxlen=200; pool.py:406,731). `sum(game_lengths) / len(game_lengths)`. Correct rolling mean.
- **B — Eff-n:** 200-game window; distinct games. PASS.
- **C — Planner-semantics:** Game length is ply count / 2 (compound moves; pool.py:729). Unit is compound turns. Named `avg_game_length` — no unit label in event. Minor ambiguity (turns vs plies) but consistent with codebase convention. PASS.
- **D–F:** No band; measures correctly; not redundant. PASS.
- **Empirical:** Range 21.0–41.8 in banked sample. Plausible.
- **Verdict: CORRECT — keep**

---

### sims_per_sec
- **Emit site:** `hexo_rl/training/events.py:254`
- **Event:** `iteration_complete`
- **A — Formula:** `n_simulations * len(games_batch) / elapsed` (pool.py:717-720). Instantaneous rate per drain cycle (replaces previous value each drain). Correct computation for that interval. `aggregates=true` in inventory is misleading — it covers multiple games in one drain but is NOT a rolling or cumulative average; each reading replaces the last. Naming is accurate (sims per wall-second).
- **B — Eff-n:** Throughput metric; not a comparison. No CI concern. PASS.
- **C–F:** No planner semantics; no band; measures throughput correctly; not redundant. PASS.
- **Empirical:** Range 3426–7428 in banked sample. Consistent with RTX 4060 self-play rates.
- **Verdict: CORRECT — keep**

---

### buffer_size
- **Emit site:** `hexo_rl/training/events.py:255`
- **Event:** `iteration_complete`
- **A — Formula:** `buffer.size` (live ReplayBuffer size). Correct.
- **B–F:** No aggregation; no planner semantics; no band; measures buffer occupancy; not redundant with `buffer_capacity`.
- **Note:** Also emitted in `system_stats` event from pool.py:811 (separate buffer-update pathway) and in `train_step_summary` structlog (events.py:287). The pool.py secondary emit creates a schema collision: `system_stats` from pool.py carries `{buffer_size, buffer_capacity}` while `system_stats` from gpu_monitor.py carries `{gpu_util_pct, vram_used_gb, ram_*, cpu_util_pct, rss_gb}`. Dashboard consumers cannot distinguish them by event name alone. The primary emit (events.py:255) is clean.
- **Verdict: CORRECT — keep** (primary emit; schema collision is a dashboard hygiene issue separate from stat correctness)

---

### buffer_capacity
- **Emit site:** `hexo_rl/training/events.py:256`
- **Event:** `iteration_complete`
- **A — Formula:** `buffer.capacity`. Correct.
- **B–F:** PASS all. Same schema-collision caveat as `buffer_size`.
- **Verdict: CORRECT — keep**

---

### corpus_selfplay_frac
- **Emit site:** `hexo_rl/training/events.py:257`
- **Event:** `iteration_complete`
- **A — Formula:** `round(1.0 - w_pre, 4)` where `w_pre` is the pretrained-corpus mixing weight. This is the **selfplay fraction of the training batch mix**, not the fraction of the replay buffer that is selfplay data. The name `corpus_selfplay_frac` is slightly ambiguous (could be read as "fraction of corpus that is selfplay" or "selfplay fraction of corpus mix"). The formula is unambiguous and consistent with codebase intent.
- **B–F:** No aggregation concern; no planner semantics; no band; construct validity OK (training mix selfplay weight); not redundant with `corpus_fraction` (buffer_composition event, different quantity: buffer occupancy fraction). PASS.
- **Verdict: CORRECT — keep**

---

### batch_fill_pct
- **Emit site:** `hexo_rl/training/events.py:258`
- **Event:** `iteration_complete`
- **A — Formula:** `min((reqs / (fwd * max(bs, 1))) * 100.0, 100.0)` where `reqs=total_requests, fwd=forward_count, bs=batch_size` (pool.py:445-452). This is the lifetime-cumulative average batch occupancy as percentage. Formula is correct for that interpretation. Emitted without rounding (banked log shows full float: `50.86220349563047`). Not rounded at emit site unlike most other stats in this event.
- **B–F:** Throughput diagnostic; no CI needed; no planner semantics; no band; not redundant. PASS.
- **Verdict: CORRECT — keep**

---

### quiescence_fires_per_step
- **Emit site:** `hexo_rl/training/events.py:228`
- **Event:** `training_step`
- **A — Formula:** `qfire_delta = _cur_qfire - self._last_quiescence_fires` (step_coordinator.py:1266-1268). This is the delta in Rust quiescence-fire counter **since the last log-interval boundary**, emitted every `log_interval=10` steps. The stat is named `quiescence_fires_per_step` but the value is fires-per-10-steps (fires-per-interval). It is NOT divided by `log_interval` before emit. **Axis A naming fail:** a consumer reading the name assumes per-step rate; the actual value is 10× higher than a per-step rate. Banked sample: values 2452, 1071, 1382, 940, 435 → true per-step rates 245, 107, 138, 94, 43. **Fix:** divide by `log_interval` at emit, or rename to `quiescence_fires_this_interval`.
- **B–F:** PASS (no aggregation concern beyond the naming issue; no planner semantics; no band; construct valid conditional on correct interpretation; not redundant).
- **Verdict: BIASED — fix** (divide `qfire_delta` by `log_interval` before emit, or rename)

---

### colony_extension_stone_count
- **Emit site:** `hexo_rl/selfplay/pool.py:770`
- **Event:** `game_complete`
- **A — Formula:** `_ext_count` from `_compute_colony_extension(move_history)` (instrumentation.py:86-114). Counts stones at hex-distance > 6 from any opponent stone. Correct implementation.
- **B — Eff-n:** Per-game scalar; no aggregation. PASS.
- **C — Planner-semantics:** Board geometry metric; no planner semantics. PASS.
- **D — Band calibration:** No band. PASS.
- **E — Construct validity:** Correctly measures colony extension stone count for the completed game. PASS.
- **F — Redundancy:** Not fully redundant with `colony_extension_stone_total` or `colony_extension_fraction` (different quantities). PASS.
- **Empirical:** Values 0 in early games (early training, few stones far from opponent). Correct.
- **Verdict: CORRECT — keep**

---

### colony_extension_stone_total
- **Emit site:** `hexo_rl/selfplay/pool.py:771`
- **Event:** `game_complete`
- **A — Formula:** `_ext_total` = denominator from `_compute_colony_extension` (instrumentation.py:106-113). Count of stones with at least one opponent stone on board (excludes uncontested single-player positions). Correct.
- **B–F:** Per-game; no planner semantics; no band; measures correctly; not redundant with count/fraction.
- **Empirical:** 25–41 in banked sample (consistent with game lengths 21–42 compound moves * 2 stones).
- **Verdict: CORRECT — keep**

---

### colony_extension_fraction
- **Emit site:** `hexo_rl/selfplay/pool.py:772`
- **Event:** `game_complete`
- **A — Formula:** `ext_count / ext_total if ext_total > 0 else 0.0` (instrumentation.py:178). Correct per-game fraction.
- **B — Eff-n:** Per-game (aggregates=false, confirmed). PASS.
- **C — Planner-semantics:** Geometric metric. PASS.
- **D — Band calibration:** No band on this stat (has_band=false). The 0.15/0.25 thresholds documented in PREREG §3 seed6 appear in monitoring scripts (`check_phase_c.py:187`, `d1m_monitor.py:779`) NOT in this stat's emit path. Seed 6 re-derived: thresholds are in B5-layer monitoring scripts, not enforced gates in B2 emit. PASS.
- **E — Construct validity:** Fraction of stones colonially extended this game. Directly measures the mechanism of interest. PASS.
- **F — Redundancy:** With `stone_count` and `stone_total`, fraction is derivable but not emitted as a raw fraction elsewhere. PASS.
- **Empirical:** 0.0 in all 381 banked game_complete events (early training, no colony behavior yet). Formula present and correct.
- **Verdict: CORRECT — keep**

---

### terminal_reason
- **Emit site:** `hexo_rl/selfplay/pool.py:776`
- **Event:** `game_complete`
- **A — Formula:** `_TR_NAMES.get(int(terminal_reason), "unknown")` (pool.py:755-756). Maps Rust u8 {0→"six_in_a_row", 1→"colony", 2→"ply_cap", 3→"other_draw"} to string. Correct.
- **B–F:** Per-game categorical; no aggregation; no planner semantics; no band; measures terminal condition correctly; not redundant (terminal_reason_counts in instrumentation accumulates separately).
- **Empirical:** Values {"ply_cap", "six_in_a_row"} in banked sample. No "colony" or "other_draw" (early training, consistent with expected regime).
- **Verdict: CORRECT — keep**

---

### model_version_min
- **Emit site:** `hexo_rl/selfplay/pool.py:777`
- **Event:** `game_complete`
- **A — Formula:** `int(mv_min)` from Rust drain tuple (pool.py:726-727). Minimum model version used across all moves in one game. Correct.
- **B–F:** Per-game; no planner semantics; no band; staleness diagnostic; not redundant with max/distinct/range_size (different quantity).
- **Empirical:** All 0 in banked sample (single model version 0 throughout early training). Correct.
- **Verdict: CORRECT — keep**

---

### model_version_max
- **Emit site:** `hexo_rl/selfplay/pool.py:778`
- **Event:** `game_complete`
- **A — Formula:** `int(mv_max)` from Rust drain tuple. Maximum model version used in one game. Correct.
- **B–F:** Same as `model_version_min`. PASS.
- **Verdict: CORRECT — keep**

---

### model_version_distinct
- **Emit site:** `hexo_rl/selfplay/pool.py:779`
- **Event:** `game_complete`
- **A — Formula:** `int(mv_distinct)` from Rust drain. Count of distinct model versions used across all moves in one game. Correct (Rust computes distinct set).
- **B — Eff-n:** This IS the distinct count — correctly uses distinct versions, not move count. PASS.
- **C–F:** No planner semantics; no band; measures correctly; not redundant.
- **Verdict: CORRECT — keep**

---

### model_version_range_size
- **Emit site:** `hexo_rl/selfplay/pool.py:780`
- **Event:** `game_complete`
- **A — Formula:** `int(mv_max - mv_min)` (pool.py:780). Verified: range_size == max - min for all 381 banked games. Correct.
- **B–F:** Per-game; no planner semantics; no band; staleness metric; **Axis F:** trivially derivable from `model_version_max - model_version_min` (both emitted in same event). Mild redundancy but emitted for dashboard convenience.
- **Verdict: REDUNDANT** with `model_version_max` + `model_version_min` (trivially computed). **Canonical:** `model_version_max - model_version_min`. **Action: drop** if payload size matters; keep for dashboard convenience otherwise.

---

### stride5_run_p90
- **Emit site:** `hexo_rl/selfplay/pool.py:782`
- **Event:** `game_complete`
- **A — Formula:** Rolling P90 of `stride5_run_max` over last 50 games (instrumentation.py:173-174). Formula: `sr_hist[max(0, int(len(sr_hist)*0.9)-1)]`. Verified correct for typical n (matches numpy-style P90 index at n=10,20,50). `aggregates=false` in inventory is **wrong** — this is a rolling aggregate over 50 games, emitted per-game. The stat itself emitted as per-game event but its value reflects a 50-game window.
- **B — Eff-n:** Window = 50 distinct games. Reasonable. PASS.
- **C — Planner-semantics:** Stride-5 geometry detector; no planner semantics. PASS.
- **D — Band calibration:** No band. PASS.
- **E — Construct validity:** Measures worst-case stride-5 pattern prevalence in recent games. PASS.
- **F — Redundancy:** Not redundant. PASS.
- **Empirical:** Values 2–4 in banked sample. Consistent with low-colony early training.
- **Verdict: CORRECT — keep** (inventory `aggregates` label is wrong but that's a metadata error, not a stat error; formula is correct)

---

### row_max_density
- **Emit site:** `hexo_rl/selfplay/pool.py:784`
- **Event:** `game_complete`
- **A — Formula:** `int(_row_max_density)` from `_compute_stride5_metrics` return value (instrumentation.py:63, pool.py:735). Maximum stone count on any single hex row in any of three axes (q/r/s). Correct per-game computation.
- **B–F:** Per-game; no planner semantics; no band; valid density metric; not redundant.
- **Empirical:** Range 7–18 in banked sample. Plausible for games 21–42 compound turns.
- **Verdict: CORRECT — keep**

---

### corpus_fraction
- **Emit site:** `hexo_rl/training/step_coordinator.py:1346`
- **Event:** `buffer_composition`
- **A — Formula:** `max(0.0, 1.0 - (sp_pushed / buffer.size))` (pool.py:567). Fraction of live buffer from preloaded corpus (not selfplay). Correct. Gated by `cfg.instrumentation_enabled` (step_coordinator.py:1312) — **absent from both banked logs** (instrumentation disabled in these runs).
- **B — Eff-n:** Buffer-snapshot metric; no n_eff concern. PASS.
- **C–F:** No planner semantics; no band; correctly measures buffer corpus fraction; not redundant with `corpus_selfplay_frac` (different dimension: buffer vs batch mix).
- **Coverage gap:** Never emitted in banked sample. Emit site exists (step_coordinator.py:1344-1349) but `instrumentation_enabled=False` gates it.
- **Verdict: CORRECT — keep** (coverage gap noted; stat formula is correct when it fires)

---

### worker_draw_rate_per_worker
- **Emit site:** `hexo_rl/training/step_coordinator.py:1353`
- **Event:** `worker_draw_rate` (field `per_worker`)
- **A — Formula:** Rolling last-50-game draw rate per worker (instrumentation.py:197-204). `sum(dq)/len(dq)` over deque(maxlen=50). Correct rolling mean. Gated by same `instrumentation_enabled` — **absent from both banked logs**.
- **B — Eff-n:** 50 games per worker window; distinct games. PASS.
- **C–F:** No planner semantics; no band; measures per-worker draw rate correctly; not redundant.
- **Coverage gap:** Never emitted in banked sample.
- **Verdict: CORRECT — keep** (coverage gap noted)

---

### mv_median_range
- **Emit site:** `hexo_rl/training/step_coordinator.py:1358`
- **Event:** `model_version_summary` (field `median_range`)
- **A — Formula:** `ranges_sorted[n // 2]` (instrumentation.py:228). For even n this returns the **upper** of the two middle elements, not the true median (which averages both). Upward bias of up to +0.5 integer units. Fix: use `ranges_sorted[(n-1) // 2]` for lower median or interpolate. **Axis A weak fail.** Gated by `instrumentation_enabled` — **absent from both banked logs**.
- **B–F:** Coverage gap; formula bias is only Axis A concern. No planner semantics; no band; staleness diagnostic; not redundant.
- **Coverage gap:** Never emitted in banked sample.
- **Verdict: BIASED — fix** (use `(n-1)//2` or average of two middle elements)

---

### mv_p90_range
- **Emit site:** `hexo_rl/training/step_coordinator.py:1358`
- **Event:** `model_version_summary` (field `p90_range`)
- **A — Formula:** `ranges_sorted[max(0, int(n * 0.9) - 1)]` (instrumentation.py:229). Verified: matches numpy-style P90 index for n ∈ {10, 20, 50, 100, 200}. Correct. Gated — **absent from banked logs**.
- **B–F:** PASS. Coverage gap.
- **Verdict: CORRECT — keep** (coverage gap noted)

---

### mv_max_range
- **Emit site:** `hexo_rl/training/step_coordinator.py:1358`
- **Event:** `model_version_summary` (field `max_range`)
- **A — Formula:** `ranges_sorted[-1]` (instrumentation.py:230). Max of per-game version ranges over rolling 200 games. Correct. Gated — **absent from banked logs**.
- **B–F:** PASS. Coverage gap.
- **Verdict: CORRECT — keep** (coverage gap noted)

---

### mv_median_distinct
- **Emit site:** `hexo_rl/training/step_coordinator.py:1358`
- **Event:** `model_version_summary` (field `median_distinct`)
- **A — Formula:** `distincts_sorted[n // 2]` (instrumentation.py:231). Same upper-half median bias as `mv_median_range` — upward bias ≤ +0.5 integer units for even n. **Axis A weak fail.** Gated — **absent from banked logs**.
- **B–F:** Coverage gap; formula bias is the only Axis A concern.
- **Verdict: BIASED — fix** (use `(n-1)//2`)

---

### mv_spearman_rho_range_vs_draw
- **Emit site:** `hexo_rl/training/step_coordinator.py:1358`
- **Event:** `model_version_summary` (field `spearman_rho_range_vs_draw`)
- **A — Formula:** `scipy.stats.spearmanr(ranges, is_draw)` where `is_draw = 1 if winner_code == 0 else 0` (instrumentation.py:232-240). winner_code==0 = draw (pool.py:612: `_WINNER_NAMES = ("draw","x","o")`). n>=10 guard in place. Correct formula.
- **B — Eff-n:** 200-game rolling window; n>=10 guard; no CI reported (diagnostic correlation). PASS.
- **C–F:** No planner semantics; no band; construct valid; not redundant. PASS. Gated — **absent from banked logs**.
- **Verdict: CORRECT — keep** (coverage gap noted)

---

### gpu_util_pct
- **Emit site:** `hexo_rl/monitoring/gpu_monitor.py:119`
- **Event:** `system_stats` (gpu-variant)
- **A — Formula:** `float(util.gpu)` from `pynvml` (gpu_monitor.py:100). Standard GPU utilization percentage. Correct.
- **B–F:** System poll metric; no aggregation concern; no planner semantics; no band; measures GPU correctly; not redundant.
- **Schema note:** `system_stats` event name collides with pool.py:810-813 which emits `system_stats` with `{buffer_size, buffer_capacity}`. Dashboard consumers see two schemas under one event name.
- **Empirical:** 859 system_stats events with gpu fields in banked sample. Values 62.0 in sample.
- **Verdict: CORRECT — keep**

---

### vram_used_gb
- **Emit site:** `hexo_rl/monitoring/gpu_monitor.py:119`
- **Event:** `system_stats` (gpu-variant)
- **A — Formula:** `mem.used / 1e9` (gpu_monitor.py:101). Bytes to GB conversion. Correct (1 GB = 1e9 bytes, consistent with SI prefix convention used throughout codebase).
- **B–F:** System poll; no aggregation; no planner semantics; no band; correct measurement; not redundant.
- **Empirical:** ~1.01 GB in banked sample (early training with small buffer). Correct.
- **Verdict: CORRECT — keep**

---

### rss_gb
- **Emit site:** `hexo_rl/monitoring/gpu_monitor.py:128`
- **Event:** `system_stats` (gpu-variant)
- **A — Formula:** `_PROCESS.memory_info().rss / 1e9` (gpu_monitor.py:114). Process RSS in GB. Correct.
- **B–F:** System poll; no aggregation; no planner semantics; no band; correct; not redundant.
- **Empirical:** ~6.32 GB in banked sample.
- **Verdict: CORRECT — keep**

---

### cpu_util_pct
- **Emit site:** `hexo_rl/monitoring/gpu_monitor.py:128`
- **Event:** `system_stats` (gpu-variant)
- **A — Formula:** `psutil.cpu_percent(interval=None)` (gpu_monitor.py:109). Returns CPU utilization since last call (non-blocking). Correct.
- **B–F:** System poll; no aggregation; no planner semantics; no band; correct; not redundant.
- **Empirical:** 0.0 in one banked sample (non-blocking call can return 0 on first call after boot).
- **Verdict: CORRECT — keep**

---

### disk_free_gb
- **Emit site:** `hexo_rl/monitoring/disk_guard.py:60`
- **Event:** `disk_free`
- **A — Formula:** `shutil.disk_usage(path).free / 1e9` (disk_guard.py:58-60). Correct bytes-to-GB conversion.
- **B — Eff-n:** System measurement; no aggregation; no CI concern. PASS.
- **C — Planner-semantics:** Filesystem metric; no planner semantics. PASS.
- **D — Band calibration:** has_band=true. Bands: `warn_gb=10.0` (default), `fail_gb=5.0` (default) (disk_guard.py:33-34). These are filesystem management thresholds, not regime-specific health bands. They are **config-driven** (can be overridden per deployment) and **engineering-sound** (10 GB warn / 5 GB abort is a reasonable operational floor). Not borrowed from another regime; not inverted; not invented from context. PASS.
- **E — Construct validity:** Measures available disk space correctly. PASS.
- **F — Redundancy:** Not duplicate of any other stat. PASS.
- **Empirical:** 463.48 GB in banked sample.
- **Verdict: CORRECT — keep**

---

## Summary table

| Stat | Verdict | Action |
|---|---|---|
| win_rate_p0 | CORRECT | keep |
| win_rate_p1 | CORRECT | keep |
| draw_rate | CORRECT | keep |
| games_total | CORRECT | keep |
| games_this_iter | CORRECT | keep |
| games_per_hour | CORRECT | keep |
| positions_per_hour | CORRECT | keep |
| avg_game_length | CORRECT | keep |
| sims_per_sec | CORRECT | keep |
| buffer_size | CORRECT | keep |
| buffer_capacity | CORRECT | keep |
| corpus_selfplay_frac | CORRECT | keep |
| batch_fill_pct | CORRECT | keep |
| quiescence_fires_per_step | BIASED | fix (divide by log_interval or rename) |
| colony_extension_stone_count | CORRECT | keep |
| colony_extension_stone_total | CORRECT | keep |
| colony_extension_fraction | CORRECT | keep |
| terminal_reason | CORRECT | keep |
| model_version_min | CORRECT | keep |
| model_version_max | CORRECT | keep |
| model_version_distinct | CORRECT | keep |
| model_version_range_size | REDUNDANT | drop (= max - min, both emitted) |
| stride5_run_p90 | CORRECT | keep |
| row_max_density | CORRECT | keep |
| corpus_fraction | CORRECT | keep |
| worker_draw_rate_per_worker | CORRECT | keep |
| mv_median_range | BIASED | fix (upper-half median) |
| mv_p90_range | CORRECT | keep |
| mv_max_range | CORRECT | keep |
| mv_median_distinct | BIASED | fix (upper-half median) |
| mv_spearman_rho_range_vs_draw | CORRECT | keep |
| gpu_util_pct | CORRECT | keep |
| vram_used_gb | CORRECT | keep |
| rss_gb | CORRECT | keep |
| cpu_util_pct | CORRECT | keep |
| disk_free_gb | CORRECT | keep |

---

## Coverage gaps (stats not present in banked sample)

The following B2 stats are gated by `cfg.instrumentation_enabled=False` and are **absent from all banked logs**. Emit sites verified to exist in source. Coverage gap is not a formula error but is noted:

- `corpus_fraction` — `buffer_composition` event never emitted
- `worker_draw_rate_per_worker` — `worker_draw_rate` event never emitted
- `mv_median_range`, `mv_p90_range`, `mv_max_range`, `mv_median_distinct`, `mv_spearman_rho_range_vs_draw` — `model_version_summary` event never emitted

**Schema collision:** `system_stats` event is emitted by both `gpu_monitor.py:128` (GPU/CPU/RAM fields) and `pool.py:811` (buffer_size/buffer_capacity fields). Same event name, incompatible schemas, both present in banked logs (859 gpu-type, 869 buffer-type). Dashboard consumers must filter by field presence.

---

## Seeds re-derived

### Seed 6 (partial ownership — colony_extension_fraction in B2)
**Claim:** `colony` 0.15/0.25 bands mislabeled as gates not heuristics — Axis D.

**Re-derived from source:**
- `colony_extension_fraction` (B2): `has_band=false` in inventory. No band defined at the B2 emit site (pool.py:772) or in `_compute_colony_extension`. The 0.15 threshold appears only in monitoring scripts: `check_phase_c.py:187` (`warn=0.10, crit=0.15` in a bar-display helper) and `d1m_monitor.py:779` (comment string "investigate ≥0.15"). Neither enforces a gate or abort.
- **Finding:** For the B2 stat `colony_extension_fraction`, seed 6's claim does NOT apply — the stat has no band at all. The 0.15/0.25 thresholds are in B5-layer monitoring scripts and are cosmetic display thresholds, not enforced gates on the emit path. B2 stat passes Axis D.
- **Outcome:** CONFIRMED at B5 layer (monitoring scripts), FALSIFIED for B2 stat itself. Ownership of seed 6's substantive finding belongs to B5.

### No other §3 seeds fall primarily in B2
- Seed 2 (depth ~3.0/3.4 literal): owned by B3 (`mcts_mean_depth`).
- Seed 7 (alt_spread NaN): owned by B5 (`value_spread_canary.py`).
- Seeds 1,3,4,5,8: owned by B3/B4/B5.
