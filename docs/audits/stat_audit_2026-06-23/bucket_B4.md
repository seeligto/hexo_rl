# Bucket B4 — eval stats audit (D-STATAUDIT Phase 1)

Source pin: worktree `52067631`. Read-only. Empirical claims use §4 banked logs only.

## Cross-cutting facts (load-bearing for every verdict below)

- **win_rate formula** (`hexo_rl/eval/evaluator.py:231`): `wr = (win_count + 0.5*draw_count)/n_games`.
  Draw-as-half, denominator = total games. Correct. EVERY `wr_*` stat is this value.
- **CI formula** (`hexo_rl/eval/gate_logic.py:74-97` `_binomial_ci`): Wilson score interval,
  z from `norm.ppf`, clamped [0,1], width→0 only at wins∈{0,n}. Math CORRECT. EVERY `ci_*` is this.
- **Eff-n (Axis B)**: `_binomial_ci(wins, n)` uses the RAW game count `n`, never a distinct-game count.
  BUT the production eval loop (`evaluator.py:202-217`) injects diversity: color alternates by `i%2`
  AND `eval_random_opening_plies` default = 4 (`defaults.py:32-33`) AND eval_temperature default = 0.5
  (NOT temp-0). So distinct openings ⇒ n_eff ≈ n for the temp-0.5 arms. Re-derived against the
  banked round-robin (`rr_5rung.jsonl`): 26–37 distinct game-signatures per 40-game pair — diversity
  IS injected, n_eff NOT collapsed to ~2. The §D-ARGMAX n_eff→2 pathology needs argmax+temp-0+FIXED
  opening; the live eval breaks all three. See "Seeds re-derived → seed5".
- **EvalResult source** (`evaluator.py:57-64`): `win_rate, win_count, n_games, colony_wins, draw_count`.

---

## Per-stat findings

### wr_best — `eval_pipeline.py:452` (computed `opponent_runners.py:362`)
- A: `er.win_rate` = draw-half formula, correct. PASS.
- B: n_games default 200 (`_run_best` L326) — promotion-grade n. CI computed (`ci_best`). Distinct-game
  diversity present (opening plies + alternating color). PASS.
- C: best opponent is another MODEL playing its own MCTS at `model_sims`/`opponent_sims`; this is a
  H2H-vs-anchor WR, not a PUCT-visit heuristic. Gumbel-neutral (WR is outcome-based). PASS.
- D: feeds the promotion gate via `evaluate_gate` with `promotion_winrate=0.55` AND `ci_lo>0.5`
  (`gate_logic.py:120-122`). Band is regime-correct (vs-anchor self-play bar). PASS.
- E: H2H-vs-champion is the canonical strength instrument; not a Goodhart proxy. PASS.
- F: distinct (only stat measuring vs rotating champion). PASS.
- **Verdict: CORRECT. keep.**

### ci_best — `eval_pipeline.py:452` (`gate_logic.py:74`)
- A: Wilson, correct. PASS. B: built on n=200; the ONLY CI that gates promotion (`ci_lo>0.5`). PASS.
- D: `ci_lo>0.5` guard is the §101.a gate — calibrated to this regime. PASS.
- **Verdict: CORRECT. keep.**

### colony_wins_best — `eval_pipeline.py:452` (`evaluator.py:225`)
- A: raw integer count of wins that ALSO pass `is_colony_win` — a subset of win_count. Correct count.
- B: raw count, no band, no CI; honest as a count. PASS.
- F: NOT redundant with wr_best (wr is all wins; this is the colony-shaped subset). Carries the
  anti-colony-objective signal. Denominator (n=200) recoverable from the paired wr arm. PASS.
- **Verdict: CORRECT. keep.**

### wr_sealbot — `eval_pipeline.py:452` (`opponent_runners.py:152`)
- A: draw-half WR, correct. PASS.
- B: n_games default 50 (`_run_sealbot` L133) — n below the n≈400 promotion bar, but SealBot is a
  MONITOR arm (does not gate promotion; the gating arm is wr_best). 50 acceptable for trend. PASS.
- C: outcome-based WR vs an external fixed bot, Gumbel-neutral. PASS.
- D: see `sealbot_gate_passed` below — the attached green band (≥0.5) is mis-set; wr_sealbot ITSELF
  carries no band on its own row, the band lives in the derived flag. Magnitude reality: SealBot at
  fair temp-0.5 / 128-sims is ~18% (seed4). PASS for the raw rate; the band failure is charged to
  `sealbot_gate_passed`.
- E: real external-strength signal. PASS.
- **Verdict: CORRECT. keep.** (band defect quarantined in `sealbot_gate_passed`.)

### ci_sealbot — `eval_pipeline.py:452`
- A: Wilson, correct. B: n=50, honest CI width reported. No gate dependency. PASS all.
- **Verdict: CORRECT. keep.**

### colony_wins_sealbot — `eval_pipeline.py:452`
- Raw colony-win subset count vs SealBot. Same shape as colony_wins_best. PASS all.
- **Verdict: CORRECT. keep.**

### sealbot_gate_passed — `eval_pipeline.py:452` (`opponent_runners.py:155`)
- A: `er.win_rate >= 0.5` — a boolean threshold flag. Formula trivial.
- D (FAIL): the 0.5 bar is a borrowed PUCT-style "beat the bot" threshold. SealBot at the live fair
  regime (temp-0.5, model_sims default 128) is realistically a ~18% WR opponent for in-distribution
  HeXO models (seed4: green-at-0.55/0.5 borrowed from the wr_best self-play bar). A 0.5 absolute
  green is unreachable in this regime → the flag is ~always False ⇒ a dead/inverted band that reads
  as "always failing" rather than tracking SLOPE. It is also NOT consumed by any gate (`promoted`
  ignores it; only wr_best/strength/robustness gate) so it is band-only signal with a mis-set band.
- E: as an absolute green it mis-measures (named "gate_passed" but gates nothing and is mis-thresholded).
- **Verdict: BIASED (Axis D). fix:** make it slope-based (improvement vs prior round) or re-band to the
  regime-real SealBot WR distribution (green ≈ sustained rise above the ~0.18 baseline), and rename to
  drop "gate" (it is not a gate). Keep the flag only as a trend marker.

### wr_random — `eval_pipeline.py:452` (`opponent_runners.py:120`)
- A: draw-half WR, correct. B: n default 20; sanity-floor opponent (expect ~1.0), low n acceptable for
  a floor check, no CI gate. C/E: outcome-based, Gumbel-neutral, real. F: distinct (lowest rung). PASS all.
- **Verdict: CORRECT. keep.**

### ci_random — `eval_pipeline.py:452`
- Wilson on n=20. No band, honest width. PASS all. **Verdict: CORRECT. keep.**

### colony_wins_random — `eval_pipeline.py:452`
- Raw colony subset count. PASS all. **Verdict: CORRECT. keep.**

### wr_bootstrap_anchor — `eval_pipeline.py:452` (`opponent_runners.py:312`)
- A: draw-half WR vs frozen bootstrap model. Correct. PASS.
- B: n default 100. Feeds the bootstrap-FLOOR gate (`eval_pipeline.py:372-378`) as a hard AND-condition
  (`>= min_winrate`, default 0.45) — but that gate is a simple point-estimate threshold with NO CI
  guard (unlike wr_best). For a floor at 0.45 on n=100 the CI half-width ≈ ±0.10 — a single noisy
  round can false-pass/false-block the floor. Minor (it only blocks, paired with wr_best's CI gate).
- C: model-vs-frozen-model outcome WR, Gumbel-neutral. PASS.
- D: floor 0.45 is regime-derived (anti-Class-5 collapse guard, §155 T2). PASS.
- E: catches the "beats champion but collapses vs fixed reference" pathology — construct-valid. PASS.
- **Verdict: CORRECT. keep.** (note the floor-gate lacks a CI guard; advisory, not a verdict-flip.)

### ci_bootstrap_anchor — `eval_pipeline.py:452`
- Wilson on n=100. Computed but NOT consumed by the floor gate (point-estimate only). Honest as a
  reported CI; the gap is in the gate, not the stat. PASS all. **Verdict: CORRECT. keep.**

### colony_wins_bootstrap_anchor — `eval_pipeline.py:452`
- Raw colony subset count. PASS all. **Verdict: CORRECT. keep.**

### wr_argmax_n — `eval_pipeline.py:452` (`opponent_runners.py:224`, `evaluator.py:349`)
- A: draw-half WR; model plays at `model_sims=1` (≈ policy argmax) vs SealBot. Correct.
- B (Axis B, the §D-ARGMAX lens): n default 20. The MODEL side is argmax (deterministic), BUT
  opening-plies=4 + alternating color + SealBot's own think-time variation still inject distinct
  games (re-derivation below). So this arm is LESS collapsed than the pure-argmax-fixed-opening case,
  yet most fragile of the B4 arms: drop opening_plies or temp→0 on BOTH sides and n_eff collapses.
  CI on n=20 is wide (~±0.20) and reported. As a DRIFT detector (divergence vs MCTS-128) the
  point-comparison is the intended read, not an absolute strength claim.
- C: this arm deliberately strips MCTS (n_sims=1) to isolate the POLICY head — it is intentionally
  NOT a search-health stat; Gumbel-neutral (it bypasses search entirely). PASS.
- D: thresholds (~18% rise / ~28% floor) live in the comment, not a coded band on this stat. Soft.
- E: construct-valid DRIFT detector (policy-vs-value divergence, §170). PASS.
- **Verdict: CORRECT. keep.** (advisory: pin the diversity injection — if a variant sets
  opening_plies=0 this arm's CI becomes √(copies)-overconfident per §D-ARGMAX.)

### ci_argmax_n — `eval_pipeline.py:452`
- Wilson on n=20, honest width, no gate. The n_eff caveat above transfers but CI IS reported and is
  wide. PASS (no false "CI-resolved" claim in code). **Verdict: CORRECT. keep.**

### colony_wins_argmax_n — `eval_pipeline.py:452`
- Raw colony subset count. PASS all. **Verdict: CORRECT. keep.**

### wr_nnue — `eval_pipeline.py:452` (`opponent_runners.py:188`)
- A: draw-half WR vs Hammerhead NNUE. Correct. B: n default 100, CI reported, default-OFF arm.
  C/E: outcome WR vs external bot, Gumbel-neutral; reads "SealBot-overfit vs general plateau"
  (construct-valid second rung). F: distinct opponent. PASS all.
- **Verdict: CORRECT. keep.**

### ci_nnue — `eval_pipeline.py:452`
- Wilson on n=100, honest. PASS all. **Verdict: CORRECT. keep.**

### colony_wins_nnue — `eval_pipeline.py:452`
- Raw colony subset count. PASS all. **Verdict: CORRECT. keep.**

### offwindow_forced_win_rate — `result_types.py:94` (computed `offwindow_probe.py:159`, written `opponent_runners.py:399`)
- A: `ow/n` where `ow = sum(off_window_win)`, `off_window_win = adv_won AND win_cell_off_window`
  (`offwindow_probe.py:132`). Binding cell = the move that LANDS the win (`forced_win_detector.py:343`
  uses `max(win_cells, key=cheb-to-center)` = completing cell) — this is the turn-vs-ply-correct cell
  per the §D-COHERENCE corollary. Formula CORRECT.
- B: default n=100; rate then back-projected `forced = round(rate*n)` for a Wilson CI (`L394-395`).
  Honest. PASS.
- C: outcome/geometry based, Gumbel-neutral. PASS.
- D: consumed as a robustness BLOCK in `decide_promotion` (`gate_logic.py:64-68`) at
  `robustness_threshold` default 0.06; missing measurement = pass (not a false block). Band regime-set.
- E: measures genuine off-window exploitability vs the model's own MCTS — the right adversarial
  instrument (§D-COHERENCE: a fixed-bot WR would false-clear this). Construct-valid. PASS.
- **Verdict: CORRECT. keep.**

### offwindow_strict_forced_rate — `result_types.py:95` (`offwindow_probe.py:160`)
- A: `strict/n`, `strict = adv_won AND win_cell_off_window AND model_had_inwindow_block is False`
  (`offwindow_probe.py:133`). Correct: the strict variant requires the model had no in-window block
  available. Formula correct. B: n=100, no separate CI (it is the stricter companion to the banded
  rate above). C/E: same adversarial construct, Gumbel-neutral. PASS.
- F: NOT redundant — it is a strictly-tighter conditioned subset of offwindow_forced_win_rate
  (different denominator-condition), carries the "no-excuse" exploitability. PASS.
- **Verdict: CORRECT. keep.**

### strength_aggregate — `result_types.py:100`
- A (FAIL — coverage/phantom): grepped ALL of `hexo_rl/` — there is NO write site
  (`results["strength_aggregate"] = ...` returns nothing; only `.get()` reads in `eval_pipeline.py:390`,
  `alert_rules.py:242`, `step_coordinator.py:1087`). result_types.py:97-99 itself documents it as
  "operator-gated follow-up" — i.e. the producer is unbuilt. In the audited tree this stat is NEVER
  emitted, so the `decide_promotion` branch where `strength_aggregate REPLACES wr_best`
  (`gate_logic.py:57-62`) is DEAD CODE; promotion always falls through to wr_best. A stat that is
  consumed by a gate but never produced is an Axis-A "intended definition not implemented" + an
  Objective-A coverage gap (the documented fixed-reference strength axis does not exist this run).
- B/C/D/E/F: not reachable — nothing emitted.
- **Verdict: WRONG (Axis A — definition not implemented; phantom). drop** the stat from the tracked
  inventory until a producer exists (or build the producer). As-is it is a silent no-op gate branch.

### strength_cycle_density — `result_types.py:101`
- A (FAIL — phantom, same as above): NO write site. Read only at `step_coordinator.py:1090` (defaults
  to 0.0 via `.get(...,0.0)`) and referenced by `alert_rules.py:213-235` non-transitivity abort. The
  abort therefore ALWAYS sees 0.0 ⇒ the cycle-density abort can never fire — it is permanently
  green by construction (a band that cannot trip). Axis A (unimplemented) + Axis D (a calibration band
  on a constant-0 input is meaningless).
- **Verdict: WRONG (Axis A — phantom). drop** until a producer emits real ladder-cycle density.

### elo_estimate — `eval_pipeline.py:440`
- A: `ratings[ckpt_pid][0]` — the Bradley-Terry point estimate from `compute_ratings` (B3-owned BT math).
  Extraction correct (point estimate; the (lo,hi) is discarded for this scalar but kept in `ratings`).
- B: BT fit over `get_all_pairwise(run_id)`. The pairwise rows are themselves WR-over-n records; BT
  rating is anchored to `checkpoint_0` (`bt_cfg.anchor_player`). The point-estimate scalar carries NO
  CI on its own row (the CI lives in `results["ratings"][name]["ci"]`), so a consumer reading only
  `elo_estimate` sees no uncertainty — minor (CI IS surfaced elsewhere).
- C: BT over outcome WRs, Gumbel-neutral. PASS. E: relative strength index, construct-valid IF the
  pairwise graph is connected; sparse early rounds give unstable Elo (the BT module's concern, B3).
- F: not redundant (only stat giving a single relative-strength scalar across all opponents). PASS.
- **Verdict: CORRECT. keep.** (advisory: surface the BT CI alongside the scalar; band-less point
  estimate invites over-reading early sparse rounds.)

### value_fc2_weight_abs_max — `eval_pipeline.py:293` (`_g4_value_head_band_check` L75-88)
- A: `value_fc2.weight.abs().max()`, rounded 4dp. Correct measurement of the value-head output-layer
  weight magnitude. Returns nan/True for MagicMock (test-only guard). PASS.
- B: not an aggregate over games (single weight read). No eff-n concern. PASS.
- C: a NN-weight probe, independent of search regime. Gumbel-neutral. PASS.
- D: band [0.154, 0.462] = baseline 0.308 ±50% (`L66-72`). Derived from the v7full bootstrap
  measurement, not borrowed/inverted. WARNING-only (does not gate). Reasonable. PASS.
- E (note): this is an internal-metric proxy (weight magnitude), exactly the Axis-E class "internal
  metric moves while H2H regresses." But it is WARNING-only and explicitly a value-head DRIFT canary,
  not a gate or a strength claim — so it does not Goodhart any lever. PASS as-bounded.
- **Verdict: CORRECT. keep.**

### g4_value_head_band_pass — `eval_pipeline.py:294`
- A: `_G4_VALUE_FC2_BAND_LO <= val <= _G4_VALUE_FC2_BAND_HI`, the band flag for the stat above. Correct.
- D: same regime-derived band, warning-only. PASS. F: derived trivially from value_fc2_weight_abs_max
  + the two band constants — but it is the surfaced/actionable form (the raw weight is the recoverable
  one), so this is the canonical alert, not the redundant copy. PASS.
- **Verdict: CORRECT. keep.**

### eval_games — `eval_pipeline.py:288` (incremented per arm, e.g. `opponent_runners.py:123`)
- A: running sum of `n` across all arms that fired this round. Correct accumulator; it counts TOTAL
  games (random+sealbot+...+best), a coverage/throughput number, not a strength stat. PASS.
- B/C/D/E: not a strength aggregate, no band, Gumbel-neutral. PASS. F: distinct (coverage counter). PASS.
- **Verdict: CORRECT. keep.**

### promoted — `eval_pipeline.py:398`
- A: `decision.promoted` from `decide_promotion` = (strength_ok AND robustness_ok), where strength_ok
  falls back to `wr_best_promoted` = (wr_best≥0.55 AND ci_lo>0.5 AND bootstrap_floor). Correct
  conjunction; fail-safe on missing arms (missing wr_best ⇒ no promote; missing robustness ⇒ pass).
- B: rests on wr_best's n=200 + CI gate — adequate. PASS.
- C: outcome-gated, Gumbel-neutral. PASS. D: bands inherited from the gating stats, regime-set. PASS.
- E: the actual deployment decision; construct-valid by definition. Caveat: the strength_aggregate
  REPLACE branch is dead (phantom producer) so "promoted" is currently the legacy wr_best decision —
  charged to strength_aggregate, not here. PASS.
- **Verdict: CORRECT. keep.**

### eval_round_wall_sec — `result_types.py:70`
- A: wall-clock of the eval round, set by `step_coordinator._run_eval` (sizes the close-out drain
  budget, §D-LOOPFIX W1). Diagnostic timing, not a strength/aggregate stat. No band. PASS all axes.
- **Verdict: CORRECT. keep.**

---

## Seeds re-derived (B4-owned)

### seed4 — `wr_sealbot` green-at-0.55 + absolute-not-slope (Axis D) — CONFIRMED
Re-derived from source: `opponent_runners.py:155` sets `sealbot_gate_passed = er.win_rate >= 0.5`
(the "green" bar is 0.5 here, the gating-doc 0.55 elsewhere — either way a self-play-style bar applied
to an EXTERNAL bot). Live fair regime is temp-0.5 (`defaults.py:32`) + model_sims 128 (`L29`), where a
realistic in-distribution SealBot WR is ~18%, so a 0.5 absolute green is effectively unreachable ⇒ the
flag reads ~always-False and is ABSOLUTE not SLOPE-based. The §4 banked event logs contain ZERO eval
events (both `events_*.jsonl` hold only game_complete/iteration_complete — short pre-eval runs), so I
could not pull a live wr_sealbot distribution; flagged "not present in banked sample" and verified the
emit site + threshold from source instead. Outcome: CONFIRMED as a mis-set band (BIASED, Axis D).

### seed5 — §D-ARGMAX n_eff→2 under argmax+no-plies (Axis B) — INVERTED (for the live eval path)
The documented pathology: deterministic argmax+temp-0+fixed-opening collapses distinct games to ~2/pair,
making a Wilson/BT CI over the raw count √(copies)-overconfident. Re-derived against the banked
round-robin `rr_5rung.jsonl`: 26–37 DISTINCT game-signatures per 40-game pair — diversity present, NOT
collapsed to 2. Source check: the production eval loop (`evaluator.py:202-217`) injects
`eval_random_opening_plies=4` (`defaults.py:33`) + alternating color (`i%2`) + temp 0.5, breaking all
three collapse preconditions. CI code (`_binomial_ci`) does still key on the raw `n` (no distinct-game
dedup), so the pathology RETURNS for any variant that sets opening_plies=0 / temp 0 on both sides
(notably the `argmax_n` arm's MODEL side is already deterministic — it survives only because openings +
SealBot variation remain). Outcome: INVERTED for the as-shipped eval defaults; the un-deduped-n latent
risk is real and noted on `ci_argmax_n` / `wr_argmax_n`.

### seed8 — `forced_win_conversion` 0.686→1.0 "recovery" on partial evidence (Axis E + coverage) — STALE
`forced_win_conversion` lives in `forced_win_detector.py:392` (a replay-EMA diagnostic), NOT a tracked
B4 eval-pipeline stat — it is never written into `EvalRoundResult`. The B4 robustness path uses
`offwindow_forced_win_rate` / `offwindow_strict_forced_rate` (different stats, audited CORRECT above).
The §4 banked logs carry NO `forced_win_trend` / `forced_win_conversion` events (verified: only
game_complete/iteration_complete in both event logs), so the "0.686→1.0 recovery" cannot be reproduced
from the banked sample at all. The detector code itself (`forced_win_detector.py:374-382`) only updates
the EMA on games with ≥1 forced win and `n` counts only those games — so a near-zero `n` makes a "1.0
conversion" an advisory-only readout whose CI spans the gate (the docstring L362-367 says so). Outcome:
the seed's "recovery" claim is not re-derivable here (no banked emit, stat not in B4 surface) ⇒ STALE
doc-claim relative to the audited eval inventory. Construct-validity caveat (small-n conversion=1.0 is
non-informative) is real and lives in the detector module (B5/diagnostics scope), not B4.
