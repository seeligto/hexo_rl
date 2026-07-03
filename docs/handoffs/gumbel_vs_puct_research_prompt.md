# Research prompt — WHY Gumbel-100 plays differently than PUCT-600 (and what it means)

Paste this into a fresh Claude Code session as the task. Prefix: "Read CLAUDE.md and
docs/07_PHASE4_SPRINT_LOG.md (§D-GUMBELSIMS) first." It is a DATA-DIVE: investigate the actual
games + stats, not just re-run evals. Goal: explain the behavioral divergence behind the Phase-3
verdict, and surface the raw data for human inspection.

## What we already know (Phase 3, matched 18w, v6_live2_ls, 15k steps, from bootstrap)
Two arms, ONE variable = self-play search: Arm-Gumbel (`gumbel_mcts`, m=32, n_sims_full=100) vs
Arm-PUCT (n_sims_full=600). Results (`reports/gumbelsims/PHASE3_RESULT.md`):

- **Strength (round_robin, distinct-game CI):** PUCT-600 stronger — Elo +154 vs Gumbel +65; PUCT
  beats Gumbel head-to-head **60%** (72-48). Transitive (anchor < Gumbel < PUCT).
- **Off-window coherence (exploit_probe / KClusterMCTSBot):** INVERTED — **Gumbel DEFENDED (0/100),
  PUCT FORCEABLE (17/100).** The weaker arm is more off-window-robust.
- **vs SealBot:** Gumbel 32% ≥ PUCT 24% (CIs overlap; non-transitive vs the H2H).
- **Game character (training self-play, ~7500 games each):**
  | | Gumbel-18w | PUCT-18w |
  |---|---|---|
  | draw_rate | **0.000** | **0.236** |
  | plies mean/median/p90 | 34 / 31 / 53 | 92 / 85 / **150 (cap)** |
  | policy_entropy (spread) | 2.31 | 2.52 |
  | self-play entropy | 2.48 | 2.99 |
- **Throughput (tuned, vast):** Gumbel-opt 4.5× PUCT (host-tuning-contingent). An equal-GPU-hours
  affordability arm (Gumbel-opt, tspg=1.0, w32/ib64/wait2/lf8) is RUNNING — compare its final to the
  done PUCT-15k.

**Working story to test/refute:** Gumbel-100 = shallow search (100 sims / 32 candidates ≈ 3 sims
each) → peaked/decisive policies → short, drawless games → weaker in deep direct combat but stays
off-window-defended (never drawn into the long off-window-forcing sequences). PUCT-600 = deep search
→ cautious/spread policies → long, drawy games that hit the 150-cap → stronger direct play but the
long games expose a 17% off-window blind spot.

### THE DEPTH FINDING + the `m` lever (central — likely the strength fix)
Measured `mcts_mean_depth` (self-play): **Gumbel-100/m32 = 2.83** vs **PUCT-600 = 3.52** (Gumbel-opt
also 2.83). **Gumbel searches SHALLOWER, not deeper** — Gumbel's sim-efficiency is *policy improvement*
(good ROOT action from few sims), NOT tactical depth. Sequential Halving spreads the 100 sims across
m=32 root candidates (≈3 each) → BREADTH, shallow per line; PUCT concentrates 600 down the principal
variation → DEPTH. This is the mechanism behind the strength loss (worst in the 0–30-ply tactical
bucket, 68% PUCT).

**The levers — TWO ways to buy depth, with different costs (map the 2D (m,n) Pareto frontier).**
Depth = f(n, m): sims-per-candidate ≈ n/m. PUCT-600 sits at depth 3.5; Gumbel-100/m32 ≈ 3 sims/cand
→ depth 2.8.
- **lower m (keep n=100):** depth ↑, breadth ↓ (fewer candidates → narrower improved-policy target),
  **throughput UNCHANGED** (n fixed → the 4.5× multiplier holds). The FREE lever — but the policy-
  improvement guarantee weakens as m shrinks. Phase-1's policy-optimal m=32 likely diverges from the
  strength-optimal m (lower, deeper).
- **raise n (keep m=32):** depth ↑, breadth unchanged, **throughput ↓** (n=200 ≈ 2.25×, n=400 ≈ ~1×
  = PUCT-parity, multiplier gone). The costly lever (Phase-1's "no cheap knee" — strength buys with
  sims, but you pay throughput).
- **combinations** (e.g. m=8/n=150) interpolate — the goal is the Pareto frontier: the HIGHEST-
  throughput (m,n) that reaches PUCT-comparable depth/strength while keeping a usable improved-policy
  breadth.

**Sweep:** the 2D grid m ∈ {4,8,16,32}, n ∈ {100,200,400,(600)} on the armc/v6_live2_ls generator —
measure (a) `mcts_mean_depth` (via `MCTSTree.last_search_stats()[0]`), (b) matched-position value-
regret vs a deep reference (n=800), (c) improved-policy entropy/breadth, (d) pos/hr (∝1/n; flat across
m). Find the frontier, then a SHORT training A/B at the best affordable (m,n) vs PUCT-600 to test
whether deeper-Gumbel recovers strength while keeping a throughput multiplier + the off-window edge.
NB: more training does NOT deepen the search (depth = f(n,m)) — the running affordability arm's 2.8
depth ceiling stands regardless of how long it trains; only (m,n) moves it.

## Central questions
1. **Why drawless vs 24% draws?** Is Gumbel genuinely more decisive, or is it resigning/blundering
   into quick losses (short games could be either decisive skill OR fast blunders)? Check WHO wins
   the short Gumbel games and HOW (winning line length, last-move tactics) vs the long PUCT draws.
2. **Why the off-window inversion?** Two hypotheses — (H_exposure) PUCT's longer games simply REACH
   more off-window-forcing positions (forcing-position-rate 0.34 vs 0.17), so it's opportunity not
   skill; (H_defense) Gumbel's broader Top-k candidate sampling actually teaches a better off-window
   defense. Discriminate: at MATCHED game-length / matched forcing-positions, is Gumbel still more
   defended? Replay the 17 PUCT off-window losses (exploit_probe games) and characterize them.
3. **Where does Gumbel lose to PUCT?** In the 360 round_robin games, split Gumbel-vs-PUCT games by
   length/phase: does PUCT win via long-game attrition (its strength) while Gumbel wins the short
   decisive ones? Is the 60% H2H concentrated in long games?
4. **Spread / cluster structure:** does Gumbel's play fragment the board into more/fewer clusters
   (k-windows) than PUCT? Compute per-position cluster counts (the v6_live2_ls multi-window legal
   set) over both arms' games — colony (many small clusters) vs coherent (few). Relate to
   colony_extension_fraction (Gumbel 0.006 mean / 0.385 max vs PUCT ~0).
5. **Affordability:** when the Gumbel-opt arm lands, does ~2.5× more + fresher (tspg=1.0) training
   close the 89-Elo gap while KEEPING the drawless/decisive character + off-window defense?
6. **Can depth recover strength for free? (the highest-value lever)** At fixed n=100 (throughput
   unchanged), does lowering m (deeper search) recover strength toward PUCT-600 — and where does the
   depth↔policy-breadth tradeoff optimum sit? This is the cheapest path to GUMBEL-SIMS-POSITIVE: same
   4.5× throughput, more depth. See "THE DEPTH FINDING + the `m` lever" above for the proposed sweep.

## Specific analyses (produce numbers + a few annotated example games)
- **Game-length distribution** per arm (training self-play + the 360 eval games), split by winner /
  draw / who-vs-who. Histogram + percentiles.
- **Decisiveness vs blunder:** for short Gumbel games, the winning-line length, and value-trajectory
  (does the eval/value head see the win coming, or is it a late swing?). Use the move-lists.
- **Off-window forensics:** replay the PUCT exploit_probe forceable games (the off-window losses) —
  what tactic forces them; does the same tactic fail vs Gumbel's model? (exploit_probe stores games.)
- **Cluster/spread time-series:** per ply, count legal-set clusters (k-windows) + policy entropy, for
  representative games of each arm. Gumbel-decisive vs PUCT-spread.
- **Per-phase strength:** bucket round_robin Gumbel-vs-PUCT games by length; win-rate by bucket.

## Data sources (all present)
- **Eval games w/ full move-lists:** `reports/p3_rr/*.jsonl` (360 games; labels s0k=anchor,
  s15k=Gumbel, s15001=PUCT; keys: p1/p2, p1_step/p2_step, winner, plies, moves, play_command).
  Aggregate: `reports/p3_rr_agg/` (win_matrix.csv, ratings.csv).
- **Off-window games + summaries:** `reports/p3_coherence/{gumbel,puct}.{json,summary.json}`.
- **SealBot:** `reports/p3_sealbot/{gumbel,puct}.log` (+ `reports/eval/*_sealbot.json`).
- **Training self-play replays (sampled, sample_rate=50):** vast `logs/replays/games_2026-06-*.jsonl`
  (filter by run timestamp — Gumbel-18w log `p3_gumbel_2026...071024Z`, PUCT-18w `...135010Z`,
  Gumbel-opt `...T065022Z`). Pull with rsync.
- **Training stat streams:** the run `.log` files on vast — `game_complete` events (plies, winner,
  colony_extension_fraction, sims_per_sec) + `train_step` events (policy_entropy,
  policy_entropy_selfplay, value_loss, value_accuracy_masked, components/cluster counts).
- **Board/cluster tooling:** `engine` `Board.with_encoding_name("v6_live2_ls")` + `legal_moves()` to
  replay move-lists and compute per-position cluster counts; `hexo_rl.eval.k_cluster_mcts_bot` for
  the no-drop view; `scripts/coherence_tempstrength.py` / `forced_win_detector` for off-window/forced.
- **Verdicts/context:** `reports/gumbelsims/{PHASE1_RESULT,PHASE3_RESULT}.md`, `reports/thr_opt/REPORT.md`,
  sprint log §D-GUMBELSIMS.

## Deliverable
`reports/investigations/gumbel_vs_puct_games_<date>.md`: the mechanism (decisive-shallow vs
cautious-deep), discriminated H_exposure vs H_defense for the off-window inversion, per-phase
strength split, 3-5 annotated example games (a short Gumbel win, a long PUCT draw, a PUCT off-window
loss), and the affordability read once Gumbel-opt lands. Flag any data artifact (e.g. short games =
blunders not skill) that would change the Phase-3 interpretation. Keep CLAUDE.md re-validation
discipline: a clean-metric flat result with a pre-registered gain is a confound, not a win.
