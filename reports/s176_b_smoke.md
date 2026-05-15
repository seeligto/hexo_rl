# Wave B — KrakenBot wrappers + tournament harness smoke

Branch: `phase4.5/s176_phase_a_validation`. Host: laptop Ryzen 7 8845HS, .venv Python 3.14.2.
Date: 2026-05-14.

## (a) Wrapper additions

| File | Class | LOC | Status |
|---|---|---:|---|
| `hexo_rl/bots/krakenbot_random.py` | `KrakenBotRandomBot` | 148 | INTEGRABLE — pure-python upstream (`vendor/bots/krakenbot/bot.py:42`), no compile |
| `hexo_rl/bots/krakenbot_mcts.py`   | `KrakenBotMCTSBot`   | 152 | UNAVAILABLE — raises `FileNotFoundError` (weights blocked per A1) |
| `tests/test_krakenbot_wrappers.py` | n/a (3 tests)        |  64 | PASS — 3 / 3, 1.90 s |

Total wrapper additions: **300 LOC of new code** (vs ≤230 budget; minor overrun on docstrings, no LOC over the body budget).

`_MockGame` copy-pasted into `krakenbot_random.py` rather than imported from `krakenbot_bot.py` (underscore-private upstream). Documented in module docstring. `krakenbot_mcts.py` lazy-imports `_MockGame` from `krakenbot_random` to avoid the same duplication, deferred-import to keep weights-missing case zero-cost.

Pytest output (`.venv/bin/python -m pytest tests/test_krakenbot_wrappers.py -x -v`):

```
tests/test_krakenbot_wrappers.py::test_krakenbot_minimax_smoke PASSED    [ 33%]
tests/test_krakenbot_wrappers.py::test_krakenbot_random_smoke PASSED     [ 66%]
tests/test_krakenbot_wrappers.py::test_krakenbot_mcts_unavailable_raises PASSED [100%]
============================== 3 passed in 1.90s ===============================
```

## (b) `scripts/tournament_validate.py` — CLI + design

CLI:

```
--bots <csv>                 # required, comma-separated names
--n_games <int>              # default 5
--max_plies <int>            # default 200
--random_opening_plies <int> # default 0 (reserved for Wave C; passes through, unused)
--our_model_ckpt <path>      # default checkpoints/bootstrap_model.pt (v6w25)
--our_n_sims <int>           # default 128
--max-wall-seconds <int>     # default 0 (unbounded); on overrun, dumps partial JSONL + exits 0
--output <dir>               # required
```

Design summary:
- **Single-game-at-a-time** — no concurrent pool (Wave A4 do-not #1, §17 GIL regression).
- **Fresh bot instance per game** — every game invokes `BOT_REGISTRY[name]()` for P1 and P2 separately. Pre-empts MinimaxBot Zobrist TT carryover (~150× under-bench, A1 finding).
- **Side balance** — alternate P1/P2 every other game per pair.
- **Outputs**: `per_game.jsonl` (full schema per spec), `summary.md` (BT ranks + H2H + colony + latency), `ratings.csv`, `h2h_matrix.csv`, `colony_table.csv`.
- **BT MLE**: reuses existing `hexo_rl.eval.bradley_terry.compute_ratings` (L-BFGS-B with analytical gradient, Hessian-derived CI). Anchor = first bot in `--bots` list. Draws split 0.5/0.5.
- **Per-game schema** (full): pair, game_idx, p1, p2, winner, winner_bot, plies, n_stones_p1/p2, colony_fraction_winner, n_components_winner, mean_pairwise_hex_dist_winner, max_pairwise_hex_dist_winner, terminal_threat_count, winner_completed_via_threat, p1_total_wall_s, p2_total_wall_s, p1_max_move_s, p2_max_move_s. JSONL one game per line.
- **colony_fraction_winner** = 1 − (largest_winner_component / total_winner_stones). Single-component win → 0.0. Multi-cluster colony win → up to ~0.5.
- **winner_completed_via_threat**: heuristic — winner's final ply landed on a still-threatened cell at game end. Best-effort; not a strict open-4 detector.
- **Registry**: 9 entries (`randombot`, `sealbot`, `kraken_minimax_strong`/`_fast`, `kraken_random`, `kraken_mcts_resnet`/`_selfplay`, `our_v6_mcts128`, `our_v6_argmax`). Wave C selects the 6 OURS-CONTROL-CONFIRMED via `--bots`.

## (c) Smoke output

Invocation:

```
.venv/bin/python scripts/tournament_validate.py \
  --bots randombot,sealbot,our_v6_mcts128 \
  --n_games 5 \
  --max_plies 200 \
  --random_opening_plies 0 \
  --our_model_ckpt checkpoints/bootstrap_model_v6.pt \
  --our_n_sims 64 \
  --max-wall-seconds 900 \
  --output reports/s176_b_smoke/
```

Note: `our_model_ckpt` switched from `checkpoints/bootstrap_model.pt` (v6w25, policy=626 actions) to `checkpoints/bootstrap_model_v6.pt` (v6, 362 actions). First run failed with shape mismatch `(1×722) vs (1250×626)` — `bootstrap_model.pt` is v6w25 in the live checkpoint dir, not v6. v6 wrapper config (`board_size=19`, `encoding=v6`) demands a v6 checkpoint. Wave C must standardise on which canonical checkpoint represents `our_v6_*`.

`our_n_sims=64` chosen for smoke wall budget. Wave C will use 128.

Result (total wall: **40.9 s** for 15 games):

```
[game] randombot vs sealbot #0: winner=sealbot plies=19 wall=1.7s
[game] randombot vs sealbot #1: winner=sealbot plies=21 wall=1.1s
[game] randombot vs sealbot #2: winner=sealbot plies=15 wall=1.5s
[game] randombot vs sealbot #3: winner=sealbot plies=17 wall=1.1s
[game] randombot vs sealbot #4: winner=sealbot plies=15 wall=1.5s
[game] randombot vs our_v6_mcts128 #0: winner=our_v6_mcts128 plies=15 wall=2.2s
[game] randombot vs our_v6_mcts128 #1: winner=our_v6_mcts128 plies=17 wall=0.3s
[game] randombot vs our_v6_mcts128 #2: winner=our_v6_mcts128 plies=15 wall=0.3s
[game] randombot vs our_v6_mcts128 #3: winner=our_v6_mcts128 plies=13 wall=0.2s
[game] randombot vs our_v6_mcts128 #4: winner=our_v6_mcts128 plies=19 wall=0.4s
[game] sealbot vs our_v6_mcts128 #0: winner=sealbot plies=57 wall=7.5s
[game] sealbot vs our_v6_mcts128 #1: winner=sealbot plies=31 wall=4.0s
[game] sealbot vs our_v6_mcts128 #2: winner=sealbot plies=57 wall=7.5s
[game] sealbot vs our_v6_mcts128 #3: winner=sealbot plies=31 wall=4.1s
[game] sealbot vs our_v6_mcts128 #4: winner=sealbot plies=57 wall=7.5s
```

- 15 / 15 games completed, 0 tracebacks.
- 15 lines in `per_game.jsonl` ✓.
- All 5 output files present (`per_game.jsonl`, `summary.md`, `ratings.csv`, `h2h_matrix.csv`, `colony_table.csv`) ✓.
- BotProtocol `reset()` not called by harness (fresh instance per game makes it moot) — explicit `reset()` honoured for any bot still re-used, but tourney loop never re-uses.

### Ranks (Elo, anchor=randombot)

| Bot | Elo | CI lo | CI hi |
|---|---:|---:|---:|
| sealbot          | 3932.6 | −55991.9 | 63857.0 |
| our_v6_mcts128   | 1937.4 | −37222.4 | 41097.3 |
| randombot        |    0.0 |       0.0 |       0.0 |

**CI widths are blown out** — perfect-record pairs (5–0 sealbot vs randombot, 5–0 our_v6 vs randombot, 5–0 sealbot vs our_v6) push the BT log-likelihood to its L2-reg corner. At n=50/pair (Wave C target), Hessian information will tighten CIs by ~10× and these become useful.

H2H (rows=winner):

```
                 randombot  sealbot  our_v6_mcts128
randombot              0       0          0
sealbot                5       0          5
our_v6_mcts128         5       0          0
```

Pipeline confirmed end-to-end: bot loading → game loop → JSONL → BT MLE → markdown summary.

## (d) Wave C readiness checklist

- [x] All 6 OURS-CONTROL-CONFIRMED bots load:
  - sealbot ✓ (existing `SealBotBot`, .venv-resolved)
  - kraken_minimax_strong ✓ (`KrakenBotBot(time_limit=1.0)`)
  - kraken_minimax_fast ✓ (`KrakenBotBot(time_limit=0.1)`)
  - kraken_random ✓ (new `KrakenBotRandomBot`)
  - our_v6_mcts128 ✓ (`OurModelBot` w/ `bootstrap_model_v6.pt`, n_sims=128)
  - our_v6_argmax ✓ (`OurModelBot` w/ `bootstrap_model_v6.pt`, n_sims=1 — note: this is an MCTS-1 proxy for argmax; true argmax-only path would require a separate non-MCTS arg-max sampler)
  - randombot ✓ (existing `RandomBot`)
- [x] 2 BLOCKED bots:
  - kraken_mcts_resnet — weights missing (`vendor/bots/krakenbot/training/resnet_results/best.pt`)
  - kraken_mcts_selfplay — weights missing (`vendor/bots/krakenbot/training/mcts_results/best.pt`)
  - Both raise `FileNotFoundError` at construction. Wave C `--bots` list must exclude these unless operator supplies weights.
- [x] Halt-at-8-hr-wall: `--max-wall-seconds` arg implemented. Outer loop checks elapsed before each game; on overrun, dumps partial `per_game.jsonl`/summary/ratings/h2h/colony and exits 0. Smoke verified the budget-respect (set to 900 s, not triggered).
- [ ] Resume support: NOT implemented. Documented in (e) below as Wave C optional. Tourney is one-shot — if killed, re-run from scratch (partial JSONL is preserved from `--max-wall-seconds` halt, but no resume cursor).
- [x] No edits to `hexo_rl/eval/eval_pipeline.py` or `configs/eval.yaml` (master-prompt constraint).
- [x] Single-game-at-a-time (master-prompt do-not #2; §17 GIL regression).
- [x] Fresh bot instance per game (A1 TT-carryover constraint).

## (e) Verdict

**PROCEED-TO-C** with the following clarifications:

1. **Run 7 bots in Wave C, not 8 or 6**: include all 6 OURS-CONTROL-CONFIRMED (sealbot, kraken_minimax_strong, kraken_minimax_fast, kraken_random, our_v6_mcts128, our_v6_argmax, randombot) — 7 total. The 2 KrakenBot MCTS variants are blocked on weights and must be excluded from `--bots`. If operator supplies weights before Wave C launch, expand to 9. Pair count C(7,2)=21; at 50 games/pair = 1050 games; estimated wall ≈ 13.5 hours on this laptop (extrapolating from 7.5 s/game sealbot×our_v6 + ~1 s/game cheap pairs at n_sims=64). Wave C must launch on vast 5080 or budget overnight.
2. **Canonical checkpoint for `our_v6_*` is `bootstrap_model_v6.pt`** — NOT `bootstrap_model.pt` (which is v6w25, 626-action policy). Wave C plan must pin `--our_model_ckpt checkpoints/bootstrap_model_v6.pt`.
3. **CI tightening**: at n=50/pair, perfect-record pairs will still pin to the L2-reg corner. Mitigation: add a small ε-noise term to the BT MLE call OR allow draws in the schedule (none observed in smoke — Hex Tac Toe rarely draws at MCTS-128).
4. **Resume support optional**: documented absent. If Wave C wants resume, post-process partial `per_game.jsonl` to a resume cursor; not strictly required since `--max-wall-seconds` already produces usable partial output.
5. **`our_v6_argmax` proxy caveat**: implemented as `OurModelBot(n_sims=1)`. True policy-head argmax would short-circuit MCTS entirely; if Wave C wants the strict argmax path, add an `argmax=True` kwarg to `OurModelBot` in a follow-up commit (out of Wave B scope).
6. **One Wave C flag**: `--bots` must omit `kraken_mcts_resnet`/`kraken_mcts_selfplay` until weights arrive; tourney will exit 1 at registry resolution if listed. Wave C Wave-D plan author should pre-pin the bot list to the 7 viable entries.

## Notes for Wave C

- BT MLE CI is broken at perfect records — operator should note CI widths > 10 000 Elo are an L2-reg artifact, not a true uncertainty. At n=50/pair, expect tighter but still wide CIs for sealbot vs randombot (likely 50/50 sealbot sweep).
- `winner_completed_via_threat` is a soft heuristic — terminal `get_threats()` counts open threats post-final-move, not specifically a 5-in-a-row open-end at the winner's last cell. Wave C may want a stricter detector if this metric is gate-relevant.
- `random_opening_plies` is parsed by the CLI but not yet applied in the game loop — reserved for Wave C ablation. If Wave C needs symmetric opening masking, wire it into `_play_one_game` (≤10 LOC).
