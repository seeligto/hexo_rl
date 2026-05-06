# Phase 4.0 architecture baseline

Starting config for self-play RL (do not exceed without benchmarking):

- Network: 12 residual blocks × 128 channels, GroupNorm(8), SE blocks on every block (§99).
- Input: 8 planes (§131, post §97 18-plane build). Chain-length planes (Q13) stored in ReplayBuffer `chain_planes` sub-buffer.
- Value head: global avg + max pooling → Linear(2C → 256) → ReLU → Linear(256 → 1) → tanh.
  Loss: BCE on the pre-tanh logit against `(z+1)/2`.
- Auxiliary heads (training only — never called from InferenceServer / evaluator / MCTS):
  - opp_reply — mirror of policy head, cross-entropy, weight 0.15.
  - ownership — Conv(1×1) → tanh → (19×19), spatial MSE, weight 0.1.
  - threat — Conv(1×1) → raw logit → (19×19), BCEWithLogitsLoss with
    `pos_weight = threat_pos_weight` (default 59.0, Q19), weight 0.1.
  - chain_head — Conv(1×1) → (6, 19, 19), smooth-L1 (Huber), weight
    `aux_chain_weight: 1.0` (§92; target comes from the replay-buffer
    chain sub-buffer post-§97, not from the input slice).
- Temperature: per-compound-move quarter-cosine schedule with hard
  `temp_min: 0.05` floor at compound_move ≥ 15 (Rust:
  `compute_move_temperature` in `engine/src/game_runner/worker_loop.rs:20-31`;
  call site at `worker_loop.rs:346-353`). §36 text reconciled against
  code (see §70 C.1 resolution in `docs/07_PHASE4_SPRINT_LOG.md`).
  **§156 R12 verdict (2026-05-06):** cosine schedule is the sole
  load-bearing knob behind the §155 R10 91% draw lock under v7full
  warm-start. Variant opt-out at `temperature_threshold_compound_moves: 0`
  + `temp_min: 0.5` (see `configs/variants/w4c_smoke_v7_5080.yaml`).
  Top-level default unchanged pending cold-start data — §157 Gate 5 S2
  (variant-pinned, comment-only at top level).
- ZOI: candidate moves restricted to hex-distance ≤ 5 of last 16 moves
  (fallback to full legal set if < 3 candidates) — post-search move
  selection only; does not reduce MCTS tree branching (§77).
- Checkpoint loading: pre-§99 (BatchNorm) checkpoints refuse to load —
  `normalize_model_state_dict_keys` raises `RuntimeError` rather than
  silently corrupting trunk weights via `strict=False`. Retrain from
  `bootstrap_model.pt` when crossing §99.
- torch.compile: landed §116 (2026-04-24, reduce-overhead). Production
  selfplay uses **trace** not compile per §124 (2026-04-25) — compile
  and trace deliver same selfplay throughput within noise; trace wins
  on simplicity (no Dynamo guard cost, no cudagraph TLS thread issue).
  See `docs/rules/perf-targets.md` for benchmark methodology. compile
  re-enable for selfplay gated on Q35 dispatch fix.
- Replay buffer: start at 250K samples, grow toward 1M as training
  stabilises (§79). HEXB v6 on-disk format. v5 + v4 hard-rejected at load post-§131 P1(b).
  buffer_sample_raw target is ≤ 1,550 µs (§113 recalibration) — see
  `docs/rules/perf-targets.md`.
- Graduation gate (§101, §101.a, post-§157 three-part): self-play
  workers consume `inf_model` weights, which track the `best_model`
  anchor (not `trainer.model`). Sync fires only on graduation or on
  cold-start load. Gate AND-combines:
  (1) `wr_best ≥ promotion_winrate` (default 0.55 over 400 games;
  raised 200→400 per calibration 2026-04-17),
  (2) `ci_lo > 0.5` (binomial 95% CI; cuts false-positive rate at
  n=400 from ~9% to <1% under null), and
  (3) `wr_bootstrap_anchor ≥ min_winrate` (default 0.45) when
  `gating.bootstrap_floor.enabled = true` (default true post-§157
  Gate 5 S1, anchored to frozen `bootstrap_model_v7full.pt`; missing
  measurement = failure, defensive). Predicate at
  `hexo_rl/eval/eval_pipeline.py:401-444`.
  Promotion copies from the `eval_model` snapshot (the one that was
  actually scored), not from drifted `trainer.model`. Eval cadence
  split via per-opponent
  `stride`: effective eval_interval is 5000 steps (`training.yaml`
  overrides `eval.yaml` per §101 H1); best_checkpoint every 5000
  (`stride: 1`), SealBot every 20000 (`stride: 4`), random every 5000
  (`stride: 1`).
- Selective policy loss (§100): per-move coin-flip chooses full-search
  (600 sims) vs quick-search (100 sims). Policy / opp_reply losses
  gated on `is_full_search=1`; value / chain / ownership / threat
  losses apply to all rows. Mutex with game-level `fast_prob` enforced
  at pool init (`fast_prob: 0.0` base; `full_search_prob: 0.25`).
- ELO benchmark target: SealBot (replaces Ramora0 as external reference).
- Gumbel MCTS (per-variant, not per-host). `configs/selfplay.yaml` base
  is `gumbel_mcts: false, completed_q_values: false`; enable via
  `--variant`:
  - `gumbel_full` — Gumbel root search + completed-Q targets. Desktop
    Phase 4.0 sustained run (`gumbel_full`, n_workers=10 per §81 D3).
  - `gumbel_targets` — PUCT search + completed-Q targets (P3 sweep
    winner per §69; `max_game_moves: 200` post-§76; `inf_bs=64,
    wait_ms=4` post-§90).
  Gumbel provides root noise by construction; Dirichlet is additionally
  applied post-§73 in both branches.

Resolved before / during Phase 4.0:

- [x] Open Question 6: sequential vs compound action space
- [x] Open Question 5: supervised→self-play transition schedule
- [x] Q13: chain-length planes (§92 landed as input, §97 moved to aux
      sub-buffer)
- [x] Q17: self-play mode collapse (Dirichlet port §73, commit `71d7e6e`)
- [x] Q19: threat-head BCE class imbalance (`threat_pos_weight = 59.0`;
      §92 landing)
- [x] Q25: throughput variance from old input layout (reverted by §97; legacy
      payload no longer exists)
- [ ] Open Question 2: value aggregation strategy (min/mean/attention) — HIGH
- [ ] Q3, Q8, Q9, Q10, Q15, Q16, Q18, Q21 — see `docs/06_OPEN_QUESTIONS.md`
