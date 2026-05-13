<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §70 — Phase 4.0 Overnight Run — Mode Collapse Diagnosis

**Status:** RESOLVED at §73 — Dirichlet root noise ported to Rust `engine/src/game_runner.rs` (commit `71d7e6e`). Q17 closed. Diagnostics complete 2026-04-09; run `dcf8cbba5b9f485987880055e9cb6ea7` PAUSED at `checkpoint_00017428.pt`. Full artefacts: `archive/diagnosis_2026-04-10/`.

#### Verdict

Root cause: **no Dirichlet root noise on the Rust training path.** The PyO3 `apply_dirichlet_to_root` (`engine/src/lib.rs:454`, impl `engine/src/mcts/mod.rs:321-337`) was only wired through the Python `SelfPlayWorker` (eval / benchmark / `OurModelBot`). When the training self-play loop migrated to `engine::SelfPlayRunner` on 2026-03-30 (two days after the Dirichlet feature landed), the injection was not carried across — **unported feature, not regression**. Runtime trace under feature `debug_prior_trace`: 30/30 records `site: game_runner`, 0/30 `site: apply_dirichlet_to_root`.

Result: MCTS rubber-stamps a sharp prior → self-play enters deterministic fixed point → training targets match network outputs → no gradient signal to break the equilibrium. Dashboard `policy_entropy` averaged the high-entropy pretrain stream with the collapsed selfplay stream and masked the collapse for 16,880 steps.

#### Evidence — round-robin eval (collapse signature)

| Matchup | Score | Game length |
|---|---|---|
| ckpt_13000 vs ckpt_14000 | 100/0 P1 | exactly 25 moves, carbon-copy |
| ckpt_14000 vs ckpt_15000 | 50/0 P1 + 50 draws | 31-33 moves, carbon-copy |
| **ckpt_15000 vs RandomBot** | **50/0 P1** | **11-33, varied** |

RandomBot anchor proves the net has real knowledge; the *self-play distribution* collapsed. τ=1.0 sampling check (20 games) confirmed temperature sampling functionally correct.

#### Diagnostic A — static audit + feature-gated runtime trace (collapsed)

Static audit + git archaeology + 30-worker runtime trace under `debug_prior_trace` feature (gated on `HEXO_PRIOR_TRACE_PATH`) all confirm: zero `apply_dirichlet_to_root` calls on the Rust training path; 30/30 records `site: game_runner`. Python path Dirichlet is functionally correct, just dead code for training. Variant disclosure: trace under `gumbel_targets`; behaviour identical to `baseline_puct` (only `completed_q_values` differs). Artefacts: `archive/diagnosis_2026-04-10/diag_A_*`.

#### Diagnostic B — raw policy sharpness across checkpoints (collapsed)

500 positions stratified by phase, K=0 window. Per-checkpoint summary:

| Checkpoint | H(π) mean | top-1 mean | eff. support |
|---|---|---|---|
| bootstrap_model.pt | 2.665 | 0.379 | 21.48 |
| ckpt_13000-17428 (band) | 1.49-1.70 | 0.50-0.54 | 5.8-9.7 |

Stuck **fixed point**, not progressive collapse — entire post-bootstrap band sits within 0.21 nats. Mid-game bucket (cm 10-24) is worst: p10 down to 0.08 nats. `best_model.pt` ≡ `bootstrap_model.pt` (SHA-256 `ed07ecbe6a73` on first conv) — never promoted during P3. Restart point selection should be based on **buffer composition** not entropy rank; fresh bootstrap is the cleanest option. Artefacts: `archive/diagnosis_2026-04-10/diag_B_sharpness.md`.

#### Diagnostic C — temperature schedule + MCTS visit distribution (collapsed)

Temperature schedule (`engine/src/game_runner.rs:510-515`):

```
τ(cm) = max(temp_min, cos(π/2 · cm / threshold))   if cm < threshold
τ(cm) = temp_min                                   if cm ≥ threshold
```

| compound_move | 0 | 5 | 10 | 14 | 15 | 16 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|
| τ | 1.0000 | 0.8660 | 0.5000 | 0.1045 | 0.0500 | 0.0500 | 0.0500 | 0.0500 |

Doc reconciliation: §36 block + `docs/01_architecture.md` updated 2026-04-19 to match Rust quarter-cosine-per-compound-move formula and `selfplay.playout_cap.{temperature_threshold_compound_moves, temp_min}` config keys. Legacy ply-based `get_temperature` in `hexo_rl/selfplay/utils.py` retained for eval-adjacent paths only (`OurModelBot`, `benchmark_mcts`). See `reports/c_series_doc_fixes_2026-04-19.md`.

MCTS visit stats (30 cm=0 records from training trace): H(π_prior) mean 1.340, H(π_visits) mean 1.213, Δ 0.127 nats; top-1 visit fraction 0.526; effective support ~3.4 children. MCTS sharpens the prior by only 0.13 nats — picks among top 3 children and rubber-stamps. 30 records are all cm=0 / empty board (`GAME_RUNNER_CAP` saturated under 14 worker contention); per-game-phase variation not in this data but sufficient to demonstrate the rubber-stamp behaviour.

#### Locked parameters (still apply post-§73)

- Temperature: `temp_min = 0.05`, `temperature_threshold_compound_moves = 15`.
- `entropy_reg_weight = 0.01`.
- Dirichlet injection params on Python eval path: `epsilon = 0.25` over `n_children` (`hexo_rl/selfplay/worker.py:138-145`); §73 port carries these into Rust.

#### Follow-up landed elsewhere

- §73: Dirichlet port to Rust training path (commit `71d7e6e`).
- Monitoring split (`policy_entropy_pretrain` / `policy_entropy_selfplay`) tracked under Q17 remediation.

---

