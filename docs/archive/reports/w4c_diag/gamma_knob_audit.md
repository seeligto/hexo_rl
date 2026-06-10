# §143 — γ-knob audit and W4C smoke v3 recommendation

**Date:** 2026-05-01
**Inputs:** `reports/w4c_diag/encoding_audit.md` (§142), `reports/w4c_diag/policy_diagnosis.md` (§141)
**Scope:** Read-only audit. Configs not modified. Tabulates current values, verifies §142 protocol changes (decay_steps 20K→200K, max_game_moves 200→100) landed in commit `e4c8b29`, recommends γ knob set for W4C smoke v3.

---

## 1. γ knob inventory (current values, sources, scope)

| knob                                          | current value          | source file:line                                         | scope                                                           |
|-----------------------------------------------|------------------------|----------------------------------------------------------|-----------------------------------------------------------------|
| selfplay temperature (initial)                | **1.0** (hardcoded)    | `engine/src/game_runner/worker_loop.rs:20-31`            | Rust self-play worker, all variants                             |
| selfplay temperature (floor `temp_min`)       | **0.05**               | `configs/selfplay.yaml:87` (`playout_cap.temp_min`)      | Rust worker `compute_move_temperature`                          |
| temperature schedule shape                    | quarter-cosine anneal  | `worker_loop.rs:20-31` (`compute_move_temperature`)      | hardcoded shape; threshold + floor knob-controlled              |
| `temperature_threshold_compound_moves`        | **15**                 | `configs/selfplay.yaml:86`                               | Rust self-play production path (cm at which τ clamps to floor)  |
| `mcts.temperature_threshold_ply`              | **30**                 | `configs/selfplay.yaml:26`                               | **EVAL/BOT ONLY** (`hexo_rl/selfplay/utils.py:32-34`, called from `our_model_bot.py`); NOT used by self-play production |
| `dirichlet_alpha`                             | **0.05**               | `configs/selfplay.yaml:22`                               | Rust self-play, root noise concentration                        |
| `epsilon` (= `dirichlet_epsilon`)             | **0.25**               | `configs/selfplay.yaml:24` → `pool.py:154`               | Rust self-play, root noise mass ratio                           |
| `dirichlet_enabled`                           | **true**               | `configs/selfplay.yaml:25`                               | Rust self-play, on/off gate                                     |
| Dirichlet site                                | root only              | `engine/src/mcts/mod.rs:373-393`, called `worker_loop.rs:489-492` | applied per move except intermediate plies (`moves_remaining==1 && ply>0`) |
| `max_game_moves` (PLIES)                      | **100**                | `configs/selfplay.yaml:58`                               | base                                                            |
| `max_game_moves` (variants)                   | **100**                | `gumbel_targets.yaml:23`, `gumbel_targets_desktop.yaml:11`, `gumbel_targets_5080_24t.yaml:10`, `gumbel_targets_epyc5090_48t.yaml:13` | host overrides       |
| `mixing.initial_pretrained_weight`            | **0.8**                | `configs/training.yaml:135`                              | trainer pretrain mixing                                         |
| `mixing.min_pretrained_weight`                | **0.1**                | `configs/training.yaml:134`                              | trainer pretrain mixing floor                                   |
| `mixing.decay_steps`                          | **200_000**            | `configs/training.yaml:133`                              | trainer pretrain mixing decay constant                          |
| pretrain weight formula                       | `max(min, init·exp(-step/decay))` | `hexo_rl/training/loop.py:575-576`            | trainer                                                         |

**Knobs flagged as hardcoded / non-configurable:**
- Initial selfplay temperature τ₀ = 1.0 — hardcoded at `worker_loop.rs:20-31`. No config knob.
- Cosine schedule shape — hardcoded. No knob.
- Dirichlet skip on intermediate plies — hardcoded at `worker_loop.rs:400, 484`. No knob.
- `temperature_threshold_ply` (the Python knob) is NOT used by self-play. Only `our_model_bot.py` reads it. Self-play uses `temperature_threshold_compound_moves`.

---

## 2. §142 protocol-change verification (commit `e4c8b29`)

| change                                         | claimed in §142 protocol  | verified at                                              | status |
|------------------------------------------------|---------------------------|----------------------------------------------------------|--------|
| `mixing.decay_steps`: 20_000 → 200_000         | training.yaml             | `configs/training.yaml:133` = `200_000`                  | ✅ landed |
| `selfplay.max_game_moves`: 200 → 100           | selfplay.yaml + variants  | `configs/selfplay.yaml:58` = 100                         | ✅ landed |
| variant override (laptop)                      | gumbel_targets.yaml       | `configs/variants/gumbel_targets.yaml:23` = 100          | ✅ landed |
| variant override (desktop)                     | gumbel_targets_desktop    | `configs/variants/gumbel_targets_desktop.yaml:11` = 100  | ✅ landed |
| variant override (5080 24t)                    | gumbel_targets_5080_24t   | `configs/variants/gumbel_targets_5080_24t.yaml:10` = 100 | ✅ landed |
| variant override (epyc5090 48t)                | gumbel_targets_epyc5090_48t | `configs/variants/gumbel_targets_epyc5090_48t.yaml:13` = 100 | ✅ landed |

Commit footprint matches `git show e4c8b29 --stat` (6 files, +11 −18). No drift.

**Stale variants NOT touched** (still at `max_game_moves: 200`, intentionally — calibration / sweep / phase118 historical):
`baseline_puct.yaml:10`, `calib_R1..R4.yaml`, `gumbel_full.yaml:20`, `sweep_*ch.yaml`, `_sweep_template.yaml`. None of these is the Phase 4.0 production launch path.

---

## 3. Pretrain weight trajectory (decay_steps = 200_000)

Formula: `w(step) = max(0.1, 0.8 · exp(−step / 200_000))` (loop.py:575-576).

| step    | w_pre    |
|--------:|---------:|
|       0 | 0.8000   |
|    1000 | 0.7960   |
|    2500 | 0.7901   |
|    5000 | 0.7802   |
|  **5500** | **0.7783** |
|   10000 | 0.7610   |
|   50000 | 0.6230   |
|  200000 | 0.2943   |

**Floor ≈ 0.78 at step 5500 — confirmed.** With decay_steps=200K the bootstrap corpus retains ~97% of its initial mixing weight through the W4C smoke window (≤10K steps), preserving the bootstrap-v6 policy distribution while self-play accumulates. The §141 finding (policy head intact, locus is search/encoding) makes this floor desirable: corpus stays load-bearing, self-play targets are not allowed to dominate before the encoding-fragmentation pathology can corrupt them.

---

## 4. Temperature trajectory (cosine anneal, current vs proposed thresholds)

Cosine schedule from `worker_loop.rs:20-31`:
```
τ(cm) = max(temp_min, cos(π/2 · cm / threshold))   if cm < threshold
τ(cm) = temp_min                                   otherwise
```
where `cm` = compound move (= `(ply + 1) // 2`), `temp_min = 0.05`.

| cm | ply~ | τ @ thr=15 (current) | τ @ thr=12 (proposed) | τ @ thr=10 |
|---:|-----:|---------------------:|----------------------:|-----------:|
|  0 |    0 | 1.000 | 1.000 | 1.000 |
|  4 |    8 | 0.914 | 0.866 | 0.809 |
|  7 |   14 | 0.743 | 0.609 | 0.454 |
|  8 |   16 | 0.669 | 0.500 | 0.309 |
| 10 |   20 | 0.500 | 0.259 | **0.050** |
| 12 |   24 | 0.309 | **0.050** | 0.050 |
| 13 |   26 | 0.208 | 0.050 | 0.050 |
| 15 |   30 | **0.050** | 0.050 | 0.050 |
| 20 |   40 | 0.050 | 0.050 | 0.050 |

**Current state:** τ ≈ 0.31 at ply 24 (the §142 fragmentation entry boundary at ply 31 is reached while temperature is still ~0.21). The model still has meaningful sampling randomness in the very plies where the encoding window starts dropping stones, so visit-count sampling actively selects fragmenting moves.

---

## 5. Recommendations — W4C smoke v3 γ-knob set

§142 ranked Option γ (tighten exploration) as the cheapest mitigation that does not invalidate the bootstrap-v6 distribution. The §141 finding that the policy head is intact (top-1 agree 69% corpus, 90% threat, Spearman ρ ≥ 0.66 on real positions) means we can lean on the prior more aggressively — less noise, less random-walk exploration.

### γ.1 — Tighten temperature drop

| knob | current | recommended | new τ trajectory |
|---|---|---|---|
| `playout_cap.temperature_threshold_compound_moves` | 15 (≈ greedy by ply 30) | **10** (greedy by ply 20) | τ=0.05 at ply 20, τ=0.46 at ply 14 |

**Rationale:** §142 fragmentation pivot at ply 31 (single-window) and ply 36 (any-cluster, ≥5%). Forcing greedy floor by ply 20 means MCTS visit-count sampling is effectively argmax through the entire window in which the encoding boundary is crossed. Removes the explore-into-fragmentation feedback loop while preserving full exploration during the opening (ply 0–14 still has τ ≥ 0.45). Cheaper alternative: thr=12 (greedy by ply 24, τ=0.31 at ply 20). Pick **thr=10** as the more decisive cut — §142 self-play games already showed near-deterministic bot opposition (SealBot lines stay inside window), so we want self-play to mimic that regime as soon as the opening transposition is over.

**No new knob needed.** `selfplay_temperature_drop_ply` does not exist in the Rust path; the analog is `temperature_threshold_compound_moves`. Do not edit `mcts.temperature_threshold_ply` (Python eval-only knob).

### γ.2 — Lower Dirichlet root noise mass

| knob | current | recommended |
|---|---|---|
| `mcts.epsilon` (= `dirichlet_epsilon`) | 0.25 | **0.10** |
| `mcts.dirichlet_alpha` | 0.05 | unchanged (Go-regime per §115) |
| `mcts.dirichlet_enabled` | true | unchanged |

**Rationale:** §141 quantified policy-head preservation: top-1 mass on legal cells = 0.436 (corpus) and 0.597 (threat); Spearman ρ ≥ 0.66 on every real-position category. The 25% noise mass calibration is the AlphaZero default for from-scratch training where the policy is initially uniform. Bootstrap-v6 already carries useful priors; 25% noise overrides them at exactly the cells the bootstrap distinguished. Drop to 10% — enough to prevent root prior collapse, light enough to let the §141-validated head guide search. Combined with γ.1, this also cuts the second §142 mechanism (far-cell exploration whose Q is uncalibrated).

α stays at 0.05 — already tuned for hex branching (§115, §116 #4). Changing α and ε together would confound the smoke verdict.

### γ.3 — Max game moves

| knob | current | recommended |
|---|---|---|
| `selfplay.max_game_moves` | 100 | **100 (UNCHANGED, per operator)** |

**Rationale:** §142 originally suggested 100 → 80. Operator deferred. Keep 100. Cap still truncates the random-walk tail (was 200 pre-§142). γ.1 + γ.2 alone should reduce mid-game fragmentation; if the smoke fails on `pct_outside_any med < 30% at ply 100`, revisit a 100 → 80 follow-up.

### γ.4 — Pretrain mixing schedule

No change. `decay_steps=200_000`, `initial=0.8`, `min=0.1` — already corrected in `e4c8b29`. Floor at step 5500 = 0.78 retains bootstrap-v6 signal through the W4C smoke window.

---

## 6. Implementation checklist

Edit set is minimal — two values in one file:

1. **`configs/selfplay.yaml:24`** — `epsilon: 0.25` → `epsilon: 0.10`
   - update inline comment if desired (§143 reference)
2. **`configs/selfplay.yaml:86`** — `temperature_threshold_compound_moves: 15` → `temperature_threshold_compound_moves: 10`
   - inline comment is currently absent for this line; add `# §143: was 15; greedy floor by ply 20 to skip the §142 fragmentation pivot at ply 31`

No variant overrides needed: the `gumbel_targets*.yaml` set inherits `playout_cap` and `mcts` blocks from `selfplay.yaml`. Variants only override `max_game_moves`, `n_workers`, `inference_*`, `playout_cap.fast_prob`, `completed_q_values`, `gumbel_mcts`. (Verified via `grep -rn 'epsilon\|temperature_threshold_compound' configs/variants/` — no hits.)

No Rust changes needed. `temp_min`, `temperature_threshold_compound_moves`, `dirichlet_alpha`, `dirichlet_epsilon` all flow through `pool.py:139-155` into `SelfPlayRunner` ctor (`engine/src/game_runner/mod.rs:137`). Recompile not required — Python-only edit.

**Validation gate (per §142 §7):**
- 5K-step smoke from `bootstrap_model.pt`
- SealBot WR ≥ 10% (midpoint between ckpt_5500 1.3% and bootstrap 24%)
- `pct_outside_any med < 30% at ply 100` (re-run `scripts/diag_encoding_window_audit.py`)
- if both pass → fold γ.1+γ.2 into Phase 4.0 launch
- if SealBot WR < 5% → fall back to Option α (cap `LEGAL_MOVE_RADIUS` in `engine/src/board/moves.rs:9`)

---

## 7. Caveats

- The current schedule is a *cosine anneal*, not a step function. "Greedy by ply X" means τ reaches the 0.05 floor at ply X; τ values just below ply X are still ≥ 0.05 + small. The audit table in §4 makes the trajectory explicit.
- `mcts.temperature_threshold_ply: 30` is dead config for self-play, live config for `our_model_bot.py`. Leaving it untouched is intentional — bot eval and self-play use different temperature paths. If a future cleanup wants to delete the Python path, that is a separate scope.
- The §141 colony divergence (top-1 agree 18.5%, ρ=0.41) is *not* a regression per `feedback_colony_fraction.md`. γ-knob tuning should not be measured against colony positions; use corpus_midgame, threat, and SealBot positions for the smoke verdict.
- This audit assumes the Rust self-play path is the only training path. The Python `hexo_rl/selfplay/worker.py` exists but is wrapped by `our_model_bot.py` (eval) — verified by grep. If a non-eval caller appears, `mcts.temperature_threshold_ply` becomes load-bearing and the audit needs to widen.
