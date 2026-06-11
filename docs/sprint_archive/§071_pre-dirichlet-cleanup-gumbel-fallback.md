<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §71 — Pre-Dirichlet-Fix Cleanup & Gumbel Fallback Verification — 2026-04-10

Cleanup and instrumentation pass before the Dirichlet port session. No training runs
started. Commits are: `chore(scripts)`, `feat(monitoring)`, `diag(gumbel)`,
`docs(sprint)` — four independent commits, each self-contained.

---

#### 1. Gumbel path verification

**Static audit** (`archive/verify_gumbel_2026-04-10/diag_static.md`):

Three questions answered with line-number citations from
`engine/src/game_runner.rs`:

- **Q1 — noise freshness:** `GumbelSearchState::new()` is called inside
  the per-move loop (`line 362`) at `line 459`. Gumbel(0,1) values are
  drawn from a per-worker RNG (`let mut rng = rng()`, `line 344`) for
  every root child on every move — fresh per move, not cached across
  games or moves.

- **Q2 — noise is added, not discarded:** The Gumbel vector is added
  to `log_prior[i]` at candidate-selection time (`line 70`) and again
  at every Sequential Halving phase (`line 130`:
  `gumbel[i] + log_prior[i] + sigma(Q)`). It is not discarded.

- **Q3 — `effective_m` formula:** `line 445`:
  `gumbel_m.min(game_sims).min(tree.root_n_children())`. Matches the
  §61/§62 spec; no hardcoded constants.

**Runtime trace** (`archive/verify_gumbel_2026-04-10/diag_trace.jsonl`,
`verdict.md`): 30 records captured under `gumbel_mcts: true` from
`checkpoint_00015000.pt`.

Key finding: visit arrays differ across workers at ply=0 (top-1 visit
cells: workers 0/1/2 → cells 13/15/15 — disagreement). The
`top_visit_fraction` mean is **0.24** vs **0.65** in the PUCT §70 trace
— a 41 percentage-point reduction, confirming Sequential Halving spreads
budget across multiple candidates rather than rubber-stamping the prior.

**Verdict: Gumbel noise is functionally active on the training path.
Switching to `gumbel_mcts: true` is a validated fallback remediation if
the Dirichlet port encounters issues.**

---

#### 2. Policy-entropy split monitoring

The §70 diagnosis found that `policy_entropy` was 2.54 nats
(healthy-looking) during the P3 overnight run while the selfplay stream
was stuck at ~1.5 nats in a fixed-point. The pretrain corpus (~63%
buffer share) masked the selfplay collapse for ~13k training steps.

Changes landed in `feat(monitoring)`:

- **`trainer.py`:** `_train_on_batch` and `train_step_from_tensors`
  accept `n_pretrain: int = 0`. After the combined entropy is computed,
  splits `log_policy` by index: `[:n_pretrain]` → pretrain,
  `[n_pretrain:]` → selfplay. On the single-buffer path
  (`n_pretrain == 0`) the pretrain field is `NaN` and selfplay equals
  the combined metric.

- **`scripts/train.py`:** Passes `n_pretrain=n_pre` to
  `train_step_from_tensors`. Emits `policy_entropy_pretrain` and
  `policy_entropy_selfplay` in the `training_step` event and structlog
  `train_step` entry alongside the existing `policy_entropy` (retained
  for backwards compatibility).

- **Terminal dashboard:** New `entropy` subsection row:
  `entropy  combined X.XX  │  pretrain X.XX  │  selfplay X.XX  (collapse < 1.5 nats)`.
  Selfplay colored red (`< 1.5 nats`), yellow (`1.5–2.0`), green (`≥ 2.0`).
  Separate `selfplay entropy X.XX — selfplay mode collapse` alert.

- **Web dashboard:** `collapse_threshold_nats: 1.5` added to
  `/api/monitoring-config`. Chart.js entropy panel gains pretrain
  (blue dashed) and selfplay (red dashed) traces; second reference line
  at 1.5 nats labeled "collapse threshold".

Smoke test (2026-04-10): `policy_entropy_pretrain: 2.74`,
`policy_entropy_selfplay: 2.75` visible in JSONL after 24s of smoke
from `bootstrap_model.pt`. Both fields present; dashboards render
without crash.

---

#### 3. Known-good checkpoint inventory

> Entropy values from `diag_B_sharpness.md` (K=0 cluster window only,
> ~several tenths below min-pool; see K=0 caveat in that file).
> Weight fingerprints are SHA-prefix of the first layer weight tensor.

| Checkpoint | weight_fp | H(π) mean (K=0) | Use / Status |
|---|---|---|---|
| `bootstrap_model.pt` | `ed07ecbe6a73` | 2.665 | **Primary restart point** for Phase 4.0 post-Dirichlet-fix. Clean pretrained weights, highest entropy anchor. |
| `checkpoint_00013000.pt` | — | 1.666 | Earliest post-collapse; no less collapsed than later checkpoints. Do not restart from. |
| `checkpoint_00010000.pt` | — | — | Does not exist in current checkpoints/ (was never saved). |
| `best_model.pt` | `ed07ecbe6a73` | 2.665 | **Identical to `bootstrap_model.pt`** (same weight fingerprint). Was initialised at training start but never promoted. Do not treat as an independent reference. |
| collapsed run ckpt 13k–17k | — | 1.49–1.70 | Fixed-point collapse, no entropy gradient. Retained for forensics. **Do not restart from.** |

---

#### 4. Pre-run checklist

Walk this checklist before launching the next Phase 4.0 sustained run:

```
[ ] Dirichlet ported to engine/src/game_runner.rs, unit-tested
[ ] debug_prior_trace re-run confirms apply_dirichlet_to_root
    records appear on the training path (inverse of the §70 proof)
[ ] checkpoints/replay_buffer.bin archived to .bak
[ ] collapsed checkpoints (13k-17k) moved to
    checkpoints/collapsed_2026-04-09/ (do not delete)
[ ] make test.all passes
[ ] make bench.full passes all 10 §66 targets (or environmental
    noise explicitly acknowledged per 2026-04-09 bench run notes)
[ ] policy_entropy_pretrain and _selfplay fields visible in the
    JSONL log of a smoke run
[ ] both dashboards render the split entropy without error
[ ] 2-hour smoke from bootstrap_model.pt produces non-identical
    self-play games (eval_diagnostic.py with --temperature 1.0,
    >= 5 distinct game lengths in 20 games)
[ ] 6-hour entropy-checkpoint plan written: selfplay-stream H
    must be above 1.5 nats and trending up or stable at the
    6-hour mark, otherwise pause and investigate
```

---

