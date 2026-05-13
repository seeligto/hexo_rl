<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §67 — LR scheduler bug fix + total_steps / decay_steps co-design + named Gumbel variants

**Problem:** Both desktop and laptop runs were using the hardcoded `T_max = 50_000` fallback in `trainer.py:145` (L2 from April architecture review). With `decay_steps = 300_000`, the LR collapsed to `eta_min = 1e-5` at step 50K while the mixing weight was still bootstrap-heavy. Self-play never dominated. Both runs are unrecoverable and restart from `bootstrap_model.pt`.

**Fix — trainer.py:** Removed all silent defaults. `_build_scheduler` now raises `ValueError` if `total_steps` or `eta_min` is absent from config. Resolution order for `total_steps`:
1. `--iterations` CLI flag (sets `combined_config["total_steps"]` before Trainer init)
2. `config["total_steps"]` (from `training.yaml`)
3. `ValueError` listing both options

**Fix — configs/training.yaml:**

| Key | Old | New | Rationale |
|---|---|---|---|
| `total_steps` | absent (50K fallback) | `200_000` | Sets LR horizon explicitly |
| `eta_min` | absent (1e-5 fallback) | `2e-4` | ~10% of peak lr=0.002; prevents LR floor being too low |
| `mixing.decay_steps` | `300_000` | `70_000` | Co-designed with total_steps (see below) |

**total_steps / decay_steps co-design rationale:**

| Phase | Steps | What's happening |
|---|---|---|
| Bootstrap dominant | 0 → 70K | pretrain_weight decays 0.8 → 0.1; replay buffer fills with self-play data |
| Self-play dominant | 70K → 200K | pretrain_weight = 0.1 (floor); model trains almost entirely on its own games |
| LR floor | ~200K | cosine reaches `eta_min = 2e-4`; scheduler done |

Rule of thumb: `decay_steps ≈ 0.35 × total_steps`. Keeps bootstrap phase at ~35% of run length, leaving 65% for self-play consolidation before LR collapses.

**Named Gumbel variants (`configs/variants/`):**

Three named override files, each deep-merged on top of `selfplay.yaml` via `--variant`:

| Variant | `gumbel_mcts` | `completed_q_values` | Host |
|---|---|---|---|
| `gumbel_full` | true | true | Desktop (RTX 3070) |
| `gumbel_targets` | false | true | Laptop / cloud |
| `baseline_puct` | false | false | Ablation baseline |

`selfplay.yaml` reverted to `gumbel_mcts: false`, `completed_q_values: false` (explicit baseline). Variant must be specified explicitly — no implicit activation.

**Usage:**
```bash
make train VARIANT=gumbel_full          # desktop
make train VARIANT=gumbel_targets       # laptop
make train.resume VARIANT=gumbel_full   # resume with same variant
# Or directly:
python scripts/train.py --checkpoint checkpoints/bootstrap_model.pt --variant gumbel_full
```

**Restart plan:** Both hosts restart from `bootstrap_model.pt` at step 0. The dual-host run is an informal comparison of root search strategy (Gumbel vs PUCT) with policy target type held constant (`completed_q_values: true` on both). Desktop uses `gumbel_full`; laptop uses `gumbel_targets`.

---

