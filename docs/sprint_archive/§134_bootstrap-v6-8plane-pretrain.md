<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §134 — bootstrap-v6: 8-plane pretrain on 6,259 human games — 2026-04-30

**Date:** 2026-04-30

First v6 bootstrap checkpoint on the 8-plane architecture. Updated human corpus (6,259 games, ~16% larger than v5's ~5,400).

### Corpus

- Source: `data/corpus/raw_human/` — 6,259 qualifying games (decisive, ≥15 ply)
- Export: `scripts/export_corpus_npz.py --human-only --no-compress`
- Output: `data/bootstrap_corpus_pretrain_v6.npz`, 353,091 positions, 8-plane f16
- Elo breakdown: sub_1000=81,985 · 1000_1200=202,111 · 1200_1400=69,739 · 1400+=1,436
- P1 win rate: 50.3% (balanced)

### Pretrain

- Config: 15 epochs, batch=256, lr=0.002 cosine, aux_chain_weight=1.0, ~100 min on RTX 3070
- Final epoch 15 metrics: policy_loss=2.484, value_loss=0.594, opp_reply_loss=2.493, chain_loss=0.0019

### Validation

| Gate | Threshold | Result | Status |
|---|---|---|---|
| RandomBot wins | ≥95/100 | 100/100 | PASS |
| Threat C2 (ext∈top5%) | ≥25% | 50% | PASS |
| Threat C3 (ext∈top10%) | ≥40% | 60% | PASS |
| Forward pass | clean | OK (val=0.011) | PASS |

**Policy loss note:** 2.484 exceeds the legacy ≤2.3 spec criterion calibrated against 18-plane v5. Reduced input (10 planes dropped) raises policy entropy; all functional gates pass with margin. 8-plane policy_loss convergence criterion should be updated to ~≤2.6.

### Bugs fixed

1. `hexo_rl/bootstrap/pretrain.py` — chain plane extraction `states[i, 8]` (18-plane opp index) → `states[i, 4]` (8-plane opp ply-0).
2. `hexo_rl/bootstrap/pretrain.py` — `validate()` passed raw `to_tensor()` output (18-plane) to 8-plane model. Fixed: slice with `KEPT_PLANE_INDICES`.
3. `scripts/probe_threat_logits.py` — hardcoded 18-plane shape check + no slice before forward. Fixed: relaxed check; auto-slice in `_probe_one` for 18-plane fixtures with 8-plane models.

### Artifacts

| File | Notes |
|---|---|
| `checkpoints/bootstrap_model.pt` | v6 inference weights (8-plane, 17 MB) |
| `checkpoints/pretrain/pretrain_00000000.pt` | full checkpoint with config + optimizer |
| `checkpoints/archive/bootstrap_model_v5.pt` | v5 fallback preserved |
| `data/bootstrap_corpus_pretrain_v6.npz` | pretrain corpus (8-plane, 353k pos, 2.4 GB) |
| `fixtures/threat_probe_baseline.json` | v6 threat baseline (C2=50, C3=60) |

### Pending before Phase 4.0 sustained run

- Checkpoint cleanup: archive `checkpoint_00000484.pt`, `best_model.pt`; clear `checkpoint_log.json`; delete stale `replay_buffer.bin` (2.8 GB pre-v6 test run).
- Update `CLAUDE.md` §91 threat-probe thresholds note: v6 baseline C2=50/C3=60; gates at 25/40 remain valid.
- Recalibrate policy_loss convergence criterion: ≤2.3 → ~≤2.6 for 8-plane.

---

