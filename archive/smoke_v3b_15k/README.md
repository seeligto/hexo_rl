# smoke_v3b @ 15k steps

**Archived:** 2026-04-16  
**Reason:** Pre-architectural-changes snapshot — KrakenBot analysis complete, changes incoming

---

## Run summary

| Item | Value |
|------|-------|
| Steps | 15,000 (run stopped at 15,272; 15k is milestone) |
| Variant | `gumbel_targets` (PUCT search + completed Q-value targets, no Gumbel root) |
| Bootstrap | `checkpoints/bootstrap_model.pt` |
| Training run IDs | `d687569c` (steps ~0–9,300) · `10cc8d56` (steps ~13,600–15,272) |
| Draw rate @ 15k | ~39.4% (stable, well below 70% kill threshold) |
| Policy loss @ 15k | ~1.07 (crossed below 1.0 intra-window; trending down) |
| Player balance | 52.7% X / 47.3% O (from eval log at step 5k) |

### Eval results (step 5,000 checkpoint — from `logs/eval_step5000.log`)

| Opponent | W / G | Win rate |
|----------|-------|---------|
| RandomBot | 20 / 20 | 100.0% |
| SealBot (t=0.5s) | 0 / 50 | 0.0% (expected) |
| best\_checkpoint | 1 / 50 | 2.0% |

Bradley-Terry ratings (step 5k): SealBot +2298 >> best\_ckpt +464 >> player\_56 0 >> ckpt\_5k -212 >> random -2550

### Draw rate trend

| Step | Draw rate | Notes |
|------|-----------|-------|
| 68 | 25% | First measurement |
| 1712 | 34.4% | Stable plateau begin |
| 1928 | 35.3% | Buffer capped (250K) |
| 2616 | 36.2% | RSS stable confirmed |
| 3804 | 40.7% | Acceleration phase |
| 4392 | 43.2% | Approaching 45% flag |
| 5003 | 44.7% | Predecessor run killed (Phase 3 probe FAIL) |
| 9280 | 41.5% | Current run (d687569c), re-stabilised |
| 15230 | 39.4% | Final 15k milestone |

Draw rate is well-behaved. The draw kill threshold is 70%.

---

## Probe trend (threat-logit probe, C1/C2/C3)

| Checkpoint | C1 contrast | C2 ext∈top5 | C3 ext∈top10 | Verdict |
|-----------|-------------|-------------|--------------|---------|
| predecessor 5k | +5.055 | 20% | 35% | FAIL (C2/C3) |
| 5,500 | +6.157 | 20% | 20% | FAIL |
| 6,000 | +4.517 | 20% | 25% | FAIL |
| 9,000 | +6.184 | 20% | 20% | FAIL |
| 13,500 | +4.418 | 20% | 25% | FAIL |

**Pattern:** Threat detector (C1) strong throughout — contrast well above +0.38 threshold.  
Policy (C2/C3) not yet steering toward threat extensions. Expected at this stage with 74–80% bootstrap mix.  
Policy should catch up after bootstrap weight decays further (decay floor 10% at step ~20k).

Kill conditions: only applied at specific milestone probes (5k on predecessor run). Current run not killed on probe; decision was to continue past 15k to observe policy improvement.

---

## Model config (exact values from configs/)

```yaml
# model.yaml
board_size: 19
in_channels: 24          # 2 stone + 14 history + 2 scalar + 6 Q13 chain planes
res_blocks: 12
filters: 128
se_reduction_ratio: 4    # SE blocks on every residual block

# training.yaml (key values)
batch_size: 256
lr: 0.002
weight_decay: 0.0001
grad_clip: 1.0
lr_schedule: cosine
total_steps: 200_000
mixing:
  initial_pretrained_weight: 0.8
  decay_steps: 20_000
  min_pretrained_weight: 0.1

# selfplay.yaml (key values)
n_simulations: 400
c_puct: 1.5
fpu_reduction: 0.25
dirichlet_alpha: 0.3
n_workers: 14
max_game_moves: 200      # plies (stone placements)

# configs/variants/gumbel_targets.yaml (active overrides)
selfplay:
  gumbel_mcts: false
  completed_q_values: true
  max_game_moves: 200
  playout_cap:
    fast_prob: 0.0       # fast games disabled (draws at 94% on this variant)
```

**Policy head:** flat 362-dim log-softmax (single-stone, sequential)  
**Value head:** dual avg+max pool → FC256 → tanh (BCE loss on logit)  
**Aux heads active:** opp reply (0.15), ownership (0.1), threat (0.1 / pos\_weight=59), Q13 chain (1.0)

---

## Hardware / throughput @ step ~9,360

| Metric | Value |
|--------|-------|
| Hardware | Ryzen 7 8845HS + RTX 4060 Laptop |
| games/hr | ~345 |
| sims/sec | ~3K |
| GPU util | 83–95% |
| VRAM | 2.8 / 8.6 GB |
| RAM (RSS) | ~14 GB |
| Buffer | 250,000 / 250,000 (full) |
| Batch fill | 91% |

---

## Upcoming changes (reason for archiving)

Based on KrakenBot analysis (`docs/analysis/krakenbot_deep_dive.md`):

1. **Remove Q13 chain INPUT planes** — 24ch → 18ch input. Keep chain as aux output target.
   - Experiment C ablation confirmed these hurt (draw rate +35pp); archiving pre-change.
2. **BatchNorm → GroupNorm** — batch-size-independent normalisation for leaf eval
3. **Selective policy loss** — only apply on full-search positions (needs full-search flag in buffer)
4. **Graduation gate** — new model must beat anchor at ≥76% to replace it
5. Eventually: bilinear pair policy head (N×N joint distribution)

---

## Files in this archive

```
checkpoints/
  checkpoint_00005000.pt   — 50MB, step 5k milestone
  checkpoint_00010000.pt   — 50MB, step 10k milestone
  checkpoint_00015000.pt   — 50MB, step 15k milestone
  best_model.pt            — 17MB, best model at time of archive (inference head only)
  inference_only.pt        — 17MB, inference-only weights (no optimiser state)

configs/
  model.yaml               — network architecture
  training.yaml            — optimiser, schedule, buffer mixing
  selfplay.yaml            — MCTS, workers, playout cap
  monitoring.yaml          — dashboard config
  eval.yaml                — eval pipeline
  variants/
    gumbel_targets.yaml    — ACTIVE variant for this run

logs/
  train_run1_d687569c.jsonl              — structured training log, steps ~0–9,300
  train_run2_10cc8d56.jsonl             — structured training log, steps ~13,600–15,272
  eval_step5000.log                     — eval results vs Random/SealBot/best at step 5k
  smoke_v3b_session_5k_predecessor.json — full monitoring data from the 5k predecessor run

eval/
  results.db               — SQLite Bradley-Terry eval results DB

probes/
  probe_ckpt5000_predecessor.md   — C1/C2/C3 @ step 5k (predecessor run, FAIL)
  probe_ckpt5500.md               — C1/C2/C3 @ step 5.5k (FAIL)
  probe_ckpt6000.md               — C1/C2/C3 @ step 6k (FAIL)
  probe_ckpt6000b.md              — C1/C2/C3 @ step 6k repeat (FAIL)
  probe_ckpt9000.md               — C1/C2/C3 @ step 9k (FAIL)
  probe_ckpt13500.md              — C1/C2/C3 @ step 13.5k (FAIL)
```
