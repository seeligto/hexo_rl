# §S181-AUDIT Wave 1 Track B — launch + analysis spec

## Goal

Pin the live training-loop gradient flow + buffer composition + trunk
feature drift across the early-collapse window where 86% of V_spread
loss happens (steps 0-2k under §S180b recipe).

## Branch + commits

`phase4.5/s181_track_b_instrumented` off PR-C HEAD (`7cd0dc0`).

Code landed (this branch):

- `hexo_rl/training/track_b_attribution.py` — per-source gradient L2
  attribution helper (B1).
- `hexo_rl/training/track_b_buffer_snapshot.py` — position-class
  snapshot helper (B2).
- `scripts/structural_diagnosis/track_b/trunk_feature_drift.py` —
  post-run trunk drift analysis (B3).
- `hexo_rl/training/trainer.py` (edit) — wire B1 flag +
  attribution call before main backward.
- `hexo_rl/training/step_coordinator.py` (edit) — wire B2 snapshot
  at checkpoint-cadence buffer-save site.
- `configs/variants/v6_botmix_s181_track_b.yaml` — instrumented
  variant inheriting §S180b 3-knob recipe with B1+B2 enabled +
  checkpoint_interval=500.
- `hexo_rl/training/tests/test_track_b_attribution.py` — 8 unit
  tests.
- `hexo_rl/training/tests/test_track_b_buffer_snapshot.py` — 4 unit
  tests.

All tests PASS (43/43 including PR-C INV pins).

## Launch (operator-mediated)

Spec:

| field | value |
|---|---|
| host | vast.ai `REMOTE_HOST:REMOTE_PORT` (5080) |
| workdir | `$REPO_ROOT/` |
| anchor | `checkpoints/bootstrap_model_v6.pt` (SHA `7ab77d2c…`) |
| variant | `v6_botmix_s181_track_b` |
| iterations | 3000 |
| checkpoint_interval | 500 (→ ladder 500, 1000, …, 3000) |
| wall (est) | ~6 h (1.3× §S180b due to B1 attribution) |
| cost (est) | ~$1.50 |
| canary | dual-bank V_spread fires at every checkpoint (PR-C in scope) |

Operator pre-flight:

1. `tmux kill-session -t s181_b 2>/dev/null` (if reused)
2. Confirm `data/bootstrap_corpus_v6.npz` + `data/bot_corpus_s178_sealbot_vs_v6.npz` present on vast
3. `rm checkpoints/replay_buffer.bin` (clean start)
4. Verify branch `phase4.5/s181_track_b_instrumented` is checked out + rebuilt (`make build`)
5. `make test-py` green
6. Launch:
   ```bash
   tmux new -d -s s181_b 'python scripts/train.py \
     --checkpoint checkpoints/bootstrap_model_v6.pt \
     --variant v6_botmix_s181_track_b \
     --iterations 3000 2>&1 | tee logs/s181_b_$(date +%Y%m%d_%H%M).log'
   ```

Live monitoring:
- `tail -F logs/s181_b_*.log | grep -E 'per_source_grad_norm|buffer_position_class_snapshot|value_spread|sealbot|grad_norm|nan_or_inf'`
- Hard-abort if `grad_norm > 10` for 5 consecutive steps (already configured)
- Soft-abort signal if dual canary `both_pass: false` for 2 consecutive checkpoints (manual operator call — no auto-kill)

## Post-run analysis

### B1 — per-source gradient attribution

Source: `per_source_grad_norm` JSONL events. Per step:

```json
{
  "event": "per_source_grad_norm",
  "step": <int>,
  "n_pretrain": <int>, "n_recent": <int>, "n_uniform_self": <int>,
  "trunk_pretrain": <float L2>, "trunk_recent": <float>, "trunk_uniform_self": <float>,
  "value_pretrain": <float>, "value_recent": <float>, "value_uniform_self": <float>,
  "policy_pretrain": <float>, "policy_recent": <float>, "policy_uniform_self": <float>
}
```

Analysis (one Jupyter notebook OR a small script):

1. Compute per-source TOTAL pull = sum of L2 norms across (trunk, value, policy) per step.
2. Compute per-source SHARE = source_total / sum(all sources) per step.
3. Plot share trajectory across 3000 steps.
4. At step 500/1000/2000/3000 record per-source share + per-group share.

### B2 — buffer composition

Source: `buffer_position_class_snapshot` events at steps 500, 1000, …, 3000.

Plot colony_frac / extension_frac / neither_frac trajectory across the 6 ckpts.
Compare against anchor expectation: at step 0, buffer is empty + warmed
with selfplay; at step 2k+, selfplay positions accumulate. If colony_frac
crosses 50% by step 2k, feedback loop confirmed.

### B3 — trunk feature drift

Post-run command (operator-side after the run completes):

```bash
python -m scripts.structural_diagnosis.track_b.trunk_feature_drift \
  --ckpt-dir checkpoints/ \
  --include-anchor checkpoints/bootstrap_model_v6.pt \
  --output audit/structural/track_b/B3_trunk_drift.json
```

Reads each ckpt, forwards the alt-bank 40-position fixture through the
TRUNK output (pre-head), computes per-class centroid + intra-class
variance + inter-class distance + ratio. Outputs JSON ladder.

Flag V-B-D if `inter_centroid_dist` at step 1000 ≤ 50% of step-0 value.

## Aggregation — V-B verdict (LITERAL, L13 guard)

Decision tree (apply in order; first match wins):

1. If any source share ≥ 60% across steps 500-2000 (B1) → **V-B-A** (source-targeted lever).
2. Else if (B1) all three sources between 25-45% across steps 500-2000 AND colony-pushing → **V-B-B** (multi-source damping needed).
3. Else if (B2) colony_frac > 50% by step 2000 → **V-B-C** (feedback loop confirmed).
4. Else if (B3) inter_centroid_dist at step 1000 ≤ 50% of step-0 → **V-B-D** (trunk co-adaptation).
5. Else → **V-B-E** (no clean match; escalate, no real-run launch).

Verdicts may stack — record all that apply; primary = first match in the
tree. Output `audit/structural/track_b/B_aggregation.md` with:

- All four traces (B1 share-trajectory, B1 group-trajectory, B2 class-fractions, B3 centroid distances)
- Verdict + supporting numbers
- Mechanism narrative (which lever in REAL_RUN_RECIPE this points at)

## Stop conditions during the run

- Any standard §S180b hard-abort fires (already wired)
- Dual canary `both_pass: false` at 2 consecutive checkpoints — operator call
- B1 attribution returns NaN for >100 consecutive steps — implies graph corruption; restart

## Cross-references

- `docs/07_PHASE4_SPRINT_LOG.md` §S181-AUDIT — Track A verdicts + L47/L48/L49
- `configs/variants/v6_botmix_s180b_3knob_escalation.yaml` — base recipe
- `audit/structural/track_a/A6_aggregation.md` — A6 source decomposition
- `audit/structural/track_c/C_LITE_1_v7full_reference.md` — v7full anchor candidate
- `scripts/structural_diagnosis/track_a/position_classifier.py` — classifier used by B2
