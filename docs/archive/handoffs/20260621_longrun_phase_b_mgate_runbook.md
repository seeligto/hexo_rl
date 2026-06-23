# §D-LONGRUN-READY Phase B — m-gate Runbook

**Goal:** Discriminate whether m=8, m=16, or m=32 gives stronger per-step Gumbel training.
Provides the routing decision for Phase C (Gumbel@best-m vs PUCT-600).

**Prerequisite:** Phase A2 complete — `checkpoints/bootstrap_model_v6_live2_8300.pt` exists
and SHA256 is filled into the three configs below.

---

## Pre-flight (mandatory before ANY launch)

```bash
cd /workspace/hexo_rl
source .venv/bin/activate

# 1. Verify A2 bootstrap exists and SHA matches configs
ls -lh checkpoints/bootstrap_model_v6_live2_8300.pt

ACTUAL=$(sha256sum checkpoints/bootstrap_model_v6_live2_8300.pt | awk '{print $1}')
CONFIG=$(grep expected_anchor_sha256 configs/variants/phase_b_mgate_m8.yaml | awk '{print $2}' | tr -d '"')
[ "$ACTUAL" = "$CONFIG" ] && echo "SHA OK" || echo "SHA MISMATCH — do not launch"

# 2. Eval opponents import (the Arm-C lesson)
python -c "from hammerhead import Bot; print('hammerhead OK')"
python -c "from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot; print('KClusterMCTSBot OK')"
# If either fails → build hammerhead first (maturin develop --release)

# 3. 2-game smoke per arm (optional but recommended — verifies the full loop)
# Run with --iterations 2 to confirm launch + eval pipeline works end-to-end
# (skip if time-constrained; the random eval fires at first eval_interval anyway)
```

---

## Launch sequence (run each arm sequentially on vast)

Run one arm at a time to avoid GPU contention. Each arm is independent — the order
doesn't matter, but m=8 first is recommended (likely the best arm, and the one most
different from the banked m=32/Gumbel-opt).

### Arm m=8

```bash
cd /workspace/hexo_rl
source .venv/bin/activate

rm -f checkpoints/best_model.pt checkpoints/best_model.pt.bak
python scripts/train.py \
  --checkpoint checkpoints/bootstrap_model_v6_live2_8300.pt \
  --variant phase_b_mgate_m8 \
  --iterations 12000 \
  --checkpoint-dir checkpoints/phase_b_m8
```

Watch: `draw_rate` should stay ≪ 0.55; `grad_norm` ≪ 10.0.
Bank final checkpoint before starting next arm:
```bash
cp checkpoints/phase_b_m8/checkpoint_00012000.pt checkpoints/bank_m8_final.pt
```

### Arm m=16

```bash
rm -f checkpoints/best_model.pt checkpoints/best_model.pt.bak
python scripts/train.py \
  --checkpoint checkpoints/bootstrap_model_v6_live2_8300.pt \
  --variant phase_b_mgate_m16 \
  --iterations 12000 \
  --checkpoint-dir checkpoints/phase_b_m16
```

### Arm m=32

```bash
rm -f checkpoints/best_model.pt checkpoints/best_model.pt.bak
python scripts/train.py \
  --checkpoint checkpoints/bootstrap_model_v6_live2_8300.pt \
  --variant phase_b_mgate_m32 \
  --iterations 12000 \
  --checkpoint-dir checkpoints/phase_b_m32
```

---

## Reads (do on laptop after all three arms complete)

Pull the final checkpoints from vast:
```bash
# From laptop:
for m in 8 16 32; do
  rsync -avz -e "ssh -p 13053 -i ~/.ssh/vast_hexo" \
    "root@ssh6.vast.ai:/workspace/hexo_rl/checkpoints/phase_b_m${m}/" \
    "~/Work/Hexo/hexo_rl/checkpoints/phase_b_m${m}/"
done
```

### Read 1: strength head-to-head among m-arms

```bash
# 4-model round_robin: anchor + m8-12k + m16-12k + m32-12k
# (anchor = bootstrap_model_v6_live2_8300.pt — same warmstart for all arms)
python scripts/eval_round_robin.py \
  --models \
    checkpoints/bootstrap_model_v6_live2_8300.pt \
    checkpoints/phase_b_m8/checkpoint_00012000.pt \
    checkpoints/phase_b_m16/checkpoint_00012000.pt \
    checkpoints/phase_b_m32/checkpoint_00012000.pt \
  --names "anchor" "m8-12k" "m16-12k" "m32-12k" \
  --encoding v6_live2_ls \
  --n-games 100 \
  --opening-plies 0 \
  --temperature 0.0
# Distinct-game CI (post-980bc4d round_robin). Use deduped bootstrap CI.
```

### Read 2: each m-arm vs BANKED PUCT-15k (DO NOT re-run PUCT)

```bash
# Reuse the PUCT-15k checkpoint from reports/p3_rr_agg/ (or checkpoints/p3_puct/)
PUCT_15K=checkpoints/p3_puct/checkpoint_00015000.pt  # adjust to actual path

for m in 8 16 32; do
  python scripts/eval_round_robin.py \
    --models \
      checkpoints/phase_b_m${m}/checkpoint_00012000.pt \
      $PUCT_15K \
    --names "m${m}-12k" "puct-15k" \
    --encoding v6_live2_ls \
    --n-games 100 \
    --opening-plies 0 \
    --temperature 0.0 \
    --output-dir reports/phase_b_mgate/m${m}_vs_puct15k/
done
```

### Read 3: depth + value-regret per arm

```bash
# Pull training logs from vast and check avg_tree_depth and value_regret telemetry
# These are emitted by emit_value_pred_at_ply_cap and the Gumbel telemetry.
# Look for: m8 deeper (higher avg_depth) + lower value_regret
# If depth scales monotonically m8>m16>m32, the proxy relationship holds.
```

### Read 4: pos/hr per arm

```bash
# From the training logs: steps/hr should be ~equal across m-arms at fixed n=100.
# If m=8 is significantly slower → the fixed-n assumption breaks → flag before
# interpreting the strength read (throughput cost changes the comparison).
grep "steps_per_hr\|pos_per_hr\|throughput" checkpoints/phase_b_m8/*.log | tail -5
grep "steps_per_hr\|pos_per_hr\|throughput" checkpoints/phase_b_m16/*.log | tail -5
grep "steps_per_hr\|pos_per_hr\|throughput" checkpoints/phase_b_m32/*.log | tail -5
```

---

## Pre-registered routing decision

After the four reads:

| Verdict | Criterion | Phase C routing |
|---|---|---|
| **GUMBEL-EARNS-IT** | Best m-arm per-step Elo ≥ PUCT-15k at 12k steps, AND cheaper-per-strength | Gumbel@best-m long run |
| **GUMBEL-PARITY-ONLY** | Best m-arm weaker per-step vs PUCT-15k, edge ≈ cost-margin only | PUCT-600 long run |
| **GUMBEL-WORSE** | No m-arm improves on m=32's parity verdict | PUCT-600 long run |

**Note:** the bar is PUCT-15k (banked), not PUCT-50k. The question is whether m=8
flips the per-step weakness observed at m=32 in 15k vs 15k. If m8-12k ties or beats
PUCT-15k at matched steps (both 12k-15k), Gumbel earns the long run.

**Report the routing decision to the operator** before launching Phase C.

---

## Files produced by this step

| Artifact | Path |
|---|---|
| m=8 final checkpoint | `checkpoints/phase_b_m8/checkpoint_00012000.pt` |
| m=16 final checkpoint | `checkpoints/phase_b_m16/checkpoint_00012000.pt` |
| m=32 final checkpoint | `checkpoints/phase_b_m32/checkpoint_00012000.pt` |
| Read 2 reports | `reports/phase_b_mgate/m*_vs_puct15k/` |
| Routing verdict | operator decision, then Phase C |
