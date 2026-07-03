# §D-LONGRUN-READY Phase A2 — Bootstrap Re-Pretrain Spec

**Goal:** Fresh v6_live2-family bootstrap pretrained on the 8300-game corpus.
Provides the warmstart for Phase B m-gate arms. Replaces the old
`bootstrap_model_v6_live2.pt` (pretrained on smaller human-only corpus).

**Prerequisite:** Phase A1 complete (`/workspace/hexo_rl` is the clean tree).

**Cost:** supervised pretrain on 473k positions, 30 epochs → ~20-40 min on vast 5080.
Cheap vs the Phase B training runs.

---

## Input corpus

| Field | Value |
|---|---|
| Path | `data/bootstrap_corpus_v6_live2.npz` |
| Games | 8 300 |
| Positions | 473 049 |
| P1 win rate | 50.2% (balanced) |
| Corpus sha (first 8) | `8f7115ab` |

**Verify before running:**
```bash
cd /workspace/hexo_rl
sha256sum data/bootstrap_corpus_v6_live2.npz | cut -c1-8
# Must print: 8f7115ab
```

If it does NOT match → stop. The corpus may have been corrupted or swapped.

---

## Sanity checks (run before pretrain)

```bash
cd /workspace/hexo_rl
source .venv/bin/activate

# 1. Corpus loads without error
python -c "
import numpy as np
d = np.load('data/bootstrap_corpus_v6_live2.npz', allow_pickle=False)
print('keys:', list(d.keys()))
print('positions:', d['obs'].shape[0])
# P1 win rate
wins = (d['outcome'] == 1).sum()
total = d['outcome'].shape[0]
print(f'P1 win rate: {wins/total:.3f}')
"
# Expected: positions ≈ 473049, P1 win rate ≈ 0.502

# 2. Filter check — no silent corpus drop (§114 lesson)
# If positions printed << 473049, the corpus filter is dropping games. Stop and
# investigate before running the pretrain.

# 3. Encoding resolves correctly
python -c "from hexo_rl.encoding import lookup; e = lookup('v6_live2'); print('in_channels:', e.in_channels)"
# Expected: in_channels: 4
```

---

## Pretrain command

```bash
cd /workspace/hexo_rl
source .venv/bin/activate

MALLOC_ARENA_MAX=2 python -m hexo_rl.bootstrap.pretrain \
  --encoding v6_live2 \
  --corpus-npz data/bootstrap_corpus_v6_live2.npz \
  --epochs 30 \
  --inference-out checkpoints/bootstrap_model_v6_live2_8300.pt
```

**Notes:**
- `--encoding v6_live2` (not v6_live2_ls): both have identical 4-plane wire shape;
  the pretrained weights transfer directly to v6_live2_ls training (same network).
- `--epochs 30`: matches the original bootstrap pretrain; 473k × 30 = ~14M gradient steps.
- `--inference-out`: write to a DISTINCT filename so the old bootstrap is not overwritten.
- Run under tmux in case the SSH session drops.

---

## Post-pretrain validation

```bash
# 1. File exists and is non-trivial
ls -lh checkpoints/bootstrap_model_v6_live2_8300.pt
# Expected: ~5-15 MB

# 2. SHA256 — record this; it goes into the phase_b_mgate_*.yaml configs
sha256sum checkpoints/bootstrap_model_v6_live2_8300.pt
# Record the full sha256 as BOOTSTRAP_8300_SHA256

# 3. Encoding auto-detection works (Phase B configs will use this checkpoint)
python -c "
from hexo_rl.model.utils import load_model_with_encoding
m, enc = load_model_with_encoding('checkpoints/bootstrap_model_v6_live2_8300.pt')
print('encoding:', enc.name, 'in_channels:', enc.in_channels)
"
# Expected: encoding: v6_live2, in_channels: 4

# 4. Value head calibration smoke (should not be flat/saturated)
python -c "
import torch
from hexo_rl.model.utils import load_model_with_encoding
m, enc = load_model_with_encoding('checkpoints/bootstrap_model_v6_live2_8300.pt')
m.eval()
x = torch.zeros(4, enc.in_channels, 19, 19)  # 4 dummy boards
with torch.no_grad():
    lp, v, vl = m(x)
print(f'value range: {v.min():.3f}..{v.max():.3f}')
print(f'policy entropy: {(-lp.exp() * lp).sum(-1).mean():.2f} nats')
"
# Expected: value range should span [-0.5, 0.5] (not flat ≈0); entropy 5-9 nats
# If value = 0.000 for ALL inputs → value head not initialised (bad)
```

---

## After pretrain: update Phase B configs

The three m-gate configs have `expected_anchor_sha256: "PLACEHOLDER_FILL_AFTER_A2"`.
Replace the placeholder with the actual sha256:

```bash
BOOTSTRAP_SHA=$(sha256sum checkpoints/bootstrap_model_v6_live2_8300.pt | awk '{print $1}')
echo "Bootstrap sha256: $BOOTSTRAP_SHA"

# Update all three phase_b_mgate configs:
for m in 8 16 32; do
  sed -i "s/PLACEHOLDER_FILL_AFTER_A2/${BOOTSTRAP_SHA}/" \
    configs/variants/phase_b_mgate_m${m}.yaml
done

# Verify
grep expected_anchor_sha256 configs/variants/phase_b_mgate_m8.yaml
```

---

## Deliverable

| Artifact | Path | Status |
|---|---|---|
| New bootstrap model | `checkpoints/bootstrap_model_v6_live2_8300.pt` | produced by this step |
| SHA256 | recorded in phase_b_mgate configs | pin after production |

**Gate:** SHA256 recorded, value-calibration smoke passed → A2 done.
→ Proceed to Phase B: `docs/handoffs/longrun_phase_b_mgate_runbook.md`
