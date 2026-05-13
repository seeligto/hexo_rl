<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §95 — Experiment C: chain-plane input ablation (2026-04-16)

### Motivation

Experiment A (§94, aux_chain_weight=0) did NOT reduce draw rate.
Draw rate 47.7% at step 10312 vs smoke_v3b 44.7% at 5003 — within noise,
marginally worse. Chain aux confirmed NOT the primary driver.

**Remaining hypothesis:** The 6 chain-length input planes (18-23) themselves
prime the policy toward colony extension. The residual tower learned to use
chain-plane values to route gradient toward extending existing chains, independent
of the aux head. Removing the loss signal (Exp A) did not break that routing —
the input features remained, so the learned "extend chain = high value" shortcut
still activates in self-play.

### Chain planes: stored in buffer, NOT recomputed at sample time

Audit finding (confirmed reading Rust + Python): chain planes are computed at
**encode time** (`encode_state_to_buffer` in `engine/src/board/state.rs`, called
during self-play when games are pushed to the replay buffer) and stored as part of
the 24-plane state tensor. They are NOT recomputed at sample time. The Rust symmetry
augmentation path (`apply_symmetry_24plane` in `engine/src/replay_buffer/sym_tables.rs`)
applies coordinate permutation + axis-plane remap to the stored chain planes, but does
NOT recompute them from stones.

Consequence: zeroing at the point of use (trainer + inference server) is sufficient —
zeroed planes (all zeros) are invariant under any symmetry transform, so augmentation
cannot reintroduce signal.

### Design

Zero planes 18-23 AFTER decode from buffer / after H2D transfer — do NOT remove planes
from the architecture. `in_channels=24` stays. The 6 zeroed planes provide zero gradient
to the network via input; the conv weights for those input channels receive no gradient
from states but the trunk is otherwise unchanged.

**`zero_chain_planes: bool`** config flag (default false in `configs/training.yaml`).
Set to `true` in `configs/variants/gumbel_targets.yaml` for Experiment C.

### Wiring (3 locations)

1. **`hexo_rl/training/trainer.py`** — `_train_on_batch()`: zero `states_t[:, 18:24]`
   after H2D transfer, before model forward. Covers training path.

2. **`hexo_rl/selfplay/inference_server.py`** — `__init__()`: read and store
   `_zero_chain_planes`; `run()`: zero `tensor[:, 18:24]` after Rust→Python batch
   extraction, before model forward. Covers self-play inference path.

3. **`scripts/probe_threat_logits.py`** — `main()`: `--zero-chain-planes` CLI flag;
   zero `positions["states"][:, 18:24]` before `probe_positions()`. Ensures probe
   uses same inputs as trained model.

**Replay buffer augmentation path:** no change required. Zeroed planes are invariant
under the 12-fold hex symmetry transform (zero → zero). Stored buffer values remain
non-zero but are always masked at point of use.

### Config

```yaml
# configs/training.yaml (base):
zero_chain_planes: false   # ablation default; set true in variant for Exp C

# configs/variants/gumbel_targets.yaml (Experiment C only):
training:
  zero_chain_planes: true
# aux_chain_weight: 0.0 already in training.yaml from Exp A
```

### Success criteria at step 5000

| Outcome | Draw rate | Interpretation |
|---------|-----------|----------------|
| PRIMARY PASS | < 35% | Chain INPUT planes confirmed as cause |
| PARTIAL | 35–45% | Partial contribution; combined fix needed |
| NULL | > 45% | Chain planes NOT the cause; buffer dilution next |

### Test

`tests/test_zero_chain_planes.py` — 5 tests:
- Trainer zeroes planes 18-23 when flag true (captures model input)
- Trainer preserves planes 18-23 when flag false
- Only planes 18-23 are zeroed (planes 0-17 unchanged)
- InferenceServer zeroes planes 18-23 when flag true
- InferenceServer preserves planes 18-23 when flag false

All 5 pass; full test suite (755 tests) passes.

### Checkpoints

Experiment A final checkpoint backed up to `checkpoints/exp_a_backup/checkpoint_00010000.pt`.
Full experiment A data at `checkpoints/chain_planes_no_chain_weight/`.

Fresh run from `checkpoints/bootstrap_model.pt`. Same monitoring script as Exp A
(`scripts/monitor_experiment_a.sh`), pointed at new JSONL.

### Commits

- `feat(env): zero_chain_planes config flag for input ablation experiment (Exp C §95)`
- `docs(sprint): §95 Experiment C — chain plane input ablation`

---

