<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §99 — BatchNorm → GroupNorm migration (2026-04-16)

**Motivation:** MCTS leaf eval runs at batch=1. BatchNorm in eval mode uses
running statistics accumulated during training — these drift from the live
distribution as the model updates during self-play, introducing a
train/inference gap. GroupNorm computes per-sample statistics from fixed
channel groups, so behaviour is identical at batch=1 and batch=256.
KrakenBot uses GroupNorm(8, 128) throughout.

**Changes (`feat/groupnorm`):**

- `hexo_rl/model/network.py`:
  - `ResidualBlock.bn1/bn2` → `gn1/gn2` (`GroupNorm(_GN_GROUPS, filters)`)
  - `Trunk.input_bn` → `input_gn` (`GroupNorm(_GN_GROUPS, filters)`)
  - `policy_bn` and `opp_reply_bn` removed (2 output channels; GN(8,2) fails,
    normalization has negligible effect at 2 channels before flatten→linear)
  - `_GN_GROUPS = 8` module constant; `assert filters % _GN_GROUPS == 0` guard
    in `ResidualBlock.__init__`

- `hexo_rl/training/trainer.py`:
  - Removed BN running-stats reset from the NaN guard (GN has no running stats)

- `hexo_rl/training/checkpoints.py`:
  - `normalize_model_state_dict_keys` now raises `RuntimeError` on any
    checkpoint containing pre-GN key patterns (`.input_bn.`, `.bn1.`, etc.)
    to prevent silent trunk corruption via `strict=False` loading

**Checkpoint compatibility:** BROKEN. All pre-§99 checkpoints (including
smoke_v3b) contain BatchNorm keys and will be rejected at load time with a
clear error. Retrain from scratch.

**Benchmark:** Run `make bench` after this change. GN pool size differs from BN;
verify NN inference (batch=64) and latency (batch=1) targets still pass.

Note: benchmark methodology changed (§98 action items resolved) — runtime is
now 2 min with 90 s warmup, making results more representative of real
throughput. Prior baselines (1 min / shorter warmup) are not directly
comparable. Fresh `make bench` on this branch establishes the new GN baseline.

