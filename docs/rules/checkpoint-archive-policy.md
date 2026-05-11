# Checkpoint Archive Policy

Deliberately-unstamped checkpoint directories are allowlisted in both the
T10 migration script and the audit CLI.  They are **not** stamped and do
**not** count as audit failures.

## Rationale

These directories contain obsolete or broken training artifacts that have
no active consumers.  Stamping them would add noise without value.

## Allowlisted prefixes

| Prefix | Reason |
|---|---|
| `checkpoints/broken/` | Corrupt / un-loadable checkpoints |
| `checkpoints/collapsed_*` | Training runs that collapsed (archived) |
| `checkpoints/olod_20k/` | Old 20k-step legacy sweep artifacts |
| `checkpoints/chain_planes*` | Abandoned chain-planes experiment |
| `checkpoints/v9_s3/` | Superseded v9_s3 smoke run |
| `checkpoints/pretrain/` | Obsolete pretrain sweep artifacts |
| `checkpoints/w4c_smoke_v7_laptop_preflight/` | Pre-§150 laptop preflight (dead) |

## Adding a new dead dir

1. Add the prefix to `_DEAD_DIR_PATTERNS` in
   `scripts/migrations/2026_05_09_stamp_artifact_metadata.py`.
2. Add the prefix to `_DEAD_CKPT_PREFIXES` in
   `hexo_rl/encoding/audit.py`.
3. Update the table above.

Both sources must stay in sync.
