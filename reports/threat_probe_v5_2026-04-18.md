# Threat-probe baseline regeneration v4 → v5 (2026-04-17)

## Motivation

`fixtures/threat_probe_baseline.json` v4 (generated 2026-04-16) was anchored
to a `bootstrap_model.pt` that was subsequently replaced. After §99
(BatchNorm → GroupNorm) and the follow-up bootstrap refresh, the live
bootstrap produces threat-head outputs that bear little resemblance to the
recorded v4 numbers. `make probe.latest` therefore compared every trained
checkpoint against a stale reference. Goal: re-anchor the baseline against
the live 18-plane post-§97/§99 bootstrap, schema v4 → v5.

## Phase 1 — NPZ compatibility (Option A: slice)

`fixtures/threat_probe_positions.npz` was shape `(20, 24, 19, 19)` — built
pre-§97. The current probe script asserts `states.shape[1:] == (18, 19, 19)`
so would not load it at all.

Per `GameState.to_tensor()` post-§97 (hexo_rl/env/game_state.py:186-242) and
the §92→§97 sprint history, the plane layout is:

| planes | content |
|---|---|
| 0     | current player stones @ t |
| 1-7   | current player stones @ t-1 … t-7 |
| 8     | opponent stones @ t |
| 9-15  | opponent stones @ t-1 … t-7 |
| 16    | `moves_remaining == 2` flag (broadcast) |
| 17    | `ply % 2` parity (broadcast) |
| 18-23 | (pre-§97) chain-length planes — removed post-§97 |

Empirical validation on the existing NPZ:

| channels | min | max | unique | interpretation |
|---|---|---|---|---|
| 0-15  | 0 | 1 | {0, 1}   | stone presence (binary) |
| 16-17 | 1 | 1 | {1}      | scalar planes (all fixture positions are P1-to-move @ ply=7) |
| 18-23 | 0 | 0.67 | fractional (1/6 scale) | chain-length (deprecated input) |

Planes 0-17 are bit-identical to the current `to_tensor()` output; planes
18-23 are the deprecated chain planes. **Option A selected** — slice in
place to `states[:, :18]`, preserving every probe position, every ext/ctrl
index, every phase label. No regeneration from game records.

SHA-256 changes (expected):

```
before: f02c529f20586ec4b0131371fdfe69389b4a0a4d962fa7550ccd2ba05af58cfa
after:  a2b24b3af80928857340016d3e06d71b13632bea629b7b21991f8b7b589d9921
```

## Phase 2 — Baseline regeneration

Command:

```
.venv/bin/python scripts/probe_threat_logits.py \
  --checkpoint checkpoints/bootstrap_model.pt \
  --write-baseline \
  --output reports/probes/bootstrap_v5_regen.md
```

`BASELINE_SCHEMA_VERSION` bumped `4 → 5` in
`scripts/probe_threat_logits.py:78`.

Observed v5 values:

| metric | value |
|---|---|
| `ext_logit_mean`  | +0.0800 |
| `ctrl_logit_mean` | +0.0277 |
| `contrast_mean`   | +0.0523 |
| `ext_in_top5_pct` | 20% |
| `ext_in_top10_pct`| 20% |

## Phase 3 — Sanity check vs §91 C1

C1 criterion: `contrast_mean ≥ max(0.38, 0.8 × bootstrap_contrast)`.

With `bootstrap_contrast = +0.052`, the multiplicative term is `0.042`, so
the absolute floor `0.38` binds. **C1 threshold unchanged in practice**
(previously bound as well, since v4's `−0.937` made `0.8×` negative).

### Before / after

| metric | v4 (2026-04-16 bootstrap) | v5 (2026-04-17 bootstrap) | Δ |
|---|---|---|---|
| `ext_logit_mean`  | +0.2171 | +0.0800 | −0.137 |
| `ctrl_logit_mean` | +1.1538 | +0.0277 | −1.126 |
| `contrast_mean`   | −0.9366 | +0.0523 | +0.989 |
| `ext_in_top5_pct` | 20% | 20% | 0 |
| `ext_in_top10_pct`| 20% | 20% | 0 |

### "Markedly different" flag (> ±0.3)

The +0.989 contrast shift triggers the task's flag. **Root cause is
bootstrap file substitution, not probe-position instability.**

1. `checkpoints/bootstrap_model.pt` mtime is 2026-04-17 10:43, newer than
   the v4 commit (0da0b7c, 2026-04-16 19:40). Bootstrap was regenerated
   between v4 and now.
2. `ctrl_logit_mean` fell by ~1.1 nats in isolation while `ext_logit_mean`
   moved only −0.14. If chain planes had been confounding the threat head
   at the control cell, we would expect a coupled shift across ext/ctrl
   when we drop them; instead we see an asymmetric collapse consistent
   with retrained weights.
3. Policy top-K membership is identical (20%/20%) — the fixture's
   geometry is stable across the migration; only the model that scores it
   changed.

Takeaway: the probe benchmark set is sound; baselines must be
regenerated whenever `bootstrap_model.pt` is physically replaced.

## Verification

1. Schema bump confirmed:
   ```
   python -c "import json; d=json.load(open('fixtures/threat_probe_baseline.json')); assert d['version']==5, d['version']"
   ```
   → exit 0.

2. Round-trip self-test: `make probe.bootstrap` → exit 0; baseline written
   bit-identical to the regeneration above.

3. `make probe.latest` full end-to-end exercise **not performable** in this
   session: no post-§99 (GroupNorm) trained checkpoint is available.
   `checkpoints/saved/checkpoint_*.pt` are pre-§99 BN and refuse to load
   via `normalize_model_state_dict_keys` (§99 safety rail). Probe will be
   exercised properly at the next sustained training run's step-5k gate.

## Files changed

| path | change |
|---|---|
| `fixtures/threat_probe_positions.npz` | sliced 24-plane → 18-plane (planes 18-23 dropped) |
| `fixtures/threat_probe_baseline.json` | regenerated v4 → v5 |
| `scripts/probe_threat_logits.py` | `BASELINE_SCHEMA_VERSION: 4 → 5` |
| `docs/07_PHASE4_SPRINT_LOG.md` | §100.d entry |

## What was NOT changed

- C1/C2/C3 thresholds (`THRESH_CONTRAST_FLOOR = 0.38`, `25.0`, `40.0`)
- C4 drift warning threshold (`5.0`)
- Per-position content of the NPZ (side_to_move, ext_cell_idx,
  control_cell_idx, game_phase)
- Probe position count (`n=20`)
- Fixture generation script (`scripts/generate_threat_probe_fixtures.py`)
