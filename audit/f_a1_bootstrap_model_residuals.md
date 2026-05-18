# F-A1 — bootstrap_model.pt Residuals Classification

**Branch.** phase4.5/s179c_fa1_prep
**Master at branch creation.** f34295a
**IMPL-A commit (canonical handled).** c1173a8
**Date.** 2026-05-18

## Summary table

| # | path:line | classification | fix recommendation |
|---|-----------|----------------|--------------------|
| 1 | `hexo_rl/bootstrap/pretrain_trainer.py:302` | SCRIPT (write path, no foot-gun) | LEAVE — default OUTPUT dest for new pretrain; writing `bootstrap_model.pt` is intended. Caller can override via `--inference-out`. |
| 2 | `scripts/smoke_selfplay_gumbel.py:37` | SCRIPT (recurring, broken default) | FIX — update default `--checkpoint` from `bootstrap_model.pt` → `bootstrap_model_v6.pt` (canonical v6 anchor). |
| 3 | `scripts/tournament_validate.py:598` | SCRIPT (recurring, broken default) | FIX — update default `--our_model_ckpt` from `bootstrap_model.pt` → `bootstrap_model_v6.pt`. |
| 4 | `Makefile:313,324` | SCRIPT / kill-criterion tooling (broken) | FIX — replace hardcoded `checkpoints/bootstrap_model.pt` with `$(BOOTSTRAP)` in `probe.bootstrap` and `probe.latest` targets. |

## Per-reference detail

### 1. `hexo_rl/bootstrap/pretrain_trainer.py:302` — SCRIPT (LEAVE)

```python
# line 292 (docstring)
# Also writes a weights-only file (default: checkpoints/bootstrap_model.pt)
# for the eval pipeline. Pass ``inf_out`` to override that path — used by
# --resume runs that should not clobber the canonical bootstrap model
# until eval confirms uplift.
...
# line 302 (actual default)
inf_path = inf_out if inf_out is not None else Path("checkpoints") / "bootstrap_model.pt"
save_inference_weights(self.model, inf_path)
```

**Why this classification.** This is an OUTPUT path, not a read path. Running
`make pretrain` (or `python -m hexo_rl.bootstrap.pretrain`) without
`--inference-out` writes a new file to `checkpoints/bootstrap_model.pt`. That
new file would be freshly-trained weights — the expected output of a pretrain
run. The F-A1 entry itself notes "no foot-gun". The file `bootstrap_model.pt`
is absent from checkpoints/ (quarantined by IMPL-A to
`archive_quarantine/bootstrap_model_random_init_v6w25.pt`), so the next pretrain
run would simply create a new trained file at that path — which is harmless and
intended. Changing the default output name would require updating
`reports/bootstrap_model_pt_provenance.md` §5 recommendation and
`pretrain_cli.py` help text and `pretrain_trainer.py` docstring — no real
benefit over the `--inference-out` override that already exists.

**Fix recommendation.** LEAVE.

---

### 2. `scripts/smoke_selfplay_gumbel.py:37` — SCRIPT (FIX)

```python
# line 7 (docstring example — also updated)
#     .venv/bin/python scripts/smoke_selfplay_gumbel.py \
#         --variant gumbel_full --checkpoint checkpoints/bootstrap_model.pt \
...
# line 37 (argument default)
p.add_argument("--checkpoint", default="checkpoints/bootstrap_model.pt")
```

**Why this classification.** Recurring utility script for read-only selfplay
smoke. Running without `--checkpoint` override loads `checkpoints/bootstrap_model.pt`
which no longer exists (quarantined by c1173a8). Runtime error: FileNotFoundError
or PyTorch load failure. Not a one-off migration script — used regularly during
development smokes.

**Fix recommendation.** Update default to `checkpoints/bootstrap_model_v6.pt`
(canonical v6 anchor, matches `BOOTSTRAP ?=` in Makefile). Applied in this commit.

---

### 3. `scripts/tournament_validate.py:598` — SCRIPT (FIX)

```python
# lines 595-599
parser.add_argument(
    "--our_model_ckpt",
    type=str,
    default="checkpoints/bootstrap_model.pt",
    help="Checkpoint path for our_v6_* bots.",
)
```

**Why this classification.** Recurring tournament validation script, used for
H2H evaluation and tourney runs. Default `--our_model_ckpt` loads a file that no
longer exists. Running without `--our_model_ckpt` override yields runtime failure.
Production utility (not one-off migration).

**Fix recommendation.** Update default to `checkpoints/bootstrap_model_v6.pt`.
Applied in this commit.

---

### 4. `Makefile:313,324` — SCRIPT / kill-criterion tooling (FIX)

```make
# probe.bootstrap target (line 310-316)
probe.bootstrap: ## Probe bootstrap_model.pt; save fixtures/threat_probe_baseline.json + report
    $(PY) scripts/probe_threat_logits.py \
        --checkpoint checkpoints/bootstrap_model.pt \   # ← hardcoded absent path

# probe.latest target (line 318-325)
probe.latest: ## Threat-logit probe against latest checkpoint; PASS/FAIL step-5k kill criterion
    $(PY) scripts/probe_threat_logits.py \
        --checkpoint "$(CHECKPOINT_LATEST)" \
        --baseline-checkpoint checkpoints/bootstrap_model.pt \   # ← hardcoded absent path
```

**Why this classification.** `probe.bootstrap` and `probe.latest` are kill-criterion
tooling. `probe.latest` runs at every 5K checkpoint and its baseline comparison to
`bootstrap_model.pt` gates promotion decisions. Both targets hardcode the absent
path directly rather than using the `$(BOOTSTRAP)` Makefile knob that IMPL-A
already corrected for `make train` / `make smoke` / `make eval`. Invoking either
target would fail immediately with a file-not-found error. This is not a one-off
script — it is the primary threat-probe kill criterion workflow tool.

**Fix recommendation.** Replace hardcoded `checkpoints/bootstrap_model.pt` with
`$(BOOTSTRAP)` in both targets. `BOOTSTRAP` defaults to `checkpoints/bootstrap_model_v6.pt`
(set by IMPL-A at Makefile:11). This makes `probe.bootstrap` and `probe.latest`
consistent with all other Makefile targets that already use `$(BOOTSTRAP)`.
Applied in this commit.

---

## Decision

**surgical-fix-SCRIPT** — 3 of 4 residuals are broken-default load paths (absent
file since IMPL-A quarantine). Fix is low-risk text-only (default string updates +
Makefile variable substitution). Ref 1 (pretrain write path) is LEAVE per F-A1
note "no foot-gun".

No CODE-CRITICAL classification applies (no hot-path Python/Rust production code
that silently uses the bad anchor in the `make train` pipeline — those were all
closed by IMPL-A c1173a8).

## Commit (if any)

Applied with commit on `phase4.5/s179c_fa1_prep` — see git log.
Subject: `fix(anchors): drop residual bootstrap_model.pt references`
