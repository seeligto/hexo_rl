<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §84 — Fix eval checkpoint retention (2026-04-13)

### Symptom (pre-existing, first noted in §71)

§71 footnote: "checkpoint_00010000.pt does not exist in current checkpoints/ (was never
saved)". Same symptom on the laptop gumbel_targets run at step ~19K: ckpt_5000, ckpt_10000,
ckpt_15000 absent from disk. Eval DB (results.db) has BT ratings for all three players but
weight files are gone — re-eval, sharpness sweeps, and post-hoc forensics impossible.

### Root cause

Three config values created a perfect eviction storm:

```
checkpoint_interval: 500    # save every 500 steps
max_checkpoints_kept: 10    # keep 10 most recent
eval_interval: 5000         # eval at 5k, 10k, 15k ...

5000 / 500 = 10 = max_checkpoints_kept
```

`prune_checkpoints()` (`checkpoints.py:53–79`) kept the N largest step numbers and deleted
everything else via `Path.unlink()` — no exemption for eval-step checkpoints. After exactly
10 more rolling saves, each eval checkpoint was evicted by the next eval.

`best_checkpoint` promotion overwrites `best_model.pt` (weights-only, `torch.save()`). It
does not rename or copy the numbered checkpoint, so promotion offered no protection.

Eval DB stores player names (`"checkpoint_5000"`), not file paths — DB records intact,
only the weight files were gone.

### Fix

Two-tier retention: eval steps permanent, rolling window unchanged at 10.

- `checkpoints.py`: `prune_checkpoints()` gains `preserve_predicate: Optional[Callable[[int], bool]]`.
  Steps matching the predicate are excluded from the rotation pool entirely.
- `trainer.py` `save_checkpoint()`: builds predicate `lambda s: s > 0 and s % eval_interval == 0`
  from config each call (not a frozen set — tracks `eval_interval` if it changes mid-run).
  Guarded by `preserve_eval_checkpoints` config key (default `True`).
- `configs/training.yaml`: `preserve_eval_checkpoints: true` added.
- `tests/test_trainer.py`: `test_eval_checkpoints_not_pruned` — 30 fake files, eval_interval=5000,
  max_kept=10; asserts all 3 eval checkpoints present + correct 10 rolling present + older rolling absent.

### Recovery for live laptop run

ckpt_5000/10000/15000 unrecoverable — accept loss. Restart sequence: let run hit step 20000
naturally, eval completes, then graceful stop before step 25000 (old rotation window). On
restart, new code loads config, `prune_checkpoints` sees ckpt_20000 on disk, predicate
exempts it. ckpt_20000 becomes the first permanent forensic anchor.

**Commit:** `fix(training): preserve eval checkpoints`

