<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §68 — Eval DB run_id bug fix + broken-run cleanup

`EvalPipeline` stored `self.run_id` but never passed it to 5 `db.get_or_create_player()` / `db.insert_match()` call sites in `run_evaluation()` — every run's eval collapsed onto `run_id=""` in the ratings DB. Fix: thread `run_id=self.run_id` through all 5 calls. Reference opponents (SealBot, random_bot) keep `run_id=""` as shared anchors; pairwise/history queries already match run-specific players plus empty-`run_id` references.

**Broken-run cleanup (§67 scheduler poison):** archived to `archive/checkpoints.broken-202604/` (10 checkpoints, best_model.pt, replay_buffer.bin, log) and `archive/eval.broken-202604/results.db`. Kept: `bootstrap_model.pt`, `checkpoints/pretrain/`, `runs/*/games/`, logs, corpus.

---

