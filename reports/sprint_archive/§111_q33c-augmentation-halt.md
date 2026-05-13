<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §111 — Q33-C augmentation discriminator — 2026-04-21 (HALT)

Follow-up to §110. Q33-B left two candidate explanations for the
`pe_self ≈ 5.36` fixed point: (E1) healthy self-play distribution shift —
stronger model plays harder positions → `pe_self` on those is naturally
high, or (E2) augmentation blur — 12-fold symmetry mis-rotates policy
targets vs inputs, pinning batch `pe_self` near uniform. Plan: mirror the
Q33-B `gumbel_targets` smoke with augmentation disabled, compare `pe_A`
(with aug) vs `pe_B` (no aug), apply |pe_A − pe_B| thresholds.

**Outcome: HALT.** The augmentation toggle is Python-API-only. Audit
confirms:

- No `augment` / `apply_sym` / `symmetry` config key in `configs/`.
- `engine/src/replay_buffer/mod.rs:192-207` exposes `sample_batch`
  with `augment: bool` as a mandatory positional PyO3 argument.
- `hexo_rl/training/trainer.py:247` default arg `augment: bool = True`;
  production `loop.py:424` calls `trainer.train_step(buffer, recent_buffer=…)`
  which inherits the default.
- `hexo_rl/training/batch_assembly.py` hard-codes `True` at 5 sites
  (lines 232, 265, 271, 323, 333) — not driven by any flag.

Per the task prompt's explicit branch for this case: "If it's only
reachable from Python API (not config), document and halt — canonical-only
split instrumentation would be the next step but that requires code
changes out of scope here." No smokes were launched.

Report: `reports/q33c_augmentation_discriminator_2026-04-21.md`. The
report documents the audit, argues against a /tmp monkey-patch workaround
on scope grounds, and specifies a minimal "plumb `training.augment:
bool` through `loop.py` + `trainer.train_step` + `assemble_mixed_batch`"
follow-up task that would unlock the discriminator within a small-code
scope.

Secondary findings surfaced during the audit, worth keeping on the radar:

- **Static-audit candidate.** `engine/src/replay_buffer/sym_tables.rs:333`
  already tests coordinate consistency for every symmetry + every
  source axis. A parity test targeting the *policy scatter* (delta
  target at cell (0,0) rotated under sym k) is a cheap verification
  that would falsify E2 without running a smoke. Not done in this task
  (out of scope), but queued as a zero-runtime check for the next
  Q37 owner.
- **Log-analysis candidate.** Q33 `gumbel_targets` 20K drift
  (`pe_self: 4.62 → 5.54`) vs mean-game-length-in-window correlation
  on the existing `runs/c51d245de55c4a4bb39ac418397669bd/` logs.
  Non-zero correlation weakens E2 and strengthens E1; zero correlation
  is the opposite. Zero-runtime, pure log analysis.

**Effect on Phase 4.5 gating:** unchanged from §110. Q37 remains HIGH /
blocking. Phase 4.5 bootstrap-strengthening work is not justified on
the premise of moving `pe_self` off ~5.4; it is justified on independent
grounds (value quality, opening coverage). The `pe_self` interpretation
remains to be discriminated before Phase 4.5 commits serious GPU-days
to a bootstrap rebuild.

**Q33 / Q37 updates** (`docs/06_OPEN_QUESTIONS.md`):

- Q33 unchanged — WATCH, re-framed post-Q33-B still accurate.
- Q37 gains a "Q33-C HALT" note pointing at the report and the
  minimal-code-change follow-up scope; priority stays HIGH / blocking.

### Commits

- `docs(sprint): §111 Q33-C augmentation discriminator (halt)`
- `docs(q33-c): halt report + Q37 update (no verdict, toggle gap documented)`

---

