<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §151 — Numba @njit audit (audit-only) — 2026-05-04

Scope: 18 Python files (eval, training, bootstrap, env, utils, scripts, monitoring).
Audit-only: no code changes, no benchmarks, no installs.

**Verdict: NO-GO.** No qualifying hot-path Python loop; architectural rule
already honoured (hot loops in Rust, Python = numpy + torch glue).

Findings:
- Two lukewarm candidates — `batch_assembly._augment_recent_rows` policy/aux
  scatter (lines 234–238, per training step) and `pretrain.make_augmented_collate`
  policy scatter (lines 156–157, per pretrain batch) — are **Rust-port-instead**
  candidates, not Numba candidates. Both fold naturally into extending
  `engine.apply_symmetries_batch` to also scatter policy / ownership /
  winning_line per row.
- All other Python loops are cold-path (eval ~hourly, corpus build one-shot,
  analysis one-shot, validate once-per-pretrain) or torch / I/O bound.
- §92 C3 settled the prior strongest candidate (`_compute_chain_planes`) in
  favour of numpy-vectorized — out-of-scope for this audit.

Rationale against Numba adoption: third-toolchain cost (LLVM, CI surface,
wheel-build complexity, fourth FFI boundary) on top of Cargo + PyO3 + maturin +
torch.compile (Triton). No measured bottleneck justifies it.

Follow-up (deferred, predicated on bench delta):
  Extend `engine/src/replay_buffer/sample.rs::apply_symmetries_batch` to also
  scatter policy (362), ownership (361 u8), winning_line (361 u8) per row.
  Removes the two Python scatter loops in one go. Trigger: only if a
  `make bench` delta or training-step profiler surfaces the scatter as
  measurable. Currently unmeasured. Tracked at `/tmp/refactor_todos.md`.

Report: `/tmp/numba_audit_report.md`

---

