<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §83 — quiescence_fire_count instrumentation (2026-04-12)

No instrumentation existed to measure whether the quiescence value override actually fires during self-play. Added `pub quiescence_fire_count: AtomicU64` on `MCTSTree` (reset in `new_game()`); `fetch_add(1, Relaxed)` at all 4 firing branches in `apply_quiescence`. `SelfPlayRunner` accumulates `mcts_quiescence_fires` per-search; emitted as `quiescence_fires_per_step` in the training event. `tests/test_gumbel_mcts.py::TestQuiescenceFireCount` validates getter + reset. Zero performance impact (relaxed atomic on post-search path). Commits `4124faa`, `ad79be7`.

---

