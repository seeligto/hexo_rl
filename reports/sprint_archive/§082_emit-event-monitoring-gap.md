<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §82 — emit_event monitoring gap: ownership_loss + threat_loss (2026-04-12)

Both losses written to structlog JSONL since §58 but absent from `emit_event()` in `scripts/train.py` → invisible on dashboards. Fix: added `"loss_ownership"` and `"loss_threat"` (default 0.0) to the `train_step` event. Commit `d6a293e`.

---

