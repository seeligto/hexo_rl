<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §75 — Fast game disable for gumbel_targets (2026-04-10)

Draw-rate investigation (`reports/draw_rate_investigation_2026-04-10/`): 100% of draws are 150-ply timeouts. Low-sim games (fast_prob=0.25, 50 sims, τ=1.0, PUCT) hit 94.4% draw vs 3.7% on standard games — colony-extension behaviour in the viewer. Fix: `fast_prob: 0.0` in `configs/variants/gumbel_targets.yaml`. `gumbel_full.yaml` unchanged (Gumbel SH effective in low-sim regime, §71). Resumed from ckpt_25008.

---

