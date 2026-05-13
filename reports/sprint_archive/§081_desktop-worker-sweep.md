<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §81 — Desktop Worker-Count Sweep 2026-04-12

Laptop P3 winner (n_workers=14, burst=16) caused 97% worker-idle on Ryzen 7 3700x via GIL burst stalls (§77). D1–D5 sweep found ceiling at **D3: n_workers=10, wait_ms=5.0, burst=8 → ~334 gph**. D5 (12w) regressed to 307 gph (declining) as batch_fill rose 78→90% — inference server backs up, GIL/callback boundary saturated. §69's 400 gph gate unreachable on Zen2; laptop gate (659K pos/hr on Zen4) does not backport. `configs/variants/gumbel_targets_desktop.yaml` locks in D3. Sweep yamls deleted. Sustained run resumes from `ckpt_30851` with 250K buffer (§79) at ~180K filled.

---

