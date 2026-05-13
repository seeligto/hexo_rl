<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §79 — Initial buffer increased 100K → 250K (2026-04-12)

§40b had reduced 250K → 100K as a draw-collapse stability measure; collapse resolved at §40, CLAUDE.md still said 250K — config was the stale artifact. At 100K with ~48% self-play the model sees ~600 games of context, too thin to generalise beyond colony patterns.

**Schedule:** `[{step:0, 250K}, {step:300K, 500K}, {step:1M, 1M}]`. Growth tiers shift right vs §40b. Steps 300K and 1M exceed `total_steps: 200_000` — apply only on extended runs.

**Memory budget** (14,458 B/entry × Rust + Python-mirror): 250K ≈ 5.05 GB, 500K ≈ 10.1 GB, 1M ≈ 20.2 GB. 32 GB RAM → 250K leaves ~19 GB headroom; +2.98 GB vs 100K.

**Resume safety:** `load_from_path` reads `min(saved_size, self.capacity)` into pre-allocated capacity — no resize. Old 100K checkpoints load cleanly into 250K buffer.

---

