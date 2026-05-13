<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §129 — Disk-budget guard + checkpoint/game-record pruning — 2026-04-28

Added lightweight disk-space monitoring to prevent silent run failure when the vast.ai NVMe fills. `DiskGuard` background thread polls `shutil.disk_usage` every 60 s, emits `disk_free` events to the monitoring fan-out, warns at < 10 GB, and sends SIGTERM at < 5 GB (triggering the existing graceful shutdown path — buffer is saved before exit). Checkpoint pruning gained `keep_all` (disables pruning for debug runs) and `anchor_every_steps=5000` (permanent anchors at every 5k-step boundary, complementing the existing `preserve_eval_checkpoints` logic). Game records (daily JSONL, ~400 bytes/record) auto-archive to `tar.gz` when total exceeds 10k records — effectively free at ~4 MB uncompressed. Structlog switched to `RotatingFileHandler` with gzip rotation at 500 MB / file. Footprint headroom for a 500k-step run: ~9 GB delta on a 100 GB box (5.1 GB eval anchors + 2.5 GB replay buffer + 0.6 GB logs), leaving 85+ GB free — well above the 10 GB warn threshold. 7 new tests pass (3 disk guard, 4 checkpoint prune); 887 existing tests unaffected.

---

