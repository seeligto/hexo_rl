<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §78 — /analyze Policy Viewer (2026-04-11)

Interactive debugging tool — inspect raw network priors on arbitrary positions (§70 mode collapse was invisible until 16k steps).

**Scope** (branch `feat/policy-viewer`, 4 commits):

1. Rust PyO3 — `forced_root_child` getter/setter, `get_root_children_info()`, `get_improved_policy()`, `get_top_visits` → 4-tuple (+q_value).
2. `hex_canvas.js` ES module extracted from `viewer.html` for reuse.
3. `/api/analyze` Blueprint — checkpoint LRU (max 3, mtime stale check), Python-driven MCTS (PUCT + Gumbel SH), `ThreadPoolExecutor(1)`. `model_loader.py` loads checkpoints without importing Trainer.
4. `/analyze` SPA — sidebar, policy heatmap, visit overlay, deep-link (`?moves=<base64>&checkpoint=<path>`).

**Key decisions.** Python-driven MCTS (not Rust `analyze_position`) — avoids FFI callback complexity; PyMCTSTree already exposes `select_leaves`/`expand_and_backup`. Gumbel SH in `/analyze` uses raw Q (not `completed_q_values`) — interactive-only; production SH in `engine/src/game_runner.rs` stays authoritative. `model_loader.py` duplicates `_extract_model_state` / `_infer_model_hparams` from Trainer to sidestep optimizer/scheduler imports; sync test added.

**Post-review fixes:** deep-link XSS (typeof validation), BOARD_SIZE from checkpoint metadata (was hardcoded 19), checkpoint path-traversal guard, dead var cleanup, `analyze_bp.checkpoint_dir` configurable.

---

