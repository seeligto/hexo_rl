# Future Refactors — Not In This Pass

## Completed on 2026-04-02

- **Renamed `native_core/` → `engine/`** — including PyO3 module name
- **Renamed `python/` → `hexo_rl/`** — all import paths updated
- **Renamed `hexo_rl/logging/` → `hexo_rl/monitoring/`** — `setup.py` → `configure.py`
- **Removed `Rust` prefix from exported types** — `ReplayBuffer`, `SelfPlayRunner`, `InferenceBatcher`

## Completed on 2026-04-13 (§86 / §87 / §88 / §93)

- **Split `engine/src/replay_buffer/` into `mod` + `storage` + `push` + `sample` + `persist` + `sym_tables`** (§86). Old `sampling.rs` merged into `sample.rs`.
- **Split `engine/src/game_runner.rs` into `game_runner/{mod, worker_loop, gumbel_search, records}.rs`** (§86).
- **Gated `pyo3/extension-module` behind a Cargo feature** so `cargo test` links libpython without `--no-default-features` (§87).
- **Split `scripts/train.py` into `scripts/train.py` + `hexo_rl/training/{loop, batch_assembly, aux_decode}.py`** (§88). `scripts/train.py` dropped from 1,132 to 319 lines.
- **Extracted the 12-fold augmentation kernel into Rust (`apply_symmetry_24plane`)** and exposed via PyO3 (`engine.apply_symmetry`, `apply_symmetries_batch`, `compute_chain_planes`) (§93 C8/C10). Pretrain augmentation now routes through the Rust kernel; Python `_apply_hex_sym` deleted.
- **Deleted dead `TensorBuffer` and `SelfPlayWorker.play_game()`** (§93 C9.5) after confirming no live path referenced them.
- **Consolidated hex coordinate helpers into `hexo_rl/utils/coordinates.py`** (`flat_to_axial`, `axial_to_flat`, `cell_to_flat`, `axial_distance`) (§93 C11).

---

## Deferred — do when the time is right

### Split large files

These files have grown past a comfortable single-responsibility boundary.
Split only when their scope makes navigation painful — not before.

- `hexo_rl/monitoring/web_dashboard.py` — Flask server, SocketIO logic, and route handlers in one file.
  Split into `web_dashboard/server.py`, `web_dashboard/routes.py`, `web_dashboard/socket_handlers.py` if it keeps growing. Root-level `dashboard.py` no longer exists.
- `hexo_rl/bootstrap/corpus_analysis.py` — analysis logic mixed with CLI argument parsing.
  Split analysis functions into a library module; keep CLI as thin wrapper.
- `hexo_rl/bootstrap/pretrain.py` — dataset loading, training loop, and validation in one file.
  Split into `pretrain_train.py` and `pretrain_validate.py` if it keeps growing.
- `scripts/benchmark.py` — MCTS bench, inference bench, buffer bench, and worker bench all inline.
  Extract each benchmark into `hexo_rl/benchmark/` submodules; keep `scripts/benchmark.py` as dispatcher.

### Deduplicate Python / Rust symmetry tables (partial)

State+chain augmentation scatter: **dedup complete** per §93 C8/C10.
`engine.apply_symmetries_batch` is the single source of truth; pretrain
calls it via `make_augmented_collate`. `tests/test_pretrain_aug.py`
(F1) and `tests/test_chain_plane_rust_parity.py` (F2) guard the
dedup against drift.

### Game replay storage — DONE

Full game records are persisted to `runs/<run_id>/games/<game_id>.json`
via `hexo_rl/monitoring/game_recorder.py`. Viewable at
`/viewer/game/<id>` with threat overlay, MCTS visit heatmap, and
scrubber. In-memory index capped at `viewer_max_memory_games: 50`;
disk rotated to `viewer_max_disk_games: 1000` oldest-first.

### Corpus pipeline metadata

`hexo_rl/corpus/pipeline.py` currently discards source metadata (which games came from
humans vs SealBot vs random). Add a `source_id` field to the NPZ schema so training-time
sampling can weight sources differently without re-running the full export.

### Evaluation results pagination

`hexo_rl/eval/results_db.py` loads all historical match results into memory for
Bradley-Terry recomputation. At 1M+ matches this will be slow. Add a windowed query
(last N matches per pair) as an alternative computation mode.

### Anchor snapshot lineage (R4, audit 2026-04-18)

`checkpoints/best_model.pt` is overwritten on each graduation and no DB row
records the graduation event itself, so the weights that defined anchor-at-step-N
are unrecoverable after the next promotion. Elo back-fitting and ablation replay
cannot reconstruct the chain.

Fix shape: write `checkpoints/anchors/anchor_{step}.pt` on promotion (append-only)
and add a `graduations` table `(run_id, promoted_at_step, prev_anchor_step,
wr_best, ci_lo, ci_hi, timestamp)`. Prune by run_id/age policy.

See `reports/elo_db_anchor_audit_2026-04-18.md` for context.

### py-spy flame graph on live training

Blocked on `py-spy` Python 3.14 support (0.4.1 fails with "Failed to find python
version from target process"). Re-attempt when upstream lands. Expected to confirm
NN forward dominates wall-time; if otherwise, reopen the worker-parallelism
hypothesis. Tracked as Q18 in `docs/06_OPEN_QUESTIONS.md`.

---

# Docs drift — deferred LOW/NOTE (post-W5)

Findings from `reports/master_review_2026-04-18/verdict.md` Section 4. Skipped as low ROI; fix opportunistically when touching the referenced file.

| ID | File | One-line claim |
|---|---|---|
| G-006 | CLAUDE.md bench section | References wrong baseline authority |
| ~~G-007~~ | ~~configs/training.yaml (chain-aux comment)~~ | **DONE** — swept by PROMPT20 24-plane audit |
| ~~G-008~~ | ~~engine/src/lib.rs (to_tensor docstring)~~ | **DONE** — swept by PROMPT20 24-plane audit |
| G-009 | docs/01_architecture.md | MCTSTree PyO3 kwargs example wrong |
| G-010 | docs/01_architecture.md | `Board(size=19)` + `apply_move(row, col, player)` API example wrong |
| G-011 | docs/03_tooling.md | `buffer.push(...)` stale 4-arg signature |
| G-012 | docs/01_architecture.md | `draw_reward` config key doesn't exist (is `draw_value`) |
| G-013 | docs/07_PHASE4_SPRINT_LOG.md §36 | Temperature config keys don't exist |
| G-014 | CLAUDE.md repo layout | Missing docs, krakenbot_bot.py, scripts |
| G-015 | docs/06_OPEN_QUESTIONS.md Q14 | "Blocked on submodule add" — submodule present |
| G-016 | docs/05_community_integration.md | Stale `python/…` path prefix |
| G-017 | docs/05_community_integration.md | `serve_bot.py` / `bot_cli.py` don't exist |
| G-018 | docs/08_DASHBOARD_SPEC.md | "train.py emits events" — moved to loop.py (§88) |
| G-019 | docs/08_DASHBOARD_SPEC.md | Changelog labels 2026-04-18 as §101 — actual §104 |
| G-020 | docs/02_roadmap.md | Phase 4.0 exit criterion wrong worker throughput number |
| G-021 | docs/00_agent_context.md | MCTS target stale (≥140k sim/s) |
| G-022 | docs/07_PHASE4_SPRINT_LOG.md §5 | Still references "50 games vs best" |
| G-023 | docs/07_PHASE4_SPRINT_LOG.md §67 | decay_steps internal inconsistency |
| G-024 | docs/07_PHASE4_SPRINT_LOG.md §73 | Grep-proof line numbers swapped (self-flagged) |
| G-025 | docs/09_VIEWER_SPEC.md | Last-updated 2026-04-03 pre §78 /analyze |
| G-026 | docs/07_PHASE4_SPRINT_LOG.md §3 | Replay-buffer file-split pointer slightly stale |
| G-027 | docs/00_agent_context.md | Tensor-encoding description "8 history + 2 meta" differs from docs/01 |
