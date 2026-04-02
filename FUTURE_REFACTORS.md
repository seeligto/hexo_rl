# Future Refactors — Not In This Pass

## Completed on 2026-04-02

- **Renamed `native_core/` → `engine/`** — including PyO3 module name
- **Renamed `python/` → `hexo_rl/`** — all import paths updated
- **Renamed `hexo_rl/logging/` → `hexo_rl/monitoring/`** — `setup.py` → `configure.py`
- **Removed `Rust` prefix from exported types** — `ReplayBuffer`, `SelfPlayRunner`, `InferenceBatcher`

---

## Deferred — do when the time is right

### Split large files

These files have grown past a comfortable single-responsibility boundary.
Split only when their scope makes navigation painful — not before.

- `dashboard.py` (root) — Flask server, SocketIO logic, and route handlers all in one file.
  Split into `dashboard/server.py`, `dashboard/routes.py`, `dashboard/socket_handlers.py`.
- `hexo_rl/bootstrap/corpus_analysis.py` — analysis logic mixed with CLI argument parsing.
  Split analysis functions into a library module; keep CLI as thin wrapper.
- `hexo_rl/bootstrap/pretrain.py` — dataset loading, training loop, and validation in one file.
  Split into `pretrain_train.py` and `pretrain_validate.py` if it keeps growing.
- `scripts/benchmark.py` — MCTS bench, inference bench, buffer bench, and worker bench all inline.
  Extract each benchmark into `hexo_rl/benchmark/` submodules; keep `scripts/benchmark.py` as dispatcher.

### Deduplicate Python / Rust symmetry tables

The 12-fold hex augmentation symmetry tables are defined in both:
- Rust: `engine/src/replay_buffer/sym_tables.rs`
- Python: `hexo_rl/selfplay/policy_projection.py` (for policy mapping back to global coords)

They should agree by construction, but divergence is possible. Long-term: generate both from
a single source-of-truth (a codegen script or a shared constant file). Not urgent while both
tables are small and tested independently.

### Game replay storage

Currently self-play games are processed into (state, policy, outcome) triples and discarded.
No full game record is kept. This blocks:
- Post-hoc review of interesting games
- Opening book construction from self-play data
- Debugging value estimation errors on specific positions

Add an optional `game_replay_path` config key. When set, serialize full move sequences
(not tensors) to a compact binary format (e.g. one u16 per move × game length) alongside
the replay buffer. Keep this off by default — storage cost is low but I/O adds overhead.

### Corpus pipeline metadata

`hexo_rl/corpus/pipeline.py` currently discards source metadata (which games came from
humans vs SealBot vs random). Add a `source_id` field to the NPZ schema so training-time
sampling can weight sources differently without re-running the full export.

### Evaluation results pagination

`hexo_rl/eval/results_db.py` loads all historical match results into memory for
Bradley-Terry recomputation. At 1M+ matches this will be slow. Add a windowed query
(last N matches per pair) as an alternative computation mode.
