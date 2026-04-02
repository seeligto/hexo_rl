# Future Refactors — Not In This Pass

No deferred renames remain from the 2026-04-02 cleanup audit.
The following were completed on 2026-04-02:

- **Renamed `native_core/` → `engine/`** — including PyO3 module name
- **Renamed `python/` → `hexo_rl/`** — all import paths updated
- **Renamed `hexo_rl/logging/` → `hexo_rl/monitoring/`** — `setup.py` → `configure.py`
- **Removed `Rust` prefix from exported types** — `ReplayBuffer`, `SelfPlayRunner`, `InferenceBatcher`
