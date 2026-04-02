# Future Refactors — Not In This Pass

The following refactors were considered during the 2026-04-02 cleanup audit
but deferred because they would break imports across the codebase without
functional benefit at this stage.

- **Do NOT rename `python/` → `hexo/`** — every import path in scripts/,
  tests/, and configs references `python.`. Renaming is pure churn until
  the package is published or the namespace conflicts.

- **Do NOT rename `native_core/` → `engine/`** — maturin, Cargo.toml, and
  PyO3 module registration all reference `native_core`. Renaming touches
  build config, CI, and every `from native_core import` line.

Both can be revisited when the project reaches packaging/distribution phase.
