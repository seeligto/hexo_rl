<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §87 — gate pyo3 extension-module behind cargo feature (2026-04-13)

Removed `extension-module` from `[features] default` so bare `cargo test` links libpython without `--no-default-features --features test-with-python`. Works because Rust tests don't call `Python::with_gil()` — no interpreter bootstrap needed. `maturin develop` reads `features = ["extension-module"]` from `pyproject.toml` and activates it explicitly. `test-with-python` retained as escape hatch. Commit `chore(build): gate pyo3 extension-module behind cargo feature`.

---

