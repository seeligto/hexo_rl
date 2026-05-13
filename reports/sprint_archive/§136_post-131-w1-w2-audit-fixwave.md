<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §136 — Post-§131 W1+W2 audit fix wave — 2026-04-30

**Scope:** 19 commits. Correctness fixes, dead-code sweep, full doc-drift
alignment for 8-plane / bootstrap-v6 era. Q49 RNG independence audit
(COUPLED-NEGLIGIBLE).

**Correctness fixes (W1A):**
- `9b44650` `fix(smoke)`: `smoke_w4_step1_rotation` force `strict=True` load +
  read `in_channels` from checkpoint (was silent garbage trunk weights with
  `strict=False` against bootstrap-v6).
- `3ff3ffa` `fix(eval)`: `windowing_diagnostic` slice 18→8 via
  `KEPT_PLANE_INDICES` before forward (was `RuntimeError: expected 8 channels,
  got 18` on every `probe_windowing.py` invocation).

**Repo hygiene (W1B):**
- `f848054` `chore(repo)`: `.gitignore` scratch patterns (`*.bak`,
  `debug_print*.py`, `test_pool*.py`, `print_*.py`, `parse_yaml.py`,
  `test_benchmark.py`, `test_game.py`, `profile_epyc_pyspy.sh`).
- 18 root-level scratch files deleted, ~278 KB; cargo `target/` (1.7 GB) +
  `.torchinductor-cache-probe/` (81 MB) removed.

**Dead-code wave (W1C, 8 commits):**
- Deleted: `policy_projection`, `replay_poller`, `hexo_rl/api`,
  `opening_book`, `BootstrapDataset`, `LossResult` dataclass,
  `CorpusPipeline` + `HybridGameSource`, `regen_bootstrap_corpus.py`.
  Collateral: `FUTURE_REFACTORS.md` entry removed,
  `push_corpus_preview.py` docstring cleaned, `setup.sh` scaffolds
  updated, `docs/06_CORPUS_DESIGN.md` marked RETIRED.

**Q49 RNG audit (W1D, read-only):**
- `99cf6e7` Dirichlet × `sym_idx` share one `ThreadRng` (ChaCha12);
  coupling structural, not statistical. Correlation ≤ 2⁻¹²⁸ per PRF
  security. Verdict: **COUPLED-NEGLIGIBLE**. No remediation. **W3 UNBLOCKED.**
  Q49 marked RESOLVED in `docs/06_OPEN_QUESTIONS.md`.

**Doc alignment wave (W2A/W2B/W2C, 7 commits):**
- `CLAUDE.md`, `README.md`, `docs/01_architecture.md`, `docs/02_roadmap.md`,
  `docs/rules/{board-representation,phase-4-architecture,build-commands}.md`,
  `docs/00_agent_context.md`, `docs/03_tooling.md`,
  `scripts/probe_threat_logits.py`: all 18-plane / HEXB v5 / STATE_STRIDE 6498
  / bootstrap-v4 references replaced with 8-plane / HEXB v6 /
  STATE_STRIDE 2888 / bootstrap-v6. Q40 gating updated; Q45 (subtree-reuse
  cost-benefit re-derivation) added to active table.

**Test gate:** 937 py + 138 rs passed, exit 0.

**Open for W3:** L1 (`jit.trace` dynamic-shape), Q41/Q52 (v6 H2H +
SealBot anchor), Q43 (rotation × eval), Q44 (bench recalibration),
config-curation sweep (17 stale variants), CI YAML audit, tower-shim
retirement (2026-05-28).

**Forward-pointer §1 updated inline: 18 → 8 planes (§131 retcon).**

