# Post-Q13 Cleanup Report — 2026-04-15

Branch: `chore/post-q13-cleanup` (cut from `feat/q13-chain-planes` tip)

---

## Bloat: tracked file delta

| | Before | After |
|---|---|---|
| Tracked files | ~295 (est. pre-branch) | 273 |
| Lines removed | — | ~11,928 (Commit B) + 42 (Commit C) |

**Commit B removals (22 files):**
- `archive/sweep_2026-04-13_gpu_util/run_{a,b,c}/{train.jsonl,dmon.log,train.log}` (9 files)
- `archive/dirichlet_port_2026-04-10/trace.jsonl`
- `archive/verify_gumbel_2026-04-10/diag_trace.jsonl`
- `archive/diagnosis_2026-04-10/diag_A_trace_{training,worker}.jsonl`
- `archive/bench_investigation_2026-04-09/{cold_run_1,cold_run_3}.txt`
- `archive/gumbel_vs_puct_bench_2026-04-09/{baseline_puct_run{1,2},gumbel_full_run{1,2},run_meta}.txt`
- `docs/07_PHASE4_SPRINT_LOG_BACKUP.md`
- `.github/PULL_REQUEST_q13.md`

All dirs verified: `verdict.md` / `results.md` / `summary.md` present before cut.

**Commit C removals (1 file):**
- `docs/06_MEMORY_MCP.md` — zero external references. Superseded by CLAUDE.md MCP section.

**Kept (live refs exist):**
- `docs/CORPUS_REPORT.md` — referenced sprint log + `scripts/analyze_human_corpus.py`
- `docs/q12_s_ordering_audit.md` — referenced sprint log line 1868

---

## Section D — benchmark.py target sync

| Metric | Old target | New target | Source |
|---|---|---|---|
| NN inference batch=64 | 8,500 | **8,250** | CLAUDE.md post-rebaseline §72 |
| Worker throughput | 625,000 | **500,000** | CLAUDE.md post-rebaseline §72 |
| Buffer aug (1,400) | unchanged | unchanged | Section F pending |

Also added `aux_chain_weight: 1.0  # Q21: currently degenerate (aux_target = input slice)` comment to `configs/training.yaml`.

---

## Section E — debug_assert

`encode_state_to_buffer` gains a `debug_assert_eq!(out.len(), 24 * TOTAL_CELLS, ...)` at the end of the function. Free in release builds; catches shape drift in dev/test. 131+4 Rust tests pass.

---

## Section F — buffer aug fuse result

**Change:** fused two-loop scatter in `apply_symmetry_24plane` into a single loop over `0..N_PLANES` using a pre-built `src_plane_lookup [[usize; 24]; 12]` table stored in `SymTables`. No branch on plane index inside the loop.

**F1 parity:** 3/3 PASS (byte-exact vs ReplayBuffer path).

| Metric | Pre-fuse (§92 C6) | Post-fuse (n=5, 2026-04-15) | Delta |
|---|---|---|---|
| aug median (µs) | ~1,742 | **1,637** | −105 µs (−6%) |
| aug IQR (µs) | — | ±95 | — |
| Target (µs) | 1,400 | 1,400 | — |

**Decision:** 1,637 µs > 1,500 µs relaxation floor → **STOP**. Fuse is committed (correct, measurable improvement) but target not relaxed. Needs profiling before any relaxation.

---

## Section G — worker throughput variance

`make bench` run immediately after Section F bench (machine not fully cooled).

| Metric | n=5 result | Prior baseline |
|---|---|---|
| Median pos/hr | 463,201 | 659,983 |
| IQR | ±241,194 (**52%**) | ±8.6% |
| Range | 428.6k–781.2k | — |
| GPU temp at run start | not logged (hot from bench) | — |

**Decision (IQR 52% >> 15% threshold):** variance is real and architectural.

**Q25 opened** at HIGH priority. Gate: do NOT launch sustained self-play until Q25 is diagnosed.

Q25 hypotheses (in priority order):
1. Thermal drift within 5-run window (RTX 3070, desktop, no monitoring)
2. 24-plane state assembly adds InferenceBatcher queue-fill jitter
3. InferenceBatcher queue sized for 18-plane states; 33% larger payload → worker stalls
4. Q18 CUDA stream contamination compounding with thermal variance

---

## Commits (chore/post-q13-cleanup)

| Hash | Commit |
|---|---|
| afe8ef2 | chore(gitignore): exclude raw logs from archive and reports/mcts_depth_*/ |
| 680053e | chore(repo): remove raw logs and transient artifacts from archive per bloat audit |
| 1d77106 | chore(docs): remove one-off audit docs subsumed by sprint log |
| f0fbbbd | fix(bench): sync benchmark.py targets to post-rebaseline CLAUDE.md values |
| b1d1f5e | feat(buffer): debug_assert on encode_state_to_buffer output length |
| ee904ee | perf(buffer): fuse plane-scatter loops in apply_symmetry_24plane |
| 40d80f6 | chore(bench): worker throughput n=5 result + open Q25 |

---

## Open questions opened

- **Q25** (HIGH): worker throughput IQR 52% — diagnose before sustained self-play. See `docs/06_OPEN_QUESTIONS.md`.

## Test state

- Rust: 131 + 4 = **135 tests PASS**
- Python: **722 tests PASS**, 1 deselected (slow marker), 4 errors pre-existing (`engineio` missing, unrelated to this branch)
- F1 parity: **3/3 PASS**
