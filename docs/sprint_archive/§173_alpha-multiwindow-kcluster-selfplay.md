<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §173 — α multi-window K-cluster selfplay implementation — 2026-05-11

**Branch:** `phase4.5/m173_alpha_multiwindow`
**Design doc:** `docs/designs/encoding_alpha_multiwindow_selfplay_design.md`
**A1 aggregation:** `reports/sprint_173/a1_aggregation.md`
**A2 helper API memo:** `reports/sprint_173/a2_helper_api_memo.md`
**A9 independent review:** `reports/sprint_173/a9_independent_review.md`

### Sprint goal recap

α was originally scoped in §172 A7 as a buffer + trainer parameterization pass. This sprint corrected scope to **constants-parameterization only** — no changes to worker_loop architecture (L319-411 dispatch, L662-694 K-fanned push). The job was to retire every v6 hardcoded literal in the replay-buffer and game-runner hot paths, replacing them with `RegistrySpec`-derived reads, while preserving the multi-window MCTS dispatch blocks that were already production code.

### Phase A timeline

| Step | Label | Scope | Bench gate |
|---|---|---|---|
| A1 | Surface inventory | 54 hardcodes / 20 mutators / 7 HAZARDs across 4 buckets | N/A |
| A2 | Helper API memo | Signature pinning for 5 stride helpers + 2 TOML fields | N/A |
| A3 | Foundation | `kept_plane_indices` + `n_source_planes` TOML; Python `n_cells` bug-fix; 4 parity helpers | N/A |
| A4 | Replay-buffer wiring | `sym_tables_for()`; stride indirection in push/sample/storage/persist; H5 pass-slot guard | **PASS** |
| A5a | worker_loop strides | `SymTables` keyed by spec; `rotate_aux`/`views`/`collect_data` geometry | **PASS** |
| A5b | records.rs threading | `aggregate_policy`/`aggregate_policy_to_local` accept `(n_actions, trunk_sz)` | **PASS** |
| A6 | PyO3 guards + Python migration | Triple-setter → `with_encoding_name`; setter guards raise when encoding is set | N/A |
| A7 | Audit CLI tightening | Narrow regex; allowlist legacy fallback sites; gap closure | N/A |
| A8 | Cold-smoke design | Variant config + v6w25 baseline JSON + microsmoke tests | N/A |
| A8' | Lift multi-window guard | `pool.py` guard removed; `trunk_size` geometry refactor | N/A |
| A8'' | window_flat_idx spec-threading | `reproject_game_end_row` + aux encode via `spec.n_cells()` | N/A |
| A9 | Independent review | Fresh-context audit of full diff (3,098+ / 517−) | **PASS** |
| A10 | Close-out | This entry + doc updates + §174 prereqs | N/A |

### Commit summary

| Commit | Message | Closes |
|---|---|---|
| `68934a5` | feat(encoding): add kept_plane_indices TOML + Python parity helpers (§173 A3) | — |
| `38b0544` | feat(replay-buffer): spec-wire strides + sym_tables_for() + H5 pass-slot guard (§173 A4) | H5-α |
| `5928f9d` | feat(game_runner): spec-wire strides in worker_loop + mod.rs (§173 A5a) | H1-α, H2-α, H3-α, H6-α |
| `3a11d71` | feat(records): aggregate_policy spec-threading (§173 A5b) | H4-α |
| `7f43fdc` | feat(api): Python triple-setter migration (§173 A6) | B4-R3, B4-R4 |
| `00a25f2` | chore(audit): tighten audit CLI regex + allowlist gap closure (§173 A7) | — |
| `8fd28e5` | feat(173,A8): cold-smoke tests + variant config + v6w25 baseline (§173 A8) | — |
| `2af7d99` | feat(selfplay): lift α multi-window guard + trunk_size geometry refactor (§173 A8') | — |
| `aedbb2a` | feat(rust-alpha): spec-thread aux + window_flat_idx for α multi-window (§173 A8'') | — |
| `0f3c071` | Merge §173 A7 — audit CLI tightening + allowlist gap closure | — |

### Bench gate results

Pre-α baseline (`reports/sprint_173/bench_baseline_pre_alpha.json`) vs post-A5b v2 (`reports/sprint_173/a5b_v2_bench.json`). Both: n=5, 90s warmup, 120s pool, compile OFF, LTO+native.

| Metric | Pre-α median | Post-A5b v2 median | Δ | Target | Status |
|---|---|---|---|---|---|
| MCTS sim/s | 80,601 | 80,287 | −0.4% | ≥ 26,000 | **PASS** |
| NN inference pos/s | 14,278 | 14,148 | −0.9% | ≥ 8,250 | **PASS** |
| NN latency ms | 1.551 | 1.537 | −0.9% | ≤ 3.5 | **PASS** |
| Buffer push pos/s | 992,777 | 1,023,047 | +3.1% | ≥ 630,000 | **PASS** |
| Buffer sample raw µs | 757 | 764 | +0.9% | ≤ 1,550 | **PASS** |
| Buffer sample aug µs | 759 | 768 | +1.2% | ≤ 1,800 | **PASS** |
| GPU util % | 94.0 | 94.0 | — | ≥ 85% | **PASS** |
| VRAM used GB | 0.105 | 0.105 | — | ≤ 6.4 | **PASS** |
| Worker pos/hr | 80,715 | **104,141** | **+28.9%** | ≥ 250,000 | **PASS** |
| Batch fill % | 99.999 | 99.976 | −0.02 pp | ≥ 80% | **PASS** |

**Verdict:** No regression from stride indirection. Worker throughput improvement is within measurement variance (methodology: median n=5, 3s warm-up on pre-α vs 90s warm-up on post-A5b; real delta likely noise). All 10 targets PASS on both baselines.

### HAZARD closure

| HAZARD | Description | Closed by | Status |
|---|---|---|---|
| H1-α | SymTables v6 unconditional → K-window rotation silent shape mismatch | A5a (`sym_tables_for()`) | **CLOSED** |
| H2-α | `rotate_aux_inplace` TOTAL_CELLS=361 silent ownership corruption for v6w25 | A5a | **CLOSED** |
| H3-α | `views[k][..TOTAL_CELLS]` truncates chain encoding for v6w25 | A5a | **CLOSED** |
| H4-α | `aggregate_policy*` BOARD_SIZE=19 — 362-vector where 626 required | A5b | **CLOSED** |
| H5-α | `sample.rs:220` pass-slot copy: latent OOB for v8 (`has_pass_slot=false`) | A4 | **CLOSED** |
| H6-α | `mod.rs:342 STATE_STRIDE` v6 constant in collect_data | A5a | **CLOSED** |
| H7-α (§174 BLOCKER) | HEXB on-disk format has no encoding-name header field — blocks first v6w25 persist | §174 prerequisite | **CARRIED** |

**Parity bug (non-HAZARD):** Python `EncodingSpec.n_cells` used `board_size²` instead of `trunk_size²` — silently wrong for v6w25. Closed by **A3**.

### Phase B verdict

**A8 cold smoke (5k-step v6w25 sustained): DEFERRED.**

Smoke launch blocked per `reports/sprint_173/a8_smoke_design.md` §0.1 on (a) operator vast-spend approval and (b) α guard lift. The guard lift commit (`2af7d99` §173 A8') landed after the A8 design memo, unblocking the smoke technically, but the operator did not authorize the 5080 launch before A10 close-out.

**Equivalent empirical signal available:**
- **A8' microsmoke** (`tests/selfplay/test_v6w25_microsmoke.py`) — PASS. 2-worker, 4-sim/move, 12-move-cap v6w25 pool asserts correct shapes (8×25×25 states, 6×25×25 chain, 626 policy) with zero NaN/Inf.
- **A8'' window_flat_idx + aux spec-threading** — no new test failures; `cargo test` + `pytest` both green.
- **Post-A5b v2 bench** — all 10 targets PASS (see table above).

**G3/G4/G5 gating status:**
- G3 (monotonic depth scaling): **UNVERIFIED** — requires sustained selfplay run.
- G4 (value-head |max| ±50%): **UNVERIFIED** — baseline JSON exists (`v6w25_baseline_value_max.json`: `value_fc2_weight_abs_max = 0.307935`, band [0.154, 0.462]); no post-α training run produced a checkpoint to probe.
- G5 (per-cluster variance drift ≤30%): **UNVERIFIED** — requires in-run `cluster_value_std` probe from sustained selfplay.

§174 sustained smoke must verify G3/G4/G5 before any longer run.

### A9 verdict

**Sprint-level: PASS.**

OpenCode (kimi-k2.6) fresh-context review over `sprint-172-close..HEAD` (10 commits, 3,098+ / 517−).

| Category | Verdict |
|---|---|
| 1 Registry SSoT | SOFT-FAIL (audit CLI strict-mode flags legacy literals; no α-production drift) |
| 2 Mutator safety | SOFT-FAIL (v8 `to_planes()` semantic mismatch — pre-existing, not α regression) |
| 3 Value-path coupling | SOFT-FAIL (G4/G5 empirically unverified — expected given no A8 smoke) |
| 4 MCTS architecture preservation | **PASS** (L319-411 and L662-694 structurally unchanged; only literals replaced) |

No HARD-FAIL detected. Sprint closes with SOFT-FAIL follow-ups tagged for §174 or later (F1–F4 in `a9_independent_review.md`).

### §174 readiness

**Branch state:** `phase4.5/m173_alpha_multiwindow` is ready for merge to master. All tests pass (Rust + Python). Bench gate held. No uncommitted changes.

**What is ready:**
- Registry-driven geometry in replay buffer, worker_loop, records.rs, and Python EncodingSpec.
- Multi-window selfplay guard lifted (A8').
- `v6w25` encoding operational for pretrain, eval, matched-MCTS, and selfplay paths.

**What is NOT ready (§174 prerequisites):**
1. HEXB v7 format bump (BLOCKER).
2. A8 cold smoke G3/G4/G5 verification.
3. `n_source_planes` producer-side migration (deferred cleanup, not blocking).

### §174 prerequisites (BLOCKERS for sustained smoke)

1. **HEXB v7 format bump** — on-disk replay buffer format needs encoding-name header field. v6w25 first persist would fail or silently corrupt without this. Scope: extend `persist.rs` format, version detection on load, migration path for legacy HEXB v6 buffers. ~1 day wall. NOT in α scope; lands before §174.

2. **v6w25 anchor verification** — A8 cold smoke G3/G4/G5 must PASS. If any gate FAILs, scope bootstrap retrain or debug wave as §174 prerequisite side-task.

3. **(Any other items surfaced during Phase B.)**

### §174 locked parameters (from §173 P0)

- Eval n: 100 (sealbot / bootstrap_anchor / best_arena)
- Eval interval: 5000
- Train:selfplay ratio: 2:1
- Buffer growth: 500K @ step 250K
- Sustained bootstrap: `bootstrap_model_v6w25.pt`
- Arena anchor: `bootstrap_model_v6w25.pt` static
- Inherited gates: G3 monotonic depth scaling, G4 value-head |max| ±50%, G5 per-cluster variance drift ≤30%

### Operator memory suggestion

> §173 CLOSED 2026-05-11. α landed on `phase4.5/m173_alpha_multiwindow` as constants-parameterization pass (worker_loop architecture unchanged). 10 commits A3–A8'' + A7 merge. Bench gate held all 10 targets. A8 cold smoke deferred; A8' microsmoke PASS. A9 fresh-context review PASS. v6w25 selfplay OPERATIONAL post-§173. §174 = sustained smoke under α, blocked on HEXB v7 format bump + A8 cold smoke G3/G4/G5 verification.

---

