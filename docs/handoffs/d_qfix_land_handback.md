# D-QFIX-LAND — Handback

Branch `worktree-d-qfix-land` off canonical HEAD `e132e67`. Committed to canonical repo only — **NOT deployed to the live training host** (the remote Gumbel run continues on its own binary). Two refactors landed; the one run-risk change (Dirichlet-off) is **deferred, not landed**.

## Commits (clean bisect, no Co-Authored-By)
```
19dfcfe  refactor(mcts): explicit InteriorSelector planner split        (A1)
dd7bbc4  refactor(mcts): consolidate qtransform into shared completed_q module (A2a)
e132e67  (base, canonical HEAD)
```
Each compiles standalone (`cargo check --tests` per commit); final full `cargo test -p engine` on the branch tip = all binaries `ok`, **0 failed** (lib block 226+ tests, A1 4/4, golden 2/2, INV19 4/4).

---

## What landed

### A2a — `refactor(mcts): consolidate qtransform into shared completed_q module` (LANDED)
- New `engine/src/mcts/completed_q.rs` (152 LOC): `CqChild`, `CqAgg`, `v_mix`, `improved_policy_masses`, `prior_fallback_masses`. The two completed-Q sites — `get_improved_policy` (S1) + `get_improved_policy_ls` (S2) in `policy.rs` — route through it; each keeps its OWN scatter (S1 dense `Vec<f32>`; S2 ragged `LegalSetPolicy{dense, overflow}`). Off-window/overflow handling is the only real S1↔S2 divergence and stays caller-side. `policy.rs` **784 → 672** (shrank).
- FMA numeric form preserved verbatim (`sigma_scale.mul_add(completed_q, log_prior)`, `(sum_n_f/vps).mul_add(policy_weighted_q, v_hat)`).
- **Golden evidence (byte-identical):** `engine/src/mcts/golden_tests.rs` + `engine/tests/golden/completed_q/golden_bits.txt` pin `f32::to_bits()` across nominal + adversarial fixtures (extreme q=±1, NaN/±Inf, prior=0 floor, all-unvisited `sum_n==0` incl. S2 overflow-norm, single legal move, `visited_prior_sum<=1e-8` else-branch, mr==1 perspective flip, S2 off-window covered/dropped). Post-refactor S1+S2 bits == HEAD; S3 (`gumbel_search::score`) bits unchanged. REVIEW **independently regenerated** the goldens from a clean-HEAD worktree and they byte-matched the committed file → genuine HEAD provenance, not captured-from-refactored. `tests/test_completed_q.py` adds Rust↔Python-oracle parity (`atol=1e-5`).
- `engine/src/game_runner/gumbel_search.rs` (S3) **untouched** — confirmed empty diff vs HEAD. S3 is NOT a completed-Q copy (q_hat=0 unvisited, no v_mix); folding it in would change Gumbel root selection. Left separate by design.

### A1 — `refactor(mcts): explicit InteriorSelector planner split` (LANDED)
- `InteriorSelector{Puct, GumbelImproved}` enum (`mcts/mod.rs`) replaces the hardcoded `if cur==0` branch in `selection.rs::select_one_leaf`. The `forced_root_child` short-circuit (Gumbel root-forcing) stays OUTSIDE the selector match → independent of the interior rule.
- **`Puct` arm byte-identical** to the prior branch: both root-else and interior paths call `pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value)`. Golden/identity tests in `tests.rs` confirm.
- **`GumbelImproved` = documented PLACEHOLDER**, NOT default: delegates to PUCT, warns once via `std::sync::Once`. Wired + config-selectable (the toggle the future interior A/B needs); the real interior-Gumbel rule is explicitly future work.
- **Config hard-read end-to-end:** `mcts.interior_selector` (default `"puct"` in yaml) → `pool.py` subscripts `mcts_cfg["interior_selector"]` (KeyError on missing, **no `.get` default**) → `#[pyo3(get,set)]` on `SelfPlayRunnerConfig` (NOT a positional ctor arg — **INV19 surface unchanged, INV19 tests pass**) → `InteriorSelector::from_config_str` (panics on unknown variant) → set on `tree.interior_selector` after `MCTSTree::new_full` in `worker_loop/inner.rs`.
- `selection.rs` 252 → ~302. `configs/selfplay.yaml` +1 key.
- **Bench-harness fix (folded into A1):** the new required key broke `make bench` — `scripts/benchmark.py` builds a *synthetic* `mcts` config and omitted `interior_selector` → `WorkerPool` init `KeyError` → zeroed worker throughput → bench exit 1 (a harness failure, NOT a perf regression). Fixed by threading the key into the synthetic config. All other `WorkerPool` callers load the real `selfplay.yaml` (now carries the key) and were unaffected — verified by enumerating every `WorkerPool(` call site.

---

## Performance (no regression)
- Baseline (HEAD, idle host): **MCTS sim/s 94,758** (±777). [laptop RTX 4060; the dispatch's 73k floor is a vast-5080 number — here the gate is no-regression vs local baseline, and 73k is also cleared with margin.]
- REVIEW measured HEAD vs refactored **back-to-back on the same host**: 90,124 → 89,410 = **−0.8%, within noise**. `completed_q.rs` fns already `#[inline]`; `get_improved_policy` runs once/move (not per-sim); A1 adds one resolved match branch on the hot path with no measurable cost. **No `#[inline]` rescue or revert needed.**
- Post-fix `make bench`: worker throughput recovered (33,625 pos/hr), **"All checks PASS — Phase 4.5 exit criteria met."**

---

## PARKED / NULL — the A2b divergence-unification list
The dispatch's load-bearing premise was that the qtransform copies had **drifted** on two axes, making "collapse to one" a behavior change needing a gated A2b unification. **Both claimed divergences are FALSE** — verified three ways (DESIGN agent line-by-line, dispatcher grep, REVIEW independent HEAD-regen):
- **#2 (q-range [-1,1] vs [0,1]):** FALSE. No site clamps `[0,1]`. All sites `clamp(-1.0, 1.0)`.
- **#3 (v_mix source: backed-up W/N vs raw-NN root value):** FALSE. S1 (`policy.rs:158`) and S2 (`policy.rs:378`) both use `v_hat = root.w_value / root.n_visits`. No raw-NN site exists in the production path.

**Consequence:** the two completed-Q sites were **byte-identical in their math** before A2a — so A2a is a genuine dedup, not a silent behavior change, and **A2b has nothing to unify → CANCELLED.** This is a real null result: it kills the "copy-drift → ~2× target-floor inflation (§104)" attribution for the math path. The only residual numeric-form gap is the Python oracle (f64, non-FMA) vs Rust (f32, FMA) — handled as a parity *tolerance* (`atol=1e-5`), not a unification. If a future pass wants A2b, redefine it as "FMA-align the Python oracle," which is the sole genuine gap.

---

## DEFERRED (not run this session)

### Phase 5 — Dirichlet-off eval gate (NOT LANDED, NOT RUN)
The only run-risk change. **Not evaluated** — the eval harness run was pre-empted by a host crash (below). No Dirichlet change was committed; HEAD behavior is unchanged. The eval is fully specced in `docs/handoffs/d_qfix_land_design.md` §3.4:
- Harness `scripts/gumbel_sims_smoke.py` on a banked checkpoint, paired `dirichlet=True` vs `False`, reading the completed-Q improved-policy target.
- **Pre-registered CONFIRM** (both required): off-branch target-floor entropy drops **≥1.8×** (Dirichlet-OFF = sharper target) AND the diversity-abort triad stays in band (distinct-game fraction ≥0.5, opening top-1 mass ≤0.6, early-game entropy floor) — measured on **distinct games** (§D-ARGMAX), not replicated argmax.
- FAIL either → PARK as cosmetic. **Run this on the remote vast host** (GPU eval, no training/restart).

### Phase 4 — RED-TEAM (INTERRUPTED, partial)
The adversarial pass (A2a byte-identity on inputs the goldens miss; A1 silent-mis-thread hunt; Dirichlet-spec gameability) was **interrupted mid-run by the host crash**. It had authored an adversarial fixture set (`adv_redteam_tests.rs`, since removed as throwaway) and was building when the machine cut. **A2a + A1 are REVIEW-validated** (purity + identity + perf), but the independent adversarial confirmation is incomplete. **Re-run RED-TEAM on the remote host** before treating A1's non-default `GumbelImproved` arm as battle-tested.

---

## Incident — host hard-crash (2026-06-23 16:06:08)
The dev laptop **hard-reset** during the RED-TEAM agent's `cargo test --release` (LTO, `codegen-units=1`, `target-cpu=native`). Kernel log ends cleanly at 16:06:08 with **no OOM-kill, no panic, no thermal-shutdown message, no clean-shutdown sequence** → signature of a **hardware protective power cut** (CPU/VRM thermal or power-delivery limit, faster than the kernel can log). Proximate cause: sustained heavy compile load — the ~6th heavy build stacked on ~1h45m of near-continuous CPU+GPU work (builds, `make test`, two 12-min `make bench` runs, IMPL/REVIEW builds). The adversarial *test logic* was not a memory bomb. Collateral: the crash + RED-TEAM's HEAD-comparison left the A1 threading reverted to HEAD; **re-applied from the design spec** (the enum survived in `mcts/mod.rs`; A2a was untouched) and re-verified.
- **Mitigation applied for the remainder:** all subsequent builds run **serially (never parallel) and core-capped (`CARGO_BUILD_JOBS=4`), debug not LTO-release, no `make bench`.** No further crash.
- **Recommendation:** for the deferred RED-TEAM + Phase-5 work, run on the **remote vast host**, or keep local builds throttled (`-j4`, serial, monitor temps).

---

## File map as landed (all < 1000 LOC)
| File | LOC | note |
|---|---|---|
| `engine/src/mcts/completed_q.rs` (new) | 152 | A2a shared module |
| `engine/src/mcts/policy.rs` | 672 | A2a (was 784) |
| `engine/src/mcts/golden_tests.rs` (new) | 291 | A2a byte-identity harness |
| `engine/tests/golden/completed_q/golden_bits.txt` (new) | — | committed golden bits |
| `engine/src/mcts/selection.rs` | ~302 | A1 (was 252) |
| `engine/src/mcts/mod.rs` | ~314 | A2a mod-lines + A1 enum |
| `engine/src/game_runner/config.rs` | ~322 | A1 PyO3 field |
| `engine/src/game_runner/mod.rs`, `worker_loop/{inner,mod,params}.rs` | <1000 each | A1 threading |
| `engine/src/game_runner/gumbel_search.rs` | 308 | **untouched** (S3) |
| `hexo_rl/selfplay/pool.py`, `configs/selfplay.yaml`, `scripts/benchmark.py` | — | A1 config + bench fix |
| `tests/test_completed_q.py` | +92 | Rust↔oracle parity |

Note: `worker_loop/inner.rs` is ~1306 LOC — pre-existing (1298 at HEAD + ~4 A1 threading lines), outside this dispatch's new-file scope.

## Read-only phase integrity (git-diff confirmations)
- DESIGN (Phase 1) + REVIEW (Phase 3): tracked-source tree left clean (verified `git status` after each; only artifacts = design doc, golden capture, `.venv` symlink).
- `gumbel_search.rs` (S3): empty diff vs HEAD at every checkpoint.
- IMPL diff scope matched the locked §3.2 file map at every boundary; no stray edits.

## A3 / A4 (SH-schedule fn, masked_argmax helper)
Not landed — fold-in-only-if-free, and they were not free of friction relative to A1/A2a. Deferred to a later pass (not load-bearing).
