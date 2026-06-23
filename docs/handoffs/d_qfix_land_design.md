# D-QFIX-LAND — Design (Phase 1)

Branch `worktree-d-qfix-land` @ canonical HEAD `e132e67`. Read-only design. No code.
Dispatcher-verified independently (grep + line reads) — see §Verification.

## 5-line summary
1. **Divergence between the two completed-Q sites (S1↔S2) = ZERO in the math.** Both clamp `[-1,1]`, both use `v_hat=W/N`, both FMA-fuse, identical sigma. Only off-window/overflow handling + output type differ. A2a = pure dedup, not a refuse-to-unify tightrope.
2. **Site S3 (`gumbel_search.rs::score`) is NOT a 4th completed-Q copy.** It is `gumbel + log_prior + sigma(q_hat)` with `q_hat=0` for unvisited (no v_mix). Routing it through completed-Q = category error. Stays UNTOUCHED.
3. **Dispatch divergences #2 (q-range [-1,1] vs [0,1]) and #3 (v_mix backed-up vs raw-NN) are BOTH FALSE.** No site clamps `[0,1]`; no site uses raw-NN root value. All use `clamp(-1,1)` + `W/N`. **A2b unification has nothing to unify — CANCELLED, recorded as a null.**
4. **File map:** new `engine/src/mcts/completed_q.rs` (~150 LOC); A1 extends `selection.rs` in-place (252→~300). policy.rs 784→~560. All files <1000.
5. **Parallelism: A2a and A1 SERIALIZE** (both touch `mcts/mod.rs`). Land A2a first (golden-proven), then A1 on top.

---

## Verification (dispatcher, independent of design agent)
- `grep -nE "clamp\(0|0\.0, *1\.0\)|clip\(.*0\.0.*1\.0"` over policy.rs / gumbel_search.rs / completed_q.py → **no `[0,1]` clamp**. Only `clamp(-1.0,1.0)`. #2 FALSE confirmed.
- `v_hat = root.w_value / root.n_visits as f32;` at policy.rs **L158 (S1)** and **L378 (S2)** — identical. L157 comment: `// v_hat: root value estimate (W/N from root node)`. #3 FALSE confirmed.
- S1/S2 sigma `(c_visit + max_n) * c_scale` L180/L386; FMA `sigma_scale.mul_add(completed_q, log_prior)` L193/210/227 and L396/413/427; v_mix FMA `(sum_n_f/visited_prior_sum).mul_add(policy_weighted_q, v_hat)` L169/L382 — identical. Byte-identity confirmed.

---

## 3.1 — DIVERGENCE TABLE (gating artifact)
Sites:
- **S1** `engine/src/mcts/policy.rs::get_improved_policy` L84–235. Dense `Vec<f32>` output.
- **S2** `engine/src/mcts/policy.rs::get_improved_policy_ls` L297–437. `LegalSetPolicy{dense, overflow}` output. Comment: completed-Q math "FROZEN".
- **S3** `engine/src/game_runner/gumbel_search.rs::score` L116–132. **Different computation** (Gumbel SH candidate score; q_hat=0 unvisited; no v_mix/softmax).
- **S4** `hexo_rl/selfplay/completed_q.py::compute_improved_policy` L17–111. Python oracle (tests/docs), NOT production.

| Axis | S1 | S2 | S3 | S4 | A2a action |
|---|---|---|---|---|---|
| q clamp range | `clamp(-1,1)` | `clamp(-1,1)` | `clamp(-1,1)` on q_hat | `clip(-1,1)` | IDENTICAL — #2 FALSE |
| v_mix source (v_hat) | `root.w_value/root.n_visits` (L158) | same (L378) | N/A (q_hat=0) | passed-in W/N | IDENTICAL S1=S2 — #3 FALSE |
| v_mix formula | `(1/(1+sum_n))·FMA(sum_n/vps, pwq, v_hat)` L169 | same L382 | N/A | non-FMA L70-72 | S1=S2 identical; S4 non-FMA (parity tol) |
| sigma scale | `(c_visit+max_n)*c_scale` L180 | same L386 | same L129 then ·q_hat | same L87 | scale sub-expr identical all 4 |
| logit numeric form | FMA `mul_add` L193/210/227 | FMA L396/413/427 | plain mul ×3 + add L129/131 | f64 non-FMA L93 | S1=S2 FMA-identical; S3 differs; S4 oracle |
| unvisited fill | `v_mix.clamp(-1,1)` L189 | same L393 | `q_hat=0` L124 | `v_mix` clip L84 | S1=S2 identical; S3 fundamentally different |
| sum_n==0 edge | normalized prior L143-155 | normalized prior + overflow L357-376 | N/A | prior/uniform L50-59 | S1 vs S2 differ ONLY in overflow routing |
| degenerate guards | `max_logit==-inf→ret`, `sum_exp<=0→ret` L196/213 | same L401/416 | NaN→Equal in sort | softmax | S1=S2 identical |
| log_prior floor | `prior.max(1e-8).ln()` L191 | same L395 | same (ctor) | passed-in | identical |
| q_sign perspective | `mr==1?-1:1` L103 ·w_value | same L312 | `root_mr==1?-raw:raw` L122 | caller root-persp | S1=S2 identical; S3 same inline |
| **off-window/overflow** | drop if `action>=n_actions` L118 | keep-if-covered/drop L332; ragged overflow map | N/A | `legal_mask` arg | **THE one real S1↔S2 divergence — kept in caller** |
| output type | `Vec<f32>` dense | `LegalSetPolicy{dense,overflow}` | `f32` scalar | `np.ndarray` | per-site caller scatter |

**Verdict on dispatch's claimed divergences:** #2 FALSE, #3 FALSE. **PARKED for A2b: nothing** — there is no math divergence. A2b CANCELLED (recorded as a null; the "drifted copies / §104 ~2× target-floor inflation via copy-drift" premise is falsified for the math path). The only genuine residual numeric-form gap is S4 oracle f64-non-FMA vs Rust f32-FMA → handled as parity tolerance, not unification.

**What A2a MUST preserve:** the single S1↔S2 divergence = off-window handling + output container. Preserved by keeping the **scatter/output stage in the caller**; the shared fn returns per-child masses, each caller scatters into its own container (dense `Vec<f32>` vs ragged `LegalSetPolicy`).

**S3 ruling (flagged category error):** S3 = `gumbel_values[i] + log_priors[i] + sigma`, `sigma=(c_visit+max_n)*c_scale*q_hat`, `q_hat=0` unvisited. No v_mix, no completed-Q vector, no softmax. Consolidating into completed-Q would force a v_mix unvisited-fill where the paper requires q_hat=0 → would change Gumbel root selection. **S3 stays separate; do NOT extract a shared sigma helper** (1-line saving, risks an FMA-vs-mul byte change on the SH hot path). gumbel_search.rs L116–132 untouched by A2a (regression-golden only).

---

## 3.2 — FILE MAP (locked; operator constraint <1000 LOC/file)

### New: `engine/src/mcts/completed_q.rs` (~150 LOC)
Free functions over borrowed child slices (no MCTSTree coupling; isolation-testable). Caller pre-extracts `CqChild{visits,prior,q_val}` + `CqAgg{sum_n,max_n,visited_prior_sum,policy_weighted_q,v_hat}` in its single child-scan, calls shared fn, scatters masses into its own container.
- `v_mix(&CqAgg) -> f32` — FROZEN FMA form matching L169/L382.
- `improved_policy_masses(&[CqChild], &CqAgg, c_visit, c_scale) -> Vec<f32>` — the L180–228 body, scatter removed. Empty vec ⇒ degenerate (caller returns empty container).
- `prior_fallback_masses(&[CqChild]) -> Vec<f32>` — sum_n==0 normalized-prior path (matches S1 L143-155 AND S2 L357-376; they normalize identically).

Register `mod completed_q;` in `engine/src/mcts/mod.rs` after `pub mod policy;`. Fns `pub(super)` / free fns on borrowed slices.

**Caller shape after refactor (S1 and S2):** scan pool → build `Vec<CqChild>` + `CqAgg` + coord side-table → call shared fn → scatter masses by coord/flat-index into the caller's container. Off-window/overflow decision + coord→action mapping STAY in each caller (the one real divergence). Shared fn never sees `n_actions` or window geometry.

### `gumbel_search.rs::score` — UNTOUCHED (S3 separate; no shared helper). See §3.1 ruling.
### Python `completed_q.py` — UNTOUCHED (parity oracle). Contract = Rust↔Python parity via golden fixtures, `atol=1e-5`.
### A1 — extend `selection.rs` in-place (NOT a new interior_select.rs). 252→~300 LOC.

### Projected LOC
| File | now | after | <1000? |
|---|---|---|---|
| `mcts/completed_q.rs` (new) | 0 | ~150 | yes |
| `mcts/policy.rs` | 784 | ~560 | yes |
| `mcts/selection.rs` | 252 | ~300 | yes |
| `mcts/mod.rs` | — | +~10 | yes |
| `game_runner/gumbel_search.rs` | 308 | 308 (untouched) | yes |
| `game_runner/config.rs` | ~310 | +~8 | yes |
| `hexo_rl/selfplay/completed_q.py` | 111 | 111 (untouched) | yes |

policy.rs 784→~560 **CONFIRMS extraction REDUCES** the file.

---

## 3.3 — A1 DESIGN

### Branch A1 replaces: `selection.rs::select_one_leaf` L123–138
```
let best = if cur == 0 {
    if let Some(forced) = self.forced_root_child { forced }
    else { pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value) }
} else { pick_best_puct(self, first, n_ch, cur, parent_n, fpu_value) };
```
Current behavior = **PUCT everywhere, Gumbel-forced override at root only**. The `forced_root_child` override is Gumbel's mechanism and stays OUTSIDE the new match (independent of selector).

### Enum + field
```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InteriorSelector { Puct, GumbelImproved } // Puct == current behavior; GumbelImproved wired, NOT default
```
Field `interior_selector: InteriorSelector` on `MCTSTree` (mcts/mod.rs near `forced_root_child` L85). Replacement keeps the `forced_root_child` short-circuit, then `match self.interior_selector { Puct => pick_best_puct(..), GumbelImproved => pick_best_gumbel_improved(..) }` in BOTH the root-else and the interior branch.

**`Puct` arm golden-identical** to today. **`GumbelImproved` body = documented PLACEHOLDER** (delegates to `pick_best_puct` with a `// PLACEHOLDER: interior-Gumbel rule is future work` comment + one-time `eprintln` warning when selected) — so it is wired + config-selectable but cannot silently masquerade as a real different rule. Real interior-Gumbel semantics are OUT OF SCOPE for land-now.

### Config path (hard-read, no literal default in source)
1. yaml `mcts.interior_selector: puct` (string) → hard-read in the python config loader (`KeyError` on missing, NO `.get(default)`).
2. PyO3: add `interior_selector: String` to `SelfPlayRunnerConfig` via the **`#[pyo3(get,set)]` pattern** (config.rs L94–102 template), NOT a positional `new()` arg (positional ctor is INV19-pinned; adding an arg breaks ~10 call sites + byte-equivalence invariant). Parse string→enum at `SelfPlayRunner::new`, panic on unknown variant.
3. Thread → `SelfPlayRunner` fields → `worker_loop` → set `tree.interior_selector` immediately after `MCTSTree::new_full` in `worker_loop/inner.rs` (lower blast radius than extending `new_full`).

### Default-arm golden
Rust `#[test]`: tree with `interior_selector=Puct`, run `select_one_leaf`, assert selected-leaf path byte-identical to HEAD snapshot. Extend `mcts/tests.rs` forced-child cohort (L579–709).

---

## 3.4 — DIRICHLET-EVAL SPEC (Phase 5; eval-only, no training/restart)

### Where Dirichlet hits the Gumbel root
- Apply: `policy.rs::apply_dirichlet_to_root` L453–489 (`child.prior=(1-ε)·prior+ε·noise`, FMA L474). Noise: `dirichlet.rs::sample_dirichlet`.
- Production call: `worker_loop/inner.rs` ~L747–753 — after root expansion, before `GumbelSearchState::new`, gated `dirichlet_enabled && !is_intermediate_ply`. Perturbs priors feeding BOTH Gumbel-Top-k sampling AND the completed-Q `log_prior` in the recorded target.
- Eval call: `hexo_rl/eval/gumbel_search_py.py::_apply_root_dirichlet` L52–58, gated by `dirichlet` arg of `run_gumbel_on_board` (applied before candidate select + target read).

### Target head read
The completed-Q improved-policy target `tree.get_improved_policy(c_visit,c_scale)` (harness reads at gumbel_search_py.py L108). Training metrics: `policy_target_entropy_fullsearch` + `policy_target_kl_uniform_fullsearch` (trainer.py L109-110/L150-152). Eval computes the same two scalars from the harness `improved_policy` vector, paired `dirichlet=True` vs `False`.

### Harness / checkpoint
`scripts/gumbel_sims_smoke.py` — already loads a banked ckpt, builds positions by self-play rollout, runs `run_gumbel_on_board(..., dirichlet=False)`. Phase 5 adds a paired `dirichlet=True` pass + entropy/KL per position + diversity triad. **§D-ARGMAX:** dedupe byte-identical sequences, bootstrap CI over DISTINCT games, inject opening diversity.

### Pre-registered CONFIRM (BOTH required)
1. **Target-floor entropy drops ≥1.8× turning Dirichlet OFF:** `entropy_on / entropy_off ≥ 1.8` (Dirichlet-OFF = sharper/lower-entropy target). State sign explicitly in eval.
2. **Diversity-abort stays in band** (distinct games): distinct-game fraction ≥0.5, opening top-1 mass ≤0.6, early-game entropy ≥ floor (calibrate from a HEAD baseline run). ANY out-of-band → the entropy delta is a collapsed-regime artifact → **ABORT, do not confirm.**

FAIL either → PARK as cosmetic; record the null (kills the §104→Dirichlet attribution).

---

## 3.5 — IMPL PARALLELISM → SERIALIZE (A2a then A1)
Overlap = `engine/src/mcts/mod.rs` (A2a adds `mod completed_q;`; A1 adds enum/field + edits `new_full`). Disjoint lines but the tree's central module is touched by both → clean serial chain over a parallel merge. **Land A2a first** (blast radius: completed_q.rs + policy.rs + 1 mod line + python parity test; golden-provable in isolation). **Then A1** rebased on proven-byte-identical A2a.

---

## 3.6 — GOLDEN-CAPTURE PLAN
Capture as **exact float bits** (`f32::to_bits()` Rust / `struct.pack` Python) — NOT `abs()<1e-6` — so an FMA→non-FMA regression is caught (sub-ULP but nonzero on x86 without contraction).

Existing infra: `policy.rs` tests L531–784 (`setup_improved_policy_tree` L680–717 builds `(visits,w_value,prior)` children); `gumbel_search.rs` tests L175–308; `tests/test_completed_q.py` L29–180.

Per-site capture:
- **S1**: full dense `Vec<f32>` improved-policy, all `to_bits()`.
- **S2**: `dense` (to_bits) + `overflow` map (sorted key, value to_bits). Needs a fixture with one off-window **covered** and one **uncovered** child (exercise keep/drop L332 + scatter L432) — IMPL must add an off-window fixture variant (stub `is_covered`/cluster centers).
- **S3**: per-candidate `score(offset,max_n).to_bits()`. Regression guard only (A2a must NOT touch it).
- **S4**: `np.ndarray` f64 via `.tobytes()`. Parity reference.

Adversarial fixtures (RED-TEAM): (1) extreme q=±1 clamp rails; (2) NaN-prone q / prior=0 floor → guards return empty container byte-identically; (3) all-unvisited root (sum_n==0) prior-fallback incl. S2 overflow normalization; (4) single legal move; (5) v_mix activation w/ partial visits incl. the `visited_prior_sum<=1e-8` else-branch; (6) mr==1 perspective flip.

Mechanics: HEAD capture test prints/serializes `to_bits()` to `engine/tests/golden/completed_q/`. Post-refactor parity test asserts `to_bits()` equality (Rust↔Rust bit-exact). Python parity test asserts Rust f32→f64 vs S4 oracle within `atol=1e-5` (cross-language NOT bit-exact).

---

## Open items → IMPL / RED-TEAM
1. **A2b CANCELLED** — no divergence to unify (dispatcher-confirmed). Recorded as a null in handback.
2. S2 off-window golden needs a minimal `is_covered`/cluster-center stub.
3. `GumbelImproved` arm = documented placeholder (delegates to PUCT + once-warning); real rule out of scope.
4. Phase-5 diversity bands need a HEAD baseline run before pre-registration is final.
