<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §93 — Q13 fix-up + F1 root cause + F3 dead code removed + pretrain v3b (2026-04-15)

**What.** Ten-commit fix-up on the `feat/q13-chain-planes` branch:
C8 extracted the Rust augmentation kernel and exposed it to Python via
three PyO3 bindings (`apply_symmetry`, `apply_symmetries_batch`,
`compute_chain_planes`); C9 landed three byte-exact parity guards for
F1/F2/F3; C9.5 deleted the dead `TensorBuffer` assembler surfaced by the
F3 guard; C10 routed pretrain augmentation through the Rust kernel
(eliminating the broken `_apply_hex_sym` path that corrupted chain
planes in pretrain v3); C11 consolidated four hex coordinate helpers
into `hexo_rl/utils/coordinates.py` with round-trip tests; C12–C15
landed the W1–W4 cleanups from the review (broken-four + triple-axis
test cases, optional `legal_mask` on chain loss, dashboard wiring for
three aux losses, `encode_planes_to_buffer` rename and 18-plane
docstring cleanup). C16 regenerated the bootstrap from scratch via
the corrected pipeline as pretrain v3b, and this entry (C17) records
the outcome.

**Why.** §92 landed the 24-plane break atomically as C3, and the C6
pretrain v3 produced a working 24-plane `bootstrap_model.pt` — but the
`reports/review_q13_q19_landing_26_04_14.md` post-landing audit caught
F1: `hexo_rl/bootstrap/pretrain.py:55-133::_apply_hex_sym` scattered
state tensors by pure coordinate permutation, with neither the
`axis_perm` remap for planes 18..23 nor the `(row=q, col=r)` coordinate
convention used by `_compute_chain_planes` and the Rust `SymTables`.
Result: pretrain v3's 15 epochs saw chain planes that contradicted the
stones in 11 of every 12 augmented samples, and the trunk learned
whatever cross-axis garbage came out. Phase 4.0 self-play cannot start
from a bootstrap whose Q13 signal is randomised.

The review also drafted an F3 tensor-buffer parity guard. That guard
caught real divergence (`TensorBuffer.assemble()` still produced
(K, 18, 19, 19) post-§92) — but the live-path trace
(`reports/tensor_buffer_live_path_26_04_15.md`) showed the divergence
was in *dead code*: `SelfPlayWorker.play_game()` is the only caller and
no production path reaches it (the live self-play loop is the Rust
`SelfPlayRunner` via `WorkerPool`). Zero self-play checkpoints were
corrupted through F3; C9.5 deleted the dead code outright rather than
rewriting it. "One assembler is better than two."

**Commit sequence (details in git log + `reports/q13_fix_26_04_15.md`):**

| Commit | Scope |
|---|---|
| C8 | Extract `apply_symmetry_24plane<T: Copy>` kernel from `ReplayBuffer::apply_sym`; expose via PyO3 as `engine.apply_symmetry`, `apply_symmetries_batch`, `compute_chain_planes`. Thread-local `SymTables`; raises `SymTables` + `encode_chain_planes` to `pub`. |
| C9 | Add F1/F2/F4 guards. F1: `test_pretrain_aug.py` — buffer-vs-binding byte-exact parity over 4,000 draws. F2: `test_chain_plane_rust_parity.py` — Python vs Rust `_compute_chain_planes` across 21 positions (open/blocked 3/4, XX.X.XX, triple-axis, edge runs, near-five). F4: oracle-note comment in `test_chain_plane_augmentation.py`. |
| C9.5 | Delete dead `TensorBuffer`, `SelfPlayWorker.play_game`, and their tests. F3 guard retired — zero corrupted checkpoints, live path is Rust `SelfPlayRunner`. |
| C10 | Route pretrain augmentation through `engine.apply_symmetries_batch`. Delete `_apply_hex_sym` / `_precompute_hex_syms`. New `make_augmented_collate`; 12×362 policy scatter table. 20-batch timing probe at launch. |
| C11 | Consolidate hex-coord helpers into `hexo_rl/utils/coordinates.py` (`flat_to_axial`, `axial_to_flat`, `cell_to_flat`, `axial_distance`). 28 tests. Migrate 5 call sites. |
| C12 | Test-coverage gaps: triple-axis-intersection + XX.X.XX broken-four chain-value pins. |
| C13 | `compute_chain_loss` gains optional `legal_mask`; default path byte-exact unchanged. |
| C14 | Surface `loss_chain / loss_ownership / loss_threat` in terminal + web dashboards. |
| C15 | Rename `Board::encode_18_planes_to_buffer` → `encode_planes_to_buffer`; update `get_cluster_views` doc comment. |
| C16 | Pretrain v3b + `threat_probe_baseline v4`. v3 archived as `bootstrap_model_v3_broken_aug.pt`. |
| C17 | §93 sprint log landing. |

**F1 root cause.** Pre-C10 `_apply_hex_sym` had two bugs: (1) no `axis_perm` remap on planes 18..23; (2) `(col=q, row=r)` convention in `_precompute_hex_syms` vs `(row=q, col=r)` in `_compute_chain_planes` / Rust `SymTables`. Both eliminated by routing through `apply_symmetry_24plane<f32>` — same kernel the ReplayBuffer uses, with `axis_perm` derived from hex basis transform and pinned by `test_chain_plane_augmentation.py`.

**Pretrain v3b results (current production bootstrap).** 15 epochs × 779 batches at batch_size=256, ~40 min on RTX 3070. End-to-end DataLoader ~32.7 ms/batch (numpy↔tensor boundary dominates; Rust scatter sub-ms).

| metric | gate | v3b | note |
|---|---|---|---|
| policy_loss (final) | ≤ 2.47 | **2.1758** | matches v3 — corpus + optimiser unchanged |
| value_loss (final) | ≤ 0.59 | **0.4990** | |
| opp_reply_loss (final) | — | **2.1846** | |
| chain_loss (final) | ≤ 0.01 | **0.0018** | degenerate plateau (Q21: aux target is slice-equivalent) |
| 100-game RandomBot greedy wins | ≥ 95 | **100/100** | PASS |

**The v3→v3b win is correctness, not aux-scalar.** Chain planes are now byte-exactly consistent with stones under every augmentation (F1 fix). Whether that uplifts tactical sharpening is a Phase 4.0 sustained-run question, not a pretrain-loss-scalar question.

**Threat-probe baseline v4.** `fixtures/threat_probe_baseline.json` regenerated against v3b bootstrap; schema v3 → v4. Contrast −0.9366 — same untrained-head noise-band as v3. `probe_threat_logits.py --write-baseline` returns exit 0 by construction; §91 C1 relative gate applies to post-self-play checkpoints only.

**Downgraded expectations carry over from §92.** Q21 (wider-window aux target) parked. **Q22** (chain-plane Rust port deleting Python `_compute_chain_planes` and its ~80 µs/call cost) parked — F2 parity guard pins the two paths together. **Q23** (tensor-assembler consolidation) **closed** by C9.5 — only `GameState.to_tensor()` + `encode_state_to_buffer` remain.

**Guards snapshot:**

| Guard | File | Coverage |
|---|---|---|
| F1 pretrain-aug parity | `tests/test_pretrain_aug.py` | 3 positions × 12 syms, 4,000-draw buffer coverage |
| F2 chain-plane parity | `tests/test_chain_plane_rust_parity.py` | 21 hand-picked positions, byte-exact |
| F4 invariance oracle | `tests/test_chain_plane_augmentation.py` | 4 positions × 12 syms, independent Python oracle |

**Reports:** `reports/review_q13_q19_landing_26_04_14.md` (F1-F7/W1-W4 audit); `reports/tensor_buffer_live_path_26_04_15.md` (F3 trace); `reports/q13_fix_26_04_15.md` (C8–C17 landing summary).

---

