# §P5-CT_PINF_probes

_Relocated from `docs/07_PHASE4_SPRINT_LOG.md` (D-DOCS-DEBLOAT split, 2026-06-23). Scope: §P5-CT compound-turn defect probe + §P-INF inference attribution. Verbatim; falsified-register rows also consolidated into the sprint-log index register section._

## §P5-CT — compound-turn defect probe (CF-1 banked, CF-2 gated)

**Date 2026-05-28.** Discriminate compound-turn-adjacent defects from the
§S150 loop-regression alternative as proximate driver of the colony /
value-head discrimination collapse. Backing audit
`audit/structural/compound_turn_pipeline_audit.md`; investigation report
`reports/investigations/compound_turn_cf1_cf2_20260528.md`.

**ARM 1 — CF-1 BANKED (free, code-complete).** `expand_and_backup_single`
(backup.rs:223) hardcoded the terminal value `-1.0`, which is a sign
**inversion** for a **first-stone win**: `apply_move` (core.rs:528-532) keeps
the player on stone-1 (mr 2→1, no flip), so the `check_win` leaf's
side-to-move is still the winner → should be `+1.0`. Fix derives the sign
from `board.moves_remaining` (`mr==1`⇒+1.0, `mr==2`⇒-1.0). TDD discriminator
in `engine/src/mcts/tests.rs` (`test_cf1_stone1_win_scored_as_win` FAILS on
old code / passes on new; `test_cf1_stone2_win_still_scored_as_loss_to_mover`
proves no Case-B regression). Placed crate-internal because
`expand_and_backup_single` + `Board::cells/last_move` are `pub(crate)`
(brief's `engine/tests/inv*.rs` placement impossible). Independent review:
all 4 checks PASS. Bench (focused `mcts_sims_cpu_only` pre/post, laptop):
−1.4% / +0.8% / +1.2% — within noise, MCTS ≥73k floor PASS.

**ARM 2 — CF-2 GATED (pretrain operator/GPU-pending).** v6 drops turn-phase
planes 16/17 (registry.toml:78); v8 keeps them but a v8 smoke is a
**confounded + argmax-degenerate** CF-2 test → REJECTED. Clean recipe = new
`v6tp` registry entry keeping 16/17 (channels encoder already emits them — no
encode.rs change) + ~30-epoch §150 pretrain. **Sequencing:** run CF-1-fixed
v6 smoke FIRST; escalate to v6tp pretrain only on FAIL/SPLIT.

**CF-4 DEFERRED:** not "free" — needs a per-row ply field across the hot-path
record tuple + PyO3 boundary; ply-index aux already inert (L58).

**Pre-registered verdict (V_spread@2k primary):** PASS = V_spread>+0.10
through 5k AND SealBot WR within 10pp of peak @20k; FAIL = V_spread≤0 by 2k
(L44 sig) → pivot to §S150 hunt; SPLIT = filler-first policy improves but
V_spread collapses → CF-1 banked as correctness-only. **E1 vs E2 UNDECIDED**
pending smoke. Cosine temp stays OFF (L9).

**L61 — a hardcoded terminal sign is a compound-turn trap.** Negamax
terminal values must be derived from the leaf's side-to-move, never
hardcoded, in any engine where a move can fail to flip the player (multi-ply
turns). The `-1.0` constant was correct for every turn-final win and silently
inverted only first-stone wins — a class invisible to single-stone-game
intuition. **Why:** caught by re-deriving the sign from `apply_move`'s
turn-structure, not from the "side-to-move just lost" heuristic. **How to
apply:** any future MCTS terminal-handling edit must case on
`moves_remaining`; pin with a stone-1-vs-stone-2 unit test.

### Phase 5a — productionized CF-1 + CF-2 + CF-4 (2026-05-28)

Operator decision after the cheap arm passed: productionize all three as a
5-stage surgical wave on `master` (zero file-overlap Rust↔Python). Six
commits `fd25a37..fc0308c`: CF-1 sign fix (B1.1), v6tp registry (B1.2), CF-4
ply-index emission through `collect_data` (B1.3), v6tp Python mirror + smoke
config (B2.1), v6tp 10-plane corpus export + pretrain recipe (B2.2), CF-4
pool.py wiring (B2.3).

- **CF-4 was NOT pool.py-only** as the brief implied: `collect_data` carried
  no per-row ply index and no game_id, so the Rust record tuple had to emit
  it (the "not free" cost the probe deferred). Landed via
  `RecordTuple→WorkerResultRow→collect_data` (8→9 arrays). Bench-cold
  (per-move, not per-sim). Inert on the baseline (`ply_index_weight: 0.0`).
- **BENCH-GATE PASS:** controlled A/B on `mcts_sims_cpu_only` — no change at
  n=100/400/800; ≈257k sim/s vs 73k floor.
- **REVIEW PASS** (2 parallel fresh-context): Rust sign/plumbing correct;
  Python is a **clean single-variable delta** vs v6 (corpus seed-stable,
  policies/outcomes encoding-independent, augmentation plane-generic, fresh
  10-plane pretrain, lever stack OFF).
- **Caveat (reviewer):** CF-1 is a global engine change ⇒ a v6tp smoke is not
  comparable to the *historical* §175 v6 datapoint; attributing V_spread
  recovery to CF-2 alone needs a co-built **CF-1-only v6 control**.
- **AGGREGATION (operator GPU-pending):** `scripts/p5a_v6tp_pretrain.sh` →
  `--variant v6tp_p5a_smoke --iterations 30000`. SUCCESS = SealBot WR ≥25%
  past 30k AND V_spread >+0.10 @20k. FAILURE (V_spread collapse despite the
  plane) → Phase 5b **TD-λ** (NOT aux heads). Report:
  `reports/investigations/compound_turn_cf1_cf2_20260528.md` §8.

#### Phase 5a smoke VERDICT — QUALIFIED SUCCESS (2026-05-29, ran on vast)

30k smoke completed (lever stack OFF). Full data + reasoning: report §9.
- **SealBot WR (sampled T=0.5, in-run):** 22/18/21/21/22 % (5k–25k) — flat
  ~21% plateau, above §150 anchor 17.4%, far above §S178/§S181 collapse (0–5%).
- **SealBot WR (greedy T=0, standalone 30k ckpt): 33%** [24.6, 42.7] — clears
  the ≥25% bar, ≈2× anchor.
- **Temperature isolated as the dominant factor** of the 33-vs-21 gap: same
  KClusterMCTSBot path 0.33@T0 → 0.23@T0.5 (−10pp); path effect (KCluster vs
  ModelPlayer @T0.5) ~2pp within noise. Confirms L21/L22 — sampled flatness,
  not weak argmax.
- **V_spread (T3):** net-positive throughout (+0.22..+0.59 vs §S181's −0.016),
  degraded in the tail (+0.6→+0.22, −0.31 excursion @25.5k) — attractor
  pressing late but not winning.
- **Colony = decision, not spam:** stride5_p90=4 (vs 60), colony_ext_frac=0.0
  (vs 0.40); colony share of wins 36%→82% is legit meta per
  [[feedback_colony_is_meta_not_kill_signal]].
- **Net:** colony attractor SUPPRESSED (the §S181 target). Favors E1
  (compound-turn defects were contributing); E2 (§S150 loop-regression)
  weakened. Open: CF-1 (global) vs CF-2 (turn-phase plane) attribution →
  **CF-1-only v6 control** (`v6_p5a_control`, running) resolves it; eval at
  BOTH T=0 and T=0.5.
- **L62 — always eval at both T=0 (greedy strength) and T=0.5 (sampled-policy
  health); a single temperature misreads the model** (here: 33% vs 22% on the
  same checkpoint). **L63 — the encoding registry is NOT the single source of
  truth it claims; ~14 modules hardcode "v6≡8 planes" (WIRE_CHANNELS/
  BUFFER_CHANNELS/KEPT_PLANE_INDICES/{8,11} literals). Needs a dedicated
  encoding-width audit wave** (report §9.7).

### Next-session PROMPT 2 — H-PLANE-MISMATCH CONFIRMED + hardcode ledger (2026-05-29)

Two no-GPU deliverables, run in parallel with the CF-1 control. Report
`reports/investigations/hplane_mismatch_20260529.md`; probe
`scripts/structural_diagnosis/hplane_activation_dump.py`.

**H-PLANE-MISMATCH — CONFIRMED.** v6-family wire planes 1-3/9-11 (my/opp history
t-1..t-3) are LIVE in pretrain (Python `to_tensor` fills them from the
`move_history` deque; corpus mean-abs ≈ 0.04 ≈ the live t0 planes, nonzero in
94-99.8% of rows) but EXACTLY ZERO in self-play (Rust `encode_state_to_buffer` +
`_channels` zero them; pinning test). Matched-sample dump: corpus history-sum
0.0434 vs self-play 0.0000. So 6 of 8 (v6) / 6 of 10 (v6tp) wire planes carry full
mass in pretrain and zero in RL — a transfer cliff. Fresh-context review PASS
(re-dumped Rust path → history 0.000000). **Correction:** the zeroing is only the
history planes; turn-phase 16/17 are live on both paths.

**CRITICAL caveat — NOT a colony cause.** The mismatch is INVARIANT across all
v6-family runs; §150 (17.4%, no collapse) and §175/§S178/§S181 (collapse) share
it. It cannot be the colony differentiator — it is a constant **regression-class
baseline handicap**, scoped against/before Phase 6, NOT a colony remedy.

**Recommendation:** register `v6_live2` (`kept_plane_indices=[0,8,16,17]`, 4
planes = my/opp t0 + turn-phase) — drops exactly the mismatched history planes,
makes pretrain==selfplay. Gate a fresh-pretrain **MCTS-matched** smoke vs v6tp
(probe gates can't validate dynamic equivariance, L2). Populated-history-on-Rust
is infeasible (split-responsibility).

**Hardcode ledger (L63 follow-through):**
`audit/structural/encoding_width_hardcode_ledger.md` — **P0=2**
(`orchestrator.py:286` in_channels fallback=18; `generate_bot_corpus.py` hardcoded
v6), P1=6 (diagnostic/probe surfaces). Every reactively-fixed LIVE path verified
clean. Ledger only (clean bisect); fix wave is separate, P0s first; add v6tp as
the non-8-plane regression canary.

**L64 — a documented encoding asymmetry that is constant across runs is a
baseline handicap, never a per-run failure differentiator.** Don't attribute a
divergent failure (colony in some runs, not others) to an invariant. **Why:** the
history-plane shift read as colony-relevant in the brief but is identical in
collapsing and non-collapsing runs. **How to apply:** before proposing an
invariant as a cause, confirm it varies across the outcomes you're explaining; if
it doesn't, it's a floor, not a trigger.

### PROMPT 1 — CF-1 control launched on vast; second opponent DROPPED (2026-05-29)

- **CF-1-only v6 control LAUNCHED on vast** (5080, `phase4.5/p5a_v6tp`). Pretrain
  (`p5a_v6_control_pretrain.sh`, 8-plane v6, seed 42) running; a guarded chain
  (`scripts/_chain_v6ctl_smoke.sh`, tmux `v6ctlsmoke`) auto-launches the 30k
  `v6_p5a_control` smoke on `PRETRAIN_DONE`. v6tp 30k artifacts archived (vast
  `checkpoints/v6tp_archive/` + pulled to laptop `checkpoints/v6tp_archive/`)
  BEFORE the control clobbers the shared `checkpoints/` dir — this preserves the
  QUALIFIED-SUCCESS v6tp 30k model regardless.
- **Second-opponent disambiguation DROPPED (operator, 2026-05-29).** KrakenBot was
  the only non-SealBot bundled bot and is too weak to serve as a strong second
  opponent: its MinimaxBot emits illegal-move sentinels mid-game so the wrapper
  falls back to a neighbor-2 heuristic (early v6tp@T0 vs Kraken ≈90% reflected the
  weakened opponent, not model strength). No other strong bundled bot exists; the
  only alternative is `CommunityAPIBot` against a live `explore.htttx.io/bots/<name>`
  endpoint (needs a URL + the bot online). A trial `run_sealbot_eval --opponent`
  generalization + its eval were **reverted** (uncommitted) — the kraken arm is not
  pursued. The v6tp ~21% sampled / 33% greedy therefore stays read as
  distance-from-SealBot, NOT validated as a general plateau.
- **Attribution verdict (CF-1 vs CF-2) BLOCKED** on the control 30k results
  (~13hr). Process per report §9.6 + PROMPT 1 — spam-signal primary, V_spread
  DEMOTED (operator note 2). CF-1 + CF-4 banked as correctness regardless.

### v6_live2 encoding LANDED (H-PLANE-MISMATCH fix scaffolding, 2026-05-29)

Registered `v6_live2` (`kept_plane_indices=[0,8,16,17]`, 4 planes = my/opp t0 +
turn-phase) = v6tp minus the dead history planes. Scope/gate:
`reports/investigations/v6_live2_scoping_20260529.md`. Landed the no-GPU
scaffolding + made it actually runnable (NOT yet pretrained/smoked — GPU queued
behind the control). Code uncommitted.

- **Wiring** (all green): registry.toml entry + Python `_REGISTERED_NAMES` +
  resolvers detector (`in_ch==4 → v6_live2`) + corpus/anchor path maps + export
  `--encoding v6_live2` + `configs/variants/v6_live2_smoke.yaml` (`in_channels:4`)
  + `scripts/p5a_v6_live2_pretrain.sh`. Engine rebuilt (`make build`). Verified:
  audit parity, lookup, 4-plane model construct+forward, detector (neutral label
  → v6_live2), export slice → (T,4,19,19), round-trip + 16 Rust registry tests.
- **Fresh-context review caught 6 run-blockers** the static wiring missed — all
  one root cause the hardcode-ledger UNDER-COUNTED: chain-plane recompute hardcodes
  the **opponent t0 stone at corpus/buffer slot 4** (the v6 position), but v6_live2
  has opp at slot 1. Fixed via a registry-derived `opp_stone_slot(spec)` helper
  (`hexo_rl/encoding/resolvers.py`) at all 5 recompute sites
  (`batch_assembly.py` ×3: load_pretrained / load_bot / `_augment_recent_rows`;
  `pretrain_dataset.py` collate) + the checkpoint_loader allow-list & spec branch
  (`checkpoint_loader.py`). **Backward-compat pinned:** `opp_stone_slot`==4 for
  v6/v6tp/v6w25/v8/v7full, ==1 only for v6_live2. Regression test
  `tests/test_v6_live2_wiring.py`. Full Python suite green except 2 PRE-EXISTING
  `test_analyze_api` failures (confirmed via stash — unrelated to v6_live2).
- **L65 — the encoding-width ledger's grep patterns missed the `states[:, 4]`
  opponent-plane-index class.** A hardcode audit keyed on identifier names
  (WIRE_CHANNELS/KEPT_PLANE_INDICES/{8,11}) will not catch a bare positional
  slice like `states[:, 4]` that encodes "opp lives at plane 4". **Why:** the
  ledger reported all live paths clean, yet a 4-plane run hard-crashed at 5 such
  sites. **How to apply:** when adding an encoding with a NEW plane COUNT, grep
  positional slices (`[:, N]`, `[i, N]`) over state tensors too, and prefer a
  dry end-to-end run / fresh-context trace over a name-grep ledger alone.

### FINAL VERDICTS — PROMPT 1 + PROMPT 2 (2026-05-31)

All data in: CF-1-only v6 control (killed @20k, verdict decisive) + v6_live2 30k
smoke (ADOPT). Handoff: `reports/investigations/v6_live2_session_handoff_20260530.md`.

**PROMPT 1 — CF-2 (turn-phase planes 16/17) is LOAD-BEARING.** The CF-1-only v6
control COLLAPSED vs SealBot (9%→1%→3% @ 5/10/15k) with the self-anchor *rising*
(0.47→0.63) — the anchor↑/sealbot↓ colony-capture divergence (L34) — while v6tp
held ~21% and v6_live2 matched/beat it. CF-1's terminal-sign fix alone does NOT
suppress the attractor; the turn-phase signal carries v6tp. **Keep the plane.**
Collapse manifested as the WR/anchor divergence, NOT `colony_extension_fraction`
(stayed 0.0) — a strength collapse vs SealBot, not extension-spam. CF-1 + CF-4
banked as correctness regardless. **Second opponent DROPPED** (KrakenBot too weak —
illegal-move fallbacks); SealBot-style-specificity NOT disambiguated (only
`CommunityAPIBot` vs a live htttx.io endpoint remains). Residual = **policy
flatness** (10pp T0/T0.5 gap: v6tp 33/21, v6_live2 40/20) — policy-target hygiene,
deferred. CF-5/CF-6 UNFIXED/UNDETERMINED (gate any future KataGo FPU).

**PROMPT 2 — H-PLANE-MISMATCH CONFIRMED → v6_live2 ADOPTED.** History planes
1-3/9-11 live in pretrain (mean-abs ≈ live t0 ≈ 0.04), exactly zero in self-play —
a 6-plane pretrain↔selfplay cliff. INVARIANT across collapse/non-collapse runs ⇒
baseline handicap, not the colony cause (L64). Fix realized: `v6_live2=[0,8,16,17]`
(drop dead history, keep stones + turn-phase), 30k smoke = **ADOPT** —
**greedy 40% [31,50] (n=100) > v6tp 33%**; sampled ~0.20 ≈ v6tp ~0.21 (within CI,
trajectory 0.15/0.20/0.20/0.29/0.16); anchor 0.45→0.52 + best 0.37→0.54 climbing
with SealBot (genuine self-improvement, no colony-capture); spam clean throughout;
threat head healthy (C1 PASS 3.870 @ 8.5k vs bootstrap 0.063). Ledger: P0=2, P1=6
+ the missed positional-slice class (L65, fixed). Ledger P0s + a proper
de-hardcoding sweep still owed.

**Synthesis — production encoding = v6_live2.** `[0,8,16,17]` literally IS
PROMPT 1's answer (keep the load-bearing turn-phase = CF-2) + PROMPT 2's answer
(drop the history cliff). Simplest of the three (4 planes vs v6tp's 10),
matches-or-beats v6tp vs SealBot, no spam. **Adopt v6_live2.** 30k model archived
`checkpoints/v6_live2_rl/checkpoint_00030000.pt`; vast run hung in its final
in-run eval (instance reset) — training completed, model safe. Open: commit the
arc, ledger-P0 sweep, policy-flatness + real-second-opponent (deferred).

### De-hardcoding sweep — `resolve_arch` resolver + INV pin (2026-05-31) — PASS

Owed ledger-P0 sweep closed as a clean, test-pinned, one-commit-per-site arc
(NOT reactive). Design (question-first, operator-confirmed): REJECT
checkpoint shape-sniffing; ADOPT ONE registry-derived resolver
`hexo_rl/encoding/resolvers.py::resolve_arch(name) -> ArchSpec` {in_channels,
kept_indices, cur_stone_slot, opp_stone_slot, k_max, policy_logit_count,
history_planes, turn_phase_planes} — explicit, by name, every field from
`lookup(name)`. Folded the pre-existing `opp_stone_slot` + new `cur_stone_slot`
onto a shared `_kept_slot_of`. Rust mirror: `impl RegistrySpec` accessors
(cur/opp_stone_slot, history/turn_phase_planes) — init-time only, NO MCTS
hot-path call site ⇒ **no bench gate** (confirmed; 196 Rust tests green).

14 commits `0e89ccd..05e3365` on `phase4.5/v6_live2`. Sites routed:
- **P0-1** `orchestrator.py` fresh-run `in_channels` (was literal 18) →
  `_resolve_fresh_in_channels` → `resolve_arch(enc).in_channels`.
- **P0-2** `generate_bot_corpus.py` (was hardcoded-v6 end-to-end, no flag) →
  `--encoding` + `_resolve_generator_encoding` (19×19-single-window guard) +
  `spec.kept_plane_indices` slice threaded through factory/play/save.
- **P1-1..6** early_game_probe / build_value_probe_fixture / value_probe
  (plane-count skip→NaN guard) / windowing_diagnostic / analyze_api /
  v6_argmax_bot — each slices the RESOLVED encoding's kept set (or
  registry-scanned by `in_channels`), never v6's 8.
- **cur-slot** `batch_assembly.py` `pre/bot_states[:, 0]` → `cur_stone_slot`.
- **2 NEW L65-class finds beyond the ledger** (the grep caught what name-grep
  missed): `structural_diagnosis/track_a/{position_classifier,a3_h_bank}.py`
  hardcoded opp at v6 slot 4 (`state[4]` / `states[:, 4]`) — routed via
  plane-count→registry slot derivation.

**INV pin (both facets, GREEN):** (1) `resolve_arch == registry` parametrized
over ALL registered encodings (count-agnostic via `_load()`); (2)
`test_inv_no_positional_plane_slice` greps the live tree for bare `[:, <int>]`
plane slices, fails on any not in the documented SOURCE-layout allowlist
(game_state encoder writes, axis_distribution source read, dataset_v8 native
builder, bench synthetic input, policy index). **Teeth verified** by injecting
`arr[:, 4]` → RED, restore → GREEN.

**Verdict = PASS.** All P0+P1 + new finds resolver-routed; INV green; full suite
**1733 py + 196 rs green, 0 fail** (the anticipated `test_analyze_api` 400==200
failures did not occur). Fresh-context review: NONE surviving on live paths, 3
spot-checks PASS. **L66 — a name-grep hardcode ledger is structurally blind to
positional slices (`states[:, N]`) and leading-axis plane reads (`state[N]`);
ship a registry-derived resolver AND a grep-INV with TEETH (inject-and-revert)
so the next new-plane-count encoding fails a test, not a run** (refines L65 from
"grep positional slices too" to "pin them in CI"). Note: repo has no configured
formatter (no `[tool.*]`/pre-commit); `ruff format` would reflow ~1188 lines vs
the hand-aligned house style — deliberately NOT applied; new code matches
surrounding style; all ruff F401/F841/E402 in touched files are pre-existing.
Acceptable residuals (P2, unchanged): module-level `BUFFER_CHANNELS =
lookup("v6").n_planes` v6-family default consts; Rust `sym_tables.rs` v6
fallbacks (guarded). PARTIAL/none — no site needed a deeper refactor.

---

## §P-INF — inference attribution: GPU-bound vs FFI/dispatch (Rust-rewrite question)

**Date 2026-05-31.** Settle empirically whether self-play inference wall-clock is
GPU-bound or dispatch/FFI/GIL-bound **before** anyone specs moving inference to Rust
(per L18/L39/§S186: a tall line is a question, not headroom). Report
`reports/investigations/inference_attribution_2026-05-31.md`.

**Premise (code-confirmed).** PyO3 is crossed **per fused GPU batch, not per-leaf**:
worker submit (`submit_batch_and_wait_rust`, inference_bridge.rs:177) is `pub(crate)`
Rust→Rust (called worker_loop/inner.rs:534); the only per-batch Python-facing crossings
are `next_inference_batch` (fetch) + `submit_inference_results` (return). 2 FFI crossings
amortise over 64–192 leaves.

**Method.** Real WorkerPool selfplay, `diagnostics.perf_timing=true` +
**`perf_sync_cuda=true`** (mandatory — without the post-H2D/post-forward
`cuda.synchronize()`, `forward_us` collapses to async launch and GPU time mis-attributes
to the `.cpu()` D2H sync). 5-bucket attribution; `submit_us` added (perf-gated) to time
the 2nd crossing so Σ5 = full cycle. Driver `scripts/perf/inference_attribution_probe.py`,
laptop 4060, `v6_live2_smoke_laptop`, bootstrap_model_v6_live2.pt.

**Result (2 independent runs, per-batch p50).** forward/RT = **83.2% / 79.98%**;
**FFI=(fetch+submit)/RT = 5.7% / 7.8%**; h2d ~1.8%, d2h ~2%. Sum-check closes (untimed
tail 0.90% via independent inter-emit timestamps). Non-forward residual is dominated by
**batch-fill stall** (22% of batches hit the 16 ms `max_wait` timeout; high-fetch_wait ⇒
*lower* batch_n = worker starvation), not dispatch.

**Verdict — Rust inference REJECTED on evidence.** The `forward ≥ 80%` clause is knife-edge
(reviewer's run 79.98% < gate; **not** post-hoc moved) so the literal E1 gate is
INCONCLUSIVE-leaning-GPU-bound — but the **decision** rests on the FFI clause only (Rust can
touch nothing else): FFI <8% on both runs, and an **upper bound** (both ran half-full batches
under sync; production batch-fill ~99% ⇒ fuller batches + no stall ⇒ FFI fraction strictly
smaller). The 9 ms GPU forward is untouchable by a rewrite; §124 TorchScript trace already
captured the dispatch win in the Python server. If selfplay throughput is ever the target, the
only recoverable lever is the batch-fill stall (feeders: n_workers↑ / max_wait↓ / batch↓),
config-side. Concurs with §090, §124, §125 (80.4% forward on 4080S), L18.

---

