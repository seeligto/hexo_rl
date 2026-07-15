# S7 blocker fixes — F1 / F2 / F3 closed — 2026-07-15

**Base:** worktree `gnn-integration` HEAD `2a94aa1` (clean at start). Closes the three blockers
named in `reports/probes/gnn_integration/S7_smoke_gate.md` (Part-1 FAIL: F1+F2; Part-2 FAIL: F3).
Machine: laptop RTX 4060 Max-Q (CUDA available — all CUDA-gated tests genuinely ran).

**Status: ALL THREE FIXED + tested + proof-run past the Part-1 crash sites.**

---

## F1 — `run4_gnn.yaml` inherited-`<auto>` corpus prefill (config)

**Fix:** `configs/variants/run4_gnn.yaml` now declares `mixing.pretrained_buffer_path: null`
explicitly (with a full rationale comment at the key). `null` (not the `"<auto>"` literal) skips
`expand_auto_paths` entirely → no `_pretrained_buffer_path_auto_resolved` provenance stamp → 
`load_pretrained_buffer` takes its `if not pretrained_path: return None` no-op branch
(`hexo_rl/training/batch_assembly.py:210-213`). Real corpus-mix OFF per `run4_gnn_design.md`
Decision 1 — the sha-pin gate (`gnn_axis_v1` deliberately unpinned in `_CORPUS_SHA_PINS`,
`hexo_rl/encoding/resolvers.py:163-167`) is never reached.

**Inherited-default audit** (every base yaml in `orchestrator._BASE_CONFIGS` + the run4 rulings —
corpus-mix OFF, bot-mix 0 — checked against what the merge switches ON by default):

| Base default | Verdict for a graph fresh-init run |
|---|---|
| `training.yaml mixing.pretrained_buffer_path: "<auto>"` | **THE F1 LANDMINE** — fixed (explicit `null`) |
| `training.yaml mixing.bot_corpus_path: null` + `bot_batch_share: 0.0` | Safe as inherited (`load_bot_corpus_buffer` no-ops on null path) — AND the variant already declares `bot_batch_share: 0` (F1 preserve-ckpt-baked). No change needed. |
| `training.yaml mixing.{initial_pretrained_weight 0.8, min_pretrained_weight 0.1, decay_steps 200k, pretrain_max_samples 0}` | Inert whenever `pretrained_buffer_path` is falsy — `step_coordinator`'s mixed-batch branch requires `pretrained_buffer is not None` (step_coordinator.py:923-925). Documented in the yaml comment; NOT declared (declaring dead knobs would imply they act). |
| `training.yaml mixing.buffer_persist: true` + base `buffer_persist_path` | Already namespaced per-lineage by the variant (`replay_buffer_run4_gnn.hexg`, §RUN3-STEP0 law). Safe. |
| `training.yaml aux weights (aux_opp_reply 0.15, ownership 0.2, threat 0.2, chain 1.0, uncertainty 0.1, entropy_reg 0.005)` | Inherited non-zero — but `_train_on_graph_batch`'s standing §6.3 guard covers `aux_opp_reply/uncertainty/ownership/threat/chain/ply_index` loud-raise… on the **trainer path only, and only for weights ≠ 0**. Verified NOT a launch blocker (proof-run reached the training loop; the raise fires at first train step *if* it fires). **NOTE:** entropy_reg_weight (0.005) is NOT in the guard's list — flagged for the S7 re-run to observe first real train_step (see "residual watch items"). |
| `selfplay.yaml legal_move_radius_jitter: true` | Dead code for ALL registry-spec encodings since §172/§173 (memory: legal-move-radius-jitter-dead-code) — inert, no declaration needed. |
| `selfplay.yaml solver_enabled/seed_fraction/gumbel_mcts/forced_win_policy` | All default OFF. Safe. |
| `monitoring.yaml web_dashboard: true` | Behavioral but representation-agnostic (run3 rode it); launch scripts pass `--no-web-dashboard` when isolating. Not a graph-specific trap. |
| `game_replay.yaml enabled: true` | Representation-agnostic recorder; ran fine in proof-run. Safe. |
| `model.yaml value_head_type: scalar` / `in_channels: 8` | Already explicitly shadowed by the variant (pre-existing, commit-B). |
| `training.yaml ema.enabled: false`, `per_class_target_temperature.enabled: false`, `torch_compile: false` | All OFF by default. Safe. |

**Extra landmines found beyond the named F1:** none launch-blocking. One residual watch item
(entropy_reg_weight outside the §6.3 aux guard — silently unused on the graph path rather than
loud) — a wart, not a blocker; the §6.3 guard's own design decision only names the six head-aux
weights.

**Test (sha-pin gate never reached):** `tests/test_run4_gnn_launch_path.py` — 3 tests. Drives the
REAL merge entrypoint (`orchestrator.load_train_config` + `flatten_config_and_resolve_encoding`)
against the on-disk yaml (pattern: `tests/test_run3_corpus_launch_path.py`), asserts: raw yaml
declares explicit `null` (regression-pinned, fails if the key is ever removed again); resolved
config carries no corpus path, no auto-resolve provenance flag; **the actual consumer
`load_pretrained_buffer` returns `None`** on the resolved config (no raise — the exact S7 Part-1
first-crash call); `bot_batch_share` resolves 0. Plain unit tests (no integration marker): pure
config resolution, runs in the default `not slow and not integration` gate — the class of miss
named in memory `launch-path-needs-integration-gate` is closed by testing the config contract in
the default suite AND by the proof-run below.

---

## F2 — `cuda_warmup` dense-only (code gap)

**Fix:** `hexo_rl/training/lifecycle.py::cuda_warmup` is now representation-aware,
hoisted-single-branch (house pattern — one `_representation` read, one if/else inside the SAME
`torch.no_grad()` + `torch.autocast(device_type="cuda")` context both paths share):

- New optional `spec` kwarg (default `None` → `getattr(spec, "representation", "grid")` → grid —
  every pre-S7 caller/test signature-compatible, dense path **byte-identical** (the grid branch
  body is untouched, same dummy-width logic, same autocast, same synchronize).
- Graph branch: `_graph_warmup_batch(spec, device)` builds a minimal 1-position batch through the
  SAME production seam a real graph step rides — `HexgBuffer.push_graph_position` →
  `sample_graph_batch(1, augment=False)` (C1 native builder) → `collate_graph_batch`
  (full 18-assertion resolver) → `stone_mask_from_batch` — then calls
  `inf_model.forward_batch(...)`. Reuses the existing collate module + the buffer-driven
  tiny-batch pattern from `tests/training/test_gnn_hexg_buffer.py`; the synthetic position is
  self-contained (one stone + empty neighbor; no dependency on the WP-A `wpa_positions.json`
  fixture, which may be absent on a launch host). fp16 semantics: same autocast context as dense
  (matches the trainer's own `autocast(...)` around `forward_batch`).
- Call site `hexo_rl/training/loop.py:129` → `cuda_warmup(inf_model, device, board_size,
  spec=_arch.spec)` (`InfModelArch.spec` is the WP-4 threaded registry spec — the same single
  dispatch discriminant `build_net` uses; no new resolution).

**Test:** `tests/training/test_cuda_warmup_representation.py` — 4 tests, all
`skipif(not torch.cuda.is_available())` (ran for real on this CUDA laptop):
graph warmup does not raise + `forward_batch` spy called once (touches the model);
omitting `spec` on a GnnNet still raises `NotImplementedError` (proves the graph branch is
spec-gated, not a masking try/except); dense warmup calls bare `forward` exactly once, never
`forward_batch` (byte-identical behavior pin); dense pre-S7 call signature (no `spec` kwarg)
still valid.

---

## F3 — graph ckpt → EVALFAIR book radius=None (code gap + PINNED RULING)

**PINNED CONTROLLER RULING (implemented exactly):** graph checkpoints resolve to the STANDARD
EVALFAIR d5 book at **r=5** (the D-LADDER instrument convention) via an EXPLICIT mapping at the
resolution site — not a silent default: an info event `graph_ckpt_evalfair_book_r5` names the
mapping; an optional CLI/config override `--graph-eval-book-radius` is accepted and validates
against the available book map. A dense ckpt's resolution is byte-identical.

**Where:** the ONE resolution authority `scripts/evalfair/core.py::radius_from_checkpoint`
(every consumer — `stage2_d5_eval`, `run_arm`, `retro_slope`, `run_retro_ckpt`,
`head_vs_krakenbot`, `head_vs_strix`, `headtohead` — funnels through it, so the mapping lands
once, not per-site):

- Schedule-scan result non-None → returned unchanged (dense curriculum path **byte-identical**;
  the F3 branch is only reachable when the scan found nothing).
- Scan `None` + ckpt's own `config['encoding']` resolves to `representation == "graph"`
  (`_ckpt_is_graph_representation` — best-effort, never raises; absent/unregistered encoding →
  conservative "not graph") → return `GRAPH_EVALFAIR_BOOK_RADIUS` (=5) or the operator override,
  logging `graph_ckpt_evalfair_book_r5` (fields: radius, overridden, msg) — the required info
  event, never silent.
- Scan `None` + NOT graph → `None` exactly as before — a dense ckpt with a stripped/missing
  schedule still hits `require_offline_radius`'s HARD-ERROR / `resolve_book_for_radius`'s raise
  (the F3 mapping can never mask a genuine dense-side bug).
- Comment at the site (verbatim requirement): whole-board net → book radius shapes opening
  diversity only; r=5 keeps numbers comparable with the dense d5 instrument; operator-overridable.

**Threading (the S7 Part-2 "no override is exposed" gap):**
`run_arm(..., graph_eval_book_radius_override=None)` (new kwarg, default `None` = byte-identical)
→ its own internal `radius_from_checkpoint` call; `stage2_d5_eval(...,
graph_eval_book_radius_override=None)` → BOTH its book-selection `radius_from_checkpoint` AND
`run_arm` (so the two resolutions agree); `run_pull_eval(...)` param; `mantis_pull_eval.py` CLI
flag `--graph-eval-book-radius` (int, default None). `run_arm`'s pre-existing
`require_offline_radius(radius, radius_stage_override=None, ...)` call is deliberately unchanged
— `radius` is already the graph-mapped concrete value by then (mapping lives in the ONE resolver,
not duplicated at the guard site); a site comment records why. An unknown override radius is
validated by the existing book map (`resolve_book_for_radius` raises "No book registered for
radius=N") — no duplicated validation.

**Test:** `tests/test_s7_f3_graph_eval_book_radius.py` — 12 tests (synthetic-minimal ckpt dicts,
mirroring `tests/test_confres_6d_offline_radius.py`'s established stub pattern; cheaper than S7's
full Trainer mint and sufficient — the fix is pure config-resolution logic, the graph forward is
covered elsewhere): graph ckpt → r=5 default + override honored; dense no-schedule stays `None`
(override inert for dense); dense WITH schedule unchanged; unregistered/absent encoding → `None`,
no raise; graph ckpt composes to the r5 book via `resolve_book_for_radius`; **unknown override
radius raises** ("No book registered for radius=7"); `run_arm` completes on a graph ckpt with a
staged r5 book and NO override (the exact pre-F3 HARD-ERROR path); `run_arm` threads an explicit
override to its own internal resolution (book pinned r4 + override 4 → guard agrees); dense
unresolvable-radius HARD-ERROR regression re-asserted post-F3; `stage2_d5_eval` threads the param
through to `run_arm` (spy).

---

## Evidence — proof-run (S7 Part-1 entry command, this host, post-fix)

`scripts/train.py --variant <byte-copy of run4_gnn.yaml + scratch buffer_persist_path only>
--iterations 3 --min-buffer-size 32 --checkpoint-dir <scratch> --log-dir <scratch> --run-name
s7f_verify --no-dashboard --no-web-dashboard` (temp variant archived at
`<scratch>/s7_verify/_s7f_verify_run4_gnn.yaml`, removed from the repo tree; NOTE — unlike S7's
smoke, NO `pretrained_buffer_path` unblock override was needed: the committed yaml carries it now).

Observed (full log `<scratch>/s7_verify/stdout.log`):

1. config resolution + startup event: merged config shows `mixing.pretrained_buffer_path: null`,
   `bot_batch_share: 0` — **F1 crash site passed** (no sha-pin raise; no corpus load attempted);
2. `train_encoding_resolved encoding_name=gnn_axis_v1` → `gnn_warmstart_loaded loaded_keys=46
   verified_tensors=46` (OQ-5 live-fire again) → model built 286,082 params dist65;
3. `cuda_warmup_start` → **`cuda_warmup_done elapsed_sec=0.3`** — **F2 crash site passed** (S7
   died here with `NotImplementedError: Module [GnnNet] is missing the required "forward"`);
4. `worker_pool_started n_workers=14` → `selfplay_stall_watchdog_armed` → main-loop `warmup`
   events (GPU 67-83% util on live graph self-play inference) — **the training loop was reached**,
   past every S7 Part-1 blocker;
5. killed cleanly at ~2 min (SIGTERM to the one launch PID, verified by /proc cmdline match — no
   pkill patterns): graceful close-out ran (`terminal_eval_skipped_on_interrupt`,
   `checkpoint_saved step=0`, `buffer_saved`, `session_end`), process exited in 1s, zero orphaned
   python processes (`pgrep -x python` = 0). No optimizer step was reached inside the 2-min budget
   (laptop graph self-play needs >2 min for a first full game at 600 sims; the dispatcher's
   "starts stepping OR ~2 min" bound applied) — steps-fire is separately proven by
   `test_gnn_train_step.py`'s 3-step CUDA leg (S7 Part-3 PASS, unchanged).

## Residual watch items (NOT blockers — for the S7 gate re-run)

- `value_spread_canary_failed: 'GnnNet' object has no attribute 'in_channels'` — WARNING-level,
  fired once in the close-out path (canary probes `model.in_channels`, dense-only attr). Logged
  and swallowed by design; a graph-aware canary (or a skip on graph specs) is a small follow-up.
- `entropy_reg_weight: 0.005` inherited non-zero is OUTSIDE `_train_on_graph_batch`'s §6.3
  aux-guard list — verify at the gate re-run's first real train step whether it's silently unused
  (likely, the graph loss is `policy + value` only) and decide declare-0 vs guard-extend.
- Aux weights (`aux_opp_reply` 0.15 / `ownership` 0.2 / `threat` 0.2 / `chain` 1.0 /
  `uncertainty` 0.1) inherit non-zero and WILL hit the §6.3 loud-raise at the FIRST train step of
  a real run — S7's re-run must confirm whether the yaml needs explicit zeros (the guard is
  deliberately loud; zeroing them in the variant is the likely fix, one commit, but that is the
  gate re-run's finding to make, not smuggled in here).

## Verification summary

| Suite | Result |
|---|---|
| NEW `tests/test_run4_gnn_launch_path.py` (F1) | **3 passed** |
| NEW `tests/training/test_cuda_warmup_representation.py` (F2, real CUDA) | **4 passed** |
| NEW `tests/test_s7_f3_graph_eval_book_radius.py` (F3) | **12 passed** |
| `tests/training tests/selfplay tests/model -q -m "not slow and not integration"` | **261 passed, 13 deselected** |
| `tests/eval -q -m "not slow and not integration"` | **90 passed, 2 skipped, 5 deselected** |
| F3 regression: `tests/test_confres_6d_offline_radius.py` + `scripts/evalfair/tests/` (incl. `test_retro_slope_fixes.py`, `test_radius_fold.py`) | **79 passed, 3 deselected** (run with the two new root tests) |
| GNN regression: `test_orchestrator_gnn_build/buffer`, `test_lifecycle_gnn_build`, `test_trainer_ckpt_load_gnn_resume` | **18 passed** |

**Total new tests: 19/19 green. Zero pre-existing failures introduced.**

Files changed (uncommitted, per instruction — nothing git-added):
`configs/variants/run4_gnn.yaml`, `hexo_rl/training/lifecycle.py`, `hexo_rl/training/loop.py`,
`scripts/evalfair/core.py`, `scripts/eval/mantis_pull_eval.py`; new tests
`tests/test_run4_gnn_launch_path.py`, `tests/training/test_cuda_warmup_representation.py`,
`tests/test_s7_f3_graph_eval_book_radius.py`.

---

## Follow-up (same day, coordinator order): watch item (1) closed — §6.3 weights zeroed

The residual watch items above flagged a KNOWN first-train-step crash (inherited nonzero aux
weights → §6.3 loud-raise) plus the entropy_reg_weight silent-no-op wart. Closed now rather than
letting the gate re-run rediscover them:

**yaml (`configs/variants/run4_gnn.yaml`):** every key the §6.3 guard checks is now declared 0
explicitly, one block with rationale (GnnNet ships policy + dist65 only, design §1.3 DROP):
`aux_opp_reply_weight, uncertainty_weight, ownership_weight, threat_weight, aux_chain_weight,
ply_index_weight` (base defaults 0.15/0.1/0.2/0.2/1.0/0.0 — ply_index pinned despite base-0, F1
preserve-ckpt-baked) **+ `entropy_reg_weight: 0.0`** (base 0.005).

**entropy_reg_weight ruling: BOTH — added to the guard AND zeroed in the yaml.** Verified read
site: `entropy_reg_weight` is consumed ONLY by the dense `_train_on_batch`
(`trainer.py` `entropy_weight = float(self.config.get("entropy_reg_weight", 0.0))`); the graph
loss body (`policy_ce + binned_value`) never reads it — on a graph config it was exactly the
silent-no-op knob class §6.3's philosophy forbids ("a positive weight with no consumer is a
config ERROR, not a silent no-op"). Zeroing in the yaml alone would have fixed run4 but left the
knob silently ignorable for every FUTURE graph variant; guard-adding makes the silent state
unconstructable. Consequence accepted: any future graph variant must declare it 0 (loud,
self-explaining error otherwise).

**Guard refactor (`hexo_rl/training/trainer.py`):** the inline §6.3 key tuple hoisted to a
module-level constant `GRAPH_FORBIDDEN_NONZERO_WEIGHTS` (now 7 keys, incl. entropy_reg_weight);
`_train_on_graph_batch` iterates the constant. Single source of truth shared with the test — the
test enumerates the SAME tuple the guard checks, so guard and test can never drift.

**Test (`tests/test_run4_gnn_launch_path.py`):** new
`test_run4_gnn_resolves_every_graph_forbidden_weight_to_zero` — imports
`GRAPH_FORBIDDEN_NONZERO_WEIGHTS`, asserts the constant still covers at least the 7 named keys
(a shrink would weaken guard AND test — same-source cuts both ways), then asserts every listed
key resolves to 0 in the REAL merged run4_gnn config (same real-entrypoint drive as the other F1
tests; also re-exercises the abort-on-warning variant validator — passed, no namespace shadows
from the new flat keys).

**Re-verification:** `tests/test_run4_gnn_launch_path.py` **4 passed** (was 3);
`tests/training/test_gnn_train_step.py -m integration` **10 passed** (guard change regression —
graph train steps still fire, fp16 CUDA leg included); `tests/training tests/selfplay tests/model
-q -m "not slow and not integration"` **261 passed, 13 deselected** (unchanged). New-test total
now **20/20**.

Additional file changed: `hexo_rl/training/trainer.py` (constant hoist + guard list + message).
