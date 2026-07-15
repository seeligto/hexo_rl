# S7 — the 3-part integration smoke gate (OQ-7 close-out attempt) — 2026-07-15

**Worktree HEAD:** `2a94aa1` (`feat(gnn-int): WP-5b commit B — trainer graph path + corpus export +
warm-start + run4 yaml (bench-gated, WP5B-COMPLETE)`). Machine: laptop RTX 4060 Max-Q, idle before
each part (GPU 3% / 2 MiB used, no other python/torch procs).

**OVERALL VERDICT: FAIL — Part-1 FAIL, Part-2 FAIL, Part-3 PASS.**

---

## 0. Frozen-spec discovery (per "doc governs" instruction)

`docs/designs/run4_gnn_design.md` OQ-7 states the gate's 3 parts + PASS thresholds were **"not
pinned in-repo"** as of the design doc's DRAFT date (2026-07-14) and only lists *candidate* parts.
The actual **pinned** articulation lives in `docs/designs/gnn_training_path_design.md` §9
("Smoke-gate mapping (OQ-7 parts 1+3)"), which maps: Part 1 = WP-1/WP-B ragged contract +
adversarial coverage; Part 2 = WP-3 step-7 inference-seam round-trip; Part 3 = WP-5 training-path
leg (5 named pytest files). This numbering is a permutation of, not identical to, the dispatcher's
restated ordering (training-launch / eval-boundary / formal-suites) — content-mapped 1:1 by task
(dispatcher-Part-1 ≈ doc-Part-3 train-leg; dispatcher-Part-2 ≈ doc-Part-2 inference-seam; dispatcher
Part-3 ≈ doc-Part-1 ragged contract). `configs/variants/run4_gnn.yaml`'s own header comment
(committed at HEAD, WP-5b commit B) explicitly says the yaml exists to "give **the S7 integration
smoke gate** a real on-disk config to exercise (**stronger than** the inline `SelfPlayRunnerConfig`
commit-A's own smoke used)" — i.e. the yaml's own author intent confirms S7 includes a REAL
production-entrypoint launch beyond the pinned pytest files, matching the dispatcher's restatement.
I therefore ran the dispatcher's 3 parts as literally specified, using the doc's pinned test/tool
names wherever the dispatcher pointed at "whatever the design doc names."

---

## PART 1 — training-launch smoke: **FAIL**

**Gate criteria (as given):** all losses finite every step; watchdog thread/process alive during the
run; checkpoint written at/after step 3; checkpoint reloads clean.

**Command run** (production entrypoint, fresh-init, scratch-isolated):

```
.venv/bin/python scripts/train.py \
  --variant _s7smoke_run4_gnn \
  --iterations 3 --min-buffer-size 32 \
  --checkpoint-dir <scratch>/part1_ckpt --log-dir <scratch>/part1_logs \
  --run-name s7_part1 --no-dashboard --no-web-dashboard
```

`_s7smoke_run4_gnn.yaml` = byte-copy of the committed `configs/variants/run4_gnn.yaml` with two
isolation/unblock overrides (kept out of the repo, not git-added; archived at
`<scratch>/s7_smoke/_s7smoke_run4_gnn.yaml`):
1. `mixing.buffer_persist_path` → scratch path (dispatcher's explicit isolation instruction).
2. `mixing.pretrained_buffer_path: null` — **required to get past config load at all** (see Finding
   F1 below); `--min-buffer-size 32` (default 256) only for laptop wall-clock speed, not correctness.

No `--checkpoint` flag (per the yaml's own comment: BC-prefit rides the `gnn_warm_start` seam, not
`--checkpoint`, matching run4's INIT=BC-prefit ruling).

**Observed:** launch progressed cleanly through config resolution → encoding gate
(`gnn_axis_v1` resolved) → **BC-prefit warm-start fired and landed-verified (OQ-5 live-fire): 46/46
tensors** (`gnn_warmstart_loaded loaded_keys=46 verified_tensors=46`) → model built (286,082 params,
`value_head_type=dist65`) → buffer/corpus init → **crashed before the self-play worker pool started**,
i.e. before a single training step, before the watchdog armed, before any checkpoint:

```
File ".../hexo_rl/training/loop.py", line 129, in run_training_loop
    cuda_warmup(inf_model, device, board_size)
File ".../hexo_rl/training/lifecycle.py", line 173, in cuda_warmup
    inf_model(_dummy)
File ".../torch/nn/modules/module.py", line 403, in _forward_unimplemented
    raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')
NotImplementedError: Module [GnnNet] is missing the required "forward" function
```

**Blocker (F2, real bug, reproduced twice):** `hexo_rl/training/lifecycle.py::cuda_warmup` (called
unconditionally from `hexo_rl/training/loop.py:129`, with **no representation guard anywhere in the
call chain** — grepped, none exists) builds a dense `(1, C, board_size, board_size)` dummy tensor and
calls `inf_model(_dummy)`, assuming a CNN-shaped `forward()`. `GnnNet` implements only
`forward_batch(wire)` (graph input), never `forward()` — `nn.Module`'s default `forward` raises
`NotImplementedError`. This codepath was never exercised by any prior WP-1..WP-5b test because every
existing GNN smoke drives `Trainer.train_step()` / `InferenceServer._run_graph_loop` directly, never
`scripts/train.py`'s full `run_training_loop`. This is exactly the class of gap a real
production-entrypoint launch is supposed to catch (vs. the narrower pinned pytest, which does not
reach this line) — the dispatcher's insistence on a REAL launch (not just the pytest) was correct and
caught a genuine hole. **No `--no-cuda-warmup` / config knob exists to route around it** — I did not
patch `lifecycle.py` (out of scope for a gate-run task; would be inventing a workaround and calling it
a pass).

**Finding F1 (separate, upstream of F2, also real):** before hitting F2, the FIRST launch attempt
(without the `pretrained_buffer_path: null` override) hard-failed even earlier, at
`hexo_rl/training/batch_assembly.py::load_pretrained_buffer`:
`ValueError: <auto> corpus for encoding 'gnn_axis_v1' requires a sha pin; add to _CORPUS_SHA_PINS or
use an explicit path.` The committed `configs/variants/run4_gnn.yaml` does **not** declare
`mixing.pretrained_buffer_path`, so it inherits `configs/training.yaml`'s base default `"<auto>"` —
which resolves through the single-resolver corpus-sha-pin gate and hard-fails for `gnn_axis_v1` (no
registered pin; WP-5b commit B's own AS-BUILT Deviation 1 already flagged that no real corpus/sha
exists in this worktree to mint one). This directly contradicts run4_gnn_design.md Decision 1
("INIT: BC-prefit, **corpus-mix OFF**") and the yaml's own commit-B delta doc law ("F1
preserve-ckpt-baked: DECLARE, don't inherit") — the committed yaml is missing an explicit
`mixing.pretrained_buffer_path: null`/none declaration. **This is a real launch-blocker in the
checked-in `run4_gnn.yaml`, independent of my smoke isolation — a genuine production launch from this
yaml as committed would hit this exact crash first.**

**Ckpt reload leg:** not reached (no checkpoint was ever written — `<scratch>/part1_ckpt/` is empty).

**Verdict: FAIL.** Blockers: F1 (`run4_gnn.yaml` missing `mixing.pretrained_buffer_path` disable,
easy 1-line yaml fix) then F2 (`cuda_warmup` not representation-aware, a real code gap in
`hexo_rl/training/lifecycle.py`, not a config issue).

---

## PART 2 — eval-boundary: **FAIL**

**Gate criteria (as given):** eval runs end-to-end (strength not the gate); pairs complete; reported
eff_n is honest (deduped distinct games).

**Precondition note:** the task asks to take "the part-1 graph checkpoint" through
`mantis_pull_eval`. Part 1 produced **no checkpoint** (F2 above). To still exercise Part 2's own
plumbing as a diagnostic (this does **not** change Part 1's FAIL verdict), I minted a substitute
`gnn_axis_v1` checkpoint via the exact same `Trainer`/`GnnNet`/`HexgBuffer` machinery already
independently validated PASS in Part 3 (`tests/training/test_gnn_train_step.py`): fresh `GnnNet` →
BC-prefit warm-start (46/46 verified) → 3 `train_step`s on the WP-A frozen `wpa_positions.json` set
(finite losses every step) → `save_checkpoint` at step 3. This bypasses only the orthogonal,
already-isolated F2 `cuda_warmup` bug (a CUDA-kernel-precompile call, unrelated to checkpoint
content/correctness) — everything else is the real, gate-tested code path.

**Command run** (direct call to the function `mantis_pull_eval.py`'s Stage 2 delegates to, since
`mantis_pull_eval.py`'s CLI exposes no `n_pairs`/pair-count override — confirmed by reading its full
`argparse` block, `scripts/eval/mantis_pull_eval.py:832-910`; no `--pairs`/`--n-pairs`/`--radius-stage`
flag exists anywhere in the mantis→stage2→run_arm chain):

```python
from scripts.eval.mantis_pull_eval import stage2_d5_eval
stage2_d5_eval(
    ckpt_path="<scratch>/part2_ckpt/checkpoint_00000003.pt",
    book_r4="tests/fixtures/opening_books/evalfair_r4_v2.json",
    book_r5="tests/fixtures/opening_books/evalfair_r5_v2.json",
    out_dir="<scratch>/part2_out", workers=1, n_boot=200,
    expect_encoding="gnn_axis_v1", n_pairs=4,
)
```

**Observed — fails immediately, before any game is played:**

```
File ".../scripts/eval/mantis_pull_eval.py", line 242, in stage2_d5_eval
    book = resolve_book_for_radius(radius, books_by_radius, ckpt_path)
File ".../scripts/evalfair/retro_slope.py", line 137, in resolve_book_for_radius
    raise ValueError(...)
ValueError: No book registered for radius=None (ckpt=.../checkpoint_00000003.pt).
Available radii: [4, 5]. Pass the matching --book-rNone path.
```

**Blocker (F3, real, structural, not an artifact of the substitute checkpoint):**
`scripts/evalfair/core.py::radius_from_checkpoint` resolves a checkpoint's opening-book radius from
`config["selfplay"]["legal_move_radius_schedule"]` (`hexo_rl/config/resolve/radius.py
resolve_radius_from_schedule`) — a windowed-curriculum concept the whole-board GNN representation has
none of. **The committed `run4_gnn.yaml` declares no `legal_move_radius_schedule`** (confirmed by
reading it — graph runs have no radius curriculum at all), so **any** `gnn_axis_v1` checkpoint,
including a genuine future Part-1 product, resolves `radius=None`. `resolve_book_for_radius` has no
`None`-radius entry (`evalfair_r4_v2`/`evalfair_r5_v2` are keyed `{4, 5}` only) and raises before
`run_arm`'s own `require_offline_radius` hard-error gate is even reached. **No override is exposed**:
`stage2_d5_eval`'s signature has no `radius_stage_override` parameter, `run_arm` hardcodes
`radius_stage_override=None` at its one `require_offline_radius` call site (not threaded from any
caller), and `mantis_pull_eval.py`'s CLI has no `--radius-stage` flag. This is a genuine, unresolved
compatibility gap between the GNN training path and the EVALFAIR d5 instrument — `gnn_integration_scope.md`
§C5 scoped "ModelPlayer path... needs C3... 3-7 pd" for instrument compatibility, and C3 (searched-GNN
via `InferenceServer`) IS landed (WP-3 steps 6-7), but this specific book-radius coupling was never
closed. **Not fixable by a flag from outside the eval scripts as currently written** — a real fix
needs either a graph-specific book keyed off no-radius, or a code change to `radius_from_checkpoint`/
`resolve_book_for_radius`/`run_arm` to accept/special-case `representation=="graph"`.

**Verdict: FAIL.** Zero pairs ran; eff_n honesty/dedupe was never reached (nothing to dedupe — the
eval never starts). Blocker: F3 (`scripts/evalfair/core.py::radius_from_checkpoint` /
`scripts/evalfair/retro_slope.py::resolve_book_for_radius` have no graph/no-radius path, and no
override is threaded through `mantis_pull_eval.py`'s CLI or `stage2_d5_eval`'s signature).

---

## PART 3 — formal ragged round-trip + adversarial-payload suites: **PASS**

All named suites run, exact commands + counts recorded verbatim:

| Suite | Command | Result |
|---|---|---|
| Byte-parity oracle (WP-1, 1,696-position harness) | `pytest tests/test_hexo_graph_parity.py -q` | **1 passed** (90.1s) |
| 18-assertion collate contract + 9 ADV payloads (ADV-1a,1b,2a,2b,3,4,7,8,9) | `pytest tests/selfplay/test_graph_collate.py -q` | **19 passed** |
| HEXG buffer round-trip + ADV-7 aug-roundtrip enforcement (`AugRoundTripMismatch`) | `pytest tests/training/test_gnn_hexg_buffer.py -q` | **14 passed** |
| Graduated ADV-A/B/D (push-guard non-finite/out-of-range/illegal-visit-coord) | `pytest tests/selfplay/test_gnn_record_dispatch.py -q -m integration` | **2 passed** (ADV-C finiteness folded in) |
| OQ-7 part-3 train-leg, incl. **fp16=True/False parametrize** on real CUDA (BREAK-1 regression guard) + checkpoint round-trip + `encoding_name`/`schema_version` stamp assert | `pytest tests/training/test_gnn_train_step.py -q -m integration` | **10 passed** |
| BC-prefit loader landed-verify (OQ-5) + ADV dropped/corrupted-tensor raises | `pytest tests/training/test_gnn_bc_warmstart.py -q` | **10 passed** |
| Corpus export parity/held-out/provenance ADV cases | `pytest tests/test_gnn_hexg_corpus_export.py -q` | **9 passed** |
| WP-3 step-7 inference-seam round-trip smoke (doc's own "Part 2") | `pytest tests/selfplay/test_gnn_seam_smoke.py -q -m integration` | **1 passed** |
| Orchestrator/lifecycle/ckpt-load graph-build regression sanity | `pytest tests/test_orchestrator_gnn_build.py tests/test_orchestrator_gnn_buffer.py tests/training/test_lifecycle_gnn_build.py tests/training/test_trainer_ckpt_load_gnn_resume.py -q -m "not slow"` | **18 passed** |
| Rust HEXG round-trip (32 tests, incl. `push_read_roundtrip`, `persist_roundtrip_byte_identical`, `adv7_desync_is_caught_by_the_canary`, cross-format LOUD-FAIL both directions) | `cargo test --package engine hexg --lib` | **32 passed** |
| Collection sanity | `pytest --collect-only -q -m "not slow and not integration"` | **2750/2779 collected, 0 errors** |

**Total: 116/116 passed across 11 suites, 0 failures.** No ADV case was skipped; all 9 named
`gnn_ragged_contract_v1.md` ADV payloads (1a,1b,2a,2b,3,4,7,8,9) plus the graduated ADV-A/B/C/D
push-guard set are present and green.

**One honest gap flagged (not a failure, a documentation/coverage mismatch):**
`tests/selfplay/test_gnn_seam_smoke.py`'s own docstring claims "the SAME seam runs on CUDA under fp16
autocast when present," but the test hardcodes `device = torch.device("cpu")` (line 153) with no
`parametrize` — the fp16/CUDA leg of *this specific* seam-smoke test is **not actually exercised**,
despite the docstring. The fp16 leg **is** genuinely covered elsewhere and DID run on real CUDA here
(`test_gnn_train_step.py`'s `fp16=True` parametrize, gated `skipif(not torch.cuda.is_available())` —
this laptop has CUDA, so it ran and passed, exercising exactly the run4 launch regime and the BREAK-1
`ragged_policy_ce` scatter-dtype regression guard). Net: the fp16/CUDA leg dispatcher asked for is
covered by a different file than its own docstring implies; no functional gap, just a stale comment
in `test_gnn_seam_smoke.py`.

**Verdict: PASS.**

---

## Summary table

| Part | Criterion | Observed | Verdict |
|---|---|---|---|
| 1 — training launch | finite losses/step, watchdog alive, ckpt@≥3, reload clean | crashed pre-pool at `cuda_warmup` (`GnnNet` has no `forward()`); upstream config gap (F1) hit first without an override; zero steps, zero checkpoint, watchdog never armed | **FAIL** |
| 2 — eval-boundary | eval runs end-to-end, pairs complete, eff_n honest | `resolve_book_for_radius` raises immediately — graph ckpts resolve `radius=None`, no book/override exists for that; zero pairs | **FAIL** |
| 3 — formal suites | named suites run, PASS | 116/116 across 11 suites (9 ADV + graduated ADV-A/B/C/D + fp16 CUDA leg + byte-parity oracle + Rust round-trip), 0 failures | **PASS** |

**OVERALL: FAIL-part1, FAIL-part2** (Part 3 PASS). Per run4_gnn_design.md §5/OQ-7, the gate is NOT
PASS — run4 launch remains blocked on this criterion until F1 (yaml one-liner), F2 (`cuda_warmup`
representation-guard), and F3 (radius/book resolution for graph checkpoints in the EVALFAIR d5 path)
are fixed and the gate is re-run.

## Artifacts (scratch, not committed)

`<scratch>/s7_smoke/` (`/tmp/claude-1000/-home-timmy-Work-Hexo-hexo-rl/2c5d9d06-3416-4182-85c9-8c7837722c89/scratchpad/s7_smoke/`):
`_s7smoke_run4_gnn.yaml` (the isolation-patched variant copy used for Part 1), `part1_stdout.log`
(full traceback), `part1_ckpt/` (empty), `mint_part2_ckpt.py` + `part2_ckpt/checkpoint_00000003.pt`
(Part-2 diagnostic mint), `part2_eval_boundary.py` (Part-2 driver + traceback).

---
---

# Re-run after blocker fixes — 2026-07-15 (same day)

**Context:** coordinator landed uncommitted fixes for F1/F2/F3 + zeroed the 7
`GRAPH_FORBIDDEN_NONZERO_WEIGHTS` in the yaml (`reports/probes/gnn_integration/S7_blocker_fixes.md`;
touched: `configs/variants/run4_gnn.yaml`, `hexo_rl/training/{lifecycle,loop,trainer}.py`,
`scripts/evalfair/core.py`, `scripts/eval/mantis_pull_eval.py`, +3 new test files). Re-ran the frozen
3-part gate under the same contract as run 1. Machine idle before each part; verified zero orphaned
python processes and GPU released after every leg.

**RE-RUN OVERALL VERDICT: FAIL — Part-1 FAIL-as-written (capacity, semantics-PASS at labeled bs=64
deviation), Part-2 FAIL (new structural blocker F8), Part-3 PASS.** All three original blockers
(F1/F2/F3) are CONFIRMED CLEARED; the gate now fails on issues those fixes could not have reached.

---

## RE-RUN PART 1 — training-launch smoke: **FAIL as-written; all gate semantics PASS at a labeled bs=64 deviation**

**Command** (identical contract to run 1: byte-copy of the now-fixed committed yaml + scratch
`buffer_persist_path`, fresh-init, `--iterations 3 --min-buffer-size 32/64`, scratch
checkpoint/log dirs, single PID):

```
.venv/bin/python scripts/train.py --variant _s7rerun_run4_gnn --iterations 3 \
  --min-buffer-size 32 --checkpoint-dir <scratch>/rerun/part1_ckpt \
  --log-dir <scratch>/rerun/part1_logs --run-name s7_rerun_part1 \
  --no-dashboard --no-web-dashboard
```

### Attempt 1 — crashed at `resolve_anchor` (cross-lineage best_model.pt trap, §RUN3-STEP0 class)

`cuda_warmup_done elapsed_sec=0.3` (**F2 CONFIRMED CLEARED**), no corpus raise (**F1 CONFIRMED
CLEARED**) — then:

```
File ".../hexo_rl/training/anchor.py", line 520, in resolve_anchor
    if _inf_base.in_channels == best_model.in_channels and _inf_vht == _anc_vht:
AttributeError: 'GnnNet' object has no attribute 'in_channels'
```

**Forensics (F5, two legs, both real):**
- **F5a (config, §RUN3-STEP0 law violation):** `run4_gnn.yaml` namespaces `buffer_persist_path` but
  NOT `eval_pipeline.gating.best_model_path` — it inherits the SHARED `checkpoints/best_model.pt`
  (`configs/eval.yaml:146`). The blocker-fix **proof-run itself planted the trap**: it took the
  fresh-init anchor branch (no best_model existed at 19:56), then wrote its step-0 GnnNet weights to
  the shared path (`best_model_initialized path=checkpoints/best_model.pt step=0`, proof-run log).
  My re-run 11 minutes later found that file and took the anchor-exists branch. Exact replay of the
  run3 STEP0-FAIL mechanism; run3's own fix is the template (`run3_dist65.yaml:172` pins a
  per-lineage `best_model_path`). **The committed run4 yaml needs the same 1-line namespace.**
- **F5b (code):** `anchor.py:520`'s arch-match sync reads `.in_channels` unconditionally —
  dense-only attr, absent on `GnnNet`. Fires on ANY graph run that resolves an existing anchor —
  i.e. **every run4 session except the very first** (any resume / restart-after-promotion), even
  with correct namespacing. Masked in the proof-run only because no anchor existed yet.

**Mitigation applied for the re-run (within the original "outputs at scratch" contract):** archived
the contaminating artifact to `checkpoints/stale_artifacts_s7_20260715/best_model_proofrun_gnn_step0.pt`
(§RUN3-STEP0 precedent: archive, never delete) + added a scratch
`eval_pipeline.gating.best_model_path` override to the re-run variant (labeled, archived at
`<scratch>/rerun/_s7rerun_run4_gnn.yaml`).

### Attempt 2 — as-written (inherited `batch_size: 256`): **CUDA OOM at the FIRST train step (F6)**

Everything upstream now green: warmup 0.3s → anchor fresh-init to scratch → `worker_pool_started
n_workers=14` → `selfplay_stall_watchdog_armed enabled=true timeout_sec=1800` → **live graph
self-play WORKED** (5 `game_complete`, buffer filled past min in ~80s, GPU ~80%) → first train step:

```
File ".../hexo_rl/bots/strix_v1_net.py", line 66, in forward
    msg = (x.index_select(0, src) + self.lin(edge_attr)).relu()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.47 GiB.
GPU 0 has a total capacity of 7.65 GiB ... 6.49 GiB is allocated by PyTorch
```

**F6 (capacity, dev-host):** base-inherited `batch_size: 256` × self-play-distribution graphs (mean
~490/2932 nodes/edges) × GINE E×H message intermediates retained for backward, CONCURRENT with the
live inference server, exceeds the 8 GiB 4060. The run4 launch host is the 16 GiB 5080 —
**unmeasured for this concurrency** (WP-A's "torch runs bs=256 in 213 ms" was inference-only, no
backward, no concurrent self-play). The OQ-2 5080 rider must cover the TRAIN-step memory envelope,
not just inference throughput. Secondary observation, resolved by the diagnostic leg below: during
the ~18s OOM window the inference server logged **147× `NonFiniteModelOutput` (values NaN/Inf,
probs finite)** and self-play had run clean for the prior 80s — at bs=64 these failures are **0**
over the whole run, so the NaN storm is memory-pressure-coupled fallout, NOT an intrinsic fp16
value-head defect; the die-loud guard behaved as designed.

### Attempt 3 — diagnostic leg, labeled deviation `batch_size: 64` (+ `--min-buffer-size 64`): **ALL GATE SEMANTICS PASS**

| Gate criterion | Observed |
|---|---|
| All losses finite every step | step1 loss 9.549/policy 5.402/value 4.148; step2 9.137/4.989/4.148; step3 8.032/4.516/3.516 — all finite. (`grad_norm` step-1 = `inf` = documented GradScaler first-step overflow-backoff, repo-convention-accepted (`test_gnn_train_step.py` docstring); steps 2-3 finite: 18.42, 14.32. `fp16_scale` settled 32768.) |
| Watchdog alive | `selfplay_stall_watchdog_armed enabled=true timeout_sec=1800.0` at pool start; coordinator loop advanced games+steps throughout (watchdog check runs per loop iteration); no stall fire. |
| Ckpt written at/after step 3 | `checkpoint_saved step=3 checkpoint_path=.../checkpoint_00000003.pt` + `inference_only.pt` + `buffer_saved positions=147`. |
| Ckpt reloads clean | `Trainer.load_checkpoint` → gated loader resolved `metadata['encoding_name']=gnn_axis_v1`, `schema_version=1`; rebuilt `GnnNet`, step=3; `torch.allclose` over all 50 tensors TRUE; all tensors finite. |
| Clean shutdown, single PID | self-stopped via `--iterations 3` (`session_end final_step=3 games_played=5 buffer_size=147 elapsed_sec=125.8`, policy loss 5.40→4.52); zero orphaned python processes; GPU back to 2 MiB. No kill needed. |
| No shared-artifact writes | `checkpoints/best_model.pt` NOT recreated (scratch override held); buffer + best_model + ckpts all under scratch. |

**Observed but non-fatal, new finding F7:** at the step-3 eval round, ALL in-loop eval opponents
failed: `eval_opponent_failed opponent=random/sealbot/best error="'GnnNet' object has no attribute
'in_channels'"` ×3 → `evaluation_round_complete eval_games=0` (+ `value_fc2_weight_abs_max=NaN`
from the same dense-attr class — the `value_spread_canary_failed` sibling). Caught-per-opponent, so
the run survives, but **the in-loop eval/promotion pipeline is a silent no-op for graph runs** —
every eval round will complete with 0 games. Same `.in_channels` class as F5b/F8; run4 OQ-8's
"promotion gate through the GNN" remains genuinely open.

**Part-1 verdict: FAIL as-written** (F6 capacity on the pre-registered laptop host; the committed
yaml's inherited bs=256 cannot run the gate on the 4060). The gate's SEMANTIC criteria — finite
losses, watchdog, ckpt write/reload, clean stop, launch-blocker-free path — **all PASS** at the
clearly-labeled bs=64 deviation, so the training PATH is sound; what fails is a capacity knob on
the dev host. Disposition (coordinator's call): either sanction a laptop-smoke batch-size in the
gate spec, or re-run Part-1-as-written on the 5080 when the box frees.

---

## RE-RUN PART 2 — eval-boundary: **FAIL (new structural blocker F8; F3 CONFIRMED CLEARED)**

This time a REAL Part-1 graph checkpoint existed (`<scratch>/rerun/part1_ckpt/checkpoint_00000003.pt`,
minted by the actual production launch). Command (same driver contract as run 1):

```python
stage2_d5_eval(ckpt_path=<part1 ckpt>, book_r4=evalfair_r4_v2.json, book_r5=evalfair_r5_v2.json,
               out_dir=<scratch>, workers=1, n_boot=2000, expect_encoding="gnn_axis_v1", n_pairs=4)
```

**F3 CONFIRMED CLEARED:** `graph_ckpt_evalfair_book_r5 radius=5 overridden=False` fired at BOTH
resolution sites (stage2's book selection + `run_arm`'s internal re-resolution, agreeing), the r5
book loaded, the gated loader passed `expect_encoding=gnn_axis_v1`, and pair play STARTED.

**New blocker (F8, structural, one layer deeper):** first head move crashes —

```
File ".../hexo_rl/eval/gumbel_search_py.py", line 58, in _infer_and_expand
    policies, values = engine.infer_batch(leaves)
File ".../hexo_rl/selfplay/inference.py", line 82, in infer_batch
    if tensor.shape[1] != self.model.in_channels:
AttributeError: 'GnnNet' object has no attribute 'in_channels'
```

The offline EVALFAIR deploy head (`deploy_strength_eval.get_move` → `run_gumbel_on_board` →
`InferenceEngine.infer_batch`) encodes positions to DENSE planes and forwards through the dense
seam — **there is no graph branch in the offline searched-eval inference engine.** This is exactly
the gap `gnn_integration_scope.md` §C5's red-team pre-registered ("there is NO
searched-GNN-through-our-stack path today... the deploy-regime promotion gate cannot run on a
raw-policy adapter") and run4 OQ-8 left open. WP-3's graph seam landed in the SELF-PLAY
`InferenceServer._run_graph_loop` — the offline eval `InferenceEngine` is a different, dense-only
seam that was never graduated. The building blocks exist (`GnnBcBot`/`strix_v1_bot` board→graph
build + `forward_batch` + legal-set projection, all exercised by `test_gnn_seam_smoke.py`), but
wiring them into `hexo_rl/selfplay/inference.py::infer_batch` (or a `GnnV1Bot` deploy adapter) is
real dev work, out of a gate-run's scope.

**Pairs completed: 0/4. eff_n: raw n/a, deduped n/a — never reached (no games played).**

**Part-2 verdict: FAIL.** Blocker F8 (offline searched-eval inference seam dense-only —
`hexo_rl/selfplay/inference.py:82` + the whole `gumbel_search_py` head path). F3 is closed and its
fix verified live.

---

## RE-RUN PART 3 — formal suites: **PASS (all counts re-recorded post-fix)**

Fixes touched trainer/lifecycle/eval paths — full re-run, exact counts:

| Suite | Command | Result (re-run) | vs run 1 |
|---|---|---|---|
| Byte-parity oracle | `pytest tests/test_hexo_graph_parity.py -q` | **1 passed** (94.7s) | = |
| 18-assertion collate + 9 ADV | `pytest tests/selfplay/test_graph_collate.py -q` | **19 passed** | = |
| HEXG buffer + ADV-7 aug | `pytest tests/training/test_gnn_hexg_buffer.py -q` | **14 passed** | = |
| BC warm-start landed-verify (OQ-5) | `pytest tests/training/test_gnn_bc_warmstart.py -q` | **10 passed** | = |
| Corpus export ADV | `pytest tests/test_gnn_hexg_corpus_export.py -q` | **9 passed** | = |
| Graduated ADV-A/B/C/D | `pytest tests/selfplay/test_gnn_record_dispatch.py -q -m integration` | **2 passed** | = |
| Train-leg incl. fp16 CUDA leg | `pytest tests/training/test_gnn_train_step.py -q -m integration` | **10 passed** | = |
| WP-3 step-7 seam smoke | `pytest tests/selfplay/test_gnn_seam_smoke.py -q -m integration` | **1 passed** | = |
| Orchestrator/lifecycle/ckpt-load | `pytest tests/test_orchestrator_gnn_build.py tests/test_orchestrator_gnn_buffer.py tests/training/test_lifecycle_gnn_build.py tests/training/test_trainer_ckpt_load_gnn_resume.py -q -m "not slow"` | **18 passed** | = |
| Rust HEXG round-trip | `cargo test --package engine hexg --lib` | **32 passed** | = |
| **NEW fix suites (F1/F2/F3 + §6.3)** | `pytest tests/test_run4_gnn_launch_path.py tests/training/test_cuda_warmup_representation.py tests/test_s7_f3_graph_eval_book_radius.py -q` | **20 passed** (4+4+12; CUDA legs ran for real) | new |
| Collection sanity | `pytest --collect-only -q -m "not slow and not integration"` | **2770/2799 collected, 0 errors** | +20 (new test files) |

**Total: 136/136 across 12 suites, 0 failures. Part-3 verdict: PASS.**

---

## Re-run summary

| Part | Criterion | Observed | Verdict |
|---|---|---|---|
| 1 | finite losses / watchdog / ckpt write+reload / clean single-PID stop | F1+F2 cleared; NEW F5a/b (shared best_model.pt trap + `anchor.py:520` dense-only sync — hit then mitigated-at-scratch); as-written bs=256 → OOM at first train step (F6); at labeled bs=64: 3 steps finite, watchdog armed+live, ckpt@3 reloads 50/50 allclose, clean self-stop 125.8s, zero orphans. F7: in-loop eval opponents all fail on `.in_channels` (eval rounds = 0 games, non-fatal). | **FAIL as-written / semantics-PASS @bs=64** |
| 2 | 4 EVALFAIR pairs vs d5 end-to-end, honest eff_n | F3 cleared+verified live (`graph_ckpt_evalfair_book_r5 radius=5` both sites); play starts; first head move dies in dense-only `InferenceEngine.infer_batch` (F8). 0/4 pairs; eff_n raw n/a / deduped n/a. | **FAIL** |
| 3 | named suites run + recorded | 136/136 across 12 suites (incl. 20 new fix tests), 0 failures | **PASS** |

**OVERALL: FAIL** (gate not PASS per run4 §5/OQ-7). F1/F2/F3 closed and verified. **Open blockers,
in dependency order:** F5a (yaml: namespace `eval_pipeline.gating.best_model_path`, 1-line, run3
template), F5b (`anchor.py:520` representation-aware arch-sync), F6 (batch-size capacity on dev host
— knob/spec decision + OQ-2 5080 train-step memory rider), F7 (in-loop eval pipeline `.in_channels`
dense-only → graph eval rounds silently 0-game; OQ-8), F8 (offline searched-eval
`InferenceEngine.infer_batch` dense-only → EVALFAIR d5 battery cannot run on graph ckpts; the
Part-2 hard blocker). F5b/F7/F8 are one bug CLASS: dense-only `.in_channels` reads across the
eval/anchor surfaces (`anchor.py`, eval pipeline opponents, `selfplay/inference.py`) — a sweep for
`\.in_channels` consumers with a representation guard would close all three coherently.

## Re-run artifacts (scratch, not committed)

`<scratch>/s7_smoke/rerun/`: `_s7rerun_run4_gnn.yaml` (labeled overrides: scratch buffer/best_model
paths + bs=64 diagnostic), `part1_stdout_bs256_oom.log` (attempt-2 OOM + NaN-storm evidence),
`part1_stdout.log` (attempt-3 green run), `part1_ckpt/checkpoint_00000003.pt` (+`inference_only.pt`,
`checkpoint_log.json`), `part1_buffer.hexg` (147 positions), `best_model_s7rerun.pt`,
`part2_eval.py` + `part2_stdout.log` (F8 traceback + F3 event evidence). In-repo (gitignored/
untracked, deliberately left): `checkpoints/stale_artifacts_s7_20260715/best_model_proofrun_gnn_step0.pt`
(archived contaminant), `tests/fixtures/early_game_probe_gnn_axis_v1_v1.npz` (by-design
deterministic per-encoding probe fixture, auto-minted on first graph run — idempotent per its
module docstring; left untracked per do-not-git-add).
