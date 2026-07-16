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

---
---

# Re-run 3 (round-2 fixes) — 2026-07-15 (same day)

**Context:** round-2 fixes + review (PASS, `S7_round2_review.md`) + micro-fixes S-1/S-2/S-4 in the
worktree, uncommitted: F5a namespaced `best_model_path` (both variants, collision-tested); F5b/F7/F8
closed via the shared `model_representation()` primitive (`build_net.py`) — anchor arch-sync raises
`RepresentationMismatch` loud, in-loop eval loud-skips with a counter, `infer_batch` gained a graph
branch reusing the WP-3 seam; **F6 pinned ruling: `configs/variants/run4_gnn_smoke.yaml` is the S7
GATE vehicle** — a full labeled duplicate of `run4_gnn.yaml` whose ONLY deltas are capacity knobs
(`batch_size: 64`, `min_buffer_size: 64`, `buffer_capacity: 4096`) + smoke-namespaced
`buffer_persist_path`/`best_model_path`; production `run4_gnn.yaml` keeps bs=256 OPEN pending the
5080 OQ-2 train-step memory rider. Parity is test-pinned:
`tests/test_run4_gnn_launch_path.py::test_run4_gnn_smoke_resolved_config_parity_with_production`
(resolved-config equivalence on every non-labeled key incl. the §178 levers) +
`test_run4_gnn_and_smoke_variant_paths_never_collide` (smoke/prod artifact-path disjointness).

**RUN-3 OVERALL VERDICT: FAIL — Part-1 FAIL (new finding F9, fp16 numeric), Part-2 PASS, Part-3
PASS.** Every previously-named blocker (F1/F2/F3/F5a/F5b/F6-vehicle/F7/F8) is confirmed CLEARED and
observed working live. The gate now fails on exactly one thing: F9.

---

## RUN-3 PART 1 — training launch on `run4_gnn_smoke`: **FAIL (F9)**

**Command** (the pinned gate vehicle, run as-committed — no byte-copy, no CLI knob overrides beyond
output dirs; the yaml carries its own namespaced paths + min_buffer_size):

```
.venv/bin/python scripts/train.py --variant run4_gnn_smoke --iterations 3 \
  --checkpoint-dir <scratch>/run3/part1_ckpt --log-dir <scratch>/run3/part1_logs \
  --run-name s7_run3_part1 --no-dashboard --no-web-dashboard
```

**Launch path — every prior blocker observed CLEARED in one run:** warm-start 46/46 verified (F1
path clean, no corpus raise) → `cuda_warmup_done 0.3s` (F2) → `best_model_initialized
path=checkpoints/best_model_run4_gnn_smoke.pt` (F5a namespacing live — shared
`checkpoints/best_model.pt` untouched) → `worker_pool_started n_workers=14` →
`selfplay_stall_watchdog_armed enabled=true timeout_sec=1800` → live graph self-play (3 games,
buffer 81, GPU ~82%) → 3 train steps fired at bs=64, no OOM (F6 vehicle works) → step-3 eval round:
`evaluation_round_complete eval_games=0 eval_opponents_skipped=8` (F7 loud-skip counter live) →
`checkpoint_saved step=3` → `buffer_saved positions=81` → clean single-PID self-stop
(`session_end elapsed_sec=107.6`), zero orphans, GPU released.

**Gate criterion "all losses finite every step": VIOLATED —**

```
step 1: loss=NaN policy_loss=4.573 value_loss=NaN grad_norm=NaN fp16_scale=32768
step 2: loss=NaN policy_loss=4.470 value_loss=NaN grad_norm=NaN fp16_scale=16384
step 3: loss=NaN policy_loss=4.515 value_loss=NaN grad_norm=NaN fp16_scale=8192
```

plus 26× `graph_inference_forward_failed: NonFiniteModelOutput (probs finite=True, values
finite=False)` in the live inference server. Policy loss finite throughout (BREAK-1's
fp32-cast-at-entry holding); the VALUE/embedding path is what dies.

### F9 — fp16 GNN forward is non-finite on production-scale self-play graphs (NEW, the sole open Part-1 blocker)

Diagnosed to mechanism on run-3's actual saved buffer (all scripts + outputs in
`<scratch>/run3/f9_diag*.py`):

1. **Deterministic reproduction:** real `Trainer.train_step` on the run-3 buffer — `fp16=True` →
   `value_loss=NaN` every step; `fp16=False` → all finite (9.699/7.702/6.981). Same code, same
   data, same seed.
2. **Trigger = D6-augmented batches under fp16 autocast** (2×2 isolation): augment=False+fp16
   finite; augment=True+fp16 → embedding NaN (train AND eval mode — not a norm-mode effect);
   augment=True+fp32 → finite with absmax **7.16** (no magnitude problem at fp32).
3. **Mechanism:** GINE sum-aggregation intermediates sit AT the fp16 ceiling on this data — raw
   conv-stack absmax reaches **5.56e4 vs fp16 max 6.55e4** (batch of 64 self-play graphs = 818,796
   edges; run-3's three games all ran to the 150-move ply-cap → ~500-node late-game positions,
   far past the BC-corpus distribution mean 290/1294 where every existing fp16 test leg lives).
   Marginal batches tip over → `inf` → LayerNorm → NaN. The D6 rotation merely perturbs which
   batches tip; the 26 LIVE inference failures (no augmentation) are the same ceiling hit by deep
   late-game leaf positions directly.
4. **Re-adjudication of run 2:** run-2's clean bs=64 leg was DATA LUCK (5 shorter games, buffer
   under the ceiling), and run-2's "NaN storm is memory-pressure-coupled" attribution is
   RETRACTED — the bs=256 storm was F9 all along (deep-position overflow), coincident with, not
   caused by, the OOM. F9 is not introduced by round-2 (trainer/inference-server numeric path
   byte-identical since round-1); it was latent in the fp16 regime and only surfaces on
   ply-cap-deep self-play data.
5. **No corruption:** GradScaler skipped every poisoned optimizer step (fp16_scale halving 32768→
   8192 = the documented backoff). Verified: ckpt weights all finite AND rep+policy byte-equal
   46/46 to the warm-start init. The defense held; training simply cannot make progress.

**Why the suites are green while the gate fails:** every fp16 leg (`test_gnn_train_step.py`
fp16=True CUDA, seam smoke, WP-A parity) runs on WPA/BC-scale graphs below the ceiling. F9 needs
production-scale ply-cap games — exactly what only the REAL launch produces. The gate did its job.

**Fix directions (coordinator's call, not smuggled in here):** fp32-island the representation
(`autocast(enabled=False)` around the GINE stack — the WP-A bench assumed eager fp32-safe math),
or bf16 (`amp_dtype: bf16` — sm_89/sm_120 support it; 8-bit exponent removes the 65504 ceiling),
or mean-aggregation/activation-clamp (changes semantics — needs a parity re-check vs the +414
evidence net). `amp_dtype` is already a config knob (`_resolve_amp_dtype`, trainer.py) — a bf16
run4_gnn declaration may be the one-line candidate, gated on a rerun of this exact leg.

**Other gate criteria (for the record, all individually MET):** watchdog armed+live; ckpt written
at step 3; ckpt reloads clean (gated loader, `gnn_axis_v1`/schema-1, 50/50 allclose, all finite);
clean single-PID shutdown (self-stop, zero orphans, GPU released); no shared-artifact writes
(smoke-namespaced paths held; archived post-run to
`checkpoints/stale_artifacts_s7_20260715/{replay_buffer_run4_gnn_smoke_run3gate.hexg,
best_model_run4_gnn_smoke_run3gate.pt}` so the next gate run stays fresh-init).

**Part-1 verdict: FAIL** — losses-finite criterion violated by F9. Everything else about the
launch path is now clean.

---

## RUN-3 PART 2 — eval-boundary through `mantis_pull_eval` stage-2: **PASS**

**Command** (run-3's REAL Part-1 checkpoint — weights finite, = warm-start init since the scaler
skipped all steps):

```python
stage2_d5_eval(ckpt_path=<run3 part1 ckpt>, book_r4=evalfair_r4_v2.json,
               book_r5=evalfair_r5_v2.json, out_dir=<scratch>, workers=1,
               n_boot=2000, expect_encoding="gnn_axis_v1", n_pairs=4)
```

**Observed — END-TO-END COMPLETION:**

```
wr=0.0  pair_ci=[0.0,0.0]  n=8  eff_n=8  n_pairs=4  per_pair_scores=[0,0,0,0]
wall=156.2s  wall_per_move_head=2.14s  wall_per_move_sealbot=0.26s
radius=5  book_id=evalfair_r5_v2  deploy_matched=true  n_sims_effective=600 (ckpt-baked)
bad_pairs=0  censored_games=0  suffix_collisions=[]
```

| Gate criterion | Observed |
|---|---|
| Runs end-to-end | 4/4 pairs, 8/8 games completed vs SealBot d5 through the F8-fixed graph `infer_batch` (WP-3 seam reuse); `graph_ckpt_evalfair_book_r5 radius=5` fired at both resolution sites (F3); gated loader passed `expect_encoding=gnn_axis_v1` |
| Pairs complete | 4 pairs × color-swap, `bad_pairs=0`, `censored_games=0` |
| eff_n honest | **raw n=8, deduped eff_n=8** — `suffix_collisions=[]`: the §D-ARGMAX dedupe instrument RAN and found all 8 games distinct (4 distinct book openings × 2 colors) — eff_n is a measured count, not an assumed one |
| Strength (NOT the gate) | WR 0.0 — an effectively-step-0 net losing every d5 game, exactly as pre-registered ("a 3-step net will lose") |

F9 note: no non-finite failure fired during eval — head-vs-d5 games ended well short of ply-cap
depth, staying under the fp16 ceiling. A stronger GNN reaching deep positions in eval would
re-expose F9 on this path too (same forward); the F9 fix covers both.

**Part-2 verdict: PASS.**

---

## RUN-3 PART 3 — formal suites re-recorded (round-2 touched eval/anchor/inference/build_net): **PASS**

| Suite | Command | Result (run 3) | vs run 2 |
|---|---|---|---|
| Byte-parity oracle | `pytest tests/test_hexo_graph_parity.py -q` | **1 passed** (91.3s) | = |
| 18-assertion collate + 9 ADV; HEXG buffer + ADV-7; BC warm-start (OQ-5); corpus export ADV | `pytest tests/selfplay/test_graph_collate.py tests/training/test_gnn_hexg_buffer.py tests/training/test_gnn_bc_warmstart.py tests/test_gnn_hexg_corpus_export.py -q` | **52 passed** (19+14+10+9) | = |
| Graduated ADV-A/B/C/D + train-leg (fp16 CUDA leg) + seam smoke | `pytest tests/selfplay/test_gnn_record_dispatch.py tests/training/test_gnn_train_step.py tests/selfplay/test_gnn_seam_smoke.py -q -m integration` | **13 passed** (2+10+1) | = |
| Orchestrator/lifecycle/ckpt-load | `pytest tests/test_orchestrator_gnn_build.py tests/test_orchestrator_gnn_buffer.py tests/training/test_lifecycle_gnn_build.py tests/training/test_trainer_ckpt_load_gnn_resume.py -q -m "not slow"` | **18 passed** | = |
| Rust HEXG round-trip | `cargo test --package engine hexg --lib` | **32 passed** | = |
| Round-1 fix suites (F1/F2/F3) — launch_path grew: +parity pin (S-1) + path-collision + namespacing tests | `pytest tests/test_run4_gnn_launch_path.py tests/training/test_cuda_warmup_representation.py tests/test_s7_f3_graph_eval_book_radius.py -q` | **29 passed** (13+4+12) | +9 |
| **NEW round-2 suites** (F5b/F7/F8 + sweep-completeness + build_net) | `pytest tests/selfplay/test_gnn_local_inference_engine.py tests/test_eval_pipeline_graph_representation.py tests/test_s7_in_channels_sweep_completeness.py tests/training/test_anchor_graph_representation.py tests/model/test_build_net.py -q` | **35 passed** | new |
| Collection sanity | `pytest --collect-only -q -m "not slow and not integration"` | **2799/2828 collected, 0 errors** | +29 |

**Total: 180/180 across 13 suites, 0 failures. Part-3 verdict: PASS.**

---

## Run-3 summary

| Part | Criterion | Observed | Verdict |
|---|---|---|---|
| 1 | finite losses / watchdog / ckpt write+reload / clean single-PID stop | ALL prior blockers cleared live (F1/F2/F5a/F5b-path/F6-vehicle/F7-counter); watchdog armed, ckpt@3 reloads 50/50 finite, clean self-stop 107.6s, zero orphans — but loss/value_loss/grad_norm **NaN all 3 steps** + 26 live inference non-finite failures = **F9** (fp16 ceiling on production-scale graphs; mechanism pinned, fp32 finite, GradScaler prevented corruption) | **FAIL (F9)** |
| 2 | 4 EVALFAIR pairs vs d5 end-to-end, honest eff_n | 4/4 pairs, 8/8 games, wall 156s, WR 0.0 (strength not the gate); **eff_n raw=8, deduped=8, suffix_collisions=[]**; F3+F8 fixes carried it end-to-end | **PASS** |
| 3 | named suites re-run + recorded | 180/180 across 13 suites, 0 failures; collection 2799/2828, 0 errors | **PASS** |

**RUN-3 OVERALL: FAIL — single remaining blocker F9** (fp16 non-finite GNN value/embedding forward
on production-scale self-play graphs; deterministic repro + mechanism + fix candidates recorded
above). F1/F2/F3/F5a/F5b/F6/F7/F8: ALL CLOSED and live-verified. The gate is one numeric-regime
decision away from PASS: fix/declare the precision regime (bf16 / fp32-island / clamp), rerun
Part 1 on `run4_gnn_smoke`, and re-read Part 2 unchanged.

## Run-3 artifacts (scratch, not committed)

`<scratch>/s7_smoke/run3/`: `part1_stdout.log` (NaN steps + all cleared-blocker events),
`part1_ckpt/checkpoint_00000003.pt`, `f9_diag{,2,3,4,5}.py` (the F9 isolation ladder: fp16-vs-fp32 ×
augment×mode × per-layer absmax), `part2_eval.py` + `part2_stdout.log` (full RESULT_JSON).
In-repo archived: `checkpoints/stale_artifacts_s7_20260715/{replay_buffer_run4_gnn_smoke_run3gate.hexg,
best_model_run4_gnn_smoke_run3gate.pt}` (run-3's smoke-namespace outputs, archived so the next gate
run is fresh-init).

---
---

# Re-run 4 (F9 bf16) — 2026-07-16

**Context:** F9 fixed per `S7_f9_bf16_fix.md` (read first, per instruction): graph path bf16
autocast via the code-pinned `amp_dtype_for` resolver (trainer + WP-3 inference seam + cuda_warmup;
graph branch ignores the `amp_dtype` config key by design), GradScaler auto-off on bf16, throughput
parity proven (bf16 701 vs fp16 715 evals/s live-seam A/B). Re-adjudications accepted into this
run's expectations: prior runs' fast 26-29-ply games were fp16-NaN artifacts (run-3's "3 ply-cap
games" were actually organic-draw-class artifact games; its failure count 136 not 26); genuine bf16
games cost ~30+ min/wave on the 4060. Smoke vehicle re-tightened by that report:
`batch_size: 32`, `selfplay.inference_batch_size: 32`, `selfplay_stall_timeout_sec: 5400`
(all labeled, allowlisted in the parity test).

**RUN-4 OVERALL VERDICT: FAIL — Part-1 FAIL-as-pinned (capacity, F6-residual; ALL semantic gate
criteria incl. the F9 zero-non-finite criterion PASS on a labeled bs=16 diagnostic leg), Part-2
PASS, Part-3 PASS. F9 is CONFIRMED FIXED — the sole remaining gate blocker is one smoke-vehicle
capacity knob.**

---

## RUN-4 PART 1 — training launch on `run4_gnn_smoke`: **FAIL as-pinned; ALL gate semantics (incl. F9) PASS at labeled bs=16 leg**

### Attempt 1 — as-pinned vehicle (bs=32/ib=32/watchdog-5400): capacity FAIL at train-step 3

```
.venv/bin/python scripts/train.py --variant run4_gnn_smoke --iterations 3 \
  --checkpoint-dir <scratch>/run4/part1_ckpt --log-dir <scratch>/run4/part1_logs \
  --run-name s7_run4_part1 --no-dashboard --no-web-dashboard
```

Pre-run hygiene: no stray smoke-namespace artifacts, GPU idle. Launch clean (warm-start 46/46,
bf16 `cuda_warmup_done 0.3s`, namespaced anchor, pool 14, watchdog armed 5400s).

**F9 evidence (the criterion this re-run exists for): CLEAN.** Zero `NonFiniteModelOutput` across
the entire ~74-min run of continuous bf16 self-play inference. Games now GENUINE: 2×
`game_complete plies=150 winner=draw` (real ply-cap draws at T+70/74min — matching the fix
report's re-adjudicated wall-time, and unlike every fp16-artifact 26-29-ply game before). Both
fired train steps FINITE: step1 loss 5.255 (policy 5.255, value 0.000, grad 18.38), step2 8.877
(8.877, 0.000, 31.99) — `value_loss=0.0` is CORRECT here, not suspicious: both buffer games were
ply-cap draws → `value_valid=0` on every row (INV26 §178 lever) → fully-masked value batch;
`fp16_scale=1.0` static = GradScaler bypassed on bf16 as designed.

**Capacity FAIL (F6-residual):** the 3rd train step OOM'd in `loss.backward()`
(`torch.OutOfMemoryError: tried to allocate 732 MiB ... 2.68 GiB reserved but unallocated`
— fragmentation), preceded by 5 inference-server `OutOfMemoryError` events (1.0-1.2 GiB batch
allocations; logged as `graph_inference_forward_failed error_type=OutOfMemoryError` — OOM-class,
NOT NaN-class; the die-loud path worked). Root cause: GENUINE-game graphs are several times
larger than every sizing basis — training batches carried **n_legal_nodes 47825/45340 at 32
graphs ≈ 1494/graph** vs run-3's fp16-artifact ≈525/graph. The fix report's own bs 64→32
halving was calibrated against one 88-ply game; two concurrent 150-ply-cap games + a 14-game
live wave exceed 8 GiB at bs=32. Hard crash pre-checkpoint (no ckpt, no close-out). Artifacts
archived (`*_run4gate_attempt1*`).

### Attempt 2 — labeled diagnostic leg (`batch_size: 16` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`): **ALL GATE CRITERIA PASS**

Byte-copy variant `_s7run4diag_run4_gnn_smoke.yaml` (sole yaml delta: bs 32→16, labeled; archived
at `<scratch>/run4/`) + the allocator env var the OOM traceback itself recommends (targets the
observed 2.68 GiB reserved-unallocated fragmentation). Honest caveat: TWO knobs changed at once —
bs=16-without-allocator-conf was not separately measured (wall-time economics: ~70 min/attempt).

| Gate criterion | Observed |
|---|---|
| 3 finite steps | step1 loss **9.310** (policy 5.206, value 4.104, grad 22.17); step2 **10.738** (7.185, 3.553, 46.45); step3 **8.376** (5.137, 3.239, 17.10) — all strictly finite, `fp16_scale=1.0` static (scaler off on bf16). value_loss REAL this time: both games were genuine six-in-a-row WINS (107-ply `winner=o`, 140-ply `winner=x`) → `value_valid=1` rows live. n_legal_nodes 23134/18695/29368 @16 graphs — same ~1450-1835/graph genuine-data density that killed bs=32. |
| Zero NonFiniteModelOutput | **0** NaN-class events AND **0** OOM events across the full 64-min run. Combined with attempt 1: ~135 min cumulative bf16 self-play, 0 non-finite. F9 criterion MET. |
| Watchdog alive | armed 5400s at pool start; loop advanced games+steps throughout; no fire. |
| Ckpt write + clean reload | `checkpoint_saved step=3` + `buffer_saved positions=247`; reload via gated loader: `gnn_axis_v1`/schema-1, `GnnNet` step=3, 50/50 allclose, all finite; **42/46 rep+policy tensors CHANGED from warm-start init** (real bf16 optimizer steps landed — the 4 static ones are zero-grad params; contrast run-3 where the scaler skipped everything). |
| Clean single-PID shutdown | self-stopped via `--iterations 3` (`session_end elapsed_sec=3821.8 games_played=2 buffer_size=247`), zero orphaned processes, GPU released to 2 MiB. No kill needed. |
| F7 counter | `evaluation_round_complete eval_games=0 eval_opponents_skipped=8` — loud-skip live again. |

**Part-1 verdict: FAIL as-pinned** (the committed smoke vehicle's bs=32 cannot complete 3 steps on
this host with genuine-game data) — with every SEMANTIC criterion, including the F9
zero-non-finite criterion, **witnessed end-to-end through the production entrypoint** at the
labeled bs=16 leg. This is the F6 lineage's third capacity iteration (256→64→32→16), each
invalidated by the data distribution growing more genuine; the fix report's own OQ-2 rider
("train-step memory envelope must be measured on GENUINELY-played game data") is the standing
answer — disposition (coordinator): pin the smoke yaml at bs=16 (+ consider `expandable_segments`
or de-confounding it), or witness the as-pinned leg on the 5080.

---

## RUN-4 PART 2 — eval-boundary through `mantis_pull_eval` stage-2: **PASS**

Run-4's REAL Part-1 checkpoint (3 genuine bf16 gradient steps baked in), same driver contract:

```
wr=0.0  pair_ci=[0.0,0.0]  n=8  eff_n=8  n_pairs=4  per_pair_scores=[0,0,0,0]
wall=153.1s  wall_per_move_head=2.13s  wall_per_move_sealbot=0.30s
radius=5  book_id=evalfair_r5_v2  deploy_matched=true  n_sims=600 (ckpt-baked)
bad_pairs=0  censored_games=0  suffix_collisions=[]
```

- End-to-end: 4/4 pairs, 8/8 games vs SealBot d5; `graph_ckpt_evalfair_book_r5 radius=5` fired at
  both resolution sites; gated loader passed `expect_encoding=gnn_axis_v1`.
- **eff_n honest: raw n=8, deduped eff_n=8**, `suffix_collisions=[]` — dedupe instrument ran, all
  8 games distinct.
- WR 0.0 as pre-registered (3-step net; strength NOT the gate).
- Zero non-finite / zero OOM in the eval log — the bf16 fix transitively covers the offline
  searched-eval seam (`LocalInferenceEngine` → `InferenceServer.__init__`), as the fix report
  claimed by construction; now observed live.

**Part-2 verdict: PASS.**

---

## RUN-4 PART 3 — formal suites re-recorded (incl. new F9 suites): **PASS**

| Suite | Command | Result (run 4) | vs run 3 |
|---|---|---|---|
| **NEW F9 mechanism regression** (fp16 non-finite / bf16 finite on hub-fanin overflow fixture + `amp_dtype_for` tie-in, real CUDA) | `pytest tests/model/test_gnn_net_f9_bf16.py -q` | **7 passed** | new |
| Byte-parity oracle | `pytest tests/test_hexo_graph_parity.py -q` | **1 passed** (87.9s) | = |
| 18-assertion collate + 9 ADV; HEXG buffer + ADV-7; BC warm-start (OQ-5); corpus export ADV | `pytest tests/selfplay/test_graph_collate.py tests/training/test_gnn_hexg_buffer.py tests/training/test_gnn_bc_warmstart.py tests/test_gnn_hexg_corpus_export.py -q` | **52 passed** | = |
| Graduated ADV-A/B/C/D + train-leg (bf16 CUDA leg, renamed axis, grad-norm strictly-finite BOTH legs) + seam smoke | `pytest tests/selfplay/test_gnn_record_dispatch.py tests/training/test_gnn_train_step.py tests/selfplay/test_gnn_seam_smoke.py -q -m integration` | **13 passed** | = |
| Orchestrator/lifecycle/ckpt-load | `pytest tests/test_orchestrator_gnn_build.py tests/test_orchestrator_gnn_buffer.py tests/training/test_lifecycle_gnn_build.py tests/training/test_trainer_ckpt_load_gnn_resume.py -q -m "not slow"` | **18 passed** | = |
| Rust HEXG round-trip | `cargo test --package engine hexg --lib` | **32 passed** | = |
| Fix-class suites, rounds 1+2 combined (launch_path incl. smoke-parity/allowlist + cuda_warmup + F3 + local-inference + eval-pipeline-graph + in_channels-sweep + anchor-graph + build_net incl. 8 new `amp_dtype_for` tests) | `pytest tests/test_run4_gnn_launch_path.py tests/training/test_cuda_warmup_representation.py tests/test_s7_f3_graph_eval_book_radius.py tests/selfplay/test_gnn_local_inference_engine.py tests/test_eval_pipeline_graph_representation.py tests/test_s7_in_channels_sweep_completeness.py tests/training/test_anchor_graph_representation.py tests/model/test_build_net.py -q` | **69 passed** | +23 (F9 additions) |
| Dense path untouched | `pytest tests/test_trainer.py -q -m "not slow and not integration"` | **58 passed** | new to table |
| Collection sanity | `pytest --collect-only -q -m "not slow and not integration"` | **2811/2840 collected, 0 errors** | +12 |

**Total: 250/250 across 14 suites, 0 failures. Part-3 verdict: PASS.**

---

## Run-4 summary

| Part | Criterion | Observed | Verdict |
|---|---|---|---|
| 1 | 3 finite steps / watchdog / ckpt write+reload / **zero NonFiniteModelOutput** / clean single-PID stop | **F9 FIXED — 0 non-finite events over ~135 min cumulative bf16 self-play + 3 strictly-finite train steps + genuine games (107/140/150/150-ply)**; as-pinned bs=32 OOM'd backward at step 3 on genuine-game graphs (~1494 legal nodes/graph, F6-residual); labeled bs=16+expandable_segments leg met EVERY criterion end-to-end (ckpt@3 reloads 50/50, 42/46 tensors moved, clean self-stop 3821.8s, zero orphans) | **FAIL as-pinned / ALL semantics PASS @bs=16** |
| 2 | 4 EVALFAIR pairs vs d5 end-to-end, honest eff_n | 4/4 pairs, 8/8 games, wall 153s, WR 0.0; **eff_n raw=8, deduped=8, suffix_collisions=[]**; zero non-finite in eval (bf16 transitively live) | **PASS** |
| 3 | named suites + new F9 suites re-recorded | 250/250 across 14 suites, 0 failures | **PASS** |

**RUN-4 OVERALL: FAIL — sole remaining blocker = F6-residual capacity of the pinned smoke vehicle**
(bs=32 → OOM at step 3 on genuine-game data; bs=16 + `expandable_segments:True` proven end-to-end
on this host, two-knob confound flagged). **F9 is CLOSED** — zero non-finite events across both
attempts, mechanism suite green, weights actually train. Disposition options: (a) pin smoke yaml
`batch_size: 16` (fourth F6 iteration — this time sized against GENUINE-game data, which no prior
iteration was) ± adopt/de-confound the allocator conf, re-run Part-1 once more (~70 min on 4060);
(b) witness the as-pinned bs=32 leg on the 5080 (16 GiB — minutes, and doubles as the OQ-2
train-step memory rider). Part 2 + Part 3 need no re-run either way unless the yaml change touches
them (it does not).

## Run-4 artifacts (scratch, not committed)

`<scratch>/s7_smoke/run4/`: `part1_stdout_attempt1_bs32_oom.log` (as-pinned leg: F9-clean evidence
+ backward-OOM traceback + genuine 150-ply games), `part1_stdout.log` (bs=16 leg, full green run),
`_s7run4diag_run4_gnn_smoke.yaml` (the labeled one-knob diagnostic variant, removed from repo
tree), `part1_ckpt/checkpoint_00000003.pt` (+`inference_only.pt`), `part2_eval.py` +
`part2_stdout.log` (full RESULT_JSON). In-repo archived
(`checkpoints/stale_artifacts_s7_20260715/`): `*_run4gate_attempt1*` (bs=32 leg) +
`*_run4gate_bs16*` (bs=16 leg) smoke-namespace outputs — next gate run stays fresh-init.

---
---

# Re-run 5 (part-1 as-pinned witness) — 2026-07-16

**Controller ruling executed:** smoke vehicle amended in place — `configs/variants/run4_gnn_smoke.yaml`
`batch_size: 32 -> 16` + `selfplay.inference_batch_size: 32 -> 16` (labeled SMOKE OVERRIDE comments
extended: sized against GENUINE-game graphs ~1494 legal-nodes/graph vs the 525-node fp16-artifact
basis; production values untouched; 5080 OQ-2 rider owns the real capacity knee).
`tests/test_run4_gnn_launch_path.py` value asserts updated 32→16 (allowlist already covered both
keys) — **13/13 pass** post-amendment.

## Part 1 — as-pinned witness on the amended `run4_gnn_smoke`: **PASS (no env var needed)**

**DE-CONFOUND RESOLVED:** run exactly as pinned, explicitly WITHOUT `PYTORCH_CUDA_ALLOC_CONF`
(verified absent from the launch environment) — completed clean. **bs=16 alone suffices; the
allocator env var is NOT part of the smoke launch protocol.** Run-4's two-knob confound is closed:
the batch size was the operative fix, `expandable_segments` unnecessary.

```
.venv/bin/python scripts/train.py --variant run4_gnn_smoke --iterations 3 \
  --checkpoint-dir <scratch>/run5/part1_ckpt --log-dir <scratch>/run5/part1_logs \
  --run-name s7_run5_part1 --no-dashboard --no-web-dashboard
```

| Gate criterion | Observed |
|---|---|
| 3 finite steps | step1 loss **5.466** (grad 10.00), step2 **7.021** (grad 44.45), step3 **5.650** (grad 18.42) — all strictly finite; `fp16_scale=1.0` static (bf16, scaler off). `value_loss=0.0` all three steps is CORRECT for this draw: both games were genuine 150-ply ply-cap draws → `value_valid=0` fully-masked value batches (INV26 §178 lever). Non-zero value-loss coverage at the SAME batch_size=16 stands from run-4 attempt 2 (value 4.104/3.553/3.239 on genuine-win data — same code, same bs). |
| Zero NonFiniteModelOutput | **0** NaN-class AND **0** OOM-class events across the full run — cumulative bf16 evidence now ~200+ min of production self-play with zero non-finite. |
| Watchdog alive | armed 5400s, loop advanced throughout, never fired. |
| Ckpt write + clean reload | `checkpoint_saved step=3`; reload via gated loader: `gnn_axis_v1`/schema-1, `GnnNet` step=3, allclose over all tensors, all finite. |
| Clean single-PID shutdown | self-stopped via `--iterations 3`: `session_end final_step=3 games_played=2 buffer_size=300 elapsed_sec=4125.0` (**wall 68.8 min** — inside the 70-90 min budget); zero orphaned processes; GPU released to 2 MiB. |
| Hygiene | fresh-init verified pre-launch; smoke-namespace outputs archived post-run (`*_run5gate*`); F7 loud-skip counter fired again (`eval_opponents_skipped=8`). |

## Parts 2 + 3 — STAND from run 4 (explicit, per ruling)

No re-run needed or performed: the yaml deltas (bs/ib 32→16) touch only Part-1 capacity knobs —
Part 2 consumed a bs=16-trained checkpoint already (run-4 attempt 2's), and no Part-3 suite reads
the smoke yaml's capacity values except `test_run4_gnn_launch_path.py`, which was re-run
post-amendment (13/13). Standing results: **Part 2 PASS** (4/4 pairs, 8/8 games, eff_n raw=8
deduped=8, `suffix_collisions=[]`); **Part 3 PASS** (250/250 across 14 suites).

---

# S7 GATE — FINAL OVERALL VERDICT: **PASS**

| Part | Final status | Witnessed |
|---|---|---|
| 1 — 3-step training launch (production entrypoint, as-pinned `run4_gnn_smoke`) | **PASS** | run 5 (this run): finite steps, watchdog, ckpt write+reload, zero non-finite, zero OOM, clean shutdown, no env-var dependency |
| 2 — eval-boundary (`mantis_pull_eval` stage-2, 4 EVALFAIR pairs vs d5, honest eff_n) | **PASS** | run 4: 8/8 games end-to-end, eff_n raw=8/deduped=8, dedupe instrument ran |
| 3 — formal ragged/adversarial/parity suites | **PASS** | run 4: 250/250 across 14 suites (+13/13 launch-path re-run post-amendment in run 5) |

Five gate runs, nine blockers found and closed en route (F1 corpus-mix inherit, F2 dense-only
cuda_warmup, F3 graph book-radius, F5a shared best_model namespace, F5b/F7/F8 dense-only
`.in_channels` class, F6 capacity ladder 256→64→32→16 — final iteration sized against genuine-game
data, F9 fp16→bf16 numeric regime). Standing riders for run4 LAUNCH (not gate blockers): OQ-2 5080
train-step memory envelope + first-wave wall vs the production 1800s watchdog, both to be measured
on genuinely-played bf16 data; OQ-8 promotion gate through the GNN (in-loop eval loud-skips by
design — `eval_opponents_skipped` counter is the visibility hook).

## Run-5 artifacts (scratch, not committed)

`<scratch>/s7_smoke/run5/`: `part1_stdout.log` (full green as-pinned run), `part1_ckpt/
checkpoint_00000003.pt`. In-repo archived (`checkpoints/stale_artifacts_s7_20260715/`):
`*_run5gate*` smoke-namespace outputs. Repo-tree deltas this run: `configs/variants/
run4_gnn_smoke.yaml` (bs/ib 16 + comments), `tests/test_run4_gnn_launch_path.py` (asserts 16).
