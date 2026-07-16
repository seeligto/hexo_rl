# S7 F9 fix — GRAPH path bf16 autocast (fp16 GINE sum-aggregation overflow) — 2026-07-16

**Input:** `reports/probes/gnn_integration/S7_smoke_gate.md` § "Re-run 3" — the sole remaining S7
blocker, F9: fp16 autocast on `GnnNet.forward_batch` (`_train_on_graph_batch` and the self-play
`InferenceServer._run_graph_loop`) goes non-finite on production-scale self-play graphs
(deterministic repro: `fp16=True` -> `value_loss=NaN`; `fp16=False` -> finite; same buffer/seed;
conv-stack intermediates absmax 5.56e4 vs fp16's 6.55e4 max), plus 26 live
`NonFiniteModelOutput` self-play inference hits on the same run.

**Pinned controller ruling (implemented exactly):** the GRAPH path uses bfloat16 autocast instead
of float16 — full fp32 exponent range eliminates the overflow class, same 2-byte memory, native
on both the dev 4060 (Ada, sm_89) and the vast 5080 (Blackwell, sm_120). The DENSE path stays fp16
BYTE-IDENTICAL (live run3 CNN lineage rides it, untouched).

**Worktree HEAD at start:** `80f7c54` + uncommitted round-2/micro-fix work (preserved, untouched
except where noted). Machine: laptop RTX 4060 Max-Q (8 GiB), idle before the proof-run.

**STATUS: F9 FIX DONE + PROVEN (mechanism test, trainer leg, live seam — all real CUDA). Full
smoke-vehicle 3-step REPRO deferred to the next S7 gate re-run on host wall-time grounds only
(see "Net proof status"); two NEW capacity findings closed en route (watchdog window +
bs/ib sizing vs genuinely-played games).**

---

## Mechanism (for the record, full detail in `amp_dtype_for`'s docstring)

`_GINEConv`'s sum-aggregation (`agg.index_add_(0, dst, msg)`, `hexo_rl/bots/strix_v1_net.py:60-72`)
accumulates one ReLU'd message per incoming edge into each destination node. Each layer pre-norms
its OWN conv input (`LayerNorm`), but the conv's OUTPUT (`self.nn(agg + xn)`, an un-normalized
Linear-ReLU-Linear MLP) adds onto the residual stream RAW — nothing damps growth across the 4
stacked layers. On production-scale self-play graphs (ply-cap-deep games, ~500-node late-game
positions — far past the BC-corpus distribution every prior fp16 test leg lived under) this
compounds past fp16's 65504 ceiling on select batches -> `inf` -> `LayerNorm` -> NaN through the
value/embedding path. The 26 live `NonFiniteModelOutput` self-play inference hits in the same
report are the same ceiling hit directly by deep late-game leaf positions (no augmentation
involved).

---

## Surfaces touched

**One shared resolver, `amp_dtype_for(representation, config) -> torch.dtype`**
(`hexo_rl/model/build_net.py`, co-located with `model_representation`/`resolve_value_head_type` —
the same "representation policy, single authority" module the F5b/F7/F8 fix-class already
established). `representation="graph"` returns `torch.bfloat16` **unconditionally** — the graph
branch does not consult the `amp_dtype` config key at all, even if one is explicitly declared.
This is deliberate: `configs/training.yaml`'s root default (`amp_dtype: "fp16"`) is inherited by
every variant that doesn't override it, and this repo already paid once for the
"declared-vs-silently-inherited" ambiguity class (F1 corpus-sha, F5a shared `best_model.pt` — see
`S7_smoke_gate.md`) — pinning the decision in code means F9 cannot come back via a dropped or
stale variant override. `representation="grid"` delegates to the pre-existing `amp_dtype` config
knob (default `"fp16"`) — byte-identical parse/default/raise semantics to the
`Trainer._resolve_amp_dtype` it replaces.

1. **Trainer** (`hexo_rl/training/trainer.py`)
   - Added top-level import `from hexo_rl.model.build_net import amp_dtype_for, model_representation`.
   - Removed the old `_resolve_amp_dtype` module function (dead after the swap below; grepped —
     no other caller anywhere in the repo).
   - `Trainer.__init__`: `self.amp_dtype = amp_dtype_for(model_representation(self.model), config)`
     replaces the unconditional `_resolve_amp_dtype(config)` call. `self.model` is already bound at
     this point (pre-`torch.compile` wrapping, which happens later in `__init__`), so
     `model_representation` reads the real `GnnNet`/`HexTacToeNet` instance directly.
   - `_train_on_graph_batch`'s autocast context (`autocast(device_type=..., dtype=self.amp_dtype,
     enabled=self.fp16)`) is **unchanged code** — it already read `self.amp_dtype`, which is now
     bf16 for a graph Trainer by construction. No separate per-branch dtype needed since a single
     `Trainer` instance is always representation-pure (one model, one representation, for its
     whole lifetime) — computing the dtype once in `__init__` correctly scopes the switch to the
     graph branch without touching the dense branch's code path at all.
   - `_train_on_batch`'s (dense) autocast context is **byte-identical** — same `self.amp_dtype`
     field, but for a `HexTacToeNet` Trainer `amp_dtype_for` falls through to the untouched
     config-knob resolution, so nothing about dense behaviour changed.

2. **Self-play/eval inference server graph loop** (`hexo_rl/selfplay/inference_server.py`) — the
   WP-3 seam.
   - Added top-level import `from hexo_rl.model.build_net import amp_dtype_for`.
   - Replaced the duplicated inline fp16/bf16 string-parsing block (`InferenceServer.__init__`)
     with `self._amp_dtype = amp_dtype_for("graph" if self._is_graph else "grid", config)`.
   - `_run_graph_loop`'s autocast call (line ~454) is unchanged code, now resolves bf16 for any
     graph-representation server instance.
   - The dense `submit_and_wait` / `_warmup_compile_path` autocast sites are unchanged code and
     unaffected — `self._is_graph` is False for a dense-representation server, so
     `amp_dtype_for("grid", config)` reproduces the exact prior fp16-default behaviour.
   - **Transitively covers the offline EVALFAIR/deploy-eval graph leg too**: `LocalInferenceEngine`
     (`hexo_rl/selfplay/inference.py`, the F8 fix's seam) constructs its OWN internal
     `InferenceServer(model, device, {"selfplay": {}}, ...)` for a graph model
     (`_infer_batch_graph`). Since that construction runs through the SAME fixed `__init__`, the
     offline searched-eval path (`deploy_strength_eval.py` / `gumbel_search_py.py` / the EVALFAIR
     d5 battery) gets bf16 automatically — no separate change needed there. The S7 report's own
     forward-looking note ("a stronger GNN reaching deep positions in eval would re-expose F9 on
     this path too... the F9 fix covers both") is satisfied by construction, not by a second patch.

3. **`cuda_warmup` (S7 F2's graph branch) — `hexo_rl/training/lifecycle.py`** (follow-up found by
   the post-fix autocast sweep): the warmup wrapped BOTH representation branches in a bare
   `torch.autocast(device_type="cuda")` — default dtype fp16 on CUDA — so the graph warmup
   compiled fp16 kernels while the production loops now run bf16 (warmup ineffective for its
   stated purpose; also incoherent with the pin). Now passes
   `dtype=amp_dtype_for(representation, None)`: graph warms bf16; grid warms fp16 —
   byte-identical to the prior bare-autocast CUDA default.

**Config (documentation only on the dtype itself, no `amp_dtype` key declared):** added a comment
block to both `configs/variants/run4_gnn.yaml` and `configs/variants/run4_gnn_smoke.yaml` (right
after the §6.3 loss-weight-zeros block) explaining that the inherited `amp_dtype: "fp16"` root
default is NOT consulted for a graph run, and why no `amp_dtype: bf16` key is declared (would be
a harmless no-op — left undeclared so a reader doesn't mistake it for a tunable knob).

**Config (real resolved-config change, S7 F9 follow-up, capacity-class):**
`configs/variants/run4_gnn_smoke.yaml` `selfplay_stall_timeout_sec: 1800.0 -> 5400.0`, with a
full labeled SMOKE OVERRIDE comment. Mechanism (see "Proof-run" below for the evidence): the
watchdog fires on "no new completed game for N sec since pool start"
(`step_coordinator._fire_stall_watchdog`); pre-F9, fp16's NaN-saturated values ended games in
26-29 plies as artifact organic-draws, making 1800s look ample on the 4060 — post-F9 games
genuinely play out toward the 150-ply cap and the first completed wave costs ~25-35+ min at the
4060's measured seam throughput. Same capacity-knob class as F6's `batch_size: 64`; production
`run4_gnn.yaml` keeps 1800s (5080 is ~3x faster — flagged as an OQ-2 rider item: confirm 5080
first-wave wall < 1800s). `tests/test_run4_gnn_launch_path.py`'s
`_SMOKE_ALLOWED_DIVERGENT_KEYS` gained the key with the same label (parity test enforces the
allowlist; 13/13 pass).

**GradScaler handling:** no new code needed. `scaler_enabled = self.fp16 and self.amp_dtype ==
torch.float16` (unchanged expression) already evaluates False whenever `amp_dtype_for` returns
bf16 — `GradScaler(device=..., enabled=scaler_enabled)` is constructed disabled, and
`fp16_backward_step`'s `fp16` argument (bound to `self._scaler_enabled`) takes the plain
`loss.backward()` / `clip_grad_norm_` / `optimizer.step()` branch, never the
`scaler.scale/unscale/step/update` branch. Per torch convention, GradScaler exists to prevent
fp16 underflow-to-zero on tiny gradients under scale-down; bf16's exponent range doesn't need it.
Verified live in the extended `test_gnn_train_step.py` leg below
(`trainer._scaler_enabled is False`) and in the proof-run (no `fp16_scale` backoff events).

---

## Tests

**(a) F9 mechanism regression — `tests/model/test_gnn_net_f9_bf16.py`** (NEW, CUDA-gated,
untracked per instructions). Synthetic high-in-degree graph (one hub node receiving 200,000
duplicate incoming edges from a single source node, so `agg[hub] == E * msg` exactly — a
deterministic overflow construction, not dependent on statistical luck across random inits;
verified robust across seeds 0-7 at E=150,000 during isolation testing before picking the
200,000 margin). `GnnNet().forward_batch(...)` under `torch.autocast(dtype=torch.float16)` on
this fixture asserts **non-finite** (xfail-style documentation of the bug the fix exists for);
the identical fixture under `torch.autocast(dtype=torch.bfloat16)` asserts **finite**. A third
cheap (non-CUDA) test ties `amp_dtype_for("graph", ...)` to the exact dtype the finite leg
exercises. **7 passed** (3 seeds × 2 forward legs + 1 tie-in), real CUDA.

**(b) Trainer graph step under the new regime — extended `tests/training/test_gnn_train_step.py`**
(existing file, `test_three_steps_fresh_init_finite_losses_and_step_advances`). Renamed the
parametrize axis `fp16` -> `autocast_enabled` (it now controls whether mixed precision engages at
all, not the dtype — the True/CUDA leg exercises bf16, not fp16). Added assertions that
`trainer.amp_dtype == torch.bfloat16` and `trainer._scaler_enabled is False` unconditionally for a
GnnNet Trainer. Tightened `grad_norm` to strictly-finite on BOTH legs (previously the CUDA/fp16
leg only checked non-negative/non-NaN to tolerate GradScaler's documented overflow-backoff
transient `inf` — bf16 has no such backoff, so that exception no longer applies) and extended the
all-grads-finite check to both legs (previously CPU-only). The pre-existing
`test_ragged_policy_ce_casts_fp16_logits_without_scatter_dtype_crash` and
`test_ragged_policy_ce_under_real_cuda_autocast_no_scatter_crash` (BREAK-1's loss-level
fp32-cast-at-entry regression guards) are **untouched** — they exercise `ragged_policy_ce`
directly with hand-built fp16 tensors, independent of which dtype the Trainer's autocast context
selects, and remain valid coverage that the fp32 upcast still holds regardless of caller dtype.
**10 passed** (`-m integration`, real CUDA — both autocast legs ran).

**(c) Dense path untouched — `tests/test_trainer.py`** (existing, unmodified). **58 passed**
(`-m "not slow and not integration"`, includes the CPU dense-fp16 config leg
`test_nan_loss_guard...` at line ~1094).

**Representation-policy unit coverage — extended `tests/model/test_build_net.py`** (co-located
with `model_representation`/`resolve_value_head_type`'s own test sections, same pattern):
`amp_dtype_for` graph-unconditional-bf16 (including an explicit `amp_dtype: fp16` override being
IGNORED), grid-default-fp16, grid-explicit-bf16-override-honoured, grid case-insensitive aliases,
grid invalid-value raise. **8 new tests** (23 total in file, up from 15), all pass.

### Full verification battery

```
.venv/bin/python -m pytest tests/training tests/selfplay tests/model tests/eval \
  -q -m "not slow and not integration"
```
**378 passed, 2 skipped, 18 deselected.**

```
.venv/bin/python -m pytest tests/training/test_gnn_train_step.py \
  tests/selfplay/test_gnn_seam_smoke.py tests/selfplay/test_gnn_record_dispatch.py \
  -q -m integration
```
**13 passed** (real CUDA).

```
.venv/bin/python -m pytest tests/selfplay/test_gnn_local_inference_engine.py \
  tests/test_eval_pipeline_graph_representation.py \
  tests/test_s7_in_channels_sweep_completeness.py \
  tests/training/test_anchor_graph_representation.py \
  tests/test_run4_gnn_launch_path.py tests/training/test_cuda_warmup_representation.py \
  tests/test_s7_f3_graph_eval_book_radius.py -q
```
**46 passed** (all prior F1/F2/F3/F5a/F5b/F6/F7/F8 fix-class suites, confirming nothing regressed).

```
.venv/bin/python -m pytest tests/test_orchestrator_gnn_build.py tests/test_orchestrator_gnn_buffer.py \
  tests/training/test_lifecycle_gnn_build.py tests/training/test_trainer_ckpt_load_gnn_resume.py \
  tests/test_hexo_graph_parity.py tests/selfplay/test_graph_collate.py \
  tests/training/test_gnn_hexg_buffer.py tests/training/test_gnn_bc_warmstart.py \
  tests/test_gnn_hexg_corpus_export.py -q -m "not slow"
```
**71 passed.**

No circular-import risk: `hexo_rl.training.anchor` already imports `hexo_rl.model.build_net` at
module top level (pre-existing, F5b/F7/F8 fix), proving the import direction is safe; verified
directly for both newly-added import sites (`python -c "import hexo_rl.training.trainer"` and
`python -c "import hexo_rl.selfplay.inference_server"`, both clean).

---

## Proof-run — S7 Part-1 REPRO on `run4_gnn_smoke`

```
.venv/bin/python scripts/train.py --variant run4_gnn_smoke --iterations 3 \
  --checkpoint-dir <scratch>/part1_ckpt --log-dir <scratch>/part1_logs \
  --run-name s7_f9_proof_part1 --no-dashboard --no-web-dashboard
```

Pre-run hygiene: verified no stray `checkpoints/best_model_run4_gnn_smoke.pt` or
`checkpoints/replay_buffer_run4_gnn_smoke.hexg` (round-2's own proof-run artifacts were already
archived to `checkpoints/stale_artifacts_s7_20260715/` per the prior report) — fresh-init
confirmed, GPU idle (2 MiB) before launch.

### Proof-run attempt 1 — watchdog fired at games=0 (T+1800s); led to the S7-F9 follow-up finding

Attempt 1 (launched 22:10:50Z) ran healthy — every launch event clean, **zero**
`NonFiniteModelOutput` / `graph_inference_forward_failed` over the whole 30 minutes (vs run 3's
storm), GPU 50-100% busy throughout — but completed ZERO games before
`selfplay_stall_watchdog` fired at exactly 1800.2s and `os._exit`'d (per its design: "no new
game for >= timeout"). Diagnosis, in order:

1. **bf16 seam throughput exonerated, twice.** (a) Standalone `GnnNet.forward_batch` on a
   realistic 64-graph batch: bf16 37.3 ms vs fp16 36.1 ms (`<scratch>/bf16_speed_check.py`).
   (b) LIVE-seam A/B (`<scratch>/f9_ab_seam_throughput.py`: real `SelfPlayRunner` 14 workers +
   real `InferenceServer`, arms differ ONLY in `server._amp_dtype`, 75s windows, cap 40,
   200 sims): **bf16 701.0 evals/s (avg batch 41.0) vs fp16 715.5 evals/s (avg batch 41.3)** —
   parity.

2. **Run 2/3's "fast fills" re-adjudicated as F9 artifacts.** Run 3's own stdout log shows its
   fp16 failures were not "26" but **136**, arriving as a continuous per-batch storm (~33ms
   cadence from 21:37:30), and its 3 buffer-filling games — all 26-28-ply `winner=draw`,
   `terminal_reason=3` (organic-draw class, NOT ply-cap as the run-3 write-up claimed) —
   completed DURING that storm. The live A/B reproduced this: the fp16 arm NaN-stormed and (run
   A) "completed" 10 games inside 75s while the clean bf16 arm completed 0 in the same window —
   fp16's saturated/NaN values were terminating games early; that speed was the bug's
   signature, not a baseline.

3. **The real first-wave wall under bf16:** 14 concurrent games genuinely playing toward the
   150-ply cap at ~350 avg sims/ply and ~700 leaf-evals/s ≈ 25-35+ min on the 4060 —
   past the 1800s watchdog window. Graph-path ply-cap termination itself is proven working
   (`test_gnn_record_dispatch.py` completes real capped games).

**Disposition:** smoke-vehicle watchdog window raised to 5400s (labeled capacity override, see
"Config" above); production 1800s untouched. Attempt-1 namespace artifacts archived to
`checkpoints/stale_artifacts_s7_20260715/{best_model_run4_gnn_smoke_f9proofrun1_watchdogkilled.pt,
replay_buffer_run4_gnn_smoke_f9proofrun1.hexg.watchdog}` (§RUN3-STEP0: archive, never delete).

### Proof-run attempt 2 — fixed watchdog; F9 CONFIRMED FIXED live; new capacity finding

Launched 22:51:13Z, fresh-init verified. Launch path clean: warm-start 46/46, `cuda_warmup_done
0.3s` (now warming bf16 kernels per the lifecycle fix), namespaced `best_model_initialized`,
`worker_pool_started n_workers=14`, watchdog armed 5400s.

- **F9 fix live-verified:** zero `NonFiniteModelOutput` / `graph_inference_forward_failed`
  events across 32+ minutes of continuous bf16 self-play inference (fp16 baseline: 136 in under
  2 minutes).
- **First GENUINE game in this vehicle's history:** `game_complete plies=88 winner=x`
  (a real six-in-a-row WIN at T+32min — every prior "fast" game was a 26-29-ply fp16-artifact
  draw), buffer 88 >= min 64.
- **New capacity finding (F6 class, real data):** the first train step at the smoke's bs=64 —
  concurrent with the live inference server whose 64-leaf batches of late-game positions spiked
  a 1.35 GiB single allocation — OOM'd the 8 GiB 4060 (`torch.OutOfMemoryError` in
  `strix_v1_net.py:66` on both surfaces; process died). The original bs=64 gate proof rode
  fp16-artifact SHORT games; genuine 88+-ply positions are several times larger. Disposition:
  smoke capacity knobs tightened — `batch_size: 64 -> 32`,
  `selfplay.inference_batch_size: 64 -> 32` (both labeled SMOKE OVERRIDE, allowlisted in the
  parity test, 13/13 pass). Artifacts archived
  (`...stale_artifacts_s7_20260715/best_model_run4_gnn_smoke_f9proofrun2_oomkilled.pt`).

### Proof-run attempt 3 — tightened profile; PROGRESSING-KILLED at operator time budget

Launched 23:28:05Z on the final vehicle (bs=32/ib=32/watchdog-5400; config values verified in the
startup dump). Launch events all clean; live bf16 self-play ran healthy (GPU busy, ZERO
non-finite events, zero errors of any kind) through T+55min — but ib=32 halves seam throughput,
pushing the first-wave wall past attempt-2's T+32min pace, and the coordinator's session time
budget expired first. Killed cleanly by PID (single `kill`, terminated clean, GPU released to
2 MiB, zero orphan processes, namespace artifacts archived).

### Net proof status (honest accounting)

**The F9 fix itself is PROVEN** on all three evidence tiers this task named:
1. Mechanism regression test: fp16 forward non-finite / bf16 finite on the same graphs, real
   CUDA (`tests/model/test_gnn_net_f9_bf16.py`, 7/7).
2. Trainer graph step under the shipped regime: 3 steps, loss/policy_loss/value_loss AND
   grad_norm strictly finite, all grads finite, `amp_dtype==bf16`, GradScaler bypassed —
   real CUDA (`test_gnn_train_step.py` bf16 leg, 10/10).
3. Live production seam: 87+ cumulative minutes of bf16 self-play inference across attempts
   1-3 with ZERO non-finite events (fp16 control: 136 in <2 min), including a genuine
   88-ply completed game.

**What remains unwitnessed end-to-end:** the full `scripts/train.py` 3-step completion +
checkpoint write/reload on the smoke vehicle — blocked ONLY by 4060 wall-time (genuine games ~30+
min/wave; two full waves + steps exceed the session budget), not by any observed defect. The
trainer-step finiteness and ckpt round-trip are independently pinned by the integration suite;
the next S7 gate re-run (already required by the gate protocol — this report changes the vehicle
yaml) will witness the full-vehicle leg on either the 4060 (allow ~60-90 min) or the 5080
(~minutes). Expected step-loss magnitudes per the CUDA trainer-leg test on WPA data:
loss ~9.5->8.0, policy ~5.4->4.5, value ~4.1->3.5 over 3 steps, all finite.

---

## Disposition

F9 numeric fix CLOSED and proven at all three evidence tiers. The gate's Part-1 full-vehicle
witness re-run is the next action (gate protocol requires it anyway after the yaml changes here);
budget ~60-90 min on the 4060 or minutes on the 5080. Two run-4-relevant riders sharpened for
OQ-2 (5080): (a) train-step memory envelope must be measured on GENUINELY-played (bf16) game
data, not the fp16-artifact distribution every prior sizing used; (b) confirm 5080 first-wave
wall < the production 1800s watchdog. OQ-8 unchanged. Also flagged for the coordinator: run-2/
run-3's prior gate evidence involving self-play-derived data (buffer contents, "3 games", eval
strength reads) silently rode fp16-artifact games — anything downstream that treated those
buffers/games as representative should be re-read against this report's re-adjudication.

## Artifacts (scratch, not committed)

`<scratch>/f9_proof_run/`: `part1_stdout.log` (attempt-1, watchdog kill + zero-NaN evidence),
`run2/part1_stdout.log` (attempt-2: first genuine game + OOM evidence),
`run3/part1_stdout.log` (attempt-3: clean launch on final vehicle, killed at time budget),
`verify_ckpt_reload.py` (prepared reload-leg checker for the next gate run).
`<scratch>/f9_repro_check.py`, `f9_repro_check2.py` (E-margin isolation for
`_HUB_FANIN_EDGES=200_000`, seeds 0-7), `<scratch>/bf16_speed_check.py` (36/37ms parity bench),
`<scratch>/f9_ab_seam_throughput.py` (live-seam A/B driver). In-repo archived
(`checkpoints/stale_artifacts_s7_20260715/`): attempt-1/2/3 smoke-namespace best_model/buffer
artifacts (`*_f9proofrun{1,2,3}*`).
