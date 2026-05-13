<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §116 — D-ladder investigation: curr_10k catastrophic forgetting — 2026-04-23

### Trigger

Eval vs bootstrap (post-§114 sustained run, `checkpoint_00010000.pt`) reported
curr_10k losing badly. User asked for D1–D5 discriminator ladder to decide
between policy regression, value regression, or both.

### Verdict — P-regressed (distributional), V intact on corpus

Report: `reports/investigations/diag_D_20260423/VERDICT.md`

| Diag | N | Metric | Threshold | Measured | Verdict |
|---|---|---|---|---|---|
| Control Zero | 50 | boot-vs-boot WR | ~50% ±14 | 54.0% | harness clean |
| D1 (policy argmax) | 100 | curr WR ex-draws | P ≤ 10% | **6.0%** | **P-regressed** |
| D2 (curr@800 vs boot@128) | 50 | curr WR | deep ≤ 30% | **4.0%** | **deep regression** |
| D3 (KL on corpus) | 500 | mean nats | close < 0.3 | 0.181 | policies close on corpus |
| D4 (V MSE on corpus) | 500 | ratio | matched ≤ 1.0 | 1.027 | V matched |

Reconciliation: D3/D4 probe late-game corpus positions; D1/D2 probe real-game
trajectories including openings. Mismatch means the regression is
**distributional, not global**.

### Smoking gun — early-game policy collapsed to near-uniform

D3-extra early-game synthetic probe (30 samples per ply):

- Empty board (ply 0): curr argmax agreement with boot = 0%; curr H=2.87 vs boot H=3.16.
- Ply 2–7: curr entropy 5.47–5.70 nats (log(362) ≈ 5.89 = uniform), top-1 mass 0.009–0.022.
- Bootstrap retains H=3.4–4.0 with top-1 mass 0.13–0.24 on the same positions.

Curr has effectively forgotten how to open. On ply 2–7 positions the policy
head is indistinguishable from uniform over the 362-action space.

### Root cause hypothesis

Replay buffer during sustained run under-covered early-game positions. Policy
head drifted toward uniform on ply < 15 as training distribution concentrated
in mid/late-game. Once openings became random, self-play games entered a
degenerate regime where curr lost by ply 15–20, reinforcing late-game training
but never correcting the opening policy.

Circumstantial:
- Bootstrap corpus capped POSITION_END=150 but has no lower cap — early-game
  should be represented but replay buffer composition during sustained run was
  not audited.
- MCTS pool overflow (`next_free=199999`) logged during D2 — uniform prior
  produces tree fan-out without convergence.

### Actions

**Immediate:** revert live checkpoint to `bootstrap_model.pt`. Do not promote
any checkpoint from this sustained run.

**Follow-up (ordered by cost):**

1. Re-run D1 on ckpt_5000/7000/9000 — locate forgetting onset step.
2. Audit replay buffer composition by ply / phase during sustained run.
3. Verify Dirichlet noise is enabled at root (§112 port) in the sustained-run config.
4. Defer D5 per-head ablations — pathology is distributional, not head-specific.

### Artifacts

- `reports/investigations/diag_D_20260423/VERDICT.md`
- Scripts: `scripts/diag_games.py`, `scripts/diag_forward.py`, `scripts/diag_argmax_agreement.py`, `scripts/diag_early_game.py`

---

## §116 — torch.compile Retry: GO on reduce-overhead (2026-04-23)

**Branch:** `probe/torch-compile-retry-20260423`
**Status:** Probe complete. Landing pending AC-power bench gate.

### Summary

Both §32 blockers are resolved in Python 3.14.2 + PyTorch 2.11.0+cu130:

| Blocker (§32) | Status |
|---|---|
| TLS crash on Py3.14 (§30) | **Gone** — PT2.11 fixes Py3.14 CUDA thread-local storage |
| 27 GB Triton JIT spike on first forward | **Gone** — 59.5 MB peak; 6.4 s compile |

All three modes work. **`reduce-overhead` is the landing target.**

### Measurements (battery — ratios valid, absolutes depressed)

| Metric | Eager | default | reduce-overhead | max-autotune-no-cudagraphs |
|---|---|---|---|---|
| Throughput batch=64 (pos/s) | 2,529 | 3,665 | **3,788** | 3,744 |
| Throughput speedup vs eager | 1.00× | 1.45× | **1.50×** | 1.48× |
| Latency batch=1 (mean ms) | 3.553 | 2.844 | **1.897** | 3.007 |
| Latency speedup vs eager | 1.00× | 1.25× | **1.87×** | 1.18× |
| Compile time | — | 11.8 s | **6.4 s** | 29.9 s |
| Graph breaks | 0 | 0 | **0** | 0 |

`reduce-overhead` latency (1.897 ms) matches the AC-power baseline (1.84 ms)
within battery variance — confirms it was the mode used in the existing baseline.

### Technical notes

- `triton.cudagraphs = False` — PT2.11 does not activate CUDA graph replay on
  RTX 4060 Laptop (20 SMs). Gains come from Triton kernel fusion across
  GroupNorm + ReLU + SE + residual add.
- `Not enough SMs to use max_autotune_gemm mode` — informational; does not
  affect correctness or block compile.
- Divergence vs eager: policy abs_max=1.53e-3, value abs_max=1.34e-3 — within
  fp16 tolerance, MCTS-safe (no systematic bias, random-sign fp16 noise).
- Prior +3% estimate (§32) was against already-compiled `default` baseline.
  True eager → reduce-overhead gain is 1.50× throughput / 1.87× latency.

### Landing steps

1. `configs/training.yaml`: set `torch_compile: true`, add `torch_compile_mode: reduce-overhead`.
2. `hexo_rl/selfplay/inference_server.py __init__`: after `self.model.eval()`, call
   `self.model = compile_model(self.model, mode=config.get("torch_compile_mode", "reduce-overhead"))`
   guarded by `if config.get("torch_compile", False):`.
3. `hexo_rl/model/network.py compile_model()`: already accepts `mode` arg — no change.
4. Run `make bench` with AC power. Verify all 10 perf targets pass.
   Expected: NN inference ≥6,500 pos/s; NN latency ≤3.5 ms.
5. Commit: `perf(inference): re-enable torch.compile reduce-overhead (§32 blockers fixed in PT2.11)`.
6. Update `perf-targets.md` baseline after AC bench.
7. Train path (`trainer.py`): defer. Validate inference stability over 1K steps first.

### Artifacts

- Report: `reports/investigations/torch_compile_retry_20260423/report.md`
- Raw data: `reports/investigations/torch_compile_retry_20260423/data.json`
- Dynamo logs: `reports/investigations/torch_compile_retry_20260423/logs/`
- Probe script: `scripts/probe_torch_compile.py`

### §116.a — Landing on master 2026-04-24, then revert on resume deadlock

Landing sequence (master):

1. `1e2d82b perf(compile): enable torch.compile reduce-overhead (§116 GO)`
   flipped `torch_compile: false → true`, added `torch_compile_mode: reduce-overhead`, read mode from config in `trainer.py`, `loop.py`, `benchmark.py` (all three had hardcoded `mode="default"` that would have silently lost the §116 gains).
2. `41ffad5 fix(compile): resume path + best_model unwrap` — two runtime
   fixes discovered on the first resume attempt:
   (i) `best_model = best_ref.model` captured the `torch._dynamo.OptimizedModule`
   wrapper under the new live config, so every downstream `best_model.state_dict()`
   call emitted `_orig_mod.*`-prefixed keys and failed on the unwrapped
   `_inf_base.load_state_dict(...)` target. Fixed by unwrapping once at
   creation.
   (ii) `scripts/train.py` `config_overrides` propagated `torch_compile`
   from the live config onto pre-§116 checkpoints but not
   `torch_compile_mode`, so the resumed trainer silently fell back to
   `mode="default"` despite the new YAML setting. Fixed by propagating
   both.

3. **Second resume deadlock.** With mode-override fix in place, the second
   resume entered `futex_do_wait` on all 78 Python threads at step 6002
   immediately after `buffer_warmup_ended`: the trainer had just issued its
   first gradient step while the inference server was JIT-compiling the
   inf_model, and both `torch._dynamo` contexts hung. GPU 0 %,
   `InductorSubproc` idle on its pipe. Unrelated to our own code — most
   likely Triton compile-cache lock contention or a Py3.14 dynamo
   thread-safety edge when two OptimizedModule contexts compile
   concurrently; the §116 probe only exercised single-model compile, not
   the trainer + inference_server topology.

4. `e102a0a revert(compile): disable torch_compile after resume deadlock`
   flipped the YAML flag back to `false`. **The code-correctness fixes
   (`1e2d82b` mode-plumbing + `41ffad5` OptimizedModule unwrap) stay** —
   they are independently correct and wire up a future compile re-enable
   once the deadlock is root-caused.

### Re-enable preconditions

Do not re-flip `torch_compile: true` without at least one of:

- A repro harness for the trainer+inference dual-JIT deadlock so the
  failure mode is observable before it hits a 6000-step checkpoint.
- Sequencing the two compiles (trainer first, inference server deferred
  until first train_step emits), OR a single shared `_dynamo.config.cache_size_limit`
  workaround that proves cache-lock contention was the cause.
- Bench verification with a **training loop** smoke (not just the
  batch=64 synthetic probe in `probe_torch_compile.py`).

