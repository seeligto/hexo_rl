<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §124 — InferenceServer dispatch fix: TorchScript trace + bench methodology shift

**Date:** 2026-04-25
**Commits:** `1ab2e01` (trace + tests + narrow sweep), follow-up commit (bench methodology + perf-targets + this entry).
**Reference hardware:** laptop AMD Ryzen 7 8845HS + RTX 4060 Laptop GPU.

### Verdict

TorchScript-trace InferenceServer forward at `__init__` via `torch.jit.trace`; gated by `selfplay.trace_inference` (default `true`); falls back to untraced module if trace raises (e.g. `torch.compile`-wrapped). ScriptModule shares parameter storage so `load_state_dict_safe` mutation propagates without re-tracing. Also merges D2H (`cat(probs, value)` → single `.cpu().numpy()`). Eliminates ~32.6% CPython `nn.Module._call_impl` dispatch cost per forward (py-spy 200 Hz, 180 s, n_workers=10 on 3070); binding constraint on dispatch-bound hardware (EPYC 4080 S 60% GPU-util lock — see `feedback_compile_selfplay_dispatch_bound.md`). Closes `project_stall_diagnostic_deferred.md`; Q35 (ReplayBuffer GIL) won't move pos/hr.

**Why trace not compile:** trace wins on simplicity (no Dynamo guard cost, no cudagraph TLS thread issue, no Triton 27 GB spike on PT 2.10) and ~matches compile throughput on GPU-bound hardware while lifting dispatch-bound hardware. They're alternatives, not stackable (`compiled._orig_mod` unwrap verified but not implemented).

### Local 3070 smoke (90 s warmup + 180 s steady, n_workers=10, no py-spy)

| Path | pos/hr | fwd/s | batch_fill | inf/s |
|---|---|---|---|---|
| trace OFF | 122,800 | 73.7 | 97.5 % | 4,600 |
| **trace ON** | **164,600** | **94.5** | **87.9 %** | **5,316** |

**+34% pos/hr on 3070.** Always profile-compare without py-spy attached for absolute numbers — py-spy is fine for proportional breakdowns only.

### Bench methodology shift: compile OFF + trace ON is the production gate

§123 set `make bench` to compile-on under the assumption it matched production; today's sweep showed compile *regresses* selfplay pos/hr ~4% on EPYC 4080 S — production variants set `torch_compile: false`. Bench must reflect production.

| make target | compile | trace | When to use |
|---|---|---|---|
| `make bench`         | OFF | ON (default) | Phase 4.5 gate; matches production |
| `make bench.compile` | ON  | falls back   | Engineering datum: peak NN compute |
| `make bench.fast`    | OFF | ON (default) | Cold-cache quick check |

`scripts/sweep_epyc4080.py` passes `--no-compile`; sweep YAML writes `torch_compile: false`. NN inference target lowered 6,500 → 4,000 pos/s (compile-off loses Inductor fusion; tracks `min(observed × 0.85, prior)`).

### Laptop 4060 baseline (`make bench` compile-off + trace-on, NEW PRODUCTION BASELINE)

MCTS 66,926 sim/s · NN inf 4,859 pos/s · NN lat 2.56 ms · Buffer push 615,183 pos/s · GPU 100% · Worker 177,799 pos/hr (batch_fill 99.2%) — all gates PASS. Compile-on datum 186,832 pos/hr (within IQR of trace-on). Worker bimodality (1 of 5 runs 0 games) is pre-existing §102 startup race, not trace-induced.

### Follow-ups

- **Python 3.14 deprecation.** `torch.jit.trace` deprecated on Py 3.14+. `torch.export` verified bit-identical / ~equivalent perf locally; migrate when PyTorch removes jit.trace. pytest.ini suppresses deprecation warnings.
- **EPYC 4080 S validation sweep.** CLOSED 2026-04-25 — see §125.

---

