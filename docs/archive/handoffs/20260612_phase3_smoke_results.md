# D-RERUNPREP Phase 3 — GPU smoke RESULTS (vast 5080, 2026-06-11)

> **RESOLUTION (2026-06-12) — and a CORRECTION.** The F1 "fp16 W2-pin" diagnosis below was
> **WRONG on the mechanism**. A cheap GPU discriminator (run before the re-smoke) showed the anchor
> is **fp32** and OLD==NEW hashing (`4198d5cb`). The real cause was a **host bootstrap DRIFT**:
> vast's `bootstrap_model_v6_live2.pt` (`4198d5cb`) ≠ the committed pin (`aba28e10`, set from the
> laptop's drifted copy; the file is gitignored / out-of-band). The genuine **code** bug is
> **W2-VACUOUS** — the launch fresh-init path skipped the pin entirely. Fixes: `verify_launch_anchor_pin`
> (closes W2-VACUOUS, fail-closed); `checkpoint_state_sha256` (DRY/source-hash robustness, not the
> cause); F2 telemetry; F3 script. **Data fix:** de-facto bootstrap = vast's `4198d5cb` (golong +
> all vast runs used it) → re-pinned configs/runbook to `4198d5cb`, synced hosts. 12 anchor + 16
> closeout tests green, full suite 1998 passed. Re-smoke on the fixed code + correct pin is the
> remaining gate. (The fp16 narrative in §F1 below is retained as the as-diagnosed record, struck
> through by this banner.)

**Verdict: FAIL** (do NOT launch the 50k). But the criteria *table* the script printed is
mostly its own bugs — the real signal, mined from `run.log` / `resume.log`, is below.
Artifacts on vast: `reports/phase3_armc_smoke_20260611T205257Z/` (run.log 1.49 MB, resume.log,
closeout_test.log, exploit_probe.*).

## Real findings (ranked)

### F1 — BLOCKER: the §D-LOOPFIX **W2 incumbent-pin is non-functional**
The marquee W2 safety claim ("HARD-FAILS launch if best_model.pt is anything else") does not
work as deployed.
- **Exists-path false-fail.** The resume HARD-FAILED: `RuntimeError: best_model.pt resolved to
  4198d5cb… but config pinned aba28e10…` — for a `best_model.pt` that was a byte copy of the
  pinned bootstrap. Root cause: the pin (`scripts/anchor_sha256.py`) hashes the **on-disk fp32
  file**; the runtime assert (`anchor.py:332`, `state_dict_sha256(best_model.state_dict())`)
  hashes the **loaded anchor model, which is fp16** (`config fp16:true`). Keys match (147=147),
  so it's pure precision — and `fp32→fp16` is lossy, so **no cast reconciles them**. The pin can
  never equal the runtime hash for the same logical weights → it false-rejects a *correct*
  incumbent.
- **Launch-path vacuum.** The runbook's preflight `rm -f best_model.pt{,.bak}` sends
  `resolve_anchor` down the `anchor_fresh_init_no_bootstrap` branch (`anchor.py:~413-430`), which
  seeds the anchor from `trainer.inference_state_dict()` (= the `--checkpoint` bootstrap) and
  **never runs the pin check**. So on the intended launch the pin provides **zero** protection;
  the real protection is the `rm` + fresh-init, not the pin.
- Also: that branch *warns* `anchor_fresh_init_no_bootstrap` with `tried=_BOOTSTRAP_ANCHOR_CANDIDATES`
  → `bootstrap_model_v6_live2.pt` is **not** a recognised anchor candidate, so even the "load the
  bootstrap as the pinned anchor" path isn't wired.

**Why Phase 1 missed it:** B2 mocked `load_best_model_resilient` (never exercised the real
fp16 load); B3 verified only that the *file* sha == pin (static), never that the *runtime model*
sha == pin. This is precisely the static→runtime gap the GPU smoke exists to close.

**Run-validity impact:** the 50k would still *run and be valid* — fresh-init seeds the anchor
from the bootstrap (terminal `wr_best=0.605`, not 1.0 → a real incumbent, not random) — and the
specific as-run bug (stale golong incumbent) is prevented by the `rm`+fresh-init. But the W2
*proof* the re-run's premise rests on is absent.

**Fix scope: TICKET (not XS).** Pick a canonical sha domain (e.g. hash the anchor in the same
dtype on both sides, or compute the pin from the post-load runtime model, or keep the anchor
model fp32 for identity) AND make the fresh-init/launch path actually verify the pin. Needs its
own design + review. Do NOT inline-patch.

### F2 — XS: `terminal_eval_complete` structlog line omits `completed`/`terminal`
Stage A (closeout integration test) failed on this. The terminal eval *completed correctly*
(`wr_best=0.605`); only the JSONL telemetry line lacks the field the test (and the watch sheet)
key on. Diagnosis + 2-kwarg patch: `docs/archive/handoffs/20260612_phase3_finding_terminal_eval_completed.md`.

### F3 — the self-check script (`run_phase3_armc_smoke.sh`) has bugs → false FAILs
- `grepc()` returns `"0\n0"` on zero matches (`grep -c` exits 1 → `|| echo 0` adds a second 0),
  breaking `[ … -ge 1 ]` (the `line 52/59: integer expression expected` errors).
- Wrong tokens: searched `anchor_identity` in the main run (it's `anchor_fresh_init_no_bootstrap`
  there), `ignore_stride.*True` (not logged in structlog), pool-stop (no such event token).
- crit 8 (provenance) is **unsatisfiable without a promotion** — a 500-step bootstrap-vs-bootstrap
  run never promotes, so no `save_best_model_atomic` → no `.provenance.json`. The smoke must
  FORCE a promotion (or lower the bar) to test the sidecar.
Must fix the script before any re-smoke, else the table stays untrustworthy.

## What actually PASSED (corroborated from run.log — the lifecycle works)
| Criterion | Real status | Evidence |
|---|---|---|
| 1 iteration_limit_reached | ✅ | event present, stop at 500 |
| 2 drain budgeted, WARN-never-kill | ✅ | `final_eval_drain_waiting budget_sec=900` → `final_eval_drain_timeout … "proceeding"` (warned, **not killed**) |
| 4 terminal full-battery | ✅ (core) | `terminal_eval_complete` ×1, `step=500, wr_best=0.605` (ran the battery); only the `completed` field missing (F2) |
| 5 resume ignores terminal record | ✅ (core) | resume emitted `anchor_identity`, **no** `terminal_eval_complete` re-ingest (it then died on F1's pin, a separate issue) |
| 6 ≥1 n=400 promotion decision | ✅ | best_checkpoint n=400 evaluated |
| 9 exploit_probe + KClusterMCTSBot | ✅ | `exploit_probe.summary.json`: exploit arm awr 1.0, control 0.0, verdict FORCEABLE (n=2 → wiring proof only, not strength) |

The training→eval→drain→terminal→exit lifecycle is sound. The blockers are the **W2 pin (F1)**
and the **telemetry/test-harness issues (F2/F3)**, not the run mechanics.

## Recommendation: **NO-GO** until F1 is resolved
The entire premise of the §D-LOOPFIX re-run is "the loop is now PROVEN fixed." The smoke proved a
marquee fix (W2 pin) is **not functioning**. The run would be incidentally valid (rm+fresh-init),
but launching a ~4-day/~$67 run on a violated premise is exactly what this sweep exists to stop.
Path to GO: (1) fix F1 [design ticket], (2) apply F2 [XS patch], (3) fix F3 [script], (4) re-run
the Phase-3 smoke (now with a forced-promotion case for provenance), (5) then launch.

cost note: the smoke also gives a real throughput datum — 500 steps + 2-3 full eval rounds at
`eval_interval=200` ran ~21:10→23:55 (~2h45m), eval-dominated; the real `eval_interval=12500`
run has proportionally far less eval overhead. Re-derive the $/wall-clock from the run.log step
timestamps before quoting the cost estimate.
