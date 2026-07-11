# E1 promotion-gate CUDA isolation — design note

**Status:** design + implementation (CONFRES batch-7 seam addition, 2026-07-11).
**Owner:** CONFRES completion lead (WP1 of D-H CONSOLIDATE).
**Related:** `docs/designs/selfplay_stall_watchdog_design.md` (the fail-fast SYMPTOM guard, shipped);
this note is the ROOT fix that guard was deferring.

---

## 1. Problem — the run2 eval-boundary livelock

`StepCoordinator` kicks the in-loop promotion-gate eval in a **daemon thread**
(`step_coordinator.py:1329` `_run_eval` → `eval_pipeline.run_evaluation`). That thread runs GPU
forwards on `self.eval_model` CONCURRENTLY with self-play, which drives its own GPU forwards
through the `InferenceServer` (`hexo_rl/selfplay/inference_server.py`). Two Python threads issuing
CUDA forwards into the same process/context is the run2 wedge: at a 250k-class eval boundary the
eval thread and the self-play inference threads deadlocked on the GPU for ~45h — `games_completed`
frozen, the main loop spinning in `waiting_for_games` forever (memory `run2-stall-watchdog`).

The self-play stall watchdog (exit 42 after `selfplay_stall_timeout_sec` with no new game) makes
this **fail fast** instead of hanging 45h — but it is a SYMPTOM guard. The ROOT cause is the
shared CUDA state between the eval-thread forwards and the self-play inference forwards.

## 2. Options

**Option A (PREFERRED) — run the promotion-gate eval as a SUBPROCESS.** The eval boundary spawns a
child `python -m hexo_rl.eval.promotion_gate_worker` with its OWN CUDA context (a fresh process =
a fresh context; no shared allocator, no cross-thread CUDA serialization). The child loads the
candidate + best weights from checkpoint files, runs the eval battery, and writes its
`EvalRoundResult` back as a JSONL sidecar file. The parent reads the sidecar (the bridge-sidecar
pattern — NEVER stderr, which the 64KB pipe-deadlock class would wedge on a chatty eval). This is
the same isolation self-play already gets from its worker processes.

**Option B — serialize.** Pause self-play forwards during the gate window (drain the inference
server, hold new forwards, run the eval on the main context, resume). Correct but couples the two
subsystems tightly, stalls self-play for the FULL eval wall-time (4–7h at deep boundaries —
exactly when throughput matters), and the pause/resume handshake is a new deadlock surface of its
own.

**Choice: Option A.** Simplest isolation, no self-play stall, matches the existing worker-process
pattern, and the failure mode of a crashed child is a clean non-zero exit + an absent/partial
sidecar the parent can detect (vs Option B's silent-hang risk).

## 3. Design (Option A)

```
StepCoordinator eval boundary
  │
  ├─ (existing) load candidate weights into eval_model, snapshot step
  ├─ write candidate + best weights to a run-scoped tmp dir (already on disk as checkpoints)
  ├─ spawn: python -m hexo_rl.eval.promotion_gate_worker \
  │           --candidate <ckpt> --best <ckpt> --config <json> --radius <int> \
  │           --result <sidecar.jsonl>
  │        (own process → own CUDA context; NO shared InferenceServer state)
  ├─ the worker runs eval_pipeline.run_evaluation and writes ONE json line
  │  {"event":"promotion_gate_result", ...EvalRoundResult...} to <sidecar.jsonl>, then exits 0
  └─ parent reads <sidecar.jsonl>, parses the EvalRoundResult, feeds it to the promotion logic
     (identical downstream: same EvalRoundResult shape the in-thread path produced)
```

**Bridge = file, never stderr.** The worker's stdout/stderr go to a log file (for forensics); the
RESULT travels via the `--result` JSONL sidecar. A partial/absent sidecar + a non-zero child exit
= a broken eval → the parent flags it LOUD (the existing `_eval_broken` path) and disables
promotions for that round, exactly as the in-thread crash path does.

**Compatibility.** The subprocess path is behind a config flag
(`eval_pipeline.subprocess_isolation: true`), default matching current behavior until validated on
vast, so the golden-run byte-purity holds (the in-thread path is unchanged when the flag is off).
The flag ON is the run3 posture.

**Watchdog interaction.** The subprocess isolation removes the concurrent-GPU-forward deadlock the
watchdog was catching. The watchdog STAYS as defense-in-depth (a hung child, an OOM, a driver
wedge still freeze `games_completed`); `selfplay_stall_timeout_sec` is unchanged. The regression
smoke (below) runs with the watchdog ARMED to prove the boundary crosses cleanly under it.

## 4. Regression smoke (joins the integration gate)

`tests/test_promotion_gate_isolation_smoke.py` (`@pytest.mark.slow @pytest.mark.integration`):

- Launch a short `train.py` run that CROSSES an eval boundary of the 250k-trigger class (a small
  `eval_interval` so the boundary falls inside `--iterations`), with the self-play stall watchdog
  ARMED and `selfplay_stall_timeout_sec` set EXPLICITLY (a small but non-trivial value).
- PASS =
  1. the eval boundary is crossed (an `evaluation_start` / promotion-gate event fires),
  2. `games_completed` ADVANCES after the boundary (self-play resumed — no wedge),
  3. the process exits 0 (NOT the watchdog's exit 42 — no stall).

This is the exact run2 failure geometry (eval boundary + armed watchdog) proven to cross cleanly.

## 5. Rollout

- Land behind the default-off flag (byte-pure); the CONFRES golden-run 3-regime byte-purity is
  unaffected.
- Operator flips `subprocess_isolation: true` for the run3 launch after the vast smoke confirms the
  boundary crosses with the flag ON at a real 250k-class boundary.
