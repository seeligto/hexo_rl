# Self-play stall watchdog — design + incident record

**Date:** 2026-07-11. **Branch:** `hotfix/selfplay-watchdog` (off `fc97cc82`, the live run2 commit).
**Trigger:** run2_mw_fresh on vast wedged silently for ~45h; the monitor read "last event
157009s ago". This doc records the root-cause forensics and the watchdog that turns a silent
wedge into a fast, loud, monitorable failure.

---

## 1. Incident forensics (run2_mw_fresh, 2026-07-09)

The run was **not dead and not crashed** — it was **livelocked**. Evidence chain, traced
end-to-end from the live process on vast (PID 266593, alive in tmux `run2`):

1. Main training thread alive and looping — logged `waiting_for_games` every 5s because
   `new_games = pool.games_completed - last_train_game_count <= 0`. `pool.games_completed`
   **frozen at 60032** since **Jul 9 12:45 (step 250032)**.
2. Self-play Rust workers stopped completing games — **38 threads blocked in `futex_wait`**
   on the inference waiter condvars (`engine/src/inference_bridge.rs`).
3. Those inference results never came: **2 threads spinning at 100% CPU each** (confirmed
   live: +300 ticks / 3s each; pure userspace, empty `/proc/<tid>/syscall` = CUDA spin-sync
   busy-poll), while the **GPU sat at 0%**.
4. One spinner (**TID 376906**) was the **eval daemon thread**, spawned at the **step-250000
   boundary** (`StepCoordinator.step()` → `threading.Thread(_run_eval)`). The other (**TID
   266720**) was the self-play `InferenceServer` thread.
5. **Proof:** the eval round kicked off at 250000 **never completed** — last
   `evaluation_round_complete` is step **225000**; the 250000 round hung.

**Root cause (class):** the in-process async evaluation runs its **own** `LocalInferenceEngine`
GPU forwards on the eval daemon thread (`hexo_rl/eval/eval_batcher.py`), concurrent with the
self-play `InferenceServer` thread's forwards, sharing CUDA execution state (default
stream/driver). At the 250000 boundary that concurrent cross-thread GPU use wedged: both
threads spin in CUDA sync busy-waits (100% CPU, GPU idle) and self-play blocks on inference
forever. `torch_compile=false` for this run, so it is **not** the cudagraph path — it is the
raw concurrent-forward hazard. It is an **intermittent race**: eval boundaries at 200k and
225k passed fine. This is the same *class* of cross-thread CUDA hazard already documented at
`hexo_rl/training/lifecycle.py:76-81` (the reduce-overhead cudagraph deadlock at step 6002,
bench fix c26b9b4) — but that mitigation only covered `torch.compile`.

Not observed / ruled out: no panic, no Python traceback, no CUDA error, no OOM (`dmesg`
clean), no process death.

---

## 2. Watchdog design (this change)

**Goal:** bound the damage. A silent indefinite wedge (45h, ~$10 of GPU doing nothing, and
worse — invisible science loss) becomes a **fast, loud, fail-fast exit** that the monitor's
stale-detection surfaces and the operator/launch-loop restarts. Mechanism-agnostic: it
catches *any* self-play death (this eval-wedge, a worker crash, an inference-thread death),
not just this one CUDA race.

**Detection signal:** `pool.games_completed` stops advancing. Ground-truth "self-play is
alive" — it froze for 45h in this incident.

**Location:** `StepCoordinator.step()`, right after `self._games_played =
self.pool.games_completed`. Runs every loop iteration, all branches (warmup, waiting,
training).

**Logic:**
```python
if self._games_played > self._watchdog_last_games:        # progress → reset
    self._watchdog_last_games = self._games_played
    self._watchdog_last_progress_time = self._clock.now()
elif cfg.selfplay_stall_timeout_sec > 0:                  # frozen → check age
    stalled = self._clock.now() - self._watchdog_last_progress_time
    if stalled >= cfg.selfplay_stall_timeout_sec:
        self._fire_stall_watchdog(stalled)
```

**Fire action (`_fire_stall_watchdog`):**
1. LOUD `self._logger.error("selfplay_stall_watchdog", ...)` — carries `step`,
   `games_completed`, `stalled_for_sec`, `threshold_sec`, and
   **`eval_in_flight=self.is_eval_in_flight()`** (records the eval-wedge signature).
2. Best-effort **CPU-only** `_try_save_buffer(..., "selfplay_stall_watchdog", ...)` — verified
   GPU-free (`buffer_persist.try_save_buffer` writes numpy to disk, never raises, no-ops when
   `buffer_persist` disabled). Wrapped in try/except so it can never block the exit.
3. `self._exit_fn(SELFPLAY_STALL_EXIT_CODE)` — `os._exit`, guaranteed to kill even with a
   CUDA-wedged daemon thread (a clean shutdown would try to save a checkpoint via the wedged
   GPU and hang). The last periodic checkpoint (every 500 steps) already captured state at
   the wedge, so no progress is lost.

**Seams:**
- Config: `StepCoordinatorConfig.selfplay_stall_timeout_sec: float = 1800.0` (30 min). Far
  beyond any legitimate zero-games gap: even during a heavy 4-7h eval round self-play only
  *slows*, it does not stop (games kept completing at every prior healthy eval boundary).
  `<= 0` disables. Default ON so the live run is protected.
- `SELFPLAY_STALL_EXIT_CODE = 42` — distinct nonzero so the launch script's `RUN2_EXITED`
  sentinel distinguishes a watchdog abort from other failures.
- `StepCoordinator.__init__(..., exit_fn: Callable[[int], None] = os._exit)` — injectable so
  tests verify firing without killing pytest.
- Init state: `self._watchdog_last_games = pool.games_completed`,
  `self._watchdog_last_progress_time = clock.now()`.
- Plumbed in `hexo_rl/training/loop.py` via the existing `_lifecycle_knob(name, default)`
  pattern.

**False-positive analysis:** the only way to legitimately produce zero new games for 30 min
post-warmup is a full self-play starvation — itself a problem worth flagging. Warmup produces
games (progress resets the timer). Eval rounds run self-play concurrently. The knob is
operator-tunable if a legitimate long pause is ever discovered.

**Tests** (`tests/training/test_step_coordinator.py`, existing FakeClock/Mock harness):
fires after threshold with frozen games (asserts `exit_fn(42)` + loud event +
`eval_in_flight` recorded); does not fire while games advance; `<= 0` disables; buffer-save
attempted before exit; does not fire before threshold.

---

## 3. Deferred: root fix (separate bench-gated cycle)

The watchdog bounds damage; it does not eliminate the race. Eliminating the concurrent
cross-thread GPU contention is a separate, **perf-sensitive** change requiring `make bench`
and its own design cycle. Candidate approaches, to be evaluated there:

- **Dedicated CUDA stream for eval** — isolate eval forwards from the self-play default
  stream. Smallest change; still shares the caching allocator.
- **Serialize forwards** — a global inference lock so eval and self-play never run model
  forwards concurrently. Mechanism-agnostic; costs eval-round latency (eval blocks self-play
  inference).
- **Subprocess eval** — eval in its own process + CUDA context; full isolation, largest change
  (IPC for results + checkpoint hand-off).

The exact CUDA primitive could not be confirmed on the live box (the vast container lacks
`CAP_SYS_PTRACE`, so py-spy/gdb are unavailable). The chosen root fix should be robust to the
mechanism, which argues for serialize-forwards or subprocess isolation over a stream-only
tweak.
