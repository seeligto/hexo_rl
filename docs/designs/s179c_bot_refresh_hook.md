# §S179c — Bot Corpus Refresh Hook Activation Design

**Status.** DESIGN. Implementation deferred pending §S178 sustained run verdict.
**Position.** Escalation lever §3.4 in s178_design.md ladder.
**Branch.** phase4.5/s179c_fa1_prep
**Master at branch creation.** f34295a

---

## 0. Context

§S178 uses a STATIC bot corpus (`data/bot_corpus_s178_sealbot_vs_v6.npz`, 700 games / 21,899
positions, SealBot vs `bootstrap_model_v6.pt`). The refresh hook is fully wired but inert: when
`cfg.bot_corpus_refresh_enabled` is False, `hexo_rl/training/step_coordinator.py:651-666` emits a
warning-only log and does not regenerate. §S178 design §3.4 (`docs/designs/S178_design.md:163-184`)
catalogues the aging risk: model diverges from v6.pt rapidly during first 5-10K steps, and the
bot signal weakens as model surpasses v6.pt strength. §S179c flips `enabled: true` and adds:

1. async subprocess launcher (non-blocking trainer pause ≤200ms);
2. atomic NPZ swap (write-rename pattern, POSIX rename guarantee);
3. hot-reload of `bot_buffer` in `batch_assembly.py:243-336` (close old, open new).

Anchor stickiness is the trigger pin: the refresh runs against the CURRENT `best_model.pt`
(see `hexo_rl/training/anchor.py:35-58`), not v6.pt, so the bot games match where the policy now
lives. This converts the bot slot from a static counter-example pool to a moving training
adversary.

---

## 1. Trigger conditions

### 1.1 Eval event that qualifies

Source-of-truth promotion gate is already wired:
`hexo_rl/eval/eval_pipeline.py:297-299` — `promotion_winrate=0.55` against `best_checkpoint`,
combined with `bootstrap_floor.min_winrate=0.45` at `eval_pipeline.py:299,307`. The
`promoted_step` value reaches the refresh hook via
`step_coordinator.py:636-643` (`_drain_pending_eval` returns `(thread, best_model_step)`;
caller assigns `promoted_step = self._best_model_step` when delta).

**Trigger rule (V179c-1 wire):** refresh fires when

```
promoted_step is not None
AND (best_model_winrate − previous_best_model_winrate) ≥ 0.05   # 5pp delta on next-anchor eval
AND (self._train_step − self._last_bot_refresh_step) ≥ cooldown_steps
AND self._n_refreshes_so_far < max_regens
```

The 5pp delta is computed from `eval_result[0]["wr_best"]` minus a one-slot history kept on
`StepCoordinator` (`self._last_best_wr: float`). 5pp is the same magnitude as
`promotion_winrate` (0.55) over `bootstrap_floor` (0.45) — the existing gating contract — so
a regen costs compute only on a meaningfully stronger anchor, not noise crossings. Mere
promotion is insufficient: it can fire on +1pp where the bot pool is still informative.

**Rationale.** SealBot WR @ MCTS-128 jitter is wide (e.g. §175 sampled-eval n=20 Wilson CIs span
~12pp per `MEMORY.md project_bootstrap_argmax_drift_check_20260511`). Δ ≥ 5pp on n=200
(`eval.yaml:752`, best_checkpoint `n_games: 200`) holds at Wilson95 against 1pp noise.

### 1.2 Cooldown

```
refresh.cooldown_steps: 25_000   # default; existing key in step_coordinator.py:174
```

25K steps matches one §178 promotion cadence. With `eval_interval: 5000` and ~ 770 games/hr
selfplay, 25K steps is ~ 5 promotion-eligible rounds — promotions arriving inside cooldown
are still valid for `best_model.pt` updates but do not re-fire the regen.

Smaller (e.g. 10K) risks thrashing: ~ 10hr single-thread regen (`scripts/generate_bot_corpus.py`
n_games=700 elapsed_sec=2786 per
`data/bot_corpus_s178_sealbot_vs_v6.npz.metadata.json:14` extrapolated to 4× sims) would
overlap the next promotion window. Larger (e.g. 50K) defeats the §3.4 lever: anchor drift
during 50K steps invalidates the bot pool.

### 1.3 Hard cap

```
refresh.max_regens: 6
```

Bounds total GPU contention: 6 × ~ 2-10hr = ~ 12-60hr regen compute over a 100K-step run
(`v6_botmix_s178.yaml:27` `total_steps: 100_000`). At 5K-step `eval_interval`, refresh windows
are 0, 25K, 50K, 75K, 100K — five eligible eval boundaries plus one manual lever = 6.

### 1.4 Manual trigger CLI

For operator-forced refresh outside the cooldown/threshold:

```bash
# Touch a control file; coordinator polls at top of each step.
touch /tmp/hexo_rl_force_bot_refresh
# Coordinator reads + deletes file → fires regen on next eval boundary, regardless of cooldown.
```

Read path: `step_coordinator.py` top of `step()` checks `Path("/tmp/hexo_rl_force_bot_refresh").is_file()`,
deletes file, sets `force_refresh: bool = True` flag carried into the refresh gate at
`step_coordinator.py:651` (replaces the cooldown predicate).

Rejected alternative: SIGUSR1 handler. Already burned on §S178 launch (signal handlers installed
by caller against shared `ShutdownState`, per `step_coordinator.py:218-221`). File-based sentinel
is robust + scriptable.

---

## 2. Async regen launcher

### 2.1 subprocess.Popen vs asyncio

**Decision: `subprocess.Popen` (non-blocking poll loop).**

Trainer process model rationale:

- Trainer step loop is single-threaded synchronous (`step_coordinator.py:444-870`); eval already
  runs as `threading.Thread(daemon=True)` at `step_coordinator.py:704`. No asyncio event loop
  in the process — adopting `asyncio.create_subprocess_exec` would require either
  (a) a dedicated event-loop thread (~50 LOC scaffolding), or
  (b) per-step `asyncio.run()` calls that block.
- `subprocess.Popen` with `proc.poll()` integrates into the existing per-step structure: one
  `Popen` at fire time, then `proc.poll()` at each subsequent eval boundary until terminal.
  Maps 1:1 to existing `self._eval_thread` lifecycle pattern.
- Regen is wall-clock-long (minutes-hours, not ms), so async/sync overhead is irrelevant; what
  matters is non-blocking trainer pause at launch.

**Rejected `asyncio.create_subprocess_exec`:** adds dependency on event-loop installation in the
training process. Eval thread proves the threading pattern works; introducing asyncio for one
caller is unjustified.

### 2.2 PID tracking + zombie reap

State on `StepCoordinator`:

```python
self._refresh_proc: subprocess.Popen | None = None
self._refresh_started_step: int = 0
self._refresh_target_anchor_sha: str | None = None    # current best_model.pt sha at fire time
self._n_refreshes_so_far: int = 0
self._last_best_wr: float | None = None
```

Per-step poll:

```python
if self._refresh_proc is not None:
    rc = self._refresh_proc.poll()
    if rc is None:
        pass                                    # still running
    elif rc == 0:
        self._logger.info("bot_corpus_regen_complete", ...)
        # Stage the swap (§3) + hot-reload (§4) here.
        self._refresh_proc = None
    else:
        self._logger.warning("bot_corpus_regen_failed", returncode=rc, ...)
        self._refresh_proc = None               # zombie reaped via poll()
```

Per Python stdlib `subprocess.Popen.poll`: once returncode is set, the child is reaped on the
next OS-level `wait()` which `poll()` performs implicitly. No leaked zombies.

### 2.3 Failure handling

Three failure classes:

| Class | Detection | Recovery |
|---|---|---|
| Regen subprocess crash (rc != 0) | `Popen.poll()` returncode non-zero | Log `bot_corpus_regen_failed`; retain CURRENT NPZ in place (no swap); decrement `_n_refreshes_so_far` so retry is possible at next cooldown |
| NPZ corrupt write (`save_corpus` truncated by SIGKILL) | Sidecar sha256 mismatch on post-rename verify (§3.2) | Reject the swap; rename `.tmp` to `.corrupt-<ts>`; retain current NPZ; log + warn |
| Disk full | `OSError` from `save_corpus` inside subprocess → non-zero rc | Same as crash; subprocess prints `FATAL` and exits 2 per `generate_bot_corpus.py:393-395` pattern |

### 2.4 Event-bus logging

Per `docs/08_DASHBOARD_SPEC.md:13-20`, `emit_event()` is the single call site (passive observer).
New event types (all carry the standard `ts` field added by `emit_event` per
`docs/08_DASHBOARD_SPEC.md:48`):

```json
{ "event": "bot_corpus_regen_requested", "step": 30000, "trigger": "best_model_promotion",
  "wr_delta": 0.07, "anchor_sha": "ab12cd34", "subprocess_pid": 42313 }

{ "event": "bot_corpus_regen_complete", "step": 32500, "returncode": 0,
  "elapsed_sec": 8400, "n_positions": 22150, "new_npz_sha": "deadbeef..." }

{ "event": "bot_corpus_regen_failed", "step": 32500, "returncode": 2, "reason": "no_decisive_games" }

{ "event": "bot_corpus_hot_reload", "step": 32500, "n_positions": 22150, "reload_sec": 1.7 }
```

Passive observer: emit-only, never read back. Renderer pattern-matches in
`hexo_rl/monitoring/` (out of scope).

---

## 3. Atomic NPZ swap

### 3.1 Write-rename pattern

Subprocess writes to `data/bot_corpus_s178_sealbot_vs_v6.npz.NEW.tmp`; coordinator (NOT the
subprocess) performs the rename when subprocess exits 0. Two reasons coordinator owns the
rename: (a) subprocess can SIGKILL between save and rename; (b) coordinator already gates the
hot-reload, so the rename and reload must be sequential under one owner.

Sequence on `rc == 0`:

```
1. Verify <tmp>.metadata.json exists and reports n_positions > 0.
2. Compute sha256(<tmp>) → compare to metadata.sha256 (save_corpus auto-writes both at
   hexo_rl/bootstrap/corpus_io.py:106-155).
3. Read CURRENT canonical NPZ sha (from the existing sidecar) for forensic log.
4. os.rename(canonical, canonical + ".bak")              # rotate old
5. os.rename(<tmp>, canonical)                            # atomic swap
6. os.rename(<tmp>.metadata.json, canonical.metadata.json)
7. Emit `bot_corpus_swap_committed` event.
8. Schedule hot-reload (§4).
```

POSIX `rename(2)` is atomic when source and destination are on the same filesystem. From
`man 2 rename` (POSIX 1003.1-2017): "If newpath already exists, it will be atomically replaced".
Confirmed at `hexo_rl/training/anchor.py:46-58` (`save_best_model_atomic`): the canonical
in-tree pattern is `path.tmp` → torch.save → load-verify → `path.replace(bak)` → `tmp.replace(path)`.
We reuse this pattern unmodified for the NPZ.

`Path.replace()` (Python stdlib) wraps `os.rename` with cross-platform semantics; safe on Linux
training hosts. **Pin: same filesystem.** Subprocess writes inside `data/` (same FS as canonical
`data/bot_corpus_s178_sealbot_vs_v6.npz`); rejecting cross-FS path arguments at config
validation is the defence. INV-S179c-2 below.

### 3.2 SHA verification post-rename

After step 5, immediately re-compute sha256 of the canonical path and compare to the metadata
sidecar `sha256` field. Mismatch → roll back: restore `.bak` → canonical, move bad file aside
as `.corrupt-<ts>` (mirrors `anchor.py:61-67` quarantine pattern). Refuse hot-reload.

### 3.3 Backup retention

Old NPZ retained as `data/bot_corpus_s178_sealbot_vs_v6.npz.bak` for one refresh cycle —
overwritten on next refresh. Provides one-step rollback for the operator (manual
`mv *.bak <canonical>` if a refresh produces a regression visible at next eval). 22MB file ×
1 retained copy = trivial disk cost.

---

## 4. Hot-reload

### 4.1 Reload trigger: event-driven, NOT mtime polling

`batch_assembly.py` (the load entry point at `batch_assembly.py:243-336`) does NOT auto-watch
the NPZ. The COORDINATOR triggers reload directly after the swap completes.

**Rationale.** `batch_assembly.assemble_mixed_batch` is called per training step
(`step_coordinator.py:548-555`); polling NPZ mtime each call is wasteful, and the coordinator
already owns the reload moment (post-swap). Direct call is simpler than file-watch.

### 4.2 Close-old / open-new

```python
# Inside StepCoordinator, after swap committed:
old_size = self.bot_buffer.size if self.bot_buffer is not None else 0
self.bot_buffer = None              # drop reference; Rust Drop closes the buffer
import gc; gc.collect()             # eagerly free the ~22MB Rust allocation
self.bot_buffer = load_bot_corpus_buffer(
    self.mixing_cfg, self.full_config, self._event_emitter,
    self.buffer.size, self.buffer.capacity,
)
self._logger.info("bot_corpus_hot_reload",
    step=self._train_step,
    old_n_positions=old_size,
    new_n_positions=(self.bot_buffer.size if self.bot_buffer else 0),
)
```

`engine::ReplayBuffer` (`engine/src/replay_buffer/mod.rs:74-100`) owns flat Vecs; dropping the
Python handle releases them via Rust Drop. No explicit close API needed.

### 4.3 Read-write lock pattern

**Locked: single-writer / single-reader trainer process.** No locking primitive needed.

`assemble_mixed_batch` (`batch_assembly.py:476-555`) reads `self.bot_buffer.sample_batch` ONCE
per training step, fully on the trainer thread. Eval runs on a separate daemon thread but does
NOT touch `self.bot_buffer` (see `_run_eval` body at `step_coordinator.py:684-702`). Therefore
the hot-reload swap site is:

```
trainer thread, end of one training step  →  swap buffer reference  →  next training step uses new buffer
```

No partial-batch hazard: the read in step N has already returned numpy arrays (allocated copies
per `mod.rs:54-65` "owned NumPy arrays") before the swap fires. Step N+1 reads the new buffer.

INV-S179c-3 below pins this: swap fires at the same point in the step lifecycle where the
batch read has already completed (right after eval drain, before next iteration).

### 4.4 Reload cost upper bound

Current NPZ: 22K positions, 35MB on disk (`data/bot_corpus_s178_sealbot_vs_v6.npz` per
`metadata.json` `n_positions: 21899`).

Measured proxy: `load_pretrained_buffer` at `batch_assembly.py:122-238` loads 50K+ positions
in ~30s wall-clock (per existing `corpus_loaded` log emission). Bot corpus is ~ 0.44× that
size → upper bound **2 seconds** wall-clock load. Budget: **≤ 5 sec** (2.5× safety margin).

Trainer pauses for those 5 sec (single-threaded step loop). Acceptable: training is paused for
~ 60 sec each eval already (`step_coordinator.py:636-708` eval drain + kickoff). 5 sec extra
every 25K steps is a negligible overhead.

---

## 5. Boundary discipline

### 5.1 Scope

Refresh applies ONLY to `self.bot_buffer` (the §178 bot-corpus Rust ReplayBuffer constructed at
`batch_assembly.py:324`). Specifically:

- **Human corpus (`self.pretrained_buffer`)**: NEVER touched by refresh. Cited:
  `batch_assembly.py:122-238` is a different function with a different NPZ path key
  (`mixing.pretrained_buffer_path`); refresh hook only re-invokes `load_bot_corpus_buffer`.
- **Self-play buffer (`self.buffer`)**: NEVER touched. It is a separate Rust ReplayBuffer
  filled by `WorkerPool.push_game` callbacks; refresh has no path to it.
- **Recent buffer (`self.recent_buffer`)**: NEVER touched. Python RecentBuffer for recency
  weighting; lives independently in `hexo_rl/training/recency_buffer.py`.

### 5.2 INV pins

**INV-S179c-1.** Refresh hook never touches human corpus or selfplay buffer instances.
Test: instrument `load_pretrained_buffer` call counter; assert it stays at 1 (single startup
load) across an entire run with N refreshes. Equivalent assertion on `self.buffer is unchanged`
identity-comparison across refresh.

**INV-S179c-2.** Refresh target path must be on same filesystem as canonical NPZ; reject
cross-FS configurations at startup. Implementation: at config-load time, assert
`Path(bot_corpus_path).parent.resolve()` and `Path(bot_corpus_path).parent.resolve()` share
`os.stat().st_dev`. Cross-FS rename is not atomic.

**INV-S179c-3.** Hot-reload swap site is post-eval-drain, pre-next-iteration. No swap inside
a training step's batch-assembly window. Test: monkeypatch `assemble_mixed_batch` to record
the `id(self.bot_buffer)` it sees per call; assert identity stable WITHIN one step.

**INV-S179c-4.** Refresh-disabled run is bitwise identical to §S178 baseline. Test: run
fixture-seeded smoke (n_steps=200) with `enabled: false` against frozen baseline trace; assert
zero divergence on loss values, buffer sizes, batch contents.

---

## 6. Config surface

New keys (extend existing block at `configs/variants/v6_botmix_s178.yaml:44-48`):

```yaml
mixing:
  # Existing §S178 keys retained:
  bot_corpus_path: "data/bot_corpus_s178_sealbot_vs_v6.npz"
  bot_batch_share: 0.15

  # §S179c refresh block (existing skeleton at yaml:44-48 + NEW fields):
  bot_corpus_refresh:
    enabled: false                   # DEFAULT OFF — matches §S178 wiring
    trigger:
      metric: best_model_winrate     # NEW — explicit trigger source
      threshold: 0.05                # NEW — Δ vs previous best wr to fire (5pp)
    cooldown_steps: 25_000           # existing
    max_regens: 6                    # NEW — hard cap per run
    min_new_games: 200               # existing skeleton; passed to subprocess --n-games
    # Subprocess argument template — fills target_anchor at fire time.
    regen_command:
      script: "scripts/generate_bot_corpus.py"
      args:
        n_games: 200                 # match min_new_games
        max_plies: 150
        random_opening_plies: 4
        think_seconds: 0.5
        anchor_n_sims: 200
        anchor_temperature: 0.5
```

`enabled: false` default matches the existing wiring at
`hexo_rl/training/step_coordinator.py:173`. A §S179c-enabled variant (e.g.
`configs/variants/v6_botmix_s179.yaml`, NEW) flips `enabled: true` and lands as a separate
commit per §9 task split.

Config plumbing extends `hexo_rl/training/loop.py:243-249` to read the new fields with safe
defaults:

```python
bot_corpus_refresh_enabled=bool(mixing_cfg.get("bot_corpus_refresh", {}).get("enabled", False)),
bot_corpus_refresh_cooldown=int(mixing_cfg.get("bot_corpus_refresh", {}).get("cooldown_steps", 25_000)),
bot_corpus_refresh_threshold=float(
    mixing_cfg.get("bot_corpus_refresh", {}).get("trigger", {}).get("threshold", 0.05)
),
bot_corpus_refresh_max_regens=int(mixing_cfg.get("bot_corpus_refresh", {}).get("max_regens", 6)),
```

---

## 7. Pre-registered verdicts

| ID | Hypothesis | PASS | FAIL | NULL |
|---|---|---|---|---|
| **V179c-1** | Regen subprocess starts + completes without blocking trainer | Max trainer pause at launch ≤ 200ms (measured: `t_pre_Popen` vs `t_post_Popen` in fire path); subprocess `Popen.poll() is None` immediately after start | Trainer pause > 200ms at launch, OR subprocess fails to start | — |
| **V179c-2** | Hot-reload swaps buffer without dropped batch reads | Batch read error count = 0 across entire run (`assemble_mixed_batch` exception counter); INV-S179c-3 (identity stable within step) passes | Any `AttributeError` / `RuntimeError` from `bot_buffer.sample_batch` during a step in which a refresh fired | — |
| **V179c-3** | SealBot WR @ step 30K post-refresh ≥ §S178 step-30K baseline | Wilson95 lower bound of refresh-on WR ≥ §S178 step-30K WR point estimate, n=100 SealBot games (matches eval.yaml `sealbot.n_games: 50`, double for tight CI) | Wilson95 upper bound of refresh-on WR < §S178 baseline (strict regression) | Overlap of CIs |
| **V179c-4** | colony_frac @ step 30K does not regress | Refresh-on colony_frac ≤ §S178 step-30K colony_frac + 5pp tolerance | Refresh-on colony_frac ≥ §S178 step-30K + 10pp | 5-10pp delta |
| **V179c-5** | Regen runs against PROMOTED model (not stale anchor) | Sidecar `extra.anchor_sha256` of refreshed NPZ matches `best_model.pt` sha at fire time; verified by post-swap sha comparison | Sidecar anchor_sha does not match `best_model.pt` at fire time | — |

All verdicts pre-registered before code lands. Freeze at variant-yaml commit SHA per L13
(see `docs/07_PHASE4_SPRINT_LOG.md:637`).

---

## 8. Risk register

| # | Risk | Likelihood | Mitigation | Source |
|---|---|---|---|---|
| 1 | Regen compute steals GPU from trainer | HIGH | Subprocess runs SealBot @ 0.5s/move (CPU-only per `generate_bot_corpus.py:301-322`) + anchor MCTS @ 200 sims (GPU bursts ~ 1-2GB VRAM). At 5080+9900X (primary host) GPU contention reduces selfplay throughput ~ 5-15% during regen window. Acceptable for ~ 2-10hr per 25K-step window. Operator-driven on idle GPU windows in worst case. | `data/bot_corpus_s178_sealbot_vs_v6.npz.metadata.json` elapsed_sec=2786 baseline |
| 2 | NPZ corrupt-write during atomic rename failure | LOW | SHA verify post-rename (§3.2); `.bak` retained for one cycle; quarantine pattern from `anchor.py:61-67` | §3 + `anchor.py:35-58` |
| 3 | Buffer instance lifetime race during hot-reload | LOW | Single-writer/single-reader (§4.3); swap site is post-eval-drain pre-next-step; INV-S179c-3 pin | §4.3 |
| 4 | Bot strength drift mid-run changes corrective signal distribution (feature/bug?) | MEDIUM (feature, monitor) | Documented §3.4 in S178_design.md:163-184 — refreshing against newer/stronger anchor TEACHES "what SealBot punishes about current weights" instead of "what SealBot punishes about v6.pt weights". V179c-3/V179c-4 catch regression if the signal shift hurts colony reduction. Risk #4 in §S178 risk register (`S178_design.md:313`) is the inverse risk this lever addresses. | §S178 risk #4, L8 head-drift |
| 5 | Refresh fires during a transient WR-noise crossing (5pp threshold gamed by eval noise) | MEDIUM | n=200 best_checkpoint eval (`eval.yaml:752`) at Wilson95 holds 5pp delta against ~ 1pp noise. Threshold is configurable; raise to 7pp if §S178 verdict shows tight WR distribution | `eval.yaml:752`, MEMORY `project_175_eval_fix` Wilson CI ~ 12pp at n=20 |
| 6 | Stale subprocess on training crash leaves zombie + lock on `<canonical>.NEW.tmp` | LOW | Coordinator's `flush_pending_eval()` final-block (`step_coordinator.py:215-216` pattern) extended: SIGTERM the subprocess on shutdown; unlink stale `.tmp` files. Detect on next launch and unlink + warn | §2.2 zombie reap |
| 7 | Filesystem write fails between rename of canonical and rename of sidecar (§3.1 steps 5-6) | LOW | Order matters: canonical-first, sidecar-second. If sidecar rename fails post canonical rename, the run has an unsidecar'd NPZ. Mitigation: `corpus_io.py:178-250` load path requires sidecar (`f"{npz_path} has no metadata sidecar"`). Rollback: rename `.bak` back to canonical + log fatal | `corpus_io.py:204-205` |
| 8 | `min_new_games: 200` produces small pool (~ 6K positions) → behavioural cloning overfit | MEDIUM | 200 games × ~30 plies-decisive = ~ 6K positions vs §S178's 22K. With `bot_batch_share=0.15` at batch=256, n_bot ≈ 38; sample-with-replacement on 6K is fine. Operator override: raise to 400 if entropy probes flag overfit | §S178 risk #2, design §4.3 |
| 9 | Refresh interacts with `best_checkpoint` rotation: regen-target model may itself rotate before swap completes | LOW | Snapshot `target_anchor_sha` at fire time (§2.2). At rare race (model promoted DURING regen), the next refresh fires on next promotion + cooldown. V179c-5 catches this: sidecar anchor_sha matches AT FIRE TIME, not at completion. Acceptable — promotions are 5K-step granular, refresh window is 25K | §1.1 trigger rule |

---

## 9. Implementation cost estimate

### 9.1 LOC + file list

| # | Task | Files | LOC est |
|---|---|---|---|
| TC1 | StepCoordinator state extension (`_refresh_proc`, `_last_best_wr`, `_n_refreshes_so_far`, `_refresh_target_anchor_sha`) + fire path | `hexo_rl/training/step_coordinator.py` (extend §178 T7 block at lines 645-666) | ~ 80 |
| TC2 | Refresh poll integration (per-step `Popen.poll()` + swap + hot-reload) | `hexo_rl/training/step_coordinator.py` (new method `_tick_bot_refresh`) | ~ 60 |
| TC3 | Atomic swap + sha verify helper | `hexo_rl/training/batch_assembly.py` (NEW function `swap_bot_corpus_atomic`) | ~ 50 |
| TC4 | `generate_bot_corpus.py` subprocess CLI compatibility (add `--target-anchor-from-best-model` flag, read `best_model.pt` path from arg) | `scripts/generate_bot_corpus.py` (extend `_parse_args`) | ~ 20 |
| TC5 | Config plumbing | `hexo_rl/training/loop.py` (extend lines 243-249) + `configs/variants/v6_botmix_s179.yaml` (NEW) | ~ 30 + 150 (yaml) |
| TC6 | Force-trigger sentinel | `hexo_rl/training/step_coordinator.py` (top of `step()`) | ~ 15 |
| TC7 | Dashboard events | `docs/08_DASHBOARD_SPEC.md` extension + emit-site additions (no new module) | ~ 30 |
| TC8 | Tests: hot-reload, swap atomicity, INV-S179c-1..4, refresh-disabled bitwise identity | `tests/test_step_coordinator_bot_refresh.py` (NEW) + extend `tests/test_step_coordinator_bot_share.py` | ~ 250 |
| TC9 | Sprint log entry + L24+ candidate slot | `docs/07_PHASE4_SPRINT_LOG.md` | ~ 60 |

**Total: ~ 745 LOC across ~ 9 commits.**

### 9.2 Estimated dev hours

- TC1 + TC2 + TC6: tight scope, well-defined diff against existing hook → ~ 4 hr
- TC3: atomic-swap reuse of `anchor.py` pattern → ~ 2 hr
- TC4: small CLI extension → ~ 1 hr
- TC5: config + variant + plumbing → ~ 2 hr
- TC7: dashboard schema + emit sites → ~ 1 hr
- TC8: test suite (5 INVs + 2 happy-path + 2 failure paths) → ~ 6 hr
- TC9: sprint log → ~ 1 hr
- Bench gate (positions/hr regression check on hot-reload overhead) → ~ 2 hr
- Operator-run sustained smoke (200-step refresh-fire smoke + 5K-step refresh-cycle smoke) → ~ 4 hr operator + ~ 2 hr Claude

**Total: ~ 19-25 dev hours + ~ 4 hr operator.**

### 9.3 Test coverage

INV-S179c-4 (refresh-disabled = bitwise identical to §S178) is the linchpin: blocks regression
from the wiring extension itself. Test pattern:

```python
def test_refresh_disabled_bitwise_identical(tmp_path, snapshot):
    """INV-S179c-4: refresh enabled=False produces identical trace vs §S178 frozen baseline."""
    coord = _make_coordinator(
        config_overrides={"bot_corpus_refresh_enabled": False},
        logger=Mock(),
    )
    # Run 200 fixture-seeded steps; capture loss, buffer sizes, batch checksums.
    trace = _run_n_steps(coord, n=200)
    snapshot.assert_match(trace, "s178_baseline_n200.json")
```

Bench gate: confirm hot-reload (5-sec pause × N refreshes ≈ 30 sec total over 100K-step run) is
≤ 0.1% throughput overhead vs §S178 sustained. Acceptable per
`docs/rules/perf-targets.md` 10-metric gate methodology.

---

## 10. Forward pointers

- §S179a (visit-count CE policy target) and §S179b (policy surprise weighting) are sibling
  escalation levers per `reports/s178_design.md §3.3-3.4` ladder.
- §S179c implementation gated behind §S178 verdict V178-3 (colony_frac ≤ 50% at step 30K).
  V178-3 PASS → defer S179c. V178-3 FAIL/NULL → proceed.
- §S179c does NOT block on Source B subprocess infra (`S178_design.md:254`, deferred). Refresh
  reuses Source A pattern (single SealBot vs single anchor).

---

## 11. Open hygiene items (NOT §S179c-blocking)

| # | Item | Path |
|---|---|---|
| HC1 | Subprocess GPU contention measurement on 5080 + 4060 reference hardware | benchmark fixture |
| HC2 | Sidecar metadata schema_version bump if `extra.refresh_lineage` field added (track chain of regens) | `hexo_rl/bootstrap/corpus_io.py:106-155` |
| HC3 | Force-trigger sentinel path configurable (currently hardcoded `/tmp/hexo_rl_force_bot_refresh`) | `step_coordinator.py` |
| HC4 | Multi-process regen (n_workers > 1, deferred per `generate_bot_corpus.py:310-315`) | `scripts/generate_bot_corpus.py` |

---

## 12. Locked decisions

- Async model: `subprocess.Popen` + per-step `poll()` (NOT asyncio).
- Swap owner: coordinator (NOT subprocess).
- Reload trigger: direct call from coordinator (NOT mtime polling).
- Lock model: single-writer / single-reader (NO mutex).
- Default state: `enabled: false` (matches §S178 wiring).
- Trigger threshold: 5pp Δ best_model WR (matches `promotion_winrate` over `bootstrap_floor`).
- Cooldown: 25K steps. Max regens: 6 per run.
- Backup retention: one cycle (`.bak`).
- Filesystem invariant: same-FS rename (INV-S179c-2).
