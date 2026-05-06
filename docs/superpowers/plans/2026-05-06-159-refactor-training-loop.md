# §159 Refactor — `hexo_rl/training/loop.py` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose 1465-LOC god module into 5 files (~400 + ~800) per `audit/SUMMARY.md` L9 #1 and `docs/refactor-template.md`. Zero behavior change. All tests pass. Bench gate skipped (orchestration, not hot-path — per template L113).

**Architecture:** Pure-move refactor. Each new file owns a single subsystem responsibility; `loop.py` keeps `run_training_loop` public API and the inner `_run_loop` step coordinator. Re-export shims preserve `_drain_pending_eval` import path until the final commit retires them. Tests updated only in the final commit.

**Tech Stack:** Python 3.11, torch, structlog, threading.signal — no new deps.

---

## Decision boundary — name resolution (USER-CONFIRMED 2026-05-06)

Kickoff names a new `training/checkpoint.py`. Confirmed deviation:

**`training/checkpoints.py` (plural, 176 LOC) stays untouched.** It owns trainer-side state I/O (`save_inference_weights`, `normalize_model_state_dict_keys`, full-state save/load/prune). Industry-standard "checkpoint" naming, multiple cross-package importers, well-named for its job.

**`training/anchor.py` (singular, NEW)** owns the best_model.pt anchor lifecycle. Naming matches existing log-event vocabulary throughout the codebase: `anchor_load_failed`, `anchor_quarantined`, `anchor_recovered_from_bak`, `anchor_loaded_from_bootstrap`, `anchor_persisted_from_fallback`, `anchor_fresh_init_no_bootstrap`, constant `_BOOTSTRAP_ANCHOR_CANDIDATES`. The anchor is conceptually distinct: graduation-gate reference, promoted on win, has fallback chain + quarantine semantics — not a general checkpoint.

`checkpoint_log.json` referenced in the kickoff's split spec is not produced by `loop.py` today — no rotation logic exists in scope; that line was aspirational and is dropped from this plan.

## Decision boundary — anchor + eval-pipeline split (USER-CONFIRMED 2026-05-06)

Originally planned to defer L318-456 (anchor + eval-pipeline setup) to §16x. **User overrode: split now.** Two separable concerns inside that block:

A. **EvalPipeline construction** (L321-341, ~40 LOC, pure setup) → new `eval/pipeline_setup.py::build_eval_pipeline`
B. **Anchor resolution** (L343-436, ~95 LOC, threads trainer + inf_model + config; mutates inf_model state_dict on arch-match sync) → extends `training/anchor.py` with `resolve_anchor` orchestration on top of the I/O primitives

`AnchorState` dataclass = `(best_model, best_model_step, best_model_path)` returned from `resolve_anchor`.

---

## File layout (post-refactor)

| File | LOC est | Responsibility |
|------|---------|----------------|
| `hexo_rl/training/loop.py` | ~280 | `run_training_loop` skeleton, inner `_run_loop` step coordinator, `_suppress_semaphore_leak_warning` |
| `hexo_rl/training/anchor.py` (NEW) | ~260 | I/O primitives (`save_best_model_atomic`, `load_best_model_resilient`, `_quarantine_corrupt`, `_try_load_anchor`) + orchestration (`resolve_anchor`, `AnchorState`) |
| `hexo_rl/training/signals.py` (NEW) | ~80 | `ShutdownState` dataclass + `install_signal_handlers()` |
| `hexo_rl/training/orchestrator.py` (NEW) | ~350 | `drain_pending_eval`, `try_save_buffer`, `replay_pretrain_events`, `emit_axis_distribution`, `emit_training_events` |
| `hexo_rl/training/lifecycle.py` (NEW) | ~250 | `build_inference_model`, `cuda_warmup`, `cuda_stream_audit`, `build_eval_model`, `LoopSubsystems` |
| `hexo_rl/eval/pipeline_setup.py` (NEW) | ~50 | `build_eval_pipeline` (eval.yaml load + variant override + EvalPipeline ctor) |

Caller inventory (Gate 1):
- `scripts/train.py:87` — imports `run_training_loop`. Public API preserved → no edit.
- `tests/test_training_loop_graduation.py:203,266` — imports `_drain_pending_eval` from `hexo_rl.training.loop`. Re-export shim during refactor; final commit moves test imports to `hexo_rl.training.orchestrator`.

---

## Branch + preamble

- [ ] **Step 1: Create refactor branch**

```bash
git checkout master && git pull --ff-only && git checkout -b refactor/training-loop-split
```

- [ ] **Step 2: Verify baseline LOC + tests**

```bash
wc -l hexo_rl/training/loop.py
make test 2>&1 | tail -5
```

Expected: `1464` (matches audit's 1465 — within ±10%, audit valid). Tests pass.

If `wc -l` differs >10% from 1465 → STOP, audit stale (template Gate 1).

- [ ] **Step 3: Confirm caller inventory matches plan**

```bash
rg -n 'from hexo_rl\.training\.loop|import.*training\.loop' --type py
```

Expected exactly:
```
scripts/train.py:87:from hexo_rl.training.loop import run_training_loop
tests/test_training_loop_graduation.py:203:    from hexo_rl.training.loop import _drain_pending_eval
tests/test_training_loop_graduation.py:266:    from hexo_rl.training.loop import _drain_pending_eval
```

If extra callers appear → STOP, surface to user (template Gate 3 stop signal: "caller count grows").

---

### Task 1: Extract `training/anchor.py`

**Files:**
- Create: `hexo_rl/training/anchor.py`
- Modify: `hexo_rl/training/loop.py` (delegate to anchor, keep names re-exported)

Leaf-most extract: anchor helpers reach nothing else in `loop.py`; only Trainer + torch.

- [ ] **Step 1: Create `hexo_rl/training/anchor.py`**

Contents — verbatim copy of L42–L179 from current `loop.py`, with imports rebuilt at top of new file:

```python
"""Best-model anchor management — atomic save, resilient load, quarantine.

Owns ``best_model.pt`` lifecycle: torch.save round-trip verify + .bak rotation
on save; best → .bak → bootstrap_*.pt fallback chain on load; corrupt-anchor
quarantine with timestamp suffix.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from hexo_rl.training.trainer import Trainer

log = structlog.get_logger(__name__)

# Bootstrap candidates tried (in order) when no usable best_model.pt exists.
# Fresh runs anchor against the trained bootstrap, not a random fresh-init
# copy of trainer.model.
_BOOTSTRAP_ANCHOR_CANDIDATES: tuple[str, ...] = (
    "checkpoints/bootstrap_model_v7full.pt",
    "checkpoints/bootstrap_model.pt",
)


def save_best_model_atomic(model: torch.nn.Module, path: Path) -> None:
    # body identical to current loop._save_best_model_atomic
    ...


def _quarantine_corrupt(path: Path) -> Path:
    # body identical to current loop._quarantine_corrupt
    ...


def _try_load_anchor(
    candidate: Path,
    *,
    checkpoint_dir: str,
    device: torch.device,
    fallback_config: dict[str, Any],
) -> Optional[Trainer]:
    # body identical to current loop._try_load_anchor
    ...


def load_best_model_resilient(
    best_model_path: Path,
    *,
    checkpoint_dir: str,
    device: torch.device,
    config: dict[str, Any],
) -> Optional[Trainer]:
    # body identical to current loop._load_best_model_resilient,
    # internal calls rewritten to use _try_load_anchor + _quarantine_corrupt
    # within this module; _BOOTSTRAP_ANCHOR_CANDIDATES references local const.
    ...
```

**The agent must paste the exact function bodies from `loop.py:51-179`.** Two public renames:
- `_save_best_model_atomic` → `save_best_model_atomic` (drop leading underscore — it's now a module-public API)
- `_load_best_model_resilient` → `load_best_model_resilient` (same reason)
- `_try_load_anchor` and `_quarantine_corrupt` keep underscores — module-private.

- [ ] **Step 2: Update `loop.py` to delegate**

In `hexo_rl/training/loop.py`:

Replace L42–L179 (the `_BOOTSTRAP_ANCHOR_CANDIDATES` constant and four helper functions) with:

```python
from hexo_rl.training.anchor import (
    _BOOTSTRAP_ANCHOR_CANDIDATES,
    load_best_model_resilient as _load_best_model_resilient,
    save_best_model_atomic as _save_best_model_atomic,
)
```

Keep the underscore-prefixed local names so the rest of `loop.py` (L356, L374, L432, L434, L1117) needs zero edits.

- [ ] **Step 3: Run full test suite**

```bash
make test 2>&1 | tail -10
```

Expected: same pass count as Step 2 of branch preamble. Any failure → revert this commit, re-attempt.

- [ ] **Step 4: Commit**

```bash
git add hexo_rl/training/anchor.py hexo_rl/training/loop.py
git commit -m "refactor(training): extract best-model anchor to anchor.py (§159)"
```

---

### Task 2: Extract `training/signals.py`

**Files:**
- Create: `hexo_rl/training/signals.py`
- Modify: `hexo_rl/training/loop.py` L557-571

Decision: shutdown is communicated via mutable list closures today (`_running`, `_stop_count`, `_shutdown_save`). Replace with a small dataclass returned from the installer. Pure structural change — no semantics shift.

- [ ] **Step 1: Create `hexo_rl/training/signals.py`**

```python
"""Signal handler registration for the self-play training loop.

Wraps SIGINT/SIGTERM → cooperative shutdown state. Two presses force-exit;
one press flips ``running=False`` and ``shutdown_save=True`` so the loop
saves a checkpoint before returning.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class ShutdownState:
    running: bool = True
    stop_count: int = 0
    shutdown_save: bool = False


def install_signal_handlers(state: ShutdownState) -> None:
    """Install SIGINT/SIGTERM handlers that flip ``state``.

    Two consecutive signals force-exit; one signal sets ``running=False``
    and ``shutdown_save=True``. The training loop is responsible for
    polling ``state`` between iterations.
    """

    def _stop(sig: int, frame: Any) -> None:
        state.stop_count += 1
        if state.stop_count >= 2:
            sys.exit(1)
        log.info(
            "shutdown_requested",
            msg="finishing current step… press Ctrl+C again to force",
        )
        state.shutdown_save = True
        state.running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
```

- [ ] **Step 2: Replace handler block in `loop.py`**

In `hexo_rl/training/loop.py` L557-571, replace:

```python
    # ── Graceful shutdown ──────────────────────────────────────────────────────
    _running = [True]
    _stop_count = [0]
    _shutdown_save = [False]

    def _stop(sig: int, frame: Any) -> None:
        _stop_count[0] += 1
        if _stop_count[0] >= 2:
            sys.exit(1)
        log.info("shutdown_requested", msg="finishing current step… press Ctrl+C again to force")
        _shutdown_save[0] = True
        _running[0] = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)
```

with:

```python
    # ── Graceful shutdown ──────────────────────────────────────────────────────
    from hexo_rl.training.signals import ShutdownState, install_signal_handlers
    _shutdown = ShutdownState()
    install_signal_handlers(_shutdown)
```

Then **rewrite every reference** in `loop.py`:
- `_running[0]` → `_shutdown.running` (sites: L668, L783, L815)
- `_shutdown_save[0]` → `_shutdown.shutdown_save` (sites: L673, L1018)
- `_stop_count[...]` references — only inside the deleted closure, no remaining sites

Delete the now-unused `import signal` and `import sys` from the top of `loop.py` ONLY IF no other use remains. (`sys.exit` is now in signals.py; `signal.signal` is gone from loop.py. Confirm with `rg '\bsys\.|signal\.' hexo_rl/training/loop.py` before deleting; if any other use is found, keep them.)

- [ ] **Step 3: Run tests**

```bash
make test 2>&1 | tail -10
```

Expected: same pass count.

- [ ] **Step 4: Commit**

```bash
git add hexo_rl/training/signals.py hexo_rl/training/loop.py
git commit -m "refactor(training): extract signal handlers to signals.py (§159)"
```

---

### Task 3: Extract `training/orchestrator.py`

**Files:**
- Create: `hexo_rl/training/orchestrator.py`
- Modify: `hexo_rl/training/loop.py` (delegate, keep `_drain_pending_eval` re-export shim for test)

Move five functions from end of `loop.py`:
- `_drain_pending_eval` (L1075-1129) → public `drain_pending_eval`
- `_try_save_buffer` (L1132-1153) → public `try_save_buffer`
- `_replay_pretrain_events` (L1156-1191) → public `replay_pretrain_events`
- `_emit_axis_distribution` (L1194-1285) → public `emit_axis_distribution`
- `_emit_training_events` (L1288-1455) → public `emit_training_events`

`drain_pending_eval` references `_save_best_model_atomic` (now in `anchor.py`) and `pool._inference_server.load_state_dict_safe` (private API — preserved as-is, the coupling is real and predates this refactor; surface in follow-ups).

- [ ] **Step 1: Create `hexo_rl/training/orchestrator.py`**

```python
"""Self-play loop side-channel orchestration: eval drain, axis-distribution,
buffer save, pretrain-replay, training-event emission.

Each function is called from training/loop.py at a specific cadence; none
hold loop state. Extracted from training/loop.py per §159 refactor. No
behavior change.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any, Optional

import structlog

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.early_game_probe import (
    EARLY_GAME_ENTROPY_WARN_THRESHOLD,
    EarlyGameProbe,
)
from hexo_rl.monitoring.events import emit_event
from hexo_rl.monitoring.gpu_monitor import GPUMonitor
from hexo_rl.training.anchor import save_best_model_atomic

log = structlog.get_logger(__name__)


def drain_pending_eval(
    eval_thread: Optional[threading.Thread],
    eval_result: list[Optional[dict[str, Any]]],
    eval_model: Optional[HexTacToeNet],
    best_model: Optional[HexTacToeNet],
    best_model_path: Path,
    best_model_step: Optional[int],
    pool: Any,
    train_step: int,
) -> tuple[Optional[threading.Thread], Optional[int]]:
    # body identical to current loop._drain_pending_eval, with
    # _save_best_model_atomic call updated to save_best_model_atomic
    ...


def try_save_buffer(
    buffer: Any,
    mixing_cfg: dict[str, Any],
    trigger: str,
    recent_buffer: Optional[Any] = None,
) -> None:
    # body identical to current loop._try_save_buffer
    ...


def replay_pretrain_events(args: argparse.Namespace) -> None:
    # body identical to current loop._replay_pretrain_events
    ...


def emit_axis_distribution(
    train_step: int,
    pool: Any,
    config: dict[str, Any],
    baseline: dict[str, float],
    tb_writer: Any,
) -> Optional[float]:
    # body identical to current loop._emit_axis_distribution
    ...


def emit_training_events(
    train_step: int,
    loss_info: dict[str, float],
    w_pre: float,
    games_played: int,
    last_iter_games: int,
    pool: Any,
    buffer: Any,
    gpu_monitor: GPUMonitor,
    config: dict[str, Any],
    mcts_config: dict[str, Any],
    capacity: int,
    games_per_hour_fn: Any,
    qfire_delta: int,
    early_game_probe: Optional[EarlyGameProbe] = None,
    trainer_model: Optional[Any] = None,
) -> None:
    # body identical to current loop._emit_training_events
    ...
```

Paste the exact function bodies from `loop.py:1075-1455`.

- [ ] **Step 2: Replace bodies in `loop.py` with re-export shim**

Delete L1075-1455 from `loop.py`. Replace with:

```python
# ── Re-exports preserved during §159 refactor ─────────────────────────────────
# tests/test_training_loop_graduation.py imports `_drain_pending_eval` directly.
# Shim retired in the final §159 commit once tests are updated.
from hexo_rl.training.orchestrator import (
    drain_pending_eval as _drain_pending_eval,
    try_save_buffer as _try_save_buffer,
    replay_pretrain_events as _replay_pretrain_events,
    emit_axis_distribution as _emit_axis_distribution,
    emit_training_events as _emit_training_events,
)
```

The five existing call sites inside `run_training_loop` (`_drain_pending_eval`, `_try_save_buffer`, `_replay_pretrain_events`, `_emit_axis_distribution`, `_emit_training_events`) need NO edit — they bind to the imported alias.

`_suppress_semaphore_leak_warning` stays in `loop.py` — it's the post-loop teardown, not orchestration.

- [ ] **Step 3: Run tests**

```bash
make test 2>&1 | tail -10
```

Expected: same pass count. Tests still importing `_drain_pending_eval` from `loop.py` resolve via shim.

- [ ] **Step 4: Commit**

```bash
git add hexo_rl/training/orchestrator.py hexo_rl/training/loop.py
git commit -m "refactor(training): extract orchestrator hooks to orchestrator.py (§159)"
```

---

### Task 4: Extract `eval/pipeline_setup.py`

**Files:**
- Create: `hexo_rl/eval/pipeline_setup.py`
- Modify: `hexo_rl/training/loop.py` L318-341 (replace with builder call)

Pure move of the eval-yaml load + variant deep-merge + `EvalPipeline` construction. No anchor logic — that's Task 5.

- [ ] **Step 1: Create `hexo_rl/eval/pipeline_setup.py`**

```python
"""Build the EvalPipeline from eval.yaml + per-variant overrides.

Sweep / variant configs may declare an ``eval_pipeline:`` block to override
eval cost (n_games, model_sims, opponent enables, eval_interval). This
module deep-merges that block onto the eval.yaml load before constructing
EvalPipeline; without that merge, sweep mode would pay the production
~134 min/round eval cost regardless of how short the sweep is.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from hexo_rl.eval.eval_pipeline import EvalPipeline

log = structlog.get_logger(__name__)


def build_eval_pipeline(
    config: dict[str, Any],
    device: torch.device,
    run_id: str,
    train_cfg: dict[str, Any],
) -> tuple[Optional[EvalPipeline], dict[str, Any], int]:
    """Load eval.yaml, apply variant override, build EvalPipeline.

    Returns ``(eval_pipeline, eval_ext_config, eval_interval)``:
      - ``eval_pipeline`` is None when ``configs/eval.yaml`` is absent or its
        ``eval_pipeline.enabled`` flag is False.
      - ``eval_ext_config`` is the merged eval.yaml content (empty dict if
        the file is absent). Caller reads ``gating.best_model_path`` from it.
      - ``eval_interval`` is the cadence to use for eval ticks. Falls back
        to ``train_cfg["eval_interval"]`` → ``config["eval_interval"]`` →
        100 when no eval pipeline is built.
    """
    from hexo_rl.utils.config import load_config as _load_config, _deep_merge

    eval_yaml_path = Path("configs/eval.yaml")
    eval_pipeline: EvalPipeline | None = None
    eval_ext_config: dict[str, Any] = {}
    eval_interval: int = int(
        train_cfg.get("eval_interval", config.get("eval_interval", 100))
    )

    if eval_yaml_path.exists():
        eval_ext_config = _load_config(str(eval_yaml_path))
        main_ep_override = config.get("eval_pipeline", {})
        if main_ep_override:
            _deep_merge(
                eval_ext_config.setdefault("eval_pipeline", {}),
                main_ep_override,
            )
        ep_cfg = eval_ext_config.get("eval_pipeline", {})
        if ep_cfg.get("enabled", False):
            eval_pipeline = EvalPipeline(eval_ext_config, device, run_id=run_id)
            eval_interval = int(ep_cfg.get("eval_interval", 1000))
            log.info(
                "eval_pipeline_enabled",
                interval=eval_interval,
                overrides_applied=bool(main_ep_override),
            )

    return eval_pipeline, eval_ext_config, eval_interval
```

Body is verbatim copy of `loop.py` L318-341 logic. The `_eval_interval_cfg` initial value matches the original L324 default; the post-build override matches L339-341.

- [ ] **Step 2: Replace block in `loop.py`**

In `hexo_rl/training/loop.py`, replace L318-341 (the entire `# ── Eval pipeline` block from the comment through the EvalPipeline construct) with:

```python
    # ── Eval pipeline ─────────────────────────────────────────────────────────
    from hexo_rl.eval.pipeline_setup import build_eval_pipeline
    eval_pipeline, eval_ext_config, _eval_interval_cfg = build_eval_pipeline(
        config, device, run_id, train_cfg,
    )
```

The downstream block at L343 (`best_model_path = Path(eval_ext_config.get(...))`) and L350 (`if eval_pipeline is not None:`) need NO edit — they bind to the locals just unpacked.

The `EvalPipeline` import at L319 is moved into `pipeline_setup.py`. Confirm the original L319 `from hexo_rl.eval.eval_pipeline import EvalPipeline` is no longer needed in `loop.py`:

```bash
rg -n 'EvalPipeline' hexo_rl/training/loop.py
```

If only type-hint usage remains, keep the import as type-only or use `if TYPE_CHECKING`. If the import is fully unused, delete it.

- [ ] **Step 3: Run tests**

```bash
make test 2>&1 | tail -10
```

Expected: same pass count.

- [ ] **Step 4: Commit**

```bash
git add hexo_rl/eval/pipeline_setup.py hexo_rl/training/loop.py
git commit -m "refactor(eval): extract eval-pipeline builder to pipeline_setup.py (§159)"
```

---

### Task 5: Extend `training/anchor.py` with `resolve_anchor`

**Files:**
- Modify: `hexo_rl/training/anchor.py` (add `AnchorState` + `resolve_anchor`)
- Modify: `hexo_rl/training/loop.py` L343-456 (replace anchor-resolution block with single call)

Builds on Task 1 (anchor I/O) and Task 4 (eval pipeline tuple). `resolve_anchor` mutates `inf_model` (state_dict load on arch-match) — documented in docstring. AnchorState bundles the three values currently passed as separate locals.

- [ ] **Step 1: Add `AnchorState` + `resolve_anchor` to `hexo_rl/training/anchor.py`**

Append to `hexo_rl/training/anchor.py`:

```python
from dataclasses import dataclass

from hexo_rl.model.network import HexTacToeNet


@dataclass
class AnchorState:
    """Resolved best-model anchor + provenance.

    ``best_model`` is None only when no eval pipeline is configured (the
    caller supplies eval_pipeline=None) — this matches the pre-refactor
    invariant where best_model stayed None outside the eval branch.
    ``best_model_step`` is None when no anchor step was recoverable from
    the loaded checkpoint.
    """
    best_model: Optional[HexTacToeNet]
    best_model_step: Optional[int]
    best_model_path: Path


def resolve_anchor(
    *,
    eval_pipeline: Any,                      # EvalPipeline | None
    eval_ext_config: dict[str, Any],
    inf_model: torch.nn.Module,              # mutated when arch matches
    trainer: Trainer,
    args: Any,                               # argparse.Namespace
    config: dict[str, Any],
    device: torch.device,
    board_size: int,
    res_blocks: int,
    filters: int,
    in_channels: int,
    input_channels: Any,
    se_reduction_ratio: int,
) -> AnchorState:
    """Resolve the best-model anchor and sync ``inf_model`` to it.

    Steps:
      1. Resolve ``best_model_path`` from eval.yaml gating config.
      2. If eval_pipeline is None → return AnchorState(None, None, path).
      3. Try resilient load (best.pt → .bak → bootstrap candidates).
      4. On success: unwrap torch.compile, persist to live path if recovered
         from a fallback, sync inf_model when in_channels match (preserves
         the input_channel_index buffer when present), warn on
         trainer.step ≠ best_model_step (M2 invariant).
      5. On total failure: fresh-init from trainer.model, save atomically.

    Mutates ``inf_model`` via ``load_state_dict`` when the architecture
    matches the loaded anchor. Sweep variants intentionally leave inf_model
    on trainer.model weights (arch-mismatch logged, no sync).
    """
    best_model_path = Path(
        eval_ext_config.get("eval_pipeline", {}).get("gating", {}).get(
            "best_model_path", "checkpoints/best_model.pt"
        )
    )
    best_model: HexTacToeNet | None = None
    best_model_step: int | None = None
    if eval_pipeline is None:
        return AnchorState(None, None, best_model_path)

    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    best_ref = load_best_model_resilient(
        best_model_path,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        config=config,
    )
    if best_ref is not None:
        best_model = getattr(best_ref.model, "_orig_mod", best_ref.model)
        best_model.eval()
        best_model_step = best_ref.step
        if not best_model_path.exists():
            save_best_model_atomic(best_model, best_model_path)
            log.info("anchor_persisted_from_fallback", path=str(best_model_path))
        _inf_base = getattr(inf_model, "_orig_mod", inf_model)
        if _inf_base.in_channels == best_model.in_channels:
            _best_sd = best_model.state_dict()
            _inf_idx = getattr(_inf_base, "input_channel_index", None)
            if "input_channel_index" not in _best_sd and _inf_idx is not None:
                _best_sd = dict(_best_sd)
                _best_sd["input_channel_index"] = _inf_idx.detach().clone()
            _inf_base.load_state_dict(_best_sd)
        else:
            log.info(
                "inf_model_anchor_arch_mismatch_skip_sync",
                inf_in_channels=_inf_base.in_channels,
                anchor_in_channels=best_model.in_channels,
                msg="inf_model starts from trainer.model (sweep variant)",
            )
        log.info("best_model_loaded", path=str(best_model_path), step=best_model_step)
        if best_model_step is not None and trainer.step != best_model_step:
            log.warning(
                "resume_anchor_step_mismatch",
                trainer_step=trainer.step,
                best_model_step=best_model_step,
                msg=(
                    "trainer.model and best_model.pt were loaded from different "
                    "training steps. First eval will compare the current trainer "
                    "weights against this anchor; confirm this is intended."
                ),
            )
    else:
        log.warning(
            "anchor_fresh_init_no_bootstrap",
            tried=list(_BOOTSTRAP_ANCHOR_CANDIDATES),
            msg=(
                "No anchor or bootstrap available — initialising best_model.pt "
                "from current trainer.model. Drop a bootstrap_model.pt (or one "
                "of _BOOTSTRAP_ANCHOR_CANDIDATES) into checkpoints/ to anchor "
                "wr_best meaningfully."
            ),
        )
        base_model = getattr(trainer.model, "_orig_mod", trainer.model)
        best_model = HexTacToeNet(
            board_size=board_size, res_blocks=res_blocks, filters=filters,
            in_channels=in_channels, input_channels=input_channels,
            se_reduction_ratio=se_reduction_ratio,
        ).to(device)
        best_model.load_state_dict(base_model.state_dict())
        best_model.eval()
        save_best_model_atomic(best_model, best_model_path)
        best_model_step = trainer.step
        log.info(
            "best_model_initialized",
            path=str(best_model_path),
            step=best_model_step,
        )

    return AnchorState(best_model, best_model_step, best_model_path)
```

The body is a verbatim restructure of `loop.py` L343-436 — every log call, every conditional, the M2 step-mismatch warn, the fresh-init fallback. Only changes: `_save_best_model_atomic` → `save_best_model_atomic` (Task 1 rename), encapsulated `if eval_pipeline is None` early-return for the no-eval branch.

- [ ] **Step 2: Replace anchor block in `loop.py`**

In `hexo_rl/training/loop.py`, replace L343-440 (the `best_model_path = Path(...)` line through the closing `_eval_thread = ...; _eval_result = [None]` declarations — but KEEP those last two declarations in `loop.py`) with:

```python
    # ── Anchor resolution ─────────────────────────────────────────────────────
    from hexo_rl.training.anchor import resolve_anchor
    _anchor = resolve_anchor(
        eval_pipeline=eval_pipeline,
        eval_ext_config=eval_ext_config,
        inf_model=inf_model,
        trainer=trainer,
        args=args,
        config=config,
        device=device,
        board_size=board_size,
        res_blocks=res_blocks,
        filters=filters,
        in_channels=in_channels,
        input_channels=input_channels,
        se_reduction_ratio=se_reduction_ratio,
    )
    best_model = _anchor.best_model
    best_model_step = _anchor.best_model_step
    best_model_path = _anchor.best_model_path

    _eval_thread: threading.Thread | None = None
    _eval_result: list[dict | None] = [None]
```

The downstream eval-side model alloc block at L443-456 stays untouched (becomes Task 6's lifecycle move).

- [ ] **Step 3: Run tests**

```bash
make test 2>&1 | tail -10
```

Expected: same pass count. The graduation tests in `tests/test_training_loop_graduation.py` exercise `_drain_pending_eval` directly — they don't run `resolve_anchor`. If a test fails referencing `best_model_path`, surface — likely a hidden-coupling discovery.

- [ ] **Step 4: Commit**

```bash
git add hexo_rl/training/anchor.py hexo_rl/training/loop.py
git commit -m "refactor(training): extract resolve_anchor orchestration to anchor.py (§159)"
```

---

### Task 6: Extract `training/lifecycle.py`

**Files:**
- Create: `hexo_rl/training/lifecycle.py`
- Modify: `hexo_rl/training/loop.py` (replace setup blocks + teardown blocks)

Lifecycle bundles subsystem boot + matching teardown. Two surface options: (a) functions that return handles + a separate teardown function, or (b) a context-manager class. Pick (a) — minimum coupling, no `with` block needed inside `run_training_loop`, easy to test piece-wise.

Move:
- Inference model build (L238-272) → `build_inference_model(trainer, device, config) -> torch.nn.Module`
- Eval-side model alloc (L449-456) → `build_eval_model(trainer, config, device) -> HexTacToeNet`
- GPU monitor (L458-460), Disk guard (L462-472), TB writer (L530-538), dashboards (L540-555), early-game probe (L474-487), value probe (L489-516) → `LoopSubsystems` dataclass + `build_subsystems(args, config, device, run_id) -> LoopSubsystems` + `LoopSubsystems.teardown(self) -> None`

Anchor + eval pipeline setup (L318-457) STAYS in `loop.py`. Rationale: reaches into trainer + eval_yaml load + eval_pipeline construction + `inf_model.load_state_dict` + log emission with run-specific state. Splitting it cleanly needs a separate pass — surface as follow-up. Keeping it in `loop.py` keeps this commit a pure move.

- [ ] **Step 1: Create `hexo_rl/training/lifecycle.py`**

```python
"""Training-loop subsystem lifecycle: model builds + monitor/probe/dashboard
boot and teardown.

Owns construction of the inference model, eval-side model, and a
``LoopSubsystems`` bundle covering GPU monitor, disk guard, dashboards,
probes, and TensorBoard writer. Anchor (best_model.pt) management lives
in training/anchor.py; eval pipeline construction stays in
training/loop.py because it threads run-specific state.

Extracted from training/loop.py per §159 refactor. No behavior change.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.monitoring.disk_guard import DiskGuard
from hexo_rl.monitoring.early_game_probe import EarlyGameProbe
from hexo_rl.monitoring.events import register_renderer
from hexo_rl.monitoring.gpu_monitor import GPUMonitor
from hexo_rl.monitoring.metrics_writer import MetricsWriter
from hexo_rl.monitoring.value_probe import ValueProbe
from hexo_rl.training.trainer import Trainer

log = structlog.get_logger(__name__)


def build_inference_model(
    trainer: Trainer,
    device: torch.device,
    config: dict[str, Any],
) -> torch.nn.Module:
    """Build inference-side HexTacToeNet, copy trainer weights, optionally
    torch.compile (mode="default" — see thread-safety note in original
    block, line numbers L260-271 of pre-refactor loop.py).
    """
    # body — paste from loop.py L238-272 verbatim, including comment block
    # explaining mode="default" forced for InferenceServer thread safety.
    # `inf_model` ends as the return value.
    ...


def cuda_warmup(inf_model: torch.nn.Module, device: torch.device, board_size: int) -> None:
    """Force CUDA kernel JIT compile on the main thread before workers start.
    Without this the first worker call blocks 90-120s mid-startup.
    """
    # body from loop.py L275-291
    ...


def cuda_stream_audit(config: dict[str, Any], device: torch.device) -> None:
    """B4 perf probe — log default vs current stream pointer when
    ``diagnostics.perf_timing`` is enabled. Q18 hypothesis check."""
    # body from loop.py L293-310
    ...


def build_eval_model(
    trainer: Trainer,
    config: dict[str, Any],
    device: torch.device,
) -> HexTacToeNet:
    """Allocate the eval-side HexTacToeNet once for reuse across rounds.
    L1: avoids 30 MB CUDA activation churn per eval round.
    """
    # body from loop.py L449-456
    ...


@dataclass
class LoopSubsystems:
    gpu_monitor: GPUMonitor
    disk_guard: DiskGuard
    early_game_probe: Optional[EarlyGameProbe]
    value_probe: Optional[ValueProbe]
    value_probe_interval: int
    composition_interval: int
    instrumentation_enabled: bool
    axis_baseline: dict[str, float]
    tb_writer: Optional[MetricsWriter]
    dashboards: list[Any] = field(default_factory=list)

    def teardown(self) -> None:
        """Stop all subsystems. Safe to call multiple times."""
        try:
            self.gpu_monitor.stop()
            self.gpu_monitor.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.disk_guard.stop()
        except Exception:
            pass
        for d in self.dashboards:
            try:
                d.stop()
            except Exception:
                pass


def build_subsystems(
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
    run_id: str,
) -> LoopSubsystems:
    """Build and start GPU monitor, disk guard, probes, TB writer, dashboards."""
    # body — sequential paste from loop.py L458-555 + L489-528, returning a
    # populated LoopSubsystems. Order matters (gpu_monitor.start() before
    # disk_guard.start() before probes; dashboards last so render thread
    # sees a populated event registry).
    ...
```

Paste exact bodies from listed line ranges of `loop.py`.

- [ ] **Step 2: Wire `loop.py` to `lifecycle.py`**

In `hexo_rl/training/loop.py`, replace L238-555 (inference model build through dashboard renderers, EXCEPT the eval-pipeline + anchor setup at L318-456) with calls to lifecycle helpers.

Concretely, after the docstring + arg-unpacking section ending at L237, the body becomes:

```python
    from hexo_rl.training.lifecycle import (
        LoopSubsystems,
        build_eval_model,
        build_inference_model,
        build_subsystems,
        cuda_stream_audit,
        cuda_warmup,
    )

    inf_model = build_inference_model(trainer, device, config)
    board_size = int(trainer.config.get("board_size", 19))
    cuda_warmup(inf_model, device, board_size)
    cuda_stream_audit(config, device)

    # ── Worker pool (unchanged) ────────────────────────────────────────────
    from hexo_rl.selfplay.pool import WorkerPool
    pool = WorkerPool(inf_model, config, device, buffer)
    if recent_buffer is not None:
        pool.recent_buffer = recent_buffer

    # ── Eval pipeline + anchor setup (UNCHANGED — kept in loop.py) ─────────
    # … L318-447 stay here verbatim (eval_yaml load + EvalPipeline construct +
    #   anchor resilient load + best_model_path setup) …

    # ── Eval-side model ────────────────────────────────────────────────────
    eval_model: HexTacToeNet | None = None
    if eval_pipeline is not None:
        eval_model = build_eval_model(trainer, config, device)

    # ── All other subsystems ───────────────────────────────────────────────
    subsys = build_subsystems(args, config, device, run_id)
    gpu_monitor = subsys.gpu_monitor
    disk_guard = subsys.disk_guard
    early_game_probe = subsys.early_game_probe
    value_probe = subsys.value_probe
    value_probe_interval = subsys.value_probe_interval
    composition_interval = subsys.composition_interval
    instrumentation_enabled = subsys.instrumentation_enabled
    _axis_baseline = subsys.axis_baseline
    _tb_writer = subsys.tb_writer
    dashboards = subsys.dashboards
```

The local-name aliases (`gpu_monitor = subsys.gpu_monitor`, etc.) keep the existing 30+ call sites in `_run_loop` unchanged — pure move guarantee.

In the `finally:` block (L1037-1045), replace:

```python
        pool.stop()
        gpu_monitor.stop()
        gpu_monitor.join(timeout=2.0)
        disk_guard.stop()
        for d in dashboards:
            try:
                d.stop()
            except Exception:
                pass
```

with:

```python
        pool.stop()
        subsys.teardown()
```

- [ ] **Step 3: Run tests**

```bash
make test 2>&1 | tail -10
```

Expected: same pass count. If any test that constructed real `GPUMonitor` / `DiskGuard` mocks fails → revert + surface. Most tests do not exercise lifecycle directly (graduation test bypasses `run_training_loop`).

- [ ] **Step 4: Commit**

```bash
git add hexo_rl/training/lifecycle.py hexo_rl/training/loop.py
git commit -m "refactor(training): extract subsystem lifecycle to lifecycle.py (§159)"
```

---

### Task 7: Update tests, retire shim, sanity-check

**Files:**
- Modify: `tests/test_training_loop_graduation.py:203,266`
- Modify: `hexo_rl/training/loop.py` (remove orchestrator re-export shim)

- [ ] **Step 1: Update test imports**

```bash
sed -i 's|from hexo_rl.training.loop import _drain_pending_eval|from hexo_rl.training.orchestrator import drain_pending_eval as _drain_pending_eval|' tests/test_training_loop_graduation.py
```

Verify:

```bash
rg -n '_drain_pending_eval|drain_pending_eval' tests/test_training_loop_graduation.py
```

Expected: imports point at `hexo_rl.training.orchestrator`; the local alias keeps the test bodies unchanged.

- [ ] **Step 2: Remove re-export shim from `loop.py`**

In `hexo_rl/training/loop.py`, find the block added in Task 3 Step 2:

```python
from hexo_rl.training.orchestrator import (
    drain_pending_eval as _drain_pending_eval,
    try_save_buffer as _try_save_buffer,
    replay_pretrain_events as _replay_pretrain_events,
    emit_axis_distribution as _emit_axis_distribution,
    emit_training_events as _emit_training_events,
)
```

The `run_training_loop` body still references these as `_drain_pending_eval` etc., so the local-alias form **stays** — but the comment about "shim retired in final commit" is now stale. Update the comment to:

```python
# ── Orchestrator hooks ────────────────────────────────────────────────────────
from hexo_rl.training.orchestrator import (
    drain_pending_eval as _drain_pending_eval,
    try_save_buffer as _try_save_buffer,
    replay_pretrain_events as _replay_pretrain_events,
    emit_axis_distribution as _emit_axis_distribution,
    emit_training_events as _emit_training_events,
)
```

(The aliases are not a shim; they're now the canonical-internal naming. Tests no longer depend on `_drain_pending_eval` resolving from `loop.py`.)

- [ ] **Step 3: Final test pass**

```bash
make test 2>&1 | tail -20
```

Expected: pass count matches branch baseline. Any new failure → revert this commit's test edit, debug, retry.

- [ ] **Step 4: Verify final LOC**

```bash
wc -l hexo_rl/training/loop.py hexo_rl/training/anchor.py hexo_rl/training/signals.py hexo_rl/training/orchestrator.py hexo_rl/training/lifecycle.py hexo_rl/eval/pipeline_setup.py
```

Targets (template Gate 4):
- `loop.py`: 1465 → ~280 (≤ 350 OK)
- `anchor.py`: ~260
- `signals.py`: ~80
- `orchestrator.py`: ~350
- `lifecycle.py`: ~250
- `eval/pipeline_setup.py`: ~50
- Sum new files: ~990 (within ±20%)

If `loop.py` > 350 LOC or new-file sum > 1200 → surface to user, do not auto-fix.

- [ ] **Step 5: Diff stat sanity**

```bash
git diff master --stat
```

Expected: 5 created files (`anchor.py`, `signals.py`, `orchestrator.py`, `lifecycle.py`, `eval/pipeline_setup.py`), 1 heavily modified (`loop.py`), 1 lightly modified (`tests/test_training_loop_graduation.py`).

- [ ] **Step 6: Commit test+shim cleanup**

```bash
git add tests/test_training_loop_graduation.py hexo_rl/training/loop.py
git commit -m "refactor(training): point tests at orchestrator module (§159)"
```

---

### Task 8: Sprint log draft + follow-ups capture

**Files:**
- Create: `/tmp/sprint_log_159_refactor_training_loop_draft.md`
- Create / append: `/tmp/refactor_followups.md`

- [ ] **Step 1: Capture follow-ups discovered during extract**

Append to `/tmp/refactor_followups.md` (create if missing):

```markdown
## §159 follow-ups (separate commits, AFTER refactor lands)

1. **`pool._inference_server.load_state_dict_safe` reach** (orchestrator.py
   `drain_pending_eval`). Cross-module private access predates §159; kept
   verbatim. Candidate for a `WorkerPool.promote_anchor(state_dict)` public
   method in §16x.

2. **`resolve_anchor` arg-list bloat.** Six model-arch ints (board_size,
   res_blocks, filters, in_channels, input_channels, se_reduction_ratio)
   passed flat. Same group repeats in `build_inference_model` and
   `build_eval_model`. Candidate: `ModelArchSpec` frozen dataclass with
   `from_trainer(trainer)` constructor, threaded through lifecycle.py +
   anchor.py. Pure cleanup — out of §159 pure-move scope.

3. **Possible signal-handler bug — surface only.** ShutdownState dataclass now
   holds three fields; lifecycle teardown does not consult `_shutdown.shutdown_save`
   to decide between graceful and rushed teardown. Behavior unchanged because the
   pre-refactor code didn't either, but it's worth a look.

4. **Possible dashboard coupling.** `dashboards.append(td)` after `td.start()` —
   if `td.start()` raises, the dashboard isn't tracked for teardown. Not a §159
   concern; pre-existing.

5. **Possible anchor atomicity tightening.** `save_best_model_atomic` does
   `path.replace(bak)` then `tmp.replace(path)`. Window between the two is
   small but non-zero. fsync would close it; out of §159 scope.

6. **Lifecycle test coverage.** No unit test for `build_subsystems` /
   `LoopSubsystems.teardown` ordering, nor for `resolve_anchor` branching
   (eval=None / fresh-init / arch-mismatch). python_health.md row 1 calls
   these out as "new unit tests needed". Land separately.

7. **Eval pipeline + anchor coupling.** `resolve_anchor` takes
   `eval_pipeline` (only to early-return when None) and `eval_ext_config`
   (for `gating.best_model_path`). Tighter would be
   `resolve_anchor(best_model_path, ...)` — pull the path resolution to the
   call site. Pure cleanup, out of §159 pure-move scope.
```

- [ ] **Step 2: Draft sprint log entry**

Create `/tmp/sprint_log_159_refactor_training_loop_draft.md`:

```markdown
## §159 — Refactor: training/loop.py split (2026-05-06)

**Audit candidate:** `audit/SUMMARY.md` L9 #1 (CRITICAL).

**Outcome:** `hexo_rl/training/loop.py` 1464 → ~280 LOC. Five new files sum
~990 LOC: `training/anchor.py` (260; I/O + `resolve_anchor`),
`training/signals.py` (80), `training/orchestrator.py` (350),
`training/lifecycle.py` (250), `eval/pipeline_setup.py` (50).

**Caller updates:** 1 production (`scripts/train.py` — no edit, public API
preserved), 1 test file (`tests/test_training_loop_graduation.py` — import
path updated from `loop` to `orchestrator`).

**Tests:** `make test` clean, identical pass count to pre-refactor master.

**Bench:** Skipped per `docs/refactor-template.md` L113 — orchestration, not
hot-path.

**Naming decisions:** `training/anchor.py` chosen over kickoff's
`training/checkpoint.py` to avoid clash with existing `training/checkpoints.py`
(plural; trainer-state I/O). "Anchor" matches existing log-event vocabulary
(`anchor_loaded`, `anchor_quarantined`, etc.).

**Follow-ups in `/tmp/refactor_followups.md`:** `pool._inference_server`
reach (cross-module private), `ModelArchSpec` consolidation, lifecycle test
coverage, anchor atomicity fsync, eval+anchor arg coupling.

**Commits on `refactor/training-loop-split`:**
1. `refactor(training): extract best-model anchor to anchor.py (§159)`
2. `refactor(training): extract signal handlers to signals.py (§159)`
3. `refactor(training): extract orchestrator hooks to orchestrator.py (§159)`
4. `refactor(eval): extract eval-pipeline builder to pipeline_setup.py (§159)`
5. `refactor(training): extract resolve_anchor orchestration to anchor.py (§159)`
6. `refactor(training): extract subsystem lifecycle to lifecycle.py (§159)`
7. `refactor(training): point tests at orchestrator module (§159)`
8. `docs(sprint): §159 refactor training/loop.py landed`  ← LANDS WITH USER GO
```

- [ ] **Step 3: Surface to user — DO NOT COMMIT YET**

Print to chat:
```
§159 refactor draft ready.
- TARGET: hexo_rl/training/loop.py 1464 → ~280 LOC
- 5 new files (anchor.py / signals.py / orchestrator.py / lifecycle.py /
  eval/pipeline_setup.py), ~990 LOC sum
- Branch refactor/training-loop-split with 7 commits ready
- Sprint log draft: /tmp/sprint_log_159_refactor_training_loop_draft.md
- Follow-ups: /tmp/refactor_followups.md (7 items)
- Tests pass: <N>/<N>
- Naming: anchor.py confirmed (matches log-event vocabulary, distinct from
  existing checkpoints.py)

Awaiting go to land final docs commit.
```

Stop here. User says go → Task 9 fires.

---

### Task 9: Land sprint log (executes only after user "go")

- [ ] **Step 1: Move draft into sprint log**

Open `docs/07_PHASE4_SPRINT_LOG.md`, append the §159 entry from
`/tmp/sprint_log_159_refactor_training_loop_draft.md` at the end of the file.

- [ ] **Step 2: Commit**

```bash
git add docs/07_PHASE4_SPRINT_LOG.md
git commit -m "docs(sprint): §159 refactor training/loop.py landed"
```

- [ ] **Step 3: Final state**

Branch `refactor/training-loop-split` is ready to merge to master. **Do not
push or merge** without user confirmation per `docs/rules/workflow.md`.

```bash
git log --oneline master..HEAD
```

Expected: 8 commits, all `refactor(training):`, `refactor(eval):`, or `docs(sprint):` prefixed.

---

## Hard constraints (lifted from `docs/refactor-template.md`)

- No behavior change during move commits.
- Public API (`run_training_loop`) identical until final commit.
- No bench gate (orchestration, not hot-path) — skipped per L113.
- No file outside the listed paths gets touched.
- No config defaults modified.
- No test assertions modified.

## Stop signals

- Test failure during any extract → revert that commit, debug, retry. Do **not**
  paper over with a try/except.
- Caller count grows beyond Step 3 of preamble → audit boundary wrong, STOP.
- Hidden coupling discovered (e.g. `_run_loop` reaching into orchestrator
  helper internals beyond the documented signature) → STOP, surface.
- Any LOC budget overflow > 20% → surface to user before committing.
