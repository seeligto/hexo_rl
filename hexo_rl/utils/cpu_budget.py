"""CPU thread budget detection + per-library env defaults.

Imported VERY early — before numpy / torch / any BLAS-using library — by
``scripts/train.py`` and ``scripts/benchmark.py``. Stdlib-only so the
import-order constraint is trivially satisfied; the package-level
``hexo_rl.utils.__init__`` is also kept torch-free for the same reason.

PyTorch / NumPy / OpenBLAS / MKL all read OMP_NUM_THREADS (and sibling
vars) at native-runtime initialisation, which happens during their
import. The env vars must be set in os.environ before ``import numpy``
or ``import torch``. Without this, on a rented container with N threads
carved out of an M-thread host (vast.ai 42-of-128, AWS Spot, ...) every
BLAS op tries to grab M threads against the N-slot cgroup, manifesting
as 100 % container CPU + ~60 % GPU util + self-play workers starved of
inference dispatches.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any


_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_INTEROP_THREADS",
)


def detect_cpu_budget() -> int:
    """Return the smallest of (host nproc, sched affinity, cgroup quotas).

    Falls through to 1 if every detection fails (paranoid but harmless).
    """
    candidates: list[int] = []
    n = os.cpu_count()
    if n:
        candidates.append(n)
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    # cgroup v2: /sys/fs/cgroup/cpu.max  ("max <period>" or "<quota> <period>")
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if parts and parts[0] != "max":
            q, p = int(parts[0]), int(parts[1])
            if q > 0 and p > 0:
                candidates.append(max(1, math.ceil(q / p)))
    except (FileNotFoundError, ValueError, IndexError, PermissionError):
        pass
    # cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_{quota,period}_us
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read().strip())
        if q > 0 and p > 0:
            candidates.append(max(1, math.ceil(q / p)))
    except (FileNotFoundError, ValueError, PermissionError):
        pass
    return min(candidates) if candidates else 1


def apply_auto_thread_budget(*, log_prefix: str = "[hexo_rl]", silent: bool = False) -> dict[str, Any]:
    """Set per-library thread caps in os.environ based on detected cgroup budget.

    Idempotent — guarded by ``_HEXO_THREAD_BUDGET_APPLIED``.

    Operator overrides (precedence high → low):
      1. ``_HEXO_THREAD_BUDGET_APPLIED`` already set → no-op.
      2. ``HEXO_THREAD_BUDGET=N`` → forces per-lib value to N.
      3. Pre-existing per-var env (e.g. ``OMP_NUM_THREADS=6``) → respected
         via ``setdefault``; only fills the vars that are NOT already set.

    Default per-lib value: ``clamp(budget // 4, 1, 8)``. Leaves most of the
    cgroup for Rust self-play workers (~1 vCPU each), gives BLAS / batch
    assembly / trainer backward a small parallel slice, caps so a 128-vCPU
    bare-metal host doesn't grant 32-thread BLAS for tiny ops.

    Returns a dict summarising what was applied. ``log_prefix`` controls
    the optional one-line stderr summary; pass ``silent=True`` to suppress.
    """
    if "_HEXO_THREAD_BUDGET_APPLIED" in os.environ:
        return {
            "cpu_budget": detect_cpu_budget(),
            "applied": False,
            **{v: os.environ.get(v, "") for v in _THREAD_ENV_VARS},
        }

    budget = detect_cpu_budget()
    forced = os.environ.get("HEXO_THREAD_BUDGET")
    per_lib = int(forced) if forced else max(1, min(budget // 4, 8))
    for v in _THREAD_ENV_VARS:
        os.environ.setdefault(v, str(per_lib))
    os.environ["_HEXO_THREAD_BUDGET_APPLIED"] = "1"

    if not silent:
        msg = (
            f"{log_prefix} cpu_budget={budget} per_lib={per_lib} "
            f"omp={os.environ['OMP_NUM_THREADS']} mkl={os.environ['MKL_NUM_THREADS']} "
            f"interop={os.environ['TORCH_INTEROP_THREADS']}"
        )
        try:
            print(msg, file=sys.stderr)
        except (OSError, ValueError):
            pass

    return {
        "cpu_budget": budget,
        "per_lib": per_lib,
        "applied": True,
        **{v: os.environ[v] for v in _THREAD_ENV_VARS},
    }


def apply_torch_interop_cap() -> None:
    """Apply ``torch.set_num_interop_threads`` from env after torch is imported.

    Inter-op threads have no env hook in PyTorch — must be set programmatically
    BEFORE any parallel torch work. Call this from script entry points right
    after ``import torch``. No-op if no relevant env var is set.
    """
    n = int(os.environ.get("TORCH_INTEROP_THREADS", os.environ.get("OMP_NUM_THREADS", "0")))
    if n <= 0:
        return
    try:
        import torch as _torch  # local: callers have already imported torch
        _torch.set_num_interop_threads(n)
    except (RuntimeError, ImportError):
        # RuntimeError: parallel runtime already started (set in a prior call).
        # ImportError: torch not actually present (impossible in caller path,
        # but defensive — keep the function side-effect-only).
        pass
