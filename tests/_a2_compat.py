"""§S181 FU-2 A2 — pytest helper: graceful skip on pre-A2 anchor incompatibility.

A2 (multi-scale avg-pool value head, 2026-05-23) shifted ``value_fc1``
input dim from ``2*filters`` (GAP+GMP) to ``5*filters`` (multi-scale
avg-pool). All checkpoints in the repo at A2 land time were stamped
under the old shape; loading any of them into an A2 model fails the
guard in ``hexo_rl.eval.checkpoint_loader`` /
``hexo_rl.viewer.model_loader``.

Tests that load on-disk anchors call ``a2_load_or_skip(loader, *args,
**kwargs)`` here — the wrapper translates the A2 RuntimeError into a
``pytest.skip`` with a clear message, leaving other RuntimeError paths
untouched. Remove this helper once the repo's anchors are all A2.
"""
from __future__ import annotations

from typing import Any, Callable

import pytest


def is_a2_incompat_error(exc: BaseException) -> bool:
    """True iff the RuntimeError is the A2 value_fc1 shape guard firing."""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return "value_fc1" in msg and "A2" in msg


def a2_load_or_skip(loader: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call loader(*args, **kwargs); skip the test on the A2 guard error."""
    try:
        return loader(*args, **kwargs)
    except RuntimeError as exc:
        if is_a2_incompat_error(exc):
            pytest.skip(
                f"pre-§S181-FU-2 A2 anchor incompatible with multi-scale "
                f"avg-pool value head — re-pretrain anchor under A2 to "
                f"re-enable this test. Underlying error: {exc}"
            )
        raise
