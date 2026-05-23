"""§S181-AUDIT Wave 1 Track B — selfplay buffer position-class snapshot.

At checkpoint cadence sample N positions from the live ReplayBuffer,
classify each (colony / extension / neither) using the Track A
classifier, and emit a `buffer_position_class_snapshot` event with
per-class fractions + per-class mean value_target.

Goal (V-B-C). Confirm-or-refute: at step ≥ 2k the selfplay buffer
becomes colony-heavy (model bias → biased selfplay → reinforced bias).

INSPECTION-ONLY. The classifier is a pure NumPy function over the
canonical (n_planes, H, W) state tensor. Buffer sampling uses
`buffer.sample_batch(N, augment=False)` so no aug-RNG variance enters
the classification.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


def snapshot_buffer_position_classes(
    buffer: Any, step: int, n_sample: int = 5000,
) -> dict[str, float] | None:
    """Sample + classify + emit. Returns the payload dict (or None on failure).

    Args:
        buffer: A `ReplayBuffer` (Rust binding). Must expose `.size`,
            `.sample_batch(n, augment)`.
        step:    Training step at which the snapshot fires.
        n_sample: Target sample count; clamped down to `buffer.size`.

    Returns:
        Dict mirroring the emitted event, or None if the buffer is
        empty or sampling failed.
    """
    if buffer.size <= 0:
        return None
    n = min(n_sample, buffer.size)
    try:
        states, _chain, _policies, outcomes, _own, _wl, _ifs = (
            buffer.sample_batch(n, False)
        )
    except Exception as exc:  # noqa: BLE001 — snapshot must not break the loop
        log.warning("track_b_buffer_snapshot_sample_failed",
                    step=step, error=str(exc))
        return None

    # Classifier lives in scripts/ — make sure the repo root is on path.
    try:
        from scripts.structural_diagnosis.track_a.position_classifier import (
            classify_state,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("track_b_buffer_snapshot_classifier_unavailable",
                    step=step, error=str(exc))
        return None

    states_np = np.asarray(states)
    classes = np.empty(states_np.shape[0], dtype=object)
    for i in range(states_np.shape[0]):
        classes[i] = classify_state(states_np[i])

    outcomes_np = np.asarray(outcomes, dtype=np.float64).reshape(-1)
    n_total = int(states_np.shape[0])
    if n_total == 0:
        return None

    def _frac_and_mean(mask: np.ndarray) -> tuple[int, float, float]:
        cnt = int(mask.sum())
        frac = cnt / n_total
        mean_val = (
            float(outcomes_np[mask].mean()) if cnt > 0 else float("nan")
        )
        return cnt, frac, mean_val

    n_col, col_frac, col_mean = _frac_and_mean(classes == "colony")
    n_ext, ext_frac, ext_mean = _frac_and_mean(classes == "extension")
    n_nei, nei_frac, nei_mean = _frac_and_mean(classes == "neither")

    payload = dict(
        event="buffer_position_class_snapshot",
        step=step,
        n_sampled=n_total,
        buffer_size=int(buffer.size),
        colony_n=n_col,
        extension_n=n_ext,
        neither_n=n_nei,
        colony_frac=round(col_frac, 4),
        extension_frac=round(ext_frac, 4),
        neither_frac=round(nei_frac, 4),
        colony_mean_value_target=round(col_mean, 4) if np.isfinite(col_mean) else None,
        extension_mean_value_target=round(ext_mean, 4) if np.isfinite(ext_mean) else None,
        neither_mean_value_target=round(nei_mean, 4) if np.isfinite(nei_mean) else None,
    )

    try:
        from hexo_rl.monitoring.events import emit_event
        emit_event(payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("track_b_buffer_snapshot_emit_failed",
                    step=step, error=str(exc))

    log.info("buffer_position_class_snapshot",
             step=step, n=n_total,
             colony_frac=payload["colony_frac"],
             extension_frac=payload["extension_frac"],
             neither_frac=payload["neither_frac"])
    return payload
