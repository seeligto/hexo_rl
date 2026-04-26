"""IQR-aware comparison + bimodality detection for sweep cells.

Per-cell measurement is noisy (§102/§124: pos/hr IQR can hit ±143k with 80%
of runs in a startup-race tail). We need:

1. ``compare_iqr(A, B)`` — three-way compare that declares TIE when the
   median delta is smaller than the larger of the two IQRs. The search
   strategies in :mod:`strategies` interpret a tie as "shrink symmetrically"
   (ternary) or "favor cheaper" (grid/bisect) — see their docs.
2. ``bimodal_from_raw(raw, median)`` — flags startup-race cells where the
   slow-tail run is far below the median. Triggers on the §125-style
   ``[0, 0, 180k, 185k, 192k]`` pattern.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class CellResult:
    """Outcome of one bench cell. ``raw`` is optional; min/max/IQR are always set.

    The runner builds CellResults from parsed bench stdout (median + IQR +
    min + max + n) since the JSON report does not preserve raw runs. Tests
    construct CellResults directly with raw lists.
    """

    median: float
    iqr: float
    min: float
    max: float
    n_runs: int
    raw: tuple[float, ...] = field(default_factory=tuple)
    bimodal: bool = False

    def __post_init__(self) -> None:
        if self.raw:
            object.__setattr__(self, "n_runs", len(self.raw))


def compare_iqr(a: CellResult, b: CellResult, *, min_iqr: float = 0.0) -> int:
    """Return +1 if a > b, -1 if a < b, 0 if within combined IQR (TIE).

    ``min_iqr`` is a floor on the tie band — protects against
    suspiciously-low IQR readings (e.g. n_runs=2 collapsing both runs to
    the same value).
    """
    delta = a.median - b.median
    band = max(a.iqr, b.iqr, min_iqr)
    if abs(delta) < band:
        return 0
    return 1 if delta > 0 else -1


def bimodal_from_raw(raw: Sequence[float], median: float | None = None) -> bool:
    """Flag the §125 startup-race pattern.

    Triggers when **both**:
      * ``max(raw) > 5 * min(raw)`` — large dynamic range across runs
      * ``min(raw) < 0.2 * median`` — slow-tail run is far below median

    With ``min == 0`` (worker startup race produces zero-pos/hr runs) the
    first condition is vacuously true, so the rule reduces to the
    min/median check — exactly the §125 [0, 0, 180k, 185k, 192k] case.
    """
    if not raw:
        return False
    lo = min(raw)
    hi = max(raw)
    med = median if median is not None else statistics.median(raw)
    if med <= 0:
        return False
    cond_range = (lo == 0) or (hi > 5 * lo)
    cond_floor = lo < 0.2 * med
    return cond_range and cond_floor


def upper_mode_filter(raw: Sequence[float], median: float) -> tuple[float, ...]:
    """For a re-evaluated bimodal cell: return only runs at >= 0.5 * median.

    Matches the §125 spec: when a cell stays bimodal after a re-run,
    treat IQR as the spread of the upper-mode runs. The runner uses this
    to compute IQR for comparison while logging the cell as BIMODAL.
    """
    if median <= 0:
        return tuple(raw)
    return tuple(x for x in raw if x >= 0.5 * median)
