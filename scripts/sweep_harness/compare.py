"""IQR-aware comparison for sweep cells.

Per-cell measurement is noisy (§102/§124: pos/hr IQR can hit ±143k with 80%
of runs in a startup-race tail). We need ``compare_iqr(A, B)`` — a three-way
compare that declares TIE when the median delta is smaller than the larger of
the two IQRs. The search strategies in :mod:`strategies` interpret a tie as
"shrink symmetrically" (ternary) or "favor cheaper" (grid/bisect).

§128 (2026-04-28): positions_generated counter is continuous (increments once
per ply); the startup-race burst pattern ([0, 0, 180k, 185k, 192k]) that
drove bimodality detection and retry logic cannot occur. Bimodal detection
removed — was dead code.
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
